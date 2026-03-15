"""Main evaluation pipeline.

Orchestrates the full end-to-end flow:
1. Scrape target -> 2. Build intent model -> 3. Generate personas ->
4. Orchestrate evaluation -> 5. Apply specialist lenses (parallelized) ->
6. Generate reports
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from preflight.core.intent_modeler import IntentModeler
from preflight.core.llm import LLMClient
from preflight.core.orchestrator import Orchestrator
from preflight.core.persona_generator import PersonaGenerator
from preflight.core.seed_input import SeedInputGenerator
from preflight.core.progress import PipelineProgress
from preflight.core.repo_analyzer import RepoAnalyzer
from preflight.core.schemas import RunConfig, RunResult
from preflight.lenses.auth_lens import AuthLens
from preflight.lenses.design_lens import DesignLens
from preflight.lenses.institutional_lens import InstitutionalLens
from preflight.lenses.responsive_lens import ResponsiveLens
from preflight.lenses.trust_lens import TrustLens
from preflight.reporting.handoff import HandoffGenerator
from preflight.reporting.report_generator import ReportGenerator
from preflight.runners.web_runner import WebRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout and cap constants
# ---------------------------------------------------------------------------

STEP_TIMEOUT_SECONDS = {
    "repo": 60,
    "scrape": 30,
    "intent": 30,
    "personas": 20,
    "evaluate": 300,  # 5 min cap for full evaluation
    "lenses": 120,  # 2 min cap for all lenses combined
    "reports": 30,
    "handoff": 15,
}

MAX_JOURNEYS_PER_AGENT = 3  # Cap journeys to avoid long runs
MAX_AGENTS = 6  # Cap persona count


async def _with_timeout(coro, timeout_sec: float, label: str):
    """Run a coroutine with a timeout, logging on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning("Step '%s' timed out after %ds", label, timeout_sec)
        return None


async def run_pipeline(config: RunConfig) -> RunResult:
    """Execute the complete Preflight evaluation pipeline."""
    llm = LLMClient(
        provider=config.llm_provider,
        model=config.llm_model if config.llm_model != "gemini-2.0-flash" else None,
        tier=config.llm_tier,
    )

    # Determine which steps will run
    inst_lens = InstitutionalLens(llm)
    progress = PipelineProgress(
        has_repo=bool(config.repo_url),
        has_design=config.design_review,
        has_institutional=True,  # Decided after intent model, but show in plan
    )

    progress.show_plan()

    # Step 1a: Analyze repository (if provided)
    repo_insights = None
    if config.repo_url:
        progress.start_step("repo", config.repo_url)
        repo_analyzer = RepoAnalyzer(llm)
        repo_insights = await _with_timeout(
            repo_analyzer.analyze(config.repo_url, config.github_token_env),
            STEP_TIMEOUT_SECONDS["repo"],
            "repo",
        )
        if repo_insights:
            progress.complete_step(
                "repo",
                f"Found: {repo_insights.product_name}, "
                f"{len(repo_insights.claimed_features)} features, "
                f"{len(repo_insights.routes_or_pages)} routes"
            )
        else:
            progress.complete_step("repo", "Timed out — continuing without repo insights")

    # Step 1b: Scrape landing page (with accessibility tree for input detection)
    progress.start_step("scrape", config.target_url)
    web_runner = WebRunner(llm, config.output_dir)
    scrape_result = await _with_timeout(
        web_runner.scrape_landing_page(config.target_url, include_a11y_tree=True),
        STEP_TIMEOUT_SECONDS["scrape"],
        "scrape",
    )
    if scrape_result is None:
        page_content = "(scrape timed out)"
        accessibility_tree = ""
    elif isinstance(scrape_result, tuple):
        page_content, accessibility_tree = scrape_result
    else:
        page_content = scrape_result
        accessibility_tree = ""
    progress.complete_step("scrape", f"Loaded {len(page_content)} chars of visible content")

    # Step 2: Build Product Intent Model
    progress.start_step("intent")
    modeler = IntentModeler(llm)
    intent = await modeler.build_intent_model(
        config, page_content, repo_insights, accessibility_tree=accessibility_tree,
    )
    progress.update_stats(product=intent.product_name)
    progress.complete_step(
        "intent",
        f"{intent.product_name} ({intent.product_type}), "
        f"confidence={intent.confidence:.0%}, "
        f"{len(intent.critical_journeys)} journeys identified"
    )

    # Cap journeys to prevent long runs
    if len(intent.critical_journeys) > MAX_JOURNEYS_PER_AGENT * 2:
        logger.info(
            "Capping journeys from %d to %d",
            len(intent.critical_journeys),
            MAX_JOURNEYS_PER_AGENT * 2,
        )
        intent.critical_journeys = intent.critical_journeys[:MAX_JOURNEYS_PER_AGENT * 2]

    # Step 3: Generate agent personas
    progress.start_step("personas")
    persona_gen = PersonaGenerator(llm)
    agents = await persona_gen.generate_personas(intent, config)
    # Cap agent count
    if len(agents) > MAX_AGENTS:
        logger.info("Capping agents from %d to %d", len(agents), MAX_AGENTS)
        agents = agents[:MAX_AGENTS]
    progress.update_stats(agents=len(agents))
    progress.complete_step("personas", f"{len(agents)} personas: " + ", ".join(a.name for a in agents[:4]))

    # Step 3b: Generate seed inputs for input-first products
    if intent.input_first:
        seed_gen = SeedInputGenerator(llm)
        for agent in agents:
            agent.seed_inputs = seed_gen.generate_for_persona(intent, agent)
        total_seeds = sum(len(a.seed_inputs) for a in agents)
        logger.info("Generated %d total seed inputs for %d agents", total_seeds, len(agents))

    # Step 4: Orchestrate evaluation (with timeout)
    progress.start_step("evaluate", f"{len(agents)} agents x {len(intent.critical_journeys)} journeys")
    orchestrator = Orchestrator(llm, config.output_dir)

    result = await _with_timeout(
        orchestrator.run(config, intent, agents),
        STEP_TIMEOUT_SECONDS["evaluate"],
        "evaluate",
    )

    if result is None:
        # Evaluation timed out — create minimal result
        result = RunResult(
            config=config,
            intent_model=intent,
            agents=agents,
            started_at=datetime.now(tz=timezone.utc),
        )
        logger.warning("Evaluation timed out — running lenses with partial results")

    progress.update_stats(issues=len(result.issues))
    progress.complete_step("evaluate", f"{len(result.issues)} issues found across {len(agents)} agents")

    # Step 5: Specialist lenses — run in parallel for speed
    progress.start_step("lenses", "Running design, trust, responsive, auth lenses in parallel")

    async def run_design():
        if not config.design_review:
            return []
        design_lens = DesignLens(llm, config.output_dir)
        return await design_lens.review(result, config.design_guidance)

    async def run_trust():
        trust_lens = TrustLens(llm)
        issues, scorecard = await trust_lens.review(result)
        return issues, scorecard

    async def run_responsive():
        responsive_lens = ResponsiveLens(llm, output_dir=config.output_dir)
        return await responsive_lens.review(result)

    async def run_auth():
        auth_lens = AuthLens(llm)
        return await auth_lens.review(result)

    # Execute all lenses concurrently with timeout
    lens_tasks = [
        asyncio.create_task(run_design()),
        asyncio.create_task(run_trust()),
        asyncio.create_task(run_responsive()),
        asyncio.create_task(run_auth()),
    ]

    try:
        done, pending = await asyncio.wait(
            lens_tasks,
            timeout=STEP_TIMEOUT_SECONDS["lenses"],
        )
        # Cancel any timed-out lenses
        for task in pending:
            task.cancel()
            logger.warning("Lens task timed out and was cancelled")
    except Exception as e:
        logger.warning("Lens parallelization error: %s", e)
        done = set()

    # Collect results from completed lenses
    lens_summary_parts = []
    for task in done:
        try:
            task_result = task.result()
            if task_result is None:
                continue

            # Design lens returns list[Issue]
            if isinstance(task_result, list):
                result.issues.extend(task_result)
                lens_summary_parts.append(f"{len(task_result)} design")

            # Trust lens returns (list[Issue], TrustScorecard)
            elif isinstance(task_result, tuple) and len(task_result) == 2:
                issues_or_score, second = task_result
                if isinstance(issues_or_score, list):
                    # Trust: (issues, scorecard)
                    result.issues.extend(issues_or_score)
                    if hasattr(second, 'overall_score'):
                        lens_summary_parts.append(
                            f"trust={second.overall_score:.0%}"
                        )
                    else:
                        # Responsive: (issues, score)
                        result.scores["responsive_score"] = second
                        lens_summary_parts.append(
                            f"responsive={second:.0%}"
                        )
                else:
                    # Responsive returns (issues, float)
                    result.issues.extend(issues_or_score)
                    result.scores["responsive_score"] = second
        except Exception as e:
            logger.warning("Error collecting lens result: %s", e)

    progress.complete_step("lenses", ", ".join(lens_summary_parts) or "complete")

    # Institutional lens runs separately (may be skipped)
    if inst_lens.should_run(intent, config.institutional_review):
        progress.start_step("institutional")
        inst_issues = await _with_timeout(
            inst_lens.review(result),
            60,
            "institutional",
        )
        if inst_issues:
            result.issues.extend(inst_issues)
        progress.complete_step("institutional", f"{len(inst_issues or [])} governance issues")

    # Re-sort all issues
    result.issues.sort(
        key=lambda i: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(i.severity.value, 5),
            -i.confidence,
        ),
    )

    result.completed_at = datetime.now(tz=timezone.utc)
    progress.update_stats(issues=len(result.issues))

    # Step 6: Generate reports
    progress.start_step("reports")
    reporter = ReportGenerator(config.output_dir)
    paths = reporter.generate_all(result)
    progress.complete_step("reports", "report.md, report.html, report.json")

    # Step 7: Generate developer handoff
    progress.start_step("handoff")
    handoff_gen = HandoffGenerator(config.output_dir)
    handoff_paths = handoff_gen.generate_all(result, repo_insights)
    progress.complete_step("handoff", "HANDOFF.md, handoff.json")

    # Final summary
    duration = str(result.completed_at - result.started_at).split(".")[0]
    progress.show_summary(duration=duration)

    return result
