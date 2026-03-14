"""Main evaluation pipeline.

Orchestrates the full end-to-end flow:
1. Scrape target → 2. Build intent model → 3. Generate personas →
4. Orchestrate evaluation → 5. Apply specialist lenses → 6. Generate reports
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from humanqa.core.intent_modeler import IntentModeler
from humanqa.core.llm import LLMClient
from humanqa.core.orchestrator import Orchestrator
from humanqa.core.persona_generator import PersonaGenerator
from humanqa.core.progress import PipelineProgress
from humanqa.core.repo_analyzer import RepoAnalyzer
from humanqa.core.schemas import RunConfig, RunResult
from humanqa.lenses.auth_lens import AuthLens
from humanqa.lenses.design_lens import DesignLens
from humanqa.lenses.institutional_lens import InstitutionalLens
from humanqa.lenses.trust_lens import TrustLens
from humanqa.reporting.handoff import HandoffGenerator
from humanqa.reporting.report_generator import ReportGenerator
from humanqa.runners.web_runner import WebRunner

logger = logging.getLogger(__name__)


async def run_pipeline(config: RunConfig) -> RunResult:
    """Execute the complete HumanQA evaluation pipeline."""
    llm = LLMClient(provider=config.llm_provider, model=config.llm_model)

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
        repo_insights = await repo_analyzer.analyze(
            config.repo_url, config.github_token_env
        )
        progress.complete_step(
            "repo",
            f"Found: {repo_insights.product_name}, "
            f"{len(repo_insights.claimed_features)} features, "
            f"{len(repo_insights.routes_or_pages)} routes"
        )

    # Step 1b: Scrape landing page
    progress.start_step("scrape", config.target_url)
    web_runner = WebRunner(llm, config.output_dir)
    page_content = await web_runner.scrape_landing_page(config.target_url)
    progress.complete_step("scrape", f"Loaded {len(page_content)} chars of visible content")

    # Step 2: Build Product Intent Model
    progress.start_step("intent")
    modeler = IntentModeler(llm)
    intent = await modeler.build_intent_model(config, page_content, repo_insights)
    progress.update_stats(product=intent.product_name)
    progress.complete_step(
        "intent",
        f"{intent.product_name} ({intent.product_type}), "
        f"confidence={intent.confidence:.0%}, "
        f"{len(intent.critical_journeys)} journeys identified"
    )

    # Step 3: Generate agent personas
    progress.start_step("personas")
    persona_gen = PersonaGenerator(llm)
    agents = await persona_gen.generate_personas(intent, config)
    progress.update_stats(agents=len(agents))
    progress.complete_step("personas", f"{len(agents)} personas: " + ", ".join(a.name for a in agents[:4]))

    # Step 4: Orchestrate evaluation
    progress.start_step("evaluate", f"{len(agents)} agents × {len(intent.critical_journeys)} journeys")
    orchestrator = Orchestrator(llm, config.output_dir)

    # Patch orchestrator to report per-agent progress
    original_run = orchestrator.run

    async def run_with_progress(cfg, intent_model, agent_list):
        result = await original_run(cfg, intent_model, agent_list)
        return result

    result = await orchestrator.run(config, intent, agents)
    progress.update_stats(issues=len(result.issues))
    progress.complete_step("evaluate", f"{len(result.issues)} issues found across {len(agents)} agents")

    # Step 5: Specialist lenses
    if config.design_review:
        progress.start_step("design")
        design_lens = DesignLens(llm, config.output_dir)
        design_issues = await design_lens.review(result, config.design_guidance)
        result.issues.extend(design_issues)
        progress.complete_step("design", f"{len(design_issues)} design issues")

    progress.start_step("trust")
    trust_lens = TrustLens(llm)
    trust_issues, trust_scorecard = await trust_lens.review(result)
    result.issues.extend(trust_issues)
    progress.complete_step(
        "trust",
        f"Trust score: {trust_scorecard.overall_score:.0%}, {len(trust_issues)} issues"
    )

    # Auth/login flow evaluation (no credentials needed)
    progress.start_step("auth")
    auth_lens = AuthLens(llm)
    auth_issues = await auth_lens.review(result)
    result.issues.extend(auth_issues)
    progress.complete_step("auth", f"{len(auth_issues)} auth/login issues")

    if inst_lens.should_run(intent, config.institutional_review):
        progress.start_step("institutional")
        inst_issues = await inst_lens.review(result)
        result.issues.extend(inst_issues)
        progress.complete_step("institutional", f"{len(inst_issues)} governance issues")

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
