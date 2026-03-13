"""Main evaluation pipeline.

Orchestrates the full end-to-end flow:
1. Scrape target → 2. Build intent model → 3. Generate personas →
4. Orchestrate evaluation → 5. Apply specialist lenses → 6. Generate reports
"""

from __future__ import annotations

import logging
from datetime import datetime

from humanqa.core.intent_modeler import IntentModeler
from humanqa.core.llm import LLMClient
from humanqa.core.orchestrator import Orchestrator
from humanqa.core.persona_generator import PersonaGenerator
from humanqa.core.repo_analyzer import RepoAnalyzer
from humanqa.core.schemas import RunConfig, RunResult
from humanqa.lenses.design_lens import DesignLens
from humanqa.lenses.institutional_lens import InstitutionalLens
from humanqa.reporting.report_generator import ReportGenerator
from humanqa.runners.web_runner import WebRunner

logger = logging.getLogger(__name__)


async def run_pipeline(config: RunConfig) -> RunResult:
    """Execute the complete HumanQA evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("HumanQA evaluation starting for %s", config.target_url)
    logger.info("=" * 60)

    llm = LLMClient(provider=config.llm_provider, model=config.llm_model)

    # Step 1a: Analyze repository (if provided)
    repo_insights = None
    if config.repo_url:
        logger.info("Step 1a: Analyzing repository %s ...", config.repo_url)
        repo_analyzer = RepoAnalyzer(llm)
        repo_insights = await repo_analyzer.analyze(
            config.repo_url, config.github_token_env
        )
        logger.info(
            "Repo analysis complete: %s (confidence=%.0f%%)",
            repo_insights.product_name,
            repo_insights.repo_confidence * 100,
        )

    # Step 1b: Scrape landing page for intent modeling
    logger.info("Step 1b: Scraping target product...")
    web_runner = WebRunner(llm, config.output_dir)
    page_content = await web_runner.scrape_landing_page(config.target_url)

    # Step 2: Build Product Intent Model
    logger.info("Step 2: Building product intent model...")
    modeler = IntentModeler(llm)
    intent = await modeler.build_intent_model(config, page_content, repo_insights)
    logger.info(
        "Product identified as: %s (%s), confidence=%.0f%%",
        intent.product_name, intent.product_type, intent.confidence * 100,
    )

    # Step 3: Generate agent personas
    logger.info("Step 3: Generating agent personas...")
    persona_gen = PersonaGenerator(llm)
    agents = await persona_gen.generate_personas(intent, config)
    logger.info("Generated %d agent personas", len(agents))

    # Step 4: Orchestrate evaluation
    logger.info("Step 4: Running evaluation with %d agents...", len(agents))
    orchestrator = Orchestrator(llm, config.output_dir)
    result = await orchestrator.run(config, intent, agents)

    # Step 5: Apply specialist lenses
    logger.info("Step 5: Applying specialist lenses...")

    # Design lens
    if config.design_review:
        logger.info("  Running design review...")
        design_lens = DesignLens(llm, config.output_dir)
        design_issues = await design_lens.review(result, config.design_guidance)
        result.issues.extend(design_issues)
        logger.info("  Design review found %d issues", len(design_issues))

    # Institutional lens
    inst_lens = InstitutionalLens(llm)
    if inst_lens.should_run(intent, config.institutional_review):
        logger.info("  Running institutional/governance review...")
        inst_issues = await inst_lens.review(result)
        result.issues.extend(inst_issues)
        logger.info("  Institutional review found %d issues", len(inst_issues))
    else:
        logger.info("  Institutional review skipped (not relevant)")

    # Re-sort all issues after adding lens results
    result.issues.sort(
        key=lambda i: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(i.severity.value, 5),
            -i.confidence,
        ),
    )

    result.completed_at = datetime.now(tz=__import__("datetime").timezone.utc)

    # Step 6: Generate reports
    logger.info("Step 6: Generating reports...")
    reporter = ReportGenerator(config.output_dir)
    paths = reporter.generate_all(result)
    logger.info("Reports generated: %s", paths)

    # Summary
    logger.info("=" * 60)
    logger.info("HumanQA evaluation complete")
    logger.info("  Product: %s", intent.product_name)
    logger.info("  Issues found: %d", len(result.issues))
    logger.info("  Agents used: %d", len(result.agents))
    logger.info("  Duration: %s", result.completed_at - result.started_at)
    logger.info("  Output: %s", config.output_dir)
    logger.info("=" * 60)

    return result
