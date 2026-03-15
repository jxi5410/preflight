"""Product Intent Modeler.

Builds a Product Intent Model from visible product surfaces (landing page,
onboarding, help docs, app store description, UI copy) and user-provided context.
No code inspection.
"""

from __future__ import annotations

import logging

from preflight.core.llm import LLMClient
from preflight.core.schemas import (
    FeatureExpectation,
    ProductIntentModel,
    InstitutionalRelevance,
    RepoInsights,
    RunConfig,
)

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are a product analyst for Preflight, an AI QA evaluation system.

Your job is to understand what a product IS and DOES by examining its visible surfaces only.
You must NEVER reference source code, repos, or internal architecture.

You will receive:
- Page content scraped from the product's landing/home page
- Any user-supplied brief or context
- Optional focus flows and persona hints
- Optional repository insights (product claims extracted from README, docs, etc.)
- Optional accessibility tree showing the page's interactive elements

When repository insights are provided, cross-reference them with what you see in the UI.
Build a feature expectation list: features the repo claims should exist.
Use recent changes to prioritize testing areas (recently changed = higher risk).

From this, infer:
- product_name: The product's name
- product_type: Category (e.g. "B2B SaaS dashboard", "consumer marketplace", "developer tool")
- target_audience: Who this is for (list of audience segments)
- primary_jobs: What users hire this product to do (list of jobs-to-be-done)
- user_expectations: What users would reasonably expect from a product like this
- critical_journeys: The most important user flows to test
- trust_sensitive_actions: Actions where trust, accuracy, or safety matter most
- institutional_relevance: "none" | "low" | "moderate" | "high" — would serious professionals/institutions use this?
- institutional_reasoning: Why you assigned that relevance level
- input_first: true if the product's primary interaction requires user input before showing content (e.g. search engines, AI tools, URL analyzers). Strong signals: prominent input/textarea as main element, input with submit button as primary CTA, placeholder like "Search...", "Enter URL...", "Ask anything...", page has very little content besides the input. false otherwise.
- input_type: If input_first is true, one of: "search", "prompt", "url", "code", "data", "free_text". Empty string if input_first is false.
- input_placeholder: If input_first is true, the actual placeholder text from the input field. Empty string if not applicable.
- assumptions: What you're assuming that could be wrong
- confidence: 0.0-1.0 how confident you are in this model
- feature_expectations: list of {{feature_name, source}} objects for features the product claims to offer

Respond with valid JSON matching the schema exactly."""

INTENT_PROMPT_TEMPLATE = """Analyze this product and build a Product Intent Model.

## Product URL
{url}

## Scraped Landing Page Content
{page_content}

## Accessibility Tree (interactive elements)
{accessibility_tree}

## User-Supplied Brief
{brief}

## Focus Flows (if any)
{focus_flows}

## Persona Hints (if any)
{persona_hints}

## Repository Insights (if available)
{repo_insights}

Respond with a JSON object with these fields:
product_name, product_type, target_audience (list), primary_jobs (list),
user_expectations (list), critical_journeys (list), trust_sensitive_actions (list),
institutional_relevance ("none"|"low"|"moderate"|"high"), institutional_reasoning (string),
input_first (bool), input_type (string: "search"|"prompt"|"url"|"code"|"data"|"free_text"|""),
input_placeholder (string),
assumptions (list), confidence (float 0-1),
feature_expectations (list of {{"feature_name": "...", "source": "..."}})."""


class IntentModeler:
    """Infers product purpose from visible surfaces."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _format_repo_insights(self, repo_insights: RepoInsights | None) -> str:
        """Format repo insights for inclusion in the prompt."""
        if repo_insights is None:
            return "(no repository analysis available)"

        parts: list[str] = []
        if repo_insights.product_name:
            parts.append(f"Product: {repo_insights.product_name}")
        if repo_insights.description:
            parts.append(f"Description: {repo_insights.description}")
        if repo_insights.tech_stack:
            parts.append(f"Tech stack: {', '.join(repo_insights.tech_stack)}")
        if repo_insights.claimed_features:
            parts.append("Claimed features:\n" + "\n".join(
                f"  - {f}" for f in repo_insights.claimed_features
            ))
        if repo_insights.routes_or_pages:
            parts.append("Routes/pages:\n" + "\n".join(
                f"  - {r}" for r in repo_insights.routes_or_pages[:20]
            ))
        if repo_insights.recent_changes:
            parts.append("Recent changes (prioritize testing these areas):\n" + "\n".join(
                f"  - {c}" for c in repo_insights.recent_changes[:10]
            ))
        if repo_insights.known_issues:
            parts.append("Known open issues:\n" + "\n".join(
                f"  - {i}" for i in repo_insights.known_issues[:10]
            ))
        if repo_insights.documentation_summary:
            parts.append(f"Documentation summary: {repo_insights.documentation_summary}")

        return "\n\n".join(parts) if parts else "(repository analysis found no useful data)"

    async def build_intent_model(
        self,
        config: RunConfig,
        page_content: str,
        repo_insights: RepoInsights | None = None,
        accessibility_tree: str = "",
    ) -> ProductIntentModel:
        """Build a Product Intent Model from scraped content, config, and optional repo insights."""
        prompt = INTENT_PROMPT_TEMPLATE.format(
            url=config.target_url,
            page_content=page_content[:12000],
            accessibility_tree=accessibility_tree[:5000] if accessibility_tree else "(not available)",
            brief=config.brief or "(none provided)",
            focus_flows=", ".join(config.focus_flows) if config.focus_flows else "(none)",
            persona_hints=", ".join(config.persona_hints) if config.persona_hints else "(none)",
            repo_insights=self._format_repo_insights(repo_insights),
        )

        logger.info("Building product intent model for %s", config.target_url)

        try:
            data = self.llm.complete_json(prompt, system=INTENT_SYSTEM_PROMPT)
            # Map institutional_relevance string to enum
            ir_raw = data.get("institutional_relevance", "none")
            data["institutional_relevance"] = InstitutionalRelevance(ir_raw)

            # Normalize input_first fields
            data.setdefault("input_first", False)
            data.setdefault("input_type", "")
            data.setdefault("input_placeholder", "")

            # Extract feature expectations from LLM response
            raw_expectations = data.pop("feature_expectations", [])
            feature_expectations = [
                FeatureExpectation(
                    feature_name=fe.get("feature_name", ""),
                    source=fe.get("source", ""),
                )
                for fe in raw_expectations
                if isinstance(fe, dict) and fe.get("feature_name")
            ]

            model = ProductIntentModel(
                **data,
                repo_insights=repo_insights,
                feature_expectations=feature_expectations,
            )
            logger.info(
                "Intent model built: %s (%s), confidence=%.2f, input_first=%s, features=%d",
                model.product_name,
                model.product_type,
                model.confidence,
                model.input_first,
                len(model.feature_expectations),
            )
            return model
        except Exception as e:
            logger.error("Failed to build intent model: %s", e)
            return ProductIntentModel(
                product_name="Unknown",
                product_type="Unknown",
                confidence=0.1,
                assumptions=["Intent modeling failed; using minimal defaults"],
                repo_insights=repo_insights,
            )
