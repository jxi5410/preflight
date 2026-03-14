"""Responsive/Mobile Layout Lens — evaluates mobile responsiveness and visual layout.

Mandatory vision-based evaluation for mobile viewports:
- Desktop-vs-mobile comparison (same pages, different viewports)
- Touch target sizing
- Text readability at mobile scale
- Horizontal overflow / scroll issues
- Navigation adaptation (hamburger menu, etc.)
"""

from __future__ import annotations

import logging

from humanqa.core.llm import LLMClient
from humanqa.core.schemas import (
    Evidence,
    Issue,
    IssueCategory,
    Platform,
    RunResult,
    Severity,
)

logger = logging.getLogger(__name__)

RESPONSIVE_SYSTEM_PROMPT = """You are a mobile responsiveness evaluator for HumanQA.

You compare desktop and mobile evaluation results to identify responsive design issues.

Focus on:
1. **Layout breaks**: Content that overflows, overlaps, or becomes unreadable on mobile
2. **Touch targets**: Buttons/links smaller than 44x44px on mobile
3. **Text sizing**: Text too small to read on mobile (< 14px effective)
4. **Navigation**: Desktop nav that doesn't adapt to mobile (no hamburger/drawer)
5. **Horizontal scroll**: Pages wider than viewport forcing horizontal scroll
6. **Image scaling**: Images that don't scale down or break layout
7. **Form usability**: Inputs too small, dropdowns unusable on touch
8. **Viewport issues**: Content hidden or cut off on smaller screens

You MUST cite specific evidence for every finding.

Respond with JSON:
{
  "issues": [
    {
      "title": "...",
      "severity": "critical|high|medium|low|info",
      "confidence": 0.0-1.0,
      "user_impact": "...",
      "observed_facts": ["..."],
      "inferred_judgment": "...",
      "repair_brief": "...",
      "affects_viewport": "mobile|both"
    }
  ],
  "responsive_score": 0.0-1.0,
  "summary": "..."
}"""

RESPONSIVE_EVAL_PROMPT = """Evaluate mobile responsiveness by comparing desktop and mobile findings.

## Product: {product_name} ({product_type})
## Target URL: {target_url}

## Desktop Issues (viewport 1440x900)
{desktop_issues}

## Mobile Issues (viewport 390x844)
{mobile_issues}

## Coverage
{coverage_summary}

Compare desktop vs mobile findings. Identify:
1. Issues that only appear on mobile (responsive breakage)
2. Issues worse on mobile than desktop
3. Missing mobile adaptations (navigation, touch targets)
4. Mobile-specific UX problems

Rate overall responsive quality 0.0-1.0."""


class ResponsiveLens:
    """Evaluates mobile responsiveness and visual layout."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def review(self, result: RunResult) -> tuple[list[Issue], float]:
        """Review mobile responsiveness by comparing desktop/mobile findings.

        Returns (issues, responsive_score).
        """
        # Separate desktop and mobile issues
        desktop_issues = [
            i for i in result.issues
            if i.platform == Platform.web
        ]
        mobile_issues = [
            i for i in result.issues
            if i.platform == Platform.mobile_web
        ]

        if not mobile_issues and not desktop_issues:
            return [], 1.0  # No data to compare

        # Format issues for comparison
        desktop_text = self._format_issues(desktop_issues, "desktop")
        mobile_text = self._format_issues(mobile_issues, "mobile")

        if not desktop_text and not mobile_text:
            return [], 1.0

        # Coverage summary
        mobile_pages = sum(
            1 for e in result.coverage.entries
            if "mobile" in e.agent_id.lower()
        )
        desktop_pages = len(result.coverage.entries) - mobile_pages

        coverage = (
            f"Desktop pages: {desktop_pages}, Mobile pages: {mobile_pages}, "
            f"Total: {len(result.coverage.entries)}"
        )

        prompt = RESPONSIVE_EVAL_PROMPT.format(
            product_name=result.intent_model.product_name,
            product_type=result.intent_model.product_type,
            target_url=result.config.target_url,
            desktop_issues=desktop_text or "(no desktop issues)",
            mobile_issues=mobile_text or "(no mobile issues)",
            coverage_summary=coverage,
        )

        try:
            data = self.llm.complete_json(prompt, system=RESPONSIVE_SYSTEM_PROMPT)
        except Exception as e:
            logger.warning("Responsive lens evaluation failed: %s", e)
            return [], 0.5

        issues: list[Issue] = []
        for raw in data.get("issues", []):
            sev = raw.get("severity", "medium")
            try:
                severity = Severity(sev)
            except ValueError:
                severity = Severity.medium

            viewport = raw.get("affects_viewport", "mobile")
            platform = Platform.mobile_web if viewport == "mobile" else Platform.web

            issues.append(Issue(
                title=raw.get("title", "Responsive issue"),
                severity=severity,
                confidence=raw.get("confidence", 0.7),
                platform=platform,
                category=IssueCategory.responsive,
                agent="responsive_lens",
                user_impact=raw.get("user_impact", ""),
                observed_facts=raw.get("observed_facts", []),
                inferred_judgment=raw.get("inferred_judgment", ""),
                repair_brief=raw.get("repair_brief", ""),
                likely_product_area="Layout/Responsive",
            ))

        responsive_score = data.get("responsive_score", 0.5)

        # Flag if no mobile evaluation was done at all
        if not mobile_issues:
            issues.append(Issue(
                title="No mobile viewport evaluation performed",
                severity=Severity.medium,
                confidence=0.9,
                platform=Platform.mobile_web,
                category=IssueCategory.responsive,
                agent="responsive_lens",
                user_impact="Mobile users may encounter untested layout issues",
                observed_facts=["No mobile viewport (390x844) evaluation was included in this run"],
                inferred_judgment="Mobile testing should be mandatory for web products",
                repair_brief="Ensure at least one agent uses mobile_web viewport",
                likely_product_area="Layout/Responsive",
            ))
            responsive_score = min(responsive_score, 0.3)

        logger.info(
            "Responsive lens: %d issues, score=%.1f",
            len(issues), responsive_score,
        )
        return issues, responsive_score

    @staticmethod
    def _format_issues(issues: list[Issue], label: str) -> str:
        """Format issues for prompt context."""
        if not issues:
            return ""
        lines = []
        for i in issues[:20]:  # Cap for token budget
            lines.append(
                f"- [{i.severity.value}] {i.title} "
                f"(category={i.category.value}, area={i.likely_product_area})"
            )
        return f"### {label.title()} ({len(issues)} issues)\n" + "\n".join(lines)
