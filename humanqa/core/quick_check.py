"""Quick Check — lightweight, single-pass evaluation for MCP and CI.

Returns a fast assessment of a URL without the full multi-agent pipeline.
Designed for <30s turnaround: scrape + single LLM call + structured result.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from humanqa.core.llm import LLMClient
from humanqa.core.schemas import Severity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QuickIssue(BaseModel):
    """A single finding from a quick check."""
    title: str
    severity: str = "medium"
    category: str = "functional"
    confidence: float = 0.7
    user_impact: str = ""


class QuickCheckResult(BaseModel):
    """Result of a quick check — fast, lightweight assessment."""
    url: str
    product_name: str = ""
    product_type: str = ""
    issues: list[QuickIssue] = Field(default_factory=list)
    summary: str = ""
    score: float = 0.0  # 0.0 (terrible) to 1.0 (no issues)
    checked_at: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Quick check function
# ---------------------------------------------------------------------------

QUICK_CHECK_PROMPT = """\
You are HumanQA, an AI QA evaluation system. Analyze this web page content
and provide a quick assessment of quality issues a real user would encounter.

URL: {url}
{focus_section}
Page content:
{content}

Respond with JSON:
{{
  "product_name": "name of the product",
  "product_type": "saas | marketing | ecommerce | content | other",
  "issues": [
    {{
      "title": "short issue title",
      "severity": "critical | high | medium | low | info",
      "category": "functional | ux | ui | performance | trust | accessibility | copy | auth",
      "confidence": 0.8,
      "user_impact": "what the user experiences"
    }}
  ],
  "summary": "1-2 sentence overall assessment",
  "score": 0.75
}}

Focus on real, observable problems. Be specific and evidence-based.
Do not invent issues you cannot see in the content.
Return 0-10 issues, prioritized by severity.
"""


async def quick_check(
    url: str,
    focus: str | None = None,
    llm: LLMClient | None = None,
    tier: str = "balanced",
) -> QuickCheckResult:
    """Run a quick, single-pass evaluation of a URL.

    Args:
        url: The URL to check.
        focus: Optional focus area (e.g. "checkout flow", "accessibility").
        llm: Optional pre-configured LLM client.
        tier: Model tier to use if creating a new LLM client.

    Returns:
        QuickCheckResult with issues and summary.
    """
    import time

    start = time.monotonic()

    if llm is None:
        llm = LLMClient(tier=tier)

    # Step 1: Scrape the page
    from humanqa.runners.web_runner import WebRunner

    runner = WebRunner(llm, "/tmp/humanqa_quick")
    try:
        content = await runner.scrape_landing_page(url)
    except Exception as e:
        logger.warning("Quick check scrape failed for %s: %s", url, e)
        content = f"(Failed to load page: {e})"

    if not content:
        content = "(Page returned no visible content)"

    # Cap content to avoid token overflow
    if len(content) > 8000:
        content = content[:8000] + "\n...(truncated)"

    # Step 2: Single LLM call
    focus_section = f"Focus area: {focus}\n" if focus else ""
    prompt = QUICK_CHECK_PROMPT.format(
        url=url,
        content=content,
        focus_section=focus_section,
    )

    try:
        data = llm.complete_json(prompt, tier="fast")
    except Exception as e:
        logger.warning("Quick check LLM call failed: %s", e)
        elapsed = time.monotonic() - start
        return QuickCheckResult(
            url=url,
            summary=f"Quick check failed: {e}",
            score=0.5,
            duration_seconds=round(elapsed, 1),
        )

    elapsed = time.monotonic() - start

    # Step 3: Parse into schema
    issues = []
    for raw_issue in data.get("issues", []):
        issues.append(QuickIssue(
            title=raw_issue.get("title", "Unknown issue"),
            severity=raw_issue.get("severity", "medium"),
            category=raw_issue.get("category", "functional"),
            confidence=raw_issue.get("confidence", 0.7),
            user_impact=raw_issue.get("user_impact", ""),
        ))

    return QuickCheckResult(
        url=url,
        product_name=data.get("product_name", ""),
        product_type=data.get("product_type", ""),
        issues=issues,
        summary=data.get("summary", ""),
        score=max(0.0, min(1.0, data.get("score", 0.5))),
        duration_seconds=round(elapsed, 1),
    )
