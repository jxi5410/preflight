"""Quick Check — lightweight, single-pass evaluation for MCP and CI.

Returns a fast assessment of a URL without the full multi-agent pipeline.
Designed for <30s turnaround: scrape + single LLM call + structured result.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from preflight.core.llm import LLMClient
from preflight.core.schemas import Severity

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
    input_first: bool = False
    input_type: str = ""
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
You are Preflight, an AI QA evaluation system. Analyze this web page content
and provide a quick assessment of quality issues a real user would encounter.

URL: {url}
{focus_section}
Page content:
{content}

Accessibility tree (interactive elements):
{accessibility_tree}
{input_first_section}
Respond with JSON:
{{
  "product_name": "name of the product",
  "product_type": "saas | marketing | ecommerce | content | other",
  "input_first": false,
  "input_type": "",
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

Set input_first to true if the product's primary interaction requires user input
before showing content (search engines, AI tools, URL analyzers, etc.).
If input_first is true, set input_type to one of: search, prompt, url, code, data, free_text.

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

    # Step 1: Scrape the page (with a11y tree for input detection)
    from preflight.runners.web_runner import WebRunner

    runner = WebRunner(llm, "/tmp/preflight_quick")
    accessibility_tree = ""
    try:
        scrape_result = await runner.scrape_landing_page(url, include_a11y_tree=True)
        if isinstance(scrape_result, tuple):
            content, accessibility_tree = scrape_result
        else:
            content = scrape_result
    except Exception as e:
        logger.warning("Quick check scrape failed for %s: %s", url, e)
        content = f"(Failed to load page: {e})"

    if not content:
        content = "(Page returned no visible content)"

    # Cap content to avoid token overflow
    if len(content) > 8000:
        content = content[:8000] + "\n...(truncated)"

    # Step 2: Single LLM call (with input detection)
    focus_section = f"Focus area: {focus}\n" if focus else ""
    prompt = QUICK_CHECK_PROMPT.format(
        url=url,
        content=content,
        accessibility_tree=accessibility_tree[:3000] if accessibility_tree else "(not available)",
        focus_section=focus_section,
        input_first_section="",
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

    is_input_first = data.get("input_first", False)
    input_type = data.get("input_type", "")

    # Step 2b: If input-first detected, run a heuristic seed input and re-evaluate
    if is_input_first and input_type:
        from preflight.core.seed_input import get_heuristic_seed_input

        seed = get_heuristic_seed_input(input_type)
        try:
            seed_content = await _quick_check_seed_input(runner, url, seed.input_text)
            if seed_content:
                seed_prompt = QUICK_CHECK_PROMPT.format(
                    url=url,
                    content=seed_content,
                    accessibility_tree="(after seed input submission)",
                    focus_section=focus_section,
                    input_first_section=(
                        f"\nThis is an input-first product. A seed input '{seed.input_text}' "
                        f"was typed and submitted. The content below shows the results.\n"
                        f"Evaluate both the input UX and the quality of results.\n"
                    ),
                )
                try:
                    seed_data = llm.complete_json(seed_prompt, tier="fast")
                    # Merge seed input issues with landing page issues
                    for raw_issue in seed_data.get("issues", []):
                        data.setdefault("issues", []).append(raw_issue)
                    # Update summary if seed provided one
                    if seed_data.get("summary"):
                        data["summary"] = (
                            data.get("summary", "")
                            + f" After typing '{seed.input_text}': "
                            + seed_data["summary"]
                        )
                except Exception as e:
                    logger.warning("Quick check seed evaluation failed: %s", e)
        except Exception as e:
            logger.warning("Quick check seed input failed: %s", e)

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
        input_first=is_input_first,
        input_type=input_type,
        issues=issues,
        summary=data.get("summary", ""),
        score=max(0.0, min(1.0, data.get("score", 0.5))),
        duration_seconds=round(elapsed, 1),
    )


async def _quick_check_seed_input(
    runner: "WebRunner",
    url: str,
    seed_text: str,
) -> str:
    """Navigate to URL, type seed input, submit, return resulting page content.

    Uses Playwright directly for deterministic interaction (no LLM).
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(1500)

            # Find input field
            input_el = None
            for selector in [
                'input[type="search"]',
                'input[type="text"]',
                "textarea",
                "input:not([type])",
                '[role="searchbox"]',
                '[role="textbox"]',
            ]:
                try:
                    el = page.locator(selector).first
                    if await el.is_visible(timeout=1000):
                        input_el = el
                        break
                except Exception:
                    continue

            if not input_el:
                return ""

            await input_el.clear()
            await input_el.fill(seed_text)

            # Try submit button, then Enter
            submitted = False
            for selector in [
                'button[type="submit"]',
                "form button",
            ]:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=1000):
                        await btn.click()
                        submitted = True
                        break
                except Exception:
                    continue

            if not submitted:
                await input_el.press("Enter")

            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            await page.wait_for_timeout(2000)

            text = await page.evaluate("() => document.body.innerText")
            return text[:8000]
        except Exception as e:
            logger.warning("Seed input interaction failed: %s", e)
            return ""
        finally:
            await browser.close()
