"""Responsive/Mobile Layout Lens — evaluates mobile responsiveness and visual layout.

Mandatory vision-based evaluation for mobile viewports:
- Desktop-vs-mobile comparison (same pages, different viewports)
- Touch target sizing
- Text readability at mobile scale
- Horizontal overflow / scroll issues
- Navigation adaptation (hamburger menu, etc.)

This lens independently captures screenshots at desktop (1440x900) and mobile
(390x844) viewports using Playwright, then sends both to the LLM via
complete_json_with_vision for side-by-side visual comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import async_playwright

from preflight.core.llm import LLMClient
from preflight.core.schemas import (
    Issue,
    IssueCategory,
    Platform,
    RunResult,
    Severity,
)

logger = logging.getLogger(__name__)

RESPONSIVE_SYSTEM_PROMPT = """You are a mobile responsiveness evaluator for Preflight.

You are receiving TWO screenshots of the same page:
1. Desktop viewport (1440x900)
2. Mobile viewport (390x844)

IMPORTANT: Base ALL findings on what you literally see in these screenshots.
Compare the two images side by side to identify responsive design issues.

Focus on:
1. **Layout breaks**: Content that overflows, overlaps, or becomes unreadable on mobile
2. **Touch targets**: Buttons/links that appear too small for finger taps on mobile
3. **Text sizing**: Text too small to read on mobile
4. **Navigation**: Desktop nav that doesn't adapt to mobile (no hamburger/drawer)
5. **Horizontal scroll**: Content wider than the mobile viewport
6. **Image scaling**: Images that don't scale down or break layout
7. **Form usability**: Inputs too small, dropdowns unusable on touch
8. **Cut-off content**: Elements clipped or hidden on mobile that are visible on desktop
9. **Spacing**: Padding/margins that don't adapt between viewports

You MUST cite specific evidence for every finding by referencing what you see
in the desktop vs mobile screenshots.

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

RESPONSIVE_EVAL_PROMPT = """Evaluate mobile responsiveness by comparing the desktop and mobile screenshots.

## Product: {product_name} ({product_type})
## Target URL: {target_url}

The first image is the DESKTOP screenshot (1440x900 viewport).
The second image is the MOBILE screenshot (390x844 viewport).

Compare them and identify:
1. Layout elements that break or overflow on mobile
2. Text that becomes unreadable on mobile
3. Navigation that doesn't adapt (missing hamburger menu, etc.)
4. Touch targets that are too small on mobile
5. Content that is cut off or hidden on mobile
6. Spacing/padding issues between viewports

Rate overall responsive quality 0.0-1.0."""


class ResponsiveLens:
    """Evaluates mobile responsiveness by capturing and comparing viewport screenshots."""

    def __init__(self, llm: LLMClient, output_dir: str = "./artifacts"):
        self.llm = llm
        self.output_dir = Path(output_dir)

    async def review(self, result: RunResult) -> tuple[list[Issue], float]:
        """Capture desktop and mobile screenshots, then compare via vision LLM.

        Returns (issues, responsive_score).
        """
        target_url = result.config.target_url

        # Capture screenshots at both viewports
        desktop_bytes = await self._capture_viewport(
            url=target_url,
            width=1440,
            height=900,
            label="desktop",
        )
        mobile_bytes = await self._capture_viewport(
            url=target_url,
            width=390,
            height=844,
            label="mobile",
            mobile_ua=True,
        )

        if not desktop_bytes and not mobile_bytes:
            logger.warning("Could not capture any screenshots for responsive review")
            return [], 0.5

        # Build image list for vision
        images: list[tuple[bytes, str]] = []
        if desktop_bytes:
            images.append((desktop_bytes, "image/png"))
        if mobile_bytes:
            images.append((mobile_bytes, "image/png"))

        prompt = RESPONSIVE_EVAL_PROMPT.format(
            product_name=result.intent_model.product_name,
            product_type=result.intent_model.product_type,
            target_url=target_url,
        )

        try:
            data = self.llm.complete_json_with_vision(
                prompt, images=images, system=RESPONSIVE_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.warning("Responsive lens vision evaluation failed: %s", e)
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

        logger.info(
            "Responsive lens: %d issues, score=%.1f",
            len(issues), responsive_score,
        )
        return issues, responsive_score

    async def _capture_viewport(
        self,
        url: str,
        width: int,
        height: int,
        label: str,
        mobile_ua: bool = False,
    ) -> bytes | None:
        """Capture a screenshot at the specified viewport size."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)

                context_kwargs: dict = {"viewport": {"width": width, "height": height}}
                if mobile_ua:
                    context_kwargs["user_agent"] = (
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                        "Mobile/15E148 Safari/604.1"
                    )

                context = await browser.new_context(**context_kwargs)
                page = await context.new_page()

                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(2000)  # Let JS render

                screenshot_bytes = await page.screenshot(full_page=True, timeout=10000)

                # Save to disk for reference
                self.output_dir.mkdir(parents=True, exist_ok=True)
                out_path = self.output_dir / f"responsive-{label}-{width}x{height}.png"
                out_path.write_bytes(screenshot_bytes)

                await browser.close()
                return screenshot_bytes

        except Exception as e:
            logger.warning("Failed to capture %s viewport (%dx%d): %s", label, width, height, e)
            return None
