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
2. Mobile viewport (390x844, iPhone)

IMPORTANT: Base ALL findings on what you literally see in these screenshots.
Focus primarily on the MOBILE screenshot. Most users are on phones.

═══════════════════════════════════════════════════════
MOBILE-SPECIFIC CHECKS (highest priority)
═══════════════════════════════════════════════════════

A. CONTENT HIDDEN BY HEADERS: Look at the TOP of the mobile screenshot. Is any
   page content (text, cards, list items) partially hidden behind a fixed navigation
   bar, search bar, or sticky header? If the first visible content item appears to
   start mid-sentence or mid-element, this is a cut-off bug.

B. OVERLAPPING ELEMENTS: Are there any buttons, panels, filters, or UI components
   that visually overlap each other? Elements should never stack on top of each
   other unless it's an intentional modal with a backdrop.

C. PANELS WITHOUT CLOSE BUTTONS: Are there any open panels, sidebars, filter
   drawers, or overlays that have no visible close button (X) and no obvious way
   to dismiss them?

D. ZOOM/SCALE ISSUES: Does the initial view show an appropriate amount of content?
   If a map or content area appears extremely zoomed in with very little visible,
   the default zoom is wrong for mobile.

E. TOUCH TARGET OVERLAP: Are any clickable elements so close together that a finger
   tap would likely hit the wrong one?

F. HORIZONTAL OVERFLOW: Can you see a horizontal scrollbar or content extending
   beyond the right edge of the 390px viewport?

═══════════════════════════════════════════════════════
GENERAL RESPONSIVE CHECKS
═══════════════════════════════════════════════════════

1. **Layout breaks**: Content that overflows, overlaps, or becomes unreadable on mobile
2. **Touch targets**: Buttons/links that appear too small for finger taps on mobile
3. **Text sizing**: Text too small to read on mobile
4. **Navigation**: Desktop nav that doesn't adapt to mobile (no hamburger/drawer)
5. **Image scaling**: Images that don't scale down or break layout
6. **Form usability**: Inputs too small, dropdowns unusable on touch
7. **Spacing**: Padding/margins that don't adapt between viewports

You MUST cite specific evidence for every finding by referencing what you see
in the desktop vs mobile screenshots. Every mobile layout problem (A-F above)
is at least "high" severity.

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
The second image is the MOBILE screenshot (390x844 viewport, iPhone).

FOCUS ON THE MOBILE SCREENSHOT FIRST. Compare with desktop to identify issues.

Check these specific mobile problems (most common bugs):
1. Content hidden behind fixed headers/nav bars — look at the TOP of mobile
2. Overlapping elements — panels, filters, buttons stacking on each other
3. Panels/overlays with no close button or dismiss mechanism
4. Map or content areas too zoomed in for initial mobile view
5. Touch targets too close together for finger taps
6. Horizontal overflow beyond the 390px viewport edge
7. Navigation that doesn't adapt (full desktop nav crammed into mobile)
8. Text too small to read without zooming

For each issue, describe EXACTLY where on the mobile screen it appears.
Rate overall responsive quality 0.0-1.0."""


MOBILE_DETAIL_PROMPT = """Look at this mobile screenshot (390px wide, iPhone).

Check these specific issues — they are the most common mobile bugs:

A. CONTENT HIDDEN BY HEADERS: Is any page content (text, cards, list items) partially hidden behind a fixed navigation bar, search bar, or sticky header at the top of the screen? If the first visible content item appears to start mid-sentence or mid-element, this is a cut-off bug.

B. OVERLAPPING ELEMENTS: Are there any buttons, panels, filters, or UI components that visually overlap each other? Elements should never stack on top of each other unless it's an intentional modal.

C. PANELS WITHOUT CLOSE BUTTONS: Are there any open panels, sidebars, filter drawers, or overlays that have no visible close button (X) and no obvious way to dismiss them?

D. ZOOM/SCALE ISSUES: Does the initial view show an appropriate amount of content? If a map or content area appears extremely zoomed in with very little visible, the default zoom is wrong for mobile.

E. TOUCH TARGET OVERLAP: Are any clickable elements so close together that a finger tap would likely hit the wrong one?

F. HORIZONTAL OVERFLOW: Can you see a horizontal scrollbar or content extending beyond the right edge?

For each issue found, describe EXACTLY where on the screen it is (top-left, center, behind the nav bar, etc.) and what element is affected.

Respond with JSON: {"mobile_issues": [{"title": "...", "severity": "critical|high|medium|low", "confidence": 0.0-1.0, "user_impact": "...", "observed_facts": ["..."], "inferred_judgment": "...", "repair_brief": "...", "affects_viewport": "mobile"}], "mobile_score": 0.0-1.0}"""


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

        # Dedicated mobile detail check — second call on mobile screenshot only
        if mobile_bytes:
            try:
                mobile_data = self.llm.complete_json_with_vision(
                    MOBILE_DETAIL_PROMPT,
                    images=[(mobile_bytes, "image/png")],
                )
                mobile_extra = mobile_data.get("mobile_issues", [])
                if mobile_extra:
                    existing_titles = {r.get("title", "") for r in data.get("issues", [])}
                    for raw in mobile_extra:
                        if raw.get("title", "") not in existing_titles:
                            data.setdefault("issues", []).append(raw)
                    logger.info(
                        "Mobile detail check added %d issues to responsive lens",
                        len(mobile_extra),
                    )
            except Exception as e:
                logger.warning("Responsive lens mobile detail check failed: %s", e)

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
