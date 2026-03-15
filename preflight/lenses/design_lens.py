"""Design Lens — Specialist design critique.

Evaluates UI/design quality from captured screenshots and page content.
Assesses hierarchy, spacing, readability, CTA prominence, visual polish,
consistency, and brand coherence. No code inspection.
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

DESIGN_REVIEW_SYSTEM = """You are a senior product designer conducting a design review for Preflight.

You evaluate products from their visible UI only — you are looking at actual screenshots.
You never reference source code.

IMPORTANT: You are receiving actual screenshot images. Base ALL findings on what you
literally see in these images. Do not invent observations — only report what is visible.

Assess across these dimensions:
1. Alignment — Are elements aligned to a consistent grid? Look for misaligned text, buttons, cards.
2. Spacing — Is whitespace consistent? Are there areas that are too cramped or too sparse?
3. Sizing — Are elements appropriately sized? Look for oversized/undersized text, buttons, images.
4. Cut-off content — Is any text, image, or UI element clipped, truncated, or overflowing its container?
5. Visual hierarchy — Is it clear what's most important on each screen?
6. Readability — Is text legible, well-sized, properly contrasted?
7. CTA prominence — Are calls to action obvious and well-placed?
8. Visual polish — Does it look professional or rough/unfinished?
9. Consistency — Are patterns, colors, typography consistent across screens?
10. Brand coherence — Does the visual language match the product's positioning?

## EVIDENCE ANCHORING (MANDATORY)

Every finding MUST cite specific evidence. Findings without anchored evidence will be rejected.

Each finding must reference at least ONE of:
- **Screenshot reference**: "In screenshot {filename}, [specific observation]"
- **Element reference**: A specific UI element with measurable detail
  Example: "The submit button has 8px padding on a 390px viewport"
- **Measurement**: A quantifiable observation
  Example: "Heading text is 12px on desktop, below readable minimum of 16px"
- **Observed absence**: An explicit negative observation
  Example: "No hover state visible on any interactive element in the navigation bar"

Bad (will be rejected):
- "The design feels cluttered" (no specific element)
- "Colors don't work well" (no specific reference)

Good:
- "In screenshot hero-page.png, the primary CTA 'Get Started' uses the same visual weight as secondary links, reducing its prominence"
- "The navigation bar has 6 items plus a dropdown, but on the 390px mobile viewport they overflow without a hamburger menu"

Severity scale:
- critical: unusable (blocks core function)
- high: significantly hurts experience
- medium: noticeable quality issue
- low: polish item
- info: suggestion

Respond with JSON: {"design_issues": [...], "design_strengths": [...], "overall_assessment": "..."}"""

DESIGN_REVIEW_PROMPT = """Review the design quality of this product.

## Product
{product_name} ({product_type})
Target audience: {target_audience}

## Screenshots provided
You are receiving desktop screenshots (1440x900) AND mobile screenshots (390x844).
Evaluate both viewports for design quality. Note any design issues that appear
only on mobile or are worse on mobile.

{screenshot_list}

## Page descriptions from evaluation
{page_descriptions}

## Design guidance (if provided)
{design_guidance}

For each issue found, provide:
- title: Clear design issue title
- severity: critical | high | medium | low | info
- confidence: 0.0-1.0
- user_impact: How this affects real users
- observed_facts: What you literally see (list)
- inferred_judgment: Your design assessment
- likely_product_area: Where in the product
- repair_brief: What to fix

Also provide:
- design_strengths: What's working well (list of strings)
- overall_assessment: 2-3 sentence summary of design quality"""


class DesignLens:
    """Specialist design review from captured artifacts."""

    def __init__(self, llm: LLMClient, output_dir: str = "./artifacts"):
        self.llm = llm
        self.output_dir = Path(output_dir)

    async def review(
        self,
        run_result: RunResult,
        design_guidance: str | None = None,
    ) -> list[Issue]:
        """Run design review on collected artifacts from a run."""
        intent = run_result.intent_model

        # Gather screenshot references and load actual image files
        screenshot_names = []
        for issue in run_result.issues:
            screenshot_names.extend(issue.evidence.screenshots)
        # Also check artifact dir for additional screenshots
        if self.output_dir.exists():
            for f in self.output_dir.glob("*.png"):
                if f.name not in screenshot_names:
                    screenshot_names.append(f.name)

        # Load actual screenshot files for vision evaluation (desktop)
        images: list[tuple[bytes, str]] = []
        for name in screenshot_names[:8]:  # Cap to manage token budget
            path = self.output_dir / name
            if path.exists():
                try:
                    images.append((path.read_bytes(), "image/png"))
                except Exception as e:
                    logger.debug("Could not read screenshot %s: %s", name, e)

        # Capture mobile viewport screenshots for design evaluation
        target_url = run_result.config.target_url
        mobile_bytes = await self._capture_mobile_screenshot(target_url)
        if mobile_bytes:
            images.append((mobile_bytes, "image/png"))
            screenshot_names.append("design-mobile-390x844.png")

        # Build page descriptions from coverage and issues
        page_descs = []
        for entry in run_result.coverage.entries:
            page_descs.append(
                f"- {entry.screen_name or entry.url} (status: {entry.status}, "
                f"issues: {entry.issues_found})"
            )

        prompt = DESIGN_REVIEW_PROMPT.format(
            product_name=intent.product_name,
            product_type=intent.product_type,
            target_audience=", ".join(intent.target_audience),
            screenshot_list="\n".join(f"- {s}" for s in screenshot_names[:20]) or "(none captured)",
            page_descriptions="\n".join(page_descs[:30]) or "(none)",
            design_guidance=design_guidance or "(none provided)",
        )

        try:
            if images:
                data = self.llm.complete_json_with_vision(
                    prompt, images=images, system=DESIGN_REVIEW_SYSTEM,
                )
            else:
                logger.warning("No screenshot images available — falling back to text-only design review")
                data = self.llm.complete_json(prompt, system=DESIGN_REVIEW_SYSTEM)

            design_issues = []
            for raw in data.get("design_issues", []):
                sev = raw.get("severity", "medium")
                try:
                    severity = Severity(sev)
                except ValueError:
                    severity = Severity.medium

                design_issues.append(Issue(
                    title=raw.get("title", "Design issue"),
                    severity=severity,
                    confidence=raw.get("confidence", 0.7),
                    platform=Platform.web,
                    category=IssueCategory.design,
                    agent="design_lens",
                    user_impact=raw.get("user_impact", ""),
                    observed_facts=raw.get("observed_facts", []),
                    inferred_judgment=raw.get("inferred_judgment", ""),
                    likely_product_area=raw.get("likely_product_area", ""),
                    repair_brief=raw.get("repair_brief", ""),
                ))

            logger.info(
                "Design review complete: %d issues, %d strengths noted",
                len(design_issues),
                len(data.get("design_strengths", [])),
            )
            return design_issues

        except Exception as e:
            logger.error("Design review failed: %s", e)
            return []

    async def _capture_mobile_screenshot(self, url: str) -> bytes | None:
        """Capture a screenshot at mobile viewport (390x844) for design evaluation."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 390, "height": 844},
                    user_agent=(
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                        "Mobile/15E148 Safari/604.1"
                    ),
                )
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(2000)

                screenshot_bytes = await page.screenshot(full_page=True, timeout=10000)

                # Save to disk
                self.output_dir.mkdir(parents=True, exist_ok=True)
                out_path = self.output_dir / "design-mobile-390x844.png"
                out_path.write_bytes(screenshot_bytes)

                await browser.close()
                return screenshot_bytes

        except Exception as e:
            logger.warning("Failed to capture mobile screenshot for design review: %s", e)
            return None
