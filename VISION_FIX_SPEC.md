# CRITICAL FIX: Design and Mobile Evaluation Must Use Vision

## Problem

Design lens and responsive lens are fundamentally broken — they evaluate visual quality WITHOUT looking at screenshots. They use `complete_json` (text only) instead of `complete_json_with_vision`. This is like asking someone to review a website's design while blindfolded.

Additionally, mobile evaluation only happens IF the LLM generates a persona with `mobile_web` preference, which is unreliable. There's no guarantee any mobile testing occurs.

## Fix 1: Design Lens Must Use Vision

File: `preflight/lenses/design_lens.py`

**Current:** Calls `self.llm.complete_json(prompt, ...)` — text only, no screenshots sent.

**Fix:** 
1. Collect ALL screenshot files from the artifacts directory
2. Load the most important ones (landing page, key screens — up to 5 images to stay within token limits)
3. Call `self.llm.complete_json_with_vision(prompt, images=[(bytes, "image/png"), ...], ...)` 
4. The prompt must explicitly ask about visual design: alignment, spacing, sizing, hierarchy, cut-off elements, consistency, color, typography, CTA prominence, visual polish

**Updated design review prompt must include:**
```
You are looking at actual screenshots of this product. Evaluate the VISUAL design:

1. ALIGNMENT: Are elements properly aligned? Any misaligned text, buttons, images, or cards?
2. SPACING: Is spacing consistent? Any elements too close together or too far apart?
3. SIZING: Are elements appropriately sized? Any oversized or undersized components?
4. VISUAL HIERARCHY: Is it clear what's most important on each page?
5. CUT-OFF CONTENT: Is any text, image, or component cut off or partially hidden?
6. CONSISTENCY: Are similar elements styled consistently across screens?
7. TYPOGRAPHY: Is text readable? Appropriate font sizes and weights?
8. COLOR: Is the color palette cohesive? Sufficient contrast for readability?
9. COMPONENT STATES: Do buttons, inputs, and interactive elements look properly styled?
10. OVERALL POLISH: Does this look professional or rough/unfinished?

For each issue, describe EXACTLY where on the screen the problem is (top-left, center, below the hero section, etc.) and what specifically is wrong.
```

## Fix 2: Responsive Lens Must Take Its Own Screenshots

File: `preflight/lenses/responsive_lens.py`

**Current:** Compares text descriptions of desktop vs mobile issues from other agents. Never captures its own screenshots. If no mobile agent ran, it has nothing to compare.

**Fix:**
1. The responsive lens must INDEPENDENTLY capture screenshots at both viewports:
   - Desktop: 1440x900
   - Mobile: 390x844
2. For the target URL and up to 3 key pages from coverage, capture BOTH viewports
3. Send desktop+mobile screenshot pairs to the LLM via vision in a single call
4. Ask the LLM to compare side-by-side and identify responsive breakages

**Implementation:**
```python
async def review(self, result: RunResult) -> tuple[list[Issue], float]:
    """Capture screenshots at both viewports and compare via vision."""
    urls_to_check = self._get_key_urls(result)  # Target URL + top pages from coverage
    
    all_issues = []
    for url in urls_to_check[:4]:  # Cap at 4 pages
        desktop_screenshot = await self._capture_screenshot(url, width=1440, height=900)
        mobile_screenshot = await self._capture_screenshot(url, width=390, height=844)
        
        issues = await self._compare_viewports(
            url, desktop_screenshot, mobile_screenshot, result.intent_model
        )
        all_issues.extend(issues)
    
    return all_issues, self._calculate_score(all_issues)

async def _capture_screenshot(self, url: str, width: int, height: int) -> bytes:
    """Capture a screenshot at a specific viewport size."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": width, "height": height})
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(2000)
        screenshot = await page.screenshot(full_page=True)
        await browser.close()
        return screenshot

async def _compare_viewports(self, url, desktop_bytes, mobile_bytes, intent):
    """Send both screenshots to LLM and ask for responsive comparison."""
    prompt = f"""Compare these two screenshots of the same page at {url}.
    
LEFT IMAGE: Desktop viewport (1440x900)
RIGHT IMAGE: Mobile viewport (390x844)

Product: {intent.product_name} ({intent.product_type})

Identify ALL responsive design problems:
1. Content visible on desktop but CUT OFF, HIDDEN, or MISSING on mobile
2. Text that TRUNCATES or OVERFLOWS its container on mobile
3. Elements that OVERLAP or MISALIGN on mobile
4. Touch targets too small (buttons/links < 44px) on mobile
5. HORIZONTAL SCROLLING on mobile (content wider than 390px viewport)
6. Navigation that doesn't adapt (no hamburger/drawer menu)
7. Images that don't resize or break layout on mobile
8. Forms that are difficult to use on mobile
9. Critical information pushed below the fold on mobile
10. Any visual element that looks BROKEN on mobile but fine on desktop

Be specific about WHERE on the screen each issue occurs.
Respond with JSON: {{"issues": [...], "responsive_score": 0.0-1.0}}"""

    return self.llm.complete_json_with_vision(
        prompt,
        images=[(desktop_bytes, "image/png"), (mobile_bytes, "image/png")],
        system=RESPONSIVE_SYSTEM_PROMPT,
    )
```

## Fix 3: Guarantee At Least One Mobile Persona

File: `preflight/core/persona_generator.py`

After generating personas, check if ANY have `device_preference == Platform.mobile_web`. If none do, force the last persona to be a mobile user:

```python
# After persona generation:
has_mobile = any(p.device_preference == Platform.mobile_web for p in personas)
if not has_mobile and len(personas) > 1:
    # Convert last persona to mobile
    personas[-1].device_preference = Platform.mobile_web
    personas[-1].behavioral_style += " (mobile user — tests on phone)"
    logger.info("Forced mobile persona: %s", personas[-1].name)
```

## Fix 4: Design Lens Must Also Check Mobile

The design lens should capture and evaluate BOTH desktop and mobile screenshots separately. Mobile design issues are often completely different from desktop ones.

In the design lens review method:
1. Capture desktop screenshots (1440x900) of key pages
2. Capture mobile screenshots (390x844) of the same pages
3. Send BOTH sets to the LLM
4. Ask specifically: "Are there design issues that ONLY appear on mobile?"

## Build Order

1. Update `design_lens.py`: load actual screenshots, use `complete_json_with_vision`
2. Update `responsive_lens.py`: capture own screenshots at both viewports, compare via vision
3. Update `persona_generator.py`: guarantee at least one mobile persona
4. Update design lens to also check mobile viewport
5. Tests
6. Push

## Key Constraint

Every visual evaluation MUST use `complete_json_with_vision` with actual screenshot bytes. Using `complete_json` (text-only) for any design, layout, or responsive evaluation is a bug.
