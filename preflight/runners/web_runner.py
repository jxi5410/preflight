"""Web Runner — Playwright-based external UI evaluation.

Evaluates a product through real browser interaction using:
- Accessibility tree snapshots for semantic page understanding
- Vision-based evaluation (screenshots sent to LLM)
- Deterministic action execution (LLM plans, Playwright executes)
- Multi-step journey execution with plan-execute-judge-adapt loop
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from playwright.async_api import async_playwright, Page, BrowserContext

from preflight.core.actions import (
    ACTION_PLAN_PROMPT_TEMPLATE,
    ACTION_PLAN_SYSTEM_PROMPT,
    execute_action,
)
from preflight.core.llm import LLMClient
from preflight.core.schemas import (
    Action,
    AgentPersona,
    CoverageEntry,
    CoverageMap,
    Evidence,
    Issue,
    IssueCategory,
    JourneyStep,
    PageSnapshot,
    Platform,
    RunConfig,
    ScreenshotEvidence,
    Severity,
)
from preflight.runners.page_snapshot import capture_snapshot, snapshot_to_prompt_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVALUATION_SYSTEM_PROMPT = """You are a QA evaluation agent for Preflight. You are acting as a specific user persona
interacting with a product through its UI.

You will receive:
- A screenshot of the current page (as an image)
- The accessibility tree (semantic page structure)
- Performance metrics and console errors
- Your persona details and journey context

Your job:
1. Evaluate what you see from your persona's perspective
2. Identify real issues with specific, anchored evidence
3. Separate observed facts from inferred judgments

## EVIDENCE ANCHORING (MANDATORY)

You MUST cite evidence for every finding. A finding without anchored evidence will be rejected.

Every finding must reference at least ONE of:
- **Screenshot reference**: "In screenshot {step_ref}, [description of what is visible]"
- **Element reference**: An element by accessible name/role from the accessibility tree
  Example: "The element [role=button, name='Submit'] has no visible disabled state"
- **Measurement**: A specific metric from the page state
  Example: "Page load took {load_time_ms}ms (measured), exceeding the 3000ms budget"
- **Observed absence**: An explicit negative observation
  Example: "No element with role 'navigation' or label containing 'menu' was found on this page"

Bad (will be rejected):
- "The page feels slow" (no measurement)
- "Navigation is confusing" (no specific element or screenshot reference)
- "There might be accessibility issues" (no specific element cited)

Good:
- "In screenshot step-3, the submit button [role=button, name='Submit'] has 8px padding, making it a small touch target on mobile"
- "Page load took 4200ms (measured), exceeding reasonable threshold for a SaaS app"
- "No element with role 'navigation' or label containing 'audit' was found on this page"
- "Console error: 'Failed to fetch /api/data' — the data table shows a loading spinner indefinitely"

## Response format

For each issue found, provide:
- title: Clear issue title
- severity: critical | high | medium | low | info
- confidence: 0.0-1.0
- category: functional | ux | ui | performance | trust | design | accessibility | auth | responsive
- user_impact: How this affects a real user
- observed_facts: What you literally see/measure — MUST include evidence anchor (list)
- inferred_judgment: What you conclude from the evidence
- hypotheses: Possible explanations (list)
- likely_product_area: Where in the product this lives
- repair_brief: What a developer should fix
- evidence_ref: Screenshot step reference (e.g. "step-3")

Respond with JSON: {"issues": [...], "persona_reaction": "...", "confidence_level": 0.0-1.0}"""

EVALUATION_PROMPT_TEMPLATE = """## Persona
Name: {persona_name} | Role: {persona_role}
Goals: {persona_goals}
Patience: {patience_level} | Expertise: {expertise_level}
Style: {behavioral_style}

## Journey: {journey}
## Step {step_number} of max {max_steps}

## Current Page State
{page_context}

## Action Just Taken
{action_description}

## Previous Actions
{previous_actions}

Evaluate this page from your persona's perspective. The screenshot is attached as an image.
Find issues with specific evidence. Respond with JSON."""

# Default max steps per journey (exploration cap)
DEFAULT_MAX_STEPS = 8

# Timeout for individual page navigation
PAGE_NAVIGATION_TIMEOUT_MS = 20000

# Timeout for individual action execution
ACTION_TIMEOUT_MS = 10000


class WebRunner:
    """Runs web-based evaluation using Playwright with vision + a11y + deterministic actions."""

    def __init__(self, llm: LLMClient, output_dir: str = "./artifacts"):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        config: RunConfig,
        persona: AgentPersona,
        journeys: list[str],
        coverage: CoverageMap,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> tuple[list[Issue], CoverageMap]:
        """Run evaluation for a single persona across assigned journeys."""
        all_issues: list[Issue] = []

        async with async_playwright() as p:
            # Choose viewport based on persona device preference
            if persona.device_preference == Platform.mobile_web:
                viewport = {"width": 390, "height": 844}
                user_agent = (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                    "Mobile/15E148 Safari/604.1"
                )
            else:
                viewport = {"width": 1440, "height": 900}
                user_agent = None

            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport=viewport,
                user_agent=user_agent,
            )

            # Enable performance observer for LCP/CLS
            await context.add_init_script("""
                window.__preflight_perf = {errors: [], networkErrors: 0};
                if (typeof PerformanceObserver !== 'undefined') {
                    new PerformanceObserver((list) => {}).observe({type: 'largest-contentful-paint', buffered: true});
                    new PerformanceObserver((list) => {}).observe({type: 'layout-shift', buffered: true});
                }
            """)

            page = await context.new_page()

            # Collect console errors
            console_errors: list[str] = []
            page.on("console", lambda msg: (
                console_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type in ("error", "warning") else None
            ))

            # Track network errors
            network_error_count = 0

            def on_response(response):
                nonlocal network_error_count
                if response.status >= 400:
                    network_error_count += 1

            page.on("response", on_response)

            try:
                # Navigate to target
                start = time.monotonic()
                await page.goto(
                    config.target_url,
                    wait_until="domcontentloaded",
                    timeout=PAGE_NAVIGATION_TIMEOUT_MS,
                )
                load_time_ms = int((time.monotonic() - start) * 1000)

                # Handle credentials if provided
                if config.credentials:
                    await self._attempt_login(page, config)

                # Execute each journey with multi-step loop
                for journey in journeys:
                    logger.info(
                        "  Agent %s starting journey: %s",
                        persona.name, journey,
                    )
                    journey_issues, steps = await self._execute_journey(
                        page=page,
                        persona=persona,
                        journey=journey,
                        config=config,
                        console_errors=console_errors,
                        network_error_count=network_error_count,
                        initial_load_time_ms=load_time_ms,
                        max_steps=max_steps,
                    )
                    all_issues.extend(journey_issues)

                    coverage.entries.append(CoverageEntry(
                        url=page.url,
                        screen_name=await page.title(),
                        agent_id=persona.id,
                        flow=journey,
                        status="visited",
                        issues_found=len(journey_issues),
                    ))

                    logger.info(
                        "  Journey '%s' complete: %d steps, %d issues",
                        journey, len(steps), len(journey_issues),
                    )

                    # Navigate back to start for next journey
                    if len(journeys) > 1:
                        try:
                            await page.goto(
                                config.target_url,
                                wait_until="domcontentloaded",
                                timeout=15000,
                            )
                        except Exception:
                            pass

            except Exception as e:
                logger.error("Web evaluation failed for %s: %s", persona.name, e)
                all_issues.append(Issue(
                    title=f"Evaluation blocked: {str(e)[:100]}",
                    severity=Severity.critical,
                    category=IssueCategory.functional,
                    agent=persona.id,
                    user_impact="Could not complete evaluation — product may be unreachable or broken",
                    observed_facts=[str(e)],
                    platform=Platform.web,
                ))
            finally:
                await browser.close()

        return all_issues, coverage

    async def _execute_journey(
        self,
        page: Page,
        persona: AgentPersona,
        journey: str,
        config: RunConfig,
        console_errors: list[str],
        network_error_count: int,
        initial_load_time_ms: int,
        max_steps: int,
    ) -> tuple[list[Issue], list[JourneyStep]]:
        """Execute a multi-step journey using plan-execute-judge-adapt loop."""
        issues: list[Issue] = []
        steps: list[JourneyStep] = []
        previous_actions: list[str] = [f"Navigated to {config.target_url}"]

        for step_num in range(1, max_steps + 1):
            # 1. CAPTURE: Take a snapshot of the current page
            snapshot = await capture_snapshot(
                page=page,
                output_dir=self.output_dir,
                snapshot_name=f"{persona.id}-{journey[:20]}-step{step_num:02d}",
                console_errors=console_errors,
                network_error_count=network_error_count,
                load_time_ms=initial_load_time_ms if step_num == 1 else 0,
            )

            # 2. PLAN: Ask LLM to plan next action(s)
            plan = await self._plan_actions(
                snapshot=snapshot,
                persona=persona,
                journey=journey,
                step_number=step_num,
                max_steps=max_steps,
                previous_actions=previous_actions,
            )

            actions = plan.get("actions", [])
            journey_complete = plan.get("journey_complete", False)

            if journey_complete or not actions:
                # Journey is done — do one final evaluation
                step_issues = await self._judge_snapshot(
                    snapshot=snapshot,
                    persona=persona,
                    journey=journey,
                    step_number=step_num,
                    max_steps=max_steps,
                    action_description="Journey complete — final evaluation",
                    previous_actions=previous_actions,
                )
                issues.extend(step_issues)
                steps.append(JourneyStep(
                    step_number=step_num,
                    action=Action(type="screenshot", reason="Final evaluation"),
                    snapshot_before=None,
                    snapshot_after=snapshot,
                    screenshot_path=snapshot.screenshot_path,
                    issues_found=[i.id for i in step_issues],
                    persona_reaction=plan.get("persona_reaction", ""),
                    confidence_level=plan.get("confidence_level", 0.5),
                ))
                break

            # 3. EXECUTE each planned action
            for action_data in actions:
                action = Action(
                    type=action_data.get("type", "screenshot"),
                    target=action_data.get("target", ""),
                    value=action_data.get("value"),
                    reason=action_data.get("reason", ""),
                )

                snapshot_before = snapshot
                success = await execute_action(page, action)

                action_desc = (
                    f"{action.type}: {action.target}"
                    + (f" = {action.value}" if action.value else "")
                    + (f" ({'ok' if success else 'FAILED'})")
                )
                previous_actions.append(action_desc)

                # Capture snapshot after action
                snapshot = await capture_snapshot(
                    page=page,
                    output_dir=self.output_dir,
                    snapshot_name=f"{persona.id}-{journey[:20]}-step{step_num:02d}-after",
                    console_errors=console_errors,
                    network_error_count=network_error_count,
                )

                # 4. JUDGE: Evaluate the new state with vision
                step_issues = await self._judge_snapshot(
                    snapshot=snapshot,
                    persona=persona,
                    journey=journey,
                    step_number=step_num,
                    max_steps=max_steps,
                    action_description=action_desc,
                    previous_actions=previous_actions,
                )
                issues.extend(step_issues)

                steps.append(JourneyStep(
                    step_number=step_num,
                    action=action,
                    snapshot_before=snapshot_before,
                    snapshot_after=snapshot,
                    screenshot_path=snapshot.screenshot_path,
                    issues_found=[i.id for i in step_issues],
                    persona_reaction=plan.get("persona_reaction", ""),
                    confidence_level=plan.get("confidence_level", 0.5),
                ))

        return issues, steps

    async def _plan_actions(
        self,
        snapshot: PageSnapshot,
        persona: AgentPersona,
        journey: str,
        step_number: int,
        max_steps: int,
        previous_actions: list[str],
    ) -> dict:
        """Ask LLM to plan the next action(s) based on current page state."""
        prompt = ACTION_PLAN_PROMPT_TEMPLATE.format(
            persona_name=persona.name,
            persona_role=persona.role,
            persona_goals=", ".join(persona.goals),
            patience_level=persona.patience_level,
            expertise_level=persona.expertise_level,
            journey=journey,
            url=snapshot.url,
            title=snapshot.title,
            accessibility_tree=snapshot.accessibility_tree[:5000],
            previous_actions="\n".join(previous_actions[-10:]),
            step_number=step_number,
            max_steps=max_steps,
        )

        try:
            # Use vision if we have a screenshot
            if snapshot.screenshot_base64:
                screenshot_bytes = base64.b64decode(snapshot.screenshot_base64)
                return self.llm.complete_json_with_vision(
                    prompt,
                    images=[(screenshot_bytes, "image/png")],
                    system=ACTION_PLAN_SYSTEM_PROMPT,
                )
            else:
                return self.llm.complete_json(prompt, system=ACTION_PLAN_SYSTEM_PROMPT)
        except Exception as e:
            logger.warning("Action planning failed: %s", e)
            return {"actions": [], "journey_complete": True, "persona_reaction": "Planning failed"}

    async def _judge_snapshot(
        self,
        snapshot: PageSnapshot,
        persona: AgentPersona,
        journey: str,
        step_number: int,
        max_steps: int,
        action_description: str,
        previous_actions: list[str],
    ) -> list[Issue]:
        """Evaluate a page snapshot from the persona's perspective using vision."""
        page_context = snapshot_to_prompt_context(snapshot)

        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            persona_name=persona.name,
            persona_role=persona.role,
            persona_goals=", ".join(persona.goals),
            patience_level=persona.patience_level,
            expertise_level=persona.expertise_level,
            behavioral_style=persona.behavioral_style or "standard",
            journey=journey,
            step_number=step_number,
            max_steps=max_steps,
            page_context=page_context,
            action_description=action_description,
            previous_actions="\n".join(previous_actions[-10:]),
        )

        try:
            # Send screenshot as vision input alongside text context
            if snapshot.screenshot_base64:
                screenshot_bytes = base64.b64decode(snapshot.screenshot_base64)
                data = self.llm.complete_json_with_vision(
                    prompt,
                    images=[(screenshot_bytes, "image/png")],
                    system=EVALUATION_SYSTEM_PROMPT,
                )
            else:
                data = self.llm.complete_json(prompt, system=EVALUATION_SYSTEM_PROMPT)

            return self._parse_issues(
                data, persona, snapshot, step_number,
            )
        except Exception as e:
            logger.error("Page evaluation failed: %s", e)
            return []

    def _parse_issues(
        self,
        data: dict,
        persona: AgentPersona,
        snapshot: PageSnapshot,
        step_number: int,
    ) -> list[Issue]:
        """Parse LLM evaluation response into Issue objects."""
        raw_issues = data.get("issues", [])
        issues: list[Issue] = []

        for raw in raw_issues:
            # Map category string to enum
            cat = raw.get("category", "functional")
            try:
                category = IssueCategory(cat)
            except ValueError:
                category = IssueCategory.functional

            sev = raw.get("severity", "medium")
            try:
                severity = Severity(sev)
            except ValueError:
                severity = Severity.medium

            platform = (
                Platform.mobile_web
                if persona.device_preference == Platform.mobile_web
                else Platform.web
            )

            evidence_ref = raw.get("evidence_ref", f"step-{step_number}")
            screenshot_file = Path(snapshot.screenshot_path).name if snapshot.screenshot_path else ""

            # Build rich screenshot evidence with caption
            screenshot_evidence: list[ScreenshotEvidence] = []
            if screenshot_file:
                viewport_str = ""
                if persona.device_preference == Platform.mobile_web:
                    viewport_str = "390x844"
                else:
                    viewport_str = "1440x900"
                screenshot_evidence.append(ScreenshotEvidence(
                    path=screenshot_file,
                    caption=raw.get("title", ""),
                    step_ref=evidence_ref,
                    viewport=viewport_str,
                ))

            issues.append(Issue(
                title=raw.get("title", "Untitled issue"),
                severity=severity,
                confidence=raw.get("confidence", 0.7),
                platform=platform,
                category=category,
                agent=persona.id,
                user_impact=raw.get("user_impact", ""),
                repro_steps=raw.get("repro_steps", [
                    f"Navigate to {snapshot.url}",
                    f"At {evidence_ref}: {raw.get('title', '')}",
                ]),
                expected=raw.get("expected", ""),
                actual=raw.get("actual", ""),
                observed_facts=raw.get("observed_facts", []),
                inferred_judgment=raw.get("inferred_judgment", ""),
                hypotheses=raw.get("hypotheses", []),
                evidence=Evidence(
                    screenshots=[screenshot_file] if screenshot_file else [],
                    screenshot_evidence=screenshot_evidence,
                ),
                likely_product_area=raw.get("likely_product_area", ""),
                repair_brief=raw.get("repair_brief", ""),
            ))

        return issues

    async def scrape_landing_page(
        self, url: str, include_a11y_tree: bool = False,
    ) -> str | tuple[str, str]:
        """Scrape visible text from a URL for intent modeling.

        Args:
            url: The URL to scrape.
            include_a11y_tree: If True, also returns the accessibility tree.

        Returns:
            Page text string, or (page_text, a11y_tree) tuple if include_a11y_tree.
        """
        from preflight.core.actions import get_accessibility_tree

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)  # Let JS render
                text = await page.evaluate("() => document.body.innerText")
                text = text[:15000]
                if include_a11y_tree:
                    a11y_tree = await get_accessibility_tree(page)
                    return text, a11y_tree
                return text
            except Exception as e:
                logger.error("Failed to scrape %s: %s", url, e)
                if include_a11y_tree:
                    return f"(Failed to load: {e})", ""
                return f"(Failed to load: {e})"
            finally:
                await browser.close()

    async def _attempt_login(self, page: Page, config: RunConfig) -> None:
        """Try to fill login forms if credentials provided."""
        if not config.credentials:
            return
        try:
            # Use accessible strategies for login
            email_strategies = [
                lambda: page.get_by_label("Email").first,
                lambda: page.get_by_role("textbox", name="email").first,
                lambda: page.get_by_placeholder("Email").first,
                lambda: page.locator('input[type="email"]').first,
            ]
            for get_el in email_strategies:
                try:
                    el = get_el()
                    if await el.is_visible(timeout=2000):
                        await el.fill(config.credentials.email or "")
                        break
                except Exception:
                    continue

            pwd_strategies = [
                lambda: page.get_by_label("Password").first,
                lambda: page.get_by_role("textbox", name="password").first,
                lambda: page.locator('input[type="password"]').first,
            ]
            for get_el in pwd_strategies:
                try:
                    el = get_el()
                    if await el.is_visible(timeout=2000):
                        await el.fill(config.credentials.password or "")
                        break
                except Exception:
                    continue

            # Submit
            submit_strategies = [
                lambda: page.get_by_role("button", name="Log in").first,
                lambda: page.get_by_role("button", name="Sign in").first,
                lambda: page.get_by_role("button", name="Login").first,
                lambda: page.locator('button[type="submit"]').first,
            ]
            for get_el in submit_strategies:
                try:
                    el = get_el()
                    if await el.is_visible(timeout=2000):
                        await el.click()
                        await page.wait_for_load_state("domcontentloaded", timeout=10000)
                        break
                except Exception:
                    continue

        except Exception as e:
            logger.warning("Login attempt failed: %s", e)
