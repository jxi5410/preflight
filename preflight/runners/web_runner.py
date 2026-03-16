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
    AbandonmentEvent,
    Action,
    AgentPersona,
    CoverageEntry,
    CoverageMap,
    EmotionalState,
    Evidence,
    ExplorationDetour,
    Issue,
    IssueCategory,
    JourneyStep,
    PageSnapshot,
    Platform,
    ProductIntentModel,
    RunConfig,
    ScreenshotEvidence,
    SeedInput,
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

INPUT_FIRST_EVALUATION_ADDENDUM = """
This product requires user input to function. You typed: "{seed_input}"

Evaluate the results:
- Did the product respond appropriately to this input?
- Are the results relevant to what was typed?
- How long did results take to appear? Was there a loading indicator?
- Is it clear what the results mean and how to act on them?
- If no results were found, is the empty state helpful?
- Can the user easily modify their input and try again?
- Does the product suggest alternatives or corrections?"""

# Default max steps per journey (exploration cap)
DEFAULT_MAX_STEPS = 8

# Timeout for individual page navigation
PAGE_NAVIGATION_TIMEOUT_MS = 20000

# Timeout for individual action execution
ACTION_TIMEOUT_MS = 10000


class WebRunner:
    """Runs web-based evaluation using Playwright with vision + a11y + deterministic actions."""

    def __init__(self, llm: LLMClient, output_dir: str = "./artifacts", memory_context: str = ""):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_context = memory_context

    async def evaluate(
        self,
        config: RunConfig,
        persona: AgentPersona,
        journeys: list[str],
        coverage: CoverageMap,
        max_steps: int = DEFAULT_MAX_STEPS,
        intent_model: ProductIntentModel | None = None,
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

                # Handle input-first products: execute seed inputs before journeys
                is_input_first = (
                    intent_model is not None
                    and intent_model.input_first
                    and persona.seed_inputs
                )
                if is_input_first:
                    seed_issues = await self._execute_seed_inputs(
                        page=page,
                        persona=persona,
                        config=config,
                        console_errors=console_errors,
                        network_error_count=network_error_count,
                        load_time_ms=load_time_ms,
                    )
                    all_issues.extend(seed_issues)

                    coverage.entries.append(CoverageEntry(
                        url=page.url,
                        screen_name=await page.title(),
                        agent_id=persona.id,
                        flow="seed_input_evaluation",
                        status="visited",
                        issues_found=len(seed_issues),
                    ))

                    # Navigate back to start for regular journeys
                    try:
                        await page.goto(
                            config.target_url,
                            wait_until="domcontentloaded",
                            timeout=15000,
                        )
                    except Exception:
                        pass

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

    async def _execute_seed_inputs(
        self,
        page: Page,
        persona: AgentPersona,
        config: RunConfig,
        console_errors: list[str],
        network_error_count: int,
        load_time_ms: int,
    ) -> list[Issue]:
        """Execute seed inputs for input-first products.

        For each seed input: find the primary input field, type the text,
        submit, wait for results, evaluate.
        """
        all_issues: list[Issue] = []

        for idx, seed_input in enumerate(persona.seed_inputs):
            logger.info(
                "  Agent %s trying seed input %d: %s",
                persona.name, idx + 1, seed_input.input_text[:50],
            )

            try:
                # Find the primary input field using accessibility-based strategies
                input_el = None
                input_strategies = [
                    lambda: page.get_by_role("searchbox").first,
                    lambda: page.get_by_role("textbox").first,
                    lambda: page.locator("textarea").first,
                    lambda: page.locator('input[type="text"]').first,
                    lambda: page.locator('input[type="search"]').first,
                    lambda: page.locator("input:not([type])").first,
                ]
                for get_el in input_strategies:
                    try:
                        el = get_el()
                        if await el.is_visible(timeout=2000):
                            input_el = el
                            break
                    except Exception:
                        continue

                if input_el is None:
                    logger.warning("No input field found for seed input")
                    all_issues.append(Issue(
                        title="Cannot find primary input field",
                        severity=Severity.high,
                        category=IssueCategory.functional,
                        agent=persona.id,
                        user_impact="User cannot interact with the product's main input",
                        observed_facts=["No visible input, textarea, or searchbox found on page"],
                        platform=Platform.web,
                    ))
                    break

                # Clear and type the seed input
                await input_el.clear()
                await input_el.fill(seed_input.input_text)

                # Try to submit: look for a submit button, then fall back to Enter
                submitted = False
                submit_strategies = [
                    lambda: page.get_by_role("button", name="Search").first,
                    lambda: page.get_by_role("button", name="Submit").first,
                    lambda: page.get_by_role("button", name="Go").first,
                    lambda: page.get_by_role("button", name="Send").first,
                    lambda: page.locator('button[type="submit"]').first,
                    lambda: page.locator("form button").first,
                ]
                for get_btn in submit_strategies:
                    try:
                        btn = get_btn()
                        if await btn.is_visible(timeout=1000):
                            await btn.click()
                            submitted = True
                            break
                    except Exception:
                        continue

                if not submitted:
                    await input_el.press("Enter")

                # Wait for results to load
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    await page.wait_for_timeout(2000)  # Extra time for async results
                except Exception:
                    pass

                # Capture and evaluate the results
                snapshot = await capture_snapshot(
                    page=page,
                    output_dir=self.output_dir,
                    snapshot_name=f"{persona.id}-seed{idx + 1:02d}",
                    console_errors=console_errors,
                    network_error_count=network_error_count,
                    load_time_ms=load_time_ms if idx == 0 else 0,
                )

                step_issues = await self._judge_seed_input_result(
                    snapshot=snapshot,
                    persona=persona,
                    seed_input=seed_input,
                    step_number=idx + 1,
                )
                all_issues.extend(step_issues)

                # Navigate back for next input
                if idx < len(persona.seed_inputs) - 1:
                    try:
                        await page.goto(
                            config.target_url,
                            wait_until="domcontentloaded",
                            timeout=15000,
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error("Seed input execution failed: %s", e)
                all_issues.append(Issue(
                    title=f"Seed input failed: {seed_input.input_text[:50]}",
                    severity=Severity.medium,
                    category=IssueCategory.functional,
                    agent=persona.id,
                    user_impact="Product may not handle user input correctly",
                    observed_facts=[str(e)],
                    platform=Platform.web,
                ))

        # Also test empty input submission
        try:
            await page.goto(
                config.target_url,
                wait_until="domcontentloaded",
                timeout=15000,
            )
            # Find and submit empty
            input_el = None
            for get_el in [
                lambda: page.get_by_role("searchbox").first,
                lambda: page.get_by_role("textbox").first,
                lambda: page.locator("textarea").first,
            ]:
                try:
                    el = get_el()
                    if await el.is_visible(timeout=2000):
                        input_el = el
                        break
                except Exception:
                    continue

            if input_el:
                await input_el.clear()
                await input_el.press("Enter")
                await page.wait_for_timeout(2000)
                snapshot = await capture_snapshot(
                    page=page,
                    output_dir=self.output_dir,
                    snapshot_name=f"{persona.id}-seed-empty",
                    console_errors=console_errors,
                    network_error_count=network_error_count,
                )
                empty_input = SeedInput(
                    input_text="",
                    purpose="Test empty input error handling",
                    expected_outcome="Product should show a helpful message or validation",
                )
                empty_issues = await self._judge_seed_input_result(
                    snapshot=snapshot,
                    persona=persona,
                    seed_input=empty_input,
                    step_number=len(persona.seed_inputs) + 1,
                )
                all_issues.extend(empty_issues)
        except Exception as e:
            logger.debug("Empty input test failed: %s", e)

        return all_issues

    async def _judge_seed_input_result(
        self,
        snapshot: PageSnapshot,
        persona: AgentPersona,
        seed_input: SeedInput,
        step_number: int,
    ) -> list[Issue]:
        """Evaluate results after a seed input submission."""
        page_context = snapshot_to_prompt_context(snapshot)
        input_addendum = INPUT_FIRST_EVALUATION_ADDENDUM.format(
            seed_input=seed_input.input_text or "(empty input)",
        )

        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            persona_name=persona.name,
            persona_role=persona.role,
            persona_goals=", ".join(persona.goals),
            patience_level=persona.patience_level,
            expertise_level=persona.expertise_level,
            behavioral_style=persona.behavioral_style or "standard",
            journey="seed input evaluation",
            step_number=step_number,
            max_steps=len(persona.seed_inputs) + 1,
            page_context=page_context,
            action_description=f"Typed '{seed_input.input_text or '(empty)'}' and submitted",
            previous_actions=[f"Typed: {seed_input.input_text or '(empty)'}"],
        ) + input_addendum

        try:
            if snapshot.screenshot_base64:
                screenshot_bytes = base64.b64decode(snapshot.screenshot_base64)
                data = self.llm.complete_json_with_vision(
                    prompt,
                    images=[(screenshot_bytes, "image/png")],
                    system=EVALUATION_SYSTEM_PROMPT,
                )
            else:
                data = self.llm.complete_json(prompt, system=EVALUATION_SYSTEM_PROMPT)

            return self._parse_issues(data, persona, snapshot, step_number)
        except Exception as e:
            logger.error("Seed input evaluation failed: %s", e)
            return []

    @staticmethod
    def _check_abandonment(persona: AgentPersona, step_num: int, last_action: str, screenshot_path: str | None = None) -> AbandonmentEvent | None:
        """Check if the persona should abandon the journey based on emotional state."""
        state = persona.emotional_state
        cb = persona.cognitive_behavior
        confusing_steps = sum(1 for e in persona.emotional_timeline if e.dimension == "frustration" and e.new_value > e.old_value)

        if state.frustration > 0.8 and confusing_steps >= cb.patience_threshold:
            return AbandonmentEvent(
                step_index=step_num,
                reason="frustrated",
                persona_thought="I've had enough. Nothing is working the way I expect and I'm done trying.",
                emotional_state_at_abandonment=state.model_copy(),
                last_action=last_action,
                screenshot_path=screenshot_path,
            )
        if state.engagement < 0.2:
            return AbandonmentEvent(
                step_index=step_num,
                reason="bored",
                persona_thought="I've lost interest. This doesn't seem worth my time anymore.",
                emotional_state_at_abandonment=state.model_copy(),
                last_action=last_action,
                screenshot_path=screenshot_path,
            )
        if state.confidence < 0.2:
            return AbandonmentEvent(
                step_index=step_num,
                reason="lost",
                persona_thought="I have no idea what to do next. I feel completely lost.",
                emotional_state_at_abandonment=state.model_copy(),
                last_action=last_action,
                screenshot_path=screenshot_path,
            )
        return None

    @staticmethod
    def _get_attention_instruction(persona: AgentPersona) -> str:
        """Get attention-simulation instructions based on cognitive behavior."""
        span = persona.cognitive_behavior.attention_span
        if span == "scanner":
            return "You only look at headlines, buttons, and images — skip body text entirely."
        elif span == "reader":
            return "You read everything carefully, including body text, labels, and fine print."
        else:  # skimmer
            return "You read the first sentence of each section and scan headings."

    @staticmethod
    def _should_detour(persona: AgentPersona) -> bool:
        """Determine if a curious persona should deviate from the journey path."""
        import random
        if persona.cognitive_behavior.exploration_style == "curious":
            return random.random() < 0.3  # 30% chance
        return False

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
            # CHECK ABANDONMENT before each step
            abandonment = self._check_abandonment(
                persona, step_num,
                previous_actions[-1] if previous_actions else "none",
                None,
            )
            if abandonment:
                persona.abandonment_events.append(abandonment)
                issues.append(Issue(
                    title=f"Persona abandoned journey: {abandonment.reason}",
                    severity=Severity.high,
                    confidence=0.9,
                    platform=Platform.web if persona.device_preference != Platform.mobile_web else Platform.mobile_web,
                    category=IssueCategory.ux,
                    agent=persona.id,
                    user_impact=f"User gave up at step {step_num}: {abandonment.persona_thought}",
                    observed_facts=[
                        f"Frustration: {persona.emotional_state.frustration:.2f}",
                        f"Confidence: {persona.emotional_state.confidence:.2f}",
                        f"Engagement: {persona.emotional_state.engagement:.2f}",
                        f"Last action: {abandonment.last_action}",
                    ],
                    likely_product_area="User Flow",
                    repair_brief="Investigate why users abandon at this point in the journey",
                ))
                logger.info("  Agent %s abandoned journey '%s' at step %d: %s",
                           persona.name, journey, step_num, abandonment.reason)
                break

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

            # CHECK NON-LINEAR EXPLORATION for curious personas
            if self._should_detour(persona) and actions:
                detour = ExplorationDetour(
                    step_index=step_num,
                    detour_target=actions[0].get("target", "unknown") if isinstance(actions[0], dict) else "unknown",
                    reason="I noticed something interesting and want to explore it before continuing",
                )
                persona.exploration_detours.append(detour)
                logger.info("  Agent %s taking detour at step %d", persona.name, step_num)

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

        # Inject learned context from prior runs
        system = EVALUATION_SYSTEM_PROMPT
        if self.memory_context:
            system += f"\n\n## LEARNED CONTEXT FROM PRIOR RUNS\n{self.memory_context}"

        try:
            # Send screenshot as vision input alongside text context
            if snapshot.screenshot_base64:
                screenshot_bytes = base64.b64decode(snapshot.screenshot_base64)
                data = self.llm.complete_json_with_vision(
                    prompt,
                    images=[(screenshot_bytes, "image/png")],
                    system=system,
                )
            else:
                data = self.llm.complete_json(prompt, system=system)

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
