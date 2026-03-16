"""Tests for abandonment modeling and non-linear exploration."""

import pytest

from preflight.core.schemas import (
    AbandonmentEvent,
    AgentPersona,
    CognitiveBehavior,
    EmotionalEvent,
    EmotionalState,
    ExplorationDetour,
)
from preflight.runners.web_runner import WebRunner


class TestAbandonmentEvent:
    """Test AbandonmentEvent model."""

    def test_basic_event(self):
        event = AbandonmentEvent(
            step_index=5,
            reason="frustrated",
            persona_thought="I've had enough. Nothing works.",
            emotional_state_at_abandonment=EmotionalState(
                frustration=0.9, confidence=0.1,
            ),
            last_action="click: Submit button (FAILED)",
        )
        assert event.reason == "frustrated"
        assert event.emotional_state_at_abandonment.frustration == 0.9

    def test_with_screenshot(self):
        event = AbandonmentEvent(
            step_index=3,
            reason="lost",
            persona_thought="I'm completely lost.",
            emotional_state_at_abandonment=EmotionalState(),
            last_action="navigate: /settings",
            screenshot_path="/tmp/screenshot.png",
        )
        assert event.screenshot_path == "/tmp/screenshot.png"


class TestExplorationDetour:
    """Test ExplorationDetour model."""

    def test_basic_detour(self):
        detour = ExplorationDetour(
            step_index=2,
            detour_target="Pricing",
            reason="I noticed a Pricing link and want to check it",
        )
        assert detour.step_index == 2
        assert detour.returned_to_journey is False

    def test_detour_with_return(self):
        detour = ExplorationDetour(
            step_index=4,
            detour_target="About page",
            reason="Curious about the company",
            returned_to_journey=True,
        )
        assert detour.returned_to_journey is True


class TestAbandonmentCheck:
    """Test abandonment logic in WebRunner."""

    def test_high_frustration_triggers_abandonment(self):
        persona = AgentPersona(
            name="Frustrated User", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(frustration=0.85, confidence=0.3),
            cognitive_behavior=CognitiveBehavior(patience_threshold=2),
            emotional_timeline=[
                EmotionalEvent(step_index=1, trigger="error", dimension="frustration",
                              old_value=0.0, new_value=0.4, persona_thought="Annoying"),
                EmotionalEvent(step_index=2, trigger="confusing", dimension="frustration",
                              old_value=0.4, new_value=0.85, persona_thought="This is terrible"),
            ],
        )
        result = WebRunner._check_abandonment(persona, 3, "click: button")
        assert result is not None
        assert result.reason == "frustrated"

    def test_low_engagement_triggers_abandonment(self):
        persona = AgentPersona(
            name="Bored User", role="User", persona_type="power_user",
            emotional_state=EmotionalState(engagement=0.15),
        )
        result = WebRunner._check_abandonment(persona, 4, "scroll: page")
        assert result is not None
        assert result.reason == "bored"

    def test_low_confidence_triggers_abandonment(self):
        persona = AgentPersona(
            name="Lost User", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(confidence=0.15),
        )
        result = WebRunner._check_abandonment(persona, 2, "navigate: /unknown")
        assert result is not None
        assert result.reason == "lost"

    def test_no_abandonment_when_ok(self):
        persona = AgentPersona(
            name="Happy User", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(
                confidence=0.7, frustration=0.2, engagement=0.8,
            ),
        )
        result = WebRunner._check_abandonment(persona, 1, "click: Start")
        assert result is None

    def test_frustration_without_enough_steps_no_abandonment(self):
        persona = AgentPersona(
            name="User", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(frustration=0.9),
            cognitive_behavior=CognitiveBehavior(patience_threshold=5),
            emotional_timeline=[
                EmotionalEvent(step_index=1, trigger="error", dimension="frustration",
                              old_value=0.0, new_value=0.9, persona_thought="Bad"),
            ],
        )
        # Only 1 frustration increase but patience_threshold is 5
        result = WebRunner._check_abandonment(persona, 2, "click: button")
        assert result is None


class TestAttentionSimulation:
    """Test attention span instructions."""

    def test_scanner_instruction(self):
        persona = AgentPersona(
            name="Scanner", role="User", persona_type="first_time_user",
            cognitive_behavior=CognitiveBehavior(attention_span="scanner"),
        )
        instruction = WebRunner._get_attention_instruction(persona)
        assert "headlines" in instruction.lower()
        assert "skip body text" in instruction.lower()

    def test_reader_instruction(self):
        persona = AgentPersona(
            name="Reader", role="User", persona_type="power_user",
            cognitive_behavior=CognitiveBehavior(attention_span="reader"),
        )
        instruction = WebRunner._get_attention_instruction(persona)
        assert "read everything" in instruction.lower()

    def test_skimmer_instruction(self):
        persona = AgentPersona(
            name="Skimmer", role="User", persona_type="first_time_user",
            cognitive_behavior=CognitiveBehavior(attention_span="skimmer"),
        )
        instruction = WebRunner._get_attention_instruction(persona)
        assert "first sentence" in instruction.lower()


class TestDetourLogic:
    """Test non-linear exploration probability."""

    def test_curious_persona_can_detour(self):
        persona = AgentPersona(
            name="Curious", role="User", persona_type="first_time_user",
            cognitive_behavior=CognitiveBehavior(exploration_style="curious"),
        )
        # Run multiple times — at least one should detour (probabilistic)
        import random
        random.seed(42)
        detours = sum(WebRunner._should_detour(persona) for _ in range(100))
        assert detours > 0  # Should get some detours with 30% probability
        assert detours < 100  # Not all

    def test_linear_persona_never_detours(self):
        persona = AgentPersona(
            name="Linear", role="User", persona_type="first_time_user",
            cognitive_behavior=CognitiveBehavior(exploration_style="linear"),
        )
        for _ in range(50):
            assert WebRunner._should_detour(persona) is False

    def test_goal_driven_persona_never_detours(self):
        persona = AgentPersona(
            name="GoalDriven", role="User", persona_type="power_user",
            cognitive_behavior=CognitiveBehavior(exploration_style="goal-driven"),
        )
        for _ in range(50):
            assert WebRunner._should_detour(persona) is False


class TestPersonaAbandonmentFields:
    """Test that persona model has abandonment and detour fields."""

    def test_persona_has_abandonment_events(self):
        persona = AgentPersona(
            name="A", role="r", persona_type="first_time_user",
        )
        assert persona.abandonment_events == []

    def test_persona_has_exploration_detours(self):
        persona = AgentPersona(
            name="A", role="r", persona_type="first_time_user",
        )
        assert persona.exploration_detours == []

    def test_abandonment_event_added(self):
        persona = AgentPersona(
            name="A", role="r", persona_type="first_time_user",
        )
        event = AbandonmentEvent(
            step_index=3, reason="frustrated",
            persona_thought="Done.", emotional_state_at_abandonment=EmotionalState(),
            last_action="click: Submit",
        )
        persona.abandonment_events.append(event)
        assert len(persona.abandonment_events) == 1
