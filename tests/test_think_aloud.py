"""Tests for think-aloud narration."""

import pytest

from preflight.core.schemas import (
    Action,
    AgentPersona,
    CognitiveBehavior,
    EmotionalEvent,
    EmotionalState,
    JourneyStep,
    RunConfig,
    RunResult,
)
from preflight.runners.web_runner import WebRunner


class TestJourneyStepThinkAloud:
    """Test that JourneyStep includes think_aloud field."""

    def test_default_empty(self):
        step = JourneyStep(
            step_number=1,
            action=Action(type="click", target="Sign Up"),
        )
        assert step.think_aloud == ""

    def test_with_narration(self):
        step = JourneyStep(
            step_number=2,
            action=Action(type="click", target="Submit"),
            think_aloud="Okay, I see a big search bar in the center. I'll try typing 'London restaurants'...",
        )
        assert "search bar" in step.think_aloud

    def test_think_aloud_serialization(self):
        step = JourneyStep(
            step_number=1,
            action=Action(type="navigate", target="/home"),
            think_aloud="This page looks clean. I can see the main heading clearly.",
        )
        data = step.model_dump()
        restored = JourneyStep(**data)
        assert restored.think_aloud == step.think_aloud


class TestPersonaJourneySteps:
    """Test that persona stores journey steps."""

    def test_persona_has_journey_steps(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
        )
        assert persona.journey_steps == []

    def test_persona_accumulates_steps(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
        )
        step = JourneyStep(
            step_number=1,
            action=Action(type="click", target="Start"),
            think_aloud="Let me click this button...",
        )
        persona.journey_steps.append(step)
        assert len(persona.journey_steps) == 1
        assert persona.journey_steps[0].think_aloud == "Let me click this button..."


class TestEmotionalUpdate:
    """Test emotional state updates from LLM responses."""

    def test_apply_emotional_update(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(confidence=0.7, frustration=0.0),
        )
        data = {
            "think_aloud": "This is confusing",
            "emotional_update": {
                "confidence": 0.4,
                "frustration": 0.5,
            },
        }
        WebRunner._apply_emotional_update(persona, data, step_number=1)
        assert persona.emotional_state.confidence == 0.4
        assert persona.emotional_state.frustration == 0.5
        assert len(persona.emotional_timeline) == 2  # Both changed

    def test_no_update_when_missing(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(confidence=0.7),
        )
        data = {"think_aloud": "Looks fine"}
        WebRunner._apply_emotional_update(persona, data, step_number=1)
        assert persona.emotional_state.confidence == 0.7
        assert len(persona.emotional_timeline) == 0

    def test_small_changes_ignored(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(confidence=0.7),
        )
        data = {
            "think_aloud": "Minor",
            "emotional_update": {"confidence": 0.72},  # Only 0.02 change
        }
        WebRunner._apply_emotional_update(persona, data, step_number=1)
        # Change < 0.05, should be ignored
        assert persona.emotional_state.confidence == 0.7
        assert len(persona.emotional_timeline) == 0

    def test_values_clamped(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(frustration=0.5),
        )
        data = {
            "think_aloud": "Extreme",
            "emotional_update": {"frustration": 1.5},  # Over 1.0
        }
        WebRunner._apply_emotional_update(persona, data, step_number=1)
        assert persona.emotional_state.frustration == 1.0

    def test_emotional_event_recorded(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
            emotional_state=EmotionalState(trust=0.5),
        )
        data = {
            "think_aloud": "This looks professional and trustworthy",
            "emotional_update": {"trust": 0.8},
        }
        WebRunner._apply_emotional_update(persona, data, step_number=3)
        assert len(persona.emotional_timeline) == 1
        event = persona.emotional_timeline[0]
        assert event.step_index == 3
        assert event.dimension == "trust"
        assert event.old_value == 0.5
        assert event.new_value == 0.8


class TestReportThinkAloud:
    """Test that report generator includes think-aloud section."""

    def test_report_includes_transcript_section(self):
        from preflight.reporting.report_generator import ReportGenerator
        import tempfile

        agent = AgentPersona(
            name="Alex", role="New user", persona_type="first_time_user",
            journey_steps=[
                JourneyStep(
                    step_number=1,
                    action=Action(type="navigate", target="/home"),
                    think_aloud="I see a clean landing page with a big CTA button.",
                ),
                JourneyStep(
                    step_number=2,
                    action=Action(type="click", target="Sign Up"),
                    think_aloud="I'll click Sign Up because that's what I need to do.",
                ),
            ],
            emotional_timeline=[
                EmotionalEvent(
                    step_index=1, trigger="clean design", dimension="trust",
                    old_value=0.5, new_value=0.7, persona_thought="Looks good.",
                ),
            ],
        )

        result = RunResult(
            config=RunConfig(target_url="https://example.com"),
            agents=[agent],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            path = gen.generate_markdown(result)
            with open(path) as f:
                content = f.read()

            assert "Think-Aloud Transcripts" in content
            assert "I see a clean landing page" in content
            assert "Emotional Timeline" in content
            assert "trust" in content
