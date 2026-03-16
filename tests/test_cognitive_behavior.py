"""Tests for cognitive behavior profiles."""

import pytest
from pydantic import ValidationError

from preflight.core.schemas import (
    AgentPersona,
    CognitiveBehavior,
)


class TestCognitiveBehavior:
    """Test CognitiveBehavior model."""

    def test_defaults(self):
        cb = CognitiveBehavior()
        assert cb.attention_span == "skimmer"
        assert cb.patience_threshold == 3
        assert cb.exploration_style == "linear"
        assert cb.error_tolerance == "medium"
        assert cb.jargon_comfort == "some"
        assert cb.comparison_anchors == []

    def test_scanner_profile(self):
        cb = CognitiveBehavior(
            attention_span="scanner",
            patience_threshold=2,
            exploration_style="goal-driven",
            error_tolerance="low",
            jargon_comfort="none",
            comparison_anchors=["Google", "Bing"],
        )
        assert cb.attention_span == "scanner"
        assert cb.patience_threshold == 2
        assert len(cb.comparison_anchors) == 2

    def test_reader_profile(self):
        cb = CognitiveBehavior(
            attention_span="reader",
            patience_threshold=8,
            exploration_style="curious",
            error_tolerance="high",
            jargon_comfort="fluent",
            comparison_anchors=["VS Code", "JetBrains", "Vim"],
        )
        assert cb.attention_span == "reader"
        assert cb.patience_threshold == 8

    def test_comparison_anchors_list(self):
        cb = CognitiveBehavior(
            comparison_anchors=["Notion", "Slack", "Trello"],
        )
        assert "Notion" in cb.comparison_anchors
        assert len(cb.comparison_anchors) == 3

    def test_serialization_roundtrip(self):
        cb = CognitiveBehavior(
            attention_span="scanner",
            exploration_style="curious",
            jargon_comfort="fluent",
        )
        data = cb.model_dump()
        cb2 = CognitiveBehavior(**data)
        assert cb2.attention_span == "scanner"
        assert cb2.exploration_style == "curious"


class TestPersonaWithCognitiveBehavior:
    """Test cognitive behavior integration with persona model."""

    def test_persona_has_cognitive_behavior_default(self):
        persona = AgentPersona(
            name="Test", role="User", persona_type="first_time_user",
        )
        assert isinstance(persona.cognitive_behavior, CognitiveBehavior)

    def test_persona_with_custom_cognitive_behavior(self):
        persona = AgentPersona(
            name="Dev", role="Developer", persona_type="power_user",
            cognitive_behavior=CognitiveBehavior(
                attention_span="skimmer",
                exploration_style="curious",
                error_tolerance="high",
                jargon_comfort="fluent",
                comparison_anchors=["GitHub", "GitLab"],
            ),
        )
        assert persona.cognitive_behavior.attention_span == "skimmer"
        assert persona.cognitive_behavior.jargon_comfort == "fluent"
        assert "GitHub" in persona.cognitive_behavior.comparison_anchors

    def test_different_persona_types_should_have_different_profiles(self):
        novice_cb = CognitiveBehavior(
            attention_span="scanner",
            patience_threshold=2,
            exploration_style="linear",
            error_tolerance="low",
            jargon_comfort="none",
        )
        expert_cb = CognitiveBehavior(
            attention_span="reader",
            patience_threshold=8,
            exploration_style="curious",
            error_tolerance="high",
            jargon_comfort="fluent",
        )
        assert novice_cb.attention_span != expert_cb.attention_span
        assert novice_cb.patience_threshold != expert_cb.patience_threshold
        assert novice_cb.error_tolerance != expert_cb.error_tolerance
        assert novice_cb.jargon_comfort != expert_cb.jargon_comfort
