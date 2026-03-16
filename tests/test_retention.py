"""Tests for retention verdict."""

import pytest
from pydantic import ValidationError

from preflight.core.schemas import (
    AgentPersona,
    RetentionVerdict,
    RunConfig,
    RunResult,
)


class TestRetentionVerdict:
    """Test RetentionVerdict model."""

    def test_positive_verdict(self):
        verdict = RetentionVerdict(
            persona_id="agent-abc",
            would_use_again=True,
            would_recommend=True,
            confidence_in_verdict=0.9,
            primary_reason="The product solves my exact problem efficiently.",
            dealbreakers=[],
            delighters=["Fast load times", "Clean UI", "Good onboarding"],
            comparison_note="Compared to Notion, this is more focused and simpler.",
            overall_sentiment="positive",
            persona_closing_thought="Overall, I really like this product and would use it daily.",
        )
        assert verdict.would_use_again is True
        assert verdict.overall_sentiment == "positive"
        assert len(verdict.delighters) == 3

    def test_negative_verdict(self):
        verdict = RetentionVerdict(
            persona_id="agent-xyz",
            would_use_again=False,
            would_recommend=False,
            confidence_in_verdict=0.85,
            primary_reason="Too many bugs and confusing UI.",
            dealbreakers=["Error messages everywhere", "No clear navigation"],
            delighters=[],
            overall_sentiment="negative",
            persona_closing_thought="I wouldn't come back to this product. Too frustrating.",
        )
        assert verdict.would_use_again is False
        assert len(verdict.dealbreakers) == 2

    def test_mixed_verdict(self):
        verdict = RetentionVerdict(
            persona_id="agent-mix",
            would_use_again=True,
            would_recommend=False,
            confidence_in_verdict=0.5,
            primary_reason="Has potential but needs polish.",
            dealbreakers=["Slow performance"],
            delighters=["Unique feature set"],
            overall_sentiment="mixed",
            persona_closing_thought="I'd try it again but wouldn't recommend it yet.",
        )
        assert verdict.overall_sentiment == "mixed"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            RetentionVerdict(
                persona_id="x",
                would_use_again=True,
                would_recommend=True,
                confidence_in_verdict=1.5,  # Over 1.0
                primary_reason="",
                overall_sentiment="positive",
                persona_closing_thought="",
            )

    def test_comparison_note_optional(self):
        verdict = RetentionVerdict(
            persona_id="agent-x",
            would_use_again=True,
            would_recommend=True,
            confidence_in_verdict=0.7,
            primary_reason="Good product.",
            overall_sentiment="positive",
            persona_closing_thought="Solid.",
        )
        assert verdict.comparison_note is None

    def test_serialization_roundtrip(self):
        verdict = RetentionVerdict(
            persona_id="agent-rt",
            would_use_again=True,
            would_recommend=False,
            confidence_in_verdict=0.6,
            primary_reason="Decent but not great.",
            dealbreakers=["No mobile app"],
            delighters=["Good pricing"],
            comparison_note="Better than Tool X",
            overall_sentiment="neutral",
            persona_closing_thought="It's okay.",
        )
        data = verdict.model_dump()
        restored = RetentionVerdict(**data)
        assert restored.persona_id == "agent-rt"
        assert restored.comparison_note == "Better than Tool X"


class TestRunResultWithRetention:
    """Test RunResult includes retention verdicts."""

    def test_default_empty(self):
        result = RunResult(config=RunConfig(target_url="https://example.com"))
        assert result.retention_verdicts == []

    def test_with_verdicts(self):
        result = RunResult(
            config=RunConfig(target_url="https://example.com"),
            retention_verdicts=[
                RetentionVerdict(
                    persona_id="a", would_use_again=True, would_recommend=True,
                    confidence_in_verdict=0.8, primary_reason="Great",
                    overall_sentiment="positive", persona_closing_thought="Love it.",
                ),
                RetentionVerdict(
                    persona_id="b", would_use_again=False, would_recommend=False,
                    confidence_in_verdict=0.7, primary_reason="Too buggy",
                    overall_sentiment="negative", persona_closing_thought="No thanks.",
                ),
            ],
        )
        assert len(result.retention_verdicts) == 2
        assert result.retention_verdicts[0].would_use_again is True
        assert result.retention_verdicts[1].would_use_again is False


class TestRetentionInReport:
    """Test that report generator includes retention verdicts."""

    def test_report_includes_retention_section(self):
        from preflight.reporting.report_generator import ReportGenerator
        import tempfile

        agent = AgentPersona(
            name="Alex", role="New user", persona_type="first_time_user",
        )
        verdict = RetentionVerdict(
            persona_id=agent.id,
            would_use_again=True,
            would_recommend=False,
            confidence_in_verdict=0.7,
            primary_reason="Good but needs work.",
            dealbreakers=["Slow load times"],
            delighters=["Clean design"],
            comparison_note="Better than Tool X for this use case.",
            overall_sentiment="mixed",
            persona_closing_thought="I'd come back but wouldn't tell friends yet.",
        )

        result = RunResult(
            config=RunConfig(target_url="https://example.com"),
            agents=[agent],
            retention_verdicts=[verdict],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            path = gen.generate_markdown(result)
            with open(path) as f:
                content = f.read()

            assert "Retention Verdicts" in content
            assert "Would Use Again" in content
            assert "Dealbreakers" in content
            assert "Delighters" in content
            assert "Better than Tool X" in content
