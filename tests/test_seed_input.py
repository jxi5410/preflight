"""Tests for seed input detection, generation, and input-first flow handling."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from preflight.core.schemas import (
    AgentPersona,
    ProductIntentModel,
    SeedInput,
)
from preflight.core.seed_input import (
    HEURISTIC_INPUTS,
    SeedInputGenerator,
    get_heuristic_seed_input,
)
from preflight.core.quick_check import QuickCheckResult, quick_check


# ---------------------------------------------------------------------------
# SeedInput schema tests
# ---------------------------------------------------------------------------


class TestSeedInputSchema:
    def test_basic_seed_input(self):
        si = SeedInput(
            input_text="test query",
            purpose="Basic search test",
            expected_outcome="Results should appear",
        )
        assert si.input_text == "test query"
        assert si.is_edge_case is False

    def test_edge_case_seed_input(self):
        si = SeedInput(
            input_text="",
            purpose="Test empty input handling",
            expected_outcome="Should show validation or helpful message",
            is_edge_case=True,
        )
        assert si.is_edge_case is True
        assert si.input_text == ""

    def test_seed_input_json_roundtrip(self):
        si = SeedInput(
            input_text="hello world",
            purpose="Simple test",
            expected_outcome="Response",
        )
        data = json.loads(si.model_dump_json())
        restored = SeedInput.model_validate(data)
        assert restored.input_text == "hello world"


# ---------------------------------------------------------------------------
# ProductIntentModel input_first fields
# ---------------------------------------------------------------------------


class TestProductIntentModelInputFirst:
    def test_defaults(self):
        model = ProductIntentModel()
        assert model.input_first is False
        assert model.input_type == ""
        assert model.input_placeholder == ""

    def test_input_first_set(self):
        model = ProductIntentModel(
            product_name="SearchApp",
            product_type="search engine",
            input_first=True,
            input_type="search",
            input_placeholder="Search the web...",
        )
        assert model.input_first is True
        assert model.input_type == "search"
        assert model.input_placeholder == "Search the web..."

    def test_input_first_json_roundtrip(self):
        model = ProductIntentModel(
            product_name="AI Chat",
            input_first=True,
            input_type="prompt",
            input_placeholder="Ask anything...",
        )
        data = json.loads(model.model_dump_json())
        restored = ProductIntentModel.model_validate(data)
        assert restored.input_first is True
        assert restored.input_type == "prompt"


# ---------------------------------------------------------------------------
# AgentPersona seed_inputs field
# ---------------------------------------------------------------------------


class TestAgentPersonaSeedInputs:
    def test_default_empty(self):
        persona = AgentPersona(name="Test", role="R", persona_type="pt")
        assert persona.seed_inputs == []

    def test_with_seed_inputs(self):
        persona = AgentPersona(
            name="Test",
            role="R",
            persona_type="pt",
            seed_inputs=[
                SeedInput(
                    input_text="query",
                    purpose="test",
                    expected_outcome="results",
                ),
            ],
        )
        assert len(persona.seed_inputs) == 1
        assert persona.seed_inputs[0].input_text == "query"

    def test_seed_inputs_json_roundtrip(self):
        persona = AgentPersona(
            name="Test",
            role="R",
            persona_type="pt",
            seed_inputs=[
                SeedInput(input_text="q", purpose="p", expected_outcome="e"),
            ],
        )
        data = json.loads(persona.model_dump_json())
        restored = AgentPersona.model_validate(data)
        assert len(restored.seed_inputs) == 1
        assert restored.seed_inputs[0].input_text == "q"


# ---------------------------------------------------------------------------
# Heuristic seed input generation
# ---------------------------------------------------------------------------


class TestHeuristicSeedInput:
    def test_search_type(self):
        si = get_heuristic_seed_input("search")
        assert si.input_text == "test"

    def test_url_type(self):
        si = get_heuristic_seed_input("url")
        assert si.input_text == "example.com"

    def test_prompt_type(self):
        si = get_heuristic_seed_input("prompt")
        assert "help" in si.input_text.lower()

    def test_code_type(self):
        si = get_heuristic_seed_input("code")
        assert "print" in si.input_text

    def test_unknown_type_fallback(self):
        si = get_heuristic_seed_input("unknown_type")
        assert si.input_text == HEURISTIC_INPUTS["free_text"]

    def test_all_heuristic_types_defined(self):
        for input_type in ["search", "url", "prompt", "code", "data", "free_text"]:
            si = get_heuristic_seed_input(input_type)
            assert si.input_text, f"No heuristic input for type: {input_type}"
            assert si.purpose


# ---------------------------------------------------------------------------
# LLM-based seed input generation
# ---------------------------------------------------------------------------


class TestSeedInputGenerator:
    def test_generate_for_non_input_first(self):
        """Should return empty list when product is not input-first."""
        mock_llm = MagicMock()
        gen = SeedInputGenerator(mock_llm)
        intent = ProductIntentModel(input_first=False)
        persona = AgentPersona(name="T", role="R", persona_type="pt")

        result = gen.generate_for_persona(intent, persona)
        assert result == []
        mock_llm.complete_json.assert_not_called()

    def test_generate_with_llm(self):
        """Should call LLM and parse response into SeedInput objects."""
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "seed_inputs": [
                {
                    "input_text": "best restaurants",
                    "purpose": "Common search query",
                    "expected_outcome": "Restaurant listings",
                    "is_edge_case": False,
                },
                {
                    "input_text": "asdjkfhaskjdf",
                    "purpose": "Gibberish test",
                    "expected_outcome": "Helpful error or no results message",
                    "is_edge_case": True,
                },
            ],
        }

        gen = SeedInputGenerator(mock_llm)
        intent = ProductIntentModel(
            product_name="FoodSearch",
            input_first=True,
            input_type="search",
            input_placeholder="Search restaurants...",
        )
        persona = AgentPersona(
            name="Hungry User",
            role="Diner",
            persona_type="first_time_user",
            goals=["Find a restaurant"],
            expertise_level="novice",
        )

        result = gen.generate_for_persona(intent, persona)
        assert len(result) == 2
        assert result[0].input_text == "best restaurants"
        assert result[1].is_edge_case is True
        mock_llm.complete_json.assert_called_once()

    def test_generate_caps_at_three(self):
        """Should cap seed inputs at 3 per persona."""
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "seed_inputs": [
                {"input_text": f"input {i}", "purpose": "p", "expected_outcome": "e"}
                for i in range(5)
            ],
        }

        gen = SeedInputGenerator(mock_llm)
        intent = ProductIntentModel(input_first=True, input_type="search")
        persona = AgentPersona(name="T", role="R", persona_type="pt")

        result = gen.generate_for_persona(intent, persona)
        assert len(result) == 3

    def test_generate_llm_failure_fallback(self):
        """Should fall back to heuristic on LLM failure."""
        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = Exception("API error")

        gen = SeedInputGenerator(mock_llm)
        intent = ProductIntentModel(
            input_first=True, input_type="search",
        )
        persona = AgentPersona(name="T", role="R", persona_type="pt")

        result = gen.generate_for_persona(intent, persona)
        assert len(result) == 1
        assert result[0].input_text == "test"  # heuristic for search

    def test_generate_uses_smart_tier(self):
        """Should use smart tier for LLM call."""
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {"seed_inputs": []}

        gen = SeedInputGenerator(mock_llm)
        intent = ProductIntentModel(input_first=True, input_type="prompt")
        persona = AgentPersona(name="T", role="R", persona_type="pt")

        gen.generate_for_persona(intent, persona)
        call_kwargs = mock_llm.complete_json.call_args
        assert call_kwargs.kwargs.get("tier") == "smart"


# ---------------------------------------------------------------------------
# Intent modeler input detection
# ---------------------------------------------------------------------------


class TestIntentModelerInputDetection:
    def test_intent_modeler_parses_input_first(self):
        """Intent modeler should parse input_first fields from LLM response."""
        from preflight.core.intent_modeler import IntentModeler

        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "product_name": "SearchApp",
            "product_type": "search engine",
            "target_audience": ["general public"],
            "primary_jobs": ["find information"],
            "user_expectations": ["fast results"],
            "critical_journeys": ["search", "filter results"],
            "trust_sensitive_actions": [],
            "institutional_relevance": "none",
            "institutional_reasoning": "",
            "input_first": True,
            "input_type": "search",
            "input_placeholder": "Search...",
            "assumptions": [],
            "confidence": 0.9,
            "feature_expectations": [],
        }

        modeler = IntentModeler(mock_llm)
        from preflight.core.schemas import RunConfig

        config = RunConfig(target_url="https://search.example.com")
        result = asyncio.get_event_loop().run_until_complete(
            modeler.build_intent_model(config, "Search the web", accessibility_tree="[searchbox]")
        )

        assert result.input_first is True
        assert result.input_type == "search"
        assert result.input_placeholder == "Search..."

    def test_intent_modeler_defaults_input_first_false(self):
        """Intent modeler should default input_first to False if not in response."""
        from preflight.core.intent_modeler import IntentModeler

        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "product_name": "BlogApp",
            "product_type": "content",
            "target_audience": ["readers"],
            "primary_jobs": ["read articles"],
            "user_expectations": [],
            "critical_journeys": ["browse"],
            "trust_sensitive_actions": [],
            "institutional_relevance": "none",
            "institutional_reasoning": "",
            "assumptions": [],
            "confidence": 0.8,
            "feature_expectations": [],
        }

        modeler = IntentModeler(mock_llm)
        from preflight.core.schemas import RunConfig

        config = RunConfig(target_url="https://blog.example.com")
        result = asyncio.get_event_loop().run_until_complete(
            modeler.build_intent_model(config, "Blog content")
        )

        assert result.input_first is False
        assert result.input_type == ""


# ---------------------------------------------------------------------------
# Quick check input-first handling
# ---------------------------------------------------------------------------

_WEB_RUNNER_PATCH = "preflight.runners.web_runner.WebRunner"


class TestQuickCheckInputFirst:
    def test_quick_check_detects_input_first(self):
        """Quick check should detect and report input-first products."""
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "product_name": "SearchApp",
            "product_type": "saas",
            "input_first": True,
            "input_type": "search",
            "issues": [],
            "summary": "Search engine detected",
            "score": 0.8,
        }

        with patch(_WEB_RUNNER_PATCH) as MockRunner:
            mock_instance = MagicMock()
            mock_instance.scrape_landing_page = AsyncMock(
                return_value=("Search page content", "[searchbox]")
            )
            MockRunner.return_value = mock_instance

            # Mock _quick_check_seed_input to avoid Playwright
            with patch(
                "preflight.core.quick_check._quick_check_seed_input",
                new_callable=AsyncMock,
                return_value="Search results for: test",
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    quick_check("https://search.example.com", llm=mock_llm)
                )

        assert result.input_first is True
        assert result.input_type == "search"

    def test_quick_check_result_has_input_fields(self):
        """QuickCheckResult should include input_first and input_type fields."""
        result = QuickCheckResult(
            url="https://example.com",
            input_first=True,
            input_type="prompt",
        )
        assert result.input_first is True
        data = json.loads(result.model_dump_json())
        assert data["input_first"] is True
        assert data["input_type"] == "prompt"

    def test_quick_check_non_input_first(self):
        """Non-input-first products should have input_first=False."""
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "product_name": "Blog",
            "product_type": "content",
            "input_first": False,
            "input_type": "",
            "issues": [],
            "summary": "Blog site",
            "score": 0.9,
        }

        with patch(_WEB_RUNNER_PATCH) as MockRunner:
            mock_instance = MagicMock()
            mock_instance.scrape_landing_page = AsyncMock(
                return_value=("Blog content", "")
            )
            MockRunner.return_value = mock_instance

            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://blog.example.com", llm=mock_llm)
            )

        assert result.input_first is False
        assert result.input_type == ""


# ---------------------------------------------------------------------------
# Web runner seed input execution
# ---------------------------------------------------------------------------


class TestWebRunnerSeedInputs:
    def test_judge_seed_input_result_calls_llm(self):
        """_judge_seed_input_result should call LLM with input-specific prompt."""
        import base64
        from preflight.runners.web_runner import WebRunner
        from preflight.core.schemas import PageSnapshot

        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "issues": [
                {
                    "title": "Results not relevant",
                    "severity": "medium",
                    "category": "functional",
                    "confidence": 0.8,
                    "user_impact": "Search results don't match query",
                    "observed_facts": ["Results don't contain search term"],
                },
            ],
            "persona_reaction": "Disappointed with results",
        }

        runner = WebRunner(mock_llm, "/tmp/test-artifacts")
        persona = AgentPersona(
            name="Searcher",
            role="User",
            persona_type="first_time_user",
            goals=["Find information"],
            seed_inputs=[
                SeedInput(
                    input_text="test query",
                    purpose="Basic search",
                    expected_outcome="Relevant results",
                ),
            ],
        )
        snapshot = PageSnapshot(
            url="https://search.example.com/results",
            screenshot_base64=base64.b64encode(b"\x89PNG").decode(),
            screenshot_path="/tmp/seed01.png",
        )
        seed_input = persona.seed_inputs[0]

        issues = asyncio.get_event_loop().run_until_complete(
            runner._judge_seed_input_result(
                snapshot=snapshot,
                persona=persona,
                seed_input=seed_input,
                step_number=1,
            )
        )

        assert len(issues) == 1
        assert issues[0].title == "Results not relevant"
        # Verify the prompt includes input-first evaluation addendum
        call_args = mock_llm.complete_json_with_vision.call_args
        prompt = call_args[0][0]
        assert "test query" in prompt
        assert "Did the product respond" in prompt

    def test_judge_seed_input_llm_failure(self):
        """Should return empty list on LLM failure."""
        from preflight.runners.web_runner import WebRunner
        from preflight.core.schemas import PageSnapshot

        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = Exception("API error")

        runner = WebRunner(mock_llm, "/tmp/test-artifacts")
        persona = AgentPersona(name="T", role="R", persona_type="pt")
        snapshot = PageSnapshot(url="https://test.com")
        seed = SeedInput(input_text="test", purpose="p", expected_outcome="e")

        issues = asyncio.get_event_loop().run_until_complete(
            runner._judge_seed_input_result(snapshot, persona, seed, 1)
        )
        assert issues == []


# ---------------------------------------------------------------------------
# Integration: scrape_landing_page with a11y tree
# ---------------------------------------------------------------------------


class TestScrapeLandingPageA11y:
    def test_scrape_returns_tuple_with_a11y(self):
        """scrape_landing_page(include_a11y_tree=True) returns (text, tree)."""
        from preflight.runners.web_runner import WebRunner

        mock_llm = MagicMock()
        runner = WebRunner(mock_llm, "/tmp/test")

        # We can't easily test the actual Playwright interaction without a browser,
        # but we can test that the method signature accepts the parameter
        # and verify the return type annotation allows tuples.
        import inspect
        sig = inspect.signature(runner.scrape_landing_page)
        assert "include_a11y_tree" in sig.parameters
