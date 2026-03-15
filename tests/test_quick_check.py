"""Tests for quick_check module and MCP server tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from preflight.core.quick_check import QuickCheckResult, QuickIssue, quick_check
from preflight.core.schemas import (
    AgentPersona,
    CoverageEntry,
    CoverageMap,
    Issue,
    IssueCategory,
    ProductIntentModel,
    RunConfig,
    RunResult,
    Severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> RunConfig:
    defaults = {"target_url": "https://example.com"}
    defaults.update(kwargs)
    return RunConfig(**defaults)


def _make_issue(title="Test Issue", severity="medium", category="functional",
                confidence=0.8, agent="tester", **kwargs) -> Issue:
    return Issue(
        title=title,
        severity=Severity(severity),
        category=IssueCategory(category),
        confidence=confidence,
        agent=agent,
        **kwargs,
    )


def _make_run_result(issues=None, run_id="run-test", **kwargs) -> RunResult:
    return RunResult(
        run_id=run_id,
        config=_make_config(),
        intent_model=ProductIntentModel(product_name="TestApp", product_type="saas"),
        issues=issues or [],
        agents=[AgentPersona(name="Tester", role="QA", persona_type="power_user")],
        coverage=CoverageMap(entries=[
            CoverageEntry(url="https://example.com", status="visited"),
        ]),
        **kwargs,
    )


# Patch target for WebRunner — it's imported locally inside quick_check
_WEB_RUNNER_PATCH = "preflight.runners.web_runner.WebRunner"


def _mock_web_runner(return_content="Test page content"):
    """Create a mock WebRunner that returns the given content from scrape."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.scrape_landing_page = AsyncMock(return_value=return_content)
    mock_cls.return_value = mock_instance
    return mock_cls


# ===========================================================================
# QuickCheckResult schema tests
# ===========================================================================

class TestQuickCheckResult:
    """Tests for the QuickCheckResult schema."""

    def test_quick_issue_schema(self):
        issue = QuickIssue(
            title="Button broken",
            severity="high",
            category="functional",
            confidence=0.9,
            user_impact="Cannot submit form",
        )
        assert issue.title == "Button broken"
        assert issue.severity == "high"
        assert issue.confidence == 0.9

    def test_quick_issue_defaults(self):
        issue = QuickIssue(title="Test")
        assert issue.severity == "medium"
        assert issue.category == "functional"
        assert issue.confidence == 0.7
        assert issue.user_impact == ""

    def test_quick_check_result_schema(self):
        result = QuickCheckResult(
            url="https://example.com",
            product_name="Example",
            product_type="saas",
            issues=[QuickIssue(title="Bug")],
            summary="Found 1 issue",
            score=0.8,
            duration_seconds=5.2,
        )
        assert result.url == "https://example.com"
        assert result.product_name == "Example"
        assert len(result.issues) == 1
        assert result.score == 0.8
        assert result.duration_seconds == 5.2

    def test_quick_check_result_defaults(self):
        result = QuickCheckResult(url="https://example.com")
        assert result.product_name == ""
        assert result.issues == []
        assert result.score == 0.0
        assert result.checked_at  # Should have a timestamp

    def test_quick_check_result_roundtrip(self):
        result = QuickCheckResult(
            url="https://example.com",
            product_name="Test",
            issues=[QuickIssue(title="Bug", severity="high")],
            score=0.75,
        )
        data = json.loads(result.model_dump_json())
        result2 = QuickCheckResult.model_validate(data)
        assert result2.url == "https://example.com"
        assert result2.issues[0].title == "Bug"
        assert result2.score == 0.75

    def test_quick_check_result_score_in_json(self):
        result = QuickCheckResult(url="https://x.com", score=0.85)
        data = json.loads(result.model_dump_json())
        assert data["score"] == 0.85


# ===========================================================================
# quick_check function tests
# ===========================================================================

def _mock_playwright():
    """Create a mock async_playwright that returns fake screenshots."""
    mock_pw = MagicMock()
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()

    mock_pw.return_value.__aenter__ = AsyncMock(return_value=MagicMock(
        chromium=MagicMock(launch=AsyncMock(return_value=mock_browser))
    ))
    mock_pw.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.close = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="Test page content")
    mock_page.screenshot = AsyncMock(return_value=b"\x89PNG fake screenshot")

    return mock_pw


_PLAYWRIGHT_PATCH = "playwright.async_api.async_playwright"


class TestQuickCheckFunction:
    """Tests for the quick_check async function."""

    def test_quick_check_with_mocked_llm(self):
        """Quick check should work with a mocked LLM returning valid JSON."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "product_name": "TestApp",
            "product_type": "saas",
            "desktop_issues": [
                {
                    "title": "Missing alt text on hero image",
                    "severity": "medium",
                    "category": "accessibility",
                    "confidence": 0.85,
                    "user_impact": "Screen readers skip the main image",
                }
            ],
            "mobile_issues": [],
            "summary": "Generally good, one accessibility issue",
            "score": 0.8,
        }

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        assert result.url == "https://example.com"
        assert result.product_name == "TestApp"
        assert len(result.issues) >= 1
        assert any(i.title == "Missing alt text on hero image" for i in result.issues)
        assert result.score == 0.8
        assert result.duration_seconds >= 0

    def test_quick_check_with_focus(self):
        """Focus area should be passed to the LLM prompt."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "product_name": "TestApp",
            "product_type": "saas",
            "desktop_issues": [],
            "mobile_issues": [],
            "summary": "Login flow looks good",
            "score": 0.95,
        }

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", focus="login flow", llm=mock_llm)
            )

        # Verify focus was included in the first vision call prompt
        first_call = mock_llm.complete_json_with_vision.call_args_list[0]
        prompt = first_call[0][0]
        assert "login flow" in prompt
        assert result.score == 0.95

    def test_quick_check_llm_failure(self):
        """Quick check should handle LLM failures gracefully."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.side_effect = Exception("API error")

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        assert result.url == "https://example.com"
        assert "failed" in result.summary.lower()
        assert result.score == 0.5
        assert result.issues == []

    def test_quick_check_score_clamped(self):
        """Score should be clamped to 0.0-1.0."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "product_name": "Test",
            "product_type": "saas",
            "desktop_issues": [],
            "mobile_issues": [],
            "summary": "Great",
            "score": 1.5,  # Out of range
        }

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        assert result.score == 1.0  # Clamped to max

    def test_quick_check_uses_fast_tier(self):
        """Quick check should use the fast tier for vision LLM calls."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "product_name": "Test",
            "product_type": "saas",
            "desktop_issues": [],
            "mobile_issues": [],
            "summary": "OK",
            "score": 0.9,
        }

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        # Verify fast tier was used
        call_kwargs = mock_llm.complete_json_with_vision.call_args
        assert call_kwargs.kwargs.get("tier") == "fast"

    def test_quick_check_viewport_field(self):
        """Issues should have correct viewport field set."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "product_name": "Test",
            "product_type": "saas",
            "desktop_issues": [
                {"title": "Desktop bug", "severity": "low", "category": "ui",
                 "confidence": 0.7, "user_impact": "Minor"}
            ],
            "mobile_issues": [
                {"title": "Mobile cutoff", "severity": "high", "category": "responsive",
                 "confidence": 0.9, "user_impact": "Content hidden"}
            ],
            "summary": "Issues on both viewports",
            "score": 0.6,
        }

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        desktop = [i for i in result.issues if i.viewport == "desktop"]
        mobile = [i for i in result.issues if i.viewport == "mobile"]
        assert len(desktop) >= 1
        assert len(mobile) >= 1
        assert desktop[0].title == "Desktop bug"
        assert mobile[0].title == "Mobile cutoff"

    def test_quick_check_mobile_detail_pass_runs(self):
        """A second dedicated mobile detail check should run when mobile screenshot exists."""
        mock_llm = MagicMock()
        # First call: general evaluation
        mock_llm.complete_json_with_vision.side_effect = [
            {
                "product_name": "Test",
                "product_type": "saas",
                "desktop_issues": [],
                "mobile_issues": [],
                "summary": "OK",
                "score": 0.9,
            },
            # Second call: mobile detail check
            {
                "mobile_issues": [
                    {"title": "Header covers content", "severity": "high",
                     "category": "responsive", "confidence": 0.9,
                     "user_impact": "First list item hidden behind nav bar"}
                ],
                "mobile_score": 0.5,
            },
        ]

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        # Should have called vision twice (general + mobile detail)
        assert mock_llm.complete_json_with_vision.call_count == 2
        # The mobile detail issue should appear in results
        assert any(i.title == "Header covers content" for i in result.issues)

    def test_quick_check_deduplicates_issues(self):
        """Duplicate issues from general and mobile detail passes should be deduped."""
        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.side_effect = [
            {
                "product_name": "Test",
                "product_type": "saas",
                "desktop_issues": [],
                "mobile_issues": [
                    {"title": "Overlap bug", "severity": "high",
                     "category": "responsive", "confidence": 0.9,
                     "user_impact": "Elements overlap"}
                ],
                "summary": "OK",
                "score": 0.7,
            },
            # Mobile detail returns same issue
            {
                "mobile_issues": [
                    {"title": "Overlap bug", "severity": "high",
                     "category": "responsive", "confidence": 0.9,
                     "user_impact": "Elements overlap"}
                ],
                "mobile_score": 0.6,
            },
        ]

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()):
            result = asyncio.get_event_loop().run_until_complete(
                quick_check("https://example.com", llm=mock_llm)
            )

        # Should only have one copy of "Overlap bug"
        overlap_issues = [i for i in result.issues if i.title == "Overlap bug"]
        assert len(overlap_issues) == 1


# ===========================================================================
# MCP Server tool tests (test the _impl functions directly)
# ===========================================================================

class TestMCPServerTools:
    """Tests for MCP server tool implementation functions."""

    def test_mcp_quick_check_tool(self):
        """MCP quick_check tool should return valid JSON."""
        from preflight.mcp_server import preflight_quick_check

        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "product_name": "TestApp",
            "product_type": "saas",
            "desktop_issues": [{"title": "Bug", "severity": "low", "category": "ux",
                        "confidence": 0.7, "user_impact": "Minor"}],
            "mobile_issues": [],
            "summary": "Mostly good",
            "score": 0.9,
        }

        with patch(_PLAYWRIGHT_PATCH, _mock_playwright()), \
             patch("preflight.core.quick_check.LLMClient", return_value=mock_llm):

            result_json = asyncio.get_event_loop().run_until_complete(
                preflight_quick_check(url="https://example.com", focus="", tier="balanced")
            )

        data = json.loads(result_json)
        assert data["url"] == "https://example.com"
        assert data["product_name"] == "TestApp"
        assert len(data["issues"]) >= 1

    def test_mcp_get_report_missing(self):
        """MCP get_report should return error for missing reports."""
        from preflight.mcp_server import preflight_get_report

        result_json = asyncio.get_event_loop().run_until_complete(
            preflight_get_report(run_dir="/nonexistent/path", format="markdown")
        )
        data = json.loads(result_json)
        assert "error" in data

    def test_mcp_get_report_success(self, tmp_path):
        """MCP get_report should return report content."""
        from preflight.mcp_server import preflight_get_report

        (tmp_path / "report.md").write_text("# Preflight Report\n\nTest content")

        result = asyncio.get_event_loop().run_until_complete(
            preflight_get_report(run_dir=str(tmp_path), format="markdown")
        )
        assert "# Preflight Report" in result
        assert "Test content" in result

    def test_mcp_get_report_json(self, tmp_path):
        """MCP get_report should return raw JSON for json format."""
        from preflight.mcp_server import preflight_get_report

        report_data = _make_run_result(issues=[_make_issue("Bug")])
        (tmp_path / "report.json").write_text(report_data.model_dump_json(indent=2))

        result = asyncio.get_event_loop().run_until_complete(
            preflight_get_report(run_dir=str(tmp_path), format="json")
        )
        data = json.loads(result)
        assert data["run_id"] == "run-test"

    def test_mcp_get_report_invalid_format(self):
        """MCP get_report should reject unknown formats."""
        from preflight.mcp_server import preflight_get_report

        result_json = asyncio.get_event_loop().run_until_complete(
            preflight_get_report(run_dir="./artifacts", format="pdf")
        )
        data = json.loads(result_json)
        assert "error" in data
        assert "pdf" in data["error"]

    def test_mcp_compare_success(self, tmp_path):
        """MCP compare should return structured comparison."""
        from preflight.mcp_server import preflight_compare

        # Create baseline and current dirs with reports
        for name, issues in [("baseline", [_make_issue("Old Bug")]),
                             ("current", [_make_issue("New Bug")])]:
            d = tmp_path / name
            d.mkdir()
            r = _make_run_result(run_id=name, issues=issues)
            (d / "report.json").write_text(r.model_dump_json())

        result_json = asyncio.get_event_loop().run_until_complete(
            preflight_compare(
                baseline_dir=str(tmp_path / "baseline"),
                current_dir=str(tmp_path / "current"),
            )
        )
        data = json.loads(result_json)
        assert data["baseline_run_id"] == "baseline"
        assert data["current_run_id"] == "current"
        assert "summary" in data
        assert "markdown" in data

    def test_mcp_compare_missing_baseline(self):
        """MCP compare should return error for missing baseline."""
        from preflight.mcp_server import preflight_compare

        result_json = asyncio.get_event_loop().run_until_complete(
            preflight_compare(
                baseline_dir="/nonexistent/baseline",
                current_dir="/nonexistent/current",
            )
        )
        data = json.loads(result_json)
        assert "error" in data

    def test_mcp_compare_missing_current(self, tmp_path):
        """MCP compare should return error for missing current."""
        from preflight.mcp_server import preflight_compare

        d = tmp_path / "baseline"
        d.mkdir()
        r = _make_run_result(run_id="baseline")
        (d / "report.json").write_text(r.model_dump_json())

        result_json = asyncio.get_event_loop().run_until_complete(
            preflight_compare(
                baseline_dir=str(tmp_path / "baseline"),
                current_dir="/nonexistent/current",
            )
        )
        data = json.loads(result_json)
        assert "error" in data

    def test_mcp_server_has_register_function(self):
        """MCP server module should expose _register_mcp_tools."""
        from preflight.mcp_server import _register_mcp_tools
        assert callable(_register_mcp_tools)

    def test_mcp_main_requires_mcp_package(self):
        """main() should fail gracefully without mcp package."""
        from preflight.mcp_server import _HAS_MCP
        if not _HAS_MCP:
            from preflight.mcp_server import main
            with pytest.raises(SystemExit):
                main()


# ===========================================================================
# CLI check command tests
# ===========================================================================

class TestCLICheck:
    """Tests for the CLI check command."""

    def test_cli_check_help(self):
        from click.testing import CliRunner
        from preflight.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["check", "--help"])
        assert result.exit_code == 0
        assert "--focus" in result.output
        assert "--tier" in result.output
        assert "--json-output" in result.output

    def test_cli_check_exists_in_group(self):
        from preflight.cli import main

        assert "check" in main.commands
