"""Tests for Phase 4: Reporting, comparison, GitHub export, webhook, CLI."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from humanqa.core.schemas import (
    AgentPersona,
    CoverageEntry,
    CoverageMap,
    Evidence,
    Issue,
    IssueCategory,
    Platform,
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


# ===========================================================================
# comparison.py
# ===========================================================================

class TestComparison:
    """Tests for run-to-run comparison."""

    def test_new_issues_detected(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(issues=[], run_id="baseline")
        current = _make_run_result(
            issues=[_make_issue("New bug")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        assert len(result.new_issues) == 1
        assert result.new_issues[0].title == "New bug"
        assert len(result.resolved_issues) == 0

    def test_resolved_issues_detected(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Old bug")],
            run_id="baseline",
        )
        current = _make_run_result(issues=[], run_id="current")
        result = compare_runs(baseline, current)
        assert len(result.resolved_issues) == 1
        assert result.resolved_issues[0].title == "Old bug"

    def test_persistent_issues_detected(self):
        from humanqa.reporting.comparison import compare_runs

        issue = _make_issue("Persistent bug", severity="medium")
        baseline = _make_run_result(issues=[issue], run_id="baseline")
        current = _make_run_result(
            issues=[_make_issue("Persistent bug", severity="medium")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        assert len(result.persistent_issues) == 1
        assert len(result.new_issues) == 0
        assert len(result.resolved_issues) == 0

    def test_regressed_issues_severity_increase(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Bug", severity="low")],
            run_id="baseline",
        )
        current = _make_run_result(
            issues=[_make_issue("Bug", severity="critical")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        assert len(result.regressed_issues) == 1
        base_issue, curr_issue = result.regressed_issues[0]
        assert base_issue.severity == Severity.low
        assert curr_issue.severity == Severity.critical

    def test_severity_decrease_not_regression(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Bug", severity="critical")],
            run_id="baseline",
        )
        current = _make_run_result(
            issues=[_make_issue("Bug", severity="low")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        assert len(result.regressed_issues) == 0
        assert len(result.persistent_issues) == 1

    def test_matching_uses_title_and_category(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Same title", category="functional")],
            run_id="baseline",
        )
        current = _make_run_result(
            issues=[_make_issue("Same title", category="ux")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        # Different category = different issue
        assert len(result.new_issues) == 1
        assert len(result.resolved_issues) == 1

    def test_case_insensitive_matching(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Login Button Broken")],
            run_id="baseline",
        )
        current = _make_run_result(
            issues=[_make_issue("login button broken")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        assert len(result.persistent_issues) == 1
        assert len(result.new_issues) == 0

    def test_comparison_summary_property(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Old"), _make_issue("Persistent")],
            run_id="baseline",
        )
        current = _make_run_result(
            issues=[_make_issue("New"), _make_issue("Persistent")],
            run_id="current",
        )
        result = compare_runs(baseline, current)
        assert "New: 1" in result.summary
        assert "Resolved: 1" in result.summary

    def test_to_markdown_output(self):
        from humanqa.reporting.comparison import compare_runs

        baseline = _make_run_result(
            issues=[_make_issue("Resolved bug", severity="high")],
            run_id="baseline-1",
        )
        current = _make_run_result(
            issues=[_make_issue("New bug", severity="critical")],
            run_id="current-1",
        )
        result = compare_runs(baseline, current)
        md = result.to_markdown()
        assert "# HumanQA Run Comparison" in md
        assert "baseline-1" in md
        assert "current-1" in md
        assert "New Issues" in md
        assert "Resolved Issues" in md

    def test_load_run_result_file_not_found(self, tmp_path):
        from humanqa.reporting.comparison import load_run_result

        with pytest.raises(FileNotFoundError):
            load_run_result(str(tmp_path / "nonexistent"))

    def test_load_run_result_success(self, tmp_path):
        from humanqa.reporting.comparison import load_run_result

        result = _make_run_result(run_id="loaded-run")
        json_path = tmp_path / "report.json"
        json_path.write_text(result.model_dump_json(indent=2))

        loaded = load_run_result(str(tmp_path))
        assert loaded.run_id == "loaded-run"
        assert loaded.config.target_url == "https://example.com"

    def test_duplicate_titles_keep_highest_confidence(self):
        from humanqa.reporting.comparison import _build_issue_map

        issues = [
            _make_issue("Dup", confidence=0.5),
            _make_issue("Dup", confidence=0.9),
        ]
        m = _build_issue_map(issues)
        assert len(m) == 1
        assert list(m.values())[0].confidence == 0.9


# ===========================================================================
# github_export.py
# ===========================================================================

class TestGitHubExport:
    """Tests for GitHub issue export."""

    def test_format_issue_body_contains_sections(self):
        from humanqa.reporting.github_export import format_issue_body

        issue = _make_issue(
            "Test Bug",
            severity="high",
            user_impact="Users can't login",
            repro_steps=["Go to login", "Enter credentials", "Click submit"],
            repair_brief="Fix the auth handler",
            observed_facts=["Button renders but is non-functional"],
        )
        body = format_issue_body(issue)
        assert "**Severity:** high" in body
        assert "## User Impact" in body
        assert "Users can't login" in body
        assert "## Evidence" in body
        assert "## Repro Steps" in body
        assert "1. Go to login" in body
        assert "Repair Brief" in body
        assert "HumanQA" in body

    def test_format_issue_body_with_screenshots(self):
        from humanqa.reporting.github_export import format_issue_body

        issue = _make_issue(
            "Visual Bug",
            evidence=Evidence(screenshots=["screenshot1.png", "screenshot2.png"]),
        )
        body = format_issue_body(issue)
        assert "## Screenshots" in body
        assert "screenshot1.png" in body

    def test_issue_labels_include_severity_and_category(self):
        from humanqa.reporting.github_export import issue_labels

        issue = _make_issue("Bug", severity="critical", category="accessibility")
        labels = issue_labels(issue)
        assert "humanqa" in labels
        assert "priority: critical" in labels
        assert "type: accessibility" in labels

    def test_export_dry_run(self):
        from humanqa.reporting.github_export import export_issues_via_gh

        issues = [_make_issue("Bug A"), _make_issue("Bug B")]
        results = export_issues_via_gh(issues, repo="owner/repo", dry_run=True)
        assert len(results) == 2
        assert all(r["status"] == "dry_run" for r in results)
        assert all("[HumanQA]" in r["title"] for r in results)

    def test_export_min_severity_filter(self):
        from humanqa.reporting.github_export import export_issues_via_gh

        issues = [
            _make_issue("Critical", severity="critical"),
            _make_issue("Low", severity="low"),
            _make_issue("Info", severity="info"),
        ]
        results = export_issues_via_gh(
            issues, repo="owner/repo", dry_run=True, min_severity="high",
        )
        # Only critical passes the "high" threshold
        assert len(results) == 1
        assert "Critical" in results[0]["title"]

    def test_export_normalizes_repo_url(self):
        from humanqa.reporting.github_export import export_issues_via_gh

        issues = [_make_issue("Bug")]
        results = export_issues_via_gh(
            issues, repo="https://github.com/owner/repo", dry_run=True,
        )
        assert len(results) == 1

    @patch("humanqa.reporting.github_export.subprocess.run")
    def test_export_calls_gh_cli(self, mock_run):
        from humanqa.reporting.github_export import export_issues_via_gh

        mock_run.return_value = MagicMock(
            returncode=0, stdout="https://github.com/owner/repo/issues/42\n",
        )
        issues = [_make_issue("Real Bug")]
        results = export_issues_via_gh(issues, repo="owner/repo")
        assert len(results) == 1
        assert results[0]["status"] == "created"
        assert "issues/42" in results[0]["url"]
        mock_run.assert_called_once()

    @patch("humanqa.reporting.github_export.subprocess.run")
    def test_export_handles_gh_error(self, mock_run):
        from humanqa.reporting.github_export import export_issues_via_gh

        mock_run.return_value = MagicMock(
            returncode=1, stderr="auth required",
        )
        issues = [_make_issue("Bug")]
        results = export_issues_via_gh(issues, repo="owner/repo")
        assert results[0]["status"] == "error"
        assert "auth required" in results[0]["error"]

    @patch("humanqa.reporting.github_export.subprocess.run",
           side_effect=FileNotFoundError)
    def test_export_handles_missing_gh(self, mock_run):
        from humanqa.reporting.github_export import export_issues_via_gh

        issues = [_make_issue("Bug A"), _make_issue("Bug B")]
        results = export_issues_via_gh(issues, repo="owner/repo")
        assert len(results) == 1  # Breaks after first FileNotFoundError
        assert "gh CLI not found" in results[0]["error"]

    def test_export_summary_text(self):
        from humanqa.reporting.github_export import export_summary

        results = [
            {"status": "created"}, {"status": "created"},
            {"status": "dry_run"},
            {"status": "error"},
        ]
        summary = export_summary(results)
        assert "2 issues created" in summary
        assert "1 issues (dry run)" in summary
        assert "1 errors" in summary

    def test_export_summary_empty(self):
        from humanqa.reporting.github_export import export_summary

        assert export_summary([]) == "No issues to export"


# ===========================================================================
# report_generator.py (HTML report)
# ===========================================================================

class TestReportGenerator:
    """Tests for report generation including HTML."""

    def test_generate_markdown(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[
            _make_issue("Critical Bug", severity="critical"),
            _make_issue("Low Bug", severity="low"),
        ])
        path = gen.generate_markdown(result)
        content = Path(path).read_text()
        assert "# HumanQA Evaluation Report" in content
        assert "Critical Bug" in content
        assert "CRITICAL" in content

    def test_generate_json(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[_make_issue("Bug")])
        path = gen.generate_json(result)
        data = json.loads(Path(path).read_text())
        assert data["run_id"] == "run-test"
        assert len(data["issues"]) == 1

    def test_generate_html(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(
            issues=[
                _make_issue("HTML Bug", severity="high", category="ux"),
            ],
            scores={"trust_score": 0.85},
        )
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "<!DOCTYPE html>" in html
        assert "TestApp" in html
        assert "HTML Bug" in html
        assert "filterIssues" in html  # JavaScript filter

    def test_generate_html_with_screenshots(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[
            _make_issue("Visual", evidence=Evidence(screenshots=["shot.png"])),
        ])
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "shot.png" in html
        assert "lightbox" in html.lower()

    def test_generate_repair_briefs(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[
            _make_issue("Fix This", severity="high", repair_brief="Update the handler"),
            _make_issue("Info only", severity="info"),
        ])
        path = gen.generate_repair_briefs(result)
        briefs_dir = Path(path)
        # Info-level issues should be skipped
        brief_files = list(briefs_dir.glob("*.md"))
        assert len(brief_files) == 1
        content = brief_files[0].read_text()
        assert "Fix This" in content
        assert "Update the handler" in content

    def test_generate_all(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[_make_issue("Bug")])
        paths = gen.generate_all(result)
        assert "markdown" in paths
        assert "json" in paths
        assert "html" in paths
        assert "repair_briefs" in paths
        assert Path(paths["markdown"]).exists()
        assert Path(paths["html"]).exists()


# ===========================================================================
# webhook.py
# ===========================================================================

class TestWebhook:
    """Tests for webhook/Slack summary."""

    def test_build_summary_text(self):
        from humanqa.reporting.webhook import build_summary_text

        result = _make_run_result(issues=[
            _make_issue("Bug A", severity="critical"),
            _make_issue("Bug B", severity="high"),
            _make_issue("Bug C", severity="medium"),
        ])
        text = build_summary_text(result)
        assert "HumanQA Report" in text
        assert "1 Critical" in text
        assert "1 High" in text
        assert "1 Medium" in text
        assert "Bug A" in text  # Top issue

    def test_build_summary_no_issues(self):
        from humanqa.reporting.webhook import build_summary_text

        result = _make_run_result(issues=[])
        text = build_summary_text(result)
        assert "No issues found" in text

    def test_build_summary_with_scores(self):
        from humanqa.reporting.webhook import build_summary_text

        result = _make_run_result(
            issues=[_make_issue("Bug")],
            scores={"trust_score": 0.75},
        )
        text = build_summary_text(result)
        assert "Trust: 75%" in text

    def test_build_slack_payload(self):
        from humanqa.reporting.webhook import build_slack_payload

        result = _make_run_result(issues=[_make_issue("Bug")])
        payload = build_slack_payload(result)
        assert "text" in payload
        assert "HumanQA Report" in payload["text"]

    def test_build_slack_payload_with_report_url(self):
        from humanqa.reporting.webhook import build_slack_payload

        result = _make_run_result(issues=[])
        payload = build_slack_payload(result, report_url="https://reports.example.com/1")
        assert "https://reports.example.com/1" in payload["text"]
        assert "Full Report" in payload["text"]

    @pytest.mark.asyncio
    async def test_send_webhook_success(self):
        from humanqa.reporting.webhook import send_webhook

        result = _make_run_result(issues=[_make_issue("Bug")])

        mock_response = MagicMock(status_code=200)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("humanqa.reporting.webhook.httpx.AsyncClient", return_value=mock_client):
            success = await send_webhook("https://hooks.slack.com/test", result)
        assert success is True

    @pytest.mark.asyncio
    async def test_send_webhook_failure(self):
        from humanqa.reporting.webhook import send_webhook

        result = _make_run_result(issues=[])

        mock_response = MagicMock(status_code=500, text="Internal Server Error")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("humanqa.reporting.webhook.httpx.AsyncClient", return_value=mock_client):
            success = await send_webhook("https://hooks.slack.com/test", result)
        assert success is False

    @pytest.mark.asyncio
    async def test_send_webhook_exception(self):
        from humanqa.reporting.webhook import send_webhook

        result = _make_run_result(issues=[])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("humanqa.reporting.webhook.httpx.AsyncClient", return_value=mock_client):
            success = await send_webhook("https://hooks.slack.com/test", result)
        assert success is False

    def test_severity_emoji_mapping(self):
        from humanqa.reporting.webhook import SEVERITY_EMOJI

        assert len(SEVERITY_EMOJI) == 5
        assert "critical" in SEVERITY_EMOJI
        assert "info" in SEVERITY_EMOJI


# ===========================================================================
# cli.py
# ===========================================================================

class TestCLI:
    """Tests for CLI exit codes and command structure."""

    def test_check_exit_code_no_threshold(self):
        from humanqa.cli import _check_exit_code

        result = _make_run_result(issues=[_make_issue("Bug", severity="critical")])
        assert _check_exit_code(result, None) == 0

    def test_check_exit_code_critical_found(self):
        from humanqa.cli import _check_exit_code

        result = _make_run_result(issues=[_make_issue("Bug", severity="critical")])
        assert _check_exit_code(result, "critical") == 1

    def test_check_exit_code_below_threshold(self):
        from humanqa.cli import _check_exit_code

        result = _make_run_result(issues=[_make_issue("Bug", severity="low")])
        assert _check_exit_code(result, "high") == 0

    def test_check_exit_code_at_threshold(self):
        from humanqa.cli import _check_exit_code

        result = _make_run_result(issues=[_make_issue("Bug", severity="medium")])
        assert _check_exit_code(result, "medium") == 1

    def test_check_exit_code_no_issues(self):
        from humanqa.cli import _check_exit_code

        result = _make_run_result(issues=[])
        assert _check_exit_code(result, "critical") == 0

    def test_check_exit_code_multiple_severities(self):
        from humanqa.cli import _check_exit_code

        result = _make_run_result(issues=[
            _make_issue("Info", severity="info"),
            _make_issue("Low", severity="low"),
            _make_issue("Medium", severity="medium"),
        ])
        assert _check_exit_code(result, "high") == 0
        assert _check_exit_code(result, "medium") == 1
        assert _check_exit_code(result, "low") == 1

    def test_cli_group_exists(self):
        from click.testing import CliRunner
        from humanqa.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "HumanQA" in result.output

    def test_cli_version(self):
        from click.testing import CliRunner
        from humanqa.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_cli_run_help(self):
        from click.testing import CliRunner
        from humanqa.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--fail-on" in result.output
        assert "--webhook" in result.output

    def test_cli_compare_help(self):
        from click.testing import CliRunner
        from humanqa.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "BASELINE_DIR" in result.output
        assert "CURRENT_DIR" in result.output

    def test_cli_export_issues_help(self):
        from click.testing import CliRunner
        from humanqa.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["export-issues", "--help"])
        assert result.exit_code == 0
        assert "--repo" in result.output
        assert "--dry-run" in result.output
        assert "--min-severity" in result.output

    def test_cli_compare_missing_dir(self):
        from click.testing import CliRunner
        from humanqa.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["compare", "/nonexistent/a", "/nonexistent/b"])
        assert result.exit_code == 1

    def test_cli_compare_valid_dirs(self, tmp_path):
        from click.testing import CliRunner
        from humanqa.cli import main

        # Create two run dirs with report.json
        for name in ["baseline", "current"]:
            d = tmp_path / name
            d.mkdir()
            r = _make_run_result(
                run_id=name,
                issues=[_make_issue(f"Issue in {name}")],
            )
            (d / "report.json").write_text(r.model_dump_json())

        runner = CliRunner()
        result = runner.invoke(main, [
            "compare",
            str(tmp_path / "baseline"),
            str(tmp_path / "current"),
        ])
        assert result.exit_code == 0
        assert "Run Comparison" in result.output
