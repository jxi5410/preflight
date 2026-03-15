"""Preflight MCP Server — exposes Preflight as tools for Claude Code and other MCP clients.

Tools:
    preflight_quick_check  — Fast single-pass evaluation (~30s)
    preflight_evaluate     — Full multi-agent evaluation pipeline
    preflight_get_report   — Retrieve a previously generated report
    preflight_compare      — Compare two runs to detect regressions

Entry point:
    preflight-mcp (console_scripts in pyproject.toml)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP SDK setup — uses FastMCP from the mcp package
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False


# ---------------------------------------------------------------------------
# Tool implementations (standalone async functions, testable without mcp)
# ---------------------------------------------------------------------------

async def _quick_check_impl(
    url: str,
    focus: str = "",
    tier: str = "balanced",
) -> str:
    """Quick check a URL — fast, single-pass QA evaluation (~30s).

    Returns a lightweight assessment with issues, score, and summary.
    Use this for rapid feedback during development or PR review.

    Args:
        url: The product URL to evaluate.
        focus: Optional focus area (e.g. "checkout flow", "accessibility", "login").
        tier: Model tier — balanced (default), budget, premium, or openai.
    """
    from preflight.core.quick_check import quick_check

    result = await quick_check(
        url=url,
        focus=focus or None,
        tier=tier,
    )
    return result.model_dump_json(indent=2)


async def _evaluate_impl(
    url: str,
    repo_url: str = "",
    brief: str = "",
    focus_flows: str = "",
    tier: str = "balanced",
    output_dir: str = "./artifacts",
    fail_on: str = "",
) -> str:
    """Run a full Preflight evaluation — multi-agent, multi-lens QA pipeline.

    This is a comprehensive evaluation that takes 2-5 minutes. It generates
    personas, runs browser-based evaluations, applies specialist lenses
    (design, trust, auth, responsive), deduplicates findings, and produces
    full reports.

    Args:
        url: The product URL to evaluate.
        repo_url: Optional GitHub repo URL for deeper product understanding.
        brief: Optional product description to guide evaluation.
        focus_flows: Comma-separated focus flows (e.g. "login,checkout,settings").
        tier: Model tier — balanced (default), budget, premium, or openai.
        output_dir: Directory for report output (default: ./artifacts).
        fail_on: Return error status if issues at this severity or above found
                 (critical, high, medium, low). Empty = always succeed.
    """
    from preflight.core.llm import get_tier_config
    from preflight.core.pipeline import run_pipeline
    from preflight.core.schemas import RunConfig

    tier_provider, tier_models = get_tier_config(tier)

    config = RunConfig(
        target_url=url,
        repo_url=repo_url or None,
        brief=brief or None,
        focus_flows=[f.strip() for f in focus_flows.split(",") if f.strip()],
        output_dir=output_dir,
        llm_provider=tier_provider,
        llm_model=tier_models.smart,
        llm_tier=tier,
    )

    result = await run_pipeline(config)

    # Build summary response
    severity_counts: dict[str, int] = {}
    for issue in result.issues:
        sev = issue.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    summary = {
        "run_id": result.run_id,
        "url": url,
        "issues_total": len(result.issues),
        "severity_counts": severity_counts,
        "scores": result.scores,
        "output_dir": output_dir,
        "reports": {
            "html": f"{output_dir}/report.html",
            "markdown": f"{output_dir}/report.md",
            "json": f"{output_dir}/report.json",
            "handoff": f"{output_dir}/HANDOFF.md",
        },
        "top_issues": [
            {
                "title": issue.title,
                "severity": issue.severity.value,
                "category": issue.category.value,
                "confidence": issue.confidence,
                "user_impact": issue.user_impact,
            }
            for issue in result.issues[:10]
        ],
    }

    # Check fail_on threshold
    if fail_on:
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        threshold = sev_order.get(fail_on, 3)
        has_failure = any(
            sev_order.get(i.severity.value, 5) <= threshold
            for i in result.issues
        )
        summary["ci_status"] = "FAIL" if has_failure else "PASS"
        summary["fail_on"] = fail_on

    return json.dumps(summary, indent=2)


async def _get_report_impl(
    run_dir: str = "./artifacts",
    format: str = "markdown",
) -> str:
    """Retrieve a previously generated Preflight report.

    Args:
        run_dir: Path to the run artifacts directory (default: ./artifacts).
        format: Report format — markdown, json, html, or handoff.
    """
    format_files = {
        "markdown": "report.md",
        "json": "report.json",
        "html": "report.html",
        "handoff": "HANDOFF.md",
    }

    if format not in format_files:
        return json.dumps({
            "error": f"Unknown format '{format}'. Available: {', '.join(format_files.keys())}"
        })

    file_path = Path(run_dir) / format_files[format]

    if not file_path.exists():
        return json.dumps({
            "error": f"Report not found at {file_path}. Run an evaluation first.",
            "available": [
                str(f.name)
                for f in Path(run_dir).glob("*")
                if f.is_file()
            ] if Path(run_dir).exists() else [],
        })

    content = file_path.read_text()

    # For JSON format, parse and return structured
    if format == "json":
        return content

    # For text formats, return directly
    return content


async def _compare_impl(
    baseline_dir: str,
    current_dir: str,
) -> str:
    """Compare two Preflight runs to detect regressions and progress.

    Returns new issues, resolved issues, regressions (severity increased),
    and persistent issues.

    Args:
        baseline_dir: Path to the baseline run artifacts directory.
        current_dir: Path to the current run artifacts directory.
    """
    from preflight.reporting.comparison import compare_runs, load_run_result

    try:
        baseline = load_run_result(baseline_dir)
    except FileNotFoundError:
        return json.dumps({
            "error": f"Baseline report not found in {baseline_dir}"
        })

    try:
        current = load_run_result(current_dir)
    except FileNotFoundError:
        return json.dumps({
            "error": f"Current report not found in {current_dir}"
        })

    result = compare_runs(baseline, current)

    comparison = {
        "baseline_run_id": result.baseline_run_id,
        "current_run_id": result.current_run_id,
        "summary": result.summary,
        "new_issues": [
            {
                "title": i.title,
                "severity": i.severity.value,
                "category": i.category.value,
            }
            for i in result.new_issues
        ],
        "resolved_issues": [
            {
                "title": i.title,
                "severity": i.severity.value,
                "category": i.category.value,
            }
            for i in result.resolved_issues
        ],
        "regressed_issues": [
            {
                "title": curr.title,
                "from_severity": base.severity.value,
                "to_severity": curr.severity.value,
            }
            for base, curr in result.regressed_issues
        ],
        "persistent_count": len(result.persistent_issues),
        "markdown": result.to_markdown(),
    }

    return json.dumps(comparison, indent=2)


# ---------------------------------------------------------------------------
# Public aliases (used by tests and direct callers)
# ---------------------------------------------------------------------------

preflight_quick_check = _quick_check_impl
preflight_evaluate = _evaluate_impl
preflight_get_report = _get_report_impl
preflight_compare = _compare_impl


# ---------------------------------------------------------------------------
# MCP tool registration (only when mcp package is available)
# ---------------------------------------------------------------------------

def _register_mcp_tools(server: "FastMCP") -> None:
    """Register all tool functions with the MCP server."""

    @server.tool()
    async def preflight_quick_check(
        url: str,
        focus: str = "",
        tier: str = "balanced",
    ) -> str:
        """Quick check a URL — fast, single-pass QA evaluation (~30s).

        Returns a lightweight assessment with issues, score, and summary.
        Use this for rapid feedback during development or PR review.

        Args:
            url: The product URL to evaluate.
            focus: Optional focus area (e.g. "checkout flow", "accessibility", "login").
            tier: Model tier — balanced (default), budget, premium, or openai.
        """
        return await _quick_check_impl(url=url, focus=focus, tier=tier)

    @server.tool()
    async def preflight_evaluate(
        url: str,
        repo_url: str = "",
        brief: str = "",
        focus_flows: str = "",
        tier: str = "balanced",
        output_dir: str = "./artifacts",
        fail_on: str = "",
    ) -> str:
        """Run a full Preflight evaluation — multi-agent, multi-lens QA pipeline.

        This is a comprehensive evaluation that takes 2-5 minutes.

        Args:
            url: The product URL to evaluate.
            repo_url: Optional GitHub repo URL for deeper product understanding.
            brief: Optional product description to guide evaluation.
            focus_flows: Comma-separated focus flows (e.g. "login,checkout,settings").
            tier: Model tier — balanced (default), budget, premium, or openai.
            output_dir: Directory for report output (default: ./artifacts).
            fail_on: CI gate threshold (critical, high, medium, low).
        """
        return await _evaluate_impl(
            url=url, repo_url=repo_url, brief=brief, focus_flows=focus_flows,
            tier=tier, output_dir=output_dir, fail_on=fail_on,
        )

    @server.tool()
    async def preflight_get_report(
        run_dir: str = "./artifacts",
        format: str = "markdown",
    ) -> str:
        """Retrieve a previously generated Preflight report.

        Args:
            run_dir: Path to the run artifacts directory (default: ./artifacts).
            format: Report format — markdown, json, html, or handoff.
        """
        return await _get_report_impl(run_dir=run_dir, format=format)

    @server.tool()
    async def preflight_compare(
        baseline_dir: str,
        current_dir: str,
    ) -> str:
        """Compare two Preflight runs to detect regressions and progress.

        Args:
            baseline_dir: Path to the baseline run artifacts directory.
            current_dir: Path to the current run artifacts directory.
        """
        return await _compare_impl(baseline_dir=baseline_dir, current_dir=current_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the Preflight MCP server."""
    if not _HAS_MCP:
        print(
            "MCP server requires the 'mcp' package. Install with:\n"
            "  pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    server = FastMCP("Preflight")
    _register_mcp_tools(server)
    server.run()


if __name__ == "__main__":
    main()
