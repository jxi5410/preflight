"""HumanQA CLI — main entry point.

Usage:
    humanqa run <url> [options]
    humanqa schedule <url> --cron "0 2 * * *" [options]
    humanqa compare <baseline_dir> <current_dir>
    humanqa export-issues --repo <repo> --run <run_dir>
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

from humanqa.core.schemas import Credentials, RunConfig

console = Console()

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def _check_exit_code(result, fail_on: str | None) -> int:
    """Determine exit code based on --fail-on threshold.

    Returns 0 if no issues at or above threshold, 1 otherwise.
    """
    if not fail_on:
        return 0

    threshold = SEVERITY_ORDER.get(fail_on, 3)
    for issue in result.issues:
        if SEVERITY_ORDER.get(issue.severity.value, 5) <= threshold:
            return 1
    return 0


@click.group()
@click.version_option(version="0.2.0")
def main():
    """HumanQA — External-experience AI QA system."""
    pass


@main.command()
@click.argument("url")
@click.option("--repo", "-r", default=None, help="GitHub repo URL for product context")
@click.option("--github-token-env", default="GITHUB_TOKEN", help="Env var name for GitHub token")
@click.option("--brief", "-b", default=None, help="Product brief / description")
@click.option("--credentials", "-c", default=None, help="JSON credentials: '{\"email\": \"...\", \"password\": \"...\"}'")
@click.option("--focus", "-f", default=None, help="Comma-separated focus flows")
@click.option("--personas", default=None, help="Comma-separated persona hints")
@click.option("--output", "-o", default="./artifacts", help="Output directory")
@click.option("--provider", default="anthropic", help="LLM provider: anthropic | openai")
@click.option("--model", default=None, help="LLM model override")
@click.option("--institutional", default="auto", help="Institutional review: auto | on | off")
@click.option("--no-design", is_flag=True, help="Skip design review")
@click.option("--fail-on", default=None, type=click.Choice(["critical", "high", "medium", "low"]),
              help="Exit with code 1 if issues at this severity or above are found")
@click.option("--webhook", default=None, help="Webhook URL for posting summary (e.g. Slack)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    url: str,
    repo: str | None,
    github_token_env: str,
    brief: str | None,
    credentials: str | None,
    focus: str | None,
    personas: str | None,
    output: str,
    provider: str,
    model: str | None,
    institutional: str,
    no_design: bool,
    fail_on: str | None,
    webhook: str | None,
    verbose: bool,
):
    """Run an evaluation against a product URL."""
    setup_logging(verbose)

    # Parse credentials
    creds = None
    if credentials:
        try:
            creds_data = json.loads(credentials)
            creds = Credentials(**creds_data)
        except (json.JSONDecodeError, Exception) as e:
            console.print(f"[red]Invalid credentials JSON: {e}[/red]")
            sys.exit(1)

    config = RunConfig(
        target_url=url,
        repo_url=repo,
        github_token_env=github_token_env,
        credentials=creds,
        brief=brief,
        persona_hints=[p.strip() for p in personas.split(",")] if personas else [],
        focus_flows=[f.strip() for f in focus.split(",")] if focus else [],
        output_dir=output,
        llm_provider=provider,
        llm_model=model or ("claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o"),
        institutional_review=institutional,
        design_review=not no_design,
    )

    console.print(f"\n[bold]HumanQA[/bold] evaluating: [cyan]{url}[/cyan]")
    if repo:
        console.print(f"  Repository: [cyan]{repo}[/cyan]")
    console.print()

    from humanqa.core.pipeline import run_pipeline

    result = asyncio.run(run_pipeline(config))

    # Print summary
    console.print(f"\n[bold green]Evaluation complete![/bold green]")
    console.print(f"  Issues found: [bold]{len(result.issues)}[/bold]")

    severity_counts = {}
    for issue in result.issues:
        sev = issue.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    for sev in ["critical", "high", "medium", "low", "info"]:
        count = severity_counts.get(sev, 0)
        if count:
            color = {"critical": "red", "high": "red", "medium": "yellow", "low": "blue", "info": "dim"}.get(sev, "white")
            console.print(f"    [{color}]{sev}: {count}[/{color}]")

    console.print(f"\n  Reports: {output}/")
    console.print(f"    report.md — Human-readable report")
    console.print(f"    report.html — Interactive HTML report")
    console.print(f"    report.json — Machine-readable export")
    console.print(f"    repair_briefs/ — Coding agent handoffs")
    console.print(f"    HANDOFF.md — Developer handoff document")
    console.print(f"    handoff.json — Machine-readable handoff")

    # Webhook
    if webhook:
        from humanqa.reporting.webhook import send_webhook
        console.print(f"\n  Sending webhook to {webhook}...")
        sent = asyncio.run(send_webhook(webhook, result))
        if sent:
            console.print("  [green]Webhook sent successfully[/green]")
        else:
            console.print("  [red]Webhook failed[/red]")

    # CI exit code
    exit_code = _check_exit_code(result, fail_on)
    if fail_on and exit_code:
        console.print(f"\n[red]CI FAIL: Issues found at severity >= {fail_on}[/red]")
    sys.exit(exit_code)


@main.command()
@click.argument("url")
@click.option("--cron", default="0 2 * * *", help="Cron expression (default: 2 AM daily)")
@click.option("--brief", "-b", default=None, help="Product brief")
@click.option("--credentials", "-c", default=None, help="JSON credentials")
@click.option("--output", "-o", default="./artifacts", help="Base output directory")
@click.option("--provider", default="anthropic", help="LLM provider")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def schedule(
    url: str,
    cron: str,
    brief: str | None,
    credentials: str | None,
    output: str,
    provider: str,
    verbose: bool,
):
    """Schedule recurring evaluation runs."""
    setup_logging(verbose)

    creds = None
    if credentials:
        try:
            creds_data = json.loads(credentials)
            creds = Credentials(**creds_data)
        except Exception as e:
            console.print(f"[red]Invalid credentials JSON: {e}[/red]")
            sys.exit(1)

    config = RunConfig(
        target_url=url,
        credentials=creds,
        brief=brief,
        output_dir=output,
        llm_provider=provider,
    )

    from humanqa.scheduling.scheduler import RunScheduler

    scheduler = RunScheduler(output)
    scheduler.load_schedule()
    job_id = scheduler.schedule(config, cron_expression=cron)

    console.print(f"\n[bold green]Scheduled![/bold green]")
    console.print(f"  Job ID: {job_id}")
    console.print(f"  Cron: {cron}")
    console.print(f"  Target: {url}")
    console.print(f"\nStarting scheduler (Ctrl+C to stop)...\n")

    scheduler.start()
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop()
        console.print("\nScheduler stopped.")


@main.command()
@click.argument("baseline_dir")
@click.argument("current_dir")
@click.option("--output", "-o", default=None, help="Output file for comparison report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def compare(baseline_dir: str, current_dir: str, output: str | None, verbose: bool):
    """Compare two HumanQA runs to detect regressions."""
    setup_logging(verbose)

    from humanqa.reporting.comparison import compare_runs, load_run_result

    try:
        baseline = load_run_result(baseline_dir)
        current = load_run_result(current_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    result = compare_runs(baseline, current)

    console.print(f"\n[bold]HumanQA Run Comparison[/bold]")
    console.print(f"  Baseline: {result.baseline_run_id}")
    console.print(f"  Current: {result.current_run_id}")
    console.print()

    if result.new_issues:
        console.print(f"  [red]New issues: {len(result.new_issues)}[/red]")
        for issue in result.new_issues[:5]:
            console.print(f"    + [{issue.severity.value}] {issue.title}")
    else:
        console.print("  [green]No new issues[/green]")

    if result.resolved_issues:
        console.print(f"  [green]Resolved: {len(result.resolved_issues)}[/green]")

    if result.regressed_issues:
        console.print(f"  [red]Regressed: {len(result.regressed_issues)}[/red]")
        for base, curr in result.regressed_issues[:5]:
            console.print(f"    ! {curr.title}: {base.severity.value} -> {curr.severity.value}")

    if result.persistent_issues:
        console.print(f"  [yellow]Persistent: {len(result.persistent_issues)}[/yellow]")

    # Write report if output specified
    if output:
        from pathlib import Path
        Path(output).write_text(result.to_markdown())
        console.print(f"\n  Report written to {output}")


@main.command("export-issues")
@click.option("--repo", required=True, help="GitHub repo (owner/repo or URL)")
@click.option("--run", "run_dir", required=True, help="Path to HumanQA run artifacts")
@click.option("--min-severity", default="low", type=click.Choice(["critical", "high", "medium", "low"]),
              help="Minimum severity to export")
@click.option("--dry-run", is_flag=True, help="Format issues without creating them on GitHub")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def export_issues(repo: str, run_dir: str, min_severity: str, dry_run: bool, verbose: bool):
    """Export HumanQA findings as GitHub issues."""
    setup_logging(verbose)

    from humanqa.reporting.comparison import load_run_result
    from humanqa.reporting.github_export import export_issues_via_gh, export_summary

    try:
        result = load_run_result(run_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Exporting HumanQA issues to GitHub[/bold]")
    console.print(f"  Repo: {repo}")
    console.print(f"  Run: {result.run_id} ({len(result.issues)} issues)")
    console.print(f"  Min severity: {min_severity}")
    if dry_run:
        console.print("  [yellow]DRY RUN — no issues will be created[/yellow]")
    console.print()

    results = export_issues_via_gh(
        result.issues,
        repo=repo,
        dry_run=dry_run,
        min_severity=min_severity,
    )

    summary = export_summary(results)
    console.print(f"  {summary}")

    for r in results:
        if r.get("status") == "created":
            console.print(f"  [green]Created:[/green] {r['title']} -> {r['url']}")
        elif r.get("status") == "dry_run":
            console.print(f"  [dim]Would create:[/dim] {r['title']}")
        elif r.get("status") == "error":
            console.print(f"  [red]Error:[/red] {r['title']}: {r.get('error', '')}")


@main.command()
@click.argument("run_dir")
@click.option("--repo", "-r", default=None, help="GitHub repo URL for file mapping")
@click.option("--github-token-env", default="GITHUB_TOKEN", help="Env var name for GitHub token")
@click.option("--output", "-o", default=None, help="Output directory (defaults to run_dir)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def handoff(run_dir: str, repo: str | None, github_token_env: str, output: str | None, verbose: bool):
    """Generate a developer handoff from a completed run."""
    setup_logging(verbose)

    from humanqa.reporting.comparison import load_run_result
    from humanqa.reporting.handoff import HandoffGenerator

    try:
        result = load_run_result(run_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Optionally analyze repo for file mapping
    repo_insights = None
    if repo:
        from humanqa.core.llm import LLMClient
        from humanqa.core.repo_analyzer import RepoAnalyzer

        console.print(f"  Analyzing repo for file mapping: [cyan]{repo}[/cyan]")
        llm = LLMClient(
            provider=result.config.llm_provider,
            model=result.config.llm_model,
        )
        analyzer = RepoAnalyzer(llm)
        repo_insights = asyncio.run(
            analyzer.analyze(repo, github_token_env)
        )

    out_dir = output or run_dir
    generator = HandoffGenerator(out_dir)
    handoff_obj = generator.generate(result, repo_insights)
    paths = generator.generate_all(result, repo_insights)

    console.print(f"\n[bold green]Handoff generated![/bold green]")
    console.print(f"  Tasks: [bold]{len(handoff_obj.tasks)}[/bold]")
    console.print(f"  Feature gaps: [bold]{len(handoff_obj.feature_gaps)}[/bold]")
    console.print(f"\n  Output:")
    for name, path in paths.items():
        console.print(f"    {path}")


if __name__ == "__main__":
    main()
