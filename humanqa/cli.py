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


@click.group(invoke_without_command=True)
@click.version_option(version="0.2.0")
@click.pass_context
def main(ctx):
    """HumanQA — External-experience AI QA system.

    Run with no arguments for interactive mode, or use a subcommand.
    """
    if ctx.invoked_subcommand is None:
        _interactive_run()


def _interactive_run():
    """Interactive mode — asks the user for URL and repo."""
    console.print("\n[bold]Welcome to HumanQA[/bold] — your team of AI QA companions\n")

    url = console.input("[bold]What's the product URL?[/bold] (e.g. https://your-product.com): ").strip()
    if not url:
        console.print("[red]No URL provided. Exiting.[/red]")
        sys.exit(1)
    if not url.startswith("http"):
        url = "https://" + url

    repo = console.input(
        "[bold]GitHub repo URL?[/bold] (for deeper product understanding, or press Enter to skip): "
    ).strip() or None

    brief = console.input(
        "[bold]Brief product description?[/bold] (or press Enter to let us figure it out): "
    ).strip() or None

    focus = console.input(
        "[bold]Any specific flows to focus on?[/bold] (comma-separated, or press Enter for auto): "
    ).strip() or None

    mode = console.input(
        "[bold]Run mode?[/bold] [dim](full=complete evaluation, quick=fast check)[/dim]\n"
        "  Mode [full]: "
    ).strip().lower() or "full"

    if mode == "quick":
        console.print(f"\n[bold]HumanQA Quick Check[/bold] — [cyan]{url}[/cyan]\n")
        setup_logging(verbose=True)
        from humanqa.core.quick_check import quick_check as run_quick_check
        qc_result = asyncio.run(run_quick_check(url, focus=focus))
        console.print(f"  Score: [bold]{qc_result.score:.0%}[/bold]")
        console.print(f"  Time: {qc_result.duration_seconds}s")
        if qc_result.issues:
            console.print(f"  Issues: [bold]{len(qc_result.issues)}[/bold]")
            for issue in qc_result.issues:
                color = {
                    "critical": "red", "high": "red", "medium": "yellow",
                    "low": "blue", "info": "dim",
                }.get(issue.severity, "white")
                console.print(f"    [{color}][{issue.severity}][/{color}] {issue.title}")
        else:
            console.print("  [green]No issues found[/green]")
        if qc_result.summary:
            console.print(f"\n  [dim]{qc_result.summary}[/dim]")
        return

    console.print(
        "\n[bold]Model tier?[/bold] "
        "[dim](balanced=Gemini default, budget=Gemini lite, premium=Claude, openai=GPT-4o)[/dim]"
    )
    tier = console.input("  Tier [balanced]: ").strip().lower() or "balanced"
    if tier not in ("balanced", "budget", "premium", "openai"):
        console.print(f"[yellow]Unknown tier '{tier}', using balanced[/yellow]")
        tier = "balanced"

    console.print()

    from humanqa.core.llm import get_tier_config
    tier_provider, tier_models = get_tier_config(tier)

    config = RunConfig(
        target_url=url,
        repo_url=repo,
        brief=brief,
        focus_flows=[f.strip() for f in focus.split(",")] if focus else [],
        output_dir="./artifacts",
        llm_provider=tier_provider,
        llm_model=tier_models.smart,
        llm_tier=tier,
    )

    console.print(f"[bold]HumanQA[/bold] evaluating: [cyan]{url}[/cyan]")
    if repo:
        console.print(f"  Repository: [cyan]{repo}[/cyan]")
    console.print()

    setup_logging(verbose=True)

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

    console.print(f"\n  Reports: ./artifacts/")
    console.print(f"    report.md — Human-readable report")
    console.print(f"    report.html — Interactive HTML report")
    console.print(f"    HANDOFF.md — Ready for Claude Code / Codex")
    console.print(f"    handoff.json — Machine-readable handoff")


@main.command()
@click.argument("url")
@click.option("--repo", "-r", default=None, help="GitHub repo URL for product context")
@click.option("--github-token-env", default="GITHUB_TOKEN", help="Env var name for GitHub token")
@click.option("--brief", "-b", default=None, help="Product brief / description")
@click.option("--credentials", "-c", default=None, help="JSON credentials: '{\"email\": \"...\", \"password\": \"...\"}'")
@click.option("--focus", "-f", default=None, help="Comma-separated focus flows")
@click.option("--personas", default=None, help="Comma-separated persona hints")
@click.option("--output", "-o", default="./artifacts", help="Output directory")
@click.option("--provider", default="gemini", help="LLM provider: gemini | anthropic | openai")
@click.option("--model", default=None, help="LLM model override")
@click.option("--tier", default="balanced",
              type=click.Choice(["balanced", "budget", "premium", "openai"]),
              help="Model tier: balanced (Gemini, default) | budget (Gemini lite+flash) | premium (Claude) | openai")
@click.option("--institutional", default="auto", help="Institutional review: auto | on | off")
@click.option("--no-design", is_flag=True, help="Skip design review")
@click.option("--fail-on", default=None, type=click.Choice(["critical", "high", "medium", "low"]),
              help="Exit with code 1 if issues at this severity or above are found")
@click.option("--handoff", "handoff_format", default=None,
              type=click.Choice(["claude-code", "codex", "cursor", "generic"]),
              help="Handoff format: claude-code | codex | cursor | generic")
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
    tier: str,
    institutional: str,
    no_design: bool,
    fail_on: str | None,
    handoff_format: str | None,
    webhook: str | None,
    verbose: bool,
):
    """Run an evaluation against a product URL."""
    setup_logging(verbose)

    # Resolve provider from tier if not explicitly overridden
    from humanqa.core.llm import get_tier_config
    tier_provider, tier_models = get_tier_config(tier)
    effective_provider = provider if model else tier_provider
    effective_model = model or tier_models.smart

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
        llm_provider=effective_provider,
        llm_model=effective_model,
        llm_tier=tier,
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
@click.option("--focus", "-f", default=None, help="Focus area (e.g. 'checkout flow', 'accessibility')")
@click.option("--tier", default="balanced",
              type=click.Choice(["balanced", "budget", "premium", "openai"]),
              help="Model tier")
@click.option("--json-output", "json_out", is_flag=True, help="Output raw JSON instead of formatted text")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def check(url: str, focus: str | None, tier: str, json_out: bool, verbose: bool):
    """Quick check a URL — fast, single-pass evaluation (~30s).

    Examples:

        humanqa check https://your-product.com

        humanqa check https://your-product.com --focus "login flow"
    """
    setup_logging(verbose)

    from humanqa.core.quick_check import quick_check as run_quick_check

    result = asyncio.run(run_quick_check(url, focus=focus, tier=tier))

    if json_out:
        console.print(result.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]HumanQA Quick Check[/bold] — {result.url}")
    if result.product_name:
        console.print(f"  Product: {result.product_name} ({result.product_type})")
    console.print(f"  Score: [bold]{result.score:.0%}[/bold]")
    console.print(f"  Time: {result.duration_seconds}s")
    console.print()

    if result.issues:
        console.print(f"  [bold]{len(result.issues)} issues found:[/bold]")
        for issue in result.issues:
            color = {
                "critical": "red", "high": "red", "medium": "yellow",
                "low": "blue", "info": "dim",
            }.get(issue.severity, "white")
            console.print(f"    [{color}][{issue.severity}][/{color}] {issue.title}")
            if issue.user_impact:
                console.print(f"      [dim]{issue.user_impact}[/dim]")
    else:
        console.print("  [green]No issues found[/green]")

    if result.summary:
        console.print(f"\n  [dim]{result.summary}[/dim]")


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
@click.option("--format", "fmt", default="generic",
              type=click.Choice(["claude-code", "codex", "cursor", "generic"]),
              help="Handoff format: claude-code | codex | cursor | generic")
@click.option("--repo", "-r", default=None, help="GitHub repo URL for file mapping")
@click.option("--github-token-env", default="GITHUB_TOKEN", help="Env var name for GitHub token")
@click.option("--output", "-o", default=None, help="Output directory (defaults to run_dir)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def handoff(run_dir: str, fmt: str, repo: str | None, github_token_env: str, output: str | None, verbose: bool):
    """Generate a developer handoff from a completed run.

    Regenerate handoff from an existing run in a specific format:

        humanqa handoff ./artifacts --format claude-code

        humanqa handoff ./artifacts --format codex
    """
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
            tier=result.config.llm_tier,
        )
        analyzer = RepoAnalyzer(llm)
        repo_insights = asyncio.run(
            analyzer.analyze(repo, github_token_env)
        )

    out_dir = output or run_dir
    generator = HandoffGenerator(out_dir)
    handoff_obj = generator.generate(result, repo_insights)
    paths = generator.generate_all(result, repo_insights, handoff_format=fmt)

    console.print(f"\n[bold green]Handoff generated![/bold green] (format: {fmt})")
    console.print(f"  Tasks: [bold]{len(handoff_obj.tasks)}[/bold]")
    console.print(f"  Feature gaps: [bold]{len(handoff_obj.feature_gaps)}[/bold]")
    console.print(f"  Estimated scope: {handoff_obj.total_estimated_hours}")
    console.print(f"\n  Output:")
    for name, path in paths.items():
        console.print(f"    {path}")


if __name__ == "__main__":
    main()
