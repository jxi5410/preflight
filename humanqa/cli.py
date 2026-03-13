"""HumanQA CLI — main entry point.

Usage:
    humanqa run <url> [options]
    humanqa schedule <url> --cron "0 2 * * *" [options]
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


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.group()
@click.version_option(version="0.1.0")
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
    console.print(f"    report.json — Machine-readable export")
    console.print(f"    repair_briefs/ — Coding agent handoffs")


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


if __name__ == "__main__":
    main()
