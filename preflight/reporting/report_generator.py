"""Report Generator — Human-readable and machine-readable outputs.

Produces:
- Markdown report for humans (PMs, designers, engineers)
- JSON export for machines
- Per-issue repair briefs for coding agents (Claude Code / Codex)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import jinja2

from preflight.core.performance import score_explanation
from preflight.core.schemas import Issue, RunResult, Severity

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates all output artifacts from a run result."""

    def __init__(self, output_dir: str = "./artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, result: RunResult, memory_context: str = "") -> dict[str, str]:
        """Generate all report formats. Returns dict of format -> filepath."""
        paths = {}
        paths["markdown"] = self.generate_markdown(result, memory_context=memory_context)
        paths["json"] = self.generate_json(result)
        paths["html"] = self.generate_html(result)
        paths["repair_briefs"] = self.generate_repair_briefs(result)
        return paths

    def generate_markdown(self, result: RunResult, memory_context: str = "") -> str:
        """Generate human-readable markdown report."""
        intent = result.intent_model
        issues = result.issues

        # Count by severity
        severity_counts = {}
        for i in issues:
            severity_counts[i.severity.value] = severity_counts.get(i.severity.value, 0) + 1

        # Count by category
        category_counts = {}
        for i in issues:
            category_counts[i.category.value] = category_counts.get(i.category.value, 0) + 1

        # Build report
        lines = []
        lines.append(f"# Preflight Evaluation Report")
        lines.append(f"**Run ID:** {result.run_id}")
        lines.append(f"**Date:** {result.started_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"**Target:** {result.config.target_url}")
        lines.append("")

        # Executive summary
        lines.append("## Executive Summary")
        lines.append("")
        critical_high = severity_counts.get("critical", 0) + severity_counts.get("high", 0)
        total = len(issues)
        lines.append(
            f"Preflight evaluated **{intent.product_name}** ({intent.product_type}) "
            f"using {len(result.agents)} agent personas across "
            f"{len(result.coverage.entries)} screens/states. "
            f"Found **{total} issues** total: "
            f"**{critical_high} critical/high**, "
            f"{severity_counts.get('medium', 0)} medium, "
            f"{severity_counts.get('low', 0)} low, "
            f"{severity_counts.get('info', 0)} informational."
        )
        lines.append("")

        # Product understanding
        lines.append("## Product Understanding")
        lines.append("")
        lines.append(f"**What we think this is:** {intent.product_name} — {intent.product_type}")
        lines.append(f"**Target audience:** {', '.join(intent.target_audience)}")
        lines.append(f"**Primary jobs:** {', '.join(intent.primary_jobs)}")
        lines.append(f"**Confidence:** {intent.confidence:.0%}")
        if intent.assumptions:
            lines.append(f"**Assumptions:** {'; '.join(intent.assumptions)}")
        lines.append("")

        # Who tested
        lines.append("## Agent Team")
        lines.append("")
        for agent in result.agents:
            journeys = ", ".join(agent.assigned_journeys) if agent.assigned_journeys else "(auto)"
            lines.append(
                f"- **{agent.name}** ({agent.persona_type}) — "
                f"{agent.device_preference.value}, patience={agent.patience_level} | "
                f"Journeys: {journeys}"
            )
        lines.append("")

        # Issues by severity
        lines.append("## Findings")
        lines.append("")

        for severity in ["critical", "high", "medium", "low", "info"]:
            sev_issues = [i for i in issues if i.severity.value == severity]
            if not sev_issues:
                continue

            label = severity.upper()
            lines.append(f"### {label} ({len(sev_issues)})")
            lines.append("")

            for issue in sev_issues:
                lines.append(f"#### {issue.id}: {issue.title}")
                lines.append(f"**Category:** {issue.category.value} | "
                           f"**Platform:** {issue.platform.value} | "
                           f"**Confidence:** {issue.confidence:.0%}")
                if issue.user_impact:
                    lines.append(f"**User impact:** {issue.user_impact}")
                lines.append("")

                if issue.observed_facts:
                    lines.append("**Observed facts:**")
                    for fact in issue.observed_facts:
                        lines.append(f"- {fact}")
                    lines.append("")

                if issue.inferred_judgment:
                    lines.append(f"**Judgment:** {issue.inferred_judgment}")
                    lines.append("")

                if issue.repro_steps:
                    lines.append("**Repro steps:**")
                    for j, step in enumerate(issue.repro_steps, 1):
                        lines.append(f"{j}. {step}")
                    lines.append("")

                if issue.expected and issue.actual:
                    lines.append(f"**Expected:** {issue.expected}")
                    lines.append(f"**Actual:** {issue.actual}")
                    lines.append("")

                if issue.repair_brief:
                    lines.append(f"**Fix brief:** {issue.repair_brief}")
                    lines.append("")

                if issue.evidence.screenshots:
                    for s in issue.evidence.screenshots:
                        lines.append(f"![screenshot]({s})")
                    lines.append("")

                lines.append("---")
                lines.append("")

        # Think-Aloud Transcripts
        has_think_aloud = any(
            step.think_aloud
            for agent in result.agents
            for step in agent.journey_steps
        )
        if has_think_aloud:
            lines.append("## Think-Aloud Transcripts")
            lines.append("")
            for agent in result.agents:
                agent_steps = [s for s in agent.journey_steps if s.think_aloud]
                if not agent_steps:
                    continue
                lines.append(f"### {agent.name} ({agent.persona_type})")
                lines.append("")
                for step in agent_steps:
                    lines.append(f"**Step {step.step_number}** — {step.action.type}: {step.action.target}")
                    lines.append(f"> {step.think_aloud}")
                    lines.append("")
                    if step.screenshot_path:
                        lines.append(f"![step-{step.step_number}]({step.screenshot_path})")
                        lines.append("")

                # Show emotional timeline
                if agent.emotional_timeline:
                    lines.append("**Emotional Timeline:**")
                    lines.append("")
                    for event in agent.emotional_timeline:
                        lines.append(
                            f"- Step {event.step_index}: {event.dimension} "
                            f"{event.old_value:.2f} → {event.new_value:.2f} "
                            f"({event.trigger[:60]})"
                        )
                    lines.append("")

                lines.append("---")
                lines.append("")

        # Institutional readiness (if applicable)
        inst_issues = [i for i in issues if i.category.value == "institutional_trust"]
        if inst_issues or result.scores.get("institutional_readiness") is not None:
            lines.append("## Institutional Readiness")
            lines.append("")
            readiness = result.scores.get("institutional_readiness_label", "not assessed")
            lines.append(f"**Readiness level:** {readiness}")
            lines.append(f"**Institutional issues found:** {len(inst_issues)}")
            lines.append("")

        # Coverage summary
        lines.append("## Coverage Summary")
        lines.append("")
        visited = len([e for e in result.coverage.entries if e.status == "visited"])
        failed = len([e for e in result.coverage.entries if e.status == "failed"])
        lines.append(f"- Screens visited: {visited}")
        lines.append(f"- Failed navigations: {failed}")
        lines.append(f"- Total coverage entries: {len(result.coverage.entries)}")
        lines.append("")

        # Learning context
        if memory_context:
            lines.append("## Learning Context")
            lines.append("")
            lines.append(
                "This evaluation incorporated feedback from prior runs. "
                "Known false positives were suppressed and evaluation "
                "thresholds were adjusted based on user feedback."
            )
            lines.append("")

        # Category breakdown
        lines.append("## Issue Breakdown by Category")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        # Write to file
        report_text = "\n".join(lines)
        report_path = self.output_dir / "report.md"
        report_path.write_text(report_text)
        logger.info("Markdown report written to %s", report_path)
        return str(report_path)

    def generate_json(self, result: RunResult) -> str:
        """Generate machine-readable JSON export."""
        json_path = self.output_dir / "report.json"
        json_path.write_text(result.model_dump_json(indent=2))
        logger.info("JSON report written to %s", json_path)
        return str(json_path)

    def generate_html(self, result: RunResult) -> str:
        """Generate self-contained interactive HTML report."""
        intent = result.intent_model
        issues = result.issues

        severity_counts = {}
        for i in issues:
            severity_counts[i.severity.value] = severity_counts.get(i.severity.value, 0) + 1

        categories = sorted({i.category.value for i in issues})
        agents = sorted({i.agent for i in issues if i.agent})

        # Generate performance budget explanation
        perf_explanation = ""
        if intent.product_type:
            perf_explanation = score_explanation(intent.product_type).split("\n")[1]  # Just the explanation line

        template_dir = Path(__file__).parent / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(["html"]),
        )
        template = env.get_template("report.html")

        html = template.render(
            run_id=result.run_id,
            started_at=result.started_at.strftime("%Y-%m-%d %H:%M UTC"),
            config=result.config,
            intent=intent,
            issues=issues,
            total_issues=len(issues),
            severity_counts=severity_counts,
            categories=categories,
            agents=agents,
            scores=result.scores,
            coverage_count=len(result.coverage.entries),
            issue_groups=result.issue_groups,
            perf_explanation=perf_explanation,
        )

        html_path = self.output_dir / "report.html"
        html_path.write_text(html)
        logger.info("HTML report written to %s", html_path)
        return str(html_path)

    def generate_repair_briefs(self, result: RunResult) -> str:
        """Generate per-issue repair briefs for coding agents."""
        briefs_dir = self.output_dir / "repair_briefs"
        briefs_dir.mkdir(parents=True, exist_ok=True)

        for issue in result.issues:
            if issue.severity.value in ("info",):
                continue  # Skip info-level for repair briefs

            brief = []
            brief.append(f"# Repair Brief: {issue.id}")
            brief.append("")
            brief.append(f"## Issue: {issue.title}")
            brief.append(f"**Severity:** {issue.severity.value}")
            brief.append(f"**Category:** {issue.category.value}")
            brief.append(f"**Platform:** {issue.platform.value}")
            brief.append(f"**Confidence:** {issue.confidence:.0%}")
            brief.append("")

            brief.append("## User Impact")
            brief.append(issue.user_impact or "(not specified)")
            brief.append("")

            if issue.repro_steps:
                brief.append("## Repro Steps")
                for j, step in enumerate(issue.repro_steps, 1):
                    brief.append(f"{j}. {step}")
                brief.append("")

            if issue.expected:
                brief.append(f"## Expected Behavior")
                brief.append(issue.expected)
                brief.append("")

            if issue.actual:
                brief.append(f"## Actual Behavior")
                brief.append(issue.actual)
                brief.append("")

            if issue.observed_facts:
                brief.append("## Evidence")
                for fact in issue.observed_facts:
                    brief.append(f"- {fact}")
                brief.append("")

            if issue.evidence.screenshots:
                brief.append("## Screenshots")
                for s in issue.evidence.screenshots:
                    brief.append(f"- {s}")
                brief.append("")

            brief.append("## Likely Product Area")
            brief.append(issue.likely_product_area or "(unknown)")
            brief.append("")

            brief.append("## Suggested Fix")
            brief.append(issue.repair_brief or "(no specific suggestion)")
            brief.append("")

            brief.append("## Regression Test Suggestion")
            brief.append(
                f"After fixing, verify: navigate through the repro steps above and confirm "
                f"the expected behavior is met. Automate this as an e2e test covering "
                f"'{issue.likely_product_area or 'the affected area'}'."
            )

            brief_path = briefs_dir / f"{issue.id}.md"
            brief_path.write_text("\n".join(brief))

        logger.info("Repair briefs written to %s", briefs_dir)
        return str(briefs_dir)
