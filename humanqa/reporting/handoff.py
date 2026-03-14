"""Handoff Generator — produces HANDOFF.md and handoff.json for developer handoff.

Transforms a HumanQA run result into an actionable developer document that
maps issues to tasks, identifies feature gaps, and summarizes coverage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from humanqa.core.file_mapper import map_issue_to_files
from humanqa.core.schemas import (
    FeatureGap,
    Handoff,
    HandoffTask,
    RepoInsights,
    RunResult,
    Severity,
)

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

EFFORT_THRESHOLDS = {
    "critical": "large",
    "high": "medium",
    "medium": "medium",
    "low": "small",
    "info": "small",
}


class HandoffGenerator:
    """Generates developer handoff documents from a run result."""

    def __init__(self, output_dir: str = "./artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: RunResult,
        repo_insights: RepoInsights | None = None,
    ) -> Handoff:
        """Build a Handoff from the run result."""
        tasks = self._build_tasks(result, repo_insights)
        feature_gaps = self._build_feature_gaps(result)
        coverage = self._build_coverage_summary(result)

        severity_counts = {}
        for issue in result.issues:
            sev = issue.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        critical_high = severity_counts.get("critical", 0) + severity_counts.get("high", 0)
        summary = (
            f"Found {len(result.issues)} issues ({critical_high} critical/high). "
            f"{len(tasks)} actionable tasks, {len(feature_gaps)} feature gaps identified."
        )

        handoff = Handoff(
            run_id=result.run_id,
            product_name=result.intent_model.product_name,
            target_url=result.config.target_url,
            summary=summary,
            tasks=tasks,
            feature_gaps=feature_gaps,
            coverage_summary=coverage,
        )
        return handoff

    def generate_all(
        self,
        result: RunResult,
        repo_insights: RepoInsights | None = None,
    ) -> dict[str, str]:
        """Generate HANDOFF.md and handoff.json. Returns dict of format -> path."""
        handoff = self.generate(result, repo_insights)
        paths = {}
        paths["handoff_md"] = self._write_markdown(handoff)
        paths["handoff_json"] = self._write_json(handoff)
        return paths

    def _build_tasks(
        self, result: RunResult, repo_insights: RepoInsights | None
    ) -> list[HandoffTask]:
        """Convert issues into actionable HandoffTasks."""
        tasks: list[HandoffTask] = []
        for issue in result.issues:
            if issue.severity == Severity.info:
                continue  # Skip info-level issues

            likely_files = map_issue_to_files(issue, repo_insights)
            task = HandoffTask(
                issue_id=issue.id,
                title=issue.title,
                severity=issue.severity,
                category=issue.category,
                likely_files=likely_files,
                repair_brief=issue.repair_brief,
                repro_steps=issue.repro_steps,
                expected=issue.expected,
                actual=issue.actual,
                effort_estimate=EFFORT_THRESHOLDS.get(issue.severity.value, "medium"),
            )
            tasks.append(task)

        # Sort by severity then title
        tasks.sort(key=lambda t: (SEVERITY_ORDER.get(t.severity.value, 5), t.title))
        return tasks

    def _build_feature_gaps(self, result: RunResult) -> list[FeatureGap]:
        """Identify gaps between claimed features and observed behavior."""
        gaps: list[FeatureGap] = []
        expectations = result.intent_model.feature_expectations

        for feat in expectations:
            if feat.verified is True:
                continue  # Feature works as claimed

            # Find related issues
            related = []
            feat_lower = feat.feature_name.lower()
            for issue in result.issues:
                if (
                    feat_lower in issue.title.lower()
                    or feat_lower in issue.likely_product_area.lower()
                ):
                    related.append(issue.id)

            if feat.verified is False:
                status = "broken" if related else "missing"
            else:
                status = "partial" if related else "missing"

            gaps.append(
                FeatureGap(
                    feature_name=feat.feature_name,
                    source=feat.source,
                    status=status,
                    details=f"Feature '{feat.feature_name}' from {feat.source} was not verified"
                    if not related
                    else f"Feature '{feat.feature_name}' has {len(related)} related issue(s)",
                    related_issues=related,
                )
            )

        return gaps

    def _build_coverage_summary(self, result: RunResult) -> dict[str, object]:
        """Build a coverage summary dict."""
        entries = result.coverage.entries
        visited = len([e for e in entries if e.status == "visited"])
        failed = len([e for e in entries if e.status == "failed"])
        pending = len([e for e in entries if e.status == "pending"])
        return {
            "total_entries": len(entries),
            "visited": visited,
            "failed": failed,
            "pending": pending,
            "visited_urls": sorted(result.coverage.visited_urls()),
        }

    def _write_markdown(self, handoff: Handoff) -> str:
        """Write HANDOFF.md."""
        lines: list[str] = []
        lines.append("# Developer Handoff")
        lines.append("")
        lines.append(f"**Run ID:** {handoff.run_id}")
        lines.append(f"**Product:** {handoff.product_name}")
        lines.append(f"**Target:** {handoff.target_url}")
        lines.append(f"**Generated:** {handoff.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(handoff.summary)
        lines.append("")

        # Tasks
        if handoff.tasks:
            lines.append("## Tasks")
            lines.append("")
            lines.append("| # | Severity | Category | Title | Effort | Files |")
            lines.append("|---|----------|----------|-------|--------|-------|")
            for i, task in enumerate(handoff.tasks, 1):
                files_str = ", ".join(task.likely_files[:3]) if task.likely_files else "-"
                lines.append(
                    f"| {i} | {task.severity.value} | {task.category.value} | "
                    f"{task.title} | {task.effort_estimate} | {files_str} |"
                )
            lines.append("")

            # Task details
            lines.append("### Task Details")
            lines.append("")
            for task in handoff.tasks:
                lines.append(f"#### [{task.severity.value.upper()}] {task.title}")
                lines.append(f"**Issue ID:** {task.issue_id}")
                if task.likely_files:
                    lines.append(f"**Likely files:** {', '.join(task.likely_files)}")
                if task.repair_brief:
                    lines.append(f"**Fix brief:** {task.repair_brief}")
                if task.repro_steps:
                    lines.append("**Repro steps:**")
                    for j, step in enumerate(task.repro_steps, 1):
                        lines.append(f"  {j}. {step}")
                if task.expected:
                    lines.append(f"**Expected:** {task.expected}")
                if task.actual:
                    lines.append(f"**Actual:** {task.actual}")
                lines.append("")

        # Feature gaps
        if handoff.feature_gaps:
            lines.append("## Feature Gaps")
            lines.append("")
            lines.append("| Feature | Source | Status | Related Issues |")
            lines.append("|---------|--------|--------|----------------|")
            for gap in handoff.feature_gaps:
                related = ", ".join(gap.related_issues) if gap.related_issues else "-"
                lines.append(f"| {gap.feature_name} | {gap.source} | {gap.status} | {related} |")
            lines.append("")

        # Coverage
        cov = handoff.coverage_summary
        if cov:
            lines.append("## Coverage")
            lines.append("")
            lines.append(f"- Visited: {cov.get('visited', 0)}")
            lines.append(f"- Failed: {cov.get('failed', 0)}")
            lines.append(f"- Pending: {cov.get('pending', 0)}")
            visited_urls = cov.get("visited_urls", [])
            if visited_urls:
                lines.append("")
                lines.append("**Visited URLs:**")
                for url in visited_urls:
                    lines.append(f"- {url}")
            lines.append("")

        md_text = "\n".join(lines)
        md_path = self.output_dir / "HANDOFF.md"
        md_path.write_text(md_text)
        logger.info("HANDOFF.md written to %s", md_path)
        return str(md_path)

    def _write_json(self, handoff: Handoff) -> str:
        """Write handoff.json."""
        json_path = self.output_dir / "handoff.json"
        json_path.write_text(handoff.model_dump_json(indent=2))
        logger.info("handoff.json written to %s", json_path)
        return str(json_path)
