"""Agent Team Orchestrator.

Coordinates agent runs across the product. Assigns journeys, manages coverage map,
minimises redundant exploration, and deduplicates findings across agents.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from preflight.core.llm import LLMClient
from preflight.core.performance import (
    evaluate_snapshot_performance,
    performance_results_to_issues,
    summarize_performance,
)
from preflight.core.schemas import (
    AgentPersona,
    CoverageMap,
    FirstImpressionResult,
    Issue,
    IssueCategory,
    IssueGroup,
    PageSnapshot,
    Platform,
    ProductIntentModel,
    RetentionVerdict,
    RunConfig,
    RunResult,
    Severity,
)
from preflight.lenses.first_impression_lens import FirstImpressionLens
from preflight.runners.web_runner import WebRunner
from preflight.runners.mobile_runner import MobileRunner

logger = logging.getLogger(__name__)

DEDUP_SYSTEM_PROMPT = """You are a deduplication engine for Preflight.

Your job is to cluster semantically similar issues from multiple QA agents.
Two issues are duplicates if they describe the same underlying problem, even if:
- They use different wording
- They were found by different agents
- They have different severity levels
- One is more specific than the other

Two issues are NOT duplicates if they:
- Affect different pages or flows
- Describe different root causes
- Are in different categories AND about different UI elements

Respond with JSON: {"clusters": [{"indices": [0, 3], "reason": "Both describe the same missing form validation"}]}

Each issue index should appear in at most one cluster. Issues that are unique should NOT appear in any cluster."""

DEDUP_PROMPT = """Cluster these issues by semantic similarity. Group duplicates together.

## Issues
{issue_list}

Return clusters of duplicate issues. Each cluster should contain the indices of issues
that describe the same underlying problem. Issues with no duplicates should be omitted."""

COMPARATIVE_SYSTEM_PROMPT = """You are the comparative analysis engine for Preflight.

After multiple personas have evaluated a product, you compare their findings to identify:

1. **Convergence findings**: Issues noticed by multiple personas. These are high-signal —
   if 3 of 5 personas stumbled on the same problem, it's likely a real and impactful issue.
2. **Persona-specific findings**: Issues only one persona type noticed. These are valuable
   because they reveal blind spots — e.g., only the compliance reviewer noticed a missing
   audit trail, or only the novice user got confused by the onboarding.

For convergence findings:
- Summarize what the personas agreed on
- Note how many personas encountered it
- Recommend severity weighting: more personas = higher severity

For persona-specific findings:
- Explain why only this persona type would notice
- Assess whether this is a niche concern or a blind spot in other personas

Respond with JSON:
{
  "convergence_findings": [
    {
      "title": "...",
      "description": "...",
      "personas_affected": ["persona1", "persona2"],
      "convergence_count": 3,
      "recommended_severity": "high",
      "evidence_summary": "..."
    }
  ],
  "persona_specific_findings": [
    {
      "title": "...",
      "description": "...",
      "persona": "...",
      "why_only_this_persona": "...",
      "recommended_severity": "medium"
    }
  ],
  "cross_persona_summary": "2-3 sentence overall assessment"
}"""

COMPARATIVE_PROMPT = """Compare findings across personas. Identify convergence and unique insights.

## Agents who participated
{agent_descriptions}

## All findings (after deduplication)
{findings_by_agent}

## Coverage summary
{coverage_summary}

Analyze cross-persona patterns. What did multiple personas agree on? What did only specific
persona types catch?"""

RETENTION_SYSTEM_PROMPT = """You are simulating a real user's final verdict after using a product.

Based on the full experience — first impression, journey walkthrough, emotional state changes,
and any abandonment events — determine whether this persona would return to this product.

Respond with JSON:
{
  "would_use_again": true/false,
  "would_recommend": true/false,
  "confidence_in_verdict": 0.0-1.0,
  "primary_reason": "One sentence explaining the main reason",
  "dealbreakers": ["list of things that would prevent return"],
  "delighters": ["list of things that would bring them back"],
  "comparison_note": "Compared to [product], this is..." or null,
  "overall_sentiment": "positive" | "neutral" | "negative" | "mixed",
  "persona_closing_thought": "First person closing thought..."
}"""

RETENTION_PROMPT = """You are {persona_name}, a {persona_role}.

## Your Experience Summary

### First Impression
{first_impression}

### Journey Think-Aloud Transcript
{think_aloud_transcript}

### Emotional Timeline
{emotional_timeline}

### Abandonment Events
{abandonment_events}

### Your Profile
- Expertise: {expertise_level}
- Goals: {persona_goals}
- You compare products to: {comparison_anchors}

Now give your final verdict: would you come back to this product? Would you recommend it?
Be honest and specific. Speak in first person as {persona_name}."""

JOURNEY_ASSIGNMENT_PROMPT = """You are the test planner for Preflight.

Given the product's critical journeys and a team of agent personas, assign journeys
to agents so that:
1. Every critical journey is covered by at least one agent
2. Different agents test from their unique perspective
3. Minimize redundant coverage — don't assign the same journey to similar personas
4. Assign mobile-focused journeys to agents with mobile device preference
5. Assign trust/institutional journeys to institutional personas

## Critical Journeys
{journeys}

## Agent Team
{agents}

Respond with JSON: {{"assignments": [{{"agent_id": "...", "journeys": ["...", "..."]}}]}}"""


def compute_error_signature(issue: Issue) -> str:
    """Compute a normalized fingerprint for deduplication.

    Combines category, product area, and normalized title keywords
    into a stable key that identifies the same underlying problem
    regardless of wording differences.
    """
    import re

    # Normalize title: lowercase, strip punctuation, sort words
    title_words = sorted(set(re.sub(r"[^a-z0-9 ]", "", issue.title.lower()).split()))
    # Keep top 5 distinctive words (skip very short words)
    title_key = "|".join(w for w in title_words if len(w) >= 3)[:100]
    area = issue.likely_product_area.lower().strip()
    cat = issue.category.value
    return f"{cat}:{area}:{title_key}"


def group_issues(issues: list[Issue]) -> list[IssueGroup]:
    """Group issues by category + product area into IssueGroups."""
    from collections import defaultdict

    buckets: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        area = issue.likely_product_area.strip() or "General"
        key = f"{issue.category.value}:{area.lower()}"
        buckets[key].append(issue)

    groups: list[IssueGroup] = []
    for key, bucket in buckets.items():
        if len(bucket) < 2:
            continue  # Only group when there are multiple related issues

        # Highest severity in the group
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        best_sev = min(bucket, key=lambda i: sev_order.get(i.severity.value, 5)).severity

        area = bucket[0].likely_product_area.strip() or "General"
        cat = bucket[0].category

        group = IssueGroup(
            title=f"{cat.value.replace('_', ' ').title()} issues in {area}",
            description=f"{len(bucket)} related {cat.value} issues in the {area} area",
            category=cat,
            severity=best_sev,
            issue_ids=[i.id for i in bucket],
            product_area=area,
            issue_count=len(bucket),
        )

        for issue in bucket:
            issue.group_id = group.id

        groups.append(group)

    return groups


class Orchestrator:
    """Coordinates the full evaluation run across all agents."""

    def __init__(self, llm: LLMClient, output_dir: str = "./artifacts"):
        self.llm = llm
        self.output_dir = output_dir
        self.web_runner = WebRunner(llm, output_dir)
        self.mobile_runner = MobileRunner(llm, output_dir)
        self.first_impression_lens = FirstImpressionLens(llm)
        self._collected_snapshots: list[PageSnapshot] = []

    async def run(
        self,
        config: RunConfig,
        intent: ProductIntentModel,
        agents: list[AgentPersona],
    ) -> RunResult:
        """Execute the full evaluation pipeline."""
        result = RunResult(
            config=config,
            intent_model=intent,
            agents=agents,
            started_at=datetime.now(tz=__import__("datetime").timezone.utc),
        )

        # Step 1: Assign journeys to agents
        assignments = await self._assign_journeys(intent, agents)

        # Step 2: First-impression evaluation (runs BEFORE journeys)
        first_impressions: list[FirstImpressionResult] = []
        try:
            landing_snapshot = await self._capture_landing_snapshot(config)
            if landing_snapshot:
                for agent in agents:
                    fi_result = await self.first_impression_lens.evaluate(
                        persona=agent, intent=intent, snapshot=landing_snapshot,
                    )
                    first_impressions.append(fi_result)
                result.first_impressions = first_impressions
        except Exception as e:
            logger.warning("First-impression evaluation failed: %s", e)

        # Step 3: Run web evaluations
        coverage = CoverageMap()
        all_issues: list[Issue] = []

        # Convert first-impression results to issues
        if first_impressions:
            fi_issues = self.first_impression_lens.results_to_issues(first_impressions)
            all_issues.extend(fi_issues)

        # Run agents sequentially to avoid overwhelming the target
        # (parallel option available but sequential is safer default)
        for agent in agents:
            agent_journeys = assignments.get(agent.id, intent.critical_journeys[:2])
            agent.assigned_journeys = agent_journeys

            if agent.device_preference in (Platform.web, Platform.mobile_web):
                if agent.device_preference == Platform.mobile_web:
                    issues, coverage = await self.mobile_runner.evaluate_mobile_web(
                        config, agent, agent_journeys, coverage,
                    )
                else:
                    issues, coverage = await self.web_runner.evaluate(
                        config, agent, agent_journeys, coverage,
                        intent_model=intent,
                    )
                all_issues.extend(issues)
                logger.info(
                    "Agent %s found %d issues on %s",
                    agent.name, len(issues), agent.device_preference.value,
                )

            elif agent.device_preference == Platform.mobile_app:
                issues, coverage = await self.mobile_runner.evaluate_native_app(
                    config, agent, agent_journeys, coverage,
                )
                all_issues.extend(issues)

        # Step 4: Run at least one mobile critical path if not already covered
        mobile_covered = any(
            a.device_preference in (Platform.mobile_web, Platform.mobile_app)
            for a in agents
        )
        if not mobile_covered and agents:
            logger.info("No mobile agent assigned — running mobile critical path with first agent")
            mobile_issues, coverage = await self.mobile_runner.evaluate_mobile_web(
                config, agents[0], intent.critical_journeys[:1], coverage,
            )
            all_issues.extend(mobile_issues)

        # Step 5: Evaluate performance budgets from collected snapshots
        perf_issues, perf_scores = self._evaluate_performance(
            self._collected_snapshots, intent.product_type,
        )
        all_issues.extend(perf_issues)

        # Step 6: Compute error signatures for dedup
        for issue in all_issues:
            if not issue.error_signature:
                issue.error_signature = compute_error_signature(issue)

        # Step 7: Deduplicate
        deduped = self._deduplicate_issues(all_issues)

        # Step 8: Comparative evaluation across personas
        comparative_issues = self._comparative_evaluation(deduped, agents, coverage)
        deduped.extend(comparative_issues)

        # Step 9: Rank by severity and confidence
        ranked = sorted(
            deduped,
            key=lambda i: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(i.severity.value, 5),
                -i.confidence,
            ),
        )

        # Step 10: Group related issues
        issue_groups = group_issues(ranked)

        result.issues = ranked
        result.issue_groups = issue_groups
        result.coverage = coverage
        result.scores.update(perf_scores)

        # Step 11: Retention verdicts — would each persona come back?
        retention_verdicts = await self._evaluate_retention(agents, result)
        result.retention_verdicts = retention_verdicts

        result.completed_at = datetime.now(tz=__import__("datetime").timezone.utc)

        return result

    async def _assign_journeys(
        self,
        intent: ProductIntentModel,
        agents: list[AgentPersona],
    ) -> dict[str, list[str]]:
        """Use LLM to intelligently assign journeys to agents."""
        if not intent.critical_journeys:
            # No journeys inferred — everyone gets generic exploration
            return {a.id: ["general_exploration"] for a in agents}

        agents_desc = "\n".join(
            f"- {a.id}: {a.name} ({a.persona_type}, {a.device_preference.value}, "
            f"patience={a.patience_level}, expertise={a.expertise_level})"
            for a in agents
        )

        prompt = JOURNEY_ASSIGNMENT_PROMPT.format(
            journeys="\n".join(f"- {j}" for j in intent.critical_journeys),
            agents=agents_desc,
        )

        try:
            data = self.llm.complete_json(prompt, tier="fast")
            assignments_raw = data.get("assignments", [])
            result: dict[str, list[str]] = {}
            for entry in assignments_raw:
                agent_id = entry.get("agent_id", "")
                journeys = entry.get("journeys", [])
                result[agent_id] = journeys
            return result
        except Exception as e:
            logger.warning("Journey assignment failed, using round-robin: %s", e)
            # Fallback: round-robin
            result = {}
            for i, agent in enumerate(agents):
                start = i % len(intent.critical_journeys)
                result[agent.id] = [
                    intent.critical_journeys[start],
                    intent.critical_journeys[(start + 1) % len(intent.critical_journeys)],
                ]
            return result

    def _comparative_evaluation(
        self,
        issues: list[Issue],
        agents: list[AgentPersona],
        coverage: CoverageMap,
    ) -> list[Issue]:
        """Compare findings across personas to generate convergence and persona-specific insights."""
        if len(agents) < 2 or not issues:
            return []

        # Group issues by agent
        agent_map = {a.id: a for a in agents}
        findings_by_agent: dict[str, list[Issue]] = {}
        for issue in issues:
            findings_by_agent.setdefault(issue.agent, []).append(issue)

        agent_descs = "\n".join(
            f"- {a.id}: {a.name} ({a.persona_type}, expertise={a.expertise_level}, "
            f"patience={a.patience_level})"
            for a in agents
        )

        findings_text_parts = []
        for agent_id, agent_issues in findings_by_agent.items():
            agent_name = agent_map.get(agent_id, AgentPersona(
                name=agent_id, role="", persona_type="",
            )).name
            lines = [f"\n### {agent_name} ({agent_id})"]
            for iss in agent_issues[:15]:  # Cap per agent for token budget
                lines.append(
                    f'  - [{iss.severity.value}] "{iss.title}" '
                    f"(category={iss.category.value}, confidence={iss.confidence})"
                )
            findings_text_parts.append("\n".join(lines))

        coverage_summary = f"{len(coverage.entries)} pages visited, {len(coverage.visited_urls())} unique URLs"

        prompt = COMPARATIVE_PROMPT.format(
            agent_descriptions=agent_descs,
            findings_by_agent="\n".join(findings_text_parts),
            coverage_summary=coverage_summary,
        )

        try:
            data = self.llm.complete_json(prompt, system=COMPARATIVE_SYSTEM_PROMPT, tier="fast")
        except Exception as e:
            logger.warning("Comparative evaluation failed: %s", e)
            return []

        comparative_issues: list[Issue] = []

        # Process convergence findings
        for finding in data.get("convergence_findings", []):
            count = finding.get("convergence_count", 2)
            sev_str = finding.get("recommended_severity", "medium")
            try:
                severity = Severity(sev_str)
            except ValueError:
                severity = Severity.medium

            personas_affected = finding.get("personas_affected", [])
            comparative_issues.append(Issue(
                title=f"[Convergence: {count} personas] {finding.get('title', 'Cross-persona finding')}",
                severity=severity,
                confidence=min(0.5 + count * 0.1, 1.0),  # More personas = higher confidence
                platform=Platform.web,
                category=IssueCategory.ux,
                agent="comparative_analysis",
                user_impact=finding.get("description", ""),
                observed_facts=[
                    f"Found by {count} of {len(agents)} personas",
                    f"Personas affected: {', '.join(personas_affected)}",
                    finding.get("evidence_summary", ""),
                ],
                inferred_judgment=f"Cross-persona convergence suggests this is a significant issue",
                likely_product_area="Cross-cutting",
                repair_brief=finding.get("description", ""),
            ))

        # Process persona-specific findings
        for finding in data.get("persona_specific_findings", []):
            sev_str = finding.get("recommended_severity", "low")
            try:
                severity = Severity(sev_str)
            except ValueError:
                severity = Severity.low

            comparative_issues.append(Issue(
                title=f"[{finding.get('persona', 'Specialist')} only] {finding.get('title', 'Persona-specific finding')}",
                severity=severity,
                confidence=0.7,
                platform=Platform.web,
                category=IssueCategory.ux,
                agent="comparative_analysis",
                user_impact=finding.get("description", ""),
                observed_facts=[
                    f"Only detected by persona: {finding.get('persona', 'unknown')}",
                    f"Reason: {finding.get('why_only_this_persona', '')}",
                ],
                inferred_judgment=finding.get("why_only_this_persona", ""),
                likely_product_area="Persona-specific",
                repair_brief=finding.get("description", ""),
            ))

        logger.info(
            "Comparative evaluation: %d convergence, %d persona-specific findings",
            len(data.get("convergence_findings", [])),
            len(data.get("persona_specific_findings", [])),
        )

        return comparative_issues

    async def _evaluate_retention(
        self,
        agents: list[AgentPersona],
        result: RunResult,
    ) -> list[RetentionVerdict]:
        """Generate retention verdicts for each persona."""
        verdicts: list[RetentionVerdict] = []

        # Build first-impression lookup
        fi_map = {fi.persona_id: fi for fi in result.first_impressions}

        for agent in agents:
            # Build think-aloud transcript
            transcript_parts = []
            for step in agent.journey_steps:
                if step.think_aloud:
                    transcript_parts.append(f"Step {step.step_number}: {step.think_aloud}")
            transcript = "\n".join(transcript_parts) if transcript_parts else "(no transcript)"

            # Build emotional timeline
            timeline_parts = []
            for event in agent.emotional_timeline:
                timeline_parts.append(
                    f"Step {event.step_index}: {event.dimension} "
                    f"{event.old_value:.2f} → {event.new_value:.2f} ({event.trigger[:60]})"
                )
            timeline = "\n".join(timeline_parts) if timeline_parts else "(no emotional changes)"

            # Build first impression text
            fi = fi_map.get(agent.id)
            fi_text = (
                f"Clarity: {fi.clarity_score}/10 — {fi.clarity_explanation}\n"
                f"Gut reaction: {fi.gut_reaction}\n"
                f"Would continue: {fi.would_continue}"
            ) if fi else "(no first impression recorded)"

            # Build abandonment events
            abandonment_text = "\n".join(
                f"Step {ae.step_index}: {ae.reason} — {ae.persona_thought}"
                for ae in agent.abandonment_events
            ) if agent.abandonment_events else "(no abandonment events)"

            cb = agent.cognitive_behavior
            prompt = RETENTION_PROMPT.format(
                persona_name=agent.name,
                persona_role=agent.role,
                first_impression=fi_text,
                think_aloud_transcript=transcript[:3000],
                emotional_timeline=timeline,
                abandonment_events=abandonment_text,
                expertise_level=agent.expertise_level,
                persona_goals=", ".join(agent.goals),
                comparison_anchors=", ".join(cb.comparison_anchors) if cb.comparison_anchors else "(none)",
            )

            try:
                data = self.llm.complete_json(prompt, system=RETENTION_SYSTEM_PROMPT, tier="fast")
                verdict = RetentionVerdict(
                    persona_id=agent.id,
                    would_use_again=data.get("would_use_again", False),
                    would_recommend=data.get("would_recommend", False),
                    confidence_in_verdict=max(0.0, min(1.0, float(data.get("confidence_in_verdict", 0.5)))),
                    primary_reason=data.get("primary_reason", ""),
                    dealbreakers=data.get("dealbreakers", []),
                    delighters=data.get("delighters", []),
                    comparison_note=data.get("comparison_note"),
                    overall_sentiment=data.get("overall_sentiment", "neutral"),
                    persona_closing_thought=data.get("persona_closing_thought", ""),
                )
                verdicts.append(verdict)
            except Exception as e:
                logger.warning("Retention verdict failed for %s: %s", agent.name, e)
                verdicts.append(RetentionVerdict(
                    persona_id=agent.id,
                    would_use_again=False,
                    would_recommend=False,
                    confidence_in_verdict=0.0,
                    primary_reason="Retention evaluation failed",
                    overall_sentiment="neutral",
                    persona_closing_thought="Unable to evaluate.",
                ))

        logger.info("Retention verdicts: %d/%d would use again",
                    sum(1 for v in verdicts if v.would_use_again), len(verdicts))
        return verdicts

    async def _capture_landing_snapshot(self, config: RunConfig) -> PageSnapshot | None:
        """Capture a snapshot of the landing page for first-impression evaluation."""
        try:
            from playwright.async_api import async_playwright
            from preflight.runners.page_snapshot import capture_snapshot
            from pathlib import Path

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1440, "height": 900})
                try:
                    await page.goto(
                        config.target_url,
                        wait_until="domcontentloaded",
                        timeout=20000,
                    )
                    await page.wait_for_timeout(2000)
                    snapshot = await capture_snapshot(
                        page=page,
                        output_dir=Path(self.output_dir),
                        snapshot_name="landing-page",
                    )
                    return snapshot
                finally:
                    await browser.close()
        except Exception as e:
            logger.warning("Failed to capture landing snapshot: %s", e)
            return None

    def add_snapshots(self, snapshots: list[PageSnapshot]) -> None:
        """Register snapshots collected during evaluation for performance analysis."""
        self._collected_snapshots.extend(snapshots)

    def _evaluate_performance(
        self,
        snapshots: list[PageSnapshot],
        product_type: str,
    ) -> tuple[list[Issue], dict]:
        """Evaluate performance metrics from collected snapshots against budgets."""
        if not snapshots:
            return [], {}

        all_perf_results = []
        all_perf_issues: list[Issue] = []
        seen_urls: set[str] = set()

        for snapshot in snapshots:
            # Evaluate each unique URL only once (use first snapshot per URL)
            if snapshot.url in seen_urls:
                continue
            seen_urls.add(snapshot.url)

            results = evaluate_snapshot_performance(snapshot, product_type)
            all_perf_results.extend(results)
            issues = performance_results_to_issues(results, snapshot)
            all_perf_issues.extend(issues)

        scores = summarize_performance(all_perf_results) if all_perf_results else {}
        return all_perf_issues, scores

    def _deduplicate_issues(self, issues: list[Issue]) -> list[Issue]:
        """Remove near-duplicate issues using a two-pass strategy.

        Pass 1 (fast): Group by error_signature — exact-match fingerprints
        that merge issues with identical category+area+keywords.

        Pass 2 (LLM): Send remaining issues for semantic clustering to catch
        near-duplicates that differ in wording but share root cause.

        Falls back to title-string matching if LLM dedup fails.
        """
        if not issues:
            return []

        # Pass 1: Signature-based dedup (fast, deterministic)
        after_sig = self._deduplicate_by_signature(issues)
        logger.info(
            "Signature dedup: %d -> %d issues", len(issues), len(after_sig),
        )

        # Pass 2: LLM semantic dedup on remaining issues
        if len(after_sig) <= 3:
            return after_sig

        try:
            return self._deduplicate_with_llm(after_sig)
        except Exception as e:
            logger.warning("LLM dedup failed, falling back to title matching: %s", e)
            return self._deduplicate_by_title(after_sig)

    @staticmethod
    def _deduplicate_by_signature(issues: list[Issue]) -> list[Issue]:
        """Fast dedup pass: merge issues with identical error_signature."""
        if not issues:
            return []

        sig_map: dict[str, Issue] = {}
        for issue in issues:
            sig = issue.error_signature
            if not sig:
                # No signature — keep as-is (will be handled by LLM pass)
                sig = issue.id  # unique, so no merging

            if sig in sig_map:
                existing = sig_map[sig]
                if issue.confidence > existing.confidence:
                    issue.observed_facts.append(
                        f"Also reported by agent: {existing.agent}"
                    )
                    sig_map[sig] = issue
                else:
                    existing.observed_facts.append(
                        f"Also reported by agent: {issue.agent}"
                    )
            else:
                sig_map[sig] = issue

        return list(sig_map.values())

    def _deduplicate_with_llm(self, issues: list[Issue]) -> list[Issue]:
        """Use LLM to cluster semantically similar issues."""
        # Process in batches of 30 to stay within token limits
        batch_size = 30
        all_deduped: list[Issue] = []

        for batch_start in range(0, len(issues), batch_size):
            batch = issues[batch_start:batch_start + batch_size]
            if len(batch) <= 1:
                all_deduped.extend(batch)
                continue

            issue_summaries = []
            for i, issue in enumerate(batch):
                issue_summaries.append(
                    f'{i}: [{issue.severity.value}] "{issue.title}" '
                    f'(agent={issue.agent}, category={issue.category.value}, '
                    f'confidence={issue.confidence})'
                )

            prompt = DEDUP_PROMPT.format(
                issue_list="\n".join(issue_summaries),
            )

            data = self.llm.complete_json(prompt, system=DEDUP_SYSTEM_PROMPT, tier="fast")
            clusters = data.get("clusters", [])

            if not clusters:
                # LLM returned no clusters — keep all
                all_deduped.extend(batch)
                continue

            used_indices: set[int] = set()
            for cluster in clusters:
                indices = cluster.get("indices", [])
                if not indices:
                    continue

                # Validate indices
                valid_indices = [idx for idx in indices if 0 <= idx < len(batch)]
                if not valid_indices:
                    continue

                # Pick the highest-confidence issue as the representative
                cluster_issues = [batch[idx] for idx in valid_indices]
                best = max(cluster_issues, key=lambda x: x.confidence)

                # Annotate with other agents who found similar issues
                other_agents = [
                    ci.agent for ci in cluster_issues
                    if ci.id != best.id and ci.agent != best.agent
                ]
                for agent_name in other_agents:
                    best.observed_facts.append(f"Also reported by agent: {agent_name}")

                all_deduped.append(best)
                used_indices.update(valid_indices)

            # Add any issues not included in any cluster
            for i, issue in enumerate(batch):
                if i not in used_indices:
                    all_deduped.append(issue)

        return all_deduped

    @staticmethod
    def _deduplicate_by_title(issues: list[Issue]) -> list[Issue]:
        """Fallback dedup: group by normalized title string."""
        if not issues:
            return []

        seen: dict[str, Issue] = {}
        for issue in issues:
            key = issue.title.lower().strip()
            if key in seen:
                if issue.confidence > seen[key].confidence:
                    existing_agent = seen[key].agent
                    issue.observed_facts.append(
                        f"Also reported by agent: {existing_agent}"
                    )
                    seen[key] = issue
                else:
                    seen[key].observed_facts.append(
                        f"Also reported by agent: {issue.agent}"
                    )
            else:
                seen[key] = issue

        return list(seen.values())
