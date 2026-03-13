"""Institutional Lens — Governance, provenance, and auditability review.

Evaluates whether a product meets the standards a serious professional or
institution would require: source verification, provenance, data freshness,
audit trails, role separation, and governance controls.

Operates only on visible product surfaces and captured artifacts.
Uses a structured checklist for deterministic verification, with LLM for
nuanced judgment where needed.
"""

from __future__ import annotations

import logging
from typing import Any

from humanqa.core.llm import LLMClient
from humanqa.core.schemas import (
    ChecklistResult,
    Evidence,
    InstitutionalRelevance,
    Issue,
    IssueCategory,
    Platform,
    ProductIntentModel,
    ProvenanceScore,
    RunResult,
    Severity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured checklist definitions
# ---------------------------------------------------------------------------

INSTITUTIONAL_CHECKLIST = [
    {
        "name": "audit_trail",
        "label": "Audit trail visible",
        "search_terms": ["history", "activity", "log", "audit", "changelog", "recent changes"],
        "severity_if_failed": "high",
        "description": "Look for elements indicating activity history, audit logs, or change records",
    },
    {
        "name": "version_history",
        "label": "Version history",
        "search_terms": ["version", "revision", "v1", "v2", "draft", "published"],
        "severity_if_failed": "medium",
        "description": "Look for versioning, revision indicators, or draft/published status",
    },
    {
        "name": "source_attribution",
        "label": "Source attribution",
        "search_terms": ["source", "citation", "reference", "cited", "from", "based on", "powered by"],
        "severity_if_failed": "high",
        "description": "On key outputs, check for source citations, attribution, or reference links",
    },
    {
        "name": "data_freshness",
        "label": "Data freshness markers",
        "search_terms": ["updated", "as of", "last modified", "refreshed", "ago", "timestamp"],
        "severity_if_failed": "medium",
        "description": "Look for timestamps, update indicators, or freshness markers on data",
    },
    {
        "name": "role_indicators",
        "label": "Role indicators",
        "search_terms": ["admin", "role", "permission", "viewer", "editor", "owner", "member"],
        "severity_if_failed": "medium",
        "description": "Look for user role display, permissions UI, or access level indicators",
    },
    {
        "name": "confirmation_dialogs",
        "label": "Confirmation dialogs for risky actions",
        "search_terms": ["confirm", "are you sure", "cannot be undone", "delete", "remove", "cancel"],
        "severity_if_failed": "high",
        "description": "Verify that destructive/risky actions are gated by confirmation dialogs",
    },
    {
        "name": "error_quality",
        "label": "Error message quality",
        "search_terms": ["error", "failed", "try again", "something went wrong", "oops"],
        "severity_if_failed": "medium",
        "description": "Check whether error messages are helpful and actionable",
    },
    {
        "name": "privacy_indicators",
        "label": "Privacy indicators",
        "search_terms": ["privacy", "privacy policy", "data handling", "gdpr", "cookie", "consent"],
        "severity_if_failed": "medium",
        "description": "Look for privacy policy links, data handling disclosures, consent mechanisms",
    },
    {
        "name": "export_capability",
        "label": "Export/download capability",
        "search_terms": ["export", "download", "csv", "pdf", "save", "backup"],
        "severity_if_failed": "low",
        "description": "Verify that users can extract/export their data",
    },
]

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

INSTITUTIONAL_SYSTEM = """You are an institutional/governance reviewer for HumanQA.

You evaluate whether a product is trustworthy enough for professional and institutional use.
You look at the product ONLY through its visible UI — never source code.

## EVIDENCE ANCHORING (MANDATORY)

Every finding MUST cite specific evidence. Findings without anchored evidence will be rejected.

Each finding must reference at least ONE of:
- **Element reference**: A specific UI element or section by name/role
- **Observed absence**: An explicit search that found nothing
- **Screenshot reference**: A specific page description
- **Existing issue reference**: Corroboration from other findings

Respond with JSON: {"institutional_issues": [...], "institutional_strengths": [...],
"readiness_level": "not_ready|early|developing|mature", "readiness_summary": "..."}"""

INSTITUTIONAL_PROMPT = """Conduct an institutional/governance review of this product.

## Product
{product_name} ({product_type})
Institutional relevance: {institutional_relevance}
Reasoning: {institutional_reasoning}

## Structured Checklist Results
{checklist_results}

## Trust-sensitive actions identified
{trust_actions}

## Issues already found (for context)
{existing_issues}

## Page descriptions from evaluation
{page_descriptions}

## Accessibility tree content from visited pages
{a11y_content}

The structured checklist above has already verified presence/absence of key elements.
Focus your review on:
1. Issues the checklist flagged as "fail" — provide deeper analysis
2. Governance gaps the checklist couldn't detect (workflow issues, contextual problems)
3. Overall institutional readiness assessment

For each institutional issue, provide:
- title: Clear issue title
- severity: critical | high | medium | low | info
- confidence: 0.0-1.0
- subcategory: source_provenance | data_integrity | auditability | governance_control | professional_trust
- user_impact: How this affects an institutional user
- observed_facts: What you literally see or don't see (list)
- inferred_judgment: Your governance assessment
- hypotheses: Possible explanations (list)
- likely_product_area: Where in the product
- repair_brief: What to fix for institutional readiness

Also provide:
- institutional_strengths: What governance aspects work well
- readiness_level: "not_ready" | "early" | "developing" | "mature"
- readiness_summary: 2-3 sentence assessment"""

PROVENANCE_SYSTEM = """You are a provenance scoring engine for HumanQA.

For each product output area, evaluate how well the product attributes sources,
shows data freshness, and distinguishes facts from inferences.

## EVIDENCE ANCHORING (MANDATORY)
Every score must cite specific UI elements or observed absences.

Score scale (0-5):
0: No provenance at all — outputs appear with no source, date, or attribution
1: Minimal — timestamps exist but no source attribution
2: Basic — some outputs cite sources but inconsistently
3: Good — most outputs cite sources, timestamps present
4: Strong — sources cited with specificity, freshness shown, fact/inference distinction
5: Full provenance — complete source trail, timestamps, confidence indicators, methodology shown

Respond with JSON: {"outputs": [{"output_name": "...", "score": 0-5, "sources_cited": bool,
"sources_specific": bool, "freshness_shown": bool, "evidence": [...], "details": "..."}]}"""

PROVENANCE_PROMPT = """Score the provenance quality for this product's key outputs.

## Product: {product_name} ({product_type})

## Page content and accessibility tree
{page_content}

## Coverage
{coverage_summary}

Identify the product's key output areas (dashboards, reports, generated content,
recommendations, data displays) and score each for provenance quality."""

GOVERNANCE_SYSTEM = """You are a governance flow tester for HumanQA.

For each risky action identified, evaluate whether proper governance controls exist:
- Confirmation dialog before destructive actions
- Undo/recovery option
- Role/permission checks
- Approval workflows for sensitive operations

## EVIDENCE ANCHORING (MANDATORY)
Every finding must cite specific UI elements or observed absences from the accessibility tree.

Respond with JSON: {"governance_results": [{"action": "...", "has_confirmation": bool,
"has_undo": bool, "has_role_check": bool, "has_approval_flow": bool, "evidence": [...],
"missing_gates": [...], "severity": "...", "details": "..."}]}"""

GOVERNANCE_PROMPT = """Test governance controls for risky actions in this product.

## Product: {product_name} ({product_type})

## Trust-sensitive actions identified
{trust_actions}

## Accessibility tree content from visited pages
{a11y_content}

## Page descriptions
{page_descriptions}

## Existing issues (for context)
{existing_issues}

For each trust-sensitive action, evaluate what governance controls are visible in the UI.
If no trust-sensitive actions were identified, look for common risky patterns:
delete, remove, export, payment, admin, settings changes."""


class InstitutionalLens:
    """Governance, provenance, and auditability review.

    Uses a structured checklist for deterministic verification,
    provenance scoring for data outputs, and governance flow testing
    for risky actions. LLM provides nuanced judgment on top.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def should_run(self, intent: ProductIntentModel, override: str = "auto") -> bool:
        """Determine if institutional review is relevant."""
        if override == "on":
            return True
        if override == "off":
            return False
        return intent.institutional_relevance in (
            InstitutionalRelevance.moderate,
            InstitutionalRelevance.high,
        )

    async def review(self, run_result: RunResult) -> list[Issue]:
        """Run full institutional review: checklist + provenance + governance + LLM."""
        intent = run_result.intent_model

        if not self.should_run(intent, run_result.config.institutional_review):
            logger.info(
                "Institutional review skipped (relevance: %s)",
                intent.institutional_relevance.value,
            )
            return []

        all_issues: list[Issue] = []

        # Gather page content for analysis
        a11y_content = self._gather_a11y_content(run_result)
        page_descs = self._gather_page_descriptions(run_result)

        # Step 1: Run structured checklist
        checklist_results = self.run_checklist(a11y_content, page_descs)
        checklist_issues = self._checklist_to_issues(checklist_results)
        all_issues.extend(checklist_issues)
        logger.info(
            "Institutional checklist: %d/%d checks failed",
            sum(1 for r in checklist_results if r.status == "fail"),
            len(checklist_results),
        )

        # Step 2: Provenance scoring
        provenance_scores = await self.score_provenance(intent, a11y_content, run_result)
        provenance_issues = self._provenance_to_issues(provenance_scores)
        all_issues.extend(provenance_issues)

        # Step 3: Governance flow testing
        governance_issues = await self.test_governance_flows(
            intent, a11y_content, page_descs, run_result,
        )
        all_issues.extend(governance_issues)

        # Step 4: LLM-based nuanced review (builds on checklist results)
        llm_issues = await self._llm_review(
            intent, run_result, checklist_results, a11y_content, page_descs,
        )
        all_issues.extend(llm_issues)

        # Store scores
        self._update_scores(run_result, checklist_results, provenance_scores)

        return all_issues

    # ------------------------------------------------------------------
    # Structured Checklist
    # ------------------------------------------------------------------

    def run_checklist(
        self,
        a11y_content: str,
        page_descriptions: str,
    ) -> list[ChecklistResult]:
        """Run the structured institutional checklist against page content.

        Searches accessibility trees and page descriptions for indicators
        of each governance capability. This is deterministic, not LLM-based.
        """
        combined = (a11y_content + "\n" + page_descriptions).lower()
        results: list[ChecklistResult] = []

        for check in INSTITUTIONAL_CHECKLIST:
            found_terms: list[str] = []
            for term in check["search_terms"]:
                if term.lower() in combined:
                    found_terms.append(term)

            if found_terms:
                results.append(ChecklistResult(
                    check_name=check["name"],
                    status="pass",
                    evidence=[f"Found indicator(s): {', '.join(found_terms)}"],
                    details=check["description"],
                    severity_if_failed=check["severity_if_failed"],
                ))
            else:
                results.append(ChecklistResult(
                    check_name=check["name"],
                    status="fail",
                    evidence=[
                        f"Searched for: {', '.join(check['search_terms'])}",
                        "None found in any visited page's accessibility tree or content",
                    ],
                    details=check["description"],
                    severity_if_failed=check["severity_if_failed"],
                ))

        return results

    def _checklist_to_issues(self, results: list[ChecklistResult]) -> list[Issue]:
        """Convert failed checklist items into Issue objects."""
        issues: list[Issue] = []
        for r in results:
            if r.status != "fail":
                continue

            label = next(
                (c["label"] for c in INSTITUTIONAL_CHECKLIST if c["name"] == r.check_name),
                r.check_name,
            )

            try:
                severity = Severity(r.severity_if_failed)
            except ValueError:
                severity = Severity.medium

            issues.append(Issue(
                title=f"Institutional check failed: {label}",
                severity=severity,
                confidence=0.9,
                platform=Platform.web,
                category=IssueCategory.institutional_trust,
                agent="institutional_checklist",
                user_impact=f"Institutional users expect {label.lower()} to be present",
                observed_facts=r.evidence,
                inferred_judgment=f"No indicators of {label.lower()} found in any visited page",
                likely_product_area="Governance",
                repair_brief=r.details,
            ))

        return issues

    # ------------------------------------------------------------------
    # Provenance Scoring
    # ------------------------------------------------------------------

    async def score_provenance(
        self,
        intent: ProductIntentModel,
        a11y_content: str,
        run_result: RunResult,
    ) -> list[ProvenanceScore]:
        """Score provenance quality for the product's key outputs."""
        coverage_summary = f"{len(run_result.coverage.entries)} pages visited"

        prompt = PROVENANCE_PROMPT.format(
            product_name=intent.product_name,
            product_type=intent.product_type,
            page_content=a11y_content[:6000],
            coverage_summary=coverage_summary,
        )

        try:
            data = self.llm.complete_json(prompt, system=PROVENANCE_SYSTEM)
            scores: list[ProvenanceScore] = []
            for raw in data.get("outputs", []):
                scores.append(ProvenanceScore(
                    output_name=raw.get("output_name", "Unknown"),
                    score=max(0, min(5, raw.get("score", 0))),
                    sources_cited=raw.get("sources_cited", False),
                    sources_specific=raw.get("sources_specific", False),
                    freshness_shown=raw.get("freshness_shown", False),
                    evidence=raw.get("evidence", []),
                    details=raw.get("details", ""),
                ))
            logger.info("Provenance scoring: %d outputs scored", len(scores))
            return scores
        except Exception as e:
            logger.warning("Provenance scoring failed: %s", e)
            return []

    def _provenance_to_issues(self, scores: list[ProvenanceScore]) -> list[Issue]:
        """Convert low provenance scores into Issue objects."""
        issues: list[Issue] = []
        for s in scores:
            if s.score >= 3:
                continue

            if s.score <= 1:
                severity = Severity.high
            else:
                severity = Severity.medium

            issues.append(Issue(
                title=f"Low provenance score ({s.score}/5): {s.output_name}",
                severity=severity,
                confidence=0.85,
                platform=Platform.web,
                category=IssueCategory.institutional_trust,
                agent="provenance_scorer",
                user_impact=(
                    f"The '{s.output_name}' output area lacks adequate source attribution. "
                    f"Sources cited: {s.sources_cited}, specific: {s.sources_specific}, "
                    f"freshness shown: {s.freshness_shown}."
                ),
                observed_facts=s.evidence,
                inferred_judgment=s.details,
                likely_product_area=s.output_name,
                repair_brief=(
                    f"Add source attribution and freshness indicators to {s.output_name}. "
                    f"Current score: {s.score}/5."
                ),
            ))

        return issues

    # ------------------------------------------------------------------
    # Governance Flow Testing
    # ------------------------------------------------------------------

    async def test_governance_flows(
        self,
        intent: ProductIntentModel,
        a11y_content: str,
        page_descriptions: str,
        run_result: RunResult,
    ) -> list[Issue]:
        """Test governance controls for trust-sensitive actions."""
        existing_summary = "\n".join(
            f"- [{i.severity.value}] {i.title}"
            for i in run_result.issues[:15]
        )

        prompt = GOVERNANCE_PROMPT.format(
            product_name=intent.product_name,
            product_type=intent.product_type,
            trust_actions="\n".join(
                f"- {a}" for a in intent.trust_sensitive_actions
            ) or "(none identified — look for common risky patterns)",
            a11y_content=a11y_content[:5000],
            page_descriptions=page_descriptions,
            existing_issues=existing_summary or "(none yet)",
        )

        try:
            data = self.llm.complete_json(prompt, system=GOVERNANCE_SYSTEM)
            issues: list[Issue] = []

            for result in data.get("governance_results", []):
                missing = result.get("missing_gates", [])
                if not missing:
                    continue

                sev_str = result.get("severity", "medium")
                try:
                    severity = Severity(sev_str)
                except ValueError:
                    severity = Severity.medium

                issues.append(Issue(
                    title=f"Missing governance gate: {result.get('action', 'risky action')}",
                    severity=severity,
                    confidence=0.8,
                    platform=Platform.web,
                    category=IssueCategory.institutional_trust,
                    agent="governance_tester",
                    user_impact=(
                        f"The action '{result.get('action', '')}' lacks: {', '.join(missing)}"
                    ),
                    observed_facts=result.get("evidence", []),
                    inferred_judgment=result.get("details", ""),
                    likely_product_area="Governance",
                    repair_brief=f"Add {', '.join(missing)} for '{result.get('action', '')}'",
                ))

            logger.info("Governance testing: %d missing gates found", len(issues))
            return issues

        except Exception as e:
            logger.warning("Governance flow testing failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # LLM-based nuanced review
    # ------------------------------------------------------------------

    async def _llm_review(
        self,
        intent: ProductIntentModel,
        run_result: RunResult,
        checklist_results: list[ChecklistResult],
        a11y_content: str,
        page_descriptions: str,
    ) -> list[Issue]:
        """Run LLM-based institutional review, informed by checklist results."""
        existing_summary = "\n".join(
            f"- [{i.severity.value}] {i.title} ({i.category.value})"
            for i in run_result.issues[:20]
        )

        checklist_text = self._format_checklist_results(checklist_results)

        prompt = INSTITUTIONAL_PROMPT.format(
            product_name=intent.product_name,
            product_type=intent.product_type,
            institutional_relevance=intent.institutional_relevance.value,
            institutional_reasoning=intent.institutional_reasoning,
            checklist_results=checklist_text,
            trust_actions="\n".join(
                f"- {a}" for a in intent.trust_sensitive_actions
            ) or "(none)",
            existing_issues=existing_summary or "(none yet)",
            page_descriptions=page_descriptions,
            a11y_content=a11y_content[:4000],
        )

        try:
            data = self.llm.complete_json(prompt, system=INSTITUTIONAL_SYSTEM)

            issues = []
            for raw in data.get("institutional_issues", []):
                sev = raw.get("severity", "medium")
                try:
                    severity = Severity(sev)
                except ValueError:
                    severity = Severity.medium

                issues.append(Issue(
                    title=raw.get("title", "Institutional issue"),
                    severity=severity,
                    confidence=raw.get("confidence", 0.7),
                    platform=Platform.web,
                    category=IssueCategory.institutional_trust,
                    agent="institutional_lens",
                    user_impact=raw.get("user_impact", ""),
                    observed_facts=raw.get("observed_facts", []),
                    inferred_judgment=raw.get("inferred_judgment", ""),
                    hypotheses=raw.get("hypotheses", []),
                    likely_product_area=raw.get("likely_product_area", ""),
                    repair_brief=raw.get("repair_brief", ""),
                ))

            readiness = data.get("readiness_level", "unknown")
            logger.info(
                "LLM institutional review: %d issues, readiness=%s",
                len(issues), readiness,
            )

            # Store readiness
            readiness_scores = {
                "not_ready": 0.0, "early": 0.25,
                "developing": 0.5, "mature": 0.85,
            }
            run_result.scores["institutional_readiness"] = readiness_scores.get(readiness, 0.0)
            run_result.scores["institutional_readiness_label"] = readiness

            return issues

        except Exception as e:
            logger.error("LLM institutional review failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_a11y_content(run_result: RunResult) -> str:
        """Extract accessibility tree content from run coverage and issues."""
        parts: list[str] = []
        for entry in run_result.coverage.entries[:20]:
            parts.append(f"Page: {entry.screen_name or entry.url} (status: {entry.status})")
        # Include observed facts from issues as additional content
        for issue in run_result.issues[:30]:
            for fact in issue.observed_facts:
                parts.append(fact)
        return "\n".join(parts) if parts else "(no accessibility content captured)"

    @staticmethod
    def _gather_page_descriptions(run_result: RunResult) -> str:
        """Build page descriptions from coverage entries."""
        return "\n".join(
            f"- {e.screen_name or e.url} (status: {e.status}, issues: {e.issues_found})"
            for e in run_result.coverage.entries[:20]
        ) or "(none)"

    @staticmethod
    def _format_checklist_results(results: list[ChecklistResult]) -> str:
        """Format checklist results for inclusion in LLM prompt."""
        lines: list[str] = []
        for r in results:
            label = next(
                (c["label"] for c in INSTITUTIONAL_CHECKLIST if c["name"] == r.check_name),
                r.check_name,
            )
            status_icon = {"pass": "PASS", "fail": "FAIL", "not_applicable": "N/A"}.get(
                r.status, "?"
            )
            evidence_str = "; ".join(r.evidence) if r.evidence else ""
            lines.append(f"[{status_icon}] {label}: {evidence_str}")
        return "\n".join(lines)

    @staticmethod
    def _update_scores(
        run_result: RunResult,
        checklist_results: list[ChecklistResult],
        provenance_scores: list[ProvenanceScore],
    ) -> None:
        """Store institutional scores in run result."""
        total = len(checklist_results)
        passed = sum(1 for r in checklist_results if r.status == "pass")
        run_result.scores["institutional_checklist_total"] = float(total)
        run_result.scores["institutional_checklist_passed"] = float(passed)
        if total > 0:
            run_result.scores["institutional_checklist_ratio"] = passed / total

        if provenance_scores:
            avg_score = sum(s.score for s in provenance_scores) / len(provenance_scores)
            run_result.scores["provenance_avg_score"] = avg_score
