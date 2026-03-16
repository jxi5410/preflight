"""Core data schemas for Preflight.

All structured data models: product intent, personas, issues, evidence, reports.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class Platform(str, Enum):
    web = "web"
    mobile_web = "mobile_web"
    mobile_app = "mobile_app"


class IssueCategory(str, Enum):
    functional = "functional"
    ux = "ux"
    ui = "ui"
    performance = "performance"
    trust = "trust"
    institutional_trust = "institutional_trust"
    design = "design"
    accessibility = "accessibility"
    copy = "copy"
    auth = "auth"  # Login/auth flow issues
    responsive = "responsive"  # Mobile responsiveness issues


class InstitutionalRelevance(str, Enum):
    none = "none"
    low = "low"
    moderate = "moderate"
    high = "high"


# ---------------------------------------------------------------------------
# Run Configuration
# ---------------------------------------------------------------------------

class Credentials(BaseModel):
    email: str | None = None
    password: str | None = None
    token: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Everything needed to invoke a run."""
    target_url: str
    repo_url: str | None = None
    github_token_env: str = "GITHUB_TOKEN"
    mobile_target: str | None = None
    credentials: Credentials | None = None
    brief: str | None = None
    persona_hints: list[str] = Field(default_factory=list)
    focus_flows: list[str] = Field(default_factory=list)
    design_guidance: str | None = None
    institutional_review: str = "auto"  # auto | on | off
    design_review: bool = True
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    llm_tier: str = "balanced"  # balanced | budget | premium | openai
    output_dir: str = "./artifacts"


# ---------------------------------------------------------------------------
# Repo Insights
# ---------------------------------------------------------------------------

class RepoInsights(BaseModel):
    """Structured understanding extracted from a GitHub repository."""
    product_name: str = ""
    description: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    claimed_features: list[str] = Field(default_factory=list)
    routes_or_pages: list[str] = Field(default_factory=list)
    recent_changes: list[str] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)
    configuration_hints: list[str] = Field(default_factory=list)
    documentation_summary: str = ""
    repo_confidence: float = 0.0
    is_public: bool | None = None  # None = unknown, True = public, False = private


class FeatureExpectation(BaseModel):
    """A feature the product claims to have, with verification status."""
    feature_name: str
    source: str = ""  # e.g. "README", "CHANGELOG", "docs/", "GitHub issue"
    verified: bool | None = None  # None = not yet checked, True/False = checked


# ---------------------------------------------------------------------------
# Product Intent Model
# ---------------------------------------------------------------------------

class ProductIntentModel(BaseModel):
    """What the system infers the product is and does."""
    product_name: str = ""
    product_type: str = ""
    target_audience: list[str] = Field(default_factory=list)
    primary_jobs: list[str] = Field(default_factory=list)
    user_expectations: list[str] = Field(default_factory=list)
    critical_journeys: list[str] = Field(default_factory=list)
    trust_sensitive_actions: list[str] = Field(default_factory=list)
    institutional_relevance: InstitutionalRelevance = InstitutionalRelevance.none
    institutional_reasoning: str = ""
    input_first: bool = False
    input_type: str = ""  # search, prompt, url, code, data, free_text
    input_placeholder: str = ""
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    raw_signals: dict[str, Any] = Field(default_factory=dict)
    repo_insights: RepoInsights | None = None
    feature_expectations: list[FeatureExpectation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Emotional State
# ---------------------------------------------------------------------------

class EmotionalState(BaseModel):
    """Tracks a persona's emotional state during a session."""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="How confident the persona feels about what to do next")
    frustration: float = Field(default=0.0, ge=0.0, le=1.0, description="Accumulated frustration from confusion, errors, dead ends")
    trust: float = Field(default=0.5, ge=0.0, le=1.0, description="How much the persona trusts this product")
    engagement: float = Field(default=0.7, ge=0.0, le=1.0, description="How interested/invested the persona is in continuing")
    delight: float = Field(default=0.0, ge=0.0, le=1.0, description="Positive surprise or satisfaction moments")


class EmotionalEvent(BaseModel):
    """A moment where the persona's emotional state shifted."""
    step_index: int
    trigger: str  # What caused the shift (e.g., "confusing label", "fast load time", "error message")
    dimension: str  # Which emotion changed (confidence, frustration, trust, engagement, delight)
    old_value: float
    new_value: float
    persona_thought: str  # What the persona "thought" at this moment, in first person


# ---------------------------------------------------------------------------
# Cognitive Behavior
# ---------------------------------------------------------------------------

class CognitiveBehavior(BaseModel):
    """How this persona thinks and behaves when using a product."""
    attention_span: str = Field(default="skimmer", description="'scanner' (reads headlines only), 'skimmer' (reads first lines), 'reader' (reads everything)")
    patience_threshold: int = Field(default=3, description="How many confusing/failed steps before the persona considers abandoning")
    exploration_style: str = Field(default="linear", description="'linear' (follows obvious path), 'curious' (clicks around), 'goal-driven' (goes straight for target)")
    error_tolerance: str = Field(default="medium", description="'low' (one error and they're suspicious), 'medium' (a couple errors are ok), 'high' (keeps trying)")
    jargon_comfort: str = Field(default="some", description="'none' (confused by technical terms), 'some' (knows basics), 'fluent' (expects precision)")
    comparison_anchors: list[str] = Field(default_factory=list, description="Products this persona has used before and will compare against, e.g. ['Notion', 'Slack']")


# ---------------------------------------------------------------------------
# Agent / Persona
# ---------------------------------------------------------------------------

class SeedInput(BaseModel):
    """A contextually appropriate input for a persona to try."""
    input_text: str
    purpose: str
    expected_outcome: str
    is_edge_case: bool = False


class AgentPersona(BaseModel):
    """A dynamically generated user agent."""
    id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    name: str
    role: str
    persona_type: str  # e.g. first_time_user, power_user, risk_compliance_reviewer
    goals: list[str] = Field(default_factory=list)
    expectations: list[str] = Field(default_factory=list)
    patience_level: str = "moderate"  # low | moderate | high
    expertise_level: str = "intermediate"  # novice | intermediate | expert
    behavioral_style: str = ""
    device_preference: Platform = Platform.web
    assigned_journeys: list[str] = Field(default_factory=list)
    seed_inputs: list[SeedInput] = Field(default_factory=list)
    cognitive_behavior: CognitiveBehavior = Field(default_factory=CognitiveBehavior)
    emotional_state: EmotionalState = Field(default_factory=EmotionalState)
    emotional_timeline: list[EmotionalEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actions (deterministic interaction engine)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """A single deterministic browser action planned by the LLM."""
    type: str  # navigate, click, fill_form, search, scroll, wait_for, screenshot, go_back
    target: str = ""  # accessibility label, URL, or field name
    value: str | None = None  # For fill_form/search: the text to enter
    reason: str = ""  # Why the persona is doing this


# ---------------------------------------------------------------------------
# Page Snapshot
# ---------------------------------------------------------------------------

class PageSnapshot(BaseModel):
    """Structured capture of a page state at a point in time."""
    url: str = ""
    title: str = ""
    accessibility_tree: str = ""  # Serialized accessibility tree
    screenshot_base64: str = ""  # Base64-encoded screenshot PNG
    screenshot_path: str = ""  # Path to saved screenshot file
    console_errors: list[str] = Field(default_factory=list)
    network_error_count: int = 0
    load_time_ms: int = 0
    lcp_ms: float | None = None  # Largest Contentful Paint
    cls_score: float | None = None  # Cumulative Layout Shift
    page_text: str = ""  # Visible text content (fallback when a11y tree unavailable)


# ---------------------------------------------------------------------------
# Journey Step
# ---------------------------------------------------------------------------

class JourneyStep(BaseModel):
    """One step in a multi-step journey execution."""
    step_number: int
    action: Action
    snapshot_before: PageSnapshot | None = None
    snapshot_after: PageSnapshot | None = None
    screenshot_path: str = ""
    issues_found: list[str] = Field(default_factory=list)  # Issue IDs
    persona_reaction: str = ""
    confidence_level: float = 0.5


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class ScreenshotEvidence(BaseModel):
    """A screenshot with contextual metadata."""
    path: str  # File path or filename
    caption: str = ""  # Human-readable description of what this shows
    step_ref: str = ""  # e.g. "step-3"
    timestamp_ms: int = 0  # When captured relative to journey start
    viewport: str = ""  # e.g. "1440x900" or "390x844"


class Evidence(BaseModel):
    screenshots: list[str] = Field(default_factory=list)
    screenshot_evidence: list[ScreenshotEvidence] = Field(default_factory=list)
    trace: str | None = None
    logs: list[str] = Field(default_factory=list)
    har: str | None = None
    video: str | None = None


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    """A single finding from an evaluation run."""
    id: str = Field(default_factory=lambda: f"ISS-{uuid.uuid4().hex[:6].upper()}")
    title: str
    severity: Severity = Severity.medium
    confidence: float = 0.8
    platform: Platform = Platform.web
    category: IssueCategory = IssueCategory.functional
    agent: str = ""
    user_impact: str = ""
    repro_steps: list[str] = Field(default_factory=list)
    expected: str = ""
    actual: str = ""
    observed_facts: list[str] = Field(default_factory=list)
    inferred_judgment: str = ""
    hypotheses: list[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)
    likely_product_area: str = ""
    repair_brief: str = ""
    error_signature: str = ""  # Dedup key: normalized fingerprint of the issue
    group_id: str = ""  # ID of the IssueGroup this belongs to


class IssueGroup(BaseModel):
    """A cluster of related issues grouped by theme or root cause."""
    id: str = Field(default_factory=lambda: f"GRP-{uuid.uuid4().hex[:6].upper()}")
    title: str  # Human-readable group name
    description: str = ""
    category: IssueCategory = IssueCategory.functional
    severity: Severity = Severity.medium  # Highest severity in group
    issue_ids: list[str] = Field(default_factory=list)
    product_area: str = ""
    issue_count: int = 0


# ---------------------------------------------------------------------------
# Institutional Checklist
# ---------------------------------------------------------------------------

class ChecklistResult(BaseModel):
    """Result of a single institutional checklist verification."""
    check_name: str
    status: str = "not_checked"  # pass | fail | not_applicable | not_checked
    evidence: list[str] = Field(default_factory=list)
    details: str = ""
    severity_if_failed: str = "medium"  # Severity to assign if status is "fail"


class ProvenanceScore(BaseModel):
    """Provenance scoring for a product output."""
    output_name: str
    score: int = 0  # 0 (no provenance) to 5 (full source trail with timestamps)
    sources_cited: bool = False
    sources_specific: bool = False
    freshness_shown: bool = False
    evidence: list[str] = Field(default_factory=list)
    details: str = ""


class TrustSignal(BaseModel):
    """A single trust signal check result."""
    signal_name: str
    present: bool | None = None  # None = not checked
    details: str = ""
    evidence: list[str] = Field(default_factory=list)


class TrustScorecard(BaseModel):
    """Aggregate trust signal inventory."""
    signals: list[TrustSignal] = Field(default_factory=list)
    overall_score: float = 0.0  # 0.0 to 1.0
    summary: str = ""


# ---------------------------------------------------------------------------
# Coverage Map
# ---------------------------------------------------------------------------

class CoverageEntry(BaseModel):
    url: str = ""
    screen_name: str = ""
    agent_id: str = ""
    flow: str = ""
    status: str = "pending"  # pending | visited | failed | skipped
    issues_found: int = 0
    timestamp: datetime | None = None


class CoverageMap(BaseModel):
    entries: list[CoverageEntry] = Field(default_factory=list)

    def visited_urls(self) -> set[str]:
        return {e.url for e in self.entries if e.status == "visited"}

    def failed_urls(self) -> set[str]:
        return {e.url for e in self.entries if e.status == "failed"}

    def pending_flows(self) -> list[str]:
        return list({e.flow for e in self.entries if e.status == "pending"})


# ---------------------------------------------------------------------------
# Run Result
# ---------------------------------------------------------------------------

class FixOption(BaseModel):
    """One possible approach to fixing an issue."""
    approach: str  # Short label, e.g. "Quick patch" or "Proper refactor"
    description: str = ""  # What to do
    trade_offs: str = ""  # Pros/cons of this approach
    estimated_effort: str = "moderate"  # quick_fix | moderate | significant


class HandoffTask(BaseModel):
    """A single actionable task for an AI coding tool."""
    task_number: int
    issue_id: str
    severity: str
    title: str
    description: str = ""
    likely_files: list[str] = Field(default_factory=list)
    repro_steps: list[str] = Field(default_factory=list)
    expected_behavior: str = ""
    fix_guidance: str = ""
    fix_options: list[FixOption] = Field(default_factory=list)
    verification: str = ""
    evidence_screenshots: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)
    blocks: list[int] = Field(default_factory=list)
    estimated_complexity: str = "moderate"  # quick_fix | moderate | significant


class FeatureGap(BaseModel):
    """A feature claimed in repo but not found in UI."""
    feature: str
    source: str  # Where the claim comes from
    claim: str = ""  # What was claimed
    ui_status: str = "not_found"  # not_found | partial | different


class Handoff(BaseModel):
    """Complete handoff package for an AI coding tool."""
    handoff_version: str = "1.0"
    run_id: str
    product_name: str
    repo_url: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    target_url: str
    tasks: list[HandoffTask] = Field(default_factory=list)
    feature_gaps: list[FeatureGap] = Field(default_factory=list)
    total_estimated_hours: str = ""
    summary: str = ""


class RunResult(BaseModel):
    """Complete output of a single evaluation run."""
    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    completed_at: datetime | None = None
    config: RunConfig
    intent_model: ProductIntentModel = Field(default_factory=ProductIntentModel)
    agents: list[AgentPersona] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    issue_groups: list[IssueGroup] = Field(default_factory=list)
    coverage: CoverageMap = Field(default_factory=CoverageMap)
    summary: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
