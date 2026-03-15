"""Tests for all refinement features.

Tests increments 1-10:
1. ScreenshotEvidence with captions
2. IssueGroup schema + clustering logic
3. Aggressive deduplication (error signatures)
4. Adaptive scoring (product-type-aware budgets)
5. HTML report improvements
6. Repo visibility detection
7. Handoff fix options
8. Login/auth flow evaluation
9. Mobile responsiveness evaluation
10. Run time optimizations
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from humanqa.core.schemas import (
    AgentPersona,
    CoverageEntry,
    CoverageMap,
    Evidence,
    FeatureExpectation,
    FixOption,
    Handoff,
    HandoffTask,
    Issue,
    IssueCategory,
    IssueGroup,
    Platform,
    ProductIntentModel,
    RepoInsights,
    RunConfig,
    RunResult,
    ScreenshotEvidence,
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
# 1. ScreenshotEvidence with captions
# ===========================================================================

class TestScreenshotEvidence:
    """Tests for ScreenshotEvidence schema."""

    def test_screenshot_evidence_fields(self):
        se = ScreenshotEvidence(
            path="screenshots/step-03.png",
            caption="Submit button has no hover state",
            step_ref="step-3",
            timestamp_ms=4500,
            viewport="1440x900",
        )
        assert se.path == "screenshots/step-03.png"
        assert se.caption == "Submit button has no hover state"
        assert se.step_ref == "step-3"
        assert se.timestamp_ms == 4500
        assert se.viewport == "1440x900"

    def test_screenshot_evidence_defaults(self):
        se = ScreenshotEvidence(path="shot.png")
        assert se.caption == ""
        assert se.step_ref == ""
        assert se.timestamp_ms == 0
        assert se.viewport == ""

    def test_evidence_has_screenshot_evidence_list(self):
        ev = Evidence(
            screenshots=["shot.png"],
            screenshot_evidence=[
                ScreenshotEvidence(path="shot.png", caption="Login page"),
            ],
        )
        assert len(ev.screenshot_evidence) == 1
        assert ev.screenshot_evidence[0].caption == "Login page"

    def test_evidence_backward_compat(self):
        ev = Evidence(screenshots=["a.png", "b.png"])
        assert ev.screenshot_evidence == []
        assert len(ev.screenshots) == 2

    def test_screenshot_evidence_roundtrip(self):
        se = ScreenshotEvidence(
            path="test.png",
            caption="Test caption",
            step_ref="step-5",
            viewport="390x844",
        )
        data = json.loads(se.model_dump_json())
        se2 = ScreenshotEvidence.model_validate(data)
        assert se2.caption == "Test caption"
        assert se2.viewport == "390x844"


# ===========================================================================
# 2. IssueGroup schema + clustering logic
# ===========================================================================

class TestIssueGroup:
    """Tests for IssueGroup schema and clustering."""

    def test_issue_group_fields(self):
        group = IssueGroup(
            title="UX issues in checkout",
            description="3 related UX issues in the checkout area",
            category=IssueCategory.ux,
            severity=Severity.high,
            issue_ids=["ISS-001", "ISS-002", "ISS-003"],
            product_area="checkout",
            issue_count=3,
        )
        assert group.id.startswith("GRP-")
        assert group.issue_count == 3
        assert group.severity == Severity.high

    def test_issue_group_defaults(self):
        group = IssueGroup(title="General")
        assert group.issue_ids == []
        assert group.issue_count == 0
        assert group.category == IssueCategory.functional

    def test_issue_has_group_id(self):
        issue = _make_issue("Bug", group_id="GRP-ABC123")
        assert issue.group_id == "GRP-ABC123"

    def test_issue_has_error_signature(self):
        issue = _make_issue("Bug", error_signature="functional:login:broken|button")
        assert issue.error_signature == "functional:login:broken|button"

    def test_group_issues_creates_groups(self):
        from humanqa.core.orchestrator import group_issues

        issues = [
            _make_issue("Bug A", category="ux", likely_product_area="checkout"),
            _make_issue("Bug B", category="ux", likely_product_area="checkout"),
            _make_issue("Bug C", category="functional", likely_product_area="login"),
        ]
        groups = group_issues(issues)
        # Should create one group for ux:checkout (2 issues)
        # functional:login has only 1 issue, no group
        assert len(groups) == 1
        assert groups[0].issue_count == 2
        assert groups[0].category == IssueCategory.ux
        # Issues should be linked to group
        assert issues[0].group_id == groups[0].id
        assert issues[1].group_id == groups[0].id
        assert issues[2].group_id == ""

    def test_group_issues_empty(self):
        from humanqa.core.orchestrator import group_issues

        assert group_issues([]) == []

    def test_group_issues_no_duplicates(self):
        from humanqa.core.orchestrator import group_issues

        issues = [
            _make_issue("A", category="ux", likely_product_area="a"),
            _make_issue("B", category="functional", likely_product_area="b"),
        ]
        groups = group_issues(issues)
        assert len(groups) == 0

    def test_run_result_has_issue_groups(self):
        result = _make_run_result(
            issue_groups=[IssueGroup(title="Test group")],
        )
        assert len(result.issue_groups) == 1


# ===========================================================================
# 3. Aggressive deduplication (error signatures)
# ===========================================================================

class TestErrorSignatureDedup:
    """Tests for error signature computation and signature-based dedup."""

    def test_compute_error_signature(self):
        from humanqa.core.orchestrator import compute_error_signature

        issue = _make_issue(
            "Submit button broken on checkout",
            category="functional",
            likely_product_area="checkout",
        )
        sig = compute_error_signature(issue)
        assert sig.startswith("functional:checkout:")
        assert "broken" in sig
        assert "button" in sig
        assert "checkout" in sig

    def test_same_issue_same_signature(self):
        from humanqa.core.orchestrator import compute_error_signature

        issue1 = _make_issue("Button broken", category="ux", likely_product_area="login")
        issue2 = _make_issue("Button broken", category="ux", likely_product_area="login")
        assert compute_error_signature(issue1) == compute_error_signature(issue2)

    def test_different_category_different_signature(self):
        from humanqa.core.orchestrator import compute_error_signature

        issue1 = _make_issue("Broken button", category="ux", likely_product_area="login")
        issue2 = _make_issue("Broken button", category="functional", likely_product_area="login")
        assert compute_error_signature(issue1) != compute_error_signature(issue2)

    def test_signature_dedup(self):
        from humanqa.core.orchestrator import Orchestrator

        issues = [
            _make_issue("Bug A", error_signature="sig1", agent="agent1", confidence=0.8),
            _make_issue("Bug A dup", error_signature="sig1", agent="agent2", confidence=0.9),
            _make_issue("Bug B", error_signature="sig2", agent="agent1"),
        ]
        deduped = Orchestrator._deduplicate_by_signature(issues)
        assert len(deduped) == 2
        # Higher confidence kept for sig1
        sig1_issue = [i for i in deduped if i.error_signature == "sig1"][0]
        assert sig1_issue.confidence == 0.9
        assert "Also reported by agent: agent1" in sig1_issue.observed_facts

    def test_signature_dedup_no_sig(self):
        from humanqa.core.orchestrator import Orchestrator

        issues = [
            _make_issue("Bug A", agent="a"),
            _make_issue("Bug B", agent="b"),
        ]
        # No signatures — each gets unique key, no merging
        deduped = Orchestrator._deduplicate_by_signature(issues)
        assert len(deduped) == 2


# ===========================================================================
# 4. Adaptive scoring (product-type-aware budgets)
# ===========================================================================

class TestAdaptiveScoring:
    """Tests for product-type-aware performance budgets."""

    def test_classify_saas(self):
        from humanqa.core.performance import classify_product_type

        assert classify_product_type("SaaS Dashboard") == "saas_app"
        assert classify_product_type("analytics platform") == "saas_app"

    def test_classify_marketing(self):
        from humanqa.core.performance import classify_product_type

        assert classify_product_type("Landing Page") == "marketing_site"
        assert classify_product_type("marketing website") == "marketing_site"

    def test_classify_ecommerce(self):
        from humanqa.core.performance import classify_product_type

        assert classify_product_type("ecommerce platform") == "ecommerce"
        assert classify_product_type("online store") == "ecommerce"
        assert classify_product_type("marketplace") == "ecommerce"

    def test_classify_content(self):
        from humanqa.core.performance import classify_product_type

        assert classify_product_type("documentation site") == "content_site"
        assert classify_product_type("blog") == "content_site"
        assert classify_product_type("wiki") == "content_site"

    def test_classify_mobile(self):
        from humanqa.core.performance import classify_product_type

        assert classify_product_type("mobile web app") == "mobile_web"

    def test_classify_default(self):
        from humanqa.core.performance import classify_product_type

        assert classify_product_type("unknown thing") == "default"

    def test_ecommerce_budget_stricter_cls(self):
        from humanqa.core.performance import BUDGETS

        ecom = BUDGETS["ecommerce"]
        default = BUDGETS["default"]
        # Ecommerce should have stricter CLS threshold
        assert ecom["cls_score"][1] < default["cls_score"][1]

    def test_content_budget_tighter_lcp(self):
        from humanqa.core.performance import BUDGETS

        content = BUDGETS["content_site"]
        default = BUDGETS["default"]
        assert content["lcp_ms"][0] < default["lcp_ms"][0]

    def test_score_explanation_includes_category(self):
        from humanqa.core.performance import score_explanation

        explanation = score_explanation("SaaS Dashboard")
        assert "saas_app" in explanation
        assert "SaaS" in explanation

    def test_score_explanation_default(self):
        from humanqa.core.performance import score_explanation

        explanation = score_explanation("unknown product")
        assert "default" in explanation.lower()


# ===========================================================================
# 5. HTML report improvements
# ===========================================================================

class TestHTMLReportImprovements:
    """Tests for HTML report clickable cards, inline screenshots, search."""

    def test_html_has_search_bar(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[_make_issue("Bug")])
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "search-bar" in html
        assert "Search issues" in html

    def test_html_has_clickable_cards(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[_make_issue("Bug", severity="critical")])
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "filterBySeverity" in html
        assert "onclick" in html

    def test_html_has_screenshot_gallery(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        ev = Evidence(
            screenshots=["shot.png"],
            screenshot_evidence=[
                ScreenshotEvidence(path="shot.png", caption="Login page broken"),
            ],
        )
        result = _make_run_result(issues=[_make_issue("Bug", evidence=ev)])
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "screenshot-gallery" in html
        assert "screenshot-caption" in html

    def test_html_has_score_explanations(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(
            issues=[_make_issue("Bug")],
            scores={"trust_score": 0.8},
        )
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "score-explanation" in html

    def test_html_has_group_chips(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(
            issues=[_make_issue("Bug")],
            issue_groups=[IssueGroup(title="UX in checkout", issue_count=3)],
        )
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "group-chip" in html
        assert "UX in checkout" in html

    def test_html_lightbox_has_caption(self, tmp_path):
        from humanqa.reporting.report_generator import ReportGenerator

        gen = ReportGenerator(output_dir=str(tmp_path))
        result = _make_run_result(issues=[_make_issue("Bug")])
        path = gen.generate_html(result)
        html = Path(path).read_text()
        assert "lightbox-caption" in html


# ===========================================================================
# 6. Repo visibility detection
# ===========================================================================

class TestRepoVisibility:
    """Tests for repo visibility detection."""

    def test_repo_insights_has_is_public(self):
        insights = RepoInsights(product_name="Test", is_public=True)
        assert insights.is_public is True

    def test_repo_insights_default_unknown(self):
        insights = RepoInsights(product_name="Test")
        assert insights.is_public is None

    def test_repo_insights_private(self):
        insights = RepoInsights(product_name="Test", is_public=False)
        assert insights.is_public is False

    def test_repo_insights_roundtrip(self):
        insights = RepoInsights(product_name="Test", is_public=True)
        data = json.loads(insights.model_dump_json())
        assert data["is_public"] is True
        insights2 = RepoInsights.model_validate(data)
        assert insights2.is_public is True


# ===========================================================================
# 7. Handoff fix options
# ===========================================================================

class TestHandoffFixOptions:
    """Tests for fix options in handoff tasks."""

    def test_fix_option_schema(self):
        opt = FixOption(
            approach="Quick patch",
            description="Fix the immediate symptom",
            trade_offs="Fast but might not address root cause",
            estimated_effort="quick_fix",
        )
        assert opt.approach == "Quick patch"
        assert opt.estimated_effort == "quick_fix"

    def test_fix_option_defaults(self):
        opt = FixOption(approach="Fix")
        assert opt.description == ""
        assert opt.trade_offs == ""
        assert opt.estimated_effort == "moderate"

    def test_handoff_task_has_fix_options(self):
        task = HandoffTask(
            task_number=1,
            issue_id="ISS-001",
            severity="high",
            title="Bug",
            fix_options=[
                FixOption(approach="Quick", estimated_effort="quick_fix"),
                FixOption(approach="Proper", estimated_effort="significant"),
            ],
        )
        assert len(task.fix_options) == 2

    def test_handoff_generates_fix_options(self):
        from humanqa.reporting.handoff import HandoffGenerator

        result = _make_run_result(issues=[
            _make_issue("Critical bug", severity="critical"),
        ])
        gen = HandoffGenerator("/tmp/test_fix_opts")
        handoff = gen.generate(result)
        # Critical issues should get multiple fix options
        assert len(handoff.tasks[0].fix_options) >= 2

    def test_handoff_fix_options_in_markdown(self, tmp_path):
        from humanqa.reporting.handoff import HandoffGenerator

        result = _make_run_result(issues=[
            _make_issue("Big bug", severity="critical"),
        ])
        gen = HandoffGenerator(str(tmp_path))
        paths = gen.generate_all(result)
        md = Path(paths["handoff_md"]).read_text()
        assert "Fix options" in md

    def test_handoff_fix_options_in_json(self, tmp_path):
        from humanqa.reporting.handoff import HandoffGenerator

        result = _make_run_result(issues=[
            _make_issue("Bug", severity="high"),
        ])
        gen = HandoffGenerator(str(tmp_path))
        paths = gen.generate_all(result)
        data = json.loads(Path(paths["handoff_json"]).read_text())
        assert "fix_options" in data["tasks"][0]
        assert len(data["tasks"][0]["fix_options"]) >= 1

    def test_accessibility_gets_aria_option(self):
        from humanqa.reporting.handoff import HandoffGenerator

        result = _make_run_result(issues=[
            _make_issue("Missing alt text", severity="medium", category="accessibility"),
        ])
        gen = HandoffGenerator("/tmp/test_a11y_opts")
        handoff = gen.generate(result)
        approaches = [o.approach for o in handoff.tasks[0].fix_options]
        assert any("ARIA" in a or "semantic" in a.lower() for a in approaches)


# ===========================================================================
# 8. Login/auth flow evaluation
# ===========================================================================

class TestAuthLens:
    """Tests for login/auth flow evaluation lens."""

    def test_auth_category_exists(self):
        assert IssueCategory.auth == "auth"

    def test_auth_lens_no_auth_pages(self):
        """Auth lens should return empty when no auth pages found."""
        from humanqa.lenses.auth_lens import AuthLens

        mock_llm = MagicMock()
        lens = AuthLens(mock_llm)

        result = _make_run_result(issues=[])
        import asyncio
        issues = asyncio.get_event_loop().run_until_complete(lens.review(result))
        assert issues == []

    def test_auth_lens_detects_auth_urls(self):
        """Auth lens should detect login-related URLs in coverage."""
        from humanqa.lenses.auth_lens import AuthLens

        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "issues": [
                {
                    "title": "No password reset link",
                    "severity": "medium",
                    "confidence": 0.8,
                    "user_impact": "Users cannot recover accounts",
                    "observed_facts": ["No forgot password link found"],
                }
            ],
            "has_auth_page": True,
        }
        lens = AuthLens(mock_llm)

        result = _make_run_result(issues=[])
        result.coverage.entries.append(
            CoverageEntry(url="https://example.com/login", status="visited")
        )

        import asyncio
        issues = asyncio.get_event_loop().run_until_complete(lens.review(result))
        assert len(issues) >= 1
        assert issues[0].category == IssueCategory.auth
        assert issues[0].agent == "auth_lens"

    def test_auth_lens_handles_llm_failure(self):
        from humanqa.lenses.auth_lens import AuthLens

        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = Exception("LLM error")
        lens = AuthLens(mock_llm)

        result = _make_run_result(issues=[])
        result.coverage.entries.append(
            CoverageEntry(url="https://example.com/signin", status="visited")
        )

        import asyncio
        issues = asyncio.get_event_loop().run_until_complete(lens.review(result))
        assert issues == []


# ===========================================================================
# 9. Mobile responsiveness evaluation
# ===========================================================================

class TestResponsiveLens:
    """Tests for mobile responsiveness evaluation lens."""

    def test_responsive_category_exists(self):
        assert IssueCategory.responsive == "responsive"

    def test_responsive_lens_flags_missing_mobile(self):
        """Should flag when no mobile evaluation was performed."""
        from humanqa.lenses.responsive_lens import ResponsiveLens

        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "issues": [],
            "responsive_score": 0.5,
            "summary": "No mobile data",
        }
        lens = ResponsiveLens(mock_llm)

        result = _make_run_result(issues=[
            _make_issue("Desktop bug", platform=Platform.web),
        ])

        import asyncio
        issues, score = asyncio.get_event_loop().run_until_complete(lens.review(result))
        # Should have at least the "no mobile viewport" warning
        mobile_warnings = [i for i in issues if "mobile viewport" in i.title.lower()]
        assert len(mobile_warnings) >= 1
        assert score <= 0.3

    def test_responsive_lens_with_mobile_data(self):
        from humanqa.lenses.responsive_lens import ResponsiveLens

        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "issues": [
                {
                    "title": "Touch targets too small",
                    "severity": "medium",
                    "confidence": 0.8,
                    "user_impact": "Hard to tap on mobile",
                    "observed_facts": ["Buttons 28x28px"],
                    "affects_viewport": "mobile",
                }
            ],
            "responsive_score": 0.7,
        }
        lens = ResponsiveLens(mock_llm)

        result = _make_run_result(issues=[
            _make_issue("Mobile bug", platform=Platform.mobile_web),
        ])

        import asyncio
        issues, score = asyncio.get_event_loop().run_until_complete(lens.review(result))
        assert len(issues) >= 1
        assert any(i.category == IssueCategory.responsive for i in issues)
        assert score == 0.7

    def test_responsive_lens_handles_llm_failure(self):
        from humanqa.lenses.responsive_lens import ResponsiveLens

        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = Exception("LLM error")
        lens = ResponsiveLens(mock_llm)

        result = _make_run_result(issues=[
            _make_issue("Bug", platform=Platform.web),
        ])

        import asyncio
        issues, score = asyncio.get_event_loop().run_until_complete(lens.review(result))
        assert issues == []
        assert score == 0.5

    def test_responsive_empty_issues_returns_1(self):
        from humanqa.lenses.responsive_lens import ResponsiveLens

        mock_llm = MagicMock()
        lens = ResponsiveLens(mock_llm)

        result = _make_run_result(issues=[])

        import asyncio
        issues, score = asyncio.get_event_loop().run_until_complete(lens.review(result))
        assert score == 1.0


# ===========================================================================
# 10. Run time optimizations
# ===========================================================================

class TestRunTimeOptimizations:
    """Tests for timeouts, caps, and parallelization constants."""

    def test_step_timeouts_defined(self):
        from humanqa.core.pipeline import STEP_TIMEOUT_SECONDS

        assert "repo" in STEP_TIMEOUT_SECONDS
        assert "scrape" in STEP_TIMEOUT_SECONDS
        assert "evaluate" in STEP_TIMEOUT_SECONDS
        assert "lenses" in STEP_TIMEOUT_SECONDS
        # Evaluate should have the longest timeout
        assert STEP_TIMEOUT_SECONDS["evaluate"] >= STEP_TIMEOUT_SECONDS["lenses"]

    def test_max_agents_cap(self):
        from humanqa.core.pipeline import MAX_AGENTS

        assert MAX_AGENTS >= 4
        assert MAX_AGENTS <= 10

    def test_max_journeys_cap(self):
        from humanqa.core.pipeline import MAX_JOURNEYS_PER_AGENT

        assert MAX_JOURNEYS_PER_AGENT >= 2
        assert MAX_JOURNEYS_PER_AGENT <= 5

    def test_default_max_steps_reduced(self):
        from humanqa.runners.web_runner import DEFAULT_MAX_STEPS

        assert DEFAULT_MAX_STEPS <= 10
        assert DEFAULT_MAX_STEPS >= 5

    def test_page_navigation_timeout(self):
        from humanqa.runners.web_runner import PAGE_NAVIGATION_TIMEOUT_MS

        assert PAGE_NAVIGATION_TIMEOUT_MS <= 30000
        assert PAGE_NAVIGATION_TIMEOUT_MS >= 10000

    def test_with_timeout_returns_none_on_timeout(self):
        import asyncio
        from humanqa.core.pipeline import _with_timeout

        async def slow_task():
            await asyncio.sleep(10)
            return "done"

        result = asyncio.get_event_loop().run_until_complete(
            _with_timeout(slow_task(), 0.01, "test")
        )
        assert result is None

    def test_with_timeout_returns_result_on_success(self):
        import asyncio
        from humanqa.core.pipeline import _with_timeout

        async def fast_task():
            return "done"

        result = asyncio.get_event_loop().run_until_complete(
            _with_timeout(fast_task(), 5.0, "test")
        )
        assert result == "done"


# ===========================================================================
# 11. Multi-provider tiered model support
# ===========================================================================

class TestTieredModelSupport:
    """Tests for multi-provider tiered model configuration."""

    def test_tier_presets_exist(self):
        from humanqa.core.llm import TIER_PRESETS

        assert "balanced" in TIER_PRESETS
        assert "budget" in TIER_PRESETS
        assert "premium" in TIER_PRESETS
        assert "openai" in TIER_PRESETS

    def test_balanced_uses_gemini(self):
        from humanqa.core.llm import get_tier_config

        provider, models = get_tier_config("balanced")
        assert provider == "gemini"
        assert "gemini" in models.fast
        assert "gemini" in models.smart

    def test_budget_uses_flash_lite(self):
        from humanqa.core.llm import get_tier_config

        provider, models = get_tier_config("budget")
        assert provider == "gemini"
        assert models.fast == "gemini-2.5-flash"
        assert models.smart == "gemini-3-flash"

    def test_premium_uses_claude(self):
        from humanqa.core.llm import get_tier_config

        provider, models = get_tier_config("premium")
        assert provider == "anthropic"
        assert "claude" in models.smart

    def test_openai_tier(self):
        from humanqa.core.llm import get_tier_config

        provider, models = get_tier_config("openai")
        assert provider == "openai"
        assert models.fast == "gpt-4.1"
        assert models.smart == "gpt-5.4"

    def test_invalid_tier_raises(self):
        from humanqa.core.llm import get_tier_config

        with pytest.raises(ValueError, match="Unknown tier"):
            get_tier_config("nonexistent")

    def test_default_tier_is_balanced(self):
        from humanqa.core.llm import DEFAULT_TIER

        assert DEFAULT_TIER == "balanced"

    def test_llm_client_with_tier_balanced(self):
        from humanqa.core.llm import LLMClient

        client = LLMClient(tier="balanced")
        assert client.provider == "gemini"
        assert "gemini" in client.model
        assert client._resolve_model("fast") == "gemini-2.0-flash"
        assert client._resolve_model("smart") == "gemini-2.0-flash"

    def test_llm_client_with_tier_premium(self):
        from humanqa.core.llm import LLMClient

        client = LLMClient(tier="premium")
        assert client.provider == "anthropic"
        assert "claude" in client.model

    def test_llm_client_model_override_with_tier(self):
        from humanqa.core.llm import LLMClient

        client = LLMClient(tier="balanced", model="custom-model")
        # Model override should apply to both tiers
        assert client._resolve_model("fast") == "custom-model"
        assert client._resolve_model("smart") == "custom-model"

    def test_llm_client_without_tier(self):
        from humanqa.core.llm import LLMClient

        # No tier, explicit provider — should work like before
        client = LLMClient(provider="gemini")
        assert client.provider == "gemini"
        assert client._resolve_model("fast") == client.model
        assert client._resolve_model("smart") == client.model

    def test_gemini_deferred_init_without_key(self):
        """Gemini client should defer init when no API key is set."""
        import os
        from humanqa.core.llm import LLMClient

        # Ensure no Gemini key
        old_keys = {}
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            old_keys[k] = os.environ.pop(k, None)

        try:
            client = LLMClient(tier="balanced")
            assert client._client is None
            assert client._gemini_deferred is True
        finally:
            for k, v in old_keys.items():
                if v is not None:
                    os.environ[k] = v

    def test_run_config_default_tier(self):
        config = RunConfig(target_url="https://example.com")
        assert config.llm_tier == "balanced"
        assert config.llm_provider == "gemini"
        assert config.llm_model == "gemini-2.0-flash"

    def test_run_config_custom_tier(self):
        config = RunConfig(
            target_url="https://example.com",
            llm_tier="premium",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
        )
        assert config.llm_tier == "premium"

    def test_cli_has_tier_option(self):
        from humanqa.cli import main
        cmd = main.commands["run"]
        param_names = [p.name for p in cmd.params]
        assert "tier" in param_names

    def test_cli_tier_choices(self):
        from humanqa.cli import main
        cmd = main.commands["run"]
        tier_param = [p for p in cmd.params if p.name == "tier"][0]
        assert "balanced" in tier_param.type.choices
        assert "budget" in tier_param.type.choices
        assert "premium" in tier_param.type.choices
        assert "openai" in tier_param.type.choices

    def test_fast_tier_tagged_in_orchestrator(self):
        """Verify orchestrator dedup calls use fast tier."""
        from humanqa.core.orchestrator import Orchestrator
        import inspect

        source = inspect.getsource(Orchestrator._deduplicate_with_llm)
        assert 'tier="fast"' in source

    def test_fast_tier_tagged_in_persona_generator(self):
        from humanqa.core.persona_generator import PersonaGenerator
        import inspect

        source = inspect.getsource(PersonaGenerator.generate_personas)
        assert 'tier="fast"' in source

    def test_smart_tier_default_in_web_runner(self):
        """Web runner vision calls should NOT have tier='fast'."""
        from humanqa.runners.web_runner import WebRunner
        import inspect

        source = inspect.getsource(WebRunner._judge_snapshot)
        assert 'tier="fast"' not in source
