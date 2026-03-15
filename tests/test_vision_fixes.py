"""Tests for vision-based evaluation fixes.

Verifies that design_lens, responsive_lens use complete_json_with_vision,
and that persona_generator guarantees a mobile_web persona.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import pytest

from preflight.core.schemas import (
    AgentPersona,
    CoverageEntry,
    CoverageMap,
    Evidence,
    Issue,
    IssueCategory,
    Platform,
    ProductIntentModel,
    RunConfig,
    RunResult,
    Severity,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fix 1: DesignLens uses complete_json_with_vision when screenshots exist
# ---------------------------------------------------------------------------

class TestDesignLensVision:
    """Design lens must send actual screenshot images via vision API."""

    def _make_llm(self):
        llm = MagicMock()
        llm.complete_json_with_vision.return_value = {
            "design_issues": [
                {
                    "title": "Misaligned heading",
                    "severity": "medium",
                    "confidence": 0.8,
                    "user_impact": "Looks sloppy",
                    "observed_facts": ["Heading is 5px off-grid"],
                    "inferred_judgment": "Alignment issue",
                    "likely_product_area": "Header",
                    "repair_brief": "Align to grid",
                }
            ],
            "design_strengths": ["Good color palette"],
            "overall_assessment": "Decent design with minor issues.",
        }
        llm.complete_json.return_value = llm.complete_json_with_vision.return_value
        return llm

    def _make_run_result(self):
        return RunResult(
            config=RunConfig(target_url="https://example.com"),
            intent_model=ProductIntentModel(
                product_name="TestApp",
                product_type="SaaS",
                target_audience=["developers"],
            ),
            issues=[],
            coverage=CoverageMap(entries=[
                CoverageEntry(url="https://example.com", status="visited"),
            ]),
        )

    def test_uses_vision_when_screenshots_exist(self):
        """When screenshot PNGs exist in output_dir, complete_json_with_vision must be called."""
        from preflight.lenses.design_lens import DesignLens

        mock_llm = self._make_llm()
        run_result = self._make_run_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake screenshot files
            (Path(tmpdir) / "page-01.png").write_bytes(b"\x89PNG fake image data")
            (Path(tmpdir) / "page-02.png").write_bytes(b"\x89PNG fake image data 2")

            lens = DesignLens(llm=mock_llm, output_dir=tmpdir)

            with patch("preflight.lenses.design_lens.async_playwright") as mock_pw:
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()
                mock_pw.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                mock_page.goto = AsyncMock()
                mock_page.wait_for_timeout = AsyncMock()
                mock_page.screenshot = AsyncMock(return_value=b"\x89PNG mobile")

                issues = _run(lens.review(run_result))

            # MUST use vision API, not text-only
            mock_llm.complete_json_with_vision.assert_called_once()
            mock_llm.complete_json.assert_not_called()

            # Should have passed images (2 desktop from files + 1 mobile captured)
            call_args = mock_llm.complete_json_with_vision.call_args
            images = call_args.kwargs["images"]
            assert len(images) == 3
            assert all(mime == "image/png" for _, mime in images)

            assert len(issues) == 1
            assert issues[0].title == "Misaligned heading"

    def test_falls_back_to_text_when_no_screenshots_and_capture_fails(self):
        """When no screenshots exist and mobile capture fails, falls back to complete_json."""
        from preflight.lenses.design_lens import DesignLens

        mock_llm = self._make_llm()
        run_result = self._make_run_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            lens = DesignLens(llm=mock_llm, output_dir=tmpdir)

            with patch("preflight.lenses.design_lens.async_playwright") as mock_pw:
                # Make Playwright fail so no mobile screenshot captured
                mock_pw.return_value.__aenter__.side_effect = Exception("No browser")

                issues = _run(lens.review(run_result))

            # Falls back to text-only
            mock_llm.complete_json.assert_called_once()
            mock_llm.complete_json_with_vision.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 3: PersonaGenerator guarantees at least one mobile_web persona
# ---------------------------------------------------------------------------

class TestPersonaGeneratorMobileGuarantee:
    """After generation, at least one persona must have device_preference=mobile_web."""

    def test_adds_mobile_persona_when_none_generated(self):
        """If LLM generates all web personas, generator must inject a mobile one."""
        from preflight.core.persona_generator import PersonaGenerator

        mock_llm = MagicMock()
        # LLM returns all-web personas
        mock_llm.complete_json.return_value = [
            {
                "name": "Alex, First-time User",
                "role": "New user",
                "persona_type": "first_time_user",
                "goals": ["Sign up"],
                "expectations": ["Easy onboarding"],
                "patience_level": "moderate",
                "expertise_level": "novice",
                "behavioral_style": "Cautious",
                "device_preference": "web",
            },
            {
                "name": "Sam, Power User",
                "role": "Experienced user",
                "persona_type": "power_user",
                "goals": ["Advanced features"],
                "expectations": ["Keyboard shortcuts"],
                "patience_level": "low",
                "expertise_level": "expert",
                "behavioral_style": "Efficient",
                "device_preference": "web",
            },
        ]

        gen = PersonaGenerator(llm=mock_llm)
        config = RunConfig(target_url="https://example.com")
        intent = ProductIntentModel(
            product_name="TestApp",
            product_type="SaaS",
            target_audience=["developers"],
            primary_jobs=["manage tasks"],
            critical_journeys=["onboarding"],
        )

        personas = _run(gen.generate_personas(intent, config))

        mobile_personas = [p for p in personas if p.device_preference == Platform.mobile_web]
        assert len(mobile_personas) >= 1, "Must guarantee at least one mobile_web persona"

    def test_keeps_existing_mobile_persona(self):
        """If LLM already generates a mobile persona, don't add a duplicate."""
        from preflight.core.persona_generator import PersonaGenerator

        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = [
            {
                "name": "Alex, Mobile User",
                "role": "Mobile user",
                "persona_type": "mobile_user",
                "goals": ["Browse on phone"],
                "expectations": ["Works on mobile"],
                "patience_level": "low",
                "expertise_level": "intermediate",
                "behavioral_style": "Quick",
                "device_preference": "mobile_web",
            },
            {
                "name": "Sam, Desktop User",
                "role": "Desktop user",
                "persona_type": "power_user",
                "goals": ["Advanced features"],
                "expectations": ["Keyboard shortcuts"],
                "patience_level": "low",
                "expertise_level": "expert",
                "behavioral_style": "Efficient",
                "device_preference": "web",
            },
        ]

        gen = PersonaGenerator(llm=mock_llm)
        config = RunConfig(target_url="https://example.com")
        intent = ProductIntentModel(
            product_name="TestApp",
            product_type="SaaS",
            target_audience=["developers"],
            primary_jobs=["manage tasks"],
            critical_journeys=["onboarding"],
        )

        personas = _run(gen.generate_personas(intent, config))

        mobile_personas = [p for p in personas if p.device_preference == Platform.mobile_web]
        assert len(mobile_personas) == 1, "Should not duplicate existing mobile persona"
        assert len(personas) == 2


# ---------------------------------------------------------------------------
# Fix 2 & 4: ResponsiveLens captures its own screenshots and uses vision
# ---------------------------------------------------------------------------

class TestResponsiveLensVision:
    """Responsive lens must capture desktop+mobile screenshots and use vision."""

    def _make_llm(self):
        llm = MagicMock()
        llm.complete_json_with_vision.return_value = {
            "issues": [
                {
                    "title": "Nav overflows on mobile",
                    "severity": "high",
                    "confidence": 0.9,
                    "user_impact": "Can't navigate on phone",
                    "observed_facts": ["Nav items extend beyond viewport"],
                    "inferred_judgment": "Missing responsive nav",
                    "repair_brief": "Add hamburger menu",
                    "affects_viewport": "mobile",
                }
            ],
            "responsive_score": 0.4,
            "summary": "Poor mobile adaptation.",
        }
        return llm

    def _make_result(self):
        return RunResult(
            config=RunConfig(target_url="https://example.com"),
            intent_model=ProductIntentModel(
                product_name="TestApp",
                product_type="SaaS",
            ),
            issues=[],
            coverage=CoverageMap(),
        )

    def test_captures_screenshots_at_both_viewports(self):
        """ResponsiveLens must independently capture screenshots at 1440x900 and 390x844."""
        from preflight.lenses.responsive_lens import ResponsiveLens

        mock_llm = self._make_llm()
        result = self._make_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            lens = ResponsiveLens(llm=mock_llm, output_dir=tmpdir)

            with patch("preflight.lenses.responsive_lens.async_playwright") as mock_pw:
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()

                mock_pw.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                mock_page.goto = AsyncMock()
                mock_page.wait_for_load_state = AsyncMock()
                mock_page.wait_for_timeout = AsyncMock()
                mock_page.screenshot = AsyncMock(return_value=b"\x89PNG fake screenshot")

                issues, score = _run(lens.review(result))

                # Must have called vision API
                mock_llm.complete_json_with_vision.assert_called_once()

                # Should have passed 2 images (desktop + mobile)
                call_args = mock_llm.complete_json_with_vision.call_args
                images = call_args.kwargs["images"]
                assert len(images) == 2, "Must send both desktop and mobile screenshots"

                # Verify viewport sizes were used (two new_context calls)
                assert mock_browser.new_context.call_count == 2

    def test_returns_issues_from_vision(self):
        """Vision-based review should parse issues correctly."""
        from preflight.lenses.responsive_lens import ResponsiveLens

        mock_llm = self._make_llm()
        result = self._make_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            lens = ResponsiveLens(llm=mock_llm, output_dir=tmpdir)

            with patch("preflight.lenses.responsive_lens.async_playwright") as mock_pw:
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()

                mock_pw.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                mock_page.goto = AsyncMock()
                mock_page.wait_for_load_state = AsyncMock()
                mock_page.wait_for_timeout = AsyncMock()
                mock_page.screenshot = AsyncMock(return_value=b"\x89PNG fake screenshot")

                issues, score = _run(lens.review(result))

                assert len(issues) == 1
                assert issues[0].title == "Nav overflows on mobile"
                assert issues[0].category == IssueCategory.responsive
                assert score == 0.4


# ---------------------------------------------------------------------------
# Fix 4: DesignLens also evaluates mobile viewport
# ---------------------------------------------------------------------------

class TestDesignLensMobileViewport:
    """Design lens must also capture and evaluate mobile viewport screenshots."""

    def test_captures_mobile_screenshots_for_design_review(self):
        """DesignLens should capture mobile viewport screenshots when target_url is available."""
        from preflight.lenses.design_lens import DesignLens

        mock_llm = MagicMock()
        mock_llm.complete_json_with_vision.return_value = {
            "design_issues": [],
            "design_strengths": ["Good mobile layout"],
            "overall_assessment": "Clean design.",
        }

        run_result = RunResult(
            config=RunConfig(target_url="https://example.com"),
            intent_model=ProductIntentModel(
                product_name="TestApp",
                product_type="SaaS",
                target_audience=["developers"],
            ),
            issues=[],
            coverage=CoverageMap(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Put a desktop screenshot
            (Path(tmpdir) / "desktop-01.png").write_bytes(b"\x89PNG desktop")

            lens = DesignLens(llm=mock_llm, output_dir=tmpdir)

            with patch("preflight.lenses.design_lens.async_playwright") as mock_pw:
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()

                mock_pw.return_value.__aenter__.return_value.chromium.launch = AsyncMock(return_value=mock_browser)
                mock_browser.new_context = AsyncMock(return_value=mock_context)
                mock_context.new_page = AsyncMock(return_value=mock_page)
                mock_page.goto = AsyncMock()
                mock_page.wait_for_load_state = AsyncMock()
                mock_page.wait_for_timeout = AsyncMock()
                mock_page.screenshot = AsyncMock(return_value=b"\x89PNG mobile screenshot")

                issues = _run(lens.review(run_result))

                # Should have launched a mobile browser
                mock_browser.new_context.assert_called_once()
                context_kwargs = mock_browser.new_context.call_args.kwargs
                assert context_kwargs.get("viewport") == {"width": 390, "height": 844}

                # Vision should include both desktop (from file) and mobile (captured)
                mock_llm.complete_json_with_vision.assert_called_once()
                call_args = mock_llm.complete_json_with_vision.call_args
                images = call_args.kwargs["images"]
                assert len(images) >= 2, "Must include both desktop and mobile screenshots"
