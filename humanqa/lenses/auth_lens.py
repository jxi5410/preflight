"""Auth/Login Flow Lens — evaluates login page as product surface.

Evaluates login/auth pages WITHOUT credentials:
- Login page design and UX quality
- Error message quality (submit empty form, bad format)
- Trust signals (HTTPS, password visibility toggle, etc.)
- Accessibility of auth forms
- Social login / SSO options visibility
"""

from __future__ import annotations

import logging

from humanqa.core.llm import LLMClient
from humanqa.core.schemas import (
    Evidence,
    Issue,
    IssueCategory,
    Platform,
    RunResult,
    Severity,
)

logger = logging.getLogger(__name__)

AUTH_EVAL_SYSTEM_PROMPT = """You are a login/auth flow evaluator for HumanQA.

You evaluate login and authentication pages as product surfaces WITHOUT
attempting to log in. You assess:

1. **Login page design**: Is it clear, professional, and trustworthy?
2. **Error handling quality**: What happens with empty submissions or invalid formats?
3. **Trust signals**: HTTPS indicator, password visibility toggle, "remember me",
   forgot password link, privacy policy link
4. **Accessibility**: Form labels, focus states, keyboard navigation, ARIA attributes
5. **Social/SSO options**: Are alternative login methods available and discoverable?
6. **Security UX**: Password requirements shown upfront? Rate limiting indication?
   Account lockout warnings?

You MUST cite evidence for every finding (screenshot reference, element reference,
or observed absence).

Respond with JSON:
{
  "issues": [
    {
      "title": "...",
      "severity": "critical|high|medium|low|info",
      "confidence": 0.0-1.0,
      "user_impact": "...",
      "observed_facts": ["..."],
      "inferred_judgment": "...",
      "repair_brief": "..."
    }
  ],
  "login_page_detected": true|false,
  "trust_assessment": "summary of trust signals present/absent",
  "auth_ux_score": 0.0-1.0
}"""

AUTH_EVAL_PROMPT = """Evaluate this page as a login/auth surface.

## Page URL: {url}
## Page Title: {title}

## Accessibility Tree (semantic structure)
{accessibility_tree}

## Page Text Content
{page_text}

## Console Errors
{console_errors}

Evaluate the login/auth experience. Focus on:
- Is this a login page? If not, note that and evaluate what auth elements exist.
- Quality of the login form (labels, placeholders, error states)
- Trust signals visible on the page
- Error path quality (what would happen with bad input?)
- Accessibility of the auth flow
- Missing standard auth features (forgot password, etc.)

Do NOT attempt to log in. Evaluate the page as-is."""


class AuthLens:
    """Evaluates login/auth pages as product surfaces."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def review(
        self,
        result: RunResult,
        login_snapshot: object | None = None,
    ) -> list[Issue]:
        """Review auth-related pages found during evaluation.

        Generates issues for login page quality, error handling,
        trust signals, and auth UX without requiring credentials.
        """
        issues: list[Issue] = []

        # Look for login-related pages in coverage
        auth_urls = []
        for entry in result.coverage.entries:
            url_lower = entry.url.lower()
            if any(kw in url_lower for kw in (
                "login", "signin", "sign-in", "auth", "register",
                "signup", "sign-up", "forgot", "reset-password",
            )):
                auth_urls.append(entry.url)

        if not auth_urls and not login_snapshot:
            # Check if any issues reference auth flows
            has_auth_issues = any(
                any(kw in i.title.lower() for kw in ("login", "auth", "sign"))
                for i in result.issues
            )
            if not has_auth_issues:
                return []

        # Evaluate login page quality from existing data
        issues.extend(self._evaluate_auth_surface(result, auth_urls))

        return issues

    def _evaluate_auth_surface(
        self,
        result: RunResult,
        auth_urls: list[str],
    ) -> list[Issue]:
        """Evaluate auth pages using LLM analysis of collected data."""
        # Collect auth-related evidence from existing issues and coverage
        auth_context_parts = []
        for issue in result.issues:
            title_lower = issue.title.lower()
            if any(kw in title_lower for kw in ("login", "auth", "password", "sign")):
                auth_context_parts.append(
                    f"Existing finding: {issue.title} ({issue.severity.value}) - {issue.user_impact}"
                )

        if auth_urls:
            auth_context_parts.append(f"Auth pages found: {', '.join(auth_urls)}")

        if not auth_context_parts:
            return []

        prompt = f"""Evaluate the authentication experience for this product.

## Product: {result.intent_model.product_name}
## Target URL: {result.config.target_url}

## Auth-Related Context
{chr(10).join(auth_context_parts)}

## Known Auth Pages
{chr(10).join(auth_urls) if auth_urls else '(none found — product may not have a login page)'}

Based on this context, identify auth/login UX issues. Focus on:
- Is a login page discoverable from the main product?
- Are there trust signals on auth pages?
- Are error paths likely to be helpful?
- Is password reset available?
- Are there accessibility concerns in auth flows?

Respond with JSON: {{"issues": [...], "has_auth_page": true/false}}"""

        try:
            data = self.llm.complete_json(prompt, system=AUTH_EVAL_SYSTEM_PROMPT)
        except Exception as e:
            logger.warning("Auth lens evaluation failed: %s", e)
            return []

        issues: list[Issue] = []
        for raw in data.get("issues", []):
            sev = raw.get("severity", "medium")
            try:
                severity = Severity(sev)
            except ValueError:
                severity = Severity.medium

            issues.append(Issue(
                title=raw.get("title", "Auth flow issue"),
                severity=severity,
                confidence=raw.get("confidence", 0.7),
                platform=Platform.web,
                category=IssueCategory.auth,
                agent="auth_lens",
                user_impact=raw.get("user_impact", ""),
                observed_facts=raw.get("observed_facts", []),
                inferred_judgment=raw.get("inferred_judgment", ""),
                repair_brief=raw.get("repair_brief", ""),
                likely_product_area="Authentication",
            ))

        has_auth = data.get("has_auth_page", True)
        if not has_auth and not auth_urls:
            issues.append(Issue(
                title="No login/auth page discoverable from product surface",
                severity=Severity.info,
                confidence=0.6,
                platform=Platform.web,
                category=IssueCategory.auth,
                agent="auth_lens",
                user_impact="Users may not find how to log in or create an account",
                observed_facts=["No auth-related URLs found during evaluation"],
                inferred_judgment="Login page may be at a non-standard URL or behind a redirect",
                likely_product_area="Authentication",
            ))

        logger.info("Auth lens: %d issues found", len(issues))
        return issues
