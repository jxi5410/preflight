"""File Mapper — maps issues to likely source files using repo structure.

Uses routes, tech stack, product area, and framework conventions to suggest
which files a developer should look at when fixing an issue.
"""

from __future__ import annotations

import re

from humanqa.core.schemas import Issue, RepoInsights


class FileMapper:
    """Maps issues to likely source files using repo structure."""

    def __init__(self, repo_insights: RepoInsights | None = None):
        self.insights = repo_insights

    def map_issue_to_files(
        self,
        issue: Issue,
        repo_insights: RepoInsights | None = None,
    ) -> list[str]:
        """Return likely file paths for an issue based on:
        - issue.likely_product_area (e.g., "checkout_form")
        - repo_insights.routes_or_pages (e.g., ["app/checkout/page.tsx"])
        - issue.repro_steps (URLs visited -> route mapping)
        - repo_insights.tech_stack (framework conventions)
        """
        insights = repo_insights or self.insights
        if not insights:
            return []

        candidates: list[str] = []
        product_area = issue.likely_product_area.lower().strip()
        title_lower = issue.title.lower()

        # Match against known routes/pages
        for route in insights.routes_or_pages:
            route_lower = route.lower()
            if product_area and _fuzzy_match(product_area, route_lower):
                candidates.append(route)
                continue
            if _title_matches_route(title_lower, route_lower):
                candidates.append(route)

        # Match against repro step URLs
        for step in issue.repro_steps:
            for route in insights.routes_or_pages:
                if route != "/" and route.lower() in step.lower():
                    if route not in candidates:
                        candidates.append(route)

        # Infer from category + tech stack
        category_hints = _category_file_hints(issue.category.value, insights.tech_stack)
        candidates.extend(category_hints)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique[:10]


def _fuzzy_match(area: str, route: str) -> bool:
    """Check if a product area loosely matches a route path."""
    area_words = set(re.split(r"[\s_\-/]+", area))
    route_words = set(re.split(r"[\s_\-/]+", route.strip("/")))
    overlap = area_words & route_words
    return bool(overlap - {"", "the", "a", "an", "page", "view", "screen"})


def _title_matches_route(title: str, route: str) -> bool:
    """Check if issue title keywords match a route."""
    route_parts = set(re.split(r"[\s_\-/]+", route.strip("/")))
    route_parts -= {"", "index", "page", "layout"}
    if not route_parts:
        return False
    for part in route_parts:
        if len(part) >= 3 and part in title:
            return True
    return False


def _category_file_hints(category: str, tech_stack: list[str]) -> list[str]:
    """Suggest generic file patterns based on issue category and tech stack."""
    hints: list[str] = []
    stack_lower = " ".join(tech_stack).lower()

    if category == "accessibility":
        if "react" in stack_lower or "next" in stack_lower:
            hints.append("src/components/")
        hints.append("(check ARIA attributes and semantic HTML)")
    elif category == "performance":
        if "next" in stack_lower:
            hints.append("next.config.js")
        hints.append("(check network requests and bundle size)")
    elif category in ("ui", "design"):
        if "tailwind" in stack_lower:
            hints.append("tailwind.config.js")
        hints.append("(check CSS/styling files)")

    return hints


# Convenience function for backward compatibility
def map_issue_to_files(issue: Issue, insights: RepoInsights | None) -> list[str]:
    """Convenience wrapper around FileMapper."""
    return FileMapper(insights).map_issue_to_files(issue)
