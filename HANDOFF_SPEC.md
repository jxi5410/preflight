# AI Coding Tool Handoff — Implementation Spec

**Purpose:** Make HumanQA's output directly executable by AI coding tools (Claude Code, Codex, etc.) with zero reformatting. The user reviews and confirms; the coding tool executes.

---

## The Problem

Current repair briefs are markdown files written for humans. An engineer reads them, interprets them, then tells Claude Code what to fix. That's friction. The handoff should be:

```
HumanQA finds issues → User reviews → User says "fix these" → AI coding tool executes
```

Not:

```
HumanQA finds issues → User reads report → User reformulates as prompts → AI coding tool executes
```

---

## What to Build

### 1. Handoff File Generator (`reporting/handoff.py`)

Generates a single `HANDOFF.md` file designed to be consumed directly by an AI coding tool.

**Output format — a structured task list the coding tool can execute top-to-bottom:**

```markdown
# HumanQA Handoff — [Product Name]
Generated: 2026-03-14 02:00 UTC | Run: run-abc123

## Context
Product: [name] ([type])
Repo: https://github.com/user/repo
Tech stack: Next.js, TypeScript, Tailwind, PostgreSQL
Target URL: https://product.com

## Summary
15 issues found: 2 critical, 4 high, 6 medium, 3 low
Estimated scope: ~3-4 hours of implementation work

---

## Task 1 of 9 — CRITICAL
### Checkout form silently fails on invalid card input
**What's wrong:** Entering an invalid credit card number and clicking "Pay" shows no error message. The button becomes disabled with no feedback. User has no idea what happened.
**Where to look:** Likely in the checkout/payment component. Route structure suggests `app/checkout/page.tsx` or similar. Look for form validation and error state handling.
**User impact:** Users cannot complete purchases. Revenue-blocking.
**Repro:** Navigate to /checkout → Fill form with invalid card "4111111111111112" → Click "Pay" → Observe: no error, button grays out
**Expected:** Clear inline error message explaining the card number is invalid
**Fix guidance:** Add client-side card validation (Luhn check) with inline error display. Ensure the submit handler surfaces API validation errors in the UI.
**Verify fix:** After fixing, the same repro steps should show an error message. Write an e2e test: fill checkout with invalid card → assert error message visible.
**Evidence:** screenshots/step-7-checkout-fail.png

---

## Task 2 of 9 — CRITICAL
### [next issue...]

---

## Dependency Notes
- Fix Task 1 (checkout form) before Task 5 (payment confirmation page) — Task 5 depends on successful checkout
- Tasks 3 and 4 are independent and can be done in parallel

## Feature Gaps (Repo claims vs UI reality)
These features are documented in the README but not found in the UI:
- Dark mode toggle (README says "supports dark/light theme")
- CSV export on dashboard (README says "export your data")

## Verification Checklist
After all fixes, re-run: `humanqa run https://product.com --repo https://github.com/user/repo`
Expected: Critical and high issues should not reappear.
```

### 2. JSON Task Format (`handoff.json`)

Machine-parseable version for tools that prefer structured input:

```json
{
  "handoff_version": "1.0",
  "run_id": "run-abc123",
  "product": {
    "name": "ProductName",
    "repo": "https://github.com/user/repo",
    "tech_stack": ["Next.js", "TypeScript", "Tailwind"],
    "target_url": "https://product.com"
  },
  "tasks": [
    {
      "task_number": 1,
      "issue_id": "ISS-A1B2C3",
      "severity": "critical",
      "title": "Checkout form silently fails on invalid card input",
      "description": "Entering an invalid credit card number and clicking Pay shows no error message.",
      "likely_files": ["app/checkout/page.tsx", "components/PaymentForm.tsx"],
      "repro_steps": ["Navigate to /checkout", "Fill form with invalid card", "Click Pay"],
      "expected_behavior": "Clear inline error message explaining the card number is invalid",
      "fix_guidance": "Add client-side card validation with inline error display. Surface API errors in UI.",
      "verification": "Fill checkout with invalid card → assert error message visible",
      "evidence_screenshots": ["screenshots/step-7-checkout-fail.png"],
      "depends_on": [],
      "blocks": [5]
    }
  ],
  "feature_gaps": [
    {
      "feature": "Dark mode toggle",
      "source": "README",
      "claim": "supports dark/light theme",
      "ui_status": "not_found"
    }
  ],
  "dependency_graph": {
    "1": {"blocks": [5]},
    "3": {"parallel_with": [4]}
  }
}
```

### 3. CLI Integration

```bash
# Default: generates all formats including handoff
humanqa run https://product.com --repo https://github.com/user/repo

# Explicit handoff-only output
humanqa run https://product.com --handoff claude-code

# Just regenerate handoff from existing run
humanqa handoff ./artifacts/report.json --format claude-code
humanqa handoff ./artifacts/report.json --format codex
humanqa handoff ./artifacts/report.json --format cursor
```

**`--handoff` flag values:**
- `claude-code` — optimized for Claude Code (HANDOFF.md with CLAUDE.md conventions)
- `codex` — optimized for Codex (structured JSON tasks)
- `cursor` — optimized for Cursor (markdown with file references)
- `generic` — works with any tool (HANDOFF.md)

### 4. Tool-Specific Formatting

**Claude Code format** — generates a file following CLAUDE.md conventions:
- Tasks as numbered action items
- File paths relative to repo root
- Verification commands included
- Can be fed directly: `claude "$(cat .humanqa/HANDOFF.md)"`

**Codex format** — generates JSON with task objects that map to Codex's task structure

**Generic format** — markdown that any LLM coding tool can parse

### 5. Likely File Mapping

The key to frictionless handoff is telling the coding tool **where to look**.

Use the repo analyzer's route/page structure to map issues to likely files:

```python
class FileMapper:
    """Maps issues to likely source files using repo structure."""
    
    def map_issue_to_files(
        self,
        issue: Issue,
        repo_insights: RepoInsights,
    ) -> list[str]:
        """Return likely file paths for an issue based on:
        - issue.likely_product_area (e.g., "checkout_form")
        - repo_insights.routes_or_pages (e.g., ["app/checkout/page.tsx"])
        - issue.repro_steps (URLs visited → route mapping)
        - repo_insights.tech_stack (framework conventions)
        """
```

### 6. Dependency Detection

Order tasks by:
1. Severity (critical first)
2. Dependencies (if issue B is on a page that requires issue A's flow to work, A comes first)
3. Isolation (independent issues grouped so they can be parallelized)

Use LLM to infer dependencies from repro steps and product area overlap.

### 7. Scope Estimation

For each task, estimate rough complexity:
- **Quick fix** (< 30 min): Missing error message, wrong copy, styling issue
- **Moderate** (30 min - 2 hr): Form validation, state management fix, new component state
- **Significant** (2-4 hr): New feature gap, missing flow, architectural issue

Include total estimated scope in the handoff summary.

---

## Schema Additions

Add to `schemas.py`:

```python
class HandoffTask(BaseModel):
    """A single actionable task for an AI coding tool."""
    task_number: int
    issue_id: str
    severity: str
    title: str
    description: str
    likely_files: list[str] = Field(default_factory=list)
    repro_steps: list[str] = Field(default_factory=list)
    expected_behavior: str = ""
    fix_guidance: str = ""
    verification: str = ""
    evidence_screenshots: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)
    blocks: list[int] = Field(default_factory=list)
    estimated_complexity: str = "moderate"  # quick_fix | moderate | significant
    
class FeatureGap(BaseModel):
    """A feature claimed in repo but not found in UI."""
    feature: str
    source: str  # Where the claim comes from
    claim: str  # What was claimed
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
```

---

## Pipeline Integration

In `pipeline.py`, after report generation:

```python
# Step 7: Generate AI coding tool handoff
handoff_gen = HandoffGenerator(llm, config.output_dir)
handoff = await handoff_gen.generate(result)
# Writes: HANDOFF.md, handoff.json
```

---

## Success Criteria

1. `humanqa run` produces a `HANDOFF.md` that Claude Code can execute directly
2. User workflow is: run HumanQA → skim HANDOFF.md → paste into Claude Code → confirm
3. Tasks include likely file paths (not just "somewhere in the codebase")
4. Tasks are ordered by severity and dependency
5. Feature gaps (repo claims vs UI reality) are listed as separate tasks
6. Each task has a verification step the coding tool can run after fixing
7. Total scope estimate helps user decide whether to fix all or prioritize
