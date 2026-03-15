# Self-Learning Feedback Loops — Spec

**Purpose:** Make Preflight improve with every run. It learns which findings matter, how to write better repair briefs, and what "normal" looks like for each product. No fine-tuning — just accumulated context that makes each run smarter than the last.

---

## Architecture: Product Memory

Each product Preflight evaluates gets a persistent memory store. This is a local directory that accumulates signals across runs.

```
~/.preflight/memory/
  {product_slug}/              # e.g. "londonai-network"
    config.json                # Product-specific settings
    feedback.json              # User ratings on findings
    run_history.json           # Summary of every run
    dismissed_patterns.json    # Issue patterns the user has marked as noise
    product_baseline.json      # What "normal" looks like (screenshots, known states)
    learned_priorities.json    # What the user cares about most
```

The product slug is derived from the target URL: `https://LondonAI.network` → `londonai-network`.

---

## Feedback Loop 1: Learn From User Ratings

### How it works

After each run, the report includes a feedback mechanism. The user can rate findings:
- **Useful** — this was a real issue worth fixing
- **Noise** — this isn't a real problem, stop reporting it
- **Critical miss** — you should have caught something you didn't (user describes what)

### CLI feedback command

```bash
# Review and rate findings from the last run
preflight feedback ./artifacts

# Interactive: shows each finding, user rates it
# Preflight found: "Missing privacy policy link"
# Rate: [u]seful / [n]oise / [s]kip > u
#
# Preflight found: "Button color inconsistency on hover"  
# Rate: [u]seful / [n]oise / [s]kip > n
#
# Anything we missed? (describe or press Enter to skip) > 
# The contact form doesn't show a success message after submission
```

### Storage: `feedback.json`

```json
{
  "ratings": [
    {
      "run_id": "run-abc123",
      "issue_id": "ISS-A1B2C3",
      "issue_title": "Missing privacy policy link",
      "category": "trust",
      "rating": "useful",
      "timestamp": "2026-03-15T10:00:00Z"
    },
    {
      "run_id": "run-abc123",
      "issue_id": "ISS-D4E5F6",
      "issue_title": "Button color inconsistency on hover",
      "category": "design",
      "rating": "noise",
      "timestamp": "2026-03-15T10:00:05Z"
    }
  ],
  "missed_issues": [
    {
      "run_id": "run-abc123",
      "description": "Contact form doesn't show success message after submission",
      "timestamp": "2026-03-15T10:01:00Z"
    }
  ]
}
```

### How it affects the next run

Before generating the final report, inject feedback context into the issue synthesis prompt:

```
## Learned preferences for this product

The user has previously rated these types of findings:
- USEFUL: trust issues (privacy policy, data handling) — 5 times rated useful
- USEFUL: functional issues (form behavior, error handling) — 3 times rated useful  
- NOISE: design issues (color consistency, hover states) — 4 times rated noise
- NOISE: minor copy issues — 2 times rated noise

The user previously reported these missed issues:
- "Contact form doesn't show success message after submission"
- "Mobile navigation doesn't close after selecting a menu item"

Adjust your severity scoring:
- Boost severity for issue types the user consistently rates as useful
- Reduce severity (or suppress) for issue types consistently rated as noise
- Actively look for patterns similar to previously missed issues
```

---

## Feedback Loop 2: Learn From Fixes

### How it works

When Preflight runs again after the developer has fixed issues, it compares against the previous run. Issues that disappeared were successfully fixed. Issues that persist were not. New issues may be regressions.

This data feeds back into how Preflight writes repair briefs:
- If an issue was fixed after one run, the repair brief was probably good
- If an issue persisted across 3+ runs, the repair brief may need to be more specific
- If a fix introduced new issues, the repair brief should include regression warnings

### Storage: `run_history.json`

```json
{
  "runs": [
    {
      "run_id": "run-abc123",
      "timestamp": "2026-03-14T15:00:00Z",
      "issue_count": 42,
      "severity_counts": {"critical": 2, "high": 8, "medium": 20, "low": 12},
      "issue_ids": ["ISS-A1B2C3", "ISS-D4E5F6", ...],
      "issue_summaries": [
        {"id": "ISS-A1B2C3", "title": "Missing privacy policy", "severity": "high", "category": "trust"}
      ]
    },
    {
      "run_id": "run-def456",
      "timestamp": "2026-03-15T15:00:00Z",
      "issue_count": 35,
      "resolved_from_previous": ["ISS-A1B2C3", "ISS-X7Y8Z9"],
      "new_since_previous": ["ISS-NEW001", "ISS-NEW002"],
      "persistent": ["ISS-D4E5F6", ...]
    }
  ]
}
```

### How it affects the next run

Inject run history into the report generation prompt:

```
## Run history for this product

This is run #4 for this product. Trends:
- Run 1 (Mar 14): 42 issues (2 critical, 8 high)
- Run 2 (Mar 15): 35 issues — 10 resolved, 3 new
- Run 3 (Mar 16): 28 issues — 9 resolved, 2 new  
- This run: evaluating...

Persistent issues (not fixed across 3+ runs):
- "Button color inconsistency on hover" — persisted 3 runs, likely intentional or low priority
- "No loading indicator on search" — persisted 3 runs, repair brief may need more detail

Previously resolved issues to watch for regression:
- "Missing privacy policy" — fixed in run 2, verify still fixed

When writing repair briefs for persistent issues, be MORE specific than previous briefs.
When an issue reappears after being resolved, mark it as a REGRESSION with high severity.
```

---

## Feedback Loop 3: Learn the Product Baseline

### How it works

After the first run, Preflight saves a "baseline" of what the product looks like — key page states, navigation structure, expected elements. On subsequent runs, it uses this baseline to distinguish between "this is how the product always looks" (not an issue) and "this changed since last time" (potential regression or improvement).

### Storage: `product_baseline.json`

```json
{
  "established_at": "2026-03-14T15:00:00Z",
  "last_updated": "2026-03-15T15:00:00Z",
  "pages": {
    "/": {
      "title": "LDN/ai — London AI Community",
      "key_elements": ["navigation bar", "hero section", "events list", "footer"],
      "known_states": ["Events section shows upcoming events", "Hero has signup CTA"]
    },
    "/events": {
      "title": "Events — LDN/ai",
      "key_elements": ["event cards", "filter bar", "pagination"]
    }
  },
  "known_design_choices": [
    "Dark theme with purple accents — intentional brand choice, not a bug",
    "Events page uses card grid layout"
  ],
  "known_limitations": [
    "No search functionality — confirmed by user as not-yet-built feature"
  ]
}
```

### How it affects the next run

Inject baseline into evaluation prompts:

```
## Product baseline (from previous runs)

This product has been evaluated before. Known states:
- Homepage has: navigation bar, hero section, events list, footer
- Events page has: event cards, filter bar, pagination

Known design choices (don't flag these):
- Dark theme with purple accents is intentional
- Card grid layout on events page is intentional

Known limitations (user confirmed, don't flag unless changed):
- No search functionality (not yet built)

Focus on:
- Changes from the baseline (new elements, missing elements, layout shifts)
- Issues NOT in the known limitations list
- Regressions from previously working states
```

---

## Feedback Loop 4: Learn Evaluation Patterns

### How it works

Track which prompts and evaluation strategies produce the most useful findings. Over time, optimize the approach.

### Storage: `learned_priorities.json`

```json
{
  "high_value_categories": ["trust", "functional", "responsive"],
  "low_value_categories": ["design_minor", "copy"],
  "effective_personas": [
    {"type": "first_time_user", "avg_useful_findings": 4.2},
    {"type": "skeptical_buyer", "avg_useful_findings": 3.8},
    {"type": "mobile_user", "avg_useful_findings": 5.1}
  ],
  "ineffective_personas": [
    {"type": "power_user", "avg_useful_findings": 0.8, "avg_noise": 3.2}
  ],
  "high_value_flows": ["onboarding", "login", "checkout"],
  "low_value_flows": ["footer_links", "about_page"]
}
```

### How it affects the next run

- Prioritize persona types that historically produce useful findings
- Deprioritize or skip persona types that produce mostly noise
- Spend more time on flows that yield high-value findings
- Skip flows that consistently produce nothing useful

---

## Implementation

### New module: `preflight/core/memory.py`

```python
class ProductMemory:
    """Persistent memory for a product across evaluation runs."""
    
    def __init__(self, product_url: str, memory_dir: str = "~/.preflight/memory"):
        self.slug = self._url_to_slug(product_url)
        self.base_dir = Path(memory_dir).expanduser() / self.slug
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    # Feedback
    def record_rating(self, issue_id: str, issue_title: str, category: str, rating: str): ...
    def record_missed_issue(self, description: str): ...
    def get_feedback_context(self) -> str: ...
    
    # Run history
    def record_run(self, result: RunResult): ...
    def get_run_history_context(self) -> str: ...
    def get_persistent_issues(self, min_runs: int = 3) -> list[dict]: ...
    def get_resolved_issues(self) -> list[dict]: ...
    
    # Baseline
    def update_baseline(self, result: RunResult): ...
    def get_baseline_context(self) -> str: ...
    
    # Learned priorities
    def update_priorities(self): ...  # Called after feedback
    def get_priority_context(self) -> str: ...
    
    # Combined context for prompts
    def get_full_context(self) -> str:
        """Returns all learned context formatted for injection into LLM prompts."""
        sections = []
        feedback = self.get_feedback_context()
        if feedback:
            sections.append(feedback)
        history = self.get_run_history_context()
        if history:
            sections.append(history)
        baseline = self.get_baseline_context()
        if baseline:
            sections.append(baseline)
        priorities = self.get_priority_context()
        if priorities:
            sections.append(priorities)
        return "\n\n".join(sections)
```

### Pipeline integration

In `pipeline.py`, load memory at the start and inject context into prompts:

```python
async def run_pipeline(config: RunConfig) -> RunResult:
    memory = ProductMemory(config.target_url)
    learned_context = memory.get_full_context()
    
    # Pass learned_context to intent modeler, persona generator, 
    # evaluation prompts, and report generation
    
    # ... run pipeline ...
    
    # After completion, record this run
    memory.record_run(result)
    memory.update_baseline(result)
```

### CLI commands

```bash
# Rate findings from last run
preflight feedback ./artifacts

# View what Preflight has learned about a product
preflight memory https://LondonAI.network

# Reset memory for a product (start fresh)
preflight memory https://LondonAI.network --reset

# Show run history
preflight memory https://LondonAI.network --history
```

### Interactive feedback after run

After a run completes, optionally ask:

```
Evaluation complete! 28 issues found.

Would you like to rate the findings? This helps Preflight learn what matters to you.
[y]es / [n]o / [l]ater > y

1/28: [HIGH] Missing privacy policy link in footer
      Rate: [u]seful / [n]oise / [s]kip > u

2/28: [MEDIUM] Button hover state inconsistency on nav bar  
      Rate: [u]seful / [n]oise / [s]kip > n
      (Noted: will reduce priority for minor design consistency issues)

3/28: [HIGH] Login form shows no error on invalid email
      Rate: [u]seful / [n]oise / [s]kip > u

... (user can Ctrl+C to stop rating at any time)

Anything we missed? (describe, or Enter to skip) >
The mobile hamburger menu doesn't close after tapping a link

Thanks! Preflight will remember these preferences for next time.
Feedback saved to ~/.preflight/memory/londonai-network/
```

---

## How Learning Manifests

### Run 1 (no memory): 
Generic evaluation. 42 issues, mix of useful and noise.

### Run 2 (with feedback from run 1):
- Suppresses issue types user marked as noise
- Boosts severity for categories user found useful
- Actively looks for patterns similar to missed issues
- Compares against run 1: flags regressions, notes resolved issues
- 28 issues, higher signal-to-noise ratio

### Run 5 (significant memory):
- Knows the product's design language — stops flagging intentional choices
- Knows which personas produce the best findings — allocates more time to them
- Knows which flows are high-value — focuses evaluation there
- Repair briefs are more specific because persistent issues get enhanced descriptions
- 15 issues, almost all genuinely actionable

### Run 10+:
- Essentially a custom QA team trained on this specific product
- Only reports on changes, regressions, and genuinely new issues
- Repair briefs reference historical patterns ("this broke before in run 7 when...")
- Persona mix is optimized for this product's specific weaknesses

---

## Build Order

1. `preflight/core/memory.py` — ProductMemory class with all storage methods
2. `preflight feedback` CLI command — interactive rating of findings
3. `preflight memory` CLI command — view/reset learned context
4. Wire memory into pipeline.py — load at start, record at end
5. Inject feedback context into evaluation prompts (web_runner, lenses)
6. Inject run history context into report generation and issue synthesis
7. Inject baseline context into evaluation prompts
8. Inject priority context into persona generator and journey planner
9. Auto-prompt for feedback after interactive runs (optional, user can skip)
10. Tests
11. Push

## Constraints

- All memory is LOCAL. Never sent to any server. Privacy first.
- Memory files are human-readable JSON. Users can inspect and edit them.
- First run with no memory must work exactly as well as today — memory only improves things.
- Memory injection adds context to prompts, never replaces core evaluation logic.
- Keep memory context under 2000 tokens to avoid inflating LLM costs.
- Feedback is always optional. User is never forced to rate findings.
