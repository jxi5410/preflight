# Preflight

External-experience AI QA system. Evaluates shipped products like real users would — through the UI only, no code inspection — and produces evidence-backed findings plus repair briefs for coding agents (Claude Code, Codex, Cursor).

## What It Does

1. **Understands your product** — infers purpose, audience, and critical flows from visible surfaces + optional repo analysis
2. **Generates realistic test personas** — dynamically creates a team of user agents tailored to your product (4-6 agents, capped for speed)
3. **Evaluates through the UI** — runs web (Playwright) and mobile (Playwright emulation / Maestro) interactions as real users would
4. **Applies specialist lenses in parallel** — design critique, trust assessment, auth/login flow review, mobile responsiveness check, institutional/governance review
5. **Deduplicates aggressively** — two-pass dedup using error signatures (fast) then LLM semantic clustering
6. **Groups related issues** — clusters findings by category and product area into issue groups
7. **Produces actionable reports** — interactive HTML report with clickable severity cards, inline screenshot galleries, search, and score explanations
8. **Generates developer handoffs** — prioritized tasks with fix options, file mapping, dependency graphs, and verification steps

## Quick Start

```bash
# Install from PyPI
pip install preflight-qa
playwright install chromium

# Quick check (~1-2 min) — fast single-pass evaluation
preflight check https://your-product.com

# Full evaluation with repo context
preflight run https://your-product.com --repo https://github.com/user/repo

# Interactive mode (prompts for everything)
preflight
```

## Supported LLM Providers

Preflight works with any major LLM provider. Pick the one you already use — you just need an API key.

### Quick Setup

```bash
# Google Gemini (default — cheapest, recommended)
export GOOGLE_API_KEY=your-key    # Free from aistudio.google.com

# OpenAI
export OPENAI_API_KEY=sk-...      # From platform.openai.com

# Anthropic Claude
export ANTHROPIC_API_KEY=sk-...   # From console.anthropic.com
```

Then choose your tier when running:

```bash
preflight check https://myapp.com                    # Uses default (balanced/Gemini)
preflight check https://myapp.com --tier balanced     # Gemini 3 Flash + 3.1 Pro
preflight check https://myapp.com --tier budget       # Gemini 2.5 Flash + 3 Flash
preflight check https://myapp.com --tier openai       # GPT-4.1 + GPT-5.4
preflight check https://myapp.com --tier premium      # Claude Sonnet + Opus
```

Or specify models directly:

```bash
preflight run https://myapp.com --provider google --model gemini-3.1-pro-preview
preflight run https://myapp.com --provider openai --model gpt-5.4
preflight run https://myapp.com --provider anthropic --model claude-sonnet-4-20250514
```

### Tier Comparison

| Tier | Provider | Fast Model | Smart Model | Cost/Run | Best For |
|------|----------|-----------|-------------|----------|----------|
| **balanced** (default) | Google | Gemini 3 Flash | Gemini 3.1 Pro | ~$0.50-0.80 | Daily use, most users |
| **budget** | Google | Gemini 2.5 Flash | Gemini 3 Flash | ~$0.20-0.40 | High volume, cost-sensitive |
| **openai** | OpenAI | GPT-4.1 | GPT-5.4 | ~$2-4 | OpenAI ecosystem users |
| **premium** | Anthropic | Claude Sonnet 4.6 | Claude Opus 4.6 | ~$3-5 | Best quality, complex products |

Preflight uses a **tiered model approach** — cheap fast models handle per-page evaluations and screenshot analysis, while smarter models handle product understanding, persona generation, and report synthesis. This keeps costs low without sacrificing judgment quality.

### More Examples

```bash
# Quick check with focus
preflight check https://your-product.com --focus "login flow"

# Full run with options
preflight run https://your-product.com \
  --brief "B2B SaaS dashboard for financial analytics" \
  --credentials '{"email": "test@example.com", "password": "test123"}' \
  --focus "onboarding,search,export" \
  --output ./my-report

# Generate handoff from existing run
preflight handoff ./artifacts --format claude-code

# Compare runs for regressions
preflight compare ./baseline ./current

# Export issues to GitHub
preflight export-issues --repo user/repo --run ./artifacts

# Schedule overnight runs
preflight schedule https://your-product.com --cron "0 2 * * *"
```

## Configuration

Create `preflight.yaml` or pass options via CLI:

```yaml
target:
  url: https://your-product.com
  credentials:
    email: test@example.com
    password: test123

options:
  brief: "Financial analytics dashboard"
  focus_flows:
    - onboarding
    - search
    - export
  personas_hint: "enterprise finance users"
  institutional_review: auto  # auto | on | off
  design_review: true

llm:
  tier: balanced  # balanced | budget | openai | premium
  # Or specify directly:
  # provider: google    # google | openai | anthropic
  # model: gemini-3.1-pro-preview

output:
  dir: ./reports
  formats:
    - markdown
    - json
    - html
    - repair_briefs
```

## Environment Variables

```bash
# LLM Provider (set one based on your chosen tier)
GOOGLE_API_KEY=...          # For balanced/budget tiers (free from aistudio.google.com)
OPENAI_API_KEY=sk-...       # For openai tier
ANTHROPIC_API_KEY=sk-...    # For premium tier

# Optional
GITHUB_TOKEN=ghp_...        # For private repo analysis and GitHub issue export
PREFLIGHT_OUTPUT_DIR=./reports  # Default: ./artifacts
```

## MCP Server (Claude Code / AI Tool Integration)

Preflight exposes an MCP (Model Context Protocol) server so AI coding tools like Claude Code can use it as a tool.

### Setup

Add to your Claude Code MCP configuration (`~/.claude/claude_desktop_config.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "preflight": {
      "command": "preflight-mcp",
      "env": {
        "GOOGLE_API_KEY": "your-key"
      }
    }
  }
}
```

Use `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` instead if you prefer those providers.

### Available Tools

| Tool | Description | Speed |
|------|-------------|-------|
| `preflight_quick_check` | Fast single-pass evaluation | ~30s |
| `preflight_evaluate` | Full multi-agent QA pipeline | 2-5 min |
| `preflight_get_report` | Retrieve existing reports | Instant |
| `preflight_compare` | Compare runs for regressions | Instant |

### Usage in Claude Code

```
"Quick check my staging site at https://staging.myapp.com"
"Run a full evaluation on https://myapp.com with repo https://github.com/org/myapp"
"Show me the Preflight report from the last run"
"Compare the current run against ./baseline for regressions"
```

See [HUMANQA_SKILL.md](HUMANQA_SKILL.md) for the full integration guide.

## Architecture

```
preflight/
├── core/
│   ├── intent_modeler.py    # Infers product purpose from visible surfaces
│   ├── persona_generator.py # Generates tailored user agent team
│   ├── orchestrator.py      # Coordinates agents, dedup, issue grouping
│   ├── pipeline.py          # End-to-end pipeline with timeouts & parallelization
│   ├── performance.py       # Product-type-aware performance budgets
│   ├── file_mapper.py       # Maps issues to likely source files
│   ├── repo_analyzer.py     # GitHub repo analysis (visibility, tech stack, routes)
│   ├── schemas.py           # All data models (issues, groups, evidence, agents)
│   ├── actions.py           # Deterministic browser action engine
│   ├── quick_check.py       # Fast single-pass evaluation for MCP/CI
│   ├── progress.py          # Visual progress tracker
│   └── llm.py               # LLM abstraction (Anthropic / OpenAI / Gemini)
├── runners/
│   ├── web_runner.py        # Playwright-based web evaluation (desktop + mobile)
│   ├── mobile_runner.py     # Mobile web emulation / Maestro native app testing
│   └── page_snapshot.py     # Page state capture (a11y tree, screenshots, metrics)
├── lenses/
│   ├── design_lens.py       # Design/UI quality review
│   ├── trust_lens.py        # Trust signal detection
│   ├── auth_lens.py         # Login/auth flow evaluation (no credentials needed)
│   ├── responsive_lens.py   # Mobile responsiveness & layout comparison
│   └── institutional_lens.py # Governance/provenance/auditability review
├── reporting/
│   ├── report_generator.py  # Markdown, JSON, interactive HTML reports
│   ├── handoff.py           # Developer handoff (HANDOFF.md + handoff.json)
│   ├── comparison.py        # Run-to-run regression comparison
│   ├── github_export.py     # Export issues to GitHub Issues
│   ├── webhook.py           # Slack/webhook notifications
│   └── templates/
│       └── report.html      # Interactive HTML report template
├── mcp_server.py            # MCP server (Claude Code / AI tool integration)
└── scheduling/
    └── scheduler.py         # Cron-based scheduled runs
```

## Evaluation Pipeline

```
Scrape landing page ──► Build intent model ──► Generate personas (max 6)
         │                                            │
    Analyze repo ◄─── (optional)                      ▼
                                          Orchestrate evaluation
                                          (8 steps/journey cap, 5min timeout)
                                                      │
                                                      ▼
                                    ┌─────── Specialist lenses (parallel) ──────┐
                                    │  Design  │  Trust  │ Responsive │  Auth   │
                                    └──────────┴────────┴────────────┴─────────┘
                                                      │
                                        Institutional lens (if applicable)
                                                      │
                                                      ▼
                                          Deduplicate & group issues
                                          (signature pass → LLM pass)
                                                      │
                                          ┌───────────┼───────────┐
                                          ▼           ▼           ▼
                                      report.html  HANDOFF.md  repair_briefs/
                                      report.md    handoff.json
                                      report.json
```

## Key Features

### Evidence-Backed Findings
Every issue cites specific evidence — screenshot references, element selectors from the accessibility tree, measured metrics, or documented absences. Findings without anchored evidence are rejected.

### Screenshot Evidence with Captions
Screenshots include contextual metadata: captions describing what's shown, step references linking to the journey, viewport dimensions, and timestamps.

### Adaptive Performance Budgets
Performance thresholds adapt to product type:
- **Marketing sites** — tight LCP and load budgets (first impressions matter)
- **SaaS apps** — lenient LCP, strict CLS (interactivity over raw speed)
- **E-commerce** — strictest CLS budgets (layout shifts hurt conversion)
- **Content sites** — tightest LCP (fast text rendering)
- **Mobile web** — accounts for slower networks

### Login/Auth Evaluation
Evaluates login pages as product surfaces without requiring credentials — assesses form quality, error handling, trust signals, password reset availability, and accessibility.

### Mobile Responsiveness
Compares desktop vs mobile findings to catch responsive design issues: layout breaks, undersized touch targets, text readability, navigation adaptation, and horizontal overflow.

### Developer Handoff with Fix Options
Each handoff task includes multiple fix strategies with trade-offs:
- **Critical issues** get both quick-patch and proper-fix options
- **Accessibility issues** get ARIA/semantic HTML suggestions
- **Performance issues** get optimization pass options

### Repo Visibility Detection
Automatically detects whether the GitHub repo is public or private, informing trust evaluation and handoff context.

## Output

Each run produces:
- `report.html` — Interactive HTML report with filters, search, clickable severity cards, inline screenshot galleries, and score explanations
- `report.md` — Human-readable prioritized findings
- `report.json` — Machine-readable full export
- `HANDOFF.md` — Developer handoff with tasks, fix options, dependencies
- `handoff.json` — Machine-readable handoff for AI coding tools
- `repair_briefs/` — Per-issue technical guides for coding agents
- Screenshots — Evidence images referenced in reports

## Core Principle

This system **never inspects source code**. All evaluation happens through the observable user experience — the same surface real users see.

## License

MIT
