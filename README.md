# Marketing Analytics Daily Master

A reproducible marketing intelligence pipeline and Streamlit dashboard for daily aggregated KPIs. The system ingests canonical CSV exports (`sales_day`, `adspend_day`, `orders`, `engagement_day`, `views_day`), rebuilds gap-free KPI series, runs quality and anomaly agents, enforces a no-bull brief schema, and renders a Plotly-based UI without ever reading raw PII rows.

## Table of contents
- [What this is](#what-this-is)
- [Data inputs](#data-inputs)
- [How it works (agents + pipeline)](#how-it-works-agents--pipeline)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Windows (PowerShell)](#windows-powershell)
  - [macOS / Linux (bash)](#macos--linux-bash)
- [.env and environment variables](#env-and-environment-variables)
- [Runbook](#runbook)
  - [Quickstart (smoke ETL + brief + UI)](#quickstart-smoke-etl--brief--ui)
  - [Daily operator flow](#daily-operator-flow)
  - [Weekly deep-dive](#weekly-deep-dive)
- [Common manage.py commands](#common-managepy-commands)
- [Artifacts & file conventions](#artifacts--file-conventions)
- [Dashboard guide](#dashboard-guide)
- [Verification checklist](#verification-checklist)
- [Troubleshooting](#troubleshooting)
- [Development notes](#development-notes)

## What this is
Purpose. Deliver an explainable, operator-friendly view of daily growth KPIs by combining cleaned CSV exports with automated quality gates, anomaly detection, and a verified founder brief.

Design goals. Zero raw PII in the UI, Markdown-first brief (with optional JSON verifier path), two commands from data ingestion to executive brief, and a cache-friendly dashboard that runs the same on Windows/macOS/Linux.

What you get. A local pipeline that writes reusable artifacts under `reports/daily_master/`, JSON-only payloads that can feed other tooling, and a Streamlit dashboard with Overview, Trends, Anomalies, Efficiency, Margin, Cohorts, and Settings pages.

## Data inputs
- `Data/master/MASTER_DAILY_JOINED.csv` (or configured path) containing joined daily KPIs.
- Core columns (daily grain): revenue, ad spend, orders, views, reach/likes/comments/shares/follows/saves, discounts, returns, shipping, duties, tax.
- Canonical mapping stored in `configs/daily_master.json`.
- Optional: channel/campaign/SKU level columns for downstream agents (efficiency, creative, product).
- Optional: Shopify line item export (e.g., `Data/orders_export_*.csv`) with `created_at`, `lineitem_title`, `lineitem_quantity`, and `line_total`. `lineitem_sku` may be blank; the ETL normalises titles, assigns a `variant` fallback, and groups hero SKUs by product name when codes are missing.

## How it works (agents + pipeline)
```
configs/daily_master.json
        |
        v
Intake Agent ---> Quality Gate ---> KPI Aggregator ---> Anomaly Detective ---> Attribution
    |                    |                    |                     |                    |
    |                    |                    |                     |                    +-- efficiency.json (optional)
    |                    |                    |                     +-- anomalies.json + anomalies_notes.md
    |                    |                    +-- kpi_summary.json + trend.png
    |                    +-- quality_report.json/.md, data_quality.json
    +-- series.jsonl, shape.json, llm_payload.json
                                 |
                                 v
Margin & Leakage ---> Narrative & Action ---> Verifier
        |                    |                     |
        +-- margin_waterfall.json  +-- brief.md / brief_raw.txt (MD)  +-- brief_draft.json  +-- brief_verified.json + brief_notes.md (verified)
```
All agents operate on aggregated daily rows. The LLM stage defaults to a Markdown-only Responses call that clamps opportunity lifts to 40% of the 28-day revenue baseline. The legacy JSON + verifier path remains available behind `--generate-brief` when needed.

## Repository layout
```
Marketing_proj_clean/
  configs/
    daily_master.json           # canonical daily mapping
  Marketing_analytics/
    ai.py                       # schema + brief routing
    daily_master.py             # ETL orchestrator
    dashboard.py                # Streamlit pages
    quality.py, anomaly.py, ... # supporting agents
  reports/
    daily_master/               # generated artifacts (JSON/MD only)
  manage.py                     # CLI wrapper (etl/brief/ui/...)
  dashboard.py                  # entry point (calls Marketing_analytics.dashboard.main)
  requirements.txt
  README.md (this file)
```

## Requirements
- Python 3.10+ (tested on 3.12)
- pandas, numpy, plotly, streamlit, pytest (installed via `requirements.txt`)
- Optional: OpenAI and Anthropic Python SDKs for LLM stages
- No external database; artifacts are local JSON/CSV/MD

## Installation
### Windows (PowerShell)
```
git clone <repo-url>
cd Marketing_proj_clean
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
### macOS / Linux (bash)
```
git clone <repo-url>
cd Marketing_proj_clean
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## .env and environment variables
Populate a `.env` (or export variables in your shell) before running the brief. Environment variables always override `.env` values.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for MD-only brief drafting via OpenAI Responses API. | unset |
| `OPENAI_MODEL_ANALYSIS` | Optional model override (`gpt-5`, `gpt-5-mini`, etc.). | `gpt-5` |
| `DEFAULT_SEASONALITY` | Hero SKU seasonality multiplier used in MD reports. | `1.20` |
| `DEFAULT_SAFETY_STOCK` | Safety stock multiplier used in MD reports. | `1.15` |
| `OPENAI_MAX_OUTPUT_TOKENS` | Caps Responses output tokens (MD + verified flows). | `3000` |
| `OPENAI_SERIES_DAYS` | (Verified flow only) Max trailing days passed to the JSON payload. | `90` |
| `OPENAI_TOP_ANOMALIES` | (Verified flow only) Anomaly runs sent to the draft. | `10` |
| `OPENAI_REASONING_EFFORT` | (Verified flow) Optional reasoning budget for o3/gpt-5 models. | unset |
| `ANTHROPIC_API_KEY` | (Verified flow) Enables Anthropic verifier. | unset |
| `ANTHROPIC_MODEL_VERIFIER` | (Verified flow) Anthropic model id. | `claude-opus-4-1-20250805` |
| `BRIEF_SKIP_VERIFIER` | (Verified flow) Set to `1` to skip Anthropic verification. | `0` |

`.env` example:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL_ANALYSIS=gpt-5
ANTHROPIC_API_KEY=sk-anthropic...
```

## Runbook
### Quickstart (smoke ETL + brief + UI)
```
python -m venv .venv && .\.venv\Scripts\activate         # or source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py etl --quick                                # smoke KPI artifacts
.\.venv312\Scripts\python.exe cli\marketing_analysis.py --config configs\daily_master.json --generate-brief --md-only
python manage.py ui                                         # launches Streamlit dashboard
```
Artifacts land in `reports/daily_master/`, including `series.jsonl`, `quality_report.md`, `anomalies.json`, `brief.md`, and `brief_raw.txt`.

### Daily operator flow
1. Activate virtualenv and export API keys.
2. `python manage.py etl --quick` (fast smoke; use `--full` for full write).
3. `python manage.py schema --check` (should print `OK`).
4. `.\.venv312\Scripts\python.exe cli\marketing_analysis.py --config configs\daily_master.json --generate-brief --md-only`.
5. `python manage.py ui` and review Overview, Anomalies, Margin pages.
6. Note caveats from `quality_report.md` and action items from `brief.md`.

### Weekly deep-dive
1. `python manage.py etl --full --debug` to capture stage timestamps.
2. Review `reports/daily_master/quality_report.md` for WARN/FAIL rules.
3. Open dashboard, export trend/efficiency/margin screenshots.
4. `python manage.py brief --md` to refresh `brief.md` (add `--verify` if you need the JSON route).
5. Archive outputs or `python manage.py clean --yes` after backups.

## Common manage.py commands
| Command | Description |
| --- | --- |
| `python manage.py schema --check` | Lints the brief schema (fails fast on missing `type/properties/required/additionalProperties:false`). |
| `python manage.py etl [--quick|--full] [--debug]` | Runs the daily master ETL. `--quick` is KPI smoke (default). `--full` forces the full pipeline. `--debug` prints stage banners + artifact timestamps. |
| `python manage.py brief --md` | Markdown-only brief writer (outputs `brief.md` + `brief_raw.txt`). Use `python manage.py brief [--model MODEL] [--days N] [--tokens N]` for the legacy JSON/verifier route. |
| `python manage.py ui` | Launches the Streamlit dashboard (`streamlit run dashboard.py`). |
| `python manage.py status` | Prints `status.json` stage/pct/time if present. |
| `python manage.py doctor` | Checks Python/version info, package presence, config/artifact paths, and key availability. |
| `python manage.py clean [--yes]` | Deletes `reports/daily_master/*` (asks for confirmation unless `--yes`). |
| `python manage.py test` | Runs `pytest -q`. |
| `python manage.py models` | Calls `/v1/models` via OpenAI SDK and highlights `OPENAI_MODEL_ANALYSIS`. |

Exit codes: `0` success, `1` failure (schema violations, pipeline errors, missing config/API keys), `2` invalid CLI usage.

## Artifacts & file conventions
All outputs live under `reports/daily_master/`.

| File | Purpose |
| --- | --- |
| `series.jsonl` | Gap-free daily KPI records (date, revenue, ad_spend, orders, views, derived ratios, engagement). |
| `shape.json` | Dataset shape, min/max dates, null counts. |
| `data_quality.json` / `quality_report.md` | Quality gate verdicts (PASS/WARN/FAIL) with rule details. |
| `llm_payload.json` | Aggregated 7/28/90-day windows with derived MER/ROAS/CAC/AOV. |
| `anomalies.json` / `anomalies_notes.md` | STL residual anomaly runs with context and operator notes. |
| `margin_waterfall.json` | Daily revenue-to-contribution breakdown when margin agent runs. |
| `sku_series.jsonl` | Per-day hero SKU rollup (date, sku_code, sku, product_title, variant, units, revenue, shares, 28-day run rate). Falls back to normalized product titles when SKU codes are missing. |
| `efficiency.json` | 28-day paid-channel snapshot with raw revenue/spend/orders per channel/creative (`scope: "paid"`). Downstream tables compute MER/CAC/CTR with zero guards. |
| `brief.md` | Markdown brief produced by the MD-only flow. |
| `brief_raw.txt` | Exact model output for audits (MD-only flow). |
| `brief_draft.json` | (Verified flow) Raw OpenAI Responses draft. |
| `brief_verified.json` | (Verified flow) Post-verifier brief (clamped + Anthropic feedback). |
| `brief_notes.md` | (Verified flow) Verifier status and notes. |
| `trend.png` | Optional static trend image surfaced in the dashboard. |
| `status.json` | Progress telemetry for long-running jobs (optional). |

## Dashboard guide
- **Overview**: KPI tiles (Revenue, Ad Spend, Orders, MER, CAC, AOV) with 7/28-day deltas; revenue + MER trend with MA(7/28); quality banner; verified brief card.
- **Trends & Seasonality**: Revenue/MER trend with date filters, optional `trend.png`, and weekly heatmap built from `series.jsonl`.
- **Anomalies**: Top |z| anomalies table, Plotly markers on revenue chart, link to `anomalies_notes.md`.
- **Efficiency**: Renders `efficiency.json` (channel/campaign MER/CAC) when present; otherwise shows a friendly placeholder.
- **Margin & Leakage**: Plotly waterfall for `margin_waterfall.json` and raw JSON fallback.
- **Cohorts & LTV**: Displays cohort/LTV artifacts when the optional agent writes them.
- **Settings & Run**: Artifact directory, cache refresh button, tail of `etl.log`, and read-only environment knob table.

All reads are cached via `st.cache_data` keyed by file path + mtime; missing files trigger informative `st.info` placeholders.

## Verification checklist
- `python manage.py schema --check` -> prints `OK` (exit 0).
- `python manage.py etl --quick` -> refreshes `series.jsonl`, `data_quality.json`, `anomalies.json`.
- `python manage.py brief --md` -> writes `reports/daily_master/brief.md` + `brief_raw.txt` (requires `OPENAI_API_KEY`).
- `python manage.py test` -> pytest suite green.
- `python manage.py ui` -> dashboard loads in under ~3 seconds with placeholders for missing optional artifacts.

## Troubleshooting
| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `[Brief] Schema check failed - fix these first:` | (Verified flow) Schema linter detected a non-strict object node. | Update `Marketing_analytics/ai.py::build_brief_schema()` so every object sets `type/properties/required/additionalProperties:false`; rerun schema check. |
| `OPENAI_API_KEY environment variable is required` | API key not exported. | `setx`/`export` the key or load `.env` before running `manage.py brief`. |
| `context_length_exceeded` | Payload too large for selected model. | Reduce `OPENAI_SERIES_DAYS`, lower `OPENAI_TOP_ANOMALIES`, or switch to higher-context model. |
| Missing `series.jsonl`/`quality_report.json` | ETL not run or failed. | `python manage.py etl --quick` and review stdout for errors. |
| Dashboard shows placeholders only | Optional artifacts absent (e.g., efficiency, margin). | Run corresponding agents (via ETL) or accept placeholders. |
| `python manage.py ui` fails on Windows execution policy | PowerShell script execution disabled. | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in elevated PowerShell, then retry. |

## Development notes
- Code style: keep patches ASCII, use succinct comments for non-obvious logic.
- Do not alter metric math or artifact shapes without coordinating with downstream agents.
- Tests: `python manage.py test` (pytest -q) covers schema, router, anomalies, pipeline, and quality gates.
- Dashboard reads aggregated artifacts only; ensure new data stays anonymized.
- `BRIEF_JSON_SCHEMA` (verified flow) lives in `Marketing_analytics/ai.py`; keep schema updates synchronized with tests (`tests/test_brief_schema_strict.py`) when you use the JSON route.

