
# COPY-PASTE PROMPTS FOR CODEX (Data/ aware)
**Generated:** 2025-09-25 02:00:48 UTC

---
GLOBAL SESSION SYSTEM PROMPT

GLOBAL SESSION SYSTEM PROMPT — Data/ layout aware

You are a Senior Analytics Engineer. You will harden a pandas + Streamlit marketing analytics repo. **Context is king**: stay grounded in the files under `Data/` and the config I provide. Never invent numbers, TAM, geos, or subscription strategies that aren’t present.

**Folder contract (do not change unless asked):**
- `Data/master/MASTER_DAILY_JOINED.csv` is the authoritative input for PR1–PR4.
- `Data/inputs/` contains per-source dailies (adspend, sales, orders, views, engagement) used for QA and future joins.
- `Data/docs/` holds the data dictionary, export manifest, and sample config.
- `Data/_archive/` contains “NYU case study” and “results by day.xlsx - * - CLEANED.csv” files; these MUST NOT be read by the pipeline.

**Config contract:**
- Use `configs/daily_master.json` as the single source of truth for column mappings and derived KPI names.
- Only write artifacts under `reports/daily_master/`.
- Never send PII to any LLM; only aggregated daily KPIs.

**Golden rules:**
- Work PR-by-PR. For each PR return: summary, unified DIFFs, commands to run, expected artifacts with sample shapes.
- Cap opportunity lifts to **20–40% of 28-day revenue baseline** unless anomalies prove structural shift (then show evidence).
- If a dependency is missing, propose the smallest stable addition and show `requirements` diff.

---

PR0

PR0 — Repo Intake & Understanding (No Code Changes)

Goal: Confirm we can read from the new `Data/` layout and that `configs/daily_master.json` mappings match `Data/master/MASTER_DAILY_JOINED.csv`. DO NOT edit files yet.

Tasks:
1) Repo map: concise tree (ignore .git, .venv, __pycache__, dist/build). Identify CLI entry points, pipeline, dashboard, config, tests.
2) Data inventory (Data/):
   - `master/MASTER_DAILY_JOINED.csv`, `master/MASTER_DAILY_CURATED.csv` (optional)
   - `inputs/adspend_day - DAILY.csv`, `inputs/sales_day - DAILY.csv`, `inputs/orders - DAILY.csv`, `inputs/views_day - DAILY.csv`, `inputs/engagement_day - DAILY.csv`
   For each: print rows and [min_date, max_date]; flag missing days vs calendar and duplicate dates.
3) Config validation (concept only):
   Confirm in `master/MASTER_DAILY_JOINED.csv` the columns exist for:
     revenue → sales_day__net_sales
     total_sales → sales_day__total_sales
     orders → sales_day__orders
     ad_spend → adspend_day__amount_spent_usd
     views → views_day__primary
     Optional: engagement_day__{reach,likes,comments,shares,follows,saves}
               sales_day__{discounts,returns,shipping_charges,duties,taxes}
4) Scratch KPI profile (read-only):
   Compute per-day revenue, ad_spend, orders, views (+ optional fields), and derived MER/ROAS, CAC, AOV with zero-division guards. Summarize 7/28/90-day averages and revenue slope over the last 6–8 weeks.
5) Quality notes: freshness, missing days, duplicates, non-negativity.
6) Confirmed understanding bullets + Unknowns.

Deliverable: paste `INTAKE_REPORT.md`.


PR1

PR1 — Config & Data Contract (Data/ aware)

Goal: Make `configs/daily_master.json` canonical; enforce schema/types; compute zero-guarded KPIs; write LLM-safe artifacts.

Inputs:
- Config at `configs/daily_master.json` with:
  data_root = "Data"
  data_path = "Data/master/MASTER_DAILY_JOINED.csv"
  date_column = "date"
  mappings: { revenue: sales_day__net_sales, total_sales: sales_day__total_sales, orders: sales_day__orders, ad_spend: adspend_day__amount_spent_usd, views: views_day__primary, optional engagement & sales components }
  derived: { roas: revenue/ad_spend, aov: revenue/orders }
  artifacts_dir = "reports/daily_master"

Tasks:
1) Implement a robust config loader (dataclass or pydantic) and `--config` flag; allow optional `--data_root` override.
2) Validate required columns; enforce dtypes; fill calendar gaps between min and max dates (sales stay NaN; ad/engagement may be 0); zero-guard derived KPIs.
3) Write artifacts under `reports/daily_master/`:
   - series.jsonl (one row/day with all mapped + derived fields)
   - shape.json (rows, min/max dates, null counts)
   - data_quality.json (schema/range/freshness checks per rule)
   - llm_payload.json (AGGREGATES ONLY — no PII)
   KpiRow schema identical to earlier plan.
4) Tests: `tests/test_config_contract.py` validates presence of required fields, date dtype/monotonicity, zero-division guards, and artifact creation.

Output: DIFFs, test file(s). Commands:
- `python cli/marketing_analysis.py --config configs/daily_master.json`
- `pytest -q`


PR2
PR2 — Metrics Registry

Same as earlier; no Data/ path changes. Ensure all KPI math uses the registry and tests cover zero-division & NaN propagation.

PR3
PR3 — Quality Gate

Run checks against the Data/master input; write quality_report.md/json; stop on FAIL when `quality.stop_on_fail` is true.

PR4
PR4 — Anomaly Detection

STL or rolling z-scores on revenue and MER. Artifacts under reports/daily_master/anomalies.json.

PR5
PR5 — Attribution & Efficiency (day-level)

If group columns exist in the master (channel/campaign), compute MER/CAC by group; otherwise output overall summary.

PR6
PR6 — Creative Intelligence

Use cleaned creative files if present (but DO NOT read from Data/_archive). Elasticity & fatigue markers; creatives.json.

PR7
PR7 — Margin & Leakage

Build daily waterfall from optional mapped columns; reconcile to revenue; margin.json + margin_waterfall.csv.

PR8
PR8 — LLM Router & Verifier (No‑BS Brief)

Router: 4o‑mini / o3 / Sonnet 4; Inputs strictly from artifacts; schema-validated outputs; verifier clamps overblown claims.

PR9
PR9 — Streamlit App

Five pages reading artifacts from reports/daily_master; no PII; smooth startup.

PR10
PR10 — CI & DX

CI workflow + pre-commit + .env.example; README quickstart updated for Data/ layout.
