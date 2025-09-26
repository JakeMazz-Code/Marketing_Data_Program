
# CODEX_PROMPTS.md — Master & Micro Prompts (Updated)
**Generated:** 2025-09-25 01:37:47 UTC

These are exact, copy-paste prompts for a coding LLM. They assume the daily-master bundle and the final two files have been integrated via `configs/daily_master.json`. Sources & mappings follow your package notes and sample config. fileciteturn0file0 fileciteturn0file1

---

## MASTER PROMPT — "Ship the No‑BS Founder Pack"
**Role**: You are a senior analytics engineer. Improve a pandas + Streamlit repo that reads a **daily master** CSV and emits KPIs, JSON artifacts, charts, and a founder brief via an LLM. Your job is to make small, surgical PRs that pass CI and produce grounded insights. **Never invent metrics or TAM.**

**Inputs available**
- Master joined daily panel with: `sales_day__*`, `adspend_day__*`, `orders__*`, `engagement_day__*`, `views_day__*`.
- Promoted config at `configs/daily_master.json` declaring mappings and derived KPIs (MER/ROAS, CAC, AOV).
- Streamlit shell, reporting utilities, optional AI summarizer.

**Tasks**
1) Promote & enforce config + data contract; emit `llm_payload.json` with only aggregated KPIs.
2) Add a metrics registry; refactor callers.
3) Add a quality gate; block bad runs; write `quality_report.md`.
4) Add STL-based anomaly detection; write `anomalies.json`.
5) Add attribution/efficiency tables; degrade gracefully when channel/campaign missing.
6) Add creative intelligence (if creative fields present); otherwise skip.
7) Add margin/leakage waterfall from daily sales components.
8) Implement LLM router + verifier that clamps unrealistic lifts and cites baselines.
9) Build Streamlit pages reading the artifacts.
10) Add CI (ruff/black/mypy/pytest) and `.env.example`.

**Output requirements**
- Provide unified diffs per file modified; add unit tests & fixtures.
- Ensure `python cli/marketing_analysis.py --config configs/daily_master.json` runs end‑to‑end and `streamlit run dashboard.py` renders all pages.
- No PII to LLM; use the data contract.

**Acceptance**: Follow `PATCHES.md` acceptance criteria verbatim.

---

## MICRO PROMPTS — One PR at a Time

### PR1 — Config & Contract
Implement `configs/daily_master.json` loading in `pipeline.py`. Required columns: `date`, `sales_day__net_sales`→`revenue`, `sales_day__orders`→`orders`, `adspend_day__amount_spent_usd`→`ad_spend`, `views_day__primary`→`views`. Optional engagement and sales components map exactly as in the config. Compute MER/ROAS, CAC, AOV with zero‑guards. Export `reports/daily_master/llm_payload.json` (aggregated only). Add `tests/test_config_contract.py` to validate schema, types, and guards.

### PR2 — Metrics Registry
Create `metrics_registry.py` exposing `compute(metric_name, df)` for MER, ROAS, CAC, AOV (+ placeholder for contribution margin). Add tests for 0 spend, 0 orders, and NaN propagation. Refactor existing KPI code to use the registry.

### PR3 — Quality Gate
Create `quality.py` w/ checks: schema/type, missing dates vs calendar, freshness (data includes last N days), numeric ranges, duplicates. Write `quality_report.md` + JSON; block on FAIL and attach caveats on WARN. Add `tests/test_quality.py` with fixtures for each rule.

### PR4 — Anomalies
Create `anomaly.py` using STL on revenue and MER. Compute residual z-scores; flag |z|≥2.5; join with spend changes and engagement spikes for likely causes. Export `anomalies.json`. Add synthetic tests.

### PR5 — Attribution & Efficiency
Create `efficiency.py`. If columns `channel`/`campaign` exist, group daily MER/CAC and produce ranked reallocation suggestions (shift $ from bottom to top performers). If missing, produce overall summary only. Export `efficiency.json`. Tests for both branches.

### PR6 — Creative Intelligence
Create `creative.py`. Compute elasticity of purchase proxy vs views and a fatigue marker when impressions↑ but ROAS↓. Export `creatives.json`. Tests with toy data.

### PR7 — Margin & Leakage
Create `margin.py`. Build a daily waterfall using `discount_amount`, `returns_amount`, `shipping_amount`, `duties_amount`, `tax_amount`, reconciling to `revenue`. Export `margin.json` and `margin_waterfall.csv`. Tests verify reconciliation.

### PR8 — LLM Router & Verifier
In `ai.py`, add router: 4o‑mini (default), o3 (diagnostic), Claude Sonnet 4 (long context). Enforce a strict JSON schema for the **No‑BS Executive Brief**; implement a **Verifier** that recomputes KPIs from `llm_payload.json`, clamps uplift to 20–40% of the 28‑day baseline unless anomalies indicate structural changes, and requires cited baselines. Add `tests/test_ai_shapes.py`.

### PR9 — Streamlit Pages
Create `streamlit/pages/1_Overview.py`, `2_Channel_Efficiency.py`, `3_Creatives.py`, `4_Margin_Leakage.py`, `5_Anomalies.py`. Pages read artifacts only, render tiles/charts/tables, and display caveats. No PII. Add a smoke test for `dashboard.py` startup.

### PR10 — CI & DX
Add `.github/workflows/ci.yml` running ruff/black/mypy/pytest on PRs. Add `.pre-commit-config.yaml`. Add `.env.example` documenting `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` and warn not to include PII in prompts.

---

## “No‑BS Executive Brief” (paste into your LLM prompts folder)
**System**: You are a marketing analyst for a small apparel brand. Use only the provided daily KPIs. **Never propose TAM/geo/subscription unless present in inputs.** Quantify only with provided metrics. If evidence is insufficient, say so.

**User**:
Inputs: `series` (KpiRow list, last 90 days), `quality_report`, optional `efficiency`, `creatives`, `margin`, `anomalies`.
Tasks:
1) Name **three growth opportunities** grounded in MER, CAC, AOV, channel mix, or proven creative lift. Provide **pess/base/optimistic** ranges tied to the 28‑day baseline.
2) State whether **revenue is accelerating or slowing**. Show 7‑ vs 28‑day trend and driver (spend/mix/conversion proxy).
3) If channels exist, give **budget shifts** ($ from X to Y) with incremental revenue math.
4) If creatives exist, list which **to scale/refresh** with evidence.
5) If margin inputs exist, list **leakage** and quick fixes.
6) End with **three next actions** with owner and “why now.”
Constraints: No single opportunity >40% of 28‑day revenue unless anomalies indicate structural change; use absolute dates; output **valid JSON** keys: `topline`, `kpis`, `opportunities`, `diagnostics`, `actions`.

## “Verifier” (strict)
**System**: You are a reviewer. Reject any claim not supported by the data.

**User**:
Given `series`, `draft_brief_json`, `quality_report`: recompute KPIs; flag any claim >40% of 28‑day baseline w/o causal evidence; fix % deltas; ensure tone is grounded. Return `status: APPROVED` or `status: REDLINE` with `issues[]` and `patched_brief_json`.
