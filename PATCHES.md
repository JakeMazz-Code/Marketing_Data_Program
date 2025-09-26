
# PATCHES.md — Updated Patches (Includes sales_day + views_day)
**Generated:** 2025-09-25 01:37:47 UTC

This list reflects the final two files and the promoted config. Mappings and file expectations align with your **Marketing Data Package** and **sample config**. fileciteturn0file0 fileciteturn0file1

---

## 1) Promote Config & Contracts
**Paths**: `configs/daily_master.json`, `config.py`, `pipeline.py`, `tests/test_config_contract.py`  
**Steps**
- Add `configs/daily_master.json` (content provided in this drop) and load in `pipeline.py`.
- Required fields: `date`, `sales_day__net_sales`→`revenue`, `sales_day__orders`→`orders`, `adspend_day__amount_spent_usd`→`ad_spend`, `views_day__primary`→`views`.
- Optional: reach/likes/comments/shares/follows/saves; discounts/returns/shipping/duties/taxes.
- Compute derived KPIs with zero‑division guards: MER/ROAS, CAC, AOV.
- Emit `reports/daily_master/llm_payload.json` (aggregated only).
**Acceptance**
- `python cli/marketing_analysis.py --config configs/daily_master.json` completes; schema validated; `llm_payload.json` exists and matches registry.

## 2) Metrics Registry
**Paths**: `metrics_registry.py`, `metrics.py`, `tests/test_metrics_registry.py`  
**Steps**
- Centralize MER, ROAS, CAC, AOV (+ contribution margin placeholder). Explicit NaN/zero semantics.
**Acceptance**
- Unit tests for edge cases; all callers refactored to use the registry.

## 3) Quality Gate
**Paths**: `quality.py`, `tests/test_quality.py`  
**Checks**
- Schema/type validation; missing-day range vs calendar; freshness (data includes the last N days); non-negative numeric ranges; duplicates by date; currency assumption = USD.
- Attach WARN caveats; FAIL blocks downstream stages.
**Acceptance**
- `quality_report.md` produced; pipeline stops on FAIL; tests cover key rules.

## 4) Anomalies
**Paths**: `anomaly.py`, `tests/test_anomaly.py`  
**Steps**
- STL on revenue & MER; |z|≥2.5 threshold; correlate w/ spend change and engagement spikes.
**Acceptance**
- Synthetic anomalies detected; low false positives.

## 5) Attribution & Efficiency
**Paths**: `efficiency.py`, `tests/test_efficiency.py`  
**Steps**
- When `channel`/`campaign` columns exist, compute MER/CAC and rank; otherwise degrade to overall metrics.
**Acceptance**
- Ranked table with suggested reallocation deltas; JSON exported.

## 6) Creative Intelligence
**Paths**: `creative.py`, `tests/test_creative.py`  
**Steps**
- Elasticity (views→purchase proxy) and fatigue detection (impressions↑, ROAS↓). Leaderboard JSON for Streamlit.
**Acceptance**
- Outputs reconcile with inputs; tests pass on toy data.

## 7) Margin & Leakage
**Paths**: `margin.py`, `tests/test_margin.py`  
**Steps**
- Daily waterfall from discount/returns/shipping/taxes/duties; reconcile to revenue.
**Acceptance**
- Waterfall totals match daily revenue; leakage table exported.

## 8) LLM Router & Verifier
**Paths**: `ai.py`, `prompts/`, `tests/test_ai_shapes.py`  
**Steps**
- Router: 4o‑mini (default); o3 (diagnostic); Sonnet 4 (long context). Force structured JSON schema.
- Verifier: recompute KPIs from `llm_payload.json`; clamp claims to bounded uplift unless anomalies justify; require cited baselines.
**Acceptance**
- Briefs pass verifier; no “$10M lift” style hallucinations.

## 9) Streamlit App
**Paths**: `dashboard.py`, `streamlit/pages/*.py`  
**Steps**
- Implement 5 pages reading `reports/daily_master/*` artifacts; no PII displayed.
**Acceptance**
- `streamlit run dashboard.py` succeeds; interactions are smooth.

## 10) CI & DX
**Paths**: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `pyproject.toml`  
**Steps**
- ruff/black/mypy/pytest in CI; minimal coverage threshold; pre‑commit hooks.
**Acceptance**
- PRs require passing CI; formatting & typing enforced.

## 11) Security & PII
**Paths**: `.env.example`, `.gitignore`, `docs/security.md`  
**Steps**
- Document keys (OPENAI_API_KEY, ANTHROPIC_API_KEY). Add redaction utility for any future customer-level content.
**Acceptance**
- No PII in LLM payloads; security checklist in PR template.
