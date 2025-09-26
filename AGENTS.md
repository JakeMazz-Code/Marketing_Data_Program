
# AGENTS.md — Updated Agent System (Daily Master + Final Files)
**Generated:** 2025-09-25 01:37:47 UTC

This version incorporates your final two files: **sales_day - DAILY.csv** and **views_day - DAILY.csv**, in addition to ad spend, orders, and engagement. The agents operate only on **aggregated daily KPIs** and never transmit PII.

> **Core inputs (daily grain):** sales_day, adspend_day, orders, engagement_day, views_day. The canonical mappings come from your **sample config** (promoted to `configs/daily_master.json`). fileciteturn0file0 fileciteturn0file1

---

## Router & Guardrails
- **Routing**
  - Daily/weekly brief + structuring → OpenAI **GPT‑4o‑mini** (default)
  - Deep “what changed & why” → OpenAI **o3**
  - Read-long-docs audits → Anthropic **Claude Sonnet 4** (1M ctx)
- **Grounding**
  - Inputs: date, revenue, ad_spend, orders, views, reach/likes/comments/shares/follows/saves, and derived MER/ROAS, CAC, AOV.
  - **No TAM/geo/subscription claims** unless present in inputs.
  - **Bounded uplift**: proposals must be tied to 28–90 day baselines; if data is insufficient → “insufficient evidence.”

**Shared schema**
```yaml
KpiRow:
  date: YYYY-MM-DD
  revenue: float?           # sales_day__net_sales
  total_sales: float?       # sales_day__total_sales
  orders: int?              # sales_day__orders
  ad_spend: float?          # adspend_day__amount_spent_usd
  views: int?               # views_day__primary
  reach: int?               # engagement_day__reach
  likes: int?
  comments: int?
  shares: int?
  follows: int?
  saves: int?
  discount_amount: float?   # sales_day__discounts
  returns_amount: float?    # sales_day__returns
  shipping_amount: float?   # sales_day__shipping_charges
  duties_amount: float?     # sales_day__duties
  tax_amount: float?        # sales_day__taxes
  mer: float?               # revenue / ad_spend if ad_spend>0
  roas: float?              # alias of mer
  cac: float?               # ad_spend / orders if orders>0
  aov: float?               # revenue / orders if orders>0
```
Mappings & file usage are defined in `configs/daily_master.json`. fileciteturn0file1

---

## Agents (10)
1) **Intake Agent** — load `MASTER_DAILY_JOINED.csv`, apply mappings, compute MER/ROAS, CAC, AOV with zero‑guards; fill missing calendar dates; output `series.jsonl`, `shape.json`, `data_quality.json`.
2) **Quality Gate** — schema/type/range/freshness/missing-day checks; writes `quality_report.md` + JSON.
3) **KPI Aggregator** — 7/28‑day rollups, slopes, WoW/MoM deltas; exports `kpi_summary.json` and `trend.png`.
4) **Anomaly Detective** — STL residuals on revenue & MER; flags |z|≥2.5; correlates with spend deltas; outputs `anomalies.json`.
5) **Attribution & Efficiency** — channel/campaign day-level MER/CAC if columns exist; ranked reallocation suggestions.
6) **Creative Intelligence** — elasticity (views→purchase proxy), fatigue markers; outputs creative leaderboard.
7) **Product & Hero SKU** *(optional)* — when SKU arrives; identifies demand-pull, bundles; naive Q4/Q1 forecast needs COGS or a margin %.
8) **Margin & Leakage** — daily waterfall using discounts/returns/shipping/taxes/duties; reconciles to revenue.
9) **Narrative & Action** — “No‑BS” founder brief (grounded, bounded impact, cites exact dates/metrics).
10) **Verifier** — recomputes KPIs from series; clamps unrealistic lifts; returns APPROVED/REDLINE with fixes.

---

## Streamlit Pages
- **1_Overview** — KPI tiles; 7/28‑day trend; caveats banner from Quality Gate.
- **2_Channel_Efficiency** — MER/CAC by channel/campaign; reallocation suggestions.
- **3_Creatives** — leaderboard; fatigue markers; quick test ideas.
- **4_Margin_Leakage** — daily waterfall; top leakage days.
- **5_Anomalies** — flagged dates; spend/engagement context.

All pages read JSON artifacts from `reports/daily_master/*` and avoid PII.
