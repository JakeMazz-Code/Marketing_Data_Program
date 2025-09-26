# DASHBOARD.md — Streamlit Overview

The Streamlit app reads only the JSON/MD artifacts under `reports/daily_master/` and displays aggregated KPIs without ever touching raw rows or PII.

## Pages
- **Overview** – KPI tiles (Revenue, Ad Spend, Orders, MER, CAC, AOV) with 7/28-day deltas, revenue/MER trend with MA(7/28), quality banner, and a verified-brief snapshot.
- **Trends & Seasonality** – Revenue & MER trend with date picker, optional uploaded `trend.png`, and a weekly heatmap built from `series.jsonl`.
- **Anomalies** – Top |z| anomalies table, revenue timeline markers, and `anomalies_notes.md` link.
- **Efficiency** – Renders `efficiency.json` (channels/campaigns, reallocation guidance) when available; otherwise shows a friendly placeholder.
- **Margin & Leakage** – Visualises `margin_waterfall.json` when present and falls back to raw JSON.
- **Cohorts & LTV** – Displays cohort/LTV artifacts when the optional agent writes them.
- **Settings & Run** – Artifact directory, cache-refresh button, tail of `etl.log`, and read-only environment knob table.

## Performance & Fallbacks
- All file reads are wrapped in `st.cache_data` keyed by `path + mtime`.
- Missing artifacts return `st.info("Not available yet.")` so pages stay usable on first run.
- Charts use Plotly for fast interactivity; missing columns simply suppress the chart.
- No PII or raw-level data is loaded; everything comes from `series.jsonl`, `llm_payload.json`, or other aggregated artifacts.

## Running the App
```
python manage.py ui
```
This proxies to `streamlit run dashboard.py`. The entry script simply calls `Marketing_analytics.dashboard.main()`.

