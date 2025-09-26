# Marketing Data Package (ZIP)
**Generated:** 2025-09-25 00:50:28

This package contains:
- Master daily datasets (CURATED and JOINED)
- Per‑source daily aggregates
- Cleaned source extracts
- A data dictionary (column lineage) and an export manifest
- A **sample config** you can drop into your pipeline

---

## Priority Order (Start Here)
1) **MASTER_DAILY_CURATED.csv** — lean daily KPI panel for quick analysis
   - Columns: 
   - Use for: trendlines, core KPIs (Revenue, Spend, Orders, Views), quick ROAS/AOV.
   - Ideal for dashboards & sanity checks.

2) **MASTER_DAILY_JOINED.csv** — comprehensive daily panel (all sources merged by date)
   - Many columns grouped by prefixes:
   (No columns detected – if this shows empty, re-open in Excel/Sheets to view.
   - Use for: deeper analysis, building feature sets, feeding your report builder.
   - Tip: filter to the prefixes you care about; keep `date` as your join key.

3) **Per‑source daily aggregates**
   - `adspend_day - DAILY.csv` — ad spend by date (USD)
   - `sales_day - DAILY.csv` — revenue components by date (gross, discounts, returns, net sales, taxes, shipping, duties, etc.)
   - `orders - DAILY.csv` — orders export aggregated by `paid_at` date (plus a `row_count` for volume)
   - `engagement_day - DAILY.csv` — post-level engagement summed by date (views, reach, likes, shares, comments, follows, saves)
   - `views_day - DAILY.csv` — normalized “organic + paid views by day” (fixed from a nonstandard CSV)

4) **Cleaned source extracts**
   - Normalized headers, types, and encodings; 1:1 with the raw files you uploaded.
   - Helpful for drill-downs and re-aggregation.

5) **Metadata & helpers**
   - `DATA_DICTIONARY_columns_mapping.csv` — maps joined columns back to original source columns (lineage).
   - `EXPORT_MANIFEST.csv` — every file emitted in this package.
   - `Loaded_files__shape_summary_.csv` — quick shape overview.

6) **Sample pipeline config**
   - `daily_master.sample.json` — points at the JOINED master and maps core fields (revenue, spend, orders, attention, discounts, etc.).

---

## How to Read the Files
- **Dates**: ISO `YYYY-MM-DD` at daily grain. (They reflect the timezone provided by each source export; no TZ shifting applied.)
- **Currencies**: USD fields are numeric (symbols/commas stripped). 
- **Row counts**: Many aggregates include `__row_count` indicating # of records contributing to that day.
- **Prefixes**:
  - `sales_day__*` → Shopify/commerce daily sales table (net/gross, discounts, returns…)
  - `adspend_day__*` → Paid media spend
  - `orders__*` → Orders export aggregated by day
  - `engagement_day__*` → Social engagement metrics
  - `views_day__*` → Normalized daily view counts from the “organic + paid views” export

---

## How to Use in Your Pipeline
- Quick fix: point `configs/daily_master.json.data_path` to **MASTER_DAILY_JOINED.csv** and paste mappings from `daily_master.sample.json`.
- Minimal config example (JSON):
{
  "data_path": "/mnt/data/MASTER_DAILY_JOINED.csv",
  "date_column": "date",
  "mappings": {
    "revenue": "sales_day__net_sales",
    "total_sales": "sales_day__total_sales",
    "orders": "sales_day__orders",
    "ad_spend": "adspend_day__amount_spent_usd",
    "views": "views_day__primary",
    "reach": "engagement_day__reach",
    "likes": "engagement_day__likes",
    "comments": "engagement_day__comments",
    "shares": "engagement_day__shares",
    "follows": "engagement_day__follows",
    "saves": "engagement_day__saves",
    "discount_amount": "sales_day__discounts",
    "returns_amount": "sales_day__returns",
    "shipping_amount": "sales_day__shipping_charges",
    "duties_amount": "sales_day__duties",
    "tax_amount": "sales_day__taxes"
  },
  "derived": {
    "roas": "revenue / ad_spend",
    "aov": "revenue / orders"
  }
}

- CLI run:
```bash
python cli/marketing_analysis.py --config configs/daily_master.json --enable-ai
```

- To add **campaign/channel/creative/product/COGS** sections: export or compute those columns at daily grain and join them into the master; then add their names to the `mappings` block in the config.

---

## Notes
- Some TikTok files you provided are **not daily** (e.g., broken down by age/gender/ad only). They’re cleaned here for reference but aren’t merged into the daily master unless a date field is present.
- Any `Unnamed_*` columns came from unusual headers in the original sheets; kept in the full master for completeness.
