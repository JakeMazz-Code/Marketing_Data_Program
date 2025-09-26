# Data Quality Report
**Status:** PASS

## Summary
- Rows: 96
- Date range: 2025-06-21 -> 2025-09-24
- Freshness threshold (days): 7
- Missing dates: 0
- Duplicate dates: 0

## Rules
- [PASS] R1 Schema: All required columns present
- [PASS] R2 Types: Datetime index is daily, naive, and monotonic
- [PASS] R3 Freshness: Latest date 2025-09-24 within freshness threshold (7 days)
- [PASS] R4 Calendar: No missing dates between min and max
- [PASS] R5 Duplicates: No duplicate dates detected
- [PASS] R6 Range: Revenue, ad_spend, orders, and views are non-negative