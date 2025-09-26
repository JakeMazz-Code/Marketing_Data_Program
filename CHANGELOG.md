# CHANGELOG

## 2025-09-25 — Schema Gate ? Dashboard v2 ? manage.py
- Hardened `lint_brief_schema` messaging and wired the schema gate through `python manage.py schema --check` (now mandatory before briefs).
- Rebuilt `manage.py` with verbs for `etl`, `brief`, `ui`, `status`, `doctor`, `clean`, `test`, and `models`, plus debug stage logging and verifier skip support.
- Added a root `dashboard.py` entry point and rewrote `Marketing_analytics/dashboard.py` as a multi-page Plotly app (cached artifact reads, graceful fallbacks, verified-brief card).
- Refreshed operator docs (`README`, `COMMANDS`, `DASHBOARD`, `LLM_RESPONSES`, `RUN_STEPS`, `TROUBLESHOOTING`) and created this changelog.

