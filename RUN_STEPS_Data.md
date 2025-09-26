
# RUN_STEPS_Data.md — Step-by-step with Data/ folder
**Generated:** 2025-09-25 02:00:48 UTC

1) Create folders (if not already): 
   - Data/master/, Data/inputs/, Data/docs/, Data/_archive/
   Move files per DATA_LAYOUT_AND_KEEP_GUIDE.md.

2) Save the updated config at `configs/daily_master.json` (see attached).

3) Codex PRs:
   - Paste **GLOBAL SESSION SYSTEM PROMPT**.
   - Paste **PR0**; verify `INTAKE_REPORT.md` reflects Data/ paths.
   - Paste **PR1**; run:
     ```bash
     python cli/marketing_analysis.py --config configs/daily_master.json
     ```
     Confirm artifacts in `reports/daily_master/`.
   - Paste **PR2 → PR10** sequentially, running `pytest -q` after each PR; run `streamlit run dashboard.py` at PR9.

4) Deliverables:
   - Streamlit screenshots/URL, `brief_verified.json`, and `quality_report.md`.
