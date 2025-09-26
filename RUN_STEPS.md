# RUN_STEPS.md — Operator Playbook

## Daily Routine (˜10 minutes)
1. **Activate environment** – `.\.venv\Scripts\activate` (or `source .venv/bin/activate`).
2. **Smoke ETL** – `python manage.py etl --quick`. Confirm `series.jsonl`, `data_quality.json`, and `anomalies.json` refreshed.
3. **Generate brief** – `python manage.py brief --model gpt-5 --days 120 --tokens 3000`. Ensure `brief_verified.json` is updated.
4. **Launch dashboard** – `python manage.py ui` and skim Overview, Anomalies, Margin pages for red flags.
5. **Log outcomes** – Append key notes to `reports/daily_master/status.json` (if used) or your ops tracker.

## Weekly Deep-Dive (˜30 minutes)
1. Run `python manage.py etl --full --debug` to capture stage timings and artifact timestamps.
2. Review `quality_report.md` for any WARN/FAIL and address upstream data issues.
3. Open the dashboard and export screenshots of:
   - KPI tiles + revenue/MER trend.
   - Anomalies table.
   - Margin waterfall (if present).
4. Re-run `python manage.py brief --verify` and inspect `brief_notes.md` for verifier feedback.
5. File action items using the top three `actions` from `brief_verified.json`.

## Monthly Hygiene
- `python manage.py doctor` before major changes or key rotations.
- `python manage.py clean --yes` when archiving outputs (after exporting essentials).
- `python manage.py test` to keep the pytest suite green.
- Update `configs/daily_master.json` if new KPI columns or channels land.

