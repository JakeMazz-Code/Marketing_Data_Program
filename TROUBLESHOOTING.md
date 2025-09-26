# TROUBLESHOOTING.md — Common Failures

| Error | What it means | Fix |
| --- | --- | --- |
| `invalid_json_schema` | `python manage.py schema --check` found an object without strict `type/properties/required/additionalProperties:false`. | Edit `Marketing_analytics/ai.py::build_brief_schema()` so every object sets the strict fields, then rerun the check. |
| `429` or `RateLimitError` | OpenAI throttled the Responses call. | Wait for the backoff to finish, reduce `OPENAI_SERIES_DAYS`, or upgrade API capacity; rerun `python manage.py brief`. |
| `context_length_exceeded` | Payload was too large for the selected model. | Lower `OPENAI_SERIES_DAYS`, trim anomalies via `OPENAI_TOP_ANOMALIES`, or choose a model with a larger context window. |
| Missing `series.jsonl` / `quality_report.json` | ETL artifacts were not generated. | Run `python manage.py etl --quick`; inspect stdout for failures. |
| `[ETL] Pipeline failed: ...` | Daily master execution raised an error (bad config, missing columns, etc.). | Re-run with `python manage.py etl --full --debug` to see stage logs and fix the underlying data/config issue. |
| `Running scripts is disabled on this system` (Windows) | PowerShell execution policy blocks Streamlit/CLI. | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in an elevated PowerShell, then retry. |

