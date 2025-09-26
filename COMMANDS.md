# Commands

## manage.py verbs
| Command | What it does | Example |
| --- | --- | --- |
| `python manage.py schema --check` | Lints the brief JSON schema (fails fast on any missing `type`, `properties`, `required`, or `additionalProperties:false`). | `python manage.py schema --check` |
| `python manage.py etl [--quick|--full] [--debug]` | Runs the daily master ETL. `--quick` is the fast smoke (default), `--full` is an explicit full run, and `--debug` prints stage banners + artifact timestamps. | `python manage.py etl --full --debug` |
| `python manage.py brief [--model MODEL] [--days N] [--tokens N] [--verify|--no-verify]` | Generates the OpenAI Responses draft, runs the verifier (unless `--no-verify`), and writes `brief_draft.json`, `brief_verified.json`, `brief_notes.md`. | `python manage.py brief --model gpt-5 --days 90 --tokens 2500` |
| `python manage.py ui` | Launches `streamlit run dashboard.py`. | `python manage.py ui` |
| `python manage.py status` | Prints `status.json` stage/pct/timestamp if present. | `python manage.py status` |
| `python manage.py doctor` | Checks Python/package versions, env keys, and artifact directories. | `python manage.py doctor` |
| `python manage.py clean [--yes]` | Clears `reports/daily_master/*` (asks for confirmation unless `--yes`). | `python manage.py clean --yes` |
| `python manage.py test` | Runs `pytest -q`. | `python manage.py test` |
| `python manage.py models` | Lists `/v1/models` via OpenAI API and highlights `OPENAI_MODEL_ANALYSIS`. | `python manage.py models` |

## Environment knobs
| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for the Responses draft. | _unset_ |
| `OPENAI_MODEL_ANALYSIS` | Responses model used for the brief router. | `gpt-5` |
| `OPENAI_SERIES_DAYS` | Max trailing days sent to the LLM payload. | `90` (overridden by CLI `--days`) |
| `OPENAI_MAX_OUTPUT_TOKENS` | Caps the Responses reply tokens. | `3000` |
| `OPENAI_TOP_ANOMALIES` | Max anomaly runs included in payload. | `10` |
| `OPENAI_REASONING_EFFORT` | Optional reasoning hint for o3/gpt-5 classes. | _unset_ |
| `ANTHROPIC_API_KEY` | Enables Anthropics verifier. | _unset_ |
| `ANTHROPIC_MODEL_VERIFIER` | Anthropic model for verification. | `claude-opus-4-1-20250805` |
| `BRIEF_SKIP_VERIFIER` | When `1`, skips the Anthropic verifier and writes the draft as verified. | `0` |

## Exit codes
- `0` – success.
- `1` – failure (schema violations, pipeline/brief errors, missing config/api keys).
- `2` – bad CLI usage (e.g., using `--quick` and `--full` together).

## Artifact map (`reports/daily_master/`)
| Artifact | Description |
| --- | --- |
| `series.jsonl` | Daily KPI time series (date, revenue, ad_spend, orders, ratios, engagement). |
| `shape.json` | Dimensions + null counts for the joined dataset. |
| `data_quality.json` / `quality_report.md` | Quality gate summary with PASS/WARN/FAIL verdicts. |
| `anomalies.json` / `anomalies_notes.md` | STL residual anomalies and notes. |
| `llm_payload.json` | Aggregated windows (7/28/90 day sums + ratios). |
| `brief_draft.json` / `brief_verified.json` / `brief_notes.md` | Executive brief artifacts. |
| `trend.png` (optional) | External trend visual used on the dashboard. |
| `status.json` (optional) | ETL/brief progress for long-running jobs. |

