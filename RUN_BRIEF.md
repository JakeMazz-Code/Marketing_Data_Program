# Daily Master Brief Runbook

This runbook covers the Markdown-only executive brief flow. The MD path makes a single OpenAI call, writes `brief.md`
and `brief_raw.txt`, and never retries for JSON schema errors.

## 1. Environment variables

Set the following before running the command (PowerShell syntax shown; use `export` on Bash):

```powershell
$Env:OPENAI_API_KEY = "sk-your-key"
$Env:OPENAI_MODEL_ANALYSIS = "gpt-5"   # optional; defaults to gpt-5 with Responses route
$Env:DEFAULT_SEASONALITY = "1.20"      # optional; overrides hero SKU seasonality multiplier
$Env:DEFAULT_SAFETY_STOCK = "1.15"     # optional; overrides safety stock multiplier
```

`OPENAI_MODEL_ANALYSIS` is only needed if you want a specific model (e.g., `gpt-5.1-mini`). Leave the seasonality and
safety values unset to use the defaults baked into the report (1.20 and 1.15 respectively).

## 2. Generate the Markdown brief

Run the command from the repo root after the daily master ETL has written fresh artifacts:

```powershell
.\.venv312\Scripts\python.exe cli\marketing_analysis.py --config configs\daily_master.json --generate-brief --md-only
```

On Bash/macOS:

```bash
./.venv312/Scripts/python.exe cli/marketing_analysis.py --config configs/daily_master.json --generate-brief --md-only
```

The CLI prints three log lines of interest:

- `[Brief] missing:...` when sections fall back to "Not available in artifacts." (harmless)
- `[Brief] MD-only: model='gpt-5' route='responses' (reasoning=high)` indicating the OpenAI route used
- `[Brief] MD-only: wrote` with the absolute paths to `brief.md` and `brief_raw.txt`

## 3. Output files

Both files live under `reports/daily_master/`:

- `brief.md` – polished Markdown with the required seven sections in order, followed by **Additional Insights Uncovered (Model Reasoning)** and the legacy summary sections.
- `brief_raw.txt` – exact model output (useful for diffing or audits).

If optional artifacts such as efficiency or creatives are missing, the brief keeps the section headings and inserts
"Not available in artifacts.". The log lists those sections as `missing:`.

## 4. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `[Brief] Failed: OPENAI_API_KEY environment variable is required` | Key not exported | Set `OPENAI_API_KEY` before running the command. |
| Brief contains `{{ADDITIONAL_INSIGHTS}}` | Model response returned empty text | Re-run the command; the fixture replaces the placeholder when the call succeeds. |
| Command reports stale data | ETL artifacts outdated | Re-run the ETL (`python -m cli.marketing_analysis --config configs/daily_master.json --until write`) first. |

To use the original verified JSON flow, omit `--md-only` and ensure Anthropic credentials are available. The MD-only path
is recommended for daily operations because it avoids schema retries and writes human-readable Markdown instantly.
