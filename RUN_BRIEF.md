# Daily Master Brief Runbook

This guide walks through generating the executive brief artifacts after the daily master pipeline finishes. The steps cover both Windows PowerShell and Bash shells.

## 1. Required Environment Variables

Set the following environment variables before running the brief command (never commit keys to source control):

- `OPENAI_API_KEY`
- `OPENAI_MODEL_ANALYSIS`
- `OPENAI_SERIES_DAYS` (optional, defaults to 90)
- `OPENAI_TOP_ANOMALIES` (optional, defaults to 10)
- `OPENAI_MAX_OUTPUT_TOKENS` (optional)
- `OPENAI_REASONING_EFFORT` (only for o3/gpt-5 reasoning routes)
- `ANTHROPIC_API_KEY` (optional; enables external verification)
- `ANTHROPIC_MODEL_VERIFIER` (optional; defaults to a Claude Opus variant)

### PowerShell

```powershell
$Env:OPENAI_API_KEY = "sk-your-key"
$Env:OPENAI_MODEL_ANALYSIS = "gpt-4-0613"   # or another available model
$Env:OPENAI_SERIES_DAYS = "90"
$Env:OPENAI_TOP_ANOMALIES = "10"
$Env:ANTHROPIC_API_KEY = ""  # omit or remove if not available
```

### Bash (macOS/Linux)

```bash
export OPENAI_API_KEY="sk-your-key"
export OPENAI_MODEL_ANALYSIS="gpt-4-0613"
export OPENAI_SERIES_DAYS=90
export OPENAI_TOP_ANOMALIES=10
export ANTHROPIC_API_KEY=""   # leave unset if you do not have a key
```

> ⚠️ Rotate any API key that was ever printed or pasted into a shared console.

## 2. Choosing a Model

1. List the available models in your account. For the official OpenAI CLI this looks like:
   ```bash
   openai models list
   ```
2. Pick a Chat Completions model (`gpt-4`, `gpt-4-0613`, `gpt-3.5-turbo`) **or** a Responses family model (`gpt-4o`, `gpt-4o-mini`, `o3-*`, `gpt-5-*`, `omni-*`).
3. Set `OPENAI_MODEL_ANALYSIS` to that value. The router will automatically pick the correct API path and strip unsupported parameters.

## 3. Generate Artifacts and Brief

1. Produce the daily master artifacts (run from the repo root):
   ```bash
   python -m cli.marketing_analysis --config configs/daily_master.json
   ```
2. Generate the brief after artifacts exist:
   ```bash
   python -m cli.marketing_analysis --config configs/daily_master.json --generate-brief
   ```
   The CLI prints the model route (Chat vs Responses) and the series window used for each attempt.
3. The following files appear under `reports/daily_master/`:
   - `brief_draft.json`
   - `brief_verified.json`
   - `brief_notes.md`

## 4. Reading `brief_notes.md`

Open `reports/daily_master/brief_notes.md` and check:

- `Status: APPROVED` or `Status: REDLINE`
- Listed issues (e.g., lifts clamped over the 40% baseline)
- The “Verifier notes” section indicating whether Anthropic approved the draft or a local clamp fallback was used.

## 5. Rotating Leaked Keys

If an API key was ever echoed to the console or written to a log, rotate it immediately via the provider dashboard and update the environment variable with the new key.

## 6. Tuning Anomaly Noise

If anomaly notes dominate the brief:

1. Review `reports/daily_master/anomalies.json`.
2. Adjust the anomaly sensitivity in your daily master config (e.g., increase the z-score threshold or reduce lookback windows).
3. Re-run the artifacts and brief commands.

## 7. Updating the Model Route

When your account gains new models (e.g., o3/gpt-5 access):

1. Set `OPENAI_MODEL_ANALYSIS` to the new model name.
2. Optionally set `OPENAI_REASONING_EFFORT` (e.g., `medium`) for reasoning models.
3. Re-run the brief command; the router automatically switches to the Responses API with the correct payload format.

Keep this runbook close to your operations documentation so analysts can self-serve the brief generation process.
