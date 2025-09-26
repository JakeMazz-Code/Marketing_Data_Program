# LLM_RESPONSES.md — Brief Flow

## Execution Path
1. `python manage.py brief` calls `lint_brief_schema()` and aborts if any object node is missing `type`, `properties`, `required`, or `additionalProperties:false`.
2. The router loads `series.jsonl`, `llm_payload.json`, `quality_report.json`, and `anomalies.json` to build a payload limited to the last 90–120 days.
3. OpenAI Responses API is invoked with:
   - `text.format` set to the strict JSON schema (`json_schema`, `strict: true`).
   - System prompt enforcing “No TAM/geo/subscription claims” and 20–40% uplift bounds.
   - `OPENAI_SERIES_DAYS`, `OPENAI_TOP_ANOMALIES`, and `OPENAI_MAX_OUTPUT_TOKENS` controlling payload size.
4. On failure the router retries with a smaller series window ladder (90 ? 60 ? 45 ? 30 ? 21 ? 14) and exponential backoff for 429s.
5. Verification runs through Anthropic if `ANTHROPIC_API_KEY` is available; otherwise the local clamp keeps each opportunity =40% of the 28-day revenue baseline. Set `BRIEF_SKIP_VERIFIER=1` or `--no-verify` to skip the External check in safe environments.
6. Outputs are written to `reports/daily_master/brief_draft.json`, `brief_verified.json`, and `brief_notes.md`.

## Rate Limit & Error Handling
| Error | Router behaviour | Operator fix |
| --- | --- | --- |
| `429` or `RateLimitError` | Waits `min(2^attempt, 30)` seconds, shrinks series window, retries. | Check key usage, reduce `OPENAI_SERIES_DAYS`, or upgrade plan. |
| `context_length_exceeded` | Shrinks payload (series + anomalies) and retries. | Lower `OPENAI_SERIES_DAYS` or use a higher-context model. |
| Non-JSON response | Attempts to extract fenced JSON; if parsing fails, retries with smaller payload. | Typically resolves automatically; otherwise inspect the system prompt. |

## Schema Rules (strict)
- Every object node sets `type: "object"`.
- `properties` enumerate only allowed keys.
- `required` lists every property key.
- `additionalProperties` is `false` at every object or nested `items` definition.
- Array `items` that emit objects also follow the same rules.

## Models & Defaults
- Draft: `OPENAI_MODEL_ANALYSIS` (default `gpt-5` via Responses API).
- Verifier: `ANTHROPIC_MODEL_VERIFIER` (default `claude-opus-4-1-20250805`).
- Token cap: `OPENAI_MAX_OUTPUT_TOKENS` (CLI `--tokens`).
- Series window: `OPENAI_SERIES_DAYS` (CLI `--days`).

