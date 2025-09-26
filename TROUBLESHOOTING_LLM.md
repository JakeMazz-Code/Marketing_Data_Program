# LLM Troubleshooting Guide

This document maps the most common brief-generation failures to quick fixes.

## Error Messages & Fixes

| Error Snippet | Likely Cause | Fix |
| --- | --- | --- |
| `unexpected keyword argument response_format` | Chat Completions model receiving Responses-only parameters. | Remove `response_format` or switch to a Responses-capable model (`gpt-4o`, `o3`, `gpt-5`, `omni`). The router now auto-detects, but double-check `OPENAI_MODEL_ANALYSIS`. |
| `Invalid value: text ... expected input_text` | Responses API content block sent with the wrong type. | Ensure each content block uses `{ "type": "input_text", "text": "..." }`. The router handles this automatically; upgrade if you see this. |
| `Unsupported parameter: temperature` or `reasoning.effort` | Chat model rejecting parameters meant for other routes. | Remove `temperature` / `reasoning.effort` when using Chat Completions. Set `OPENAI_MODEL_ANALYSIS` to a Responses model if you need reasoning controls. |
| `context_length_exceeded` | Payload too large for the selected model. | Reduce `OPENAI_SERIES_DAYS`, increase model context, or let the router retry (it now steps down 90 → 60 → 45 → 30 → 21 → 14 days automatically). |
| `OpenAI response did not contain JSON content.` | The model answered with plain text instead of structured JSON. | The router now forces tool calls or parses fallback JSON blocks. If it recurs, tighten the prompt or verify the model supports tool usage. |

## Payload Controls

- `OPENAI_SERIES_DAYS`: Maximum trailing days of daily metrics to send (default 90). Lower it if the model has a small context window.
- `OPENAI_TOP_ANOMALIES`: Maximum number of anomaly runs included (default 10).
- `OPENAI_REASONING_EFFORT`: Optional reasoning budget for o3/gpt-5-class models.

## Verification Path

- If `ANTHROPIC_API_KEY` is set and reachable, the draft is reviewed by Anthropic.
- If the key is missing or the call fails, local clamps ensure each opportunity’s base lift stays ≤ 40% of the 28-day revenue baseline, and `brief_notes.md` records that fallback.

Always rotate any API key that appears in terminal output.
