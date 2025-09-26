"""AI-assisted summarisation utilities and LLM brief router."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore[import]
    from openai import RateLimitError  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[misc]
    class RateLimitError(Exception):
        pass

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None  # type: ignore[misc]

from Marketing_analytics.config import AISummaryConfig
from Marketing_analytics.daily_master import load_daily_master_settings


class AISummarizer:
    """Generate narrative insights using a configured language model."""

    def __init__(self, config: AISummaryConfig) -> None:
        self.config = config

    def _get_api_key(self) -> str:
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"API key not found in environment variable '{self.config.api_key_env}'."
            )
        return api_key

    def _frame_snapshot(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(no data)"
        trimmed = frame.head(self.config.max_table_rows)
        try:
            return trimmed.to_markdown(index=False)
        except Exception:  # pragma: no cover - fallback if tabulate missing
            return trimmed.to_string(index=False)

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:,.4f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _render_prompt(self, context: Dict[str, object]) -> str:
        return self.config.prompt_template.format(
            overall_metrics=context.get("overall_metrics", "(no metrics)"),
            campaign_table=context.get("campaign_table", "(no data)"),
            channel_table=context.get("channel_table", "(no data)"),
            segment_table=context.get("segment_table", "(no data)"),
            product_table=context.get("product_table", "(no data)"),
            creative_table=context.get("creative_table", "(no data)"),
            margin_table=context.get("margin_table", "(no data)"),
            timeline_table=context.get("timeline_table", "(no data)"),
            customer_value_table=context.get("customer_value_table", "(no data)"),
        )

    def _call_openai(self, prompt: str) -> str:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Install `openai` to use this provider.")
        client = OpenAI(api_key=self._get_api_key())
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
        )
        message = response.choices[0].message
        if isinstance(message.content, str):
            return message.content
        return "".join(
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
            for block in message.content  # type: ignore[union-attr]
        )

    def _call_anthropic(self, prompt: str) -> str:
        if anthropic is None:
            raise RuntimeError(
                "anthropic package is not installed. Install `anthropic` to use this provider."
            )
        client = anthropic.Anthropic(api_key=self._get_api_key())
        response = client.messages.create(
            model=self.config.model,
            system=self.config.system_prompt,
            max_completion_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_blocks).strip()

    def generate_summary(
        self,
        *,
        overall: Dict[str, object],
        campaign: pd.DataFrame,
        channel: pd.DataFrame,
        segment: pd.DataFrame,
        product: pd.DataFrame,
        creative: pd.DataFrame,
        margin: pd.DataFrame,
        timeline: pd.DataFrame,
        customer_value: pd.DataFrame,
    ) -> Optional[Dict[str, object]]:
        if not self.config.is_enabled():
            return None

        context = {
            "overall_metrics": "\n".join(
                f"- {key.replace('_', ' ').title()}: {self._format_value(value)}"
                for key, value in overall.items()
            )
            if overall
            else "(no metrics)",
            "campaign_table": self._frame_snapshot(campaign),
            "channel_table": self._frame_snapshot(channel),
            "segment_table": self._frame_snapshot(segment),
            "product_table": self._frame_snapshot(product),
            "creative_table": self._frame_snapshot(creative),
            "margin_table": self._frame_snapshot(margin),
            "timeline_table": self._frame_snapshot(timeline),
            "customer_value_table": self._frame_snapshot(customer_value),
        }
        prompt = self._render_prompt(context)

        try:
            if self.config.provider.lower() == "anthropic":
                summary_text = self._call_anthropic(prompt)
            else:
                summary_text = self._call_openai(prompt)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"AI summary generation failed: {exc}")
            return None

        summary_text = summary_text.strip()
        if not summary_text:
            return None

        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "prompt": prompt,
            "markdown": summary_text,
        }


def _load_artifacts(artifacts_dir: str | Path) -> Dict[str, Any]:
    """Load JSON-based artifacts needed for LLM briefing."""

    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_path}")

    result: Dict[str, Any] = {}

    # Mandatory JSONs
    llm_payload_path = artifacts_path / "llm_payload.json"
    quality_report_path = artifacts_path / "quality_report.json"
    if not llm_payload_path.exists():
        raise FileNotFoundError("llm_payload.json is required for LLM routing.")
    if not quality_report_path.exists():
        raise FileNotFoundError("quality_report.json is required for LLM routing.")

    result["llm_payload"] = json.loads(llm_payload_path.read_text(encoding="utf-8"))
    result["quality_report"] = json.loads(quality_report_path.read_text(encoding="utf-8"))

    # Optional JSON artifacts
    optional_json_files = {
        "anomalies": "anomalies.json",
        "efficiency": "efficiency.json",
        "margin": "margin.json",
        "creatives": "creatives.json",
    }
    for key, filename in optional_json_files.items():
        path = artifacts_path / filename
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result[key] = None
        else:
            result[key] = None

    # Series data for metric context (JSON lines)
    series_path = artifacts_path / "series.jsonl"
    if series_path.exists():
        result["series"] = pd.read_json(series_path, lines=True)
    else:
        result["series"] = pd.DataFrame()

    return result



def build_brief_schema() -> Dict[str, Any]:
    opportunity_lift = {
        "type": "object",
        "properties": {
            "pess": {"type": "number"},
            "base": {"type": "number"},
            "optim": {"type": "number"},
        },
        "required": ["pess", "base", "optim"],
        "additionalProperties": False,
    }

    anomaly_item = {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "count": {"type": "integer"},
            "peak_z": {"type": "number"},
            "direction_at_peak": {"type": "string"},
            "spend_change_pct": {"type": ["number", "null"]},
            "engagement_change_pct": {"type": ["number", "null"]},
            "note_at_peak": {"type": ["string", "null"]},
        },
        "required": [
            "metric",
            "start_date",
            "end_date",
            "count",
            "peak_z",
            "direction_at_peak",
            "spend_change_pct",
            "engagement_change_pct",
            "note_at_peak"
        ],
        "additionalProperties": False,
    }

    generic_diag_item = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "value": {"type": ["number", "null"]},
            "note": {"type": ["string", "null"]},
        },
        "required": ["label", "value", "note"],
        "additionalProperties": False,
    }

    action_item = {
        "type": "object",
        "properties": {
            "owner": {"type": "string"},
            "action": {"type": "string"},
        },
        "required": ["owner", "action"],
        "additionalProperties": False,
    }

    schema = {
        "type": "object",
        "properties": {
            "topline": {
                "type": "object",
                "properties": {
                    "period": {"type": "string"},
                    "trend": {"enum": ["accelerating", "slowing", "flat"]},
                    "driver": {"enum": ["spend", "mix", "conversion", "unclear"]},
                },
                "required": ["period", "trend", "driver"],
                "additionalProperties": False,
            },
            "kpis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "current": {"type": "number"},
                        "d7": {"type": "number"},
                        "d28": {"type": "number"},
                        "change_wow_pct": {"type": "number"},
                    },
                    "required": ["metric", "current", "d7", "d28", "change_wow_pct"],
                    "additionalProperties": False,
                },
            },
            "opportunities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "why_now": {"type": "string"},
                        "expected_lift_usd": opportunity_lift,
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"enum": ["low", "med", "high"]},
                    },
                    "required": ["title", "why_now", "expected_lift_usd", "assumptions", "confidence"],
                    "additionalProperties": False,
                },
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    "anomalies": {"type": "array", "items": anomaly_item},
                    "efficiency": {"type": "array", "items": generic_diag_item},
                    "creatives": {"type": "array", "items": generic_diag_item},
                    "margin": {"type": "array", "items": generic_diag_item},
                },
                "required": ["anomalies", "efficiency", "creatives", "margin"],
                "additionalProperties": False,
            },
            "actions": {
                "type": "array",
                "items": action_item,
            },
        },
        "required": ["topline", "kpis", "opportunities", "diagnostics", "actions"],
        "additionalProperties": False,
    }

    return schema

BRIEF_JSON_SCHEMA: Dict[str, Any] = build_brief_schema()


def lint_brief_schema(schema: Dict[str, Any]) -> List[str]:
    """Validate that the brief schema is strict at every object node."""
    errors: List[str] = []

    def _visit(node: Dict[str, Any], pointer: str) -> None:
        if not isinstance(node, dict):
            errors.append(f"{pointer}: schema node must be an object definition")
            return
        node_type = node.get("type")
        if node_type != "object":
            errors.append(f"{pointer}: type must be 'object'")
            return
        properties = node.get("properties")
        if not isinstance(properties, dict):
            errors.append(f"{pointer}: properties must be a dictionary")
            properties = {}
        required = node.get("required")
        if not isinstance(required, list):
            expected = sorted(properties.keys())
            errors.append(f"{pointer}: required must list all properties = {expected}")
            required_set = set()
        else:
            if not all(isinstance(item, str) for item in required):
                errors.append(f"{pointer}: required entries must be strings")
            required_set = set(required)
        prop_keys = list(properties.keys())
        if prop_keys and required_set != set(prop_keys):
            errors.append(f"{pointer}: required must list all properties = {prop_keys}")
        additional = node.get("additionalProperties")
        if additional is not False:
            errors.append(f"{pointer}: additionalProperties must be false")
        for key, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue
            prop_type = prop_schema.get("type")
            if prop_type == "object" or (prop_type is None and "properties" in prop_schema):
                _visit(prop_schema, f"{pointer}/{key}")
            if prop_type == "array":
                items = prop_schema.get("items")
                if isinstance(items, dict) and (items.get("type") == "object" or "properties" in items):
                    _visit(items, f"{pointer}/{key}/items")

    _visit(schema, "#")
    return errors



def _summarize_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compress consecutive anomaly days into runs and rank by |z|."""

    if df is None or df.empty:
        return []

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", "metric"])
    if working.empty:
        return []

    working = working.sort_values(["metric", "date"]).reset_index(drop=True)
    runs: List[Dict[str, Any]] = []

    def _compile_run(metric: str, rows: List[pd.Series]) -> Dict[str, Any]:
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        peak_idx = frame["z"].abs().idxmax()
        peak_row = frame.loc[peak_idx]
        start_date = frame["date"].min()
        end_date = frame["date"].max()

        def _to_float(value: Any) -> Optional[float]:
            try:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return None
                return float(value)
            except Exception:
                return None

        return {
            "metric": metric,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "count": int(frame.shape[0]),
            "peak_z": float(peak_row["z"]),
            "direction_at_peak": str(peak_row.get("direction") or ("up" if peak_row["z"] >= 0 else "down")),
            "spend_change_pct": _to_float(peak_row.get("spend_change_pct")),
            "engagement_change_pct": _to_float(peak_row.get("engagement_change_pct")),
            "note_at_peak": peak_row.get("note"),
        }

    for metric, group in working.groupby("metric"):
        group = group.sort_values("date")
        run_rows: List[pd.Series] = []
        previous_date: Optional[pd.Timestamp] = None
        for _, row in group.iterrows():
            current_date = row["date"]
            if not run_rows:
                run_rows.append(row)
            else:
                assert previous_date is not None
                if (current_date - previous_date).days == 1:
                    run_rows.append(row)
                else:
                    runs.append(_compile_run(metric, run_rows))
                    run_rows = [row]
            previous_date = current_date
        if run_rows:
            runs.append(_compile_run(metric, run_rows))

    runs.sort(key=lambda item: abs(item["peak_z"]), reverse=True)
    return runs


def _compute_baselines(series: pd.Series) -> Dict[str, Optional[float | str]]:
    """Compute trailing averages for the supplied series."""

    if series.empty:
        return {"d7_avg": None, "d28_avg": None, "last_date": None}

    ordered = series.sort_index()
    ordered = ordered.dropna()
    if ordered.empty:
        return {"d7_avg": None, "d28_avg": None, "last_date": None}

    last_date = ordered.index.max()
    d7 = ordered.tail(7).mean() if len(ordered) >= 1 else np.nan
    d28 = ordered.tail(28).mean() if len(ordered) >= 1 else np.nan

    def _clean(value: float) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)

    return {
        "d7_avg": _clean(d7),
        "d28_avg": _clean(d28),
        "last_date": last_date.strftime("%Y-%m-%d") if isinstance(last_date, pd.Timestamp) else str(last_date),
    }


class LLMRouter:
    """Route drafts through OpenAI then verify/clamp with Anthropic (or local rules)."""

    _REASONING_HINTS = ("o3", "gpt-5")
    _SERIES_KEYS = ["date", "revenue", "ad_spend", "orders", "mer", "roas", "cac", "aov"]

    def __init__(self, artifacts: Dict[str, Any]) -> None:
        self.artifacts = artifacts
        self.openai_model = os.getenv("OPENAI_MODEL_ANALYSIS", "gpt-5")
        self.openai_max_tokens = self._safe_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "3000"))
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL_VERIFIER", "claude-opus-4-1-20250805")
        self.anthropic_max_tokens = self._safe_int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "3000"))
        self.series_days_default = max(self._safe_int(os.getenv("OPENAI_SERIES_DAYS", "90")), 1)
        self.top_anomalies = max(self._safe_int(os.getenv("OPENAI_TOP_ANOMALIES", "10")), 1)
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip()
        self._last_series_window: Optional[int] = None
        self._route_cache: Optional[str] = None

    @staticmethod
    def _safe_int(value: Optional[str]) -> int:
        try:
            if value is None:
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _select_route(self, client: Any) -> str:
        flag = os.getenv("OPENAI_USE_RESPONSES", "").strip().lower()
        responses_attr = getattr(client, "responses", None)
        has_responses = responses_attr is not None and hasattr(responses_attr, "create")
        if flag in {"1", "true", "yes"} and has_responses:
            return "responses"
        return "chat"

    def _call_chat_api(
        self,
        client: Any,
        schema: Dict[str, Any],
        request_payload: Dict[str, Any],
        token_cap: Optional[int],
    ) -> Any:
        system_prompt = (
            "No-BS brief; only use provided KPIs; never invent TAM/geo/subscriptions; use absolute dates. "
            "Reply with valid JSON that matches the provided schema exactly."
        )
        user_payload = {
            "schema": schema,
            "payload": request_payload,
        }
        kwargs: Dict[str, Any] = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
            ],
        }
        if token_cap is not None:
            kwargs["max_completion_tokens"] = token_cap
        return client.chat.completions.create(**kwargs)

    # ----------------------------
    # OpenAI draft
    # ----------------------------

    def draft_brief(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        schema = self._brief_schema()
        violations = lint_brief_schema(schema)
        if violations:
            print("[Brief] Schema check failed – fix these first:")
            for violation in violations:
                print(f"- {violation}")
            raise RuntimeError("Brief schema failed strict validation; see log for details.")
        if OpenAI is None:
            raise RuntimeError("openai package not installed; cannot draft brief.")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required to draft the brief.")

        client = OpenAI(api_key=api_key)
        route = self._select_route(client)
        self._route_cache = route
        max_retries = max(1, self._safe_int(os.getenv("OPENAI_MAX_RETRIES", "6")))
        series_ladder = self._series_retry_ladder(payload.get("series", []))
        anomaly_ladder = self._anomaly_retry_ladder()
        last_error: Optional[Exception] = None
        token_cap = self.openai_max_tokens if self.openai_max_tokens > 0 else None
        token_cap_str = token_cap if token_cap is not None else "auto"

        for attempt in range(1, max_retries + 1):
            series_days = series_ladder[min(attempt - 1, len(series_ladder) - 1)]
            anomaly_limit = anomaly_ladder[min(attempt - 1, len(anomaly_ladder) - 1)]
            request_payload, used_days = self._shrink_payload(payload, series_days, anomaly_limit)
            self._last_series_window = used_days
            print(
                f"[Brief] Attempt {attempt}: model='{self.openai_model}' route={route} "
                f"series_window_days={used_days} tokens={token_cap_str}"
            )
            try:
                if route == "responses":
                    response = self._call_responses_api(client, schema, request_payload)
                else:
                    response = self._call_chat_api(client, schema, request_payload, token_cap)
            except RateLimitError as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                delay = self._rate_limit_delay(attempt, exc)
                print(f"[Brief] 429; retrying in {delay:.1f}s with smaller payload...")
                time.sleep(delay)
                continue
            except Exception as exc:  # pragma: no cover - network/SDK errors
                last_error = exc
                if self._is_context_error(exc) and attempt < max_retries:
                    print("[Brief] context_length_exceeded detected; reducing series window and retrying...")
                    continue
                raise self._friendly_error(exc)

            try:
                return self._parse_json_payload(response)
            except RuntimeError as parse_err:
                last_error = parse_err
                if attempt >= max_retries:
                    break
                print(f"[Brief] Non-JSON response from {route}; shrinking payload and retrying...")
                continue

        assert last_error is not None  # pragma: no cover - defensive
        raise self._friendly_error(last_error)


    # ----------------------------
    # OpenAI routing helpers
    # ----------------------------

    def route_name(self) -> str:
        return self._route_cache or "responses"

    def _series_retry_ladder(self, series: Iterable[Any]) -> List[int]:
        counts = [self.series_days_default, 60, 45, 30, 21, 14]
        series_list = list(series)
        total_available = len(series_list)
        ladder = []
        for count in counts:
            if count <= 0:
                continue
            ladder.append(min(count, total_available if total_available else count))
        if not ladder:
            ladder = [total_available or 14]
        # Remove duplicates while preserving order
        deduped: List[int] = []
        for value in ladder:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _anomaly_retry_ladder(self) -> List[int]:
        top = max(self.top_anomalies, 1)
        ladder: List[int] = [top]
        for candidate in (6, 4):
            candidate = max(candidate, 1)
            value = min(top, candidate)
            if value > 0 and value not in ladder:
                ladder.append(value)
        if ladder[-1] != 1:
            ladder.append(1)
        return ladder


    def _shrink_payload(self, payload: Dict[str, Any], window_days: int, anomaly_limit: int) -> Tuple[Dict[str, Any], int]:
        working = deepcopy(payload)
        series = working.get("series") or []
        if series:
            trimmed = self._trim_series(series, window_days)
            working["series"] = trimmed
            window_days = len(trimmed)
        anomalies = working.get("anomaly_runs") or []
        if anomalies and anomaly_limit:
            working["anomaly_runs"] = self._trim_anomalies(anomalies, anomaly_limit)
        working["series_window_days"] = window_days
        return working, window_days


    def _trim_series(self, series: Iterable[Dict[str, Any]], window_days: int) -> List[Dict[str, Any]]:
        records = list(series)
        if not records:
            return []
        try:
            records.sort(key=lambda item: item.get("date"))
        except Exception:
            pass
        trimmed = records[-window_days:]
        minimal: List[Dict[str, Any]] = []
        for row in trimmed:
            entry = {key: row.get(key) for key in self._SERIES_KEYS if row.get(key) is not None}
            if "date" in row:
                entry["date"] = row["date"]
            minimal.append(entry)
        return minimal

    def _trim_anomalies(self, anomalies: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        records = [dict(item) for item in anomalies if isinstance(item, dict)]
        try:
            records.sort(key=lambda item: abs(float(item.get("peak_z") or item.get("z") or 0.0)), reverse=True)
        except Exception:
            pass
        limit = max(int(limit or 0), 1)
        return records[:limit]

    # ----------------------------
    # OpenAI call implementations
    # ----------------------------

    def _brief_schema(self) -> Dict[str, Any]:
        return deepcopy(BRIEF_JSON_SCHEMA)


    def _call_responses_api(
        self,
        client: Any,
        schema: Dict[str, Any],
        request_payload: Dict[str, Any],
    ) -> Any:
        system_prompt = (
            "No-BS brief; only use provided KPIs; never invent TAM/geo/subscriptions; use absolute dates."
            " Cap opportunity lifts to 20-40% of the 28-day revenue baseline unless an anomaly run"
            " clearly indicates a structural change and you cite it explicitly."
        )
        input_items = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(request_payload, separators=(",", ":"))},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.openai_model,
            "input": input_items,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Brief",
                    "strict": True,
                    "schema": schema,
                }
            },
            "store": False,
        }
        if self.openai_max_tokens > 0:
            kwargs["max_output_tokens"] = self.openai_max_tokens
        if self.reasoning_effort and self._supports_reasoning():
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return client.responses.create(**kwargs)

    def _parse_json_payload(self, response: Any) -> Dict[str, Any]:
        for candidate in self._response_text_candidates(response):
            try:
                return self._ensure_json_object(candidate)
            except RuntimeError:
                continue
        raise RuntimeError("OpenAI response did not contain JSON content.")


    def _response_text_candidates(self, response: Any) -> List[Any]:
        candidates: List[Any] = []
        if response is None:
            return candidates
        if isinstance(response, (str, dict)):
            candidates.append(response)
            return candidates
        text_value = getattr(response, "output_text", None)
        if text_value:
            candidates.append(text_value)
        output = getattr(response, "output", None)
        if output:
            for item in output or []:
                if isinstance(item, dict):
                    content = item.get("content") or []
                else:
                    content = getattr(item, "content", []) or []
                for part in content:
                    if isinstance(part, dict):
                        text_part = part.get("text") or part.get("output_text")
                    else:
                        text_part = getattr(part, "text", None)
                    if text_part:
                        candidates.append(text_part)
        choices = getattr(response, "choices", None)
        if choices:
            for choice in choices or []:
                message = getattr(choice, "message", None)
                if message is None and isinstance(choice, dict):
                    message = choice.get("message")
                if not message:
                    continue
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    candidates.append(content)
                    continue
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text_part = part.get("text") or part.get("value") or part.get("content")
                        else:
                            text_part = getattr(part, "text", None)
                        if text_part:
                            candidates.append(text_part)
        return candidates


    def _rate_limit_delay(self, attempt: int, exc: Optional[Exception]) -> float:
        if exc is not None:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None) if response else None
            if isinstance(headers, dict):
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
                if retry_after is not None:
                    try:
                        return float(retry_after)
                    except (TypeError, ValueError):
                        pass
        return max(4.0 * attempt, 1.0)



    def _supports_reasoning(self) -> bool:
        model_lower = (self.openai_model or "").lower()
        return any(hint in model_lower for hint in self._REASONING_HINTS)

    # ----------------------------
    # JSON parsing utilities
    # ----------------------------

    def _ensure_json_object(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            raw = raw.strip()
        if not raw:
            raise RuntimeError("OpenAI response did not contain JSON content.")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
            candidate = self._extract_json_block(raw)
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    raise RuntimeError("Failed to parse JSON from OpenAI response.")
        raise RuntimeError("OpenAI response did not contain JSON content.")

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            body = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(body)
        depth = 0
        start = None
        for idx, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif char == "}":
                if depth:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start : idx + 1]
        return None

    # ----------------------------
    # Error handling
    # ----------------------------

    @staticmethod
    def _is_context_error(exc: Exception) -> bool:
        message = str(getattr(exc, "message", "") or str(exc)).lower()
        code = str(getattr(exc, "code", "") or "").lower()
        return "context_length_exceeded" in message or "context_length_exceeded" in code

    @staticmethod
    def _is_json_missing_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "did not contain json" in message or "failed to parse json" in message

    @staticmethod
    def _friendly_error(exc: Exception) -> RuntimeError:
        message = str(exc)
        lower = message.lower()
        if "response_format" in lower and "unexpected" in lower:
            message += " - Hint: remove response_format when calling Chat Completions or switch to the Responses route."
        if "expected input_text" in lower:
            message += " - Hint: Responses API requires content parts with type 'input_text'."
        if "temperature" in lower or "reasoning.effort" in lower:
            message += " - Hint: temperature/reasoning parameters are unsupported on Chat models."
        if "context_length_exceeded" in lower:
            message += " - Hint: payload too large; reduce OPENAI_SERIES_DAYS or OPENAI_TOP_ANOMALIES."
        return RuntimeError(message)

    # ----------------------------
    # Anthropic verification + clamps
    # ----------------------------

    def verify_brief(self, payload: Dict[str, Any], draft: Dict[str, Any]) -> Dict[str, Any]:
        verified = deepcopy(draft)
        issues: List[str] = []
        notes: List[str] = []

        baseline_total = self._revenue_28_total()
        clamp_limit = baseline_total * 0.40 if baseline_total is not None else None

        opportunities = verified.get("opportunities", []) or []
        for opportunity in opportunities:
            lifts = opportunity.get("expected_lift_usd") or {}
            for key in ("pess", "base", "optim"):
                value = lifts.get(key)
                if (
                    clamp_limit is not None
                    and isinstance(value, (int, float))
                    and value > clamp_limit + 1e-6
                ):
                    lifts[key] = float(clamp_limit)
                    issues.append(
                        f"Clamped {opportunity.get('title', 'opportunity')} {key} uplift to"
                        f" ${clamp_limit:,.2f} (40% of 28-day revenue baseline)."
                    )
            opportunity["expected_lift_usd"] = lifts
        verified["opportunities"] = opportunities

        diagnostics = verified.get("diagnostics") or {}
        diagnostics.setdefault("anomalies", payload.get("anomaly_runs", []))
        diagnostics.setdefault("efficiency", [])
        diagnostics.setdefault("creatives", [])
        diagnostics.setdefault("margin", [])
        verified["diagnostics"] = diagnostics

        feedback, verifier_note = self._anthropic_verifier_feedback(payload, draft, verified)
        if verifier_note:
            notes.append(verifier_note)
        anthropic_feedback = feedback
        if anthropic_feedback:
            issues.append(f"Anthropic feedback: {anthropic_feedback}")

        status = "REDLINE" if issues else "APPROVED"
        return {"status": status, "issues": issues, "brief_verified": verified, "notes": notes}

    def _revenue_28_total(self) -> Optional[float]:
        llm_payload = self.artifacts.get("llm_payload") or {}
        try:
            total = llm_payload["windows"]["28d"]["revenue_sum"]
        except Exception:  # pragma: no cover - defensive
            return None
        if total is None:
            return None
        try:
            return float(total)
        except Exception:
            return None

    def _anthropic_verifier_feedback(
        self,
        payload: Dict[str, Any],
        draft: Dict[str, Any],
        verified: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        if anthropic is None:
            return None, "Anthropic verifier SDK not installed; applied local 40% clamp rules."
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return None, "Anthropic API key not available; applied local 40% clamp rules."
        client = anthropic.Anthropic(api_key=api_key)
        instructions = (
            "Verify the executive brief for numerical sanity. Ensure opportunities respect the"
            " <=40% of 28-day revenue baseline rule unless justified. Highlight any remaining"
            " inconsistencies or data mismatches."
        )
        content = json.dumps({
            "payload": payload,
            "draft": draft,
            "verified": verified,
        })
        try:
            response = client.messages.create(
                model=self.anthropic_model,
                system=instructions,
                max_tokens=self.anthropic_max_tokens,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # pragma: no cover - network/availability issues
            return None, f"Anthropic verification unavailable ({exc.__class__.__name__}); applied local clamp."

        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        feedback = "\n".join(text_blocks).strip()
        return (feedback or None, None)


def _prepare_brief_payload(
    settings: Any,
    artifacts: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    series_df_obj = artifacts.get("series")
    series_df: pd.DataFrame = series_df_obj if isinstance(series_df_obj, pd.DataFrame) else pd.DataFrame()
    if series_df.empty:
        raise RuntimeError("series.jsonl is required to build the LLM payload.")

    mappings = settings.mappings
    derived = settings.derived

    series_df = series_df.copy()
    series_df["date"] = pd.to_datetime(series_df["date"], errors="coerce")
    series_df = series_df.dropna(subset=["date"]).sort_values("date")
    numeric_cols = [
        mappings.revenue,
        mappings.total_sales,
        mappings.orders,
        mappings.ad_spend,
        mappings.views,
        derived.mer,
        derived.roas,
        derived.aov,
        mappings.engagement_reach,
        mappings.engagement_likes,
        mappings.engagement_comments,
        mappings.engagement_shares,
        mappings.engagement_follows,
        mappings.engagement_saves,
    ]
    for col in numeric_cols:
        if col and col in series_df.columns:
            series_df[col] = pd.to_numeric(series_df[col], errors="coerce")

    if mappings.orders in series_df.columns and mappings.ad_spend in series_df.columns:
        orders = series_df[mappings.orders].astype(float)
        spend = series_df[mappings.ad_spend].astype(float)
        series_df["__cac"] = np.where(orders > 0, spend / orders, np.nan)
    else:
        series_df["__cac"] = np.nan

    if len(series_df) > 120:
        series_trimmed = series_df.tail(120)
    else:
        series_trimmed = series_df

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, float) and np.isnan(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    series_records: List[Dict[str, Any]] = []
    for _, row in series_trimmed.iterrows():
        record: Dict[str, Any] = {
            "date": row["date"].strftime("%Y-%m-%d"),
            "revenue": _as_float(row.get(mappings.revenue)),
            "ad_spend": _as_float(row.get(mappings.ad_spend)),
            "orders": _as_float(row.get(mappings.orders)),
            "views": _as_float(row.get(mappings.views)),
            "total_sales": _as_float(row.get(mappings.total_sales)),
            "mer": _as_float(row.get(derived.mer)),
            "roas": _as_float(row.get(derived.roas)),
            "aov": _as_float(row.get(derived.aov)),
            "cac": _as_float(row.get("__cac")),
        }
        optional_map = {
            "reach": mappings.engagement_reach,
            "likes": mappings.engagement_likes,
            "comments": mappings.engagement_comments,
            "shares": mappings.engagement_shares,
            "follows": mappings.engagement_follows,
            "saves": mappings.engagement_saves,
        }
        for key, col in optional_map.items():
            if col and col in series_trimmed.columns:
                record[key] = _as_float(row.get(col))
        series_records.append(record)

    series_df = series_df.set_index("date")

    baselines = {
        "revenue": _compute_baselines(pd.to_numeric(series_df[mappings.revenue], errors="coerce"))
        if mappings.revenue in series_df.columns
        else {"d7_avg": None, "d28_avg": None, "last_date": None},
        "mer": _compute_baselines(pd.to_numeric(series_df[derived.mer], errors="coerce"))
        if derived.mer in series_df.columns
        else {"d7_avg": None, "d28_avg": None, "last_date": None},
        "cac": _compute_baselines(pd.to_numeric(series_df["__cac"], errors="coerce")),
        "aov": _compute_baselines(pd.to_numeric(series_df[derived.aov], errors="coerce"))
        if derived.aov in series_df.columns
        else {"d7_avg": None, "d28_avg": None, "last_date": None},
    }

    quality_report = artifacts["quality_report"]
    caveats: List[str] = []
    for rule in quality_report.get("rules", []):
        if rule.get("status") in {"WARN", "FAIL"}:
            detail = rule.get("detail")
            if not detail:
                detail = rule.get("name")
            caveats.append(f"{rule.get('name', 'rule')}: {detail}")

    anomalies_raw = artifacts.get("anomalies")
    anomalies_df = pd.DataFrame(anomalies_raw) if isinstance(anomalies_raw, list) else pd.DataFrame()
    anomaly_runs = _summarize_anomalies(anomalies_df)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "series": series_records,
        "quality": {
            "status": quality_report.get("status", "UNKNOWN"),
            "caveats": caveats,
        },
        "anomaly_runs": anomaly_runs,
        "baselines": baselines,
        "llm_windows": artifacts["llm_payload"].get("windows", {}),
    }

    return payload, len(series_records)


def generate_verified_brief(config_path: str) -> Dict[str, str]:
    """Load artifacts, draft a brief with OpenAI, verify with Anthropic, and persist outputs."""

    settings = load_daily_master_settings(Path(config_path))

    md_only_env = os.getenv("OPENAI_MD_ONLY", "").strip().lower()
    if md_only_env in {"1", "true", "yes"}:
        return generate_brief_md(config_path)

    artifacts = _load_artifacts(settings.artifacts_dir)

    payload, _ = _prepare_brief_payload(settings, artifacts)

    router = LLMRouter(artifacts)
    draft = router.draft_brief(payload)
    if router._last_series_window:
        print(
            f"[Brief] Draft completed via {router.route_name()} route using {router.openai_model} "
            f"with {router._last_series_window} days of series data."
        )
    skip_verifier = os.getenv("BRIEF_SKIP_VERIFIER", "").strip().lower() in {"1", "true", "yes"}
    if skip_verifier:
        verification = {"status": "SKIPPED", "issues": ["Verifier skipped via BRIEF_SKIP_VERIFIER"], "brief_verified": draft, "notes": []}
    else:
        verification = router.verify_brief(payload, draft)

    artifacts_dir = settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    draft_path = artifacts_dir / "brief_draft.json"
    verified_path = artifacts_dir / "brief_verified.json"
    notes_path = artifacts_dir / "brief_notes.md"

    draft_path.write_text(json.dumps(draft, indent=2), encoding="utf-8")
    verified_path.write_text(json.dumps(verification["brief_verified"], indent=2), encoding="utf-8")

    status = verification.get("status", "UNKNOWN")
    notes_lines = [f"Status: {status}", ""]
    issues = [issue for issue in verification.get("issues", []) if issue]
    if issues:
        notes_lines.append("Issues:")
        notes_lines.extend(f"- {issue}" for issue in issues)
    else:
        notes_lines.append("Approved as-is")

    extra_notes = [note for note in verification.get("notes", []) if note]
    if extra_notes:
        notes_lines.append("")
        notes_lines.append("Verifier notes:")
        notes_lines.extend(f"- {note}" for note in extra_notes)

    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

    return {
        "draft": str(draft_path),
        "verified": str(verified_path),
        "notes": str(notes_path),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    status = getattr(exc, "status", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if response_status == 429:
            return True
    message = str(exc).lower()
    return "429" in message and "rate" in message



def generate_brief_md(config_path: str) -> Dict[str, str]:
    """Generate a Markdown-only executive brief using a single OpenAI call."""

    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install `openai` to use this provider.")

    settings = load_daily_master_settings(Path(config_path))
    artifacts = _load_artifacts(settings.artifacts_dir)

    payload, series_window_days = _prepare_brief_payload(settings, artifacts)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to draft the brief.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL_ANALYSIS", "").strip() or "gpt-5-chat-latest"

    max_tokens_env = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
    try:
        max_tokens = int(max_tokens_env) if max_tokens_env else 3000
    except ValueError:
        max_tokens = 3000
    if max_tokens <= 0:
        max_tokens = 3000

    system_prompt = (
        "You are a marketing analyst. Write a concise Markdown brief for executives with headings, bullets, "
        "and absolute dates. Do not include code fences."
    )

    serialized_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    brief_text = ""
    last_exc: Optional[Exception] = None
    backoff = [8, 12, 16]
    route_used = "chat"

    for attempt in range(len(backoff)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": serialized_payload},
                ],
                max_completion_tokens=max_tokens,
                )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("OpenAI returned no choices for the MD brief request.")
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            if message is None:
                raise RuntimeError("OpenAI response did not include a message content block.")
            content = getattr(message, "content", None)
            if isinstance(content, str):
                brief_text = content
            else:
                parts: List[str] = []
                for part in content or []:  # type: ignore[assignment]
                    if isinstance(part, dict):
                        text_part = part.get("text") or part.get("value") or part.get("content")
                    else:
                        text_part = getattr(part, "text", None)
                    if text_part:
                        parts.append(str(text_part))
                brief_text = "\n".join(parts)
            break
        except RateLimitError as exc:
            last_exc = exc
            if attempt == len(backoff) - 1:
                raise
            time.sleep(backoff[min(attempt, len(backoff) - 1)])
        except Exception as exc:  # pragma: no cover - network/runtime issues
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt != len(backoff) - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
                continue
            raise
    else:  # pragma: no cover - defensive
        if last_exc is not None:
            raise last_exc

    brief_text = (brief_text or "").strip()
    if not brief_text:
        if last_exc is not None:
            raise RuntimeError("OpenAI returned empty brief text.") from last_exc
        raise RuntimeError("OpenAI returned empty brief text.")

    artifacts_dir = settings.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    brief_md_path = artifacts_dir / "brief.md"
    brief_raw_path = artifacts_dir / "brief_raw.txt"
    brief_md_path.write_text(brief_text, encoding="utf-8")
    brief_raw_path.write_text(brief_text, encoding="utf-8")

    print(
        f"[Brief] MD-only: model='{model}' route={route_used} "
        f"series_window_days={series_window_days} tokens={max_tokens}"
    )

    return {
        "brief_md": str(brief_md_path),
        "brief_raw": str(brief_raw_path),
    }







