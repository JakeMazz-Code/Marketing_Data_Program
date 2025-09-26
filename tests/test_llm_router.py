import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from Marketing_analytics.ai import LLMRouter


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL_ANALYSIS",
        "OPENAI_SERIES_DAYS",
        "OPENAI_TOP_ANOMALIES",
        "OPENAI_MAX_OUTPUT_TOKENS",
        "OPENAI_MAX_RETRIES",
        "OPENAI_USE_RESPONSES",
    ]:
        monkeypatch.delenv(key, raising=False)


def _payload(days: int = 5) -> Dict[str, Any]:
    series = []
    for idx in range(days):
        series.append(
            {
                "date": f"2025-01-{idx + 1:02d}",
                "revenue": 1000.0 + idx,
                "ad_spend": 200.0,
                "orders": 10 + idx,
                "mer": 5.0,
                "roas": 5.0,
                "cac": 20.0,
                "aov": 100.0,
            }
        )
    anomalies = [
        {"peak_z": 4.0, "metric": "revenue"},
        {"peak_z": 3.0, "metric": "mer"},
    ]
    return {
        "series": series,
        "anomaly_runs": anomalies,
        "quality": {"status": "PASS", "caveats": []},
        "baselines": {},
        "llm_windows": {},
    }


def test_responses_invoked_with_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_ANALYSIS", "gpt-4o-mini")

    captured: Dict[str, Any] = {}

    class FakeResponses:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            content = json.dumps(
                {
                    "topline": {"period": "2025-01", "trend": "accelerating", "driver": "spend"},
                    "kpis": [],
                    "opportunities": [],
                    "diagnostics": {},
                    "actions": [],
                }
            )
            return SimpleNamespace(output_text=content, output=[])

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    monkeypatch.setattr("Marketing_analytics.ai.OpenAI", FakeOpenAI)

    router = LLMRouter({})
    result = router.draft_brief(_payload())

    assert result["topline"]["driver"] == "spend"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["store"] is False
    fmt = captured["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "Brief"


def test_context_shrinks_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_ANALYSIS", "gpt-5")
    monkeypatch.setenv("OPENAI_SERIES_DAYS", "90")

    calls: List[int] = []

    class FakeResponses:
        def __init__(self) -> None:
            self.invocations = 0

        def create(self, **kwargs: Any) -> Any:
            self.invocations += 1
            calls.append(kwargs["input"][1])
            if self.invocations == 1:
                raise RuntimeError("context_length_exceeded: token limit")
            content = json.dumps(
                {
                    "topline": {"period": "2025", "trend": "flat", "driver": "mix"},
                    "kpis": [],
                    "opportunities": [],
                    "diagnostics": {},
                    "actions": [],
                }
            )
            return SimpleNamespace(output_text=content, output=[])

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    monkeypatch.setattr("Marketing_analytics.ai.OpenAI", FakeOpenAI)

    router = LLMRouter({})
    long_payload = _payload(days=120)
    result = router.draft_brief(long_payload)

    assert result["topline"]["trend"] == "flat"
    assert router._last_series_window == 60
    assert len(calls) == 2


def test_verify_local_clamp_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    artifacts = {"llm_payload": {"windows": {"28d": {"revenue_sum": 100000.0}}}}
    router = LLMRouter(artifacts)

    payload = {"anomaly_runs": [], "quality": {"status": "PASS"}}
    draft = {
        "opportunities": [
            {
                "title": "Massive scale",
                "why_now": "Room to scale hero campaign",
                "expected_lift_usd": {"pess": 40000.0, "base": 80000.0, "optim": 100000.0},
                "assumptions": [],
                "confidence": "low",
            }
        ],
        "diagnostics": {},
    }

    verification = router.verify_brief(payload, draft)

    verified_opp = verification["brief_verified"]["opportunities"][0]
    assert verified_opp["expected_lift_usd"]["base"] <= 40000.0
    assert any("local 40%" in note.lower() for note in verification.get("notes", []))
