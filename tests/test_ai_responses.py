from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from Marketing_analytics.ai import LLMRouter, RateLimitError


class _StubResponses:
    def __init__(self, queue: List[Any]) -> None:
        self._queue = queue
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError("No more stub responses available")
        result = self._queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _StubOpenAI:
    def __init__(self, queue: List[Any]) -> None:
        self.responses = _StubResponses(queue)


def _make_payload() -> Dict[str, Any]:
    return {
        "series": [
            {
                "date": "2025-09-20",
                "revenue": 1000.0,
                "ad_spend": 400.0,
                "orders": 40,
                "mer": 2.5,
                "roas": 2.5,
                "cac": 10.0,
                "aov": 25.0,
            }
        ],
        "anomaly_runs": [
            {"date": "2025-09-10", "metric": "revenue", "z": 3.2, "direction": "up"},
            {"date": "2025-09-11", "metric": "revenue", "z": 2.9, "direction": "up"},
        ],
    }


def _make_brief() -> Dict[str, Any]:
    return {
        "topline": {"period": "2025-09-15..2025-09-21", "trend": "accelerating", "driver": "spend"},
        "kpis": [
            {"metric": "Revenue", "current": 1000.0, "d7": 900.0, "d28": 850.0, "change_wow_pct": 0.05},
            {"metric": "MER", "current": 2.5, "d7": 2.3, "d28": 2.1, "change_wow_pct": 0.08},
        ],
        "opportunities": [
            {
                "title": "Scale prospecting",
                "why_now": "Efficiency steady; room to scale",
                "expected_lift_usd": {"pess": 15000.0, "base": 20000.0, "optim": 25000.0},
                "assumptions": ["Auction steady"],
                "confidence": "med",
            }
        ],
        "diagnostics": {},
        "actions": [
            {"owner": "Ops", "action": "Prepare reallocation"}
        ],
    }


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_SERIES_DAYS", raising=False)
    monkeypatch.delenv("OPENAI_TOP_ANOMALIES", raising=False)
    monkeypatch.delenv("OPENAI_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_MAX_RETRIES", raising=False)
    monkeypatch.delenv("OPENAI_USE_RESPONSES", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_ANALYSIS", "gpt-5")


def test_responses_happy_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    brief = _make_brief()
    response = type("Resp", (), {"output_text": json.dumps(brief), "output": []})()
    stub_client_holder: Dict[str, _StubOpenAI] = {}

    def fake_openai(api_key: str) -> _StubOpenAI:
        client = _StubOpenAI([response])
        stub_client_holder["client"] = client
        return client

    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    monkeypatch.setattr("Marketing_analytics.ai.OpenAI", fake_openai)

    router = LLMRouter({})
    result = router.draft_brief(_make_payload())

    captured = capsys.readouterr().out
    assert "route=responses" in captured
    assert result["topline"]["trend"] == "accelerating"
    assert stub_client_holder["client"].responses.calls


def test_responses_text_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    brief = _make_brief()
    item = type(
        "Item",
        (),
        {
            "content": [
                {"text": json.dumps(brief)}
            ]
        },
    )
    response = type("Resp", (), {"output_text": "", "output": [item]})()

    def fake_openai(api_key: str) -> _StubOpenAI:
        return _StubOpenAI([response])

    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    monkeypatch.setattr("Marketing_analytics.ai.OpenAI", fake_openai)

    router = LLMRouter({})
    result = router.draft_brief(_make_payload())
    assert result["opportunities"][0]["expected_lift_usd"]["base"] == 20000.0


def test_responses_rate_limit_retry(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    brief = _make_brief()
    success = type("Resp", (), {"output_text": json.dumps(brief), "output": []})()

    def _make_rate_limit(retry_after: str | None) -> RateLimitError:
        headers = {"Retry-After": retry_after} if retry_after else {}
        response = type("_Response", (), {"headers": headers, "request": None, "status_code": 429})()
        return RateLimitError("Rate limited", response=response, body=None)

    sequence: List[Any] = [_make_rate_limit("4"), _make_rate_limit(None), success]

    def fake_openai(api_key: str) -> _StubOpenAI:
        return _StubOpenAI(sequence)

    sleeps: List[float] = []

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setenv("OPENAI_MAX_RETRIES", "5")
    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    monkeypatch.setattr("Marketing_analytics.ai.OpenAI", fake_openai)
    monkeypatch.setattr("Marketing_analytics.ai.time.sleep", fake_sleep)

    router = LLMRouter({})
    result = router.draft_brief(_make_payload())

    captured = capsys.readouterr().out
    assert "429; retrying in" in captured
    assert len(sleeps) == 2
    assert pytest.approx(sleeps[0]) == 4.0
    assert sleeps[1] == pytest.approx(8.0)
    assert result["topline"]["period"] == "2025-09-15..2025-09-21"
