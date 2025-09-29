from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

import Marketing_analytics.ai as ai_module
from tests.md_only_test_utils import base_artifacts, build_series, patch_inputs, stub_settings


class _FakeResponses:
    def __init__(self, log: List[dict]) -> None:
        self.log = log

    def create(self, **kwargs):
        self.log.append(kwargs)
        template = kwargs["input"][1]["content"]
        output = template.replace("{{ADDITIONAL_INSIGHTS}}", "- Insight 1")
        return SimpleNamespace(output_text=output)


class _FakeOpenAI:
    def __init__(self, api_key: str, tracker: List["_FakeOpenAI"]) -> None:
        assert api_key == "sk-test"
        self._responses_calls: List[dict] = []
        self.responses = _FakeResponses(self._responses_calls)
        self.chat = SimpleNamespace(completions=None)
        tracker.append(self)


@pytest.fixture()
def md_only_context(tmp_path, monkeypatch):
    series_df = build_series()
    settings = stub_settings(tmp_path)
    artifacts = base_artifacts(series_df)
    patch_inputs(monkeypatch, settings, artifacts)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_ANALYSIS", "gpt-5")
    monkeypatch.delenv("OPENAI_MAX_OUTPUT_TOKENS", raising=False)

    created: List[_FakeOpenAI] = []

    def _factory(api_key: str) -> _FakeOpenAI:
        return _FakeOpenAI(api_key, created)

    monkeypatch.setattr(ai_module, "OpenAI", _factory)
    monkeypatch.setattr(ai_module.time, "sleep", lambda _seconds: None)

    return SimpleNamespace(tmp_path=tmp_path, clients=created, artifacts=artifacts, settings=settings)
