from types import SimpleNamespace

import pandas as pd

import Marketing_analytics.ai as ai_module


def _empty_router() -> ai_module.LLMRouter:
    artifacts = {
        "llm_payload": {},
        "quality_report": {"rules": []},
        "series": pd.DataFrame(),
    }
    return ai_module.LLMRouter(artifacts)


def test_router_defaults_to_chat_when_flag_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_USE_RESPONSES", raising=False)
    monkeypatch.setenv("OPENAI_MODEL_ANALYSIS", "gpt-5")
    router = _empty_router()
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda *_args, **_kwargs: None))
    assert router._select_route(client) == "chat"


def test_router_uses_responses_when_flag_enabled(monkeypatch):
    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    router = _empty_router()
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda *_args, **_kwargs: None))
    assert router._select_route(client) == "responses"
    monkeypatch.delenv("OPENAI_USE_RESPONSES", raising=False)


def test_router_falls_back_to_chat_without_responses_attr(monkeypatch):
    monkeypatch.setenv("OPENAI_USE_RESPONSES", "true")
    router = _empty_router()
    client = SimpleNamespace()
    assert router._select_route(client) == "chat"
    monkeypatch.delenv("OPENAI_USE_RESPONSES", raising=False)
