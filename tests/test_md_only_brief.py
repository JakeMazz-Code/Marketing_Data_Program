from types import SimpleNamespace

import pandas as pd

import Marketing_analytics.ai as ai_module


def _stub_settings(tmp_path):
    mappings = SimpleNamespace(
        revenue="revenue",
        total_sales="total_sales",
        orders="orders",
        ad_spend="ad_spend",
        views="views",
        engagement_reach=None,
        engagement_likes=None,
        engagement_comments=None,
        engagement_shares=None,
        engagement_follows=None,
        engagement_saves=None,
    )
    derived = SimpleNamespace(mer="mer", roas="roas", aov="aov")
    return SimpleNamespace(artifacts_dir=tmp_path, mappings=mappings, derived=derived)


def _stub_artifacts():
    series_df = pd.DataFrame(
        [
            {
                "date": "2024-08-01",
                "revenue": 100.0,
                "total_sales": 120.0,
                "orders": 4,
                "ad_spend": 40.0,
                "views": 800,
                "mer": 2.5,
                "roas": 2.5,
                "aov": 25.0,
            },
            {
                "date": "2024-08-02",
                "revenue": 110.0,
                "total_sales": 130.0,
                "orders": 5,
                "ad_spend": 44.0,
                "views": 900,
                "mer": 2.5,
                "roas": 2.5,
                "aov": 22.0,
            },
        ]
    )
    return {
        "series": series_df,
        "llm_payload": {"windows": {"28d": {"revenue_sum": 3100.0}}},
        "quality_report": {"status": "PASS", "rules": []},
        "anomalies": [],
    }


def test_generate_brief_md_writes_brief(tmp_path, monkeypatch):
    settings = _stub_settings(tmp_path)
    artifacts = _stub_artifacts()
    monkeypatch.setattr(ai_module, "load_daily_master_settings", lambda _: settings)
    monkeypatch.setattr(ai_module, "_load_artifacts", lambda _: artifacts)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL_ANALYSIS", "gpt-5")

    calls = []

    class FakeChat:
        def __init__(self) -> None:
            self.count = 0

        def create(self, **kwargs):
            self.count += 1
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="## Brief\n- Hello"))]
            )

    fake_chat = FakeChat()

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            assert api_key == "sk-test"
            self.chat = SimpleNamespace(completions=fake_chat)

            def _raise(*_args, **_kwargs):
                raise AssertionError("responses.create should not be called for MD-only")

            self.responses = SimpleNamespace(create=_raise)

    monkeypatch.setattr(ai_module, "OpenAI", FakeOpenAI)

    result = ai_module.generate_brief_md("fake-config.json")

    brief_md = (tmp_path / "brief.md").read_text(encoding="utf-8")
    brief_raw = (tmp_path / "brief_raw.txt").read_text(encoding="utf-8")

    assert brief_md == "## Brief\n- Hello"
    assert brief_raw == "## Brief\n- Hello"
    assert result["brief_md"].endswith("brief.md")
    assert fake_chat.count == 1
    assert calls[0]["model"] == "gpt-5"
    assert calls[0]["messages"][0]["role"] == "system"


def test_generate_brief_md_handles_rate_limit_retry(tmp_path, monkeypatch):
    settings = _stub_settings(tmp_path)
    artifacts = _stub_artifacts()
    monkeypatch.setattr(ai_module, "load_daily_master_settings", lambda _: settings)
    monkeypatch.setattr(ai_module, "_load_artifacts", lambda _: artifacts)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL_ANALYSIS", raising=False)

    slept = []
    monkeypatch.setattr(ai_module.time, "sleep", lambda seconds: slept.append(seconds))

    class FakeRateLimitError(Exception):
        pass

    monkeypatch.setattr(ai_module, "RateLimitError", FakeRateLimitError)

    class FlakyChat:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError("rate limited")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="### Recovered\n- Second try"))]
            )

    flaky_chat = FlakyChat()

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            assert api_key == "sk-test"
            self.chat = SimpleNamespace(completions=flaky_chat)
            self.responses = SimpleNamespace(create=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(ai_module, "OpenAI", FakeOpenAI)

    result = ai_module.generate_brief_md("fake-config.json")

    text = (tmp_path / "brief.md").read_text(encoding="utf-8")
    assert "Second try" in text
    assert flaky_chat.calls == 2
    assert slept == [8]
    assert result["brief_raw"].endswith("brief_raw.txt")
