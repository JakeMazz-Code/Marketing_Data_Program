from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest



from Marketing_analytics.ai import LLMRouter, generate_verified_brief


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _write_series(path: Path) -> None:
    records = [
        {
            "date": "2025-09-19",
            "sales_day__net_sales": 9500.0,
            "sales_day__total_sales": 10200.0,
            "sales_day__orders": 190,
            "adspend_day__amount_spent_usd": 3200.0,
            "views_day__primary": 58000,
            "mer": 2.96875,
            "roas": 2.96875,
            "aov": 50.0,
        },
        {
            "date": "2025-09-20",
            "sales_day__net_sales": 10500.0,
            "sales_day__total_sales": 11200.0,
            "sales_day__orders": 205,
            "adspend_day__amount_spent_usd": 3400.0,
            "views_day__primary": 60000,
            "mer": 3.088235,
            "roas": 3.088235,
            "aov": 51.2195,
        },
        {
            "date": "2025-09-21",
            "sales_day__net_sales": 11000.0,
            "sales_day__total_sales": 11800.0,
            "sales_day__orders": 210,
            "adspend_day__amount_spent_usd": 3500.0,
            "views_day__primary": 62000,
            "mer": 3.142857,
            "roas": 3.142857,
            "aov": 52.3809,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_generate_verified_brief_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "Data" / "master"
    data_root.mkdir(parents=True)
    master_csv = data_root / "MASTER_DAILY_JOINED.csv"
    master_csv.write_text("date\n", encoding="utf-8")

    reports_dir = tmp_path / "reports" / "daily_master"
    reports_dir.mkdir(parents=True)

    # Mandatory artifacts
    llm_payload = {
        "windows": {
            "7d": {"revenue_sum": 70000.0, "ad_spend_sum": 21000.0},
            "28d": {"revenue_sum": 100000.0, "ad_spend_sum": 33000.0},
            "90d": {"revenue_sum": 300000.0, "ad_spend_sum": 99000.0},
        }
    }
    (reports_dir / "llm_payload.json").write_text(json.dumps(llm_payload, indent=2), encoding="utf-8")

    quality_report = {
        "status": "PASS",
        "rules": [
            {"name": "R3 Freshness", "status": "PASS", "detail": "Within threshold"},
            {"name": "R6 Range", "status": "WARN", "detail": "Spend includes weekend dip"},
        ],
    }
    (reports_dir / "quality_report.json").write_text(json.dumps(quality_report, indent=2), encoding="utf-8")

    anomalies = [
        {
            "date": "2025-09-10",
            "metric": "revenue",
            "z": 3.4,
            "direction": "up",
            "spend_change_pct": 0.22,
            "engagement_change_pct": 0.05,
            "note": "Revenue spike tied to flash sale",
        },
        {
            "date": "2025-09-11",
            "metric": "revenue",
            "z": 3.1,
            "direction": "up",
            "spend_change_pct": 0.18,
            "engagement_change_pct": 0.04,
            "note": "Carry-over from flash sale",
        },
        {
            "date": "2025-09-15",
            "metric": "mer",
            "z": -2.9,
            "direction": "down",
            "spend_change_pct": 0.40,
            "engagement_change_pct": -0.10,
            "note": "Spend spike with flat revenue",
        },
    ]
    (reports_dir / "anomalies.json").write_text(json.dumps(anomalies, indent=2), encoding="utf-8")

    _write_series(reports_dir / "series.jsonl")

    config = {
        "data_root": str(tmp_path / "Data"),
        "data_path": str(master_csv),
        "date_column": "date",
        "mappings": {
            "revenue": "sales_day__net_sales",
            "total_sales": "sales_day__total_sales",
            "orders": "sales_day__orders",
            "ad_spend": "adspend_day__amount_spent_usd",
            "views": "views_day__primary",
        },
        "derived": {"roas": "roas", "aov": "aov"},
        "artifacts_dir": str(reports_dir),
    }
    config_path = tmp_path / "daily_master.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def fake_draft(self: LLMRouter, payload: Dict[str, object]) -> Dict[str, object]:  # type: ignore[override]
        assert payload["series"]
        last_entry = payload["series"][-1]
        assert "mer" in last_entry
        return {
            "topline": {
                "period": "2025-09-15..2025-09-21",
                "trend": "accelerating",
                "driver": "spend",
            },
            "kpis": [
                {
                    "metric": "Revenue",
                    "current": last_entry["revenue"],
                    "d7": 9800.0,
                    "d28": 9600.0,
                    "change_wow_pct": 0.12,
                },
                {
                    "metric": "MER",
                    "current": last_entry["mer"],
                    "d7": 2.85,
                    "d28": 2.60,
                    "change_wow_pct": 0.08,
                },
            ],
            "opportunities": [
                {
                    "title": "Scale Prospecting 50%",
                    "why_now": "Spend efficiency steady; room to scale",
                    "expected_lift_usd": {"pess": 45000.0, "base": 60000.0, "optim": 75000.0},
                    "assumptions": ["Auction stays stable"],
                    "confidence": "med",
                }
            ],
            "diagnostics": {
                "anomalies": payload["anomaly_runs"],
                "efficiency": [],
                "creatives": [],
                "margin": [],
            },
            "actions": [
                {
                    "owner": "Ops",
                    "action": "Prep budget reallocation",
                    "impact": "high",
                    "difficulty": "med",
                    "due": "2025-10-01",
                }
            ],
        }

    monkeypatch.setattr(LLMRouter, "draft_brief", fake_draft)
    monkeypatch.setattr(LLMRouter, "_anthropic_verifier_feedback", lambda *args, **kwargs: (None, None))

    result = generate_verified_brief(str(config_path))

    draft_path = Path(result["draft"])
    verified_path = Path(result["verified"])
    notes_path = Path(result["notes"])

    assert draft_path.exists()
    assert verified_path.exists()
    assert notes_path.exists()

    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    assert {"topline", "kpis", "opportunities", "diagnostics", "actions"}.issubset(verified)

    opp = verified["opportunities"][0]
    assert opp["expected_lift_usd"]["base"] == pytest.approx(40000.0)
    assert len(verified["diagnostics"]["anomalies"]) > 0

    kpi_metrics = {entry["metric"] for entry in verified["kpis"]}
    assert "MER" in kpi_metrics

    notes_text = notes_path.read_text(encoding="utf-8")
    assert "REDLINE" in notes_text







