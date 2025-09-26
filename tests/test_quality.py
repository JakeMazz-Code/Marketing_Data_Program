from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from Marketing_analytics.quality import run_quality_checks, write_quality_artifacts


BASE_CFG: Dict[str, object] = {
    "date_column": "date",
    "mappings": {
        "revenue": "revenue",
        "total_sales": "total_sales",
        "orders": "orders",
        "ad_spend": "ad_spend",
        "views": "views",
    },
    "quality": {
        "freshness_days": 7,
        "allow_missing_dates": False,
        "stop_on_fail": True,
    },
}


def _make_df(dates, **cols) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    for key, values in cols.items():
        frame[key] = values
    return frame.set_index("date")


def test_fail_missing_required_column():
    today = pd.Timestamp.today().normalize()
    df = _make_df([today], revenue=[100])
    report = run_quality_checks(df, BASE_CFG)
    assert report.status == "FAIL"
    assert any(rule.status == "FAIL" for rule in report.rules if rule.name == "R1 Schema")


def test_fail_stale_freshness():
    df = _make_df([
        "2024-01-01",
        "2024-01-02",
    ],
        revenue=[100, 110],
        total_sales=[100, 110],
        orders=[1, 1],
        ad_spend=[10, 10],
        views=[1000, 1000],
    )
    report = run_quality_checks(df, BASE_CFG)
    assert report.status == "FAIL"
    assert any(rule.status == "FAIL" for rule in report.rules if rule.name == "R3 Freshness")


def test_fail_duplicate_dates():
    today = pd.Timestamp.today().normalize()
    df = _make_df([
        today,
        today,
    ],
        revenue=[100, 110],
        total_sales=[100, 110],
        orders=[1, 1],
        ad_spend=[10, 10],
        views=[1000, 1000],
    )
    report = run_quality_checks(df, BASE_CFG)
    assert report.status == "FAIL"
    assert any(rule.status == "FAIL" for rule in report.rules if rule.name == "R5 Duplicates")


def test_fail_negative_values():
    today = pd.Timestamp.today().normalize()
    df = _make_df([
        today,
        today + pd.Timedelta(days=1),
    ],
        revenue=[100, 110],
        total_sales=[100, 110],
        orders=[1, 1],
        ad_spend=[10, -5],
        views=[1000, 800],
    )
    report = run_quality_checks(df, BASE_CFG)
    assert report.status == "FAIL"
    assert any(rule.status == "FAIL" for rule in report.rules if rule.name == "R6 Range")


def test_warn_missing_dates_allowed():
    cfg = {
        **BASE_CFG,
        "quality": {
            **BASE_CFG["quality"],
            "allow_missing_dates": True,
        },
    }
    today = pd.Timestamp.today().normalize()
    df = _make_df([
        today,
        today + pd.Timedelta(days=2),
    ],
        revenue=[100, 110],
        total_sales=[100, 110],
        orders=[1, 1],
        ad_spend=[10, 10],
        views=[1000, 800],
    )
    report = run_quality_checks(df, cfg)
    assert report.status == "WARN"
    rule = next(r for r in report.rules if r.name == "R4 Calendar")
    assert rule.status == "WARN"


def test_pass_and_writes(tmp_path: Path):
    today = pd.Timestamp.today().normalize()
    df = _make_df([
        today,
        today + pd.Timedelta(days=1),
    ],
        revenue=[100, 110],
        total_sales=[100, 110],
        orders=[1, 2],
        ad_spend=[10, 12],
        views=[1000, 1100],
    )
    cfg = {
        "date_column": BASE_CFG["date_column"],
        "mappings": dict(BASE_CFG["mappings"]),
        "quality": dict(BASE_CFG["quality"]),
    }
    report = run_quality_checks(df, cfg)
    assert report.status == "PASS"
    paths = write_quality_artifacts(report, tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()

