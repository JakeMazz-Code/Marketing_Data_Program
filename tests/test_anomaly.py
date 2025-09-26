from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from Marketing_analytics import anomaly


def _default_cfg(columns: Dict[str, str | None]) -> Dict[str, object]:
    cfg = {
        "min_rows": 35,
        "period": 7,
        "z_threshold": 2.5,
        "columns": columns,
    }
    return cfg


def _synthetic_series(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    cycle = np.sin(2 * math.pi * np.arange(periods) / 7)
    revenue = 1200 + 150 * cycle
    ad_spend = 400 + 40 * np.cos(2 * math.pi * np.arange(periods) / 14)
    views = 9000 + 500 * np.sin(2 * math.pi * np.arange(periods) / 30)

    # Inject anomalies
    revenue[50] += 500
    revenue[70] -= 450

    df = pd.DataFrame(
        {
            "revenue": revenue,
            "ad_spend": ad_spend,
            "views": views,
        },
        index=dates,
    )
    df["mer"] = df["revenue"] / df["ad_spend"]
    return df


def test_stl_detects_spike_and_dip():
    df = _synthetic_series()
    cfg = _default_cfg(
        {
            "revenue": "revenue",
            "mer": "mer",
            "ad_spend": "ad_spend",
            "views": "views",
            "reach": None,
        }
    )

    anomalies_df = anomaly.detect_anomalies(df, cfg)

    spike_date = df.index[50].strftime("%Y-%m-%d")
    dip_date = df.index[70].strftime("%Y-%m-%d")

    revenue_rows = anomalies_df[anomalies_df["metric"] == "revenue"]
    assert spike_date in revenue_rows["date"].tolist()
    assert dip_date in revenue_rows["date"].tolist()
    assert {
        (row["date"], row["direction"])
        for _, row in revenue_rows.iterrows()
        if row["date"] in {spike_date, dip_date}
    } == {(spike_date, "up"), (dip_date, "down")}

    mer_rows = anomalies_df[anomalies_df["metric"] == "mer"]
    assert spike_date in mer_rows["date"].tolist()
    assert dip_date in mer_rows["date"].tolist()

    methods = anomalies_df.attrs.get("methods", {})
    assert methods.get("revenue") in {"stl", "ma", "both"}
    assert methods.get("mer") in {"stl", "ma", "both"}


def test_flat_series_produces_no_anomalies():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame(
        {
            "revenue": np.full(len(dates), 1000.0),
            "ad_spend": np.full(len(dates), 250.0),
            "views": np.full(len(dates), 8000.0),
        },
        index=dates,
    )
    df["mer"] = df["revenue"] / df["ad_spend"]

    cfg = _default_cfg(
        {
            "revenue": "revenue",
            "mer": "mer",
            "ad_spend": "ad_spend",
            "views": "views",
            "reach": None,
        }
    )

    anomalies_df = anomaly.detect_anomalies(df, cfg)
    assert anomalies_df.empty
    assert anomalies_df.attrs.get("note") is None


def test_fallback_moving_average(monkeypatch):
    df = _synthetic_series()
    cfg = _default_cfg(
        {
            "revenue": "revenue",
            "mer": "mer",
            "ad_spend": "ad_spend",
            "views": "views",
            "reach": None,
        }
    )

    monkeypatch.setattr(anomaly, "_stl_available", lambda: False)

    anomalies_df = anomaly.detect_anomalies(df, cfg)
    methods = anomalies_df.attrs.get("methods", {})
    assert methods
    assert all(method == "ma" for method in methods.values())
    assert not anomalies_df.empty
