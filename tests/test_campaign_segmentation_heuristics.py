from __future__ import annotations

import pandas as pd
import pytest

from Marketing_analytics.md_brief import _aggregate_segments


def test_campaign_segmentation_heuristics():
    df = pd.DataFrame(
        [
            {
                "campaign": "Prospecting - Broad",
                "adset": "Cold LAA",
                "revenue": 1500.0,
                "ad_spend": 500.0,
                "orders": 30,
            },
            {
                "campaign": "Retarget - Cart DPA",
                "adset": "Warm Cart",
                "revenue": 900.0,
                "ad_spend": 150.0,
                "orders": 25,
            },
            {
                "campaign": "Newsletter - Loyalty",
                "adset": "VIP",
                "revenue": 400.0,
                "ad_spend": 80.0,
                "orders": 10,
            },
        ]
    )

    summary = _aggregate_segments(df)
    assert not summary.empty

    tof_row = summary.loc[summary["__segment"] == "Prospecting / TOF"].iloc[0]
    bof_row = summary.loc[summary["__segment"] == "Retargeting / BOF"].iloc[0]

    assert tof_row["__segment"] == "Prospecting / TOF"
    assert bof_row["__segment"] == "Retargeting / BOF"
    assert pytest.approx(tof_row["revenue"], rel=1e-6) == 1500.0
    assert pytest.approx(tof_row["ad_spend"], rel=1e-6) == 500.0
    assert pytest.approx(bof_row["revenue"], rel=1e-6) == 900.0
    assert pytest.approx(bof_row["ad_spend"], rel=1e-6) == 150.0

