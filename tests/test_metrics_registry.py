from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


from Marketing_analytics.metrics_registry import compute_one, compute_series, list_metrics


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sales_day__net_sales": [100.0, 80.0, 60.0, np.nan],
            "adspend_day__amount_spent_usd": [50.0, 0.0, 20.0, np.nan],
            "sales_day__orders": [5.0, 4.0, 0.0, np.nan],
        },
        index=pd.Index(["normal", "zero_spend", "zero_orders", "nan_den"], name="day"),
    )


def test_compute_series_zero_guards(sample_frame: pd.DataFrame) -> None:
    result = compute_series(sample_frame, ["MER", "ROAS", "CAC", "AOV"])

    assert pytest.approx(result.loc["normal", "MER"], rel=1e-9) == 2.0
    assert pytest.approx(result.loc["normal", "ROAS"], rel=1e-9) == 2.0
    # Zero spend -> NaN for MER/ROAS
    assert np.isnan(result.loc["zero_spend", "MER"])
    assert np.isnan(result.loc["zero_spend", "ROAS"])
    # Zero orders -> NaN for CAC/AOV
    assert np.isnan(result.loc["zero_orders", "CAC"])
    assert np.isnan(result.loc["zero_orders", "AOV"])
    # NaN denominators propagate
    assert np.isnan(result.loc["nan_den", "CAC"])
    assert np.isnan(result.loc["nan_den", "AOV"])

    mask = sample_frame["adspend_day__amount_spent_usd"].astype(float) > 0
    assert np.allclose(result.loc[mask, "MER"], result.loc[mask, "ROAS"], atol=1e-12)
    assert np.allclose(
        result.loc[mask, "MER"] * sample_frame.loc[mask, "adspend_day__amount_spent_usd"],
        sample_frame.loc[mask, "sales_day__net_sales"],
        atol=1e-9,
    )


def test_compute_one_matches_series(sample_frame: pd.DataFrame) -> None:
    cac_series = compute_one(sample_frame, "cac")
    frame = compute_series(sample_frame, ["CAC"])
    pd.testing.assert_series_equal(cac_series, frame["CAC"], check_names=False)


def test_list_metrics() -> None:
    metrics = list_metrics()
    assert set(metrics) == {"MER", "ROAS", "CAC", "AOV"}
