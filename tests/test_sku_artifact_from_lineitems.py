from __future__ import annotations

from pathlib import Path

import pandas as pd

from Marketing_analytics.daily_extras import generate_extra_artifacts


def test_sku_artifact_from_lineitems(tmp_path):
    data_root = tmp_path / "Data"
    data_root.mkdir()

    rows = [
        {"created_at": "2025-08-01", "lineitem_title": "Glow Serum", "lineitem_quantity": 3, "line_total": 150.0},
        {"created_at": "2025-08-01", "lineitem_title": "Hydra Mask", "lineitem_quantity": 2, "line_total": 90.0},
        {"created_at": "2025-08-01", "lineitem_title": "Vitamin Mist", "lineitem_quantity": 1, "line_total": 45.0},
        {"created_at": "2025-08-02", "lineitem_title": "Glow Serum", "lineitem_quantity": 4, "line_total": 200.0},
        {"created_at": "2025-08-02", "lineitem_title": "Hydra Mask", "lineitem_quantity": 3, "line_total": 135.0},
        {"created_at": "2025-08-02", "lineitem_title": "Vitamin Mist", "lineitem_quantity": 2, "line_total": 90.0},
    ]
    export_path = data_root / "orders_export.csv"
    pd.DataFrame(rows).to_csv(export_path, index=False)

    artifacts_dir = tmp_path / "reports"
    results = generate_extra_artifacts(data_root, artifacts_dir, line_item_globs=[str(export_path)])

    sku_result = results["sku_series"]
    assert sku_result.path is not None

    frame = pd.read_json(Path(sku_result.path), lines=True)
    assert frame.shape[0] >= 6

    expected_columns = [
        "date", "sku_code",
        "product_title",
        "variant",
        "source",
        "revenue",
        "units",
        "revenue_28d",
        "units_28d",
        "run_rate_units_28d",
        "revenue_share_28d",
        "units_share_28d",
    ]
    for column in expected_columns:
        assert column in frame.columns, f"missing column {column}"
        assert frame[column].notna().any(), f"column {column} should have values"

    assert frame["product_title"].str.contains("Glow Serum").any()
    assert frame["sku"].isna().all()