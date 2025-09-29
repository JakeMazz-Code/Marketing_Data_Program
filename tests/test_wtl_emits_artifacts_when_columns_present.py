from __future__ import annotations

from pathlib import Path

import pandas as pd

from Marketing_analytics.daily_extras import generate_extra_artifacts


def test_wtl_emits_artifacts_when_columns_present(tmp_path):
    data_root = tmp_path / "Data"
    data_root.mkdir()
    records = [
        {
            "date": "2025-09-01",
            "channel": "Meta",
            "campaign": "Prospecting - LAA",
            "adset": "Cold_Broad",
            "creative_id": "CR-M-1",
            "sku": "SKU-001",
            "product_title": "Glow Serum",
            "units": 18,
            "orders": 15,
            "revenue": 1450.0,
            "ad_spend": 380.0,
            "views": 18000,
            "clicks": 520,
            "discounts": 60.0,
            "returns": 40.0,
            "shipping": 32.0,
            "taxes": 18.0,
            "duties": 6.0,
            "on_hand": 210,
            "on_order": 65,
            "lead_time_days": 20,
        },
        {
            "date": "2025-09-02",
            "channel": "TikTok",
            "campaign": "Retarget - DPA",
            "adset": "Warm_Close",
            "creative_id": "CR-T-4",
            "sku": "SKU-002",
            "product_title": "Hydra Mask",
            "units": 12,
            "orders": 11,
            "revenue": 980.0,
            "ad_spend": 210.0,
            "views": 12500,
            "clicks": 410,
            "discounts": 25.0,
            "returns": 15.0,
            "shipping": 28.0,
            "taxes": 14.0,
            "duties": 4.0,
            "on_hand": 150,
            "on_order": 45,
            "lead_time_days": 17,
        },
    ]
    pd.DataFrame.from_records(records).to_csv(data_root / "daily.csv", index=False)

    artifacts_dir = tmp_path / "reports"
    results = generate_extra_artifacts(data_root, artifacts_dir)

    expected_columns = {
        "sku_series": {"date", "sku_code", "sku", "product_title", "variant", "source", "units", "orders", "revenue", "revenue_28d", "units_28d", "revenue_share_28d", "units_share_28d", "run_rate_units_28d"},
        "channel_series": {
            "date",
            "channel",
            "revenue",
            "ad_spend",
            "orders",
            "views",
            "clicks",
            "mer",
            "roas",
            "cac",
            "aov",
            "rpm",
            "ctr",
        },
        "campaign_series": {
            "date",
            "channel",
            "campaign",
            "adset",
            "revenue",
            "ad_spend",
            "orders",
            "views",
            "clicks",
            "mer",
            "roas",
            "cac",
            "aov",
            "rpm",
            "ctr",
        },
        "creative_series": {
            "date",
            "channel",
            "creative_id",
            "views",
            "clicks",
            "ad_spend",
            "revenue",
            "mer",
            "roas",
            "rpm",
            "ctr",
        },
        "margin_series": {
            "date",
            "revenue",
            "discounts",
            "returns",
            "shipping",
            "taxes",
            "duties",
        },
        "inventory_snapshot": {
            "date",
            "sku",
            "product_title",
            "on_hand",
            "on_order",
            "lead_time_days",
        },
    }

    for key, expected in expected_columns.items():
        result = results[key]
        assert result.path is not None, f"{key} should be written"
        artifact_path = Path(result.path)
        assert artifact_path.exists()
        frame = pd.read_json(artifact_path, lines=True)
        assert not frame.empty
        assert expected.issubset(set(frame.columns))
