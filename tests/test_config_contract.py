from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
CONFIG_PATH = PROJECT_ROOT / "configs" / "daily_master.json"

from Marketing_analytics.daily_master import load_daily_master_settings, run_daily_master




def test_config_required_mappings_present(tmp_path):
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["artifacts_dir"] = str(tmp_path / "reports")
    tmp_config = tmp_path / "daily_master.json"
    tmp_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    settings = load_daily_master_settings(tmp_config, data_root_override=DATA_ROOT)

    mappings = settings.mappings
    assert mappings.revenue == "sales_day__net_sales"
    assert mappings.total_sales == "sales_day__total_sales"
    assert mappings.orders == "sales_day__orders"
    assert mappings.ad_spend == "adspend_day__amount_spent_usd"
    assert mappings.views == "views_day__primary"

    assert settings.derived.mer == "mer"
    assert settings.data_path.exists(), "Data path should resolve to existing joined master"
    assert settings.artifacts_dir == (tmp_path / "reports").resolve()


def test_daily_master_artifacts_and_zero_guards(tmp_path):
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    artifacts_dir = tmp_path / "reports"
    payload["artifacts_dir"] = str(artifacts_dir)
    tmp_config = tmp_path / "daily_master.json"
    tmp_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    settings = load_daily_master_settings(tmp_config, data_root_override=DATA_ROOT)
    result = run_daily_master(settings)

    for key in {"series", "shape", "data_quality", "llm_payload", "anomalies", "anomalies_notes"}:
        path = Path(result[key])
        assert path.exists(), f"Expected artifact {key} at {path}"
    quality = result.get("quality_report")
    assert quality is not None
    assert Path(quality["json"]).exists()
    assert Path(quality["markdown"]).exists()

    series_path = Path(result["series"])
    series_df = pd.read_json(series_path, lines=True)
    parsed_dates = pd.to_datetime(series_df["date"], errors="coerce")
    assert parsed_dates.notna().all(), "date column must be parseable"
    assert parsed_dates.is_monotonic_increasing, "dates should be sorted"

    roas_col = settings.derived.roas
    aov_col = settings.derived.aov
    mer_col = settings.derived.mer

    assert mer_col in series_df.columns, "MER column should be present in series output"
    assert (series_df[mer_col].dropna() != float("inf")).all(), "MER should be finite"
    assert (series_df[roas_col].dropna() != float("inf")).all(), "ROAS should be finite"
    assert (series_df[aov_col].dropna() != float("inf")).all(), "AOV should be finite"

    # Ensure zero guard by checking a known zero-spend day
    zero_spend_days = series_df.loc[series_df[settings.mappings.ad_spend] == 0]
    if not zero_spend_days.empty:
        assert zero_spend_days[roas_col].isna().all(), "ROAS should be NaN when spend is zero"
        assert zero_spend_days[mer_col].isna().all(), "MER should be NaN when spend is zero"

    anomalies_path = Path(result["anomalies"])
    anomalies_payload = json.loads(anomalies_path.read_text(encoding="utf-8"))
    assert isinstance(anomalies_payload, list), "Anomalies artifact should be a JSON list"
    notes_path = Path(result["anomalies_notes"])
    assert notes_path.read_text(encoding="utf-8").strip(), "Anomaly notes should not be empty"
