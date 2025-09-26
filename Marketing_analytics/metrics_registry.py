"""Centralised marketing KPI registry.

All metric computations pull column mappings from ``configs/daily_master.json`` so
ratios remain consistent across pipeline stages, reporting surfaces, and LLM-facing
artifacts. Supported metrics (case-insensitive):

* ``MER``  = revenue / ad_spend (alias of ROAS)
* ``ROAS`` = revenue / ad_spend (alias of MER)
* ``CAC``  = ad_spend / orders
* ``AOV``  = revenue / orders

Each ratio is zero guarded: denominators ``<= 0`` or missing values yield ``NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "daily_master.json"


@dataclass(frozen=True)
class MetricColumns:
    revenue: str
    ad_spend: str
    orders: str


_REQUIRED_KEYS = {"revenue", "ad_spend", "orders"}
_SYNONYMS = {
    "revenue": {"revenue"},
    "ad_spend": {"ad_spend", "media_spend"},
    "orders": {"orders", "purchases"},
}


def list_metrics() -> List[str]:
    """Return the list of supported metric names."""

    return ["MER", "ROAS", "CAC", "AOV"]


def compute_series(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Compute multiple KPI series against *df* using canonical mappings."""

    metrics_upper = [metric.upper() for metric in metrics]
    unknown = [m for m in metrics_upper if m not in {"MER", "ROAS", "CAC", "AOV"}]
    if unknown:
        raise ValueError(f"Unsupported metrics requested: {unknown}")

    columns = _load_metric_columns()
    revenue = _ensure_float(df, columns.revenue)
    ad_spend = _ensure_float(df, columns.ad_spend)
    orders = _ensure_float(df, columns.orders)

    results: Dict[str, pd.Series] = {}
    for metric in metrics_upper:
        if metric in {"MER", "ROAS"}:
            series = _safe_ratio(revenue, ad_spend)
        elif metric == "CAC":
            series = _safe_ratio(ad_spend, orders)
        elif metric == "AOV":
            series = _safe_ratio(revenue, orders)
        else:  # pragma: no cover - already guarded above
            raise ValueError(metric)
        results[metric] = series

    frame = pd.DataFrame(results, index=df.index)
    return frame[[m for m in metrics_upper]]


def compute_one(df: pd.DataFrame, metric: str) -> pd.Series:
    """Compute a single KPI series."""

    frame = compute_series(df, [metric])
    return frame.iloc[:, 0]


def _ensure_float(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"Expected column '{column}' in dataframe")
    series = pd.to_numeric(df[column], errors="coerce")
    return series.astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    mask = denominator > 0
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    if mask.any():
        result.loc[mask] = numerator.loc[mask].astype(float) / denominator.loc[mask].astype(float)
    return result


@lru_cache(maxsize=4)
def _load_metric_columns(config_path: Path | None = None) -> MetricColumns:
    path = (config_path or DEFAULT_CONFIG_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    mappings = payload.get("mappings") or {}

    resolved: Dict[str, str] = {}
    for key in _REQUIRED_KEYS:
        values = _SYNONYMS.get(key, {key})
        column = None
        for candidate in values:
            column = mappings.get(candidate) or mappings.get(candidate.upper())
            if column:
                break
        if not column:
            raise KeyError(f"Mapping for '{key}' not found in {path}")
        resolved[key] = column

    return MetricColumns(
        revenue=resolved["revenue"],
        ad_spend=resolved["ad_spend"],
        orders=resolved["orders"],
    )


def get_metric_columns() -> MetricColumns:
    """Expose resolved metric column names."""

    return _load_metric_columns()

