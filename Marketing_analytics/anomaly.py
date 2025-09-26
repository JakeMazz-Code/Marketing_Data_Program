"""Revenue & MER anomaly detection helpers.

The detector favours STL decomposition when statsmodels is available; otherwise it
falls back to a moving-average residual z-score strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard exercised in tests
    from statsmodels.tsa.seasonal import STL  # type: ignore
except Exception:  # pragma: no cover - statsmodels optional dependency
    STL = None  # type: ignore


_DEFAULT_COLUMNS = [
    "date",
    "metric",
    "z",
    "direction",
    "spend_change_pct",
    "engagement_change_pct",
    "note",
]


@dataclass(frozen=True)
class AnomalyResult:
    residuals: pd.Series
    z_score: pd.Series
    method: str


def detect_anomalies(df: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    """Detect revenue/MER anomalies using STL with MA fallback corroboration."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Anomaly detector expects a DatetimeIndex")

    cfg = cfg or {}
    min_rows = int(cfg.get("min_rows", 35))
    period = int(cfg.get("period", 7))
    threshold = float(cfg.get("z_threshold", 2.5))
    columns = dict(cfg.get("columns", {}))

    revenue_col = columns.get("revenue")
    mer_col = columns.get("mer")
    spend_col = columns.get("ad_spend")
    views_col = columns.get("views")
    reach_col = columns.get("reach")

    missing_columns = [name for name in (revenue_col, mer_col, spend_col) if not name]
    if missing_columns:
        raise ValueError("Anomaly config missing required column mappings")

    frame = df.copy().sort_index()
    revenue_series = pd.to_numeric(frame[revenue_col], errors="coerce")
    mer_series = pd.to_numeric(frame[mer_col], errors="coerce")
    spend_series = pd.to_numeric(frame[spend_col], errors="coerce")

    available = revenue_series.dropna()
    if len(available) < min_rows:
        result = _empty_anomaly_frame()
        result.attrs["note"] = f"Insufficient history: need >= {min_rows} revenue rows, found {len(available)}."
        return result

    target_engagement_col = reach_col if reach_col and reach_col in frame.columns else views_col
    engagement_series = None
    if target_engagement_col and target_engagement_col in frame.columns:
        engagement_series = pd.to_numeric(frame[target_engagement_col], errors="coerce")

    spend_change = spend_series.pct_change().replace([np.inf, -np.inf], np.nan)
    engagement_change = None
    if engagement_series is not None:
        engagement_change = engagement_series.pct_change().replace([np.inf, -np.inf], np.nan)

    anomalies = []
    methods: Dict[str, str] = {}

    for metric_name, series in (("revenue", revenue_series), ("mer", mer_series)):
        cleaned = series.dropna()
        if len(cleaned) < min_rows:
            continue

        stl_result = _compute_stl_result(cleaned, period)
        ma_result = _compute_ma_result(cleaned)

        if stl_result is None and ma_result is None:
            continue

        candidate_methods = []
        if stl_result is not None:
            candidate_methods.append("stl")
        if ma_result is not None:
            candidate_methods.append("ma")
        default_method = "both" if len(candidate_methods) == 2 else candidate_methods[0]
        methods[metric_name] = default_method

        aligned_z_stl = (
            stl_result.z_score.reindex(frame.index)
            if stl_result is not None
            else pd.Series(np.nan, index=frame.index, dtype="float64")
        )
        aligned_residuals_stl = (
            stl_result.residuals.reindex(frame.index)
            if stl_result is not None
            else pd.Series(np.nan, index=frame.index, dtype="float64")
        )
        stl_mask = (
            aligned_z_stl.abs() >= threshold
            if stl_result is not None
            else pd.Series(False, index=frame.index)
        )

        aligned_z_ma = (
            ma_result.z_score.reindex(frame.index)
            if ma_result is not None
            else pd.Series(np.nan, index=frame.index, dtype="float64")
        )
        aligned_residuals_ma = (
            ma_result.residuals.reindex(frame.index)
            if ma_result is not None
            else pd.Series(np.nan, index=frame.index, dtype="float64")
        )
        ma_threshold = (threshold + 0.2) if stl_result is not None else threshold
        ma_mask = (
            aligned_z_ma.abs() >= ma_threshold
            if ma_result is not None
            else pd.Series(False, index=frame.index)
        )

        combined_mask = stl_mask | ma_mask
        if not combined_mask.any():
            continue

        metric_triggers: set[str] = set()

        for ts in frame.index[combined_mask]:
            flag_stl = bool(stl_mask.loc[ts]) if stl_result is not None else False
            flag_ma = bool(ma_mask.loc[ts]) if ma_result is not None else False

            if flag_stl:
                metric_triggers.add("stl")
            if flag_ma:
                metric_triggers.add("ma")

            z_stl = aligned_z_stl.loc[ts] if stl_result is not None else np.nan
            z_ma = aligned_z_ma.loc[ts] if ma_result is not None else np.nan
            resid_stl = aligned_residuals_stl.loc[ts] if stl_result is not None else np.nan
            resid_ma = aligned_residuals_ma.loc[ts] if ma_result is not None else np.nan

            if flag_stl and flag_ma:
                use_stl = not pd.isna(z_stl) and (pd.isna(z_ma) or abs(z_stl) >= abs(z_ma))
                if use_stl:
                    z_value = float(z_stl)
                    residual_value = resid_stl
                else:
                    z_value = float(z_ma)
                    residual_value = resid_ma
            elif flag_stl:
                z_value = float(z_stl)
                residual_value = resid_stl
            else:
                z_value = float(z_ma)
                residual_value = resid_ma

            if pd.isna(residual_value):
                if flag_stl and not pd.isna(resid_stl):
                    residual_value = resid_stl
                elif flag_ma and not pd.isna(resid_ma):
                    residual_value = resid_ma
                else:
                    residual_value = 0.0

            direction = "up" if residual_value > 0 else "down"

            spend_pct_raw = spend_change.loc[ts]
            spend_pct = float(spend_pct_raw) if not pd.isna(spend_pct_raw) else None
            if engagement_change is not None:
                engagement_raw = engagement_change.loc[ts]
                engagement_pct = float(engagement_raw) if not pd.isna(engagement_raw) else None
            else:
                engagement_pct = None
            note = _make_note(metric_name, direction, spend_pct, engagement_pct)
            anomalies.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "metric": metric_name,
                    "z": float(z_value) if not pd.isna(z_value) else float("nan"),
                    "direction": direction,
                    "spend_change_pct": spend_pct,
                    "engagement_change_pct": engagement_pct,
                    "note": note,
                }
            )

        if metric_triggers:
            if "stl" in metric_triggers and "ma" in metric_triggers:
                methods[metric_name] = "both"
            elif "stl" in metric_triggers:
                methods[metric_name] = "stl"
            elif "ma" in metric_triggers:
                methods[metric_name] = "ma"

    result_df = pd.DataFrame(anomalies, columns=_DEFAULT_COLUMNS)
    if not result_df.empty:
        result_df = result_df.sort_values(["date", "metric"], ignore_index=True)
    result_df.attrs["methods"] = methods
    return result_df


def _empty_anomaly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_DEFAULT_COLUMNS)


def _compute_stl_result(series: pd.Series, period: int) -> Optional[AnomalyResult]:
    if not _stl_available():
        return None
    if len(series) < max(period * 2, 28):
        return None
    try:
        residuals = _stl_residuals(series, period)
        z = _robust_z(residuals)
    except Exception:
        return None
    if not z.notna().any():
        return None
    return AnomalyResult(residuals=residuals, z_score=z, method="stl")


def _compute_ma_result(series: pd.Series, window: int = 28) -> Optional[AnomalyResult]:
    residuals, z_std = _moving_average_z(series, window=window)
    z_robust = _robust_z(residuals)
    z = z_robust if z_robust.notna().any() else z_std
    if residuals.isna().all() and z.isna().all():
        return None
    return AnomalyResult(residuals=residuals, z_score=z, method="ma")


def _stl_available() -> bool:
    return STL is not None


def _stl_residuals(series: pd.Series, period: int) -> pd.Series:
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return pd.Series(result.resid, index=series.index, dtype="float64")


def _robust_z(residuals: pd.Series) -> pd.Series:
    resid = residuals.astype(float)
    median = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - median))
    if not np.isfinite(mad) or mad <= 0 or mad < 1e-9:
        return pd.Series(np.nan, index=resid.index, dtype="float64")
    scale = 1.4826 * mad
    z = (resid - median) / scale
    return pd.Series(z, index=resid.index, dtype="float64")


def _moving_average_z(series: pd.Series, window: int = 28) -> tuple[pd.Series, pd.Series]:
    window = max(window, 7)
    mean = series.rolling(window=window, min_periods=window // 2).mean()
    residuals = series - mean
    std = residuals.rolling(window=window, min_periods=window // 2).std(ddof=0)
    std = std.where(std >= 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = residuals / std
    z = z.replace([np.inf, -np.inf], np.nan)
    return residuals, z


def _make_note(metric: str, direction: str, spend_pct: Optional[float], engagement_pct: Optional[float]) -> str:
    title = "Revenue" if metric == "revenue" else metric.upper()
    noun = "spike" if direction == "up" else "dip"
    components = []
    if spend_pct is not None:
        components.append(f"spend {_format_pct(spend_pct)}")
    if engagement_pct is not None:
        components.append(f"engagement {_format_pct(engagement_pct)}")
    if components:
        if len(components) == 2:
            trailer = f"concurrent {components[0]} and {components[1]}"
        else:
            trailer = components[0]
        return f"{title} {noun} w/ {trailer}"
    return f"{title} {noun} without supporting spend/engagement change"


def _format_pct(value: float) -> str:
    return f"{value:+.0%}"
