"""Daily Master config and artifact pipeline (Data/ aware).

This module implements PR1: load `configs/daily_master.json`, validate schema/dtypes
against Data/master (or Data fallback), fill calendar gaps with strict rules, compute
zero-guarded KPIs, and write LLM-safe aggregated artifacts under reports/daily_master/.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import json
import numpy as np
import pandas as pd

from Marketing_analytics.anomaly import detect_anomalies
from Marketing_analytics.metrics_registry import compute_series
from Marketing_analytics.quality import run_quality_checks, write_quality_artifacts


# ----------------------------
# Config models & loader
# ----------------------------


@dataclass(slots=True)
class DailyMappings:
    revenue: str
    total_sales: str
    orders: str
    ad_spend: str
    views: str
    # Optional engagement & sales components
    engagement_reach: Optional[str] = None
    engagement_likes: Optional[str] = None
    engagement_comments: Optional[str] = None
    engagement_shares: Optional[str] = None
    engagement_follows: Optional[str] = None
    engagement_saves: Optional[str] = None
    sales_discounts: Optional[str] = None
    sales_returns: Optional[str] = None
    sales_shipping: Optional[str] = None
    sales_duties: Optional[str] = None
    sales_taxes: Optional[str] = None


@dataclass(slots=True)
class DailyDerived:
    mer: str = "mer"  # revenue / ad_spend (MER)
    roas: str = "roas"  # revenue / ad_spend
    aov: str = "aov"  # revenue / orders


@dataclass(slots=True)
class DailyAnomalySettings:
    min_rows: int = 35
    period: int = 7
    z_threshold: float = 2.5


@dataclass(slots=True)
class DailyMasterSettings:
    config_path: Path
    data_root: Path
    data_path: Path
    date_column: str
    mappings: DailyMappings
    derived: DailyDerived
    anomalies: DailyAnomalySettings
    artifacts_dir: Path
    freshness_days: int = 7
    allow_missing_dates: bool = True
    stop_on_fail: bool = False


def _as_path(base: Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p)


def is_daily_master_config(config_path: Path) -> bool:
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and "mappings" in payload and "date_column" in payload


def load_daily_master_settings(config_path: Path, *, data_root_override: Optional[Path] = None) -> DailyMasterSettings:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("daily_master config must be a JSON object")

    project_root = config_path.parent.parent

    data_root_raw = payload.get("data_root", "Data")
    data_root = Path(data_root_raw)
    if not data_root.is_absolute():
        data_root = (config_path.parent / data_root).resolve()
        if not data_root.exists():
            candidate = (project_root / data_root_raw).resolve()
            if candidate.exists():
                data_root = candidate
    if data_root_override is not None:
        data_root = data_root_override.resolve()

    data_path_raw = payload.get("data_path", "master/MASTER_DAILY_JOINED.csv")
    data_path = Path(data_path_raw)
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()
        if not data_path.exists():
            candidate = (project_root / data_path_raw).resolve()
            if candidate.exists():
                data_path = candidate
    # Fallbacks to honor layout contract
    if not data_path.exists():
        candidate = data_root / "master" / "MASTER_DAILY_JOINED.csv"
        alt = data_root / "MASTER_DAILY_JOINED.csv"
        if candidate.exists():
            data_path = candidate
        elif alt.exists():
            data_path = alt

    date_column = str(payload.get("date_column", "date"))
    mappings_payload_raw = payload.get("mappings") or {}
    # Accept synonyms from config and normalize to dataclass fields
    key_map = {
        # required
        "revenue": "revenue",
        "total_sales": "total_sales",
        "orders": "orders",
        "ad_spend": "ad_spend",
        "views": "views",
        # engagement optional
        "engagement_reach": "engagement_reach",
        "reach": "engagement_reach",
        "engagement_likes": "engagement_likes",
        "likes": "engagement_likes",
        "engagement_comments": "engagement_comments",
        "comments": "engagement_comments",
        "engagement_shares": "engagement_shares",
        "shares": "engagement_shares",
        "engagement_follows": "engagement_follows",
        "follows": "engagement_follows",
        "engagement_saves": "engagement_saves",
        "saves": "engagement_saves",
        # sales components
        "sales_discounts": "sales_discounts",
        "discount_amount": "sales_discounts",
        "sales_returns": "sales_returns",
        "returns_amount": "sales_returns",
        "sales_shipping": "sales_shipping",
        "shipping_amount": "sales_shipping",
        "sales_duties": "sales_duties",
        "duties_amount": "sales_duties",
        "sales_taxes": "sales_taxes",
        "tax_amount": "sales_taxes",
    }
    mappings_payload: Dict[str, Optional[str]] = {}
    for k, v in mappings_payload_raw.items():
        norm = key_map.get(k, None)
        if norm:
            mappings_payload[norm] = v

    required = [
        "revenue",
        "total_sales",
        "orders",
        "ad_spend",
        "views",
    ]
    for key in required:
        if not mappings_payload.get(key):
            raise ValueError(f"Required mapping '{key}' missing from config")
    mappings = DailyMappings(**mappings_payload)

    derived_payload = payload.get("derived") or {}
    # If expressions are provided (e.g., "revenue / ad_spend"), keep default names
    roas_name = derived_payload.get("roas")
    aov_name = derived_payload.get("aov")
    def _clean_name(expr: Optional[str], default: str) -> str:
        if not isinstance(expr, str):
            return default
        return default if ("/" in expr or " " in expr) else expr
    derived = DailyDerived(roas=_clean_name(roas_name, "roas"), aov=_clean_name(aov_name, "aov"))

    anomalies_payload = payload.get("anomalies") or {}
    if isinstance(anomalies_payload, dict):
        anomaly_min_rows = int(anomalies_payload.get("min_rows", 35))
        anomaly_period = int(anomalies_payload.get("period", 7))
        anomaly_threshold = float(anomalies_payload.get("z_threshold", 2.5))
    else:
        anomaly_min_rows, anomaly_period, anomaly_threshold = 35, 7, 2.5
    anomalies = DailyAnomalySettings(
        min_rows=anomaly_min_rows,
        period=anomaly_period,
        z_threshold=anomaly_threshold,
    )

    artifacts_dir_raw = payload.get("artifacts_dir", "reports/daily_master")
    artifacts_dir = Path(artifacts_dir_raw)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (project_root / artifacts_dir).resolve()

    quality_payload = payload.get("quality") or {}
    freshness_days = int(quality_payload.get("freshness_days", 7)) if isinstance(quality_payload, dict) else 7
    allow_missing_dates = bool(quality_payload.get("allow_missing_dates", True)) if isinstance(quality_payload, dict) else True
    stop_on_fail = bool(quality_payload.get("stop_on_fail", False)) if isinstance(quality_payload, dict) else False

    return DailyMasterSettings(
        config_path=config_path.resolve(),
        data_root=data_root,
        data_path=data_path,
        date_column=date_column,
        mappings=mappings,
        derived=derived,
        anomalies=anomalies,
        artifacts_dir=artifacts_dir,
        freshness_days=freshness_days,
        allow_missing_dates=allow_missing_dates,
        stop_on_fail=stop_on_fail,
    )


# ----------------------------
# Core processing
# ----------------------------


def _enforce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _full_calendar_index(dates: pd.Series) -> pd.DatetimeIndex:
    min_d = dates.min()
    max_d = dates.max()
    return pd.date_range(min_d, max_d, freq="D")



def _null_counts(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in cols if c in df.columns}


def run_daily_master(settings: DailyMasterSettings) -> Dict[str, str]:
    # Load
    df = pd.read_csv(settings.data_path)
    if settings.date_column not in df.columns:
        raise AssertionError(f"Date column '{settings.date_column}' not found in input")
    df[settings.date_column] = pd.to_datetime(df[settings.date_column], errors="coerce")
    df = df.dropna(subset=[settings.date_column]).sort_values(settings.date_column)

    m = settings.mappings
    required_cols = [m.revenue, m.total_sales, m.orders, m.ad_spend, m.views]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Required mapped columns missing from data: {missing}")

    # Enforce numeric types
    numeric_candidates = [
        m.revenue,
        m.total_sales,
        m.orders,
        m.ad_spend,
        m.views,
        m.engagement_reach,
        m.engagement_likes,
        m.engagement_comments,
        m.engagement_shares,
        m.engagement_follows,
        m.engagement_saves,
        m.sales_discounts,
        m.sales_returns,
        m.sales_shipping,
        m.sales_duties,
        m.sales_taxes,
    ]
    _enforce_numeric(df, [c for c in numeric_candidates if c])

    # Calendar reindex
    df = df.set_index(settings.date_column)
    full_idx = _full_calendar_index(df.index)
    df = df.reindex(full_idx)
    df.index.name = settings.date_column

    quality_cfg = {
        "date_column": settings.date_column,
        "mappings": asdict(m),
        "quality": {
            "freshness_days": settings.freshness_days,
            "allow_missing_dates": settings.allow_missing_dates,
            "stop_on_fail": settings.stop_on_fail,
        },
    }
    quality_report = run_quality_checks(df.copy(), quality_cfg)
    quality_paths = write_quality_artifacts(quality_report, settings.artifacts_dir)
    if quality_report.status == "FAIL" and settings.stop_on_fail:
        failing = ", ".join(rule.name for rule in quality_report.rules if rule.status == "FAIL") or "unknown"
        raise RuntimeError(f"Data quality gate failed (rules: {failing})")

    # Fill rules: sales remain NaN; ad/engagement may be 0
    ad_like = [m.ad_spend]
    engagement_like = [m.views, m.engagement_reach, m.engagement_likes, m.engagement_comments,
                       m.engagement_shares, m.engagement_follows, m.engagement_saves]
    for c in ad_like + [c for c in engagement_like if c]:
        if c and c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Derived KPIs
    derived_mer_col = settings.derived.mer
    derived_roas_col = settings.derived.roas
    derived_aov_col = settings.derived.aov
    kpi_frame = compute_series(df, ['MER', 'ROAS', 'AOV'])
    df[derived_mer_col] = kpi_frame['MER']
    df[derived_roas_col] = kpi_frame['ROAS']
    df[derived_aov_col] = kpi_frame['AOV']

    anomalies_cfg = {
        "min_rows": settings.anomalies.min_rows,
        "period": settings.anomalies.period,
        "z_threshold": settings.anomalies.z_threshold,
        "columns": {
            "revenue": m.revenue,
            "mer": derived_mer_col,
            "ad_spend": m.ad_spend,
            "views": m.views,
            "reach": m.engagement_reach,
        },
    }
    anomalies_df = detect_anomalies(df, anomalies_cfg)

    # Build series for output
    out_cols = [
        m.revenue,
        m.total_sales,
        m.orders,
        m.ad_spend,
        m.views,
        derived_mer_col,
        derived_roas_col,
        derived_aov_col,
    ]
    optional_cols = [
        m.engagement_reach,
        m.engagement_likes,
        m.engagement_comments,
        m.engagement_shares,
        m.engagement_follows,
        m.engagement_saves,
        m.sales_discounts,
        m.sales_returns,
        m.sales_shipping,
        m.sales_duties,
        m.sales_taxes,
    ]
    out_cols.extend([c for c in optional_cols if c])

    # Prepare artifacts dir
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    anomalies_path = settings.artifacts_dir / "anomalies.json"
    anomalies_notes_path = settings.artifacts_dir / "anomalies_notes.md"
    if anomalies_df.empty:
        anomalies_path.write_text("[]\n", encoding="utf-8")
    else:
        anomalies_payload = anomalies_df.to_dict(orient="records")
        anomalies_path.write_text(json.dumps(anomalies_payload, indent=2), encoding="utf-8")
    attr_note = anomalies_df.attrs.get("note") if hasattr(anomalies_df, "attrs") else None
    if anomalies_df.empty:
        if attr_note:
            note_lines = [str(attr_note)]
        else:
            note_lines = [f"No anomalies detected (|z| >= {settings.anomalies.z_threshold:.2f})."]
    else:
        note_lines = []
        if attr_note:
            note_lines.append(str(attr_note))
        methods = anomalies_df.attrs.get("methods", {})
        if methods:
            method_text = ", ".join(f"{metric}:{method}" for metric, method in methods.items())
            note_lines.append(f"Detection methods: {method_text}.")
        note_lines.append(f"Detected {len(anomalies_df)} anomalies (|z| >= {settings.anomalies.z_threshold:.2f}).")
    anomalies_notes_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    # series.jsonl
    series_path = settings.artifacts_dir / "series.jsonl"
    with series_path.open("w", encoding="utf-8") as fh:
        for ts, row in df[out_cols].iterrows():
            record = {"date": ts.strftime("%Y-%m-%d")}
            for c in out_cols:
                val = row.get(c)
                if pd.isna(val):
                    record[c] = None
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    record[c] = float(val)
                else:
                    record[c] = val
            fh.write(json.dumps(record) + "\n")

    # shape.json
    shape_payload = {
        "rows": int(len(df)),
        "min_date": df.index.min().strftime("%Y-%m-%d"),
        "max_date": df.index.max().strftime("%Y-%m-%d"),
        "null_counts": _null_counts(df, out_cols),
        "date_monotonic": True,
    }
    shape_path = settings.artifacts_dir / "shape.json"
    shape_path.write_text(json.dumps(shape_payload, indent=2), encoding="utf-8")

    # data_quality.json
    today = datetime.now(timezone.utc).date()
    max_date = df.index.max().date()
    freshness_days = (today - max_date).days
    gaps = pd.date_range(df.index.min(), df.index.max()).difference(df.index)
    freshness_threshold = settings.freshness_days
    allow_missing = settings.allow_missing_dates
    dq = {
        "checks": [
            {"name": "schema_required_columns", "status": "pass", "details": required_cols},
            {"name": "date_monotonic", "status": "pass"},
            {"name": "non_negative_spend", "status": "pass" if (df[m.ad_spend] >= 0).all() else "fail"},
            {"name": "freshness_days", "status": "pass" if freshness_days <= freshness_threshold else "warn", "value": freshness_days},
            {
                "name": "gap_count",
                "status": "pass"
                if len(gaps) == 0 or allow_missing
                else ("fail" if settings.stop_on_fail else "warn"),
                "value": int(len(gaps)),
            },
        ]
    }
    dq_path = settings.artifacts_dir / "data_quality.json"
    dq_path.write_text(json.dumps(dq, indent=2), encoding="utf-8")

    # llm_payload.json (aggregates only)
    def _agg_window(days: int) -> Dict[str, Optional[float]]:
        frame = df.tail(days)
        rev = float(frame[m.revenue].sum(skipna=True)) if m.revenue in frame.columns else None
        spend = float(frame[m.ad_spend].sum(skipna=True)) if m.ad_spend in frame.columns else None
        orders = float(frame[m.orders].sum(skipna=True)) if m.orders in frame.columns else None
        views = float(frame[m.views].sum(skipna=True)) if m.views in frame.columns else None
        metric_input = pd.DataFrame({
            m.revenue: [rev if rev is not None else np.nan],
            m.ad_spend: [spend if spend is not None else np.nan],
            m.orders: [orders if orders is not None else np.nan],
        })
        ratios = compute_series(metric_input, ['MER', 'ROAS', 'AOV'])
        mer_value = ratios['MER'].iat[0]
        roas_value = ratios['ROAS'].iat[0]
        aov_value = ratios['AOV'].iat[0]
        mer = None if pd.isna(mer_value) else float(mer_value)
        roas = None if pd.isna(roas_value) else float(roas_value)
        aov = None if pd.isna(aov_value) else float(aov_value)
        return {
            "revenue_sum": rev,
            "ad_spend_sum": spend,
            "orders_sum": orders,
            "views_sum": views,
            settings.derived.mer: mer,
            settings.derived.roas: roas,
            settings.derived.aov: aov,
        }

    llm_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "date_range": {
            "min": df.index.min().strftime("%Y-%m-%d"),
            "max": df.index.max().strftime("%Y-%m-%d"),
        },
        "windows": {
            "7d": _agg_window(7),
            "28d": _agg_window(28),
            "90d": _agg_window(90),
        },
    }
    llm_path = settings.artifacts_dir / "llm_payload.json"
    llm_path.write_text(json.dumps(llm_payload, indent=2), encoding="utf-8")

    return {
        "series": str(series_path),
        "shape": str(shape_path),
        "data_quality": str(dq_path),
        "llm_payload": str(llm_path),
        "quality_report": quality_paths,
        "anomalies": str(anomalies_path),
        "anomalies_notes": str(anomalies_notes_path),
    }


def run_daily_master_from_config(config_path: Path, *, data_root_override: Optional[Path] = None) -> Dict[str, str]:
    settings = load_daily_master_settings(Path(config_path), data_root_override=data_root_override)
    return run_daily_master(settings)



