from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import json
import pandas as pd

RuleStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(slots=True)
class RuleResult:
    name: str
    status: RuleStatus
    detail: str
    sample_rows: Optional[int] = None


@dataclass(slots=True)
class QualityReport:
    status: RuleStatus
    rules: List[RuleResult]
    stats: Dict[str, object]


def run_quality_checks(df: pd.DataFrame, cfg: Dict[str, object]) -> QualityReport:
    quality_cfg = cfg.get("quality", {}) if isinstance(cfg, dict) else {}
    mappings = cfg.get("mappings", {}) if isinstance(cfg, dict) else {}

    freshness_days = int(quality_cfg.get("freshness_days", 7)) if isinstance(quality_cfg, dict) else 7
    allow_missing = bool(quality_cfg.get("allow_missing_dates", True)) if isinstance(quality_cfg, dict) else True
    stop_on_fail = bool(quality_cfg.get("stop_on_fail", False)) if isinstance(quality_cfg, dict) else False

    rules: List[RuleResult] = []

    # R1 Schema
    required_fields = []
    if isinstance(mappings, dict):
        required_fields.extend(
            mappings.get(key)
            for key in ("revenue", "total_sales", "orders", "ad_spend", "views")
        )
    required_columns = [col for col in required_fields if isinstance(col, str) and col]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        rules.append(
            RuleResult(
                name="R1 Schema",
                status="FAIL",
                detail=f"Missing required columns: {', '.join(missing_cols)}",
                sample_rows=len(missing_cols),
            )
        )
    else:
        rules.append(
            RuleResult(
                name="R1 Schema",
                status="PASS",
                detail="All required columns present",
                sample_rows=None,
            )
        )

    # R2 Types
    type_status: RuleStatus = "PASS"
    type_details: List[str] = []
    if not isinstance(df.index, pd.DatetimeIndex):
        type_status = "FAIL"
        type_details.append("Index is not a DatetimeIndex")
        dt_index = None
    else:
        dt_index = df.index
        if dt_index.tz is not None:
            type_status = "FAIL"
            type_details.append("Datetime index must be timezone naive")
        if not dt_index.equals(dt_index.normalize()):
            type_status = "FAIL"
            type_details.append("Timestamps contain intra-day components")
        if not dt_index.is_monotonic_increasing:
            type_status = "FAIL"
            type_details.append("Dates are not monotonically increasing")
    rules.append(
        RuleResult(
            name="R2 Types",
            status=type_status,
            detail="; ".join(type_details) if type_details else "Datetime index is daily, naive, and monotonic",
            sample_rows=None,
        )
    )

    min_date = dt_index.min() if dt_index is not None and len(dt_index) else None
    max_date = dt_index.max() if dt_index is not None and len(dt_index) else None

    # R3 Freshness
    if max_date is None or pd.isna(max_date):
        rules.append(
            RuleResult(
                name="R3 Freshness",
                status="FAIL",
                detail="No valid dates available to assess freshness",
                sample_rows=None,
            )
        )
    else:
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=freshness_days)
        latest = max_date.date()
        if latest < cutoff:
            rules.append(
                RuleResult(
                    name="R3 Freshness",
                    status="FAIL",
                    detail=f"Latest date {latest} older than freshness cutoff {cutoff}",
                    sample_rows=None,
                )
            )
        else:
            rules.append(
                RuleResult(
                    name="R3 Freshness",
                    status="PASS",
                    detail=f"Latest date {latest} within freshness threshold ({freshness_days} days)",
                    sample_rows=None,
                )
            )

    # R4 Calendar continuity
    missing_dates: List[str] = []
    if min_date is not None and max_date is not None and dt_index is not None:
        expected = pd.date_range(min_date, max_date, freq="D")
        missing = expected.difference(dt_index)
        missing_dates = [ts.strftime("%Y-%m-%d") for ts in missing]
    if missing_dates:
        status = "WARN" if allow_missing else "FAIL"
        preview = ", ".join(missing_dates[:5]) + (" ..." if len(missing_dates) > 5 else "")
        rules.append(
            RuleResult(
                name="R4 Calendar",
                status=status,
                detail=f"Missing dates detected ({len(missing_dates)}): {preview}",
                sample_rows=len(missing_dates),
            )
        )
    else:
        rules.append(
            RuleResult(
                name="R4 Calendar",
                status="PASS",
                detail="No missing dates between min and max",
                sample_rows=None,
            )
        )

    # R5 Duplicates
    duplicate_count = int(dt_index.duplicated().sum()) if dt_index is not None else 0
    if duplicate_count > 0:
        rules.append(
            RuleResult(
                name="R5 Duplicates",
                status="FAIL",
                detail=f"Found {duplicate_count} duplicate date rows",
                sample_rows=duplicate_count,
            )
        )
    else:
        rules.append(
            RuleResult(
                name="R5 Duplicates",
                status="PASS",
                detail="No duplicate dates detected",
                sample_rows=None,
            )
        )

    # R6 Range sanity (non-negative)
    numeric_keys = ["revenue", "ad_spend", "orders", "views"]
    negative_columns: Dict[str, int] = {}
    for key in numeric_keys:
        col = mappings.get(key) if isinstance(mappings, dict) else None
        if isinstance(col, str) and col in df.columns:
            negatives = df[col].dropna() < 0
            count = int(negatives.sum())
            if count > 0:
                negative_columns[col] = count
    if negative_columns:
        detail = ", ".join(f"{col} ({count})" for col, count in negative_columns.items())
        rules.append(
            RuleResult(
                name="R6 Range",
                status="FAIL",
                detail=f"Negative values detected in: {detail}",
                sample_rows=sum(negative_columns.values()),
            )
        )
    else:
        rules.append(
            RuleResult(
                name="R6 Range",
                status="PASS",
                detail="Revenue, ad_spend, orders, and views are non-negative",
                sample_rows=None,
            )
        )

    has_fail = any(rule.status == "FAIL" for rule in rules)
    has_warn = any(rule.status == "WARN" for rule in rules)

    if stop_on_fail and has_fail:
        overall_status: RuleStatus = "FAIL"
    elif has_fail or has_warn:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    stats = {
        "rows": int(len(df)),
        "min_date": min_date.strftime("%Y-%m-%d") if isinstance(min_date, pd.Timestamp) and not pd.isna(min_date) else None,
        "max_date": max_date.strftime("%Y-%m-%d") if isinstance(max_date, pd.Timestamp) and not pd.isna(max_date) else None,
        "freshness_days": freshness_days,
        "missing_dates": len(missing_dates),
        "duplicate_dates": duplicate_count,
    }

    return QualityReport(status=overall_status, rules=rules, stats=stats)


def write_quality_artifacts(report: QualityReport, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    report_dict = asdict(report)
    json_path = output_dir / "quality_report.json"
    json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    lines = [
        "# Data Quality Report",
        f"**Status:** {report.status}",
        "",
        "## Summary",
        f"- Rows: {report.stats.get('rows')}",
        f"- Date range: {report.stats.get('min_date')} -> {report.stats.get('max_date')}",
        f"- Freshness threshold (days): {report.stats.get('freshness_days')}",
        f"- Missing dates: {report.stats.get('missing_dates')}",
        f"- Duplicate dates: {report.stats.get('duplicate_dates')}",
        "",
        "## Rules",
    ]
    for rule in report.rules:
        sample = f" (count={rule.sample_rows})" if rule.sample_rows else ""
        lines.append(f"- [{rule.status}] {rule.name}: {rule.detail}{sample}")
    markdown_path = output_dir / "quality_report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(markdown_path)}
