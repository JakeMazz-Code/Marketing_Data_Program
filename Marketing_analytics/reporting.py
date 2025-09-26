"""Reporting helpers for the marketing analytics template."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

import pandas as pd

from Marketing_analytics.config import AnalysisSettings
from Marketing_analytics.data_loader import DatasetBundle
from Marketing_analytics.models import PropensityResults


def _frame_to_json_records(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("", encoding="utf-8")
    else:
        df.to_csv(path, index=False)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def build_summary_payload(
    *,
    settings: AnalysisSettings,
    bundle: DatasetBundle,
    overall: Dict[str, float | int | None],
    campaign: pd.DataFrame,
    channel: pd.DataFrame,
    segment: pd.DataFrame,
    product: pd.DataFrame,
    creative: pd.DataFrame,
    margin: pd.DataFrame,
    timeline: pd.DataFrame,
    customer_value: pd.DataFrame,
    model: PropensityResults | None,
    ai_summary: Optional[Dict[str, object]] = None,
    ai_summary_path: Optional[Path] = None,
) -> Dict:
    payload: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat(),
        "data_path": str(settings.data_path),
        "rows_analyzed": int(len(bundle.frame)),
        "column_mapping": asdict(bundle.mapping),
        "overall_metrics": overall,
        "report_version": "marketing-analytics-template/1.0",
    }

    if not campaign.empty:
        payload["top_campaigns"] = _frame_to_json_records(campaign.head(10))
    if not channel.empty:
        payload["top_channels"] = _frame_to_json_records(channel.head(10))
    if not segment.empty:
        payload["top_segments"] = _frame_to_json_records(segment.head(10))
    if not product.empty:
        payload["top_products"] = _frame_to_json_records(product.head(10))
    if not creative.empty:
        payload["top_creatives"] = _frame_to_json_records(creative.head(10))
    if not margin.empty:
        payload["margin_diagnostics"] = _frame_to_json_records(margin)
    if not timeline.empty:
        payload["timeline"] = _frame_to_json_records(timeline)
    if not customer_value.empty:
        payload["customer_value_distribution"] = json.loads(
            customer_value.describe(include="all").to_json(date_format="iso")
        )

    if model:
        payload["propensity_metrics"] = {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in model.metrics.items()
        }
        payload["propensity_features"] = _frame_to_json_records(model.feature_importance)

    if ai_summary:
        payload["ai_summary"] = {
            "provider": ai_summary.get("provider"),
            "model": ai_summary.get("model"),
            "output_path": ai_summary_path.name if ai_summary_path else None,
            "markdown": ai_summary.get("markdown"),
        }

    return payload


def build_markdown_report(
    *,
    settings: AnalysisSettings,
    overall: Dict[str, float | int | None],
    campaign: pd.DataFrame,
    channel: pd.DataFrame,
    segment: pd.DataFrame,
    product: pd.DataFrame,
    creative: pd.DataFrame,
    margin: pd.DataFrame,
    timeline: pd.DataFrame,
    model: PropensityResults | None,
    ai_summary: Optional[str] = None,
) -> str:
    lines = [
        "# Marketing Performance Summary",
        "",
        f"**Dataset:** `{settings.data_path.name}`",
        f"**Rows analyzed:** {overall.get('rows', 'unknown')}" if "rows" in overall else "",
        "",
        "## Key Metrics",
    ]
    for key, label in [
        ("customer_count", "Customers"),
        ("conversion_count", "Conversions"),
        ("conversion_rate", "Conversion rate"),
        ("total_revenue", "Revenue"),
        ("total_spend", "Spend"),
        ("net_margin", "Net margin"),
        ("refund_amount", "Refunds"),
        ("discount_amount", "Discounts"),
        ("shipping_cost", "Shipping cost"),
        ("units_sold", "Units"),
        ("avg_order_value", "Average order value"),
        ("cost_per_conversion", "Cost per conversion"),
    ]:
        value = overall.get(key)
        if value is None:
            continue
        if key.endswith("rate") or key == "roi" or "rate" in key:
            formatted = f"{value:.2%}"
        elif any(token in key for token in ["value", "revenue", "spend", "cost", "margin", "discount", "refund"]):
            formatted = f"${value:,.2f}"
        else:
            formatted = f"{value:,.0f}"
        lines.append(f"- **{label}:** {formatted}")

    sections = [
        ("## Campaign performance", campaign),
        ("## Channel performance", channel),
        ("## Segment performance", segment),
        ("## Product performance", product),
        ("## Creative performance", creative),
        ("## Post-purchase & margin diagnostics", margin),
    ]
    for title, frame in sections:
        lines.extend(["", title, ""])
        lines.append(dataframe_to_markdown(frame))

    if not timeline.empty:
        lines.extend(["", "## Timeline", ""])
        lines.append(dataframe_to_markdown(timeline))

    if model:
        lines.extend(["", "## Propensity model", ""])
        lines.append("Model trained to predict positive response probability.")
        for key, value in model.metrics.items():
            if isinstance(value, float):
                formatted = f"{value:.4f}" if "rate" in key or "roc" in key or "accuracy" in key else f"{value:.3f}"
            else:
                formatted = str(value)
            lines.append(f"- **{key}:** {formatted}")
        lines.extend(["", "### Top features", ""])
        lines.append(dataframe_to_markdown(model.feature_importance.head(20)))

    if ai_summary:
        lines.extend(["", "## AI-generated insights", "", ai_summary])

    return "\n".join(line for line in lines if line is not None)



