"""Computation of core metrics for marketing analytics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from Marketing_analytics.config import ColumnMapping
from Marketing_analytics.data_loader import DatasetBundle
from Marketing_analytics.metrics_registry import compute_series, get_metric_columns


def _safe_sum(df: pd.DataFrame, column: str | None) -> float | None:
    if not column or column not in df.columns:
        return None
    return float(df[column].fillna(0).sum())


def _safe_mean(df: pd.DataFrame, column: str | None) -> float | None:
    if not column or column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    return float(series.mean())


def _safe_rate(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator / denominator)


def _optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)



def _select_dimension(bundle: DatasetBundle, *candidates: Optional[str]) -> Optional[str]:
    for candidate in candidates:
        if bundle.available(candidate):
            return candidate
    return None


def overall_snapshot(bundle: DatasetBundle) -> Dict[str, float | int | None]:
    df = bundle.frame
    mapping = bundle.mapping

    customer_count = df[mapping.customer_id].nunique()
    campaign_count = df[mapping.campaign].nunique() if bundle.available(mapping.campaign) else None
    channel_count = df[mapping.channel].nunique() if bundle.available(mapping.channel) else None

    conversions = _safe_sum(df, mapping.response)
    total_spend = _safe_sum(df, mapping.spend)
    total_revenue = _safe_sum(df, mapping.revenue)
    total_units = _safe_sum(df, mapping.units)
    total_cogs = _safe_sum(df, mapping.cogs)
    total_refunds = _safe_sum(df, mapping.refund_amount)
    total_discounts = _safe_sum(df, mapping.discount_amount)
    total_shipping = _safe_sum(df, mapping.shipping_cost)

    net_margin = None
    if "net_margin" in df.columns:
        net_margin = float(df["net_margin"].fillna(0).sum())
    elif mapping.gross_margin and mapping.gross_margin in df.columns:
        net_margin = _safe_sum(df, mapping.gross_margin)
    elif total_revenue is not None and total_cogs is not None:
        net_margin = total_revenue - total_cogs

    metrics: Dict[str, float | int | None] = {
        "customer_count": customer_count,
        "campaign_count": campaign_count,
        "channel_count": channel_count,
        "total_spend": total_spend,
        "total_revenue": total_revenue,
        "conversion_count": conversions,
        "units_sold": total_units,
        "net_margin": net_margin,
        "refund_amount": total_refunds,
        "discount_amount": total_discounts,
        "shipping_cost": total_shipping,
    }

    if conversions is not None and customer_count:
        metrics["conversion_rate"] = _safe_rate(conversions, customer_count)

    if total_spend is not None and total_revenue is not None:
        metrics["roi"] = _safe_rate(total_revenue - total_spend, total_spend)

    if total_revenue is not None and conversions not in {None, 0}:
        metrics["avg_order_value"] = _safe_rate(total_revenue, conversions)

    if total_units not in {None, 0} and conversions not in {None, 0}:
        metrics["units_per_conversion"] = _safe_rate(total_units, conversions)

    if total_spend not in {None, 0} and conversions not in {None, 0}:
        metrics["cost_per_conversion"] = _safe_rate(total_spend, conversions)

    if bundle.available(mapping.touch_value):
        metrics["avg_engagement_score"] = _safe_mean(df, mapping.touch_value)

    for flag_field, label in [
        (mapping.return_flag, "return_count"),
        (mapping.exchange_flag, "exchange_count"),
        (mapping.defect_flag, "defect_count"),
    ]:
        if flag_field and flag_field in df.columns:
            metrics[label] = float(df[flag_field].fillna(0).sum())

    try:
        metric_cols = get_metric_columns()
        required_cols = [metric_cols.revenue, metric_cols.ad_spend, metric_cols.orders]
        if all(bundle.available(col) for col in required_cols):
            aggregate_df = pd.DataFrame({
                metric_cols.revenue: [df[metric_cols.revenue].fillna(0).sum()],
                metric_cols.ad_spend: [df[metric_cols.ad_spend].fillna(0).sum()],
                metric_cols.orders: [df[metric_cols.orders].fillna(0).sum()],
            })
            ratios = compute_series(aggregate_df, ['MER', 'ROAS', 'CAC', 'AOV']).iloc[0]
            metrics['mer'] = _optional_float(ratios['MER'])
            metrics['roas'] = _optional_float(ratios['ROAS'])
            metrics['cac'] = _optional_float(ratios['CAC'])
            metrics['aov'] = _optional_float(ratios['AOV'])
    except (FileNotFoundError, KeyError):
        pass

    return metrics


def _group_summary(
    df: pd.DataFrame,
    by: str,
    mapping: ColumnMapping,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    group = df.groupby(by, dropna=False)
    summary = pd.DataFrame(index=group.size().index)
    summary["records"] = group.size()

    if mapping.response and mapping.response in df.columns:
        conversions = group[mapping.response].sum(min_count=1)
        audience = group[mapping.response].count()
        summary["conversion_count"] = conversions
        summary["conversion_rate"] = conversions / audience.replace(0, np.nan)

    if mapping.revenue and mapping.revenue in df.columns:
        summary["revenue"] = group[mapping.revenue].sum(min_count=1)
    if mapping.spend and mapping.spend in df.columns:
        summary["spend"] = group[mapping.spend].sum(min_count=1)
    if mapping.touch_value and mapping.touch_value in df.columns:
        summary["avg_engagement"] = group[mapping.touch_value].mean()
    if mapping.units and mapping.units in df.columns:
        summary["units"] = group[mapping.units].sum(min_count=1)
    if mapping.discount_amount and mapping.discount_amount in df.columns:
        summary["discount_amount"] = group[mapping.discount_amount].sum(min_count=1)
    if mapping.shipping_cost and mapping.shipping_cost in df.columns:
        summary["shipping_cost"] = group[mapping.shipping_cost].sum(min_count=1)
    if mapping.refund_amount and mapping.refund_amount in df.columns:
        summary["refund_amount"] = group[mapping.refund_amount].sum(min_count=1)
    if mapping.cogs and mapping.cogs in df.columns:
        summary["cogs"] = group[mapping.cogs].sum(min_count=1)
    if "net_margin" in df.columns:
        summary["net_margin"] = group["net_margin"].sum(min_count=1)
    elif mapping.gross_margin and mapping.gross_margin in df.columns:
        summary["net_margin"] = group[mapping.gross_margin].sum(min_count=1)

    if "revenue" in summary.columns and "spend" in summary.columns:
        spend = summary["spend"].replace(0, np.nan)
        summary["roi"] = (summary["revenue"] - summary["spend"]) / spend

    summary = summary.reset_index().rename(columns={by: "dimension"})
    if top_n:
        for priority in ("revenue", "net_margin", "conversion_count", "records"):
            if priority in summary.columns:
                summary = summary.sort_values(priority, ascending=False).head(top_n)
                break
    return summary


def campaign_performance(bundle: DatasetBundle, *, top_n: int | None = None) -> pd.DataFrame:
    if not bundle.available(bundle.mapping.campaign):
        return pd.DataFrame()
    return _group_summary(bundle.frame, bundle.mapping.campaign, bundle.mapping, top_n=top_n)


def channel_performance(bundle: DatasetBundle, *, top_n: int | None = None) -> pd.DataFrame:
    if not bundle.available(bundle.mapping.channel):
        return pd.DataFrame()
    return _group_summary(bundle.frame, bundle.mapping.channel, bundle.mapping, top_n=top_n)


def segment_performance(bundle: DatasetBundle, *, top_n: int | None = None) -> pd.DataFrame:
    if not bundle.available(bundle.mapping.segment):
        return pd.DataFrame()
    return _group_summary(bundle.frame, bundle.mapping.segment, bundle.mapping, top_n=top_n)


def product_performance(bundle: DatasetBundle, *, top_n: int | None = None) -> pd.DataFrame:
    dimension = _select_dimension(bundle, bundle.mapping.product_sku, bundle.mapping.product_name)
    if not dimension:
        return pd.DataFrame()
    return _group_summary(bundle.frame, dimension, bundle.mapping, top_n=top_n)


def creative_performance(bundle: DatasetBundle, *, top_n: int | None = None) -> pd.DataFrame:
    dimension = _select_dimension(bundle, bundle.mapping.creative_id, bundle.mapping.creative_name)
    if not dimension:
        return pd.DataFrame()
    return _group_summary(bundle.frame, dimension, bundle.mapping, top_n=top_n)


def time_series_summary(bundle: DatasetBundle, *, freq: str) -> pd.DataFrame:
    mapping = bundle.mapping
    if not mapping.event_timestamp or mapping.event_timestamp not in bundle.frame.columns:
        return pd.DataFrame()

    df = bundle.frame.set_index(mapping.event_timestamp)
    resampler = df.resample(freq)

    frame = pd.DataFrame(index=resampler.size().index)

    if mapping.response and mapping.response in df.columns:
        frame["conversions"] = resampler[mapping.response].sum()
    if mapping.revenue and mapping.revenue in df.columns:
        frame["revenue"] = resampler[mapping.revenue].sum()
    if mapping.spend and mapping.spend in df.columns:
        frame["spend"] = resampler[mapping.spend].sum()
    if mapping.units and mapping.units in df.columns:
        frame["units"] = resampler[mapping.units].sum()
    if "net_margin" in df.columns:
        frame["net_margin"] = resampler["net_margin"].sum()

    frame["unique_customers"] = resampler[mapping.customer_id].nunique()
    frame = frame.reset_index().rename(columns={mapping.event_timestamp: "period"})

    if mapping.response and mapping.response in df.columns:
        denominator = resampler[mapping.customer_id].nunique().replace(0, np.nan)
        frame["conversion_rate"] = frame["conversions"].values / denominator.values

    return frame


def customer_value_distribution(bundle: DatasetBundle) -> pd.DataFrame:
    mapping = bundle.mapping
    df = bundle.frame

    value_columns: list[str] = []
    for candidate in [
        mapping.revenue,
        mapping.order_value,
        mapping.touch_value,
        mapping.units,
        mapping.cogs,
        mapping.discount_amount,
        mapping.shipping_cost,
        mapping.refund_amount,
    ]:
        if candidate and candidate in df.columns:
            value_columns.append(candidate)
    if "net_margin" in df.columns:
        value_columns.append("net_margin")
    if mapping.gross_margin and mapping.gross_margin in df.columns:
        value_columns.append(mapping.gross_margin)

    value_columns = list(dict.fromkeys(value_columns))
    if not value_columns:
        return pd.DataFrame()

    agg = df.groupby(mapping.customer_id)[value_columns].agg(["mean", "sum", "count"]).reset_index()
    agg.columns = ["_".join(filter(None, col)).rstrip("_") for col in agg.columns.to_flat_index()]
    return agg


def margin_diagnostics(bundle: DatasetBundle) -> pd.DataFrame:
    mapping = bundle.mapping
    df = bundle.frame
    issues = []
    total_records = len(df)

    def _sum_if_available(frame: pd.DataFrame, column: Optional[str]) -> float | None:
        if not column:
            return None
        if column in frame.columns:
            return float(frame[column].fillna(0).sum())
        return None

    for column, label in [
        (mapping.return_flag, "returns"),
        (mapping.exchange_flag, "exchanges"),
        (mapping.defect_flag, "defects"),
    ]:
        if not column or column not in df.columns:
            continue
        flagged = df[df[column].fillna(0) > 0]
        if flagged.empty:
            continue
        record_count = int(len(flagged))
        issue: Dict[str, object] = {
            "issue": label,
            "records": record_count,
            "share_of_orders": record_count / total_records if total_records else None,
            "revenue": _sum_if_available(flagged, mapping.revenue),
            "spend": _sum_if_available(flagged, mapping.spend),
            "units": _sum_if_available(flagged, mapping.units),
            "refund_amount": _sum_if_available(flagged, mapping.refund_amount),
            "discount_amount": _sum_if_available(flagged, mapping.discount_amount),
            "shipping_cost": _sum_if_available(flagged, mapping.shipping_cost),
            "cogs": _sum_if_available(flagged, mapping.cogs),
        }
        if "net_margin" in flagged.columns:
            issue["net_margin"] = float(flagged["net_margin"].fillna(0).sum())
        elif mapping.gross_margin and mapping.gross_margin in flagged.columns:
            issue["net_margin"] = _sum_if_available(flagged, mapping.gross_margin)
        issues.append(issue)

    if not issues:
        return pd.DataFrame()

    return pd.DataFrame(issues).sort_values("records", ascending=False)


