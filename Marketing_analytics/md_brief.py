"""Markdown-only brief construction helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from Marketing_analytics.etl_text_clean import normalize_product_title
_TEXT_SYNONYMS: Dict[str, set[str]] = {
    "channel": {"channel", "platform", "source"},
    "campaign": {"campaign", "campaign_name"},
    "adset": {"adset", "adset_name"},
    "sku": {"sku", "product_sku", "lineitem_sku", "product_id"},
    "product_title": {"product_title", "lineitem_title", "product_name", "title"},
    "creative_id": {"creative_id", "post_id", "ad_id", "ig_id"},
}

_NUMERIC_SYNONYMS: Dict[str, set[str]] = {
    "revenue": {"revenue", "purchase_value", "total_made", "conversion_value"},
    "ad_spend": {"ad_spend", "spend", "cost"},
    "orders": {"orders", "purchases", "conversions"},
    "views": {"video_views", "views", "impressions"},
    "units": {"units", "quantity"},
    "clicks": {"clicks", "link_clicks"},
}

_TOF_KEYWORDS = {"prospecting", "broad", "cold", "lookalike", "laa", "new"}
_BOF_KEYWORDS = {"retarget", "remarket", "bof", "viewed", "cart", "purchase"}
_OWNED_KEYWORDS = {"email", "sms", "newsletter", "crm", "loyalty", "owned", "retention", "vip"}


@dataclass(slots=True)
class MDOnlyTemplateResult:
    """Container for MD-only brief scaffolding."""

    template: str
    missing_sections: List[str]
    metrics: Dict[str, Any]
    latest_date: Optional[datetime]


# ----------------------------
# Generic helpers
# ----------------------------


def _clean_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def _safe_sum(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    total = series.sum(min_count=1)
    return _clean_float(total)


def _safe_div(
    numerator: Optional[float],
    denominator: Optional[float],
    *,
    allow_inf: bool = False,
) -> Optional[float]:
    num = _clean_float(numerator)
    den = _clean_float(denominator)
    if num is None or den is None:
        return None
    if den == 0:
        if allow_inf and num > 0:
            return math.inf
        return None
    return num / den


def _format_currency(value: Optional[float], *, decimals: int = 2) -> str:
    num = _clean_float(value)
    if num is None or math.isinf(num):
        return "n/a"
    return f"${num:,.{decimals}f}"


def _format_ratio(value: Optional[float]) -> str:
    num = _clean_float(value)
    if num is None or math.isinf(num):
        return "n/a"
    return f"{num:,.2f}x"


def _format_number(value: Optional[float], *, decimals: int = 0) -> str:
    num = _clean_float(value)
    if num is None:
        return "n/a"
    return f"{num:,.{decimals}f}"


def _format_percent(value: Optional[float]) -> str:
    num = _clean_float(value)
    if num is None or math.isinf(num):
        return "n/a"
    return f"{num * 100:.1f}%"


def _markdown_table(headers: Iterable[str], rows: List[List[str]]) -> str:
    header_line = " | ".join(headers)
    separator = " | ".join(["---" for _ in headers])
    body = [" | ".join(row) for row in rows]
    return "\n".join([header_line, separator, *body])


def _build_anomaly_section(anomalies: Any, notes: Any) -> List[str]:
    lines: List[str] = []
    if isinstance(anomalies, list) and anomalies:
        sorted_anomalies = sorted(
            anomalies, key=lambda item: abs(float(item.get('peak_z', 0.0) or 0.0)), reverse=True
        )[:5]
        for entry in sorted_anomalies:
            metric = str(entry.get('metric', 'metric')).replace('_', ' ').title()
            start = entry.get('start_date') or entry.get('date') or entry.get('start')
            end = entry.get('end_date') or entry.get('end')
            if start and end and start != end:
                date_label = f"{start} to {end}"
            else:
                date_label = start or 'unknown date'
            peak_z = entry.get('peak_z')
            if peak_z is None:
                detail = f"- {metric} anomaly around {date_label}."
            else:
                try:
                    detail = f"- {metric} anomaly around {date_label} (|z|={float(peak_z):.2f})."
                except (TypeError, ValueError):
                    detail = f"- {metric} anomaly around {date_label}."
            lines.append(detail)
    else:
        lines.append('- No anomalies triggered in the last 28 days.')

    if isinstance(notes, (list, tuple)):
        for note in notes:
            if note:
                lines.append(f"- {str(note)}")
    elif isinstance(notes, str) and notes.strip():
        lines.append(notes.strip())

    return lines


def _build_quality_section(report: Any) -> List[str]:
    if not isinstance(report, dict):
        return ['- Quality report unavailable.']

    status = str(report.get('status', 'UNKNOWN')).upper()
    rules = report.get('rules')
    lines: List[str] = []

    if status == 'PASS':
        lines.append('- Data checks: PASS (fresh + schema aligned).')
    else:
        lines.append(f"- Data checks: {status}.")

    if isinstance(rules, list) and rules:
        for rule in rules[:5]:
            rule_status = str(rule.get('status', '')).upper()
            rule_name = rule.get('name') or rule.get('id') or 'check'
            detail = rule.get('detail') or rule.get('message')
            if rule_status == 'PASS':
                continue
            message = f"   - {rule_name}: {rule_status}"
            if detail:
                message += f" - {detail}"
            lines.append(message)
    elif not lines:
        lines.append('- No quality rules reported.')

    return lines


def _build_kpi_table(windows: Dict[int, Dict[str, Optional[float]]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for horizon in (7, 28, 90):
        metrics = windows.get(horizon, {})
        rows.append([
            f"{horizon}-day",
            _format_currency(metrics.get('revenue_sum')),
            _format_currency(metrics.get('ad_spend_sum')),
            _format_number(metrics.get('orders_sum'), decimals=0),
            _format_ratio(metrics.get('mer')),
            _format_currency(metrics.get('cac')),
            _format_currency(metrics.get('aov')),
        ])
    return rows


def _window_slice(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty or days <= 0:
        return df.iloc[0:0]
    return df.tail(days)


def _filter_date_window(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["date"] >= start
    if end is not None:
        mask &= df["date"] <= end
    return df.loc[mask].copy()


def _compute_window_metrics(df: pd.DataFrame, window: int) -> Dict[str, Optional[float]]:
    slice_df = _window_slice(df, window)
    if slice_df.empty:
        return {
            "revenue_sum": None,
            "spend_sum": None,
            "orders_sum": None,
            "views_sum": None,
            "mer": None,
            "cac": None,
            "aov": None,
            "days": 0,
            "daily_revenue": None,
        }
    revenue = _safe_sum(slice_df.get("revenue"))
    spend = _safe_sum(slice_df.get("ad_spend"))
    orders = _safe_sum(slice_df.get("orders"))
    views = _safe_sum(slice_df.get("views"))
    days_available = slice_df.shape[0]
    daily_revenue = _safe_div(revenue, days_available)
    return {
        "revenue_sum": revenue,
        "spend_sum": spend,
        "orders_sum": orders,
        "views_sum": views,
        "mer": _safe_div(revenue, spend, allow_inf=True),
        "cac": _safe_div(spend, orders),
        "aov": _safe_div(revenue, orders),
        "days": days_available,
        "daily_revenue": daily_revenue,
    }


def _compute_revenue_slope(df: pd.DataFrame) -> Optional[float]:
    if "revenue" not in df.columns:
        return None
    working = df[["date", "revenue"]].dropna()
    working = working.sort_values("date")
    if working.shape[0] < 2:
        return None
    lookback = min(56, working.shape[0])
    working = working.tail(lookback)
    days = (working["date"] - working["date"].min()).dt.days.astype(float)
    revenue = working["revenue"].astype(float)
    if np.allclose(revenue, revenue.iloc[0]):
        return 0.0
    slope, _intercept = np.polyfit(days.to_numpy(), revenue.to_numpy(), 1)
    return float(slope)


def _summarise_margin(df: pd.DataFrame, window: int) -> Dict[str, Any]:
    window_df = _window_slice(df, window)
    revenue_sum = _safe_sum(window_df.get("revenue"))
    components: List[Dict[str, Any]] = []
    for label in ["discounts", "returns", "shipping", "duties", "taxes"]:
        if label not in df.columns:
            continue
        series = window_df[label]
        total = _safe_sum(series)
        if total is None:
            continue
        share = None
        if revenue_sum and revenue_sum > 0:
            share = total / revenue_sum
        top_day = None
        if not series.dropna().empty:
            idx = series.idxmax()
            if idx in df.index:
                top_day = {
                    "date": df.loc[idx, "date"].strftime("%Y-%m-%d"),
                    "value": _clean_float(series.loc[idx]),
                }
        components.append({
            "label": label,
            "total": total,
            "share": share,
            "top_day": top_day,
        })
    components.sort(key=lambda item: abs(item["total"]), reverse=True)
    leading = components[0] if components else None
    return {
        "revenue_sum": revenue_sum,
        "components": components,
        "leading_component": leading,
    }


def _ensure_synonym_columns(df: pd.DataFrame) -> None:
    lower_lookup = {str(col).lower(): col for col in df.columns}

    for target, options in {**_TEXT_SYNONYMS, **_NUMERIC_SYNONYMS}.items():
        if target in df.columns:
            continue
        for option in options:
            match = lower_lookup.get(option)
            if match is not None:
                df[target] = df[match]
                break

    for target in _NUMERIC_SYNONYMS:
        if target in df.columns:
            df[target] = pd.to_numeric(df[target], errors="coerce")

    for text_col in ("channel", "campaign", "adset", "sku", "product_title", "creative_id"):
        if text_col in df.columns:
            df[text_col] = (
                df[text_col]
                .astype(str)
                .replace({"nan": "", "None": ""})
                .str.strip()
            )


def _normalize_channel(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("meta", "facebook", "instagram", "ig")):
        return "Meta"
    if "tiktok" in lowered:
        return "TikTok"
    if any(keyword in lowered for keyword in ("email", "sms", "klaviyo", "mailchimp")):
        return "Email/SMS"
    if any(keyword in lowered for keyword in ("organic", "seo", "direct", "referral", "blog")):
        return "Organic"
    return "Other"


def _segment_label(campaign: str, adset: str) -> str:
    combined = f"{campaign or ''} {adset or ''}".lower()
    tokens = {token for token in re.split(r'[^a-z0-9]+', combined) if token}
    has_tof = any(token in tokens for token in _TOF_KEYWORDS)
    has_bof = any(token in tokens for token in _BOF_KEYWORDS)
    has_owned = any((keyword in tokens) or (keyword in combined) for keyword in _OWNED_KEYWORDS)
    if has_tof and not has_bof:
        return "Prospecting / TOF"
    if has_bof and not has_tof:
        return "Retargeting / BOF"
    if has_tof and has_bof:
        return "Full Funnel / Mixed"
    if has_owned:
        return "Owned / CRM"
    if tokens and any(token in {"brand", "awareness", "upper"} for token in tokens):
        return "Prospecting / TOF"
    if tokens and any(token in {"loyalty", "vip", "winback"} for token in tokens):
        return "Owned / CRM"
    return "Needs Naming Hygiene"


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "n/a"


def _aggregate_channels(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "channel" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    for col in ["revenue", "ad_spend", "orders", "views", "clicks"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    working["channel"] = working["channel"].apply(_normalize_channel)
    working = working[working["channel"].notna()]
    if working.empty:
        return pd.DataFrame()
    agg_cols = {col: "sum" for col in ["revenue", "ad_spend", "orders", "views", "clicks"] if col in working.columns}
    summary = working.groupby("channel", dropna=False).agg(agg_cols).reset_index()
    return summary.fillna(0.0)



def _aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    working = df.copy()
    for col in ["revenue", "ad_spend", "orders"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    if "campaign" not in working.columns and "adset" not in working.columns:
        return pd.DataFrame()
    working["campaign"] = working.get("campaign", "").fillna("")
    working["adset"] = working.get("adset", "").fillna("")
    working = working[~((working["campaign"].str.strip() == "") & (working["adset"].str.strip() == ""))]
    if working.empty:
        return pd.DataFrame()
    working["__segment"] = working.apply(
        lambda row: _segment_label(str(row.get("campaign", "")), str(row.get("adset", ""))),
        axis=1,
    )
    agg_cols = {col: "sum" for col in ["revenue", "ad_spend", "orders"] if col in working.columns}
    summary = working.groupby("__segment", dropna=False).agg(agg_cols).reset_index()
    return summary.fillna(0.0)




def _has_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()



def _channel_efficiency_section(
    channel_df: pd.DataFrame,
    efficiency_payload: Optional[Dict[str, Any]],
    window_28: pd.DataFrame,
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    explainer = (
        "MER (same as ROAS here) = Revenue / Ad Spend; CAC = Spend / Orders; Revenue Share shows each channel's contribution over the last 28 days."
    )

    rows: List[List[str]] = []
    note_lines: List[str] = []
    top_channel: Optional[Dict[str, Any]] = None
    channel_df_missing = channel_df is None or channel_df.empty

    data_source: List[Dict[str, Any]] = []
    scope_note = None
    if efficiency_payload and isinstance(efficiency_payload, dict):
        candidates = efficiency_payload.get('by_channel')
        if isinstance(candidates, list) and candidates:
            data_source = candidates
            scope_note = efficiency_payload.get('scope')

    if data_source:
        total_revenue = sum(float(item.get('revenue') or 0.0) for item in data_source)
        for item in data_source:
            channel_name = str(item.get('channel', 'Unknown') or 'Unknown')
            revenue = float(item.get('revenue') or 0.0)
            spend = float(item.get('spend') or 0.0)
            orders = float(item.get('orders') or 0.0)
            roas = revenue / spend if spend > 0 else None
            cac = spend / orders if orders > 0 else None
            share = revenue / total_revenue if total_revenue else None
            rows.append([
                channel_name,
                _format_currency(revenue),
                _format_currency(spend),
                _format_number(orders, decimals=0),
                _format_ratio(roas),
                _format_currency(cac),
                _format_percent(share),
            ])
        leading = max(data_source, key=lambda item: float(item.get('revenue') or 0.0))
        top_channel = {
            'name': str(leading.get('channel', 'Unknown') or 'Unknown'),
            'revenue': float(leading.get('revenue') or 0.0),
            'roas': (float(leading.get('revenue') or 0.0) / float(leading.get('spend') or 1.0)) if float(leading.get('spend') or 0.0) > 0 else None,
            'share': (float(leading.get('revenue') or 0.0) / total_revenue) if total_revenue else None,
        }
        site_revenue = window_28['revenue'].sum(min_count=1) if 'revenue' in window_28.columns else np.nan
        site_spend = window_28['ad_spend'].sum(min_count=1) if 'ad_spend' in window_28.columns else np.nan
        if total_revenue and site_revenue and site_revenue > 0:
            divergence = abs(total_revenue - site_revenue) / site_revenue
            if divergence > 0.25 or (total_revenue >= site_revenue and all(float(item.get('spend') or 0.0) <= 0 for item in data_source)):
                note_lines.append('Channel scope appears to differ from sitewide totals; values shown are paid-only / incomplete spend.')
    else:
        summary = _aggregate_channels(channel_df)
        if summary.empty or 'revenue' not in summary.columns or 'ad_spend' not in summary.columns:
            message = f"{explainer}\nNot available in artifacts. (missing: channel/ad_spend/orders)"
            return message, True, None
        total_revenue = float(summary['revenue'].sum())
        ordered = summary.sort_values('revenue', ascending=False).reset_index(drop=True)
        for _, row in ordered.iterrows():
            channel = str(row.get('channel', 'Other') or 'Other')
            revenue = float(row.get('revenue', 0.0) or 0.0)
            spend = float(row.get('ad_spend', 0.0) or 0.0)
            orders_val = float(row.get('orders', 0.0) or 0.0) if 'orders' in summary.columns else None
            roas = _safe_div(revenue, spend, allow_inf=True)
            cac = _safe_div(spend, orders_val) if orders_val is not None else None
            share = revenue / total_revenue if total_revenue else None
            rows.append([
                channel,
                _format_currency(revenue),
                _format_currency(spend),
                _format_number(orders_val, decimals=0) if orders_val is not None else 'n/a',
                _format_ratio(roas),
                _format_currency(cac),
                _format_percent(share),
            ])
        lead = ordered.iloc[0]
        top_channel = {
            'name': str(lead.get('channel', 'Other') or 'Other'),
            'revenue': float(lead.get('revenue', 0.0) or 0.0),
            'share': (float(lead.get('revenue', 0.0) or 0.0) / total_revenue) if total_revenue else None,
            'roas': _safe_div(float(lead.get('revenue', 0.0) or 0.0), float(lead.get('ad_spend', 0.0) or 0.0), allow_inf=True),
        }

    table = _markdown_table(
        ["Channel", "Revenue", "Spend", "Orders", "ROAS", "CAC", "Revenue Share"],
        rows,
    ) if rows else None

    body_lines = [explainer]
    if table:
        body_lines.append(table)
        if channel_df_missing:
            body_lines.append('Not available in artifacts. (missing: channel/ad_spend/orders)')
    else:
        body_lines.append('Not available in artifacts. (missing: channel/ad_spend/orders)')
    if scope_note:
        body_lines.append(f"Scope noted as {scope_note}.")
    if note_lines:
        body_lines.extend(note_lines)
    elif scope_note == 'paid':
        body_lines.append('- Paid channels only.')

    return "\n".join(body_lines), not bool(rows), top_channel
def _customer_segments_section(campaign_df: pd.DataFrame) -> Tuple[str, bool]:
    explainer = (
        "Segments are inferred from campaign/ad set names: Prospecting/TOF (cold audiences) versus Retargeting/BOF (warm audiences)."
    )
    summary = _aggregate_segments(campaign_df)
    if summary.empty or "revenue" not in summary.columns or "ad_spend" not in summary.columns:
        return f"{explainer}\nNot available in artifacts. (missing: campaign/adset text)", True

    total_revenue = float(summary["revenue"].sum()) if "revenue" in summary.columns else 0.0

    lines: List[str] = [explainer]
    for _, row in summary.sort_values("revenue", ascending=False).iterrows():
        segment = row["__segment"]
        revenue = float(row.get("revenue", 0.0) or 0.0)
        spend = float(row.get("ad_spend", 0.0) or 0.0)
        orders_val = float(row.get("orders", 0.0) or 0.0) if "orders" in summary.columns else None
        share = revenue / total_revenue if total_revenue else None
        roas = _safe_div(revenue, spend, allow_inf=True)
        cac = _safe_div(spend, orders_val) if orders_val is not None else None

        lines.append(
            f"- {segment}: Revenue {_format_currency(revenue)} ({_format_percent(share)} share), MER {_format_ratio(roas)}, CAC {_format_currency(cac)}."
        )

        if segment == "Prospecting / TOF":
            lines.append(
                "   - Growth move: keep cold spend flowing where MER holds above baseline; watch CAC drift on broad audiences."
            )
            lines.append(
                "   - Next step: recycle TOF winners into fresh BOF journeys within 48 hours."
            )
        elif segment == "Retargeting / BOF":
            lines.append(
                "   - Growth move: fuel BOF with the latest TOF creative so CAC stays efficient."
            )
            lines.append(
                "   - Next step: use BOF to push post-purchase offers and subscription add-ons."
            )
        else:
            lines.append(
                "   - Growth move: tag campaigns with TOF/BOF keywords to expose funnel performance."
            )
            lines.append(
                "   - Next step: split mixed audiences by intent (viewed vs purchased) to rebuild clarity."
            )

    return "\n".join(lines), False





def _hero_skus_section(
    sku_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    seasonality: float,
    safety_stock: float,
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    explainer = (
        f"Inventory plan = run rate (units / 28 days) x 45-day horizon x seasonality {seasonality:.2f} x safety stock {safety_stock:.2f}."
    )
    if sku_df is None or sku_df.empty:
        return (f"{explainer}\nNot available in artifacts. (missing: sku/product_title metrics)", True, None)

    working = sku_df.copy()
    if 'date' not in working.columns:
        return (f"{explainer}\nNot available in artifacts. (missing: date column)", True, None)
    working['date'] = pd.to_datetime(working['date'], errors='coerce')
    working = working.dropna(subset=['date'])
    if working.empty:
        return (f"{explainer}\nNot available in artifacts. (missing: valid dates)", True, None)

    if 'product_title' in working.columns:
        working['product_title'] = working['product_title'].apply(normalize_product_title)
    else:
        working['product_title'] = None

    latest_date = working['date'].max()
    start_28 = latest_date - pd.Timedelta(days=27)
    recent = working[working['date'] >= start_28].copy()
    if recent.empty:
        last_label = latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else str(latest_date)
        msg = f"{explainer}\nNot available in artifacts. (stale: no SKU rows in last 28 days; latest {last_label})"
        return (msg, True, None)

    for col in ['revenue', 'units', 'orders']:
        if col in recent.columns:
            recent[col] = pd.to_numeric(recent[col], errors='coerce')

    units_source = None
    if 'units' in recent.columns and recent['units'].fillna(0).sum() > 0:
        units_source = 'units'
        units_column = 'units'
    elif 'orders' in recent.columns and recent['orders'].fillna(0).sum() > 0:
        units_source = 'orders'
        recent['units_proxy'] = recent['orders']
        units_column = 'units_proxy'
    else:
        units_column = None

    # Use only sku_code for labeling; if missing, rely on product_title only
    recent['sku_value'] = recent.get('sku_code')

    def _label_from_row(row: pd.Series) -> str:
        title_val = normalize_product_title(row.get('product_title', '') or '')
        sku_val = row.get('sku_value')
        if isinstance(sku_val, (float, np.floating)) and pd.isna(sku_val):
            sku_val = None
        if isinstance(sku_val, str):
            sku_val = sku_val.strip()
        if sku_val:
            return f"{sku_val} — {title_val}" if title_val else sku_val
        return title_val or 'Top SKU'

    recent['__label'] = recent.apply(_label_from_row, axis=1)

    agg_map: Dict[str, str] = {'revenue': 'sum'}
    if units_column:
        agg_map[units_column] = 'sum'
    grouped = recent.groupby('__label', as_index=False).agg(agg_map)
    grouped['sku_code'] = recent.groupby('__label')['sku_value'].first().values
    grouped['product_title'] = recent.groupby('__label')['product_title'].first().values

    total_revenue = grouped['revenue'].sum()
    grouped['revenue_share'] = grouped['revenue'] / total_revenue if total_revenue else np.nan

    if units_column and grouped[units_column].notna().any():
        total_units = grouped[units_column].sum()
        grouped['units_total'] = grouped[units_column]
        grouped['units_share'] = grouped[units_column] / total_units if total_units else np.nan
        grouped['run_rate'] = grouped[units_column] / 28.0
        grouped['q4_buy'] = grouped['run_rate'].apply(
            lambda val: math.ceil(val * 45.0 * seasonality * safety_stock) if pd.notna(val) and val > 0 else None
        )
    else:
        grouped['units_total'] = np.nan
        grouped['units_share'] = np.nan
        grouped['run_rate'] = np.nan
        grouped['q4_buy'] = None

    if units_column:
        grouped = grouped.sort_values([units_column, 'revenue'], ascending=[False, False])
    else:
        grouped = grouped.sort_values('revenue', ascending=False)

    display_df = grouped.head(5).copy()

    rows: List[List[str]] = []
    top_snapshot: Optional[Dict[str, Any]] = None

    inventory_lookup: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    if inventory_df is not None and not inventory_df.empty:
        inv = inventory_df.copy()
        inv['date'] = pd.to_datetime(inv.get('date'), errors='coerce')
        inv = inv.dropna(subset=['date'])
        if not inv.empty:
            latest_inv = inv[inv['date'] == inv['date'].max()]
            for _, row in latest_inv.iterrows():
                sku_val = str(row.get('sku', '') or '').strip()
                title_val = normalize_product_title(row.get('product_title', '') or '')
                inventory_lookup[(sku_val, title_val)] = {
                    'on_hand': _clean_float(row.get('on_hand')),
                    'on_order': _clean_float(row.get('on_order')),
                    'lead_time': _clean_float(row.get('lead_time_days')),
                }

    for _, row in display_df.iterrows():
        label = row['__label']
        sku_val = str(row.get('sku_code') or '').strip()
        title_val = normalize_product_title(row.get('product_title') or '')
        revenue = float(row.get('revenue') or 0.0)
        revenue_share = row.get('revenue_share')
        units_total = row.get('units_total') if units_column else None
        units_share = row.get('units_share') if units_column else None
        run_rate = row.get('run_rate') if units_column else None
        recommended = row.get('q4_buy') if units_column else None

        inv_info = (
            inventory_lookup.get((sku_val, title_val))
            or inventory_lookup.get((sku_val, ''))
            or inventory_lookup.get(('', title_val))
        )
        inv_note = ''
        if inv_info:
            parts: List[str] = []
            on_hand = inv_info.get('on_hand')
            on_order = inv_info.get('on_order')
            if on_hand is not None:
                parts.append(f"on hand {int(on_hand):,}")
            if on_order is not None and on_order > 0:
                parts.append(f"on order {int(on_order):,}")
            if parts:
                inv_note = ' (' + '; '.join(parts) + ')'

        units_display = _format_number(units_total, decimals=0) if units_total is not None else 'n/a'
        if units_source == 'orders' and units_total is not None:
            units_display = f"{units_display} (orders proxy)"
        revenue_share_display = _format_percent(revenue_share)
        units_share_display = _format_percent(units_share) if units_column else 'n/a'
        run_rate_display = _format_number(run_rate, decimals=2) if units_column else 'n/a'
        recommended_display = (
            f"{_format_number(recommended, decimals=0)}{inv_note}" if recommended else f"Indicative only{inv_note}"
        )

        rows.append([
            label,
            _format_currency(revenue),
            units_display,
            revenue_share_display,
            units_share_display,
            run_rate_display,
            recommended_display,
        ])

        if top_snapshot is None:
            top_snapshot = {
                'name': label,
                'revenue': revenue,
                'share': revenue_share,
                'run_rate': run_rate,
                'recommended': recommended,
                'units_source': units_source or 'none',
            }

    if not rows:
        return (f"{explainer}\nNot available in artifacts. (missing: sku/product_title metrics)", True, None)

    table = _markdown_table(
        [
            'SKU / Product',
            'Revenue (28d)',
            'Units (28d)',
            'Revenue Share',
            'Units Share',
            'Run Rate / day',
            'Rec. 45d Units',
        ],
        rows,
    )

    body_lines = [
        '## Hero SKUs & Q4/Q1 Inventory Plan',
        explainer,
        table,
    ]

    if units_source is None:
        body_lines.append('- Unit counts unavailable; recommendations shown as indicative.')
    elif units_source == 'orders':
        body_lines.append('- Unit counts derived from orders (proxy). Validate with operations before finalising inventory buys.')

    if recent.get('sku_code') is not None and recent['sku_code'].isna().all():
        body_lines.append('- Title-based grouping (SKU code not supplied).')

    return "\n".join(body_lines), False, top_snapshot

def _creative_performance_section(
    creative_df: pd.DataFrame,
    campaign_df: pd.DataFrame,
    efficiency_payload: Optional[Dict[str, Any]],
) -> Tuple[str, bool]:
    explainer = (
        "RPM = Revenue / Views x 1,000; ROAS = Revenue / Ad Spend; CTR = Clicks / Views. We highlight the top assets over the last 28 days."
    )

    rows: List[List[str]] = []
    data_source: List[Dict[str, Any]] = []
    if efficiency_payload and isinstance(efficiency_payload, dict):
        candidates = efficiency_payload.get('by_creative')
        if isinstance(candidates, list) and candidates:
            data_source = candidates

    if data_source:
        scored: List[Dict[str, Any]] = []
        for item in data_source:
            revenue = float(item.get('revenue') or 0.0)
            spend = float(item.get('spend') or 0.0)
            views = float(item.get('views') or 0.0)
            clicks = float(item.get('clicks') or 0.0)
            rpm = (revenue / views) * 1000 if views > 0 else None
            roas = revenue / spend if spend > 0 else None
            ctr = clicks / views if views > 0 and clicks > 0 else (0.0 if clicks == 0 and views > 0 else None)
            scored.append({
                'creative_id': str(item.get('creative_id', 'unknown') or 'unknown'),
                'channel': str(item.get('channel', 'Unknown') or 'Unknown'),
                'revenue': revenue,
                'rpm': rpm,
                'roas': roas,
                'ctr': ctr,
            })
        key = 'revenue' if any(entry['revenue'] for entry in scored) else 'rpm'
        scored.sort(key=lambda entry: entry.get(key) or 0.0, reverse=True)
        for entry in scored[:5]:
            rows.append([
                entry['creative_id'],
                entry['channel'],
                _format_currency(entry['revenue']),
                _format_number(entry['rpm'], decimals=2),
                _format_ratio(entry['roas']),
                _format_percent(entry['ctr']),
            ])
        table = _markdown_table(
            ["Creative", "Channel", "Revenue", "Revenue/1k Views", "ROAS", "CTR"],
            rows,
        )
        body_lines = [explainer, table]
        if creative_df is None or creative_df.empty:
            body_lines.append('Not available in artifacts. (missing: creative_id/views)')
        return "\n".join(body_lines), False

    numeric_cols = ["revenue", "ad_spend", "views", "clicks"]

    if creative_df is not None and not creative_df.empty and "creative_id" in creative_df.columns:
        df = creative_df.copy()
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "channel" in df.columns:
            df["channel"] = df["channel"].apply(_normalize_channel)
        df = df[df["creative_id"].notna()]
        if not df.empty:
            agg_dict = {col: "sum" for col in numeric_cols if col in df.columns}
            summary = df.groupby(["creative_id", "channel"], dropna=False).agg(agg_dict).reset_index()
            if not summary.empty:
                summary = summary.fillna(0.0)
                summary["rpm"] = summary.apply(
                    lambda row: (row["revenue"] / row["views"]) * 1000 if row.get("views", 0) else None,
                    axis=1,
                )
                summary["roas"] = summary.apply(
                    lambda row: _safe_div(row["revenue"], row.get("ad_spend"), allow_inf=True),
                    axis=1,
                )
                summary["ctr"] = summary.apply(
                    lambda row: _safe_div(row["clicks"], row["views"]) if row.get("views", 0) else None,
                    axis=1,
                )
                metric_key = "rpm" if summary["rpm"].notna().any() else "roas"
                summary = summary.sort_values(
                    [metric_key, "revenue"], ascending=[False, False]
                ).head(5)

                rows = []
                for _, row in summary.iterrows():
                    rows.append([
                        str(row.get("creative_id")),
                        str(row.get("channel", "n/a")),
                        _format_currency(row.get("revenue")),
                        _format_number(row.get("rpm"), decimals=2),
                        _format_ratio(row.get("roas")),
                        _format_percent(row.get("ctr")),
                    ])

                table = _markdown_table(
                    ["Creative", "Channel", "Revenue", "Revenue/1k Views", "ROAS", "CTR"],
                    rows,
                )
                return f"{explainer}\n{table}", False

    if campaign_df is not None and not campaign_df.empty and "campaign" in campaign_df.columns:
        df = campaign_df.copy()
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        summary = df.groupby("campaign", dropna=False).agg(
            {col: "sum" for col in numeric_cols if col in df.columns}
        ).reset_index()
        if not summary.empty:
            summary = summary.fillna(0.0)
            summary["rpm"] = summary.apply(
                lambda row: (row["revenue"] / row["views"]) * 1000 if row.get("views", 0) else None,
                axis=1,
            )
            summary["roas"] = summary.apply(
                lambda row: _safe_div(row["revenue"], row.get("ad_spend"), allow_inf=True),
                axis=1,
            )
            summary["ctr"] = summary.apply(
                lambda row: _safe_div(row["clicks"], row["views"]) if row.get("views", 0) else None,
                axis=1,
            )
            metric_key = "rpm" if summary["rpm"].notna().any() else "roas"
            summary = summary.sort_values(
                [metric_key, "revenue"], ascending=[False, False]
            ).head(5)
            rows = []
            for _, row in summary.iterrows():
                rows.append([
                    str(row.get("campaign")),
                    _format_currency(row.get("revenue")),
                    _format_number(row.get("rpm"), decimals=2),
                    _format_ratio(row.get("roas")),
                    _format_percent(row.get("ctr")),
                ])
            table = _markdown_table(
                ["Campaign", "Revenue", "Revenue/1k Views", "ROAS", "CTR"],
                rows,
            )
            body = [
                explainer,
                "Creative IDs missing; campaign-level proxy shown instead.",
                table,
            ]
            return "\n".join(body), False

    return f"{explainer}\nNot available in artifacts. (missing: creative_id/views)", True



def build_md_only_template(
    settings: Any,
    artifacts: Dict[str, Any],
    *,
    seasonality: float,
    safety_stock: float,
) -> MDOnlyTemplateResult:
    series_df_obj = artifacts.get("series")
    if not isinstance(series_df_obj, pd.DataFrame) or series_df_obj.empty:
        raise RuntimeError("series.jsonl is required to build the MD-only brief.")

    series_df = series_df_obj.copy()
    series_df["date"] = pd.to_datetime(series_df["date"], errors="coerce")
    series_df = series_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    _ensure_synonym_columns(series_df)

    mappings = settings.mappings
    rename_map: Dict[str, str] = {}

    def _map(source: Optional[str], target: str) -> None:
        if not source or source not in series_df.columns:
            return
        if target in series_df.columns:
            series_df[target] = series_df[target].fillna(series_df[source])
            return
        rename_map[source] = target

    _map(mappings.revenue, "revenue")
    _map(getattr(mappings, "total_sales", None), "total_sales")
    _map(mappings.orders, "orders")
    _map(mappings.ad_spend, "ad_spend")
    _map(mappings.views, "views")
    _map(getattr(mappings, "engagement_reach", None), "reach")
    _map(getattr(mappings, "engagement_likes", None), "likes")
    _map(getattr(mappings, "engagement_comments", None), "comments")
    _map(getattr(mappings, "engagement_shares", None), "shares")
    _map(getattr(mappings, "engagement_follows", None), "follows")
    _map(getattr(mappings, "engagement_saves", None), "saves")
    _map(getattr(mappings, "sales_discounts", None), "discounts")
    _map(getattr(mappings, "sales_returns", None), "returns")
    _map(getattr(mappings, "sales_shipping", None), "shipping")
    _map(getattr(mappings, "sales_duties", None), "duties")
    _map(getattr(mappings, "sales_taxes", None), "taxes")

    if rename_map:
        series_df = series_df.rename(columns=rename_map)

    numeric_cols = [
        "revenue",
        "total_sales",
        "orders",
        "ad_spend",
        "views",
        "reach",
        "likes",
        "comments",
        "shares",
        "follows",
        "saves",
        "discounts",
        "returns",
        "shipping",
        "duties",
        "taxes",
        "units",
        "clicks",
    ]
    for col in numeric_cols:
        if col in series_df.columns:
            series_df[col] = pd.to_numeric(series_df[col], errors="coerce")

    windows = {
        7: _compute_window_metrics(series_df, 7),
        28: _compute_window_metrics(series_df, 28),
        90: _compute_window_metrics(series_df, 90),
    }

    revenue_slope = _compute_revenue_slope(series_df)
    latest_date = series_df["date"].max() if not series_df.empty else None
    start_28 = (latest_date - pd.Timedelta(days=27)) if isinstance(latest_date, pd.Timestamp) else None

    def _prepare_artifact(name: str, numeric_cols: Iterable[str]) -> pd.DataFrame:
        obj = artifacts.get(name)
        if not isinstance(obj, pd.DataFrame) or obj.empty:
            return pd.DataFrame()
        df_art = obj.copy()
        if "date" in df_art.columns:
            df_art["date"] = pd.to_datetime(df_art["date"], errors="coerce")
            df_art = df_art.dropna(subset=["date"]).sort_values("date")
        for col in numeric_cols:
            if col in df_art.columns:
                df_art[col] = pd.to_numeric(df_art[col], errors="coerce")
        return df_art

    sku_artifact = _prepare_artifact("sku_series", ["revenue", "units", "orders"])
    channel_artifact = _prepare_artifact("channel_series", ["revenue", "ad_spend", "orders", "views", "clicks"])
    campaign_artifact = _prepare_artifact("campaign_series", ["revenue", "ad_spend", "orders", "views", "clicks"])
    creative_artifact = _prepare_artifact("creative_series", ["revenue", "ad_spend", "views", "clicks"])
    margin_artifact = _prepare_artifact("margin_series", ["revenue", "discounts", "returns", "shipping", "taxes", "duties"])
    inventory_artifact = _prepare_artifact("inventory_snapshot", ["on_hand", "on_order", "lead_time_days"])

    efficiency_payload = artifacts.get("efficiency") if isinstance(artifacts.get("efficiency"), dict) else None

    def _window_artifact(df_art: pd.DataFrame) -> pd.DataFrame:
        if df_art is None or df_art.empty:
            return pd.DataFrame()
        if start_28 is None or "date" not in df_art.columns:
            return df_art.copy()
        return _filter_date_window(df_art, start_28, latest_date)

    sku_window = _window_artifact(sku_artifact)
    channel_window = _window_artifact(channel_artifact)
    campaign_window = _window_artifact(campaign_artifact)
    creative_window = _window_artifact(creative_artifact)
    margin_window = _window_artifact(margin_artifact)

    margin_summary = _summarise_margin(margin_window if not margin_window.empty else series_df, 28)
    window_28_df = _window_slice(series_df, 28)

    channel_section, channel_missing, top_channel = _channel_efficiency_section(channel_window, efficiency_payload, window_28_df)
    segment_section, segment_missing = _customer_segments_section(campaign_window)
    hero_section, hero_missing, hero_snapshot = _hero_skus_section(sku_window, inventory_artifact, seasonality, safety_stock)
    creative_section, creative_missing = _creative_performance_section(creative_window, campaign_window, efficiency_payload)

    missing_sections: List[str] = []
    if channel_missing:
        missing_sections.append("marketing_efficiency")
    if segment_missing:
        missing_sections.append("customer_segments")
    if hero_missing:
        missing_sections.append("hero_skus")
    if creative_missing:
        missing_sections.append("creative_performance")

    segment_summary = _aggregate_segments(campaign_window)
    total_seg_revenue = float(segment_summary["revenue"].fillna(0).sum()) if not segment_summary.empty and "revenue" in segment_summary.columns else 0.0
    tof_row: Optional[Dict[str, Any]] = None
    bof_row: Optional[Dict[str, Any]] = None
    if not segment_summary.empty:
        if (segment_summary["__segment"] == "Prospecting / TOF").any():
            tof_row = segment_summary.loc[segment_summary["__segment"] == "Prospecting / TOF"].iloc[0].to_dict()
        if (segment_summary["__segment"] == "Retargeting / BOF").any():
            bof_row = segment_summary.loc[segment_summary["__segment"] == "Retargeting / BOF"].iloc[0].to_dict()
    top_hero = hero_snapshot

    mer7 = windows[7]["mer"]
    mer28 = windows[28]["mer"]
    cac7 = windows[7]["cac"]
    cac28 = windows[28]["cac"]
    avg7 = windows[7]["daily_revenue"]
    avg28 = windows[28]["daily_revenue"]
    slope_direction = "up" if (revenue_slope or 0) > 0 else "down" if (revenue_slope or 0) < 0 else "flat"

    opportunities_lines: List[str] = []
    if top_hero:
        opportunities_lines.append(
            f"- Lead product **{top_hero['name']}** drove {_format_currency(top_hero['revenue'])} ({_format_percent(top_hero['share'])} of 28-day revenue)."
        )
    if top_channel:
        opportunities_lines.append(
            f"- {top_channel['name']} is {_format_percent(top_channel['share'])} of paid revenue at ROAS {_format_ratio(top_channel['roas'])}."
        )
    opportunities_lines.append(
        f"- MER {_format_ratio(mer28)} vs {_format_ratio(mer7)} (28d vs 7d); CAC {_format_currency(cac28)} vs {_format_currency(cac7)}."
    )

    revenue_trend_lines = [
        f"- Seasonality-adjusted trend: revenue slope {slope_direction} ({_format_currency(revenue_slope, decimals=0)} per day).",
        f"- 7d revenue {_format_currency(windows[7]['revenue_sum'])} vs 28d {_format_currency(windows[28]['revenue_sum'])}.",
        f"- MER delta (7d minus 28d): {('n/a' if mer7 is None or mer28 is None else f'{mer7 - mer28:.2f}')}",
    ]

    segment_lines = []
    if tof_row:
        tof_revenue = float(tof_row.get('revenue', 0.0) or 0.0)
        share = tof_revenue / total_seg_revenue if total_seg_revenue else None
        segment_lines.append(f"- Prospecting / TOF = {_format_currency(tof_revenue)} ({_format_percent(share)} of campaign revenue).")
    if bof_row:
        bof_revenue = float(bof_row.get('revenue', 0.0) or 0.0)
        share = bof_revenue / total_seg_revenue if total_seg_revenue else None
        segment_lines.append(f"- Retargeting / BOF = {_format_currency(bof_revenue)} ({_format_percent(share)} of campaign revenue).")
    if not segment_lines:
        segment_lines.append('- Campaign names lack clear TOF/BOF markers. Add intent keywords to expose funnel mix.')

    margin_lines = []
    components = {comp['label']: comp for comp in margin_summary.get('components', [])}
    for key, label in [('discounts', 'Discounts'), ('returns', 'Returns'), ('shipping', 'Shipping'), ('taxes', 'Taxes'), ('duties', 'Duties')]:
        comp = components.get(key)
        if comp and comp.get('total') is not None:
            share_text = _format_percent(comp.get('share')) if comp.get('share') is not None else 'n/a'
            margin_lines.append(f"- {label}: {_format_currency(comp.get('total'))} ({share_text} of revenue).")
    if not margin_lines:
        margin_lines.append('- Margin components unavailable; reconcile discounts/returns/shipping feeds.')

    metrics_plain = [        'Metrics in Plain English',
        '- MER (same as ROAS) = Revenue / Ad Spend.',
        '- CAC = Ad Spend / Orders.',
        '- AOV = Revenue / Orders.',
        '- RPM = Revenue / Views x 1,000.',
        '- CTR = Clicks / Views.',
        '- Run Rate = Units / 28 days.',
    ]

    segment_content = segment_section if segment_missing else "\n".join(segment_lines)

    templates: List[str] = [
        "## Metrics in Plain English",
        "\n".join(metrics_plain[1:]),
        "",
        "## What are our three biggest opportunities for growth? (What/Why/How next week)",
        "\n".join(opportunities_lines),
        "",
        "## Accounting for seasonality... is the revenue accelerating or slowing down? (7/28/90 + slope + MER)",
        "\n".join(revenue_trend_lines),
        "",
        "## What customer segments do we have... how do we get more? (infer TOF/BOF from names; if ambiguous, instruct naming hygiene)",
        segment_content,
        "",
        "## What products truly pull demand (hero SKUs)... inventory for Q4/Q1? (table from sku_series.jsonl)",
        hero_section,
        "",
        "## How can we improve margins? What's killing margin post-purchase? (28-day discounts/returns/shipping/taxes)",
        "\n".join(margin_lines),
        "",
        "## Does marketing work? Which parts add incremental value? (organic vs paid vs SMS) (use efficiency.by_channel, scope noted)",
        channel_section,
        "",
        "## Which creatives actually move product on IG? (use by_creative, RPM/ROAS/CTR with guards)",
        creative_section,
        "",
        "## Additional Insights (Model Reasoning)",
        "{{ADDITIONAL_INSIGHTS}}",
        "",
        "## Executive Summary",
        "\n".join([
            "Quick view of revenue, efficiency, and order value across time horizons.",
            f"- 7-day Revenue: {_format_currency(windows[7]['revenue_sum'])}; MER {_format_ratio(windows[7]['mer'])}; CAC {_format_currency(windows[7]['cac'])}; AOV {_format_currency(windows[7]['aov'])}",
            f"- 28-day Revenue: {_format_currency(windows[28]['revenue_sum'])}; MER {_format_ratio(windows[28]['mer'])}; CAC {_format_currency(windows[28]['cac'])}; AOV {_format_currency(windows[28]['aov'])}",
            f"- 90-day Revenue: {_format_currency(windows[90]['revenue_sum'])}; MER {_format_ratio(windows[90]['mer'])}; CAC {_format_currency(windows[90]['cac'])}; AOV {_format_currency(windows[90]['aov'])}",
        ]),
        "",
        "## Key Trends",
        "\n".join([
            f"- Revenue slope trend: {slope_direction} ({_format_currency(revenue_slope, decimals=0)} per day)",
            f"- MER delta (7d minus 28d): {('n/a' if mer7 is None or mer28 is None else f'{mer7 - mer28:.2f}')}",
            f"- CAC delta (7d minus 28d): {('n/a' if cac7 is None or cac28 is None else f'{cac7 - cac28:.2f}')}",
        ]),
        "",
    ]

    anomaly_lines = _build_anomaly_section(artifacts.get("anomalies"), artifacts.get("anomalies_notes"))
    quality_lines = _build_quality_section(artifacts.get("quality_report"))
    kpi_rows = _build_kpi_table(windows)

    templates.extend([
        "## Anomalies",
        "\n".join(anomaly_lines),
        "",
        "## Data Quality & Caveats",
        "\n".join(quality_lines),
        "",
        "## KPI Appendix (7/28/90-day)",
        _markdown_table(
            ["Window", "Revenue", "Spend", "Orders", "MER", "CAC", "AOV"],
            kpi_rows,
        ),
    ])

    template = "\n".join(templates).strip() + "\n"

    metrics = {
        "windows": windows,
        "revenue_slope": revenue_slope,
        "margin": margin_summary,
        "latest_date": latest_date,
        "seasonality": seasonality,
        "safety_stock": safety_stock,
    }

    return MDOnlyTemplateResult(
        template=template,
        missing_sections=missing_sections,
        metrics=metrics,
        latest_date=latest_date,
    )
