"""Data preparation utilities for raw marketing exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from Marketing_analytics.data_loader import _normalize_column_name


@dataclass(slots=True)
class CleanedTables:
    orders: Optional[pd.DataFrame]
    meta_campaigns: Optional[pd.DataFrame]
    tiktok_campaigns: Optional[pd.DataFrame]
    instagram_posts: Optional[pd.DataFrame]
    daily_views: Optional[pd.DataFrame]
    daily_sales: Optional[pd.DataFrame]
    daily_spend: Optional[pd.DataFrame]


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return series
    cleaned = (
        series.astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        .str.replace(r"\'", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _series_with_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        series = _to_numeric(df[column])
        return series.fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def clean_shopify_orders(frame: pd.DataFrame) -> pd.DataFrame:
    """Return line-level Shopify orders with normalized columns."""

    df = frame.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]

    value_columns = [
        "subtotal",
        "shipping",
        "taxes",
        "total",
        "lineitem_price",
        "lineitem_compare_at_price",
        "lineitem_discount",
        "discount_amount",
        "refunded_amount",
    ]
    for column in value_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])

    if "lineitem_quantity" in df.columns:
        df["lineitem_quantity"] = _to_numeric(df["lineitem_quantity"]).fillna(0).astype(int)

    if "created_at" in df.columns:
        df["created_at"] = _to_datetime(df["created_at"])

    df["order_id"] = df.get("name", "").astype(str).str.replace("#", "", regex=False).str.strip()
    if df["order_id"].eq("").all():
        df["order_id"] = df.index.astype(str)

    df["customer_id"] = df["order_id"]
    df["event_timestamp"] = df.get("created_at")
    df["product_name"] = df.get("lineitem_name")
    df["product_sku"] = df.get("lineitem_sku")

    price = _series_with_default(df, "lineitem_price")
    line_discount = _series_with_default(df, "lineitem_discount")
    revenue = price - line_discount
    fallback_revenue = _series_with_default(df, "subtotal")
    df["revenue"] = revenue.where(revenue.notna() & (revenue != 0), fallback_revenue)
    df["discount_amount_value"] = _series_with_default(df, "discount_amount")
    df["shipping_cost"] = _series_with_default(df, "shipping")
    df["tax_amount"] = _series_with_default(df, "taxes")
    df["refund_amount"] = _series_with_default(df, "refunded_amount")
    df["return_flag"] = (df["refund_amount"] > 0).astype(int)
    df["units"] = _series_with_default(df, "lineitem_quantity", default=1).fillna(1).astype(int)
    df["spend"] = 0.0
    df["response_flag"] = 1.0

    def _first_nonempty(primary: str, secondary: str, default: str) -> pd.Series:
        primary_series = df.get(primary)
        secondary_series = df.get(secondary)
        result = (
            primary_series.astype(str).str.strip() if primary_series is not None else pd.Series(index=df.index, dtype="object")
        )
        result = result.replace({"": np.nan, "nan": np.nan})
        if secondary_series is not None:
            secondary_clean = secondary_series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
            result = result.fillna(secondary_clean)
        return result.fillna(default)

    df["campaign"] = _first_nonempty("tags", "source", "shopify")
    df["channel"] = df.get("source", "shopify")
    df["channel"] = df["channel"].fillna("shopify")
    df["segment"] = _first_nonempty("shipping_province", "billing_province", "unknown")
    df["country"] = _first_nonempty("shipping_country", "billing_country", "unknown")

    cleaned = df[
        [
            "customer_id",
            "order_id",
            "event_timestamp",
            "campaign",
            "channel",
            "segment",
            "country",
            "product_name",
            "product_sku",
            "units",
            "revenue",
            "discount_amount_value",
            "shipping_cost",
            "tax_amount",
            "refund_amount",
            "return_flag",
            "spend",
            "response_flag",
            "currency",
        ]
    ].rename(
        columns={
            "discount_amount_value": "discount_amount",
            "response_flag": "response",
        }
    )

    cleaned["currency"] = cleaned["currency"].fillna("USD")
    cleaned = cleaned.dropna(subset=["event_timestamp"])

    return cleaned


def clean_meta_campaigns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]

    date_columns = ["reporting_starts", "reporting_ends", "ends"]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    numeric_columns: Iterable[str] = [
        "amount_spent_usd",
        "clicks_all",
        "reach",
        "impressions",
        "results",
        "adds_to_cart",
        "adds_to_cart_conversion_value",
        "direct_website_purchases",
        "frequency",
        "cost_per_results",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])

    df["channel"] = "meta"
    df = df.rename(
        columns={
            "reporting_starts": "start_date",
            "reporting_ends": "end_date",
            "campaign_name": "campaign",
            "amount_spent_usd": "spend",
            "results": "conversions",
            "adds_to_cart": "add_to_cart",
            "adds_to_cart_conversion_value": "add_to_cart_value",
            "direct_website_purchases": "purchases",
        }
    )

    keep_columns = [
        "start_date",
        "end_date",
        "campaign",
        "channel",
        "spend",
        "impressions",
        "reach",
        "clicks_all",
        "ctr_all",
        "cpc_all_usd",
        "frequency",
        "conversions",
        "result_indicator",
        "purchases",
        "purchase_roas_return_on_ad_spend",
    ]
    existing_columns = [col for col in keep_columns if col in df.columns]
    return df[existing_columns]


def clean_tiktok_campaigns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]

    numeric_columns = [
        "cost",
        "impressions",
        "clicks_(destination)",
        "purchases_(website)",
        "purchase_value_(website)",
        "checkouts_initiated_(website)",
        "adds_to_cart_website",
        "video_views",
        "video_views_at_50%",
        "video_views_at_100%",
        "2-second_video_views",
        "6-second_video_views",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])
    df["channel"] = "tiktok"
    df = df.rename(
        columns={
            "ad_name": "ad_name",
            "cost": "spend",
            "clicks_destination": "clicks",
            "purchases_website": "purchases",
            "purchase_value_website": "purchase_value",
            "payment_completion_roas_website": "roas",
        }
    )
    keep_columns = [
        "ad_name",
        "channel",
        "spend",
        "impressions",
        "clicks",
        "ctr_destination",
        "cpc_destination",
        "purchases",
        "purchase_value",
        "roas",
        "adds_to_cart_website",
    ]
    existing_columns = [col for col in keep_columns if col in df.columns]
    return df[existing_columns]


def clean_instagram_posts(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]

    date_columns = ["publish_time", "date"]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    numeric_columns = ["views", "reach", "likes", "shares", "follows", "comments", "saves"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])

    rename_map = {
        "post_id": "post_id",
        "account_username": "account",
        "description": "description",
        "publish_time": "publish_time",
        "views": "views",
        "reach": "reach",
        "likes": "likes",
        "shares": "shares",
        "follows": "follows",
        "comments": "comments",
        "saves": "saves",
    }
    existing_columns = [col for col in rename_map if col in df.columns]
    return df[existing_columns].rename(columns=rename_map)


def clean_daily_views(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if isinstance(df.iloc[0, 0], str) and df.iloc[0, 0].startswith("sep="):
        df = df.iloc[1:]
    header_row = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = header_row
    df = df.rename(columns={"Date": "date", "Primary": "views"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["views"] = _to_numeric(df["views"])
    return df.dropna(subset=["date"])


def clean_daily_sales(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    numeric_columns = [
        "orders",
        "gross_sales",
        "discounts",
        "returns",
        "net_sales",
        "shipping_charges",
        "taxes",
        "total_sales",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = _to_numeric(df[column])
    return df[[col for col in ["day"] + numeric_columns if col in df.columns]].dropna(subset=["day"])


def clean_daily_spend(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [_normalize_column_name(col) for col in df.columns]
    date_col = df.columns[0]
    spend_col = df.columns[1] if len(df.columns) > 1 else "amount"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[spend_col] = _to_numeric(df[spend_col])
    df = df.rename(columns={date_col: "date", spend_col: "spend"})
    return df.dropna(subset=["date"])


def write_cleaned_outputs(
    *,
    raw_dir: Path,
    output_dir: Path,
    build_pipeline_dataset: bool = True,
) -> CleanedTables:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load_csv(name: str, **kwargs) -> Optional[pd.DataFrame]:
        path = raw_dir / name
        if not path.exists():
            return None
        read_kwargs: Dict[str, object] = {"encoding": "utf-8", **kwargs}
        try:
            return pd.read_csv(path, **read_kwargs)
        except UnicodeDecodeError:
            read_kwargs["encoding"] = "latin1"
            return pd.read_csv(path, **read_kwargs)

    orders_raw = _load_csv("NYU case study file.csv")
    orders_clean = clean_shopify_orders(orders_raw) if orders_raw is not None else None
    if orders_clean is not None:
        orders_clean.to_csv(output_dir / "orders_clean.csv", index=False)

    meta_raw = _load_csv("meta.csv")
    meta_clean = clean_meta_campaigns(meta_raw) if meta_raw is not None else None
    if meta_clean is not None:
        meta_clean.to_csv(output_dir / "meta_campaigns_clean.csv", index=False)

    tiktok_raw = _load_csv("tiktok.csv")
    tiktok_clean = clean_tiktok_campaigns(tiktok_raw) if tiktok_raw is not None else None
    if tiktok_clean is not None:
        tiktok_clean.to_csv(output_dir / "tiktok_campaigns_clean.csv", index=False)

    insta_raw = _load_csv("results by day pg 1.csv")
    insta_clean = clean_instagram_posts(insta_raw) if insta_raw is not None else None
    if insta_clean is not None:
        insta_clean.to_csv(output_dir / "instagram_posts_clean.csv", index=False)

    views_raw = _load_csv("results by day pg 2.csv", header=None)
    views_clean = clean_daily_views(views_raw) if views_raw is not None else None
    if views_clean is not None:
        views_clean.to_csv(output_dir / "daily_views_clean.csv", index=False)

    sales_raw = _load_csv("results by day pg 3.csv")
    sales_clean = clean_daily_sales(sales_raw) if sales_raw is not None else None
    if sales_clean is not None:
        sales_clean.to_csv(output_dir / "daily_sales_clean.csv", index=False)

    spend_raw = _load_csv("results by day pg 4.csv")
    spend_clean = clean_daily_spend(spend_raw) if spend_raw is not None else None
    if spend_clean is not None:
        spend_clean.to_csv(output_dir / "daily_spend_clean.csv", index=False)

    if build_pipeline_dataset and orders_clean is not None:
        pipeline_columns = [
            "customer_id",
            "event_timestamp",
            "campaign",
            "channel",
            "segment",
            "product_name",
            "product_sku",
            "units",
            "revenue",
            "discount_amount",
            "shipping_cost",
            "tax_amount",
            "refund_amount",
            "return_flag",
            "spend",
            "response",
        ]
        orders_clean[pipeline_columns].to_csv(output_dir / "marketing_events.csv", index=False)

    return CleanedTables(
        orders=orders_clean,
        meta_campaigns=meta_clean,
        tiktok_campaigns=tiktok_clean,
        instagram_posts=insta_clean,
        daily_views=views_clean,
        daily_sales=sales_clean,
        daily_spend=spend_clean,
    )
