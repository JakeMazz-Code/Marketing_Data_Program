from __future__ import annotations

from types import SimpleNamespace
from datetime import datetime
from typing import Dict, Any

import pandas as pd


def build_series() -> pd.DataFrame:
    dates = pd.date_range("2024-06-01", periods=90, freq="D")
    channel_options = ["meta_paid", "tiktok_ads", "email_sends"]
    campaign_options = [
        "Prospecting - Cold LAA",
        "Retarget - Cart DPA",
        "Newsletter - Loyalty",
    ]
    adset_options = [
        "Cold_Broad",
        "BOF_Cart",
        "VIP_List",
    ]
    sku_options = ["SKU-001", "SKU-002", "SKU-003"]
    product_titles = ["Glow Serum", "Hydra Mask", "Vitamin Mist"]
    creative_options = ["CR-IG-1", "CR-TT-2", "CR-EM-3"]

    records = []
    for idx, date in enumerate(dates):
        revenue = 1200 + idx * 15
        spend = 300 + (idx % 7) * 8
        orders = 25 + (idx % 5)
        views = 15000 + idx * 120
        discounts = 40 + (idx % 3)
        returns = 18 + (idx % 4)
        shipping = 22 + (idx % 2)
        duties = 5
        taxes = 12
        units = 18 + (idx % 6)
        clicks = 95 + (idx % 9)

        channel = channel_options[idx % len(channel_options)]
        campaign = campaign_options[idx % len(campaign_options)]
        adset = adset_options[idx % len(adset_options)]
        sku = sku_options[idx % len(sku_options)]
        product = product_titles[idx % len(product_titles)]
        creative = creative_options[idx % len(creative_options)]

        records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "revenue": revenue,
                "total_sales": revenue * 1.1,
                "orders": orders,
                "ad_spend": spend,
                "views": views,
                "discounts": discounts,
                "returns": returns,
                "shipping": shipping,
                "duties": duties,
                "taxes": taxes,
                "quantity": units,
                "platform": channel,
                "campaign_name": campaign,
                "adset_name": adset,
                "product_id": sku,
                "product_name": product,
                "post_id": creative,
                "impressions": views + 500,
                "link_clicks": clicks,
            }
        )
    return pd.DataFrame.from_records(records)


def stub_settings(tmp_path) -> SimpleNamespace:
    mappings = SimpleNamespace(
        revenue="revenue",
        total_sales="total_sales",
        orders="orders",
        ad_spend="ad_spend",
        views="views",
        engagement_reach=None,
        engagement_likes=None,
        engagement_comments=None,
        engagement_shares=None,
        engagement_follows=None,
        engagement_saves=None,
        sales_discounts="discounts",
        sales_returns="returns",
        sales_shipping="shipping",
        sales_duties="duties",
        sales_taxes="taxes",
    )
    derived = SimpleNamespace(mer="mer", roas="roas", aov="aov")
    return SimpleNamespace(artifacts_dir=tmp_path, mappings=mappings, derived=derived)


def base_artifacts(series_df: pd.DataFrame) -> Dict[str, Any]:
    anomalies = [
        {
            "metric": "revenue",
            "start_date": "2024-07-04",
            "end_date": "2024-07-04",
            "peak_z": 3.1,
        }
    ]
    quality = {
        "status": "PASS",
        "rules": [],
    }

    channel_df = (
        series_df[
            [
                "date",
                "platform",
                "revenue",
                "ad_spend",
                "orders",
                "views",
                "link_clicks",
            ]
        ]
        .rename(
            columns={
                "platform": "channel",
                "link_clicks": "clicks",
            }
        )
        .copy()
    )

    campaign_df = (
        series_df[
            [
                "date",
                "platform",
                "campaign_name",
                "adset_name",
                "revenue",
                "ad_spend",
                "orders",
                "views",
                "link_clicks",
            ]
        ]
        .rename(
            columns={
                "platform": "channel",
                "campaign_name": "campaign",
                "adset_name": "adset",
                "link_clicks": "clicks",
            }
        )
        .copy()
    )

    sku_df = (
        series_df[["date", "product_id", "product_name", "quantity", "revenue", "orders"]]
        .rename(
            columns={
                "product_id": "sku",
                "product_name": "product_title",
                "quantity": "units",
            }
        )
        .copy()
    )

    creative_df = (
        series_df[["date", "platform", "post_id", "views", "link_clicks", "ad_spend", "revenue"]]
        .rename(
            columns={
                "platform": "channel",
                "post_id": "creative_id",
                "link_clicks": "clicks",
            }
        )
        .copy()
    )

    margin_df = series_df[
        ["date", "revenue", "discounts", "returns", "shipping", "duties", "taxes"]
    ].copy()

    latest_date = series_df["date"].iloc[-1]
    inventory_records = []
    for offset, (sku, group) in enumerate(series_df.groupby("product_id"), start=0):
        tail = group.iloc[-1]
        inventory_records.append(
            {
                "date": latest_date,
                "sku": sku,
                "product_title": tail["product_name"],
                "on_hand": 120 + offset * 15,
                "on_order": 35 + offset * 5,
                "lead_time_days": 18 + offset,
            }
        )
    inventory_df = pd.DataFrame(inventory_records)

    return {
        "series": series_df,
        "quality_report": quality,
        "anomalies": anomalies,
        "efficiency": {
            "as_of": pd.to_datetime(series_df["date"]).iloc[-1].strftime("%Y-%m-%d"),
            "window_days": 28,
            "scope": "paid",
            "by_channel": [
                {"channel": "Meta", "revenue": 32000.0, "spend": 8000.0, "orders": 620},
                {"channel": "TikTok", "revenue": 21000.0, "spend": 5000.0, "orders": 410},
            ],
            "by_creative": [
                {"creative_id": "CR-IG-1", "channel": "Meta", "revenue": 18000.0, "views": 85000, "spend": 4200.0, "clicks": 5400},
                {"creative_id": "CR-TT-2", "channel": "TikTok", "revenue": 12000.0, "views": 64000, "spend": 2800.0, "clicks": 4100},
            ],
        },
        "creatives": None,
        "llm_payload": {"windows": {}},
        "channel_series": channel_df,
        "campaign_series": campaign_df,
        "sku_series": sku_df,
        "creative_series": creative_df,
        "margin_series": margin_df,
        "inventory_snapshot": inventory_df,
    }




def patch_inputs(monkeypatch, settings, artifacts) -> None:
    import Marketing_analytics.ai as ai_module

    monkeypatch.setattr(ai_module, "load_daily_master_settings", lambda _path: settings)
    monkeypatch.setattr(ai_module, "_load_artifacts", lambda _dir: artifacts)

