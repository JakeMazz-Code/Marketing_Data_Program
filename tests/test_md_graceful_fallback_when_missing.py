from __future__ import annotations

from pathlib import Path

import pandas as pd

from Marketing_analytics import generate_brief_md


_MISSING_SNIPPETS = {
    "marketing": "Not available in artifacts. (missing: channel/ad_spend/orders)",
    "segments": "Not available in artifacts. (missing: campaign/adset text)",
    "skus": "Not available in artifacts. (missing: sku/product_title metrics)",
    "creatives": "Not available in artifacts. (missing: creative_id/views)",
}


def test_md_graceful_fallback_when_missing(md_only_context):
    original_series = md_only_context.artifacts["series"].copy()

    stripped_series = original_series.drop(
        columns=[
            "platform",
            "campaign_name",
            "adset_name",
            "product_id",
            "product_name",
            "quantity",
            "post_id",
            "impressions",
            "link_clicks",
        ],
        errors="ignore",
    )

    md_only_context.artifacts["series"] = stripped_series
    original_channel = md_only_context.artifacts.get("channel_series")
    original_campaign = md_only_context.artifacts.get("campaign_series")
    original_sku = md_only_context.artifacts.get("sku_series")
    original_creative = md_only_context.artifacts.get("creative_series")

    md_only_context.artifacts["channel_series"] = pd.DataFrame()
    md_only_context.artifacts["campaign_series"] = pd.DataFrame()
    md_only_context.artifacts["sku_series"] = pd.DataFrame()
    md_only_context.artifacts["creative_series"] = pd.DataFrame()


    result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    text = Path(result["brief_md"]).read_text(encoding="utf-8")

    for snippet in _MISSING_SNIPPETS.values():
        assert snippet in text, f"Expected fallback message '{snippet}'"

    md_only_context.artifacts["series"] = original_series

