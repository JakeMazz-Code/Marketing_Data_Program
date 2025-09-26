import pandas as pd

from Marketing_analytics.config import AISummaryConfig, AnalysisSettings, ColumnMapping
from Marketing_analytics.pipeline import MarketingAnalysisPipeline
from cli.prepare_dataset_template import suggest_mapping


def _build_sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shopify Customer": ["c1", "c1", "c2", "c3", "c4"],
            "Created At": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Campaign Name": ["Brand", "Brand", "Retarget", "Prospect", "Prospect"],
            "Platform": ["meta", "meta", "tiktok", "meta", "tiktok"],
            "Persona": ["Loyal", "Loyal", "New", "New", "At Risk"],
            "Conversion": [1, 0, 1, 0, 1],
            "Revenue": ["$120.50", "$0.00", "$220.00", "$0.00", "$95.25"],
            "Spend": ["$40.00", "$15.00", "$55.00", "$60.00", "$42.00"],
            "Units": [2, 0, 3, 0, 1],
            "Variant Title": ["Hero Tee", "Hero Tee", "Crew Hoodie", "Crew Hoodie", "Hero Tee"],
            "Sku": ["TEE-1", "TEE-1", "HD-1", "HD-1", "TEE-1"],
            "Ad Name": ["UGC-01", "UGC-01", "Spark-02", "Carousel-01", "UGC-02"],
            "Refund Amount": [0, 120.50, 0, 0, 0],
            "Return Flag": [0, 1, 0, 0, 0],
            "Discount": ["$5.00", "$0.00", "$12.00", "$0.00", "$0.00"],
            "Shipping": ["$4.00", "$4.00", "$6.00", "$6.00", "$4.00"],
            "COGS": ["$45.00", "$45.00", "$70.00", "$70.00", "$45.00"],
        }
    )


def test_pipeline_smoke(tmp_path):
    df = _build_sample_dataset()
    data_path = tmp_path / "dataset.csv"
    df.to_csv(data_path, index=False)

    mapping = ColumnMapping(
        customer_id="Shopify Customer",
        event_timestamp="Created At",
        campaign="Campaign Name",
        channel="Platform",
        segment="Persona",
        response="Conversion",
        revenue="Revenue",
        spend="Spend",
        units="Units",
        product_name="Variant Title",
        product_sku="Sku",
        creative_name="Ad Name",
        refund_amount="Refund Amount",
        return_flag="Return Flag",
        discount_amount="Discount",
        shipping_cost="Shipping",
        cogs="COGS",
    )

    settings = AnalysisSettings(
        data_path=data_path,
        output_dir=tmp_path / "reports",
        mapping=mapping,
        include_modeling=False,
        include_visuals=False,
        ai_summary=AISummaryConfig(enabled=False),
        minimum_rows=0,
    )

    pipeline = MarketingAnalysisPipeline(settings)
    result = pipeline.run()

    assert (settings.output_dir / "campaign_performance.csv").exists()
    assert (settings.output_dir / "margin_diagnostics.csv").exists()
    assert result["product"].shape[0] > 0
    assert result["creative"].shape[0] > 0
    assert result["margin"].shape[0] > 0
    assert result["overall"].get("total_revenue") == 435.75
    assert result["overall"].get("net_margin") is not None


def test_mapping_suggestions_handle_shopify_and_meta():
    df = _build_sample_dataset()
    mapping = suggest_mapping(df)
    assert mapping.customer_id == "Shopify Customer"
    assert mapping.event_timestamp == "Created At"
    assert mapping.product_name == "Variant Title"
    assert mapping.creative_name == "Ad Name"
    assert mapping.return_flag == "Return Flag"
    assert mapping.units == "Units"

