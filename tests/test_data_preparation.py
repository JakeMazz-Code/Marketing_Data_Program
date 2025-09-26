import pandas as pd

from Marketing_analytics.prep import (
    CleanedTables,
    clean_daily_views,
    clean_meta_campaigns,
    clean_shopify_orders,
    write_cleaned_outputs,
)


def test_clean_shopify_orders_basic():
    raw = pd.DataFrame(
        {
            "Name": ["#123", "#124"],
            "Created at": ["2025-01-01 10:00", "not a date"],
            "Lineitem quantity": ["2", "1"],
            "Lineitem price": ["$100.00", "$50.00"],
            "Lineitem discount": ["5", None],
            "Shipping": ["10", "0"],
            "Taxes": ["2", "0"],
            "Discount Amount": ["1", "0"],
            "Refunded Amount": ["0", "5"],
            "Tags": ["Launch", ""],
            "Source": ["web", "instagram"],
            "Currency": ["USD", None],
        }
    )
    cleaned = clean_shopify_orders(raw)
    assert list(cleaned.columns) == [
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
        "discount_amount",
        "shipping_cost",
        "tax_amount",
        "refund_amount",
        "return_flag",
        "spend",
        "response",
        "currency",
    ]
    assert len(cleaned) == 1  # drops the row without a valid timestamp
    row = cleaned.iloc[0]
    assert row["customer_id"] == "123"
    assert row["revenue"] == 95  # 100 - 5 discount
    assert row["discount_amount"] == 1
    assert row["return_flag"] == 0
    assert row["spend"] == 0
    assert row["currency"] == "USD"


def test_clean_daily_views_removes_sep_header():
    raw = pd.DataFrame(
        {
            0: ["sep=,", "Date", "2025-06-01", "2025-06-02"],
            1: [None, "Primary", "100", "200"],
        }
    )
    cleaned = clean_daily_views(raw)
    assert cleaned.shape == (2, 2)
    assert list(cleaned.columns) == ["date", "views"]
    assert cleaned["views"].tolist() == [100, 200]


def test_clean_meta_campaigns_channel_flag():
    raw = pd.DataFrame(
        {
            "Reporting starts": ["2025-01-01"],
            "Reporting ends": ["2025-01-07"],
            "Campaign name": ["Launch"],
            "Amount spent (USD)": ["1000"],
            "Reach": ["5000"],
            "Impressions": ["10000"],
            "Results": ["50"],
        }
    )
    cleaned = clean_meta_campaigns(raw)
    assert "channel" in cleaned.columns
    assert cleaned.loc[0, "channel"] == "meta"
    assert cleaned.loc[0, "spend"] == 1000


def test_write_cleaned_outputs(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    orders_sample = pd.DataFrame(
        {
            "Name": ["#123"],
            "Created at": ["2025-01-01"],
            "Lineitem quantity": [1],
            "Lineitem price": ["100"],
            "Currency": ["USD"],
        }
    )
    orders_sample.to_csv(raw_dir / "NYU case study file.csv", index=False)

    (raw_dir / "results by day pg 2.csv").write_text("sep=,\nDate,Primary\n2025-06-01,123\n", encoding="utf-8")

    output_dir = tmp_path / "processed"
    tables = write_cleaned_outputs(raw_dir=raw_dir, output_dir=output_dir)
    assert isinstance(tables, CleanedTables)
    assert (output_dir / "orders_clean.csv").exists()
    assert (output_dir / "marketing_events.csv").exists()
    orders = pd.read_csv(output_dir / "orders_clean.csv")
    assert orders.shape[0] == 1
    events = pd.read_csv(output_dir / "marketing_events.csv")
    assert "customer_id" in events.columns
