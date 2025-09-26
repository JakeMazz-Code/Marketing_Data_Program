"""Assist in configuring datasets for the marketing analytics template."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_bootstrap_spec = importlib.util.spec_from_file_location(
    "cli._bootstrap_runtime",
    Path(__file__).resolve().parent / "_bootstrap.py",
)
if _bootstrap_spec is None or _bootstrap_spec.loader is None:
    raise ModuleNotFoundError("Unable to load CLI bootstrap helper.")
_bootstrap = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap)
_bootstrap.ensure_package_imported()

from Marketing_analytics.config import AnalysisSettings, ColumnMapping


KEYWORDS = {
    "customer_id": ["customer_id", "client_id", "user_id", "account_id", "shopify_customer", "shopify"],
    "event_timestamp": ["event_date", "timestamp", "date", "event_time", "occurred", "created_at", "created at", "order_date"],
    "campaign": ["campaign", "offer", "promotion", "initiative", "ad_set", "campaign_name"],
    "channel": ["channel", "source", "medium", "touchpoint", "platform"],
    "segment": ["segment", "cohort", "audience", "persona", "tier", "loyalty"],
    "response": ["converted", "response", "responded", "purchase", "order", "signup", "conversion"],
    "revenue": ["revenue", "sales", "amount", "purchase_value", "gmv", "total_price", "gross_sales"],
    "spend": ["spend", "cost", "expense", "budget", "media_cost", "ad_spend"],
    "touch_value": ["engagement", "score", "rating", "interaction", "clicks", "engagement_score"],
    "order_value": ["order_value", "ticket", "basket", "lifetime_value", "ltv", "subtotal"],
    "product_name": ["product", "product_name", "item", "title", "sku_name", "variant_title"],
    "product_sku": ["sku", "variant_id", "product_sku", "sku_id", "item_sku"],
    "product_category": ["category", "collection", "product_type", "line"],
    "units": ["units", "quantity", "qty", "items", "orders", "purchases"],
    "control_flag": ["control", "test_group", "variant", "holdout"],
    "creative_name": ["creative", "ad_name", "ad name", "asset", "creative_name", "post", "video"],
    "creative_id": ["creative_id", "ad_id", "asset_id", "creative"],
    "cogs": ["cogs", "cost_of_goods", "product_cost", "unit_cost", "cost_of_goods_sold"],
    "gross_margin": ["gross_margin", "margin", "profit", "net_margin"],
    "discount_amount": ["discount", "promo", "promotion", "markdown", "discount_amount", "coupon"],
    "shipping_cost": ["shipping", "freight", "delivery_cost", "logistics_cost"],
    "refund_amount": ["refund", "return_value", "refund_amount", "chargeback"],
    "return_flag": ["return", "returned", "is_return", "refund_flag", "rma"],
    "exchange_flag": ["exchange", "swapped", "is_exchange", "exchange_flag"],
    "defect_flag": ["defect", "damaged", "faulty", "qc_fail", "defect_flag"],
}


def detect_column(df: pd.DataFrame, keywords: List[str], exclude: Optional[set[str]] = None) -> Optional[str]:
    exclude = exclude or set()
    for column in df.columns:
        norm = column.lower()
        if column in exclude:
            continue
        if any(token in norm for token in keywords):
            return column
    return None


def suggest_mapping(df: pd.DataFrame) -> ColumnMapping:
    chosen: Dict[str, Optional[str]] = {}
    used: set[str] = set()
    for field, keywords in KEYWORDS.items():
        column = detect_column(df, keywords, exclude=used)
        chosen[field] = column
        if column:
            used.add(column)
    return ColumnMapping(**chosen)


def summarize_dataset(df: pd.DataFrame, *, limit: int = 10) -> Dict[str, object]:
    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [],
    }
    for column in df.columns[:limit]:
        series = df[column]
        summary["columns"].append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "sample": series.dropna().head(3).tolist(),
            }
        )
    return summary


def build_settings(data_path: Path, mapping: ColumnMapping, output_dir: Path) -> AnalysisSettings:
    return AnalysisSettings(
        data_path=data_path,
        output_dir=output_dir,
        mapping=mapping,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a CSV and scaffold a mapping config file.")
    parser.add_argument("data", type=Path, help="Path to the CSV dataset to inspect.")
    parser.add_argument(
        "--output-config",
        type=Path,
        help="Optional path to write a JSON config compatible with marketing_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Default output directory to embed in the generated config.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of preview rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    df.columns = [col.strip() for col in df.columns]

    mapping = suggest_mapping(df)
    summary = summarize_dataset(df)

    print("\nDataset overview:\n")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Columns: {summary['column_count']:,}")
    for column_meta in summary["columns"]:
        name = column_meta["name"]
        dtype = column_meta["dtype"]
        non_null = column_meta["non_null"]
        sample = ", ".join(map(str, column_meta["sample"]))
        print(f"- {name} ({dtype}) - non-null: {non_null:,}; sample: {sample}")

    print("\nSuggested column mapping:")
    for field, value in asdict(mapping).items():
        print(f"  {field}: {value}")

    preview = df.head(args.preview_rows)
    print("\nData preview:\n")
    print(preview.to_string(index=False))

    if args.output_config:
        settings = build_settings(args.data, mapping, args.output_dir)
        payload = {
            "data_path": str(settings.data_path),
            "output_dir": str(settings.output_dir),
            "mapping": asdict(settings.mapping),
            "time_granularity": settings.time_granularity,
            "minimum_rows": settings.minimum_rows,
            "include_modeling": settings.include_modeling,
            "include_visuals": settings.include_visuals,
            "random_seed": settings.random_seed,
        }
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nConfiguration written to {args.output_config}")


if __name__ == "__main__":
    main()



