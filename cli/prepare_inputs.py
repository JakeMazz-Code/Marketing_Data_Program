"""CLI for cleaning raw marketing exports into pipeline-ready CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from Marketing_analytics.prep import write_cleaned_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw Shopify/Meta/TikTok exports and materialize processed CSVs.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the raw CSV exports (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory that will receive the cleaned CSVs (default: data/processed).",
    )
    parser.add_argument(
        "--skip-pipeline-dataset",
        action="store_true",
        help="Skip writing the combined marketing_events.csv used by the analysis pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = write_cleaned_outputs(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        build_pipeline_dataset=not args.skip_pipeline_dataset,
    )

    produced = {
        "orders": tables.orders is not None,
        "meta": tables.meta_campaigns is not None,
        "tiktok": tables.tiktok_campaigns is not None,
        "instagram": tables.instagram_posts is not None,
        "daily_views": tables.daily_views is not None,
        "daily_sales": tables.daily_sales is not None,
        "daily_spend": tables.daily_spend is not None,
    }
    print("Processed tables:")
    for label, status in produced.items():
        print(f"  {label:12s} {'ok' if status else 'missing'}")


if __name__ == "__main__":
    main()
