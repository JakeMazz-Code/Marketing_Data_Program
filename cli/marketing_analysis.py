"""General marketing analytics CLI for CSV datasets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict

_bootstrap_spec = importlib.util.spec_from_file_location(
    "cli._bootstrap_runtime",
    Path(__file__).resolve().parent / "_bootstrap.py",
)
if _bootstrap_spec is None or _bootstrap_spec.loader is None:
    raise ModuleNotFoundError("Unable to load CLI bootstrap helper.")
_bootstrap = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap)
_bootstrap.ensure_package_imported()

from Marketing_analytics import (AISummaryConfig, AnalysisSettings, ColumnMapping, MarketingAnalysisPipeline, generate_brief_md, generate_verified_brief)
from Marketing_analytics.daily_master import (
    is_daily_master_config,
    run_daily_master_from_config,
)


def parse_mapping_overrides(pairs: list[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid mapping override '{pair}'. Expected KEY=VALUE format.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() == "none":
            overrides[key] = None
        else:
            overrides[key] = value
    return overrides


def build_settings(args: argparse.Namespace) -> AnalysisSettings:
    mapping = ColumnMapping()
    overrides = parse_mapping_overrides(args.map or [])
    valid_fields = {field.name for field in fields(mapping)}
    for key, value in overrides.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown mapping field '{key}'. Valid options: {', '.join(sorted(valid_fields))}")
        setattr(mapping, key, value)

    if args.extra_numeric:
        mapping.additional_numeric = tuple(args.extra_numeric)
    if args.extra_categorical:
        mapping.additional_categorical = tuple(args.extra_categorical)

    ai_config = AISummaryConfig()
    cli_ai_requested = any(
        [
            args.enable_ai,
            bool(args.ai_provider),
            bool(args.ai_model),
            bool(args.ai_prompt_file),
            bool(args.ai_system_prompt),
        ]
    )
    if cli_ai_requested:
        ai_config.enabled = True
    if args.ai_provider:
        ai_config.provider = args.ai_provider
    if args.ai_model:
        ai_config.model = args.ai_model
    if args.ai_api_key_env:
        ai_config.api_key_env = args.ai_api_key_env
    if args.ai_temperature is not None:
        ai_config.temperature = args.ai_temperature
    if args.ai_max_output_tokens is not None:
        ai_config.max_output_tokens = args.ai_max_output_tokens
    if args.ai_max_table_rows is not None:
        ai_config.max_table_rows = args.ai_max_table_rows
    if args.ai_prompt_file:
        ai_config.prompt_template = args.ai_prompt_file.read_text(encoding="utf-8")
    if args.ai_system_prompt:
        ai_config.system_prompt = args.ai_system_prompt

    settings = AnalysisSettings(
        data_path=args.data,
        output_dir=args.output_dir,
        mapping=mapping,
        time_granularity=args.time_granularity,
        minimum_rows=args.minimum_rows,
        include_modeling=not args.no_model,
        include_visuals=not args.no_visuals,
        random_seed=args.seed,
        ai_summary=ai_config,
    )
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Template pipeline for exploring marketing/customer performance data.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("customer_churn_data.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write reports and derived artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file specifying settings and column mapping.",
    )
    parser.add_argument(
        "--map",
        action="append",
        metavar="KEY=VALUE",
        help="Override column mapping fields (e.g., --map response=converted). Use 'none' to unset.",
    )
    parser.add_argument(
        "--extra-numeric",
        nargs="*",
        default=(),
        help="Additional numeric feature columns to include in modeling.",
    )
    parser.add_argument(
        "--extra-categorical",
        nargs="*",
        default=(),
        help="Additional categorical feature columns to include in modeling.",
    )
    parser.add_argument(
        "--time-granularity",
        default="W",
        help="Pandas resample frequency for timeline summaries (e.g., 'D', 'W', 'M').",
    )
    parser.add_argument(
        "--minimum-rows",
        type=int,
        default=50,
        help="Skip modeling if fewer rows than this threshold.",
    )
    parser.add_argument("--no-model", action="store_true", help="Disable propensity modeling stage.")
    parser.add_argument("--no-visuals", action="store_true", help="Skip chart generation stage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for modeling.")

    parser.add_argument("--enable-ai", action="store_true", help="Generate an AI-authored executive summary (requires API key).")
    parser.add_argument("--ai-provider", choices=["openai", "anthropic"], help="LLM provider to use for AI summaries.")
    parser.add_argument("--ai-model", help="Model identifier for the chosen provider (e.g., gpt-4o-mini).")
    parser.add_argument("--ai-api-key-env", help="Environment variable that stores the provider API key.")
    parser.add_argument("--ai-temperature", type=float, help="Sampling temperature for the AI summary call.")
    parser.add_argument("--ai-max-output-tokens", type=int, help="Maximum tokens for the AI summary output.")
    parser.add_argument("--ai-max-table-rows", type=int, help="Rows from each table to include in the AI prompt context.")
    parser.add_argument("--ai-prompt-file", type=Path, help="Path to a file that overrides the AI prompt template.")
    parser.add_argument("--ai-system-prompt", help="Override the system prompt sent to the AI provider.")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional override for the data root when using daily_master configs.",
    )
    parser.add_argument(
        "--generate-brief",
        action="store_true",
        help="Draft + verify an executive brief from daily master artifacts.",
    )

    parser.add_argument(
        "--md-only",
        action="store_true",
        help="Write a Markdown-only brief without schema or verifier (requires --generate-brief).",
    )

    parser.add_argument(
        "--until",
        choices=["kpis", "write"],
        help="Stop the CLI after generating KPI artifacts or after writes.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )

    return parser.parse_args()

def display_console_summary(results: Dict[str, object]) -> None:
    overall = results.get("overall", {}) or {}
    report_path: Path = results.get("report_path")  # type: ignore[assignment]
    metrics_path: Path = results.get("summary_path")  # type: ignore[assignment]

    print("\nMarketing analytics run completed.\n")

    for key, label in [
        ("customer_count", "Customers"),
        ("conversion_count", "Conversions"),
        ("conversion_rate", "Conversion rate"),
        ("total_revenue", "Revenue"),
        ("total_spend", "Spend"),
        ("net_margin", "Net margin"),
        ("refund_amount", "Refunds"),
        ("discount_amount", "Discounts"),
        ("roi", "ROI"),
    ]:
        value = overall.get(key)
        if value is None:
            continue
        if key.endswith("rate") or key == "roi":
            formatted = f"{value:.2%}"
        elif "revenue" in key or "spend" in key or "margin" in key or "discount" in key or "refund" in key:
            formatted = f"${value:,.2f}"
        else:
            formatted = f"{value:,.0f}"
        print(f"  {label:20s} {formatted}")

    print("\nArtifacts:")
    print(f"  Markdown report: {report_path}")
    print(f"  Metrics summary: {metrics_path}")
    figures = results.get("figures", {})
    if figures:
        for name, filename in figures.items():
            print(f"  Figure ({name}): {filename}")
    ai_summary_path = results.get("ai_summary_path")
    if ai_summary_path:
        print(f"  AI insight markdown: {ai_summary_path}")


def main() -> None:
    args = parse_args()

    if args.md_only and not args.generate_brief:
        raise RuntimeError("--md-only requires --generate-brief.")

    if args.config and is_daily_master_config(args.config):
        # Daily Master mode: generate canonical artifacts only
        output = run_daily_master_from_config(args.config, data_root_override=args.data_root)
        print("\nDaily master artifacts written:")
        for k, v in output.items():
            print(f"  {k:18s} {v}")
        if args.generate_brief:
            if args.md_only:
                brief_paths = generate_brief_md(str(args.config))
                print(f"  brief_md        {brief_paths['brief_md']}")
                print(f"  brief_raw       {brief_paths['brief_raw']}")
                return
            else:
                brief_paths = generate_verified_brief(str(args.config))
                if 'brief_md' in brief_paths:
                    print(f"  brief_md        {brief_paths['brief_md']}")
                    print(f"  brief_raw       {brief_paths['brief_raw']}")
                else:
                    print(f"  brief_draft      {brief_paths['draft']}")
                    print(f"  brief_verified   {brief_paths['verified']}")
                    print(f"  brief_notes      {brief_paths['notes']}")
    else:
        if args.generate_brief:
            raise RuntimeError("--generate-brief requires --config pointing to a daily_master JSON.")
        if args.config:
            pipeline = MarketingAnalysisPipeline.from_config_file(args.config)
        else:
            settings = build_settings(args)
            pipeline = MarketingAnalysisPipeline(settings)

        results = pipeline.run()
        display_console_summary(results)


if __name__ == "__main__":
    main()






