"""Batch runner for the marketing analytics template."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_bootstrap_spec = importlib.util.spec_from_file_location(
    "cli._bootstrap_runtime",
    Path(__file__).resolve().parent / "_bootstrap.py",
)
if _bootstrap_spec is None or _bootstrap_spec.loader is None:
    raise ModuleNotFoundError("Unable to load CLI bootstrap helper.")
_bootstrap = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap)
_bootstrap.ensure_package_imported()

from Marketing_analytics import AISummaryConfig, AnalysisSettings, ColumnMapping, MarketingAnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one or more marketing analytics runs.")
    parser.add_argument(
        "--data",
        type=Path,
        help="CSV file to analyze using default settings (or when configs are not provided).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports"),
        help="Root directory where scenario subfolders will be written.",
    )
    parser.add_argument(
        "--scenario-name",
        help="Optional name for the scenario when using --data (creates a subfolder).",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Path to a JSON configuration file (can be provided multiple times).",
    )
    parser.add_argument("--enable-ai", action="store_true", help="Generate AI summaries for ad-hoc runs.")
    parser.add_argument("--ai-provider", choices=["openai", "anthropic"], help="LLM provider used when AI summaries are enabled.")
    parser.add_argument("--ai-model", help="Model identifier for the selected AI provider.")
    parser.add_argument("--ai-api-key-env", help="Environment variable containing the provider API key.")
    parser.add_argument("--ai-temperature", type=float, help="Sampling temperature for AI summaries.")
    parser.add_argument("--ai-max-output-tokens", type=int, help="Maximum tokens for AI summary responses.")
    parser.add_argument("--ai-max-table-rows", type=int, help="Rows from each table to include in AI prompts.")
    parser.add_argument("--ai-prompt-file", type=Path, help="Prompt template override file for AI summaries.")
    parser.add_argument("--ai-system-prompt", help="Custom system message for the AI provider.")
    return parser.parse_args()


def run_pipeline(pipeline: MarketingAnalysisPipeline) -> None:
    results = pipeline.run()
    settings = pipeline.settings
    print(f"\nRun completed for {settings.data_path} -> {settings.output_dir}")

    overall = results.get("overall", {})
    if overall:
        for key in ("customer_count", "conversion_count", "total_revenue", "total_spend", "roi"):
            value = overall.get(key)
            if value is None:
                continue
            if key == "roi":
                formatted = f"{value:.2%}"
            elif "revenue" in key or "spend" in key:
                formatted = f"${value:,.2f}"
            else:
                formatted = f"{value:,.0f}"
            print(f"  {key.replace('_', ' ').title():25s} {formatted}")
    print(f"  Summary JSON: {results['summary_path']}")
    ai_path = results.get("ai_summary_path")
    if ai_path:
        print(f"  AI insight markdown: {ai_path}")


def _build_ai_config(args: argparse.Namespace) -> AISummaryConfig:
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
    return ai_config


def main() -> None:
    args = parse_args()
    executed = False

    if args.config:
        for config_path in args.config:
            pipeline = MarketingAnalysisPipeline.from_config_file(config_path)
            run_pipeline(pipeline)
            executed = True

    if args.data:
        output_dir = args.output_root / (args.scenario_name or "ad_hoc")
        settings = AnalysisSettings(
            data_path=args.data,
            output_dir=output_dir,
            mapping=ColumnMapping(),
            ai_summary=_build_ai_config(args),
        )
        run_pipeline(MarketingAnalysisPipeline(settings))
        executed = True

    if not executed:
        raise SystemExit("No work to execute. Provide --data or at least one --config file.")


if __name__ == "__main__":
    main()

