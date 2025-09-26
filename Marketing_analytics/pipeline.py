"""High-level pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from Marketing_analytics.ai import AISummarizer
from Marketing_analytics.config import AnalysisSettings, ColumnMapping, settings_from_dict
from Marketing_analytics.data_loader import DatasetBundle, add_derived_columns, load_dataset
from Marketing_analytics.metrics import (
    campaign_performance,
    channel_performance,
    creative_performance,
    customer_value_distribution,
    margin_diagnostics,
    overall_snapshot,
    product_performance,
    segment_performance,
    time_series_summary,
)
from Marketing_analytics.models import PropensityResults, train_propensity_model
from Marketing_analytics.reporting import (
    build_markdown_report,
    build_summary_payload,
    dataframe_to_csv,
)
from Marketing_analytics.visualization import generate_visuals


class MarketingAnalysisPipeline:
    """Run the end-to-end marketing analytics template."""

    def __init__(self, settings: AnalysisSettings) -> None:
        self.settings = settings
        self.settings.resolve_paths()
        self.settings.ensure_output_tree()

    @classmethod
    def from_config_file(cls, path: Path) -> "MarketingAnalysisPipeline":
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = settings_from_dict(payload, base_path=path.parent)
        return cls(settings)

    def _prepare_dataset(self) -> DatasetBundle:
        raw = load_dataset(self.settings.data_path, self.settings.mapping)
        prepared = add_derived_columns(raw)
        return prepared

    def run(self) -> Dict[str, object]:
        bundle = self._prepare_dataset()
        df = bundle.frame
        output_dir = self.settings.output_dir

        overall = overall_snapshot(bundle)
        overall["rows"] = int(len(df))

        campaign = campaign_performance(bundle, top_n=None)
        channel = channel_performance(bundle, top_n=None)
        segment = segment_performance(bundle, top_n=None)
        product = product_performance(bundle, top_n=None)
        creative = creative_performance(bundle, top_n=None)
        margin = margin_diagnostics(bundle)
        timeline = time_series_summary(bundle, freq=self.settings.time_granularity)
        customer_value = customer_value_distribution(bundle)

        should_model = self.settings.include_modeling and len(df) >= self.settings.minimum_rows

        propensity: Optional[PropensityResults] = None
        if should_model:
            propensity = train_propensity_model(bundle, random_seed=self.settings.random_seed)

        figures = (
            generate_visuals(
                settings=self.settings,
                campaign=campaign,
                channel=channel,
                timeline=timeline,
            )
            if self.settings.include_visuals
            else {}
        )

        ai_summary = None
        ai_summary_path = None
        if self.settings.ai_summary.is_enabled():
            summarizer = AISummarizer(self.settings.ai_summary)
            ai_summary = summarizer.generate_summary(
                overall=overall,
                campaign=campaign,
                channel=channel,
                segment=segment,
                product=product,
                creative=creative,
                margin=margin,
                timeline=timeline,
                customer_value=customer_value,
            )
            if ai_summary and ai_summary.get("markdown"):
                ai_summary_path = output_dir / "ai_insights.md"
                ai_summary_path.write_text(ai_summary["markdown"], encoding="utf-8")

        summary_payload = build_summary_payload(
            settings=self.settings,
            bundle=bundle,
            overall=overall,
            campaign=campaign,
            channel=channel,
            segment=segment,
            product=product,
            creative=creative,
            margin=margin,
            timeline=timeline,
            customer_value=customer_value,
            model=propensity,
            ai_summary=ai_summary,
            ai_summary_path=ai_summary_path,
        )

        report_text = build_markdown_report(
            settings=self.settings,
            overall=overall,
            campaign=campaign,
            channel=channel,
            segment=segment,
            product=product,
            creative=creative,
            margin=margin,
            timeline=timeline,
            model=propensity,
            ai_summary=ai_summary["markdown"] if ai_summary else None,
        )

        metrics_path = output_dir / "metrics_summary.json"
        metrics_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        report_path = output_dir / "marketing_analysis_report.md"
        report_path.write_text(report_text, encoding="utf-8")

        dataframe_to_csv(campaign, output_dir / "campaign_performance.csv")
        dataframe_to_csv(channel, output_dir / "channel_performance.csv")
        dataframe_to_csv(segment, output_dir / "segment_performance.csv")
        dataframe_to_csv(product, output_dir / "product_performance.csv")
        dataframe_to_csv(creative, output_dir / "creative_performance.csv")
        dataframe_to_csv(margin, output_dir / "margin_diagnostics.csv")
        dataframe_to_csv(timeline, output_dir / "timeline_summary.csv")
        dataframe_to_csv(customer_value, output_dir / "derived" / "customer_value_distribution.csv")

        derived_dir = output_dir / "derived"
        if propensity:
            dataframe_to_csv(propensity.feature_importance, derived_dir / "propensity_feature_importance.csv")
            dataframe_to_csv(propensity.holdout_predictions, derived_dir / "propensity_holdout_predictions.csv")

        settings_snapshot = asdict(self.settings)
        settings_snapshot["data_path"] = str(self.settings.data_path)
        settings_snapshot["output_dir"] = str(self.settings.output_dir)

        return {
            "settings": settings_snapshot,
            "overall": overall,
            "campaign": campaign,
            "channel": channel,
            "segment": segment,
            "product": product,
            "creative": creative,
            "margin": margin,
            "timeline": timeline,
            "customer_value": customer_value,
            "propensity": propensity,
            "figures": figures,
            "summary_path": metrics_path,
            "report_path": report_path,
            "modeling_executed": propensity is not None,
            "modeling_threshold": self.settings.minimum_rows,
            "ai_summary": ai_summary,
            "ai_summary_path": ai_summary_path,
        }
