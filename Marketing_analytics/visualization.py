"""Visualization generation for the marketing analytics template."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from Marketing_analytics.config import AnalysisSettings

sns.set_theme(style="whitegrid")


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timeline_plot(timeline: pd.DataFrame, output_dir: Path) -> Path | None:
    if timeline.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    if "revenue" in timeline.columns:
        sns.lineplot(data=timeline, x="period", y="revenue", label="Revenue", ax=ax)
    if "spend" in timeline.columns:
        sns.lineplot(data=timeline, x="period", y="spend", label="Spend", ax=ax)
    if "conversions" in timeline.columns:
        ax2 = ax.twinx()
        sns.barplot(data=timeline, x="period", y="conversions", alpha=0.3, color="#ff7f0e", ax=ax2)
        ax2.set_ylabel("Conversions")
    ax.set_ylabel("Value")
    ax.set_title("Performance over time")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    output_path = output_dir / "figures" / "timeline.png"
    _save_plot(fig, output_path)
    return output_path


def ranking_plot(frame: pd.DataFrame, *, label: str, filename: str, output_dir: Path) -> Path | None:
    if frame.empty or "dimension" not in frame.columns:
        return None
    metrics_candidates = [col for col in ["revenue", "conversion_count", "records"] if col in frame.columns]
    if not metrics_candidates:
        return None
    metric = metrics_candidates[0]
    top = frame.nlargest(15, metric)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, x=metric, y="dimension", palette="viridis", ax=ax)
    ax.set_ylabel(label)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Top {label.lower()} by {metric.replace('_', ' ')}")
    output_path = output_dir / "figures" / filename
    _save_plot(fig, output_path)
    return output_path


def generate_visuals(
    *,
    settings: AnalysisSettings,
    campaign: pd.DataFrame,
    channel: pd.DataFrame,
    timeline: pd.DataFrame,
) -> Dict[str, str]:
    output_dir = settings.output_dir
    figures: Dict[str, str] = {}

    path = timeline_plot(timeline, output_dir)
    if path:
        figures["timeline"] = path.name

    campaign_path = ranking_plot(campaign, label="Campaign", filename="top_campaigns.png", output_dir=output_dir)
    if campaign_path:
        figures["campaign_ranking"] = campaign_path.name

    channel_path = ranking_plot(channel, label="Channel", filename="top_channels.png", output_dir=output_dir)
    if channel_path:
        figures["channel_ranking"] = channel_path.name

    return figures
