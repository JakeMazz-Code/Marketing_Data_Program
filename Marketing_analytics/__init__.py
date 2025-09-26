"""Public API for the Marketing_analytics package."""

from .ai import AISummarizer, generate_brief_md, generate_verified_brief
from .config import AISummaryConfig, AnalysisSettings, ColumnMapping
from .metrics_registry import compute_one, compute_series, list_metrics
from .pipeline import MarketingAnalysisPipeline

__all__ = [
    "AISummarizer",
    "AISummaryConfig",
    "AnalysisSettings",
    "ColumnMapping",
    "MarketingAnalysisPipeline",
    "compute_one",
    "compute_series",
    "generate_brief_md",
    "generate_verified_brief",
    "list_metrics",
]

import sys as _sys

_module = _sys.modules[__name__]
_sys.modules["marketing_analytics"] = _module
