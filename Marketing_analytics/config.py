"""Configuration models for the marketing analytics template."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import MutableMapping, Optional


DEFAULT_AI_SYSTEM_PROMPT = (
    "You are an experienced marketing analyst who writes concise, executive-ready insights."
)

DEFAULT_AI_PROMPT_TEMPLATE = """
Dataset context:

Overall metrics:
{overall_metrics}

Top campaigns (sample):
{campaign_table}

Top channels (sample):
{channel_table}

Audience segments (sample):
{segment_table}

Product leaders (sample):
{product_table}

Creative performance (sample):
{creative_table}

Post-purchase and margin diagnostics (sample):
{margin_table}

Timeline summary (sample):
{timeline_table}

Customer value distribution (sample):
{customer_value_table}

Tasks for this analysis:
1. Identify the three biggest opportunities for growth and the levers behind them.
2. Determine whether revenue is accelerating or slowing once you account for seasonality and marketing investment.
3. Describe the customer segments, call out best-performing groups, and suggest how to find more of them.
4. Surface hero SKUs/products that pull demand and highlight inventory implications heading into the next two quarters.
5. Diagnose margin headwinds post-purchase (returns, exchanges, defects, shipping, discounts) and suggest repairs.
6. Evaluate whether marketing spend is working overall and which creatives actually move product.
7. Close with specific next actions the team should validate next.
Keep the tone confident, data-driven, and avoid hallucinated numbers.
"""


@dataclass(slots=True)
class ColumnMapping:
    """Describe how raw CSV columns map onto template concepts."""

    customer_id: str = "customer_id"
    event_timestamp: Optional[str] = "event_date"
    campaign: Optional[str] = "campaign_name"
    channel: Optional[str] = "channel"
    segment: Optional[str] = "segment"
    response: Optional[str] = "converted"
    revenue: Optional[str] = "revenue"
    spend: Optional[str] = "spend"
    touch_value: Optional[str] = "engagement_score"
    order_value: Optional[str] = "order_value"
    control_flag: Optional[str] = "is_control"
    product_name: Optional[str] = None
    product_sku: Optional[str] = None
    product_category: Optional[str] = None
    units: Optional[str] = None
    creative_name: Optional[str] = None
    creative_id: Optional[str] = None
    cogs: Optional[str] = None
    gross_margin: Optional[str] = None
    discount_amount: Optional[str] = None
    shipping_cost: Optional[str] = None
    refund_amount: Optional[str] = None
    return_flag: Optional[str] = None
    exchange_flag: Optional[str] = None
    defect_flag: Optional[str] = None
    additional_numeric: tuple[str, ...] = ()
    additional_categorical: tuple[str, ...] = ()

    def required_columns(self) -> list[str]:
        return [self.customer_id]

    def known_columns(self) -> list[str]:
        cols: list[str] = [self.customer_id]
        optional = [
            self.event_timestamp,
            self.campaign,
            self.channel,
            self.segment,
            self.response,
            self.revenue,
            self.spend,
            self.touch_value,
            self.order_value,
            self.control_flag,
            self.product_name,
            self.product_sku,
            self.product_category,
            self.units,
            self.creative_name,
            self.creative_id,
            self.cogs,
            self.gross_margin,
            self.discount_amount,
            self.shipping_cost,
            self.refund_amount,
            self.return_flag,
            self.exchange_flag,
            self.defect_flag,
        ]
        cols.extend([col for col in optional if col])
        cols.extend(self.additional_numeric)
        cols.extend(self.additional_categorical)
        seen: set[str] = set()
        deduped: list[str] = []
        for col in cols:
            if col not in seen:
                seen.add(col)
                deduped.append(col)
        return deduped

    def numeric_features(self) -> list[str]:
        features = [
            col
            for col in [
                self.revenue,
                self.spend,
                self.touch_value,
                self.order_value,
                self.units,
                self.cogs,
                self.gross_margin,
                self.discount_amount,
                self.shipping_cost,
                self.refund_amount,
            ]
            if col
        ]
        features.extend(self.additional_numeric)
        seen: set[str] = set()
        deduped: list[str] = []
        for col in features:
            if col not in seen:
                seen.add(col)
                deduped.append(col)
        return deduped

    def categorical_features(self) -> list[str]:
        features = [
            col
            for col in [
                self.campaign,
                self.channel,
                self.segment,
                self.control_flag,
                self.product_name,
                self.product_sku,
                self.product_category,
                self.creative_name,
                self.creative_id,
                self.return_flag,
                self.exchange_flag,
                self.defect_flag,
            ]
            if col
        ]
        features.extend(self.additional_categorical)
        seen: set[str] = set()
        deduped: list[str] = []
        for col in features:
            if col not in seen:
                seen.add(col)
                deduped.append(col)
        return deduped

    def binary_features(self) -> list[str]:
        features = [
            col
            for col in [self.return_flag, self.exchange_flag, self.defect_flag]
            if col
        ]
        return features


@dataclass(slots=True)
class AISummaryConfig:
    """Settings for optional AI-generated narrative summaries."""

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: str = DEFAULT_AI_SYSTEM_PROMPT
    prompt_template: str = DEFAULT_AI_PROMPT_TEMPLATE
    temperature: float = 0.2
    max_output_tokens: int = 800
    max_table_rows: int = 10

    def is_enabled(self) -> bool:
        return self.enabled and bool(self.provider and self.model)


@dataclass(slots=True)
class AnalysisSettings:
    """Execution parameters for the marketing analytics pipeline."""

    data_path: Path
    output_dir: Path = Path("reports")
    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    time_granularity: str = "W"  # pandas frequency alias (e.g., 'D', 'W', 'M')
    minimum_rows: int = 50
    include_modeling: bool = True
    include_visuals: bool = True
    random_seed: int = 42
    ai_summary: AISummaryConfig = field(default_factory=AISummaryConfig)

    def resolve_paths(self) -> None:
        self.data_path = self.data_path.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()

    def ensure_output_tree(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "derived").mkdir(exist_ok=True)


def settings_from_dict(payload: MutableMapping[str, object], *, base_path: Path | None = None) -> AnalysisSettings:
    """Create :class:`AnalysisSettings` from a dictionary (e.g., parsed JSON)."""

    base = base_path or Path.cwd()

    mapping_payload = payload.get("mapping", {}) if isinstance(payload, MutableMapping) else {}
    mapping = ColumnMapping(**mapping_payload) if isinstance(mapping_payload, MutableMapping) else ColumnMapping()

    data_path_value = payload.get("data_path") if isinstance(payload, MutableMapping) else None
    if not data_path_value:
        raise ValueError("`data_path` is required in the configuration payload")

    output_dir_value = payload.get("output_dir") if isinstance(payload, MutableMapping) else None

    ai_payload = payload.get("ai_summary") if isinstance(payload, MutableMapping) else None
    ai_summary = AISummaryConfig()
    if isinstance(ai_payload, MutableMapping):
        ai_kwargs = {
            key: ai_payload.get(key)
            for key in AISummaryConfig.__dataclass_fields__.keys()
            if key in ai_payload
        }
        if "enabled" in ai_kwargs:
            ai_kwargs["enabled"] = bool(ai_kwargs["enabled"])
        ai_summary = AISummaryConfig(**ai_kwargs)

    settings = AnalysisSettings(
        data_path=Path(data_path_value),
        output_dir=Path(output_dir_value) if output_dir_value else Path("reports"),
        mapping=mapping,
        time_granularity=str(payload.get("time_granularity", "W")) if isinstance(payload, MutableMapping) else "W",
        minimum_rows=int(payload.get("minimum_rows", 50)) if isinstance(payload, MutableMapping) else 50,
        include_modeling=bool(payload.get("include_modeling", True)) if isinstance(payload, MutableMapping) else True,
        include_visuals=bool(payload.get("include_visuals", True)) if isinstance(payload, MutableMapping) else True,
        random_seed=int(payload.get("random_seed", 42)) if isinstance(payload, MutableMapping) else 42,
        ai_summary=ai_summary,
    )
    settings.resolve_paths()
    if not settings.data_path.is_absolute():
        settings.data_path = (base / settings.data_path).resolve()
    if not settings.output_dir.is_absolute():
        settings.output_dir = (base / settings.output_dir).resolve()
    return settings
