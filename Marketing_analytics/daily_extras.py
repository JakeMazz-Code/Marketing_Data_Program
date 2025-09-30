"""Supplemental artifact generation for the daily master pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Marketing_analytics.etl_text_clean import normalize_product_title, split_product_fields


_RAW_FIELD_SYNONYMS: Dict[str, set[str]] = {
    'date': {'date', 'day', 'created_at', 'processed_at', 'order_date', 'order_created_at', 'transaction_date', 'reporting_date'},
    'channel': {'channel', 'platform', 'source', 'network'},
    'campaign': {'campaign', 'campaign_name', 'ad_name', 'campaign_title', 'adgroup', 'adgroup_name', 'ad_group', 'ad_group_name'},
    'adset': {'adset', 'adset_name', 'ad_set'},
    'ad': {'ad', 'ad_name'},
    'creative_id': {'creative_id', 'post_id', 'ad_id', 'ig_id', 'tiktok_id', 'asset_id'},
    'sku': {'sku', 'product_sku', 'lineitem_sku', 'product_id', 'variant_sku'},
    'product_title': {'product_title', 'lineitem_title', 'lineitem_name', 'product_name', 'title'},
    'units': {'units', 'quantity', 'qty', 'lineitem_quantity', 'units_sold'},
    'orders': {'orders', 'purchases', 'purchases_website', 'purchases_app', 'conversions', 'order_count'},
    'revenue': {'revenue', 'purchase_value', 'purchase_value_website', 'conversion_value', 'total_made', 'gross_sales', 'net_sales', 'line_total', 'lineitem_total', 'lineitem_price_total', 'subtotal_price', 'sales', 'value'},
    'discounts': {'discounts', 'sales_discounts', 'promo_value', 'lineitem_discount'},
    'returns': {'returns', 'sales_returns', 'refunds', 'refund_amount'},
    'shipping': {'shipping', 'sales_shipping', 'shipping_revenue', 'shipping_cost'},
    'taxes': {'taxes', 'sales_taxes'},
    'duties': {'duties', 'sales_duties'},
    'ad_spend': {'ad_spend', 'spend', 'cost'},
    'views': {'views', 'impressions', 'video_views', 'plays'},
    'clicks': {'clicks', 'link_clicks'},
    'reach': {'reach', 'unique_reach'},
    'likes': {'likes', 'reactions'},
    'comments': {'comments'},
    'shares': {'shares', 'reposts'},
    'follows': {'follows', 'followers', 'subs'},
    'saves': {'saves', 'bookmarks'},
    'on_hand': {'on_hand', 'inventory_on_hand', 'qty_on_hand', 'stock_on_hand'},
    'on_order': {'on_order', 'inventory_on_order', 'incoming', 'po_on_order'},
    'lead_time_days': {'lead_time_days', 'lead_time', 'lead_time_in_days'},
}


_NUMERIC_FIELDS = {
    'revenue', 'ad_spend', 'orders', 'units', 'views', 'clicks', 'discounts', 'returns',
    'shipping', 'taxes', 'duties', 'reach', 'likes', 'comments', 'shares', 'follows', 'saves',
    'on_hand', 'on_order', 'lead_time_days'
}
_ALLOWED_NEGATIVE = {'discounts', 'returns'}

_CHANNEL_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ('Meta', ('meta', 'facebook', 'instagram', 'ig', 'fb')),
    ('TikTok', ('tiktok', 'tt')),
    ('Email/SMS', ('email', 'sms', 'klaviyo', 'mailchimp', 'attentive', 'postscript')),
    ('Organic', ('organic', 'direct', 'seo', 'site', 'owned', 'brand')),
]


def _normalize_column_name(name: str) -> str:
    lowered = str(name).strip().lower()
    lowered = lowered.replace('%', 'pct')
    tokens = []
    current = []
    for char in lowered:
        if char.isalnum():
            current.append(char)
        else:
            if current:
                tokens.append(''.join(current))
                current = []
            if char in {' ', '-', '.', '/'}:
                continue
    if current:
        tokens.append(''.join(current))
    if not tokens:
        return lowered
    return '_'.join(tokens)


def _build_lookup() -> Tuple[Dict[str, str], Dict[str, set[str]]]:
    lookup: Dict[str, str] = {}
    expanded: Dict[str, set[str]] = {}
    for canonical, names in _RAW_FIELD_SYNONYMS.items():
        norm_names = {_normalize_column_name(canonical)}
        norm_names.update({_normalize_column_name(name) for name in names})
        expanded[canonical] = norm_names
        for alias in norm_names:
            lookup.setdefault(alias, canonical)
    return lookup, expanded


_FIELD_LOOKUP, _FIELD_CANONICAL = _build_lookup()

def _prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    df = raw.copy()
    normalized_columns = [_normalize_column_name(col) for col in df.columns]
    df.columns = normalized_columns
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        canonical = _FIELD_LOOKUP.get(col)
        if canonical:
            rename_map[col] = canonical
            continue
        tokens = [token for token in col.split('_') if token]
        for size in range(min(3, len(tokens)), 0, -1):
            candidate = '_'.join(tokens[-size:])
            canonical = _FIELD_LOOKUP.get(candidate)
            if canonical:
                rename_map[col] = canonical
                break
        if col not in rename_map:
            for token in reversed(tokens):
                canonical = _FIELD_LOOKUP.get(token)
                if canonical:
                    rename_map[col] = canonical
                    break
    if rename_map:
        df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        allow_negative = col in _ALLOWED_NEGATIVE
        series = df[col]
        if series.dtype.kind in {'i', 'u', 'f'}:
            values = pd.to_numeric(series, errors='coerce')
        else:
            text = series.astype(str).str.strip()
            text = text.replace({'': np.nan, 'none': np.nan, 'nan': np.nan, 'null': np.nan})
            text = text.str.replace(r'[,$]', '', regex=True)
            text = text.str.replace(r'\(([^)]+)\)', r'-\1', regex=True)
            text = text.str.replace(r'\s+', '', regex=True)
            values = pd.to_numeric(text, errors='coerce')
        if not allow_negative:
            values = values.where(values >= 0)
        df[col] = values.astype(float)


def _coerce_date(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype='datetime64[ns]')
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    parsed = parsed.dt.tz_localize(None)
    return parsed


def _normalize_channel(value: Optional[str]) -> str:
    if value is None:
        return 'Other'
    text = str(value).strip()
    if not text:
        return 'Other'
    lowered = text.lower()
    for canonical, keywords in _CHANNEL_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return canonical
    return text.title()


def _safe_ratio(numerator: pd.Series | pd.DataFrame | float, denominator: pd.Series | pd.DataFrame | float, *, allow_inf: bool = False):
    num = numerator.astype(float) if isinstance(numerator, (pd.Series, pd.DataFrame)) else float(numerator)
    den = denominator.astype(float) if isinstance(denominator, (pd.Series, pd.DataFrame)) else float(denominator)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / den
    if isinstance(result, (pd.Series, pd.DataFrame)):
        if not allow_inf:
            result = result.replace([np.inf, -np.inf], np.nan)
        else:
            result = result.mask(~np.isfinite(result) & (num <= 0), np.nan)
    else:
        if not allow_inf and not np.isfinite(result):
            result = np.nan
        elif allow_inf and not np.isfinite(result) and num <= 0:
            result = np.nan
    return result


def _format_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        return df
    result = df.copy()
    result['date'] = pd.to_datetime(result['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return result


def _write_jsonl(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient='records')
    with path.open('w', encoding='utf-8') as fh:
        for record in records:
            clean: Dict[str, object] = {}
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    clean[key] = value.strftime('%Y-%m-%d')
                elif isinstance(value, (np.integer, int)):
                    clean[key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    clean[key] = None if pd.isna(value) else float(value)
                elif value is None:
                    clean[key] = None
                else:
                    clean[key] = value
            fh.write(json.dumps(clean) + '\n')


def _compute_rows_28d(df: pd.DataFrame) -> Tuple[int, Optional[pd.Timestamp]]:
    if 'date' not in df.columns:
        return len(df), None
    dates = pd.to_datetime(df['date'], errors='coerce')
    dates = dates.dropna()
    if dates.empty:
        return 0, None
    last_date = dates.max()
    cutoff = last_date - pd.Timedelta(days=27)
    return int((dates >= cutoff).sum()), last_date

@dataclass
class SourceFrame:
    path: Path
    frame: pd.DataFrame
    sheet: Optional[str] = None

    @property
    def label(self) -> str:
        if self.sheet:
            return f"{self.path.name}:{self.sheet}"
        return self.path.name


@dataclass
class ArtifactResult:
    name: str
    filename: str
    path: Optional[Path] = None
    rows: int = 0
    rows_28d: int = 0
    last_date: Optional[str] = None
    columns: Sequence[str] = ()
    missing_reason: Optional[str] = None
    frame: Optional[pd.DataFrame] = None


def _normalise_sku_code(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def _augment_sku_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if df.empty or 'date' not in df.columns:
        return df, False
    working = df.copy()
    working['date_ts'] = pd.to_datetime(working['date'], errors='coerce')
    working = working.dropna(subset=['date_ts'])
    if working.empty:
        return df, False

    sku_series = working.get('sku_code')
    if sku_series is None:
        sku_series = working.get('sku')
    title_series = working.get('product_title')

    keys: List[object] = []
    grouped_by_title = False
    for idx in range(len(working)):
        sku_val = sku_series.iloc[idx] if sku_series is not None else None
        if sku_val is None or (isinstance(sku_val, str) and not sku_val.strip()) or (isinstance(sku_val, float) and pd.isna(sku_val)):
            grouped_by_title = True
            if title_series is not None:
                keys.append(title_series.iloc[idx])
            else:
                keys.append('unknown')
        else:
            keys.append(sku_val)
    working['_key'] = pd.Series(keys).fillna('unknown').astype(str)
    working = working.sort_values(['_key', 'date_ts'])

    if 'units' in working.columns:
        units_roll = (
            working.set_index('date_ts')
            .groupby('_key')['units']
            .rolling('28D', min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        working['units_28d'] = units_roll.values
        working['run_rate_units_28d'] = working['units_28d'] / 28.0
    else:
        working['units_28d'] = np.nan
        working['run_rate_units_28d'] = np.nan

    if 'revenue' in working.columns:
        revenue_roll = (
            working.set_index('date_ts')
            .groupby('_key')['revenue']
            .rolling('28D', min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        working['revenue_28d'] = revenue_roll.values
    else:
        working['revenue_28d'] = np.nan

    if 'revenue' in working.columns:
        totals = working[['date_ts', 'revenue']].groupby('date_ts').sum()
        totals['revenue_28d_total'] = totals['revenue'].rolling('28D', min_periods=1).sum()
        working = working.merge(totals[['revenue_28d_total']], left_on='date_ts', right_index=True, how='left')
        working['revenue_share_28d'] = _safe_ratio(working['revenue_28d'], working['revenue_28d_total'])
    else:
        working['revenue_share_28d'] = np.nan

    if 'units' in working.columns:
        totals_units = working[['date_ts', 'units']].groupby('date_ts').sum()
        totals_units['units_28d_total'] = totals_units['units'].rolling('28D', min_periods=1).sum()
        working = working.merge(totals_units[['units_28d_total']], left_on='date_ts', right_index=True, how='left')
        working['units_share_28d'] = _safe_ratio(working['units_28d'], working['units_28d_total'])
    else:
        working['units_share_28d'] = np.nan

    update_cols = ['revenue_28d', 'units_28d', 'revenue_share_28d', 'units_share_28d', 'run_rate_units_28d']
    enriched = df.copy()
    for col in update_cols:
        enriched[col] = np.nan
    for col in update_cols:
        enriched.loc[working.index, col] = working[col]
    return enriched, grouped_by_title


def _read_delimited_with_fallback(path: Path, *, sep: str) -> pd.DataFrame:
    encodings = ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1')
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f'Unable to read {path} with common encodings: {last_error}')


def _infer_channel_from_label(label: str) -> str:
    lowered = label.lower()
    for canonical, keywords in _CHANNEL_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return canonical
    return 'Other'

class ExtraArtifactGenerator:
    def __init__(self, data_root: Path, artifacts_dir: Path, *, line_item_globs: Optional[Sequence[str]] = None) -> None:
        self.data_root = Path(data_root)
        self.artifacts_dir = Path(artifacts_dir)
        self.line_item_globs = tuple(line_item_globs or [])
        self._title_grouping_warned = False

    def run(self) -> Dict[str, ArtifactResult]:
        sources = self._discover_sources()
        results: Dict[str, ArtifactResult] = {}

        sku_result = self._build_and_write('sku_series.jsonl', self._build_sku_series(sources))
        results['sku_series'] = sku_result

        channel_result = self._build_and_write('channel_series.jsonl', self._build_channel_series(sources))
        results['channel_series'] = channel_result

        campaign_result = self._build_and_write('campaign_series.jsonl', self._build_campaign_series(sources))
        results['campaign_series'] = campaign_result

        creative_result = self._build_and_write('creative_series.jsonl', self._build_creative_series(sources))
        results['creative_series'] = creative_result

        margin_result = self._build_and_write('margin_series.jsonl', self._build_margin_series(sources))
        results['margin_series'] = margin_result

        inventory_result = self._build_and_write('inventory_snapshot.jsonl', self._build_inventory_snapshot(sources))
        results['inventory_snapshot'] = inventory_result

        efficiency_result = self._write_efficiency_summary(channel_result, creative_result)
        if efficiency_result is not None:
            results['efficiency'] = efficiency_result

        return results

    def _discover_sources(self) -> List[SourceFrame]:
        if not self.data_root.exists():
            return []
        candidates: List[Path] = []
        seen: set[Path] = set()

        for pattern in self.line_item_globs:
            for match in glob(pattern, recursive=True):
                path = Path(match)
                if path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        candidates.append(path)
                        seen.add(resolved)

        for path in sorted(self.data_root.rglob('*')):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if path.suffix.lower() not in {'.csv', '.tsv', '.xlsx', '.xls'}:
                continue
            candidates.append(path)
            seen.add(resolved)

        sources: List[SourceFrame] = []
        for path in candidates:
            suffix = path.suffix.lower()
            try:
                if suffix in {'.csv', '.tsv'}:
                    sep = ',' if suffix == '.csv' else '\t'
                    frame = _read_delimited_with_fallback(path, sep=sep)
                    prepared = _prepare_frame(frame)
                    if not prepared.empty:
                        sources.append(SourceFrame(path=path, frame=prepared))
                else:
                    workbook = pd.read_excel(path, sheet_name=None)
                    for sheet_name, frame in workbook.items():
                        prepared = _prepare_frame(frame)
                        if not prepared.empty:
                            sources.append(SourceFrame(path=path, frame=prepared, sheet=sheet_name))
            except Exception:
                continue
        return sources

    def _build_and_write(self, filename: str, payload: Tuple[pd.DataFrame, Optional[str], Dict[str, object]]) -> ArtifactResult:
        name = filename.replace('.jsonl', '')
        df, missing_reason, metadata = payload
        result = ArtifactResult(name=name, filename=filename)
        if missing_reason:
            result.missing_reason = missing_reason
            return result
        if df.empty:
            result.missing_reason = 'no_rows'
            return result

        df_sorted = df.sort_values('date') if 'date' in df.columns else df.copy()
        rows_28d, last_ts = _compute_rows_28d(df_sorted)
        formatted = _format_date_column(df_sorted)
        path = self.artifacts_dir / filename
        _write_jsonl(path, formatted)

        result.path = path
        result.rows = len(formatted)
        result.rows_28d = rows_28d
        result.columns = list(formatted.columns)
        result.frame = df_sorted
        if last_ts is not None:
            result.last_date = last_ts.strftime('%Y-%m-%d')
        elif 'date' in formatted.columns:
            last_date = formatted['date'].dropna().max()
            if isinstance(last_date, str):
                result.last_date = last_date
        if name == 'sku_series':
            grouped_by_title = bool(metadata.get('grouped_by_title'))
            if grouped_by_title and not self._title_grouping_warned:
                print('SKU artifact: title-based grouping (SKU code missing); using product_title.')
                self._title_grouping_warned = True
            as_of = result.last_date or 'n/a'
            titles = formatted['product_title'].dropna().nunique() if 'product_title' in formatted.columns else 0
            raw_sku_count = int(metadata.get('raw_sku_count', 0))
            print(f"Wrote sku_series.jsonl: {result.rows} rows (as_of={as_of}, titles={titles}, sku_codes_present={raw_sku_count})")

        return result

    def _infer_sku_source(self, source: SourceFrame) -> str:
        label = source.label.lower()
        if 'orders_export' in label or 'shopify' in label:
            return 'shopify_orders_export'
        return 'line_items'

    def _build_sku_series(self, sources: List[SourceFrame]) -> Tuple[pd.DataFrame, Optional[str], Dict[str, object]]:
        frames: List[pd.DataFrame] = []
        grouped_by_title = False
        raw_sku_count = 0
        for source in sources:
            df = source.frame
            if 'date' not in df.columns:
                continue
            if 'sku' not in df.columns and 'product_title' not in df.columns:
                continue
            subset_cols = ['date']
            for col in ['sku', 'product_title', 'units', 'orders', 'revenue', 'discounts', 'lineitem_price', 'unit_price', 'price']:
                if col in df.columns:
                    subset_cols.append(col)
            subset_cols = list(dict.fromkeys(subset_cols))
            subset = df[subset_cols].copy()
            subset['date'] = _coerce_date(subset['date'])
            subset = subset.dropna(subset=['date'])
            if subset.empty:
                continue

            if 'sku' in subset.columns:
                raw_sku_count += int(subset['sku'].notna().sum())

            raw_titles = subset.get('product_title')
            if raw_titles is None:
                raw_titles = subset.get('sku')
            bases_variants = [split_product_fields(value) for value in raw_titles]
            subset['product_title'] = [normalize_product_title(base) for base, _ in bases_variants]
            subset['variant'] = [variant.strip() if isinstance(variant, str) and variant.strip() else None for _, variant in bases_variants]
            subset['variant'] = subset['variant'].fillna('Base')

            if 'sku' in subset.columns:
                subset['sku_code'] = subset['sku'].apply(_normalise_sku_code)
            else:
                subset['sku_code'] = pd.Series([None] * len(subset), index=subset.index, dtype='object')

            sku_missing_mask = subset['sku_code'].isna()
            if sku_missing_mask.all():
                grouped_by_title = True
            elif sku_missing_mask.any():
                grouped_by_title = True

            if sku_missing_mask.any():
                fallback_titles = subset['product_title'].fillna('Unnamed Product')
                subset.loc[sku_missing_mask, 'sku_code'] = fallback_titles[sku_missing_mask].apply(
                    lambda value: normalize_product_title(value) or 'Unnamed Product'
                )

            for col in ['units', 'orders', 'revenue', 'discounts', 'lineitem_price', 'unit_price', 'price']:
                if col in subset.columns:
                    _coerce_numeric(subset, [col])

            if 'units' not in subset.columns and 'orders' in subset.columns:
                subset['units'] = subset['orders']
            elif 'units' in subset.columns and 'orders' in subset.columns:
                subset['units'] = subset['units'].fillna(subset['orders'])

            if 'orders' not in subset.columns:
                subset['orders'] = 0.0
            if 'units' not in subset.columns:
                subset['units'] = 0.0
            if 'sku' not in subset.columns:
                subset['sku'] = np.nan

            price_col = next((col for col in ['lineitem_price', 'unit_price', 'price'] if col in subset.columns), None)
            net = subset.get('revenue')
            if price_col is not None and 'units' in subset.columns:
                gross = subset[price_col].fillna(0.0) * subset['units'].fillna(0.0)
                net_candidate = gross
                if 'discounts' in subset.columns:
                    net_candidate = net_candidate - subset['discounts'].fillna(0.0)
                net = net.fillna(net_candidate) if net is not None else net_candidate
            subset['revenue'] = net.fillna(0.0) if net is not None else 0.0
            subset['revenue'] = subset['revenue'].clip(lower=0.0)

            subset['source'] = self._infer_sku_source(source)
            frames.append(subset[['date', 'sku_code', 'sku', 'product_title', 'variant', 'units', 'orders', 'revenue', 'source']])

        if not frames:
            return pd.DataFrame(), 'missing: sku/product_title or units/orders', {'grouped_by_title': False}

        combined = pd.concat(frames, ignore_index=True)
        grouped = combined.groupby(['date', 'sku_code', 'product_title', 'variant', 'source'], dropna=False).agg({
            'units': 'sum',
            'orders': 'sum',
            'revenue': 'sum',
            'sku': 'first',
        }).reset_index()
        if 'sku' not in grouped.columns:
            grouped['sku'] = np.nan
        enriched, grouping_flag = _augment_sku_metrics(grouped)
        metadata = {
            'grouped_by_title': grouped_by_title or grouping_flag,
            'raw_sku_count': int(raw_sku_count),
        }
        return enriched, None, metadata
    def _combine_frames(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if 'date' in combined.columns:
            combined['date'] = _coerce_date(combined['date'])
            combined = combined.dropna(subset=['date'])
        return combined

    def _build_channel_series(self, sources: List[SourceFrame]) -> Tuple[pd.DataFrame, Optional[str], Dict[str, object]]:
        frames: List[pd.DataFrame] = []
        for source in sources:
            df = source.frame
            if 'date' not in df.columns:
                continue
            metrics = [col for col in ['revenue', 'ad_spend', 'orders', 'views', 'clicks'] if col in df.columns]
            if not metrics:
                continue
            subset_cols = ['date'] + metrics
            if 'channel' in df.columns:
                subset_cols.append('channel')
            subset_cols = list(dict.fromkeys(subset_cols))
            subset = df[subset_cols].copy()
            subset['date'] = _coerce_date(subset['date'])
            for col in metrics:
                _coerce_numeric(subset, [col])
            channel_guess = _infer_channel_from_label(source.label)
            if 'channel' in subset.columns:
                subset['channel'] = subset['channel'].apply(_normalize_channel)
                if subset['channel'].isna().all():
                    subset['channel'] = channel_guess
                # If channel column appears numeric/empty, coerce to 'Other' and warn
                ch_series = subset['channel']
                looks_numeric = False
                if pd.api.types.is_numeric_dtype(ch_series):
                    looks_numeric = True
                else:
                    non_null = ch_series.dropna().astype(str).str.strip()
                    if not non_null.empty and (non_null.str.match(r'^\d+(?:\.\d+)?$').mean() > 0.9):
                        looks_numeric = True
                if looks_numeric:
                    examples = list(pd.unique(ch_series.dropna().astype(str)))[:5]
                    if examples:
                        print(f"Channel column appears numeric; coercing to 'Other'. Examples: {examples}")
                    subset['channel'] = 'Other'
            else:
                subset['channel'] = channel_guess
            frames.append(subset)
        combined = self._combine_frames(frames)
        if combined.empty:
            return combined, 'missing: channel metrics (revenue/ad_spend/orders)', {}
        for col in ['revenue', 'ad_spend', 'orders', 'views', 'clicks']:
            if col not in combined.columns:
                combined[col] = np.nan
        grouped = combined.groupby(['date', 'channel'], dropna=False).sum(numeric_only=True).reset_index()
        grouped['mer'] = _safe_ratio(grouped['revenue'], grouped['ad_spend'], allow_inf=True)
        grouped['roas'] = grouped['mer']
        grouped['cac'] = _safe_ratio(grouped['ad_spend'], grouped['orders'])
        grouped['aov'] = _safe_ratio(grouped['revenue'], grouped['orders'])
        grouped['rpm'] = _safe_ratio(grouped['revenue'] * 1000.0, grouped['views'])
        grouped['ctr'] = _safe_ratio(grouped['clicks'], grouped['views'])
        if 'revenue' in grouped.columns:
            totals = grouped.groupby('date')['revenue'].transform('sum')
            grouped['share'] = grouped['revenue'].div(totals.replace({0: np.nan}))
        else:
            grouped['share'] = np.nan
        return grouped, None, {}

    def _build_campaign_series(self, sources: List[SourceFrame]) -> Tuple[pd.DataFrame, Optional[str], Dict[str, object]]:
        frames: List[pd.DataFrame] = []
        for source in sources:
            df = source.frame
            if 'date' not in df.columns:
                continue
            if 'campaign' not in df.columns and 'adset' in df.columns:
                df = df.copy()
                df['campaign'] = df['adset']
            if 'campaign' not in df.columns and 'ad' in df.columns:
                df = df.copy()
                df['campaign'] = df['ad']
            if 'campaign' not in df.columns:
                continue
            metrics = [col for col in ['revenue', 'ad_spend', 'orders', 'views', 'clicks'] if col in df.columns]
            if not metrics:
                continue
            subset_cols = ['date', 'campaign'] + metrics
            if 'channel' in df.columns:
                subset_cols.append('channel')
            if 'adset' in df.columns:
                subset_cols.append('adset')
            subset_cols = list(dict.fromkeys(subset_cols))
            subset = df[subset_cols].copy()
            subset['date'] = _coerce_date(subset['date'])
            for col in metrics:
                _coerce_numeric(subset, [col])
            if 'channel' in subset.columns:
                subset['channel'] = subset['channel'].apply(_normalize_channel)
            frames.append(subset)
        combined = self._combine_frames(frames)
        if combined.empty:
            return combined, 'missing: campaign metrics', {}
        for col in ['revenue', 'ad_spend', 'orders', 'views', 'clicks']:
            if col not in combined.columns:
                combined[col] = np.nan
        grouped = combined.groupby(['date', 'channel', 'campaign', 'adset'], dropna=False).sum(numeric_only=True).reset_index()
        grouped['mer'] = _safe_ratio(grouped['revenue'], grouped['ad_spend'], allow_inf=True)
        grouped['roas'] = grouped['mer']
        grouped['cac'] = _safe_ratio(grouped['ad_spend'], grouped['orders'])
        grouped['aov'] = _safe_ratio(grouped['revenue'], grouped['orders'])
        grouped['rpm'] = _safe_ratio(grouped['revenue'] * 1000.0, grouped['views'])
        grouped['ctr'] = _safe_ratio(grouped['clicks'], grouped['views'])
        return grouped, None, {}

    def _build_creative_series(self, sources: List[SourceFrame]) -> Tuple[pd.DataFrame, Optional[str], Dict[str, object]]:
        frames: List[pd.DataFrame] = []
        for source in sources:
            df = source.frame
            if 'date' not in df.columns or 'creative_id' not in df.columns:
                continue
            metrics = [col for col in ['views', 'clicks', 'ad_spend', 'revenue'] if col in df.columns]
            if not metrics:
                continue
            subset_cols = ['date', 'creative_id'] + metrics
            if 'channel' in df.columns:
                subset_cols.append('channel')
            subset_cols = list(dict.fromkeys(subset_cols))
            subset = df[subset_cols].copy()
            subset['date'] = _coerce_date(subset['date'])
            for col in metrics:
                _coerce_numeric(subset, [col])
            channel_guess = _infer_channel_from_label(source.label)
            if 'channel' in subset.columns:
                subset['channel'] = subset['channel'].apply(_normalize_channel)
                if subset['channel'].isna().all():
                    subset['channel'] = channel_guess
            else:
                subset['channel'] = channel_guess
            frames.append(subset)
        combined = self._combine_frames(frames)
        if combined.empty:
            return combined, 'missing: creative_id/views', {}
        for col in ['views', 'clicks', 'ad_spend', 'revenue']:
            if col not in combined.columns:
                combined[col] = np.nan
        grouped = combined.groupby(['date', 'channel', 'creative_id'], dropna=False).sum(numeric_only=True).reset_index()
        grouped['mer'] = _safe_ratio(grouped['revenue'], grouped['ad_spend'], allow_inf=True)
        grouped['roas'] = grouped['mer']
        grouped['rpm'] = _safe_ratio(grouped['revenue'] * 1000.0, grouped['views'])
        grouped['ctr'] = _safe_ratio(grouped['clicks'], grouped['views'])
        if 'revenue' in grouped.columns:
            totals = grouped.groupby(['date', 'channel'])['revenue'].transform('sum')
            grouped['share'] = grouped['revenue'].div(totals.replace({0: np.nan}))
        else:
            grouped['share'] = np.nan
        return grouped, None, {}

    def _build_margin_series(self, sources: List[SourceFrame]) -> Tuple[pd.DataFrame, Optional[str], Dict[str, object]]:
        frames: List[pd.DataFrame] = []
        for source in sources:
            df = source.frame
            if 'date' not in df.columns:
                continue
            metrics = [col for col in ['revenue', 'discounts', 'returns', 'shipping', 'taxes', 'duties'] if col in df.columns]
            if not metrics:
                continue
            subset_cols = ['date'] + metrics
            subset_cols = list(dict.fromkeys(subset_cols))
            subset = df[subset_cols].copy()
            subset['date'] = _coerce_date(subset['date'])
            for col in metrics:
                _coerce_numeric(subset, [col])
            frames.append(subset)
        combined = self._combine_frames(frames)
        if combined.empty:
            return combined, 'missing: margin components (discounts/returns/etc)', {}
        grouped = combined.groupby(['date'], dropna=False).sum(numeric_only=True).reset_index()
        return grouped, None, {}

    def _build_inventory_snapshot(self, sources: List[SourceFrame]) -> Tuple[pd.DataFrame, Optional[str], Dict[str, object]]:
        frames: List[pd.DataFrame] = []
        for source in sources:
            df = source.frame
            if 'date' not in df.columns:
                continue
            if 'sku' not in df.columns and 'product_title' not in df.columns:
                continue
            metrics = [col for col in ['on_hand', 'on_order', 'lead_time_days'] if col in df.columns]
            if not metrics:
                continue
            subset_cols = ['date'] + metrics
            if 'sku' in df.columns:
                subset_cols.append('sku')
            if 'product_title' in df.columns:
                subset_cols.append('product_title')
            subset_cols = list(dict.fromkeys(subset_cols))
            subset = df[subset_cols].copy()
            subset['date'] = _coerce_date(subset['date'])
            for col in metrics:
                _coerce_numeric(subset, [col])
            frames.append(subset)
        combined = self._combine_frames(frames)
        if combined.empty:
            return combined, 'missing: inventory columns', {}
        group_cols = ['date']
        if 'sku' in combined.columns:
            group_cols.append('sku')
        if 'product_title' in combined.columns:
            group_cols.append('product_title')
        grouped = combined.groupby(group_cols, dropna=False).agg({
            col: 'mean' for col in ['on_hand', 'on_order', 'lead_time_days'] if col in combined.columns
        }).reset_index()
        return grouped, None, {}
    def _write_efficiency_summary(self, channel_result: ArtifactResult, creative_result: ArtifactResult) -> Optional[ArtifactResult]:
        channel_df = channel_result.frame
        if channel_df is None or channel_df.empty or 'date' not in channel_df.columns:
            # Emit a stub aligned to series.jsonl if possible
            series_path = self.artifacts_dir / 'series.jsonl'
            as_of = None
            if series_path.exists():
                try:
                    series_df = pd.read_json(series_path, lines=True)
                    if 'date' in series_df.columns and not series_df.empty:
                        as_of = pd.to_datetime(series_df['date'], errors='coerce').max()
                except Exception:
                    as_of = None
            if as_of is None or pd.isna(as_of):
                return None
            payload = {
                'as_of': as_of.strftime('%Y-%m-%d'),
                'window_days': 28,
                'scope': 'paid',
                'by_channel': [],
                'by_creative': [],
            }
            path = self.artifacts_dir / 'efficiency.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print("efficiency: no rows in last 28d (no channel frame); wrote empty summary")
            return ArtifactResult(name='efficiency', filename='efficiency.json', path=path, rows=0)
        working = channel_df.copy()
        working['date'] = pd.to_datetime(working['date'], errors='coerce')
        working = working.dropna(subset=['date'])
        if working.empty:
            return None
        # Align as_of with series.jsonl max(date) when available
        series_path = self.artifacts_dir / 'series.jsonl'
        as_of = None
        if series_path.exists():
            try:
                series_df = pd.read_json(series_path, lines=True)
                if 'date' in series_df.columns and not series_df.empty:
                    as_of = pd.to_datetime(series_df['date'], errors='coerce').max()
            except Exception:
                as_of = None
        if as_of is None or pd.isna(as_of):
            as_of = working['date'].max()
        cutoff = as_of - pd.Timedelta(days=27)
        window = working[(working['date'] >= cutoff) & (working['date'] <= as_of)].copy()
        if window.empty:
            payload = {
                'as_of': as_of.strftime('%Y-%m-%d'),
                'window_days': 28,
                'scope': 'paid',
                'by_channel': [],
                'by_creative': [],
            }
            path = self.artifacts_dir / 'efficiency.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            print("efficiency: no rows in last 28d; wrote empty summary")
            return ArtifactResult(name='efficiency', filename='efficiency.json', path=path, rows=0)
        for col in ['revenue', 'ad_spend', 'orders']:
            if col in window.columns:
                window[col] = pd.to_numeric(window[col], errors='coerce').fillna(0.0)
        if 'channel' not in window.columns:
            window['channel'] = 'Unknown'
        window['channel'] = window['channel'].fillna('Unknown').astype(str)
        channel_rows: List[Dict[str, object]] = []
        grouped = window.groupby('channel', dropna=False)[['revenue', 'ad_spend', 'orders']].sum().reset_index()
        for _, row in grouped.iterrows():
            channel_name = row['channel'] if str(row['channel']).strip() else 'Unknown'
            revenue = float(row.get('revenue', 0.0) or 0.0)
            spend = float(row.get('ad_spend', 0.0) or 0.0)
            orders = float(row.get('orders', 0.0) or 0.0)
            if revenue == 0 and spend == 0 and orders == 0:
                continue
            if channel_name in {'Other', 'Unknown'} and spend <= 0 and revenue > 0:
                continue
            channel_rows.append({
                'channel': channel_name,
                'revenue': round(revenue, 2),
                'spend': round(spend, 2),
                'orders': int(round(orders)) if not np.isnan(orders) else 0,
            })

        creative_df = creative_result.frame
        creative_rows: List[Dict[str, object]] = []
        if creative_df is not None and not creative_df.empty and 'date' in creative_df.columns:
            c_working = creative_df.copy()
            c_working['date'] = pd.to_datetime(c_working['date'], errors='coerce')
            c_working = c_working.dropna(subset=['date'])
            if not c_working.empty:
                c_window = c_working[(c_working['date'] >= cutoff) & (c_working['date'] <= as_of)].copy()
                for col in ['revenue', 'ad_spend', 'views', 'clicks']:
                    if col in c_window.columns:
                        c_window[col] = pd.to_numeric(c_window[col], errors='coerce').fillna(0.0)
                if not c_window.empty:
                    agg = c_window.groupby(['creative_id', 'channel'], dropna=False)[['revenue', 'ad_spend', 'views', 'clicks']].sum().reset_index()
                    for _, row in agg.iterrows():
                        creative_id = str(row.get('creative_id', '') or '').strip() or 'unknown'
                        channel_name = str(row.get('channel', 'Unknown') or 'Unknown')
                        revenue = float(row.get('revenue', 0.0) or 0.0)
                        spend = float(row.get('ad_spend', 0.0) or 0.0)
                        views = float(row.get('views', 0.0) or 0.0)
                        clicks = float(row.get('clicks', 0.0) or 0.0)
                        if revenue == 0 and spend == 0 and views == 0 and clicks == 0:
                            continue
                        creative_rows.append({
                            'creative_id': creative_id,
                            'channel': channel_name,
                            'revenue': round(revenue, 2),
                            'spend': round(spend, 2),
                            'views': int(round(views)) if not np.isnan(views) else 0,
                            'clicks': int(round(clicks)) if not np.isnan(clicks) else 0,
                        })

        payload = {
            'as_of': as_of.strftime('%Y-%m-%d'),
            'window_days': 28,
            'scope': 'paid',
            'by_channel': channel_rows,
            'by_creative': creative_rows,
        }
        path = self.artifacts_dir / 'efficiency.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"Wrote efficiency.json: as_of={payload['as_of']}, channels={len(channel_rows)}, creatives={len(creative_rows)}")
        return ArtifactResult(name='efficiency', filename='efficiency.json', path=path, rows=len(channel_rows))


def generate_extra_artifacts(data_root: Path, artifacts_dir: Path, *, line_item_globs: Optional[Sequence[str]] = None) -> Dict[str, ArtifactResult]:
    generator = ExtraArtifactGenerator(data_root, artifacts_dir, line_item_globs=line_item_globs)
    return generator.run()
