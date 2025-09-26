"""Utilities for loading and preparing marketing datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from Marketing_analytics.config import ColumnMapping


@dataclass(slots=True)
class DatasetBundle:
    """Container for the prepared marketing dataset and useful metadata."""

    frame: pd.DataFrame
    mapping: ColumnMapping
    column_aliases: Dict[str, str]

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def available(self, column: str | None) -> bool:
        return bool(column) and column in self.frame.columns


_NON_ALNUM_PATTERN = re.compile(r"[^0-9a-zA-Z]+")


def _normalize_column_name(name: str) -> str:
    normalized = _NON_ALNUM_PATTERN.sub("_", name.strip().lower())
    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    renamed = {col: _normalize_column_name(col) for col in df.columns}
    return df.rename(columns=renamed), renamed


def resolve_mapping(mapping: ColumnMapping, aliases: Dict[str, str]) -> ColumnMapping:
    """Adjust mapping attributes to reflect normalized column names."""

    normalized: Dict[str, object] = {}
    alias_lookup = {orig.lower(): new for orig, new in aliases.items()}
    for field in fields(mapping):
        column_name = getattr(mapping, field.name)
        if isinstance(column_name, tuple):
            normalized[field.name] = tuple(
                alias_lookup.get(col.lower(), _normalize_column_name(col)) for col in column_name
            )
        elif column_name:
            normalized[field.name] = alias_lookup.get(column_name.lower(), _normalize_column_name(column_name))
        else:
            normalized[field.name] = column_name
    return ColumnMapping(**normalized)


def _sanitize_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype.kind not in {"O", "U", "S"}:
        return series
    cleaned = (
        series.astype(str)
        .str.replace(r"[,%$]", "", regex=True)
        .str.replace(r"\\s", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        .replace({"": np.nan, "none": np.nan, "nan": np.nan})
    )
    return cleaned


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = _sanitize_numeric_series(df[column])
            df[column] = pd.to_numeric(df[column], errors="coerce")


def coerce_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype.kind in {"i", "u", "f"}:
        return series.fillna(0).astype(float)
    lower = series.astype(str).str.strip().str.lower()
    truthy = {"true", "yes", "1", "y", "t"}
    falsy = {"false", "no", "0", "n", "f"}
    return lower.map(lambda value: 1.0 if value in truthy else (0.0 if value in falsy else np.nan))


def load_dataset(path: str | Path, mapping: ColumnMapping) -> DatasetBundle:
    df = pd.read_csv(path)
    df, aliases = normalize_columns(df)
    resolved_mapping = resolve_mapping(mapping, aliases)

    missing = [col for col in resolved_mapping.required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

    if resolved_mapping.event_timestamp and resolved_mapping.event_timestamp in df.columns:
        df[resolved_mapping.event_timestamp] = pd.to_datetime(
            df[resolved_mapping.event_timestamp], errors="coerce"
        )

    numeric_candidates = resolved_mapping.numeric_features()
    coerce_numeric(df, numeric_candidates)

    if resolved_mapping.response and resolved_mapping.response in df.columns:
        df[resolved_mapping.response] = coerce_boolean(df[resolved_mapping.response])

    for flag in resolved_mapping.binary_features():
        if flag in df.columns:
            df[flag] = coerce_boolean(df[flag])

    df = df.dropna(subset=[resolved_mapping.customer_id])

    bundle = DatasetBundle(frame=df, mapping=resolved_mapping, column_aliases=aliases)
    return bundle


def add_derived_columns(bundle: DatasetBundle) -> DatasetBundle:
    df = bundle.frame.copy()
    mapping = bundle.mapping

    if mapping.response and mapping.response in df.columns:
        df["response_flag"] = df[mapping.response].fillna(0).astype(float)

    if mapping.revenue and mapping.revenue in df.columns and mapping.spend and mapping.spend in df.columns:
        spend = df[mapping.spend].replace(0, np.nan)
        df["roi"] = (df[mapping.revenue] - df[mapping.spend]) / spend

    if mapping.gross_margin and mapping.gross_margin in df.columns:
        df["net_margin"] = df[mapping.gross_margin]
    elif mapping.revenue and mapping.revenue in df.columns and mapping.cogs and mapping.cogs in df.columns:
        df["net_margin"] = df[mapping.revenue] - df[mapping.cogs]

    if mapping.event_timestamp and mapping.event_timestamp in df.columns:
        df["event_date"] = df[mapping.event_timestamp].dt.date
        df["event_month"] = df[mapping.event_timestamp].dt.to_period("M").dt.to_timestamp()

    return DatasetBundle(frame=df, mapping=mapping, column_aliases=bundle.column_aliases)


