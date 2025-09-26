"""Streamlit dashboard for daily master artifacts (PII-safe)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PAGE_CONFIG = {
    "page_title": "Marketing Daily Master",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

st.set_page_config(**PAGE_CONFIG)


@dataclass(slots=True)
class ArtifactBundle:
    root: Path
    series: pd.DataFrame
    quality: Optional[Dict[str, Any]]
    quality_md: str
    brief_verified: Optional[Dict[str, Any]]
    status: Optional[Dict[str, Any]]
    anomalies: pd.DataFrame
    efficiency: Optional[Any]
    margin: Optional[Any]
    cohorts: Optional[Any]
    notes_paths: Dict[str, Path]


def _candidate_roots() -> Iterable[Path]:
    cwd = Path.cwd()
    yield cwd / "reports" / "daily_master"
    yield Path(__file__).resolve().parent.parent / "reports" / "daily_master"


def _resolve_artifact_root() -> Path:
    for candidate in _candidate_roots():
        if candidate.exists():
            return candidate
    default = next(iter(_candidate_roots()))
    default.mkdir(parents=True, exist_ok=True)
    return default


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_json(path_str: str, stamp: float) -> Optional[Dict[str, Any]]:
    if not path_str or stamp <= 0:
        return None
    with Path(path_str).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _load_text(path_str: str, stamp: float) -> str:
    if not path_str or stamp <= 0:
        return ""
    return Path(path_str).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _load_jsonl(path_str: str, stamp: float) -> pd.DataFrame:
    if not path_str or stamp <= 0:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    with Path(path_str).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
    numeric_cols = [col for col in df.columns if col != "date"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _load_generic(path_str: str, stamp: float) -> Optional[Any]:
    if not path_str or stamp <= 0:
        return None
    with Path(path_str).open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return handle.read()


def _load_bundle(root: Path) -> ArtifactBundle:
    series_path = root / "series.jsonl"
    quality_json_path = root / "quality_report.json"
    quality_md_path = root / "quality_report.md"
    brief_path = root / "brief_verified.json"
    status_path = root / "status.json"
    anomalies_path = root / "anomalies.json"
    efficiency_path = root / "efficiency.json"
    margin_path = root / "margin_waterfall.json"
    if not margin_path.exists():
        margin_path = root / "margin.json"
    cohort_path = root / "cohort_summary.json"
    if not cohort_path.exists():
        cohort_path = root / "ltv_summary.json"

    series_df = _load_jsonl(str(series_path), _mtime(series_path))
    quality_payload = _load_json(str(quality_json_path), _mtime(quality_json_path))
    quality_md = _load_text(str(quality_md_path), _mtime(quality_md_path))
    brief_verified = _load_json(str(brief_path), _mtime(brief_path))
    status_payload = _load_json(str(status_path), _mtime(status_path))
    anomalies_payload = _load_generic(str(anomalies_path), _mtime(anomalies_path))
    anomalies_df = pd.DataFrame(anomalies_payload) if isinstance(anomalies_payload, list) else pd.DataFrame()
    efficiency_payload = _load_generic(str(efficiency_path), _mtime(efficiency_path))
    margin_payload = _load_generic(str(margin_path), _mtime(margin_path))
    cohort_payload = _load_generic(str(cohort_path), _mtime(cohort_path))

    notes = {
        "anomalies": root / "anomalies_notes.md",
        "brief": root / "brief_notes.md",
    }

    return ArtifactBundle(
        root=root,
        series=series_df,
        quality=quality_payload,
        quality_md=quality_md,
        brief_verified=brief_verified,
        status=status_payload,
        anomalies=anomalies_df,
        efficiency=efficiency_payload,
        margin=margin_payload,
        cohorts=cohort_payload,
        notes_paths=notes,
    )


def _format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}" if abs(value) >= 1000 else f"${value:,.2f}"


def _format_number(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.1%}"


def _metric_delta(series: pd.Series, window: int, how: str = "mean") -> Tuple[Optional[float], Optional[float]]:
    data = series.dropna()
    if data.empty:
        return None, None
    current = data.iloc[-1]
    if len(data) < window:
        return current, None
    recent = data.tail(window)
    previous = data.iloc[-2 * window : -window] if len(data) >= 2 * window else pd.Series(dtype=float)
    if how == "sum":
        current_window = recent.sum()
        previous_window = previous.sum() if not previous.empty else None
    else:
        current_window = recent.mean()
        previous_window = previous.mean() if not previous.empty else None
    delta = None
    if previous_window not in (None, 0):
        delta = (current_window - previous_window) / abs(previous_window)
    return current, delta


def _quality_banner(bundle: ArtifactBundle) -> None:
    quality = bundle.quality or {}
    status = (quality.get("status") or "UNKNOWN").upper()
    rules = quality.get("rules", []) if isinstance(quality, dict) else []
    caveat = None
    for rule in rules:
        if rule.get("status") in {"WARN", "FAIL"}:
            caveat = rule.get("detail") or rule.get("name")
            break
    if status == "PASS":
        st.success(f"Data quality: PASS{f' — {caveat}' if caveat else ''}")
    elif status == "WARN":
        st.warning(f"Data quality: WARN — {caveat or 'see quality_report.md'}")
    elif status == "FAIL":
        st.error(f"Data quality: FAIL — {caveat or 'see quality_report.md'}")
    else:
        st.info("Data quality: status unknown")


def _status_badge(bundle: ArtifactBundle) -> None:
    payload = bundle.status or {}
    if not payload:
        return
    stage = payload.get("stage", "unknown")
    pct = payload.get("pct")
    stamp = payload.get("updated_at") or payload.get("timestamp")
    pct_text = f"{pct:.0%}" if isinstance(pct, (int, float)) else "n/a"
    st.caption(f"Run status — stage: {stage} · progress: {pct_text} · updated: {stamp}")


def _render_kpi_tiles(bundle: ArtifactBundle) -> None:
    df = bundle.series
    if df.empty:
        st.info("KPI tiles will appear once `series.jsonl` is available.")
        return
    df = df.set_index("date")
    metrics = [
        ("revenue", "Revenue", "currency", "sum"),
        ("ad_spend", "Ad Spend", "currency", "sum"),
        ("orders", "Orders", "number", "sum"),
        ("mer", "MER", "number", "mean"),
        ("cac", "CAC", "currency", "mean"),
        ("aov", "AOV", "currency", "mean"),
    ]
    columns = st.columns(len(metrics))
    for column, (field, label, kind, agg) in zip(columns, metrics):
        if field not in df.columns:
            column.metric(label, "n/a", delta="")
            continue
        current, delta7 = _metric_delta(df[field], window=7, how=agg)
        _, delta28 = _metric_delta(df[field], window=28, how=agg)
        formatter = _format_currency if kind == "currency" else _format_number
        value_text = formatter(current)
        delta_text = f"?7d {_format_percent(delta7)}" if delta7 is not None else "?7d n/a"
        column.metric(label, value_text, delta=delta_text)
        caption = f"?28d {_format_percent(delta28)}" if delta28 is not None else "?28d n/a"
        column.caption(caption)


def _apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    default_start = max_date - pd.Timedelta(days=90)
    start, end = st.date_input(
        "Date range",
        value=(max(default_start, min_date), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(start, tuple):  # defensive fallback
        start, end = start
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df.loc[mask]


def _render_trend(bundle: ArtifactBundle) -> None:
    df = bundle.series.copy()
    if df.empty:
        st.info("Trend charts will render after an ETL run produces `series.jsonl`.")
        return
    df = df.sort_values("date")
    filtered = _apply_date_filter(df)
    if filtered.empty:
        st.warning("No data in selected range.")
        return
    for window in (7, 28):
        filtered[f"rev_ma_{window}"] = filtered["revenue"].rolling(window).mean()
        if "mer" in filtered:
            filtered[f"mer_ma_{window}"] = filtered["mer"].rolling(window).mean()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=filtered["date"],
            y=filtered["revenue"],
            name="Revenue",
            mode="lines",
            line=dict(color="#3778C2", width=2),
        ),
        secondary_y=False,
    )
    for window, style in ((7, "dash"), (28, "dot")):
        fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered[f"rev_ma_{window}"],
                name=f"Revenue MA{window}",
                mode="lines",
                line=dict(color="#6FA4E3", width=1.5, dash=style),
            ),
            secondary_y=False,
        )
    if "mer" in filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["mer"],
                name="MER",
                mode="lines",
                line=dict(color="#E4572E", width=2),
            ),
            secondary_y=True,
        )
        for window, style in ((7, "dash"), (28, "dot")):
            fig.add_trace(
                go.Scatter(
                    x=filtered["date"],
                    y=filtered[f"mer_ma_{window}"],
                    name=f"MER MA{window}",
                    mode="lines",
                    line=dict(color="#F18F68", width=1.5, dash=style),
                ),
                secondary_y=True,
            )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=40, t=10, b=40),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Revenue", secondary_y=False)
    fig.update_yaxes(title="MER", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_brief_card(bundle: ArtifactBundle) -> None:
    brief = bundle.brief_verified or {}
    if not brief:
        st.info("Run `python manage.py brief` to generate a verified brief snapshot.")
        return
    topline = brief.get("topline") or {}
    actions = brief.get("actions") or []
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Topline")
        period = topline.get("period", "n/a")
        trend = topline.get("trend", "n/a")
        driver = topline.get("driver", "n/a")
        st.markdown(f"**Period:** {period}\n\n**Trend:** {trend}\n\n**Driver:** {driver}")
    with col2:
        st.subheader("Top Actions")
        if not actions:
            st.caption("No actions returned.")
        else:
            for idx, action in enumerate(actions[:3], start=1):
                title = action.get("title", f"Action {idx}")
                why_now = action.get("why_now", "Context not provided.")
                st.markdown(f"**{idx}. {title}**\n{why_now}")
    notes = bundle.notes_paths.get("brief")
    if notes and notes.exists():
        st.caption(f"Verifier notes: {notes}")


def render_overview(bundle: ArtifactBundle) -> None:
    st.title("Overview")
    _quality_banner(bundle)
    _status_badge(bundle)
    st.subheader("Key KPIs")
    _render_kpi_tiles(bundle)
    st.subheader("Revenue & MER Trend")
    _render_trend(bundle)
    st.subheader("Verified Brief Snapshot")
    _render_brief_card(bundle)


def render_trends(bundle: ArtifactBundle) -> None:
    st.title("Trends & Seasonality")
    if bundle.series.empty:
        st.info("Series data not available yet.")
        return
    _render_trend(bundle)
    trend_image = bundle.root / "trend.png"
    if trend_image.exists():
        st.image(str(trend_image), caption="Uploaded trend visual", use_container_width=True)
    heat = bundle.series.copy()
    heat["week"] = heat["date"].dt.isocalendar().week
    heat["weekday"] = heat["date"].dt.day_name()
    pivot = heat.pivot_table(index="weekday", columns="week", values="revenue", aggfunc="sum")
    if not pivot.empty:
        st.subheader("Revenue coverage heatmap")
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index,
                colorscale="Blues",
                hovertemplate="Week %{x} · %{y}: ${%{z}:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)


def render_anomalies(bundle: ArtifactBundle) -> None:
    st.title("Anomalies")
    df = bundle.anomalies
    if df.empty:
        st.info("No anomalies detected yet.")
        return
    df = df.copy()
    df["abs_z"] = df["z"].abs()
    df = df.sort_values("abs_z", ascending=False)
    st.dataframe(df[["date", "metric", "z", "note"]], use_container_width=True)
    series = bundle.series.set_index("date") if not bundle.series.empty else pd.DataFrame()
    if not series.empty and "revenue" in series.columns:
        st.subheader("Revenue anomalies")
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(x=series.index, y=series["revenue"], mode="lines", name="Revenue"))
        revenue_anoms = df[df["metric"] == "revenue"]
        if not revenue_anoms.empty:
            x_vals = pd.to_datetime(revenue_anoms["date"])
            y_vals = series["revenue"].reindex(x_vals).tolist()
            fig_rev.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    marker=dict(color="#E4572E", size=10),
                    name="Revenue anomaly",
                    text=revenue_anoms["note"],
                )
            )
        fig_rev.update_layout(hovermode="x unified", height=320, margin=dict(l=40, r=40, t=20, b=40))
        st.plotly_chart(fig_rev, use_container_width=True)
    notes = bundle.notes_paths.get("anomalies")
    if notes and notes.exists():
        st.caption(f"Investigation notes: {notes}")


def render_efficiency(bundle: ArtifactBundle) -> None:
    st.title("Efficiency")
    payload = bundle.efficiency
    if not payload:
        st.info("Efficiency artifacts not available. Run the attribution agent to populate `efficiency.json`.")
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            heading = key.replace("_", " ").title()
            if isinstance(value, list) and value and isinstance(value[0], dict):
                st.subheader(heading)
                st.dataframe(pd.DataFrame(value), use_container_width=True)
            elif isinstance(value, dict):
                st.subheader(heading)
                st.json(value)
            else:
                st.markdown(f"**{heading}:** {value}")
    elif isinstance(payload, list):
        st.dataframe(pd.DataFrame(payload), use_container_width=True)
    else:
        st.write(payload)


def render_margin(bundle: ArtifactBundle) -> None:
    st.title("Margin & Leakage")
    data = bundle.margin
    if not data:
        st.info("Margin artifacts not yet produced.")
        return
    if isinstance(data, dict) and isinstance(data.get("waterfall"), list):
        waterfall = data.get("waterfall", [])
        if waterfall:
            labels = [item.get("label", "") for item in waterfall]
            values = [item.get("value", 0.0) for item in waterfall]
            measures = [item.get("type", "relative") for item in waterfall]
            fig = go.Figure()
            fig.add_trace(
                go.Waterfall(
                    name="Margin",
                    orientation="v",
                    measure=measures,
                    x=labels,
                    y=values,
                )
            )
            fig.update_layout(showlegend=False, height=360, margin=dict(l=40, r=40, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)
    st.json(data)


def render_cohorts(bundle: ArtifactBundle) -> None:
    st.title("Cohorts & LTV")
    payload = bundle.cohorts
    if not payload:
        st.info("No cohort or LTV artifacts detected yet.")
        return
    if isinstance(payload, list):
        st.dataframe(pd.DataFrame(payload), use_container_width=True)
    elif isinstance(payload, dict):
        st.json(payload)
    else:
        st.write(payload)


def render_settings(bundle: ArtifactBundle) -> None:
    st.title("Settings & Run")
    st.markdown(f"**Artifacts directory:** `{bundle.root}`")
    if st.button("Refresh cache", type="secondary"):
        st.cache_data.clear()
        st.experimental_rerun()
    log_path = bundle.root / "etl.log"
    if log_path.exists():
        st.subheader("etl.log")
        st.code(log_path.read_text(encoding="utf-8")[-4000:])
    st.subheader("Environment knobs")
    env_keys = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL_ANALYSIS",
        "OPENAI_SERIES_DAYS",
        "OPENAI_MAX_OUTPUT_TOKENS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL_VERIFIER",
    ]
    rows = []
    for key in env_keys:
        value = os.getenv(key)
        if not value:
            rows.append({"variable": key, "status": "missing"})
        elif key.endswith("API_KEY"):
            rows.append({"variable": key, "status": "set"})
        else:
            rows.append({"variable": key, "status": value})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Environment values are read-only within the app.")


def main() -> None:
    root = _resolve_artifact_root()
    bundle = _load_bundle(root)
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to",
        (
            "Overview",
            "Trends & Seasonality",
            "Anomalies",
            "Efficiency",
            "Margin & Leakage",
            "Cohorts & LTV",
            "Settings & Run",
        ),
    )
    renderer = {
        "Overview": render_overview,
        "Trends & Seasonality": render_trends,
        "Anomalies": render_anomalies,
        "Efficiency": render_efficiency,
        "Margin & Leakage": render_margin,
        "Cohorts & LTV": render_cohorts,
        "Settings & Run": render_settings,
    }[section]
    renderer(bundle)


if __name__ == "__main__":  # pragma: no cover
    main()


