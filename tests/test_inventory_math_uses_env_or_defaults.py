from __future__ import annotations

from pathlib import Path

from Marketing_analytics import generate_brief_md


def _extract_recommended_units(markdown: str) -> float:
    lines = markdown.splitlines()
    header = "SKU / Product | Revenue (28d) | Units (28d) | Revenue Share | Units Share | Run Rate / day | Rec. 45d Units"
    idx = lines.index(header)
    data_line = lines[idx + 2]
    cells = [cell.strip() for cell in data_line.split("|")]
    last_cell = cells[-1]
    number_str = last_cell.split("(")[0].strip().replace(",", "")
    return float(number_str)


def test_inventory_math_uses_env_or_defaults(md_only_context, monkeypatch):
    for name in ("SEASONALITY_MULT", "SAFETY_STOCK_MULT", "DEFAULT_SEASONALITY", "DEFAULT_SAFETY_STOCK"):
        monkeypatch.delenv(name, raising=False)

    first_result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    first_text = Path(first_result["brief_md"]).read_text(encoding="utf-8")
    default_units = _extract_recommended_units(first_text)

    monkeypatch.setenv("SEASONALITY_MULT", "1.8")
    monkeypatch.setenv("SAFETY_STOCK_MULT", "1.3")

    second_result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    second_text = Path(second_result["brief_md"]).read_text(encoding="utf-8")
    overridden_units = _extract_recommended_units(second_text)

    assert overridden_units != default_units
    assert overridden_units > default_units