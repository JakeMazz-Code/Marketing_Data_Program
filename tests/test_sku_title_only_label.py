from __future__ import annotations

from pathlib import Path

from Marketing_analytics import generate_brief_md

from tests.test_md_hero_skus_section import _section_lines


def test_sku_title_only_label(md_only_context):
    # Remove sku/sku_code to force title-only labeling
    sku_df = md_only_context.artifacts["sku_series"].copy()
    if "sku" in sku_df.columns:
        sku_df.drop(columns=["sku"], inplace=True)
    sku_df["sku_code"] = None
    md_only_context.artifacts["sku_series"] = sku_df

    result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    text = Path(result["brief_md"]).read_text(encoding="utf-8")

    heading = "## Hero SKUs & Q4/Q1 Inventory Plan"
    body = _section_lines(text, heading)

    header_line = "SKU / Product | Revenue (28d) | Units (28d) | Revenue Share | Units Share | Run Rate / day | Rec. 45d Units"
    assert header_line in body
    header_index = body.index(header_line)
    data_rows = [line for line in body[header_index + 2:] if "|" in line]
    assert data_rows, "expected at least one row in hero SKU table"

    first_label_cell = data_rows[0].split("|")[0].strip()
    # Should not show 'nan /' or start with 'nan'
    assert not first_label_cell.lower().startswith("nan")
    assert "nan /" not in first_label_cell.lower()
    # Title-only label should not include separator when sku code is missing
    assert " — " not in first_label_cell or first_label_cell.split(" — ")[0].strip() != "nan"

