from __future__ import annotations

from pathlib import Path

from Marketing_analytics import generate_brief_md


def _section_lines(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    if heading not in lines:
        raise AssertionError(f"Heading {heading!r} missing from brief")
    start = lines.index(heading) + 1
    for end in range(start, len(lines)):
        if lines[end].startswith("## "):
            return [line for line in lines[start:end] if line.strip()]
    return [line for line in lines[start:] if line.strip()]


def test_md_hero_skus_section(md_only_context):
    result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    text = Path(result["brief_md"]).read_text(encoding="utf-8")

    heading = "## Hero SKUs & Q4/Q1 Inventory Plan"
    body = _section_lines(text, heading)

    explainer_line = next((line for line in body if line.startswith("Inventory plan =")), None)
    assert explainer_line is not None

    header_line = "SKU / Product | Revenue (28d) | Units (28d) | Revenue Share | Units Share | Run Rate / day | Rec. 45d Units"
    assert header_line in body
    header_index = body.index(header_line)
    assert body[header_index + 1].startswith("---"), "table should have markdown separator"

    data_rows = [line for line in body[header_index + 2:] if "|" in line]
    first_label = data_rows[0].split('|')[0].strip()
    assert not first_label.lower().startswith('nan'), 'label should not show nan prefix'
    assert 'nan /' not in data_rows[0].lower(), 'label should not include nan / pattern'
    assert len(data_rows) >= 3, "expect at least three hero SKUs"

    last_column_values = [row.split("|")[-1].strip() for row in data_rows]
    assert any(value.startswith(tuple("0123456789")) for value in last_column_values), "expect numeric Q4 buy guidance"

    assert any("Run Rate / day" in line for line in body)
    assert any("Rec. 45d Units" in line for line in body)

    assert "Indicative only" not in last_column_values[0], "units should be available in fixtures"