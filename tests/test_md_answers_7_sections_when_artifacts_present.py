from __future__ import annotations

from pathlib import Path

from Marketing_analytics import generate_brief_md


_EXPECTED_HEADINGS = [
    "## What are our three biggest opportunities for growth? (What/Why/How next week)",
    "## Accounting for seasonality... is the revenue accelerating or slowing down? (7/28/90 + slope + MER)",
    "## What customer segments do we have... how do we get more? (infer TOF/BOF from names; if ambiguous, instruct naming hygiene)",
    "## What products truly pull demand (hero SKUs)... inventory for Q4/Q1? (table from sku_series.jsonl)",
    "## How can we improve margins? What's killing margin post-purchase? (28-day discounts/returns/shipping/taxes)",
    "## Does marketing work? Which parts add incremental value? (organic vs paid vs SMS) (use efficiency.by_channel, scope noted)",
    "## Which creatives actually move product on IG? (use by_creative, RPM/ROAS/CTR with guards)",
]


def _section_body(lines: list[str], heading_index: int, positions: list[int]) -> list[str]:
    start = positions[heading_index]
    end = positions[heading_index + 1] if heading_index + 1 < len(positions) else len(lines)
    return [line for line in lines[start + 1:end] if line.strip()]


def test_md_answers_7_sections_when_artifacts_present(md_only_context):
    result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    text = Path(result["brief_md"]).read_text(encoding="utf-8")

    lines = text.splitlines()
    positions: list[int] = []
    for heading in _EXPECTED_HEADINGS:
        assert heading in lines, f"Missing heading: {heading}"
        positions.append(lines.index(heading))

    assert positions == sorted(positions), "Required sections should appear in order"
    metrics_heading = "## Metrics in Plain English"
    assert metrics_heading in lines, 'Metrics explainer missing'
    metrics_index = lines.index(metrics_heading)
    # Metrics explainer should appear at the very top before required Q&As
    assert metrics_index < positions[0]
    additional_index = lines.index('## Additional Insights (Model Reasoning)')
    assert metrics_index < additional_index


    for idx, heading in enumerate(_EXPECTED_HEADINGS):
        body = _section_body(lines, idx, positions)
        assert body, f"Section {heading} should include narrative content"
        combined = "\n".join(body)
        assert "Not available" not in combined, f"Section {heading} unexpectedly reports missing data"
