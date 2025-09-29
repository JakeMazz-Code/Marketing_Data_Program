from __future__ import annotations

from pathlib import Path

from Marketing_analytics import generate_brief_md

from tests.test_md_hero_skus_section import _section_lines  # reuse helper


def test_md_channel_efficiency_handles_zero_spend(md_only_context):
    efficiency = md_only_context.artifacts["efficiency"]
    efficiency["by_channel"].append({"channel": "ZeroSpend", "revenue": 5000.0, "spend": 0.0, "orders": 80})

    result = generate_brief_md(str(md_only_context.tmp_path / "config.json"))
    text = Path(result["brief_md"]).read_text(encoding="utf-8")

    heading = "## Does marketing work? Which parts add incremental value? (organic vs paid vs SMS) (use efficiency.by_channel, scope noted)"
    body = _section_lines(text, heading)

    zero_row = next((line for line in body if line.startswith("ZeroSpend")), None)
    assert zero_row is not None
    assert "n/a" in zero_row.lower(), "zero-spend channels should emit n/a for ROAS or CAC"
