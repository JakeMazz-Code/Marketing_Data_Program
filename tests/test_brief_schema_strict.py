from copy import deepcopy

from Marketing_analytics.ai import BRIEF_JSON_SCHEMA, lint_brief_schema


def test_schema_passes_now() -> None:
    violations = lint_brief_schema(deepcopy(BRIEF_JSON_SCHEMA))
    assert violations == []


def test_schema_catches_missing_required() -> None:
    schema = deepcopy(BRIEF_JSON_SCHEMA)
    anomaly_items = schema["properties"]["diagnostics"]["properties"]["anomalies"]["items"]
    anomaly_items["required"].remove("peak_z")
    violations = lint_brief_schema(schema)
    assert any("#/diagnostics/anomalies/items" in msg and "required" in msg for msg in violations)


def test_schema_catches_additional_properties() -> None:
    schema = deepcopy(BRIEF_JSON_SCHEMA)
    schema["properties"]["topline"]["additionalProperties"] = True
    violations = lint_brief_schema(schema)
    assert any("#/topline" in msg and "additionalProperties" in msg for msg in violations)


def test_schema_catches_non_object_items() -> None:
    schema = deepcopy(BRIEF_JSON_SCHEMA)
    anomaly_items = schema["properties"]["diagnostics"]["properties"]["anomalies"]["items"]
    anomaly_items["type"] = "string"
    violations = lint_brief_schema(schema)
    assert any(
        "#/diagnostics/anomalies/items" in msg and "type must be 'object'" in msg for msg in violations
    )
