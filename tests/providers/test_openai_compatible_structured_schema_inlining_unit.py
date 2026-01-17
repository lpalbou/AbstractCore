from __future__ import annotations

from abstractcore.providers.openai_compatible_provider import _inline_json_schema_refs


def test_inline_json_schema_refs_removes_defs_and_refs() -> None:
    schema = {
        "$defs": {
            "Item": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        },
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "required": ["items"],
    }

    out = _inline_json_schema_refs(schema)
    assert "$defs" not in out
    assert out["properties"]["items"]["items"]["type"] == "object"
    assert out["properties"]["items"]["items"]["properties"]["x"]["type"] == "string"


def test_inline_json_schema_refs_noop_without_defs() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    out = _inline_json_schema_refs(schema)
    assert out == schema

