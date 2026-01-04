from __future__ import annotations

from abstractcore.tools.common_tools import write_file


def test_write_file_tool_schema_requires_content() -> None:
    tool_def = getattr(write_file, "_tool_definition", None)
    assert tool_def is not None
    params = getattr(tool_def, "parameters", {}) or {}

    assert "file_path" in params
    assert "content" in params

    # Required args are inferred by absence of a default in the schema.
    assert "default" not in params["content"]

    exported = tool_def.to_dict()
    required_args = exported.get("required_args")
    assert isinstance(required_args, list)
    assert "file_path" in required_args
    assert "content" in required_args


