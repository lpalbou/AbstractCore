from __future__ import annotations

from abstractcore.tools.common_tools import write_file
from abstractcore.tools.handler import UniversalToolHandler


def test_prepare_tools_for_native_includes_json_schema_required_fields() -> None:
    tool_def = getattr(write_file, "_tool_definition", None)
    assert tool_def is not None

    handler = UniversalToolHandler(model_name="gpt-4o")
    native = handler.prepare_tools_for_native([tool_def])
    assert isinstance(native, list) and native

    fn = native[0].get("function") if isinstance(native[0], dict) else None
    assert isinstance(fn, dict)

    params = fn.get("parameters")
    assert isinstance(params, dict)
    required = params.get("required")
    assert isinstance(required, list)

    assert "file_path" in required
    assert "content" in required



