from abstractcore.tools.core import ToolCall
from abstractcore.tools.registry import ToolRegistry


def test_tool_registry_marks_error_string_outputs_as_failure() -> None:
    reg = ToolRegistry()

    def tool_returns_error() -> str:
        return "Error: Something went wrong"

    reg.register(tool_returns_error)
    result = reg.execute_tool(ToolCall(name="tool_returns_error", arguments={}, call_id="c1"))

    assert result.success is False
    assert result.output == ""
    assert result.error == "Something went wrong"


def test_tool_registry_marks_cross_mark_outputs_as_failure() -> None:
    reg = ToolRegistry()

    def tool_returns_cross() -> str:
        return "❌ Permission denied: Cannot write"

    reg.register(tool_returns_cross)
    result = reg.execute_tool(ToolCall(name="tool_returns_cross", arguments={}, call_id="c1"))

    assert result.success is False
    assert result.output == ""
    assert result.error == "Permission denied: Cannot write"

