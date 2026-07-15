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


def test_tool_registry_marks_json_error_outputs_as_failure() -> None:
    reg = ToolRegistry()

    def tool_returns_json_error() -> str:
        return '{"success":false,"status_hint":"error","error":"requests is not installed","results":[]}'

    reg.register(tool_returns_json_error)
    result = reg.execute_tool(ToolCall(name="tool_returns_json_error", arguments={}, call_id="c1"))

    assert result.success is False
    assert result.output == ""
    assert result.error == "requests is not installed"


def test_explicit_success_with_message_is_never_a_failure() -> None:
    """Adversarial find (2026-07-13): {"success": True, "message": ...} was
    reported to the model as a FAILED call because any non-empty `message`
    counted as an error signal — after the tool's side effect had run."""
    reg = ToolRegistry()

    def tool_succeeds_with_message() -> dict:
        return {"success": True, "message": "Created 3 rows"}

    reg.register(tool_succeeds_with_message)
    result = reg.execute_tool(ToolCall(name="tool_succeeds_with_message", arguments={}, call_id="c1"))

    assert result.success is True
    assert result.error is None


def test_unmarked_dict_message_alone_is_not_an_error_signal() -> None:
    """`message` without success/ok/error markers is a normal payload field."""
    reg = ToolRegistry()

    def tool_returns_plain_message() -> dict:
        return {"message": "42 items processed", "items": 42}

    reg.register(tool_returns_plain_message)
    result = reg.execute_tool(ToolCall(name="tool_returns_plain_message", arguments={}, call_id="c1"))

    assert result.success is True
    assert result.error is None


def test_unmarked_dict_error_key_still_signals_failure() -> None:
    """The `error` key alone remains an honest failure signal."""
    reg = ToolRegistry()

    def tool_returns_error_dict() -> dict:
        return {"error": "disk full"}

    reg.register(tool_returns_error_dict)
    result = reg.execute_tool(ToolCall(name="tool_returns_error_dict", arguments={}, call_id="c1"))

    assert result.success is False
    assert result.error == "disk full"
