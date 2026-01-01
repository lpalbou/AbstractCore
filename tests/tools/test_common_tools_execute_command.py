from __future__ import annotations

from abstractcore.tools.common_tools import execute_command


def test_execute_command_includes_command_in_output() -> None:
    out = execute_command("echo hello", timeout=10)
    assert isinstance(out, dict)
    assert out.get("command") == "echo hello"
    assert out.get("success") is True
    rendered = str(out.get("rendered") or "")
    assert "Command:" in rendered
    assert "echo hello" in rendered

