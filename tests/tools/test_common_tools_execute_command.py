from __future__ import annotations

from abstractcore.tools.common_tools import execute_command


def test_execute_command_includes_command_in_output() -> None:
    out = execute_command("echo hello", timeout=10)
    assert "Command:" in out
    assert "echo hello" in out

