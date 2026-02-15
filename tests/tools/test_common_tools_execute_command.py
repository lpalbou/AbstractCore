from __future__ import annotations

import tempfile

from abstractcore.tools.common_tools import execute_command


def test_execute_command_includes_command_in_output() -> None:
    out = execute_command("echo hello", timeout=10)
    assert isinstance(out, dict)
    assert out.get("command") == "echo hello"
    assert out.get("success") is True
    rendered = str(out.get("rendered") or "")
    assert "Command:" in rendered
    assert "echo hello" in rendered


def test_execute_command_accepts_string_args() -> None:
    out = execute_command("echo hello", timeout="10", capture_output="true")  # type: ignore[arg-type]
    assert isinstance(out, dict)
    assert out.get("success") is True
    assert "hello" in str(out.get("stdout") or "")


def test_execute_command_parses_allow_dangerous_false_string() -> None:
    # If allow_dangerous is the string "false", it must NOT bypass the security block.
    with tempfile.NamedTemporaryFile(prefix="abstractcore_execute_command_", delete=True) as f:
        cmd = f"chmod 777 {f.name}"
        out = execute_command(cmd, allow_dangerous="false")  # type: ignore[arg-type]
        assert isinstance(out, dict)
        assert out.get("success") is False
        assert "CRITICAL SECURITY BLOCK" in str(out.get("rendered") or "")
