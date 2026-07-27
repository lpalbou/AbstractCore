from __future__ import annotations

import os
import subprocess
import tempfile
import time

import pytest

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


def test_execute_command_captures_stderr_and_nonzero_exit() -> None:
    out = execute_command("echo out && echo err >&2 && exit 4", timeout=10)
    assert out["success"] is False
    assert out["return_code"] == 4
    assert "out" in str(out.get("stdout") or "")
    assert "err" in str(out.get("stderr") or "")


@pytest.mark.skipif(os.name != "posix", reason="process-tree timeout semantics are POSIX-specific")
def test_execute_command_timeout_treekills_orphan_and_never_pins() -> None:
    """FACE 1 (runtime c5004): a shell that backgrounds a grandchild holding
    the stdout pipe and then sleeps past the timeout used to PIN the calling
    thread forever — subprocess.run kills only the shell, and the captured
    stdout read waits for pipe EOF that the orphaned grandchild keeps open.
    The Popen + start_new_session + tree-kill path must (a) return at ~timeout
    (never hang), (b) report an honest timeout, (c) leave no orphan alive."""
    marker = f"core_face1_pin_probe_{int(time.time() * 1000)}"
    # Backgrounded grandchild (holds stdout, sleeps long) + a foreground sleep
    # past the 2s timeout. The marker makes the orphan uniquely greppable.
    cmd = f"(sleep 40 # {marker}\n) & sleep 40"

    t0 = time.monotonic()
    out = execute_command(cmd, timeout=2, allow_dangerous=False)
    elapsed = time.monotonic() - t0

    # (a) never pinned — returns near the timeout + a small kill/drain grace.
    assert elapsed < 2 + 8, f"execute_command pinned for {elapsed:.1f}s (pipe-EOF hang)"
    # (b) honest timeout verdict.
    assert out["success"] is False
    assert "timeout" in str(out.get("error") or "").lower()
    # (c) no orphan survives the tree-kill.
    time.sleep(0.5)
    res = subprocess.run(["pgrep", "-fl", marker], capture_output=True, text=True)
    leftover = [ln for ln in (res.stdout or "").splitlines() if marker in ln and "pgrep" not in ln]
    assert not leftover, f"orphan survived the timeout tree-kill: {leftover}"
