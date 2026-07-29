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
    # (b) honest timeout verdict (message reads "timed out" since 2026-07-27).
    assert out["success"] is False
    err = str(out.get("error") or "").lower()
    assert ("timeout" in err) or ("timed out" in err)
    # (c) no orphan survives the tree-kill.
    time.sleep(0.5)
    res = subprocess.run(["pgrep", "-fl", marker], capture_output=True, text=True)
    leftover = [ln for ln in (res.stdout or "").splitlines() if marker in ln and "pgrep" not in ln]
    assert not leftover, f"orphan survived the timeout tree-kill: {leftover}"


# ---------------------------------------------------------------------------
# Timeout observability + clamp (2026-07-27 incident: a model passed
# timeout=30000 meaning milliseconds — an 8h20m wait — and the timeout result
# discarded every captured byte, so the model had nothing to diagnose from).
# The markers below are built with printf '%s' composition so the marker text
# appears in the OUTPUT but never in the command line that the render echoes.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="/bin/sh + signal semantics are POSIX-specific")
def test_timeout_result_includes_prekill_stdout_and_stderr() -> None:
    t0 = time.monotonic()
    out = execute_command(
        "printf 'pre_%s\\n' KILL_OUT; printf 'pre_%s\\n' KILL_ERR >&2; sleep 30",
        timeout=1,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 10, f"timeout path took {elapsed:.1f}s (drain must be fast when the tree died)"
    assert out["success"] is False
    assert "timed out after 1s and was killed (including all child processes)" in str(out["error"])
    rendered = str(out["rendered"])
    assert "Output captured before the kill:" in rendered
    assert "pre_KILL_OUT" in rendered
    assert "pre_KILL_ERR" in rendered
    # Full output also rides the structured fields for durable evidence.
    assert "pre_KILL_OUT" in str(out["stdout"])
    assert "pre_KILL_ERR" in str(out["stderr"])


@pytest.mark.skipif(os.name != "posix", reason="/bin/sh + signal semantics are POSIX-specific")
def test_timeout_prekill_output_keeps_tail_with_truncation_label() -> None:
    # ~45k chars of stdout before the sleep: over the 20000-char render cap.
    cmd = (
        "printf 'HEAD_%s\\n' MARKER; "
        "i=0; while [ $i -lt 3000 ]; do echo \"filler_line_$i\"; i=$((i+1)); done; "
        "printf 'TAIL_%s\\n' MARKER; sleep 30"
    )
    out = execute_command(cmd, timeout=1)

    assert out["success"] is False
    assert out["stdout_truncated"] is True
    # The full drained output keeps everything; the render keeps the TAIL —
    # the end of the log is where the failure usually shows.
    assert "HEAD_MARKER" in str(out["stdout"])
    assert "TAIL_MARKER" in str(out["stdout_preview"])
    assert "HEAD_MARKER" not in str(out["stdout_preview"])
    rendered = str(out["rendered"])
    assert "TAIL_MARKER" in rendered
    assert "HEAD_MARKER" not in rendered
    assert "#TRUNCATION" in rendered
    assert "showing the last 20000" in rendered


@pytest.mark.skipif(os.name != "posix", reason="/bin/sh + signal semantics are POSIX-specific")
def test_timeout_with_no_output_says_so() -> None:
    out = execute_command("sleep 30", timeout=1)
    assert out["success"] is False
    assert (
        "No output was captured before the kill (a backgrounded child "
        "writing to files or inheriting the terminal is invisible here)."
    ) in str(out["rendered"])
    assert out["stdout"] == ""
    assert out["stderr"] == ""


def test_timeout_clamp_note_on_success_render() -> None:
    # The incident value: 30000 (meant as milliseconds). The command is fast,
    # so nothing waits — the clamp only shortens the DEADLINE.
    out = execute_command("echo hi", timeout=30000)
    assert out["success"] is True
    assert (
        "Note: requested timeout 30000s exceeded the 600s tool maximum; 600s was used. "
        "For longer work, run it in the background and poll, or raise the host executor timeout."
    ) in str(out["rendered"])


@pytest.mark.skipif(os.name != "posix", reason="/bin/sh + signal semantics are POSIX-specific")
def test_timeout_clamp_note_on_timeout_render(monkeypatch: pytest.MonkeyPatch) -> None:
    # The ceiling is read at call time, so tests can lower it and exercise a
    # REAL clamped timeout without waiting 600s.
    from abstractcore.tools import common_tools

    monkeypatch.setattr(common_tools, "EXECUTE_COMMAND_MAX_TIMEOUT_S", 1.0)
    out = execute_command("printf 'clamped_%s\\n' RUN_OUT; sleep 30", timeout=5)

    assert out["success"] is False
    assert out["timeout_clamped"] is True
    assert out["requested_timeout_s"] == 5.0
    assert out["timeout_s"] == 1
    rendered = str(out["rendered"])
    assert "timed out after 1s" in rendered
    assert "Note: requested timeout 5s exceeded the 1s tool maximum; 1s was used." in rendered
    # The clamp note and the captured output ride the SAME timeout render.
    assert "clamped_RUN_OUT" in rendered


@pytest.mark.skipif(os.name != "posix", reason="uses /bin/sh sleep timing")
@pytest.mark.parametrize("bad_timeout", [0, float("nan"), float("inf")])
def test_nonsense_timeout_falls_back_to_default(bad_timeout: float) -> None:
    # With the 300s default applied, a 1.2s command succeeds. If the nonsense
    # value leaked through, timeout=0 would expire immediately and NaN/inf
    # would wait forever / hit the clamp — all observable as a non-success.
    out = execute_command("sleep 1.2; printf 'default_%s\\n' APPLIED", timeout=bad_timeout)
    assert out["success"] is True
    assert "default_APPLIED" in str(out["stdout"])
    # Falling back to the default is not a clamp; no note should appear.
    assert "tool maximum" not in str(out["rendered"])


def test_timeout_schema_teaches_seconds_and_ceiling() -> None:
    # The docstring never reaches the model; the exported JSON Schema is the
    # only channel that can say the unit and the ceiling at call time.
    definition = getattr(execute_command, "_tool_definition", None)
    assert definition is not None
    meta = definition.parameters.get("timeout")
    assert isinstance(meta, dict)
    description = str(meta.get("description") or "")
    assert "Seconds" in description
    assert "600" in description
    assert "clamped" in description


@pytest.mark.skipif(os.name != "posix", reason="process-tree semantics are POSIX-specific")
def test_timeout_drain_survives_grandchild_holding_pipe() -> None:
    # The adversarial drain shape: the shell backgrounds a grandchild that
    # inherits the captured pipes and sleeps. The tree-kill enumerates it via
    # ancestry BEFORE killing, so the pipes close and the drain returns fast
    # WITH both processes' pre-kill output.
    t0 = time.monotonic()
    out = execute_command(
        "(printf 'CHILD_%s\\n' OUT; sleep 30) & printf 'PARENT_%s\\n' OUT; sleep 30",
        timeout=1,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 12, f"drain hung for {elapsed:.1f}s despite the tree-kill"
    assert out["success"] is False
    assert "PARENT_OUT" in str(out["stdout"])
    assert "CHILD_OUT" in str(out["stdout"])


@pytest.mark.skipif(os.name != "posix", reason="signal semantics are POSIX-specific")
def test_timeout_drain_gives_up_and_salvages_when_kill_misses_a_holder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the residual hazard the 5s drain bound exists for: a pipe
    # holder the kill could not reach (fork race / ps failure / stuck in
    # uninterruptible I/O). We replace the tree-kill with a kill of ONLY the
    # direct shell; the backgrounded grandchild keeps the pipes open for 8s
    # (self-expiring, so the test leaks nothing). The drain must give up at
    # ~5s — never pin — and still salvage the bytes read before the kill.
    import signal as _signal

    from abstractcore.tools import process_tree

    def _kill_only_direct_child(proc):  # type: ignore[no-untyped-def]
        try:
            os.kill(proc.pid, _signal.SIGKILL)
        except Exception:
            pass

    monkeypatch.setattr(process_tree, "hard_kill_tree", _kill_only_direct_child)

    t0 = time.monotonic()
    out = execute_command(
        "(printf 'G_%s\\n' OUT; sleep 8) & printf 'P_%s\\n' OUT; sleep 30",
        timeout=1,
    )
    elapsed = time.monotonic() - t0

    # ~1s wait + ~5s bounded drain; never the grandchild's full 8s, and never
    # an unbounded pin.
    assert elapsed < 13, f"drain was not bounded: {elapsed:.1f}s"
    assert out["success"] is False
    rendered = str(out["rendered"])
    assert "the captured output may be incomplete" in rendered
    # Salvage: the pre-kill bytes ride the drain-stage TimeoutExpired on POSIX.
    assert "P_OUT" in str(out["stdout"])


def test_success_render_byte_stable_without_clamp() -> None:
    # Pin: when the clamp does not fire, the ordinary success render is
    # byte-identical to the pre-2026-07-27 format (reconstructed here from
    # the returned fields, since execution time varies per run).
    out = execute_command("printf 'stable_%s\\n' OUT", timeout=10)
    assert out["success"] is True
    expected = "\n".join(
        [
            f"🖥️  Command executed on {out['platform']}",
            f"💻 Command: {out['command']}",
            f"📁 Working directory: {out['working_directory']}",
            f"⏱️  Execution time: {out['duration_s']:.2f}s",
            "🔢 Return code: 0",
            "\n📤 STDOUT:\nstable_OUT\n",
            "\n✅ Command completed successfully",
        ]
    )
    assert out["rendered"] == expected
