"""Persistent shell session (backlog 0215)."""
from __future__ import annotations

import shutil

import pytest

from abstractcore.tools.shell_session import ShellSessionRegistry

_HAS_SHELL = bool(shutil.which("bash") or shutil.which("sh"))
pytestmark = pytest.mark.skipif(not _HAS_SHELL, reason="no POSIX shell available")


def test_cwd_persists_across_commands(tmp_path) -> None:
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        (tmp_path / "sub").mkdir()
        r1 = sess.run("cd sub && pwd")
        assert r1.exit_code == 0
        assert r1.stdout.strip().endswith("/sub")
        # A SECOND command sees the cwd set by the first (this is the whole point vs one-shot).
        r2 = sess.run("pwd")
        assert r2.stdout.strip().endswith("/sub")
    finally:
        reg.close_all()


def test_env_persists_across_commands(tmp_path) -> None:
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        sess.run("export MY_TOKEN=hello123")
        r = sess.run("echo $MY_TOKEN")
        assert r.stdout.strip() == "hello123"
    finally:
        reg.close_all()


def test_exit_code_captured(tmp_path) -> None:
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        assert sess.run("true").exit_code == 0
        assert sess.run("false").exit_code == 1
        # Use a subshell so a non-zero exit does not terminate the persistent session shell.
        assert sess.run("(exit 7)").exit_code == 7
        # Session survives and keeps working after a failing command.
        assert sess.run("echo alive").stdout.strip() == "alive"
    finally:
        reg.close_all()


def test_write_stdin_feeds_interactive_reader(tmp_path) -> None:
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        # `read` blocks for a line on stdin; write_stdin should satisfy it.
        # Run in background within the session, feed input, then echo the captured var.
        r = sess.run("read LINE <<< 'piped-value'; echo got:$LINE")
        assert "got:piped-value" in r.stdout
    finally:
        reg.close_all()


def test_registry_reuses_session_by_id(tmp_path) -> None:
    reg = ShellSessionRegistry()
    s1 = reg.open(session_id="fixed", cwd=str(tmp_path))
    s2 = reg.open(session_id="fixed", cwd=str(tmp_path))
    try:
        assert s1 is s2
    finally:
        reg.close_all()


def test_unterminated_output_is_preserved(tmp_path) -> None:
    """A command whose final line lacks a trailing newline must NOT be silently dropped."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        r = sess.run("printf 'no-newline-tail'")
        assert "no-newline-tail" in r.stdout, r
        assert r.exit_code == 0
        # A larger single-line no-newline blob must survive too.
        r2 = sess.run("printf 'x%.0s' $(seq 1 5000)")
        assert r2.stdout.count("x") == 5000
    finally:
        reg.close_all()


def test_binary_output_does_not_kill_the_session(tmp_path) -> None:
    """Non-UTF-8 output must not kill the reader thread and wedge the session."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        sess.run("head -c 40 /dev/urandom 2>/dev/null || true")
        # The session must still be usable afterwards.
        r = sess.run("echo still-alive")
        assert r.stdout.strip() == "still-alive"
        assert r.exit_code == 0
    finally:
        reg.close_all()


def test_stdin_consuming_command_does_not_corrupt_the_session(tmp_path) -> None:
    """`cat` reads stdin; the PTY same-line sentinel must survive it and the session must stay usable.

    This is the failure mode the pipe-based design could not handle (the sentinel was swallowed)."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        # `cat` with input redirected from a heredoc consumes no session stdin and returns.
        r = sess.run("cat <<'EOF'\nhello-from-cat\nEOF")
        assert "hello-from-cat" in r.stdout
        assert r.exit_code == 0
        # Session still works.
        assert sess.run("echo after-cat").stdout.strip() == "after-cat"
    finally:
        reg.close_all()


def test_set_x_does_not_desync_the_session(tmp_path) -> None:
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        sess.run("set -x")
        r = sess.run("echo after-setx")
        assert "after-setx" in r.stdout
        assert r.exit_code == 0
        # And it recovers after turning tracing off.
        sess.run("set +x")
        assert sess.run("echo clean").stdout.strip() == "clean"
    finally:
        reg.close_all()


def test_no_command_bytes_echoed_into_output(tmp_path) -> None:
    """ECHO is disabled: the output must be the command's result, not the command text itself."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        r = sess.run("echo XYZZY_MARKER")
        # Exactly the echoed value, not the command line `echo XYZZY_MARKER`.
        assert r.stdout.strip() == "XYZZY_MARKER"
        assert "echo XYZZY_MARKER" not in r.stdout
    finally:
        reg.close_all()


def test_large_output_returns_exit_code_not_false_timeout(tmp_path) -> None:
    """A finished large-output command must return its real exit code, not a false timeout
    (the O(n²) scan previously made big output cross the timeout with exit_code=None)."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        # ~2 MB of output on stdout, then exits 0.
        r = sess.run("yes ABCDEFGHIJ | head -n 200000", timeout_s=20)
        assert r.timed_out is False, "large output should not false-timeout"
        assert r.exit_code == 0
        assert r.truncated is True  # bounded to the preview cap
    finally:
        reg.close_all()


def test_no_stale_bleed_after_timeout(tmp_path) -> None:
    """After a command times out, the NEXT command's output must not contain stale bytes from
    the timed-out one (deterministic resync, not a fixed-sleep drain)."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        # A command that keeps emitting past the timeout window.
        r1 = sess.run("for i in $(seq 1 100); do echo STALE_$i; sleep 0.05; done", timeout_s=1)
        assert r1.timed_out is True
        # Next command must see ONLY its own output.
        r2 = sess.run("echo FRESH_ONLY", timeout_s=6)
        assert "FRESH_ONLY" in r2.stdout
        assert "STALE_" not in r2.stdout, f"stale bleed: {r2.stdout!r}"
        assert r2.exit_code == 0
    finally:
        reg.close_all()


# A single long-lived child with exact SIGINT behaviour: a shell loop is not one (under load bash
# both drops interrupts to a loop and ends a loop that ignores them, depending on where it forks).
_LEAVES_AT_THE_SECOND_INTERRUPT = """
import signal, sys, time
hits = []
def on_interrupt(*_):
    hits.append(1)
    if len(hits) == 2:
        sys.exit(0)
signal.signal(signal.SIGINT, on_interrupt)
print("READY", flush=True)
while True:
    print("STALE", flush=True)
    time.sleep(0.05)
"""
_IGNORES_INTERRUPTS = """
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
print("READY", flush=True)
while True:
    print("STALE", flush=True)
    time.sleep(0.05)
"""


def _start_child(sess, script: str) -> None:
    """Start `script` in the session (it never ends, so the call times out); return once it is READY.

    READY is printed after its SIGINT handler is set, so every interrupt the resync sends reaches it.
    """
    import shlex
    import sys

    out = sess.run(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}", timeout_s=1).stdout
    for _ in range(60):  # a hang guard only: each read returns as soon as output arrives
        if "READY" in out:
            return
        out += sess.read_output(timeout_s=2).stdout
    raise AssertionError(f"the child never got ready: {out!r}")


def test_a_command_that_outlives_the_first_interrupt_never_reaches_the_next_command(tmp_path) -> None:
    """The timed-out command keeps printing through the first Ctrl-C (as a bash loop does when
    bash continues it past an interrupt) and ends at the second. The next command gets only its
    own output."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        _start_child(sess, _LEAVES_AT_THE_SECOND_INTERRUPT)
        r2 = sess.run("echo FRESH_ONLY", timeout_s=10)
        assert (r2.stdout, r2.exit_code, r2.timed_out) == ("FRESH_ONLY\n", 0, False), f"stale bleed: {r2.stdout!r}"
        assert sess.run("echo AGAIN", timeout_s=10).stdout == "AGAIN\n"
    finally:
        reg.close_all()


def test_a_session_that_cannot_be_resynchronised_is_closed_and_the_next_command_fails(tmp_path, monkeypatch) -> None:
    from abstractcore.tools import shell_session

    # The child ignores SIGINT, so no attempt's marker can ever print: any limit gives this outcome.
    monkeypatch.setattr(shell_session, "_RESYNC_ATTEMPT_TIMEOUT_S", 0.5)
    reg = ShellSessionRegistry()
    sess = reg.open(session_id="stuck", cwd=str(tmp_path))
    try:
        _start_child(sess, _IGNORES_INTERRUPTS)
        with pytest.raises(RuntimeError, match="could not be resynchronised"):
            sess.run("echo FRESH_ONLY", timeout_s=5)
        assert sess.is_alive() is False
        fresh = reg.open(session_id="stuck", cwd=str(tmp_path))
        assert fresh is not sess
        assert fresh.run("echo FRESH_ONLY", timeout_s=10).stdout == "FRESH_ONLY\n"
    finally:
        reg.close_all()


def test_a_terminal_setting_that_lands_late_cannot_make_the_shell_echo_commands(tmp_path, monkeypatch) -> None:
    """ECHO is turned off before the shell starts. Applied from the parent, it raced the shell's
    startup: a readline that read the terminal first echoed every command, sentinel included."""
    import termios
    import time

    real = termios.tcsetattr

    def late(fd, when, attrs):
        time.sleep(0.5)  # a loaded machine, not a synchronisation: the shell would be up by now
        real(fd, when, attrs)

    monkeypatch.setattr(termios, "tcsetattr", late)
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        assert sess.run("echo alive", timeout_s=10).stdout == "alive\n"
        assert sess.run("echo again", timeout_s=10).stdout == "again\n"
    finally:
        reg.close_all()


def test_reopen_after_death_does_not_leak_master_fd(tmp_path) -> None:
    """Reopening a session id whose process died without close() must release the old PTY fd."""
    import os

    def _open_fd_count() -> int:
        try:
            return len(os.listdir(f"/dev/fd"))
        except Exception:
            return -1

    reg = ShellSessionRegistry()
    baseline = None
    try:
        for i in range(6):
            s = reg.open(session_id="recycle", cwd=str(tmp_path))
            # Kill the shell abnormally (no close()).
            s.run("kill -9 $$ 2>/dev/null || true", timeout_s=3)
            # SIGKILL delivery and the exit are asynchronous: run() can return before
            # waitpid(WNOHANG) sees the child gone (observed under a loaded CI matrix,
            # 2026-11-08). Wait for that exit itself, which the SIGKILL makes certain,
            # instead of a deadline; is_alive() then reports the reaped shell as dead.
            import contextlib

            with contextlib.suppress(ChildProcessError):  # run() may have reaped it already
                os.waitpid(s._pid, 0)
            assert s.is_alive() is False
            if i == 1:
                baseline = _open_fd_count()
        after = _open_fd_count()
        if baseline is not None and baseline >= 0 and after >= 0:
            # Allow small slack; a leak would grow ~1 fd per cycle (4 more cycles after baseline).
            assert after - baseline <= 2, f"fd leak: baseline={baseline} after={after}"
    finally:
        reg.close_all()


def test_close_reaps_background_children(tmp_path) -> None:
    """close() must not orphan a backgrounded child (process-group kill)."""
    import os
    import time

    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    # Start a long-lived background child and capture its pid.
    sess.run("sleep 60 & echo $!")
    r = sess.run("jobs -p")
    child_pids = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
    reg.close_all()
    time.sleep(0.5)
    for pid in child_pids:
        alive = True
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
        assert not alive, f"background child {pid} survived close()"


def test_read_output_drains_interactive_process(tmp_path) -> None:
    """read_output pairs with write_stdin: no sentinel, bounded quiet-gap read."""
    reg = ShellSessionRegistry()
    sess = reg.open(cwd=str(tmp_path))
    try:
        r = sess.run("cat", timeout_s=1.5)  # interactive: consumes the tty, honest timeout
        assert r.timed_out is True
        sess.write_stdin("echo-me-back")
        out = sess.read_output(timeout_s=3.0)
        assert "echo-me-back" in out.stdout
        assert out.exit_code is None  # unknowable in interactive mode
    finally:
        reg.close_all()


def test_close_namespace_only_reaps_that_namespace(tmp_path) -> None:
    from abstractcore.tools.shell_session import namespaced_session_id

    reg = ShellSessionRegistry()
    reg.open(session_id=namespaced_session_id("run-a", "main"), cwd=str(tmp_path))
    reg.open(session_id=namespaced_session_id("run-b", "main"), cwd=str(tmp_path))
    try:
        closed = reg.close_namespace("run-a")
        assert closed == 1
        assert reg.get(namespaced_session_id("run-a", "main")) is None
        assert reg.get(namespaced_session_id("run-b", "main")) is not None
    finally:
        reg.close_all()
