"""Round 12 (R12.1): REAL sandbox tests on macOS — commands run through the real tools
(`execute_command`, `shell_exec`) under /usr/bin/sandbox-exec with a generated profile.

Everything lives under a scratch HOME (tmp_path): the refused folder, the allowed ro/rw rows,
the private workspace and a scratch `.ssh` holding a MARKER (never the operator's files). A
refusal is proven by the marker NOT appearing in the output; a success would print the
marker and turn the test red without exfiltrating anything real."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from abstractcore.tools import sandbox as sb
from abstractcore.tools.common_tools import execute_command
from abstractcore.tools.sandbox import SandboxRow, SandboxSpec
from abstractcore.tools.shell_session import get_shell_session_registry
from abstractcore.tools.shell_tools import shell_exec

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not os.access(sb.SANDBOX_EXEC, os.X_OK),
    reason="macOS sandbox-exec tests: this host is not macOS with /usr/bin/sandbox-exec (Linux runs test_sandbox_linux.py)",
)

REFUSED_MARK = "R12-REFUSED-MARKER-7f3a"
SSH_MARK = "R12-SSH-MARKER-91c2"
RO_MARK = "R12-RO-MARKER-44d1"
OUTSIDE_MARK = "R12-OUTSIDE-MARKER-0b8e"


@pytest.fixture(autouse=True)
def _fresh_host():
    sb._reset_host_for_tests()
    yield
    sb._reset_host_for_tests()


@pytest.fixture
def t(tmp_path, monkeypatch):
    root = Path(os.path.realpath(tmp_path))
    home = root / "home"
    for d in ("home/.ssh", "home/outside", "priv", "refused/inner", "ro", "rw", "home/parent/child"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "refused/secret.txt").write_text(REFUSED_MARK)
    (root / "refused/inner/deep.txt").write_text(REFUSED_MARK)
    (root / "home/.ssh/id_marker").write_text(SSH_MARK)
    (root / "ro/f.txt").write_text(RO_MARK)
    (root / "home/outside/f.txt").write_text(OUTSIDE_MARK)
    (root / "home/parent/secret.txt").write_text(REFUSED_MARK)
    (root / "home/parent/child/ok.txt").write_text("CHILD-OK")
    (root / "rw/link_to_refused").symlink_to(root / "refused/secret.txt")
    monkeypatch.setenv("HOME", str(home))
    return root


def _stamp(t: Path, posture: str = "any_except_denied", default_mode: str = "rw", **kw):
    spec = SandboxSpec(
        private_workspace=str(t / "priv"),
        posture=posture,
        default_mode=default_mode,
        allowed=kw.pop("allowed", (SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw"))),
        refused=kw.pop("refused", (str(t / "refused"),)),
        builtin_refused=kw.pop("builtin_refused", (str(t / "home/.ssh"),)),
        **kw,
    )
    return spec.to_stamp()


def _run(t: Path, command: str, stamp, cwd=None):
    res = execute_command(command, working_directory=str(cwd or (t / "priv")), timeout=30, _sandbox=stamp)
    out = (res.get("stdout") or "") + (res.get("stderr") or "")
    return res, out


def _leaked(res, out) -> bool:
    """The marker's content, or the refused folder's listing (``ls`` prints names only)."""
    listing = [line.strip() for line in (res.get("stdout") or "").splitlines()]
    return REFUSED_MARK in out or "secret.txt" in listing


POSTURES = [("any_except_denied", "rw"), ("any_except_denied", "ro"), ("allowed_only", "rw")]

ESCAPES = [
    "cat {r}/secret.txt",
    "ls {r}",
    "cd {r} && ls && cat secret.txt",
    "cat {t}/rw/link_to_refused",
    "echo $(cat {r}/secret.txt)",
    "python3 -c \"print(open('{r}/secret.txt').read())\"",
    "bash -c 'cat {r}/secret.txt'",
    "find {r} -type f -exec cat {{}} \\;",
    "cp {r}/secret.txt {t}/priv/copied.txt; cat {t}/priv/copied.txt",
    "tar -cf - -C {r} . | tar -xOf -",
    "ln {r}/secret.txt {t}/priv/hard; cat {t}/priv/hard",
    "env -i /bin/cat {r}/inner/deep.txt",
    "sh -c 'cd {t}/refused/inner; cat deep.txt'",
]


@pytest.mark.parametrize("posture,default_mode", POSTURES)
@pytest.mark.parametrize("template", ESCAPES)
def test_refused_folder_is_unreachable(t, posture, default_mode, template):
    cmd = template.format(r=t / "refused", t=t)
    res, out = _run(t, cmd, _stamp(t, posture, default_mode))
    assert "return_code" in res, f"{cmd!r} never ran (blocked before the sandbox): {res.get('error')}"
    assert not _leaked(res, out), f"ESCAPE under {posture}/{default_mode}: {cmd!r} printed the marker"
    assert REFUSED_MARK not in res["rendered"]
    assert res["sandbox"]["kind"] == "macos-sandbox-exec"
    assert "Sandbox: macOS sandbox-exec" in res["rendered"]


@pytest.mark.parametrize("template", ESCAPES)
def test_escape_commands_work_without_the_refusal(t, template):
    # Positive control: the very same command DOES print the marker when the folder is not
    # refused, so the refusal above is the sandbox's doing (not a broken command).
    for f in ("copied.txt", "hard"):
        (t / "priv" / f).unlink(missing_ok=True)
    cmd = template.format(r=t / "refused", t=t)
    res, out = _run(t, cmd, _stamp(t, "any_except_denied", "rw", refused=()))
    assert _leaked(res, out), f"control failed for {cmd!r}: {res['rendered'][-400:]}"


@pytest.mark.parametrize("posture,default_mode", POSTURES)
def test_builtin_refusal_is_unreadable_under_every_posture(t, posture, default_mode):
    # Even when a row "allows" the folder: built-in refusals are absolute.
    stamp = _stamp(t, posture, default_mode, allowed=(SandboxRow(str(t / "home"), "rw"),))
    for cmd in (f"cat {t}/home/.ssh/id_marker", f"ls {t}/home/.ssh", "cat ~/.ssh/id_marker", "cd ~/.ssh && cat *"):
        _, out = _run(t, cmd, stamp)
        assert SSH_MARK not in out, cmd


@pytest.mark.parametrize("posture,default_mode", POSTURES)
def test_ro_row_readable_not_writable(t, posture, default_mode):
    stamp = _stamp(t, posture, default_mode)
    res, out = _run(t, f"cat {t}/ro/f.txt", stamp)
    assert res["success"] and RO_MARK in out
    res, _ = _run(t, f"touch {t}/ro/new.txt", stamp)
    assert not res["success"] and not (t / "ro/new.txt").exists()
    res, _ = _run(t, f"echo x >> {t}/ro/f.txt", stamp)
    assert (t / "ro/f.txt").read_text() == RO_MARK


@pytest.mark.parametrize("posture,default_mode", POSTURES)
def test_rw_row_private_workspace_and_tmpdir_writable(t, posture, default_mode):
    stamp = _stamp(t, posture, default_mode)
    res, _ = _run(t, f"echo ok > {t}/rw/w.txt && echo ok > {t}/priv/p.txt && echo ok > \"$TMPDIR/t.txt\"", stamp)
    assert res["success"], res["rendered"]
    assert (t / "rw/w.txt").read_text().strip() == "ok"
    assert (t / "priv/p.txt").read_text().strip() == "ok"
    assert (t / "priv/.tmp/t.txt").read_text().strip() == "ok"  # TMPDIR is private to the run
    res, out = _run(t, "echo $TMPDIR; python3 -c 'import tempfile;print(tempfile.gettempdir())'", stamp)
    assert out.split() == [str(t / "priv/.tmp")] * 2


def test_any_except_ro_default_cannot_write_outside_rw_rows(t):
    stamp = _stamp(t, "any_except_denied", "ro")
    res, out = _run(t, f"cat {t}/home/outside/f.txt", stamp)
    assert OUTSIDE_MARK in out  # readable: the default mode is read-only, not refused
    for cmd in (f"touch {t}/home/outside/new.txt", f"mkdir {t}/home/outside/d", f"rm {t}/home/outside/f.txt"):
        _run(t, cmd, stamp)
    assert not (t / "home/outside/new.txt").exists() and not (t / "home/outside/d").exists()
    assert (t / "home/outside/f.txt").exists()


def test_any_except_rw_default_writes_outside(t):
    res, _ = _run(t, f"touch {t}/home/outside/new.txt", _stamp(t, "any_except_denied", "rw"))
    assert res["success"] and (t / "home/outside/new.txt").exists()  # positive control


def test_allowed_only_hides_user_data_outside_rows(t):
    stamp = _stamp(t, "allowed_only")
    for cmd in (f"cat {t}/home/outside/f.txt", f"ls {t}/home/outside", "cat ~/outside/f.txt"):
        _, out = _run(t, cmd, stamp)
        assert OUTSIDE_MARK not in out, cmd
    res, _ = _run(t, f"touch {t}/home/outside/new.txt", stamp)
    assert not (t / "home/outside/new.txt").exists()
    res, out = _run(t, "ls /usr/bin >/dev/null && git --version && python3 -c 'print(6*7)'", stamp)
    assert res["success"] and "42" in out, res["rendered"]  # system roots stay usable


def test_nested_rows_most_specific_wins(t):
    # Refused parent with an allowed child (R12.2: valid; the child is reachable).
    stamp = _stamp(
        t,
        "any_except_denied",
        "rw",
        allowed=(SandboxRow(str(t / "home/parent/child"), "rw"),),
        refused=(str(t / "home/parent"),),
    )
    res, out = _run(t, f"cd {t}/home/parent/child && cat ok.txt && touch new.txt", stamp)
    assert res["success"] and "CHILD-OK" in out, res["rendered"]
    _, out = _run(t, f"cat {t}/home/parent/secret.txt; ls {t}/home/parent", stamp)
    assert REFUSED_MARK not in out and "child\n" not in out and out.count("Operation not permitted") == 2
    # A refused row inside an allowed one refuses that subtree.
    stamp2 = _stamp(t, "allowed_only", allowed=(SandboxRow(str(t / "home/parent"), "rw"),), refused=(str(t / "home/parent/child"),))
    _, out = _run(t, f"cat {t}/home/parent/child/ok.txt", stamp2)
    assert "CHILD-OK" not in out
    res, out = _run(t, f"cat {t}/home/parent/secret.txt", stamp2)
    assert REFUSED_MARK in out  # positive control: the allowed parent is readable


def test_cwd_inside_refused_folder_cannot_read(t):
    res, out = _run(t, "cat secret.txt; ls", _stamp(t), cwd=t / "refused")
    assert REFUSED_MARK not in out


def test_fail_closed_when_sandbox_exec_is_missing(t, monkeypatch):
    monkeypatch.setattr(sb, "SANDBOX_EXEC", str(t / "no-such-sandbox-exec"))
    res = execute_command(f"cat {t}/refused/secret.txt", working_directory=str(t / "priv"), _sandbox=_stamp(t))
    assert res["success"] is False
    assert res["sandbox"]["kind"] == "none"
    assert res["error"].startswith("Commands are not sandboxed on this gateway host")
    assert "Sandbox: none — commands refused on this host" in res["rendered"]
    assert REFUSED_MARK not in str(res)


def test_configured_host_env_is_the_only_env(t, monkeypatch):
    monkeypatch.setenv("AF_R12_PROVIDER_KEY", "sk-must-not-leak")
    sb.configure_host(env={"PATH": "/usr/bin:/bin", "HOME": str(t / "home"), "LANG": "C"})
    res, out = _run(t, "env", _stamp(t))
    assert res["success"]
    assert "sk-must-not-leak" not in out and "AF_R12_PROVIDER_KEY" not in out
    assert f"TMPDIR={t / 'priv/.tmp'}" in out


def test_configured_host_refuses_unstamped_command(t):
    sb.configure_host(env={"PATH": "/usr/bin:/bin"})
    res = execute_command("echo should-not-run", working_directory=str(t / "priv"))
    assert res["success"] is False and "should-not-run" not in (res.get("stdout") or "")
    assert res["error"] == sb.REFUSED_NO_SPEC


def test_model_cannot_forge_wider_stamp_flag(t, monkeypatch):
    # The env and the unsandboxed flag never come from the call.
    monkeypatch.setattr(sb, "SANDBOX_EXEC", str(t / "no-such-sandbox-exec"))
    stamp = {**_stamp(t), "unsandboxed_commands_allowed": True, "env": {"X": "1"}}
    res = execute_command("echo ran", working_directory=str(t / "priv"), _sandbox=stamp)
    assert res["success"] is False and "ran" not in (res.get("stdout") or "")


# --- persistent shell session ------------------------------------------------------------


def test_shell_session_is_sandboxed_for_its_whole_life(t):
    stamp = _stamp(t, "allowed_only")
    ns = "r12-test"
    try:
        out = shell_exec("cd /", _registry_namespace=ns, working_directory=str(t / "priv"), _sandbox=stamp)
        assert "Sandbox: macOS sandbox-exec" in out
        out = shell_exec(f"cd {t}/refused; cat secret.txt; cat {t}/refused/secret.txt", _registry_namespace=ns, _sandbox=stamp)
        assert REFUSED_MARK not in out
        out = shell_exec(f"cat {t}/home/outside/f.txt", _registry_namespace=ns, _sandbox=stamp)
        assert OUTSIDE_MARK not in out
        out = shell_exec(f"cat {t}/ro/f.txt; echo hi > {t}/rw/s.txt && echo WROTE", _registry_namespace=ns, _sandbox=stamp)
        assert RO_MARK in out and "WROTE" in out
        # A different (wider) sandbox replaces the session: it never keeps old access, and a
        # narrower call never inherits a wider session either.
        wider = _stamp(t, "any_except_denied", refused=())
        out = shell_exec(f"cat {t}/refused/secret.txt", _registry_namespace=ns, _sandbox=wider)
        assert "new shell session" in out and REFUSED_MARK in out  # positive control
        out = shell_exec(f"cat {t}/refused/secret.txt", _registry_namespace=ns, _sandbox=stamp)
        assert "new shell session" in out and REFUSED_MARK not in out
    finally:
        get_shell_session_registry().close_namespace(ns)


def test_shell_session_fails_closed(t, monkeypatch):
    monkeypatch.setattr(sb, "SANDBOX_EXEC", str(t / "no-such-sandbox-exec"))
    out = shell_exec(f"cat {t}/refused/secret.txt", _registry_namespace="r12-fc", _sandbox=_stamp(t))
    assert out.startswith("Error: Commands are not sandboxed on this gateway host")
    assert get_shell_session_registry().get("r12-fc::main") is None


def test_no_policy_library_use_is_unchanged(t):
    # No host configured and no stamp: the historical behaviour (no sandbox field, no line).
    res = execute_command("echo plain", working_directory=str(t / "priv"))
    assert res["success"] and "sandbox" not in res and "Sandbox:" not in res["rendered"]
