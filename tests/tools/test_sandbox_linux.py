"""Round 12 (R12.1): REAL sandbox tests on Linux — bubblewrap when installed, Landlock (the
allow-list posture) when bwrap is absent. Each test SKIPS WITH ITS REASON when the backend is
not available on this host; nothing passes silently. Scratch folders and markers only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from abstractcore.tools import sandbox as sb
from abstractcore.tools.common_tools import execute_command
from abstractcore.tools.sandbox import SandboxRow, SandboxSpec

MARK = "R12-LINUX-REFUSED-MARKER"
SSH_MARK = "R12-LINUX-SSH-MARKER"
OUTSIDE = "R12-LINUX-OUTSIDE-MARKER"

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux sandbox tests (bwrap/Landlock): this host is not Linux")


def _need_bwrap():
    if not sb._bwrap_path():
        pytest.skip("bubblewrap (bwrap) is not installed on this Linux host")


def _need_landlock(monkeypatch):
    if sb._landlock_abi() <= 0:
        pytest.skip("Landlock is not available on this kernel (needs >= 5.13 with Landlock enabled)")
    monkeypatch.setattr(sb, "_bwrap_path", lambda: None)  # force the Landlock path


@pytest.fixture(autouse=True)
def _fresh_host():
    sb._reset_host_for_tests()
    yield
    sb._reset_host_for_tests()


@pytest.fixture
def t(tmp_path, monkeypatch):
    root = Path(os.path.realpath(tmp_path))
    for d in ("home/.ssh", "home/outside", "priv", "refused", "ro", "rw"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "refused/secret.txt").write_text(MARK)
    (root / "home/.ssh/id_marker").write_text(SSH_MARK)
    (root / "home/outside/f.txt").write_text(OUTSIDE)
    (root / "ro/f.txt").write_text("RO")
    (root / "rw/link").symlink_to(root / "refused/secret.txt")
    monkeypatch.setenv("HOME", str(root / "home"))
    return root


def _stamp(t, posture, default_mode="rw", refused=True):
    return SandboxSpec(
        private_workspace=str(t / "priv"),
        posture=posture,
        default_mode=default_mode,
        allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw")),
        refused=(str(t / "refused"),) if refused else (),
        builtin_refused=(str(t / "home/.ssh"),),
    ).to_stamp()


ESCAPES = [
    "cat {r}/secret.txt",
    "ls {r}",
    "cd {r} && cat secret.txt",
    "cat {t}/rw/link",
    "echo $(cat {r}/secret.txt)",
    "python3 -c \"print(open('{r}/secret.txt').read())\"",
    "bash -c 'cat {r}/secret.txt'",
    "find {r} -type f -exec cat {{}} \\;",
    "cat {t}/home/.ssh/id_marker",
]


def _check_matrix(t, posture, default_mode="rw"):
    for template in ESCAPES:
        cmd = template.format(r=t / "refused", t=t)
        res = execute_command(cmd, working_directory=str(t / "priv"), _sandbox=_stamp(t, posture, default_mode))
        out = (res.get("stdout") or "") + (res.get("stderr") or "")
        assert "return_code" in res, res.get("error")
        assert MARK not in out and SSH_MARK not in out and "secret.txt" not in (res.get("stdout") or ""), cmd
    res = execute_command(f"cat {t}/ro/f.txt; touch {t}/ro/x; echo ok > {t}/rw/w; echo ok > $TMPDIR/t", working_directory=str(t / "priv"), _sandbox=_stamp(t, posture, default_mode))
    assert "RO" in res["stdout"] and not (t / "ro/x").exists()
    assert (t / "rw/w").exists() and (t / "priv/.tmp/t").exists()


@pytest.mark.parametrize("posture,default_mode", [("any_except_denied", "rw"), ("any_except_denied", "ro"), ("allowed_only", "rw")])
def test_bwrap_matrix(t, posture, default_mode):
    _need_bwrap()
    _check_matrix(t, posture, default_mode)
    res = execute_command(f"cat {t}/refused/secret.txt", working_directory=str(t / "priv"), _sandbox=_stamp(t, posture, default_mode, refused=False))
    if posture == "any_except_denied":
        assert MARK in res["stdout"]  # positive control: the refusal is the sandbox's doing


def test_bwrap_ro_default_cannot_write_outside(t):
    _need_bwrap()
    execute_command(f"touch {t}/home/outside/new", working_directory=str(t / "priv"), _sandbox=_stamp(t, "any_except_denied", "ro"))
    assert not (t / "home/outside/new").exists()


def test_landlock_allow_list_matrix(t, monkeypatch):
    _need_landlock(monkeypatch)
    # Landlock cannot refuse inside an allowed row; the refused/.ssh rows sit outside every
    # allowed row here, so the allow-list alone keeps them unreachable.
    _check_matrix(t, "allowed_only")
    res = execute_command(f"cat {t}/home/outside/f.txt", working_directory=str(t / "priv"), _sandbox=_stamp(t, "allowed_only"))
    assert OUTSIDE not in (res.get("stdout") or "")
    assert res["sandbox"]["kind"] == "linux-landlock"


def test_landlock_refuses_any_except_posture(t, monkeypatch):
    _need_landlock(monkeypatch)
    res = execute_command("echo ran", working_directory=str(t / "priv"), _sandbox=_stamp(t, "any_except_denied"))
    assert res["success"] is False and "ran" not in (res.get("stdout") or "")


CHILD = "R14-LINUX-CHILD-OK"


@pytest.fixture
def nested(t):
    """A refused parent holding an allowed child, which holds a refused grandchild (R12.2)."""
    for d in ("home/parent/child/deny",):
        (t / d).mkdir(parents=True, exist_ok=True)
    (t / "home/parent/secret.txt").write_text(MARK)
    (t / "home/parent/child/ok.txt").write_text(CHILD)
    (t / "home/parent/child/deny/x.txt").write_text(MARK)
    return t


def _nested_stamp(t, posture, default_mode="rw", grandchild=True):
    return SandboxSpec(
        private_workspace=str(t / "priv"),
        posture=posture,
        default_mode=default_mode,
        allowed=(SandboxRow(str(t / "home/parent/child"), "rw"),),
        refused=(str(t / "home/parent"),) + ((str(t / "home/parent/child/deny"),) if grandchild else ()),
        builtin_refused=(str(t / "home/.ssh"),),
    ).to_stamp()


@pytest.mark.parametrize("posture,default_mode", [("any_except_denied", "rw"), ("any_except_denied", "ro"), ("allowed_only", "rw")])
def test_bwrap_allowed_child_of_a_refused_parent(nested, posture, default_mode):
    """R14.1 (backlog 1002 item 1): the most specific row wins under bwrap too — the child is
    readable and writable, the parent's other files and the refused grandchild are not."""
    _need_bwrap()
    t = nested
    stamp = _nested_stamp(t, posture, default_mode)
    res = execute_command(f"cat {t}/home/parent/child/ok.txt && echo w > {t}/home/parent/child/new.txt && cat {t}/home/parent/child/new.txt", working_directory=str(t / "priv"), _sandbox=stamp)
    assert res.get("return_code") == 0 and CHILD in res["stdout"] and "w" in res["stdout"], res
    assert (t / "home/parent/child/new.txt").read_text().strip() == "w"
    for cmd in (f"cat {t}/home/parent/secret.txt", f"ls {t}/home/parent", f"cat {t}/home/parent/child/deny/x.txt", f"cd {t}/home/parent/child/deny && cat x.txt", f"echo leak > {t}/home/parent/leak.txt"):
        res = execute_command(cmd, working_directory=str(t / "priv"), _sandbox=stamp)
        out = (res.get("stdout") or "") + (res.get("stderr") or "")
        assert "return_code" in res, res.get("error")
        assert MARK not in out and "secret.txt" not in (res.get("stdout") or ""), (cmd, res)
    assert not (t / "home/parent/leak.txt").exists()


def test_landlock_allowed_child_of_a_refused_parent(nested, monkeypatch):
    """Landlock grants the child even though its parent is refused (Landlock cannot refuse a
    grandchild inside the child, so this spec has none)."""
    _need_landlock(monkeypatch)
    t = nested
    stamp = _nested_stamp(t, "allowed_only", grandchild=False)
    res = execute_command(f"cat {t}/home/parent/child/ok.txt && echo w > {t}/home/parent/child/new.txt", working_directory=str(t / "priv"), _sandbox=stamp)
    assert res.get("return_code") == 0 and CHILD in res["stdout"], res
    assert res["sandbox"]["kind"] == "linux-landlock"
    assert (t / "home/parent/child/new.txt").exists()
    res = execute_command(f"cat {t}/home/parent/secret.txt", working_directory=str(t / "priv"), _sandbox=stamp)
    assert MARK not in (res.get("stdout") or "") + (res.get("stderr") or "")
