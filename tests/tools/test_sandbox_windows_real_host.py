"""A smoke test of the Windows command sandbox on the REAL profile and drives of the host: the
real "user's data" computation, the real state folder and Low folders, the real profile's
folders on the way to a run's folder. It writes access-list entries and integrity labels on
the host's own folders (the profile, the folders at the root of its drives, the host Python's
folders) that stay there, so it runs ONLY where the environment says the host is disposable:
``ABSTRACTCORE_SANDBOX_REAL_HOST=1`` (CI job ``windows-sandbox``). Never set it on a machine
someone uses.

GitHub's Windows runner is an elevated administrator, which differs from a user's machine:
the user may change the permissions of the drive root and of nearly every folder at its root,
so "Deny everything" refuses more of them (never the drive that holds Windows, Program Files or
ProgramData: its folders are refused one by one instead); the token's mirror carries the
Administrators group, so outside the rows only the integrity level refuses writes to system
folders; and the hosted Python lives in a folder at the root of the drive (refused, then read
again through a ``read`` rule)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from abstractcore.tools import sandbox as sb
from abstractcore.tools.common_tools import execute_command
from abstractcore.tools.sandbox import SandboxRow, SandboxSpec

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Windows only"),
    pytest.mark.skipif(os.environ.get("ABSTRACTCORE_SANDBOX_REAL_HOST") != "1",
                       reason="writes sandbox entries on this host's real profile and drives: set ABSTRACTCORE_SANDBOX_REAL_HOST=1 on a disposable host only"),
]  # fmt: skip

MARK = "R12-REAL-HOST-MARKER-6d2f"
SHAPES = [("any_except_denied", "rw"), ("any_except_denied", "ro"), ("allowed_only", "rw")]
PROBE_KEY = "HKCU\\Software\\AbstractCoreSandboxProbe"


@pytest.fixture(scope="module")
def host():
    run = uuid.uuid4().hex[:12]
    home = Path(os.path.realpath(os.path.expanduser("~")))
    data = Path(os.path.realpath(os.environ["LOCALAPPDATA"])) / "AbstractCoreSandboxSmoke"
    documents = home / "Documents" if (home / "Documents").is_dir() else home
    paths = {
        "home": home,
        "data": data,
        "row": home / "AbstractCoreSandboxSmokeRow" / run,
        "marker": documents / f"abstractcore-sandbox-smoke-{run}.txt",
        "outside": home / f"abstractcore-sandbox-smoke-outside-{run}.txt",
    }
    paths["row"].mkdir(parents=True)
    paths["marker"].write_text(MARK)
    yield run, paths
    for key in ("marker", "outside"):
        paths[key].unlink(missing_ok=True)
    shutil.rmtree(paths["row"].parent, ignore_errors=True)
    subprocess.run(["reg", "delete", PROBE_KEY, "/f"], capture_output=True)


def _spec(paths, run, shape, n):
    priv = paths["data"] / "workspaces" / f"{run}-{n}"
    priv.mkdir(parents=True, exist_ok=True)
    return SandboxSpec(private_workspace=str(priv), posture=shape[0], default_mode=shape[1],
                       allowed=(SandboxRow(str(paths["row"]), "rw"),),
                       builtin_refused=(str(paths["data"]), str(paths["home"] / ".ssh")), builtin_allowed=(str(priv),))  # fmt: skip


def _run(spec, command, cwd=None):
    started = time.time()
    res = execute_command(command, working_directory=cwd or spec.private_workspace, timeout=600, _sandbox=spec.to_stamp())
    return res, (res.get("stdout") or "") + (res.get("stderr") or ""), time.time() - started


def _with_entries(spec) -> int:
    """Objects around this policy's rules that carry a sandbox entry: each rule's path, its
    parent, and what is directly inside a carved folder or a folder on the way."""
    import importlib.util

    loader = importlib.util.spec_from_file_location("abstractcore_sandbox_windows_real_host", sb.WINDOWS_LAUNCHER)
    w = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(w)
    seen = set()
    for rule in sb.windows_rules(spec):
        path = rule[1]
        seen.update({path, os.path.dirname(path)})
        if rule[0] in ("carve", "pass") and os.path.isdir(path):
            with os.scandir(path) as entries:
                seen.update(e.path for e in entries)
    count = 0
    for path in seen:
        try:
            count += any(a.sid.startswith("S-1-9-") for a in w._read_dacl(path)[0])
        except Exception:  # noqa: BLE001 - in use, refused to this user: not counted
            pass
    return count


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "-".join(s))
def test_the_sandbox_on_this_hosts_real_profile(host, shape):
    run, paths = host
    confined = shape != ("any_except_denied", "rw")
    spec = _spec(paths, run, shape, SHAPES.index(shape))
    res, out, first = _run(spec, "echo ok")
    assert res["success"] and res["sandbox"]["kind"] == "windows-restricted-token", out[-800:]
    _, _, second = _run(spec, "echo ok")
    print(f"\n{'/'.join(shape)}: first launch {first:.1f}s, second {second:.2f}s, objects with an entry around its rules: {_with_entries(spec)},"
          f" drives not protected: {sb.windows_unprotected_volumes() or 'none'}")
    assert second < 10, "a launch once the rules are in place only reads them"

    git = "git -c user.email=t@example.invalid -c user.name=t"
    tools = (f'echo cmd-ok && powershell -NoProfile -Command "Write-Output ps-ok" && "{sys.executable}" -c "print(\'py-ok\')"'
             f" && git init -q repo && cd repo && echo a> a.txt && {git} add -A && {git} commit -qm m && git rev-parse --show-toplevel && git status --short")  # fmt: skip
    for cwd in (spec.private_workspace, str(paths["row"])):
        res, out, _ = _run(spec, tools, cwd=cwd)
        assert res["success"] and all(m in out for m in ("cmd-ok", "ps-ok", "py-ok")), out[-800:]
        shutil.rmtree(Path(cwd) / "repo", ignore_errors=True)

    _, out, _ = _run(spec, f'type "{paths["marker"]}"')
    assert (MARK in out) == (shape[0] == "any_except_denied"), out[-400:]

    res, out, _ = _run(spec, f'echo x> "{paths["outside"]}" & reg add {PROBE_KEY} /v probe /d x /f')
    wrote = paths["outside"].exists()
    registry = subprocess.run(["reg", "query", PROBE_KEY], capture_output=True).returncode == 0
    paths["outside"].unlink(missing_ok=True)
    subprocess.run(["reg", "delete", PROBE_KEY, "/f"], capture_output=True)
    if confined:
        assert not wrote and not registry, out[-400:]
    res, out, _ = _run(spec, f'echo w> "{paths["row"]}\\w.txt" && echo p> mine.txt')
    assert res["success"] and (paths["row"] / "w.txt").exists(), out[-400:]
