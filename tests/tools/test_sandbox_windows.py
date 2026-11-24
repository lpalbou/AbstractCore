"""REAL sandbox tests on Windows: commands run through the real tool (``execute_command``)
under a restricted token, with the access-list rules and integrity labels written by the
launcher (``abstractcore/tools/sandbox_windows.py``), for the three workspace shapes.

Everything lives under pytest's ``tmp_path``: a scratch "data folder" holding the run's
private workspace (a built-in refusal with its exception, as the gateway lays it out), a
refused folder, a scratch ``.ssh`` holding a MARKER, read-only and read & write rows. A
refusal is proven by the marker NOT appearing in the output; a success would print the
marker and turn the test red without exfiltrating anything real. Rule paths are at least
two folders below ``tmp_path`` (the launcher also writes one entry on a refused row's
parent folder).

What the launcher computes from the machine is faked (``_fake_machine``): the scratch lab
is "the user's data" that "Deny everything, allow listed workspaces" refuses, its state
folder and the folder Windows labels Low are scratch folders too, and every rule path is
checked to lie in the lab before anything runs. The real computation is tested read-only
(``test_the_allow_list_floor_is_computed_from_this_machine``)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from abstractcore.tools import sandbox as sb
from abstractcore.tools.common_tools import execute_command
from abstractcore.tools.sandbox import SandboxRow, SandboxSpec

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows sandbox tests (macOS: test_sandbox_macos.py, Linux: test_sandbox_linux.py)")

REFUSED_MARK = "R12-REFUSED-MARKER-7f3a"
SSH_MARK = "R12-SSH-MARKER-91c2"
RO_MARK = "R12-RO-MARKER-44d1"
DATA_MARK = "R12-DATA-MARKER-5e0c"
OUTSIDE_MARK = "R12-OUTSIDE-MARKER-0b8e"
PY = sys.executable

RW_DEFAULT = ("any_except_denied", "rw")
RO_DEFAULT = ("any_except_denied", "ro")
ALLOW_LIST = ("allowed_only", "rw")
POSTURES = [RW_DEFAULT, RO_DEFAULT, ALLOW_LIST]
ROWS_ONLY = [RO_DEFAULT, ALLOW_LIST]  # writes limited to the rows
IDS = {RW_DEFAULT: "rw-default", RO_DEFAULT: "ro-default", ALLOW_LIST: "allow-list"}
shapes = pytest.mark.parametrize("shape", POSTURES, ids=IDS.get)
rows_only = pytest.mark.parametrize("shape", ROWS_ONLY, ids=IDS.get)
REAL_USER_DATA_ROOTS = sb._windows_user_data_roots
REAL_UNPROTECTED = sb.windows_unprotected_volumes


@pytest.fixture(autouse=True)
def _fresh_host():
    sb._reset_host_for_tests()
    yield
    sb._reset_host_for_tests()


@pytest.fixture
def t(tmp_path) -> Path:
    root = Path(os.path.realpath(tmp_path)) / "lab"
    for d in ("data/workspaces/session-1", "data/workspaces/session-2", "data/secrets", "home/.ssh", "home/outside",
              "refused/inner", "ro/sub", "rw", "parent/child/deny", "machine/LocalLow"):  # fmt: skip
        (root / d).mkdir(parents=True)
    for rel, text in (
        ("refused/secret.txt", REFUSED_MARK), ("refused/inner/deep.txt", REFUSED_MARK),
        ("home/.ssh/id_marker", SSH_MARK), ("home/outside/f.txt", OUTSIDE_MARK),
        ("data/gateway.sqlite3", DATA_MARK), ("data/secrets/token", DATA_MARK), ("data/workspaces/session-2/theirs.txt", DATA_MARK),
        ("ro/f.txt", RO_MARK), ("ro/sub/g.txt", RO_MARK),
        ("parent/secret.txt", REFUSED_MARK), ("parent/child/ok.txt", "CHILD-OK"), ("parent/child/deny/x.txt", REFUSED_MARK),
    ):  # fmt: skip
        (root / rel).write_text(text)
    return root


@pytest.fixture(autouse=True)
def _fake_machine(t, monkeypatch):
    """The lab is the user's data, its state and Low folders the launcher's: no rule, label or
    state file of these tests may land anywhere else."""
    monkeypatch.setattr(sb, "_windows_user_data_roots", lambda: (str(t),))
    monkeypatch.setattr(sb, "_windows_low_places", lambda: (str(t / "machine/LocalLow"),))
    monkeypatch.setattr(sb, "_windows_state_folder", lambda: str(t / "machine/state"))
    monkeypatch.setattr(sb, "windows_unprotected_volumes", lambda: {})  # (this host's real drives aside: see _fake_drive)
    real_rules = sb.windows_rules

    def confined(spec):
        rules = real_rules(spec)
        outside = [r for r in rules for p in r[1:] if not sb._under(p, str(t)) or os.path.normcase(p) == os.path.normcase(str(t.parent))]
        assert not outside, f"a rule outside the lab: {outside}"
        return rules

    monkeypatch.setattr(sb, "windows_rules", confined)


def _priv(t: Path) -> Path:
    return t / "data/workspaces/session-1"


def _spec(t: Path, shape=RW_DEFAULT, **kw) -> SandboxSpec:
    base = dict(
        private_workspace=str(_priv(t)),
        posture=shape[0],
        default_mode=shape[1],
        allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw"), SandboxRow(str(t / "parent/child"), "rw")),
        refused=(str(t / "refused"), str(t / "parent"), str(t / "parent/child/deny")),
        builtin_refused=(str(t / "data"), str(t / "home/.ssh")),
        builtin_allowed=(str(_priv(t)),),
    )
    base.update(kw)
    return SandboxSpec(**base)


def _session(t: Path, n: int, shape, **kw) -> SandboxSpec:
    """The spec of another session of the same gateway (its own folder in the data folder)."""
    folder = t / f"data/workspaces/session-{n}"
    folder.mkdir(exist_ok=True)
    return _spec(t, shape, private_workspace=str(folder), builtin_allowed=(str(folder),), **kw)


def _run(t: Path, command: str, spec: SandboxSpec = None, cwd=None, timeout: int = 60):
    spec = spec or _spec(t)
    res = execute_command(command, working_directory=str(cwd or spec.private_workspace), timeout=timeout, _sandbox=spec.to_stamp())
    return res, (res.get("stdout") or "") + (res.get("stderr") or "")


def _grant_self(path: Path, *flags: str) -> None:
    """Give ``path`` its own access-list entry for this user (and, with /inheritance:r, no
    inherited ones): the shape of an OpenSSH private key or a tempfile folder."""
    subprocess.run(["icacls", str(path), *flags, "/grant", f"{os.environ['USERNAME']}:F"], check=True, capture_output=True)


def _popen(t: Path, spec: SandboxSpec, command, cwd=None) -> subprocess.Popen:
    argv, env = sb.build_sandbox(spec).wrap(command, str(cwd or spec.private_workspace))
    return subprocess.Popen(argv, cwd=cwd or spec.private_workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_for(path: Path, seconds: float = 60) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline and not path.exists():
        time.sleep(0.1)
    assert path.exists(), f"{path} never appeared"


# --- the kind ----------------------------------------------------------------------------


@shapes
def test_this_host_sandboxes_every_shape(t, shape):
    assert sb.host_sandbox_kind(shape[0]) == sb.KIND_WINDOWS
    res, out = _run(t, "echo ok && cd", _spec(t, shape))
    assert res["success"] and "ok" in out and "session-1" in out
    assert res["sandbox"]["kind"] == "windows-restricted-token" and res["sandbox"]["default_mode"] == shape[1]
    assert "Sandbox: Windows restricted token" in res["rendered"]


# --- refusals ----------------------------------------------------------------------------

ESCAPES = [
    'type "{r}\\secret.txt"',
    'dir "{r}"',
    'cd /d "{r}" && dir && type secret.txt',
    'cd /d "{r}\\inner" && type deep.txt',
    'type "{r}\\inner\\deep.txt"',
    'for /f "delims=" %i in (\'type "{r}\\secret.txt"\') do @echo %i',
    'copy "{r}\\secret.txt" copied.txt & type copied.txt',
    'more < "{r}\\secret.txt"',
    'findstr /s "MARKER" "{r}\\*"',
    'powershell -NoProfile -Command "Get-Content \'{r}\\secret.txt\'"',
    'powershell -NoProfile -Command "Get-ChildItem -Recurse \'{r}\' | Get-Content"',
    '"{py}" -c "print(open(r\'{r}\\secret.txt\').read())"',
    '"{py}" -c "import os;[print(open(os.path.join(d,f)).read()) for d,_,fs in os.walk(r\'{r}\') for f in fs]"',
    '"{py}" -c "import subprocess;print(subprocess.run([\'cmd\',\'/c\',\'type\',r\'{r}\\secret.txt\'],capture_output=True,text=True).stdout)"',
    'mklink /j jump "{r}" & type jump\\secret.txt',
    'mklink /h hard.txt "{r}\\secret.txt" & type hard.txt',
    'robocopy "{r}" out /e & type out\\secret.txt',
    'type "\\\\?\\{r}\\secret.txt"',
    'type "\\\\localhost\\{share}\\secret.txt"',
    'cd /d "{t}" && type refused\\secret.txt',
    'cd /d "{r}\\.." && type refused\\secret.txt',
]


def _leaked(res, out) -> bool:
    """The marker's content, or the refused folder's listing (``dir`` ends a line with the name)."""
    return REFUSED_MARK in out or any(line.rstrip().endswith("secret.txt") for line in (res.get("stdout") or "").splitlines())


def _escape(t: Path, template: str) -> str:
    r = t / "refused"
    share = str(r).replace(":", "$", 1)  # C:\x -> C$\x (the administrative share of the drive)
    return template.format(r=r, t=t, py=PY, share=share)


@shapes
@pytest.mark.parametrize("template", ESCAPES)
def test_refused_folder_is_unreachable(t, shape, template):
    res, out = _run(t, _escape(t, template), _spec(t, shape))
    assert "return_code" in res, f"{template!r} never ran: {res.get('error')}"
    assert not _leaked(res, out), f"ESCAPE: {template!r} reached the refused folder"
    assert REFUSED_MARK not in res["rendered"]


@pytest.mark.parametrize("shape,template", [
    (shape, e) for shape in (RW_DEFAULT, RO_DEFAULT) for e in ESCAPES
    # (below Medium integrity Windows refuses a link to what the command may not write: see the next test)
    if "localhost" not in e and not (shape == RO_DEFAULT and "mklink" in e)
], ids=lambda v: IDS.get(v, v) if isinstance(v, tuple) else v)  # fmt: skip
def test_escape_commands_work_without_the_refusal(t, shape, template):
    # Positive control: the very same command DOES print the marker when the folder is not
    # refused, so the refusal above is the sandbox's doing (not a broken command).
    res, out = _run(t, _escape(t, template), _spec(t, shape, refused=(str(t / "parent"), str(t / "parent/child/deny"))))
    assert _leaked(res, out), f"control failed for {template!r}: {out[-300:]}"


@rows_only
def test_links_reach_only_what_the_command_may_write(t, shape):
    # A junction or a hard link made below Medium integrity must point to something the
    # command may write (Windows' own rule): within the rows they work, elsewhere not.
    (t / "rw/w.txt").write_text("W")
    res, out = _run(t, f'mklink /j inrow "{t}\\rw" && mklink /h hard.txt "{t}\\rw\\w.txt" && type inrow\\w.txt && type hard.txt', _spec(t, shape))
    assert res["success"] and out.count("W") >= 2, out
    os.rmdir(_priv(t) / "inrow")  # the junction only (what tools that clean the folder trip over)
    _run(t, f'mklink /j out "{t}\\home" & mklink /h out.txt "{t}\\ro\\f.txt"', _spec(t, shape))
    assert not (_priv(t) / "out").exists() and not (_priv(t) / "out.txt").exists()


@shapes
def test_a_file_with_its_own_permissions_in_a_refused_folder_is_closed_too(t, shape):
    # An OpenSSH private key: its access list does not inherit from its folder.
    _grant_self(t / "home/.ssh/id_marker", "/inheritance:r")
    for cmd in (f'type "{t}\\home\\.ssh\\id_marker"', f'dir "{t}\\home\\.ssh"', f'"{PY}" -c "print(open(r\'{t}\\home\\.ssh\\id_marker\').read())"'):
        res, out = _run(t, cmd, _spec(t, shape))
        assert SSH_MARK not in out and "id_marker" not in (res.get("stdout") or ""), cmd
    assert (t / "home/.ssh/id_marker").read_text() == SSH_MARK  # the user's own programs are untouched


@shapes
def test_builtin_refusal_is_absolute_and_its_exception_works(t, shape):
    # Even when a row "allows" them: the data folder and .ssh stay closed.
    spec = _spec(t, shape, allowed=(SandboxRow(str(t / "home"), "rw"), SandboxRow(str(t / "data/secrets"), "rw")))
    for cmd in (
        f'type "{t}\\data\\gateway.sqlite3"', f'dir "{t}\\data"', f'type "{t}\\data\\secrets\\token"', f'dir /s /b "{t}\\data\\workspaces"',
        f'type "{t}\\data\\workspaces\\session-2\\theirs.txt"', f'type "..\\session-2\\theirs.txt"', f'type "{t}\\home\\.ssh\\id_marker"',
        f'echo x> "{t}\\data\\new.txt"', f'echo x> "{t}\\data\\workspaces\\session-2\\new.txt"',
    ):  # fmt: skip
        res, out = _run(t, cmd, spec)
        assert DATA_MARK not in out and SSH_MARK not in out and "theirs.txt" not in (res.get("stdout") or ""), cmd
    assert not (t / "data/new.txt").exists() and not (t / "data/workspaces/session-2/new.txt").exists()
    res, out = _run(t, f'type "{t}\\home\\outside\\f.txt" && echo mine> note.txt && type note.txt && echo %TEMP%', spec)
    assert OUTSIDE_MARK in out and "mine" in out and str(_priv(t) / ".tmp") in out
    assert (_priv(t) / "note.txt").read_text().strip() == "mine"


@shapes
def test_content_with_its_own_grants_in_a_carved_folder_is_closed(t, shape):
    # Inherited refusals lose to a grant written on the object itself: both shapes below must
    # still be closed (a folder like the ones `tempfile` makes, a file with an explicit grant).
    _grant_self(t / "data/secrets", "/inheritance:r")
    _grant_self(t / "data/gateway.sqlite3")
    for cmd in (f'type "{t}\\data\\secrets\\token"', f'dir "{t}\\data\\secrets"', f'type "{t}\\data\\gateway.sqlite3"'):
        _, out = _run(t, cmd, _spec(t, shape))
        assert DATA_MARK not in out and "token" not in out, cmd


@shapes
def test_each_session_has_its_own_folder_in_the_same_data_folder(t, shape):
    one, two, three = _priv(t), t / "data/workspaces/session-2", t / "data/workspaces/session-3"
    (one / "mine.txt").write_text("ONE")
    res, out = _run(t, f'type mine.txt && type "{two}\\theirs.txt"', _spec(t, shape))
    assert "ONE" in out and DATA_MARK not in out
    shared = lambda: subprocess.run(["icacls", str(t / "data/workspaces")], capture_output=True, text=True).stdout.count("S-1-9-")  # noqa: E731
    entries = shared()
    # The folder closed to the first session is the second session's own.
    res, out = _run(t, f'type theirs.txt && echo two> note.txt && type "{one}\\mine.txt"', _session(t, 2, shape))
    assert DATA_MARK in out and (two / "note.txt").exists() and "ONE" not in out
    # A session folder and other content made after the rules were written: closed by inheritance,
    # and the new session's own.
    three.mkdir()
    (three / "late.txt").write_text("THREE")
    (t / "data/late").mkdir()
    (t / "data/late/x.txt").write_text(DATA_MARK)
    (t / "data/late.txt").write_text(DATA_MARK)
    res, out = _run(t, f'type "{three}\\late.txt" & type "{t}\\data\\late\\x.txt" & type "{t}\\data\\late.txt" & dir "{t}\\data\\workspaces"', _spec(t, shape))
    assert "THREE" not in out and DATA_MARK not in out  # (its name is listed: the folders on the way to a row list names)
    res, out = _run(t, f'type late.txt && echo three> note.txt && type "{one}\\mine.txt"', _session(t, 3, shape))
    assert "THREE" in out and (three / "note.txt").exists() and "ONE" not in out
    assert shared() == entries  # sessions add nothing to the folders they share


@shapes
def test_content_made_later_with_its_own_permissions_in_a_carved_folder_is_closed(t, shape):
    # After the rules are in place, the gateway makes a secrets folder the way Python makes a
    # private folder (permissions of its own, nothing inherited), a file with a grant of its
    # own, and a plain file. All three must be closed to the next command.
    _run(t, "echo first", _spec(t, shape))
    late = t / "data/late-secrets"
    os.mkdir(late, 0o700)
    (late / "sealing.key").write_text(DATA_MARK)
    _grant_self(late, "/inheritance:r")
    (t / "data/late-granted.txt").write_text(DATA_MARK)
    _grant_self(t / "data/late-granted.txt")
    (t / "data/late-plain.txt").write_text(DATA_MARK)
    for cmd in (f'type "{late}\\sealing.key"', f'dir "{late}"', f'type "{t}\\data\\late-granted.txt"', f'type "{t}\\data\\late-plain.txt"',
                f'echo x> "{late}\\new.txt"', f'echo x> "{t}\\data\\late-granted.txt"'):
        res, out = _run(t, cmd, _spec(t, shape))
        assert DATA_MARK not in out and "sealing.key" not in (res.get("stdout") or ""), cmd
    assert (t / "data/late-granted.txt").read_text() == DATA_MARK and not (late / "new.txt").exists()
    assert (late / "sealing.key").read_text() == DATA_MARK  # the gateway's own access is untouched


@shapes
def test_a_busy_file_in_the_data_folder_does_not_stop_commands(t, shape):
    # A file another process holds open without sharing it (a database, a log).
    import msvcrt

    k32 = __import__("ctypes").WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = __import__("ctypes").c_void_p
    handle = k32.CreateFileW(str(t / "data/gateway.sqlite3"), 0xC0000000, 0, None, 3, 0x80, None)
    assert handle not in (None, 2**64 - 1)
    fd = msvcrt.open_osfhandle(handle, 0)
    try:
        res, out = _run(t, f'echo ok & type "{t}\\data\\gateway.sqlite3"', _spec(t, shape))
        assert "ok" in out and DATA_MARK not in out
    finally:
        os.close(fd)
    _, out = _run(t, f'type "{t}\\data\\gateway.sqlite3"', _spec(t, shape))
    assert DATA_MARK not in out


# --- what a real profile and its folders hold, directly inside a carved folder -----------
# (the kinds found on a real profile's chain of folders; the data folder is carved under
# every shape, and its `workspaces` is a folder on the way to the run's own)


def _carve_sid(t: Path, shape) -> str:
    """The SID of the carve the data folder is part of: its own, or (allow-list) the lab's."""
    return _launcher().rule_sid("carve", str(t if shape == ALLOW_LIST else t / "data"))


def _reparse(path: Path, tag: int) -> None:
    """Make ``path`` a reparse point of ``tag`` (a cloud file placeholder, an app alias)."""
    import ctypes
    import struct

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    handle = k32.CreateFileW(str(path), 0xC0000000, 0, None, 3, 0x02000000 | 0x00200000, None)
    buf = struct.pack("<IHH", tag, 16, 0) + b"\x00" * 16
    done = ctypes.c_ulong()
    try:
        assert k32.DeviceIoControl(ctypes.c_void_p(handle), 0x000900A4, buf, len(buf), None, 0, ctypes.byref(done), None), ctypes.get_last_error()
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def _junction(link: Path, target: Path) -> None:
    """A junction shaped like the profile's legacy ones (``Application Data``...): hidden,
    system, its own listing refused to everyone."""
    subprocess.run(["cmd", "/c", "mklink", "/j", str(link), str(target)], check=True, capture_output=True)
    subprocess.run(["icacls", str(link), "/deny", "*S-1-1-0:(RD)"], check=True, capture_output=True)
    subprocess.run(["attrib", "+h", "+s", str(link), "/l"], check=True, capture_output=True)


@shapes
def test_reparse_points_that_are_not_links_are_refused_like_anything_else(t, shape):
    # OneDrive's placeholders and the Store's app aliases are reparse points of their own kind:
    # opening one opens it, not another path. 2.38.0 skipped them, leaving them readable.
    cloud = t / "data/cloud"
    cloud.mkdir()
    (cloud / "inside.txt").write_text(DATA_MARK)
    _reparse(cloud, 0x9000601A)  # IO_REPARSE_TAG_CLOUD_6
    alias = t / "data/alias.exe"
    alias.write_text(DATA_MARK)
    _reparse(alias, 0x8000001B)  # IO_REPARSE_TAG_APPEXECLINK
    res, out = _run(t, f'type "{cloud}\\inside.txt" & dir /b "{cloud}" & "{PY}" -c "print(open(r\'{cloud}\\inside.txt\').read())"', _spec(t, shape))
    assert DATA_MARK not in out and "inside.txt" not in (res.get("stdout") or "")
    w = _launcher()
    assert any(a.sid == _carve_sid(t, shape) and a.type == w.DENY for a in w._read_dacl(str(cloud))[0])
    os.remove(alias)  # (a reparse point that nothing handles: removed by its own name)


@shapes
def test_a_junction_in_a_carved_folder_is_left_alone_and_leads_nowhere_new(t, shape):
    # A lookup through a junction starts again at its target: the target's own rules apply.
    # Writing on the junction would write on its target: here a folder on the way to the row.
    jump = t / "data/Application Data"
    _junction(jump, t / "data/workspaces")
    w = _launcher()
    before = _acl(jump)
    (_priv(t) / "mine.txt").write_text("MINE")
    res, out = _run(t, f'type "{jump}\\session-1\\mine.txt" & type "{jump}\\session-2\\theirs.txt" & dir /b "{jump}\\session-2"', _spec(t, shape))
    assert "MINE" in out and DATA_MARK not in out and "theirs.txt" not in (res.get("stdout") or "")
    assert _acl(jump) == before  # nothing written on the junction ...
    carve = _carve_sid(t, shape)
    ours = sorted((a.type, a.mask) for a in w._read_dacl(str(t / "data/workspaces"))[0] if a.sid == carve and not a.flags & (w.INHERITED | w.INHERIT_ONLY))
    assert ours == sorted([(w.DENY, w.FILE_ALL & ~w.FILE_PASS), (w.ALLOW, w.FILE_PASS)])  # ... nor, through it, a refusal on its target


def test_allow_list_a_legacy_junction_of_the_profile_onto_a_folder_on_the_way(t):
    # The profile's `Application Data` points to `AppData\Roaming`, a folder on the way to
    # the host's Python under a uv install: here the lab is the profile, `data` that folder.
    jump = t / "Application Data"
    _junction(jump, t / "data")
    (_priv(t) / "mine.txt").write_text("MINE")
    res, out = _run(t, f'type "{jump}\\workspaces\\session-1\\mine.txt" & type "{jump}\\gateway.sqlite3" & type "{jump}\\secrets\\token"', _spec(t, ALLOW_LIST))
    assert "MINE" in out and DATA_MARK not in out


@shapes
def test_what_a_carved_folder_holds_hidden_long_named_or_in_use(t, shape):
    (t / "data/desktop.ini").write_text(DATA_MARK)
    subprocess.run(["attrib", "+h", "+s", str(t / "data/desktop.ini")], check=True)
    long_dir = "\\\\?\\" + str(t / "data" / ("L" * 200) / ("M" * 100))
    os.makedirs(long_dir)
    Path(long_dir, "deep.txt").write_text(DATA_MARK)
    res, out = _run(t, f'type "{t}\\data\\desktop.ini" & type "{long_dir}\\deep.txt" & "{PY}" -c "print(open(r\'{long_dir}\\deep.txt\').read())"', _spec(t, shape))
    assert "return_code" in res and DATA_MARK not in out


@shapes
def test_a_child_this_user_cannot_change_stops_commands_only_when_it_can_be_read(t, shape):
    # Readable, but its permissions are not this user's to change (the owner's own right
    # to them is taken by an OWNER RIGHTS entry): it cannot be refused, so nothing runs,
    # and the line names it.
    locked = t / "data/installer-made"
    locked.mkdir()
    (locked / "f.txt").write_text(DATA_MARK)
    _set_acl(locked, "D:P(D;;WDWO;;;{me})(A;OICI;FA;;;{me})(A;;RC;;;OW)")  # (still deletable: pytest cleans up)
    res, out = _run(t, "echo ran> ran.txt", _spec(t, shape))
    assert res.get("return_code") == 126 and "installer-made" in out and not (_priv(t) / "ran.txt").exists()
    (locked / "f.txt").unlink()
    locked.rmdir()
    # Refused to this user too, passing through included (a folder another account made): no
    # refusal is needed, and commands run.
    closed = t / "data/other-account"
    closed.mkdir()
    _set_acl(closed, "D:P(D;;0xC0021;;;{me})(A;OICI;FA;;;{me})(A;;RC;;;OW)")
    res, out = _run(t, f'echo ran> ran.txt & dir "{closed}"', _spec(t, shape))
    assert res.get("return_code") not in (None, 126) and (_priv(t) / "ran.txt").exists(), out[-300:]


@shapes
def test_children_that_come_and_go_while_a_carved_folder_is_checked(t, shape):
    # A temp folder: things are created and deleted while each launch looks at what is new.
    import threading

    stop = threading.Event()

    def churn(folder: Path):
        n = 0
        while not stop.is_set():
            p = folder / f"churn-{n % 50}"
            try:
                p.mkdir() if n % 2 else p.write_text(DATA_MARK)
                (p.rmdir() if p.is_dir() else p.unlink()) if n % 3 else None
            except OSError:
                pass
            n += 1

    threads = [threading.Thread(target=churn, args=(f,)) for f in (t / "data", t / "data/workspaces")]
    for th in threads:
        th.start()
    try:
        for _ in range(8):
            res, out = _run(t, f'echo ok & type "{t}\\data\\churn-0" & type "{t}\\data\\churn-2"', _spec(t, shape))
            assert res.get("return_code") != 126 and "ok" in out and DATA_MARK not in out, out[-300:]
    finally:
        stop.set()
        for th in threads:
            th.join()
    res, out = _run(t, f'for %f in ("{t}\\data\\churn-*") do @type "%f"', _spec(t, shape))
    assert DATA_MARK not in out  # what appeared in the meantime is refused at the next launch


# --- a path's role changes from one run to the next --------------------------------------
# A refused folder with a row inside it, carved under every shape (`profile`: under the
# allow-list it is the user's data). What is made directly inside it after a first command is
# refused by the next launch; a later policy may list it, a policy after that may not.

NO_AUTO_INHERIT = "D:(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;{me})"  # a real profile's: its folders hold unmarked copies


@shapes
@pytest.mark.parametrize("profile_acl", [None, NO_AUTO_INHERIT], ids=["inherits", "like-a-real-profile"])
def test_a_folder_made_after_a_first_run_can_take_any_role_later(t, shape, profile_acl, monkeypatch):
    profile = t / "profile"
    profile.mkdir()
    if profile_acl:
        _set_acl(profile, profile_acl)
    (profile / "keep").mkdir()
    monkeypatch.setattr(sb, "_windows_user_data_roots", lambda: (str(profile),))

    def spec(*rows):
        return _spec(t, shape, allowed=(SandboxRow(str(profile / "keep"), "rw"),) + rows, refused=(str(profile),))

    _run(t, "echo first", spec())
    made = {name: profile / name for name in ("to-rw", "to-ro", "on-the-way", "stays")}
    for name, folder in made.items():
        folder.mkdir()
        (folder / "n.txt").write_text(f"{name}-OK")
    (made["on-the-way"] / "deep").mkdir()
    (made["on-the-way"] / "deep/n.txt").write_text("DEEP-OK")
    reads = " & ".join(f'type "{f}\\n.txt"' for f in made.values()) + f' & type "{made["on-the-way"]}\\deep\\n.txt"'
    _, out = _run(t, reads, spec())  # this launch refuses what is new
    assert "-OK" not in out, out
    later = spec(SandboxRow(str(made["to-rw"]), "rw"), SandboxRow(str(made["to-ro"]), "ro"), SandboxRow(str(made["on-the-way"] / "deep"), "rw"))
    s = {k: str(v) for k, v in made.items()}
    res, out = _run(t, f'type "{s["to-rw"]}\\n.txt" & echo w> "{s["to-rw"]}\\w.txt" & type "{s["to-ro"]}\\n.txt" & echo w> "{s["to-ro"]}\\w.txt"'
                       f' & type "{s["on-the-way"]}\\deep\\n.txt" & echo w> "{s["on-the-way"]}\\deep\\w.txt" & type "{s["on-the-way"]}\\n.txt" & type "{s["stays"]}\\n.txt"', later)  # fmt: skip
    assert "to-rw-OK" in out and (made["to-rw"] / "w.txt").exists(), out
    assert "to-ro-OK" in out and not (made["to-ro"] / "w.txt").exists(), out
    assert "DEEP-OK" in out and (made["on-the-way"] / "deep/w.txt").exists(), out
    assert "on-the-way-OK" not in out and "stays-OK" not in out, out  # a folder on the way shows nothing of its own
    _, out = _run(t, reads, spec())  # the reverse: a policy that no longer lists them refuses them again
    assert "-OK" not in out, out


@shapes
def test_a_folder_with_permissions_of_its_own_in_a_refused_folder_cannot_be_a_row(t, shape):
    # No order of entries lets its run in and keeps every other run out: the command does not
    # run, and says why (2.38.0 left such a row readable to every other run).
    own = t / "parent/own"
    own.mkdir()
    (own / "n.txt").write_text(REFUSED_MARK)
    _grant_self(own, "/inheritance:r")
    _, out = _run(t, f'type "{own}\\n.txt"', _spec(t, shape))
    assert REFUSED_MARK not in out  # refused to a run that does not list it
    res, out = _run(t, "echo ran> ran.txt", _spec(t, shape, allowed=_spec(t, shape).allowed + (SandboxRow(str(own), "rw"),)))
    assert res.get("return_code") == 126 and "permissions of its own" in out and "own" in out and not (_priv(t) / "ran.txt").exists()


@shapes
@pytest.mark.parametrize("place", ["row inside a refused folder", "private workspace", "row outside refused folders"])
def test_content_of_a_read_and_write_row_takes_permission_changes_its_folder_does_not(t, shape, place):
    # Writable includes the permissions of what is inside (ssh-keygen, git, installers set
    # them); the row folder itself keeps what other runs rely on (its guard, its label,
    # another run's rule on it).
    row = {"row inside a refused folder": t / "parent/child", "private workspace": _priv(t), "row outside refused folders": t / "rw"}[place]
    spec = _spec(t, shape)
    _run(t, "echo first", spec)
    made = row / "made-later.txt"
    made.write_text("x")
    (row / "made-later-dir").mkdir()
    res, out = _run(t, f'icacls "{made}" /grant *S-1-1-0:(R) && icacls "{row}\\made-later-dir" /grant *S-1-1-0:(OI)(CI)(R) /t', spec)
    assert res["success"], out[-300:]
    assert "Everyone" in _acl(made) and "Everyone" in _acl(row / "made-later-dir")
    res, out = _run(t, f'icacls "{row}" /grant *S-1-1-0:(OI)(CI)(F) & icacls "{row}" /setintegritylevel (OI)(CI)L', spec)
    ruled = shape != RW_DEFAULT or place != "row outside refused folders"  # (no rule on it otherwise: the user's own folder, as released)
    if ruled:
        assert "Everyone" not in _acl(row) and "Low Mandatory" not in _acl(row), out[-300:]


@shapes
def test_python_makes_private_folders_in_the_rows(t, shape):
    # tempfile.mkdtemp makes its folder with permissions of its own (mode 0o700) and, on a
    # refusal, tries another name, 2**31 times: in 2.38.0's rows inside a refused folder (the
    # run's own folder and its .tmp among them) every pip install and venv hung there.
    for cwd in (_priv(t), t / "parent/child", t / "rw"):
        res, out = _run(t, f'"{PY}" -c "import tempfile,os;d=tempfile.mkdtemp(dir=\'.\');open(os.path.join(d,\'f\'),\'w\').write(\'x\');print(\'made\',d)" '
                           f'&& "{PY}" -c "import tempfile;print(\'tmp\',tempfile.mkdtemp())"', _spec(t, shape), cwd=cwd, timeout=60)  # fmt: skip
        assert res["success"] and out.count("made") == 1 and "tmp " in out, f"{cwd}: {out[-300:]}"


@shapes
def test_a_virtual_environment_with_pip_installs_from_a_local_wheel(t, shape):
    import ensurepip

    bundled = Path(ensurepip.__file__).parent / "_bundled"  # pip's own wheel: no network needed
    venv_python = _priv(t) / "v/Scripts/python.exe"
    res, out = _run(t, f'"{PY}" -m venv v && "{venv_python}" -m pip --version && "{venv_python}" -m pip install -q --no-index --find-links "{bundled}" '
                       f"--force-reinstall pip && echo installed", _spec(t, shape), timeout=240)  # fmt: skip
    assert res["success"] and "installed" in out and "pip " in out, out[-600:]


# --- the launcher's lock -------------------------------------------------------------------

SQUAT = r"""
import ctypes, sys, time
from ctypes import wintypes as W
k = ctypes.WinDLL("kernel32", use_last_error=True)
adv = ctypes.WinDLL("advapi32", use_last_error=True)
class SA(ctypes.Structure):
    _fields_ = [("n", W.DWORD), ("sd", ctypes.c_void_p), ("inherit", W.BOOL)]
sd = ctypes.c_void_p()
adv.ConvertStringSecurityDescriptorToSecurityDescriptorW("D:(D;;GA;;;WD)", 1, ctypes.byref(sd), None)
sa = SA(ctypes.sizeof(SA), sd, False)
k.CreateMutexW.restype = k.CreateFileW.restype = ctypes.c_void_p
held = [k.CreateMutexW(ctypes.byref(sa), True, "Local\\abstractcore-command-sandbox"),
        k.CreateFileW(sys.argv[2] + "\\lock", 0xC0000000, 0, None, 4, 0, None)]  # the old lock's name, the new lock
print("squatter:", [h not in (None, 2**64 - 1) for h in held], flush=True)
open(sys.argv[1], "w").write("held")
time.sleep(60)
"""


@shapes
def test_a_sandboxed_command_cannot_take_or_spoil_the_launchers_lock(t, shape):
    # A named object of the session (2.38.0's lock) could be held by any command, so every
    # later launch waited; made with a refusing access list, every later launch failed. The
    # lock is a file in the state folder, which every policy refuses.
    script, flag = _priv(t) / "squat.py", _priv(t) / "held"
    script.write_text(SQUAT)
    squatter = _popen(t, _spec(t, shape), [PY, str(script), str(flag), str(t / "machine/state")])
    try:
        _wait_for(flag)
        started = time.time()
        res, out = _run(t, "echo later-ok", _session(t, 2, shape), timeout=30)
        assert res["success"] and "later-ok" in out and time.time() - started < 15, out[-300:]
        assert squatter.poll() is None  # it was holding what it could while the later launch ran
    finally:
        squatter.kill()
        out = squatter.communicate(timeout=30)[0]
    assert "squatter: [True, False]" in out, out  # the old name could be taken, the lock file could not


def test_a_held_lock_is_waited_for_a_bounded_time_and_reported(t):
    import ctypes

    w = _launcher()
    state = t / "machine/state"
    state.mkdir(parents=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    holder = k32.CreateFileW(str(state / "lock"), 0xC0000000, 0x1, None, 4, 0, None)  # as a launch holds it
    assert holder not in (None, 2**64 - 1)
    try:
        written = ctypes.c_ulong()
        k32.WriteFile(ctypes.c_void_p(holder), b"process 4242 since 2026-11-24 10:00:00", 38, ctypes.byref(written), None)
        w.LOCK_WAIT = 0.5
        policy = w.read_policy(sb.windows_policy(sb.windows_rules(_spec(t)), _spec(t)))
        started = time.time()
        with pytest.raises(w.SandboxLaunchError, match="held by process 4242 since 2026-11-24 10:00:00"):
            w.prepare(policy)
        assert time.time() - started < 5
    finally:
        k32.CloseHandle(ctypes.c_void_p(holder))
    w.prepare(policy)  # released: the next one takes it


# --- nesting, read-only rows, the rest of the world --------------------------------------


@shapes
def test_nesting_most_specific_row_wins(t, shape):
    res, out = _run(t, f'type "{t}\\parent\\child\\ok.txt" && echo w> "{t}\\parent\\child\\w.txt"', _spec(t, shape))
    assert "CHILD-OK" in out and (t / "parent/child/w.txt").exists()  # an allowed row inside a refused one
    for cmd in (f'type "{t}\\parent\\secret.txt"', f'dir "{t}\\parent"', f'type "{t}\\parent\\child\\deny\\x.txt"', f'dir /s "{t}\\parent\\child\\deny"'):
        _, out = _run(t, cmd, _spec(t, shape))
        assert REFUSED_MARK not in out and "x.txt" not in out, cmd  # ("secret.txt" is listed: names only)


@shapes
def test_tie_refusal_wins_and_ro_beats_rw(t, shape):
    p = str(t / "rw")
    (t / "rw/f.txt").write_text(REFUSED_MARK)
    _, out = _run(t, f'type "{p}\\f.txt"', _spec(t, shape, allowed=(SandboxRow(p, "rw"),), refused=(p,)))
    assert REFUSED_MARK not in out
    (t / "ro/t.txt").write_text("T")
    q = str(t / "ro")
    _, out = _run(t, f'echo x> "{q}\\t.txt" & type "{q}\\f.txt"', _spec(t, shape, allowed=(SandboxRow(q, "rw"), SandboxRow(q, "ro"))))
    assert (t / "ro/t.txt").read_text() == "T" and RO_MARK in out


@shapes
def test_a_read_only_row_inside_a_read_and_write_row_is_read_only(t, shape):
    (t / "rw/frozen").mkdir()
    (t / "rw/frozen/f.txt").write_text(RO_MARK)
    spec = _spec(t, shape, allowed=(SandboxRow(str(t / "rw"), "rw"), SandboxRow(str(t / "rw/frozen"), "ro")))
    res, out = _run(t, f'echo x> "{t}\\rw\\w.txt" & echo x> "{t}\\rw\\frozen\\f.txt" & echo x> "{t}\\rw\\frozen\\new.txt" & type "{t}\\rw\\frozen\\f.txt"', spec)
    assert (t / "rw/w.txt").exists() and RO_MARK in out
    assert (t / "rw/frozen/f.txt").read_text() == RO_MARK and not (t / "rw/frozen/new.txt").exists()


WRITES = [
    'echo x> "{ro}\\f.txt"',
    'echo x>> "{ro}\\f.txt"',
    'echo x> "{ro}\\new.txt"',
    'del /q "{ro}\\f.txt"',
    'ren "{ro}\\f.txt" g.txt',
    'move /y "{ro}\\f.txt" "{rw}\\f.txt"',
    'mkdir "{ro}\\made"',
    'rmdir /s /q "{ro}\\sub"',
    'ren "{ro}" renamed',
    'attrib +h "{ro}\\f.txt"',
    'icacls "{ro}\\f.txt" /grant Everyone:F & echo x> "{ro}\\f.txt"',
    'takeown /f "{ro}\\f.txt" & icacls "{ro}\\f.txt" /reset & echo x> "{ro}\\f.txt"',
    'icacls "{ro}" /setintegritylevel (OI)(CI)L & echo x> "{ro}\\f.txt"',
    '"{py}" -c "open(r\'{ro}\\sub\\g.txt\',\'w\').write(\'x\')"',
    'powershell -NoProfile -Command "Set-Content \'{ro}\\f.txt\' x; Remove-Item -Recurse -Force \'{ro}\\sub\'"',
]


@shapes
@pytest.mark.parametrize("template", WRITES)
def test_read_only_row_reads_and_never_writes(t, shape, template):
    res, out = _run(t, template.format(ro=t / "ro", rw=t / "rw", py=PY) + f' & type "{t}\\ro\\f.txt"', _spec(t, shape))
    assert RO_MARK in out, f"{template!r}: the read-only row is no longer readable: {out[-300:]}"
    assert (t / "ro/f.txt").read_text() == RO_MARK and (t / "ro/sub/g.txt").read_text() == RO_MARK
    assert sorted(p.name for p in (t / "ro").iterdir()) == ["f.txt", "sub"] and not (t / "rw/f.txt").exists()
    assert (os.stat(t / "ro/f.txt").st_file_attributes & 0x2) == 0  # not hidden


@shapes
def test_refusals_cannot_be_lifted_from_inside(t, shape):
    for cmd in (
        f'icacls "{t}\\refused" /reset /t & type "{t}\\refused\\secret.txt"',
        f'icacls "{t}\\refused" /grant Everyone:F /t & type "{t}\\refused\\secret.txt"',
        f'takeown /f "{t}\\refused" /r /d y & icacls "{t}\\refused" /reset /t & type "{t}\\refused\\secret.txt"',
        f'ren "{t}\\refused" gone & type "{t}\\gone\\secret.txt"',
        f'rmdir /s /q "{t}\\refused"',
        f'icacls "{t}" /reset /t & type "{t}\\refused\\secret.txt"',
    ):
        _, out = _run(t, cmd, _spec(t, shape))
        assert REFUSED_MARK not in out, cmd
    assert (t / "refused/secret.txt").read_text() == REFUSED_MARK


def test_the_rest_of_the_world_stays_read_and_write(t):
    res, out = _run(t, f'type "{t}\\home\\outside\\f.txt" && echo w> "{t}\\home\\outside\\w.txt" && echo w> "{t}\\rw\\w.txt" && type "{t}\\ro\\f.txt"')
    assert res["success"] and OUTSIDE_MARK in out and RO_MARK in out
    assert (t / "home/outside/w.txt").exists() and (t / "rw/w.txt").exists()
    assert (t / "refused/secret.txt").read_text() == REFUSED_MARK  # and the user's own programs still read everything


def test_read_only_default_reads_everything_but_the_refusals(t):
    (t / "loose.txt").write_text(OUTSIDE_MARK)
    res, out = _run(t, f'type "{t}\\home\\outside\\f.txt" & type "{t}\\loose.txt" & dir /b "{t}" & type "{t}\\ro\\f.txt"', _spec(t, RO_DEFAULT))
    assert out.count(OUTSIDE_MARK) == 2 and RO_MARK in out and "home" in (res.get("stdout") or "")


# --- writes limited to the rows ----------------------------------------------------------

OUTSIDE_WRITES = [
    'echo x> "{t}\\home\\outside\\w.txt"',
    'echo x> "{t}\\w.txt"',
    'mkdir "{t}\\made"',
    'copy nul "{t}\\home\\w.txt"',
    '"{py}" -c "open(r\'{t}\\home\\outside\\w.txt\',\'w\').write(\'x\')"',
    'powershell -NoProfile -Command "Set-Content \'{t}\\w.txt\' x"',
    'icacls "{t}\\home\\outside" /setintegritylevel (OI)(CI)L & echo x> "{t}\\home\\outside\\w.txt"',
    'icacls "{t}\\home\\outside" /grant Everyone:F & echo x> "{t}\\home\\outside\\w.txt"',
    'move "{t}\\home\\outside\\f.txt" "{t}\\rw\\stolen.txt"',
    'del /q "{t}\\home\\outside\\f.txt"',
]


@rows_only
@pytest.mark.parametrize("template", OUTSIDE_WRITES)
def test_nothing_outside_the_rows_is_writable(t, shape, template):
    _run(t, template.format(t=t, py=PY), _spec(t, shape))
    assert not any((t / p).exists() for p in ("home/outside/w.txt", "w.txt", "made", "home/w.txt", "rw/stolen.txt"))
    assert (t / "home/outside/f.txt").read_text() == OUTSIDE_MARK
    w = _launcher()
    assert w._label(str(t / "home/outside")) is None  # no label was put on it
    assert not any(a.sid == "S-1-1-0" for a in w._read_dacl(str(t / "home/outside"))[0])


GIT = "git -c user.email=t@example.invalid -c user.name=t"


@shapes
def test_git_works_in_the_private_workspace_inside_the_data_folder(t, shape):
    # The gateway's layout: the run's folder lies in the refused data folder, carved. git turns
    # its working folder into its long form, which looks each folder above up in its parent.
    sub = _priv(t) / "repo/sub"
    sub.mkdir(parents=True)
    spec = _spec(t, shape)
    res, out = _run(t, f"git init -q && echo a> a.txt && {GIT} add -A && {GIT} commit -qm first && git status --short && git rev-parse --show-toplevel",
                    spec, cwd=_priv(t) / "repo")  # fmt: skip
    assert res["success"] and "repo" in out, out[-500:]
    res, out = _run(t, f"echo b> b.txt && {GIT} add -A && {GIT} commit -qm second && git rev-parse --show-toplevel && git log --oneline", spec, cwd=sub)
    assert res["success"] and "second" in out and "first" in out, out[-500:]
    # What the folders on the way show: names, never content.
    res, out = _run(t, f'dir /b "{t}\\data" & dir /b "{t}\\data\\workspaces" & type "{t}\\data\\gateway.sqlite3" & dir /b "{t}\\data\\workspaces\\session-2"'
                       f' & type "{t}\\data\\workspaces\\session-2\\theirs.txt" & dir /b "{t}\\data\\secrets"', spec)  # fmt: skip
    names = (res.get("stdout") or "").split()
    assert {"gateway.sqlite3", "secrets", "workspaces", "session-1", "session-2"} <= set(names), names
    assert DATA_MARK not in out and "theirs.txt" not in names and "token" not in names


@shapes
def test_git_works_in_a_row_whose_path_has_short_names_below_a_refused_folder(t, shape):
    # git's last resort for its working folder, GetLongPathNameW, looks up in its parent each
    # folder whose name could itself be a short (8.3) name: `child` in the refused `parent`
    # (`src` in a refused project, `AppData` in the profile). 2.38.0 refused listing a carved
    # folder, and git stopped with "Unable to read current working directory". (The gateway's
    # `workspaces\session-<id>` has no such name: git worked there.)
    row = t / "parent/child"
    res, out = _run(t, f"git init -q && {GIT} commit -q --allow-empty -m m && git status --short && git rev-parse --show-toplevel", _spec(t, shape), cwd=row)
    assert res["success"] and "child" in out, out[-400:]
    (row / "sub").mkdir(exist_ok=True)
    res, out = _run(t, "git rev-parse --show-prefix", _spec(t, shape), cwd=row / "sub")
    assert res["success"] and "sub/" in out, out[-400:]


def test_carved_folders_written_by_2_38_are_upgraded_in_place(t):
    # 2.38.0 refused listing a carved folder (git could not start below it). Its entries, for
    # the very same rule SIDs, are on users' and CI machines: they are replaced, not added to.
    w = _launcher()
    old_pass = w.FILE_PASS_2_38
    w.RULES["carve"] = ([(w.DENY, w.FILE_ALL & ~old_pass)], [(w.DENY, w.FILE_ALL)])
    w.FILE_PASS, w._upgraded = old_pass, lambda aces, control, sid: (aces, control)
    policy = w.read_policy(sb.windows_policy(sb.windows_rules(_spec(t)), _spec(t)))
    w.prepare(policy)  # as 2.38.0 wrote them
    carve = w.rule_sid("carve", str(t / "data"))

    def entries(path):
        return sorted((a.type, hex(a.mask)) for a in w._read_dacl(str(path))[0] if a.sid == carve and not a.flags & (w.INHERITED | w.INHERIT_ONLY))

    assert entries(t / "data") == [(w.DENY, hex(0x0F015F))] and entries(t / "data/workspaces") == [(w.ALLOW, hex(0x1000A0)), (w.DENY, hex(0x0F015F))]
    res, out = _run(t, f"git init -q && {GIT} commit -q --allow-empty -m m && git rev-parse --show-toplevel", cwd=_priv(t))
    assert res["success"], out[-500:]
    assert entries(t / "data") == [(w.DENY, hex(0x0F015E))] and entries(t / "data/workspaces") == [(w.ALLOW, hex(0x1000A1)), (w.DENY, hex(0x0F015E))]
    _, out = _run(t, f'type "{t}\\data\\gateway.sqlite3" & type "{t}\\data\\workspaces\\session-2\\theirs.txt"')
    assert DATA_MARK not in out


@rows_only
def test_the_registry_is_not_writable(t, shape):
    key = "HKCU\\Software\\AbstractCoreSandboxProbe"
    try:
        for cmd in (f"reg add {key} /v probe /d x /f", f'powershell -NoProfile -Command "New-Item -Force \'HKCU:\\Software\\AbstractCoreSandboxProbe\'"',
                    f'"{PY}" -c "import winreg;winreg.CreateKey(winreg.HKEY_CURRENT_USER,r\'Software\\AbstractCoreSandboxProbe\')"'):
            res, _ = _run(t, cmd, _spec(t, shape))
            assert res.get("return_code") not in (None, 0), cmd
            assert subprocess.run(["reg", "query", key], capture_output=True).returncode != 0, cmd
        res, out = _run(t, 'reg query "HKCU\\Environment"', _spec(t, shape))  # reading it still works
        assert res["success"], out
    finally:
        subprocess.run(["reg", "delete", key, "/f"], capture_output=True)


@shapes
def test_rows_the_private_workspace_and_its_temp_folder_are_writable_with_every_tool(t, shape):
    git = f'git -C "{t}\\rw\\repo"'
    command = (
        f'echo a> "{t}\\rw\\a.txt" && echo p> mine.txt && echo q> "%TEMP%\\q.txt"'
        f' && powershell -NoProfile -Command "Set-Content \'{t}\\rw\\b.txt\' b; Get-ChildItem \'{t}\\rw\' | Out-Null"'
        f' && "{PY}" -c "import tempfile,subprocess,sys;open(r\'{t}\\rw\\c.txt\',\'w\').write(\'c\');tempfile.mkstemp();'
        f'print(subprocess.run([sys.executable,\'-c\',\'print(7*6)\'],capture_output=True,text=True).stdout.strip())"'
        f' && git init -q "{t}\\rw\\repo" && echo r> "{t}\\rw\\repo\\r.txt" && {git} add -A'
        f' && {git} -c user.email=t@example.invalid -c user.name=t commit -qm m && {git} log --oneline'
    )
    res, out = _run(t, command, _spec(t, shape), timeout=120)
    assert res["success"], out[-600:]
    assert "42" in out and " m" in out
    assert all((t / "rw" / n).exists() for n in ("a.txt", "b.txt", "c.txt", "repo/.git/HEAD"))
    assert (_priv(t) / "mine.txt").exists() and (_priv(t) / ".tmp/q.txt").exists()


def test_a_folder_windows_labels_low_is_not_writable_under_a_read_only_default(t):
    # A Low program (and so a command limited by integrity) may write a folder labelled Low.
    low = t / "machine/LocalLow"
    subprocess.run(["icacls", str(low), "/setintegritylevel", "(OI)(CI)L"], check=True, capture_output=True)
    _run(t, f'echo x> "{low}\\w.txt" & mkdir "{low}\\made"', _spec(t, RO_DEFAULT))
    assert not (low / "w.txt").exists() and not (low / "made").exists()
    _run(t, f'echo x> "{low}\\w.txt"', _spec(t, RW_DEFAULT))  # the user's own write, where everything is writable
    assert (low / "w.txt").exists()


# --- the allow-list posture --------------------------------------------------------------


def test_allow_list_hides_user_data_outside_the_rows(t):
    (t / "loose.txt").write_text(OUTSIDE_MARK)
    spec = _spec(t, ALLOW_LIST)
    for cmd in (
        f'type "{t}\\home\\outside\\f.txt"', f'type "{t}\\loose.txt"', f'dir /s /b "{t}\\home"', f'copy "{t}\\loose.txt" c.txt & type c.txt',
        f'"{PY}" -c "print(open(r\'{t}\\home\\outside\\f.txt\').read())"',
        f'powershell -NoProfile -Command "Get-ChildItem -Recurse \'{t}\' | Get-Content"',
        f'cd /d "{t}\\home\\outside" && type f.txt', f'robocopy "{t}\\home" out /e & type out\\outside\\f.txt',
    ):
        res, out = _run(t, cmd, spec)
        stdout = res.get("stdout") or ""
        assert OUTSIDE_MARK not in out and "outside" not in stdout.replace(str(t), ""), cmd
    res, out = _run(t, f'type "{t}\\ro\\f.txt" & type "%SystemRoot%\\win.ini" & dir /b "%ProgramFiles%" & "{PY}" -c "print(6*7)"', spec)
    assert RO_MARK in out and "[fonts]" in out.lower() and "42" in out  # the rows, system roots and the host's Python


def test_allow_list_the_folders_on_the_way_to_a_row_list_names_only(t):
    # What git needs (see the launcher): the names directly inside the refused folder that
    # holds the rows; nothing below them, no content.
    (t / "loose.txt").write_text(OUTSIDE_MARK)
    res, out = _run(t, f'dir /b "{t}" & dir /b "{t}\\home" & type "{t}\\loose.txt"', _spec(t, ALLOW_LIST))
    names = (res.get("stdout") or "").split()
    assert "home" in names and "loose.txt" in names and "outside" not in names and OUTSIDE_MARK not in out


def test_allow_list_a_row_at_or_above_the_user_data_opens_it(t):
    # As on the other platforms: a row decides inside it, even over the posture's floor.
    spec = _spec(t, ALLOW_LIST, allowed=(SandboxRow(str(t / "home"), "ro"),))
    res, out = _run(t, f'type "{t}\\home\\outside\\f.txt" & echo x> "{t}\\home\\outside\\w.txt"', spec)
    assert OUTSIDE_MARK in out and not (t / "home/outside/w.txt").exists()
    assert ["carve", str(t)] in sb.windows_rules(spec)


def test_the_allow_list_floor_is_computed_from_this_machine():
    # Read-only: what "Deny everything" would refuse on this machine.
    roots = REAL_USER_DATA_ROOTS()
    home = os.path.realpath(os.path.expanduser("~"))
    # The profile is a root, or lies in one (an administrator may change C:\Users itself).
    assert any(root == home or sb._under(home, root) for root in roots), roots
    system =[os.path.realpath(os.environ[v]) for v in ("SystemRoot", "ProgramFiles", "ProgramData") if os.environ.get(v)]
    for root in roots:
        assert not any(sb._under(s, root) for s in system), (root, "would hide a system root")
        assert not any(o != root and sb._under(root, o) for o in roots)
        if root != home:
            assert sb._may_change_permissions(root), root  # every refusal can be written
    assert len(roots) == len({os.path.normcase(r) for r in roots})


# --- drives the sandbox cannot protect (no access lists, network shares) -----------------
# A scratch folder plays the drive (it is NTFS, so what the launcher does NOT write there
# shows): the volume probe is injected, the real one is checked read-only below.

def _fake_drive(t: Path, monkeypatch):
    drive = t / "exfat"
    drive.mkdir()
    why = "drive X: has no permissions (exFAT): commands can read and change it"
    monkeypatch.setattr(sb, "windows_unprotected_volumes", lambda: {str(drive) + "\\": why})
    return drive, why


@shapes
def test_an_unprotected_drive_is_shown_and_never_written(t, shape, monkeypatch):
    drive, why = _fake_drive(t, monkeypatch)
    (drive / "proj").mkdir()
    spec = _spec(t, shape, allowed=_spec(t, shape).allowed + (SandboxRow(str(drive / "proj"), "rw"),))
    res, out = _run(t, "echo ok", spec)
    assert res["success"] and "ok" in out, out[-300:]
    if shape == RW_DEFAULT:  # everything is readable and writable anyway: nothing to say
        assert "unprotected" not in res["sandbox"]
    else:
        assert res["sandbox"]["unprotected"] == [str(drive) + "\\"] and why in res["sandbox"]["unprotected_note"] and why in res["rendered"]
    w = _launcher()
    written = [a for a in w._read_dacl(str(drive / "proj"))[0] if a.sid.startswith("S-1-9-") and not a.flags & w.INHERITED]
    assert not written and w._label(str(drive / "proj")) is None  # nothing written there (what it inherits is the lab's)
    assert str(drive / "proj") not in (t / "machine/state/labelled").read_text() if (t / "machine/state/labelled").exists() else True


@shapes
@pytest.mark.parametrize("row", ["refused", "read-only"])
def test_a_refused_or_read_only_row_on_an_unprotected_drive_stops_the_command(t, shape, row, monkeypatch):
    drive, why = _fake_drive(t, monkeypatch)
    (drive / "secret").mkdir()
    kw = {"refused": _spec(t, shape).refused + (str(drive / "secret"),)} if row == "refused" else {"allowed": _spec(t, shape).allowed + (SandboxRow(str(drive / "secret"), "ro"),)}
    res, out = _run(t, "echo ran> ran.txt", _spec(t, shape, **kw))
    assert res.get("return_code") == 126 and str(drive / "secret") in out and why in out and not (_priv(t) / "ran.txt").exists(), out[-300:]


def test_this_hosts_unprotected_drives_are_never_in_the_allow_list_floor():
    # Read-only: on a host with an exFAT or FAT drive (or a mapped share), it is reported and
    # never a root of "the user's data" (no refusal could be written there).
    unprotected = REAL_UNPROTECTED()
    assert all(not any(sb._under(root, drive) for drive in unprotected) for root in REAL_USER_DATA_ROOTS())
    for drive, why in unprotected.items():
        assert why.startswith(f"drive {drive[0]}:") and "commands can read and change it" in why


# --- runs that share folders -------------------------------------------------------------


def _shared(t: Path) -> Path:
    folder = t / "shared"
    folder.mkdir(exist_ok=True)
    (folder / "f.txt").write_text(RO_MARK)
    return folder


def _owner(t: Path) -> SandboxSpec:
    """The run that has the shared folder read & write (read-only default)."""
    return _spec(t, RO_DEFAULT, allowed=(SandboxRow(str(t / "shared"), "rw"),))


def _others(t: Path):
    """Other sessions' runs that may not write the shared folder: read-only by default
    listing it read-only or not at all, the allow-list posture listing it read-only."""
    ro_row = (SandboxRow(str(t / "shared"), "ro"),)
    return {
        "ro-default, row ro": _session(t, 2, RO_DEFAULT, allowed=ro_row),
        "ro-default, no row": _session(t, 3, RO_DEFAULT, allowed=()),
        "allow-list, row ro": _session(t, 4, ALLOW_LIST, allowed=ro_row),
    }


def _try_writes(t: Path, spec: SandboxSpec, tag: str) -> str:
    s = t / "shared"
    _, out = _run(t, f'echo x> "{s}\\{tag}.txt" & echo x>> "{s}\\f.txt" & mkdir "{s}\\{tag}" & ren "{s}\\f.txt" g.txt & type "{s}\\f.txt"'
                     f' & "{PY}" -c "open(r\'{s}\\{tag}-py.txt\',\'w\')"', spec)  # fmt: skip
    return out


def _untouched(t: Path) -> bool:
    return sorted(p.name for p in (t / "shared").iterdir() if not p.name.startswith("owner")) == ["f.txt"] and (t / "shared/f.txt").read_text() == RO_MARK


def test_a_folder_one_run_writes_stays_unwritable_to_the_others_afterwards(t):
    _shared(t)
    res, out = _run(t, f'echo o> "{t}\\shared\\owner.txt"', _owner(t))
    assert (t / "shared/owner.txt").exists(), out
    for name, spec in _others(t).items():
        out = _try_writes(t, spec, "other")
        assert _untouched(t), name
        assert RO_MARK in out, f"{name}: the row it lists is readable"
    _run(t, f'echo o> "{t}\\shared\\owner2.txt"', _owner(t))
    assert (t / "shared/owner2.txt").exists()  # and still the first run's own
    _run(t, f'echo x> "{t}\\shared\\rw-default.txt"', _session(t, 5, RW_DEFAULT))
    assert (t / "shared/rw-default.txt").exists()  # where everything is writable, so is this


def test_a_folder_one_run_writes_is_unwritable_to_the_others_while_it_runs(t):
    _shared(t)
    started = _priv(t) / "started"
    owner = _popen(t, _owner(t), f'echo o> "{t}\\shared\\owner.txt" && echo s> "{started}" && "{PY}" -c "import time;time.sleep(25)"')
    try:
        _wait_for(started)
        for name, spec in _others(t).items():
            _try_writes(t, spec, "other")
            assert _untouched(t), name
        assert owner.poll() is None  # the overlap was real
    finally:
        owner.kill()
        owner.communicate(timeout=30)


@pytest.mark.parametrize("narrow_first", [True, False])
def test_rows_of_two_runs_nested_in_each_other(t, narrow_first):
    outer = t / "shared"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    wide = _spec(t, RO_DEFAULT, allowed=(SandboxRow(str(outer), "rw"),))
    narrow = _session(t, 2, RO_DEFAULT, allowed=(SandboxRow(str(inner), "rw"),))
    for spec in (narrow, wide) if narrow_first else (wide, narrow):  # who labels first
        _run(t, "echo prepare", spec)
    _run(t, f'echo n> "{inner}\\narrow.txt" & echo n> "{outer}\\narrow.txt"', narrow)
    _run(t, f'echo w> "{inner}\\wide.txt" & echo w> "{outer}\\wide.txt"', wide)
    assert (inner / "narrow.txt").exists() and not (outer / "narrow.txt").exists()
    assert (inner / "wide.txt").exists() and (outer / "wide.txt").exists()


def test_the_remaining_window_a_command_already_running_when_a_folder_is_first_labelled(t):
    # Documents the limit the launcher states: a command that started before a folder was
    # labelled for the first time is not guarded against it. A run started after it is.
    fresh = t / "fresh"
    fresh.mkdir()
    go, ready = fresh / "go", t / "data/workspaces/session-2/ready"  # go: written by the owner, readable to both
    script = t / "data/workspaces/session-2/early.py"
    script.write_text(f"import os, time\nwhile not os.path.exists(r'{go}'):\n    time.sleep(0.1)\nopen(r'{fresh}\\early.txt', 'w')\n")
    early = _popen(t, _session(t, 2, RO_DEFAULT, allowed=()), f'echo r> "{ready}" && "{PY}" "{script}"')
    try:
        _wait_for(ready)
        _run(t, f'echo o> "{fresh}\\owner.txt" & echo g> "{go}"', _spec(t, RO_DEFAULT, allowed=(SandboxRow(str(fresh), "rw"),)))
        early.communicate(timeout=60)
    finally:
        early.kill()
    assert (fresh / "owner.txt").exists()
    assert (fresh / "early.txt").exists()  # the window
    _run(t, f'echo l> "{fresh}\\late.txt"', _session(t, 3, RO_DEFAULT, allowed=()))
    assert not (fresh / "late.txt").exists()


def test_a_killed_launcher_leaves_nothing_that_lets_another_run_write(t):
    _shared(t)
    started = _priv(t) / "started"
    owner = _popen(t, _owner(t), f'echo s> "{started}" && "{PY}" -c "import time;time.sleep(120)"')
    _wait_for(started)
    owner.kill()  # as a timeout does
    owner.communicate(timeout=30)
    w = _launcher()
    # By design the label and the guard stay (written once, like the rules) ...
    assert w._label(str(t / "shared")) == (w.CHILDREN, w.WRITER)
    assert os.path.normcase(str(t / "shared")) in {os.path.normcase(p) for p in (t / "machine/state/labelled").read_text().splitlines()}
    # ... and grant nothing: not to another run, not to a Low program of the user's.
    for name, spec in _others(t).items():
        _try_writes(t, spec, "other")
        assert _untouched(t), name
    low = w.restricted_token([], 4096)
    try:
        assert w.launch(f'cmd /c "echo x> "{t}\\shared\\low.txt""', low) != 0 and not (t / "shared/low.txt").exists()
    finally:
        w.CloseHandle(low)
    _run(t, f'echo o> "{t}\\shared\\owner.txt"', _owner(t))
    assert (t / "shared/owner.txt").exists()


def test_the_list_of_labelled_folders_forgets_only_removed_folders(t):
    w = _launcher()
    state = t / "machine/state"
    w._make_state(str(state))
    kept, removed, away = t / "rw", t / "gone", t / "drive-away/proj"
    removed.mkdir()
    assert w._register(str(state), [str(kept), str(removed), str(away)]) == [str(kept), str(removed), str(away)]
    removed.rmdir()  # removed: its label and guard went with it
    assert w._register(str(state), []) == [str(kept), str(away)]  # a folder whose drive is away stays listed
    assert (state / "labelled").read_text().splitlines()[1:-1] == [str(kept), str(away)]


@shapes
def test_no_command_reads_or_changes_the_launchers_state(t, shape):
    # The list of labelled folders names other sessions' folders, and a guard lost from it
    # leaves another run's folder writable: refused to every shape, read & write by default too.
    _run(t, "echo ok", _spec(t, RO_DEFAULT))  # a list with this policy's folders in it
    listed = t / "machine/state/labelled"
    before = listed.read_text()
    res, out = _run(t, f'type "{listed}" & type nul > "{listed}" & del /q "{listed}" & ren "{t}\\machine\\state" moved & dir /b "{t}\\machine\\state"'
                       f' & "{PY}" -c "open(r\'{listed}\',\'w\')"', _spec(t, shape))  # fmt: skip
    assert listed.read_text() == before and str(t / "rw") not in (res.get("stdout") or ""), out[-300:]


@rows_only
@pytest.mark.parametrize("damage", ["emptied", "an entry removed", "deleted"])
def test_a_damaged_list_of_labelled_folders_stops_confined_commands(t, shape, damage):
    _run(t, "echo first", _spec(t, shape))
    listed = t / "machine/state/labelled"
    lines = listed.read_text().splitlines()
    {"emptied": lambda: listed.write_text(""), "an entry removed": lambda: listed.write_text("\n".join(lines[:1] + lines[2:]) + "\n"),
     "deleted": listed.unlink}[damage]()  # fmt: skip
    res, out = _run(t, "echo ran> ran.txt", _spec(t, shape))
    assert res.get("return_code") == 126 and "missing or damaged" in out and not (_priv(t) / "ran.txt").exists(), out[-300:]
    res, out = _run(t, "echo ran> ran.txt", _spec(t, RW_DEFAULT))  # the read & write default holds no guard
    assert res["success"], out[-300:]


# --- what the rules leave on the user's folders -------------------------------------------


def _launcher():
    """The launcher module (the host never imports it; the tests read its rule SIDs)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("abstractcore_sandbox_windows_under_test", sb.WINDOWS_LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _acl(path: Path) -> str:
    return subprocess.run(["icacls", str(path)], capture_output=True, text=True, check=True).stdout


def _set_acl(path: Path, sddl: str) -> None:
    """Give ``path`` exactly this access list (``{me}``: this user's SID), whatever its folder hands down."""
    w = _launcher()
    sd = w._VOID()
    convert = w._api(w._adv, "ConvertStringSecurityDescriptorToSecurityDescriptorW", w.W.BOOL, w.W.LPCWSTR, w.W.DWORD, w._P(w._VOID), w._VOID)
    assert convert(sddl.format(me=w.identity_sids()[0]), 1, w.ctypes.byref(sd), None), sddl
    assert w.SetFileSecurityW(str(path), w.DACL_INFORMATION, sd), path


@shapes
def test_entries_planted_in_advance_do_not_pass_for_a_rule(t, shape):
    # A rule's SID is no secret: something that could write a folder's access list before
    # the folder was refused may have left the rule's own entry there, after a grant to
    # everyone. The rule must be in force, not merely present.
    _set_acl(t / "refused", f"D:(A;OICI;FA;;;WD)(D;;FA;;;{_launcher().rule_sid('deny', str(t / 'refused'))})(A;;RCWD;;;OW)")
    res, out = _run(t, f'type "{t}\\refused\\secret.txt" & dir "{t}\\refused"', _spec(t, shape))
    assert not _leaked(res, out)


def test_a_read_only_row_adds_refusals_and_no_grant(t):
    # A grant to anyone but the owner on a private key makes OpenSSH refuse the key.
    _run(t, "echo ok")
    for path in (t / "ro", t / "ro/f.txt", t / "ro/sub/g.txt"):
        ours = [line for line in _acl(path).splitlines() if "S-1-9-" in line]
        assert ours and all("(DENY)" in line for line in ours), _acl(path)


def test_under_a_read_only_default_a_read_only_row_needs_no_entry(t):
    _run(t, "echo ok", _spec(t, RO_DEFAULT))
    for path in (t / "ro", t / "ro/f.txt"):
        assert "S-1-9-" not in _acl(path) and _launcher()._label(str(path)) is None


@shapes
def test_a_private_key_in_a_read_only_row_keeps_an_access_list_openssh_accepts(t, shape):
    # An OpenSSH private key: no inherited entries, and its owner holds "modify" (which does
    # not include "change permissions"). Whatever the rule adds, the only grants on the key
    # are to its owner by name: OpenSSH refuses a key that grants anything to anyone else.
    key = t / "ro/id_key"
    key.write_text(RO_MARK)
    _set_acl(key, "D:P(A;;0x1301bf;;;{me})")
    res, out = _run(t, f'type "{key}" & echo x>> "{key}"', _spec(t, shape))
    assert RO_MARK in out and key.read_text() == RO_MARK
    w = _launcher()
    user, _ = w.identity_sids()
    aces, _control = w._read_dacl(str(key))
    assert {a.sid for a in aces if a.type == w.ALLOW} == {user}, [(a.type, a.sid, hex(a.mask)) for a in aces]
    keygen = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/OpenSSH/ssh-keygen.exe"
    if keygen.exists():
        real = t / "ro/id_real"
        (t / "scratch").mkdir()
        subprocess.run([str(keygen), "-q", "-t", "ed25519", "-N", "", "-f", str(t / "scratch/id_real")], check=True, capture_output=True)
        os.replace(t / "scratch/id_real", real)  # into the read-only row, with the permissions ssh-keygen gave it
        _run(t, "echo again", _spec(t, shape, allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw/x"), "rw"))))
        check = subprocess.run([str(keygen), "-y", "-f", str(real)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr


@shapes
def test_the_owner_keeps_control_of_a_refused_folder(t, shape):
    # A folder its owner manages by being its owner only (no entry of the list gives them
    # "change permissions"): refusing it must not take that away.
    owned = t / "owned/inner"
    owned.mkdir(parents=True)
    (owned / "f.txt").write_text(REFUSED_MARK)
    _set_acl(owned, "D:PAI(A;OICI;0x1301bf;;;BU)")
    res, out = _run(t, f'type "{owned}\\f.txt"', _spec(t, shape, refused=(str(owned),)))
    assert REFUSED_MARK not in out
    again = subprocess.run(["icacls", str(owned), "/grant", "*S-1-5-32-545:(OI)(CI)F"], capture_output=True, text=True)
    assert again.returncode == 0, again.stdout + again.stderr
    assert (owned / "f.txt").read_text() == REFUSED_MARK


def test_a_row_keeps_the_permissions_it_inherited_where_it_came_from(t):
    # A folder moved on the same drive keeps what it inherited in its old place until its
    # permissions are next edited by hand. Making it a row must not be that edit.
    private = t / "private"
    private.mkdir()
    _set_acl(private, "D:PAI(A;OICI;FA;;;{me})")
    (private / "moved/sub").mkdir(parents=True)
    (private / "moved/sub/f.txt").write_text(RO_MARK)
    os.rename(private / "moved", t / "rw/moved")
    before = {p: _acl(p).count("(I)") for p in (t / "rw/moved", t / "rw/moved/sub", t / "rw/moved/sub/f.txt")}
    res, out = _run(t, f'type "{t}\\rw\\moved\\sub\\f.txt" & echo x> "{t}\\rw\\moved\\sub\\f.txt"', _spec(t, allowed=(SandboxRow(str(t / "rw/moved"), "ro"),)))
    assert RO_MARK in out and (t / "rw/moved/sub/f.txt").read_text() == RO_MARK
    for path, inherited in before.items():
        text = _acl(path)
        assert "BUILTIN\\Administrators" not in text and "NT AUTHORITY\\SYSTEM" not in text, text  # nothing of its new place
        assert text.count("(I)") >= inherited


def test_a_labelled_row_carries_the_label_and_its_guard_and_nothing_else(t):
    (t / "rw/sub").mkdir()
    (t / "rw/sub/f.txt").write_text("x")
    _run(t, "echo ok", _spec(t, RO_DEFAULT))
    w = _launcher()
    assert w._label(str(t / "rw")) == (w.CHILDREN, w.WRITER)
    assert w._label(str(t / "rw/sub/f.txt")) == (w.INHERITED, w.WRITER)
    guard = w.rule_sid("guard", str(t / "rw"))
    aces, _ = w._read_dacl(str(t / "rw"))
    assert [(a.type, a.flags & ~w.INHERITED) for a in aces if a.sid == guard] == [(w.DENY, 0), (w.DENY, w.CHILDREN | w.INHERIT_ONLY)]
    assert guard not in _acl(t)  # no entry on the folder that holds it


# --- the launcher ------------------------------------------------------------------------


@shapes
def test_exit_codes_output_and_stdin_pass_through(t, shape):
    box = sb.build_sandbox(_spec(t, shape))
    argv, env = box.wrap([PY, "-c", "import sys;d=sys.stdin.read();sys.stdout.write(d.upper());sys.stderr.write('E'*3);sys.exit(42)"], str(_priv(t)))
    proc = subprocess.run(argv, cwd=_priv(t), env=env, input="abc", capture_output=True, text=True, timeout=60)
    assert (proc.returncode, proc.stdout, proc.stderr) == (42, "ABC", "EEE")
    argv, env = box.wrap([PY, "-c", "import sys;sys.stdout.write('x'*5_000_000)"], str(_priv(t)))
    assert len(subprocess.run(argv, cwd=_priv(t), env=env, capture_output=True, timeout=60).stdout) == 5_000_000
    argv, env = box.wrap(["no-such-program-r12"], str(_priv(t)))
    proc = subprocess.run(argv, cwd=_priv(t), env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 127 and "cannot start the command" in proc.stderr


def test_a_string_command_runs_as_the_unsandboxed_tool_runs_it(t):
    # `Popen(command, shell=True)` semantics: quotes, pipes, && and environment expansion.
    res, out = _run(t, 'echo "a b" & echo c^&d & (echo one& echo two) | %SystemRoot%\\System32\\find.exe /c /v "" & echo %CD%')
    assert '"a b"' in out and "c&d" in out and "2" in out.split() and "session-1" in out


@shapes
def test_killing_the_launcher_ends_the_whole_tree(t, shape):
    box = sb.build_sandbox(_spec(t, shape))
    code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;open(\"grandchild.pid\",\"w\").write(str(__import__(\"os\").getpid()));time.sleep(120)']);time.sleep(120)"
    argv, env = box.wrap([PY, "-c", code], str(_priv(t)))
    proc = subprocess.Popen(argv, cwd=_priv(t), env=env)
    pid_file = _priv(t) / "grandchild.pid"
    deadline = time.time() + 30
    while time.time() < deadline and not (pid_file.exists() and pid_file.read_text().strip()):
        time.sleep(0.1)
    grandchild = int(pid_file.read_text())
    proc.kill()
    proc.wait(timeout=30)
    deadline = time.time() + 15
    listed = True
    while time.time() < deadline and listed:
        listed = str(grandchild) in subprocess.run(["tasklist", "/fi", f"PID eq {grandchild}", "/nh"], capture_output=True, text=True).stdout
        time.sleep(0.2)
    assert not listed, "the command's grandchild outlived its launcher"


def test_timeout_kills_the_command(t):
    started = time.time()
    res, _ = _run(t, f'"{PY}" -c "import time;time.sleep(60)"', timeout=3)
    assert res["success"] is False and time.time() - started < 30


def test_rules_are_written_once_and_a_second_launch_only_reads(t):
    box = sb.build_sandbox(_spec(t))
    argv, env = box.wrap("exit 0", str(_priv(t)))
    subprocess.run(argv, cwd=_priv(t), env=env, check=True, timeout=120)
    before = subprocess.run(["icacls", str(t / "data")], capture_output=True, text=True).stdout
    subprocess.run(argv, cwd=_priv(t), env=env, check=True, timeout=60)
    assert subprocess.run(["icacls", str(t / "data")], capture_output=True, text=True).stdout == before
    assert before.count("S-1-9-") == 2  # the carve rule's two entries, once


@rows_only
def test_rules_labels_and_the_list_are_written_once(t, shape):
    box = sb.build_sandbox(_spec(t, shape))
    argv, env = box.wrap("exit 0", str(_priv(t)))
    subprocess.run(argv, cwd=_priv(t), env=env, check=True, timeout=120)
    w = _launcher()
    watched = [t, t / "data", t / "data/workspaces", _priv(t), t / "rw", t / "parent/child"]

    def state():
        return [(_acl(p), w._label(str(p))) for p in watched] + [(t / "machine/state/labelled").read_text()]

    before = state()
    subprocess.run(argv, cwd=_priv(t), env=env, check=True, timeout=60)
    assert state() == before
    assert sorted(before[-1].splitlines()[1:-1]) == sorted([str(t / "rw"), str(t / "parent/child"), str(_priv(t))])


@shapes
def test_a_refusal_that_cannot_be_written_fails_closed(t, shape):
    # A folder this user can read but whose access list is not theirs to change (as a system
    # folder is): the refusal cannot be written, so the command does not run.
    locked = t / "rw/locked/inner"  # where it is reachable under every shape
    locked.mkdir(parents=True)
    (locked / "f.txt").write_text(REFUSED_MARK)
    subprocess.run(["icacls", str(locked), "/grant", "*S-1-3-4:(OI)(CI)(RX)", "/deny", f"{os.environ['USERNAME']}:(WDAC,WO)"], check=True, capture_output=True)
    res, out = _run(t, f'type "{locked}\\f.txt" & echo ran> ran.txt', _spec(t, shape, refused=(str(locked),)))
    assert res.get("return_code") == 126 and "the command did not run" in out and REFUSED_MARK not in out
    assert not (_priv(t) / "ran.txt").exists()


@rows_only
def test_a_row_that_cannot_be_labelled_fails_closed(t, shape):
    # The user may modify it but not take ownership (no "change owner"): no label can be put
    # on it, so the command does not run.
    row = t / "unlabelable/inner"
    row.mkdir(parents=True)
    _set_acl(row, "D:PAI(A;OICI;0x1701bf;;;{me})")  # modify + change permissions
    res, out = _run(t, "echo ran> ran.txt", _spec(t, shape, allowed=(SandboxRow(str(row), "rw"),)))
    # (under the allow-list it lies in the refused lab with a protected list: refused one step earlier)
    assert res.get("return_code") == 126 and ("cannot label" in out or "permissions of its own" in out) and "the command did not run" in out
    assert not (_priv(t) / "ran.txt").exists()


@shapes
def test_a_missing_refused_path_is_no_rule(t, shape):
    res, out = _run(t, "echo ok", _spec(t, shape, refused=(str(t / "refused"), str(t / "does/not/exist"))))
    assert res["success"] and "ok" in out


@shapes
def test_twenty_launches_at_once_share_the_rules(t, shape):
    box = sb.build_sandbox(_spec(t, shape))
    argv, env = box.wrap(f'type "{t}\\refused\\secret.txt" & echo done', str(_priv(t)))
    procs = [subprocess.Popen(argv, cwd=_priv(t), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(20)]
    outs = [p.communicate(timeout=180)[0] for p in procs]
    assert all("done" in o and REFUSED_MARK not in o for o in outs), outs[0]
    assert subprocess.run(["icacls", str(t / "refused")], capture_output=True, text=True).stdout.count("S-1-9-") == 1
