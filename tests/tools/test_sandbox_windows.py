"""REAL sandbox tests on Windows: commands run through the real tool (``execute_command``)
under a restricted token, with the access-list rules written by the launcher
(``abstractcore/tools/sandbox_windows.py``).

Everything lives under pytest's ``tmp_path``: a scratch "data folder" holding the run's
private workspace (a built-in refusal with its exception, as the gateway lays it out), a
refused folder, a scratch ``.ssh`` holding a MARKER, read-only and read & write rows. A
refusal is proven by the marker NOT appearing in the output; a success would print the
marker and turn the test red without exfiltrating anything real. Rule paths are at least
two folders below ``tmp_path`` (the launcher also writes one entry on a refused row's
parent folder)."""

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


@pytest.fixture(autouse=True)
def _fresh_host():
    sb._reset_host_for_tests()
    yield
    sb._reset_host_for_tests()


@pytest.fixture
def t(tmp_path) -> Path:
    root = Path(os.path.realpath(tmp_path)) / "lab"
    for d in ("data/workspaces/session-1", "data/workspaces/session-2", "data/secrets", "home/.ssh", "home/outside",
              "refused/inner", "ro/sub", "rw", "parent/child/deny"):  # fmt: skip
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


def _priv(t: Path) -> Path:
    return t / "data/workspaces/session-1"


def _spec(t: Path, **kw) -> SandboxSpec:
    base = dict(
        private_workspace=str(_priv(t)),
        posture="any_except_denied",
        default_mode="rw",
        allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw"), "rw"), SandboxRow(str(t / "parent/child"), "rw")),
        refused=(str(t / "refused"), str(t / "parent"), str(t / "parent/child/deny")),
        builtin_refused=(str(t / "data"), str(t / "home/.ssh")),
        builtin_allowed=(str(_priv(t)),),
    )
    base.update(kw)
    return SandboxSpec(**base)


def _run(t: Path, command: str, spec: SandboxSpec = None, cwd=None, timeout: int = 60):
    res = execute_command(command, working_directory=str(cwd or _priv(t)), timeout=timeout, _sandbox=(spec or _spec(t)).to_stamp())
    return res, (res.get("stdout") or "") + (res.get("stderr") or "")


def _grant_self(path: Path, *flags: str) -> None:
    """Give ``path`` its own access-list entry for this user (and, with /inheritance:r, no
    inherited ones): the shape of an OpenSSH private key or a tempfile folder."""
    subprocess.run(["icacls", str(path), *flags, "/grant", f"{os.environ['USERNAME']}:F"], check=True, capture_output=True)


# --- the kind ----------------------------------------------------------------------------


def test_this_host_sandboxes_the_default_posture(t):
    assert sb.host_sandbox_kind("any_except_denied") == sb.KIND_WINDOWS
    res, out = _run(t, "echo ok && cd")
    assert res["success"] and "ok" in out and "session-1" in out
    assert res["sandbox"]["kind"] == "windows-restricted-token"
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


@pytest.mark.parametrize("template", ESCAPES)
def test_refused_folder_is_unreachable(t, template):
    res, out = _run(t, _escape(t, template))
    assert "return_code" in res, f"{template!r} never ran: {res.get('error')}"
    assert not _leaked(res, out), f"ESCAPE: {template!r} reached the refused folder"
    assert REFUSED_MARK not in res["rendered"]


@pytest.mark.parametrize("template", [e for e in ESCAPES if "localhost" not in e])
def test_escape_commands_work_without_the_refusal(t, template):
    # Positive control: the very same command DOES print the marker when the folder is not
    # refused, so the refusal above is the sandbox's doing (not a broken command).
    res, out = _run(t, _escape(t, template), _spec(t, refused=(str(t / "parent"), str(t / "parent/child/deny"))))
    assert _leaked(res, out), f"control failed for {template!r}: {out[-300:]}"


def test_a_file_with_its_own_permissions_in_a_refused_folder_is_closed_too(t):
    # An OpenSSH private key: its access list does not inherit from its folder.
    _grant_self(t / "home/.ssh/id_marker", "/inheritance:r")
    for cmd in (f'type "{t}\\home\\.ssh\\id_marker"', f'dir "{t}\\home\\.ssh"', f'"{PY}" -c "print(open(r\'{t}\\home\\.ssh\\id_marker\').read())"'):
        res, out = _run(t, cmd)
        assert SSH_MARK not in out and "id_marker" not in (res.get("stdout") or ""), cmd
    assert (t / "home/.ssh/id_marker").read_text() == SSH_MARK  # the user's own programs are untouched


def test_builtin_refusal_is_absolute_and_its_exception_works(t):
    # Even when a row "allows" them: the data folder and .ssh stay closed.
    spec = _spec(t, allowed=(SandboxRow(str(t / "home"), "rw"), SandboxRow(str(t / "data/secrets"), "rw")))
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


def test_content_with_its_own_grants_in_a_carved_folder_is_closed(t):
    # Inherited refusals lose to a grant written on the object itself: both shapes below must
    # still be closed (a folder like the ones `tempfile` makes, a file with an explicit grant).
    _grant_self(t / "data/secrets", "/inheritance:r")
    _grant_self(t / "data/gateway.sqlite3")
    for cmd in (f'type "{t}\\data\\secrets\\token"', f'dir "{t}\\data\\secrets"', f'type "{t}\\data\\gateway.sqlite3"'):
        _, out = _run(t, cmd)
        assert DATA_MARK not in out and "token" not in out, cmd


def test_each_session_has_its_own_folder_in_the_same_data_folder(t):
    one, two, three = _priv(t), t / "data/workspaces/session-2", t / "data/workspaces/session-3"
    (one / "mine.txt").write_text("ONE")
    res, out = _run(t, f'type mine.txt && type "{two}\\theirs.txt"')
    assert "ONE" in out and DATA_MARK not in out
    shared = lambda: subprocess.run(["icacls", str(t / "data/workspaces")], capture_output=True, text=True).stdout.count("S-1-9-")  # noqa: E731
    entries = shared()
    # The folder closed to the first session is the second session's own.
    spec2 = _spec(t, private_workspace=str(two), builtin_allowed=(str(two),))
    res, out = _run(t, f'type theirs.txt && echo two> note.txt && type "{one}\\mine.txt"', spec2, cwd=two)
    assert DATA_MARK in out and (two / "note.txt").exists() and "ONE" not in out
    # A session folder and other content made after the rules were written: closed by inheritance,
    # and the new session's own.
    three.mkdir()
    (three / "late.txt").write_text("THREE")
    (t / "data/late").mkdir()
    (t / "data/late/x.txt").write_text(DATA_MARK)
    (t / "data/late.txt").write_text(DATA_MARK)
    res, out = _run(t, f'type "{three}\\late.txt" & type "{t}\\data\\late\\x.txt" & type "{t}\\data\\late.txt" & dir "{t}\\data\\workspaces"')
    assert "THREE" not in out and DATA_MARK not in out and "session-3" not in (res.get("stdout") or "")
    spec3 = _spec(t, private_workspace=str(three), builtin_allowed=(str(three),))
    res, out = _run(t, f'type late.txt && echo three> note.txt && type "{one}\\mine.txt"', spec3, cwd=three)
    assert "THREE" in out and (three / "note.txt").exists() and "ONE" not in out
    assert shared() == entries  # sessions add nothing to the folders they share


def test_content_made_later_with_its_own_permissions_in_a_carved_folder_is_closed(t):
    # After the rules are in place, the gateway makes a secrets folder the way Python makes a
    # private folder (permissions of its own, nothing inherited), a file with a grant of its
    # own, and a plain file. All three must be closed to the next command.
    _run(t, "echo first")
    late = t / "data/late-secrets"
    os.mkdir(late, 0o700)
    (late / "sealing.key").write_text(DATA_MARK)
    _grant_self(late, "/inheritance:r")
    (t / "data/late-granted.txt").write_text(DATA_MARK)
    _grant_self(t / "data/late-granted.txt")
    (t / "data/late-plain.txt").write_text(DATA_MARK)
    for cmd in (f'type "{late}\\sealing.key"', f'dir "{late}"', f'type "{t}\\data\\late-granted.txt"', f'type "{t}\\data\\late-plain.txt"',
                f'echo x> "{late}\\new.txt"', f'echo x> "{t}\\data\\late-granted.txt"'):
        res, out = _run(t, cmd)
        assert DATA_MARK not in out and "sealing.key" not in (res.get("stdout") or ""), cmd
    assert (t / "data/late-granted.txt").read_text() == DATA_MARK and not (late / "new.txt").exists()
    assert (late / "sealing.key").read_text() == DATA_MARK  # the gateway's own access is untouched


def test_a_busy_file_in_the_data_folder_does_not_stop_commands(t):
    # A file another process holds open without sharing it (a database, a log).
    import msvcrt

    k32 = __import__("ctypes").WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = __import__("ctypes").c_void_p
    handle = k32.CreateFileW(str(t / "data/gateway.sqlite3"), 0xC0000000, 0, None, 3, 0x80, None)
    assert handle not in (None, 2**64 - 1)
    fd = msvcrt.open_osfhandle(handle, 0)
    try:
        res, out = _run(t, f'echo ok & type "{t}\\data\\gateway.sqlite3"')
        assert "ok" in out and DATA_MARK not in out
    finally:
        os.close(fd)
    _, out = _run(t, f'type "{t}\\data\\gateway.sqlite3"')
    assert DATA_MARK not in out


# --- nesting, read-only rows, the rest of the world --------------------------------------


def test_nesting_most_specific_row_wins(t):
    res, out = _run(t, f'type "{t}\\parent\\child\\ok.txt" && echo w> "{t}\\parent\\child\\w.txt"')
    assert "CHILD-OK" in out and (t / "parent/child/w.txt").exists()  # an allowed row inside a refused one
    for cmd in (f'type "{t}\\parent\\secret.txt"', f'dir "{t}\\parent"', f'type "{t}\\parent\\child\\deny\\x.txt"'):
        _, out = _run(t, cmd)
        assert REFUSED_MARK not in out and "secret.txt" not in out, cmd


def test_tie_refusal_wins_and_ro_beats_rw(t):
    p = str(t / "rw")
    (t / "rw/f.txt").write_text(REFUSED_MARK)
    _, out = _run(t, f'type "{p}\\f.txt"', _spec(t, allowed=(SandboxRow(p, "rw"),), refused=(p,)))
    assert REFUSED_MARK not in out
    (t / "ro/t.txt").write_text("T")
    q = str(t / "ro")
    _run(t, f'echo x> "{q}\\t.txt"', _spec(t, allowed=(SandboxRow(q, "rw"), SandboxRow(q, "ro"))))
    assert (t / "ro/t.txt").read_text() == "T"


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
    '"{py}" -c "open(r\'{ro}\\sub\\g.txt\',\'w\').write(\'x\')"',
    'powershell -NoProfile -Command "Set-Content \'{ro}\\f.txt\' x; Remove-Item -Recurse -Force \'{ro}\\sub\'"',
]


@pytest.mark.parametrize("template", WRITES)
def test_read_only_row_reads_and_never_writes(t, template):
    res, out = _run(t, template.format(ro=t / "ro", rw=t / "rw", py=PY) + f' & type "{t}\\ro\\f.txt"')
    assert RO_MARK in out, f"{template!r}: the read-only row is no longer readable: {out[-300:]}"
    assert (t / "ro/f.txt").read_text() == RO_MARK and (t / "ro/sub/g.txt").read_text() == RO_MARK
    assert sorted(p.name for p in (t / "ro").iterdir()) == ["f.txt", "sub"] and not (t / "rw/f.txt").exists()
    assert (os.stat(t / "ro/f.txt").st_file_attributes & 0x2) == 0  # not hidden


def test_refusals_cannot_be_lifted_from_inside(t):
    for cmd in (
        f'icacls "{t}\\refused" /reset /t & type "{t}\\refused\\secret.txt"',
        f'icacls "{t}\\refused" /grant Everyone:F /t & type "{t}\\refused\\secret.txt"',
        f'takeown /f "{t}\\refused" /r /d y & icacls "{t}\\refused" /reset /t & type "{t}\\refused\\secret.txt"',
        f'ren "{t}\\refused" gone & type "{t}\\gone\\secret.txt"',
        f'rmdir /s /q "{t}\\refused"',
        f'icacls "{t}" /reset /t & type "{t}\\refused\\secret.txt"',
    ):
        _, out = _run(t, cmd)
        assert REFUSED_MARK not in out, cmd
    assert (t / "refused/secret.txt").read_text() == REFUSED_MARK


def test_the_rest_of_the_world_stays_read_and_write(t):
    res, out = _run(t, f'type "{t}\\home\\outside\\f.txt" && echo w> "{t}\\home\\outside\\w.txt" && echo w> "{t}\\rw\\w.txt" && type "{t}\\ro\\f.txt"')
    assert res["success"] and OUTSIDE_MARK in out and RO_MARK in out
    assert (t / "home/outside/w.txt").exists() and (t / "rw/w.txt").exists()
    assert (t / "refused/secret.txt").read_text() == REFUSED_MARK  # and the user's own programs still read everything


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


def test_entries_planted_in_advance_do_not_pass_for_a_rule(t):
    # A rule's SID is no secret: something that could write a folder's access list before
    # the folder was refused may have left the rule's own entry there, after a grant to
    # everyone. The rule must be in force, not merely present.
    w = _launcher()
    planted = f"D:(A;OICI;FA;;;WD)(D;;FA;;;{w.rule_sid('deny', str(t / 'refused'))})(A;;RCWD;;;OW)"
    sd = w._VOID()
    convert = w._api(w._adv, "ConvertStringSecurityDescriptorToSecurityDescriptorW", w.W.BOOL, w.W.LPCWSTR, w.W.DWORD, w._P(w._VOID), w._VOID)
    assert convert(planted, 1, w.ctypes.byref(sd), None) and w.SetFileSecurityW(str(t / "refused"), w.DACL_INFORMATION, sd)
    res, out = _run(t, f'type "{t}\\refused\\secret.txt" & dir "{t}\\refused"')
    assert not _leaked(res, out)


def test_a_read_only_row_adds_refusals_and_no_grant(t):
    # A grant to anyone but the owner on a private key makes OpenSSH refuse the key.
    _run(t, "echo ok")
    for path in (t / "ro", t / "ro/f.txt", t / "ro/sub/g.txt"):
        ours = [line for line in _acl(path).splitlines() if "S-1-9-" in line]
        assert ours and all("(DENY)" in line for line in ours), _acl(path)


def test_a_private_key_in_a_read_only_row_keeps_an_access_list_openssh_accepts(t):
    # An OpenSSH private key: no inherited entries, and its owner holds "modify" (which does
    # not include "change permissions"). Whatever the rule adds, the only grants on the key
    # are to its owner by name: OpenSSH refuses a key that grants anything to anyone else.
    key = t / "ro/id_key"
    key.write_text(RO_MARK)
    subprocess.run(["icacls", str(key), "/inheritance:r", "/grant:r", f"{os.environ['USERNAME']}:M"], check=True, capture_output=True)
    res, out = _run(t, f'type "{key}" & echo x>> "{key}"')
    assert RO_MARK in out and key.read_text() == RO_MARK
    w = _launcher()
    user, _ = w.identity_sids()
    aces, _control = w._read_dacl(str(key))
    assert any(a.type == w.DENY and a.sid.startswith("S-1-9-") for a in aces)
    assert {a.sid for a in aces if a.type == w.ALLOW} == {user}, [(a.type, a.sid, hex(a.mask)) for a in aces]
    keygen = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/OpenSSH/ssh-keygen.exe"
    if keygen.exists():
        real = t / "ro/id_real"
        (t / "scratch").mkdir()
        subprocess.run([str(keygen), "-q", "-t", "ed25519", "-N", "", "-f", str(t / "scratch/id_real")], check=True, capture_output=True)
        os.replace(t / "scratch/id_real", real)  # into the read-only row, with the permissions ssh-keygen gave it
        _run(t, "echo again", _spec(t, allowed=(SandboxRow(str(t / "ro"), "ro"), SandboxRow(str(t / "rw/x"), "rw"))))
        check = subprocess.run([str(keygen), "-y", "-f", str(real)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr


def test_the_owner_keeps_control_of_a_refused_folder(t):
    # A folder its owner manages by being its owner only (no entry of the list gives them
    # "change permissions"): refusing it must not take that away.
    owned = t / "owned/inner"
    owned.mkdir(parents=True)
    (owned / "f.txt").write_text(REFUSED_MARK)
    subprocess.run(["icacls", str(owned), "/inheritance:r", "/grant", "*S-1-5-32-545:(OI)(CI)M"], check=True, capture_output=True)
    res, out = _run(t, f'type "{owned}\\f.txt"', _spec(t, refused=(str(owned),)))
    assert REFUSED_MARK not in out
    again = subprocess.run(["icacls", str(owned), "/grant", "*S-1-5-32-545:(OI)(CI)F"], capture_output=True, text=True)
    assert again.returncode == 0, again.stdout + again.stderr
    assert (owned / "f.txt").read_text() == REFUSED_MARK


def test_a_row_keeps_the_permissions_it_inherited_where_it_came_from(t):
    # A folder moved on the same drive keeps what it inherited in its old place until its
    # permissions are next edited by hand. Making it a row must not be that edit.
    private = t / "private"
    private.mkdir()
    subprocess.run(["icacls", str(private), "/inheritance:r", "/grant:r", f"{os.environ['USERNAME']}:(OI)(CI)F"], check=True, capture_output=True)
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


# --- what Windows cannot express ---------------------------------------------------------


@pytest.mark.parametrize("posture,default_mode,reason", [
    ("allowed_only", "rw", sb.WINDOWS_ALLOW_LIST_UNEXPRESSIBLE),
    ("any_except_denied", "ro", sb.WINDOWS_READ_ONLY_UNEXPRESSIBLE),
])  # fmt: skip
def test_unexpressible_shapes_are_refused_before_anything_runs(t, posture, default_mode, reason):
    res, _ = _run(t, "echo ran> ran.txt", _spec(t, posture=posture, default_mode=default_mode))
    assert res["success"] is False and reason in res["error"] and res["sandbox"]["kind"] == "none"
    assert not (_priv(t) / "ran.txt").exists()


def test_host_flag_runs_the_unexpressible_shapes_unsandboxed(t):
    sb.configure_host(env=dict(os.environ), unsandboxed_commands_allowed=True)
    res, out = _run(t, "echo ran", _spec(t, posture="allowed_only"))
    assert res["success"] and "ran" in out and res["sandbox"]["kind"] == "unsandboxed"


# --- the launcher ------------------------------------------------------------------------


def test_exit_codes_output_and_stdin_pass_through(t):
    box = sb.build_sandbox(_spec(t))
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


def test_killing_the_launcher_ends_the_whole_tree(t):
    box = sb.build_sandbox(_spec(t))
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


def test_a_refusal_that_cannot_be_written_fails_closed(t):
    # A folder this user can read but whose access list is not theirs to change (as a system
    # folder is): the refusal cannot be written, so the command does not run.
    locked = t / "locked/inner"
    locked.mkdir(parents=True)
    (locked / "f.txt").write_text(REFUSED_MARK)
    subprocess.run(["icacls", str(locked), "/grant", "*S-1-3-4:(OI)(CI)(RX)", "/deny", f"{os.environ['USERNAME']}:(WDAC,WO)"], check=True, capture_output=True)
    res, out = _run(t, f'type "{locked}\\f.txt" & echo ran> ran.txt', _spec(t, refused=(str(locked),)))
    assert res.get("return_code") == 126 and "the command did not run" in out and REFUSED_MARK not in out
    assert not (_priv(t) / "ran.txt").exists()


def test_a_missing_refused_path_is_no_rule(t):
    res, out = _run(t, "echo ok", _spec(t, refused=(str(t / "refused"), str(t / "does/not/exist"))))
    assert res["success"] and "ok" in out


def test_twenty_launches_at_once_share_the_rules(t):
    box = sb.build_sandbox(_spec(t))
    argv, env = box.wrap(f'type "{t}\\refused\\secret.txt" & echo done', str(_priv(t)))
    procs = [subprocess.Popen(argv, cwd=_priv(t), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(20)]
    outs = [p.communicate(timeout=180)[0] for p in procs]
    assert all("done" in o and REFUSED_MARK not in o for o in outs)
    assert subprocess.run(["icacls", str(t / "refused")], capture_output=True, text=True).stdout.count("S-1-9-") == 1
