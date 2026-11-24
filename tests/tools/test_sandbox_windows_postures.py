"""The command sandbox's CONTRACT on Windows, for the three workspace shapes.

Real commands run through the real tool (``execute_command``, and the launcher through
``Sandbox.wrap`` where a test needs the process itself), one test per contract clause,
parametrised over:

- ``any-rw``      "Allow everything, refuse listed workspaces", read & write by default;
- ``any-ro``      the same posture, read-only by default;
- ``allow-list``  "Deny everything, allow listed workspaces".

The contract is the one macOS (``macos_profile``) and Linux (``bwrap_argv``) enforce:

- ``allow-list``: user data is unreadable except the allowed rows and the run's private
  workspace; nothing is writable except the read & write rows, the private workspace and its
  ``.tmp``; system roots and the host's Python stay readable.
- ``any-ro``: everything the user can read stays readable except refused rows and built-in
  refusals; only the read & write rows, the private workspace and its ``.tmp`` are writable.
- ``any-rw``: everything readable and writable except refused rows and built-in refusals
  (unreadable, unwritable) and read-only rows (unwritable).
- Always: refused rows and built-in refusals are unreadable and unwritable wherever they
  sit; the most specific row wins, a refusal wins a tie, read-only beats read & write on a
  tie; a built-in refusal is absolute except under ``builtin_allowed``; the command's
  environment is the host's with ``TEMP``/``TMP`` = ``<private workspace>\\.tmp``; a refusal
  that cannot be enforced means the command does not run.

Every negative check has a positive control: the same command, where the contract allows
it, does what the negative check says it must not, so a refusal is the sandbox's doing and
not a broken command. A command that never ran (the tool refused the shape) FAILS the test
with the refusal's sentence: a refusal never passes for enforcement.

Everything lives under pytest's ``tmp_path``, in ``lab`` (rule paths are at least two
folders below ``tmp_path``), whose permissions are set by the test (a temp folder's own
permissions differ between machines and are never assumed). Nothing here writes on a real
folder of the machine: the only real places a command is pointed at are the host's
``%TEMP%`` and ``HKCU\\Software``, as negative checks under the two write-confined shapes;
anything such a check manages to create is removed at once and fails the test. A guard
checks at the end of the module that the access lists of the user's profile and of the
fixed drives' top-level folders are what they were.

"User data" under the allow-list posture is the backend's design: the tests that need it
declare scratch folders as the user data roots through ONE adapter (below)."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import string
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pytest

from abstractcore.tools import sandbox as sb
from abstractcore.tools.common_tools import execute_command
from abstractcore.tools.sandbox import SandboxRow, SandboxSpec

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows sandbox contract tests (macOS: test_sandbox_macos.py, Linux: test_sandbox_linux.py)")

REFUSED_MARK = "R12-REFUSED-MARKER-7f3a"
SSH_MARK = "R12-SSH-MARKER-91c2"
RO_MARK = "R12-RO-MARKER-44d1"
DATA_MARK = "R12-DATA-MARKER-5e0c"
USER_MARK = "R12-USERDATA-MARKER-c3a9"
SHOWN_MARK = "R12-SHOWN-MARKER-2d71"
PY = sys.executable
REG_PROBE = r"Software\AbstractCoreSandboxContractProbe"


# =====================================================================================
# THE ADAPTER: how a test declares a fake "user data root"
# =====================================================================================
#
# Under "Deny everything, allow listed workspaces" the sandbox hides "user data". Which
# folders that is on Windows is the backend's design (most likely: the user's profile and
# the user-owned folders at the top of each fixed drive). The tests must never depend on,
# or cause changes to, the real ones, so they replace them with scratch folders
# (``lab\home``, shaped like a profile, and ``lab\drive``, shaped like a folder at the top
# of a drive, readable by BUILTIN\Users) and only then check what is hidden.
#
# What the backend must provide: ONE module-level function of ``abstractcore.tools.sandbox``
# that returns the user data roots (absolute paths), called in the HOST process each time a
# sandbox is built (the launcher runs in another process, where a test cannot replace
# anything). The adapter replaces it with ``lambda *args, **kwargs: (home, drive)``, so any
# signature works; if it must return something other than a sequence of path strings,
# change ``_fake_roots_value`` only.
#
# LEAD: fill in the function's name (or set the environment variable
# ABSTRACTCORE_SANDBOX_TEST_USER_DATA_FUNCTION to it). While it is empty, the tests that need
# fake user data roots SKIP, saying so; every other test runs without it. Once it is set,
# the fake roots are in force for EVERY test of this module (so no allow-list run here ever
# uses the real profile or drive folders), and a name that does not exist FAILS.
USER_DATA_ROOTS_FUNCTION = "_windows_user_data_roots"

# The backend's other two views of the real machine (``abstractcore.tools.sandbox``): the
# folders Windows already labels Low (``LocalLow``, ``Favorites``), which the read-only default
# guards, and the launcher's state folder (``%LOCALAPPDATA%\AbstractCore\command-sandbox``,
# where it lists the folders it labelled). Both are replaced by lab folders for EVERY test,
# whatever the adapter says, and a test fails before anything is launched if a rule, a folder
# to label or the state folder lies outside the lab (``_machine_is_the_lab``).
MACHINE_FUNCTIONS = ("_windows_low_places", "_windows_state_folder")
# The drives the sandbox cannot protect (``windows_unprotected_volumes``: no access lists, or a
# network share) are a view of the real machine too: none here, unless a test injects a lab
# folder as such a "drive" (``_fake_unprotected_drive``).
FAKE_LOCAL_LOW = "home/AppData/LocalLow"
# Every function of abstractcore.tools.sandbox whose name says Windows, by what it is. A new
# one fails test_every_windows_view_of_the_machine_is_known until it is classified here (and,
# if it names a real place, faked in _machine_is_the_lab).
KNOWN_WINDOWS_FUNCTIONS = {
    "_windows_user_data_roots": "faked (adapter)", "_windows_low_places": "faked", "_windows_state_folder": "faked",
    "windows_unprotected_volumes": "faked", "_windows_writes_rows": "pure", "windows_rules": "checked",
    "windows_policy": "checked", "_windows_python": "the host's interpreter (only run, never written)",
    "_may_change_permissions": "read-only probe, called by _windows_user_data_roots only",
}  # fmt: skip
# The real views, read before any test replaces them (only ever called, never launched with).
REAL_USER_DATA_ROOTS = getattr(sb, "_windows_user_data_roots", None)

ADAPTER_UNRESOLVED = (
    "the fake user data roots adapter is unresolved: set USER_DATA_ROOTS_FUNCTION at the top of "
    "test_sandbox_windows_postures.py to the abstractcore.tools.sandbox function that returns the "
    "Windows user data roots of the allow-list posture"
)


def _user_data_roots_function() -> str:
    return (os.environ.get("ABSTRACTCORE_SANDBOX_TEST_USER_DATA_FUNCTION") or USER_DATA_ROOTS_FUNCTION).strip()


def _fake_roots_value(roots: Sequence[Path]):
    return tuple(str(r) for r in roots)


def _install_fake_user_data_roots(monkeypatch, roots: Sequence[Path]) -> bool:
    """Replace the backend's user data roots by ``roots``; False while the adapter is unresolved."""
    name = _user_data_roots_function()
    if not name:
        return False
    if not callable(getattr(sb, name, None)):
        pytest.fail(f"the adapter names abstractcore.tools.sandbox.{name}, which is not a function of this abstractcore")
    value = _fake_roots_value(roots)
    monkeypatch.setattr(sb, name, lambda *args, **kwargs: value)
    return True


@pytest.fixture
def fake_user_data(t, monkeypatch) -> bool:
    """In force for every test (autouse below); the tests that NEED it call ``_need_fake_user_data``."""
    return _install_fake_user_data_roots(monkeypatch, (t / "home", t / "drive"))


def _need_fake_user_data(installed: bool) -> None:
    if not installed:
        pytest.skip(ADAPTER_UNRESOLVED)


# =====================================================================================
# Shapes, folders, helpers
# =====================================================================================


@dataclass(frozen=True)
class Shape:
    posture: str
    default_mode: str

    @property
    def id(self) -> str:
        return "allow-list" if self.posture == "allowed_only" else f"any-{self.default_mode}"

    @property
    def writes_confined(self) -> bool:
        """Only the read & write rows, the private workspace and its .tmp are writable."""
        return not (self.posture == "any_except_denied" and self.default_mode == "rw")

    @property
    def allow_list(self) -> bool:
        return self.posture == "allowed_only"


ANY_RW = Shape("any_except_denied", "rw")
ANY_RO = Shape("any_except_denied", "ro")
ALLOW_LIST = Shape("allowed_only", "rw")
ALL_SHAPES = [pytest.param(s, id=s.id) for s in (ANY_RW, ANY_RO, ALLOW_LIST)]
CONFINED_SHAPES = [pytest.param(s, id=s.id) for s in (ANY_RO, ALLOW_LIST)]
ALLOW_EVERYTHING_SHAPES = [pytest.param(s, id=s.id) for s in (ANY_RW, ANY_RO)]


@pytest.fixture(autouse=True)
def _fresh_host():
    sb._reset_host_for_tests()
    yield
    sb._reset_host_for_tests()


def _launcher():
    """The launcher module (the host never imports it; the tests use its access-list helpers)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("abstractcore_sandbox_windows_under_test", sb.WINDOWS_LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_W = None


def _w():
    global _W
    if _W is None:
        _W = _launcher()
    return _W


def _set_acl(path: Path, sddl: str) -> None:
    """Give ``path`` exactly this access list (``{me}``: this user's SID), whatever its folder hands down."""
    w = _w()
    sd = w._VOID()
    convert = w._api(w._adv, "ConvertStringSecurityDescriptorToSecurityDescriptorW", w.W.BOOL, w.W.LPCWSTR, w.W.DWORD, w._P(w._VOID), w._VOID)
    assert convert(sddl.format(me=w.identity_sids()[0]), 1, w.ctypes.byref(sd), None), sddl
    assert w.SetFileSecurityW(str(path), w.DACL_INFORMATION, sd), path


# A profile folder: the user, SYSTEM and Administrators only.
PROFILE_LIKE = "D:PAI(A;OICI;FA;;;{me})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
# A folder a user made at the top of C:\ (what C:\ hands down): Users may read it, every
# signed-in user may modify it, its creator owns it.
DRIVE_FOLDER_LIKE = "D:PAI(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;0x1200a9;;;BU)(A;OICI;0x1301bf;;;AU)(A;OICI;FA;;;{me})"


@pytest.fixture
def t(tmp_path) -> Path:
    lab = Path(os.path.realpath(tmp_path)) / "lab"
    lab.mkdir()
    _set_acl(lab, PROFILE_LIKE)
    (lab / "drive").mkdir()
    _set_acl(lab / "drive", DRIVE_FOLDER_LIKE)
    for d in ("data/workspaces/session-1", "data/workspaces/session-2", "data/secrets", "home/.ssh",
              "home/docs/inner", "drive/docs/inner", "home/refused/inner", "home/ro/sub", "home/rw",
              "home/shown/inner", "home/unlisted", "home/shared"):  # fmt: skip
        (lab / d).mkdir(parents=True, exist_ok=True)
    for rel, text in (
        ("data/gateway.sqlite3", DATA_MARK), ("data/secrets/token", DATA_MARK), ("data/workspaces/session-2/theirs.txt", DATA_MARK),
        ("home/.ssh/id_marker", SSH_MARK),
        ("home/docs/secret.txt", USER_MARK), ("home/docs/inner/deep.txt", USER_MARK),
        ("drive/docs/secret.txt", USER_MARK), ("drive/docs/inner/deep.txt", USER_MARK),
        ("home/refused/secret.txt", REFUSED_MARK), ("home/refused/inner/deep.txt", REFUSED_MARK),
        ("home/shown/secret.txt", SHOWN_MARK), ("home/shown/inner/deep.txt", SHOWN_MARK),
        ("home/ro/f.txt", RO_MARK), ("home/ro/sub/g.txt", RO_MARK),
        ("home/unlisted/u.txt", "UNLISTED-OK"),
    ):  # fmt: skip
        (lab / rel).write_text(text)
    return lab


@pytest.fixture(autouse=True)
def _fake_user_data_everywhere(fake_user_data):
    """Once the adapter is resolved, no test of this module runs with the real user data roots."""
    yield


def _inside_lab(path: str, lab: Path) -> bool:
    return sb._under(os.path.realpath(path), str(lab)) and os.path.normcase(os.path.realpath(path)) != os.path.normcase(str(lab))


@pytest.fixture(autouse=True)
def _machine_is_the_lab(t, monkeypatch, _fake_user_data_everywhere):
    """No rule, label or state file of these tests may land outside the lab. The backend's
    views of the real machine are replaced (the user data roots by the adapter, the Low places
    and the state folder here), and every rule set the sandbox builds is checked BEFORE the
    launcher runs: each rule path (and the parent a refusing rule writes on) lies in the lab,
    the folders to label are rules, and the state folder and the Low places are lab folders."""
    for name in MACHINE_FUNCTIONS:
        if hasattr(sb, name) and not callable(getattr(sb, name)):
            pytest.fail(f"abstractcore.tools.sandbox.{name} is not a function")
    if hasattr(sb, "_windows_low_places"):
        (t / FAKE_LOCAL_LOW).mkdir(parents=True, exist_ok=True)  # where it is: in the (fake) profile
        monkeypatch.setattr(sb, "_windows_low_places", lambda *a, **k: (str(t / FAKE_LOCAL_LOW),))
    if hasattr(sb, "_windows_state_folder"):
        monkeypatch.setattr(sb, "_windows_state_folder", lambda *a, **k: str(t / "machine/state"))
    if hasattr(sb, "windows_unprotected_volumes"):
        monkeypatch.setattr(sb, "windows_unprotected_volumes", lambda *a, **k: {})
    if hasattr(sb, "_windows_user_data_roots") and not _user_data_roots_function():
        pytest.fail("this abstractcore computes Windows user data roots: resolve the adapter before running these tests")
    real_rules = sb.windows_rules

    def confined(spec):
        rules = real_rules(spec)
        outside = [r for r in rules for p in r[1:] if not _inside_lab(p, t)]
        assert not outside, f"SAFETY: a sandbox rule outside the test tree, nothing was launched: {outside}"
        for name in MACHINE_FUNCTIONS + (_user_data_roots_function(), "windows_unprotected_volumes"):
            fn = getattr(sb, name, None)
            if callable(fn):
                value = fn()
                for p in [value] if isinstance(value, str) else list(value):
                    assert _inside_lab(p, t), f"SAFETY: {name}() names {p}, outside the test tree"
        return rules

    monkeypatch.setattr(sb, "windows_rules", confined)
    real_policy = sb.windows_policy

    def checked_policy(rules, *args, **kwargs):
        # What the launcher will be handed: every path in it (rules, folders to label, the
        # state folder that holds the lock, unprotected drives) lies in the lab.
        text = real_policy(rules, *args, **kwargs)
        import base64
        import json
        import zlib

        policy = json.loads(zlib.decompress(base64.urlsafe_b64decode(text)))
        paths = [p for r in policy.get("rules", []) for p in r[1:]] + list(policy.get("label") or [])
        paths += [policy["state"]] if policy.get("state") else []
        paths += list((policy.get("unprotected") or {}).keys())
        outside = [p for p in paths if not _inside_lab(p, t)]
        assert not outside, f"SAFETY: the launcher's policy names places outside the test tree, nothing was launched: {outside}"
        return text

    monkeypatch.setattr(sb, "windows_policy", checked_policy)
    yield


def _fake_unprotected_drive(t: Path, monkeypatch) -> tuple:
    """A lab folder that the sandbox takes for a drive without access lists (exFAT, a share):
    the injectable probe only, never a real drive."""
    drive = t / "fatdrive"
    (drive / "docs").mkdir(parents=True, exist_ok=True)
    (drive / "docs/f.txt").write_text("FAT-OK")
    why = "drive F: has no permissions (exFAT): commands can read and change it"
    monkeypatch.setattr(sb, "windows_unprotected_volumes", lambda *a, **k: {str(drive) + "\\": why})
    return drive, why


def _priv(t: Path) -> Path:
    return t / "data/workspaces/session-1"


def _priv2(t: Path) -> Path:
    return t / "data/workspaces/session-2"


def _spec(t: Path, shape: Shape, **kw) -> SandboxSpec:
    """The gateway's layout: the run's private workspace inside the data folder (a built-in
    refusal with that workspace as its exception), a scratch .ssh, rows in the user data."""
    base = dict(
        private_workspace=str(_priv(t)),
        posture=shape.posture,
        default_mode=shape.default_mode,
        allowed=(SandboxRow(str(t / "home/ro"), "ro"), SandboxRow(str(t / "home/rw"), "rw")),
        refused=(str(t / "home/refused"),),
        builtin_refused=(str(t / "data"), str(t / "home/.ssh")),
        builtin_allowed=(str(_priv(t)),),
    )
    base.update(kw)
    return SandboxSpec(**base)


def _spec2(t: Path, shape: Shape, **kw) -> SandboxSpec:
    """A second run of the same gateway: same data folder, its own private workspace."""
    return _spec(t, shape, private_workspace=str(_priv2(t)), builtin_allowed=(str(_priv2(t)),), **kw)


def _run(t: Path, command: str, spec: SandboxSpec, cwd=None, timeout: int = 60):
    res = execute_command(command, working_directory=str(cwd or spec.private_workspace), timeout=timeout, _sandbox=spec.to_stamp())
    return res, (res.get("stdout") or "") + (res.get("stderr") or "")


def _ran(res, what: str = "the command") -> None:
    """A refused shape is not enforcement: a command that never ran fails the test."""
    assert "return_code" in res, f"{what} never ran: {res.get('error')}"


def _box(spec: SandboxSpec) -> sb.Sandbox:
    box = sb.build_sandbox(spec)
    assert not box.refuses, f"the sandbox refuses this shape: {box.reason}"
    return box


def _same(a, b) -> bool:
    return os.path.normcase(os.path.realpath(str(a))).rstrip("\\") == os.path.normcase(os.path.realpath(str(b))).rstrip("\\")


def _listing(res) -> list:
    return [line.strip() for line in (res.get("stdout") or "").splitlines()]


# --- the guard: real folders keep their access lists -------------------------------------


def _fixed_drive_roots() -> list:
    roots = []
    get_type = ctypes.windll.kernel32.GetDriveTypeW
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if get_type(root) == 3 and os.path.isdir(root):  # DRIVE_FIXED
            roots.append(root)
    return roots


def _real_folders() -> list:
    home = os.path.expanduser("~")
    out = [home] + [os.path.join(home, *p) for p in (("AppData",), ("AppData", "Local"), ("AppData", "LocalLow"), ("AppData", "Roaming"),
                                                    ("AppData", "Local", "Temp"), ("Desktop",), ("Documents",), (".ssh",))]  # fmt: skip
    for root in _fixed_drive_roots():
        out.append(root)
        try:
            out += sorted(e.path for e in os.scandir(root) if e.is_dir(follow_symlinks=False))[:60]
        except OSError:
            pass
    return [p for p in dict.fromkeys(out) if os.path.isdir(p)]


def _security(path: str) -> str:
    """Owner, access list and integrity label of ``path``, as SDDL (read only)."""
    from ctypes import wintypes as W

    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    get = adv.GetFileSecurityW
    get.argtypes = [W.LPCWSTR, W.DWORD, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD)]
    get.restype = W.BOOL
    to_text = adv.ConvertSecurityDescriptorToStringSecurityDescriptorW
    to_text.argtypes = [ctypes.c_void_p, W.DWORD, W.DWORD, ctypes.POINTER(W.LPWSTR), ctypes.POINTER(W.ULONG)]
    to_text.restype = W.BOOL
    info = 0x1 | 0x4 | 0x10  # owner, DACL, label
    need = W.DWORD()
    get(path, info, None, 0, ctypes.byref(need))
    if not need.value:
        return f"unreadable ({ctypes.get_last_error()})"
    buf = ctypes.create_string_buffer(need.value)
    if not get(path, info, buf, need, ctypes.byref(need)):
        return f"unreadable ({ctypes.get_last_error()})"
    text = W.LPWSTR()
    if not to_text(buf, 1, info, ctypes.byref(text), None):
        return f"unconvertible ({ctypes.get_last_error()})"
    try:
        return text.value
    finally:
        ctypes.windll.kernel32.LocalFree(text)


def _acls(paths: Iterable[str]) -> Dict[str, str]:
    return {p: _security(p) for p in paths}


@pytest.fixture(scope="module", autouse=True)
def _real_folders_keep_their_access_lists():
    if sys.platform != "win32":
        yield
        return
    folders = _real_folders()
    state = sb._windows_state_folder() if hasattr(sb, "_windows_state_folder") else None
    low = list(sb._windows_low_places()) if hasattr(sb, "_windows_low_places") else []
    folders += [p for p in low if p not in folders]
    had_state = bool(state and os.path.lexists(state))
    before = _acls(folders)
    yield
    after = _acls(folders)
    changed = [p for p in folders if before[p] != after[p]]
    assert not changed, f"the sandbox changed the access list or label of real folders: {changed}"
    assert not state or os.path.lexists(state) == had_state, f"the real state folder {state} appeared or went away"


# =====================================================================================
# The kind
# =====================================================================================


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_this_host_sandboxes_the_shape(t, shape):
    assert sb.host_sandbox_kind(shape.posture) == sb.KIND_WINDOWS, f"no Windows sandbox for {shape.posture}"
    res, out = _run(t, "echo ok && cd", _spec(t, shape))
    _ran(res)
    assert res["success"] and "ok" in out and "session-1" in out
    assert res["sandbox"]["kind"] == "windows-restricted-token"
    assert res["sandbox"]["posture"] == shape.posture and res["sandbox"]["default_mode"] == shape.default_mode
    assert "Sandbox: Windows restricted token" in res["rendered"]


# =====================================================================================
# Reading: every route, every shape
# =====================================================================================

# {r}: the folder probed (holds secret.txt and inner\deep.txt); {rp}, {rn}: its parent and name.
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
    'cd /d "{rp}" && type {rn}\\secret.txt',
    'cd /d "{r}\\.." && type {rn}\\secret.txt',
]
# The administrative share needs an administrator: no control can show it reading.
CONTROLLABLE = [e for e in ESCAPES if "localhost" not in e]


def _escape(template: str, r: Path) -> str:
    share = str(r).replace(":", "$", 1)  # C:\x -> C$\x
    return template.format(r=r, rp=r.parent, rn=r.name, py=PY, share=share)


def _leaked(res, out, mark: str) -> bool:
    """The marker's content, or the folder's listing (``dir`` ends a line with the name)."""
    return mark in out or any(line.rstrip().endswith("secret.txt") for line in (res.get("stdout") or "").splitlines())


@pytest.mark.parametrize("template", CONTROLLABLE)
def test_every_route_reads_a_listed_row_under_the_allow_list(t, template):
    # Positive control of the allow-list read tests below: every route reads a listed read &
    # write row in user data. (Under "Allow everything" the controls are the routes reading
    # unlisted user data: test_user_data_stays_readable_when_everything_is_allowed.)
    spec = _spec(t, ALLOW_LIST, allowed=(SandboxRow(str(t / "home/ro"), "ro"), SandboxRow(str(t / "home/rw"), "rw"), SandboxRow(str(t / "home/shown"), "rw")))
    res, out = _run(t, _escape(template, t / "home/shown"), spec)
    _ran(res, template)
    assert _leaked(res, out, SHOWN_MARK), f"control failed under allow-list for {template!r}: {out[-300:]}"


@pytest.mark.parametrize("template", ESCAPES)
@pytest.mark.parametrize("shape", CONFINED_SHAPES)  # any-rw: test_sandbox_windows.py::test_refused_folder_is_unreachable
def test_a_refused_row_is_unreachable_by_every_route(t, shape, template):
    # Controls: the same routes read unlisted user data (any-rw, any-ro) and a listed row
    # (allow-list) in the tests around this one.
    res, out = _run(t, _escape(template, t / "home/refused"), _spec(t, shape))
    _ran(res, template)
    assert not _leaked(res, out, REFUSED_MARK), f"ESCAPE under {shape.id}: {template!r} reached the refused row"
    assert REFUSED_MARK not in res["rendered"]


@pytest.mark.parametrize("root", ["home", "drive"])
@pytest.mark.parametrize("template", ESCAPES)
def test_allow_list_hides_user_data_outside_the_rows(t, fake_user_data, template, root):
    # User data outside every row: a profile-shaped folder, and a folder at the top of a
    # drive that BUILTIN\Users may read. Control: test_user_data_stays_readable_when_everything_is_allowed.
    _need_fake_user_data(fake_user_data)
    res, out = _run(t, _escape(template, t / root / "docs"), _spec(t, ALLOW_LIST))
    _ran(res, template)
    assert not _leaked(res, out, USER_MARK), f"ESCAPE under allow-list: {template!r} read unlisted user data in {root}"


# A hard link needs write access to its target, which the read-only default refuses; and
# Windows refuses a process below Medium integrity (the write-confined shapes) a junction to a
# folder it cannot write (test_a_junction_needs_a_target_the_command_may_write). Those two
# routes read unlisted user data under any-rw only.
# (Every other route reads it under any-ro here; the implementer's module covers any-rw.)
_WRITE_ROUTES = ("mklink /h", "mklink /j")
READABLE_UNDER = [pytest.param(ANY_RO, e, id=f"any-ro-{e}") for e in CONTROLLABLE if not any(r in e for r in _WRITE_ROUTES)] + [
    pytest.param(ANY_RW, e, id=f"any-rw-{e}") for e in CONTROLLABLE if any(r in e for r in _WRITE_ROUTES)]  # fmt: skip


@pytest.mark.parametrize("shape,template", READABLE_UNDER)
def test_user_data_stays_readable_when_everything_is_allowed(t, shape, template):
    # Under "Allow everything", unlisted user data is readable by every route (and these are
    # the controls of the allow-list test above and of the refused-row test).
    res, out = _run(t, _escape(template, t / "home/docs"), _spec(t, shape))
    _ran(res, template)
    assert _leaked(res, out, USER_MARK), f"{template!r} could not read unlisted user data under {shape.id}: {out[-300:]}"


@pytest.mark.parametrize("shape", ALLOW_EVERYTHING_SHAPES)
def test_drive_shaped_user_data_stays_readable_when_everything_is_allowed(t, shape):
    r = t / "drive/docs"
    res, out = _run(t, f'type "{r}\\secret.txt" & dir /b "{r}" & "{PY}" -c "print(open(r\'{r}\\inner\\deep.txt\').read())"', _spec(t, shape))
    _ran(res)
    assert out.count(USER_MARK) == 2 and "secret.txt" in _listing(res), out[-300:]


def test_allow_list_keeps_rows_system_roots_and_python_readable(t):
    # Positive half of the allow-list posture (no fake user data needed): the rows, the
    # private workspace, the system roots and the host's Python are readable.
    (_priv(t) / "mine.txt").write_text("MINE-OK")
    (t / "home/rw/rw.txt").write_text("RW-OK")
    windir = os.environ.get("SystemRoot", r"C:\Windows")
    res, out = _run(t, f'type "{t}\\home\\ro\\f.txt" & type "{t}\\home\\rw\\rw.txt" & type mine.txt & dir /b "{windir}\\System32\\drivers\\etc" & '
                       f'dir /b "%ProgramFiles%" >nul && echo PROGRAMS-OK & "{PY}" -c "import json,sqlite3,ssl;print(6*7)"', _spec(t, ALLOW_LIST))  # fmt: skip
    _ran(res)
    for expected in (RO_MARK, "RW-OK", "MINE-OK", "hosts", "PROGRAMS-OK", "42"):
        assert expected in out, f"{expected!r} missing under allow-list: {out[-400:]}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_read_only_row_is_readable(t, shape):
    res, out = _run(t, f'type "{t}\\home\\ro\\f.txt" & dir /b /s "{t}\\home\\ro" & type "{t}\\home\\ro\\sub\\g.txt"', _spec(t, shape))
    _ran(res)
    assert out.count(RO_MARK) == 2 and any(line.endswith("g.txt") for line in _listing(res)), out[-300:]


# =====================================================================================
# Writing: every writer, every kind of place, every shape
# =====================================================================================

WRITE_OPS = [
    'type "{d}\\w_read.txt"',
    'echo NEW> "{d}\\w_over.txt"',
    'echo NEW>> "{d}\\w_app.txt"',
    'echo NEW> "{d}\\w_new.txt"',
    'del /q "{d}\\w_del.txt"',
    'ren "{d}\\w_ren.txt" w_renamed.txt',
    'move /y "{d}\\w_mvout.txt" "{priv}\\mvout_{tag}.txt"',
    'move /y "{priv}\\mvin_{tag}.txt" "{d}\\w_mvin.txt"',
    'mkdir "{d}\\w_made"',
    'rmdir /s /q "{d}\\w_rmdir"',
    'attrib +h "{d}\\w_attr.txt"',
    'icacls "{d}\\w_acl.txt" /grant *S-1-1-0:(R)',
    'takeown /f "{d}\\w_own.txt" & icacls "{d}\\w_own.txt" /reset & echo NEW> "{d}\\w_own.txt"',
    '"{py}" -c "open(r\'{d}\\w_py.txt\',\'w\').write(\'x\')"',
    '"{py}" -c "open(r\'\\\\?\\{d}\\w_q.txt\',\'w\').write(\'x\')"',
    'powershell -NoProfile -Command "Set-Content -LiteralPath \'{d}\\w_ps.txt\' x; Remove-Item -LiteralPath \'{d}\\w_psdel.txt\'"',
    'mklink /j "{priv}\\j_{tag}" "{d}" & echo NEW> "{priv}\\j_{tag}\\w_viaj.txt"',
    'cd /d "{d}" && echo NEW> w_cd.txt',  # last: it changes the folder of what follows
]


def _prepare_writes(d: Path, priv: Path, tag: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "w_read.txt").write_text(f"READ-{tag}-OK")
    for name in ("w_over.txt", "w_app.txt", "w_del.txt", "w_ren.txt", "w_mvout.txt", "w_attr.txt", "w_acl.txt", "w_own.txt", "w_psdel.txt"):
        (d / name).write_text("ORIG")
    (d / "w_rmdir").mkdir(exist_ok=True)
    (d / "w_rmdir/x.txt").write_text("ORIG")
    (priv / f"mvin_{tag}.txt").write_text("ORIG")


def _write_command(d: Path, priv: Path, tag: str) -> str:
    return " & ".join(op.format(d=d, priv=priv, tag=tag, py=PY) for op in WRITE_OPS)


def _grants_everyone(path: Path) -> bool:
    w = _w()
    aces, _ = w._read_dacl(str(path))
    return any(a.type == w.ALLOW and a.sid == "S-1-1-0" for a in aces)


def _write_effects(d: Path, priv: Path, tag: str) -> Dict[str, bool]:
    def text(p: Path) -> Optional[str]:
        try:
            return p.read_text().strip()
        except OSError:
            return None

    return {
        "overwrite": text(d / "w_over.txt") == "NEW",
        "append": "NEW" in (text(d / "w_app.txt") or ""),
        "create": (d / "w_new.txt").exists(),
        "delete": not (d / "w_del.txt").exists(),
        "rename": (d / "w_renamed.txt").exists() or not (d / "w_ren.txt").exists(),
        "move out": not (d / "w_mvout.txt").exists(),
        "move in": (d / "w_mvin.txt").exists(),
        "mkdir": (d / "w_made").exists(),
        "rmdir": not (d / "w_rmdir").exists(),
        "attrib": bool(os.stat(d / "w_attr.txt").st_file_attributes & 0x2),
        "icacls": _grants_everyone(d / "w_acl.txt"),
        "takeown + reset + write": text(d / "w_own.txt") == "NEW",
        "python": (d / "w_py.txt").exists(),
        "python \\\\?\\": (d / "w_q.txt").exists(),
        "powershell write": (d / "w_ps.txt").exists(),
        "powershell delete": not (d / "w_psdel.txt").exists(),
        "junction": (d / "w_viaj.txt").exists(),
        "cd + relative": (d / "w_cd.txt").exists(),
    }


def _probe_writes(t: Path, spec: SandboxSpec, d: Path, tag: str):
    priv = Path(spec.private_workspace)
    _prepare_writes(d, priv, tag)
    res, out = _run(t, _write_command(d, priv, tag), spec, timeout=120)
    _ran(res, f"the write probe of {tag}")
    return res, out, _write_effects(d, priv, tag)


# (target, where it is, writable under: any-rw, any-ro, allow-list)
TARGETS = {
    "private workspace": (lambda t: _priv(t) / "probe", (True, True, True)),
    "private .tmp": (lambda t: _priv(t) / ".tmp/probe", (True, True, True)),
    "read & write row": (lambda t: t / "home/rw/probe", (True, True, True)),
    "read-only row": (lambda t: t / "home/ro/probe", (False, False, False)),
    "unlisted folder": (lambda t: t / "home/unlisted/probe", (True, False, False)),
    "refused row": (lambda t: t / "home/refused/probe", (False, False, False)),
}
_SHAPE_INDEX = {ANY_RW: 0, ANY_RO: 1, ALLOW_LIST: 2}


@pytest.mark.parametrize("target", list(TARGETS))
@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_writes_follow_the_shape(t, shape, target):
    where, writable = TARGETS[target]
    expected = writable[_SHAPE_INDEX[shape]]
    d = where(t)
    tag = target.replace(" ", "_").replace("&", "and").replace(".", "")
    res, out, effects = _probe_writes(t, _spec(t, shape), d, tag)
    if expected:
        missed = [k for k, v in effects.items() if not v]
        assert not missed, f"{target} must be writable under {shape.id}; these writes failed: {missed}\n{out[-1500:]}"
    else:
        leaked = [k for k, v in effects.items() if v]
        assert not leaked, f"{target} must NOT be writable under {shape.id}; these writes went through: {leaked}"
        # Positive control: the same command, in the same shape, writes a read & write row.
        control = t / "home/rw" / f"control_{tag}"
        _, cout, ceffects = _probe_writes(t, _spec(t, shape), control, f"control_{tag}")
        # Made after the row's rule was first written: a permission change there is the
        # subject of test_permission_changes_in_a_read_and_write_row, not of this control.
        missed = [k for k, v in ceffects.items() if not v and k != "icacls"]
        assert not missed, f"control: these writes failed in a read & write row under {shape.id}: {missed}\n{cout[-1500:]}"
    readable = {"refused row": False, "unlisted folder": None if shape.allow_list else True}.get(target, True)
    if readable is not None:
        assert (f"READ-{tag}-OK" in out) == readable, f"{target} readable={not readable} under {shape.id}: {out[-400:]}"
    if target == "read-only row":
        assert (t / "home/ro/f.txt").read_text() == RO_MARK and (t / "home/ro/sub/g.txt").read_text() == RO_MARK


@pytest.mark.parametrize("shape", CONFINED_SHAPES)
def test_the_hosts_temp_folder_is_not_writable(t, shape):
    # The command's own TEMP is the private one; the host's (a real folder: this is a
    # negative check only, and anything it manages to create is removed) is not writable.
    host_temp = Path(os.path.realpath(tempfile.gettempdir()))
    assert str(t).lower().startswith(str(host_temp).lower()), "pytest's tmp_path lies in the host's TEMP (the user writes it)"
    probe = f"abstractcore-sandbox-contract-{uuid.uuid4().hex[:12]}"
    made = [host_temp / f"{probe}.txt", host_temp / f"{probe}-py.txt", host_temp / f"{probe}-ps.txt", host_temp / f"{probe}-d"]
    command = (f'echo x> "{made[0]}" & "{PY}" -c "open(r\'{made[1]}\',\'w\').write(\'x\')" & '
               f'powershell -NoProfile -Command "Set-Content -LiteralPath \'{made[2]}\' x" & mkdir "{made[3]}" & '
               f'echo x> "%TEMP%\\mine.txt" & "{PY}" -c "import tempfile;print(\'PYTEMP\',tempfile.gettempdir())" & echo TEMP=%TEMP% & echo TMP=%TMP%')  # fmt: skip
    try:
        res, out = _run(t, command, _spec(t, shape))
        _ran(res)
        created = [str(p) for p in made if p.exists()]
        assert not created, f"wrote in the host's TEMP under {shape.id}: {created}"
    finally:
        for p in made:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
    tmp = _priv(t) / ".tmp"
    # Positive control: the same writer, in the command's own TEMP, writes.
    assert (tmp / "mine.txt").exists()
    assert f"TEMP={tmp}" in out and f"TMP={tmp}" in out, out[-400:]
    assert any(line.startswith("PYTEMP ") and _same(line[7:], tmp) for line in _listing(res)), out[-400:]


@pytest.mark.parametrize("shape", CONFINED_SHAPES)
def test_the_registry_is_not_writable(t, shape):
    import winreg

    def exists() -> bool:
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PROBE))
            return True
        except OSError:
            return False

    def remove() -> None:
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, REG_PROBE)
        except OSError:
            pass

    assert not exists(), f"HKCU\\{REG_PROBE} exists before the test"
    # Control: this user may create keys there (opened for that, nothing is created) ...
    winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software", 0, winreg.KEY_CREATE_SUB_KEY))
    command = (f'reg add HKCU\\{REG_PROBE} /f & powershell -NoProfile -Command "New-Item -Path \'HKCU:\\{REG_PROBE}\'" & '
               f'"{PY}" -c "import winreg;winreg.CreateKey(winreg.HKEY_CURRENT_USER,r\'{REG_PROBE}\')" & '
               f'reg query "HKCU\\Software" >nul && echo REG-READ-OK')  # fmt: skip
    try:
        res, out = _run(t, command, _spec(t, shape))
        _ran(res)
        assert not exists(), f"a command wrote HKCU\\{REG_PROBE} under {shape.id}"
    finally:
        if exists():
            remove()
    # ... and the command can read the registry (reg.exe runs: the refusal is the sandbox's).
    assert "REG-READ-OK" in out, out[-400:]


# =====================================================================================
# Nesting and ties
# =====================================================================================

NEST = {  # folder: (marker, readable, writable); rows in _nest_rows
    "a_ro": ("NEST-A", True, False),
    "a_ro/in_rw": ("NEST-B", True, True),
    "c_rw": ("NEST-C", True, True),
    "c_rw/in_ro": ("NEST-D", True, False),
    "e_rw": ("NEST-E", True, True),
    "e_rw/mid": ("NEST-F", False, False),
    "e_rw/mid/in_rw": ("NEST-G", True, True),
    "h_rw": ("NEST-H", True, True),
    "h_rw/deny": ("NEST-I", False, False),
    "tie_refused": ("NEST-J", False, False),
    "tie_ro": ("NEST-K", True, False),
}


def _nest(root: Path) -> None:
    for folder, (mark, _, _) in NEST.items():
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "n.txt").write_text(mark)


def _nest_rows(root: Path):
    p = lambda rel: str(root / rel)  # noqa: E731
    allowed = (
        SandboxRow(p("a_ro"), "ro"), SandboxRow(p("a_ro/in_rw"), "rw"),  # rw inside ro
        SandboxRow(p("c_rw"), "rw"), SandboxRow(p("c_rw/in_ro"), "ro"),  # ro inside rw
        SandboxRow(p("e_rw"), "rw"), SandboxRow(p("e_rw/mid/in_rw"), "rw"),  # rw inside refused inside rw
        SandboxRow(p("h_rw"), "rw"),  # refused inside rw
        SandboxRow(p("tie_refused"), "rw"),  # tie: a refusal wins
        SandboxRow(p("tie_ro"), "rw"), SandboxRow(p("tie_ro"), "ro"),  # tie: read-only beats read & write
    )  # fmt: skip
    refused = (p("e_rw/mid"), p("h_rw/deny"), p("tie_refused"))
    return allowed, refused


def _nest_command(root: Path) -> str:
    parts = []
    for folder in NEST:
        f = root / folder
        parts.append(f'type "{f}\\n.txt" & echo w> "{f}\\w.txt"')
    return " & ".join(parts)


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_nesting_and_ties(t, shape):
    root = t / "home/nest"
    _nest(root)
    ctl = t / "home/nest_control"
    _nest(ctl)  # made before the first command (a folder made later: test_a_folder_made_after_a_first_run_can_become_a_row)
    allowed, refused = _nest_rows(root)
    res, out = _run(t, _nest_command(root), _spec(t, shape, allowed=allowed, refused=refused), timeout=120)
    _ran(res)
    wrong = []
    for folder, (mark, readable, writable) in NEST.items():
        if (mark in out) != readable:
            wrong.append(f"{folder}: readable={not readable}")
        if (root / folder / "w.txt").exists() != writable:
            wrong.append(f"{folder}: writable={not writable}")
    assert not wrong, f"nesting under {shape.id}: {wrong}\n{out[-1500:]}"
    # Positive control: the same command with every row read & write and nothing refused.
    res, out = _run(t, _nest_command(ctl), _spec(t, shape, allowed=tuple(SandboxRow(str(ctl / f), "rw") for f in NEST), refused=()), timeout=120)
    _ran(res)
    missed = [f for f, (mark, _, _) in NEST.items() if mark not in out or not (ctl / f / "w.txt").exists()]
    assert not missed, f"control under {shape.id}: {missed}\n{out[-1500:]}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_builtin_refusal_is_absolute_and_its_exception_works(t, shape):
    # Even under rows that "allow" them, the data folder, the other session's folder and .ssh stay closed.
    spec = _spec(t, shape, allowed=(SandboxRow(str(t / "home"), "rw"), SandboxRow(str(t / "data/secrets"), "rw"), SandboxRow(str(_priv2(t)), "rw")))
    reads = (f'type "{t}\\data\\gateway.sqlite3" & dir /b "{t}\\data" & type "{t}\\data\\secrets\\token" & type "{_priv2(t)}\\theirs.txt" & '
             f'type "..\\session-2\\theirs.txt" & type "{t}\\home\\.ssh\\id_marker" & dir /b "{t}\\home\\.ssh" & '
             f'"{PY}" -c "print(open(r\'{t}\\data\\secrets\\token\').read())"')  # fmt: skip
    writes = f' & echo x> "{t}\\data\\new.txt" & echo x> "{_priv2(t)}\\new.txt" & echo x> "{t}\\data\\secrets\\new.txt" & echo x> "{t}\\home\\.ssh\\new.txt"'
    res, out = _run(t, reads + writes, spec)
    _ran(res)
    assert DATA_MARK not in out and SSH_MARK not in out, f"a built-in refusal was read under {shape.id}: {out[-600:]}"
    assert not {"id_marker", "theirs.txt"} & set(_listing(res))
    # The data folder is on the way to the run's own folder: its names are listed, in every
    # shape (accepted limit (c): test_limit_c_names_on_the_way_to_a_row_are_listed).
    assert "gateway.sqlite3" in _listing(res), _listing(res)
    made = [p for p in (t / "data/new.txt", _priv2(t) / "new.txt", t / "data/secrets/new.txt", t / "home/.ssh/new.txt") if p.exists()]
    assert not made, f"wrote under a built-in refusal under {shape.id}: {made}"
    # The exception: the run's own folder, inside the data folder, is its own.
    res, out = _run(t, 'echo mine> note.txt && type note.txt && echo %TEMP%', spec)
    _ran(res)
    assert "mine" in out and str(_priv(t) / ".tmp") in out and (_priv(t) / "note.txt").exists()
    # Positive control: without the built-in refusals, the same commands read and write.
    open_spec = _spec(t, shape, allowed=spec.allowed + (SandboxRow(str(t / "data"), "rw"), SandboxRow(str(t / "home/.ssh"), "rw")), builtin_refused=(), builtin_allowed=())
    res, out = _run(t, reads + writes, open_spec)
    _ran(res)
    assert out.count(DATA_MARK) >= 4 and SSH_MARK in out, out[-600:]
    assert (t / "data/new.txt").exists() and (_priv2(t) / "new.txt").exists() and (t / "home/.ssh/new.txt").exists()


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_refusal_that_cannot_be_enforced_does_not_run(t, shape):
    # A folder this user can read but whose access list is not theirs to change (as a
    # system folder is). Whatever the backend does with it, the file is never read: either
    # the command does not run (126), or the refusal holds without that list. Outside the
    # user data, so that the unrefused control reads it under the allow-list too.
    locked = t / "locked/inner"
    locked.mkdir(parents=True)
    (locked / "f.txt").write_text(REFUSED_MARK)
    subprocess.run(["icacls", str(locked), "/grant", "*S-1-3-4:(OI)(CI)(RX)", "/deny", f"*{_w().identity_sids()[0]}:(WDAC,WO)"], check=True, capture_output=True)
    command = f'type "{locked}\\f.txt" & "{PY}" -c "print(open(r\'{locked}\\f.txt\').read())" & echo ran> ran.txt'
    res, out = _run(t, command, _spec(t, shape, refused=(str(t / "home/refused"), str(locked))))
    assert "return_code" in res, f"the command never reached the sandbox: {res.get('error')}"
    assert REFUSED_MARK not in out, f"an unenforceable refusal was read under {shape.id}"
    if (_priv(t) / "ran.txt").exists():
        assert res["return_code"] != 126
    else:
        assert res["return_code"] == 126 and "the command did not run" in out, out[-400:]
    # Positive control: unrefused, the same command reads it.
    (_priv(t) / "ran.txt").unlink(missing_ok=True)
    res, out = _run(t, command, _spec(t, shape))
    _ran(res)
    assert REFUSED_MARK in out


# =====================================================================================
# Isolation between runs
# =====================================================================================


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_two_runs_of_one_data_folder_each_reach_only_their_own(t, shape):
    one, two = _priv(t), _priv2(t)
    (one / "mine.txt").write_text("ONE-OK")
    (two / "theirs.txt").write_text("TWO-OK")
    s1, s2 = _spec(t, shape), _spec2(t, shape)
    res, out = _run(t, f'type mine.txt & echo w> one.txt & type "{two}\\theirs.txt" & echo x> "{two}\\from-one.txt" & dir /b "{two}"', s1)
    _ran(res)
    assert "ONE-OK" in out and (one / "one.txt").exists(), out[-400:]
    assert "TWO-OK" not in out and "theirs.txt" not in _listing(res) and not (two / "from-one.txt").exists()
    res, out = _run(t, f'type theirs.txt & echo w> two.txt & type "{one}\\mine.txt" & echo x> "{one}\\from-two.txt" & dir /b "{one}"', s2)
    _ran(res)
    assert "TWO-OK" in out and (two / "two.txt").exists(), out[-400:]
    assert "ONE-OK" not in out and "mine.txt" not in _listing(res) and not (one / "from-two.txt").exists()
    # Again after the second run: the first one has not lost its own folder.
    res, out = _run(t, "type mine.txt & echo w> again.txt", s1)
    assert "ONE-OK" in out and (one / "again.txt").exists()


# B lists the shared folder read-only, or not at all (under the write-confined shapes, an
# unlisted folder is not writable either).
B_MODES = [pytest.param(ANY_RW, "ro", id="any-rw-B-ro"), pytest.param(ANY_RO, "ro", id="any-ro-B-ro"), pytest.param(ANY_RO, "unlisted", id="any-ro-B-unlisted"),
           pytest.param(ALLOW_LIST, "ro", id="allow-list-B-ro"), pytest.param(ALLOW_LIST, "unlisted", id="allow-list-B-unlisted")]  # fmt: skip


def _ab(t: Path, shape: Shape, b_mode: str):
    x = t / "home/shared"
    a = _spec(t, shape, allowed=(SandboxRow(str(t / "home/ro"), "ro"), SandboxRow(str(x), "rw")))
    b_rows = (SandboxRow(str(t / "home/ro"), "ro"),) + ((SandboxRow(str(x), "ro"),) if b_mode == "ro" else ())
    b = _spec2(t, shape, allowed=b_rows)
    return x, a, b


def _write_x(t: Path, x: Path, spec: SandboxSpec, name: str):
    res, out = _run(t, f'echo w> "{x}\\{name}-cmd.txt" & "{PY}" -c "open(r\'{x}\\{name}-py.txt\',\'w\').write(\'x\')"', spec)
    _ran(res, f"{name}")
    return [(x / f"{name}-{k}.txt").exists() for k in ("cmd", "py")]


@pytest.mark.parametrize("shape,b_mode", B_MODES)
def test_one_folder_two_modes_one_after_the_other(t, shape, b_mode):
    x, a, b = _ab(t, shape, b_mode)
    for step, (who, spec) in enumerate((("A", a), ("B", b), ("A", a), ("B", b))):
        wrote = _write_x(t, x, spec, f"{who}{step}")
        if who == "A":
            assert all(wrote), f"A (read & write) could not write the shared folder at step {step}"
        else:
            assert not any(wrote), f"B ({b_mode}) wrote the shared folder at step {step}, after A had it read & write"


LONG = r"""
import os, sys, time
x, me, who = sys.argv[1:4]
def attempt(name):
    try:
        with open(os.path.join(x, who + "-" + name + ".txt"), "w") as f:
            f.write(who)
        print(who + "-" + name + ": wrote", flush=True)
    except OSError as exc:
        print(who + "-" + name + ": refused " + type(exc).__name__, flush=True)
attempt("before")
open(os.path.join(me, who + ".ready"), "w").close()
end = time.time() + 60
while not os.path.exists(os.path.join(me, who + ".go")) and time.time() < end:
    time.sleep(0.05)
attempt("after")
"""


def _start_long(spec: SandboxSpec, x: Path, who: str) -> subprocess.Popen:
    me = Path(spec.private_workspace)
    (me / "long.py").write_text(LONG)
    argv, env = _box(spec).wrap([PY, str(me / "long.py"), str(x), str(me), who], str(me))
    proc = subprocess.Popen(argv, cwd=me, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while time.time() < deadline and not (me / f"{who}.ready").exists():
        if proc.poll() is not None:
            pytest.fail(f"the long command of {who} ended early: {proc.stdout.read()[-600:]}")
        time.sleep(0.05)
    assert (me / f"{who}.ready").exists(), f"the long command of {who} never started"
    return proc


def _finish_long(proc: subprocess.Popen, spec: SandboxSpec, who: str) -> str:
    (Path(spec.private_workspace) / f"{who}.go").write_text("go")
    out, _ = proc.communicate(timeout=60)
    return out


@pytest.mark.parametrize("shape,b_mode", B_MODES)
def test_one_folder_two_modes_at_the_same_time(t, shape, b_mode):
    x, a, b = _ab(t, shape, b_mode)
    # A's long command runs while B's command runs.
    proc = _start_long(a, x, "a")
    try:
        assert not any(_write_x(t, x, b, "B-during-A")), f"B ({b_mode}) wrote the shared folder while A's command ran"
    finally:
        out = _finish_long(proc, a, "a")
    assert (x / "a-before.txt").exists() and (x / "a-after.txt").exists(), f"A lost its write while B ran: {out[-600:]}"
    # B's long command runs while A's command runs.
    proc = _start_long(b, x, "b")
    try:
        assert all(_write_x(t, x, a, "A-during-B")), "A could not write the shared folder while B's command ran"
    finally:
        out = _finish_long(proc, b, "b")
    assert not (x / "b-before.txt").exists() and not (x / "b-after.txt").exists(), f"B wrote the shared folder around A's run: {out[-600:]}"
    assert "b-before: refused" in out and "b-after: refused" in out, out[-600:]  # B's command did run and try
    # After both: B still cannot, A still can.
    assert not any(_write_x(t, x, b, "B-last")) and all(_write_x(t, x, a, "A-last"))


@pytest.mark.parametrize("shape,b_mode", B_MODES)
def test_a_killed_launcher_leaves_nothing_for_the_next_run(t, shape, b_mode):
    x, a, b = _ab(t, shape, b_mode)
    proc = _start_long(a, x, "a")
    proc.kill()  # the launcher, mid-command: no clean-up of its own runs
    proc.communicate(timeout=60)
    assert (x / "a-before.txt").exists()
    assert not any(_write_x(t, x, b, "B-after-kill")), f"B ({b_mode}) wrote the shared folder after A's launcher was killed"
    assert all(_write_x(t, x, a, "A-after-kill")), "A could not write its folder after its launcher was killed"
    assert not any(_write_x(t, x, b, "B-after-A")), f"B ({b_mode}) wrote the shared folder after A ran again"


# =====================================================================================
# Ordinary tools keep working
# =====================================================================================


def _tool_check(t: Path, tool: str):
    tmp = _priv(t) / ".tmp"
    if tool == "cmd":
        command = 'echo a b| findstr b & (echo one& echo two) | %SystemRoot%\\System32\\find.exe /c /v "" & set /a 6*7 & echo. & echo hidden>nul & echo shown & ver & echo t> "%TEMP%\\t.txt" & dir /b "%TEMP%"'
        return command, lambda res, out: ["a b" in out, "2" in out.split(), "42" in out, "hidden" not in out, "shown" in out]
    if tool == "powershell":
        command = 'powershell -NoProfile -Command "$ExecutionContext.SessionState.LanguageMode; [IO.Path]::GetTempPath(); \'x\' | Out-File -FilePath ps.txt; Get-Content ps.txt; 6*7"'
        return command, lambda res, out: ["FullLanguage" in out, any(_same(line, tmp) for line in _listing(res)), "42" in out, (_priv(t) / "ps.txt").exists()]
    if tool == "python":
        code = ("import tempfile,subprocess,sys;d=tempfile.gettempdir();f=tempfile.NamedTemporaryFile(delete=False);f.write(b'x');f.close();"
                "print('TMP',d);print('FILE',f.name);r=subprocess.run([sys.executable,'-c','print(6*7)'],capture_output=True,text=True);"
                "print('CHILD',r.stdout.strip());print('NULL',subprocess.run([sys.executable,'-c','print(1)'],stdout=subprocess.DEVNULL).returncode)")  # fmt: skip
        command = f'"{PY}" -c "{code}"'

        def check(res, out):
            lines = dict(line.split(" ", 1) for line in _listing(res) if " " in line)
            return [_same(lines.get("TMP", ""), tmp), _same(os.path.dirname(lines.get("FILE", "")), tmp), lines.get("CHILD") == "42", lines.get("NULL") == "0"]

        return command, check
    if tool == "git":
        if not shutil.which("git"):
            pytest.skip("git is not installed on this host")
        command = ("git init -q repo && cd repo && echo x> f.txt && git add f.txt && "
                   "git -c user.name=sandbox -c user.email=sandbox@example.invalid commit -q -m contract-commit && git log --oneline && git status --short && echo GIT-DONE")  # fmt: skip
        return command, lambda res, out: ["contract-commit" in out, "GIT-DONE" in out, (_priv(t) / "repo/.git/HEAD").exists()]
    raise AssertionError(tool)


@pytest.mark.parametrize("tool", ["cmd", "powershell", "python", "git"])
@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_ordinary_tools_keep_working(t, shape, tool):
    command, check = _tool_check(t, tool)
    res, out = _run(t, command, _spec(t, shape), timeout=120)
    _ran(res, tool)
    results = check(res, out)
    assert res["return_code"] == 0 and all(results), f"{tool} under {shape.id}: checks {results}, exit {res['return_code']}\n{out[-1200:]}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_the_command_gets_the_hosts_environment_with_its_own_temp(t, shape, monkeypatch):
    monkeypatch.setenv("AF_R12_PROVIDER_KEY", "sk-must-not-leak")
    env = {k: v for k, v in os.environ.items() if k != "AF_R12_PROVIDER_KEY"}
    env["AF_R12_HOST_VAR"] = "kept-by-the-host"
    sb.configure_host(env=env)
    res, out = _run(t, "set", _spec(t, shape))
    _ran(res)
    tmp = _priv(t) / ".tmp"
    assert "kept-by-the-host" in out and "sk-must-not-leak" not in out and "AF_R12_PROVIDER_KEY" not in out
    assert f"TEMP={tmp}" in _listing(res) and f"TMP={tmp}" in _listing(res), out[-600:]


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_exit_codes_and_standard_streams_pass_through(t, shape):
    box = _box(_spec(t, shape))
    argv, env = box.wrap([PY, "-c", "import sys;d=sys.stdin.read();sys.stdout.write(d.upper());sys.stderr.write('E'*3);sys.exit(42)"], str(_priv(t)))
    proc = subprocess.run(argv, cwd=_priv(t), env=env, input="abc", capture_output=True, text=True, timeout=60)
    assert (proc.returncode, proc.stdout, proc.stderr) == (42, "ABC", "EEE")
    argv, env = box.wrap("exit 3", str(_priv(t)))
    assert subprocess.run(argv, cwd=_priv(t), env=env, capture_output=True, timeout=60).returncode == 3
    res, out = _run(t, f'"{PY}" -c "import sys;print(\'OUT\');print(\'ERR\',file=sys.stderr);sys.exit(5)"', _spec(t, shape))
    _ran(res)
    assert res["return_code"] == 5 and res["success"] is False and "OUT" in res["stdout"] and "ERR" in res["stderr"]


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_killing_the_launcher_ends_the_whole_tree(t, shape):
    box = _box(_spec(t, shape))
    code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import os,time;open(\"grandchild.pid\",\"w\").write(str(os.getpid()));time.sleep(120)']);time.sleep(120)"
    argv, env = box.wrap([PY, "-c", code], str(_priv(t)))
    proc = subprocess.Popen(argv, cwd=_priv(t), env=env)
    pid_file = _priv(t) / "grandchild.pid"
    deadline = time.time() + 30
    while time.time() < deadline and not (pid_file.exists() and pid_file.read_text().strip()):
        time.sleep(0.05)
    assert pid_file.exists(), "the command never started its grandchild"
    grandchild = int(pid_file.read_text())
    proc.kill()
    proc.wait(timeout=30)
    deadline = time.time() + 15
    listed = True
    while time.time() < deadline and listed:
        listed = str(grandchild) in subprocess.run(["tasklist", "/fi", f"PID eq {grandchild}", "/nh"], capture_output=True, text=True).stdout
        if listed:
            time.sleep(0.1)
    assert not listed, f"the command's grandchild outlived its launcher under {shape.id}"


# =====================================================================================
# What must not change on the user's side
# =====================================================================================


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_the_user_keeps_every_right_on_every_folder_used(t, shape):
    (_priv(t) / "mine.txt").write_text("MINE")
    nest = t / "home/nest"
    _nest(nest)
    allowed, refused = _nest_rows(nest)
    spec = _spec(t, shape, allowed=allowed + (SandboxRow(str(t / "home/ro"), "ro"), SandboxRow(str(t / "home/rw"), "rw")), refused=refused + (str(t / "home/refused"),))
    res, _ = _run(t, f'type mine.txt & echo w> "{t}\\home\\rw\\w.txt" & type "{t}\\home\\ro\\f.txt"', spec)
    _ran(res)
    _run(t, "echo again", _spec2(t, shape))
    folders = [t, t / "data", t / "data/workspaces", _priv(t), _priv(t) / ".tmp", _priv2(t), t / "data/secrets", t / "home", t / "home/.ssh",
               t / "home/docs", t / "home/refused", t / "home/refused/inner", t / "home/ro", t / "home/ro/sub", t / "home/rw", t / "home/unlisted",
               t / "drive", t / "drive/docs"] + [nest / f for f in NEST]  # fmt: skip
    problems = []
    for folder in folders:
        try:
            for entry in folder.iterdir():
                if entry.is_file():
                    entry.read_bytes()
            probe = folder / f"user-{uuid.uuid4().hex[:6]}.txt"
            probe.write_text("user")
            assert probe.read_text() == "user"
            probe.unlink()
        except OSError as exc:
            problems.append(f"{folder}: {exc}")
            continue
        for args in (["/grant", "*S-1-1-0:(RX)"], ["/remove:g", "*S-1-1-0"]):
            done = subprocess.run(["icacls", str(folder), *args], capture_output=True, text=True)
            if done.returncode != 0:
                problems.append(f"{folder}: icacls {' '.join(args)}: {done.stdout.strip()} {done.stderr.strip()}")
    assert not problems, f"after {shape.id} runs the user lost rights: {problems}"
    # And the user can delete everything the runs used.
    shutil.rmtree(t)
    assert not t.exists()


# =====================================================================================
# Rows made later, permission changes, junctions
# =====================================================================================


def _carved(t: Path, shape: Shape) -> Path:
    """A refused folder with a read & write row inside (so it is carved, not closed): the user
    data root under the allow-list, a refused row with a reopened child otherwise."""
    return t / "home" if shape.allow_list else t / "home/refused"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_folder_made_after_a_first_run_can_become_a_row(t, shape):
    # A user makes a new project folder in their profile (or in a refused folder that
    # already has a row) after the first command, then adds it as a read & write workspace:
    # the contract makes it readable and writable like any row.
    carved = _carved(t, shape)
    early, late = carved / "early-project", carved / "late-project"
    early.mkdir()
    (early / "n.txt").write_text("EARLY-OK")
    first = _spec(t, shape, allowed=(SandboxRow(str(t / "home/rw"), "rw"), SandboxRow(str(t / "home/refused/inner"), "rw")))
    _ran(_run(t, "echo first", first)[0])
    late.mkdir()
    (late / "n.txt").write_text("LATE-OK")
    rows = first.allowed + (SandboxRow(str(early), "rw"), SandboxRow(str(late), "rw"))
    res, out = _run(t, f'type "{early}\\n.txt" & echo w> "{early}\\w.txt" & type "{late}\\n.txt" & echo w> "{late}\\w.txt"', _spec(t, shape, allowed=rows))
    _ran(res)
    # Control: the folder that existed at the first run works as a row.
    assert "EARLY-OK" in out and (early / "w.txt").exists(), f"control under {shape.id}: {out[-400:]}"
    assert "LATE-OK" in out and (late / "w.txt").exists(), f"a row made after the first run under {shape.id}: {out[-400:]}"


@pytest.mark.parametrize("place", ["row in user data", "row outside user data", "row inside a refused folder", "private workspace"])
@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_permission_changes_in_a_read_and_write_row(t, shape, place):
    # A read & write row is the command's to write; on macOS and Linux that includes its
    # permissions (chmod). Here: a file made in the row after its rule was written (what
    # every command after the first one meets), its access list changed with icacls.
    outer = t / "outer_rw"
    outer.mkdir()
    d = {"row in user data": t / "home/rw", "row outside user data": outer, "row inside a refused folder": t / "home/refused/inner",
         "private workspace": _priv(t)}[place]  # fmt: skip
    spec = _spec(t, shape, allowed=(SandboxRow(str(t / "home/rw"), "rw"), SandboxRow(str(outer), "rw"), SandboxRow(str(t / "home/refused/inner"), "rw")))
    _ran(_run(t, "echo first", spec)[0])
    f = d / "made-later.txt"
    f.write_text("x")
    res, out = _run(t, f'icacls "{f}" /grant *S-1-1-0:(R)', spec)
    _ran(res)
    assert _grants_everyone(f), f"icacls refused in a {place} under {shape.id}: {out[-300:]}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_junction_needs_a_target_the_command_may_write(t, shape):
    # Windows refuses a process below Medium integrity a junction to a folder it cannot write
    # (an operating-system rule for sandboxed processes). So under the write-confined shapes a
    # junction to a read-only row or to unlisted readable data cannot be made; under any-rw,
    # and to a read & write row everywhere, it can.
    spec = _spec(t, shape, allowed=(SandboxRow(str(t / "home/rw"), "rw"), SandboxRow(str(t / "home/shown"), "ro")))
    made = {}
    for target in (t / "home/rw", t / "home/shown", t / "home/docs"):
        name = f"j-{target.name}"
        res, out = _run(t, f'mklink /j {name} "{target}"', spec)
        _ran(res)
        made[target.name] = (_priv(t) / name).exists()
    assert made["rw"], "a junction to a read & write row could not be made"
    expected = not shape.writes_confined
    assert made["shown"] is expected, f"junction to a read-only row under {shape.id}: made={made['shown']}"
    if not shape.allow_list:  # unlisted user data is not reachable under the allow-list anyway
        assert made["docs"] is expected, f"junction to unlisted readable data under {shape.id}: made={made['docs']}"


# =====================================================================================
# Accepted limits (each asserts what happens, so a change in either direction is noticed)
# =====================================================================================

# B lists the folder read-only or not at all; under the allow-list an unlisted folder in the
# user data is not reachable, so only the cases where B can reach the folder show the limit.
LIMIT_B = [pytest.param(ANY_RO, "ro", id="any-ro-B-ro"), pytest.param(ANY_RO, "unlisted", id="any-ro-B-unlisted"),
           pytest.param(ALLOW_LIST, "ro", id="allow-list-B-ro"), pytest.param(ALLOW_LIST, "unlisted", id="allow-list-B-unlisted")]  # fmt: skip


def _reaches(shape: Shape, b_mode: str) -> bool:
    """Where the limit shows (measured against 6b57fd39): only where B's refusal to write rests
    on its integrity level alone, i.e. the read-only default. Under the allow-list B's
    read-only row carries a refusal of its own (it lies in the refused user data), and an
    unlisted folder there is not reachable: no write either way."""
    return shape == ANY_RO


@pytest.mark.parametrize("shape,b_mode", LIMIT_B)
def test_limit_a_a_command_running_when_a_folder_is_first_labelled_can_write_it(t, shape, b_mode):
    # (a) A command ALREADY RUNNING when a folder becomes some run's read & write row for the
    # first time can write that folder until it ends; its next launch cannot.
    x, a, b = _ab(t, shape, b_mode)
    proc = _start_long(b, x, "b")
    try:
        assert all(_write_x(t, x, a, "A-first")), "A could not write its new row"
    finally:
        out = _finish_long(proc, b, "b")
    assert "b-before: refused" in out, out[-600:]
    after = (x / "b-after.txt").exists()
    assert after is _reaches(shape, b_mode), f"limit (a) changed under {shape.id}/{b_mode}: B's running command wrote={after}\n{out[-600:]}"
    assert not any(_write_x(t, x, b, "B-next")), "B's next launch wrote the folder"


@pytest.mark.parametrize("shape,b_mode", LIMIT_B)
def test_limit_b_content_with_its_own_protected_permissions_in_a_row(t, shape, b_mode):
    # (b) Content inside a read & write row that carries its own protected permissions (what
    # Python 3.13's tempfile.mkdtemp makes) can be written by another confined run that can
    # reach it; the rest of the row cannot.
    x, a, b = _ab(t, shape, b_mode)
    assert all(_write_x(t, x, a, "A"))
    own = x / "private-tmp"
    own.mkdir()
    _set_acl(own, "D:PAI(A;OICI;FA;;;{me})(A;OICI;FA;;;SY)")
    assert not any(_write_x(t, x, b, "B-row")), "B wrote the row itself"
    res, out = _run(t, f'echo w> "{own}\\from-b.txt"', b)
    _ran(res)
    wrote = (own / "from-b.txt").exists()
    assert wrote is _reaches(shape, b_mode), f"limit (b) changed under {shape.id}/{b_mode}: wrote={wrote}: {out[-300:]}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_limit_c_names_on_the_way_to_a_row_are_listed(t, shape):
    # (c) In every shape the folders on the way to a row list the NAMES directly inside them
    # (git needs it); their content stays unreadable. Under any-rw: a refused folder holding a
    # read & write row; under the allow-list also the user data root.
    listings = {"data": f'dir /b "{t}\\data"', "refused folder holding a row": f'dir /b "{t}\\home\\refused"'}
    if shape.allow_list:
        listings["user data root"] = f'dir /b "{t}\\home"'
    spec = _spec(t, shape, allowed=(SandboxRow(str(t / "home/ro"), "ro"), SandboxRow(str(t / "home/rw"), "rw"), SandboxRow(str(t / "home/refused/inner"), "rw")))
    expect = {"data": {"gateway.sqlite3", "secrets"}, "refused folder holding a row": {"secret.txt"}, "user data root": {"docs", "refused"}}
    for what, command in listings.items():
        res, out = _run(t, command + f' & type "{t}\\data\\gateway.sqlite3" & type "{t}\\home\\docs\\secret.txt" & type "{t}\\home\\refused\\secret.txt"', spec)
        _ran(res)
        names = set(_listing(res))
        assert expect[what] <= names, f"limit (c) changed under {shape.id} for {what}: {sorted(names)}"
        assert REFUSED_MARK not in out
        assert DATA_MARK not in out
        if shape.allow_list:
            assert USER_MARK not in out


def _git_bash() -> Optional[Path]:
    git = shutil.which("git")
    if not git:
        return None
    root = Path(git).resolve().parents[1]
    for rel in ("usr/bin/bash.exe", "bin/bash.exe"):
        if (root / rel).exists():
            return root / rel
    return None


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_limit_d_git_bash_does_not_start_in_the_confined_shapes(t, shape):
    # (d) Git Bash / MSYS2 programs do not start under the write-confined shapes; git.exe does.
    bash = _git_bash()
    if bash is None:
        pytest.skip("Git Bash is not installed on this host")
    res, out = _run(t, f'"{bash}" -c "echo GITBASH-OK" & git --version', _spec(t, shape), timeout=60)
    _ran(res)
    assert "git version" in out, out[-300:]
    assert ("GITBASH-OK" in out) is (not shape.writes_confined), f"limit (d) changed under {shape.id}: {out[-400:]}"


def test_limit_e_the_allow_list_leaves_what_the_user_cannot_change_readable():
    # (e) The real user data roots (computed, never launched with): the profile is one; the
    # system roots, C:\Users\Public and ProgramData lie under none (they stay readable).
    roots = REAL_USER_DATA_ROOTS()
    home = os.path.realpath(os.path.expanduser("~"))
    assert any(_same(r, home) for r in roots), roots
    for var in ("SystemRoot", "ProgramFiles", "ProgramData", "PUBLIC"):
        if os.environ.get(var):
            place = os.path.realpath(os.environ[var])
            assert not any(sb._under(place, r) for r in roots), f"{var} ({place}) is user data: {roots}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_limit_e_already_low_places_stay_writable(t, shape):
    # (e) A folder Windows already labels Low stays writable under the confined shapes,
    # except the profile's own Low places (LocalLow, Favorites), which the read-only default
    # guards and the allow-list refuses with the profile.
    low = t / "low-place"
    low.mkdir()
    for folder in (low, t / FAKE_LOCAL_LOW):
        subprocess.run(["icacls", str(folder), "/setintegritylevel", "(OI)(CI)low"], check=True, capture_output=True)
    res, out = _run(t, f'echo w> "{low}\\w.txt" & echo w> "{t / FAKE_LOCAL_LOW}\\w.txt"', _spec(t, shape))
    _ran(res)
    assert (low / "w.txt").exists(), f"limit (e) changed under {shape.id}: an already-Low folder is no longer writable"
    assert (t / FAKE_LOCAL_LOW / "w.txt").exists() is (not shape.writes_confined), f"LocalLow under {shape.id}: {out[-300:]}"


# =====================================================================================
# The machine views, drives without permissions, the launch lock, a virtual environment
# =====================================================================================


def test_every_windows_view_of_the_machine_is_known():
    # A new function of the sandbox that looks at this machine must be classified (and, if it
    # names a real place, faked) before these tests may launch anything.
    names = {n for n in dir(sb) if callable(getattr(sb, n)) and ("windows" in n.lower()) and not n.isupper()}
    names |= {n for n in ("_may_change_permissions",) if hasattr(sb, n)}
    unknown = sorted(names - set(KNOWN_WINDOWS_FUNCTIONS))
    assert not unknown, f"unclassified Windows functions in abstractcore.tools.sandbox: {unknown}"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_drive_without_permissions_is_named_and_commands_still_run(t, shape, monkeypatch):
    # A volume that keeps no access lists (exFAT, a network share) cannot be protected: the
    # commands of a confined run still run, and the result names the drive as unprotected.
    drive, why = _fake_unprotected_drive(t, monkeypatch)
    res, out = _run(t, f'type "{drive}\\docs\\f.txt" & echo ok', _spec(t, shape))
    _ran(res)
    assert res["return_code"] == 0 and "ok" in out and "FAT-OK" in out, out[-300:]
    if shape.writes_confined:
        named = res["sandbox"].get("unprotected") or []
        assert any(_same(d, drive) for d in named) and why in json.dumps(res["sandbox"]), res["sandbox"]
        assert why in res["rendered"], res["rendered"][-400:]


@pytest.mark.parametrize("mode", ["refused", "ro"])
@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_refusal_on_a_drive_without_permissions_stops_the_command(t, shape, mode, monkeypatch):
    # A refused or read-only row there cannot hold, so the command does not run (126).
    drive, _why = _fake_unprotected_drive(t, monkeypatch)
    row = drive / "docs"
    spec = _spec(t, shape, refused=(str(t / "home/refused"), str(row))) if mode == "refused" else _spec(
        t, shape, allowed=(SandboxRow(str(t / "home/ro"), "ro"), SandboxRow(str(t / "home/rw"), "rw"), SandboxRow(str(row), "ro")))
    res, out = _run(t, f'type "{row}\\f.txt" & echo ran> ran.txt', spec)
    assert res.get("return_code") == 126 and "the command did not run" in out, f"{mode} row on an unprotected drive under {shape.id}: {out[-400:]}"
    assert "FAT-OK" not in out and not (_priv(t) / "ran.txt").exists()
    # Control: without that row, the same command runs.
    res, out = _run(t, f'type "{row}\\f.txt" & echo ran> ran.txt', _spec(t, shape))
    _ran(res)
    assert "FAT-OK" in out and (_priv(t) / "ran.txt").exists()


ATTACK = r"""
import os, sys, time
state, me = sys.argv[1], sys.argv[2]
kept, results = [], []
def attempt(name, fn):
    try:
        fn()
        results.append(name + ":DONE")
    except OSError:
        results.append(name + ":refused")
lock, listed = os.path.join(state, "lock"), os.path.join(state, "labelled")
attempt("open-lock-write", lambda: kept.append(open(lock, "a")))
attempt("open-lock-read", lambda: kept.append(open(lock, "rb")))
attempt("open-lock-rdwr", lambda: kept.append(os.open(lock, os.O_RDWR)))
attempt("delete-lock", lambda: os.remove(lock))
attempt("write-labelled", lambda: open(listed, "a").write("C:\\\\x\n"))
attempt("delete-labelled", lambda: os.remove(listed))
attempt("new-file", lambda: open(os.path.join(state, "x"), "w").close())
attempt("mkdir", lambda: os.mkdir(os.path.join(state, "d")))
attempt("list", lambda: os.listdir(state))
attempt("rename-state", lambda: os.rename(state, state + ".moved"))
print(" ".join(results), flush=True)
open(os.path.join(me, "attack.ready"), "w").close()
end = time.time() + 60
while not os.path.exists(os.path.join(me, "attack.go")) and time.time() < end:
    time.sleep(0.05)
"""


def _state_snapshot(state: Path) -> tuple:
    """(the names in the state folder, the lines of its list of labelled folders)."""
    listed = state / "labelled"
    lines = listed.read_text(encoding="utf-8", errors="replace").splitlines() if listed.exists() else []
    return sorted(p.name for p in state.iterdir()), [line for line in lines if ":\\" in line]  # the folders (not its header)


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_command_cannot_take_or_damage_the_launch_lock(t, shape):
    # The launcher's state folder (its lock, the folders it labelled) is refused to every
    # command: a command that tries to open, hold, delete, rename or fill it changes nothing,
    # and while it is still running another run's launch starts promptly.
    state = Path(sb._windows_state_folder())
    _ran(_run(t, "echo first", _spec(t, shape))[0])
    assert state.is_dir() and (state / "lock").exists(), list(state.parent.iterdir()) if state.parent.exists() else "no state folder"
    before = _state_snapshot(state)
    me = _priv(t)
    (me / "attack.py").write_text(ATTACK)
    argv, env = _box(_spec(t, shape)).wrap([PY, str(me / "attack.py"), str(state), str(me)], str(me))
    proc = subprocess.Popen(argv, cwd=me, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline and not (me / "attack.ready").exists() and proc.poll() is None:
            time.sleep(0.05)
        assert (me / "attack.ready").exists(), f"the attacking command never got going: {proc.stdout.read()[-600:] if proc.poll() is not None else ''}"
        started = time.time()
        res, out = _run(t, "echo second-run-ok", _spec2(t, shape), timeout=120)
        took = time.time() - started
        _ran(res, "the other run's launch")
        assert "second-run-ok" in out and took < 30, f"the other run's launch took {took:.1f}s: {out[-300:]}"
    finally:
        (me / "attack.go").write_text("go")
        attack_out, _ = proc.communicate(timeout=60)
    done = [w for w in attack_out.split() if w.endswith(":DONE")]
    assert not done, f"under {shape.id} a command could: {done}\n{attack_out[-600:]}"
    assert "open-lock-write:refused" in attack_out, attack_out[-600:]  # it did run and try
    names, lines = _state_snapshot(state)
    assert state.is_dir() and (state / "lock").exists() and names == before[0], (before[0], names)
    # The other run may have added its own folders to the list; nothing was taken out or forged.
    assert all(line in lines for line in before[1]) and not any("C:\\\\x" in line or line == "C:\\x" for line in lines), lines
    res, out = _run(t, "echo third-run-ok", _spec(t, shape))
    _ran(res)
    assert "third-run-ok" in out


def _probe_wheel(folder: Path) -> Path:
    """A minimal pure-Python wheel built here (pip installs it with no index and no network)."""
    import base64
    import hashlib
    import zipfile

    files = {
        "contractprobe/__init__.py": b"VALUE = 'VENV-OK'\n",
        "contractprobe-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: contractprobe\nVersion: 1.0\n",
        "contractprobe-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: contract-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = "".join(f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}\n" for name, data in files.items())
    record += "contractprobe-1.0.dist-info/RECORD,,\n"
    folder.mkdir(parents=True, exist_ok=True)
    wheel = folder / "contractprobe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("contractprobe-1.0.dist-info/RECORD", record)
    return wheel


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_a_virtual_environment_with_pip_works_in_a_read_and_write_row(t, shape):
    # python -m venv (with its own pip, from ensurepip) in a read & write row, then pip
    # installs a local wheel offline and the environment imports it.
    rw = t / "home/rw"
    wheel = _probe_wheel(rw / "wheels")
    venv = rw / "env"
    vpy = venv / "Scripts/python.exe"
    command = (f'"{PY}" -m venv "{venv}" && "{vpy}" -m pip --version && '
               f'"{vpy}" -m pip install --no-index --no-cache-dir --disable-pip-version-check "{wheel}" && '
               f'"{vpy}" -c "import contractprobe;print(contractprobe.VALUE)"')  # fmt: skip
    res, out = _run(t, command, _spec(t, shape), timeout=300)
    _ran(res)
    assert res["return_code"] == 0 and "VENV-OK" in out, f"venv + pip under {shape.id}: exit {res['return_code']}\n{out[-1500:]}"
