"""The Windows launcher of the command sandbox (see ``sandbox.py``; Windows only).

``sandbox.py`` never imports this file: it runs it, as the prefix of every sandboxed
command, with the host's own interpreter and nothing but the standard library::

    python -I sandbox_windows.py <policy> -- <program> [arguments...]
    python -I sandbox_windows.py <policy> --shell <command line>

(``<policy>``: ``sandbox.windows_policy``, the rules as compressed JSON.)

The launcher starts the command with a RESTRICTED TOKEN inside a kill-on-close job and
exits with the command's exit code. Anything it cannot set up exits 126 with one line on
stderr and the command does not run (fail closed).

The token. ``CreateRestrictedToken`` with, as restricting SIDs, a mirror of the caller's
own identity (the user and every enabled group, so the second access check Windows makes
for a restricted token grants what the first one does) plus one synthetic SID per rule;
every privilege is removed, including "bypass traverse checking", so a folder the token
may not traverse hides everything below it, whatever the files' own permissions say.

The rules (``policy["rules"]``, computed by ``sandbox.windows_rules``) are entries in the
access lists of the rule's folder for the rule's SID, ``S-1-9-<hash of action and path>``.
The SID depends only on the action and the path, so an entry is written once, stays, and
means nothing to any process that does not hold that SID:

- ``deny``   the folder itself refuses everything, traversal included: nothing below it
             can be reached by path, and nothing is written on its content;
- ``carve``  a refused folder with an allowed folder inside: it can be traversed but not
             listed, and each thing directly inside it is refused (a folder: closed like a
             ``deny`` one), now and, by inheritance, when it is created later;
- ``pass``   a folder between a carved folder and the allowed folder inside it: carved the
             same way, for the same SID;
- ``ro``     every kind of write is refused, inherited (no grant: reading is the user's);
- ``rw`` / ``read``  an allowed folder inside a refused one: modify / read, inherited.

Windows evaluates an object's own entries before the ones it inherits, and the nearest
ancestor's first, which is the nesting rule of the other backends: the most specific row
wins. An inherited refusal therefore loses to a grant written on the object itself: where
a refusal has to hold for existing content (``carve``: what is directly inside; ``ro``:
everything below), each object is checked the first time the rule is written, and the
refusal is written on the object itself where the inherited one would not be reached.
The rule's entries on its own path are written last: they are what says it is in place.
A carved folder is looked at again at every launch: what appeared directly inside it since
its last check (a folder made with permissions of its own, as a secrets folder is) is
refused before the command starts. The time of that check is kept in a named stream of the
folder itself (``<folder>:abstractcore-sandbox-<id>``), which the command cannot write.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zlib
from ctypes import wintypes as W
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

if sys.platform != "win32":  # pragma: no cover
    raise ImportError("abstractcore.tools.sandbox_windows is the Windows launcher")

EXIT_REFUSED = 126
EXIT_NOT_FOUND = 127

# --------------------------------------------------------------------------------------
# Win32
# --------------------------------------------------------------------------------------

_adv = ctypes.WinDLL("advapi32", use_last_error=True)
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_P = ctypes.POINTER
_VOID = ctypes.c_void_p


def _api(dll, name, restype, *argtypes):
    fn = getattr(dll, name)
    fn.restype, fn.argtypes = restype, list(argtypes)
    return fn


class SandboxLaunchError(RuntimeError):
    """The sandbox could not be set up; the command must not run."""


def _error(what: str, code: Optional[int] = None) -> SandboxLaunchError:
    code = ctypes.get_last_error() if code is None else code
    return SandboxLaunchError(f"{what}: {ctypes.FormatError(code).strip()} (error {code})")


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", _VOID), ("Attributes", W.DWORD)]


class _LuidAndAttributes(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("Luid", ctypes.c_longlong), ("Attributes", W.DWORD)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR), ("lpTitle", W.LPWSTR),
        ("dwX", W.DWORD), ("dwY", W.DWORD), ("dwXSize", W.DWORD), ("dwYSize", W.DWORD),
        ("dwXCountChars", W.DWORD), ("dwYCountChars", W.DWORD), ("dwFillAttribute", W.DWORD),
        ("dwFlags", W.DWORD), ("wShowWindow", W.WORD), ("cbReserved2", W.WORD), ("lpReserved2", _VOID),
        ("hStdInput", W.HANDLE), ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE),
    ]  # fmt: skip


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", _VOID)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE), ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD)]


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", W.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD),
    ]  # fmt: skip


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimits), ("IoInfo", ctypes.c_ulonglong * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]  # fmt: skip


_TreeProgress = ctypes.WINFUNCTYPE(None, W.LPWSTR, W.DWORD, _P(ctypes.c_int), _VOID, W.BOOL)

CloseHandle = _api(_k32, "CloseHandle", W.BOOL, W.HANDLE)
LocalFree = _api(_k32, "LocalFree", _VOID, _VOID)
GetCurrentProcess = _api(_k32, "GetCurrentProcess", W.HANDLE)
GetStdHandle = _api(_k32, "GetStdHandle", W.HANDLE, W.DWORD)
SetHandleInformation = _api(_k32, "SetHandleInformation", W.BOOL, W.HANDLE, W.DWORD, W.DWORD)
GetConsoleCP = _api(_k32, "GetConsoleCP", W.UINT)
CreateMutexW = _api(_k32, "CreateMutexW", W.HANDLE, _VOID, W.BOOL, W.LPCWSTR)
ReleaseMutex = _api(_k32, "ReleaseMutex", W.BOOL, W.HANDLE)
WaitForSingleObject = _api(_k32, "WaitForSingleObject", W.DWORD, W.HANDLE, W.DWORD)
CreateJobObjectW = _api(_k32, "CreateJobObjectW", W.HANDLE, _VOID, W.LPCWSTR)
SetInformationJobObject = _api(_k32, "SetInformationJobObject", W.BOOL, W.HANDLE, ctypes.c_int, _VOID, W.DWORD)
AssignProcessToJobObject = _api(_k32, "AssignProcessToJobObject", W.BOOL, W.HANDLE, W.HANDLE)
TerminateProcess = _api(_k32, "TerminateProcess", W.BOOL, W.HANDLE, W.UINT)
ResumeThread = _api(_k32, "ResumeThread", W.DWORD, W.HANDLE)
GetExitCodeProcess = _api(_k32, "GetExitCodeProcess", W.BOOL, W.HANDLE, _P(W.DWORD))
InitializeProcThreadAttributeList = _api(_k32, "InitializeProcThreadAttributeList", W.BOOL, _VOID, W.DWORD, W.DWORD, _P(ctypes.c_size_t))
UpdateProcThreadAttribute = _api(_k32, "UpdateProcThreadAttribute", W.BOOL, _VOID, W.DWORD, ctypes.c_size_t, _VOID, ctypes.c_size_t, _VOID, _VOID)
DeleteProcThreadAttributeList = _api(_k32, "DeleteProcThreadAttributeList", None, _VOID)

OpenProcessToken = _api(_adv, "OpenProcessToken", W.BOOL, W.HANDLE, W.DWORD, _P(W.HANDLE))
GetTokenInformation = _api(_adv, "GetTokenInformation", W.BOOL, W.HANDLE, ctypes.c_int, _VOID, W.DWORD, _P(W.DWORD))
CreateRestrictedToken = _api(_adv, "CreateRestrictedToken", W.BOOL, W.HANDLE, W.DWORD, W.DWORD, _VOID, W.DWORD, _VOID, W.DWORD, _VOID, _P(W.HANDLE))
CreateProcessAsUserW = _api(_adv, "CreateProcessAsUserW", W.BOOL, W.HANDLE, W.LPCWSTR, W.LPWSTR, _VOID, _VOID, W.BOOL, W.DWORD, _VOID, W.LPCWSTR, _VOID, _P(_ProcessInformation))
ConvertStringSidToSidW = _api(_adv, "ConvertStringSidToSidW", W.BOOL, W.LPCWSTR, _P(_VOID))
ConvertSidToStringSidW = _api(_adv, "ConvertSidToStringSidW", W.BOOL, _VOID, _P(W.LPWSTR))
GetLengthSid = _api(_adv, "GetLengthSid", W.DWORD, _VOID)
GetFileSecurityW = _api(_adv, "GetFileSecurityW", W.BOOL, W.LPCWSTR, W.DWORD, _VOID, W.DWORD, _P(W.DWORD))
GetSecurityDescriptorDacl = _api(_adv, "GetSecurityDescriptorDacl", W.BOOL, _VOID, _P(W.BOOL), _P(_VOID), _P(W.BOOL))
GetSecurityDescriptorControl = _api(_adv, "GetSecurityDescriptorControl", W.BOOL, _VOID, _P(W.WORD), _P(W.DWORD))
InitializeSecurityDescriptor = _api(_adv, "InitializeSecurityDescriptor", W.BOOL, _VOID, W.DWORD)
SetSecurityDescriptorDacl = _api(_adv, "SetSecurityDescriptorDacl", W.BOOL, _VOID, W.BOOL, _VOID, W.BOOL)
SetSecurityDescriptorControl = _api(_adv, "SetSecurityDescriptorControl", W.BOOL, _VOID, W.WORD, W.WORD)
SetFileSecurityW = _api(_adv, "SetFileSecurityW", W.BOOL, W.LPCWSTR, W.DWORD, _VOID)
TreeSetNamedSecurityInfoW = _api(_adv, "TreeSetNamedSecurityInfoW", W.DWORD, W.LPWSTR, ctypes.c_int, W.DWORD, _VOID, _VOID, _VOID, _VOID, W.DWORD, _TreeProgress, ctypes.c_int, _VOID)
InitializeAcl = _api(_adv, "InitializeAcl", W.BOOL, _VOID, W.DWORD, W.DWORD)
GetAce = _api(_adv, "GetAce", W.BOOL, _VOID, W.DWORD, _P(_VOID))
AddAce = _api(_adv, "AddAce", W.BOOL, _VOID, W.DWORD, W.DWORD, _VOID, W.DWORD)
AddAccessAllowedAceEx = _api(_adv, "AddAccessAllowedAceEx", W.BOOL, _VOID, W.DWORD, W.DWORD, W.DWORD, _VOID)
AddAccessDeniedAceEx = _api(_adv, "AddAccessDeniedAceEx", W.BOOL, _VOID, W.DWORD, W.DWORD, W.DWORD, _VOID)

INVALID_HANDLE = W.HANDLE(-1).value
INFINITE = 0xFFFFFFFF
TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_QUERY = 0x1, 0x2, 0x8
TOKEN_USER, TOKEN_GROUPS, TOKEN_PRIVILEGES = 1, 2, 3
SE_GROUP_ENABLED, SE_GROUP_USE_FOR_DENY_ONLY, SE_GROUP_INTEGRITY = 0x4, 0x10, 0x20
SE_FILE_OBJECT, ACL_REVISION = 1, 2
DACL_INFORMATION, PROTECTED_DACL, UNPROTECTED_DACL = 0x4, 0x80000000, 0x20000000
SE_DACL_AUTO_INHERIT_REQ, SE_DACL_AUTO_INHERITED, SE_DACL_PROTECTED = 0x0100, 0x0400, 0x1000
ALLOW, DENY = 0, 1  # ACE types
OBJECT_INHERIT, CONTAINER_INHERIT, INHERIT_ONLY, INHERITED = 0x1, 0x2, 0x8, 0x10
CHILDREN = OBJECT_INHERIT | CONTAINER_INHERIT

# File access masks.
FILE_ALL = 0x1F01FF
FILE_READ_EXECUTE = 0x1200A9
FILE_MODIFY = 0x1301BF
FILE_DELETE_CHILD = 0x40
# Every kind of write: data, append, extended attributes, delete child, attributes, DELETE,
# WRITE_DAC, WRITE_OWNER.
FILE_WRITES = 0x2 | 0x4 | 0x10 | 0x40 | 0x100 | 0x10000 | 0x40000 | 0x80000
# What a path lookup needs of an ancestor: traverse, read attributes, synchronize.
FILE_PASS = 0x20 | 0x80 | 0x100000

RULE_AUTHORITY = "S-1-9"  # resource-manager authority: never an account, never looked up
# The owner guard. An owner may always rewrite an object's access list, whatever the list
# says, unless the list has an entry for OWNER RIGHTS: its presence alone ends that implicit
# right, and the owner then has what the list grants. The sandboxed command runs as the
# owner, so every refusing rule adds one, and with it the rule's refusal of "change
# permissions" holds. The guard denies the owner nothing (its mask is a right no access list
# grants). Where the owner's control of the object WAS the implicit right alone (no entry
# grants them "change permissions"), the guard comes with a grant of exactly that right to
# the user by name, so no one is locked out of their own folder. No grant is ever made to
# OWNER RIGHTS or to a rule's SID on a refused or read-only object: OpenSSH refuses a private
# key whose access list grants anything to anyone but its owner.
OWNER_RIGHTS = "S-1-3-4"
READ_CONTROL, WRITE_DAC = 0x20000, 0x40000
OWNER_GUARD_MASK = 0x01000000  # ACCESS_SYSTEM_SECURITY: granted by privilege only, never by a list

Entry = Tuple[int, int]  # (ACE type, mask)
Template = Tuple[int, int, int]  # (ACE type, mask, inheritance flags)
# action -> (entries on the rule's own path, entries its content inherits)
RULES: Dict[str, Tuple[List[Entry], List[Entry]]] = {
    "deny": ([(DENY, FILE_ALL)], []),
    "carve": ([(DENY, FILE_ALL & ~FILE_PASS)], [(DENY, FILE_ALL)]),
    "ro": ([(DENY, FILE_WRITES)], [(DENY, FILE_WRITES)]),
    "rw": ([(ALLOW, FILE_MODIFY)], [(ALLOW, FILE_MODIFY)]),
    "read": ([(ALLOW, FILE_READ_EXECUTE)], [(ALLOW, FILE_READ_EXECUTE)]),
}
REFUSING = ("deny", "carve", "ro")


def rule_sid(action: str, path: str) -> str:
    """The SID of rule (action, path): the same for every run, different for every rule."""
    digest = hashlib.sha256(f"abstractcore-sandbox\0{action}\0{os.path.normcase(path)}".encode("utf-8")).digest()
    return RULE_AUTHORITY + "".join(f"-{int.from_bytes(digest[i:i + 4], 'little')}" for i in range(0, 16, 4))


class _Sid:
    """A binary SID owned by this object (from its string form)."""

    def __init__(self, text: str):
        raw = _VOID()
        if not ConvertStringSidToSidW(text, ctypes.byref(raw)):
            raise _error(f"invalid SID {text}")
        try:
            self._buf = ctypes.create_string_buffer(ctypes.string_at(raw, GetLengthSid(raw)))
        finally:
            LocalFree(raw)
        self.ptr = ctypes.addressof(self._buf)
        self.size = len(self._buf) - 1


def _sid_text(ptr: int) -> str:
    out = W.LPWSTR()
    if not ConvertSidToStringSidW(ptr, ctypes.byref(out)):
        raise _error("unreadable SID")
    try:
        return out.value or ""
    finally:
        LocalFree(ctypes.cast(out, _VOID))


# --------------------------------------------------------------------------------------
# Access lists
# --------------------------------------------------------------------------------------


class _Ace(NamedTuple):
    type: int
    flags: int
    mask: int
    sid: str  # "" for an entry type that carries no plain SID
    raw: bytes


def _read_dacl(path: str) -> Tuple[List[_Ace], int]:
    """(entries in order, security-descriptor control bits) of ``path``'s access list, as it is
    stored (GetNamedSecurityInfo would GUESS which entries are inherited on an object that
    does not say, by comparing with its folder as it is now)."""
    size = W.DWORD()
    GetFileSecurityW(path, DACL_INFORMATION, None, 0, ctypes.byref(size))
    sd = ctypes.create_string_buffer(size.value)
    present, defaulted, dacl = W.BOOL(), W.BOOL(), _VOID()
    control, revision = W.WORD(), W.DWORD()
    if not (
        size.value
        and GetFileSecurityW(path, DACL_INFORMATION, sd, size, ctypes.byref(size))
        and GetSecurityDescriptorDacl(sd, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted))
        and GetSecurityDescriptorControl(sd, ctypes.byref(control), ctypes.byref(revision))
    ):
        raise _error(f"cannot read the permissions of {path}")
    if not present.value or not dacl.value:
        raise SandboxLaunchError(f"{path} has no access list (everyone has full access to it)")
    aces: List[_Ace] = []
    for index in range(ctypes.c_ushort.from_address(dacl.value + 4).value):  # ACL.AceCount
        ace = _VOID()
        if not GetAce(dacl, index, ctypes.byref(ace)):
            raise _error(f"cannot read the permissions of {path}")
        kind, flags = ctypes.c_ubyte.from_address(ace.value).value, ctypes.c_ubyte.from_address(ace.value + 1).value
        size_of_ace = ctypes.c_ushort.from_address(ace.value + 2).value
        mask = W.DWORD.from_address(ace.value + 4).value
        aces.append(_Ace(kind, flags, mask, _sid_text(ace.value + 8) if kind in (ALLOW, DENY) else "", ctypes.string_at(ace.value, size_of_ace)))
    return aces, control.value


def _ours(ace: _Ace) -> bool:
    """A rule's entry or an owner guard (never a grant of access to content)."""
    return ace.sid.startswith(RULE_AUTHORITY + "-") or (ace.sid == OWNER_RIGHTS and ace.type == DENY)


def _guard(aces: Sequence[_Ace], flags: int) -> List[Tuple[str, Template]]:
    """The owner guard for an object with ``aces`` (see OWNER_RIGHTS above)."""
    user, mine = identity_sids()
    keeps_control = any(a.type == ALLOW and a.mask & WRITE_DAC and not a.flags & INHERIT_ONLY and a.sid in mine for a in aces)
    guard = [(OWNER_RIGHTS, (DENY, OWNER_GUARD_MASK, flags))]
    return guard if keeps_control else guard + [(user, (ALLOW, READ_CONTROL | WRITE_DAC, flags))]


def _pending(aces: Sequence[_Ace], sid: str, entries: Sequence[Entry], flags: int, guard: bool) -> List[Tuple[str, Template]]:
    """What ``sid`` still lacks of ``entries`` with ``flags`` among an object's own entries
    (plus the owner guard, when asked)."""
    own = [a for a in aces if not a.flags & INHERITED]
    have = {(a.type, a.mask, a.flags & (CHILDREN | INHERIT_ONLY)) for a in own if a.sid == sid}
    todo = [(sid, (kind, mask, flags)) for kind, mask in entries if (kind, mask, flags) not in have]
    if guard and not any(_ours(a) and a.sid == OWNER_RIGHTS and a.flags & (CHILDREN | INHERIT_ONLY) == flags for a in own):
        todo += _guard(aces, flags)
    return todo


def _in_force(aces: Sequence[_Ace], sid: str) -> bool:
    """``sid``'s refusal is reached before any grant that is not a rule's, and an entry for
    OWNER RIGHTS is there."""
    guarded = any(a.sid == OWNER_RIGHTS for a in aces)
    for ace in aces:
        if ace.sid == sid and ace.type == DENY:
            return guarded
        if ace.type != DENY and not _ours(ace):
            return False
    return False


def _read_rule(path: str, sid: str, refusing: bool) -> Tuple[List[_Ace], int]:
    """``path``'s access list as a rule of ``sid`` starts from. A refusing rule whose entries
    are there but not in force (written by something else, after a grant) starts over: its
    entries are dropped from what is read, so they are written again where they refuse."""
    aces, control = _read_dacl(path)
    if refusing and not _in_force(aces, sid):
        aces = [a for a in aces if a.flags & INHERITED or a.sid != sid]
    return aces, control


def _build_acl(aces: Sequence[_Ace], added: Sequence[Tuple[str, Template]]) -> ctypes.Array:
    """``aces`` plus ``added`` in the order Windows expects: the object's own refusals, its
    own grants, then what it inherits (an added entry flagged INHERITED: first of those, as
    its parent's entries are)."""
    sids = {sid: _Sid(sid) for sid, _ in added}
    size = 8 + sum(len(a.raw) for a in aces) + sum(8 + sids[sid].size for sid, _ in added)
    if size > 0xFFFF:
        raise SandboxLaunchError("the access list would be too large")
    acl = ctypes.create_string_buffer(size)
    if not InitializeAcl(acl, size, ACL_REVISION):
        raise _error("InitializeAcl")

    def add(inherited: bool, kind: int) -> bool:
        adder = AddAccessDeniedAceEx if kind == DENY else AddAccessAllowedAceEx
        return all(adder(acl, ACL_REVISION, f, mask, sids[sid].ptr) for sid, (k, mask, f) in added if k == kind and bool(f & INHERITED) == inherited)

    def keep(inherited: bool) -> bool:
        return all(AddAce(acl, ACL_REVISION, 0xFFFFFFFF, a.raw, len(a.raw)) for a in aces if bool(a.flags & INHERITED) == inherited)

    if not (add(False, DENY) and keep(False) and add(False, ALLOW) and add(True, DENY) and add(True, ALLOW) and keep(True)):
        raise _error("cannot build the access list")
    return acl


_NOT_OURS = "you must own it or be allowed to change its permissions"


def _write_here(path: str, aces: Sequence[_Ace], control: int, added: Sequence[Tuple[str, Template]]) -> None:
    """Write ``aces`` plus ``added`` as ``path``'s own access list; nothing below it is
    touched, and what it inherits stays as it is."""
    acl = _build_acl(aces, added)
    sd = ctypes.create_string_buffer(64)
    keep = control & SE_DACL_PROTECTED
    if control & SE_DACL_AUTO_INHERITED or any(a.flags & INHERITED for a in aces) or any(t[2] & INHERITED for _, t in added):
        # It holds inherited entries: say so, or Windows turns them into entries of its own.
        keep |= SE_DACL_AUTO_INHERITED | SE_DACL_AUTO_INHERIT_REQ
    if not (
        InitializeSecurityDescriptor(sd, 1)
        and SetSecurityDescriptorDacl(sd, True, acl, False)
        and SetSecurityDescriptorControl(sd, SE_DACL_AUTO_INHERIT_REQ | SE_DACL_AUTO_INHERITED | SE_DACL_PROTECTED, keep)
        and SetFileSecurityW(path, DACL_INFORMATION, sd)
    ):
        raise _error(f"cannot change the permissions of {path} ({_NOT_OURS})")


def _is_link(entry: os.DirEntry) -> bool:
    return entry.is_symlink() or bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x400)  # reparse point


def _handed_down(aces: Sequence[_Ace], to_folder: bool) -> set:
    """(type, SID, mask) of the entries a folder hands down to a folder / a file inside it."""
    bit = CONTAINER_INHERIT if to_folder else OBJECT_INHERIT
    return {(a.type, a.sid, a.mask) for a in aces if a.flags & bit and not _ours(a)}


def _copied(aces: Sequence[_Ace], control: int, handed: set) -> List[_Ace]:
    """The entries of its own that an object merely copies from its folder (``handed``), when
    that is all it has of its own: inherited entries in all but the mark (a folder made by a
    program that sets permissions itself, as Python's ``tempfile`` does). Else none."""
    theirs = [a for a in aces if not a.flags & INHERITED and not _ours(a)]
    return theirs if not control & SE_DACL_PROTECTED and all((a.type, a.sid, a.mask) in handed for a in theirs) else []


def _hand_down(folder: str) -> None:
    """Let Windows recompute, from ``folder``'s access list as it now is, what everything
    below it inherits. An object it cannot update (in use, not ours) is left as it is: what
    must hold there is checked by the caller. An object that does not inherit is not touched.
    (This is the recomputation any edit of a folder's permissions triggers: content that was
    moved here and still carried what it inherited elsewhere takes this folder's entries.)"""
    quiet = _TreeProgress(lambda _name, _status, invoke, _args, _set: invoke.__setitem__(0, 3))  # keep going
    handed = {flag: _handed_down(_read_dacl(folder)[0], flag) for flag in (True, False)}
    with os.scandir(folder) as entries:
        for entry in entries:
            if _is_link(entry):
                continue
            try:
                aces, control = _read_dacl(entry.path)
            except SandboxLaunchError:
                continue
            if not control & SE_DACL_PROTECTED:
                copied = _copied(aces, control, handed[entry.is_dir(follow_symlinks=False)])
                acl = _build_acl([a for a in aces if not a.flags & INHERITED and a not in copied], ())
                TreeSetNamedSecurityInfoW(entry.path, SE_FILE_OBJECT, DACL_INFORMATION | UNPROTECTED_DACL, None, None, acl, None, 1, quiet, 3, None)


def _readable(path: str) -> bool:
    try:
        if os.path.isdir(path):
            with os.scandir(path):
                return True
        with open(path, "rb"):
            return True
    except OSError:
        return False


def _refuse(path: str, is_dir: bool, sid: str, entries: Sequence[Entry], handed: Optional[set] = None, a_row: bool = False) -> None:
    """Make ``sid``'s refusal the first thing reached on ``path``, an object below a rule.

    ``handed`` (what its folder hands down): write the refusal as the entry the object
    INHERITS, so that a row on this very object overrides it (for the run that has the row;
    every other run is refused). That only refuses when the object has no grant of its own
    before it: entries it merely copies from its folder are marked as the inherited entries
    they are. With any other entry of its own, or when ``handed`` is None, the refusal is
    written as the object's own entry, unless it is ``a_row`` of this policy: an entry of its
    own would refuse the run it is allowed to."""
    try:
        aces, control = _read_dacl(path)
        if any(a.sid == sid and a.type == DENY for a in aces) if a_row else _in_force(aces, sid):
            return
        inherited = 0
        if handed is not None and not control & SE_DACL_PROTECTED:
            copied = _copied(aces, control, handed)
            if all(a.flags & INHERITED or _ours(a) or a in copied for a in aces):
                inherited = INHERITED
                aces = [a._replace(flags=a.flags | INHERITED, raw=a.raw[:1] + bytes([a.raw[1] | INHERITED]) + a.raw[2:]) if a in copied else a for a in aces]
        if a_row and not inherited:
            return
        flags = (CHILDREN if is_dir else 0) | inherited
        _write_here(path, aces, control, [(sid, (kind, mask, flags)) for kind, mask in entries] + _guard(aces, flags))
    except SandboxLaunchError:
        if _readable(path):
            raise  # this user can open it, and its refusal cannot be written


def _skip(paths) -> set:
    return {os.path.normcase(p) for p in paths}


def _checked_stream(folder: str, sid: str) -> str:
    """Where a carved folder keeps the time of its last check: a named stream of the folder
    itself, which a command refused the folder cannot write."""
    return f"{folder}:abstractcore-sandbox-{sid.rsplit('-', 1)[1]}"


def _born(entry: os.DirEntry) -> int:
    info = entry.stat(follow_symlinks=False)
    return getattr(info, "st_birthtime_ns", None) or info.st_ctime_ns  # creation time on Windows


def _carve(path: str, sid: str, rows: Sequence[str], passes: Sequence[str], passable: bool) -> bool:
    """A refused folder that an allowed row lies in: traversable, never listed, and each
    thing directly inside it refused (a folder inside: traversal included, so nothing below
    it needs an entry). ``passable``: a folder on the way to the row, not the refused one.

    What is created inside it later inherits the refusal, unless it is made with permissions
    of its own: so every launch looks at what appeared since the last check (False then: the
    rule itself was in place)."""
    own, below = RULES["carve"]
    if passable:
        own = own + [(ALLOW, FILE_PASS)]  # its parent's refusal is inherited here: let traversal through
    aces, control = _read_rule(path, sid, True)
    in_place = not _pending(aces, sid, own, 0, True)
    since = 0
    if in_place:
        try:
            with open(_checked_stream(path, sid), encoding="ascii") as stream:
                since = int(stream.read())
        except (OSError, ValueError):
            pass  # never checked, or a file system without streams: check everything
    else:
        first = _pending(aces, sid, below, CHILDREN | INHERIT_ONLY, True)  # what is created here later inherits it
        if first:
            _write_here(path, aces, control, first)
    started = time.time_ns()
    ruled = _skip(list(rows) + list(passes))  # the policy's own rows: their rules follow
    seen_new = False
    with os.scandir(path) as entries:
        for entry in entries:
            if _born(entry) >= since and not _is_link(entry):
                seen_new = True
                folder = entry.is_dir(follow_symlinks=False)
                _refuse(entry.path, folder, sid, below, _handed_down(aces, folder), a_row=os.path.normcase(entry.path) in ruled)
    if not in_place:
        aces, control = _read_dacl(path)
        _write_here(path, aces, control, _pending(aces, sid, own, 0, True))  # last: the rule is in place
    if seen_new or not in_place:
        try:
            with open(_checked_stream(path, sid), "w", encoding="ascii") as stream:
                stream.write(str(started))
        except OSError:
            pass
    return not in_place


def _inherit(path: str, sid: str, action: str, rows: Sequence[str]) -> bool:
    """A row whose entries its content inherits (``ro``, ``rw``, ``read``)."""
    own, below = RULES[action]
    refusing = action in REFUSING
    aces, control = _read_rule(path, sid, refusing)
    if not _pending(aces, sid, own, 0, refusing):
        return False
    if os.path.isdir(path):
        first = _pending(aces, sid, below, CHILDREN | INHERIT_ONLY, refusing)
        if first:
            _write_here(path, aces, control, first)
        _hand_down(path)
        if refusing:
            # Everything below must refuse, whatever its own access list grants.
            skipped = _skip(p for p in rows if p != path)
            stack = [path]
            while stack:
                with os.scandir(stack.pop()) as entries:
                    for entry in entries:
                        if os.path.normcase(entry.path) in skipped or _is_link(entry):
                            continue
                        folder = entry.is_dir(follow_symlinks=False)
                        _refuse(entry.path, folder, sid, below)
                        if folder:
                            stack.append(entry.path)
        aces, control = _read_dacl(path)
    _write_here(path, aces, control, _pending(aces, sid, own, 0, refusing))  # last: the rule is in place
    return True


def ensure_rule(rule: Sequence[str], rows: Sequence[str] = (), passes: Sequence[str] = ()) -> bool:
    """Write the access-list entries of one rule; False when it was already in place (one
    read). ``rows`` and ``passes``: the paths of the policy's rules, by kind. A path that
    does not exist has no rule (as with the other backends, it is not created)."""
    action, path = rule[0], rule[1]
    if not os.path.lexists(path):
        return False
    if action == "pass":
        return _carve(path, rule_sid("carve", rule[2]), rows, passes, passable=True)
    sid = rule_sid(action, path)
    refusing = action in REFUSING
    try:
        if action == "carve" and os.path.isdir(path):
            wrote = _carve(path, sid, rows, passes, passable=False)
        elif action in ("deny", "carve"):
            aces, control = _read_rule(path, sid, True)
            todo = _pending(aces, sid, RULES["deny"][0], 0, True)
            if todo:
                _write_here(path, aces, control, todo)
            wrote = bool(todo)
        else:
            wrote = _inherit(path, sid, action, rows)
    except (SandboxLaunchError, OSError):
        if refusing and not _readable(path):
            return False  # nothing to refuse: this user cannot open it anyway
        raise
    if wrote and refusing:
        # The folder that holds a refused or read-only row may not delete or rename it. Its
        # access list is not always ours to change (a drive root): then it is not ours to
        # delete from either.
        parent = os.path.dirname(path)
        if parent and parent != path:
            try:
                aces, control = _read_dacl(parent)
                todo = _pending(aces, sid, [(DENY, FILE_DELETE_CHILD)], 0, False)
                if todo:
                    _write_here(parent, aces, control, todo)
            except SandboxLaunchError:
                pass
    return wrote


def ensure_rules(rules: Sequence[Sequence[str]]) -> None:
    """Write every missing entry, one launcher at a time (two runs often share a rule)."""
    rows = [r[1] for r in rules if r[0] != "pass"]
    passes = [r[1] for r in rules if r[0] == "pass"]
    mutex = CreateMutexW(None, False, "Local\\abstractcore-command-sandbox")
    if not mutex:
        raise _error("CreateMutex")
    try:
        WaitForSingleObject(mutex, INFINITE)
        for rule in rules:
            ensure_rule(rule, rows, passes)
    finally:
        ReleaseMutex(mutex)
        CloseHandle(mutex)


def rule_sids(rules: Sequence[Sequence[str]]) -> List[str]:
    return list(dict.fromkeys(rule_sid(r[0], r[1]) for r in rules if r[0] != "pass"))


# --------------------------------------------------------------------------------------
# Token
# --------------------------------------------------------------------------------------


def _token_information(token: int, kind: int) -> ctypes.Array:
    size = W.DWORD()
    GetTokenInformation(token, kind, None, 0, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not GetTokenInformation(token, kind, buf, size, ctypes.byref(size)):
        raise _error("GetTokenInformation")
    return buf


def _identity(token: int) -> Tuple[List[int], List[ctypes.Array]]:
    """Pointers to the SIDs ``token`` acts as (the user, then every enabled group), and the
    buffers that hold them."""
    user = _token_information(token, TOKEN_USER)
    groups = _token_information(token, TOKEN_GROUPS)
    group_list = (_SidAndAttributes * W.DWORD.from_buffer(groups).value).from_buffer(groups, ctypes.sizeof(_VOID))
    sids = [_SidAndAttributes.from_buffer(user).Sid] + [
        g.Sid for g in group_list
        if g.Attributes & SE_GROUP_ENABLED and not g.Attributes & (SE_GROUP_USE_FOR_DENY_ONLY | SE_GROUP_INTEGRITY)
    ]  # fmt: skip
    return sids, [user, groups]


def _own_token(access: int) -> int:
    token = W.HANDLE()
    if not OpenProcessToken(GetCurrentProcess(), access, ctypes.byref(token)):
        raise _error("OpenProcessToken")
    return token.value


_IDENTITY: Optional[Tuple[str, frozenset]] = None


def identity_sids() -> Tuple[str, frozenset]:
    """(the user's SID, every SID this process acts as), as text (read once)."""
    global _IDENTITY
    if _IDENTITY is None:
        token = _own_token(TOKEN_QUERY)
        try:
            pointers, _buffers = _identity(token)
            texts = [_sid_text(p) for p in pointers]
            _IDENTITY = (texts[0], frozenset(texts))
        finally:
            CloseHandle(token)
    return _IDENTITY


def restricted_token(sids: Sequence[str]) -> int:
    """A primary token for the command: this process's identity mirrored as restricting
    SIDs, plus ``sids``, with no privilege left."""
    own = _own_token(TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY)
    try:
        mirror, _buffers = _identity(own)
        privileges = _token_information(own, TOKEN_PRIVILEGES)
        extra = [_Sid(s) for s in sids]
        restricting = (_SidAndAttributes * (len(mirror) + len(extra)))()
        for slot, ptr in zip(restricting, mirror + [s.ptr for s in extra]):
            slot.Sid, slot.Attributes = ptr, 0
        count = W.DWORD.from_buffer(privileges).value
        to_delete = (_LuidAndAttributes * count).from_buffer(privileges, 4) if count else None
        token = W.HANDLE()
        if not CreateRestrictedToken(own, 0, 0, None, count, to_delete, len(restricting), restricting, ctypes.byref(token)):
            raise _error("CreateRestrictedToken")
        return token.value
    finally:
        CloseHandle(own)


# --------------------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------------------

STARTF_USESTDHANDLES = 0x100
HANDLE_FLAG_INHERIT = 0x1
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x20002
CREATE_SUSPENDED, CREATE_NO_WINDOW, EXTENDED_STARTUPINFO_PRESENT = 0x4, 0x08000000, 0x00080000
JOB_EXTENDED_LIMITS = 9
JOB_KILL_ON_CLOSE, JOB_DIE_ON_UNHANDLED_EXCEPTION = 0x2000, 0x400
ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND = 2, 3


def launch(command_line: str, token: int) -> int:
    """Run ``command_line`` under ``token`` with this process's standard handles (and no
    other), in a job that dies with this process; return its exit code."""
    std = [GetStdHandle(n) for n in (W.DWORD(-10).value, W.DWORD(-11).value, W.DWORD(-12).value)]
    inherit = [h for h in dict.fromkeys(std) if h and h != INVALID_HANDLE and SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT)]
    startup = _StartupInfoEx()
    startup.StartupInfo.cb = ctypes.sizeof(startup)
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
    startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = std
    size = ctypes.c_size_t()
    InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    attributes = ctypes.create_string_buffer(size.value)
    if not InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)):
        raise _error("InitializeProcThreadAttributeList")
    handles = (W.HANDLE * max(1, len(inherit)))(*inherit)
    job = None
    process = _ProcessInformation()
    try:
        if inherit and not UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, ctypes.sizeof(W.HANDLE) * len(inherit), None, None):
            raise _error("UpdateProcThreadAttribute")
        startup.lpAttributeList = ctypes.addressof(attributes)
        job = CreateJobObjectW(None, None)
        limits = _JobExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = JOB_KILL_ON_CLOSE | JOB_DIE_ON_UNHANDLED_EXCEPTION
        if not job or not SetInformationJobObject(job, JOB_EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise _error("cannot create the job of the command")
        flags = CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT
        if not GetConsoleCP():  # no console here: a console program must not open a window
            flags |= CREATE_NO_WINDOW
        line = ctypes.create_unicode_buffer(command_line)
        if not CreateProcessAsUserW(token, None, line, None, None, bool(inherit), flags, None, None, ctypes.byref(startup), ctypes.byref(process)):
            raise OSError(ctypes.get_last_error(), ctypes.FormatError(ctypes.get_last_error()).strip())
        if not AssignProcessToJobObject(job, process.hProcess):
            code = ctypes.get_last_error()
            TerminateProcess(process.hProcess, EXIT_REFUSED)
            raise _error("cannot place the command in its job", code)
        ResumeThread(process.hThread)
        WaitForSingleObject(process.hProcess, INFINITE)
        code = W.DWORD()
        GetExitCodeProcess(process.hProcess, ctypes.byref(code))
        return ctypes.c_int32(code.value).value
    finally:
        DeleteProcThreadAttributeList(attributes)
        for handle in (process.hThread, process.hProcess, job):
            if handle:
                CloseHandle(handle)  # closing the job ends whatever the command left running


def shell_command_line(command: str) -> str:
    """What ``subprocess.Popen(command, shell=True)`` runs on Windows."""
    comspec = os.environ.get("ComSpec") or os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
    return f'{comspec} /c "{command}"'


def read_policy(text: str) -> List[List[str]]:
    """The rules of a policy argument (``sandbox.windows_policy``). Raises on anything that is
    not a list of rules with absolute paths: the policy decides what the command may reach."""
    rules = json.loads(zlib.decompress(base64.urlsafe_b64decode(text)))["rules"]
    for rule in rules:
        well_formed = isinstance(rule, list) and all(isinstance(part, str) for part in rule) and len(rule) == (3 if rule[:1] == ["pass"] else 2)
        if not (well_formed and (rule[0] in RULES or rule[0] == "pass") and all(os.path.isabs(p) for p in rule[1:])):
            raise SandboxLaunchError(f"malformed policy rule {rule!r}")
    return rules


def main(argv: Sequence[str]) -> int:
    try:
        if len(argv) == 3 and argv[1] == "--shell":
            command_line = shell_command_line(argv[2])
        elif len(argv) >= 3 and argv[1] == "--":
            command_line = subprocess.list2cmdline(argv[2:])
        else:
            raise SandboxLaunchError("usage: sandbox_windows.py <policy> (-- <program> [arguments...] | --shell <command line>)")
        rules = read_policy(argv[0])
        ensure_rules(rules)
        token = restricted_token(rule_sids(rules))
    except Exception as exc:  # noqa: BLE001 - fail closed, whatever went wrong
        _say(f"sandbox: {exc}; the command did not run")
        return EXIT_REFUSED
    # Ctrl-C and Ctrl-Break reach the command too (same console): they are its to handle,
    # and this process ends when it does.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    try:
        return launch(command_line, token)
    except OSError as exc:
        _say(f"sandbox: cannot start the command: {exc.strerror}")
        return EXIT_NOT_FOUND if exc.errno in (ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND) else EXIT_REFUSED
    except SandboxLaunchError as exc:
        _say(f"sandbox: {exc}; the command did not run")
        return EXIT_REFUSED
    finally:
        CloseHandle(token)


def _say(line: str) -> None:
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - no stderr to write to
        pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
