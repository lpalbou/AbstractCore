"""OS-level sandbox for every process-spawning tool (round 12, R12.1).

The workspace policy used to bind only the FILE tools. Commands (``execute_command``,
the persistent shell session, local helpers, ...) ran with the host process's whole
filesystem and environment, so a run whose policy refused a folder could still
``ls`` it. This module turns a run's EFFECTIVE workspace set into an operating-system
sandbox the command runs inside. Nothing here reads the command string: ``cd``,
``$(...)``, symlinks, scripts and interpreters are all stopped (or not) by the kernel.

Spec (filled by the runtime from the same workspace arguments the file tools read)::

    SandboxSpec(private_workspace, posture, default_mode, allowed, refused,
                builtin_refused, builtin_allowed, env, unsandboxed_commands_allowed)

- ``posture`` ``"allowed_only"`` ("Deny everything, allow listed workspaces"): user data
  (``/Users``, ``/Volumes``, ``/home``, ``$HOME``, ...) is unreadable except the allowed
  rows and the private workspace; nothing outside the read & write rows, the private
  workspace and its private TMPDIR is writable. System roots stay readable.
- ``posture`` ``"any_except_denied"`` ("Allow everything, refuse listed workspaces"):
  everything is readable except the refused rows; writes follow ``default_mode`` (``"ro"``
  = only the read & write rows, the private workspace and its TMPDIR are writable).
- Nesting (R12 NESTING RULE): the most specific row wins (longest real-path prefix; a
  refusal wins a tie); built-in refusals are absolute except under ``builtin_allowed``
  (the host's own exceptions, e.g. the run's folder inside the gateway data folder).

Kinds: ``macos-sandbox-exec`` (generated SBPL profile; SBPL is last-match-wins, so rules
are emitted from the least to the most specific), ``linux-bwrap`` (bind mounts, tmpfs
over refused folders), ``linux-landlock`` (allow-list posture only, kernel >= 5.13),
otherwise ``none``: the command is REFUSED (fail closed) unless the host explicitly
allowed unsandboxed commands (``unsandboxed``; the gateway's ``serve
--unsandboxed-commands`` flag, never an environment variable).

Host level: :func:`configure_host` is called once by the host at boot (the gateway's
``serve``) with the scrubbed environment every command gets and the unsandboxed flag.
Once configured, a spawning tool call that carries no run sandbox is refused.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import platform
import shutil
import sys
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

POSTURES = ("allowed_only", "any_except_denied")
MODES = ("ro", "rw")

KIND_MACOS = "macos-sandbox-exec"
KIND_BWRAP = "linux-bwrap"
KIND_LANDLOCK = "linux-landlock"
KIND_NONE = "none"
KIND_UNSANDBOXED = "unsandboxed"

KIND_LABELS = {
    KIND_MACOS: "macOS sandbox-exec",
    KIND_BWRAP: "Linux bubblewrap",
    KIND_LANDLOCK: "Linux Landlock",
    KIND_NONE: "none",
    KIND_UNSANDBOXED: "none (unsandboxed commands allowed by the host)",
}

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# The one sentence a refused command answers with (the run continues).
REFUSED_NO_SANDBOX = (
    "Commands are not sandboxed on this gateway host, so they are refused: {reason}. "
    "The host can re-enable them with `serve --unsandboxed-commands`."
)
REFUSED_NO_SPEC = (
    "This command has no workspace sandbox for its run, so it is refused "
    "(the host requires every command to run inside its run's workspaces)."
)

STAMP_ARG = "_sandbox"  # the hidden tool argument the runtime stamps (paths only)
TMPDIR_NAME = ".tmp"

# User-data roots denied under "Deny everything, allow listed workspaces" (plus $HOME).
_USER_DATA_ROOTS_DARWIN = ("/Users", "/Volumes", "/private/var/root")
_USER_DATA_ROOTS_LINUX = ("/home", "/root", "/mnt", "/media", "/srv", "/run/user")
# System roots readable under the allow-list posture on Linux (bwrap/Landlock bind only these).
_SYSTEM_ROOTS_LINUX = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc", "/opt", "/nix", "/var/lib", "/run/systemd/resolve")


class SandboxError(ValueError):
    """The spec cannot be turned into a sandbox (bad posture/mode/path)."""


# --------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxRow:
    path: str
    mode: str  # "ro" | "rw"


@dataclass(frozen=True)
class SandboxSpec:
    """A run's effective workspace set, as the sandbox enforces it (see the module doc)."""

    private_workspace: str
    posture: str = "allowed_only"
    default_mode: str = "rw"
    allowed: Tuple[SandboxRow, ...] = ()
    refused: Tuple[str, ...] = ()
    builtin_refused: Tuple[str, ...] = ()
    builtin_allowed: Tuple[str, ...] = ()
    # Host level (never stamped into a tool call, never read from one):
    env: Optional[Mapping[str, str]] = field(default=None, compare=False, repr=False)
    unsandboxed_commands_allowed: bool = False

    def to_stamp(self) -> Dict[str, Any]:
        """The JSON the runtime stamps on a tool call: paths only (no env, no host flag)."""
        return {
            "version": 1,
            "private_workspace": self.private_workspace,
            "posture": self.posture,
            "default_mode": self.default_mode,
            "allowed": [{"path": r.path, "mode": r.mode} for r in self.allowed],
            "refused": list(self.refused),
            "builtin_refused": list(self.builtin_refused),
            "builtin_allowed": list(self.builtin_allowed),
        }

    @classmethod
    def from_stamp(cls, raw: Any) -> "SandboxSpec":
        """Parse a stamp (dict or JSON string). Raises SandboxError on anything malformed:
        a broken stamp never degrades to "no sandbox"."""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception as exc:
                raise SandboxError(f"malformed sandbox stamp ({exc})") from exc
        if not isinstance(raw, Mapping):
            raise SandboxError("malformed sandbox stamp (expected an object)")

        def _paths(key: str) -> Tuple[str, ...]:
            value = raw.get(key) or []
            if not isinstance(value, (list, tuple)):
                raise SandboxError(f"malformed sandbox stamp ({key} must be a list)")
            return tuple(str(v) for v in value)

        rows: List[SandboxRow] = []
        for item in raw.get("allowed") or []:
            if not isinstance(item, Mapping):
                raise SandboxError("malformed sandbox stamp (allowed rows must be objects)")
            rows.append(SandboxRow(path=str(item.get("path") or ""), mode=str(item.get("mode") or "")))
        spec = cls(
            private_workspace=str(raw.get("private_workspace") or ""),
            posture=str(raw.get("posture") or ""),
            default_mode=str(raw.get("default_mode") or "rw"),
            allowed=tuple(rows),
            refused=_paths("refused"),
            builtin_refused=_paths("builtin_refused"),
            builtin_allowed=_paths("builtin_allowed"),
        )
        return normalize_spec(spec)


def _check_path(raw: str, *, what: str) -> str:
    text = str(raw or "")
    if not text.strip():
        raise SandboxError(f"{what}: empty path")
    if any(ord(ch) < 32 or ch == "\x7f" for ch in text):
        raise SandboxError(f"{what}: control character in path {text!r}")
    p = Path(text).expanduser()
    if not p.is_absolute():
        raise SandboxError(f"{what}: path must be absolute ({text!r})")
    return os.path.realpath(str(p))


def normalize_spec(spec: SandboxSpec) -> SandboxSpec:
    """Validate and realpath every path (the sandbox matches resolved paths: /tmp is
    /private/tmp on macOS). Raises SandboxError."""
    if spec.posture not in POSTURES:
        raise SandboxError(f"unknown posture {spec.posture!r} (expected one of {POSTURES})")
    if spec.default_mode not in MODES:
        raise SandboxError(f"unknown default_mode {spec.default_mode!r} (expected ro or rw)")
    rows: List[SandboxRow] = []
    seen: set = set()
    for row in spec.allowed:
        if row.mode not in MODES:
            raise SandboxError(f"unknown mode {row.mode!r} for {row.path!r} (expected ro or rw)")
        key = (_check_path(row.path, what="allowed"), row.mode)
        if key not in seen:
            seen.add(key)
            rows.append(SandboxRow(path=key[0], mode=key[1]))

    def _uniq(items: Iterable[str], what: str) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(_check_path(p, what=what) for p in items))

    return replace(
        spec,
        private_workspace=_check_path(spec.private_workspace, what="private_workspace"),
        allowed=tuple(rows),
        refused=_uniq(spec.refused, "refused"),
        builtin_refused=_uniq(spec.builtin_refused, "builtin_refused"),
        builtin_allowed=_uniq(spec.builtin_allowed, "builtin_allowed"),
    )


# --------------------------------------------------------------------------------------
# Host policy (process level; set once by the host at boot)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _HostPolicy:
    configured: bool = False
    env: Optional[Tuple[Tuple[str, str], ...]] = None
    unsandboxed_commands_allowed: bool = False


_HOST_LOCK = threading.Lock()
_HOST = _HostPolicy()


def configure_host(*, env: Mapping[str, str], unsandboxed_commands_allowed: bool = False) -> None:
    """Set the host's command policy ONCE (the gateway's ``serve`` at boot, audited there).

    ``env`` is the scrubbed environment every sandboxed command starts from (never
    ``os.environ``); ``unsandboxed_commands_allowed`` re-enables commands on a host with no
    sandbox. Calling again with the same values is a no-op; different values raise
    RuntimeError (process-global state must not drift after boot). After this call, a
    spawning tool call that carries no run sandbox is refused."""
    global _HOST
    if not isinstance(env, Mapping):
        raise TypeError("configure_host(env=...) needs a mapping of environment variables")
    frozen_env = tuple(sorted((str(k), str(v)) for k, v in env.items()))
    new = _HostPolicy(configured=True, env=frozen_env, unsandboxed_commands_allowed=bool(unsandboxed_commands_allowed))
    with _HOST_LOCK:
        if _HOST.configured and _HOST != new:
            raise RuntimeError("the command sandbox host policy is already configured with different values")
        _HOST = new


def _reset_host_for_tests() -> None:
    """Tests only: forget the host policy."""
    global _HOST
    with _HOST_LOCK:
        _HOST = _HostPolicy()


def host_policy() -> Dict[str, Any]:
    """The host's command policy for status surfaces (variable NAMES only, never values)."""
    h = _HOST
    return {
        "configured": h.configured,
        "unsandboxed_commands_allowed": h.unsandboxed_commands_allowed,
        "env_keys": [k for k, _ in (h.env or ())],
        "kind": host_sandbox_kind(),
    }


def _host_env() -> Optional[Dict[str, str]]:
    h = _HOST
    return dict(h.env) if h.env is not None else None


# --------------------------------------------------------------------------------------
# Kind detection
# --------------------------------------------------------------------------------------


def _bwrap_path() -> Optional[str]:
    return shutil.which("bwrap")


def _landlock_abi() -> int:
    """Landlock ABI version (0 = unavailable)."""
    if not sys.platform.startswith("linux"):
        return 0
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.syscall.restype = ctypes.c_long
        # landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)
        abi = libc.syscall(_NR_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
        return int(abi) if abi > 0 else 0
    except Exception:
        return 0


def host_sandbox_kind(posture: str = "allowed_only") -> str:
    """The kind of sandbox this host can apply for ``posture`` (cheap; no spec needed):
    "macos-sandbox-exec" | "linux-bwrap" | "linux-landlock" | "none". Landlock only
    expresses the allow-list posture, so ``any_except_denied`` without bwrap is "none"."""
    if sys.platform == "darwin":
        return KIND_MACOS if os.access(SANDBOX_EXEC, os.X_OK) else KIND_NONE
    if sys.platform.startswith("linux"):
        if _bwrap_path():
            return KIND_BWRAP
        if posture == "allowed_only" and _landlock_abi() > 0:
            return KIND_LANDLOCK
    return KIND_NONE


def _no_sandbox_reason(posture: str) -> str:
    if sys.platform == "darwin":
        return f"{SANDBOX_EXEC} is missing"
    if sys.platform.startswith("linux"):
        if posture == "allowed_only":
            return "install bubblewrap (bwrap); Landlock is not available on this kernel"
        return "install bubblewrap (bwrap); Landlock cannot express \"Allow everything, refuse listed workspaces\""
    return f"unsupported platform ({platform.system() or sys.platform})"


# --------------------------------------------------------------------------------------
# Rule model shared by the backends
# --------------------------------------------------------------------------------------


def _under(path: str, prefix: str) -> bool:
    return path == prefix or prefix == "/" or path.startswith(prefix.rstrip("/") + "/")


def _system_readable_roots() -> Tuple[str, ...]:
    """Roots a command needs readable even when they sit under denied user data: the host's
    own Python (the gateway's venv and its base interpreter)."""
    out: List[str] = []
    for raw in (sys.prefix, sys.base_prefix, os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))):
        if raw:
            real = os.path.realpath(raw)
            if real not in ("/",) and real not in out:
                out.append(real)
    return tuple(out)


def private_tmpdir(spec: SandboxSpec) -> str:
    return os.path.join(spec.private_workspace, TMPDIR_NAME)


@dataclass(frozen=True)
class _Rule:
    path: str
    action: str  # "deny" | "ro" | "rw" | "read" (a system root: readable, writes untouched)


def _ordered_rules(spec: SandboxSpec, *, user_roots: Sequence[str]) -> Tuple[List[_Rule], List[_Rule]]:
    """(base rules, builtin rules) in emission order: later rules override earlier ones.

    Base: the posture's floor (user-data denial under allow-list), then every row sorted by
    path length (least specific first; on a tie deny after ro after rw so a refusal wins).
    Builtin: the built-in refusals, then the base rules lying under a builtin_allowed entry
    re-emitted in the same order (the host's own exceptions)."""
    rows: List[_Rule] = []
    for p in _system_readable_roots():
        rows.append(_Rule(p, "read"))
    for r in spec.allowed:
        rows.append(_Rule(r.path, r.mode))
    for p in spec.refused:
        rows.append(_Rule(p, "deny"))
    rows.append(_Rule(spec.private_workspace, "rw"))
    rows.append(_Rule(private_tmpdir(spec), "rw"))
    tie = {"read": 0, "rw": 1, "ro": 2, "deny": 3}
    rows.sort(key=lambda r: (len(r.path.rstrip("/")), tie[r.action]))
    floor: List[_Rule] = []
    if spec.posture == "allowed_only":
        floor = [_Rule(p, "deny") for p in user_roots]
    builtin: List[_Rule] = [_Rule(p, "deny") for p in spec.builtin_refused]
    if spec.builtin_refused:
        # A system root is never a host exception: only the run's own rows are re-emitted.
        builtin.extend(r for r in rows if r.action != "read" and any(_under(r.path, a) for a in spec.builtin_allowed))
    return floor + rows, builtin


# --------------------------------------------------------------------------------------
# macOS: sandbox-exec + SBPL
# --------------------------------------------------------------------------------------


def _sbpl_regex(path: str) -> str:
    out = []
    for ch in path:
        out.append("\\" + ch if ch in ".^$*+?()[]{}|\\/" else ch)
    return "".join(out).replace('"', '\\"')


def _sbpl_str(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


_DEV_WRITABLE = (
    '(literal "/dev/null")',
    '(literal "/dev/zero")',
    '(literal "/dev/tty")',
    '(literal "/dev/ptmx")',
    '(literal "/dev/dtracehelper")',
    '(regex #"^/dev/ttys[0-9]+$")',
    '(regex #"^/dev/fd/[0-9]+$")',
)


_CS_DARWIN_USER_TEMP_DIR = 65537


def _darwin_user_temp_dir() -> Optional[str]:
    """confstr(_CS_DARWIN_USER_TEMP_DIR): where xcrun (the /usr/bin/python3, git, ... shims)
    keeps its lookup cache."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        buf = ctypes.create_string_buffer(1024)
        n = libc.confstr(_CS_DARWIN_USER_TEMP_DIR, buf, 1024)
        if 0 < n <= 1024:
            return os.path.realpath(buf.value.decode())
    except Exception:
        pass
    return None


def macos_profile(spec: SandboxSpec) -> str:
    """The SBPL profile for ``spec`` (see the module doc). SBPL is last-match-wins."""
    spec = normalize_spec(spec)
    home = os.path.realpath(os.path.expanduser("~"))
    user_roots = list(_USER_DATA_ROOTS_DARWIN)
    if home not in ("/",) and not any(_under(home, r) for r in user_roots):
        user_roots.append(home)
    base, builtin = _ordered_rules(spec, user_roots=user_roots)
    lines: List[str] = [
        "(version 1)",
        ";; generated by abstractcore.tools.sandbox (round 12) - last matching rule wins",
        "(allow default)",
        # Escapes through other processes: Apple Events (Terminal/Finder/System Events run
        # commands outside this sandbox) and LaunchServices (`open` starts apps unsandboxed).
        "(deny appleevent-send)",
        '(deny mach-lookup (global-name "com.apple.coreservices.launchservicesd") (global-name "com.apple.coreservices.quarantine-resolver"))',
    ]
    # Writes: only listed places, unless the posture's default is read & write.
    writes_default_rw = spec.posture == "any_except_denied" and spec.default_mode == "rw"
    if not writes_default_rw:
        lines.append('(deny file-write* (subpath "/"))')
        lines.append("(allow file-write* " + " ".join(_DEV_WRITABLE) + ")")
        xcrun_tmp = _darwin_user_temp_dir()
        if xcrun_tmp:
            # xcrun's own cache files only (never the folder): without them every shimmed
            # tool prints "couldn't create cache file" on each call.
            lines.append('(allow file-write* (regex #"^' + _sbpl_regex(xcrun_tmp.rstrip("/")) + '/xcrun_db"))')

    def _emit(rule: _Rule) -> None:
        s = _sbpl_str(rule.path)
        if rule.action == "deny":
            lines.append(f"(deny file-read* file-write* (subpath {s}))")
        elif rule.action == "read":
            lines.append(f"(allow file-read* (subpath {s}))")
        elif rule.action == "ro":
            lines.append(f"(allow file-read* (subpath {s}))")
            lines.append(f"(deny file-write* (subpath {s}))")
        else:
            lines.append(f"(allow file-read* file-write* (subpath {s}))")

    for rule in base:
        _emit(rule)
    for rule in builtin:
        _emit(rule)
    # Path lookup (`cd`, getcwd, realpath) into an allowed folder stats its ancestors:
    # metadata only (existence/attributes), never their contents.
    ancestors: List[str] = []
    for rule in base + builtin:
        if rule.action == "deny":
            continue
        parent = os.path.dirname(rule.path)
        while parent and parent != "/":
            if parent not in ancestors:
                ancestors.append(parent)
            parent = os.path.dirname(parent)
    if ancestors:
        lines.append("(allow file-read-metadata " + " ".join(f"(literal {_sbpl_str(a)})" for a in ancestors) + ")")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Linux: bubblewrap
# --------------------------------------------------------------------------------------


def bwrap_argv(spec: SandboxSpec, *, bwrap: str = "bwrap") -> List[str]:
    """bwrap arguments (without the command). Mounts apply in order: later wins."""
    spec = normalize_spec(spec)
    home = os.path.realpath(os.path.expanduser("~"))
    user_roots = list(_USER_DATA_ROOTS_LINUX)
    if home != "/" and not any(_under(home, r) for r in user_roots):
        user_roots.append(home)
    argv: List[str] = [bwrap, "--die-with-parent"]
    if spec.posture == "any_except_denied":
        argv += ["--bind" if spec.default_mode == "rw" else "--ro-bind", "/", "/"]
        floor_rules: List[_Rule] = []
    else:
        for root in _SYSTEM_ROOTS_LINUX:
            if os.path.exists(root):
                argv += ["--ro-bind", root, root]
        floor_rules = []
    argv += ["--dev", "/dev", "--proc", "/proc"]
    base, builtin = _ordered_rules(spec, user_roots=())
    os.makedirs(private_tmpdir(spec), exist_ok=True)
    # A refused folder is masked by an empty tmpfs. A more specific allowed row inside it
    # (the nesting rule: the most specific row wins) is bound AFTER the mask, and bwrap
    # creates that row's mount point inside the tmpfs, so the tmpfs must stay writable
    # until every mount is in place: the masks are made read-only at the very end
    # (`--remount-ro` changes only the tmpfs mount itself, never the rows bound inside it).
    # Remounting a mask read-only right away refused the child ("bwrap: Can't mkdir
    # …/parent/child: Read-only file system"; the 0.9.0 runtime linux-sandbox job).
    masks: List[str] = []
    mounts: List[Tuple[str, str]] = []  # (path, "mask" | "bind"), in emission order
    for rule in floor_rules + base + builtin:
        if rule.action == "deny":
            if spec.posture == "allowed_only" and not os.path.exists(rule.path):
                continue
            if os.path.isdir(rule.path):
                argv += ["--tmpfs", rule.path]
                masks.append(rule.path)
                mounts.append((rule.path, "mask"))
            elif os.path.exists(rule.path):
                argv += ["--ro-bind", "/dev/null", rule.path]
                mounts.append((rule.path, "bind"))
        elif os.path.exists(rule.path):
            rw = rule.action == "rw" or (rule.action == "read" and spec.posture == "any_except_denied" and spec.default_mode == "rw")
            argv += ["--bind" if rw else "--ro-bind", rule.path, rule.path]
            mounts.append((rule.path, "bind"))
    for path in dict.fromkeys(masks):
        # The topmost mount at this path must still be the mask (a later rule of the very
        # same path re-bound it: that bind decides, and remounting would hit the bind).
        last = [kind for (p, kind) in mounts if p == path][-1]
        if last == "mask":
            argv += ["--remount-ro", path]
    return argv


# --------------------------------------------------------------------------------------
# Linux: Landlock (allow-list posture only; applied by a tiny exec helper)
# --------------------------------------------------------------------------------------

_NR_LANDLOCK_CREATE_RULESET = 444  # identical on x86_64 and aarch64 (generic table)
_NR_LANDLOCK_ADD_RULE = 445
_NR_LANDLOCK_RESTRICT_SELF = 446
_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14
_LL_FILE_ONLY = _LL_EXECUTE | _LL_WRITE_FILE | _LL_READ_FILE | _LL_TRUNCATE


def landlock_rules(spec: SandboxSpec) -> List[Tuple[str, str]]:
    """[(path, "ro"|"rw")] the Landlock ruleset grants. Landlock only ALLOWS, so a refused
    row inside an allowed one cannot be expressed: build_sandbox refuses such a spec."""
    spec = normalize_spec(spec)
    out: List[Tuple[str, str]] = [(r, "ro") for r in _SYSTEM_ROOTS_LINUX if os.path.exists(r)]
    out += [(r, "ro") for r in _system_readable_roots()]
    out += [("/proc", "ro"), ("/dev", "ro"), ("/dev/null", "rw"), ("/dev/tty", "rw"), ("/dev/pts", "rw"), ("/dev/ptmx", "rw")]
    for r in spec.allowed:
        if not any(_under(r.path, b) and not any(_under(r.path, a) for a in spec.builtin_allowed) for b in spec.builtin_refused):
            out.append((r.path, r.mode))
    out.append((spec.private_workspace, "rw"))
    out.append((private_tmpdir(spec), "rw"))
    return out


def _landlock_unexpressible(spec: SandboxSpec) -> Optional[str]:
    if spec.posture != "allowed_only":
        return "Landlock cannot express \"Allow everything, refuse listed workspaces\""
    granted = [r.path for r in spec.allowed] + [spec.private_workspace]
    for refused in list(spec.refused) + list(spec.builtin_refused):
        for g in granted:
            if _under(refused, g) and refused != g and not any(_under(refused, a) for a in spec.builtin_allowed if a != g):
                return f"Landlock cannot refuse {refused} inside the allowed {g}"
    return None


def apply_landlock(rules: Sequence[Tuple[str, str]]) -> None:
    """Restrict THIS process (and its future children) to ``rules``. Raises OSError."""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(_NR_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi <= 0:
        raise OSError(ctypes.get_errno(), "Landlock is not available")
    handled = (1 << 13) - 1  # ABI 1: EXECUTE..MAKE_SYM
    if abi >= 2:
        handled |= _LL_REFER
    if abi >= 3:
        handled |= _LL_TRUNCATE
    file_only = _LL_FILE_ONLY & handled

    class _RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class _PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = _RulesetAttr(handled)
    ruleset_fd = libc.syscall(_NR_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    ro = (_LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR) & handled
    for path, mode in rules:
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)  # type: ignore[attr-defined]
        except OSError:
            continue
        try:
            access = handled if mode == "rw" else ro
            if not os.path.isdir(path):
                access &= file_only
            pb = _PathBeneath(access, fd)
            rc = libc.syscall(_NR_LANDLOCK_ADD_RULE, ctypes.c_int(ruleset_fd), ctypes.c_int(1), ctypes.byref(pb), ctypes.c_uint32(0))
            if rc != 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
        finally:
            os.close(fd)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0)) != 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    os.close(ruleset_fd)


def _landlock_exec_main(argv: Sequence[str]) -> None:  # pragma: no cover - runs in the child
    """`python -m abstractcore.tools.sandbox <rules-json> -- <argv...>`: restrict, then exec.
    Any failure exits 126 without running the command (fail closed)."""
    try:
        sep = list(argv).index("--")
        rules = [tuple(x) for x in json.loads(argv[0])]
        cmd = list(argv[sep + 1 :])
        apply_landlock(rules)  # type: ignore[arg-type]
    except Exception as exc:
        sys.stderr.write(f"sandbox: Landlock could not be applied ({exc}); the command did not run\n")
        os._exit(126)
    os.execvp(cmd[0], cmd)


# --------------------------------------------------------------------------------------
# The Sandbox object
# --------------------------------------------------------------------------------------

Command = Union[str, Sequence[str]]


@dataclass(frozen=True)
class Sandbox:
    kind: str
    spec: Optional[SandboxSpec]
    reason: str = ""  # why commands are refused (kind none)
    _prefix: Tuple[str, ...] = ()

    @property
    def refuses(self) -> bool:
        return self.kind == KIND_NONE

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    def refusal(self) -> str:
        if self.spec is None and self.reason == REFUSED_NO_SPEC:
            return REFUSED_NO_SPEC
        return REFUSED_NO_SANDBOX.format(reason=self.reason or "no sandbox is available")

    def rendered_line(self) -> str:
        if self.kind == KIND_NONE:
            return "Sandbox: none — commands refused on this host"
        if self.kind == KIND_UNSANDBOXED:
            return "Sandbox: none — unsandboxed commands allowed by the host (serve --unsandboxed-commands)"
        return f"Sandbox: {self.label}"

    def describe(self) -> Dict[str, Any]:
        """Evidence for the tool result / ledger. Built-in refusals are host policy and are
        counted, never listed (they may name other accounts' folders)."""
        out: Dict[str, Any] = {"kind": self.kind, "label": self.label}
        if self.reason:
            out["reason"] = self.reason
        s = self.spec
        if s is not None:
            out.update(
                {
                    "posture": s.posture,
                    "default_mode": s.default_mode,
                    "private_workspace": s.private_workspace,
                    "tmpdir": private_tmpdir(s),
                    "allowed": [{"path": r.path, "mode": r.mode} for r in s.allowed],
                    "refused": list(s.refused),
                    "builtin_refused": len(s.builtin_refused),
                }
            )
        return out

    def env_for(self, env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """The command's environment: the given env, else the spec's (the host's scrubbed
        env), else this process's (library use without a host) — plus the private TMPDIR."""
        if env is not None:
            base = dict(env)
        elif self.spec is not None and self.spec.env is not None:
            base = dict(self.spec.env)
        else:
            base = dict(os.environ)
        if self.spec is not None:
            tmp = private_tmpdir(self.spec)
            os.makedirs(tmp, exist_ok=True)
            for key in ("TMPDIR", "TMP", "TEMP"):
                base[key] = tmp
        return base

    def wrap(self, command: Command, cwd: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> Tuple[List[str], Dict[str, str]]:
        """(argv, env) to pass to ``subprocess.Popen(argv, shell=False, cwd=cwd, env=env)``.
        A string command runs through ``/bin/sh -c``. Raises SandboxError when this sandbox
        refuses commands (kind none): callers must check ``refuses`` first."""
        if self.refuses:
            raise SandboxError(self.refusal())
        argv = ["/bin/sh", "-c", command] if isinstance(command, str) else [str(a) for a in command]
        if not argv:
            raise SandboxError("empty command")
        return list(self._prefix) + argv, self.env_for(env)


def build_sandbox(spec: Optional[SandboxSpec]) -> Sandbox:
    """The sandbox for ``spec`` on this host. ``spec`` None = no run sandbox: refused when
    the host is configured (fail closed), else an unsandboxed passthrough (library use)."""
    if spec is None:
        if _HOST.configured:
            return Sandbox(kind=KIND_NONE, spec=None, reason=REFUSED_NO_SPEC)
        return Sandbox(kind=KIND_UNSANDBOXED, spec=None)
    spec = normalize_spec(spec)
    kind = host_sandbox_kind(spec.posture)
    if kind == KIND_LANDLOCK:
        why = _landlock_unexpressible(spec)
        if why:
            kind = KIND_NONE
            reason = why + "; install bubblewrap (bwrap)"
        else:
            rules = json.dumps(landlock_rules(spec))
            os.makedirs(private_tmpdir(spec), exist_ok=True)
            return Sandbox(kind=KIND_LANDLOCK, spec=spec, _prefix=(sys.executable, "-m", "abstractcore.tools.sandbox", rules, "--"))
    else:
        reason = _no_sandbox_reason(spec.posture)
    if kind == KIND_MACOS:
        return Sandbox(kind=KIND_MACOS, spec=spec, _prefix=(SANDBOX_EXEC, "-p", macos_profile(spec)))
    if kind == KIND_BWRAP:
        return Sandbox(kind=KIND_BWRAP, spec=spec, _prefix=tuple(bwrap_argv(spec, bwrap=_bwrap_path() or "bwrap")) + ("--",))
    if spec.unsandboxed_commands_allowed:
        return Sandbox(kind=KIND_UNSANDBOXED, spec=spec, reason=reason)
    return Sandbox(kind=KIND_NONE, spec=spec, reason=reason)


def sandbox_for_tool_call(stamp: Any) -> Sandbox:
    """What a spawning tool runs under, from its hidden ``_sandbox`` stamp. The env and the
    unsandboxed flag come from the HOST policy only (never from the call). A malformed stamp
    refuses (it never degrades to no sandbox)."""
    if stamp is None or stamp == "" or stamp == {}:
        return build_sandbox(None)
    try:
        spec = SandboxSpec.from_stamp(stamp)
    except SandboxError as exc:
        return Sandbox(kind=KIND_NONE, spec=None, reason=str(exc))
    spec = replace(spec, env=_host_env(), unsandboxed_commands_allowed=_HOST.unsandboxed_commands_allowed)
    return build_sandbox(spec)


def refusal_result(sandbox: Sandbox, **fields: Any) -> Dict[str, Any]:
    """The structured tool result of a refused command (the run continues)."""
    sentence = sandbox.refusal()
    return {
        "success": False,
        "error": sentence,
        "sandbox": sandbox.describe(),
        "rendered": f"{sentence}\n{sandbox.rendered_line()}",
        **fields,
    }


__all__ = [
    "KIND_BWRAP",
    "KIND_LANDLOCK",
    "KIND_MACOS",
    "KIND_NONE",
    "KIND_UNSANDBOXED",
    "REFUSED_NO_SANDBOX",
    "REFUSED_NO_SPEC",
    "STAMP_ARG",
    "Sandbox",
    "SandboxError",
    "SandboxRow",
    "SandboxSpec",
    "bwrap_argv",
    "build_sandbox",
    "configure_host",
    "host_policy",
    "host_sandbox_kind",
    "landlock_rules",
    "macos_profile",
    "normalize_spec",
    "private_tmpdir",
    "refusal_result",
    "sandbox_for_tool_call",
]


if __name__ == "__main__":  # pragma: no cover - the Landlock exec helper
    _landlock_exec_main(sys.argv[1:])
