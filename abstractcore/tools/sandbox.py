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
``windows-restricted-token`` (a restricted token, access-list entries and, where writes
are limited to the rows, integrity labels, applied by the launcher in
``sandbox_windows.py``), otherwise ``none``: the command is REFUSED (fail closed) unless
the host explicitly allowed unsandboxed commands (``unsandboxed``; the gateway's ``serve
--unsandboxed-commands`` flag, never an environment variable).

Host level: :func:`configure_host` is called once by the host at boot (the gateway's
``serve``) with the scrubbed environment every command gets and the unsandboxed flag.
Once configured, a spawning tool call that carries no run sandbox is refused.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import json
import os
import platform
import shutil
import sys
import threading
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

POSTURES = ("allowed_only", "any_except_denied")
MODES = ("ro", "rw")

KIND_MACOS = "macos-sandbox-exec"
KIND_BWRAP = "linux-bwrap"
KIND_LANDLOCK = "linux-landlock"
KIND_WINDOWS = "windows-restricted-token"
KIND_NONE = "none"
KIND_UNSANDBOXED = "unsandboxed"

KIND_LABELS = {
    KIND_MACOS: "macOS sandbox-exec",
    KIND_BWRAP: "Linux bubblewrap",
    KIND_LANDLOCK: "Linux Landlock",
    KIND_WINDOWS: "Windows restricted token",
    KIND_NONE: "none",
    KIND_UNSANDBOXED: "none (unsandboxed commands allowed by the host)",
}

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# The Windows launcher: run as a script by the host's interpreter, never imported here.
WINDOWS_LAUNCHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_windows.py")

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
# "Deny everything, allow listed workspaces" on macOS denies these roots (read and write):
# user data, mounted volumes, root's home, and the shared temp roots (/private/tmp and the
# per-user temp root /private/var/folders), where other processes leave files. System
# roots (/usr, /bin, /System, /Library, /opt, /usr/local, /Applications, ...) stay READABLE
# (commands need their binaries, libraries and frameworks); under this posture every write
# outside the listed workspaces is denied anyway.
_USER_DATA_ROOTS_DARWIN = ("/Users", "/Volumes", "/private/var/root", "/private/tmp", "/private/var/folders")
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
    try:
        real = os.path.realpath(str(p))
    except OSError as exc:
        raise SandboxError(f"{what}: not a path this host can resolve ({text!r}: {exc})") from exc
    if os.name == "nt":
        if real.startswith("\\\\?\\"):  # one folder, one spelling
            real = "\\\\" + real[8:] if real.startswith("\\\\?\\UNC\\") else real[4:]
        if not os.path.isabs(real) or ":" in os.path.splitdrive(real)[1] or real.startswith("\\\\.\\"):
            raise SandboxError(f"{what}: not a folder or file path ({text!r})")  # a data stream, a device
    return real


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
    "macos-sandbox-exec" | "linux-bwrap" | "linux-landlock" | "windows-restricted-token" |
    "none". Landlock only expresses the allow-list posture, so ``any_except_denied``
    without bwrap is "none"."""
    if sys.platform == "darwin":
        return KIND_MACOS if os.access(SANDBOX_EXEC, os.X_OK) else KIND_NONE
    if sys.platform.startswith("linux"):
        if _bwrap_path():
            return KIND_BWRAP
        if posture == "allowed_only" and _landlock_abi() > 0:
            return KIND_LANDLOCK
    if sys.platform == "win32":
        return KIND_WINDOWS
    return KIND_NONE


WINDOWS_TOO_MANY_ROWS = "this run has more workspace rows than a Windows command line can carry to the sandbox"
# The launcher takes its rules as one argument, and a Windows command line holds 32,767
# characters: the rules may use half of it.
WINDOWS_POLICY_LIMIT = 16000


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
    """``path`` is ``prefix`` or lies below it, by this platform's path rules (separators,
    and letter case on Windows). Both are real paths (``normalize_spec``)."""
    path, prefix = os.path.normcase(path), os.path.normcase(prefix)
    return path == prefix or path.startswith(prefix.rstrip(os.sep) + os.sep)


def _depth(path: str) -> int:
    """Orders nested rows: an enclosing path is always shorter."""
    return len(path.rstrip(os.sep))


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
    rows.sort(key=lambda r: (_depth(r.path), tie[r.action]))
    floor: List[_Rule] = []
    if spec.posture == "allowed_only":
        floor = [_Rule(p, "deny") for p in user_roots]
    builtin: List[_Rule] = [_Rule(p, "deny") for p in spec.builtin_refused]
    if spec.builtin_refused:
        # A system root is never a host exception: only the run's own rows are re-emitted.
        builtin.extend(r for r in rows if r.action != "read" and any(_under(r.path, a) for a in spec.builtin_allowed))
    return floor + rows, builtin


def path_readable(spec: SandboxSpec, path: str) -> bool:
    """True when ``path`` (resolved through symlinks) is readable under ``spec``, by the same
    nesting rule the profiles enforce: built-in refusals are absolute (unless under a
    built-in exception); otherwise the LONGEST row prefixing the path decides (the private
    workspace and the allowed rows, ro or rw, allow; a refused row refuses; a refusal wins a
    tie); with no row, "any_except_denied" allows and "allowed_only" refuses.

    Used by the browser probe's loopback origin (``browser_tools``): a local page may only
    pull files a command of the same run could read."""
    spec = normalize_spec(spec)
    real = os.path.realpath(str(path))
    if any(_under(real, b) for b in spec.builtin_refused) and not any(_under(real, a) for a in spec.builtin_allowed):
        return False
    best: Optional[Tuple[int, int]] = None  # (prefix length, 1 = refused wins a tie)
    allows = [spec.private_workspace, private_tmpdir(spec)] + [r.path for r in spec.allowed]
    for prefix, refused in [(a, 0) for a in allows] + [(r, 1) for r in spec.refused]:
        if _under(real, prefix):
            key = (_depth(prefix), refused)
            if best is None or key > best:
                best = key
    if best is None:
        return spec.posture == "any_except_denied"
    return best[1] == 0


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
    # xcrun's own cache files only (never the folder): without them every shimmed tool
    # (/usr/bin/python3, git, ...) prints "couldn't create cache file" on each call. Emitted
    # AFTER the rows (last match wins), because the per-user temp root is denied under
    # "Deny everything, allow listed workspaces".
    xcrun_rule = ('(allow file-read* file-write* (regex #"^' + _sbpl_regex(xcrun_tmp.rstrip("/")) + '/xcrun_db"))') if xcrun_tmp else None
    if not writes_default_rw:
        if xcrun_rule:
            lines.append(xcrun_rule)

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
    if xcrun_rule and (spec.posture == "allowed_only" or not writes_default_rw):
        lines.append(xcrun_rule)
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
            up = os.path.dirname(parent)
            if up == parent:  # a filesystem root that is not "/" (C:\ on Windows)
                break
            parent = up
    if xcrun_rule and xcrun_tmp:
        # Creating the cache file looks up its folder: metadata of the folder and its
        # ancestors only (never a listing or a read of anything else in the temp root).
        parent = xcrun_tmp.rstrip("/")
        while parent and parent != "/":
            if parent not in ancestors:
                ancestors.append(parent)
            up = os.path.dirname(parent)
            if up == parent:
                break
            parent = up
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
# Windows: restricted token + access-list entries (applied by sandbox_windows.py)
# --------------------------------------------------------------------------------------


def _windows_writes_rows(spec: SandboxSpec) -> bool:
    """Writes are limited to the rows (integrity labels), not the user's own everywhere."""
    return not (spec.posture == "any_except_denied" and spec.default_mode == "rw")


_DRIVE_FIXED = 3
_WRITE_DAC, _OPEN_EXISTING, _BACKUP_SEMANTICS = 0x40000, 3, 0x02000000


def _may_change_permissions(path: str) -> bool:
    """This user can rewrite ``path``'s access list (it can then carry a refusal). Asks
    Windows to open it for that right, which changes nothing."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    handle = k32.CreateFileW(path, _WRITE_DAC, 7, None, _OPEN_EXISTING, _BACKUP_SEMANTICS, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        return False
    k32.CloseHandle(ctypes.c_void_p(handle))
    return True


def _windows_user_data_roots() -> Tuple[str, ...]:
    """What "Deny everything, allow listed workspaces" refuses on Windows, from this machine
    (the counterpart of /Users, /Volumes, /home): the user's profile, and on every fixed
    drive the drive itself when this user may change its permissions and it holds no
    system root (an administrator may change C:\\ itself), else each folder at
    its root that this user may change and that is not a system root (Windows, Program
    Files, ProgramData stay readable, as /usr and /Library do), and the user's own folder of
    the drive's recycle bin. What this user cannot change (another account's folder, one an
    installer made, a file at the root of the drive) cannot carry a refusal and stays as
    readable as it is."""
    roots = [os.path.realpath(os.path.expanduser("~"))]
    if os.name != "nt":  # the rules computed elsewhere, as in the tests
        return tuple(roots)
    system = {os.path.normcase(os.path.realpath(os.environ[v])) for v in ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData") if os.environ.get(v)}
    kernel32 = ctypes.WinDLL("kernel32")
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive)) != _DRIVE_FIXED:
            continue
        if not any(_under(s, drive) for s in system) and _may_change_permissions(drive):
            roots.append(drive)
            continue
        try:
            with os.scandir(drive) as entries:
                tops = [e.path for e in entries if e.is_dir(follow_symlinks=False) and not e.is_symlink() and os.path.normcase(e.path) not in system]
        except OSError:
            continue
        bins = [os.path.join(drive, "$Recycle.Bin")]
        try:
            with os.scandir(bins[0]) as entries:
                tops += [e.path for e in entries if e.is_dir(follow_symlinks=False)]
        except OSError:
            pass
        roots += [p for p in tops if os.path.normcase(p) != os.path.normcase(bins[0]) and _may_change_permissions(p)]
    return tuple(r for r in dict.fromkeys(roots) if not any(o != r and _under(r, o) for o in roots))


def _windows_low_places() -> Tuple[str, ...]:
    """Folders of the profile that Windows labels Low (writable to any Low program, and so
    to a command whose writes are limited by its integrity level): refused writes under the
    read-only default. (Under the allow-list posture the profile is refused anyway.)"""
    home = os.path.realpath(os.path.expanduser("~"))
    return tuple(p for p in (os.path.join(home, "AppData", "LocalLow"), os.path.join(home, "Favorites")) if os.path.isdir(p))


def _windows_state_folder() -> str:
    """Where the launcher lists the folders it labelled (refused to every command whose
    writes are limited to its rows)."""
    local = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    return os.path.join(os.path.realpath(local), "AbstractCore", "command-sandbox")


def windows_rules(spec: SandboxSpec) -> List[List[str]]:
    """The launcher's rules for ``spec``, outermost first. Windows gives the nearest row the
    last word, so only the rows that CHANGE what their surroundings allow become rules (the
    surroundings: read & write under "Allow everything, refuse listed workspaces" with read &
    write by default, else readable and unwritable, by integrity level):

    - ``["deny", path]``   a refused folder: closed at the folder itself;
    - ``["carve", path]``  a refused folder with an allowed row inside;
    - ``["pass", folder, carved]``  a folder between the ``carved`` folder and an allowed
      row inside it (traversal only, never its content);
    - ``["ro", path]``  a read-only row (writes refused; reading is the user's own right);
    - ``["rw" | "read", path]``  a row re-opened inside a refused or read-only one (``read``:
      a read-only row inside a refused one, a system root, the host's own Python); where
      writes are limited to the rows, every read & write row is one (it is labelled);
    - ``["guard", path]``  a folder Windows labels Low: writes refused (read-only default);
    - ``carve-list`` / ``pass-list`` for ``carve`` / ``pass`` where writes are limited to the
      rows: the names directly inside are listed, nothing is read.

    Under "Deny everything, allow listed workspaces" the user's data
    (``_windows_user_data_roots``) is refused, except where a row lies at or above it.
    A built-in refusal is absolute: a row inside one is dropped unless it lies under one of
    the host's exceptions (``builtin_allowed``); so is the launcher's state folder."""
    spec = normalize_spec(spec)
    writes_rows = _windows_writes_rows(spec)
    absolutes = spec.builtin_refused + ((_windows_state_folder(),) if writes_rows else ())

    def absolute(path: str) -> bool:
        return any(_under(path, b) for b in absolutes) and not any(_under(path, a) for a in spec.builtin_allowed)

    rank = {"read": 0, "rw": 1, "ro": 2, "deny": 3}  # on one path: a refusal, then read-only, wins
    rows: Dict[str, Tuple[str, str]] = {}

    def row(path: str, action: str) -> None:
        key = os.path.normcase(path)
        if key not in rows or rank[action] > rank[rows[key][1]]:
            rows[key] = (path, action)

    for path in _system_readable_roots():
        if not absolute(path):
            row(path, "read")
    for r in spec.allowed:
        if not absolute(r.path):
            row(r.path, r.mode)
    row(spec.private_workspace, "rw")
    if spec.posture == "allowed_only":
        mine = [spec.private_workspace] + [r.path for r in spec.allowed]
        for path in _windows_user_data_roots():
            if not any(_under(path, m) for m in mine):  # a row at or above it opens it, as elsewhere
                row(path, "deny")
    for path in spec.refused + absolutes:
        row(path, "deny")

    kept: List[List[str]] = []  # [action, path], outermost first
    passes: Dict[str, List[str]] = {}
    for path, action in sorted(rows.values(), key=lambda item: _depth(item[0])):
        outer = next((k for k in reversed(kept) if _under(path, k[1])), None)
        around = {"carve": "deny", "read": "ro"}.get(outer[0], outer[0]) if outer else ("ro" if writes_rows else "rw")
        if around == ("ro" if action == "read" else action) or (action == "read" and around != "deny"):
            continue  # already what its surroundings allow
        kept.append([action, path])
        if action == "ro" and around == "deny":
            kept.append(["read", path])  # refusing writes does not re-open what is refused
        if action != "deny" and around == "deny":
            outer[0] = "carve"
            folder = os.path.dirname(path)
            while not _under(outer[1], folder):
                passes.setdefault(os.path.normcase(folder), ["pass", folder, outer[1]])
                folder = os.path.dirname(folder)
    if writes_rows:
        # The folders on the way to a row list their names (never their content): git turns
        # its working folder into its long form, which looks each folder up in its parent,
        # and every row of "Deny everything" lies in a refused folder.
        for rule in kept + list(passes.values()):
            rule[0] = {"carve": "carve-list", "pass": "pass-list"}.get(rule[0], rule[0])
    guards: List[List[str]] = []
    if spec.posture == "any_except_denied" and writes_rows:
        for path in _windows_low_places():
            outer = next((k for k in reversed(kept) if _under(path, k[1])), None)
            if outer is None or outer[0] in ("ro", "read"):  # reachable and not a row's to write
                guards.append(["guard", path])
    return kept + guards + sorted(passes.values(), key=lambda rule: _depth(rule[1]))


def windows_policy(rules: Sequence[Sequence[str]], spec: Optional[SandboxSpec] = None) -> str:
    """The launcher's policy argument: the rules as compressed JSON (paths repeat a lot).
    Where ``spec`` limits writes to its rows: the folders to label (every ``rw`` rule) and
    the launcher's state folder too."""
    policy: Dict[str, Any] = {"rules": [list(r) for r in rules]}
    if spec is not None and _windows_writes_rows(spec):
        policy.update(label=[r[1] for r in rules if r[0] == "rw"], state=_windows_state_folder())
    raw = json.dumps(policy, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode("ascii")


def _windows_python() -> str:
    """The interpreter that runs the launcher: this one, or its console twin when the host
    runs windowless (pythonw has no standard streams to hand to the command)."""
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":
        twin = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(twin):
            return twin
    return exe


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
    _shell: Tuple[str, ...] = ("/bin/sh", "-c")  # what runs a command given as one string
    _exec: Tuple[str, ...] = ()  # what introduces a command given as a list

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
        A string command runs through ``/bin/sh -c`` (the Windows sandbox: through
        ``%ComSpec% /c``, as ``Popen(command, shell=True)`` does there). Raises SandboxError
        when this sandbox refuses commands (kind none): callers must check ``refuses`` first."""
        if self.refuses:
            raise SandboxError(self.refusal())
        if isinstance(command, str):
            argv = [*self._shell, command]
        else:
            argv = [str(a) for a in command]
            if not argv:
                raise SandboxError("empty command")
            argv = [*self._exec, *argv]
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
    if kind == KIND_WINDOWS:
        policy = windows_policy(windows_rules(spec), spec)
        if len(policy) <= WINDOWS_POLICY_LIMIT:
            os.makedirs(private_tmpdir(spec), exist_ok=True)
            # -I: nothing of the command's environment or folder reaches the launcher's imports.
            prefix = (_windows_python(), "-I", WINDOWS_LAUNCHER, policy)
            return Sandbox(kind=KIND_WINDOWS, spec=spec, _prefix=prefix, _shell=("--shell",), _exec=("--",))
        kind, reason = KIND_NONE, WINDOWS_TOO_MANY_ROWS
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
    "KIND_WINDOWS",
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
    "path_readable",
    "private_tmpdir",
    "refusal_result",
    "sandbox_for_tool_call",
    "windows_policy",
    "windows_rules",
]


if __name__ == "__main__":  # pragma: no cover - the Landlock exec helper
    _landlock_exec_main(sys.argv[1:])
