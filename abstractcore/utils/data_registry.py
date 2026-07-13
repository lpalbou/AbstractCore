"""Machine-level data-home registry: discoverability for every byte the framework writes.

The problem this solves (operator-ruled, 2026-07-13): AbstractFramework packages
and apps each co-locate their own data (model caches, prompt-cache artifacts,
run/session stores, logs) — after months of experiments an operator has "no
idea what is here and where I should dig". The ruling (cache-ownership vote +
operator sign-off): data stays co-located per app, but every package REGISTERS
its data home in ONE machine-level registry at first write; one management
surface (the gateway console) enumerates ALL registered homes with live sizes
and safe purge actions; observability apps get read-only telemetry.

Division of ownership:
- core (this module): the registry primitive + safe-purge verbs. Everything
  else reads it; nothing else re-derives it.
- gateway: the one management view (console + CLI parity) calling these verbs.
- observer/continuum: read-only telemetry over `list_data_homes()`.

Safety model:
- `safe_to_purge` is OWNER-DECLARED per row and enforced here: purge REFUSES
  rows declared unsafe, naming the owner and the rule (a purge without owner
  knowledge is how a cleanup amputates an entity home). Entity homes register
  with safe_to_purge=False by construction (the gateway's rows).
- Purge deletes the CONTENTS of a registered home, never the home directory
  itself, and never follows symlinks out of it. Registration survives a purge.
- Unregistered paths cannot be purged through this API at all.
- `unregister_data_home` removes the ROW only; it never touches disk.

Registry file: ``~/.abstractframework/data_registry.json`` (override with the
``ABSTRACTFRAMEWORK_DATA_REGISTRY`` environment variable). Writes are atomic
(temp file + ``os.replace``) under a cross-process advisory lock, so
register-at-first-write from many processes is safe.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..exceptions import ConfigurationError

# The ruled, closed kind set. Governance: it widens only by owning-seat
# proposal + semantics spelling ruling + a note here (rule-4b path).
# - The original five: cache-management vote v-gtytw8 + operator sign-off.
# - "entity-home": semantics pre-ruling 2026-07-13 (commons c1297) — an entity
#   home is a LIFE (memory + book + spark + artifacts + runtime), none of the
#   five; it registers safe_to_purge=False by construction (never-purge rule
#   made registry-visible). Consumers render unknown kinds labeled-unknown,
#   never coerced.
# - "artifacts": semantics ruling 2026-07-13 (commons c1302) — content-addressed
#   artifact stores with rebuildable catalogs; purge semantics are PER-STORE
#   (prunable run-media vs load-bearing stores are separate rows, never one row
#   with a footnote). An entity home's artifacts/ stays INSIDE its entity-home
#   row — a life's verbatims are part of the life, never a separate artifacts row.
DATA_HOME_KINDS = (
    "model-cache", "prompt-cache", "runs", "sessions", "logs", "entity-home", "artifacts",
)

REGISTRY_SCHEMA_VERSION = 1

_REGISTRY_ENV = "ABSTRACTFRAMEWORK_DATA_REGISTRY"
_LOCK_SUFFIX = ".lock"
_LOCK_TIMEOUT_S = 10.0


class DataRegistryError(ConfigurationError):
    """Raised for invalid registry operations (unknown home, unsafe purge, bad kind)."""


@dataclass(frozen=True)
class DataHome:
    """One registered data home (a directory some package/app writes)."""

    name: str
    path: str
    kind: str
    owner: str
    safe_to_purge: bool
    description: str = ""
    registered_at: str = ""
    updated_at: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def registry_path() -> Path:
    """The registry file location (env-overridable for tests/deployments)."""
    override = str(os.environ.get(_REGISTRY_ENV, "") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".abstractframework" / "data_registry.json"


# ---------------------------------------------------------------------------
# Cross-process advisory lock.
#
# POSIX: fcntl.flock on a persistent lock file — kernel-released on process
# death, so there is no stale-lock state and no sweep race (two processes
# "sweeping" a stale O_EXCL lock file can BOTH acquire — adversarial find).
# Windows (no fcntl): msvcrt.locking on the same file.
# ---------------------------------------------------------------------------

class _RegistryLock:
    def __init__(self, target: Path):
        self._lock_file = target.with_name(target.name + _LOCK_SUFFIX)
        self._fd: Optional[int] = None

    def __enter__(self) -> "_RegistryLock":
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_file), os.O_CREAT | os.O_RDWR)
        try:
            self._acquire(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def _acquire(self, fd: int) -> None:
        deadline = time.time() + _LOCK_TIMEOUT_S
        try:
            import fcntl

            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return
                except OSError:
                    if time.time() > deadline:
                        raise DataRegistryError(
                            f"Data registry is locked (lock file: {self._lock_file}) — "
                            f"another process held it for >{_LOCK_TIMEOUT_S}s."
                        )
                    time.sleep(0.05)
        except ImportError:
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    if time.time() > deadline:
                        raise DataRegistryError(
                            f"Data registry is locked (lock file: {self._lock_file}) — "
                            f"another process held it for >{_LOCK_TIMEOUT_S}s."
                        )
                    time.sleep(0.05)

    def __exit__(self, *exc: Any) -> None:
        if self._fd is None:
            return
        try:
            try:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except ImportError:
                import msvcrt

                try:
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": REGISTRY_SCHEMA_VERSION, "homes": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        # Never silently regenerate over a corrupt registry (config-manager
        # lesson): refuse loudly so the operator can inspect/repair.
        raise DataRegistryError(
            f"Data registry at {path} is unreadable ({e}). Repair or remove it explicitly; "
            f"it will not be silently overwritten."
        ) from e
    if not isinstance(data, dict) or not isinstance(data.get("homes"), dict):
        raise DataRegistryError(
            f"Data registry at {path} has an unexpected shape; expected an object with a "
            f"'homes' mapping. Repair or remove it explicitly."
        )
    return data


def _save_raw(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{int(time.time_ns())}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _row_from_dict(raw: Dict[str, Any]) -> DataHome:
    return DataHome(
        name=str(raw.get("name") or ""),
        path=str(raw.get("path") or ""),
        kind=str(raw.get("kind") or ""),
        owner=str(raw.get("owner") or ""),
        safe_to_purge=bool(raw.get("safe_to_purge", False)),
        description=str(raw.get("description") or ""),
        registered_at=str(raw.get("registered_at") or ""),
        updated_at=str(raw.get("updated_at") or ""),
        meta=dict(raw.get("meta") or {}),
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_data_home(
    name: str,
    *,
    path: str,
    kind: str,
    owner: str,
    safe_to_purge: bool,
    description: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> DataHome:
    """Register (or idempotently update) a data home. The register-at-first-write verb.

    Call this when your package/app first writes to a data directory.
    Re-registering the same name upserts the row (path/kind/owner/safety may
    be corrected by the owner); `registered_at` is preserved.
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        raise DataRegistryError("Data home registration requires a non-empty name.")
    clean_kind = str(kind or "").strip().lower()
    if clean_kind not in DATA_HOME_KINDS:
        raise DataRegistryError(
            f"Unknown data-home kind '{kind}'. The ruled set is: {', '.join(DATA_HOME_KINDS)}. "
            f"Kind widenings go through the vocabulary registry, not ad-hoc strings."
        )
    clean_owner = str(owner or "").strip()
    if not clean_owner:
        raise DataRegistryError("Data home registration requires a non-empty owner (package/app id).")
    resolved = Path(str(path or "")).expanduser()
    if not str(resolved).strip() or str(resolved) in ("/", os.path.sep):
        raise DataRegistryError("Data home registration requires a real directory path.")
    resolved = resolved.resolve(strict=False)
    home_dir = Path.home().resolve(strict=False)
    if resolved == home_dir or str(resolved) == resolved.anchor:
        raise DataRegistryError(
            f"Refusing to register '{resolved}' — a data home must be a dedicated directory, "
            f"never the user home or a filesystem root."
        )

    reg_file = registry_path()
    reg_resolved = reg_file.resolve(strict=False)
    if resolved == reg_resolved.parent or resolved in reg_resolved.parents:
        raise DataRegistryError(
            f"Refusing to register '{resolved}' — it contains the data registry itself "
            f"({reg_resolved}); purging it would destroy the registry."
        )

    with _RegistryLock(reg_file):
        data = _load_raw(reg_file)
        homes: Dict[str, Any] = data["homes"]
        # Nesting guard (P0): one home purged as "safe" must never be able to
        # eat another home's data — an ancestor row bypasses the child row's
        # owner-declared safe_to_purge (entity-home amputation class). Refuse
        # ancestor/descendant overlap with any OTHER registered home.
        for other_name, other_raw in homes.items():
            if other_name == clean_name or not isinstance(other_raw, dict):
                continue
            other_path = Path(str(other_raw.get("path") or ""))
            if not str(other_path):
                continue
            if resolved == other_path or resolved in other_path.parents or other_path in resolved.parents:
                raise DataRegistryError(
                    f"Refusing to register '{clean_name}' at '{resolved}': it overlaps the "
                    f"registered home '{other_name}' at '{other_path}' (ancestor/descendant). "
                    f"Nested homes would let a purge of one bypass the other's "
                    f"safe_to_purge declaration."
                )
        prior = homes.get(clean_name) if isinstance(homes.get(clean_name), dict) else None
        row = DataHome(
            name=clean_name,
            path=str(resolved),
            kind=clean_kind,
            owner=clean_owner,
            safe_to_purge=bool(safe_to_purge),
            description=str(description or ""),
            registered_at=str(prior.get("registered_at")) if prior and prior.get("registered_at") else _now_iso(),
            updated_at=_now_iso(),
            meta=dict(meta or {}),
        )
        homes[clean_name] = row.to_dict()
        data["version"] = REGISTRY_SCHEMA_VERSION
        _save_raw(reg_file, data)
    return row


def get_data_home(name: str) -> Optional[DataHome]:
    """One registered row by name (None when absent)."""
    data = _load_raw(registry_path())
    raw = data["homes"].get(str(name or "").strip())
    return _row_from_dict(raw) if isinstance(raw, dict) else None


def list_data_homes(*, include_sizes: bool = False) -> List[Dict[str, Any]]:
    """All registered rows (dicts, JSON-ready). With `include_sizes`, each row
    gains `size_bytes` (recursive, symlinks not followed; None when the path
    is missing) and `exists`."""
    data = _load_raw(registry_path())
    out: List[Dict[str, Any]] = []
    for _, raw in sorted(data["homes"].items()):
        if not isinstance(raw, dict):
            continue
        row = _row_from_dict(raw).to_dict()
        if include_sizes:
            p = Path(row["path"])
            row["exists"] = p.is_dir()
            row["size_bytes"] = _tree_size(p) if p.is_dir() else None
        out.append(row)
    return out


def data_home_size(name: str) -> Optional[int]:
    """Recursive size in bytes of a registered home (None when path missing)."""
    home = get_data_home(name)
    if home is None:
        raise DataRegistryError(f"Unknown data home '{name}' — not in the registry.")
    p = Path(home.path)
    return _tree_size(p) if p.is_dir() else None


def unregister_data_home(name: str) -> bool:
    """Remove a row from the registry. NEVER touches disk contents."""
    clean = str(name or "").strip()
    reg_file = registry_path()
    with _RegistryLock(reg_file):
        data = _load_raw(reg_file)
        if clean not in data["homes"]:
            return False
        del data["homes"][clean]
        _save_raw(reg_file, data)
    return True


def purge_data_home(name: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Delete the CONTENTS of a registered, owner-declared-safe data home.

    Refusal lattice (loud, named):
    - unknown name → refuse (unregistered data cannot be purged via this API);
    - `safe_to_purge=False` → refuse, naming the owner and the rule;
    - path missing → no-op accounting (nothing to purge);
    - the home DIRECTORY itself is never deleted (registration survives);
    - symlinks inside are unlinked, never followed (nothing outside the home
      can be reached through this verb).

    `dry_run=True` returns the would-purge accounting without deleting —
    the console's confirm-dialog read.
    """
    home = get_data_home(name)
    if home is None:
        raise DataRegistryError(
            f"Refusing to purge '{name}': not in the data registry. Only registered homes "
            f"can be purged through this API."
        )
    if not home.safe_to_purge:
        raise DataRegistryError(
            f"Refusing to purge '{home.name}': its owner ({home.owner}) declared "
            f"safe_to_purge=false. Purging owner-protected data (e.g. entity homes) is "
            f"structurally refused here — if this data must go, that is the owner's "
            f"designed lifecycle act, not a cache cleanup."
        )

    root = Path(home.path)
    accounting = {
        "name": home.name,
        "path": str(root),
        "dry_run": bool(dry_run),
        "files_deleted": 0,
        "dirs_deleted": 0,
        "symlinks_removed": 0,
        "bytes_freed": 0,
        "errors": [],
        "skipped_protected": [],
    }
    if not root.is_dir():
        return accounting

    # Symlink-swap refusal (TOCTOU class): the path was resolved at REGISTER
    # time; if any component has since been replaced by a symlink, os.walk
    # would follow it into foreign territory (walk always scandirs `top`,
    # followlinks=False notwithstanding). Refuse loudly instead of deleting
    # through the swap.
    if root.is_symlink() or os.path.realpath(root) != home.path:
        raise DataRegistryError(
            f"Refusing to purge '{home.name}': its path '{root}' no longer resolves to the "
            f"registered location '{home.path}' (now: '{os.path.realpath(root)}'). The path "
            f"was replaced after registration — re-register the home to purge it."
        )

    # Purge-time belt for hand-edited registries: never delete another
    # registered home's subtree, and never the registry file/lock themselves.
    protected: List[Path] = [registry_path().resolve(strict=False)]
    protected.append(protected[0].with_name(protected[0].name + _LOCK_SUFFIX))
    for other in list_data_homes():
        if other["name"] != home.name:
            protected.append(Path(other["path"]))

    def _under_protected(candidate: Path) -> bool:
        """Candidate IS a protected path or lives inside one — never delete."""
        for p in protected:
            if candidate == p or p in candidate.parents:
                return True
        return False

    def _contains_protected(candidate: Path) -> bool:
        """Candidate is an ANCESTOR of a protected path — its own files may go,
        but the directory itself must survive (rmdir would require deleting
        the protected subtree first)."""
        for p in protected:
            if candidate in p.parents:
                return True
        return False

    root_resolved = root.resolve(strict=False)
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        base = Path(dirpath)
        # Never operate outside the resolved root (symlinked dirs are not walked
        # with followlinks=False, but keep the belt anyway).
        try:
            base_resolved = base.resolve(strict=False)
            if root_resolved != base_resolved and root_resolved not in base_resolved.parents:
                continue
        except OSError:
            continue
        if _under_protected(base):
            accounting["skipped_protected"].append(str(base))
            continue
        for fname in filenames:
            fpath = base / fname
            if _under_protected(fpath):
                accounting["skipped_protected"].append(str(fpath))
                continue
            try:
                is_link = fpath.is_symlink()
                size = fpath.lstat().st_size
                if not dry_run:
                    fpath.unlink()
                if is_link:
                    accounting["symlinks_removed"] += 1
                else:
                    accounting["files_deleted"] += 1
                accounting["bytes_freed"] += int(size)
            except OSError as e:
                accounting["errors"].append(f"{fpath}: {e}")
        for dname in dirnames:
            dpath = base / dname
            if _under_protected(dpath) or _contains_protected(dpath):
                accounting["skipped_protected"].append(str(dpath))
                continue
            try:
                if dpath.is_symlink():
                    if not dry_run:
                        dpath.unlink()
                    accounting["symlinks_removed"] += 1
                    continue
                if not dry_run:
                    dpath.rmdir()
                accounting["dirs_deleted"] += 1
            except OSError as e:
                accounting["errors"].append(f"{dpath}: {e}")
    return accounting


def _tree_size(root: Path) -> int:
    """Recursive apparent size (sum of lstat sizes; symlinks not followed).

    Hardlinked files count once per NAME (no inode dedup) — an estimate for a
    management view, not `du` disk-usage accounting.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for fname in filenames:
            try:
                total += (base / fname).lstat().st_size
            except OSError:
                continue
    return int(total)


def register_core_data_homes() -> List[DataHome]:
    """Register the data homes CORE knows about on this machine (idempotent).

    - Hugging Face hub cache (model downloads core triggers): safe to purge —
      models re-download on demand; the description carries the cost warning.
    - LM Studio model directory when present: REPORT-ONLY (safe_to_purge=False)
      — that directory belongs to LM Studio, never touched (agreed guardrail).
    """
    from .structured_logging import get_logger

    logger = get_logger("abstractcore.data_registry")
    rows: List[DataHome] = []
    try:
        # huggingface_hub precedence: HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE
        # (legacy) > $HF_HOME/hub > ~/.cache/huggingface/hub.
        hub_env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if hub_env:
            hub = Path(hub_env).expanduser()
        else:
            hf_home = os.environ.get("HF_HOME")
            hub = Path(hf_home).expanduser() / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
        if hub.is_dir():
            rows.append(register_data_home(
                "huggingface-hub-cache",
                path=str(hub),
                kind="model-cache",
                owner="abstractcore",
                safe_to_purge=True,
                description=(
                    "Hugging Face hub model cache (content-addressed). Safe to purge: models "
                    "re-download on demand, but large models cost bandwidth/time to restore."
                ),
            ))
    except DataRegistryError:
        raise
    except Exception as e:
        logger.warning(f"#FALLBACK: could not probe/register the Hugging Face hub cache: {e}")
    try:
        lms = Path.home() / ".lmstudio" / "models"
        if not lms.is_dir():
            legacy = Path.home() / ".cache" / "lm-studio" / "models"
            lms = legacy if legacy.is_dir() else lms
        if lms.is_dir():
            rows.append(register_data_home(
                "lmstudio-models",
                path=str(lms),
                kind="model-cache",
                owner="lmstudio",
                safe_to_purge=False,
                description=(
                    "LM Studio's own model directory — reported for visibility, never purged "
                    "by the framework (manage it from LM Studio)."
                ),
            ))
    except DataRegistryError:
        raise
    except Exception as e:
        logger.warning(f"#FALLBACK: could not probe/register the LM Studio model dir: {e}")
    return rows
