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

# The ruled, closed kind set (widenings go through the semantics registry).
DATA_HOME_KINDS = ("model-cache", "prompt-cache", "runs", "sessions", "logs")

REGISTRY_SCHEMA_VERSION = 1

_REGISTRY_ENV = "ABSTRACTFRAMEWORK_DATA_REGISTRY"
_LOCK_SUFFIX = ".lock"
_LOCK_TIMEOUT_S = 10.0
_LOCK_STALE_S = 60.0


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
# Cross-process advisory lock (portable: O_CREAT|O_EXCL lock file + stale sweep)
# ---------------------------------------------------------------------------

class _RegistryLock:
    def __init__(self, target: Path):
        self._lock_file = target.with_name(target.name + _LOCK_SUFFIX)

    def __enter__(self) -> "_RegistryLock":
        deadline = time.time() + _LOCK_TIMEOUT_S
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self._lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                finally:
                    os.close(fd)
                return self
            except FileExistsError:
                # Stale-lock sweep: a crashed process must not wedge the registry.
                try:
                    age = time.time() - self._lock_file.stat().st_mtime
                    if age > _LOCK_STALE_S:
                        self._lock_file.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise DataRegistryError(
                        f"Data registry is locked (lock file: {self._lock_file}). "
                        f"Another process held it for >{_LOCK_TIMEOUT_S}s; if no process is "
                        f"running, delete the lock file."
                    )
                time.sleep(0.05)

    def __exit__(self, *exc: Any) -> None:
        try:
            self._lock_file.unlink(missing_ok=True)
        except OSError:
            pass


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
    if resolved == home_dir or resolved == resolved.anchor or str(resolved) == str(Path(resolved.anchor)):
        raise DataRegistryError(
            f"Refusing to register '{resolved}' — a data home must be a dedicated directory, "
            f"never the user home or a filesystem root."
        )

    reg_file = registry_path()
    with _RegistryLock(reg_file):
        data = _load_raw(reg_file)
        homes: Dict[str, Any] = data["homes"]
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
        "bytes_freed": 0,
        "errors": [],
    }
    if not root.is_dir():
        return accounting

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
        for fname in filenames:
            fpath = base / fname
            try:
                size = fpath.lstat().st_size
                if not dry_run:
                    fpath.unlink()
                accounting["files_deleted"] += 1
                accounting["bytes_freed"] += int(size)
            except OSError as e:
                accounting["errors"].append(f"{fpath}: {e}")
        for dname in dirnames:
            dpath = base / dname
            try:
                if dpath.is_symlink():
                    if not dry_run:
                        dpath.unlink()
                    accounting["files_deleted"] += 1
                    continue
                if not dry_run:
                    dpath.rmdir()
                accounting["dirs_deleted"] += 1
            except OSError as e:
                accounting["errors"].append(f"{dpath}: {e}")
    return accounting


def _tree_size(root: Path) -> int:
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
    rows: List[DataHome] = []
    try:
        hf_home = os.environ.get("HF_HOME")
        hub = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"
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
    except Exception:
        pass
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
    except Exception:
        pass
    return rows
