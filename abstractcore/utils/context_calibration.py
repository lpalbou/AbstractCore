"""Empirical context-window calibration store.

The GGUF loader settles a usable `n_ctx` by walking a ladder of candidate
context sizes and probe-decoding each rung (huggingface_provider._load_gguf_model).
That walk is expensive — every failed rung allocates and tears down a full KV
cache. This store remembers WHERE a (provider, model, hardware) combination
settled so the next construction can try the known-good rung right after the
requested one, instead of re-failing the same rungs.

Honesty rule (ADR 0008): a calibration entry is a HINT, not a fact about the
current process — memory conditions change, so consumers must still probe.
The store never guesses: lookups only return exact-key matches.

Storage: `~/.abstractcore/calibration/context_calibration.json` (override dir
with ABSTRACTCORE_CALIBRATION_DIR). Atomic tmp+os.replace writes under a
cross-process advisory file lock (same mechanics as utils/data_registry.py).
Corrupt files are abandoned and rebuilt (logged once) — calibration is a
cache, never a source of truth. Both public functions are raise-free.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CALIBRATION_DIR_ENV = "ABSTRACTCORE_CALIBRATION_DIR"
_LOCK_SUFFIX = ".lock"
_LOCK_TIMEOUT_S = 5.0
# Cap: keep the file a cache, not an archive. Oldest entries (by ts) drop.
_MAX_ENTRIES = 300

_warned_corrupt = False


def calibration_path() -> Path:
    """The calibration file location (env-overridable for tests/deployments)."""
    override = str(os.environ.get(_CALIBRATION_DIR_ENV, "") or "").strip()
    if override:
        return Path(override).expanduser() / "context_calibration.json"
    return Path.home() / ".abstractcore" / "calibration" / "context_calibration.json"


def gguf_calibration_model_id(path: Any) -> str:
    """Calibration model_id for a resolved GGUF file path.

    The bare basename collides for quant-anonymous multi-part layouts
    (`UD-Q3_K_XL/model-00001-of-00003.gguf` vs `UD-Q4_K_M/model-00001-...`),
    so when the file sits in a SUBDIRECTORY of an HF hub snapshot
    (`.../snapshots/<rev>/<quant-dir>/file.gguf`) the quant dir is included:
    `<quant-dir>/<basename>`. Everywhere else (snapshot root, direct paths,
    LM Studio layouts) the basename alone is the id — including a revision
    hash or arbitrary parent dir would fracture keys across re-downloads.
    The ladder (recording) and the estimator (lookup) MUST share this
    derivation. Never raises.
    """
    try:
        p = Path(str(path))
        if p.parent.parent.parent.name == "snapshots":
            return f"{p.parent.name}/{p.name}"
        return p.name
    except Exception:
        return str(path)


class _CalibrationLock:
    """Cross-process advisory lock (fcntl.flock / msvcrt.locking) — the same
    stale-proof mechanics as data_registry._RegistryLock."""

    def __init__(self, target: Path):
        self._lock_file = target.with_name(target.name + _LOCK_SUFFIX)
        self._fd: Optional[int] = None

    def __enter__(self) -> "_CalibrationLock":
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
                        raise TimeoutError(
                            f"Context calibration store is locked (lock file: {self._lock_file})."
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
                        raise TimeoutError(
                            f"Context calibration store is locked (lock file: {self._lock_file})."
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


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    """Tolerant load: a corrupt/unreadable file starts fresh (logged once).

    Unlike the data registry, calibration is a rebuildable cache — refusing
    loudly would turn a scratch file into a load blocker.
    """
    global _warned_corrupt
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries") if isinstance(data, dict) else None
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, dict)]
    except Exception as e:
        if not _warned_corrupt:
            _warned_corrupt = True
            logger.warning(
                f"Context calibration store at {path} is unreadable ({e}); starting fresh."
            )
        return []
    if not _warned_corrupt:
        _warned_corrupt = True
        logger.warning(
            f"Context calibration store at {path} has an unexpected shape; starting fresh."
        )
    return []


def _save_entries(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{int(time.time_ns())}")
    tmp.write_text(
        json.dumps({"version": 1, "entries": entries}, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _entry_key(entry: Dict[str, Any]) -> tuple:
    def _norm_bytes(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model_id") or "").strip(),
        _norm_bytes(entry.get("device_total_bytes")),
        _norm_bytes(entry.get("ram_total_bytes")),
    )


def record_context_calibration(entry: Dict[str, Any]) -> None:
    """Record (or replace) one calibration entry. Best-effort: never raises.

    Expected entry fields: provider, model_id, requested_context,
    settled_context, device_total_bytes, ram_total_bytes, rungs_tried.
    `ts` is stamped here. An entry with the same key replaces its predecessor;
    the file is capped at a few hundred entries (oldest dropped).
    """
    try:
        if not isinstance(entry, dict):
            return
        provider = str(entry.get("provider") or "").strip().lower()
        model_id = str(entry.get("model_id") or "").strip()
        settled = entry.get("settled_context")
        if not provider or not model_id or not isinstance(settled, int) or settled <= 0:
            return
        stamped = dict(entry)
        stamped["provider"] = provider
        stamped["model_id"] = model_id
        stamped["ts"] = float(time.time())
        path = calibration_path()
        with _CalibrationLock(path):
            entries = _load_entries(path)
            key = _entry_key(stamped)
            entries = [e for e in entries if _entry_key(e) != key]
            entries.append(stamped)
            if len(entries) > _MAX_ENTRIES:
                entries.sort(key=lambda e: float(e.get("ts") or 0.0))
                entries = entries[-_MAX_ENTRIES:]
            _save_entries(path, entries)
    except Exception as e:
        logger.debug(f"record_context_calibration failed (ignored): {e}")


def lookup_context_calibration(
    provider: str,
    model_id: str,
    device_total_bytes: Optional[int],
    ram_total_bytes: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Return the calibration entry for the exact key, or None. Never raises."""
    try:
        probe = {
            "provider": provider,
            "model_id": model_id,
            "device_total_bytes": device_total_bytes,
            "ram_total_bytes": ram_total_bytes,
        }
        key = _entry_key(probe)
        if not key[0] or not key[1]:
            return None
        for entry in _load_entries(calibration_path()):
            if _entry_key(entry) == key:
                return dict(entry)
        return None
    except Exception as e:
        logger.debug(f"lookup_context_calibration failed (ignored): {e}")
        return None
