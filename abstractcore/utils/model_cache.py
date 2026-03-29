from __future__ import annotations

import os
import platform
import struct
from pathlib import Path
from typing import BinaryIO, Optional, Sequence


def _dedupe_existing_dirs(candidates: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            p2 = p.expanduser()
        except Exception:
            p2 = p
        key = str(p2)
        if key in seen:
            continue
        seen.add(key)
        try:
            if p2.is_dir():
                out.append(p2)
        except Exception:
            continue
    return out


def default_hf_hub_cache_dirs() -> list[Path]:
    """Return candidate HuggingFace Hub cache directories (best-effort)."""
    candidates: list[Path] = []

    # Explicit env vars.
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            candidates.append(Path(v.strip()))

    # HF_HOME implies <HF_HOME>/hub.
    hf_home = os.getenv("HF_HOME")
    if isinstance(hf_home, str) and hf_home.strip():
        candidates.append(Path(hf_home.strip()) / "hub")

    # Prefer huggingface_hub's constant when available.
    try:  # pragma: no cover
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore

        candidates.append(Path(str(HF_HUB_CACHE)))
    except Exception:
        pass

    # Common default.
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    return _dedupe_existing_dirs(candidates)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text()
        except Exception:
            return ""


def resolve_hf_snapshot_dir(
    repo_id: str,
    *,
    cache_dirs: Optional[Sequence[Path]] = None,
    revision: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a HuggingFace repo id to a local snapshot directory (cache-only).

    This performs no network calls; it only inspects local cache folders.
    """
    s = str(repo_id or "").strip().strip("/")
    if "/" not in s:
        return None
    folder = "models--" + s.replace("/", "--")
    bases = list(cache_dirs) if cache_dirs is not None else default_hf_hub_cache_dirs()

    best: Optional[Path] = None
    best_mtime: float = -1.0

    for base in bases:
        model_dir = base / folder
        snaps_dir = model_dir / "snapshots"
        if not snaps_dir.is_dir():
            continue

        # Explicit revision / commit hash.
        if isinstance(revision, str) and revision.strip():
            cand = snaps_dir / revision.strip()
            if cand.is_dir():
                return cand

        # Respect refs/main or refs/master when present.
        refs_dir = model_dir / "refs"
        for ref in ("main", "master"):
            rev = _read_text(refs_dir / ref).strip()
            if not rev:
                continue
            cand = snaps_dir / rev
            if cand.is_dir():
                return cand

        # Fallback: pick the most recently modified snapshot dir.
        try:
            snapshot_dirs = [d for d in snaps_dir.iterdir() if d.is_dir()]
        except Exception:
            snapshot_dirs = []
        for d in snapshot_dirs:
            try:
                m = float(d.stat().st_mtime)
            except Exception:
                continue
            if m > best_mtime:
                best_mtime = m
                best = d

    return best


def default_lmstudio_model_dirs() -> list[Path]:
    """Return candidate LM Studio model directories (best-effort).

    LM Studio has used multiple locations across versions/platforms; we check a
    few common ones plus env overrides.
    """
    candidates: list[Path] = []

    for k in ("LMSTUDIO_MODELS_DIR", "LMSTUDIO_MODEL_DIR", "LM_STUDIO_MODELS_DIR"):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            candidates.append(Path(v.strip()))

    home = Path.home()

    # Newer LM Studio builds commonly use ~/.lmstudio/models.
    candidates.append(home / ".lmstudio" / "models")

    # Older/macOS default.
    if platform.system().lower() == "darwin":
        candidates.append(home / "Library" / "Application Support" / "LM Studio" / "models")

    # Common Linux defaults.
    if platform.system().lower() == "linux":
        candidates.append(home / ".cache" / "lm-studio" / "models")
        candidates.append(home / ".cache" / "lmstudio" / "models")

    # Windows defaults.
    if platform.system().lower() == "windows":  # pragma: no cover
        local = os.getenv("LOCALAPPDATA") or ""
        roaming = os.getenv("APPDATA") or ""
        if local:
            candidates.append(Path(local) / "LM Studio" / "models")
        if roaming:
            candidates.append(Path(roaming) / "LM Studio" / "models")

    return _dedupe_existing_dirs(candidates)


def _find_child_dir_ci(parent: Path, child_name: str) -> Optional[Path]:
    target = str(child_name or "").strip().lower()
    if not target:
        return None
    try:
        for entry in parent.iterdir():
            try:
                if entry.is_dir() and entry.name.lower() == target:
                    return entry
            except Exception:
                continue
    except Exception:
        return None
    return None


def resolve_lmstudio_model_dir(
    model_id: str, *, base_dirs: Optional[Sequence[Path]] = None
) -> Optional[Path]:
    """Resolve an org/model id to a local LM Studio model directory (cache-only)."""
    s = str(model_id or "").strip().strip("/")
    if "/" not in s:
        return None
    org, name = s.split("/", 1)
    if not org or not name:
        return None

    bases = list(base_dirs) if base_dirs is not None else default_lmstudio_model_dirs()
    for base in bases:
        direct = base / org / name
        if direct.is_dir():
            return direct

        org_dir = _find_child_dir_ci(base, org)
        if org_dir is None:
            continue
        name_dir = _find_child_dir_ci(org_dir, name)
        if name_dir is not None:
            return name_dir

    return None


def resolve_lmstudio_hub_manifest(model_id: str) -> Optional[Path]:
    """Resolve an LM Studio Hub model id to a local manifest.json (best-effort)."""
    s = str(model_id or "").strip().strip("/")
    if "/" not in s:
        return None
    org, name = s.split("/", 1)
    if not org or not name:
        return None

    base = Path.home() / ".lmstudio" / "hub" / "models"
    if not base.is_dir():
        return None

    # Case-insensitive org/name matching to support case-sensitive filesystems.
    org_dir = _find_child_dir_ci(base, org) or (base / org if (base / org).is_dir() else None)
    if org_dir is None:
        return None
    name_dir = _find_child_dir_ci(org_dir, name) or (org_dir / name if (org_dir / name).is_dir() else None)
    if name_dir is None:
        return None

    manifest = name_dir / "manifest.json"
    return manifest if manifest.is_file() else None


_GGUF_MAGIC = b"GGUF"

# https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/constants.py
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

_GGUF_FIXED_SIZES: dict[int, int] = {
    _GGUF_TYPE_UINT8: 1,
    _GGUF_TYPE_INT8: 1,
    _GGUF_TYPE_BOOL: 1,
    _GGUF_TYPE_UINT16: 2,
    _GGUF_TYPE_INT16: 2,
    _GGUF_TYPE_UINT32: 4,
    _GGUF_TYPE_INT32: 4,
    _GGUF_TYPE_FLOAT32: 4,
    _GGUF_TYPE_UINT64: 8,
    _GGUF_TYPE_INT64: 8,
    _GGUF_TYPE_FLOAT64: 8,
}


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError("Unexpected EOF")
    return b


def _read_u32(f: BinaryIO) -> int:
    return int(struct.unpack("<I", _read_exact(f, 4))[0])


def _read_u64(f: BinaryIO) -> int:
    return int(struct.unpack("<Q", _read_exact(f, 8))[0])


def _read_gguf_string(f: BinaryIO) -> str:
    n = _read_u64(f)
    # Defensive guard against corrupt files.
    if n < 0 or n > 512 * 1024 * 1024:
        raise ValueError("Unreasonable GGUF string length")
    return _read_exact(f, n).decode("utf-8", errors="ignore")


def _skip_gguf_value(f: BinaryIO, value_type: int) -> None:
    if value_type == _GGUF_TYPE_STRING:
        _read_gguf_string(f)
        return

    if value_type == _GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        length = _read_u64(f)
        if elem_type == _GGUF_TYPE_STRING:
            for _ in range(length):
                _read_gguf_string(f)
            return

        elem_size = _GGUF_FIXED_SIZES.get(elem_type)
        if elem_size is None:
            raise ValueError(f"Unsupported GGUF array element type: {elem_type}")
        f.seek(elem_size * length, os.SEEK_CUR)
        return

    fixed = _GGUF_FIXED_SIZES.get(value_type)
    if fixed is None:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")
    f.seek(fixed, os.SEEK_CUR)


def read_gguf_architecture(path: Path) -> Optional[str]:
    """Read `general.architecture` from a GGUF file (best-effort, cache-only).

    Returns None if the file isn't GGUF or if the key isn't present.
    """
    try:
        with path.open("rb") as f:
            if _read_exact(f, 4) != _GGUF_MAGIC:
                return None
            _ = _read_u32(f)  # version
            _ = _read_u64(f)  # tensor_count
            kv_count = _read_u64(f)

            for _ in range(kv_count):
                key = _read_gguf_string(f)
                value_type = _read_u32(f)
                if key == "general.architecture" and value_type == _GGUF_TYPE_STRING:
                    v = _read_gguf_string(f).strip()
                    return v or None
                _skip_gguf_value(f, value_type)
    except Exception:
        return None
    return None
