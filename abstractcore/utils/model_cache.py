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


def hf_hub_cache_dirs() -> list[Path]:
    """Every Hugging Face hub cache a LOOKUP must read (no network), in priority order.

    huggingface_hub's own resolution first (`default_hf_hub_cache_dirs`:
    HF_HUB_CACHE / HF_HOME / its constant -- the cache transformers,
    sentence-transformers and `snapshot_download` read and write), then the
    `cache.huggingface_cache_dir` config key (+ `/hub`), which older setups
    pointed at a non-default location. Existing directories only.

    THE shared answer to "where are the cached models" for every scan and
    resolve in AbstractCore: sites that hard-coded
    `~/.cache/huggingface/hub` ignored a relocated cache and reported a model
    that was right there as missing.
    """
    dirs = list(default_hf_hub_cache_dirs())
    try:
        from ..config import get_config_manager

        configured_raw = str(get_config_manager().config.cache.huggingface_cache_dir or "").strip()
        if configured_raw:
            configured = Path(configured_raw).expanduser() / "hub"
            if configured.is_dir() and all(str(configured) != str(d) for d in dirs):
                dirs.append(configured)
    except Exception:
        pass
    return dirs


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


# Weight files a transformers / PEFT snapshot can carry.
HF_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.msgpack", "*.h5")


def _has_loadable_marker(snapshot: Path) -> bool:
    return (snapshot / "config.json").is_file() or (snapshot / "adapter_config.json").is_file()


def resolve_hf_load_snapshot(
    repo_id: str,
    *,
    cache_dirs: Optional[Sequence[Path]] = None,
) -> Optional[Path]:
    """The cached snapshot directory a LOAD of `repo_id` should read (cache-only).

    Why this exists: transformers resolves a repo id
    offline (`local_files_only=True`) through `refs/main` -> `snapshots/<sha>`.
    AbstractCore's own downloader pins the listed commit
    (`snapshot_download(revision=<sha>)`), and huggingface_hub writes no
    `refs/main` for a revision that already IS a commit hash. Such a snapshot is
    complete on disk yet unreachable by name offline: transformers reports
    "couldn't connect to huggingface.co ... couldn't find them in the cached
    files". Handing the loader this DIRECTORY instead of the id removes the
    `refs` lookup (and every other Hub round trip) from the load.

    Order, per cache directory (see `default_hf_hub_cache_dirs`): the snapshot
    `refs/main` (or `refs/master`) names; else the newest snapshot that holds
    a `config.json` or an `adapter_config.json` (a snapshot holding only a
    README is not a load target). None when nothing usable is cached.
    """
    s = str(repo_id or "").strip().strip("/")
    if "/" not in s:
        return None
    folder = "models--" + s.replace("/", "--")
    bases = list(cache_dirs) if cache_dirs is not None else default_hf_hub_cache_dirs()

    fallback: Optional[Path] = None
    fallback_rank: tuple = (-1, -1.0)
    for base in bases:
        model_dir = Path(base) / folder
        snaps_dir = model_dir / "snapshots"
        if not snaps_dir.is_dir():
            continue
        for ref in ("main", "master"):
            rev = _read_text(model_dir / "refs" / ref).strip()
            if rev and (snaps_dir / rev).is_dir():
                return snaps_dir / rev
        try:
            snapshot_dirs = [d for d in snaps_dir.iterdir() if d.is_dir()]
        except Exception:
            snapshot_dirs = []
        for d in snapshot_dirs:
            try:
                m = float(d.stat().st_mtime)
            except Exception:
                continue
            # A usable snapshot always beats a README-only one; newest wins within a class.
            rank = (1 if _has_loadable_marker(d) else 0, m)
            if rank > fallback_rank:
                fallback, fallback_rank = d, rank
    return fallback


def describe_hf_snapshot(snapshot: Path) -> dict:
    """What a cached snapshot can be loaded as, from its files alone (no network).

    `kind` is "model" (config.json + weights), "adapter" (adapter_config.json,
    no full model), "config_only" (config.json, no weight file) or "unknown"
    (neither config: a partial download, or a repository for another runtime).
    """
    snapshot = Path(snapshot)
    try:
        files = sorted(p.name for p in snapshot.iterdir())
    except Exception:
        files = []
    has_config = (snapshot / "config.json").is_file()
    has_weights = any(any(snapshot.glob(pattern)) for pattern in HF_WEIGHT_PATTERNS)
    adapter_config = snapshot / "adapter_config.json"
    base_model: Optional[str] = None
    if adapter_config.is_file():
        try:
            import json

            base_model = str(json.loads(_read_text(adapter_config)).get("base_model_name_or_path") or "").strip() or None
        except Exception:
            base_model = None
    if has_config and has_weights:
        kind = "model"
    elif adapter_config.is_file():
        kind = "adapter"
    elif has_config:
        kind = "config_only"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "files": files,
        "base_model": base_model,
        "has_tokenizer": (snapshot / "tokenizer_config.json").is_file() or (snapshot / "tokenizer.json").is_file(),
    }


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


_GGUF_SCALAR_FORMATS: dict[int, str] = {
    _GGUF_TYPE_UINT8: "<B",
    _GGUF_TYPE_INT8: "<b",
    _GGUF_TYPE_UINT16: "<H",
    _GGUF_TYPE_INT16: "<h",
    _GGUF_TYPE_UINT32: "<I",
    _GGUF_TYPE_INT32: "<i",
    _GGUF_TYPE_FLOAT32: "<f",
    _GGUF_TYPE_UINT64: "<Q",
    _GGUF_TYPE_INT64: "<q",
    _GGUF_TYPE_FLOAT64: "<d",
    _GGUF_TYPE_BOOL: "<B",
}


def _read_gguf_scalar(f: BinaryIO, value_type: int):
    fmt = _GGUF_SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"Not a scalar GGUF value type: {value_type}")
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]


# `<arch>.`-prefixed header keys carrying KV-cache geometry, keyed by the
# suffix (the architecture prefix varies per model family).
_GGUF_GEOMETRY_SUFFIXES: dict[str, str] = {
    ".block_count": "block_count",
    ".attention.head_count": "head_count",
    ".attention.head_count_kv": "head_count_kv",
    ".attention.key_length": "key_length",
    ".attention.value_length": "value_length",
    ".embedding_length": "embedding_length",
    ".context_length": "context_length",
}


def read_gguf_geometry(path: Path) -> Optional[dict]:
    """Read KV-cache geometry keys from a GGUF header (best-effort, cache-only).

    Same single-pass walk as `read_gguf_architecture`, additionally collecting
    `<arch>.block_count`, `<arch>.attention.head_count[_kv]`,
    `<arch>.attention.key_length` / `.value_length`, `<arch>.embedding_length`,
    and `<arch>.context_length`. Returns a dict with `architecture` plus any
    geometry keys found (values as ints), or None when the file isn't GGUF /
    nothing was readable. Missing keys are absent, never guessed.
    """
    try:
        out: dict = {}
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
                    if v:
                        out["architecture"] = v
                    continue
                field = None
                for suffix, name in _GGUF_GEOMETRY_SUFFIXES.items():
                    if key.endswith(suffix):
                        field = name
                        break
                if field is not None and value_type in _GGUF_SCALAR_FORMATS:
                    try:
                        out[field] = int(_read_gguf_scalar(f, value_type))
                    except (ValueError, TypeError):
                        pass
                    continue
                _skip_gguf_value(f, value_type)
        return out or None
    except Exception:
        return None


def read_gguf_mtp_layers(path: Path) -> Optional[int]:
    """Read `<arch>.nextn_predict_layers` from a GGUF header, or None.

    This is the ONLY honest way to know a GGUF carries a multi-token-prediction
    head. The filename is not evidence in either direction, and both errors
    happen in the wild: `Qwen3.8-27B-Q4_K_M.gguf` carries the head with no "mtp"
    anywhere in its name, while unsloth's `*-MTP-GGUF` repos put "MTP" in the
    REPO name and not in the file's.

    Reads the header only -- a few KB -- so it is safe to call before deciding
    whether to load 17 GB of weights.
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
                # Arch-prefixed key: `qwen35.nextn_predict_layers`,
                # `deepseek2.nextn_predict_layers`, ... so match the suffix
                # rather than enumerating every architecture llama.cpp supports.
                if key.endswith(".nextn_predict_layers") and value_type in (
                    _GGUF_TYPE_UINT8,
                    _GGUF_TYPE_INT8,
                    _GGUF_TYPE_UINT16,
                    _GGUF_TYPE_INT16,
                    _GGUF_TYPE_UINT32,
                    _GGUF_TYPE_INT32,
                    _GGUF_TYPE_UINT64,
                    _GGUF_TYPE_INT64,
                ):
                    size = _GGUF_FIXED_SIZES.get(value_type)
                    if not size:
                        return None
                    raw = _read_exact(f, size)
                    signed = value_type in (
                        _GGUF_TYPE_INT8,
                        _GGUF_TYPE_INT16,
                        _GGUF_TYPE_INT32,
                        _GGUF_TYPE_INT64,
                    )
                    return int.from_bytes(raw, "little", signed=signed)
                _skip_gguf_value(f, value_type)
    except Exception:
        return None
    return None


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
