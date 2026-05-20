"""Dependency-light helpers for local cached-vision catalog snapshots."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["get_local_vision_cache_catalog"]


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _framework_candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        try:
            root = candidate.expanduser().resolve()
        except Exception:
            continue
        if root not in roots:
            roots.append(root)
        if (root / "runtime").is_dir() or (root / "untracked").is_dir():
            parent = root.parent
            if parent not in roots:
                roots.append(parent)
    return roots[:16]


def _default_hf_hub_cache_dirs() -> list[Path]:
    dirs: list[str] = []
    for key in ("HF_HUB_CACHE", "HF_HUB_CACHE_DIR", "ABSTRACTCORE_VISION_HF_HUB_CACHE", "ABSTRACTVISION_HF_HUB_CACHE"):
        value = _env(key)
        if value:
            dirs.append(value)

    hf_home = _env("HF_HOME")
    if hf_home:
        dirs.append(str(Path(hf_home).expanduser() / "hub"))

    for key in ("TRANSFORMERS_CACHE", "DIFFUSERS_CACHE"):
        value = _env(key)
        if value:
            dirs.append(value)

    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore

        dirs.append(str(HF_HUB_CACHE))
    except Exception:
        dirs.append(str(Path.home() / ".cache" / "huggingface" / "hub"))

    for root in _framework_candidate_roots():
        dirs.append(str(root / "runtime" / "hf-hub"))
        quarantine = root / "runtime" / "model-quarantine"
        try:
            if quarantine.is_dir():
                dirs.extend(str(path / "hf-hub") for path in quarantine.iterdir() if path.is_dir())
        except Exception:
            pass

    out: list[Path] = []
    seen: set[str] = set()
    for raw_dir in dirs:
        path = Path(raw_dir).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            out.append(path)
    return out


def _default_local_diffusers_model_dirs() -> list[Path]:
    dirs: list[str] = []
    for key in (
        "ABSTRACTCORE_VISION_MODELS_DIR",
        "ABSTRACTCORE_VISION_MODEL_DIR",
        "ABSTRACTVISION_MODELS_DIR",
        "ABSTRACTVISION_MODEL_DIR",
    ):
        value = _env(key)
        if value:
            dirs.append(value)
    for root in _framework_candidate_roots():
        dirs.append(str(root / "untracked" / "models" / "abstractvision"))
        dirs.append(str(root / "runtime" / "models" / "abstractvision"))
        quarantine = root / "runtime" / "model-quarantine"
        try:
            if quarantine.is_dir():
                dirs.extend(str(path / "models") for path in quarantine.iterdir() if path.is_dir())
        except Exception:
            pass

    out: list[Path] = []
    seen: set[str] = set()
    for raw_dir in dirs:
        path = Path(raw_dir).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            out.append(path)
    return out


_DIFFUSERS_NON_IMAGE_TOKENS = (
    "acestep",
    "ace_step",
    "audio",
    "music",
    "sound",
    "speech",
    "spectrogram",
    "mel",
    "oobleck",
    "audioldm",
    "stableaudio",
    "stable_audio",
    "musicldm",
)

_DIFFUSERS_IMAGE_PIPELINE_TOKENS = (
    "image",
    "stable",
    "flux",
    "qwenimage",
    "qwen_image",
    "qwen-image",
    "pixart",
    "kolors",
    "kandinsky",
    "wuerstchen",
    "deepfloyd",
    "paint",
    "controlnet",
    "consistency",
    "unclip",
    "hunyuandit",
    "hunyuan",
    "lumina",
    "sana",
    "chroma",
    "omnigen",
    "hidream",
    "cogview",
    "zimage",
    "z-image",
)


def _diffusers_model_index_supports_image(model_index: Path, *, model_id: Optional[str] = None) -> bool:
    try:
        data = json.loads(model_index.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    class_name = str(data.get("_class_name") or "").strip().lower()
    model_s = str(model_id or model_index.parent.name or "").strip().lower()
    try:
        blob = json.dumps(data, sort_keys=True).lower()
    except Exception:
        blob = class_name

    for token in _DIFFUSERS_NON_IMAGE_TOKENS:
        if token in class_name or token in model_s or token in blob:
            return False
    return any(token in class_name or token in model_s for token in _DIFFUSERS_IMAGE_PIPELINE_TOKENS)


def _hf_snapshot_model_indexes(folder: Path) -> list[Path]:
    snapshots = folder / "snapshots"
    try:
        if not snapshots.is_dir():
            return []
        return [
            snap.joinpath("model_index.json")
            for snap in snapshots.iterdir()
            if snap.is_dir() and snap.joinpath("model_index.json").exists()
        ]
    except Exception:
        return []


def _is_hf_model_cached(model_id: str, cache_dirs: list[Path]) -> bool:
    model_s = str(model_id or "").strip()
    if "/" not in model_s:
        return False
    folder = "models--" + model_s.replace("/", "--")
    for base in cache_dirs:
        cached = base / folder
        for model_index in _hf_snapshot_model_indexes(cached):
            if _diffusers_model_index_supports_image(model_index, model_id=model_s):
                return True
    return False


def _discover_cached_hf_diffusers_models(cache_dirs: list[Path]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for base in cache_dirs:
        try:
            candidates = list(base.glob("models--*"))
        except Exception:
            continue
        for folder in candidates:
            name = folder.name
            if not name.startswith("models--"):
                continue
            model_id = name[len("models--") :].replace("--", "/")
            if "/" not in model_id or model_id in seen:
                continue
            if not any(_diffusers_model_index_supports_image(path, model_id=model_id) for path in _hf_snapshot_model_indexes(folder)):
                continue
            seen.add(model_id)
            out.append(model_id)
    return sorted(out)


def _discover_local_diffusers_models(model_dirs: list[Path]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for base in model_dirs:
        try:
            model_indexes = list(base.rglob("model_index.json"))
        except Exception:
            continue
        for model_index in model_indexes:
            if not _diffusers_model_index_supports_image(model_index):
                continue
            folder = model_index.parent
            name = folder.name
            if "__" in name:
                model_id = name.replace("__", "/")
            else:
                model_id = str(folder)
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            out.append(model_id)
    return sorted(out)


def _default_lmstudio_model_dirs() -> list[Path]:
    dirs: list[str] = []
    for key in ("LMSTUDIO_MODELS_DIR", "LMSTUDIO_MODEL_DIR", "LM_STUDIO_MODELS_DIR"):
        value = _env(key)
        if value:
            dirs.append(value)

    sysname = platform.system().lower()
    home = Path.home()
    if sysname == "darwin":
        dirs.append(str(home / "Library" / "Application Support" / "LM Studio" / "models"))
    elif sysname == "linux":
        dirs.append(str(home / ".cache" / "lm-studio" / "models"))
        dirs.append(str(home / ".cache" / "lmstudio" / "models"))
    elif sysname == "windows":
        local = os.getenv("LOCALAPPDATA") or ""
        roaming = os.getenv("APPDATA") or ""
        if local:
            dirs.append(str(Path(local) / "LM Studio" / "models"))
        if roaming:
            dirs.append(str(Path(roaming) / "LM Studio" / "models"))

    out: list[Path] = []
    seen: set[str] = set()
    for raw_dir in dirs:
        path = Path(raw_dir).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            out.append(path)
    return out


def _is_lmstudio_model_cached(model_id: str, cache_dirs: list[Path]) -> bool:
    model_s = str(model_id or "").strip()
    if "/" not in model_s:
        return False
    org, name = model_s.split("/", 1)
    for base in cache_dirs:
        path = base / org / name
        try:
            if path.is_dir() and any(path.iterdir()):
                return True
        except Exception:
            continue
    return False


def _load_vision_model_capabilities_registry() -> Any:
    from abstractvision import VisionModelCapabilitiesRegistry  # type: ignore

    return VisionModelCapabilitiesRegistry


def _cache_dirs_payload(
    hf_dirs: list[Path],
    local_diffusers_dirs: list[Path],
    lmstudio_dirs: list[Path],
) -> Dict[str, list[str]]:
    return {
        "huggingface": [str(path) for path in hf_dirs],
        "local_diffusers": [str(path) for path in local_diffusers_dirs],
        "lmstudio": [str(path) for path in lmstudio_dirs],
    }


def _registry_unavailable_payload(
    hf_dirs: list[Path],
    local_diffusers_dirs: list[Path],
    lmstudio_dirs: list[Path],
    *,
    error: str,
) -> Dict[str, Any]:
    return {
        "models": [],
        "registry_available": False,
        "registry_total": 0,
        "cached_total": 0,
        "cache_dirs": _cache_dirs_payload(hf_dirs, local_diffusers_dirs, lmstudio_dirs),
        "error": error,
    }


def _is_missing_abstractvision_error(exc: BaseException) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        missing_name = str(getattr(exc, "name", "") or "").strip().lower()
        if missing_name == "abstractvision":
            return True
    message = str(exc).strip()
    return message in {
        'No module named "abstractvision"',
        "No module named 'abstractvision'",
    }


def get_local_vision_cache_catalog() -> Dict[str, Any]:
    """Return a JSON-safe snapshot of locally cached vision models."""

    hf_dirs = _default_hf_hub_cache_dirs()
    local_diffusers_dirs = _default_local_diffusers_model_dirs()
    lmstudio_dirs = _default_lmstudio_model_dirs()
    try:
        registry_cls = _load_vision_model_capabilities_registry()
    except ImportError as exc:
        if _is_missing_abstractvision_error(exc):
            return _registry_unavailable_payload(
                hf_dirs,
                local_diffusers_dirs,
                lmstudio_dirs,
                error="AbstractVision is required for vision model registry endpoints. Install `abstractvision`.",
            )
        return _registry_unavailable_payload(
            hf_dirs,
            local_diffusers_dirs,
            lmstudio_dirs,
            error=f"Failed to load AbstractVision registry: {exc}",
        )
    try:
        registry = registry_cls()
    except Exception as exc:
        return _registry_unavailable_payload(
            hf_dirs,
            local_diffusers_dirs,
            lmstudio_dirs,
            error=f"Failed to initialize AbstractVision registry: {exc}",
        )

    model_ids = list(registry.list_models())
    models: list[Dict[str, Any]] = []
    seen_model_ids: set[str] = set()

    for model_id in model_ids:
        spec = registry.get(model_id)
        supported_tasks = sorted(spec.tasks.keys())
        if "text_to_image" not in spec.tasks and "image_to_image" not in spec.tasks:
            continue

        cached_in: list[str] = []
        if _is_hf_model_cached(model_id, hf_dirs):
            cached_in.append("huggingface")
        if _is_lmstudio_model_cached(model_id, lmstudio_dirs):
            cached_in.append("lmstudio")
        if not cached_in:
            continue

        models.append(
            {
                "id": model_id,
                "provider": spec.provider,
                "license": spec.license,
                "tasks": supported_tasks,
                "notes": spec.notes,
                "cached_in": cached_in,
            }
        )
        seen_model_ids.add(str(model_id))

    for model_id in _discover_cached_hf_diffusers_models(hf_dirs):
        if model_id in seen_model_ids:
            continue
        models.append(
            {
                "id": model_id,
                "provider": "huggingface",
                "license": "unknown",
                "tasks": ["image_to_image", "text_to_image"],
                "notes": "Discovered from the local Hugging Face cache (image-capable diffusers model_index.json present).",
                "cached_in": ["huggingface"],
                "discovered": True,
            }
        )
        seen_model_ids.add(model_id)

    for model_id in _discover_local_diffusers_models(local_diffusers_dirs):
        if model_id in seen_model_ids:
            continue
        models.append(
            {
                "id": model_id,
                "provider": "huggingface" if "/" in model_id else "local",
                "license": "unknown",
                "tasks": ["image_to_image", "text_to_image"],
                "notes": "Discovered from a local AbstractVision Diffusers model directory (image-capable model_index.json present).",
                "cached_in": ["local_diffusers"],
                "discovered": True,
            }
        )
        seen_model_ids.add(model_id)

    models.sort(key=lambda item: str(item.get("id") or ""))
    return {
        "models": models,
        "registry_available": True,
        "registry_total": len(model_ids),
        "cached_total": len(models),
        "cache_dirs": _cache_dirs_payload(hf_dirs, local_diffusers_dirs, lmstudio_dirs),
    }
