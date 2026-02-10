"""
OpenAI-compatible vision generation endpoints for AbstractCore Server.

This module is intentionally dependency-light:
- It does not import `abstractvision` unless the endpoints are actually used.

Design notes:
- AbstractCore Server is a gateway; vision generation is delegated to AbstractVision backends.
- This router can act as a thin "vision proxy" to any upstream that implements
  `/images/generations` and `/images/edits`, or run local backends (Diffusers / stable-diffusion.cpp).

Out-of-the-box behavior:
- If `ABSTRACTCORE_VISION_BACKEND` is not set, the router defaults to `auto`.
- In `auto` mode, the backend is inferred per-request:
  - Hugging Face repo ids like `org/model` -> Diffusers backend
  - Local `.gguf` file paths -> stable-diffusion.cpp backend
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shlex
import time
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

try:  # Optional dependency (needed only for multipart parsing).
    import multipart  # type: ignore  # noqa: F401

    _HAS_MULTIPART = True
except Exception:  # pragma: no cover
    _HAS_MULTIPART = False


router = APIRouter(tags=["vision"])

_BACKEND_CACHE_LOCK = threading.Lock()
_BACKEND_CACHE: Dict[Tuple[Any, ...], Tuple[Any, threading.Lock, float]] = {}

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_MODEL_ID: Optional[str] = None
_ACTIVE_BACKEND_KIND: Optional[str] = None
_ACTIVE_BACKEND: Any = None
_ACTIVE_CALL_LOCK: Optional[threading.Lock] = None
_ACTIVE_LOADED_AT_S: Optional[float] = None

_JOBS_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _jobs_max() -> int:
    raw = _env("ABSTRACTCORE_VISION_JOBS_MAX", "8") or "8"
    try:
        n = int(str(raw).strip())
    except Exception:
        n = 8
    return max(1, min(n, 64))


def _jobs_ttl_s() -> float:
    raw = _env("ABSTRACTCORE_VISION_JOBS_TTL_S", "600") or "600"
    try:
        v = float(str(raw).strip())
    except Exception:
        v = 600.0
    return max(10.0, min(v, 24.0 * 3600.0))


def _new_job_id() -> str:
    return uuid.uuid4().hex


def _jobs_cleanup_locked(*, now_s: float) -> None:
    ttl = _jobs_ttl_s()
    # Drop old completed jobs.
    for jid, job in list(_JOBS.items()):
        state = str(job.get("state") or "")
        if state not in {"succeeded", "failed"}:
            continue
        updated = float(job.get("updated_at_s") or 0.0)
        if updated and (now_s - updated) > ttl:
            _JOBS.pop(jid, None)

    # Enforce size bound (drop oldest completed first).
    max_entries = _jobs_max()
    if len(_JOBS) <= max_entries:
        return
    items = sorted(_JOBS.items(), key=lambda kv: float(kv[1].get("updated_at_s") or kv[1].get("created_at_s") or 0.0))
    for jid, job in items:
        if len(_JOBS) <= max_entries:
            break
        state = str(job.get("state") or "")
        if state in {"succeeded", "failed"}:
            _JOBS.pop(jid, None)

    # If still too many (all in-flight), drop the oldest anyway (best-effort).
    if len(_JOBS) > max_entries:
        items = sorted(
            _JOBS.items(),
            key=lambda kv: float(kv[1].get("created_at_s") or 0.0),
        )
        for jid, _job in items[: max(0, len(_JOBS) - max_entries)]:
            _JOBS.pop(jid, None)


def _any_inflight_job_locked() -> bool:
    return any(str(j.get("state") or "") in {"queued", "running"} for j in _JOBS.values())


def _job_update_progress(job_id: str, *, step: Optional[int], total: Optional[int], message: Optional[str] = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        prog = job.get("progress")
        if not isinstance(prog, dict):
            prog = {}
            job["progress"] = prog
        if step is not None:
            prog["step"] = int(step)
        if total is not None:
            prog["total_steps"] = int(total)
        if message is not None:
            prog["message"] = str(message)
        job["updated_at_s"] = time.time()


def _job_finish(job_id: str, *, ok: bool, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["state"] = "succeeded" if ok else "failed"
        job["updated_at_s"] = time.time()
        if ok:
            job["result"] = result
            job.pop("error", None)
        else:
            job["error"] = str(error or "Unknown error")
            job.pop("result", None)


def _get_or_create_cached_backend(key: Tuple[Any, ...], factory):
    with _BACKEND_CACHE_LOCK:
        now = time.time()
        cached = _BACKEND_CACHE.get(key)
        if cached is not None:
            backend, call_lock, _ts = cached
            _BACKEND_CACHE[key] = (backend, call_lock, now)
            return backend, call_lock

        backend = factory()
        call_lock = threading.Lock()
        _BACKEND_CACHE[key] = (backend, call_lock, now)

        max_entries_raw = _env("ABSTRACTCORE_VISION_BACKEND_CACHE_MAX", "4") or "4"
        try:
            max_entries = int(str(max_entries_raw).strip())
        except Exception:
            max_entries = 4
        max_entries = max(1, min(int(max_entries), 64))

        if len(_BACKEND_CACHE) > max_entries:
            # Evict least-recently-used backends (best-effort).
            items = sorted(_BACKEND_CACHE.items(), key=lambda kv: kv[1][2])
            for k, _ in items[: max(0, len(_BACKEND_CACHE) - max_entries)]:
                if k == key:
                    continue
                _BACKEND_CACHE.pop(k, None)

        return backend, call_lock


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _active_state() -> Dict[str, Any]:
    with _ACTIVE_LOCK:
        return {
            "model_id": _ACTIVE_MODEL_ID,
            "backend_kind": _ACTIVE_BACKEND_KIND,
            "loaded_at_s": _ACTIVE_LOADED_AT_S,
            "has_backend": _ACTIVE_BACKEND is not None,
        }


def _get_active_backend() -> Tuple[Optional[str], Optional[str], Any, Optional[threading.Lock]]:
    with _ACTIVE_LOCK:
        return _ACTIVE_MODEL_ID, _ACTIVE_BACKEND_KIND, _ACTIVE_BACKEND, _ACTIVE_CALL_LOCK


def _unload_backend_best_effort(backend: Any) -> None:
    unload = getattr(backend, "unload", None)
    if callable(unload):
        unload()

    # Extra best-effort GC to drop references ASAP.
    try:
        import gc

        gc.collect()
    except Exception:
        pass


def _unload_active_backend() -> None:
    global _ACTIVE_MODEL_ID, _ACTIVE_BACKEND_KIND, _ACTIVE_BACKEND, _ACTIVE_CALL_LOCK, _ACTIVE_LOADED_AT_S
    with _ACTIVE_LOCK:
        backend = _ACTIVE_BACKEND
        call_lock = _ACTIVE_CALL_LOCK
        _ACTIVE_MODEL_ID = None
        _ACTIVE_BACKEND_KIND = None
        _ACTIVE_BACKEND = None
        _ACTIVE_CALL_LOCK = None
        _ACTIVE_LOADED_AT_S = None

    if backend is not None:
        if call_lock is not None:
            with call_lock:
                _unload_backend_best_effort(backend)
        else:
            _unload_backend_best_effort(backend)


def _vision_backend_kind() -> str:
    raw = _env("ABSTRACTCORE_VISION_BACKEND")
    if not raw:
        return "auto"
    v = str(raw).strip().lower()
    if v in {"auto", "default"}:
        return "auto"
    if v in {"openai", "openai-compatible", "openai_compatible", "proxy", "openai_compatible_proxy"}:
        return "openai_compatible_proxy"
    if v in {"diffusers", "hf-diffusers", "huggingface-diffusers"}:
        return "diffusers"
    if v in {"sdcpp", "sd-cpp", "stable-diffusion.cpp", "stable-diffusion-cpp", "stable_diffusion_cpp"}:
        return "sdcpp"
    return v


_KNOWN_MODEL_PREFIXES: set[str] = {
    # AbstractCore providers (model ids often look like provider/model or provider/org/model).
    "openai",
    "anthropic",
    "openrouter",
    "portkey",
    "ollama",
    "lmstudio",
    "vllm",
    "openai-compatible",
    "openai_compatible",
    "huggingface",
    "hf",
    "mlx",
    # Vision backend families (AbstractVision).
    "diffusers",
    "sdcpp",
}


def _split_known_prefix(model: str) -> tuple[Optional[str], str]:
    s = str(model or "").strip()
    if not s or "/" not in s:
        return None, s
    head, tail = s.split("/", 1)
    head_s = head.strip()
    if head_s in _KNOWN_MODEL_PREFIXES:
        return head_s, tail.strip()
    return None, s


def _normalize_request_model_for_backend(request_model: Any) -> Optional[str]:
    """Normalize AbstractCore-style model ids into AbstractVision backend model ids.

    This keeps `/v1/images/*` compatible with AbstractCore's multi-provider model naming:
    - `huggingface/Qwen/Qwen-Image-2512` -> `Qwen/Qwen-Image-2512` (Diffusers)
    - `mlx/Qwen/Qwen-Image-2512` -> `Qwen/Qwen-Image-2512` (Diffusers; device selection is env-driven)

    For non-vision providers (e.g. `openai/gpt-4o-mini`), we intentionally drop the request model unless a
    vision upstream proxy is configured. This enables a "vision default" behavior via env configuration.
    """
    model = str(request_model or "").strip()
    if not model:
        return None

    prefix, rest = _split_known_prefix(model)
    if prefix is None:
        return model

    # Local generation backends: strip prefix and pass through.
    if prefix in {"huggingface", "hf", "mlx", "diffusers", "sdcpp"}:
        return rest or None

    # Other providers: only use the request model when proxying to an upstream images endpoint.
    if _env("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL"):
        return rest or None

    # No upstream: treat as "no request model" so env-configured vision defaults apply.
    return None


def _looks_like_filesystem_path(model: str) -> bool:
    s = str(model or "").strip()
    if not s:
        return False
    if s.startswith(("~", "./", "../")):
        return True
    if s.startswith(("/", "\\")):
        return True
    if s.startswith("file:"):
        return True
    # Windows drive letters (e.g. C:\path\to\file.gguf)
    if len(s) >= 3 and s[1:3] == ":\\":
        return True
    # Common local weight formats (we care most about gguf for stable-diffusion.cpp).
    if s.lower().endswith((".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
        return True
    return False


def _looks_like_hf_repo_id(model: str) -> bool:
    s = str(model or "").strip()
    if not s:
        return False
    if _looks_like_filesystem_path(s):
        return False
    if "://" in s:
        return False
    parts = s.split("/")
    if len(parts) != 2:
        return False
    org, name = parts
    return bool(org and name)


def _infer_backend_kind(request_model: Any) -> str:
    model = str(_normalize_request_model_for_backend(request_model) or "").strip()
    if model:
        if _looks_like_filesystem_path(model):
            return "sdcpp"
        # Default: treat HF-style ids as Diffusers.
        if _looks_like_hf_repo_id(model):
            return "diffusers"
        # Unknown strings: prefer local generation unless proxy is explicitly configured.
        if _env("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL"):
            return "openai_compatible_proxy"
        return "diffusers"

    # No request model: fall back to env configuration.
    if _env("ABSTRACTCORE_VISION_MODEL_ID"):
        return "diffusers"
    if _env("ABSTRACTCORE_VISION_SDCPP_MODEL") or _env("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL"):
        return "sdcpp"
    if _env("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL"):
        return "openai_compatible_proxy"
    return "auto_unconfigured"


def _effective_backend_kind(request_model: Any) -> str:
    env_kind = _vision_backend_kind()
    if env_kind == "auto":
        return _infer_backend_kind(request_model)

    model = str(request_model or "").strip()
    if model and env_kind == "sdcpp" and _looks_like_hf_repo_id(model):
        # Common misconfiguration: user set SDCPP backend but selected a Diffusers model id.
        return "diffusers"
    if model and env_kind == "diffusers" and _looks_like_filesystem_path(model):
        # Common misconfiguration: user set Diffusers backend but passed a local gguf path.
        return "sdcpp"

    return env_kind


def _default_hf_hub_cache_dirs() -> list[Path]:
    dirs: list[str] = []
    # Explicit overrides.
    for k in ("HF_HUB_CACHE", "HF_HUB_CACHE_DIR"):
        v = _env(k)
        if v:
            dirs.append(v)

    # HF_HOME implies <HF_HOME>/hub.
    hf_home = _env("HF_HOME")
    if hf_home:
        dirs.append(str(Path(hf_home).expanduser() / "hub"))

    # Other common env vars used by Transformers/Diffusers. These may or may not be hub-style dirs.
    for k in ("TRANSFORMERS_CACHE", "DIFFUSERS_CACHE"):
        v = _env(k)
        if v:
            dirs.append(v)

    # Default from huggingface_hub if available.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore

        dirs.append(str(HF_HUB_CACHE))
    except Exception:
        # Fallback: common default.
        dirs.append(str(Path.home() / ".cache" / "huggingface" / "hub"))

    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        p = Path(d).expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_dir():
            out.append(p)
    return out


def _is_hf_model_cached(model_id: str, cache_dirs: list[Path]) -> bool:
    s = str(model_id or "").strip()
    if "/" not in s:
        return False
    # HF hub cache uses folder names like: models--org--name
    folder = "models--" + s.replace("/", "--")
    for base in cache_dirs:
        snaps = base / folder / "snapshots"
        try:
            if snaps.is_dir() and any(snaps.iterdir()):
                return True
        except Exception:
            continue
    return False


def _default_lmstudio_model_dirs() -> list[Path]:
    dirs: list[str] = []
    for k in ("LMSTUDIO_MODELS_DIR", "LMSTUDIO_MODEL_DIR", "LM_STUDIO_MODELS_DIR"):
        v = _env(k)
        if v:
            dirs.append(v)

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
    for d in dirs:
        p = Path(d).expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_dir():
            out.append(p)
    return out


def _is_lmstudio_model_cached(model_id: str, cache_dirs: list[Path]) -> bool:
    s = str(model_id or "").strip()
    if "/" not in s:
        return False
    org, name = s.split("/", 1)
    for base in cache_dirs:
        p = base / org / name
        try:
            if p.is_dir() and any(p.iterdir()):
                return True
        except Exception:
            continue
    return False


def _require_upstream_base_url() -> str:
    base_url = _env("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL")
    if not base_url:
        raise HTTPException(
            status_code=501,
            detail=(
                "Vision image endpoints are not configured. "
                "Set ABSTRACTCORE_VISION_UPSTREAM_BASE_URL to an OpenAI-compatible server base URL "
                "(e.g. https://api.openai.com/v1 or http://localhost:1234/v1)."
            ),
        )
    return base_url


def _require_diffusers_model_id(request_model: Any) -> str:
    model_id = str(request_model or _env("ABSTRACTCORE_VISION_MODEL_ID") or "").strip()
    if not model_id:
        raise HTTPException(
            status_code=501,
            detail=(
                "Vision image endpoints are not configured for diffusers mode. "
                "Set ABSTRACTCORE_VISION_MODEL_ID (and optionally ABSTRACTCORE_VISION_BACKEND=diffusers), "
                "or pass `model` in the request."
            ),
        )
    return model_id


def _require_sdcpp_model_or_diffusion_model(request_model: Any) -> Tuple[Optional[str], Optional[str]]:
    req = str(request_model or "").strip()
    if req and not _looks_like_filesystem_path(req):
        raise HTTPException(
            status_code=400,
            detail=(
                "stable-diffusion.cpp backend expects a local model path (typically a .gguf file). "
                f"Got model={req!r}. If you intended to run a Hugging Face model id (e.g. 'runwayml/stable-diffusion-v1-5'), "
                "use the Diffusers backend (or set ABSTRACTCORE_VISION_BACKEND=auto)."
            ),
        )

    env_model = str(_env("ABSTRACTCORE_VISION_SDCPP_MODEL") or "").strip()
    env_diffusion = str(_env("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL") or "").strip()

    # Request model overrides env defaults.
    if req:
        # If the user configured component paths, treat the request as diffusion_model (component mode).
        component_mode = any(
            str(_env(k) or "").strip()
            for k in (
                "ABSTRACTCORE_VISION_SDCPP_VAE",
                "ABSTRACTCORE_VISION_SDCPP_LLM",
                "ABSTRACTCORE_VISION_SDCPP_LLM_VISION",
                "ABSTRACTCORE_VISION_SDCPP_CLIP_L",
                "ABSTRACTCORE_VISION_SDCPP_CLIP_G",
                "ABSTRACTCORE_VISION_SDCPP_T5XXL",
            )
        )
        return (None, req) if component_mode else (req, None)

    if env_model:
        return env_model, None
    if env_diffusion:
        return None, env_diffusion

    raise HTTPException(
        status_code=501,
        detail=(
            "Vision image endpoints are not configured for sdcpp mode. "
            "Set ABSTRACTCORE_VISION_SDCPP_MODEL (full model) or ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL "
            "(component mode), or pass a local .gguf path as `model` in the request."
        ),
    )


def _import_abstractvision() -> Tuple[Any, ...]:
    try:
        from abstractvision.backends import (  # type: ignore
            HuggingFaceDiffusersBackendConfig,
            HuggingFaceDiffusersVisionBackend,
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from abstractvision.errors import OptionalDependencyMissingError  # type: ignore
        from abstractvision.types import ImageEditRequest, ImageGenerationRequest  # type: ignore
    except Exception as e:  # pragma: no cover
        import sys

        raise HTTPException(
            status_code=501,
            detail=(
                "AbstractVision is required for vision generation endpoints. "
                "Install it into the same environment running the server (and use `python -m uvicorn ...` "
                "to ensure you are using the same interpreter). "
                f"(python={sys.executable})"
            ),
        ) from e
    return (
        OpenAICompatibleBackendConfig,
        OpenAICompatibleVisionBackend,
        HuggingFaceDiffusersBackendConfig,
        HuggingFaceDiffusersVisionBackend,
        StableDiffusionCppBackendConfig,
        StableDiffusionCppVisionBackend,
        OptionalDependencyMissingError,
        (ImageGenerationRequest, ImageEditRequest),
    )


def _parse_size(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if value is None:
        return None, None
    s = str(value).strip().lower()
    if not s:
        return None, None
    if "x" not in s:
        return None, None
    w_s, h_s = s.split("x", 1)
    try:
        return int(w_s), int(h_s)
    except Exception:
        return None, None


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except Exception:
        return None


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return v
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _resolve_backend(request_model: Any):
    normalized = _normalize_request_model_for_backend(request_model)
    req_model = str(normalized or "").strip()
    backend_kind = _effective_backend_kind(normalized)

    # Important: return "not configured" errors without requiring optional deps.
    if backend_kind == "auto_unconfigured":
        raise HTTPException(
            status_code=501,
            detail=(
                "Vision image endpoints are not configured. "
                "Either pass a vision-capable `model` in the request (recommended), or set one of:\n"
                "- ABSTRACTCORE_VISION_MODEL_ID (Diffusers)\n"
                "- ABSTRACTCORE_VISION_UPSTREAM_BASE_URL (OpenAI-compatible proxy)\n"
                "- ABSTRACTCORE_VISION_SDCPP_MODEL / ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL (stable-diffusion.cpp)"
            ),
        )

    # Validate backend-specific configuration before importing AbstractVision.
    # This keeps error messages stable and avoids optional dependency requirements for unconfigured setups.
    prevalidated: Dict[str, Any] = {"backend_kind": backend_kind}
    if backend_kind == "openai_compatible_proxy":
        base_url = _require_upstream_base_url()
        model_id = str(request_model or _env("ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID") or "").strip() or None
        prevalidated.update(
            {
                "base_url": base_url,
                "model_id": model_id,
                "timeout_s": float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "300") or "300"),
                "image_generations_path": _env("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH", "/images/generations")
                or "/images/generations",
                "image_edits_path": _env("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_EDITS_PATH", "/images/edits") or "/images/edits",
                "api_key": _env("ABSTRACTCORE_VISION_UPSTREAM_API_KEY"),
            }
        )
    elif backend_kind == "diffusers":
        model_id = _require_diffusers_model_id(request_model)
        allow_download = _env_bool("ABSTRACTCORE_VISION_ALLOW_DOWNLOAD", True)
        prevalidated.update(
            {
                "model_id": model_id,
                "device": _env("ABSTRACTCORE_VISION_DEVICE", "auto") or "auto",
                "torch_dtype": _env("ABSTRACTCORE_VISION_TORCH_DTYPE"),
                "allow_download": allow_download,
            }
        )
    elif backend_kind == "sdcpp":
        model_path, diffusion_model_path = _require_sdcpp_model_or_diffusion_model(request_model)
        extra_args = _env("ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS")
        prevalidated.update(
            {
                "sd_cli_path": _env("ABSTRACTCORE_VISION_SDCPP_BIN", "sd-cli") or "sd-cli",
                "model_path": model_path,
                "diffusion_model_path": diffusion_model_path,
                "vae": _env("ABSTRACTCORE_VISION_SDCPP_VAE"),
                "llm": _env("ABSTRACTCORE_VISION_SDCPP_LLM"),
                "llm_vision": _env("ABSTRACTCORE_VISION_SDCPP_LLM_VISION"),
                "clip_l": _env("ABSTRACTCORE_VISION_SDCPP_CLIP_L"),
                "clip_g": _env("ABSTRACTCORE_VISION_SDCPP_CLIP_G"),
                "t5xxl": _env("ABSTRACTCORE_VISION_SDCPP_T5XXL"),
                "extra_args": extra_args,
                "timeout_s": float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "3600") or "3600"),
            }
        )
    else:
        raise HTTPException(status_code=501, detail=f"Unknown vision backend kind: {backend_kind!r} (set ABSTRACTCORE_VISION_BACKEND)")

    (
        OpenAICompatibleBackendConfig,
        OpenAICompatibleVisionBackend,
        HuggingFaceDiffusersBackendConfig,
        HuggingFaceDiffusersVisionBackend,
        StableDiffusionCppBackendConfig,
        StableDiffusionCppVisionBackend,
        OptionalDependencyMissingError,
        req_types,
    ) = _import_abstractvision()
    ImageGenerationRequest, ImageEditRequest = req_types

    active_model_id, active_kind, active_backend, active_call_lock = _get_active_backend()
    if active_backend is not None and active_call_lock is not None and (not req_model or req_model == active_model_id):
        return active_backend, active_call_lock, OptionalDependencyMissingError, ImageGenerationRequest, ImageEditRequest

    if backend_kind == "openai_compatible_proxy":
        base_url = prevalidated["base_url"]
        model_id = prevalidated["model_id"]
        cfg = OpenAICompatibleBackendConfig(
            base_url=base_url,
            api_key=prevalidated["api_key"],
            model_id=model_id,
            timeout_s=prevalidated["timeout_s"],
            image_generations_path=prevalidated["image_generations_path"],
            image_edits_path=prevalidated["image_edits_path"],
        )
        key = (
            "openai_compatible_proxy",
            base_url,
            prevalidated["api_key"],
            model_id,
            prevalidated["timeout_s"],
            prevalidated["image_generations_path"],
            prevalidated["image_edits_path"],
        )
        backend, call_lock = _get_or_create_cached_backend(key, lambda: OpenAICompatibleVisionBackend(config=cfg))
        return backend, call_lock, OptionalDependencyMissingError, ImageGenerationRequest, ImageEditRequest

    if backend_kind == "diffusers":
        model_id = prevalidated["model_id"]
        allow_download = prevalidated["allow_download"]
        cfg = HuggingFaceDiffusersBackendConfig(
            model_id=model_id,
            device=prevalidated["device"],
            torch_dtype=prevalidated["torch_dtype"],
            allow_download=allow_download,
        )
        key = (
            "diffusers",
            model_id,
            prevalidated["device"],
            prevalidated["torch_dtype"],
            allow_download,
        )
        backend, call_lock = _get_or_create_cached_backend(key, lambda: HuggingFaceDiffusersVisionBackend(config=cfg))
        return backend, call_lock, OptionalDependencyMissingError, ImageGenerationRequest, ImageEditRequest

    if backend_kind == "sdcpp":
        model_path = prevalidated["model_path"]
        diffusion_model_path = prevalidated["diffusion_model_path"]
        extra_args = prevalidated["extra_args"]
        cfg = StableDiffusionCppBackendConfig(
            sd_cli_path=prevalidated["sd_cli_path"],
            model=model_path,
            diffusion_model=diffusion_model_path,
            vae=prevalidated["vae"],
            llm=prevalidated["llm"],
            llm_vision=prevalidated["llm_vision"],
            clip_l=prevalidated["clip_l"],
            clip_g=prevalidated["clip_g"],
            t5xxl=prevalidated["t5xxl"],
            extra_args=shlex.split(str(extra_args)) if extra_args else (),
            timeout_s=prevalidated["timeout_s"],
        )
        key = (
            "sdcpp",
            prevalidated["sd_cli_path"],
            model_path,
            diffusion_model_path,
            prevalidated["vae"],
            prevalidated["llm"],
            prevalidated["llm_vision"],
            prevalidated["clip_l"],
            prevalidated["clip_g"],
            prevalidated["t5xxl"],
            extra_args,
            prevalidated["timeout_s"],
        )
        backend, call_lock = _get_or_create_cached_backend(key, lambda: StableDiffusionCppVisionBackend(config=cfg))
        return backend, call_lock, OptionalDependencyMissingError, ImageGenerationRequest, ImageEditRequest

    raise HTTPException(status_code=501, detail=f"Unknown vision backend kind: {backend_kind!r} (set ABSTRACTCORE_VISION_BACKEND)")


def _import_registry() -> Any:
    try:
        from abstractvision import VisionModelCapabilitiesRegistry  # type: ignore
    except Exception as e:  # pragma: no cover
        raise HTTPException(
            status_code=501,
            detail="AbstractVision is required for vision model registry endpoints. Install `abstractvision`.",
        ) from e
    return VisionModelCapabilitiesRegistry


@router.get("/vision/models")
async def list_cached_vision_models() -> Dict[str, Any]:
    """List vision models from the AbstractVision registry that are present in local caches."""
    VisionModelCapabilitiesRegistry = _import_registry()
    reg = VisionModelCapabilitiesRegistry()

    hf_dirs = _default_hf_hub_cache_dirs()
    lms_dirs = _default_lmstudio_model_dirs()

    models: list[Dict[str, Any]] = []
    for model_id in reg.list_models():
        spec = reg.get(model_id)
        # Only list models relevant to this UI (t2i / i2i).
        supported_tasks = sorted(spec.tasks.keys())
        if "text_to_image" not in spec.tasks and "image_to_image" not in spec.tasks:
            continue

        cached_in: list[str] = []
        if _is_hf_model_cached(model_id, hf_dirs):
            cached_in.append("huggingface")
        if _is_lmstudio_model_cached(model_id, lms_dirs):
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

    models.sort(key=lambda x: str(x.get("id") or ""))
    return {
        "models": models,
        "registry_total": len(reg.list_models()),
        "cached_total": len(models),
        "active": _active_state(),
        "cache_dirs": {
            "huggingface": [str(p) for p in hf_dirs],
            "lmstudio": [str(p) for p in lms_dirs],
        },
    }


@router.get("/vision/model")
async def get_active_vision_model() -> Dict[str, Any]:
    """Get the currently loaded (in-memory) vision model for this server process."""
    return {"active": _active_state()}


@router.post("/vision/model/unload")
async def unload_active_vision_model() -> Dict[str, Any]:
    """Unload the currently active in-memory vision model (best-effort)."""
    _unload_active_backend()
    return {"ok": True, "active": _active_state()}


@router.post("/vision/model/load")
async def load_active_vision_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Unload any active model, then load the requested one into memory (best-effort)."""
    model_id = str(payload.get("model_id") or payload.get("model") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing required field: model_id")

    VisionModelCapabilitiesRegistry = _import_registry()
    reg = VisionModelCapabilitiesRegistry()
    try:
        _spec = reg.get(model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown model id: {model_id!r}") from e

    # Always switch single active model (free memory) for the playground UX.
    _unload_active_backend()

    start = time.time()
    backend_kind = _infer_backend_kind(model_id)
    if backend_kind not in {"diffusers", "sdcpp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_id!r} cannot be loaded into memory (unsupported backend kind: {backend_kind!r}).",
        )

    (
        _OpenAICompatibleBackendConfig,
        _OpenAICompatibleVisionBackend,
        HuggingFaceDiffusersBackendConfig,
        HuggingFaceDiffusersVisionBackend,
        StableDiffusionCppBackendConfig,
        StableDiffusionCppVisionBackend,
        _OptionalDependencyMissingError,
        _req_types,
    ) = _import_abstractvision()

    backend: Any = None
    try:
        if backend_kind == "diffusers":
            cfg = HuggingFaceDiffusersBackendConfig(
                model_id=model_id,
                device=_env("ABSTRACTCORE_VISION_DEVICE", "auto") or "auto",
                torch_dtype=_env("ABSTRACTCORE_VISION_TORCH_DTYPE"),
                allow_download=_env_bool("ABSTRACTCORE_VISION_ALLOW_DOWNLOAD", True),
            )
            backend = HuggingFaceDiffusersVisionBackend(config=cfg)
        else:
            # stable-diffusion.cpp: treat `model_id` as a local path when used here.
            model_path, diffusion_model_path = _require_sdcpp_model_or_diffusion_model(model_id)
            extra_args = _env("ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS")
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=_env("ABSTRACTCORE_VISION_SDCPP_BIN", "sd-cli") or "sd-cli",
                model=model_path,
                diffusion_model=diffusion_model_path,
                vae=_env("ABSTRACTCORE_VISION_SDCPP_VAE"),
                llm=_env("ABSTRACTCORE_VISION_SDCPP_LLM"),
                llm_vision=_env("ABSTRACTCORE_VISION_SDCPP_LLM_VISION"),
                clip_l=_env("ABSTRACTCORE_VISION_SDCPP_CLIP_L"),
                clip_g=_env("ABSTRACTCORE_VISION_SDCPP_CLIP_G"),
                t5xxl=_env("ABSTRACTCORE_VISION_SDCPP_T5XXL"),
                extra_args=shlex.split(str(extra_args)) if extra_args else (),
                timeout_s=float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "3600") or "3600"),
            )
            backend = StableDiffusionCppVisionBackend(config=cfg)

        call_lock = threading.Lock()
        preload = getattr(backend, "preload", None)
        if callable(preload):
            with call_lock:
                preload()

        global _ACTIVE_MODEL_ID, _ACTIVE_BACKEND_KIND, _ACTIVE_BACKEND, _ACTIVE_CALL_LOCK, _ACTIVE_LOADED_AT_S
        with _ACTIVE_LOCK:
            _ACTIVE_MODEL_ID = model_id
            _ACTIVE_BACKEND_KIND = backend_kind
            _ACTIVE_BACKEND = backend
            _ACTIVE_CALL_LOCK = call_lock
            _ACTIVE_LOADED_AT_S = time.time()

        return {
            "ok": True,
            "active": _active_state(),
            "load_ms": int((time.time() - start) * 1000),
        }
    except HTTPException:
        raise
    except Exception as e:
        if backend is not None:
            _unload_backend_best_effort(backend)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/images/generations")
async def images_generations(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    OpenAI-compatible image generation endpoint: POST /v1/images/generations

    Notes:
    - Only `response_format=b64_json` is supported.
    - In `auto` mode (default), the backend is inferred per-request based on `model`.
    """
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    response_format = str(payload.get("response_format") or "b64_json").strip().lower()
    if response_format not in {"b64_json"}:
        raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

    n = _coerce_int(payload.get("n")) or 1
    n = max(1, min(int(n), 10))

    width = _coerce_int(payload.get("width"))
    height = _coerce_int(payload.get("height"))
    if (width is None or height is None) and payload.get("size") is not None:
        w2, h2 = _parse_size(payload.get("size"))
        width = width if width is not None else w2
        height = height if height is not None else h2

    negative_prompt = payload.get("negative_prompt")
    steps = _coerce_int(payload.get("steps"))
    guidance_scale = _coerce_float(payload.get("guidance_scale"))
    seed = _coerce_int(payload.get("seed"))

    backend, call_lock, OptionalDependencyMissingError, ImageGenerationRequest, _ImageEditRequest = _resolve_backend(
        payload.get("model")
    )

    data_items = []
    for _ in range(n):
        try:
            req = ImageGenerationRequest(
                prompt=prompt,
                negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                extra={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "prompt",
                        "model",
                        "n",
                        "size",
                        "response_format",
                        "width",
                        "height",
                        "negative_prompt",
                        "seed",
                        "steps",
                        "guidance_scale",
                    }
                },
            )
            with call_lock:
                asset = backend.generate_image(req)
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
        data_items.append({"b64_json": b64})

    return {"created": int(time.time()), "data": data_items}


@router.post("/vision/jobs/images/generations")
async def jobs_images_generations(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Start an async image generation job with progress polling."""
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    response_format = str(payload.get("response_format") or "b64_json").strip().lower()
    if response_format not in {"b64_json"}:
        raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

    n = _coerce_int(payload.get("n")) or 1
    n = max(1, min(int(n), 10))

    width = _coerce_int(payload.get("width"))
    height = _coerce_int(payload.get("height"))
    if (width is None or height is None) and payload.get("size") is not None:
        w2, h2 = _parse_size(payload.get("size"))
        width = width if width is not None else w2
        height = height if height is not None else h2

    negative_prompt = payload.get("negative_prompt")
    steps = _coerce_int(payload.get("steps"))
    guidance_scale = _coerce_float(payload.get("guidance_scale"))
    seed = _coerce_int(payload.get("seed"))

    backend, call_lock, OptionalDependencyMissingError, ImageGenerationRequest, _ImageEditRequest = _resolve_backend(
        payload.get("model")
    )

    total_steps = (int(steps) * int(n)) if steps is not None else None

    now_s = time.time()
    with _JOBS_LOCK:
        _jobs_cleanup_locked(now_s=now_s)
        # Be conservative: only allow one in-flight job to reduce memory pressure.
        if _any_inflight_job_locked():
            raise HTTPException(status_code=409, detail="Another generation is already running; wait for it to finish.")

        job_id = _new_job_id()
        _JOBS[job_id] = {
            "id": job_id,
            "kind": "images/generations",
            "state": "queued",
            "created_at_s": now_s,
            "updated_at_s": now_s,
            "progress": {"step": 0, "total_steps": total_steps},
        }

    def _runner() -> None:
        try:
            _job_update_progress(job_id, step=0, total=total_steps, message="running")
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    job["state"] = "running"
                    job["updated_at_s"] = time.time()

            data_items = []
            for i in range(int(n)):
                req = ImageGenerationRequest(
                    prompt=prompt,
                    negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    extra={
                        k: v
                        for k, v in payload.items()
                        if k
                        not in {
                            "prompt",
                            "model",
                            "n",
                            "size",
                            "response_format",
                            "width",
                            "height",
                            "negative_prompt",
                            "seed",
                            "steps",
                            "guidance_scale",
                        }
                    },
                )

                offset = int(steps) * i if steps is not None else 0

                def _progress(step_i: int, total_i: Optional[int] = None) -> None:
                    # `step_i` is expected to be 0/1-based; normalize to 1..N for UI.
                    s = int(step_i)
                    if s < 0:
                        s = 0
                    if steps is not None and s > int(steps):
                        s = int(steps)
                    overall = offset + s
                    _job_update_progress(job_id, step=overall, total=total_steps)

                with call_lock:
                    fn = getattr(backend, "generate_image_with_progress", None)
                    if callable(fn):
                        asset = fn(req, progress_callback=_progress)
                    else:
                        asset = backend.generate_image(req)
                b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
                data_items.append({"b64_json": b64})

            _job_finish(job_id, ok=True, result={"created": int(time.time()), "data": data_items})
        except OptionalDependencyMissingError as e:
            _job_finish(job_id, ok=False, error=str(e))
        except Exception as e:
            _job_finish(job_id, ok=False, error=str(e))

    threading.Thread(target=_runner, name=f"vision-job-{job_id}", daemon=True).start()
    return {"job_id": job_id}


@router.get("/vision/jobs/{job_id}")
async def get_job(job_id: str, consume: Optional[bool] = False) -> Dict[str, Any]:
    """Poll a job status (optionally consume/remove it when completed)."""
    jid = str(job_id or "").strip()
    if not jid:
        raise HTTPException(status_code=400, detail="Missing job_id")

    now_s = time.time()
    with _JOBS_LOCK:
        _jobs_cleanup_locked(now_s=now_s)
        job = _JOBS.get(jid)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        data = dict(job)
        state = str(data.get("state") or "")
        if consume and state in {"succeeded", "failed"}:
            _JOBS.pop(jid, None)

    # Avoid leaking large results on in-flight polls.
    if str(data.get("state") or "") not in {"succeeded"}:
        data.pop("result", None)
    return data


if _HAS_MULTIPART:

    @router.post("/vision/jobs/images/edits")
    async def jobs_images_edits(
        prompt: str = Form(...),
        image: UploadFile = File(...),
        mask: Optional[UploadFile] = File(None),
        model: Optional[str] = Form(None),
        negative_prompt: Optional[str] = Form(None),
        seed: Optional[str] = Form(None),
        steps: Optional[str] = Form(None),
        guidance_scale: Optional[str] = Form(None),
        extra_json: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        """Start an async image edit job with progress polling."""
        prompt_s = str(prompt or "").strip()
        if not prompt_s:
            raise HTTPException(status_code=400, detail="Missing required field: prompt")

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Missing required image bytes")

        mask_bytes = await mask.read() if mask is not None else None

        extra: Dict[str, Any] = {}
        if extra_json:
            try:
                parsed = json.loads(str(extra_json))
            except Exception as e:
                raise HTTPException(status_code=400, detail="extra_json must be a JSON object string") from e
            if parsed is None:
                extra = {}
            elif isinstance(parsed, dict):
                extra = dict(parsed)
            else:
                raise HTTPException(status_code=400, detail="extra_json must be a JSON object string")

        seed_i = _coerce_int(seed)
        steps_i = _coerce_int(steps)
        guidance_f = _coerce_float(guidance_scale)

        backend, call_lock, OptionalDependencyMissingError, _ImageGenerationRequest, ImageEditRequest = _resolve_backend(model)
        total_steps = int(steps_i) if steps_i is not None else None

        now_s = time.time()
        with _JOBS_LOCK:
            _jobs_cleanup_locked(now_s=now_s)
            if _any_inflight_job_locked():
                raise HTTPException(status_code=409, detail="Another generation is already running; wait for it to finish.")

            job_id = _new_job_id()
            _JOBS[job_id] = {
                "id": job_id,
                "kind": "images/edits",
                "state": "queued",
                "created_at_s": now_s,
                "updated_at_s": now_s,
                "progress": {"step": 0, "total_steps": total_steps},
            }

        def _runner() -> None:
            try:
                _job_update_progress(job_id, step=0, total=total_steps, message="running")
                with _JOBS_LOCK:
                    job = _JOBS.get(job_id)
                    if job is not None:
                        job["state"] = "running"
                        job["updated_at_s"] = time.time()

                req = ImageEditRequest(
                    prompt=prompt_s,
                    image=bytes(image_bytes),
                    mask=bytes(mask_bytes) if mask_bytes else None,
                    negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                    seed=seed_i,
                    steps=steps_i,
                    guidance_scale=guidance_f,
                    extra=extra,
                )

                def _progress(step_i: int, total_i: Optional[int] = None) -> None:
                    s = int(step_i)
                    if s < 0:
                        s = 0
                    if total_steps is not None and s > int(total_steps):
                        s = int(total_steps)
                    _job_update_progress(job_id, step=s, total=total_steps)

                with call_lock:
                    fn = getattr(backend, "edit_image_with_progress", None)
                    if callable(fn):
                        asset = fn(req, progress_callback=_progress)
                    else:
                        asset = backend.edit_image(req)
                b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
                _job_finish(job_id, ok=True, result={"created": int(time.time()), "data": [{"b64_json": b64}]})
            except OptionalDependencyMissingError as e:
                _job_finish(job_id, ok=False, error=str(e))
            except Exception as e:
                _job_finish(job_id, ok=False, error=str(e))

        threading.Thread(target=_runner, name=f"vision-job-{job_id}", daemon=True).start()
        return {"job_id": job_id}

    @router.post("/images/edits")
    async def images_edits(
        prompt: str = Form(...),
        image: UploadFile = File(...),
        mask: Optional[UploadFile] = File(None),
        model: Optional[str] = Form(None),
        negative_prompt: Optional[str] = Form(None),
        seed: Optional[str] = Form(None),
        steps: Optional[str] = Form(None),
        guidance_scale: Optional[str] = Form(None),
        extra_json: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible image edit endpoint: POST /v1/images/edits (multipart/form-data)

        Implemented as a thin proxy over AbstractVision's OpenAI-compatible backend.
        """
        prompt_s = str(prompt or "").strip()
        if not prompt_s:
            raise HTTPException(status_code=400, detail="Missing required field: prompt")

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Missing required image bytes")

        mask_bytes = await mask.read() if mask is not None else None

        extra: Dict[str, Any] = {}
        if extra_json:
            try:
                parsed = json.loads(str(extra_json))
            except Exception as e:
                raise HTTPException(status_code=400, detail="extra_json must be a JSON object string") from e
            if parsed is None:
                extra = {}
            elif isinstance(parsed, dict):
                extra = dict(parsed)
            else:
                raise HTTPException(status_code=400, detail="extra_json must be a JSON object string")

        backend, call_lock, OptionalDependencyMissingError, _ImageGenerationRequest, ImageEditRequest = _resolve_backend(model)

        try:
            req = ImageEditRequest(
                prompt=prompt_s,
                image=bytes(image_bytes),
                mask=bytes(mask_bytes) if mask_bytes else None,
                negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                seed=_coerce_int(seed),
                steps=_coerce_int(steps),
                guidance_scale=_coerce_float(guidance_scale),
                extra=extra,
            )
            with call_lock:
                asset = backend.edit_image(req)
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
        return {"created": int(time.time()), "data": [{"b64_json": b64}]}

else:

    @router.post("/images/edits")
    async def images_edits() -> Dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail=(
                "The /v1/images/edits endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )
