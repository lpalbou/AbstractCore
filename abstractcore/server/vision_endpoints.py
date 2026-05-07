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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Body, File, Form, HTTPException, Path as FastAPIPath, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .capability_generation import ServerVisionFacade, create_capability_generation_core

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

_DEFAULT_MODEL_ALIASES = {"default", "configured-default"}
_REMOVED_VISION_MODEL_ALIASES = {"local/abstractvision", "abstractvision/default", "server/default"}


class _ProxyOptionalDependencyMissingError(Exception):
    """Placeholder exception type for the dependency-free upstream proxy path."""


@dataclass
class _ProxyGeneratedAsset:
    data: bytes
    mime_type: str = "image/png"


@dataclass
class _ProxyImageGenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class _ProxyImageEditRequest:
    prompt: str
    image: bytes
    mask: Optional[bytes] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


def _join_url(base_url: str, path: str) -> str:
    b = str(base_url or "").rstrip("/")
    p = str(path or "").strip()
    if not p:
        return b
    if not p.startswith("/"):
        p = "/" + p
    return b + p


def _sniff_mime_type(content: bytes, fallback: str) -> str:
    b = bytes(content or b"")
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(b) >= 12 and b[4:8] == b"ftyp":
        return "video/mp4"
    return str(fallback or "application/octet-stream")


def _decode_b64(s: str) -> bytes:
    raw = "".join(str(s or "").strip().split())
    pad = (-len(raw)) % 4
    if pad:
        raw = raw + ("=" * pad)
    return base64.b64decode(raw, validate=False)


def _first_data_item(resp: Dict[str, Any]) -> Dict[str, Any]:
    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


class _OpenAICompatibleImageProxyBackend:
    """Dependency-light OpenAI-compatible image proxy used by the server image."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        model_id: Optional[str],
        timeout_s: float,
        image_generations_path: str,
        image_edits_path: str,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout_s = float(timeout_s)
        self.image_generations_path = image_generations_path or "/images/generations"
        self.image_edits_path = image_edits_path or "/images/edits"

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        detail = resp.text[:1000]
        raise ValueError(f"Upstream image endpoint returned HTTP {resp.status_code}: {detail}")

    def _parse_media(self, payload: Dict[str, Any], *, fallback_mime: str) -> _ProxyGeneratedAsset:
        item = _first_data_item(payload)
        if "b64_json" in item:
            content = _decode_b64(str(item.get("b64_json") or ""))
            return _ProxyGeneratedAsset(data=content, mime_type=_sniff_mime_type(content, fallback_mime))
        if "url" in item and isinstance(item.get("url"), str):
            url = str(item.get("url"))
            with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = client.get(url)
                self._raise_for_status(resp)
                content = bytes(resp.content)
                mime = _sniff_mime_type(content, resp.headers.get("content-type") or fallback_mime)
                return _ProxyGeneratedAsset(data=content, mime_type=mime)
        raise ValueError("Invalid upstream image response: missing data[0].b64_json or data[0].url")

    def generate_image(self, request: _ProxyImageGenerationRequest) -> _ProxyGeneratedAsset:
        payload: Dict[str, Any] = {
            "prompt": request.prompt,
            "n": 1,
        }
        if self.model_id:
            payload["model"] = self.model_id
        if request.width is not None and request.height is not None:
            payload["size"] = f"{int(request.width)}x{int(request.height)}"
        if isinstance(request.extra, dict):
            payload.update({k: v for k, v in request.extra.items() if v is not None})

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                _join_url(self.base_url, self.image_generations_path),
                headers=self._headers(),
                json=payload,
            )
            self._raise_for_status(resp)
            data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid upstream image response: expected JSON object")
        return self._parse_media(data, fallback_mime="image/png")

    def edit_image(self, request: _ProxyImageEditRequest) -> _ProxyGeneratedAsset:
        fields: Dict[str, str] = {"prompt": request.prompt}
        if self.model_id:
            fields["model"] = self.model_id
        if isinstance(request.extra, dict):
            for key, value in request.extra.items():
                if value is not None:
                    fields[str(key)] = str(value)

        files: Dict[str, Tuple[str, bytes, str]] = {
            "image": ("image.png", bytes(request.image), "image/png"),
        }
        if request.mask is not None:
            files["mask"] = ("mask.png", bytes(request.mask), "image/png")

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                _join_url(self.base_url, self.image_edits_path),
                headers=self._headers(),
                data=fields,
                files=files,
            )
            self._raise_for_status(resp)
            data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid upstream image edit response: expected JSON object")
        return self._parse_media(data, fallback_mime="image/png")


class VisionModelLoadRequest(BaseModel):
    """Request body for loading a local vision generation model into memory."""

    model_id: Optional[str] = Field(
        default=None,
        description="Vision model id or local model path to load. Required unless using the `model` alias.",
        examples=["Qwen/Qwen-Image-2512"],
    )
    model: Optional[str] = Field(
        default=None,
        description="Alias for `model_id`.",
        examples=["Qwen/Qwen-Image-2512"],
    )


class ImageGenerationBody(BaseModel):
    """OpenAI-compatible image generation request body."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "openai-compatible/gpt-image-1",
                    "prompt": "A precise product photo of a red ceramic mug on a white table.",
                    "n": 1,
                    "width": 1024,
                    "height": 1024,
                    "response_format": "b64_json",
                    "negative_prompt": None,
                    "seed": None,
                    "steps": None,
                    "guidance_scale": None,
                    "quality": "low",
                    "style": None,
                    "user": "swagger-user",
                    "background": "auto",
                    "output_format": "png",
                    "output_compression": None,
                    "moderation": "auto",
                    "extra": {},
                }
            ]
        },
    )

    prompt: str = Field(..., description="Text prompt describing the image to generate.", examples=["A precise product photo of a red ceramic mug on a white table."])
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider/model image id. Omit this field to use the server's configured "
            "AbstractVision default. Explicit local models use "
            "`diffusers/default`, `diffusers/<huggingface-repo>`, "
            "or `sdcpp/default`; remote image providers use "
            "`openai-compatible/my-image-model` with a configured upstream image endpoint."
        ),
        examples=["diffusers/default", "openai-compatible/gpt-image-2"],
    )
    n: Optional[int] = Field(default=1, description="Number of images to generate. Clamped to 1..10.", examples=[1])
    width: Optional[int] = Field(default=None, description="Requested image width in pixels, backend permitting.", examples=[1024])
    height: Optional[int] = Field(default=None, description="Requested image height in pixels, backend permitting.", examples=[1024])
    response_format: Optional[str] = Field(default="b64_json", description="Response format. Only `b64_json` is currently supported.", examples=["b64_json"])
    negative_prompt: Optional[str] = Field(default=None, description="Optional negative prompt for backends that support it.")
    seed: Optional[int] = Field(
        default=None,
        description=(
            "Optional deterministic seed for local generation backends. "
            "Strict OpenAI-compatible upstreams may reject this field; for a custom upstream "
            "that supports it, pass it through `extra`."
        ),
        examples=[1234],
    )
    steps: Optional[int] = Field(
        default=None,
        description=(
            "Optional denoising/inference step count for local generation backends. "
            "For custom OpenAI-compatible upstreams, pass this through `extra`."
        ),
        examples=[20],
    )
    guidance_scale: Optional[float] = Field(
        default=None,
        description=(
            "Optional classifier-free guidance scale for local generation backends. "
            "For custom OpenAI-compatible upstreams, pass this through `extra`."
        ),
        examples=[7.5],
    )
    quality: Optional[str] = Field(default=None, description="Optional OpenAI-compatible image quality forwarded to upstream providers. GPT image models commonly support `low`, `medium`, `high`, or `auto`; DALL-E models use their own quality options.", examples=["low"])
    style: Optional[str] = Field(default=None, description="Optional OpenAI-compatible image style forwarded to upstream providers that support it, such as `vivid` or `natural` for DALL-E 3.", examples=["natural"])
    user: Optional[str] = Field(default=None, description="Optional caller/user id forwarded to upstream providers when supported.")
    background: Optional[str] = Field(default=None, description="Optional OpenAI-compatible background setting forwarded to upstream providers. GPT image models commonly support `auto`, `opaque`, or `transparent`.", examples=["auto"])
    output_format: Optional[str] = Field(default=None, description="Optional OpenAI-compatible output image format forwarded to upstream providers.", examples=["png"])
    output_compression: Optional[int] = Field(default=None, description="Optional OpenAI-compatible output compression value forwarded to upstream providers. Use this with compressed formats such as `jpeg` or `webp` when the upstream supports it.", examples=[85])
    moderation: Optional[str] = Field(default=None, description="Optional OpenAI-compatible moderation setting forwarded to upstream providers, commonly `auto` or `low` for GPT image models.", examples=["auto"])
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backend-specific parameters forwarded to the selected image backend or upstream endpoint.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_size(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "size" not in data:
            return data
        out = dict(data)
        raw_size = out.pop("size", None)
        if raw_size is None:
            return out
        width, height = _parse_size(raw_size)
        if width is None or height is None:
            raise ValueError("size must use WIDTHxHEIGHT format, for example 1024x1024")
        if out.get("width") is None:
            out["width"] = width
        if out.get("height") is None:
            out["height"] = height
        return out


def _model_payload(model: BaseModel) -> Dict[str, Any]:
    data = model.model_dump(exclude_none=True)
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        data.update(extra)
    return data


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


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        v = _env(name)
        if v is not None:
            return v
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _env_bool_first(*names: str, default: bool = False) -> bool:
    for name in names:
        v = _env(name)
        if v is not None:
            return str(v).strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _is_default_model_alias(model: Any) -> bool:
    return str(model or "").strip().lower() in _DEFAULT_MODEL_ALIASES


def _reject_removed_vision_model_alias(model: Any) -> None:
    raw = str(model or "").strip().lower()
    if raw in _REMOVED_VISION_MODEL_ALIASES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image model id {raw!r} is not supported. Omit `model` to use this server's "
                "configured AbstractVision default, or use provider/model routing such as "
                "`diffusers/default`, `diffusers/<huggingface-repo>`, `sdcpp/default`, "
                "or `openai-compatible/<model>` with a configured upstream base URL."
            ),
        )


def _diffusers_model_env() -> Optional[str]:
    return _env_first("ABSTRACTCORE_VISION_MODEL_ID", "ABSTRACTVISION_DIFFUSERS_MODEL_ID", "ABSTRACTVISION_MODEL_ID")


def _upstream_base_url_env() -> Optional[str]:
    return _env_first("ABSTRACTCORE_VISION_UPSTREAM_BASE_URL", "ABSTRACTVISION_BASE_URL")


def _upstream_model_env() -> Optional[str]:
    return _env_first("ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID", "ABSTRACTVISION_MODEL_ID")


def _sdcpp_env(*names: str) -> Optional[str]:
    core_names = [f"ABSTRACTCORE_VISION_SDCPP_{name}" for name in names]
    av_names = [f"ABSTRACTVISION_SDCPP_{name}" for name in names]
    return _env_first(*(core_names + av_names))


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
    raw = _env_first("ABSTRACTCORE_VISION_BACKEND", "ABSTRACTVISION_BACKEND")
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
    - `diffusers/default` -> configured Diffusers model from env
    - `huggingface/Qwen/Qwen-Image-2512` -> `Qwen/Qwen-Image-2512` (Diffusers)
    - `mlx/Qwen/Qwen-Image-2512` -> `Qwen/Qwen-Image-2512` (Diffusers; device selection is env-driven)
    - `sdcpp/default` -> configured stable-diffusion.cpp model from env
    - `openai-compatible/model` -> `model` for the configured upstream image endpoint

    Non-image AbstractCore providers (e.g. `openai/gpt-4o-mini`) are rejected so
    chat model ids do not silently become image-generation defaults.
    """
    model = str(request_model or "").strip()
    if not model:
        return None
    _reject_removed_vision_model_alias(model)

    prefix, rest = _split_known_prefix(model)
    if prefix is None:
        if _looks_like_hf_repo_id(model):
            raise HTTPException(
                status_code=400,
                detail=f"Image model {model!r} must use provider/model routing. Use `diffusers/{model}`.",
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image model {model!r} must use provider/model routing. "
                "Use `diffusers/<huggingface-repo>`, `sdcpp/default`, "
                "or `openai-compatible/<model>`."
            ),
        )

    # Local generation backends: strip prefix and pass through.
    if prefix in {"huggingface", "hf", "mlx", "diffusers"}:
        if _is_default_model_alias(rest):
            return None
        return rest or None
    if prefix == "sdcpp":
        if _is_default_model_alias(rest):
            return None
        return rest or None

    if prefix in {"openai-compatible", "openai_compatible"}:
        return rest or None

    raise HTTPException(
        status_code=400,
        detail=(
            f"Image model provider {prefix!r} is not supported by `/v1/images/*`. "
            "Use `diffusers/<huggingface-repo>`, `sdcpp/default`, or "
            "`openai-compatible/<model>` with a configured upstream image endpoint."
        ),
    )


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
    raw_model = str(request_model or "").strip()
    prefix, _rest = _split_known_prefix(raw_model)
    if prefix == "sdcpp":
        return "sdcpp"
    if prefix in {"huggingface", "hf", "mlx", "diffusers"}:
        return "diffusers"
    if prefix in {"openai-compatible", "openai_compatible"}:
        return "openai_compatible_proxy"

    model = str(_normalize_request_model_for_backend(request_model) or "").strip()
    if model:
        if _looks_like_filesystem_path(model):
            return "sdcpp"
        # Default: treat HF-style ids as Diffusers.
        if _looks_like_hf_repo_id(model):
            return "diffusers"
        # Unknown strings: prefer local generation unless proxy is explicitly configured.
        if _upstream_base_url_env():
            return "openai_compatible_proxy"
        return "diffusers"

    # No request model: fall back to env configuration.
    if _sdcpp_env("MODEL") or _sdcpp_env("DIFFUSION_MODEL"):
        return "sdcpp"
    if _upstream_base_url_env():
        return "openai_compatible_proxy"
    if _diffusers_model_env():
        return "diffusers"
    return "auto_unconfigured"


def _effective_backend_kind(request_model: Any) -> str:
    raw_model = str(request_model or "").strip()
    prefix, _ = _split_known_prefix(raw_model)
    if prefix == "sdcpp":
        return "sdcpp"
    if prefix in {"huggingface", "hf", "mlx", "diffusers"}:
        return "diffusers"
    if prefix in {"openai-compatible", "openai_compatible"}:
        return "openai_compatible_proxy"

    env_kind = _vision_backend_kind()
    if env_kind == "auto":
        return _infer_backend_kind(request_model)

    model = str(_normalize_request_model_for_backend(request_model) or request_model or "").strip()
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
    base_url = _upstream_base_url_env()
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
    model_id = str(_normalize_request_model_for_backend(request_model) or _diffusers_model_env() or "").strip()
    if not model_id:
        raise HTTPException(
            status_code=501,
            detail=(
                "Vision image endpoints are not configured for Diffusers mode. "
                "`diffusers/default` requires ABSTRACTCORE_VISION_MODEL_ID, "
                "ABSTRACTVISION_DIFFUSERS_MODEL_ID, or ABSTRACTVISION_MODEL_ID. "
                "Alternatively pass `diffusers/<huggingface-repo>` explicitly or configure "
                "ABSTRACTCORE_VISION_UPSTREAM_BASE_URL / ABSTRACTVISION_BASE_URL for an "
                "OpenAI-compatible image endpoint."
            ),
        )
    return model_id


def _require_sdcpp_model_or_diffusion_model(request_model: Any) -> Tuple[Optional[str], Optional[str]]:
    req = str(_normalize_request_model_for_backend(request_model) or "").strip()
    if req and not _looks_like_filesystem_path(req):
        raise HTTPException(
            status_code=400,
            detail=(
                "stable-diffusion.cpp backend expects a local model path (typically a .gguf file). "
                f"Got model={req!r}. If you intended to run a Hugging Face model id, "
                "use the Diffusers backend (or set ABSTRACTCORE_VISION_BACKEND=auto)."
            ),
        )

    env_model = str(_sdcpp_env("MODEL") or "").strip()
    env_diffusion = str(_sdcpp_env("DIFFUSION_MODEL") or "").strip()

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
                "ABSTRACTVISION_SDCPP_VAE",
                "ABSTRACTVISION_SDCPP_LLM",
                "ABSTRACTVISION_SDCPP_LLM_VISION",
                "ABSTRACTVISION_SDCPP_CLIP_L",
                "ABSTRACTVISION_SDCPP_CLIP_G",
                "ABSTRACTVISION_SDCPP_T5XXL",
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


_IMAGE_GENERATION_CORE_FIELDS = {
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
    "extra",
}


def _image_generation_request_parts(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
    width = _coerce_int(payload.get("width"))
    height = _coerce_int(payload.get("height"))
    if (width is None or height is None) and payload.get("size") is not None:
        w2, h2 = _parse_size(payload.get("size"))
        width = width if width is not None else w2
        height = height if height is not None else h2

    extra = {k: v for k, v in payload.items() if k not in _IMAGE_GENERATION_CORE_FIELDS}
    nested_extra = payload.get("extra")
    if isinstance(nested_extra, dict):
        extra.update({str(k): v for k, v in nested_extra.items() if v is not None})

    # OpenAI-compatible image endpoints expect `size`, not backend-local
    # `width`/`height`. Local generation backends still need width/height.
    if _effective_backend_kind(payload.get("model")) == "openai_compatible_proxy":
        size = payload.get("size")
        if size is None and width is not None and height is not None:
            size = f"{int(width)}x{int(height)}"
        if size is not None:
            extra.setdefault("size", str(size))
        width = None
        height = None

    return width, height, extra


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
    backend_kind = _effective_backend_kind(request_model)

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
        model_id = str(normalized or _upstream_model_env() or "").strip() or None
        prevalidated.update(
            {
                "base_url": base_url,
                "model_id": model_id,
                "timeout_s": float(_env_first("ABSTRACTCORE_VISION_TIMEOUT_S", "ABSTRACTVISION_TIMEOUT_S", default="300") or "300"),
                "image_generations_path": _env_first(
                    "ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH",
                    "ABSTRACTVISION_IMAGES_GENERATIONS_PATH",
                    default="/images/generations",
                )
                or "/images/generations",
                "image_edits_path": _env_first(
                    "ABSTRACTCORE_VISION_UPSTREAM_IMAGES_EDITS_PATH",
                    "ABSTRACTVISION_IMAGES_EDITS_PATH",
                    default="/images/edits",
                )
                or "/images/edits",
                "api_key": _env_first("ABSTRACTCORE_VISION_UPSTREAM_API_KEY", "ABSTRACTVISION_API_KEY"),
            }
        )
        key = (
            "openai_compatible_proxy",
            prevalidated["base_url"],
            prevalidated["api_key"],
            prevalidated["model_id"],
            prevalidated["timeout_s"],
            prevalidated["image_generations_path"],
            prevalidated["image_edits_path"],
        )
        backend, call_lock = _get_or_create_cached_backend(
            key,
            lambda: _OpenAICompatibleImageProxyBackend(
                base_url=prevalidated["base_url"],
                api_key=prevalidated["api_key"],
                model_id=prevalidated["model_id"],
                timeout_s=prevalidated["timeout_s"],
                image_generations_path=prevalidated["image_generations_path"],
                image_edits_path=prevalidated["image_edits_path"],
            ),
        )
        return (
            backend,
            call_lock,
            _ProxyOptionalDependencyMissingError,
            _ProxyImageGenerationRequest,
            _ProxyImageEditRequest,
        )
    elif backend_kind == "diffusers":
        model_id = _require_diffusers_model_id(request_model)
        allow_download = _env_bool_first(
            "ABSTRACTCORE_VISION_ALLOW_DOWNLOAD",
            "ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD",
            default=False,
        )
        prevalidated.update(
            {
                "model_id": model_id,
                "device": _env_first("ABSTRACTCORE_VISION_DEVICE", "ABSTRACTVISION_DIFFUSERS_DEVICE", default="auto") or "auto",
                "torch_dtype": _env_first("ABSTRACTCORE_VISION_TORCH_DTYPE", "ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE"),
                "allow_download": allow_download,
                "auto_retry_fp32": _env_bool_first(
                    "ABSTRACTCORE_VISION_AUTO_RETRY_FP32",
                    "ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32",
                    default=True,
                ),
            }
        )
    elif backend_kind == "sdcpp":
        model_path, diffusion_model_path = _require_sdcpp_model_or_diffusion_model(request_model)
        extra_args = _sdcpp_env("EXTRA_ARGS")
        prevalidated.update(
            {
                "sd_cli_path": _sdcpp_env("BIN") or "sd-cli",
                "model_path": model_path,
                "diffusion_model_path": diffusion_model_path,
                "vae": _sdcpp_env("VAE"),
                "llm": _sdcpp_env("LLM"),
                "llm_vision": _sdcpp_env("LLM_VISION"),
                "clip_l": _sdcpp_env("CLIP_L"),
                "clip_g": _sdcpp_env("CLIP_G"),
                "t5xxl": _sdcpp_env("T5XXL"),
                "extra_args": extra_args,
                "timeout_s": float(_env_first("ABSTRACTCORE_VISION_TIMEOUT_S", "ABSTRACTVISION_TIMEOUT_S", default="3600") or "3600"),
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

    if backend_kind == "diffusers":
        model_id = prevalidated["model_id"]
        allow_download = prevalidated["allow_download"]
        cfg = HuggingFaceDiffusersBackendConfig(
            model_id=model_id,
            device=prevalidated["device"],
            torch_dtype=prevalidated["torch_dtype"],
            allow_download=allow_download,
            auto_retry_fp32=prevalidated["auto_retry_fp32"],
        )
        key = (
            "diffusers",
            model_id,
            prevalidated["device"],
            prevalidated["torch_dtype"],
            allow_download,
            prevalidated["auto_retry_fp32"],
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


def _create_vision_generation_core(request_model: Any):
    backend, call_lock, OptionalDependencyMissingError, ImageGenerationRequest, ImageEditRequest = _resolve_backend(
        request_model
    )
    facade = ServerVisionFacade(
        backend=backend,
        call_lock=call_lock,
        image_generation_request_cls=ImageGenerationRequest,
        image_edit_request_cls=ImageEditRequest,
        backend_id=f"abstractcore-server:{_effective_backend_kind(request_model)}",
    )
    core = create_capability_generation_core(vision_facade=facade)
    return core, OptionalDependencyMissingError


def _first_generated_image_bytes(result: Any) -> bytes:
    image_items = getattr(result, "outputs", {}).get("image", [])
    item = image_items[0] if image_items else None
    data = getattr(item, "data", None)
    if not isinstance(data, (bytes, bytearray)):
        raise RuntimeError("Image backend returned an unexpected type (expected raw bytes).")
    return bytes(data)


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
    hf_dirs = _default_hf_hub_cache_dirs()
    lms_dirs = _default_lmstudio_model_dirs()
    try:
        VisionModelCapabilitiesRegistry = _import_registry()
        reg = VisionModelCapabilitiesRegistry()
    except HTTPException as exc:
        return {
            "models": [],
            "registry_available": False,
            "registry_total": 0,
            "cached_total": 0,
            "active": _active_state(),
            "cache_dirs": {
                "huggingface": [str(p) for p in hf_dirs],
                "lmstudio": [str(p) for p in lms_dirs],
            },
            "error": str(exc.detail),
        }

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
        "registry_available": True,
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
async def load_active_vision_model(payload: VisionModelLoadRequest = Body(...)) -> Dict[str, Any]:
    """Unload any active model, then load the requested one into memory (best-effort)."""
    data = _model_payload(payload)
    model_id = str(data.get("model_id") or data.get("model") or "").strip()
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
                device=_env_first("ABSTRACTCORE_VISION_DEVICE", "ABSTRACTVISION_DIFFUSERS_DEVICE", default="auto") or "auto",
                torch_dtype=_env_first("ABSTRACTCORE_VISION_TORCH_DTYPE", "ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE"),
                allow_download=_env_bool_first(
                    "ABSTRACTCORE_VISION_ALLOW_DOWNLOAD",
                    "ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD",
                    default=False,
                ),
                auto_retry_fp32=_env_bool_first(
                    "ABSTRACTCORE_VISION_AUTO_RETRY_FP32",
                    "ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32",
                    default=True,
                ),
            )
            backend = HuggingFaceDiffusersVisionBackend(config=cfg)
        else:
            # stable-diffusion.cpp: treat `model_id` as a local path when used here.
            model_path, diffusion_model_path = _require_sdcpp_model_or_diffusion_model(model_id)
            extra_args = _sdcpp_env("EXTRA_ARGS")
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=_sdcpp_env("BIN") or "sd-cli",
                model=model_path,
                diffusion_model=diffusion_model_path,
                vae=_sdcpp_env("VAE"),
                llm=_sdcpp_env("LLM"),
                llm_vision=_sdcpp_env("LLM_VISION"),
                clip_l=_sdcpp_env("CLIP_L"),
                clip_g=_sdcpp_env("CLIP_G"),
                t5xxl=_sdcpp_env("T5XXL"),
                extra_args=shlex.split(str(extra_args)) if extra_args else (),
                timeout_s=float(_env_first("ABSTRACTCORE_VISION_TIMEOUT_S", "ABSTRACTVISION_TIMEOUT_S", default="3600") or "3600"),
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
async def images_generations(payload: ImageGenerationBody = Body(...)) -> Dict[str, Any]:
    """
    OpenAI-compatible image generation endpoint: POST /v1/images/generations

    Notes:
    - Only `response_format=b64_json` is supported.
    - In `auto` mode (default), the backend is inferred per-request based on `model`.
    """
    payload = _model_payload(payload)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    response_format = str(payload.get("response_format") or "b64_json").strip().lower()
    if response_format not in {"b64_json"}:
        raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

    n = _coerce_int(payload.get("n")) or 1
    n = max(1, min(int(n), 10))

    negative_prompt = payload.get("negative_prompt")
    steps = _coerce_int(payload.get("steps"))
    guidance_scale = _coerce_float(payload.get("guidance_scale"))
    seed = _coerce_int(payload.get("seed"))
    width, height, extra = _image_generation_request_parts(payload)

    core, OptionalDependencyMissingError = _create_vision_generation_core(payload.get("model"))

    data_items = []
    for _ in range(n):
        try:
            result = core.generate(
                prompt,
                output={
                    "modality": "image",
                    "task": "image_generation",
                    "negative_prompt": str(negative_prompt) if negative_prompt is not None else None,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "extra": extra,
                },
            )
            image_bytes = _first_generated_image_bytes(result)
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_items.append({"b64_json": b64})

    return {"created": int(time.time()), "data": data_items}


@router.post("/vision/jobs/images/generations")
async def jobs_images_generations(payload: ImageGenerationBody = Body(...)) -> Dict[str, Any]:
    """Start an async image generation job with progress polling."""
    payload = _model_payload(payload)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    response_format = str(payload.get("response_format") or "b64_json").strip().lower()
    if response_format not in {"b64_json"}:
        raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

    n = _coerce_int(payload.get("n")) or 1
    n = max(1, min(int(n), 10))

    negative_prompt = payload.get("negative_prompt")
    steps = _coerce_int(payload.get("steps"))
    guidance_scale = _coerce_float(payload.get("guidance_scale"))
    seed = _coerce_int(payload.get("seed"))
    width, height, extra = _image_generation_request_parts(payload)

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
                    extra=extra,
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
async def get_job(
    job_id: str = FastAPIPath(..., description="Vision job id returned by `/v1/vision/jobs/images/generations` or `/v1/vision/jobs/images/edits`."),
    consume: Optional[bool] = Query(default=False, description="When true, remove a completed job from the in-memory job store after returning it."),
) -> Dict[str, Any]:
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
        prompt: str = Form(..., description="Text prompt describing the desired image edit.", examples=["Make the mug blue and keep the white background."]),
        image: UploadFile = File(..., description="Source image to edit. Upload a PNG, JPEG, or WebP file supported by the selected backend."),
        mask: Optional[UploadFile] = File(None, description="Optional mask image for inpainting/edit backends that support it. Transparent pixels mark editable areas for OpenAI-style masks."),
        model: Optional[str] = Form(
            None,
            description=(
                "Optional provider/model image id. Omit to use the server's configured AbstractVision default; "
                "use `diffusers/default`, `diffusers/<huggingface-repo>`, `sdcpp/default`, or "
                "`openai-compatible/<model>`."
            ),
            examples=["openai-compatible/gpt-image-1"],
        ),
        size: Optional[str] = Form(None, description="OpenAI-style output image size such as `1024x1024`, `1536x1024`, `1024x1536`, or `auto` when supported.", examples=["1024x1024"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported by the server response.", examples=["b64_json"]),
        negative_prompt: Optional[str] = Form(None, description="Local/backend-specific negative prompt. Strict OpenAI-compatible upstreams do not receive this top-level field unless supplied through `extra_json`.", examples=["blur, low quality"]),
        seed: Optional[str] = Form(None, description="Local/backend-specific deterministic seed. Use `extra_json` for custom OpenAI-compatible upstreams that support a seed field.", examples=["1234"]),
        steps: Optional[str] = Form(None, description="Local/backend-specific denoising/inference step count. Use `extra_json` for custom OpenAI-compatible upstreams that support a steps field.", examples=["20"]),
        guidance_scale: Optional[str] = Form(None, description="Local/backend-specific classifier-free guidance scale. Use `extra_json` for custom OpenAI-compatible upstreams that support this field.", examples=["7.5"]),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific generation parameters.", examples=['{"quality":"low","background":"auto","output_format":"png"}']),
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
        response_format_s = str(response_format or "b64_json").strip().lower()
        if response_format_s not in {"b64_json"}:
            raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")
        if size:
            extra.setdefault("size", str(size).strip())
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
        prompt: str = Form(..., description="Text prompt describing the desired image edit.", examples=["Make the mug blue and keep the white background."]),
        image: UploadFile = File(..., description="Source image to edit. Upload a PNG, JPEG, or WebP file supported by the selected backend."),
        mask: Optional[UploadFile] = File(None, description="Optional mask image for inpainting/edit backends that support it. Transparent pixels mark editable areas for OpenAI-style masks."),
        model: Optional[str] = Form(
            None,
            description=(
                "Optional provider/model image id. Omit to use the server's configured AbstractVision default; "
                "use `diffusers/default`, `diffusers/<huggingface-repo>`, `sdcpp/default`, or "
                "`openai-compatible/<model>`."
            ),
            examples=["openai-compatible/gpt-image-1"],
        ),
        size: Optional[str] = Form(None, description="OpenAI-style output image size such as `1024x1024`, `1536x1024`, `1024x1536`, or `auto` when supported.", examples=["1024x1024"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported by the server response.", examples=["b64_json"]),
        negative_prompt: Optional[str] = Form(None, description="Local/backend-specific negative prompt. Strict OpenAI-compatible upstreams do not receive this top-level field unless supplied through `extra_json`.", examples=["blur, low quality"]),
        seed: Optional[str] = Form(None, description="Local/backend-specific deterministic seed. Use `extra_json` for custom OpenAI-compatible upstreams that support a seed field.", examples=["1234"]),
        steps: Optional[str] = Form(None, description="Local/backend-specific denoising/inference step count. Use `extra_json` for custom OpenAI-compatible upstreams that support a steps field.", examples=["20"]),
        guidance_scale: Optional[str] = Form(None, description="Local/backend-specific classifier-free guidance scale. Use `extra_json` for custom OpenAI-compatible upstreams that support this field.", examples=["7.5"]),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific generation parameters.", examples=['{"quality":"low","background":"auto","output_format":"png"}']),
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

        response_format_s = str(response_format or "b64_json").strip().lower()
        if response_format_s not in {"b64_json"}:
            raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")
        if size:
            extra.setdefault("size", str(size).strip())
        core, OptionalDependencyMissingError = _create_vision_generation_core(model)

        try:
            media = [
                {
                    "type": "image",
                    "content": bytes(image_bytes),
                    "role": "source",
                }
            ]
            if mask_bytes:
                media.append(
                    {
                        "type": "image",
                        "content": bytes(mask_bytes),
                        "role": "mask",
                    }
                )
            result = core.generate(
                prompt_s,
                media=media,
                output={
                    "modality": "image",
                    "task": "image_edit",
                    "negative_prompt": str(negative_prompt) if negative_prompt is not None else None,
                    "seed": _coerce_int(seed),
                    "steps": _coerce_int(steps),
                    "guidance_scale": _coerce_float(guidance_scale),
                    "extra": extra,
                },
            )
            image_bytes_out = _first_generated_image_bytes(result)
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        b64 = base64.b64encode(image_bytes_out).decode("ascii")
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
