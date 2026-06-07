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
import hashlib
import json
import os
import shlex
import time
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Body, File, Form, HTTPException, Path as FastAPIPath, Query, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..capabilities import vision_catalog as _vision_catalog
from ..capabilities.errors import CapabilityUnavailableError
from .capability_generation import ServerVisionFacade, create_capability_generation_core

try:  # Optional dependency (needed only for multipart parsing).
    import multipart  # type: ignore  # noqa: F401

    _HAS_MULTIPART = True
except Exception:  # pragma: no cover
    _HAS_MULTIPART = False


router = APIRouter(tags=["vision"])
provider_router = APIRouter(tags=["vision"])

_BACKEND_CACHE_LOCK = threading.Lock()
_BACKEND_CACHE: Dict[Tuple[Any, ...], Tuple[Any, threading.Lock, float]] = {}
_RESIDENCY_RECORDS: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

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


@dataclass
class _ProxyImageUpscaleRequest:
    image: bytes
    resolution: Optional[Any] = None
    scale: Optional[Any] = None
    seed: Optional[int] = None
    softness: Optional[float] = None
    quantize: Optional[int] = None
    vae_tiling: Optional[bool] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class _ProxyVideoGenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_2: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class _ProxyImageToVideoRequest:
    image: bytes
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_2: Optional[float] = None
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
        text_to_video_path: Optional[str] = None,
        image_to_video_path: Optional[str] = None,
        image_to_video_mode: str = "multipart",
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout_s = float(timeout_s)
        self.image_generations_path = image_generations_path or "/images/generations"
        self.image_edits_path = image_edits_path or "/images/edits"
        self.text_to_video_path = text_to_video_path
        self.image_to_video_path = image_to_video_path
        self.image_to_video_mode = image_to_video_mode or "multipart"

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
        raise ValueError("Invalid upstream media response: missing data[0].b64_json or data[0].url")

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
            payload.update({k: v for k, v in request.extra.items() if v is not None and not callable(v)})

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
                if value is not None and not callable(value):
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

    def upscale_image(self, request: _ProxyImageUpscaleRequest) -> _ProxyGeneratedAsset:
        _ = request
        raise _ProxyOptionalDependencyMissingError(
            "image_upscale is not configured for this OpenAI-compatible vision backend."
        )

    @staticmethod
    def _video_payload_fields(request: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for attr in (
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "fps",
            "num_frames",
            "seed",
            "steps",
            "guidance_scale",
            "guidance_2",
        ):
            value = getattr(request, attr, None)
            if value is not None:
                payload[attr] = value
        if isinstance(request.extra, dict):
            payload.update({str(k): v for k, v in request.extra.items() if v is not None and not callable(v)})
        return payload

    def generate_video(self, request: _ProxyVideoGenerationRequest) -> _ProxyGeneratedAsset:
        if not self.text_to_video_path:
            raise _ProxyOptionalDependencyMissingError("text_to_video is not configured for this OpenAI-compatible vision backend.")
        payload = {
            "prompt": request.prompt,
            "n": 1,
            "response_format": "b64_json",
            **self._video_payload_fields(request),
        }
        if self.model_id:
            payload["model"] = self.model_id
        payload = {k: v for k, v in payload.items() if v is not None}
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                _join_url(self.base_url, self.text_to_video_path),
                headers=self._headers(),
                json=payload,
            )
            self._raise_for_status(resp)
            data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid upstream video response: expected JSON object")
        return self._parse_media(data, fallback_mime="video/mp4")

    def image_to_video(self, request: _ProxyImageToVideoRequest) -> _ProxyGeneratedAsset:
        if not self.image_to_video_path:
            raise _ProxyOptionalDependencyMissingError("image_to_video is not configured for this OpenAI-compatible vision backend.")
        fields = self._video_payload_fields(request)
        if self.model_id:
            fields["model"] = self.model_id

        if str(self.image_to_video_mode or "multipart").strip().lower() == "json_b64":
            payload = {
                "image_b64": base64.b64encode(bytes(request.image)).decode("ascii"),
                **fields,
            }
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(
                    _join_url(self.base_url, self.image_to_video_path),
                    headers=self._headers(),
                    json=payload,
                )
                self._raise_for_status(resp)
                data = resp.json()
        else:
            files: Dict[str, Tuple[str, bytes, str]] = {
                "image": ("image.png", bytes(request.image), "image/png"),
            }
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(
                    _join_url(self.base_url, self.image_to_video_path),
                    headers=self._headers(),
                    data={str(k): str(v) for k, v in fields.items() if v is not None},
                    files=files,
                )
                self._raise_for_status(resp)
                data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid upstream image-to-video response: expected JSON object")
        return self._parse_media(data, fallback_mime="video/mp4")


class ImageGenerationBody(BaseModel):
    """OpenAI-compatible image generation request body."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "openai-compatible/gpt-image-1",
                    "provider": "openai-compatible",
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
    provider: Optional[str] = Field(
        default=None,
        description=(
            "Optional image provider/backend hint. Prefer this field over encoding backend policy "
            "into `model`; AbstractVision resolves the concrete local engine when possible."
        ),
        examples=["mflux", "huggingface", "openai-compatible"],
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider/model image id. Omit this field to use the server's configured "
            "AbstractVision default. Explicit local models use "
            "`mlx-gen/<exact-huggingface-repo>`, `diffusers/default`, `diffusers/<huggingface-repo>`, "
            "or `sdcpp/default`; remote image providers use "
            "`openai-compatible/my-image-model` with a configured upstream image endpoint."
        ),
        examples=["mlx-gen/AbstractFramework/flux.2-klein-9b-4bit", "openai-compatible/gpt-image-2"],
    )
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional request-level base URL override for OpenAI/OpenAI-compatible image backends. "
            "Loopback URLs are allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    )
    n: Optional[int] = Field(default=1, description="Number of images to generate. Clamped to 1..10.", examples=[1])
    width: Optional[int] = Field(default=None, description="Requested image width in pixels, backend permitting.", examples=[1024])
    height: Optional[int] = Field(default=None, description="Requested image height in pixels, backend permitting.", examples=[1024])
    size: Optional[str] = Field(default=None, description="OpenAI-compatible size selector such as 1024x1024 or auto.", examples=["auto"])
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
        if str(raw_size).strip().lower() == "auto":
            out["size"] = "auto"
            return out
        width, height = _parse_size(raw_size)
        if width is None or height is None:
            raise ValueError("size must use WIDTHxHEIGHT format, for example 1024x1024")
        if out.get("width") is None:
            out["width"] = width
        if out.get("height") is None:
            out["height"] = height
        return out


class VideoGenerationBody(BaseModel):
    """OpenAI-compatible text-to-video generation request body."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": "mlx-gen/AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
                    "provider": "mlx-gen",
                    "prompt": "A slow camera move through a luminous data center.",
                    "width": 432,
                    "height": 240,
                    "fps": 24,
                    "num_frames": 41,
                    "steps": 20,
                    "guidance_scale": 4.0,
                    "guidance_2": 3.0,
                    "response_format": "b64_json",
                    "extra": {"max_sequence_length": 256},
                }
            ]
        },
    )

    prompt: str = Field(..., description="Text prompt describing the video to generate.")
    provider: Optional[str] = Field(
        default=None,
        description="Optional video provider/backend hint, e.g. `mlx-gen`, `diffusers`, or `openai-compatible`.",
        examples=["mlx-gen"],
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider/model video id. Local MLX-Gen video models use exact model ids such as "
            "`mlx-gen/AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`; remote providers use "
            "`openai-compatible/<model>` with a configured upstream video endpoint."
        ),
        examples=["mlx-gen/AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"],
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Optional request-level base URL override for OpenAI-compatible video backends.",
        examples=["http://127.0.0.1:5000/v1"],
    )
    n: Optional[int] = Field(default=1, description="Number of videos to generate. Clamped to 1..4.", examples=[1])
    width: Optional[int] = Field(default=None, description="Requested video width in pixels.", examples=[432])
    height: Optional[int] = Field(default=None, description="Requested video height in pixels.", examples=[240])
    size: Optional[str] = Field(default=None, description="Optional WIDTHxHEIGHT size selector.", examples=["432x240"])
    fps: Optional[int] = Field(default=None, description="Requested frames per second.", examples=[24])
    num_frames: Optional[int] = Field(default=None, description="Requested frame count.", examples=[41])
    frames: Optional[int] = Field(default=None, description="Alias for num_frames.", examples=[41])
    response_format: Optional[str] = Field(default="b64_json", description="Response format. Only `b64_json` is currently supported.", examples=["b64_json"])
    negative_prompt: Optional[str] = Field(default=None, description="Optional negative prompt for backends that support it.")
    seed: Optional[int] = Field(default=None, description="Optional deterministic seed for local generation backends.", examples=[1234])
    steps: Optional[int] = Field(default=None, description="Optional inference step count.", examples=[20])
    guidance_scale: Optional[float] = Field(default=None, description="Optional primary guidance scale.", examples=[4.0])
    guidance_2: Optional[float] = Field(
        default=None,
        description=(
            "Optional second-stage low-noise guidance scale for dual-transformer video models "
            "such as Wan 2.2 A14B."
        ),
        examples=[3.0],
    )
    max_sequence_length: Optional[int] = Field(default=None, description="Optional backend-specific prompt token length cap.", examples=[256])
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Backend-specific parameters forwarded to the selected video backend or upstream endpoint.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("num_frames") is None and out.get("frames") is not None:
            out["num_frames"] = out.get("frames")
        raw_size = out.get("size")
        if raw_size is not None and str(raw_size).strip().lower() != "auto":
            width, height = _parse_size(raw_size)
            if width is None or height is None:
                raise ValueError("size must use WIDTHxHEIGHT format, for example 1280x704")
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


def _progress_value_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(k): _progress_value_json_safe(v)
            for k, v in value.items()
            if isinstance(k, (str, int, float, bool)) and not callable(v)
        }
    if isinstance(value, (list, tuple)):
        return [_progress_value_json_safe(v) for v in value if not callable(v)]
    return str(value)


def _progress_event_payload(event: Any) -> Dict[str, Any]:
    if isinstance(event, dict):
        payload = {
            str(k): _progress_value_json_safe(v)
            for k, v in event.items()
            if not callable(v)
        }
    else:
        payload = {}
        for key in (
            "phase",
            "stage",
            "status",
            "message",
            "frame",
            "total_frames",
            "step",
            "total_steps",
            "progress",
            "step_progress",
            "frame_progress",
            "task",
            "timestep",
            "eta_seconds",
            "raw",
        ):
            if not hasattr(event, key):
                continue
            value = getattr(event, key, None)
            if value is not None and not callable(value):
                payload[key] = _progress_value_json_safe(value)
    if "progress" not in payload:
        frame = _coerce_int(payload.get("frame"))
        total_frames = _coerce_int(payload.get("total_frames"))
        step = _coerce_int(payload.get("step"))
        total_steps = _coerce_int(payload.get("total_steps"))
        if step is not None and total_steps:
            payload["progress"] = max(0.0, min(1.0, float(step) / float(total_steps)))
        elif frame is not None and total_frames:
            payload["progress"] = max(0.0, min(1.0, float(frame) / float(total_frames)))
    return payload


def _job_update_progress(
    job_id: str,
    *,
    step: Optional[int],
    total: Optional[int],
    message: Optional[str] = None,
    event: Optional[Dict[str, Any]] = None,
) -> None:
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
        if isinstance(event, dict):
            safe_event = _progress_event_payload(event)
            prog["last_event"] = safe_event
            event_step = _coerce_int(safe_event.get("step"))
            event_total_steps = _coerce_int(safe_event.get("total_steps"))
            event_frame = _coerce_int(safe_event.get("frame"))
            event_total_frames = _coerce_int(safe_event.get("total_frames"))
            if step is None:
                if event_step is not None:
                    prog["step"] = event_step
                if event_frame is not None:
                    prog["frame"] = event_frame
            if total is None:
                if event_total_steps is not None:
                    prog["total_steps"] = event_total_steps
                if event_total_frames is not None:
                    prog["total_frames"] = event_total_frames
            if "progress" in safe_event:
                prog["progress"] = safe_event.get("progress")
            for key in ("step_progress", "frame_progress"):
                if key in safe_event:
                    prog[key] = safe_event.get(key)
        job["updated_at_s"] = time.time()


def _job_finish(job_id: str, *, ok: bool, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["state"] = "succeeded" if ok else "failed"
        job["updated_at_s"] = time.time()
        prog = job.get("progress")
        if isinstance(prog, dict):
            prog["message"] = "succeeded" if ok else "failed"
        if ok:
            job["result"] = result
            job.pop("error", None)
        else:
            job["error"] = str(error or "Unknown error")
            job.pop("result", None)


def _get_or_create_cached_backend(key: Tuple[Any, ...], factory):
    evicted: list[Tuple[Any, threading.Lock]] = []
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
                removed = _BACKEND_CACHE.pop(k, None)
                _RESIDENCY_RECORDS.pop(k, None)
                if removed is not None:
                    evicted.append((removed[0], removed[1]))

    for evicted_backend, evicted_lock in evicted:
        with evicted_lock:
            _unload_backend_best_effort(evicted_backend)
    return backend, call_lock


def _vision_api_key_fingerprint(api_key: Optional[str]) -> Optional[str]:
    if not isinstance(api_key, str) or not api_key.strip():
        return None
    return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:16]


def _vision_request_model_from_residency_payload(payload: Dict[str, Any]) -> Optional[str]:
    provider = payload.get("provider") or payload.get("backend") or payload.get("backend_kind")
    model = payload.get("model") or payload.get("load_id") or payload.get("runtime_id")
    if isinstance(model, str) and model.strip() and isinstance(provider, str) and provider.strip():
        return str(_scoped_request_model(model, provider))
    if isinstance(model, str) and model.strip():
        return model.strip()
    if isinstance(provider, str) and provider.strip():
        return str(_scoped_request_model(None, provider))
    return None


def _vision_load_id_for_key(key: Tuple[Any, ...]) -> str:
    kind = str(key[0] if key else "").strip()
    if kind == "openai_compatible_proxy":
        model = str(key[3] if len(key) > 3 and key[3] else "default").strip()
        return f"openai-compatible/{model}"
    if kind == "diffusers":
        return f"diffusers/{key[1]}"
    if kind in {"mflux", "mlx-gen"}:
        return f"mlx-gen/{key[1] or key[2] or 'default'}"
    if kind == "sdcpp":
        model = key[2] if len(key) > 2 and key[2] else key[3] if len(key) > 3 and key[3] else "default"
        return f"sdcpp/{model}"
    if kind == "configured" and len(key) >= 5:
        return f"{key[1]}/{key[4] or 'default'}"
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def _unpack_resolved_backend(resolved: Tuple[Any, ...]) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    if len(resolved) < 5:
        raise RuntimeError("Vision backend resolver returned an invalid result.")
    return (
        resolved[0],
        resolved[1],
        resolved[2],
        resolved[3],
        resolved[4],
        resolved[5] if len(resolved) > 5 else None,
        resolved[6] if len(resolved) > 6 else None,
        resolved[7] if len(resolved) > 7 else None,
    )


def _normalize_vision_residency_task(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": None,
        "image": "image_generation",
        "vision": "image_generation",
        "image_generation": "image_generation",
        "text_to_image": "text_to_image",
        "t2i": "text_to_image",
        "image_to_image": "image_to_image",
        "i2i": "image_to_image",
        "image_edit": "image_to_image",
        "image_upscale": "image_upscale",
        "upscale": "image_upscale",
        "upscale_image": "image_upscale",
        "super_resolution": "image_upscale",
        "video_generation": "video_generation",
        "text_to_video": "text_to_video",
        "t2v": "text_to_video",
        "image_to_video": "image_to_video",
        "i2v": "image_to_video",
    }
    return aliases.get(raw, raw or None)


def _vision_record_tasks(task: Optional[str]) -> list[str]:
    task = _normalize_vision_residency_task(task)
    if task in {"video_generation", "text_to_video", "image_to_video"}:
        return ["video_generation", "text_to_video", "image_to_video"]
    if task == "image_upscale":
        return ["image_upscale"]
    return ["image_generation", "text_to_image", "image_to_image"]


def _vision_record_for_key(
    key: Tuple[Any, ...],
    *,
    cache_ts: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = dict(meta or {})
    kind = str(key[0] if key else "").strip()
    loaded_at = meta.get("loaded_at_s") or cache_ts or time.time()
    last_used_at = cache_ts or meta.get("last_used_at_s") or loaded_at
    pinned = bool(meta.get("pinned"))
    source = str(meta.get("source") or ("explicit_preload" if pinned else "request"))
    options: Dict[str, Any] = {}
    provider = kind
    model = "default"
    backend_kind = kind
    isolation = "in_process"
    scope = "process"

    if kind == "openai_compatible_proxy":
        provider = "openai-compatible"
        backend_kind = "openai-compatible"
        model = str(key[3] if len(key) > 3 and key[3] else "default")
        options = {
            "base_url": key[1] if len(key) > 1 else None,
            "timeout_s": key[4] if len(key) > 4 else None,
        }
        isolation = "remote"
        if not pinned:
            source = "configured"
    elif kind == "diffusers":
        provider = "huggingface"
        backend_kind = "diffusers"
        model = str(key[1] if len(key) > 1 else "default")
        options = {
            "device": key[2] if len(key) > 2 else None,
            "torch_dtype": key[3] if len(key) > 3 else None,
            "allow_download": key[4] if len(key) > 4 else None,
            "auto_retry_fp32": key[5] if len(key) > 5 else None,
        }
    elif kind in {"mflux", "mlx-gen"}:
        provider = "mlx-gen"
        backend_kind = "mlx-gen"
        model = str(key[1] if len(key) > 1 and key[1] else key[2] if len(key) > 2 and key[2] else "default")
        options = {
            "base_model": key[2] if len(key) > 2 else None,
            "model_dir": key[3] if len(key) > 3 else None,
            "allow_download": key[4] if len(key) > 4 else None,
        }
    elif kind == "sdcpp":
        provider = "sdcpp"
        backend_kind = "sdcpp"
        model = str(key[2] if len(key) > 2 and key[2] else key[3] if len(key) > 3 and key[3] else "default")
        options = {
            "sd_cli_path": key[1] if len(key) > 1 else None,
            "diffusion_model": key[3] if len(key) > 3 else None,
            "timeout_s": key[11] if len(key) > 11 else None,
        }
    elif kind == "configured" and len(key) >= 5:
        provider = str(key[1])
        backend_kind = str(key[2])
        model = str(key[4] or "default")
        options = {"base_url": key[3] if len(key) > 3 else None}
        isolation = "remote"
        scope = "remote"

    options = {str(k): v for k, v in options.items() if v is not None}
    load_id = _vision_load_id_for_key(key)
    # `loaded` means the backend/runtime is currently held in-process and reusable.
    loaded = isolation == "in_process" and cache_ts is not None
    state = "loaded" if loaded else "configured" if isolation == "remote" else "not_loaded"
    tasks = _vision_record_tasks(meta.get("task"))
    return {
        "runtime_id": load_id,
        "load_id": load_id,
        "task": tasks[0],
        "tasks": tasks,
        "provider": provider,
        "model": model,
        "backend_kind": backend_kind,
        "state": state,
        "loaded": loaded,
        "loaded_at": int(float(loaded_at)),
        "last_used_at": int(float(last_used_at)),
        "request_count": int(meta.get("request_count") or 0),
        "pinned": pinned,
        "source": source,
        "scope": scope,
        "isolation": isolation,
        "health": "ok",
        "error": None,
        "options": options,
    }


def _vision_backend_cache_key_for_backend(backend: Any) -> Optional[Tuple[Any, ...]]:
    with _BACKEND_CACHE_LOCK:
        for key, (candidate, _call_lock, _ts) in _BACKEND_CACHE.items():
            if candidate is backend:
                return key
    return None


def _vision_configured_key(
    *,
    provider: str,
    backend_kind: str,
    base_url: Optional[str],
    api_key: Optional[str],
    model: Optional[str],
) -> Tuple[Any, ...]:
    return (
        "configured",
        str(provider or "").strip().lower(),
        str(backend_kind or "").strip().lower(),
        str(base_url or "").strip().rstrip("/"),
        str(model or "default").strip() or "default",
        _vision_api_key_fingerprint(api_key),
    )


def _vision_record_matches(record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    task = _normalize_vision_residency_task(filters.get("task"))
    if task and task not in set(record.get("tasks") or ()):
        return False
    provider = _normalize_vision_provider_filter(filters.get("provider") or filters.get("backend"))
    if provider:
        record_provider = _normalize_vision_provider_filter(record.get("provider"))
        record_backend = _normalize_vision_provider_filter(record.get("backend_kind"))
        if provider not in {record_provider, record_backend}:
            return False
    backend_kind = _normalize_vision_provider_filter(filters.get("backend_kind"))
    if backend_kind and backend_kind != _normalize_vision_provider_filter(record.get("backend_kind")):
        return False
    model = str(filters.get("model") or "").strip()
    if model and model not in {str(record.get("model") or ""), str(record.get("load_id") or ""), str(record.get("runtime_id") or "")}:
        return False
    runtime_id = str(filters.get("runtime_id") or filters.get("load_id") or "").strip()
    if runtime_id and runtime_id not in {str(record.get("runtime_id") or ""), str(record.get("load_id") or "")}:
        return False
    return True


def _vision_loaded_records(filters: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    filters = dict(filters or {})
    with _BACKEND_CACHE_LOCK:
        records = [
            _vision_record_for_key(key, cache_ts=ts, meta=_RESIDENCY_RECORDS.get(key))
            for key, (_backend, _call_lock, ts) in _BACKEND_CACHE.items()
        ]
        cached_keys = set(_BACKEND_CACHE)
        for key, meta in _RESIDENCY_RECORDS.items():
            if key not in cached_keys:
                records.append(_vision_record_for_key(key, meta=meta))
    out = [record for record in records if _vision_record_matches(record, filters)]
    return sorted(out, key=lambda item: str(item.get("load_id") or item.get("runtime_id") or ""))


def list_server_vision_loaded_models(filters: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    """List server-local image backends visible to `/v1/images/*`."""
    return _vision_loaded_records(filters)


def load_server_vision_loaded_model(
    request: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    payload = dict(request or {})
    options = payload.get("options")
    if isinstance(options, dict):
        merged = dict(options)
        merged.update({k: v for k, v in payload.items() if k != "options"})
        payload = merged
    request_model = _vision_request_model_from_residency_payload(payload)
    backend_kind = _effective_backend_kind(request_model)
    base_url = payload.get("base_url")
    model_id = str(payload.get("model") or "").strip() or None

    if backend_kind == "openai_compatible_proxy":
        base_url_s = _validate_request_base_url(base_url) if isinstance(base_url, str) and base_url.strip() else _upstream_base_url_for_request(request_model)
        configured_key = _vision_configured_key(
            provider="openai-compatible",
            backend_kind="openai-compatible",
            base_url=base_url_s,
            api_key=api_key,
            model=model_id or _upstream_model_env(),
        )
        now = time.time()
        with _BACKEND_CACHE_LOCK:
            loaded_new = configured_key not in _RESIDENCY_RECORDS
            _RESIDENCY_RECORDS[configured_key] = {
                "source": "configured",
                "task": _normalize_vision_residency_task(payload.get("task")),
                "loaded_at_s": now,
                "last_used_at_s": now,
                "pinned": False,
            }
            record = _vision_record_for_key(configured_key, meta=_RESIDENCY_RECORDS[configured_key])
        record["loaded_new"] = bool(loaded_new)
        return record

    backend, call_lock, _missing, _gen_req, _edit_req, _video_req, _i2v_req, _upscale_req = _unpack_resolved_backend(
        _resolve_backend(
            request_model,
            base_url=payload.get("base_url"),
            api_key=api_key,
        )
    )
    preload = getattr(backend, "preload", None)
    if callable(preload):
        with call_lock:
            preload()
    key = _vision_backend_cache_key_for_backend(backend)
    if key is None:
        raise RuntimeError("Vision backend was loaded but not present in the server backend cache.")
    now = time.time()
    with _BACKEND_CACHE_LOCK:
        loaded_new = key not in _RESIDENCY_RECORDS
        _RESIDENCY_RECORDS[key] = {
            "source": "explicit_preload",
            "task": _normalize_vision_residency_task(payload.get("task")),
            "loaded_at_s": now,
            "last_used_at_s": now,
            "pinned": bool(payload.get("pin", True)),
        }
        record = _vision_record_for_key(key, cache_ts=now, meta=_RESIDENCY_RECORDS[key])
    record["loaded_new"] = bool(loaded_new)
    return record


def unload_server_vision_loaded_model(request: Dict[str, Any]) -> Dict[str, Any]:
    filters = dict(request or {})
    matches = _vision_loaded_records(filters)
    if len(matches) > 1:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Ambiguous image model unload request. Specify runtime_id/load_id or provider+model.",
                    "type": "invalid_request",
                }
            },
        )
    if not matches:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": "Loaded image model not found.", "type": "not_found"}},
        )

    target = matches[0]
    target_id = str(target.get("load_id") or target.get("runtime_id") or "")
    removed_backend: Any = None
    removed_lock: Optional[threading.Lock] = None
    with _BACKEND_CACHE_LOCK:
        target_key = None
        for key, (backend, call_lock, _ts) in list(_BACKEND_CACHE.items()):
            record = _vision_record_for_key(key, meta=_RESIDENCY_RECORDS.get(key))
            if str(record.get("load_id") or record.get("runtime_id") or "") == target_id:
                target_key = key
                removed_backend = backend
                removed_lock = call_lock
                _BACKEND_CACHE.pop(key, None)
                _RESIDENCY_RECORDS.pop(key, None)
                break
        if target_key is None:
            for key in list(_RESIDENCY_RECORDS):
                record = _vision_record_for_key(key, meta=_RESIDENCY_RECORDS.get(key))
                if str(record.get("load_id") or record.get("runtime_id") or "") == target_id:
                    _RESIDENCY_RECORDS.pop(key, None)
                    break

    if removed_backend is not None:
        if removed_lock is not None:
            with removed_lock:
                _unload_backend_best_effort(removed_backend)
        else:
            _unload_backend_best_effort(removed_backend)

    out = dict(target)
    out.update({"state": "unloaded", "loaded": False, "unloaded": True})
    return out


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


def _extract_bearer_token(auth_header: Any) -> str:
    auth = str(auth_header or "").strip()
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


def _extract_secret_header(header_value: Any) -> str:
    value = str(header_value or "").strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value

_SERVER_AUTH_TOKEN_ENV_VAR = "ABSTRACTCORE_AUTH_TOKEN"


def _server_auth_enabled() -> bool:
    value = os.getenv(_SERVER_AUTH_TOKEN_ENV_VAR)
    return isinstance(value, str) and bool(value.strip())


def _request_has_server_auth(request: Request) -> bool:
    return bool(getattr(request.state, "abstractcore_server_authenticated", False))


def _is_placeholder_api_key(token: Any) -> bool:
    text = str(token or "").strip()
    return not text or text.lower() in {"not-needed", "not_needed", "notneeded", "unused", "dummy", "empty", "none"}


def _provider_api_key_from_request(request: Request) -> Optional[str]:
    for header_name in ("x-abstractcore-provider-api-key", "x-provider-api-key"):
        token = _extract_secret_header(request.headers.get(header_name))
        if not _is_placeholder_api_key(token):
            return token

    # When server auth is enabled, `Authorization` is reserved for the server
    # auth token and must never be forwarded upstream.
    if _server_auth_enabled():
        return None

    # Legacy/Swagger UI convenience: when server auth is disabled, treat
    # `Authorization: Bearer <token>` as the upstream provider key.
    token = _extract_bearer_token(request.headers.get("authorization"))
    if _is_placeholder_api_key(token):
        return None
    return token


def _is_loopback_host(host: str) -> bool:
    h = str(host or "").strip().lower().strip("[]")
    return h in {"localhost", "127.0.0.1", "::1"} or h.startswith("127.")


def _allowlist_matches_url(url: str, allowlist: list[str]) -> bool:
    parsed = urllib.parse.urlsplit(str(url or "").strip())
    host = str(parsed.hostname or "").lower()
    netloc = str(parsed.netloc or "").lower()
    full = str(url or "").strip().lower().rstrip("/")
    for item in allowlist:
        raw = str(item or "").strip().lower().rstrip("/")
        if not raw:
            continue
        if raw.startswith(("http://", "https://")):
            if full.startswith(raw):
                return True
            continue
        if raw == host or raw == netloc:
            return True
    return False


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name)
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _validate_request_base_url(base_url: Any) -> Optional[str]:
    if not isinstance(base_url, str) or not base_url.strip():
        return None
    value = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="base_url must be an absolute http(s) URL.")
    if _is_loopback_host(parsed.hostname):
        return value
    if _allowlist_matches_url(value, _csv_env("ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST")):
        return value
    raise HTTPException(
        status_code=403,
        detail=(
            "Request-level base_url overrides are restricted for security. "
            "Loopback URLs are allowed by default; set ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST "
            "to allow additional hosts or URL prefixes."
        ),
    )


def _server_has_vision_catalog_credential() -> bool:
    return bool(str(os.getenv("OPENAI_API_KEY") or "").strip())


def _server_allows_unauthenticated() -> bool:
    return _env_bool("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", default=False)


def _looks_like_openai_api(base_url: Any) -> bool:
    return "api.openai.com" in str(base_url or "").lower()


def _vision_catalog_api_key_for_base_url(base_url: Any) -> Optional[str]:
    return _env("OPENAI_API_KEY")


def _guard_vision_catalog_credentials(*, request: Request, explicit_provider_key: bool) -> None:
    if _request_has_server_auth(request) or explicit_provider_key:
        return
    if _server_allows_unauthenticated():
        return
    if not _server_has_vision_catalog_credential():
        return
    raise HTTPException(
        status_code=401,
        detail=(
            "Server-held vision/OpenAI credentials are configured, but inbound server auth was not used. "
            "Set ABSTRACTCORE_AUTH_TOKEN and send "
            "Authorization: Bearer <server-token>, or pass an explicit "
            "provider key with X-AbstractCore-Provider-API-Key for this request."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


def _vision_catalog_config_from_env() -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    env_map = {
        "ABSTRACTCORE_VISION_BACKEND": "vision_backend",
        "ABSTRACTVISION_BACKEND": "vision_backend",
        "OPENAI_BASE_URL": "vision_base_url",
        "ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID": "vision_model_id",
        "ABSTRACTCORE_VISION_MODEL_ID": "vision_model_id",
        "ABSTRACTVISION_MODEL_ID": "vision_model_id",
        "ABSTRACTCORE_VISION_MFLUX_MODEL": "vision_mflux_model",
        "ABSTRACTVISION_MFLUX_MODEL": "vision_mflux_model",
        "ABSTRACTCORE_VISION_MFLUX_BASE_MODEL": "vision_mflux_base_model",
        "ABSTRACTVISION_MFLUX_BASE_MODEL": "vision_mflux_base_model",
        "ABSTRACTCORE_VISION_MODEL_DIR": "vision_model_dir",
        "ABSTRACTVISION_MODEL_DIR": "vision_model_dir",
        "ABSTRACTCORE_VISION_MFLUX_ALLOW_DOWNLOAD": "vision_mflux_allow_download",
        "ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD": "vision_mflux_allow_download",
        "ABSTRACTCORE_VISION_TIMEOUT_S": "vision_timeout_s",
        "ABSTRACTVISION_TIMEOUT_S": "vision_timeout_s",
        "ABSTRACTCORE_VISION_MODELS_PATH": "vision_models_path",
        "ABSTRACTVISION_MODELS_PATH": "vision_models_path",
        "ABSTRACTCORE_VISION_SDCPP_MODEL": "vision_sdcpp_model",
        "ABSTRACTVISION_SDCPP_MODEL": "vision_sdcpp_model",
        "ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL": "vision_sdcpp_diffusion_model",
        "ABSTRACTVISION_SDCPP_DIFFUSION_MODEL": "vision_sdcpp_diffusion_model",
    }
    for env_name, config_name in env_map.items():
        value = _env(env_name)
        if value is not None and config_name not in config:
            config[config_name] = value
    api_key = _vision_catalog_api_key_for_base_url(config.get("vision_base_url"))
    if api_key:
        config["vision_api_key"] = api_key
    return config


def _vision_catalog_core(request: Request, *, base_url: Optional[str], api_key: Optional[str]) -> Any:
    explicit_key = str(api_key).strip() if isinstance(api_key, str) else ""
    if _is_placeholder_api_key(explicit_key):
        explicit_key = ""
    provider_api_key = explicit_key or _provider_api_key_from_request(request)
    base_url_s = _validate_request_base_url(base_url)
    _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
    config = _vision_catalog_config_from_env()
    if base_url_s:
        config["vision_base_url"] = base_url_s
        config.setdefault("vision_backend", "openai-compatible")
        if not provider_api_key:
            api_key = _vision_catalog_api_key_for_base_url(base_url_s)
            if api_key:
                config["vision_api_key"] = api_key
    if provider_api_key:
        config["vision_api_key"] = provider_api_key
    return create_capability_generation_core(**config)


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


def _mflux_model_env() -> Optional[str]:
    return _env_first(
        "ABSTRACTCORE_VISION_MFLUX_MODEL",
        "ABSTRACTVISION_MFLUX_MODEL",
        "ABSTRACTCORE_VISION_MODEL_ID",
        "ABSTRACTVISION_MODEL_ID",
    )


def _upstream_base_url_env() -> Optional[str]:
    return _env("OPENAI_BASE_URL")


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
    if v in {"mflux", "m-flux", "mlx-gen", "mlxgen"}:
        return "mlx-gen"
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
    "mlx-gen",
    "mlxgen",
    "mflux",
    "m-flux",
    "diffusers",
    "sdcpp",
}


def _normalize_provider_route_prefix(provider: Any) -> str:
    return str(provider or "").strip().lower().replace("_", "-")


def _normalize_vision_provider_filter(provider: Any) -> str:
    raw = _normalize_provider_route_prefix(provider)
    if not raw:
        return ""
    aliases = {
        "openai_compatible": "openai-compatible",
        "openai-compatible": "openai-compatible",
        "hf": "huggingface",
        "mlx": "huggingface",
        "mflux": "mlx-gen",
        "m-flux": "mlx-gen",
        "mlxgen": "mlx-gen",
        "mlx-gen": "mlx-gen",
        "sd-cpp": "sdcpp",
        "stable-diffusion.cpp": "sdcpp",
        "stable-diffusion-cpp": "sdcpp",
        "stable_diffusion_cpp": "sdcpp",
    }
    return aliases.get(raw, raw)


def _vision_filter_provider_payload(payload: Dict[str, Any], provider: Any) -> Dict[str, Any]:
    provider_norm = _normalize_vision_provider_filter(provider)
    if not provider_norm:
        return payload
    provider_lc = provider_norm.lower()

    out = dict(payload)
    out["providers"] = [
        str(p).strip()
        for p in (out.get("providers") or [])
        if isinstance(p, str) and str(p).strip().lower() == provider_lc
    ] or [provider_norm]
    out["available_providers"] = [
        str(p).strip()
        for p in (out.get("available_providers") or [])
        if isinstance(p, str) and str(p).strip().lower() == provider_lc
    ]

    models_by_provider = out.get("models_by_provider")
    if isinstance(models_by_provider, dict) and models_by_provider:
        filtered = {k: v for k, v in models_by_provider.items() if str(k).strip().lower() == provider_lc}
        out["models_by_provider"] = filtered
        out["models"] = [m for values in filtered.values() for m in (values or [])]
        return out

    models = out.get("models")
    if isinstance(models, list) and models:
        out["models"] = [
            item
            for item in models
            if str(getattr(item, "get", lambda *_: "")("provider") or "").strip().lower() == provider_lc
        ]
    return out


def _looks_like_openai_image_model(model: Any) -> bool:
    value = str(model or "").strip().lower()
    return value.startswith(("gpt-image", "dall-e"))


def _is_openai_image_request_model(request_model: Any) -> bool:
    prefix, rest = _split_known_prefix(str(request_model or "").strip())
    return prefix == "openai" and _looks_like_openai_image_model(rest)


def _payload_with_path_provider(payload: Dict[str, Any], provider: Any) -> Dict[str, Any]:
    provider_s = _normalize_provider_route_prefix(provider)
    if not provider_s:
        return payload

    out = dict(payload)
    model = out.get("model")
    if isinstance(model, str) and model.strip():
        prefix, rest = _split_known_prefix(model)
        if prefix is not None:
            out["model"] = rest
    out["provider"] = provider_s
    return out


def _split_known_prefix(model: str) -> tuple[Optional[str], str]:
    s = str(model or "").strip()
    if not s or "/" not in s:
        return None, s
    head, tail = s.split("/", 1)
    head_s = head.strip()
    if head_s in _KNOWN_MODEL_PREFIXES:
        return head_s, tail.strip()
    return None, s


def _scoped_request_model(model: Any, provider: Any = None) -> Any:
    provider_s = str(provider or "").strip()
    model_s = str(model or "").strip()
    if not provider_s:
        return model
    provider_l = provider_s.lower().replace("_", "-")
    if model_s:
        prefix, _rest = _split_known_prefix(model_s)
        if prefix is not None:
            return model_s
        return f"{provider_l}/{model_s}"
    return f"{provider_l}/default"


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
            return model
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image model {model!r} must use provider/model routing. "
                "Use `diffusers/<huggingface-repo>`, `sdcpp/default`, "
                "or `openai-compatible/<model>`."
            ),
        )

    # Local generation backends: strip prefix and pass through.
    if prefix in {"huggingface", "hf", "mlx", "diffusers", "mlx-gen", "mlxgen", "mflux", "m-flux"}:
        if _is_default_model_alias(rest):
            return None
        return rest or None
    if prefix == "sdcpp":
        if _is_default_model_alias(rest):
            return None
        return rest or None

    if prefix in {"openai-compatible", "openai_compatible"}:
        return rest or None
    if prefix == "openai" and _looks_like_openai_image_model(rest):
        return rest or None
    if prefix == "openai" and rest:
        nested_prefix, _nested_rest = _split_known_prefix(rest)
        if nested_prefix in {"openai-compatible", "openai_compatible", "openai"}:
            return _normalize_request_model_for_backend(rest)

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
    if prefix in {"mlx-gen", "mlxgen", "mflux", "m-flux"}:
        return "mlx-gen"
    if prefix == "sdcpp":
        return "sdcpp"
    if prefix in {"huggingface", "hf", "mlx", "diffusers"}:
        return "diffusers"
    if prefix in {"openai-compatible", "openai_compatible"}:
        return "openai_compatible_proxy"
    if prefix == "openai" and _looks_like_openai_image_model(_split_known_prefix(raw_model)[1]):
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
    if _mflux_model_env():
        return "mlx-gen"
    if _diffusers_model_env():
        return "diffusers"
    return "auto_unconfigured"


def _effective_backend_kind(request_model: Any) -> str:
    raw_model = str(request_model or "").strip()
    prefix, _ = _split_known_prefix(raw_model)
    if prefix in {"mlx-gen", "mlxgen", "mflux", "m-flux"}:
        return "mlx-gen"
    if prefix == "sdcpp":
        return "sdcpp"
    if prefix in {"huggingface", "hf", "mlx", "diffusers"}:
        return "diffusers"
    if prefix in {"openai-compatible", "openai_compatible"}:
        return "openai_compatible_proxy"
    if prefix == "openai" and _looks_like_openai_image_model(_split_known_prefix(raw_model)[1]):
        return "openai_compatible_proxy"
    if _looks_like_hf_repo_id(raw_model):
        return "diffusers"

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


def _require_upstream_base_url() -> str:
    base_url = _upstream_base_url_env()
    if not base_url:
        raise HTTPException(
            status_code=501,
            detail=(
                "Vision image endpoints are not configured. "
                "Set OPENAI_BASE_URL to an OpenAI-compatible server base URL "
                "(e.g. https://api.openai.com/v1 or http://localhost:1234/v1)."
            ),
        )
    return base_url


def _upstream_base_url_for_request(request_model: Any) -> str:
    base_url = _upstream_base_url_env()
    if base_url:
        return base_url
    if _is_openai_image_request_model(request_model):
        return "https://api.openai.com/v1"
    return _require_upstream_base_url()


def _upstream_api_key_for_request(request_model: Any) -> Optional[str]:
    if _is_openai_image_request_model(request_model) or _upstream_base_url_env():
        return _env("OPENAI_API_KEY")
    return None


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
                "OPENAI_BASE_URL for an OpenAI-compatible image endpoint."
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
            MFluxBackendConfig,
            MFluxVisionBackend,
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from abstractvision.errors import OptionalDependencyMissingError  # type: ignore
        from abstractvision.types import (  # type: ignore
            ImageEditRequest,
            ImageGenerationRequest,
            ImageToVideoRequest,
            ImageUpscaleRequest,
            VideoGenerationRequest,
        )
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
        MFluxBackendConfig,
        MFluxVisionBackend,
        StableDiffusionCppBackendConfig,
        StableDiffusionCppVisionBackend,
        OptionalDependencyMissingError,
        (
            ImageGenerationRequest,
            ImageEditRequest,
            VideoGenerationRequest,
            ImageToVideoRequest,
            ImageUpscaleRequest,
        ),
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
    "provider",
    "model",
    "base_url",
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


_VIDEO_GENERATION_CORE_FIELDS = {
    "prompt",
    "provider",
    "model",
    "base_url",
    "n",
    "size",
    "response_format",
    "width",
    "height",
    "fps",
    "num_frames",
    "frames",
    "negative_prompt",
    "seed",
    "steps",
    "guidance_scale",
    "guidance_2",
    "extra",
}


def _image_generation_request_parts(
    payload: Dict[str, Any],
    *,
    request_model: Any = None,
) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
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
    effective_request_model = request_model if request_model is not None else _scoped_request_model(
        payload.get("model"),
        payload.get("provider"),
    )
    if _effective_backend_kind(effective_request_model) == "openai_compatible_proxy":
        size = payload.get("size")
        if size is None and width is not None and height is not None:
            size = f"{int(width)}x{int(height)}"
        if size is not None:
            extra.setdefault("size", str(size))
        width = None
        height = None

    return width, height, extra


def _video_generation_request_parts(
    payload: Dict[str, Any],
    *,
    request_model: Any = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Dict[str, Any]]:
    width = _coerce_int(payload.get("width"))
    height = _coerce_int(payload.get("height"))
    if (width is None or height is None) and payload.get("size") is not None:
        w2, h2 = _parse_size(payload.get("size"))
        width = width if width is not None else w2
        height = height if height is not None else h2
    fps = _coerce_int(payload.get("fps"))
    num_frames = _coerce_int(payload.get("num_frames"))
    if num_frames is None:
        num_frames = _coerce_int(payload.get("frames"))

    extra = {k: v for k, v in payload.items() if k not in _VIDEO_GENERATION_CORE_FIELDS and v is not None}
    nested_extra = payload.get("extra")
    if isinstance(nested_extra, dict):
        extra.update({str(k): v for k, v in nested_extra.items() if v is not None})

    if payload.get("max_sequence_length") is not None:
        extra.setdefault("max_sequence_length", payload.get("max_sequence_length"))

    effective_request_model = request_model if request_model is not None else _scoped_request_model(
        payload.get("model"),
        payload.get("provider"),
    )
    if _effective_backend_kind(effective_request_model) == "openai_compatible_proxy" and payload.get("size") is not None:
        extra.setdefault("size", str(payload.get("size")))

    return width, height, fps, num_frames, extra


def _scoped_request_model_for_request(model: Any, provider: Any = None, *, base_url: Any = None) -> Any:
    request_model = _scoped_request_model(model, provider)
    if request_model:
        return request_model

    base_url_s = str(base_url or "").strip()
    provider_s = _normalize_provider_route_prefix(provider)
    if not base_url_s:
        return request_model

    if provider_s in {"huggingface", "hf", "diffusers", "mlx-gen", "mlxgen", "mflux", "m-flux", "sdcpp", "sd-cpp"}:
        return request_model

    remote_provider = "openai" if _looks_like_openai_api(base_url_s) else "openai-compatible"
    default_model = "gpt-image-1" if remote_provider == "openai" else "default"
    return _scoped_request_model(default_model, remote_provider)


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


def _coerce_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _blank_form_value(v: Any) -> bool:
    if v is None:
        return True
    return str(v).strip().lower() in {"", "none", "null"}


def _coerce_form_int(name: str, v: Any, *, allowed: Optional[set[int]] = None) -> Optional[int]:
    if _blank_form_value(v):
        return None
    if isinstance(v, bool):
        raise HTTPException(status_code=400, detail=f"{name} must be an integer.")
    try:
        value = int(str(v).strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer.") from e
    if allowed is not None and value not in allowed:
        allowed_s = ", ".join(str(x) for x in sorted(allowed))
        raise HTTPException(status_code=400, detail=f"{name} must be one of: {allowed_s}.")
    return value


def _coerce_form_float(
    name: str,
    v: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if _blank_form_value(v):
        return None
    if isinstance(v, bool):
        raise HTTPException(status_code=400, detail=f"{name} must be a number.")
    try:
        value = float(str(v).strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{name} must be a number.") from e
    if minimum is not None and value < minimum:
        raise HTTPException(status_code=400, detail=f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise HTTPException(status_code=400, detail=f"{name} must be <= {maximum}.")
    return value


def _coerce_form_bool(name: str, v: Any) -> Optional[bool]:
    if _blank_form_value(v):
        return None
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"{name} must be a boolean.")


def _validate_scale_factor_field(name: str, v: Optional[str]) -> Optional[str]:
    if _blank_form_value(v):
        return None
    text = str(v).strip().lower()
    numeric_text = text[:-1].strip() if text.endswith("x") else text
    try:
        value = float(numeric_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{name} must be a positive scale factor.") from e
    if value <= 0:
        raise HTTPException(status_code=400, detail=f"{name} must be positive.")
    return str(v).strip()


def _validate_upscale_resolution_field(v: Optional[str]) -> Optional[str]:
    if _blank_form_value(v):
        return None
    text = str(v).strip().lower()
    if text.endswith("x"):
        return _validate_scale_factor_field("resolution", text)
    try:
        value = int(text)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="resolution must be a positive integer shortest edge or scale factor such as 2x.",
        ) from e
    if value <= 0:
        raise HTTPException(status_code=400, detail="resolution must be positive.")
    return str(v).strip()


def _parse_upscale_form_fields(
    *,
    scale: Optional[str],
    resolution: Optional[str],
    softness: Optional[str],
    seed: Optional[str],
    quantize: Optional[str],
    vae_tiling: Optional[str],
) -> Dict[str, Any]:
    return {
        "scale": _validate_scale_factor_field("scale", scale),
        "resolution": _validate_upscale_resolution_field(resolution),
        "softness": _coerce_form_float("softness", softness, minimum=0.0, maximum=1.0),
        "seed": _coerce_form_int("seed", seed),
        "quantize": _coerce_form_int("quantize", quantize, allowed={3, 4, 5, 6, 8}),
        "vae_tiling": _coerce_form_bool("vae_tiling", vae_tiling),
    }


def _parse_extra_json_form(extra_json: Optional[str]) -> Dict[str, Any]:
    if not extra_json:
        return {}
    try:
        parsed = json.loads(str(extra_json))
    except Exception as e:
        raise HTTPException(status_code=400, detail="extra_json must be a JSON object string") from e
    if parsed is None:
        return {}
    if isinstance(parsed, dict):
        return dict(parsed)
    raise HTTPException(status_code=400, detail="extra_json must be a JSON object string")


def _resolve_backend(request_model: Any, *, base_url: Optional[str] = None, api_key: Optional[str] = None):
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
                "- OPENAI_BASE_URL (OpenAI-compatible proxy)\n"
                "- ABSTRACTCORE_VISION_SDCPP_MODEL / ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL (stable-diffusion.cpp)"
            ),
        )

    # Validate backend-specific configuration before importing AbstractVision.
    # This keeps error messages stable and avoids optional dependency requirements for unconfigured setups.
    prevalidated: Dict[str, Any] = {"backend_kind": backend_kind}
    if backend_kind == "openai_compatible_proxy":
        explicit_base_url = _validate_request_base_url(base_url)
        explicit_api_key = str(api_key).strip() if isinstance(api_key, str) and str(api_key).strip() else None
        base_url = explicit_base_url or _upstream_base_url_for_request(request_model)
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
                "text_to_video_path": _env_first(
                    "ABSTRACTCORE_VISION_UPSTREAM_VIDEOS_GENERATIONS_PATH",
                    "ABSTRACTCORE_VISION_TEXT_TO_VIDEO_PATH",
                    "ABSTRACTVISION_TEXT_TO_VIDEO_PATH",
                    default="/videos/generations",
                )
                or "/videos/generations",
                "image_to_video_path": _env_first(
                    "ABSTRACTCORE_VISION_UPSTREAM_VIDEOS_EDITS_PATH",
                    "ABSTRACTCORE_VISION_IMAGE_TO_VIDEO_PATH",
                    "ABSTRACTVISION_IMAGE_TO_VIDEO_PATH",
                    default="/videos/edits",
                )
                or "/videos/edits",
                "image_to_video_mode": _env_first(
                    "ABSTRACTCORE_VISION_IMAGE_TO_VIDEO_MODE",
                    "ABSTRACTVISION_IMAGE_TO_VIDEO_MODE",
                    default="multipart",
                )
                or "multipart",
                "api_key": explicit_api_key or _upstream_api_key_for_request(request_model),
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
            prevalidated["text_to_video_path"],
            prevalidated["image_to_video_path"],
            prevalidated["image_to_video_mode"],
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
                text_to_video_path=prevalidated["text_to_video_path"],
                image_to_video_path=prevalidated["image_to_video_path"],
                image_to_video_mode=prevalidated["image_to_video_mode"],
            ),
        )
        return (
            backend,
            call_lock,
            _ProxyOptionalDependencyMissingError,
            _ProxyImageGenerationRequest,
            _ProxyImageEditRequest,
            _ProxyVideoGenerationRequest,
            _ProxyImageToVideoRequest,
            _ProxyImageUpscaleRequest,
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
    elif backend_kind == "mlx-gen":
        model_id = str(_normalize_request_model_for_backend(request_model) or _mflux_model_env() or "").strip() or None
        prevalidated.update(
            {
                "model_id": model_id,
                "base_model": _env_first("ABSTRACTCORE_VISION_MFLUX_BASE_MODEL", "ABSTRACTVISION_MFLUX_BASE_MODEL"),
                "model_dir": _env_first("ABSTRACTCORE_VISION_MODEL_DIR", "ABSTRACTVISION_MODEL_DIR"),
                "allow_download": _env_bool_first(
                    "ABSTRACTCORE_VISION_MFLUX_ALLOW_DOWNLOAD",
                    "ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD",
                    default=False,
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
        MFluxBackendConfig,
        MFluxVisionBackend,
        StableDiffusionCppBackendConfig,
        StableDiffusionCppVisionBackend,
        OptionalDependencyMissingError,
        req_types,
    ) = _import_abstractvision()
    ImageGenerationRequest = req_types[0]
    ImageEditRequest = req_types[1]
    VideoGenerationRequest = req_types[2] if len(req_types) > 2 else None
    ImageToVideoRequest = req_types[3] if len(req_types) > 3 else None
    ImageUpscaleRequest = req_types[4] if len(req_types) > 4 else None

    active_model_id, active_kind, active_backend, active_call_lock = _get_active_backend()
    if active_backend is not None and active_call_lock is not None and (not req_model or req_model == active_model_id):
        return (
            active_backend,
            active_call_lock,
            OptionalDependencyMissingError,
            ImageGenerationRequest,
            ImageEditRequest,
            VideoGenerationRequest,
            ImageToVideoRequest,
            ImageUpscaleRequest,
        )

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
        return (
            backend,
            call_lock,
            OptionalDependencyMissingError,
            ImageGenerationRequest,
            ImageEditRequest,
            VideoGenerationRequest,
            ImageToVideoRequest,
            ImageUpscaleRequest,
        )

    if backend_kind == "mlx-gen":
        cfg = MFluxBackendConfig(
            model=prevalidated["model_id"],
            base_model=prevalidated["base_model"],
            model_dir=prevalidated["model_dir"],
            allow_download=prevalidated["allow_download"],
        )
        key = (
            "mlx-gen",
            prevalidated["model_id"],
            prevalidated["base_model"],
            prevalidated["model_dir"],
            prevalidated["allow_download"],
        )
        backend, call_lock = _get_or_create_cached_backend(key, lambda: MFluxVisionBackend(config=cfg))
        return (
            backend,
            call_lock,
            OptionalDependencyMissingError,
            ImageGenerationRequest,
            ImageEditRequest,
            VideoGenerationRequest,
            ImageToVideoRequest,
            ImageUpscaleRequest,
        )

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
        return (
            backend,
            call_lock,
            OptionalDependencyMissingError,
            ImageGenerationRequest,
            ImageEditRequest,
            VideoGenerationRequest,
            ImageToVideoRequest,
            ImageUpscaleRequest,
        )

    raise HTTPException(status_code=501, detail=f"Unknown vision backend kind: {backend_kind!r} (set ABSTRACTCORE_VISION_BACKEND)")


def _create_vision_generation_core(
    request_model: Any,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    (
        backend,
        call_lock,
        OptionalDependencyMissingError,
        ImageGenerationRequest,
        ImageEditRequest,
        VideoGenerationRequest,
        ImageToVideoRequest,
        ImageUpscaleRequest,
    ) = _unpack_resolved_backend(
        _resolve_backend(
            request_model,
            base_url=base_url,
            api_key=api_key,
        )
    )
    facade = ServerVisionFacade(
        backend=backend,
        call_lock=call_lock,
        image_generation_request_cls=ImageGenerationRequest,
        image_edit_request_cls=ImageEditRequest,
        video_generation_request_cls=VideoGenerationRequest,
        image_to_video_request_cls=ImageToVideoRequest,
        image_upscale_request_cls=ImageUpscaleRequest,
        backend_id=f"abstractcore-server:{_effective_backend_kind(request_model)}",
    )
    core = create_capability_generation_core(vision_facade=facade)
    return core, OptionalDependencyMissingError


def _is_remote_vision_request(request_model: Any) -> bool:
    return _effective_backend_kind(request_model) == "openai_compatible_proxy"


def _first_generated_image_bytes(result: Any) -> bytes:
    image_items = getattr(result, "outputs", {}).get("image", [])
    item = image_items[0] if image_items else None
    data = getattr(item, "data", None)
    if not isinstance(data, (bytes, bytearray)):
        raise RuntimeError("Image backend returned an unexpected type (expected raw bytes).")
    return bytes(data)


def _first_generated_video_bytes(result: Any) -> bytes:
    video_items = getattr(result, "outputs", {}).get("video", [])
    item = video_items[0] if video_items else None
    data = getattr(item, "data", None)
    if not isinstance(data, (bytes, bytearray)):
        raise RuntimeError("Video backend returned an unexpected type (expected raw bytes).")
    return bytes(data)


def _vision_catalog_error(exc: Exception) -> HTTPException:
    detail = {
        "available": False,
        "source": "abstractvision",
        "stale": False,
        "error": str(exc),
    }
    if isinstance(exc, CapabilityUnavailableError):
        return HTTPException(status_code=501, detail=detail)
    msg = str(exc).lower()
    if "not configured" in msg or "missing" in msg or "does not support" in msg or "does not expose" in msg:
        return HTTPException(status_code=501, detail=detail)
    return HTTPException(status_code=502, detail=detail)


def _vision_provider_model_capabilities_for_task(task: Optional[str]) -> list[str]:
    image_caps = ["text_to_image", "image_to_image", "image_generation", "image_edit", "image_upscale"]
    video_caps = ["text_to_video", "image_to_video", "video_generation"]
    normalized = _normalize_vision_residency_task(task)
    if normalized in {"text_to_video", "image_to_video", "video_generation"}:
        return video_caps
    if normalized == "image_upscale":
        return ["image_upscale"]
    if normalized in {"text_to_image", "image_to_image", "image_generation", "image_edit"}:
        return image_caps
    if normalized:
        return []
    return image_caps + video_caps


def _configured_vision_provider_model_entries(task: Optional[str]) -> list[Dict[str, Any]]:
    caps = _vision_provider_model_capabilities_for_task(task)
    if not caps:
        return []
    entries: list[Dict[str, Any]] = []

    def add(*, provider: str, model_id: Optional[str]) -> None:
        mid = str(model_id or "").strip()
        if not mid:
            return
        provider_s = str(provider or "").strip() or "openai-compatible"
        raw_id = mid
        for prefix in ("openai-compatible/", "openai/"):
            if raw_id.startswith(prefix):
                raw_id = raw_id[len(prefix) :]
                break
        routed = mid if mid.startswith(f"{provider_s}/") else f"{provider_s}/{raw_id}"
        entries.append(
            {
                "id": raw_id,
                "model": routed,
                "provider": provider,
                "routed_model": routed,
                "object": "model",
                "owned_by": provider,
                "capabilities": caps,
                "raw": {
                    "id": raw_id,
                    "provider": provider,
                    "backend": "openai-compatible",
                    "routed_model": routed,
                    "configured": True,
                },
            }
        )

    openai_model = _env_first("OPENAI_IMAGE_MODEL_ID", "OPENAI_IMAGE_MODEL")
    if openai_model and _env("OPENAI_API_KEY"):
        add(provider="openai", model_id=openai_model)

    upstream_base_url = _env("OPENAI_BASE_URL")
    upstream_model = _env_first("ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID")
    upstream_key = _vision_catalog_api_key_for_base_url(upstream_base_url)
    if upstream_base_url and upstream_model and (not _looks_like_openai_api(upstream_base_url) or upstream_key):
        add(provider="openai" if _looks_like_openai_api(upstream_base_url) else "openai-compatible", model_id=upstream_model)

    backend_kind = _vision_backend_kind()
    abstractvision_model = _env_first("ABSTRACTVISION_MODEL_ID", "ABSTRACTCORE_VISION_MODEL_ID")
    if abstractvision_model and upstream_base_url and (backend_kind == "openai_compatible_proxy" or upstream_base_url) and (
        not _looks_like_openai_api(upstream_base_url) or upstream_key
    ):
        add(provider="openai" if _looks_like_openai_api(upstream_base_url) else "openai-compatible", model_id=abstractvision_model)

    return entries


def _cached_vision_provider_model_entries(cached_catalog: Any, task: Optional[str]) -> list[Dict[str, Any]]:
    if not isinstance(cached_catalog, dict):
        return []
    values = cached_catalog.get("models")
    if not isinstance(values, list):
        return []
    entries: list[Dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or item.get("model") or "").strip()
        if not model_id:
            continue
        provider = str(item.get("provider") or "huggingface").strip() or "huggingface"
        tasks = [str(t) for t in item.get("tasks", [])] if isinstance(item.get("tasks"), list) else ["text_to_image"]
        if task and task not in tasks:
            continue
        backend = "diffusers" if provider in {"huggingface", "hf"} else provider
        routed = f"diffusers/{model_id}" if backend == "diffusers" and not model_id.startswith("diffusers/") else model_id
        entries.append(
            {
                "id": model_id,
                "model": routed,
                "provider": provider,
                "backend": backend,
                "routed_model": routed,
                "object": "model",
                "owned_by": provider,
                "capabilities": tasks,
                "raw": {
                    **item,
                    "provider": provider,
                    "backend": backend,
                    "routed_model": routed,
                    "local_cached": True,
                },
            }
        )
    return entries


def _merge_vision_provider_model_entries(*groups: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or item.get("owned_by") or "").strip()
            model = str(item.get("model") or item.get("routed_model") or item.get("id") or "").strip()
            if not model:
                continue
            key = (provider, model)
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(item))
    return out


def _vision_providers_from_model_entries(models: list[Dict[str, Any]]) -> list[str]:
    providers: list[str] = []
    seen: set[str] = set()
    for item in models:
        provider = str(item.get("provider") or item.get("owned_by") or "").strip()
        key = provider.lower()
        if not provider or key in seen:
            continue
        seen.add(key)
        providers.append(provider)
    return providers


def _vision_models_by_provider(models: list[Dict[str, Any]]) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    seen: Dict[str, set[str]] = {}
    for item in models:
        provider = str(item.get("provider") or item.get("owned_by") or "").strip()
        model = str(item.get("model") or item.get("routed_model") or item.get("id") or "").strip()
        if not provider or not model:
            continue
        values = out.setdefault(provider, [])
        provider_seen = seen.setdefault(provider, set())
        key = model.lower()
        if key in provider_seen:
            continue
        provider_seen.add(key)
        values.append(model)
    return out


async def _vision_provider_catalog(
    request: Request,
    *,
    task: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    local_catalog: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[Any], Dict[str, Any], list[Dict[str, Any]], Optional[Exception]]:
    catalog_error: Optional[Exception] = None
    core = None
    plugin_models: list[Dict[str, Any]] = []
    availability: Dict[str, Any] = {}
    try:
        core = _vision_catalog_core(request, base_url=base_url, api_key=api_key)
        try:
            if hasattr(core.vision, "available_providers"):
                raw = core.vision.available_providers(task=task)
                if isinstance(raw, dict):
                    availability = dict(raw)
        except Exception:
            availability = {}

        plugin_models = [dict(x) for x in list(core.vision.list_provider_models(task=task) or []) if isinstance(x, dict)]
    except HTTPException:
        raise
    except Exception as e:
        catalog_error = e

    try:
        cached_catalog = dict(local_catalog) if isinstance(local_catalog, dict) else await _local_vision_model_catalog()
        cached_models = _cached_vision_provider_model_entries(cached_catalog, task)
    except Exception:
        cached_models = []
    configured_models = _configured_vision_provider_model_entries(task)
    models = _merge_vision_provider_model_entries(plugin_models, cached_models, configured_models)
    return core, availability, models, catalog_error


async def _vision_providers_payload(
    *,
    request: Request,
    task: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    include_models: bool,
) -> Dict[str, Any]:
    if task is not None:
        task = str(task).strip() or None
    if task and task not in {"text_to_image", "image_to_image", "image_upscale", "text_to_video", "image_to_video"}:
        raise HTTPException(
            status_code=400,
            detail="task must be one of: text_to_image, image_to_image, image_upscale, text_to_video, image_to_video",
        )

    if not include_models:
        try:
            core = _vision_catalog_core(request, base_url=base_url, api_key=api_key)
            availability: Dict[str, Any] = {}
            try:
                if hasattr(core.vision, "available_providers"):
                    raw = core.vision.available_providers(task=task)
                    if isinstance(raw, dict):
                        availability = dict(raw)
            except Exception:
                availability = {}
            providers = [str(p).strip() for p in availability.get("providers") or [] if isinstance(p, str) and str(p).strip()]
            available_providers = [
                str(p).strip()
                for p in availability.get("available_providers") or []
                if isinstance(p, str) and str(p).strip()
            ]
            if not providers and available_providers:
                providers = list(available_providers)
            return {
                "available": True,
                "source": "abstractvision",
                "stale": False,
                "error": None,
                "backend_id": getattr(core.vision, "backend_id", None),
                "task": task,
                "providers": providers,
                "available_providers": available_providers or providers,
                "models_by_provider": {},
                "models": [],
            }
        except HTTPException:
            raise
        except Exception as e:
            raise _vision_catalog_error(e) from e

    core, availability, models, catalog_error = await _vision_provider_catalog(
        request,
        task=task,
        base_url=base_url,
        api_key=api_key,
    )

    if models or catalog_error is None:
        providers = _vision_providers_from_model_entries(models)
        if isinstance(availability.get("providers"), list):
            providers = [str(p).strip() for p in availability.get("providers") if isinstance(p, str) and str(p).strip()] or providers
        available_providers = providers
        if isinstance(availability.get("available_providers"), list):
            available_providers = [
                str(p).strip()
                for p in availability.get("available_providers")
                if isinstance(p, str) and str(p).strip()
            ] or available_providers

        models_by_provider = _vision_models_by_provider(models)
        if available_providers:
            allowed = {p.lower() for p in available_providers}
            models_by_provider = {k: v for k, v in models_by_provider.items() if k.lower() in allowed}
        return {
            "available": True,
            "source": "abstractvision",
            "stale": False,
            "error": str(catalog_error) if catalog_error is not None and models else None,
            "backend_id": getattr(core.vision, "backend_id", None) if core is not None else None,
            "task": task,
            "providers": providers,
            "available_providers": available_providers,
            "models_by_provider": models_by_provider,
            "models": list(models or []),
        }
    raise _vision_catalog_error(catalog_error) from catalog_error


@router.get("/vision/providers/")
async def list_vision_providers(
    request: Request,
    task: Optional[str] = Query(
        default=None,
        description="Optional provider catalog task filter: text_to_image, image_to_image, image_upscale, text_to_video, or image_to_video.",
        examples=["text_to_image"],
    ),
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. openai, openai-compatible, huggingface, mflux, sdcpp).",
        examples=["mflux"],
    ),
    include_models: bool = Query(
        default=False,
        description="Include full provider model catalogs (slower).",
    ),
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible vision endpoint override for catalog discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for catalog discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """List vision providers through the AbstractVision plugin boundary."""
    payload = await _vision_providers_payload(
        request=request,
        task=task,
        base_url=base_url,
        api_key=api_key,
        include_models=bool(include_models),
    )
    return _vision_filter_provider_payload(payload, provider)


@router.get("/vision/provider_models", include_in_schema=False)
async def list_vision_provider_models(
    request: Request,
    task: Optional[str] = Query(
        default=None,
        description="Optional provider catalog task filter: text_to_image, image_to_image, image_upscale, text_to_video, or image_to_video.",
        examples=["text_to_image"],
    ),
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. openai, openai-compatible, huggingface, mflux, sdcpp).",
        examples=["mflux"],
    ),
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible vision endpoint override for catalog discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for catalog discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """List full provider model catalogs (hidden from OpenAPI schema)."""
    payload = await _vision_providers_payload(
        request=request,
        task=task,
        base_url=base_url,
        api_key=api_key,
        include_models=True,
    )
    return _vision_filter_provider_payload(payload, provider)


async def _local_vision_model_catalog() -> Dict[str, Any]:
    """List vision models from the public local cached-vision catalog helper."""
    payload = dict(_vision_catalog.get_local_vision_cache_catalog())
    payload["active"] = _active_state()
    return payload


@router.get("/vision/models")
async def list_cached_vision_models(
    request: Request,
    task: Optional[str] = Query(
        default=None,
        description="Optional model catalog task filter: text_to_image, image_to_image, image_upscale, text_to_video, or image_to_video.",
        examples=["text_to_image"],
    ),
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. openai, openai-compatible, huggingface, mflux, sdcpp).",
        examples=["huggingface"],
    ),
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible vision endpoint override for model discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for model discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """List available vision provider models, including local cached models."""
    if task is not None:
        task = str(task).strip() or None
    if task and task not in {"text_to_image", "image_to_image", "image_upscale", "text_to_video", "image_to_video"}:
        raise HTTPException(
            status_code=400,
            detail="task must be one of: text_to_image, image_to_image, image_upscale, text_to_video, image_to_video",
        )

    provider_norm = str(provider or "").strip() or None

    local_catalog = await _local_vision_model_catalog()
    core, availability, models, catalog_error = await _vision_provider_catalog(
        request,
        task=task,
        base_url=base_url,
        api_key=api_key,
        local_catalog=local_catalog,
    )
    if not models and catalog_error is not None:
        raise _vision_catalog_error(catalog_error) from catalog_error

    if provider_norm:
        provider_lc = provider_norm.lower()
        models = [
            item
            for item in (models or [])
            if str(item.get("provider") or "").strip().lower() == provider_lc
        ]

    providers = _vision_providers_from_model_entries(models)
    if isinstance(availability.get("providers"), list):
        providers = [str(p).strip() for p in availability.get("providers") if isinstance(p, str) and str(p).strip()] or providers
    available_providers = providers
    if isinstance(availability.get("available_providers"), list):
        available_providers = [
            str(p).strip()
            for p in availability.get("available_providers")
            if isinstance(p, str) and str(p).strip()
        ] or available_providers

    models_by_provider = _vision_models_by_provider(models)
    if available_providers:
        allowed = {p.lower() for p in available_providers}
        models_by_provider = {k: v for k, v in models_by_provider.items() if k.lower() in allowed}
    if provider_norm and models_by_provider:
        models_by_provider = {k: v for k, v in models_by_provider.items() if k.lower() == provider_norm.lower()}
        available_providers = list(models_by_provider) or [provider_norm]
        providers = list(available_providers)
    out = dict(local_catalog)
    if provider_norm:
        out["models"] = [
            item
            for item in (list(local_catalog.get("models") or []))
            if str(item.get("provider") or "").strip().lower() == provider_norm.lower()
        ]
    out.update(
        {
            "available": True,
            "source": "abstractvision",
            "stale": False,
            "error": str(catalog_error) if catalog_error is not None and models else None,
            "backend_id": getattr(core.vision, "backend_id", None) if core is not None else None,
            "task": task,
            "providers": providers,
            "available_providers": available_providers,
            "models_by_provider": models_by_provider,
            "models": list(models or []),
            "local_models": list(out.get("models") or []),
        }
    )
    return out


async def _images_generations_impl(
    request: Request,
    payload: ImageGenerationBody,
    *,
    path_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    OpenAI-compatible image generation endpoint: POST /v1/images/generations

    Notes:
    - Only `response_format=b64_json` is supported.
    - In `auto` mode (default), the backend is inferred per-request based on `model`.
    """
    payload = _model_payload(payload)
    if path_provider:
        payload = _payload_with_path_provider(payload, path_provider)
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
    request_model = _scoped_request_model_for_request(
        payload.get("model"),
        payload.get("provider"),
        base_url=payload.get("base_url"),
    )
    width, height, extra = _image_generation_request_parts(payload, request_model=request_model)
    provider_api_key = _provider_api_key_from_request(request)
    if _is_remote_vision_request(request_model):
        _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
    core, OptionalDependencyMissingError = _create_vision_generation_core(
        request_model,
        base_url=payload.get("base_url"),
        api_key=provider_api_key,
    )

    data_items = []
    for _ in range(n):
        try:
            output_spec = {
                "modality": "image",
                "task": "image_generation",
                "negative_prompt": str(negative_prompt) if negative_prompt is not None else None,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "extra": extra,
            }
            if payload.get("provider") is not None:
                output_spec["provider"] = payload.get("provider")
            if payload.get("model") is not None:
                output_spec["model"] = payload.get("model")
            result = core.generate(
                prompt,
                output=output_spec,
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


@router.post("/images/generations")
async def images_generations(request: Request, payload: ImageGenerationBody = Body(...)) -> Dict[str, Any]:
    """
    OpenAI-compatible image generation endpoint: POST /v1/images/generations

    The request can route remotely or locally through `model` in provider/model format,
    for example `openai-compatible/gpt-image-1` or `diffusers/Qwen/Qwen-Image-2512`.
    """
    return await _images_generations_impl(request, payload)


@provider_router.post("/{provider}/v1/images/generations")
async def provider_images_generations(
    request: Request,
    provider: str = FastAPIPath(
        ...,
        description="Image provider route prefix, e.g. `openai-compatible`, `openai`, `diffusers`, `mflux`, or `sdcpp`.",
    ),
    payload: ImageGenerationBody = Body(...),
) -> Dict[str, Any]:
    """
    Provider-scoped image generation endpoint.

    Same behavior as `/v1/images/generations`, but the provider is supplied in
    the URL so OpenAI-compatible clients can use an unprefixed model name.
    """
    return await _images_generations_impl(request, payload, path_provider=provider)


async def _videos_generations_impl(
    request: Request,
    payload: VideoGenerationBody,
    *,
    path_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """OpenAI-compatible text-to-video endpoint implementation."""
    payload = _model_payload(payload)
    if path_provider:
        payload = _payload_with_path_provider(payload, path_provider)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    response_format = str(payload.get("response_format") or "b64_json").strip().lower()
    if response_format not in {"b64_json"}:
        raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

    n = _coerce_int(payload.get("n")) or 1
    n = max(1, min(int(n), 4))

    negative_prompt = payload.get("negative_prompt")
    steps = _coerce_int(payload.get("steps"))
    guidance_scale = _coerce_float(payload.get("guidance_scale"))
    guidance_2 = _coerce_float(payload.get("guidance_2"))
    seed = _coerce_int(payload.get("seed"))
    request_model = _scoped_request_model_for_request(
        payload.get("model"),
        payload.get("provider"),
        base_url=payload.get("base_url"),
    )
    width, height, fps, num_frames, extra = _video_generation_request_parts(payload, request_model=request_model)
    provider_api_key = _provider_api_key_from_request(request)
    if _is_remote_vision_request(request_model):
        _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
    core, OptionalDependencyMissingError = _create_vision_generation_core(
        request_model,
        base_url=payload.get("base_url"),
        api_key=provider_api_key,
    )

    data_items = []
    for _ in range(n):
        try:
            output_spec = {
                "modality": "video",
                "task": "text_to_video",
                "negative_prompt": str(negative_prompt) if negative_prompt is not None else None,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "guidance_2": guidance_2,
                "seed": seed,
                "format": "mp4",
                "extra": extra,
            }
            if payload.get("provider") is not None:
                output_spec["provider"] = payload.get("provider")
            if payload.get("model") is not None:
                output_spec["model"] = payload.get("model")
            result = core.generate(prompt, output=output_spec)
            video_bytes = _first_generated_video_bytes(result)
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        data_items.append({"b64_json": base64.b64encode(video_bytes).decode("ascii")})

    return {"created": int(time.time()), "data": data_items}


@router.post("/videos/generations")
async def videos_generations(request: Request, payload: VideoGenerationBody = Body(...)) -> Dict[str, Any]:
    """OpenAI-compatible text-to-video generation endpoint."""
    return await _videos_generations_impl(request, payload)


@provider_router.post("/{provider}/v1/videos/generations")
async def provider_videos_generations(
    request: Request,
    provider: str = FastAPIPath(
        ...,
        description="Video provider route prefix, e.g. `mlx-gen`, `diffusers`, or `openai-compatible`.",
    ),
    payload: VideoGenerationBody = Body(...),
) -> Dict[str, Any]:
    """Provider-scoped text-to-video generation endpoint."""
    return await _videos_generations_impl(request, payload, path_provider=provider)


@router.post("/vision/jobs/images/generations")
async def jobs_images_generations(request: Request, payload: ImageGenerationBody = Body(...)) -> Dict[str, Any]:
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
    request_model = _scoped_request_model_for_request(
        payload.get("model"),
        payload.get("provider"),
        base_url=payload.get("base_url"),
    )
    width, height, extra = _image_generation_request_parts(payload, request_model=request_model)
    provider_api_key = _provider_api_key_from_request(request)
    if _is_remote_vision_request(request_model):
        _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))

    (
        backend,
        call_lock,
        OptionalDependencyMissingError,
        ImageGenerationRequest,
        _ImageEditRequest,
        _VideoGenerationRequest,
        _ImageToVideoRequest,
        _ImageUpscaleRequest,
    ) = _unpack_resolved_backend(
        _resolve_backend(
            request_model,
            base_url=payload.get("base_url"),
            api_key=provider_api_key,
        )
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
                run_extra = dict(extra or {})

                def _event_progress(raw_event: Any) -> None:
                    event = _progress_event_payload(raw_event)
                    _job_update_progress(job_id, step=None, total=None, event=event)

                run_extra["on_progress"] = _event_progress
                req = ImageGenerationRequest(
                    prompt=prompt,
                    negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    extra=run_extra,
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


@router.post("/vision/jobs/videos/generations")
async def jobs_videos_generations(request: Request, payload: VideoGenerationBody = Body(...)) -> Dict[str, Any]:
    """Start an async text-to-video generation job with progress polling."""
    payload = _model_payload(payload)
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    response_format = str(payload.get("response_format") or "b64_json").strip().lower()
    if response_format not in {"b64_json"}:
        raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

    negative_prompt = payload.get("negative_prompt")
    steps = _coerce_int(payload.get("steps"))
    guidance_scale = _coerce_float(payload.get("guidance_scale"))
    guidance_2 = _coerce_float(payload.get("guidance_2"))
    seed = _coerce_int(payload.get("seed"))
    request_model = _scoped_request_model_for_request(
        payload.get("model"),
        payload.get("provider"),
        base_url=payload.get("base_url"),
    )
    width, height, fps, num_frames, extra = _video_generation_request_parts(payload, request_model=request_model)
    provider_api_key = _provider_api_key_from_request(request)
    if _is_remote_vision_request(request_model):
        _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))

    (
        backend,
        call_lock,
        OptionalDependencyMissingError,
        _ImageGenerationRequest,
        _ImageEditRequest,
        VideoGenerationRequest,
        _ImageToVideoRequest,
        _ImageUpscaleRequest,
    ) = _unpack_resolved_backend(
        _resolve_backend(
            request_model,
            base_url=payload.get("base_url"),
            api_key=provider_api_key,
        )
    )
    if VideoGenerationRequest is None:
        raise HTTPException(status_code=501, detail="The selected vision backend does not expose VideoGenerationRequest.")

    total_steps = int(steps) if steps is not None else int(num_frames) if num_frames is not None else None

    now_s = time.time()
    with _JOBS_LOCK:
        _jobs_cleanup_locked(now_s=now_s)
        if _any_inflight_job_locked():
            raise HTTPException(status_code=409, detail="Another generation is already running; wait for it to finish.")

        job_id = _new_job_id()
        _JOBS[job_id] = {
            "id": job_id,
            "kind": "videos/generations",
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

            run_extra = dict(extra or {})

            def _event_progress(raw_event: Any) -> None:
                event = _progress_event_payload(raw_event)
                _job_update_progress(job_id, step=None, total=None, event=event)

            run_extra["on_progress"] = _event_progress
            req = VideoGenerationRequest(
                prompt=prompt,
                negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                width=width,
                height=height,
                fps=fps,
                num_frames=num_frames,
                steps=steps,
                guidance_scale=guidance_scale,
                guidance_2=guidance_2,
                seed=seed,
                extra=run_extra,
            )

            def _progress(step_i: int, total_i: Optional[int] = None) -> None:
                s = max(0, int(step_i))
                t = _coerce_int(total_i) or total_steps
                _job_update_progress(job_id, step=s, total=t)

            with call_lock:
                fn = getattr(backend, "generate_video_with_progress", None)
                if callable(fn):
                    asset = fn(req, progress_callback=_progress)
                else:
                    asset = backend.generate_video(req)
            b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
            _job_finish(job_id, ok=True, result={"created": int(time.time()), "data": [{"b64_json": b64}]})
        except OptionalDependencyMissingError as e:
            _job_finish(job_id, ok=False, error=str(e))
        except Exception as e:
            _job_finish(job_id, ok=False, error=str(e))

    threading.Thread(target=_runner, name=f"vision-job-{job_id}", daemon=True).start()
    return {"job_id": job_id}


@router.get("/vision/jobs/{job_id}")
async def get_job(
    job_id: str = FastAPIPath(..., description="Vision job id returned by `/v1/vision/jobs/images/*` or `/v1/vision/jobs/videos/*`."),
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
        request: Request,
        prompt: str = Form(..., description="Text prompt describing the desired image edit.", examples=["Make the mug blue and keep the white background."]),
        image: UploadFile = File(..., description="Source image to edit. Upload a PNG, JPEG, or WebP file supported by the selected backend."),
        mask: Optional[UploadFile] = File(None, description="Optional mask image for inpainting/edit backends that support it. Transparent pixels mark editable areas for OpenAI-style masks."),
        reference_images: Optional[list[UploadFile]] = File(None, description="Optional additional reference images for backends that support multi-image composition or style/layout references."),
        provider: Optional[str] = Form(
            None,
            description="Optional image provider/backend hint, e.g. `openai-compatible`, `openai`, `diffusers`, `mflux`, or `sdcpp`.",
            examples=["openai-compatible"],
        ),
        model: Optional[str] = Form(
            None,
            description=(
                "Optional provider/model image id. Omit to use the server's configured AbstractVision default; "
                "use `diffusers/default`, `diffusers/<huggingface-repo>`, `sdcpp/default`, or "
                "`openai-compatible/<model>`."
            ),
            examples=["openai-compatible/gpt-image-1"],
        ),
        base_url: Optional[str] = Form(
            None,
            description=(
                "Optional request-level base URL override for OpenAI/OpenAI-compatible image backends. "
                "Loopback URLs are allowed by default; non-loopback URLs require "
                "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
            ),
            examples=["http://127.0.0.1:5000/v1"],
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
        reference_image_bytes: list[bytes] = []
        for ref in reference_images or []:
            content = await ref.read()
            if content:
                reference_image_bytes.append(bytes(content))

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
        if reference_image_bytes:
            extra.setdefault("reference_images", reference_image_bytes)
        request_model = _scoped_request_model_for_request(model, provider, base_url=base_url)
        provider_api_key = _provider_api_key_from_request(request)
        if _is_remote_vision_request(request_model):
            _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
        (
            backend,
            call_lock,
            OptionalDependencyMissingError,
            _ImageGenerationRequest,
            ImageEditRequest,
            _VideoGenerationRequest,
            _ImageToVideoRequest,
            _ImageUpscaleRequest,
        ) = _unpack_resolved_backend(
            _resolve_backend(
                request_model,
                base_url=base_url,
                api_key=provider_api_key,
            )
        )
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

                run_extra = dict(extra or {})

                def _event_progress(raw_event: Any) -> None:
                    event = _progress_event_payload(raw_event)
                    _job_update_progress(job_id, step=None, total=None, event=event)

                run_extra["on_progress"] = _event_progress
                req = ImageEditRequest(
                    prompt=prompt_s,
                    image=bytes(image_bytes),
                    mask=bytes(mask_bytes) if mask_bytes else None,
                    negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                    seed=seed_i,
                    steps=steps_i,
                    guidance_scale=guidance_f,
                    extra=run_extra,
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
        request: Request,
        prompt: str = Form(..., description="Text prompt describing the desired image edit.", examples=["Make the mug blue and keep the white background."]),
        image: UploadFile = File(..., description="Source image to edit. Upload a PNG, JPEG, or WebP file supported by the selected backend."),
        mask: Optional[UploadFile] = File(None, description="Optional mask image for inpainting/edit backends that support it. Transparent pixels mark editable areas for OpenAI-style masks."),
        reference_images: Optional[list[UploadFile]] = File(None, description="Optional additional reference images for backends that support multi-image composition or style/layout references."),
        provider: Optional[str] = Form(
            None,
            description="Optional image provider/backend hint, e.g. `openai-compatible`, `openai`, `diffusers`, `mflux`, or `sdcpp`.",
            examples=["openai-compatible"],
        ),
        model: Optional[str] = Form(
            None,
            description=(
                "Optional provider/model image id. Omit to use the server's configured AbstractVision default; "
                "use `diffusers/default`, `diffusers/<huggingface-repo>`, `sdcpp/default`, or "
                "`openai-compatible/<model>`."
            ),
            examples=["openai-compatible/gpt-image-1"],
        ),
        base_url: Optional[str] = Form(
            None,
            description=(
                "Optional request-level base URL override for OpenAI/OpenAI-compatible image backends. "
                "Loopback URLs are allowed by default; non-loopback URLs require "
                "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
            ),
            examples=["http://127.0.0.1:5000/v1"],
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
        reference_image_bytes: list[bytes] = []
        for ref in reference_images or []:
            content = await ref.read()
            if content:
                reference_image_bytes.append(bytes(content))

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
        if reference_image_bytes:
            extra.setdefault("reference_images", reference_image_bytes)
        request_model = _scoped_request_model_for_request(model, provider, base_url=base_url)
        provider_api_key = _provider_api_key_from_request(request)
        if _is_remote_vision_request(request_model):
            _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
        core, OptionalDependencyMissingError = _create_vision_generation_core(
            request_model,
            base_url=base_url,
            api_key=provider_api_key,
        )

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
                    "provider": provider,
                    "model": model,
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

    @provider_router.post("/{provider}/v1/images/edits")
    async def provider_images_edits(
        request: Request,
        provider: str = FastAPIPath(
            ...,
            description="Image provider route prefix, e.g. `openai-compatible`, `openai`, `diffusers`, `mflux`, or `sdcpp`.",
        ),
        prompt: str = Form(..., description="Text prompt describing the desired image edit.", examples=["Make the mug blue and keep the white background."]),
        image: UploadFile = File(..., description="Source image to edit. Upload a PNG, JPEG, or WebP file supported by the selected backend."),
        mask: Optional[UploadFile] = File(None, description="Optional mask image for inpainting/edit backends that support it. Transparent pixels mark editable areas for OpenAI-style masks."),
        reference_images: Optional[list[UploadFile]] = File(None, description="Optional additional reference images for backends that support multi-image composition or style/layout references."),
        model: Optional[str] = Form(
            None,
            description=(
                "Optional unprefixed model id for the provider route. If a provider/model id is supplied, "
                "the route provider takes precedence."
            ),
            examples=["gpt-image-1"],
        ),
        base_url: Optional[str] = Form(
            None,
            description=(
                "Optional request-level base URL override for OpenAI/OpenAI-compatible image backends. "
                "Loopback URLs are allowed by default; non-loopback URLs require "
                "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
            ),
            examples=["http://127.0.0.1:5000/v1"],
        ),
        size: Optional[str] = Form(None, description="OpenAI-style output image size such as `1024x1024`, `1536x1024`, `1024x1536`, or `auto` when supported.", examples=["1024x1024"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported by the server response.", examples=["b64_json"]),
        negative_prompt: Optional[str] = Form(None, description="Local/backend-specific negative prompt. Strict OpenAI-compatible upstreams do not receive this top-level field unless supplied through `extra_json`.", examples=["blur, low quality"]),
        seed: Optional[str] = Form(None, description="Local/backend-specific deterministic seed. Use `extra_json` for custom OpenAI-compatible upstreams that support a seed field.", examples=["1234"]),
        steps: Optional[str] = Form(None, description="Local/backend-specific denoising/inference step count. Use `extra_json` for custom OpenAI-compatible upstreams that support a steps field.", examples=["20"]),
        guidance_scale: Optional[str] = Form(None, description="Local/backend-specific classifier-free guidance scale. Use `extra_json` for custom OpenAI-compatible upstreams that support this field.", examples=["7.5"]),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific generation parameters.", examples=['{"quality":"low","background":"auto","output_format":"png"}']),
    ) -> Dict[str, Any]:
        return await images_edits(
            request=request,
            prompt=prompt,
            image=image,
            mask=mask,
            reference_images=reference_images,
            provider=provider,
            model=model,
            base_url=base_url,
            size=size,
            response_format=response_format,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            extra_json=extra_json,
        )

    @router.post("/images/upscale")
    async def images_upscale(
        request: Request,
        image: UploadFile = File(..., description="Source image to upscale."),
        provider: Optional[str] = Form(None, description="Optional image provider/backend hint.", examples=["mlx-gen"]),
        model: Optional[str] = Form(None, description="Optional provider/model image id.", examples=["mlx-gen/AbstractFramework/seedvr2-3b-8bit"]),
        base_url: Optional[str] = Form(None, description="Optional request-level base URL override for compatible backends."),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported.", examples=["b64_json"]),
        scale: Optional[str] = Form(None, description="Friendly scale factor, for example `2`, `2x`, or `1.5x`.", examples=["2x"]),
        resolution: Optional[str] = Form(None, description="SeedVR2 target shortest edge in pixels or scale factor such as `2x`.", examples=["2x"]),
        softness: Optional[str] = Form(None, description="SeedVR2 softness in [0.0, 1.0].", examples=["0.0"]),
        seed: Optional[str] = Form(None, description="Deterministic seed.", examples=["1234"]),
        quantize: Optional[str] = Form(None, description="Runtime quantization for official/source SeedVR2 loading: 3, 4, 5, 6, or 8. Canonical prepared q8/q4 packages do not require this field.", examples=["8"]),
        vae_tiling: Optional[str] = Form(None, description="Enable tiled VAE encode/decode for very large outputs.", examples=["false"]),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific parameters."),
    ) -> Dict[str, Any]:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Missing required image bytes")
        response_format_s = str(response_format or "b64_json").strip().lower()
        if response_format_s not in {"b64_json"}:
            raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")
        extra = _parse_extra_json_form(extra_json)
        upscale_fields = _parse_upscale_form_fields(
            scale=scale,
            resolution=resolution,
            softness=softness,
            seed=seed,
            quantize=quantize,
            vae_tiling=vae_tiling,
        )
        request_model = _scoped_request_model_for_request(model, provider, base_url=base_url)
        provider_api_key = _provider_api_key_from_request(request)
        if _is_remote_vision_request(request_model):
            _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
        core, OptionalDependencyMissingError = _create_vision_generation_core(
            request_model,
            base_url=base_url,
            api_key=provider_api_key,
        )
        try:
            result = core.generate(
                "",
                media=[
                    {
                        "type": "image",
                        "content": bytes(image_bytes),
                        "role": "source",
                    }
                ],
                output={
                    "modality": "image",
                    "task": "image_upscale",
                    "provider": provider,
                    "model": model,
                    "scale": upscale_fields["scale"],
                    "resolution": upscale_fields["resolution"],
                    "softness": upscale_fields["softness"],
                    "seed": upscale_fields["seed"],
                    "quantize": upscale_fields["quantize"],
                    "vae_tiling": upscale_fields["vae_tiling"],
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

    @provider_router.post("/{provider}/v1/images/upscale")
    async def provider_images_upscale(
        request: Request,
        provider: str = FastAPIPath(..., description="Image provider route prefix, e.g. `mlx-gen`."),
        image: UploadFile = File(..., description="Source image to upscale."),
        model: Optional[str] = Form(None, description="Optional unprefixed model id for the provider route.", examples=["AbstractFramework/seedvr2-3b-8bit"]),
        base_url: Optional[str] = Form(None, description="Optional request-level base URL override for compatible backends."),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported.", examples=["b64_json"]),
        scale: Optional[str] = Form(None, description="Friendly scale factor, for example `2`, `2x`, or `1.5x`.", examples=["2x"]),
        resolution: Optional[str] = Form(None, description="SeedVR2 target shortest edge in pixels or scale factor such as `2x`.", examples=["2x"]),
        softness: Optional[str] = Form(None, description="SeedVR2 softness in [0.0, 1.0].", examples=["0.0"]),
        seed: Optional[str] = Form(None, description="Deterministic seed.", examples=["1234"]),
        quantize: Optional[str] = Form(None, description="Runtime quantization for official/source SeedVR2 loading: 3, 4, 5, 6, or 8. Canonical prepared q8/q4 packages do not require this field.", examples=["8"]),
        vae_tiling: Optional[str] = Form(None, description="Enable tiled VAE encode/decode for very large outputs.", examples=["false"]),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific parameters."),
    ) -> Dict[str, Any]:
        return await images_upscale(
            request=request,
            image=image,
            provider=provider,
            model=model,
            base_url=base_url,
            response_format=response_format,
            scale=scale,
            resolution=resolution,
            softness=softness,
            seed=seed,
            quantize=quantize,
            vae_tiling=vae_tiling,
            extra_json=extra_json,
        )

    @router.post("/vision/jobs/images/upscale")
    async def jobs_images_upscale(
        request: Request,
        image: UploadFile = File(..., description="Source image to upscale."),
        provider: Optional[str] = Form(None, description="Optional image provider/backend hint.", examples=["mlx-gen"]),
        model: Optional[str] = Form(None, description="Optional provider/model image id.", examples=["mlx-gen/AbstractFramework/seedvr2-3b-8bit"]),
        base_url: Optional[str] = Form(None, description="Optional request-level base URL override for compatible backends."),
        scale: Optional[str] = Form(None, description="Friendly scale factor, for example `2`, `2x`, or `1.5x`.", examples=["2x"]),
        resolution: Optional[str] = Form(None, description="SeedVR2 target shortest edge in pixels or scale factor such as `2x`.", examples=["2x"]),
        softness: Optional[str] = Form(None, description="SeedVR2 softness in [0.0, 1.0].", examples=["0.0"]),
        seed: Optional[str] = Form(None, description="Deterministic seed.", examples=["1234"]),
        quantize: Optional[str] = Form(None, description="Runtime quantization for official/source SeedVR2 loading: 3, 4, 5, 6, or 8. Canonical prepared q8/q4 packages do not require this field.", examples=["8"]),
        vae_tiling: Optional[str] = Form(None, description="Enable tiled VAE encode/decode for very large outputs.", examples=["false"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported.", examples=["b64_json"]),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific parameters."),
    ) -> Dict[str, Any]:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Missing required image bytes")
        response_format_s = str(response_format or "b64_json").strip().lower()
        if response_format_s not in {"b64_json"}:
            raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")
        extra = _parse_extra_json_form(extra_json)
        upscale_fields = _parse_upscale_form_fields(
            scale=scale,
            resolution=resolution,
            softness=softness,
            seed=seed,
            quantize=quantize,
            vae_tiling=vae_tiling,
        )
        request_model = _scoped_request_model_for_request(model, provider, base_url=base_url)
        provider_api_key = _provider_api_key_from_request(request)
        if _is_remote_vision_request(request_model):
            _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
        (
            backend,
            call_lock,
            OptionalDependencyMissingError,
            _ImageGenerationRequest,
            _ImageEditRequest,
            _VideoGenerationRequest,
            _ImageToVideoRequest,
            ImageUpscaleRequest,
        ) = _unpack_resolved_backend(
            _resolve_backend(
                request_model,
                base_url=base_url,
                api_key=provider_api_key,
            )
        )
        if ImageUpscaleRequest is None:
            raise HTTPException(status_code=501, detail="The selected vision backend does not expose ImageUpscaleRequest.")

        now_s = time.time()
        with _JOBS_LOCK:
            _jobs_cleanup_locked(now_s=now_s)
            if _any_inflight_job_locked():
                raise HTTPException(status_code=409, detail="Another generation is already running; wait for it to finish.")
            job_id = _new_job_id()
            _JOBS[job_id] = {
                "id": job_id,
                "kind": "images/upscale",
                "state": "queued",
                "created_at_s": now_s,
                "updated_at_s": now_s,
                "progress": {"step": 0, "total_steps": None},
            }

        def _runner() -> None:
            try:
                _job_update_progress(job_id, step=0, total=None, message="running")
                with _JOBS_LOCK:
                    job = _JOBS.get(job_id)
                    if job is not None:
                        job["state"] = "running"
                        job["updated_at_s"] = time.time()

                run_extra = dict(extra or {})

                def _event_progress(raw_event: Any) -> None:
                    event = _progress_event_payload(raw_event)
                    _job_update_progress(job_id, step=None, total=None, event=event)

                run_extra["on_progress"] = _event_progress
                req = ImageUpscaleRequest(
                    image=bytes(image_bytes),
                    resolution=upscale_fields["resolution"],
                    scale=upscale_fields["scale"],
                    seed=upscale_fields["seed"],
                    softness=upscale_fields["softness"],
                    quantize=upscale_fields["quantize"],
                    vae_tiling=upscale_fields["vae_tiling"],
                    extra=run_extra,
                )

                def _progress(step_i: int, total_i: Optional[int] = None) -> None:
                    _job_update_progress(job_id, step=max(0, int(step_i)), total=total_i)

                with call_lock:
                    fn = getattr(backend, "upscale_image_with_progress", None)
                    if callable(fn):
                        asset = fn(req, progress_callback=_progress)
                    else:
                        asset = backend.upscale_image(req)
                b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
                _job_finish(job_id, ok=True, result={"created": int(time.time()), "data": [{"b64_json": b64}]})
            except OptionalDependencyMissingError as e:
                _job_finish(job_id, ok=False, error=str(e))
            except Exception as e:
                _job_finish(job_id, ok=False, error=str(e))

        threading.Thread(target=_runner, name=f"vision-job-{job_id}", daemon=True).start()
        return {"job_id": job_id}

    @router.post("/vision/jobs/videos/edits")
    @router.post("/vision/jobs/videos/from-image", include_in_schema=False)
    async def jobs_videos_edits(
        request: Request,
        prompt: str = Form("", description="Text prompt describing the desired image-to-video motion."),
        image: UploadFile = File(..., description="Source image / first frame to animate."),
        provider: Optional[str] = Form(None, description="Optional video provider/backend hint.", examples=["mlx-gen"]),
        model: Optional[str] = Form(None, description="Optional provider/model video id.", examples=["mlx-gen/AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"]),
        base_url: Optional[str] = Form(None, description="Optional request-level base URL override for OpenAI-compatible video backends."),
        width: Optional[str] = Form(None, description="Requested video width in pixels.", examples=["1280"]),
        height: Optional[str] = Form(None, description="Requested video height in pixels.", examples=["704"]),
        size: Optional[str] = Form(None, description="Optional WIDTHxHEIGHT size selector.", examples=["1280x704"]),
        fps: Optional[str] = Form(None, description="Requested frames per second.", examples=["24"]),
        num_frames: Optional[str] = Form(None, description="Requested frame count.", examples=["121"]),
        frames: Optional[str] = Form(None, description="Alias for num_frames.", examples=["121"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported."),
        negative_prompt: Optional[str] = Form(None, description="Optional negative prompt."),
        seed: Optional[str] = Form(None, description="Optional deterministic seed."),
        steps: Optional[str] = Form(None, description="Optional inference step count."),
        guidance_scale: Optional[str] = Form(None, description="Optional guidance scale."),
        guidance_2: Optional[str] = Form(None, description="Optional second-stage low-noise guidance scale for dual-transformer video models."),
        max_sequence_length: Optional[str] = Form(None, description="Optional backend-specific prompt token length cap."),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific generation parameters."),
    ) -> Dict[str, Any]:
        """Start an async image-to-video generation job with progress polling."""
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Missing required image bytes")

        response_format_s = str(response_format or "b64_json").strip().lower()
        if response_format_s not in {"b64_json"}:
            raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

        payload = {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "width": _coerce_int(width),
            "height": _coerce_int(height),
            "size": size,
            "fps": _coerce_int(fps),
            "num_frames": _coerce_int(num_frames) or _coerce_int(frames),
            "negative_prompt": negative_prompt,
            "seed": _coerce_int(seed),
            "steps": _coerce_int(steps),
            "guidance_scale": _coerce_float(guidance_scale),
            "guidance_2": _coerce_float(guidance_2),
            "max_sequence_length": _coerce_int(max_sequence_length),
            "extra": _parse_extra_json_form(extra_json),
        }
        request_model = _scoped_request_model_for_request(model, provider, base_url=base_url)
        width_i, height_i, fps_i, num_frames_i, extra = _video_generation_request_parts(payload, request_model=request_model)
        provider_api_key = _provider_api_key_from_request(request)
        if _is_remote_vision_request(request_model):
            _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
        (
            backend,
            call_lock,
            OptionalDependencyMissingError,
            _ImageGenerationRequest,
            _ImageEditRequest,
            _VideoGenerationRequest,
            ImageToVideoRequest,
            _ImageUpscaleRequest,
        ) = _unpack_resolved_backend(
            _resolve_backend(
                request_model,
                base_url=base_url,
                api_key=provider_api_key,
            )
        )
        if ImageToVideoRequest is None:
            raise HTTPException(status_code=501, detail="The selected vision backend does not expose ImageToVideoRequest.")

        total_steps = _coerce_int(steps) or num_frames_i
        now_s = time.time()
        with _JOBS_LOCK:
            _jobs_cleanup_locked(now_s=now_s)
            if _any_inflight_job_locked():
                raise HTTPException(status_code=409, detail="Another generation is already running; wait for it to finish.")

            job_id = _new_job_id()
            _JOBS[job_id] = {
                "id": job_id,
                "kind": "videos/edits",
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

                run_extra = dict(extra or {})

                def _event_progress(raw_event: Any) -> None:
                    event = _progress_event_payload(raw_event)
                    _job_update_progress(job_id, step=None, total=None, event=event)

                run_extra["on_progress"] = _event_progress
                req = ImageToVideoRequest(
                    image=bytes(image_bytes),
                    prompt=str(prompt or ""),
                    negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                    width=width_i,
                    height=height_i,
                    fps=fps_i,
                    num_frames=num_frames_i,
                    seed=_coerce_int(seed),
                    steps=_coerce_int(steps),
                    guidance_scale=_coerce_float(guidance_scale),
                    guidance_2=_coerce_float(guidance_2),
                    extra=run_extra,
                )

                def _progress(step_i: int, total_i: Optional[int] = None) -> None:
                    s = max(0, int(step_i))
                    t = _coerce_int(total_i) or total_steps
                    _job_update_progress(job_id, step=s, total=t)

                with call_lock:
                    fn = getattr(backend, "image_to_video_with_progress", None)
                    if callable(fn):
                        asset = fn(req, progress_callback=_progress)
                    else:
                        asset = backend.image_to_video(req)
                b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
                _job_finish(job_id, ok=True, result={"created": int(time.time()), "data": [{"b64_json": b64}]})
            except OptionalDependencyMissingError as e:
                _job_finish(job_id, ok=False, error=str(e))
            except Exception as e:
                _job_finish(job_id, ok=False, error=str(e))

        threading.Thread(target=_runner, name=f"vision-job-{job_id}", daemon=True).start()
        return {"job_id": job_id}

    async def _videos_edits_impl(
        request: Request,
        *,
        prompt: str,
        image: UploadFile,
        provider: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
        width: Optional[str],
        height: Optional[str],
        size: Optional[str],
        fps: Optional[str],
        num_frames: Optional[str],
        frames: Optional[str],
        response_format: Optional[str],
        negative_prompt: Optional[str],
        seed: Optional[str],
        steps: Optional[str],
        guidance_scale: Optional[str],
        guidance_2: Optional[str],
        max_sequence_length: Optional[str],
        extra_json: Optional[str],
    ) -> Dict[str, Any]:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Missing required image bytes")

        response_format_s = str(response_format or "b64_json").strip().lower()
        if response_format_s not in {"b64_json"}:
            raise HTTPException(status_code=400, detail="Only response_format='b64_json' is supported.")

        payload = {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "width": _coerce_int(width),
            "height": _coerce_int(height),
            "size": size,
            "fps": _coerce_int(fps),
            "num_frames": _coerce_int(num_frames) or _coerce_int(frames),
            "negative_prompt": negative_prompt,
            "seed": _coerce_int(seed),
            "steps": _coerce_int(steps),
            "guidance_scale": _coerce_float(guidance_scale),
            "guidance_2": _coerce_float(guidance_2),
            "max_sequence_length": _coerce_int(max_sequence_length),
            "extra": _parse_extra_json_form(extra_json),
        }
        request_model = _scoped_request_model_for_request(model, provider, base_url=base_url)
        width_i, height_i, fps_i, num_frames_i, extra = _video_generation_request_parts(payload, request_model=request_model)
        provider_api_key = _provider_api_key_from_request(request)
        if _is_remote_vision_request(request_model):
            _guard_vision_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
        core, OptionalDependencyMissingError = _create_vision_generation_core(
            request_model,
            base_url=base_url,
            api_key=provider_api_key,
        )

        try:
            result = core.generate(
                str(prompt or ""),
                media=[{"type": "image", "content": bytes(image_bytes), "role": "source"}],
                output={
                    "modality": "video",
                    "task": "image_to_video",
                    "provider": provider,
                    "model": model,
                    "negative_prompt": str(negative_prompt) if negative_prompt is not None else None,
                    "width": width_i,
                    "height": height_i,
                    "fps": fps_i,
                    "num_frames": num_frames_i,
                    "seed": _coerce_int(seed),
                    "steps": _coerce_int(steps),
                    "guidance_scale": _coerce_float(guidance_scale),
                    "guidance_2": _coerce_float(guidance_2),
                    "format": "mp4",
                    "extra": extra,
                },
            )
            video_bytes_out = _first_generated_video_bytes(result)
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(video_bytes_out).decode("ascii")}]}

    @router.post("/videos/edits")
    @router.post("/videos/from-image", include_in_schema=False)
    async def videos_edits(
        request: Request,
        prompt: str = Form("", description="Text prompt describing the desired image-to-video motion."),
        image: UploadFile = File(..., description="Source image / first frame to animate."),
        provider: Optional[str] = Form(None, description="Optional video provider/backend hint.", examples=["mlx-gen"]),
        model: Optional[str] = Form(None, description="Optional provider/model video id.", examples=["mlx-gen/AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"]),
        base_url: Optional[str] = Form(None, description="Optional request-level base URL override for OpenAI-compatible video backends."),
        width: Optional[str] = Form(None, description="Requested video width in pixels.", examples=["1280"]),
        height: Optional[str] = Form(None, description="Requested video height in pixels.", examples=["704"]),
        size: Optional[str] = Form(None, description="Optional WIDTHxHEIGHT size selector.", examples=["1280x704"]),
        fps: Optional[str] = Form(None, description="Requested frames per second.", examples=["24"]),
        num_frames: Optional[str] = Form(None, description="Requested frame count.", examples=["121"]),
        frames: Optional[str] = Form(None, description="Alias for num_frames.", examples=["121"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported."),
        negative_prompt: Optional[str] = Form(None, description="Optional negative prompt."),
        seed: Optional[str] = Form(None, description="Optional deterministic seed."),
        steps: Optional[str] = Form(None, description="Optional inference step count."),
        guidance_scale: Optional[str] = Form(None, description="Optional guidance scale."),
        guidance_2: Optional[str] = Form(None, description="Optional second-stage low-noise guidance scale for dual-transformer video models."),
        max_sequence_length: Optional[str] = Form(None, description="Optional backend-specific prompt token length cap."),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific generation parameters."),
    ) -> Dict[str, Any]:
        """OpenAI-compatible image-to-video endpoint."""
        return await _videos_edits_impl(
            request,
            prompt=prompt,
            image=image,
            provider=provider,
            model=model,
            base_url=base_url,
            width=width,
            height=height,
            size=size,
            fps=fps,
            num_frames=num_frames,
            frames=frames,
            response_format=response_format,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            guidance_2=guidance_2,
            max_sequence_length=max_sequence_length,
            extra_json=extra_json,
        )

    @provider_router.post("/{provider}/v1/videos/edits")
    @provider_router.post("/{provider}/v1/videos/from-image", include_in_schema=False)
    async def provider_videos_edits(
        request: Request,
        provider: str = FastAPIPath(..., description="Video provider route prefix."),
        prompt: str = Form("", description="Text prompt describing the desired image-to-video motion."),
        image: UploadFile = File(..., description="Source image / first frame to animate."),
        model: Optional[str] = Form(None, description="Optional unprefixed model id for the provider route."),
        base_url: Optional[str] = Form(None, description="Optional request-level base URL override for OpenAI-compatible video backends."),
        width: Optional[str] = Form(None, description="Requested video width in pixels.", examples=["1280"]),
        height: Optional[str] = Form(None, description="Requested video height in pixels.", examples=["704"]),
        size: Optional[str] = Form(None, description="Optional WIDTHxHEIGHT size selector.", examples=["1280x704"]),
        fps: Optional[str] = Form(None, description="Requested frames per second.", examples=["24"]),
        num_frames: Optional[str] = Form(None, description="Requested frame count.", examples=["121"]),
        frames: Optional[str] = Form(None, description="Alias for num_frames.", examples=["121"]),
        response_format: Optional[str] = Form("b64_json", description="Response format. Only `b64_json` is currently supported."),
        negative_prompt: Optional[str] = Form(None, description="Optional negative prompt."),
        seed: Optional[str] = Form(None, description="Optional deterministic seed."),
        steps: Optional[str] = Form(None, description="Optional inference step count."),
        guidance_scale: Optional[str] = Form(None, description="Optional guidance scale."),
        guidance_2: Optional[str] = Form(None, description="Optional second-stage low-noise guidance scale for dual-transformer video models."),
        max_sequence_length: Optional[str] = Form(None, description="Optional backend-specific prompt token length cap."),
        extra_json: Optional[str] = Form(None, description="Optional JSON object string with backend-specific generation parameters."),
    ) -> Dict[str, Any]:
        return await _videos_edits_impl(
            request,
            prompt=prompt,
            image=image,
            provider=provider,
            model=model,
            base_url=base_url,
            width=width,
            height=height,
            size=size,
            fps=fps,
            num_frames=num_frames,
            frames=frames,
            response_format=response_format,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            guidance_2=guidance_2,
            max_sequence_length=max_sequence_length,
            extra_json=extra_json,
        )

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

    @provider_router.post("/{provider}/v1/images/edits")
    async def provider_images_edits(
        provider: str = FastAPIPath(..., description="Image provider route prefix."),
    ) -> Dict[str, Any]:
        _ = provider
        raise HTTPException(
            status_code=501,
            detail=(
                "The /{provider}/v1/images/edits endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )

    @router.post("/images/upscale")
    async def images_upscale() -> Dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail=(
                "The /v1/images/upscale endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )

    @provider_router.post("/{provider}/v1/images/upscale")
    async def provider_images_upscale(
        provider: str = FastAPIPath(..., description="Image provider route prefix."),
    ) -> Dict[str, Any]:
        _ = provider
        raise HTTPException(
            status_code=501,
            detail=(
                "The /{provider}/v1/images/upscale endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )

    @router.post("/vision/jobs/images/upscale")
    async def jobs_images_upscale() -> Dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail=(
                "The /v1/vision/jobs/images/upscale endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )

    @router.post("/vision/jobs/videos/edits")
    @router.post("/vision/jobs/videos/from-image", include_in_schema=False)
    async def jobs_videos_edits() -> Dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail=(
                "The /v1/vision/jobs/videos/edits endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )

    @router.post("/videos/edits")
    @router.post("/videos/from-image", include_in_schema=False)
    async def videos_edits() -> Dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail=(
                "The /v1/videos/edits endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )

    @provider_router.post("/{provider}/v1/videos/edits")
    @provider_router.post("/{provider}/v1/videos/from-image", include_in_schema=False)
    async def provider_videos_edits(
        provider: str = FastAPIPath(..., description="Video provider route prefix."),
    ) -> Dict[str, Any]:
        _ = provider
        raise HTTPException(
            status_code=501,
            detail=(
                "The /{provider}/v1/videos/edits endpoint requires python-multipart for multipart/form-data parsing. "
                "Install it via: pip install \"abstractcore[server]\" (or: pip install python-multipart)."
            ),
        )
