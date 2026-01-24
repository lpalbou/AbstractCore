"""
OpenAI-compatible vision generation endpoints for AbstractCore Server.

This module is intentionally dependency-light and safe-by-default:
- It does not import `abstractvision` unless the endpoints are actually used.
- It requires explicit configuration via environment variables.

Design notes:
- AbstractCore Server is a gateway; vision generation is delegated to AbstractVision backends.
- Today AbstractVision ships an OpenAI-compatible HTTP backend; this router can act as a thin
  "vision proxy" to any upstream that implements `/images/generations` and `/images/edits`.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile


router = APIRouter(tags=["vision"])


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


def _vision_backend_kind() -> str:
    raw = _env("ABSTRACTCORE_VISION_BACKEND")
    if not raw:
        return "openai_compatible_proxy"
    v = str(raw).strip().lower()
    if v in {"openai", "openai-compatible", "openai_compatible", "proxy", "openai_compatible_proxy"}:
        return "openai_compatible_proxy"
    if v in {"diffusers", "hf-diffusers", "huggingface-diffusers"}:
        return "diffusers"
    if v in {"sdcpp", "sd-cpp", "stable-diffusion.cpp", "stable-diffusion-cpp", "stable_diffusion_cpp"}:
        return "sdcpp"
    return v


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
    model = str(_env("ABSTRACTCORE_VISION_SDCPP_MODEL") or "").strip()
    if model:
        return model, None
    diffusion_model = str(request_model or _env("ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL") or "").strip()
    if not diffusion_model:
        raise HTTPException(
            status_code=501,
            detail=(
                "Vision image endpoints are not configured for sdcpp mode. "
                "Set ABSTRACTCORE_VISION_SDCPP_MODEL (full model) or "
                "ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL (component mode), and ensure `sd-cli` is installed."
            ),
        )
    return None, diffusion_model


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
        raise HTTPException(
            status_code=501,
            detail="AbstractVision is required for vision generation endpoints. Install it via: pip install abstractvision",
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


@router.post("/images/generations")
async def images_generations(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    OpenAI-compatible image generation endpoint: POST /v1/images/generations

    This endpoint is implemented as a thin proxy over AbstractVision's OpenAI-compatible backend.
    It requires:
    - ABSTRACTCORE_VISION_UPSTREAM_BASE_URL
    - (optional) ABSTRACTCORE_VISION_UPSTREAM_API_KEY

    Notes:
    - Only `response_format=b64_json` is supported.
    - `model` is passed through as the upstream model id (or env default if omitted).
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

    backend_kind = _vision_backend_kind()
    if backend_kind == "openai_compatible_proxy":
        base_url = _require_upstream_base_url()
        model_id = str(payload.get("model") or _env("ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID") or "").strip() or None
        (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
            _HuggingFaceDiffusersBackendConfig,
            _HuggingFaceDiffusersVisionBackend,
            OptionalDependencyMissingError,
            req_types,
        ) = _import_abstractvision()
        ImageGenerationRequest, _ImageEditRequest = req_types
        cfg = OpenAICompatibleBackendConfig(
            base_url=base_url,
            api_key=_env("ABSTRACTCORE_VISION_UPSTREAM_API_KEY"),
            model_id=model_id,
            timeout_s=float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "300") or "300"),
            image_generations_path=_env("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH", "/images/generations")
            or "/images/generations",
            image_edits_path=_env("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_EDITS_PATH", "/images/edits") or "/images/edits",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)
    elif backend_kind == "diffusers":
        model_id = _require_diffusers_model_id(payload.get("model"))
        (
            _OpenAICompatibleBackendConfig,
            _OpenAICompatibleVisionBackend,
            HuggingFaceDiffusersBackendConfig,
            HuggingFaceDiffusersVisionBackend,
            _StableDiffusionCppBackendConfig,
            _StableDiffusionCppVisionBackend,
            OptionalDependencyMissingError,
            req_types,
        ) = _import_abstractvision()
        ImageGenerationRequest, _ImageEditRequest = req_types
        cfg = HuggingFaceDiffusersBackendConfig(
            model_id=model_id,
            device=_env("ABSTRACTCORE_VISION_DEVICE", "cpu") or "cpu",
            torch_dtype=_env("ABSTRACTCORE_VISION_TORCH_DTYPE"),
            allow_download=_env_bool("ABSTRACTCORE_VISION_ALLOW_DOWNLOAD", False),
        )
        backend = HuggingFaceDiffusersVisionBackend(config=cfg)
    elif backend_kind == "sdcpp":
        model_path, diffusion_model_path = _require_sdcpp_model_or_diffusion_model(payload.get("model"))
        (
            _OpenAICompatibleBackendConfig,
            _OpenAICompatibleVisionBackend,
            _HuggingFaceDiffusersBackendConfig,
            _HuggingFaceDiffusersVisionBackend,
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
            OptionalDependencyMissingError,
            req_types,
        ) = _import_abstractvision()
        ImageGenerationRequest, _ImageEditRequest = req_types

        extra_args = _env("ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS")
        cfg = StableDiffusionCppBackendConfig(
            sd_cli_path=_env("ABSTRACTCORE_VISION_SDCPP_BIN", "sd-cli") or "sd-cli",
            model=model_path,
            diffusion_model=diffusion_model_path,
            vae=_env("ABSTRACTCORE_VISION_SDCPP_VAE"),
            llm=_env("ABSTRACTCORE_VISION_SDCPP_LLM"),
            llm_vision=_env("ABSTRACTCORE_VISION_SDCPP_LLM_VISION"),
            extra_args=shlex.split(str(extra_args)) if extra_args else (),
            timeout_s=float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "3600") or "3600"),
        )
        backend = StableDiffusionCppVisionBackend(config=cfg)
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Unknown vision backend kind: {backend_kind!r} (set ABSTRACTCORE_VISION_BACKEND)",
        )

    data_items = []
    for _ in range(n):
        try:
            asset = backend.generate_image(
                ImageGenerationRequest(
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
            )
        except OptionalDependencyMissingError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
        data_items.append({"b64_json": b64})

    return {"created": int(time.time()), "data": data_items}


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

    backend_kind = _vision_backend_kind()
    if backend_kind == "openai_compatible_proxy":
        base_url = _require_upstream_base_url()
        model_id = str(model or _env("ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID") or "").strip() or None
        (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
            _HuggingFaceDiffusersBackendConfig,
            _HuggingFaceDiffusersVisionBackend,
            _StableDiffusionCppBackendConfig,
            _StableDiffusionCppVisionBackend,
            OptionalDependencyMissingError,
            req_types,
        ) = _import_abstractvision()
        _ImageGenerationRequest, ImageEditRequest = req_types
        cfg = OpenAICompatibleBackendConfig(
            base_url=base_url,
            api_key=_env("ABSTRACTCORE_VISION_UPSTREAM_API_KEY"),
            model_id=model_id,
            timeout_s=float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "300") or "300"),
            image_generations_path=_env("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH", "/images/generations")
            or "/images/generations",
            image_edits_path=_env("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_EDITS_PATH", "/images/edits") or "/images/edits",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)
    elif backend_kind == "diffusers":
        model_id = _require_diffusers_model_id(model)
        (
            _OpenAICompatibleBackendConfig,
            _OpenAICompatibleVisionBackend,
            HuggingFaceDiffusersBackendConfig,
            HuggingFaceDiffusersVisionBackend,
            _StableDiffusionCppBackendConfig,
            _StableDiffusionCppVisionBackend,
            OptionalDependencyMissingError,
            req_types,
        ) = _import_abstractvision()
        _ImageGenerationRequest, ImageEditRequest = req_types
        cfg = HuggingFaceDiffusersBackendConfig(
            model_id=model_id,
            device=_env("ABSTRACTCORE_VISION_DEVICE", "cpu") or "cpu",
            torch_dtype=_env("ABSTRACTCORE_VISION_TORCH_DTYPE"),
            allow_download=_env_bool("ABSTRACTCORE_VISION_ALLOW_DOWNLOAD", False),
        )
        backend = HuggingFaceDiffusersVisionBackend(config=cfg)
    elif backend_kind == "sdcpp":
        model_path, diffusion_model_path = _require_sdcpp_model_or_diffusion_model(model)
        (
            _OpenAICompatibleBackendConfig,
            _OpenAICompatibleVisionBackend,
            _HuggingFaceDiffusersBackendConfig,
            _HuggingFaceDiffusersVisionBackend,
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
            OptionalDependencyMissingError,
            req_types,
        ) = _import_abstractvision()
        _ImageGenerationRequest, ImageEditRequest = req_types

        extra_args = _env("ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS")
        cfg = StableDiffusionCppBackendConfig(
            sd_cli_path=_env("ABSTRACTCORE_VISION_SDCPP_BIN", "sd-cli") or "sd-cli",
            model=model_path,
            diffusion_model=diffusion_model_path,
            vae=_env("ABSTRACTCORE_VISION_SDCPP_VAE"),
            llm=_env("ABSTRACTCORE_VISION_SDCPP_LLM"),
            llm_vision=_env("ABSTRACTCORE_VISION_SDCPP_LLM_VISION"),
            extra_args=shlex.split(str(extra_args)) if extra_args else (),
            timeout_s=float(_env("ABSTRACTCORE_VISION_TIMEOUT_S", "3600") or "3600"),
        )
        backend = StableDiffusionCppVisionBackend(config=cfg)
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Unknown vision backend kind: {backend_kind!r} (set ABSTRACTCORE_VISION_BACKEND)",
        )

    try:
        asset = backend.edit_image(
            ImageEditRequest(
                prompt=prompt_s,
                image=bytes(image_bytes),
                mask=bytes(mask_bytes) if mask_bytes else None,
                negative_prompt=str(negative_prompt) if negative_prompt is not None else None,
                seed=_coerce_int(seed),
                steps=_coerce_int(steps),
                guidance_scale=_coerce_float(guidance_scale),
                extra=extra,
            )
        )
    except OptionalDependencyMissingError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    b64 = base64.b64encode(bytes(asset.data)).decode("ascii")
    return {"created": int(time.time()), "data": [{"b64_json": b64}]}
