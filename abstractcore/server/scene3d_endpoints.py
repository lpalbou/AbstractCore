"""Scene3D (3D object) generation endpoints for the AbstractCore server.

Extension endpoints (OpenAI has no 3D generation API), following the
`/v1/audio/music` precedent exactly: JSON request in, binary model bytes out,
`provider` as the capability backend selector, and 501 with an install hint
when no scene3d capability plugin is registered — never a silent stub.

Discovery deliberately rides the existing generic capability routes
(`/v1/capabilities/scene3d/providers` and `/v1/capabilities/scene3d/models`);
this module only adds generation.

Routes:
- POST /v1/scene3d/generations           (text-to-3D, or image-to-3D via image_b64)
- POST /{provider}/v1/scene3d/generations (provider-scoped alias)

Owning seat: abstract3d drafted this module per core's ruling
(agora commons #3169, 2026-07-19); abstractcore owns review and merge.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Path as FastAPIPath, Response
from pydantic import BaseModel, ConfigDict

from ..capabilities import CapabilityUnavailableError
from ..utils.structured_logging import get_logger

# Shared server-capability helpers (same host object the music route uses;
# app.py already imports these cross-module — see `_get_capability_core`
# usage at app.py's residency routes).
from .audio_endpoints import _get_capability_core, _plugin_exception_status

logger = get_logger(__name__)

router = APIRouter(tags=["scene3d"])
provider_router = APIRouter(tags=["scene3d"])

_SCENE3D_MEDIA_TYPES = {
    "glb": "model/gltf-binary",
    "obj": "model/obj",
    "zip": "application/zip",
}

# Source-image size cap for i23d (base64 field). Mirrors the audio upload cap
# (25 MiB default) so one JSON request cannot balloon server memory.
_DEFAULT_IMAGE_MAX_BYTES = 25 * 1024 * 1024


def _image_max_bytes() -> int:
    raw = str(os.getenv("ABSTRACTCORE_SERVER_SCENE3D_IMAGE_MAX_BYTES") or "").strip()
    if not raw:
        return _DEFAULT_IMAGE_MAX_BYTES
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else _DEFAULT_IMAGE_MAX_BYTES
    except Exception:
        return _DEFAULT_IMAGE_MAX_BYTES

# Options forwarded verbatim into the scene3d output spec. Everything else in
# the request body is refused loudly (422) — silently dropping a tuning knob
# would leave callers unable to tell "applied" from "ignored". Filesystem- and
# host-control options (output_dir, artifact_store, run_id, tags) are
# deliberately NOT forwardable over HTTP.
_FORWARDABLE_OPTION_FIELDS = (
    "seed",
    "image_seed",
    "device",
    "mc_resolution",
    "cleanup",
    "texture_mode",
    "texture_resolution",
    "texture_completion",
    "num_inference_steps",
    "guidance_scale",
    "octree_resolution",
    "remove_background",
    "foreground_ratio",
    "image_provider",
    "image_model",
)

_KNOWN_REQUEST_FIELDS = frozenset(
    {"prompt", "input", "text", "image_b64", "provider", "model", "task", "format", "response_format"}
    | set(_FORWARDABLE_OPTION_FIELDS)
)

_SCENE3D_RESPONSES = {
    200: {
        "description": "Generated 3D model bytes (GLB by default).",
        "content": {
            "model/gltf-binary": {},
            "model/obj": {},
            "application/zip": {},
        },
    },
    422: {"description": "Invalid request (unknown option, missing prompt/image, bad format)."},
    501: {"description": "No scene3d capability plugin registered (install hint in detail)."},
}


class Scene3DGenerationRequest(BaseModel):
    """Text-to-3D / image-to-3D request body for capability-plugin backends."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "prompt": "a ceramic teapot with a curved spout and matte glaze",
                    "provider": "triposr",
                    "format": "glb",
                },
                {
                    "image_b64": "<base64-encoded source image>",
                    "provider": "triposr",
                    "task": "image_to_scene3d",
                    "format": "glb",
                },
            ]
        },
    )

    prompt: Optional[str] = None
    input: Optional[str] = None
    text: Optional[str] = None
    image_b64: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task: Optional[str] = None
    format: Optional[str] = None
    response_format: Optional[str] = None


def _request_prompt(payload: Dict[str, Any]) -> str:
    for key in ("prompt", "input", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalized_format(payload: Dict[str, Any]) -> str:
    raw = payload.get("format") or payload.get("response_format") or "glb"
    fmt = str(raw).strip().lower() or "glb"
    if fmt not in _SCENE3D_MEDIA_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported scene3d format {fmt!r}. Supported: {', '.join(sorted(_SCENE3D_MEDIA_TYPES))}.",
        )
    return fmt


def _normalized_task(payload: Dict[str, Any], *, has_image: bool) -> str:
    raw = str(payload.get("task") or "").strip().lower().replace("-", "_")
    if raw in {"", "scene3d", "scene3d_generation"}:
        return "image_to_scene3d" if has_image else "text_to_scene3d"
    if raw in {"t23d", "text_to_scene3d"}:
        return "text_to_scene3d"
    if raw in {"i23d", "image_to_scene3d"}:
        return "image_to_scene3d"
    raise HTTPException(
        status_code=422,
        detail=f"Unsupported scene3d task {raw!r}. Use text_to_scene3d (t23d) or image_to_scene3d (i23d).",
    )


def _reject_unknown_fields(payload: Dict[str, Any]) -> None:
    unknown = sorted(k for k in payload.keys() if k not in _KNOWN_REQUEST_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown scene3d request fields: {', '.join(unknown)}. "
                f"Supported option fields: {', '.join(_FORWARDABLE_OPTION_FIELDS)}."
            ),
        )


def _scene3d_generations_impl(
    payload: Scene3DGenerationRequest,
    *,
    path_provider: Optional[str] = None,
) -> Response:
    # Reject unknown fields on the FULL dump: with exclude_none a client's
    # `"bogus_option": null` would be silently dropped before the check,
    # violating the refuse-loudly contract above.
    _reject_unknown_fields(payload.model_dump())
    data = payload.model_dump(exclude_none=True)

    prompt = _request_prompt(data)
    image_b64 = str(data.get("image_b64") or "").strip()
    has_image = bool(image_b64)
    if not prompt and not has_image:
        raise HTTPException(
            status_code=422,
            detail="Provide a prompt (text-to-3D) or image_b64 (image-to-3D).",
        )

    fmt = _normalized_format(data)
    task = _normalized_task(data, has_image=has_image)
    if task == "image_to_scene3d" and not has_image:
        raise HTTPException(status_code=422, detail="image_to_scene3d requires image_b64.")
    if task == "text_to_scene3d" and has_image:
        # Refuse the ambiguity instead of silently ignoring the image: t23d
        # does not consume a source image.
        raise HTTPException(
            status_code=422,
            detail=(
                "image_b64 was provided with task=text_to_scene3d, which ignores images. "
                "Use task=image_to_scene3d (or omit task) to reconstruct from the image."
            ),
        )
    if has_image:
        # Bounds the DECODE and backend cost, not JSON parse memory — the
        # body is already parsed by the time any handler code runs (an
        # ASGI-level body cap would be a server-wide middleware decision).
        # b64 expands ~4/3 over raw bytes; compare against the encoded budget.
        max_encoded = (_image_max_bytes() * 4) // 3 + 4
        if len(image_b64) > max_encoded:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"image_b64 exceeds the source image limit ({_image_max_bytes()} bytes decoded). "
                    "Set ABSTRACTCORE_SERVER_SCENE3D_IMAGE_MAX_BYTES to raise it."
                ),
            )

    output_spec: Dict[str, Any] = {"modality": "scene3d", "task": task, "format": fmt}
    provider = str(data.get("provider") or path_provider or "").strip()
    if provider:
        output_spec["provider"] = provider
    model = str(data.get("model") or "").strip()
    if model:
        output_spec["model"] = model
    for key in _FORWARDABLE_OPTION_FIELDS:
        if data.get(key) is not None:
            output_spec[key] = data[key]

    media = None
    if has_image:
        media = [
            {
                "type": "image",
                "content": image_b64,
                "content_format": "base64",
                "role": "source",
            }
        ]

    core = _get_capability_core()
    try:
        result = core.generate(text=prompt, media=media, output=output_spec)
        scene_items = getattr(result, "outputs", {}).get("scene3d", [])
        scene_item = scene_items[0] if scene_items else None
        model_bytes = getattr(scene_item, "data", None)
        content_type = getattr(scene_item, "content_type", None)
        backend_id = getattr(scene_item, "backend_id", None)
        model_id = getattr(scene_item, "model", None)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Scene3D generation rejected: {e}") from e
    except Exception as e:
        # Request-class refusals from the plugin must surface as 4xx, not 500.
        # abstract3d marks them with a STABLE machine-readable `error_class`
        # (its InvalidRequestError/CapabilityNotSupportedError are RuntimeError
        # subclasses, invisible to abstractcore's exception taxonomy):
        # - license_acknowledgment_required -> 403 (policy gate, not a bug);
        #   the message phrase "license acknowledgment" stays as a fallback
        #   (abstract3d contracts to keep it).
        # - invalid_request / capability_not_supported -> 422 (the caller sent
        #   an option or task the selected backend rejects, e.g. `seed` on the
        #   feed-forward triposr backend).
        # A source image that decodes but is not an image (PIL
        # UnidentifiedImageError, an OSError) is also the caller's input.
        error_class = getattr(e, "error_class", None)
        if (
            error_class == "license_acknowledgment_required"
            or "license acknowledgment" in str(e).lower()
        ):
            raise HTTPException(status_code=403, detail=f"Scene3D generation refused: {e}") from e
        if error_class in {"invalid_request", "capability_not_supported"}:
            raise HTTPException(status_code=422, detail=f"Scene3D generation rejected: {e}") from e
        if type(e).__name__ == "UnidentifiedImageError":
            raise HTTPException(
                status_code=422,
                detail="image_b64 decoded but is not a readable image file.",
            ) from e
        raise HTTPException(
            status_code=_plugin_exception_status(e), detail=f"Scene3D generation failed: {e}"
        ) from e

    if not isinstance(model_bytes, (bytes, bytearray)):
        raise HTTPException(
            status_code=500,
            detail="Scene3D backend returned an unexpected type (expected raw bytes).",
        )

    headers: Dict[str, str] = {}
    if isinstance(backend_id, str) and backend_id.strip():
        headers["X-AbstractCore-Backend-Id"] = backend_id.strip()
    if isinstance(model_id, str) and model_id.strip():
        headers["X-AbstractCore-Model"] = model_id.strip()
    headers["X-AbstractCore-Task"] = task

    return Response(
        content=bytes(model_bytes),
        media_type=str(content_type or _SCENE3D_MEDIA_TYPES[fmt]),
        headers=headers,
    )


@router.post(
    "/scene3d/generations",
    response_class=Response,
    responses=_SCENE3D_RESPONSES,
    summary="Generate a 3D object",
    description=(
        "Generate a 3D object through the scene3d capability plugin (e.g. abstract3d). "
        "Text-to-3D from `prompt`, or image-to-3D from `image_b64`. Returns raw model "
        "bytes (GLB by default). Option applicability is per backend (e.g. `seed` only on "
        "sampling backends; `mc_resolution` only on triposr) — unsupported options are "
        "refused with 422 naming the offender. Extension endpoint — OpenAI has no 3D "
        "generation API; shape follows /v1/audio/music."
    ),
)
def scene3d_generations(payload: Scene3DGenerationRequest = Body(...)):
    # Deliberately a sync `def`: FastAPI runs it on the threadpool, so a
    # multi-minute synchronous generation cannot stall the event loop (an
    # `async def` handler around the synchronous core.generate serializes
    # EVERY other request behind it for the whole generation).
    return _scene3d_generations_impl(payload)


@provider_router.post(
    "/{provider}/v1/scene3d/generations",
    response_class=Response,
    responses=_SCENE3D_RESPONSES,
    summary="Generate a 3D object (provider-scoped)",
)
def provider_scene3d_generations(
    payload: Scene3DGenerationRequest = Body(...),
    provider: str = FastAPIPath(
        ...,
        description="Scene3D backend route prefix, e.g. `triposr`, `step1x`, `hunyuan3d21`, or `trellis2`.",
    ),
):
    # Sync `def` for the same threadpool reason as the unscoped route.
    return _scene3d_generations_impl(payload, path_provider=provider)


__all__ = [
    "Scene3DGenerationRequest",
    "provider_router",
    "router",
]
