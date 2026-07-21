"""Camera control endpoints for the AbstractCore server.

Extension endpoints (OpenAI has no camera API), following the
`/v1/audio/music` precedent: JSON in, JSON or binary out, and 501 with an
install hint when no camera capability plugin is registered — never a
silent stub. The photo route answers OpenAI-images-shaped
`{created, data: [{b64_json}]}` for client familiarity (ruled, commons
c3168).

Discovery deliberately rides the existing generic capability routes
(`/v1/capabilities/camera/providers` and `/v1/capabilities/camera/models`);
this module adds device control + capture + detection.

Routes:
- GET  /v1/camera/devices          (non-invasive discovery)
- POST /v1/camera/open             (claim a device; idempotent default)
- POST /v1/camera/close            (release a device)
- GET  /v1/camera/status           (one camera or all)
- GET  /v1/camera/preview          (latest live-view JPEG)
- POST /v1/camera/photo            (capture a still; b64_json or binary)
- POST /v1/camera/video            (bounded recording; b64_json/binary/none)
- POST /v1/camera/recording/stop   (stop a running recording)
- POST /v1/camera/detection        (arm motion/lightning/meteor detection)
- POST /v1/camera/detection/stop   (disarm)
- GET  /v1/camera/events           (cursor-paginated event log)

Error mapping: `CapabilityUnavailableError` -> 501 (plugin absent, install
hint in detail); camera-state refusals (the plugin raises with honest
user-actionable text: nothing connected, busy, recording, refused by the
body) -> 409; malformed JSON is FastAPI's 422.

Every handler is SYNC-DEF BY DESIGN (2026-07-21, adversarial finding —
the audio_speech precedent in audio_endpoints.py): these handlers await
nothing and call BLOCKING capability ops (capture_video holds the full
recording duration, up to 600s; photo waits up to 300s; open up to 20s).
As `async def` they ran ON the event loop and serialized the whole server
— chat completions, health checks — behind one camera call (head-of-line
wedge class). Sync handlers run in FastAPI's threadpool instead. Threadpool
slots are finite (default 40), but the plugin's per-camera busy refusal
bounds long captures to one per camera. Do not "modernize" these back to
async without moving the blocking calls off-loop.

Owning seat: camera drafted this module per core's ruling (agora commons
c3168, 2026-07-19); abstractcore owns review and merge.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict

from ..capabilities import CapabilityUnavailableError
from ..utils.structured_logging import get_logger

# Shared server-capability host (same object the music/scene3d routes use).
from .audio_endpoints import _get_capability_core

logger = get_logger(__name__)

router = APIRouter(tags=["camera"])

_CAMERA_RESPONSES = {
    409: {"description": "Camera-state refusal (nothing connected, busy, recording, refused by the body)."},
    501: {"description": "No camera capability plugin registered (install hint in detail)."},
}


def _camera_facade() -> Any:
    """The camera facade off the shared capability host; the facade itself
    raises CapabilityUnavailableError when no plugin registered."""
    core = _get_capability_core()
    capabilities = getattr(core, "capabilities", None)
    camera = getattr(capabilities, "camera", None)
    if camera is None:
        raise HTTPException(
            status_code=501,
            detail='No camera capability is available on this server. Install: pip install "abstractcamera"',
        )
    return camera


def _run(op_name: str, fn, *args: Any, **kwargs: Any) -> Any:
    """Uniform error mapping for camera capability calls."""
    try:
        return fn(*args, **kwargs)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        # The camera plugin raises ONE exception type with honest
        # user-actionable text for every state refusal (its contract);
        # 409 says "the request was fine, the camera's state was not".
        raise HTTPException(status_code=409, detail=f"{op_name} refused: {e}") from e


class CameraOpenRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"camera_id": ""}]})
    camera_id: Optional[str] = None


class CameraAddressRequest(BaseModel):
    """`camera` is the DEVICE UID from open/status (empty = active camera)."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"camera": ""}]})
    camera: Optional[str] = None


class CameraPhotoRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"camera": "", "timeout_s": 30, "response_format": "b64_json"}]}
    )
    camera: Optional[str] = None
    timeout_s: Optional[float] = None
    response_format: str = "b64_json"  # b64_json | binary | none


class CameraVideoRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"duration_s": 5.0, "camera": "", "response_format": "none"}]}
    )
    duration_s: float
    camera: Optional[str] = None
    timeout_s: Optional[float] = None
    response_format: str = "none"  # none (metadata JSON) | b64_json | binary


class CameraStopRecordingRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"camera": "", "timeout_s": 30}]})
    camera: Optional[str] = None
    timeout_s: Optional[float] = None


class CameraDetectionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"target": "motion", "action": "photo", "sensitivity": 60}]}
    )
    camera: Optional[str] = None
    target: str = "motion"  # motion | lightning | meteor
    action: str = "photo"  # photo | video | monitor
    sensitivity: Optional[float] = None


def _capture_response(result: Dict[str, Any], *, response_format: str, what: str) -> Any:
    """Shape a capture result. b64_json answers OpenAI-images-shaped
    {created, data: [{b64_json}]} + camera metadata; binary answers raw
    bytes; none answers the metadata dict (path on the server's disk)."""
    fmt = str(response_format or "b64_json").strip().lower()
    if fmt not in {"b64_json", "binary", "none"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported response_format {fmt!r}. Use b64_json, binary, or none.",
        )
    meta = {k: v for k, v in result.items() if k != "data_b64"}
    if fmt == "none" or result.get("path") is None:
        # No local file (on-device save policy / deferred / undelivered
        # movie) has no bytes to encode — the metadata IS the honest answer.
        return {"created": int(time.time()), "camera_result": meta, "data": []}
    data_b64 = result.get("data_b64")
    if not data_b64:
        raise HTTPException(
            status_code=500,
            detail=f"The {what} completed but returned no inline content despite response_format={fmt!r}.",
        )
    if fmt == "binary":
        return Response(
            content=base64.b64decode(data_b64),
            media_type=str(result.get("content_type") or "application/octet-stream"),
            headers={"X-AbstractCore-Camera-Path": str(result.get("path") or "")},
        )
    return {"created": int(time.time()), "data": [{"b64_json": data_b64}], "camera_result": meta}


@router.get("/camera/devices", responses=_CAMERA_RESPONSES, summary="List cameras (non-invasive)")
def camera_devices() -> Dict[str, Any]:
    camera = _camera_facade()
    return {"devices": _run("camera device listing", camera.list_cameras)}


@router.post("/camera/open", responses=_CAMERA_RESPONSES, summary="Claim a camera and start its session")
def camera_open(payload: CameraOpenRequest = Body(default=CameraOpenRequest())) -> Dict[str, Any]:
    camera = _camera_facade()
    return _run("camera open", camera.open, payload.camera_id or None)


@router.post("/camera/close", responses=_CAMERA_RESPONSES, summary="Release a camera")
def camera_close(payload: CameraAddressRequest = Body(default=CameraAddressRequest())) -> Dict[str, Any]:
    camera = _camera_facade()
    return _run("camera close", camera.close, payload.camera or None)


@router.get("/camera/status", responses=_CAMERA_RESPONSES, summary="Camera status (one or all)")
def camera_status(camera: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    facade = _camera_facade()
    return _run("camera status", facade.status, camera or None)


@router.get(
    "/camera/preview",
    response_class=Response,
    responses={**_CAMERA_RESPONSES, 200: {"content": {"image/jpeg": {}}}},
    summary="Latest live-view frame (JPEG)",
)
def camera_preview(camera: Optional[str] = Query(default=None)) -> Response:
    facade = _camera_facade()
    jpeg = _run("camera preview", facade.preview_frame, camera or None)
    if not isinstance(jpeg, (bytes, bytearray)):
        raise HTTPException(status_code=500, detail="Camera preview returned an unexpected payload type.")
    return Response(content=bytes(jpeg), media_type="image/jpeg")


@router.post(
    "/camera/photo",
    responses={**_CAMERA_RESPONSES, 200: {"description": "OpenAI-images-shaped JSON, or binary with response_format=binary."}},
    summary="Capture a photo",
)
def camera_photo(payload: CameraPhotoRequest = Body(default=CameraPhotoRequest())) -> Any:
    camera = _camera_facade()
    fmt = str(payload.response_format or "b64_json").strip().lower()
    result = _run(
        "camera photo capture",
        camera.capture_photo,
        payload.camera or None,
        timeout_s=payload.timeout_s,
        include_bytes=fmt in {"b64_json", "binary"},
    )
    return _capture_response(result, response_format=fmt, what="photo")


@router.post(
    "/camera/video",
    responses={**_CAMERA_RESPONSES, 200: {"description": "Capture metadata (default), b64_json, or binary movie bytes."}},
    summary="Record a bounded video clip",
)
def camera_video(payload: CameraVideoRequest = Body(...)) -> Any:
    camera = _camera_facade()
    fmt = str(payload.response_format or "none").strip().lower()
    result = _run(
        "camera video capture",
        camera.capture_video,
        payload.duration_s,
        payload.camera or None,
        timeout_s=payload.timeout_s,
        include_bytes=fmt in {"b64_json", "binary"},
    )
    return _capture_response(result, response_format=fmt, what="video")


@router.post("/camera/recording/stop", responses=_CAMERA_RESPONSES, summary="Stop a running recording")
def camera_recording_stop(
    payload: CameraStopRecordingRequest = Body(default=CameraStopRecordingRequest()),
) -> Dict[str, Any]:
    camera = _camera_facade()
    result = _run("camera recording stop", camera.stop_recording, payload.camera or None, timeout_s=payload.timeout_s)
    return {"created": int(time.time()), "camera_result": result, "data": []}


@router.post("/camera/detection", responses=_CAMERA_RESPONSES, summary="Arm detection (motion/lightning/meteor)")
def camera_detection(payload: CameraDetectionRequest = Body(default=CameraDetectionRequest())) -> Dict[str, Any]:
    camera = _camera_facade()
    return _run(
        "camera detection arm",
        camera.start_detection,
        payload.camera or None,
        target=payload.target,
        action=payload.action,
        sensitivity=payload.sensitivity,
    )


@router.post("/camera/detection/stop", responses=_CAMERA_RESPONSES, summary="Disarm detection")
def camera_detection_stop(
    payload: CameraAddressRequest = Body(default=CameraAddressRequest()),
) -> Dict[str, Any]:
    camera = _camera_facade()
    return _run("camera detection disarm", camera.stop_detection, payload.camera or None)


@router.get("/camera/events", responses=_CAMERA_RESPONSES, summary="Camera event log (cursor-paginated)")
def camera_events(
    camera: Optional[str] = Query(default=None),
    since_id: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    kinds: Optional[List[str]] = Query(default=None),
) -> Dict[str, Any]:
    facade = _camera_facade()
    return _run(
        "camera events read",
        facade.detection_events,
        camera or None,
        since_id=since_id,
        limit=limit,
        kinds=list(kinds) if kinds else None,
    )


__all__ = [
    "CameraDetectionRequest",
    "CameraOpenRequest",
    "CameraPhotoRequest",
    "CameraVideoRequest",
    "router",
]
