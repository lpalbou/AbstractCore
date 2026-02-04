"""
OpenAI-compatible audio endpoints for AbstractCore Server.

These endpoints are intentionally dependency-light:
- They do not require an LLM provider configuration.
- They delegate to AbstractCore capability plugins (e.g. AbstractVoice).

Endpoints:
- POST /v1/audio/transcriptions (multipart; STT)
- POST /v1/audio/speech (json; TTS)
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ..capabilities.errors import CapabilityUnavailableError


router = APIRouter(tags=["audio"])

_CORE_LOCK = threading.Lock()
_CORE: Optional[Any] = None


def _get_capability_core() -> Any:
    """Create/reuse a tiny AbstractCoreInterface host for capability plugins."""
    global _CORE
    with _CORE_LOCK:
        if _CORE is not None:
            return _CORE

        from ..core.interface import AbstractCoreInterface
        from ..core.types import GenerateResponse

        class _CapabilityOnlyCore(AbstractCoreInterface):
            def generate(self, prompt: str, **kwargs):  # type: ignore[override]
                _ = kwargs
                return GenerateResponse(content=str(prompt))

            def get_capabilities(self):  # type: ignore[override]
                return []

            def unload_model(self, model_name: str) -> None:  # type: ignore[override]
                _ = model_name
                return None

        _CORE = _CapabilityOnlyCore(model="capabilities")
        return _CORE


def _require_dict(data: Any, *, where: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail=f"Invalid request body for {where}: expected a JSON object.")
    return data


@router.post("/audio/transcriptions")
async def audio_transcriptions(request: Request):
    """OpenAI-compatible STT endpoint (multipart form with file)."""
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(
            status_code=501,
            detail='python-multipart is required for /v1/audio/transcriptions. Install with: pip install "python-multipart".',
        ) from e

    file_obj = form.get("file")
    if file_obj is None:
        raise HTTPException(status_code=422, detail="Missing required form field: file")

    # Starlette's UploadFile provides .read() coroutine + filename.
    read = getattr(file_obj, "read", None)
    if not callable(read):
        raise HTTPException(status_code=422, detail="Invalid form field 'file': expected an uploaded file.")

    try:
        audio_bytes = await read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded audio file: {e}") from e

    language = form.get("language")
    language = str(language).strip() if isinstance(language, str) and language.strip() else None

    core = _get_capability_core()
    try:
        text = core.audio.transcribe(bytes(audio_bytes), language=language)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}") from e

    return {"text": str(text or "").strip()}


@router.post("/audio/speech")
async def audio_speech(request: Request):
    """OpenAI-compatible TTS endpoint (json with input text)."""
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {e}") from e

    data = _require_dict(payload, where="/v1/audio/speech")

    input_text = data.get("input")
    if input_text is None:
        input_text = data.get("text")
    if not isinstance(input_text, str) or not input_text.strip():
        raise HTTPException(status_code=422, detail="Missing required field: input (string)")

    voice = data.get("voice")
    voice = str(voice).strip() if isinstance(voice, str) and voice.strip() else None

    fmt = data.get("format")
    if fmt is None:
        fmt = data.get("response_format")
    fmt = str(fmt).strip().lower() if isinstance(fmt, str) and fmt.strip() else "wav"

    core = _get_capability_core()
    try:
        audio = core.voice.tts(str(input_text), voice=voice, format=fmt)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio synthesis failed: {e}") from e

    if not isinstance(audio, (bytes, bytearray)):
        raise HTTPException(
            status_code=500,
            detail="TTS backend returned an unexpected type (expected raw bytes).",
        )

    return Response(content=bytes(audio), media_type=f"audio/{fmt}")
