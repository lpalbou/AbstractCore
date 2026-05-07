"""
OpenAI-compatible audio endpoints for AbstractCore Server.

These endpoints are intentionally dependency-light:
- They do not require an LLM provider configuration.
- They delegate to AbstractCore capability plugins (e.g. AbstractVoice).

Endpoints:
- POST /v1/audio/transcriptions (multipart; STT)
- POST /v1/audio/translations (multipart; not yet supported)
- POST /v1/audio/speech (json; TTS)
- POST /v1/voice/clone (multipart; AbstractVoice-compatible extension)
- POST /v1/audio/music (json; text-to-music via capability plugins)
"""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ..capabilities.errors import CapabilityUnavailableError
from ..exceptions import AuthenticationError, InvalidRequestError, ModelNotFoundError, ProviderAPIError, RateLimitError


router = APIRouter(tags=["audio"])

_CORE_LOCK = threading.Lock()
_CORE: Optional[Any] = None
_PROVIDER_API_KEY_HEADERS = (
    "x-abstractcore-provider-api-key",
    "x-provider-api-key",
)
_PLACEHOLDER_API_KEYS = {"not-needed", "not_needed", "notneeded", "unused", "dummy", "empty", "none"}
_SUPPORTED_REMOTE_AUDIO_PROVIDERS = {"openai", "openrouter", "portkey", "openai-compatible"}
_LOCAL_AUDIO_MODEL_ALIASES = {
    # `local/abstractvoice` is kept as a backward-compatible alias; prefer
    # provider/model-style `abstractvoice/default` in docs and examples.
    "local/abstractvoice",
    "abstractvoice/default",
}
_DEFAULT_AUDIO_MAX_BYTES = 25 * 1024 * 1024


class AudioSpeechRequest(BaseModel):
    """OpenAI-compatible text-to-speech request body."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "openai/gpt-4o-mini-tts",
                    "input": "Hello from AbstractCore.",
                    "text": None,
                    "voice": "coral",
                    "response_format": "wav",
                    "format": None,
                    "speed": 1.0,
                    "instructions": "Speak clearly and calmly.",
                    "provider": {},
                    "base_url": None,
                }
            ]
        }
    )

    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider/model id for remote TTS routing, e.g. "
            "`openai/gpt-4o-mini-tts`, `openai/tts-1`, `portkey/default`, or "
            "`openai-compatible/my-tts-model`. If omitted, AbstractCore delegates "
            "to local capability plugins such as abstractvoice. Clients that require "
            "a model string can use `abstractvoice/default` for local plugin fallback."
        ),
        examples=["openai/gpt-4o-mini-tts", "abstractvoice/default"],
    )
    input: Optional[str] = Field(
        default=None,
        description="Text to synthesize. Required unless using the AbstractCore-compatible `text` alias.",
        examples=["Hello from AbstractCore."],
    )
    text: Optional[str] = Field(
        default=None,
        description="AbstractCore-compatible alias for `input`.",
        examples=["Hello from AbstractCore."],
    )
    voice: Optional[str] = Field(
        default=None,
        description=(
            "Voice name supported by the selected provider/backend. For OpenAI TTS, common "
            "built-in voices include `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, "
            "`nova`, `onyx`, `sage`, `shimmer`, `verse`, `marin`, and `cedar`. Defaults "
            "to `alloy` for remote OpenAI-compatible routing when omitted."
        ),
        examples=["coral"],
    )
    response_format: Optional[str] = Field(
        default=None,
        description="Requested audio format for remote providers, e.g. `mp3`, `wav`, `opus`, `aac`, `flac`, or `pcm` when supported.",
        examples=["mp3"],
    )
    format: Optional[str] = Field(
        default=None,
        description="AbstractCore-compatible alias for `response_format`; defaults to `wav` for local plugin fallback.",
        examples=["wav"],
    )
    speed: Optional[float] = Field(
        default=None,
        description="Optional speech speed multiplier when supported by the provider.",
        examples=[1.0],
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Optional style/instruction text for providers that support expressive TTS controls.",
        examples=["Speak clearly and calmly."],
    )
    provider: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional provider-routing options forwarded to OpenAI-compatible gateways such as OpenRouter.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional request-level base URL override. Use this mainly with "
            "`openai-compatible/...` or a gateway/local endpoint. If set with `openai/...`, "
            "the request is sent to that URL instead of api.openai.com. Loopback URLs are "
            "allowed by default; non-loopback URLs require ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    )


class AudioMusicRequest(BaseModel):
    """Text-to-music request body for capability-plugin backends."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "prompt": "A short calm piano loop.",
                    "input": None,
                    "text": None,
                    "lyrics": None,
                    "response_format": "wav",
                    "format": None,
                }
            ]
        },
    )

    prompt: Optional[str] = Field(
        default=None,
        description="Music prompt. Required unless using `input` or `text` alias.",
        examples=["A short calm piano loop."],
    )
    input: Optional[str] = Field(default=None, description="Alias for `prompt`.", examples=["A short calm piano loop."])
    text: Optional[str] = Field(default=None, description="Alias for `prompt`.", examples=["A short calm piano loop."])
    lyrics: Optional[str] = Field(default=None, description="Optional lyrics for backends that support vocal music.")
    response_format: Optional[str] = Field(default=None, description="Output format. Only `wav` is currently supported.", examples=["wav"])
    format: Optional[str] = Field(default=None, description="Alias for `response_format`. Only `wav` is currently supported.", examples=["wav"])


def _model_payload(model: BaseModel) -> Dict[str, Any]:
    data = model.model_dump(exclude_none=True)
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        data.update(extra)
    return data


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _capability_config_from_env() -> Dict[str, Any]:
    """Map server env to AbstractCore capability-plugin config keys."""
    config: Dict[str, Any] = {}
    env_map = {
        "ABSTRACTVOICE_LANGUAGE": "voice_language",
        "ABSTRACTVOICE_TTS_ENGINE": "voice_tts_engine",
        "ABSTRACTVOICE_STT_ENGINE": "voice_stt_engine",
        "ABSTRACTVOICE_WHISPER_MODEL": "voice_whisper_model",
        "ABSTRACTVOICE_TTS_MODEL": "voice_tts_model",
        "ABSTRACTVOICE_STT_MODEL": "voice_stt_model",
        "ABSTRACTVOICE_CLONING_ENGINE": "voice_cloning_engine",
        "ABSTRACTVOICE_REMOTE_BASE_URL": "voice_remote_base_url",
        "ABSTRACTVOICE_REMOTE_API_KEY": "voice_remote_api_key",
        "ABSTRACTVOICE_REMOTE_TIMEOUT_S": "voice_remote_timeout_s",
        "ABSTRACTVOICE_TTS_DELIVERY_MODE": "voice_tts_delivery_mode",
    }
    for env_name, config_name in env_map.items():
        value = os.getenv(env_name)
        if isinstance(value, str) and value.strip():
            config[config_name] = value.strip()
    if os.getenv("ABSTRACTVOICE_ALLOW_DOWNLOADS") is not None:
        config["voice_allow_downloads"] = _bool_env("ABSTRACTVOICE_ALLOW_DOWNLOADS", default=True)
    if os.getenv("ABSTRACTVOICE_CLONED_TTS_STREAMING") is not None:
        config["voice_cloned_tts_streaming"] = _bool_env("ABSTRACTVOICE_CLONED_TTS_STREAMING", default=True)
    if os.getenv("ABSTRACTVOICE_DEBUG") is not None:
        config["voice_debug_mode"] = _bool_env("ABSTRACTVOICE_DEBUG", default=False)
    return config


def _get_capability_core() -> Any:
    """Create/reuse a tiny BaseProvider host for capability plugins."""
    global _CORE
    with _CORE_LOCK:
        if _CORE is not None:
            return _CORE

        from .capability_generation import create_capability_generation_core

        _CORE = create_capability_generation_core(**_capability_config_from_env())
        return _CORE


def _require_dict(data: Any, *, where: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail=f"Invalid request body for {where}: expected a JSON object.")
    return data


def _server_api_key() -> str:
    return str(os.getenv("ABSTRACTCORE_SERVER_API_KEY") or "").strip()


def _server_auth_enabled() -> bool:
    return bool(_server_api_key())


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


def _is_placeholder_api_key(token: Any) -> bool:
    text = str(token or "").strip()
    return not text or text.lower() in _PLACEHOLDER_API_KEYS


def _provider_api_key_from_request(request: Request) -> Optional[str]:
    for header_name in _PROVIDER_API_KEY_HEADERS:
        token = _extract_secret_header(request.headers.get(header_name))
        if not _is_placeholder_api_key(token):
            return token

    if _server_auth_enabled():
        return None

    token = _extract_bearer_token(request.headers.get("authorization"))
    if _is_placeholder_api_key(token):
        return None
    return token


def _request_has_server_auth(request: Request) -> bool:
    return bool(getattr(request.state, "abstractcore_server_authenticated", False))


def _provider_api_key_env_var(provider: str) -> Optional[str]:
    p = str(provider or "").strip().lower()
    if p == "openai":
        return "OPENAI_API_KEY"
    if p == "openrouter":
        return "OPENROUTER_API_KEY"
    if p == "portkey":
        return "PORTKEY_API_KEY"
    if p == "openai-compatible":
        return "OPENAI_COMPATIBLE_API_KEY"
    return None


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name)
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


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


def _server_has_provider_api_key(provider: str) -> bool:
    env_var = _provider_api_key_env_var(provider)
    if env_var and str(os.getenv(env_var) or "").strip():
        return True
    try:
        from ..config import manager as config_manager_module

        cfg = getattr(config_manager_module, "_config_manager", None)
        if cfg is not None:
            runtime_config = cfg.get_provider_config(provider)
            for key_name in ("api_key", "provider_api_key"):
                if str(runtime_config.get(key_name) or "").strip():
                    return True
    except Exception:
        pass
    return False


def _guard_unauthenticated_server_provider_key_use(provider: str, *, explicit_provider_key: bool, request: Request) -> None:
    if _request_has_server_auth(request) or explicit_provider_key:
        return
    if not _server_has_provider_api_key(provider):
        return
    env_var = _provider_api_key_env_var(provider) or "provider API key"
    raise HTTPException(
        status_code=401,
        detail=(
            f"Server-held {env_var} is configured, but inbound server auth is not. "
            "Set ABSTRACTCORE_SERVER_API_KEY and send Authorization: Bearer <server-key>, "
            "or pass an explicit provider key with Authorization: Bearer <provider-key> "
            "when server auth is not configured."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


def _parse_model_string(model_string: str) -> tuple[str, str]:
    raw = str(model_string or "").strip()
    if not raw or "/" not in raw:
        raise HTTPException(
            status_code=400,
            detail="Audio provider routing requires model in provider/model format.",
        )
    provider, model = raw.split("/", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in _SUPPORTED_REMOTE_AUDIO_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio provider '{provider}' is not supported. Supported providers: {', '.join(sorted(_SUPPORTED_REMOTE_AUDIO_PROVIDERS))}.",
        )
    if not model:
        raise HTTPException(status_code=400, detail="Audio model name must not be empty.")
    return provider, model


def _is_local_audio_model(model_string: str) -> bool:
    raw = str(model_string or "").strip().lower()
    return raw in _LOCAL_AUDIO_MODEL_ALIASES


def _create_audio_provider(
    provider: str,
    model: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if provider == "openai":
        from ..providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, **kwargs)
    if provider == "openrouter":
        from ..providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(model=model, validate_model=False, **kwargs)
    if provider == "portkey":
        from ..providers.portkey_provider import PortkeyProvider

        return PortkeyProvider(model=model, **kwargs)
    if provider == "openai-compatible":
        from ..providers.openai_compatible_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model=model, **kwargs)
    raise ValueError(f"Unsupported audio provider: {provider}")


def _provider_exception_status(exc: Exception) -> int:
    if isinstance(exc, AuthenticationError):
        return 401
    if isinstance(exc, RateLimitError):
        return 429
    if isinstance(exc, ModelNotFoundError):
        return 404
    if isinstance(exc, InvalidRequestError):
        return 400
    if isinstance(exc, ProviderAPIError):
        return 502
    text = str(exc).lower()
    if "api key" in text or "authentication" in text or "unauthorized" in text:
        return 401
    return 500


def _optional_text(value: Any) -> Optional[str]:
    return str(value).strip() if isinstance(value, str) and value.strip() else None


def _optional_float(value: Any, *, field: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid {field}: expected a number.") from e


def _audio_max_bytes() -> int:
    raw = str(os.getenv("ABSTRACTCORE_SERVER_AUDIO_MAX_BYTES") or "").strip()
    if not raw:
        return _DEFAULT_AUDIO_MAX_BYTES
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else _DEFAULT_AUDIO_MAX_BYTES
    except Exception:
        return _DEFAULT_AUDIO_MAX_BYTES


def _enforce_audio_size(audio: bytes) -> None:
    limit = _audio_max_bytes()
    if len(audio) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded audio exceeds server limit ({len(audio)} bytes > {limit} bytes).",
        )


def _json_or_raw_audio_response(content: bytes, content_type: str, *, fallback_media_type: str = "application/json"):
    media_type = str(content_type or fallback_media_type).split(";", 1)[0].strip() or fallback_media_type
    if media_type == "application/json" or media_type.endswith("+json"):
        try:
            return json.loads(content.decode("utf-8"))
        except Exception:
            return {"text": content.decode("utf-8", errors="replace")}
    return Response(content=content, media_type=media_type)


@router.post("/audio/transcriptions")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe. Common formats include mp3, mp4, mpeg, mpga, m4a, wav, and webm."),
    model: Optional[str] = Form(
        default=None,
        description=(
            "Optional provider/model id for remote STT routing, e.g. `openai/gpt-4o-mini-transcribe`, "
            "`openai/whisper-1`, `openrouter/...`, `portkey/...`, or `openai-compatible/...`. "
            "If omitted, AbstractCore delegates to local capability plugins such as abstractvoice. "
            "Clients that require a model string can use `abstractvoice/default` for local plugin fallback."
        ),
        examples=["openai/gpt-4o-mini-transcribe"],
    ),
    language: Optional[str] = Form(default=None, description="Optional input language code such as `en` or `fr`.", examples=["en"]),
    prompt: Optional[str] = Form(default=None, description="Optional transcription prompt/context for providers that support it.", examples=["Technical discussion about AbstractCore endpoints."]),
    response_format: Optional[str] = Form(default=None, description="Optional provider response format such as `json`, `text`, `srt`, or `vtt`.", examples=["json"]),
    temperature: Optional[float] = Form(default=None, description="Optional sampling temperature for providers that support it.", examples=[0.0]),
    format: Optional[str] = Form(default=None, description="Optional audio format override used by providers such as OpenRouter base64 audio input.", examples=["mp3"]),
    base_url: Optional[str] = Form(
        default=None,
        description=(
            "Optional request-level base URL override. Use this mainly with "
            "`openai-compatible/...` or a gateway/local endpoint. If set with `openai/...`, "
            "the request is sent to that URL instead of api.openai.com. Loopback URLs are "
            "allowed by default; non-loopback URLs require ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    ),
):
    """OpenAI-compatible STT endpoint (multipart form with file)."""
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded audio file: {e}") from e
    _enforce_audio_size(bytes(audio_bytes))

    language = str(language).strip() if isinstance(language, str) and language.strip() else None
    model = _optional_text(model)
    if model and _is_local_audio_model(model):
        model = None
    if model:
        provider, model_name = _parse_model_string(model)
        provider_api_key = _provider_api_key_from_request(request)
        base_url_s = _validate_request_base_url(base_url)
        _guard_unauthenticated_server_provider_key_use(
            provider,
            explicit_provider_key=bool(provider_api_key),
            request=request,
        )
        filename = str(getattr(file, "filename", "") or "audio.wav")
        content_type = str(getattr(file, "content_type", "") or "application/octet-stream")
        try:
            audio_provider = _create_audio_provider(provider, model_name, api_key=provider_api_key, base_url=base_url_s)
            content, upstream_content_type = audio_provider.transcribe_audio(
                bytes(audio_bytes),
                filename=filename,
                content_type=content_type,
                language=language,
                prompt=_optional_text(prompt),
                response_format=_optional_text(response_format),
                temperature=_optional_float(temperature, field="temperature"),
                audio_format=_optional_text(format),
            )
        except Exception as e:
            raise HTTPException(status_code=_provider_exception_status(e), detail=f"Audio transcription failed: {e}") from e
        return _json_or_raw_audio_response(content, upstream_content_type)

    filename = str(getattr(file, "filename", "") or "audio.wav")
    content_type = str(getattr(file, "content_type", "") or "audio/wav")
    core = _get_capability_core()
    try:
        result = core.generate(
            media={
                "type": "audio",
                "content": bytes(audio_bytes),
                "mime_type": content_type,
                "filename": filename,
            },
            output={"modality": "text", "task": "transcription", "language": language},
        )
        text = getattr(getattr(result, "text", None), "content", None)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}") from e

    return {"text": str(text or "").strip()}


@router.post("/audio/translations")
async def audio_translations(
    file: UploadFile = File(..., description="Audio file to translate. This endpoint is reserved for OpenAI compatibility."),
    model: Optional[str] = Form(default=None, description="Requested translation model. Currently not implemented by AbstractCore Server.", examples=["openai/whisper-1"]),
):
    """OpenAI-compatible audio translations endpoint.

    AbstractCore's capability contract currently does not expose a translation operation.
    Return 501 with actionable messaging instead of silently falling back to transcription.
    """

    raise HTTPException(
        status_code=501,
        detail="audio/translations is not supported by AbstractCore Server (v1). Use /v1/audio/transcriptions instead.",
    )


_AUDIO_BINARY_RESPONSE = {"schema": {"type": "string", "format": "binary"}}


def _audio_response(content: bytes, *, media_type: str, filename_stem: str = "abstractcore-speech") -> Response:
    normalized_media_type = str(media_type or "application/octet-stream").split(";", 1)[0]
    extension = {
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/opus": "opus",
        "audio/aac": "aac",
        "audio/flac": "flac",
        "audio/pcm": "pcm",
    }.get(normalized_media_type, "bin")
    return Response(
        content=bytes(content),
        media_type=normalized_media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename_stem}.{extension}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/audio/speech",
    response_class=Response,
    responses={
        200: {
            "description": "Generated audio bytes.",
            "content": {
                "audio/wav": _AUDIO_BINARY_RESPONSE,
                "audio/mpeg": _AUDIO_BINARY_RESPONSE,
                "audio/opus": _AUDIO_BINARY_RESPONSE,
                "audio/aac": _AUDIO_BINARY_RESPONSE,
                "audio/flac": _AUDIO_BINARY_RESPONSE,
                "audio/pcm": _AUDIO_BINARY_RESPONSE,
                "application/octet-stream": _AUDIO_BINARY_RESPONSE,
            },
        }
    },
)
async def audio_speech(request: Request, payload: AudioSpeechRequest = Body(...)):
    """OpenAI-compatible TTS endpoint (json with input text)."""
    data = _model_payload(payload)

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
    fmt = str(fmt).strip().lower() if isinstance(fmt, str) and fmt.strip() else None

    model = _optional_text(data.get("model"))
    if model and _is_local_audio_model(model):
        model = None
    if model:
        provider, model_name = _parse_model_string(model)
        provider_api_key = _provider_api_key_from_request(request)
        base_url_s = _validate_request_base_url(data.get("base_url"))
        _guard_unauthenticated_server_provider_key_use(
            provider,
            explicit_provider_key=bool(provider_api_key),
            request=request,
        )
        try:
            audio_provider = _create_audio_provider(provider, model_name, api_key=provider_api_key, base_url=base_url_s)
            content, content_type = audio_provider.synthesize_speech(
                str(input_text),
                voice=voice or "alloy",
                response_format=fmt,
                speed=_optional_float(data.get("speed"), field="speed"),
                instructions=_optional_text(data.get("instructions")),
                provider=data.get("provider") if isinstance(data.get("provider"), dict) else None,
            )
        except Exception as e:
            raise HTTPException(status_code=_provider_exception_status(e), detail=f"Audio synthesis failed: {e}") from e
        return _audio_response(content, media_type=str(content_type or "application/octet-stream"))

    fmt = fmt or "wav"

    core = _get_capability_core()
    try:
        result = core.generate(
            text=str(input_text),
            output={
                "modality": "voice",
                "task": "tts",
                "voice": voice,
                "format": fmt,
                "speed": _optional_float(data.get("speed"), field="speed"),
                "instructions": _optional_text(data.get("instructions")),
                "provider": data.get("provider") if isinstance(data.get("provider"), dict) else None,
            },
        )
        voice_items = getattr(result, "outputs", {}).get("voice", [])
        audio_item = voice_items[0] if voice_items else None
        audio = getattr(audio_item, "data", None)
        content_type = getattr(audio_item, "content_type", None)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio synthesis failed: {e}") from e

    if not isinstance(audio, (bytes, bytearray)):
        raise HTTPException(
            status_code=500,
            detail="TTS backend returned an unexpected type (expected raw bytes).",
        )

    return _audio_response(bytes(audio), media_type=str(content_type or f"audio/{fmt}"))


@router.post("/voice/clone")
async def voice_clone(
    request: Request,
    file: UploadFile = File(..., description="Reference voice audio file, commonly WAV/FLAC/OGG/MP3/M4A/WEBM."),
    model: Optional[str] = Form(
        default=None,
        description=(
            "Optional provider/model id for remote voice-clone routing. Use "
            "`openai-compatible/default` for an AbstractVoice-compatible server, or "
            "`openai/default` for OpenAI custom voice creation where available. "
            "If omitted, AbstractCore delegates to the local abstractvoice plugin path."
        ),
        examples=["openai-compatible/default"],
    ),
    name: Optional[str] = Form(default=None, description="Friendly cloned voice name.", examples=["my_voice"]),
    reference_text: Optional[str] = Form(default=None, description="Transcript of the reference audio when available.", examples=["Hello from AbstractCore voice cloning."]),
    validate: Optional[bool] = Form(default=None, description="Ask compatible clone servers to validate the clone before returning.", examples=[True]),
    base_url: Optional[str] = Form(
        default=None,
        description=(
            "Optional request-level base URL for OpenAI-compatible voice endpoints. "
            "Loopback URLs are allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
        examples=["http://127.0.0.1:5000/v1"],
    ),
    clone_path: Optional[str] = Form(
        default=None,
        description=(
            "Optional provider-specific clone path. Defaults to `/voice/clone` for "
            "OpenAI-compatible servers and `/audio/voices` for OpenAI."
        ),
        examples=["/voice/clone"],
    ),
    file_field: Optional[str] = Form(
        default=None,
        description="Optional provider-specific multipart file field. Defaults to `file`; OpenAI uses `audio_sample`.",
        examples=["file"],
    ),
    consent: Optional[str] = Form(default=None, description="Optional provider-specific consent id for custom voice creation.", examples=["consent_123"]),
):
    """AbstractVoice-compatible voice-clone endpoint.

    `/v1/voice/clone` is an extension route. It mirrors AbstractVoice's
    compatible server contract and returns the upstream/local clone JSON.
    """
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded audio file: {e}") from e
    _enforce_audio_size(bytes(audio_bytes))

    model = _optional_text(model)
    if model and _is_local_audio_model(model):
        model = None

    filename = str(getattr(file, "filename", "") or "reference.wav")
    content_type = str(getattr(file, "content_type", "") or "audio/wav")

    if model:
        provider, model_name = _parse_model_string(model)
        provider_api_key = _provider_api_key_from_request(request)
        base_url_s = _validate_request_base_url(base_url)
        _guard_unauthenticated_server_provider_key_use(
            provider,
            explicit_provider_key=bool(provider_api_key),
            request=request,
        )
        try:
            audio_provider = _create_audio_provider(provider, model_name, api_key=provider_api_key, base_url=base_url_s)
            path = _optional_text(clone_path)
            if path is None:
                path = "/audio/voices" if provider == "openai" else "/voice/clone"
            field = _optional_text(file_field)
            if field is None:
                field = "audio_sample" if provider == "openai" else "file"
            out = audio_provider.clone_voice(
                bytes(audio_bytes),
                filename=filename,
                content_type=content_type,
                name=_optional_text(name),
                reference_text=_optional_text(reference_text),
                validate=validate,
                clone_path=path,
                file_field=field,
                consent=_optional_text(consent),
            )
        except Exception as e:
            raise HTTPException(status_code=_provider_exception_status(e), detail=f"Voice clone failed: {e}") from e
        return out

    core = _get_capability_core()
    try:
        result = core.generate(
            text=_optional_text(reference_text) or "",
            media={
                "type": "audio",
                "content": bytes(audio_bytes),
                "mime_type": content_type,
                "filename": filename,
                "role": "clone_sample",
            },
            output={
                "modality": "voice",
                "task": "voice_clone",
                "name": _optional_text(name),
                "reference_text": _optional_text(reference_text),
                "consent": _optional_text(consent),
                "validate": validate,
                "cloning_engine": os.getenv("ABSTRACTVOICE_CLONING_ENGINE") or None,
            },
        )
        voice_resources = getattr(result, "resources", {}).get("voice", [])
        resource = voice_resources[0] if voice_resources else None
        voice_id = getattr(resource, "resource_id", None)
        if not voice_id:
            raise RuntimeError("Voice clone backend did not return a usable voice id.")
        metadata = getattr(resource, "metadata", None)
        info = dict(metadata or {}) if isinstance(metadata, dict) else {}
        return {"ok": True, "id": str(voice_id), "voice_id": str(voice_id), "voice": info}
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {e}") from e


@router.post("/audio/music")
async def audio_music(payload: AudioMusicRequest = Body(...)):
    """Text-to-music endpoint (extension; no official OpenAI equivalent).

    Delegates to the `music` capability plugin (typically `abstractmusic`).
    Returns raw audio bytes (WAV baseline).
    """

    data = _model_payload(payload)

    prompt = data.get("prompt")
    if prompt is None:
        prompt = data.get("input")
    if prompt is None:
        prompt = data.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=422, detail="Missing required field: prompt (string)")

    lyrics = data.get("lyrics")
    lyrics = str(lyrics) if isinstance(lyrics, str) else None

    fmt = data.get("format")
    if fmt is None:
        fmt = data.get("response_format")
    fmt = str(fmt).strip().lower() if isinstance(fmt, str) and fmt.strip() else "wav"
    if fmt != "wav":
        raise HTTPException(status_code=422, detail="Only format='wav' is supported for /v1/audio/music (v1).")

    # Forward extra knobs (best-effort).
    kwargs = {k: v for k, v in data.items() if k not in {"prompt", "input", "text", "lyrics", "format", "response_format"}}

    core = _get_capability_core()
    try:
        audio = core.music.t2m(str(prompt), lyrics=lyrics, format=fmt, **kwargs)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Music generation failed: {e}") from e

    if not isinstance(audio, (bytes, bytearray)):
        raise HTTPException(status_code=500, detail="Music backend returned an unexpected type (expected raw bytes).")

    return Response(content=bytes(audio), media_type="audio/wav")
