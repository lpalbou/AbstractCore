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
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Body, File, Form, HTTPException, Path as FastAPIPath, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ..capabilities.errors import CapabilityUnavailableError
from ..exceptions import AuthenticationError, InvalidRequestError, ModelNotFoundError, ProviderAPIError, RateLimitError


router = APIRouter(tags=["audio"])
provider_router = APIRouter(tags=["audio"])

_CORE_LOCK = threading.Lock()
_CORE: Optional[Any] = None
_PROVIDER_API_KEY_HEADERS = (
    "x-abstractcore-provider-api-key",
    "x-provider-api-key",
)
_PLACEHOLDER_API_KEYS = {"not-needed", "not_needed", "notneeded", "unused", "dummy", "empty", "none"}
_SUPPORTED_REMOTE_AUDIO_PROVIDERS = {"openai", "openrouter", "portkey", "openai-compatible"}
_REMOTE_AUDIO_PROVIDER_ALIASES = {"openai", "openrouter", "portkey", "openai-compatible", "remote"}
_PLUGIN_SERVICE_UNAVAILABLE_NEEDLES = (
    "requires openai_api_key",
    "requires openai api key",
    "requires remote_api_key",
    "requires remote api key",
    "missing ace music api key",
)
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
                    "profile": None,
                    "response_format": "wav",
                    "format": None,
                    "speed": 1.0,
                    "quality_preset": "standard",
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
    profile: Optional[str] = Field(
        default=None,
        description="Optional local AbstractVoice profile id. This is ignored by strict remote OpenAI-compatible providers.",
        examples=["en_US-amy-medium"],
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
    quality_preset: Optional[str] = Field(
        default=None,
        description="Optional local AbstractVoice quality preset: low, standard, or high. Aliases are normalized by AbstractVoice.",
        examples=["standard"],
    )
    quality: Optional[str] = Field(
        default=None,
        description="Compatibility alias for quality_preset.",
        examples=["high"],
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Optional style/instruction text for providers that support expressive TTS controls.",
        examples=["Speak clearly and calmly."],
    )
    provider: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description=(
            "Optional provider-routing options forwarded to OpenAI-compatible gateways such as OpenRouter, "
            "or a local AbstractVoice engine id such as `piper` for capability-plugin routing."
        ),
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
                    "model": "acemusic/ace-step-api",
                    "backend": "acemusic",
                    "provider": "ACE Music",
                    "prompt": "A short calm piano loop.",
                    "input": None,
                    "text": None,
                    "lyrics": None,
                    "duration_s": 8,
                    "seed": 42,
                    "num_inference_steps": 27,
                    "guidance_scale": 15.0,
                    "instrumental": True,
                    "enhance_prompt": True,
                    "text_planner_mode": "auto",
                    "response_format": "wav",
                    "format": None,
                }
            ]
        },
    )

    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional music model id. For remote ACE Music this is commonly "
            "`acemusic/ace-step-api`; for local AbstractMusic backends this is commonly "
            "a Hugging Face repo id."
        ),
        examples=["acemusic/ace-step-api", "ACE-Step/acestep-v15-xl-turbo-diffusers"],
    )
    provider: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional music provider/catalog filter forwarded to the selected music backend, "
            "for example `ACE Music` or `ace-step`."
        ),
        examples=["ACE Music", "ace-step"],
    )
    backend: Optional[str] = Field(
        default=None,
        description=(
            "Optional music backend selector, e.g. `acemusic`, `remote`, `acestep`, "
            "`acestep-v15`, or `diffusers`."
        ),
        examples=["acemusic"],
    )
    task: Optional[str] = Field(
        default=None,
        description="Optional music task selector. Defaults to `music_generation` / text-to-music.",
        examples=["text_to_music"],
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Music prompt. Required unless using `input` or `text` alias.",
        examples=["A short calm piano loop."],
    )
    input: Optional[str] = Field(default=None, description="Alias for `prompt`.", examples=["A short calm piano loop."])
    text: Optional[str] = Field(default=None, description="Alias for `prompt`.", examples=["A short calm piano loop."])
    lyrics: Optional[str] = Field(default=None, description="Optional lyrics for backends that support vocal music.")
    duration_s: Optional[float] = Field(default=None, description="Requested output duration in seconds.", examples=[8.0])
    seed: Optional[int] = Field(default=None, description="Optional deterministic seed.", examples=[42])
    num_inference_steps: Optional[int] = Field(default=None, description="Optional diffusion/sampling step count.", examples=[27])
    guidance_scale: Optional[float] = Field(default=None, description="Optional classifier-free guidance scale.", examples=[15.0])
    instrumental: Optional[bool] = Field(default=None, description="Request instrumental output when supported.")
    enhance_prompt: Optional[bool] = Field(default=None, description="Enable provider/plugin prompt enhancement when supported.")
    structure_prompt: Optional[bool] = Field(default=None, description="Enable structured prompt planning when supported.")
    auto_lyrics: Optional[bool] = Field(default=None, description="Ask the backend to infer or generate lyrics when supported.")
    text_planner_mode: Optional[str] = Field(default=None, description="Music text-planner mode such as `auto`, `on`, or `off`.")
    response_format: Optional[str] = Field(default=None, description="Output format. Supported by the server contract: `wav`, `mp3`, or `flac`.", examples=["wav"])
    format: Optional[str] = Field(default=None, description="Alias for `response_format`. Supported by the server contract: `wav`, `mp3`, or `flac`.", examples=["wav"])


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
        "OPENAI_BASE_URL": "voice_remote_base_url",
        "ABSTRACTVOICE_REMOTE_TIMEOUT_S": "voice_remote_timeout_s",
        "ABSTRACTVOICE_TTS_DELIVERY_MODE": "voice_tts_delivery_mode",
        "ABSTRACTMUSIC_BACKEND": "music_backend",
        "ABSTRACTMUSIC_MODEL_ID": "music_model_id",
        "ABSTRACTMUSIC_DEVICE": "music_device",
        "ABSTRACTMUSIC_TORCH_DTYPE": "music_torch_dtype",
        "ABSTRACTMUSIC_PIPELINE_CLASS": "music_pipeline_class",
        "ABSTRACTMUSIC_NUM_INFERENCE_STEPS": "music_num_inference_steps",
        "ABSTRACTMUSIC_DURATION_S": "music_duration_s",
        "ABSTRACTMUSIC_GUIDANCE_SCALE": "music_guidance_scale",
        "ABSTRACTMUSIC_TEXT_PLANNER": "music_text_planner_mode",
        "ABSTRACTMUSIC_REVISION": "music_revision",
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

_SERVER_AUTH_TOKEN_ENV_VAR = "ABSTRACTCORE_AUTH_TOKEN"


def _server_auth_token() -> str:
    value = os.getenv(_SERVER_AUTH_TOKEN_ENV_VAR)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _server_auth_enabled() -> bool:
    return bool(_server_auth_token())


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
        return "OPENAI_API_KEY"
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
            "Set ABSTRACTCORE_AUTH_TOKEN and send "
            "Authorization: Bearer <server-token>, "
            "or pass an explicit provider key with X-AbstractCore-Provider-API-Key."
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


def _request_provider_value(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        normalized = value.strip().lower().replace("_", "-")
        if normalized == "remote":
            return "openai-compatible"
        return normalized
    return None


def _resolve_audio_request_routing(
    *,
    model: Any,
    provider: Any = None,
    path_provider: Any = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve route/body audio selectors into provider, remote model, and local model."""
    request_provider = _request_provider_value(path_provider) or _request_provider_value(provider)
    model_name = _optional_text(model)
    local_model = None

    if model_name and _is_local_audio_model(model_name):
        model_name = None

    if request_provider and model_name and "/" in model_name:
        prefix, rest = model_name.split("/", 1)
        if prefix.strip().lower().replace("_", "-") == request_provider:
            model_name = rest.strip() or None

    if request_provider and request_provider not in _REMOTE_AUDIO_PROVIDER_ALIASES:
        local_model = model_name
        model_name = None
    elif request_provider and model_name and "/" not in model_name:
        model_name = f"{request_provider}/{model_name}"

    return request_provider, model_name, local_model


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


def _plugin_exception_status(exc: Exception) -> int:
    if isinstance(exc, (AuthenticationError, RateLimitError, ModelNotFoundError, InvalidRequestError, ProviderAPIError)):
        return _provider_exception_status(exc)

    text = str(exc).lower()
    if "http 504" in text or "gateway time-out" in text or "gateway timeout" in text:
        return 504
    if "http 502" in text or "bad gateway" in text:
        return 502
    if "http 503" in text or "service unavailable" in text:
        return 503
    if "http 5" in text:
        return 502
    if "rate limit" in text or "too many requests" in text or "http 429" in text:
        return 429
    if any(needle in text for needle in _PLUGIN_SERVICE_UNAVAILABLE_NEEDLES):
        return 503
    if "api key" in text or "authentication" in text or "unauthorized" in text or "http 401" in text or "http 403" in text:
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


def _optional_int(value: Any, *, field: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid {field}: expected an integer.") from e


_MUSIC_BACKEND_ALIAS_MAP = {
    "abstractmusic:acemusic": "abstractmusic:acemusic",
    "acemusic": "abstractmusic:acemusic",
    "ace-music": "abstractmusic:acemusic",
    "acemusic-api": "abstractmusic:acemusic",
    "ace-music-api": "abstractmusic:acemusic",
    "aceapi": "abstractmusic:acemusic",
    "remote": "abstractmusic:acemusic",
    "api": "abstractmusic:acemusic",
    "abstractmusic:acestep-diffusers": "abstractmusic:acestep-diffusers",
    "acestep": "abstractmusic:acestep-diffusers",
    "ace-step": "abstractmusic:acestep-diffusers",
    "ace": "abstractmusic:acestep-diffusers",
    "acestep-diffusers": "abstractmusic:acestep-diffusers",
    "ace-step-diffusers": "abstractmusic:acestep-diffusers",
    "abstractmusic:acestep-v15": "abstractmusic:acestep-v15",
    "acestep-v15": "abstractmusic:acestep-v15",
    "ace-step-v15": "abstractmusic:acestep-v15",
    "abstractmusic:diffusers": "abstractmusic:diffusers",
    "diffusers": "abstractmusic:diffusers",
}
_MUSIC_BACKEND_ALIASES = set(_MUSIC_BACKEND_ALIAS_MAP)


def _music_selector_value(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace("_", "-")
    return None


def _music_backend_selector(*values: Any, allow_unknown: bool = False) -> Optional[str]:
    for value in values:
        text = _music_selector_value(value)
        if not text:
            continue
        if text in _MUSIC_BACKEND_ALIAS_MAP:
            return _MUSIC_BACKEND_ALIAS_MAP[text]
        if allow_unknown:
            return text
    return None


def _music_provider_selector(*values: Any) -> Optional[str]:
    for value in values:
        text = _music_selector_value(value)
        if text and text not in _MUSIC_BACKEND_ALIASES:
            return text
    return None


def _music_capability_core_for_request(data: Dict[str, Any], *, path_provider: Optional[str] = None) -> Any:
    """Return a capability host honoring request-level music backend/model overrides."""
    backend = _music_backend_selector(data.get("backend"), data.get("music_backend"), allow_unknown=True)
    if backend is None:
        backend = _music_backend_selector(path_provider)
    model = _optional_text(data.get("model") or data.get("music_model_id"))

    overrides: Dict[str, Any] = {}
    if backend:
        overrides["music_backend"] = backend
    if model:
        overrides["music_model_id"] = model

    passthrough_config = {
        "device": "music_device",
        "torch_dtype": "music_torch_dtype",
        "pipeline_class": "music_pipeline_class",
        "revision": "music_revision",
        "duration_s": "music_duration_s",
        "num_inference_steps": "music_num_inference_steps",
        "guidance_scale": "music_guidance_scale",
        "text_planner_mode": "music_text_planner_mode",
        "text_planner": "music_text_planner_mode",
    }
    for request_key, config_key in passthrough_config.items():
        value = data.get(request_key)
        if value is not None:
            overrides[config_key] = value

    if not overrides:
        return _get_capability_core()

    from .capability_generation import create_capability_generation_core

    config = _capability_config_from_env()
    config.update(overrides)
    return create_capability_generation_core(**config)


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


def _server_has_audio_catalog_credential() -> bool:
    return bool(str(os.getenv("OPENAI_API_KEY") or "").strip())


def _server_allows_unauthenticated() -> bool:
    return _bool_env("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", default=False)


def _guard_audio_catalog_credentials(*, request: Request, explicit_provider_key: bool) -> None:
    if _request_has_server_auth(request) or explicit_provider_key:
        return
    if _server_allows_unauthenticated():
        return
    if not _server_has_audio_catalog_credential():
        return
    raise HTTPException(
        status_code=401,
        detail=(
            "Server-held audio/OpenAI credentials are configured, but inbound server auth was not used. "
            "Set ABSTRACTCORE_AUTH_TOKEN and send "
            "Authorization: Bearer <server-token>, or pass an explicit "
            "provider key with X-AbstractCore-Provider-API-Key for this request."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


def _audio_catalog_core(request: Request, *, base_url: Optional[str], api_key: Optional[str]) -> Any:
    explicit_key = str(api_key).strip() if isinstance(api_key, str) else ""
    if _is_placeholder_api_key(explicit_key):
        explicit_key = ""
    provider_api_key = explicit_key or _provider_api_key_from_request(request)
    base_url_s = _validate_request_base_url(base_url)
    _guard_audio_catalog_credentials(request=request, explicit_provider_key=bool(provider_api_key))
    config = _capability_config_from_env()
    if base_url_s:
        config["voice_remote_base_url"] = base_url_s
    if provider_api_key:
        config["voice_remote_api_key"] = provider_api_key
    from .capability_generation import create_capability_generation_core

    return create_capability_generation_core(**config)


def _audio_catalog_error(exc: Exception) -> HTTPException:
    detail = {
        "available": False,
        "source": "abstractvoice",
        "stale": False,
        "error": str(exc),
    }
    if isinstance(exc, CapabilityUnavailableError):
        return HTTPException(status_code=501, detail=detail)
    msg = str(exc).lower()
    if "not configured" in msg or "missing" in msg or "does not expose" in msg or "not expose" in msg:
        return HTTPException(status_code=501, detail=detail)
    return HTTPException(status_code=502, detail=detail)


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
    provider: Optional[str] = Form(
        default=None,
        description=(
            "Optional provider/engine hint. Use this for local plugin routing such as "
            "`faster-whisper` or `transformers-asr`, or remote routing such as `openai` "
            "or `openai-compatible`."
        ),
        examples=["faster-whisper"],
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
    request_provider, model, local_model = _resolve_audio_request_routing(model=model, provider=provider)
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
            output={
                "modality": "text",
                "task": "transcription",
                "language": language,
                "provider": request_provider,
                "model": local_model,
            },
        )
        text = getattr(getattr(result, "text", None), "content", None)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=_plugin_exception_status(e), detail=f"Audio transcription failed: {e}") from e

    return {"text": str(text or "").strip()}


@provider_router.post("/{provider}/v1/audio/transcriptions")
async def provider_audio_transcriptions(
    request: Request,
    provider: str = FastAPIPath(
        ...,
        description="Audio provider route prefix, e.g. `openai`, `openrouter`, `portkey`, `openai-compatible`, or a local AbstractVoice engine.",
    ),
    file: UploadFile = File(..., description="Audio file to transcribe. Common formats include mp3, mp4, mpeg, mpga, m4a, wav, and webm."),
    model: Optional[str] = Form(
        default=None,
        description=(
            "Optional unprefixed model id for the provider route. If a provider/model id is supplied, "
            "the route provider takes precedence."
        ),
        examples=["gpt-4o-mini-transcribe"],
    ),
    provider_hint: Optional[str] = Form(
        default=None,
        alias="provider",
        description="Optional provider/engine hint. The route provider takes precedence when both are set.",
        examples=["openai"],
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
    """Provider-scoped OpenAI-compatible STT endpoint."""
    return await audio_transcriptions(
        request=request,
        file=file,
        model=model,
        provider=provider_hint or provider,
        language=language,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        format=format,
        base_url=base_url,
    )


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


def _audio_dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _audio_voice_record_provider(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    params = item.get("params") if isinstance(item.get("params"), dict) else {}
    tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
    return str(
        item.get("provider")
        or item.get("engine_id")
        or item.get("engine")
        or tags.get("provider")
        or tags.get("engine_id")
        or tags.get("engine")
        or params.get("provider")
        or params.get("engine_id")
        or params.get("engine")
        or ""
    ).strip()


def _audio_voice_record_model(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    params = item.get("params") if isinstance(item.get("params"), dict) else {}
    return str(
        item.get("model")
        or item.get("model_id")
        or params.get("model")
        or params.get("model_id")
        or params.get("model_filename")
        or ""
    ).strip()


def _audio_provider_filtered_values(response: Dict[str, Any], provider: str, *keys: str) -> list[str]:
    wanted = str(provider or "").strip().lower()
    if not wanted:
        return []
    for key in keys:
        mapping = response.get(key)
        if not isinstance(mapping, dict):
            continue
        for provider_key, values in mapping.items():
            if str(provider_key or "").strip().lower() != wanted or not isinstance(values, list):
                continue
            return _audio_dedupe_strings([str(item).strip() for item in values if isinstance(item, str) and str(item).strip()])
    return []


def _filter_audio_voice_catalog_response(
    response: Dict[str, Any],
    *,
    provider: Optional[str],
    model: Optional[str],
    providers_only: bool,
) -> Dict[str, Any]:
    wanted_provider = str(provider or "").strip().lower()
    wanted_model = str(model or "").strip().lower()
    out = dict(response)

    record_values: list[Any] = []
    for key in ("profiles", "voices", "cloned_voices"):
        values = response.get(key)
        if isinstance(values, list):
            record_values.extend(values)

    derived_providers = [
        _audio_voice_record_provider(item)
        for item in record_values
        if isinstance(item, dict) and _audio_voice_record_provider(item)
    ]
    map_providers: list[str] = []
    for key in ("tts_models_by_provider", "tts_voices_by_provider", "tts_profiles_by_provider"):
        mapping = response.get(key)
        if isinstance(mapping, dict):
            map_providers.extend(str(k).strip() for k in mapping.keys() if str(k).strip())
    providers = _audio_dedupe_strings(
        [
            str(item).strip()
            for item in list(response.get("tts_providers") or response.get("providers") or []) + derived_providers + map_providers
            if isinstance(item, str) and str(item).strip()
        ]
    )
    stt_providers = _audio_dedupe_strings(
        [str(item).strip() for item in (response.get("stt_providers") or []) if isinstance(item, str) and str(item).strip()]
    )
    if wanted_provider:
        providers = [item for item in providers if item.lower() == wanted_provider]
        stt_providers = [item for item in stt_providers if item.lower() == wanted_provider]

    def keep_record(item: Any) -> bool:
        if not isinstance(item, dict):
            return not wanted_provider and not wanted_model
        item_provider = _audio_voice_record_provider(item).lower()
        item_model = _audio_voice_record_model(item).lower()
        if wanted_provider and item_provider and item_provider != wanted_provider:
            return False
        if wanted_model and item_model and item_model != wanted_model:
            return False
        return True

    if providers_only:
        out["providers"] = providers
        out["tts_providers"] = providers
        out["stt_providers"] = stt_providers
        out["profiles"] = []
        out["voices"] = []
        out["cloned_voices"] = []
        out["available"] = bool(providers or stt_providers)
        return out

    for key in ("profiles", "voices", "cloned_voices"):
        values = response.get(key)
        if isinstance(values, list):
            out[key] = [item for item in values if keep_record(item)]
    for key in ("tts_models_by_provider", "stt_models_by_provider", "tts_voices_by_provider", "tts_profiles_by_provider"):
        mapping = response.get(key)
        if isinstance(mapping, dict) and wanted_provider:
            out[key] = {k: v for k, v in mapping.items() if str(k).strip().lower() == wanted_provider}
    if wanted_provider:
        tts_models = _audio_provider_filtered_values(response, wanted_provider, "tts_models_by_provider", "models_by_provider")
        stt_models = _audio_provider_filtered_values(response, wanted_provider, "stt_models_by_provider")
        if not tts_models and str(response.get("active_tts_provider") or response.get("provider") or "").strip().lower() == wanted_provider:
            raw_tts = response.get("tts_models") or response.get("models")
            tts_models = (
                _audio_dedupe_strings([str(item).strip() for item in raw_tts if isinstance(item, str) and str(item).strip()])
                if isinstance(raw_tts, list)
                else []
            )
        if not stt_models and str(response.get("active_stt_provider") or "").strip().lower() == wanted_provider:
            raw_stt = response.get("stt_models")
            stt_models = (
                _audio_dedupe_strings([str(item).strip() for item in raw_stt if isinstance(item, str) and str(item).strip()])
                if isinstance(raw_stt, list)
                else []
            )
        out["models"] = tts_models
        out["tts_models"] = tts_models
        if "stt_models" in out or stt_models:
            out["stt_models"] = stt_models

    out["providers"] = providers or out.get("providers") or []
    out["tts_providers"] = providers or out.get("tts_providers") or []
    out["stt_providers"] = stt_providers or out.get("stt_providers") or []
    return out


def _audio_list_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _audio_dedupe_strings([str(item).strip() for item in value if isinstance(item, str) and str(item).strip()])


def _audio_mapping_strings(value: Any) -> Dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, list[str]] = {}
    for key, items in value.items():
        provider = str(key or "").strip()
        if not provider or not isinstance(items, list):
            continue
        values = _audio_list_strings(items)
        if values:
            out[provider] = values
    return out


def _audio_provider_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _audio_provider_matches(left: Any, right: Any) -> bool:
    return bool(_audio_provider_key(left)) and _audio_provider_key(left) == _audio_provider_key(right)


def _audio_catalog_available_provider_ids(catalog: Dict[str, Any], kind: str) -> list[str]:
    available = catalog.get("available_providers")
    if isinstance(available, dict):
        values = available.get(kind)
        if isinstance(values, list):
            return _audio_list_strings(values)
    key = {
        "tts": "available_tts_providers",
        "stt": "available_stt_providers",
        "cloning": "available_cloning_providers",
    }.get(kind)
    return _audio_list_strings(catalog.get(key)) if key else []


def _audio_provider_models(models_by_provider: Dict[str, list[str]]) -> list[Dict[str, str]]:
    out: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for provider, models in models_by_provider.items():
        provider_s = str(provider or "").strip()
        if not provider_s:
            continue
        for model in models:
            model_s = str(model or "").strip()
            if not model_s:
                continue
            routed = model_s if model_s.startswith(f"{provider_s}/") else f"{provider_s}/{model_s}"
            key = (provider_s.lower(), routed.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "id": model_s,
                    "model": routed,
                    "provider": provider_s,
                    "routed_model": routed,
                    "object": "model",
                    "owned_by": provider_s,
                }
            )
    return out


def _audio_models_by_provider(
    *,
    catalog: Dict[str, Any],
    kind: str,
    providers: list[str],
    models: list[str],
) -> Dict[str, list[str]]:
    if kind == "stt":
        mapping = _audio_mapping_strings(catalog.get("stt_models_by_provider"))
    elif kind == "cloning":
        mapping = _audio_mapping_strings(catalog.get("cloning_models_by_provider"))
        if not mapping:
            compat = catalog.get("compatibility_catalog")
            providers_by_kind = compat.get("providers") if isinstance(compat, dict) else None
            cloning = providers_by_kind.get("cloning") if isinstance(providers_by_kind, dict) else None
            if isinstance(cloning, dict):
                derived: Dict[str, list[str]] = {}
                for provider_id, entry in cloning.items():
                    provider_name = str(provider_id or "").strip()
                    if not provider_name or not isinstance(entry, dict):
                        continue
                    entries = entry.get("models")
                    if not isinstance(entries, dict):
                        continue
                    provider_models = _audio_dedupe_strings(
                        [str(model_name).strip() for model_name in entries if str(model_name or "").strip() and str(model_name).strip() != "*"]
                    )
                    if provider_models:
                        derived[provider_name] = provider_models
                mapping = derived
    else:
        mapping = _audio_mapping_strings(catalog.get("tts_models_by_provider"))
    if not mapping and len(providers) == 1 and models:
        mapping[providers[0]] = list(models)
    if models and mapping:
        allowed = {str(model).strip().lower() for model in models if str(model).strip()}
        mapping = {
            provider: [model for model in provider_models if str(model).strip().lower() in allowed]
            for provider, provider_models in mapping.items()
        }
        mapping = {provider: provider_models for provider, provider_models in mapping.items() if provider_models}
    return mapping


def _audio_model_catalog_payload(
    *,
    core: Any,
    kind: str,
    catalog: Dict[str, Any],
    models: list[str],
) -> Dict[str, Any]:
    if kind == "stt":
        provider_key = "stt_providers"
        active_provider = catalog.get("active_stt_provider")
    elif kind == "cloning":
        provider_key = "cloning_providers"
        active_provider = catalog.get("active_cloning_provider")
    else:
        provider_key = "tts_providers"
        active_provider = catalog.get("active_tts_provider") or catalog.get("engine_id")

    providers = _audio_list_strings(catalog.get(provider_key))
    available_providers = _audio_catalog_available_provider_ids(
        catalog,
        "stt" if kind == "stt" else ("cloning" if kind == "cloning" else "tts"),
    )
    if available_providers:
        providers = list(available_providers)
    models_by_provider = _audio_models_by_provider(catalog=catalog, kind=kind, providers=providers, models=models)
    if available_providers and models_by_provider:
        allowed = {provider.lower() for provider in available_providers}
        models_by_provider = {k: v for k, v in models_by_provider.items() if k.lower() in allowed}
        allowed_models = {str(model).lower() for values in models_by_provider.values() for model in values}
        if allowed_models:
            models = [model for model in models if str(model).lower() in allowed_models]
    if models_by_provider:
        for provider in models_by_provider:
            if provider not in providers:
                providers.append(provider)

    payload: Dict[str, Any] = {
        "available": True,
        "source": "abstractvoice",
        "stale": False,
        "error": None,
        "backend_id": getattr(core.voice, "backend_id", None),
        "providers": providers,
        "available_providers": available_providers,
        "active_provider": active_provider,
        "models": models,
        "models_by_provider": models_by_provider,
        "provider_models": _audio_provider_models(models_by_provider),
    }
    if kind == "tts":
        payload["controls"] = {
            "speed": {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0},
            "quality_preset": {"supported": True, "values": ["low", "standard", "high"], "default": "standard"},
            "instructions": {"supported": True},
            "profile": {"supported": True},
            "voice_clone": {"supported": True},
        }
    return payload


def _audio_provider_catalog_payload(*, core: Any, catalog: Dict[str, Any], kind: str) -> Dict[str, Any]:
    if kind == "stt":
        providers = _audio_list_strings(catalog.get("stt_providers"))
        active_provider = catalog.get("active_stt_provider")
        available_providers = _audio_catalog_available_provider_ids(catalog, "stt")
    elif kind == "cloning":
        providers = _audio_list_strings(catalog.get("cloning_providers"))
        available_providers = _audio_catalog_available_provider_ids(catalog, "cloning")
        if not providers:
            providers = list(available_providers)
        active_provider = catalog.get("active_cloning_provider")
        if not active_provider:
            available = catalog.get("available_providers")
            if isinstance(available, dict):
                active_provider = available.get("active_cloning_provider")
    else:
        providers = _audio_list_strings(catalog.get("tts_providers") or catalog.get("providers"))
        active_provider = catalog.get("active_tts_provider") or catalog.get("engine_id")
        available_providers = _audio_catalog_available_provider_ids(catalog, "tts")
    if available_providers:
        providers = list(available_providers)

    return {
        "available": True,
        "source": "abstractvoice",
        "stale": False,
        "error": None,
        "backend_id": getattr(core.voice, "backend_id", None),
        "providers": providers,
        "available_providers": available_providers,
        "active_provider": active_provider,
    }


def _audio_voice_catalog_or_available(core: Any) -> Dict[str, Any]:
    try:
        if hasattr(core.voice, "voice_catalog"):
            catalog = core.voice.voice_catalog()
            if isinstance(catalog, dict):
                return dict(catalog)
    except Exception as catalog_error:
        try:
            if hasattr(core.voice, "available_providers"):
                providers = core.voice.available_providers()
                if isinstance(providers, dict):
                    return {
                        "available_providers": dict(providers),
                        "tts_providers": _audio_list_strings(providers.get("tts_providers") or providers.get("tts")),
                        "stt_providers": _audio_list_strings(providers.get("stt_providers") or providers.get("stt")),
                        "cloning_providers": _audio_list_strings(providers.get("cloning_providers") or providers.get("cloning")),
                        "active_tts_provider": providers.get("active_tts_provider"),
                        "active_stt_provider": providers.get("active_stt_provider"),
                        "active_cloning_provider": providers.get("active_cloning_provider"),
                    }
        except Exception:
            pass
        raise catalog_error
    return {}


def _audio_voice_available_providers(core: Any) -> Dict[str, Any]:
    """Return lightweight availability from the capability plugin (best-effort)."""
    try:
        if hasattr(core.voice, "available_providers"):
            payload = core.voice.available_providers()
            if isinstance(payload, dict):
                return dict(payload)
    except Exception:
        return {}
    return {}


@router.get("/audio/voices")
async def audio_voices(
    request: Request,
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for catalog discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
    provider: Optional[str] = Query(default=None, description="Optional TTS provider/engine filter."),
    model: Optional[str] = Query(default=None, description="Optional TTS model/language filter."),
    providers_only: bool = Query(default=False, description="Return provider names without model/voice lists."),
) -> Dict[str, Any]:
    """Discover configured TTS voices/profiles through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        catalog = _audio_voice_catalog_or_available(core)
        out = dict(catalog)
        out.setdefault("kind", "tts")
        out.setdefault(
            "controls",
            {
                "speed": {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0},
                "quality_preset": {"supported": True, "values": ["low", "standard", "high"], "default": "standard"},
                "instructions": {"supported": True},
                "profile": {"supported": True},
                "voice_clone": {"supported": True},
            },
        )
        out.update(
            {
                "available": True,
                "source": "abstractvoice",
                "stale": False,
                "error": None,
                "backend_id": getattr(core.voice, "backend_id", None),
            }
        )
        return _filter_audio_voice_catalog_response(out, provider=provider, model=model, providers_only=providers_only)
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


@router.get("/audio/speech/providers")
async def audio_speech_providers(
    request: Request,
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for TTS provider discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """Discover configured TTS providers through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        availability = _audio_voice_available_providers(core)
        if availability:
            catalog = {
                "tts_providers": availability.get("known_tts_providers") or availability.get("tts_providers") or availability.get("tts"),
                "available_providers": availability,
                "available_tts_providers": availability.get("tts"),
                "active_tts_provider": availability.get("active_tts_provider"),
            }
            return _audio_provider_catalog_payload(core=core, catalog=catalog, kind="tts")
        return _audio_provider_catalog_payload(core=core, catalog=_audio_voice_catalog_or_available(core), kind="tts")
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


@router.get("/audio/speech/models")
async def audio_speech_models(
    request: Request,
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. openai, openai-compatible, elevenlabs, supertonic).",
        examples=["openai"],
    ),
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for TTS model discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """Discover configured TTS model ids through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        catalog = _audio_voice_catalog_or_available(core)
        try:
            models = _audio_list_strings(list(core.voice.list_tts_models() or []))
        except Exception:
            models = _audio_list_strings(catalog.get("tts_models") or catalog.get("models"))
        if not models:
            models = _audio_list_strings(catalog.get("tts_models") or catalog.get("models"))
        payload = _audio_model_catalog_payload(core=core, kind="tts", catalog=catalog, models=models)
        provider_norm = str(provider or "").strip()
        if provider_norm and isinstance(payload.get("models_by_provider"), dict):
            provider_lc = provider_norm.lower()
            models_by_provider = {
                k: v for k, v in payload["models_by_provider"].items() if str(k).strip().lower() == provider_lc
            }
            payload["models_by_provider"] = models_by_provider
            payload["providers"] = list(models_by_provider) or [provider_norm]
            payload["available_providers"] = [p for p in payload.get("available_providers") or [] if str(p).strip().lower() == provider_lc] or payload["providers"]
            models_filtered = [m for values in models_by_provider.values() for m in values]
            payload["models"] = models_filtered
            payload["provider_models"] = _audio_provider_models(models_by_provider)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


@router.get("/audio/transcriptions/providers")
async def audio_transcription_providers(
    request: Request,
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for STT provider discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """Discover configured STT providers through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        availability = _audio_voice_available_providers(core)
        if availability:
            catalog = {
                "stt_providers": availability.get("known_stt_providers") or availability.get("stt_providers") or availability.get("stt"),
                "available_providers": availability,
                "available_stt_providers": availability.get("stt"),
                "active_stt_provider": availability.get("active_stt_provider"),
            }
            return _audio_provider_catalog_payload(core=core, catalog=catalog, kind="stt")
        return _audio_provider_catalog_payload(core=core, catalog=_audio_voice_catalog_or_available(core), kind="stt")
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


@router.get("/audio/transcriptions/models")
async def audio_transcription_models(
    request: Request,
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. openai, openai-compatible, whisper, deepgram).",
        examples=["openai"],
    ),
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for STT model discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """Discover configured STT model ids through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        catalog = _audio_voice_catalog_or_available(core)
        try:
            models = _audio_list_strings(list(core.voice.list_stt_models() or []))
        except Exception:
            models = _audio_list_strings(catalog.get("stt_models"))
        if not models:
            models = _audio_list_strings(catalog.get("stt_models"))
        payload = _audio_model_catalog_payload(core=core, kind="stt", catalog=catalog, models=models)
        provider_norm = str(provider or "").strip()
        if provider_norm and isinstance(payload.get("models_by_provider"), dict):
            provider_lc = provider_norm.lower()
            models_by_provider = {
                k: v for k, v in payload["models_by_provider"].items() if str(k).strip().lower() == provider_lc
            }
            payload["models_by_provider"] = models_by_provider
            payload["providers"] = list(models_by_provider) or [provider_norm]
            payload["available_providers"] = [p for p in payload.get("available_providers") or [] if str(p).strip().lower() == provider_lc] or payload["providers"]
            models_filtered = [m for values in models_by_provider.values() for m in values]
            payload["models"] = models_filtered
            payload["provider_models"] = _audio_provider_models(models_by_provider)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


@router.get("/voice/clone/providers")
async def voice_clone_providers(
    request: Request,
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for voice clone provider discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """Discover voice cloning providers through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        availability = _audio_voice_available_providers(core)
        if availability:
            catalog = {
                "cloning_providers": availability.get("known_cloning_providers") or availability.get("cloning_providers") or availability.get("cloning"),
                "available_providers": availability,
                "available_cloning_providers": availability.get("cloning"),
                "active_cloning_provider": availability.get("active_cloning_provider"),
            }
            return _audio_provider_catalog_payload(core=core, catalog=catalog, kind="cloning")
        return _audio_provider_catalog_payload(core=core, catalog=_audio_voice_catalog_or_available(core), kind="cloning")
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


@router.get("/voice/clone/models")
async def voice_clone_models(
    request: Request,
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. openai-compatible, openai, omnivoice, f5-tts, chroma).",
        examples=["openai-compatible"],
    ),
    base_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional OpenAI-compatible voice endpoint override for voice cloning model discovery. "
            "Loopback is allowed by default; non-loopback URLs require "
            "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST."
        ),
    ),
    api_key: Optional[str] = Query(
        default=None,
        description=(
            "Optional upstream provider API key override for discovery. Prefer "
            "`X-AbstractCore-Provider-API-Key`; this query parameter is supported for tooling/Swagger UI "
            "convenience and is redacted from server logs."
        ),
    ),
) -> Dict[str, Any]:
    """Discover voice cloning model ids (if any) through the AbstractVoice plugin boundary."""
    try:
        core = _audio_catalog_core(request, base_url=base_url, api_key=api_key)
        provider_norm = _optional_text(provider)
        catalog = _audio_voice_catalog_or_available(core)
        models = _audio_list_strings(core.voice.list_cloning_models(provider=provider_norm))
        payload = _audio_model_catalog_payload(core=core, kind="cloning", catalog=catalog, models=models)
        if provider_norm:
            filtered_mapping = {
                provider_id: provider_models
                for provider_id, provider_models in dict(payload.get("models_by_provider") or {}).items()
                if _audio_provider_matches(provider_id, provider_norm)
            }
            payload["models_by_provider"] = filtered_mapping
            payload["provider_models"] = _audio_provider_models(filtered_mapping)
            payload["models"] = _audio_dedupe_strings(
                [model_name for provider_models in filtered_mapping.values() for model_name in provider_models]
            ) or models
            payload["providers"] = list(filtered_mapping) or [
                provider_id
                for provider_id in payload.get("providers") or []
                if _audio_provider_matches(provider_id, provider_norm)
            ] or [provider_norm]
            payload["available_providers"] = [
                provider_id
                for provider_id in payload.get("available_providers") or []
                if _audio_provider_matches(provider_id, provider_norm)
            ] or payload["providers"]
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise _audio_catalog_error(e) from e


_AUDIO_SPEECH_RESPONSES = {
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
}


async def _audio_speech_impl(
    request: Request,
    payload: AudioSpeechRequest,
    *,
    path_tts_engine: Optional[str] = None,
):
    data = _model_payload(payload)
    path_provider = _request_provider_value(path_tts_engine)

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

    request_provider = path_provider or _request_provider_value(data.get("provider"))
    remote_provider_payload = data.get("provider") if isinstance(data.get("provider"), dict) else None
    local_model = None
    model = _optional_text(data.get("model"))
    if path_provider and model and "/" in model:
        _model_provider, model_tail = model.split("/", 1)
        model = model_tail.strip() or None
    if model and _is_local_audio_model(model):
        model = None
    if path_provider and path_provider in _REMOTE_AUDIO_PROVIDER_ALIASES and model and "/" not in model:
        model = f"{path_provider}/{model}"
    elif request_provider and request_provider not in _REMOTE_AUDIO_PROVIDER_ALIASES:
        local_model = model
        model = None
    elif request_provider in {"openai", "openai-compatible"} and model and "/" not in model:
        model = f"{request_provider}/{model}"
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
                provider=remote_provider_payload,
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
                "quality_preset": _optional_text(data.get("quality_preset") or data.get("quality")),
                "instructions": _optional_text(data.get("instructions")),
                "profile": _optional_text(data.get("profile")),
                "provider": request_provider,
                "model": local_model,
            },
        )
        voice_items = getattr(result, "outputs", {}).get("voice", [])
        audio_item = voice_items[0] if voice_items else None
        audio = getattr(audio_item, "data", None)
        content_type = getattr(audio_item, "content_type", None)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=_plugin_exception_status(e), detail=f"Audio synthesis failed: {e}") from e

    if not isinstance(audio, (bytes, bytearray)):
        raise HTTPException(
            status_code=500,
            detail="TTS backend returned an unexpected type (expected raw bytes).",
        )

    return _audio_response(bytes(audio), media_type=str(content_type or f"audio/{fmt}"))


@router.post(
    "/audio/speech",
    response_class=Response,
    responses=_AUDIO_SPEECH_RESPONSES,
)
async def audio_speech(request: Request, payload: AudioSpeechRequest = Body(...)):
    """OpenAI-compatible TTS endpoint (json with input text)."""
    return await _audio_speech_impl(request, payload)


@provider_router.post(
    "/{provider}/v1/audio/speech",
    response_class=Response,
    responses=_AUDIO_SPEECH_RESPONSES,
)
async def provider_audio_speech(
    request: Request,
    payload: AudioSpeechRequest = Body(...),
    provider: str = FastAPIPath(
        ...,
        description="TTS engine/provider route prefix, e.g. `piper`, `supertonic`, `openai`, or `openai-compatible`.",
    ),
):
    """Provider-scoped TTS endpoint. The route engine takes precedence over body routing fields."""
    return await _audio_speech_impl(request, payload, path_tts_engine=provider)


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
    provider: Optional[str] = Form(
        default=None,
        description=(
            "Optional provider/engine hint. Use this for local clone routing such as "
            "`omnivoice` or `f5-tts`, or remote routing such as `openai-compatible`."
        ),
        examples=["omnivoice"],
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

    request_provider, model, local_model = _resolve_audio_request_routing(model=model, provider=provider)
    if request_provider in _REMOTE_AUDIO_PROVIDER_ALIASES and not model:
        model = f"{request_provider}/default"

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
                "provider": request_provider,
                "model": local_model,
                "tts_model": local_model,
                "cloning_engine": request_provider or os.getenv("ABSTRACTVOICE_CLONING_ENGINE") or None,
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
        raise HTTPException(status_code=_plugin_exception_status(e), detail=f"Voice clone failed: {e}") from e


@provider_router.post("/{provider}/v1/voice/clone")
async def provider_voice_clone(
    request: Request,
    provider: str = FastAPIPath(
        ...,
        description="Voice-clone provider route prefix, e.g. `openai-compatible` or `openai`.",
    ),
    file: UploadFile = File(..., description="Reference voice audio file, commonly WAV/FLAC/OGG/MP3/M4A/WEBM."),
    model: Optional[str] = Form(
        default=None,
        description="Optional unprefixed model id for the provider route. Defaults to `default` when omitted.",
        examples=["default"],
    ),
    provider_hint: Optional[str] = Form(
        default=None,
        alias="provider",
        description="Optional provider/engine hint. The route provider takes precedence when both are set.",
        examples=["openai-compatible"],
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
        description="Optional provider-specific clone path.",
        examples=["/voice/clone"],
    ),
    file_field: Optional[str] = Form(
        default=None,
        description="Optional provider-specific multipart file field.",
        examples=["file"],
    ),
    consent: Optional[str] = Form(default=None, description="Optional provider-specific consent id for custom voice creation.", examples=["consent_123"]),
):
    """Provider-scoped voice-clone endpoint."""
    provider_s = _request_provider_value(provider_hint or provider)
    model_s = _optional_text(model)
    if provider_s in _REMOTE_AUDIO_PROVIDER_ALIASES and not model_s:
        model_s = "default"
    return await voice_clone(
        request=request,
        file=file,
        model=model_s,
        provider=provider_s,
        name=name,
        reference_text=reference_text,
        validate=validate,
        base_url=base_url,
        clone_path=clone_path,
        file_field=file_field,
        consent=consent,
    )


_AUDIO_MUSIC_RESPONSES = {
    200: {
        "description": "Generated music/audio bytes.",
        "content": {
            "audio/wav": _AUDIO_BINARY_RESPONSE,
            "audio/mpeg": _AUDIO_BINARY_RESPONSE,
            "audio/mp3": _AUDIO_BINARY_RESPONSE,
            "audio/flac": _AUDIO_BINARY_RESPONSE,
            "application/octet-stream": _AUDIO_BINARY_RESPONSE,
        },
    }
}


async def _audio_music_impl(payload: AudioMusicRequest, *, path_provider: Optional[str] = None):
    """Text-to-music endpoint (extension; no official OpenAI equivalent).

    Delegates to the `music` capability plugin (typically `abstractmusic`).
    Returns raw audio bytes.
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
    if fmt not in {"wav", "mp3", "flac"}:
        raise HTTPException(status_code=422, detail="format must be one of: wav, mp3, flac.")

    provider_hint = _music_provider_selector(path_provider, data.get("provider"))
    output_spec = {
        k: v
        for k, v in data.items()
        if k
        not in {
            "prompt",
            "input",
            "text",
            "lyrics",
            "format",
            "response_format",
            "task",
        }
        and v is not None
    }
    output_spec.update(
        {
            "modality": "music",
            "task": str(data.get("task") or "music_generation"),
            "lyrics": lyrics,
            "format": fmt,
        }
    )
    if provider_hint:
        output_spec["provider"] = provider_hint
    if data.get("duration_s") is not None:
        output_spec["duration_s"] = _optional_float(data.get("duration_s"), field="duration_s")
    if data.get("guidance_scale") is not None:
        output_spec["guidance_scale"] = _optional_float(data.get("guidance_scale"), field="guidance_scale")
    if data.get("num_inference_steps") is not None:
        output_spec["num_inference_steps"] = _optional_int(data.get("num_inference_steps"), field="num_inference_steps")
    if data.get("seed") is not None:
        output_spec["seed"] = _optional_int(data.get("seed"), field="seed")

    core = _music_capability_core_for_request(data, path_provider=path_provider)
    try:
        result = core.generate(text=str(prompt), output=output_spec)
        music_items = getattr(result, "outputs", {}).get("music", [])
        music_item = music_items[0] if music_items else None
        audio = getattr(music_item, "data", None)
        content_type = getattr(music_item, "content_type", None)
    except CapabilityUnavailableError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=_plugin_exception_status(e), detail=f"Music generation failed: {e}") from e

    if not isinstance(audio, (bytes, bytearray)):
        raise HTTPException(status_code=500, detail="Music backend returned an unexpected type (expected raw bytes).")

    return Response(content=bytes(audio), media_type=str(content_type or "audio/wav"))


@router.post(
    "/audio/music",
    response_class=Response,
    responses=_AUDIO_MUSIC_RESPONSES,
)
async def audio_music(payload: AudioMusicRequest = Body(...)):
    return await _audio_music_impl(payload)


@provider_router.post(
    "/{provider}/v1/audio/music",
    response_class=Response,
    responses=_AUDIO_MUSIC_RESPONSES,
)
async def provider_audio_music(
    payload: AudioMusicRequest = Body(...),
    provider: str = FastAPIPath(
        ...,
        description="Music backend/provider route prefix, e.g. `acestep`, `diffusers`, or a provider catalog id.",
    ),
):
    return await _audio_music_impl(payload, path_provider=provider)
