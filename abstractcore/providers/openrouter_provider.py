"""
OpenRouter provider (OpenAI-compatible API).

OpenRouter exposes an OpenAI-compatible API at `https://openrouter.ai/api/v1`.
This provider subclasses `OpenAICompatibleProvider` and adds:
- API key support via `OPENROUTER_API_KEY` (or AbstractCore config fallback)
- Optional OpenRouter metadata headers (`HTTP-Referer`, `X-Title`)
"""

import base64
import os
from typing import Any, Optional, Dict, Tuple

from .openai_compatible_provider import OpenAICompatibleProvider
from ..exceptions import AuthenticationError, InvalidRequestError, ModelNotFoundError, ProviderAPIError, RateLimitError


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider using OpenAI-compatible API."""

    PROVIDER_ID = "openrouter"
    PROVIDER_DISPLAY_NAME = "OpenRouter"
    BASE_URL_ENV_VAR = "OPENROUTER_BASE_URL"
    API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        validate_model: bool = True,
        **kwargs,
    ):
        self._validate_model_on_init = bool(validate_model)
        super().__init__(model=model, base_url=base_url, api_key=api_key, **kwargs)

        if not self._has_api_key():
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY or configure via "
                "`abstractcore --set-api-key openrouter <key>`."
            )

    def _has_api_key(self) -> bool:
        if self.api_key is None:
            return False
        key = str(self.api_key).strip()
        if not key:
            return False
        return key.upper() != "EMPTY"

    def _get_api_key_from_config(self) -> Optional[str]:
        try:
            from ..config.manager import get_config_manager

            cfg = get_config_manager()
            return getattr(cfg.config.api_keys, "openrouter", None)
        except Exception:
            return None

    def _validate_model(self):
        if not getattr(self, "_validate_model_on_init", True):
            return
        # Avoid unauthenticated network calls on init; OpenRouter generally requires a key.
        if not self._has_api_key():
            return
        return super()._validate_model()

    def _get_headers(self) -> Dict[str, str]:
        headers = super()._get_headers()

        # OpenRouter recommends sending these for better analytics / abuse prevention.
        site_url = os.getenv("OPENROUTER_SITE_URL")
        if isinstance(site_url, str) and site_url.strip():
            headers["HTTP-Referer"] = site_url.strip()

        app_name = os.getenv("OPENROUTER_APP_NAME")
        if isinstance(app_name, str) and app_name.strip():
            headers["X-Title"] = app_name.strip()

        return headers

    @staticmethod
    def _infer_audio_format(
        *,
        filename: str,
        content_type: str,
        audio_format: Optional[str] = None,
    ) -> str:
        if isinstance(audio_format, str) and audio_format.strip():
            return audio_format.strip().lower().lstrip(".")

        name = str(filename or "").strip().lower()
        if "." in name:
            suffix = name.rsplit(".", 1)[-1].strip()
            if suffix:
                return suffix

        ctype = str(content_type or "").strip().lower()
        if "/" in ctype:
            subtype = ctype.split("/", 1)[1].split(";", 1)[0].strip()
            if subtype in {"mpeg", "mpga"}:
                return "mp3"
            if subtype:
                return subtype

        return "wav"

    def transcribe_audio(
        self,
        audio: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "application/octet-stream",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        temperature: Optional[float] = None,
        audio_format: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bytes, str]:
        """Transcribe audio through OpenRouter's JSON/base64 STT endpoint."""
        _ = prompt, response_format
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "input_audio": {
                    "data": base64.b64encode(bytes(audio)).decode("ascii"),
                    "format": self._infer_audio_format(
                        filename=filename,
                        content_type=content_type,
                        audio_format=audio_format,
                    ),
                },
            }
            if language:
                payload["language"] = language
            if temperature is not None:
                payload["temperature"] = temperature
            provider_options = kwargs.get("provider")
            if isinstance(provider_options, dict):
                payload["provider"] = provider_options

            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                json=payload,
                headers=self._get_headers(),
            )
            self._raise_for_status(response, request_url=f"{self.base_url}/audio/transcriptions")
            return response.content, response.headers.get("content-type", "application/json")
        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio through OpenRouter: {e}")
            raise ProviderAPIError(f"OpenRouter audio transcription error: {str(e)}")
