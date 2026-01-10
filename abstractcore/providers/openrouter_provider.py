"""
OpenRouter provider (OpenAI-compatible API).

OpenRouter exposes an OpenAI-compatible API at `https://openrouter.ai/api/v1`.
This provider subclasses `OpenAICompatibleProvider` and adds:
- API key support via `OPENROUTER_API_KEY` (or AbstractCore config fallback)
- Optional OpenRouter metadata headers (`HTTP-Referer`, `X-Title`)
"""

import os
from typing import Optional, Dict

from .openai_compatible_provider import OpenAICompatibleProvider


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
        **kwargs,
    ):
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

