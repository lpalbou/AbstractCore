"""
LM Studio provider implementation (OpenAI-compatible API).

LM Studio exposes an OpenAI-compatible server (by default at `http://localhost:1234/v1`).
This provider is a thin wrapper around `OpenAICompatibleProvider` with LM Studio defaults.
"""

from typing import Optional, Any

from .openai_compatible_provider import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio provider using OpenAI-compatible API."""

    PROVIDER_ID = "lmstudio"
    PROVIDER_DISPLAY_NAME = "LMStudio"
    BASE_URL_ENV_VAR = "LMSTUDIO_BASE_URL"
    API_KEY_ENV_VAR = None
    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(
        self,
        model: str = "local-model",
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        # ADR-0027: Local LM Studio calls should default to no client-side timeout.
        #
        # We intentionally treat "timeout omitted" as "unlimited" for this provider, rather
        # than inheriting the global `abstractcore` default timeout (which may be tuned for
        # remote providers). Operators can still override via:
        # - explicit `timeout=...` when constructing the provider, or
        # - runtime provider config (ConfigurationManager.configure_provider('lmstudio', timeout=...)).
        if "timeout" in kwargs:
            timeout = kwargs.pop("timeout")

        super().__init__(model=model, base_url=base_url, timeout=timeout, **kwargs)
