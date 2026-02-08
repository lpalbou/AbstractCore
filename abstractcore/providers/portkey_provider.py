"""
Portkey provider (OpenAI-compatible gateway).

Portkey exposes an OpenAI-compatible gateway (default: ``https://api.portkey.ai/v1``)
with **header-driven routing**.  This provider subclasses
`OpenAICompatibleProvider` and adds Portkey-specific header handling.

Routing modes (mutually exclusive — first match wins)
-----------------------------------------------------
1. **Config mode** — ``x-portkey-config``
   A Portkey config encapsulates all routing (provider selection, virtual keys,
   fallbacks, load-balancing).  When a ``config_id`` is supplied, no other
   routing headers (virtual key, provider) are sent.

2. **Virtual-key mode** — ``x-portkey-virtual-key``
   Routes to a single provider using a pre-configured virtual key that
   encapsulates provider + credentials.

3. **Provider-direct mode** — ``x-portkey-provider`` + ``Authorization``
   Routes directly to a named provider using the provider's own API key.

In every mode the Portkey gateway key is sent via **both**
``x-portkey-api-key`` *and* ``Authorization: Bearer`` so it works with
standard Portkey as well as enterprise / custom gateways (e.g. Roche Galileo)
that expect the OpenAI-compatible ``Authorization`` header. In provider-direct
mode, ``Authorization`` is intentionally overwritten with the *upstream*
provider key.
"""

import os
from urllib.parse import urlparse
from typing import Any, Dict, Optional

from .openai_compatible_provider import OpenAICompatibleProvider
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


class PortkeyProvider(OpenAICompatibleProvider):
    """Portkey provider using an OpenAI-compatible gateway.

    Supports all LLM providers routed through Portkey: OpenAI, Anthropic,
    Google Gemini, Grok / xAI, AWS Bedrock, Azure, Mistral, Cohere,
    Ollama, and any other provider configured on the Portkey dashboard.
    """

    PROVIDER_ID = "portkey"
    PROVIDER_DISPLAY_NAME = "Portkey"
    BASE_URL_ENV_VAR = "PORTKEY_BASE_URL"
    API_KEY_ENV_VAR = "PORTKEY_API_KEY"
    DEFAULT_BASE_URL = "https://api.portkey.ai/v1"

    # ── Routing modes (ordered by precedence) ──────────────────────────
    _MODE_CONFIG = "config"
    _MODE_VIRTUAL_KEY = "virtual_key"
    _MODE_PROVIDER_DIRECT = "provider_direct"

    # Generation parameters that the parent injects with defaults but
    # that must NOT be sent to a Portkey backend unless the user asked.
    # (GPT-5 / o-series reject temperature ≠ 1, some reject top_p, etc.)
    _OPTIONAL_GEN_PARAMS = frozenset({
        "temperature",
        "top_p",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    })

    def __init__(
        self,
        model: str = "default",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config_id: Optional[str] = None,
        portkey_provider: Optional[str] = None,
        virtual_key: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        portkey_headers: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # Track which generation parameters the user *explicitly* set in the
        # constructor so _mutate_payload can strip unsolicited defaults.
        self._explicit_init_params: frozenset = self._explicit_param_keys(kwargs)

        # Resolve Portkey-specific values (parameter > env var > None).
        # All values are resolved eagerly so they're available for inspection,
        # but _get_headers() only emits the headers relevant to the active
        # routing mode to avoid conflicts.
        self.config_id = self._resolve_optional(config_id, "PORTKEY_CONFIG")
        self.portkey_provider = self._resolve_optional(portkey_provider, "PORTKEY_PROVIDER")
        self.virtual_key = self._resolve_optional(virtual_key, "PORTKEY_VIRTUAL_KEY")
        self.provider_api_key = self._resolve_optional(provider_api_key, "PORTKEY_PROVIDER_API_KEY")
        self.portkey_headers = self._normalize_header_map(portkey_headers)

        # Warn when conflicting routing values are detected — a common pitfall
        # when env vars like PORTKEY_VIRTUAL_KEY linger in the shell.
        self._warn_on_routing_conflicts()

        super().__init__(model=model, base_url=base_url, api_key=api_key, **kwargs)
        self._validate_base_url()

    # ── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _resolve_optional(value: Optional[str], env_var: str) -> Optional[str]:
        """Resolve a value with parameter > env-var > None precedence."""
        if value is not None:
            text = str(value).strip()
            return text or None

        env_value = os.getenv(env_var)
        if isinstance(env_value, str):
            env_text = env_value.strip()
            return env_text or None
        return None

    @staticmethod
    def _normalize_header_map(headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Normalise a raw header dict to ``{str: str}`` (strip + skip blanks)."""
        if not isinstance(headers, dict):
            return {}
        normalized: Dict[str, str] = {}
        for raw_key, raw_value in headers.items():
            if raw_key is None or raw_value is None:
                continue
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if key and value:
                normalized[key] = value
        return normalized

    @staticmethod
    def _is_set(value: Optional[str]) -> bool:
        """Return True if *value* is a non-blank, non-``"EMPTY"`` string."""
        if value is None:
            return False
        text = str(value).strip()
        return bool(text) and text.upper() != "EMPTY"

    def _get_api_key_from_config(self) -> Optional[str]:
        """Fallback: read Portkey API key from AbstractCore config."""
        try:
            from ..config.manager import get_config_manager

            cfg = get_config_manager()
            return getattr(cfg.config.api_keys, "portkey", None)
        except Exception:
            return None

    def _validate_base_url(self) -> None:
        """Validate base_url early for clearer error messages."""
        parsed = urlparse(str(self.base_url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"Invalid PORTKEY_BASE_URL '{self.base_url}'. "
                "Expected full URL like 'https://host/v1'."
            )

    @classmethod
    def _explicit_param_keys(cls, params: Dict[str, Any]) -> frozenset:
        """Return the set of generation params explicitly set by the user.

        We only treat non-None values as explicit to avoid forwarding defaults
        when callers pass `None` to mean "use provider default".
        """
        return frozenset(
            key for key, value in params.items()
            if key in cls._OPTIONAL_GEN_PARAMS and value is not None
        )

    def _uses_max_completion_tokens(self) -> bool:
        """Check if this model uses max_completion_tokens instead of max_tokens.

        Mirrors OpenAI provider heuristics to stay consistent with the core
        abstraction layer (o1 + gpt-5 family).
        """
        model_lower = self.model.lower()
        return (
            model_lower.startswith("o1") or
            "gpt-5" in model_lower or
            model_lower.startswith("gpt-o1")
        )

    def _is_reasoning_model(self) -> bool:
        """Check if this is a reasoning model with limited parameter support."""
        model_lower = self.model.lower()
        return (
            model_lower.startswith("o1") or
            "gpt-5" in model_lower or
            model_lower.startswith("gpt-o1")
        )

    def _handle_api_error(self, error: Exception) -> Exception:
        """Add Portkey-specific diagnostics for connection errors."""
        err = super()._handle_api_error(error)
        msg = str(err)
        lowered = msg.lower()
        if any(
            phrase in lowered
            for phrase in (
                "nodename nor servname provided",
                "name or service not known",
                "temporary failure in name resolution",
            )
        ):
            return type(err)(
                f"{msg} (base_url={self.base_url}). "
                "Verify PORTKEY_BASE_URL and network/DNS/VPN access."
            )
        return err

    # ── Model validation ─────────────────────────────────────────────

    def _validate_model(self):
        """Skip model validation for Portkey.

        Portkey is a routing gateway, not a model provider.  The ``/models``
        endpoint on a Portkey gateway returns the raw model catalogue of all
        backend providers — but the actual model available to the caller
        depends on the config / virtual-key routing, which may map, alias,
        or restrict model names.  Validating against ``/models`` therefore
        produces false negatives (e.g. ``gpt-5-mini`` is available through the
        config but absent from the raw catalogue).
        """
        # Intentionally no-op — let the generation request itself surface any
        # model-not-found errors from the upstream provider.
        pass

    # ── Routing mode detection ─────────────────────────────────────────

    def _routing_mode(self) -> str:
        """Determine the active routing mode (precedence: config > virtual-key > provider-direct)."""
        if self._is_set(self.config_id):
            return self._MODE_CONFIG
        if self._is_set(self.virtual_key):
            return self._MODE_VIRTUAL_KEY
        return self._MODE_PROVIDER_DIRECT

    def _warn_on_routing_conflicts(self) -> None:
        """Log a warning if multiple mutually-exclusive routing values are set.

        This catches the common case where ``PORTKEY_VIRTUAL_KEY`` lingers in
        the shell while the user intends to use a config.
        """
        active = [
            name for name, val in [
                ("config_id", self.config_id),
                ("virtual_key", self.virtual_key),
                ("portkey_provider", self.portkey_provider),
            ]
            if self._is_set(val)
        ]
        if len(active) > 1:
            mode = self._routing_mode()
            suppressed = [n for n in active if n != {
                self._MODE_CONFIG: "config_id",
                self._MODE_VIRTUAL_KEY: "virtual_key",
                self._MODE_PROVIDER_DIRECT: "portkey_provider",
            }.get(mode)]
            logger.warning(
                "PortkeyProvider routing conflict detected; suppressed headers.",
                active_routing_values=active,
                selected_mode=mode,
                suppressed_headers=suppressed,
                hint="unset unused env vars (e.g. `unset PORTKEY_VIRTUAL_KEY`)",
            )

    # ── Payload adaptation ──────────────────────────────────────────────

    def _mutate_payload(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Adapt the OpenAI-compatible payload for Portkey's multi-provider gateway.

        Portkey is a pass-through gateway — the payload is forwarded **verbatim**
        to the backend provider (OpenAI, Azure OpenAI, Anthropic, Gemini, …).

        Because we don't know which backend model sits behind the config, we
        apply two rules:

        1. **Strip unsolicited defaults** — parameters like ``temperature``,
           ``top_p``, ``max_tokens`` are only included if the user *explicitly*
           set them (in the constructor or in the ``generate()`` call).
           This matches the behaviour of the OpenAI Python SDK: if you don't
           pass ``temperature``, it's omitted from the request and the backend
           uses its own default.  Critical because GPT-5 / o-series reasoning
           models reject ``temperature ≠ 1`` and similar constraints.

        2. **Rename** ``max_tokens`` → ``max_completion_tokens`` for OpenAI
           reasoning families (o1, gpt-5) or when the user explicitly asks for
           ``max_completion_tokens``. Other models keep the legacy name.
        """
        # ── Determine which params the user explicitly asked for ───────
        # _explicit_init_params: set in __init__ from constructor kwargs
        # kwargs here: the original kwargs from the generate() call
        explicit = self._explicit_init_params | self._explicit_param_keys(kwargs)

        # ── Reasoning models: strip unsupported params even if explicit ─
        if self._is_reasoning_model():
            blocked = (
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "repetition_penalty",
            )
            explicit_blocked = [p for p in blocked if p in explicit]
            if explicit_blocked:
                logger.warning(
                    "PortkeyProvider dropped unsupported parameters for reasoning model.",
                    model=self.model,
                    dropped_params=explicit_blocked,
                )
            for param in blocked:
                payload.pop(param, None)

        # ── Strip unsolicited generation defaults ──────────────────────
        for param in ("temperature", "top_p"):
            if param in payload and param not in explicit:
                del payload[param]

        # max_tokens → max_completion_tokens (modern OpenAI API standard)
        if "max_tokens" in payload:
            explicit_token_limit = bool(
                explicit.intersection({"max_tokens", "max_output_tokens", "max_completion_tokens"})
            )
            if explicit_token_limit:
                if "max_completion_tokens" in explicit or self._uses_max_completion_tokens():
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                # Otherwise keep legacy max_tokens for non-OpenAI backends.
            else:
                # Unsolicited default — remove entirely, let the backend decide.
                del payload["max_tokens"]

        return payload

    # ── Header construction ────────────────────────────────────────────

    def _get_headers(self) -> Dict[str, str]:
        """Build HTTP headers for the active routing mode.

        Routing modes are mutually exclusive per Portkey docs:
        - config mode   → ``x-portkey-config`` only
        - virtual-key   → ``x-portkey-virtual-key`` only
        - provider-direct → ``x-portkey-provider`` + ``Authorization``

        The Portkey gateway API key (``x-portkey-api-key`` + ``Authorization``)
        is always included for gateway authentication.
        """
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        mode = self._routing_mode()

        # ── Gateway authentication (always) ────────────────────────────
        if self._is_set(self.api_key):
            api_key_str = str(self.api_key).strip()
            # x-portkey-api-key:  Portkey-native gateway auth
            # Authorization:     OpenAI-compatible auth (needed by enterprise /
            #                    custom gateways such as Roche Galileo)
            headers["x-portkey-api-key"] = api_key_str
            headers["Authorization"] = f"Bearer {api_key_str}"

        # ── Routing headers (mode-dependent) ───────────────────────────
        if mode == self._MODE_CONFIG:
            # Config encapsulates everything — no extra routing headers.
            headers["x-portkey-config"] = str(self.config_id).strip()

        elif mode == self._MODE_VIRTUAL_KEY:
            headers["x-portkey-virtual-key"] = str(self.virtual_key).strip()

        else:  # provider-direct
            if self._is_set(self.portkey_provider):
                headers["x-portkey-provider"] = str(self.portkey_provider).strip()
            if self._is_set(self.provider_api_key):
                # Override Authorization with the upstream provider key
                # (e.g. an OpenAI sk-… or Anthropic sk-ant-… key).
                headers["Authorization"] = f"Bearer {str(self.provider_api_key).strip()}"

        # ── User-supplied extra headers (escape hatch) ─────────────────
        if self.portkey_headers:
            headers.update(self.portkey_headers)

        return headers
