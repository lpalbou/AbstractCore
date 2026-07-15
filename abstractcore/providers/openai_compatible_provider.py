"""
Generic OpenAI-compatible provider for any OpenAI-compatible API endpoint.

Supports any server implementing the OpenAI API format:
- llama.cpp server
- text-generation-webui (with OpenAI extension)
- LocalAI
- FastChat
- Aphrodite
- SGLang
- Custom deployments and proxies
"""

import os
import httpx
import json
import re
import time
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Type, TYPE_CHECKING, Tuple

# Server-side Harmony parse failures of the MODEL'S OWN OUTPUT (gpt-oss on
# vLLM): "unexpected tokens remaining in message header" is the strict
# openai-harmony parser rejecting a malformed sampled header; sibling shapes
# name the harmony parser explicitly. These arrive as HTTP 400 but are
# generation races, not request errors (see _raise_for_status).
_HARMONY_GENERATION_ARTIFACT_RE = re.compile(
    r"unexpected tokens remaining in message header|HarmonyError|openai[_-]harmony",
    re.IGNORECASE,
)

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

if TYPE_CHECKING:
    from ..media.types import MediaContent


def _inline_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline local $defs/$ref references in a JSON Schema dict.

    Some OpenAI-compatible servers only partially support `$defs`/`$ref` inside
    `response_format: {type:'json_schema'}`. Inlining keeps schemas simple and
    improves compatibility for structured outputs.
    """

    defs = schema.get("$defs")
    if not isinstance(defs, dict) or not defs:
        return schema

    def _resolve(node: Any, *, seen: set[str]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref[len("#/$defs/"):]
                target = defs.get(key)
                if isinstance(key, str) and key and isinstance(target, dict):
                    if key in seen:
                        return node
                    seen.add(key)
                    resolved_target = _resolve(dict(target), seen=seen)
                    seen.remove(key)
                    if isinstance(resolved_target, dict):
                        merged: Dict[str, Any] = dict(resolved_target)
                        for k, v in node.items():
                            if k == "$ref":
                                continue
                            merged[k] = _resolve(v, seen=seen)
                        return merged

            out: Dict[str, Any] = {}
            for k, v in node.items():
                if k == "$defs":
                    continue
                out[k] = _resolve(v, seen=seen)
            return out

        if isinstance(node, list):
            return [_resolve(x, seen=seen) for x in node]

        return node

    try:
        base = {k: v for k, v in schema.items() if k != "$defs"}
        inlined = _resolve(base, seen=set())
        return inlined if isinstance(inlined, dict) and inlined else schema
    except Exception:
        return schema
from .base import BaseProvider, ThinkingControlHandling
from ..architectures.response_postprocessing import extract_reasoning_from_message
from ..core.types import GenerateResponse
from ..exceptions import (
    ProviderAPIError,
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    format_model_error,
)
from ..tools import UniversalToolHandler
from ..utils.truncation import preview_text


class OpenAICompatibleProvider(BaseProvider):
    """
    Generic provider for any OpenAI-compatible API endpoint.

    Works with any server implementing the OpenAI API format:
    - llama.cpp server
    - text-generation-webui (OpenAI extension)
    - LocalAI
    - FastChat
    - Aphrodite
    - SGLang
    - Custom deployments and proxies

    Usage:
        # Basic usage
        llm = create_llm("openai-compatible",
                        base_url="http://127.0.0.1:1234/v1",
                        model="llama-3.1-8b")

        # With API key (optional for many local servers)
        llm = create_llm("openai-compatible",
                        base_url="http://127.0.0.1:1234/v1",
                        model="my-model",
                        api_key="your-key")

        # Environment variable configuration
        export OPENAI_BASE_URL="http://127.0.0.1:1234/v1"
        export OPENAI_API_KEY="your-key"  # Optional
        llm = create_llm("openai-compatible", model="my-model")
    """

    PROVIDER_ID = "openai-compatible"
    PROVIDER_DISPLAY_NAME = "OpenAI-compatible server"
    BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
    API_KEY_ENV_VAR = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(
        self,
        model: str = "default",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        validate_model: bool = True,
        supports_chat_template_kwargs: bool = False,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.provider = self.PROVIDER_ID
        self._validate_model_on_init = bool(validate_model)
        self._supports_chat_template_kwargs = bool(supports_chat_template_kwargs)

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        self.base_url = self._resolve_base_url(base_url)

        self.api_key = self._resolve_api_key(api_key)

        # #[WARNING:TIMEOUT]
        # Get timeout value - None means unlimited timeout
        timeout_value = getattr(self, '_timeout', None)
        # Validate timeout if provided (None is allowed for unlimited)
        if timeout_value is not None and timeout_value <= 0:
            timeout_value = None  # Invalid timeout becomes unlimited

        try:
            self.client = httpx.Client(timeout=timeout_value)
        except Exception as e:
            # Fallback with default timeout if client creation fails
            try:
                fallback_timeout = None
                try:
                    from ..config.manager import get_config_manager

                    fallback_timeout = float(get_config_manager().get_default_timeout())
                except Exception:
                    fallback_timeout = 7200.0
                if isinstance(fallback_timeout, (int, float)) and float(fallback_timeout) <= 0:
                    fallback_timeout = None
                self.client = httpx.Client(timeout=fallback_timeout)
            except Exception:
                raise RuntimeError(f"Failed to create HTTP client for {self.PROVIDER_DISPLAY_NAME}: {e}")

        self._async_client = None  # Lazy-loaded async client

        # Validate model exists on server unless the caller is using an endpoint-only path
        # such as embeddings, where chat model catalogues can be incomplete.
        if self._validate_model_on_init:
            self._validate_model()

    @property
    def async_client(self):
        """Lazy-load async HTTP client for native async operations."""
        if self._async_client is None:
            timeout_value = getattr(self, '_timeout', None)
            if timeout_value is not None and timeout_value <= 0:
                timeout_value = None
            self._async_client = httpx.AsyncClient(timeout=timeout_value)
        return self._async_client

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with optional API key authentication."""
        headers = {"Content-Type": "application/json"}
        # Only add Authorization header if api_key is provided and meaningful.
        api_key = None if self.api_key is None else str(self.api_key).strip()
        if api_key and api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # Sampling parameters subject to registry-driven filtering. `max_tokens` is
    # deliberately NOT here — every request needs an output cap, so the token
    # param is handled by RENAME (token_param_name), never by dropping.
    _REGISTRY_FILTERED_SAMPLING_PARAMS = (
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
    )

    def _apply_model_parameter_constraints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce model_capabilities.json parameter constraints on the wire payload.

        Two registry-driven constraints (both honored by the OpenAI provider since
        the capability-filtering wave; this provider built payloads unconditionally,
        so a restricted model served through an OpenAI-compatible endpoint — a
        LiteLLM/proxy route to an o-series/GPT-5-class API, or a strict vLLM
        deployment — received parameters its API rejects and failed with a 400):

        1. `unsupported_parameters`: sampling params the model's API rejects are
           DROPPED silently (the registry list is the authoritative enforcement;
           upstream callers always pass temperature/top_p as defaults, so warning
           per call is noise — same convention as the OpenAI provider).
        2. `token_param_name`: the output-cap key is RENAMED (`max_tokens` ->
           `max_completion_tokens`) when the registry declares it; the cap itself
           is never dropped.

        Absent fields mean no restrictions — payloads are byte-identical for every
        model without declarations (all local/self-hosted models today).
        Runs BEFORE `_mutate_payload` so subclass hooks see the filtered payload.
        """
        for param in self._REGISTRY_FILTERED_SAMPLING_PARAMS:
            if param in payload and not self._is_parameter_supported(param):
                dropped = payload.pop(param, None)
                # Debug-level by convention (no per-call noise), but present so an
                # explicitly-passed value that vanishes is forensically traceable.
                self.logger.debug(
                    f"Dropped generation parameter '{param}'={dropped!r} for model "
                    f"'{self.model}': declared in unsupported_parameters"
                )

        token_param = self._get_token_param_name()
        if token_param != "max_tokens" and "max_tokens" in payload:
            payload[token_param] = payload.pop("max_tokens")

        return payload

    def _mutate_payload(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Provider-specific payload hook.

        This provider supports best-effort passthrough of common OpenAI-compatible
        server extensions when present:

        - `extra_body`: vLLM and other servers
        - `chat_template_kwargs`: llama.cpp/LM Studio style servers
        """
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict) and extra_body:
            existing = payload.get("extra_body")
            if isinstance(existing, dict) and existing:
                payload["extra_body"] = {**existing, **extra_body}
            else:
                payload["extra_body"] = dict(extra_body)

        chat_template_kwargs = kwargs.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
            existing_ctk = payload.get("chat_template_kwargs")
            if isinstance(existing_ctk, dict) and existing_ctk:
                payload["chat_template_kwargs"] = {**existing_ctk, **chat_template_kwargs}
            else:
                payload["chat_template_kwargs"] = dict(chat_template_kwargs)

        # LM Studio sometimes exposes model-specific "custom fields" (e.g., Qwen3.5's "Enable Thinking")
        # that can affect chat-template variables. There's no stable, public OpenAI-compatible surface
        # for these, so we support a small best-effort passthrough:
        # - callers/providers may pass `lmstudio_template_vars` to be copied to the top-level request.
        provider_id = str(getattr(self, "provider", "") or "").strip().lower()
        if provider_id == "lmstudio":
            template_vars = kwargs.get("lmstudio_template_vars")
            if isinstance(template_vars, dict) and template_vars:
                for k, v in template_vars.items():
                    if isinstance(k, str) and k.strip():
                        payload[k.strip()] = v

        return payload

    @staticmethod
    def _build_usage_dict(usage: Any) -> Dict[str, Any]:
        """Normalize OpenAI-compatible usage, preserving token detail breakdowns.

        `completion_tokens_details.reasoning_tokens` is the only billing evidence of
        invisible reasoning (e.g. grok-4-class models that reason without returning text),
        so detail dicts must be passed through rather than rebuilt away.
        """
        usage = usage if isinstance(usage, dict) else {}
        out: Dict[str, Any] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            # Keep legacy keys for backward compatibility
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        for detail_key in ("completion_tokens_details", "prompt_tokens_details"):
            details = usage.get(detail_key)
            if isinstance(details, dict) and details:
                out[detail_key] = dict(details)
        # Normalized cross-provider cache key: surfaced only when the server actually
        # reported the field (absent != 0 — "cannot report" stays distinguishable).
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
            try:
                out["cached_input_tokens"] = int(prompt_details.get("cached_tokens") or 0)
            except Exception:
                pass
        return out

    @staticmethod
    def _snake_to_camel(name: str) -> str:
        parts = [p for p in str(name or "").split("_") if p]
        if len(parts) <= 1:
            return str(name or "")
        return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        if enabled is None and level is None:
            return kwargs, ThinkingControlHandling()

        provider_id = str(getattr(self, "provider", "") or "").strip().lower()
        if provider_id not in {"lmstudio", "openai-compatible"}:
            return kwargs, ThinkingControlHandling()
        template_kwargs_supported = bool(
            provider_id == "lmstudio"
            or getattr(self, "_supports_chat_template_kwargs", False)
            or getattr(self, "supports_chat_template_kwargs", False)
        )

        # Asset-driven control surfaces (see abstractcore/architectures/thinking_controls.py).
        # Which chat-template variables a model's template understands is model knowledge and
        # lives in the registries; this hook only owns the transport (how the kwargs are sent).
        surfaces = self._thinking_control_surfaces()

        # Boolean template switch (e.g. Qwen3/Nemotron/Gemma-4 `enable_thinking`).
        #
        # Some backends do not interpret prompt-level control tokens (e.g. "/no_think") as
        # actual switches for these templates; a backend-native knob is more reliable when
        # supported.
        if surfaces.template_kwarg:
            if not template_kwargs_supported:
                return kwargs, ThinkingControlHandling()
            requested = enabled if enabled is not None else (level is not None)
            template_kwarg = surfaces.template_kwarg
            new_kwargs = dict(kwargs)
            ctk = new_kwargs.get("chat_template_kwargs")
            ctk_dict: Dict[str, Any] = dict(ctk) if isinstance(ctk, dict) else {}
            # Snake_case Jinja variable name (used by Qwen docs, llama.cpp, vLLM recipes).
            ctk_dict[template_kwarg] = bool(requested)
            # LM Studio model.yaml customFields often use camelCase keys like `enableThinking`
            # which may be forwarded to templates by some runtimes. Keep both for compatibility.
            camel_kwarg = self._snake_to_camel(template_kwarg)
            if camel_kwarg != template_kwarg:
                ctk_dict[camel_kwarg] = bool(requested)

            # Some templates (Nemotron v3) additionally expose a "low effort" switch that
            # reduces reasoning tokens while keeping thinking enabled. Map our unified "low"
            # (and "minimal") levels to this knob when declared.
            #
            # This is intentionally best-effort: if a backend/template ignores the kwarg, it
            # should not break the request.
            handled_level = False
            if surfaces.low_effort_template_kwarg and enabled is not False and level in {"minimal", "low"}:
                ctk_dict[surfaces.low_effort_template_kwarg] = True
                handled_level = True
            new_kwargs["chat_template_kwargs"] = ctk_dict

            # LM Studio's OpenAI-compatible endpoint does not document `chat_template_kwargs`,
            # and some builds historically ignored it. Replicate the same payload under the
            # common "extra_body" extension used by other OpenAI-compatible stacks (vLLM, etc.)
            # as a best-effort compatibility shim.
            if provider_id == "lmstudio":
                eb = new_kwargs.get("extra_body")
                eb_dict: Dict[str, Any] = dict(eb) if isinstance(eb, dict) else {}
                eb_ctk = eb_dict.get("chat_template_kwargs")
                eb_ctk_dict: Dict[str, Any] = dict(eb_ctk) if isinstance(eb_ctk, dict) else {}
                eb_ctk_dict.update(ctk_dict)
                eb_dict["chat_template_kwargs"] = eb_ctk_dict
                new_kwargs["extra_body"] = eb_dict

                # Best-effort LM Studio template-variable passthrough.
                #
                # Some LM Studio runtimes/models do not honor `chat_template_kwargs`, but do expose
                # a model-specific custom field (often camelCase) which ultimately sets a Jinja
                # variable. Send both snake_case and camelCase to maximize compatibility.
                tv = new_kwargs.get("lmstudio_template_vars")
                tv_dict: Dict[str, Any] = dict(tv) if isinstance(tv, dict) else {}
                tv_dict.setdefault(template_kwarg, bool(requested))
                if camel_kwarg != template_kwarg:
                    tv_dict.setdefault(camel_kwarg, bool(requested))
                new_kwargs["lmstudio_template_vars"] = tv_dict
            return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=handled_level)

        # Integer budget template kwarg (e.g. Seed-OSS `thinking_budget`).
        #
        # NOTE: This is currently best-effort and only enabled for local OpenAI-compatible
        # servers (LM Studio / generic OpenAI-compatible) to avoid breaking strict
        # third-party gateways that may reject unknown payload fields.
        if not surfaces.budget_template_kwarg or not template_kwargs_supported:
            return kwargs, ThinkingControlHandling()

        budget_map = {"low": 512, "medium": 1024, "high": 4096, "xhigh": 8192}
        if enabled is False:
            budget = 0
        elif isinstance(level, str) and level in budget_map:
            budget = budget_map[level]
        else:
            # Default when explicitly enabled without a level.
            budget = 1024

        new_kwargs = dict(kwargs)
        ctk = new_kwargs.get("chat_template_kwargs")
        ctk_dict: Dict[str, Any] = dict(ctk) if isinstance(ctk, dict) else {}
        ctk_dict[surfaces.budget_template_kwarg] = int(budget)
        new_kwargs["chat_template_kwargs"] = ctk_dict
        return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)

    def _resolve_base_url(self, base_url: Optional[str]) -> str:
        """Resolve base URL with parameter > env var > default precedence."""
        if base_url is not None:
            resolved = str(base_url).strip()
            if not resolved:
                raise ValueError("base_url cannot be empty")
            return resolved.rstrip("/")

        env_var = getattr(self, "BASE_URL_ENV_VAR", None)
        env_val = os.getenv(env_var) if isinstance(env_var, str) and env_var else None
        if isinstance(env_val, str) and env_val.strip():
            return env_val.strip().rstrip("/")

        default = getattr(self, "DEFAULT_BASE_URL", None) or ""
        return str(default).strip().rstrip("/")

    def _resolve_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Resolve API key with parameter > env var > config fallback."""
        if api_key is not None:
            # Allow callers to explicitly disable auth by passing an empty string.
            return api_key

        env_var = getattr(self, "API_KEY_ENV_VAR", None)
        env_val = os.getenv(env_var) if isinstance(env_var, str) and env_var else None
        if env_val is not None:
            return env_val

        return self._get_api_key_from_config()

    def _get_api_key_from_config(self) -> Optional[str]:
        """Optional config-manager fallback for subclasses (default: none)."""
        return None

    def _extract_error_detail(self, response: Optional[httpx.Response]) -> Optional[str]:
        """Extract a useful error message from an HTTPX response, if possible."""
        if response is None:
            return None

        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    for k in ("message", "error", "detail"):
                        v = err.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                for k in ("message", "detail"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            # If it's JSON but not a dict, stringify it.
            if data is not None:
                return json.dumps(data, ensure_ascii=False)
        except Exception:
            pass

        try:
            text = response.text
            if isinstance(text, str) and text.strip():
                # Bound size to avoid dumping huge error bodies.
                body = text.strip()
                return preview_text(body, max_chars=2000)
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_retry_after_s(response: Optional[httpx.Response]) -> Optional[float]:
        """Parse the `Retry-After` header (seconds or HTTP-date) into seconds, if present.

        The server's own requested wait is the one delay signal better than our jitter
        guess (C3 Retry-After honoring); the retry layer still caps it at its max_delay.
        """
        try:
            headers = getattr(response, "headers", None)
            if headers is None:
                return None
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is None:
                return None
            raw = str(raw).strip()
            if not raw:
                return None
            try:
                seconds = float(raw)
                return seconds if seconds >= 0 else None
            except ValueError:
                pass
            # HTTP-date form (RFC 7231): compute the delta from now.
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone

            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (dt - datetime.now(timezone.utc)).total_seconds()
            return max(delta, 0.0)
        except Exception:
            # Best-effort only: a malformed header must never mask the real error.
            return None

    def _raise_for_status(self, response: httpx.Response, *, request_url: Optional[str] = None) -> None:
        """Raise rich provider exceptions on HTTP errors."""
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            # Unit tests sometimes stub the HTTP response with only `.raise_for_status()`/`.json()`.
            # Treat as success if `.raise_for_status()` does not raise.
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            return

        if int(status_code) < 400:
            return

        detail = self._extract_error_detail(response)
        prefix = f"{self.PROVIDER_DISPLAY_NAME} API error ({status_code})"
        msg = f"{prefix}: {detail}" if detail else prefix

        status = int(status_code)
        if status in (401, 403):
            raise AuthenticationError(msg, status_code=status)
        if status == 429:
            raise RateLimitError(msg, status_code=status,
                                 retry_after_s=self._extract_retry_after_s(response))
        if status == 400:
            # Many OpenAI-compatible servers use 400 for schema/model errors.
            if detail and ("model" in detail.lower()) and ("not found" in detail.lower()):
                self._raise_model_not_found()
            # Harmony generation artifact (gpt-oss on vLLM): the MODEL's own
            # sampled output sometimes violates its Harmony template (e.g. an
            # unclosed `to=...` recipient header) and the server's strict
            # openai-harmony parser surfaces the parse failure as a 400 on a
            # perfectly valid request (vllm#23567, openai/harmony#38/#80;
            # fixed upstream by lenient parsing, vllm#28303 — not yet on all
            # deployments). The request is NOT invalid and a resample usually
            # passes, so this must be classified TRANSIENT (retryable), never
            # InvalidRequestError — otherwise every retry layer refuses and a
            # sampling race becomes a hard failure.
            if detail and _HARMONY_GENERATION_ARTIFACT_RE.search(detail):
                raise ProviderAPIError(
                    f"{msg} [transient harmony generation artifact - the model's sampled "
                    "output violated its template; a retry resamples]",
                    status_code=status,
                )
            raise InvalidRequestError(msg, status_code=status)
        if status == 404:
            # Could be endpoint misconfiguration (missing /v1) or an unknown model.
            if detail and ("model" in detail.lower()) and ("not found" in detail.lower()):
                self._raise_model_not_found()
            raise ProviderAPIError(msg if request_url is None else f"{msg} [{request_url}]", status_code=status)

        # 5xx (incl. 503 with a server-named wait): transient server-side class.
        raise ProviderAPIError(msg if request_url is None else f"{msg} [{request_url}]",
                               status_code=status,
                               retry_after_s=self._extract_retry_after_s(response))

    def _raise_model_not_found(self) -> None:
        """Raise ModelNotFoundError with a best-effort available-model list."""
        try:
            available_models = self.list_available_models(base_url=self.base_url)
        except Exception:
            available_models = []
        raise ModelNotFoundError(format_model_error(self.PROVIDER_DISPLAY_NAME, self.model, available_models))

    def _is_prompt_cache_key_rejection(self, response: Optional[httpx.Response], payload: Dict[str, Any]) -> bool:
        """True when the server 400-rejected the request BECAUSE of `prompt_cache_key`.

        Prompt caching is best-effort: most OpenAI-compatible servers ignore unknown fields,
        but some (e.g. OVH AI Endpoints) reject them outright. Callers drop the key, mark the
        provider instance, and retry once.
        """
        if "prompt_cache_key" not in payload:
            return False
        status = getattr(response, "status_code", None)
        try:
            if status is None or int(status) != 400:
                return False
        except Exception:
            return False
        detail = self._extract_error_detail(response) or ""
        return "prompt_cache_key" in detail

    def _mark_prompt_cache_key_unsupported(self) -> None:
        self._prompt_cache_key_unsupported = True
        if hasattr(self, "logger"):
            self.logger.warning(
                "#FALLBACK: server rejected 'prompt_cache_key'; retrying without it "
                "(prompt caching disabled for this provider instance)"
            )

    def _is_stream_options_rejection(self, response: Optional[httpx.Response], payload: Dict[str, Any]) -> bool:
        """True when the server 400-rejected the request BECAUSE of `stream_options`.

        `stream_options: {"include_usage": true}` is the standard OpenAI mechanism for
        usage accounting on streamed responses (a final chunk with empty `choices` and a
        `usage` object). Most servers (vLLM, LM Studio, llama.cpp) support it; strict
        servers that reject unknown fields get the same drop-and-retry treatment as
        `prompt_cache_key` so streaming itself never breaks over an accounting extra.
        """
        if "stream_options" not in payload:
            return False
        status = getattr(response, "status_code", None)
        try:
            if status is None or int(status) != 400:
                return False
        except Exception:
            return False
        detail = self._extract_error_detail(response) or ""
        return "stream_options" in detail

    def _mark_stream_options_unsupported(self) -> None:
        self._stream_options_unsupported = True
        if hasattr(self, "logger"):
            self.logger.warning(
                "#FALLBACK: server rejected 'stream_options'; retrying without it "
                "(streamed usage accounting unavailable for this provider instance)"
            )

    def _validate_model(self):
        """Validate that the model exists on the server (best-effort)."""
        # Skip validation for "default" placeholder (used by registry for model listing)
        if self.model == "default":
            return

        try:
            # Use base_url as-is (should include /v1) for model discovery
            available_models = self.list_available_models(base_url=self.base_url)
            if available_models and self.model not in available_models:
                error_message = format_model_error(self.PROVIDER_DISPLAY_NAME, self.model, available_models)
                raise ModelNotFoundError(error_message)
        except httpx.ConnectError:
            # Server not running - will fail later when trying to generate
            if hasattr(self, 'logger'):
                self.logger.debug(f"{self.PROVIDER_DISPLAY_NAME} not accessible at {self.base_url} - model validation skipped")
            pass
        except ModelNotFoundError:
            # Re-raise model not found errors
            raise
        except Exception as e:
            # Other errors (like timeout, None type errors) - continue, will fail later if needed
            if hasattr(self, 'logger'):
                self.logger.debug(f"Model validation failed with error: {e} - continuing anyway")
            pass

    def unload_model(self, model_name: str) -> None:
        """
        Close HTTP client connection.

        Note: Most OpenAI-compatible servers manage model memory automatically.
        This method only closes the HTTP client connection for cleanup.
        """
        try:
            # Close the HTTP client connection
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()

            # Close async client if it was created
            if self._async_client is not None:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._async_client.aclose())
                except RuntimeError:
                    # No running loop
                    import asyncio
                    asyncio.run(self._async_client.aclose())

        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          execute_tools: Optional[bool] = None,
                          tool_call_tags: Optional[str] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using OpenAI-compatible server"""

        # Build messages for chat completions with tool support
        chat_messages = []

        # Add tools to system prompt if provided
        final_system_prompt = system_prompt
        # Prefer native tools when the model supports them. Only inject a prompted tool list
        # when native tool calling is not available.
        if tools and self.tool_handler.supports_prompted and not self.tool_handler.supports_native:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt += f"\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Add system message if provided
        if final_system_prompt:
            chat_messages.append({
                "role": "system",
                "content": final_system_prompt
            })

        # Add conversation history
        if messages:
            chat_messages.extend(messages)

        media_enrichment = None

        # Handle media content regardless of prompt (media can be used with messages too)
        if media:
            # Get the last user message content to combine with media
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and chat_messages:
                # If no prompt, try to get text from the last user message
                for msg in reversed(chat_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break
            try:
                # Process media files into MediaContent objects first
                processed_media = self._process_media_content(media)

                # Use capability-based media handler selection
                media_handler = self._get_media_handler_for_model(self.model)

                # Create multimodal message combining text and processed media
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                # For OpenAI-compatible servers, we might get a string (embedded text) or dict (structured)
                if isinstance(multimodal_message, str):
                    # Replace the last user message with the multimodal message, or add new one
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1]["content"] = multimodal_message
                    else:
                        chat_messages.append({
                            "role": "user",
                            "content": multimodal_message
                        })
                else:
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        # Replace last user message with structured multimodal message
                        chat_messages[-1] = multimodal_message
                    else:
                        chat_messages.append(multimodal_message)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                if user_message_text:
                    chat_messages.append({
                        "role": "user",
                        "content": user_message_text
                    })
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                if user_message_text:
                    chat_messages.append({
                        "role": "user",
                        "content": user_message_text
                    })

        # Add prompt as separate message if provided (for backward compatibility)
        elif prompt and prompt.strip():
            chat_messages.append({
                "role": "user",
                "content": prompt
            })

        # Some OpenAI-compatible servers (including common LM Studio templates) require at
        # least one user message. If upstream callers accidentally route the entire prompt
        # into `system_prompt` (or omit the prompt), the server may fail during template
        # rendering ("No user query found in messages."). Add a minimal user message as a
        # best-effort fallback rather than hard-failing.
        if not any(isinstance(m, dict) and m.get("role") == "user" for m in chat_messages):
            fallback = str(prompt or "").strip() or "Continue."
            chat_messages.append({"role": "user", "content": fallback})

        # Strict-server system-message normalization (must run after ALL message building).
        chat_messages = self._normalize_system_messages_for_strict_servers(chat_messages)

        # Build request payload using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "max_tokens": max_output_tokens,
            "top_p": generation_kwargs.get("top_p", 0.9),
        }
        top_k_value = generation_kwargs.get("top_k")
        if top_k_value is not None and str(getattr(self, "provider", "") or "").strip().lower() == "lmstudio":
            payload["top_k"] = top_k_value

        # Streamed usage accounting: without this the final usage chunk is never
        # emitted and every streamed call reports usage=None (accounting goes dark).
        if stream and not getattr(self, "_stream_options_unsupported", False):
            payload["stream_options"] = {"include_usage": True}

        # Prompt caching (best-effort): pass through `prompt_cache_key` when provided.
        # Some OpenAI-compatible servers (e.g. OVH AI Endpoints) hard-reject unknown request
        # fields with a 400 instead of ignoring them; once a server rejects the key we stop
        # sending it (see _is_prompt_cache_key_rejection) so caching stays best-effort.
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if (
            isinstance(prompt_cache_key, str)
            and prompt_cache_key.strip()
            and not getattr(self, "_prompt_cache_key_unsupported", False)
        ):
            payload["prompt_cache_key"] = prompt_cache_key.strip()

        # Native tools (OpenAI-compatible): send structured tools/tool_choice when supported.
        if tools and self.tool_handler.supports_native:
            payload["tools"] = self.tool_handler.prepare_tools_for_native(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add additional generation parameters if provided (OpenAI-compatible)
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "repetition_penalty" in kwargs:
            # Some models support repetition_penalty directly
            payload["repetition_penalty"] = kwargs["repetition_penalty"]

        # Add seed if provided (many servers support seed via OpenAI-compatible API)
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            payload["seed"] = seed_value

        # Add structured output support (OpenAI-compatible format)
        # Many servers support native structured outputs using the response_format parameter
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            if isinstance(json_schema, dict) and json_schema:
                json_schema = _inline_json_schema_refs(json_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Registry-driven parameter constraints (unsupported_parameters / token_param_name)
        payload = self._apply_model_parameter_constraints(payload)

        # Provider-specific request extensions (vLLM extra_body, OpenRouter headers, etc.)
        payload = self._mutate_payload(payload, **kwargs)

        if stream:
            # Return streaming response - BaseProvider will handle tag rewriting via UnifiedStreamProcessor
            return self._stream_generate(payload)
        else:
            response = self._single_generate(payload)
            if media_enrichment:
                from ..media.enrichment import merge_enrichment_metadata

                response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

            # Execute tools if enabled and tools are present
            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    @staticmethod
    def _normalize_system_messages_for_strict_servers(chat_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make the message list valid for strict OpenAI-compatible servers.

        vLLM-class servers (e.g. OVH AI Endpoints) reject requests whose system messages are
        not at the beginning ('System message must be at the beginning.', HTTP 400) — unlike
        native OpenAI, which accepts system/developer messages anywhere. Mirroring the shipped
        Anthropic behavior:
        - a LEADING run of system messages is merged into ONE system message at index 0
          (some templates also reject multiple system messages);
        - every NON-leading system message is converted in place into a
          `<system_instruction>`-wrapped user message, DEFERRED past tool-result runs so
          assistant tool_calls / tool-result adjacency is never broken.
        Content semantics are preserved; only the transport shape changes.
        """
        if not chat_messages:
            return chat_messages
        if not any(
            isinstance(m, dict) and m.get("role") == "system"
            for i, m in enumerate(chat_messages)
            if i > 0
        ) and not (
            len(chat_messages) > 1
            and isinstance(chat_messages[0], dict)
            and chat_messages[0].get("role") == "system"
            and isinstance(chat_messages[1], dict)
            and chat_messages[1].get("role") == "system"
        ):
            return chat_messages  # fast path: nothing to normalize

        def _text(m: Dict[str, Any]) -> str:
            content = m.get("content")
            if isinstance(content, str):
                return content
            # Structured content parts (OpenAI content-part lists) must not become Python
            # reprs: extract the text parts (adversarial-review fix, 2026-07-09).
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            parts.append(t)
                    elif isinstance(part, str) and part:
                        parts.append(part)
                return "\n".join(parts)
            return str(content or "")

        # Merge the leading system run into one message.
        out: List[Dict[str, Any]] = []
        i = 0
        leading_parts: List[str] = []
        while i < len(chat_messages) and isinstance(chat_messages[i], dict) and chat_messages[i].get("role") == "system":
            leading_parts.append(_text(chat_messages[i]))
            i += 1
        merged_leading = "\n\n".join(p for p in leading_parts if p)
        if merged_leading:
            out.append({"role": "system", "content": merged_leading})

        # Convert non-leading system messages; defer flushes past tool-result runs.
        pending: List[Dict[str, Any]] = []

        def _flush() -> None:
            while pending:
                m = pending.pop(0)
                out.append({"role": "user", "content": f"<system_instruction>\n{_text(m)}\n</system_instruction>"})

        while i < len(chat_messages):
            m = chat_messages[i]
            role = m.get("role") if isinstance(m, dict) else None
            if role == "system":
                pending.append(m)
            elif role == "tool":
                out.append(m)  # never interleave inside a tool-result run
            else:
                _flush()
                out.append(m)
            i += 1
        _flush()
        return out

    def _single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Generate single response"""
        try:
            # Ensure client is available
            if not hasattr(self, 'client') or self.client is None:
                raise ProviderAPIError("HTTP client not initialized")

            # Track generation time
            start_time = time.time()
            request_url = f"{self.base_url}/chat/completions"
            response = self.client.post(
                request_url,
                json=payload,
                headers=self._get_headers()
            )
            if self._is_prompt_cache_key_rejection(response, payload):
                self._mark_prompt_cache_key_unsupported()
                payload = {k: v for k, v in payload.items() if k != "prompt_cache_key"}
                response = self.client.post(request_url, json=payload, headers=self._get_headers())
            self._raise_for_status(response, request_url=request_url)
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            # Extract response from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    message = {}

                content = message.get("content", "")
                reasoning = extract_reasoning_from_message(
                    message,
                    architecture_format=self.architecture_config,
                    model_capabilities=self.model_capabilities,
                )
                tool_calls = message.get("tool_calls")
                if tool_calls is None:
                    # Some servers surface tool calls at the choice level.
                    tool_calls = choice.get("tool_calls")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                reasoning = None
                tool_calls = None
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            metadata: Dict[str, Any] = {
                "_provider_request": {
                    "url": request_url,
                    "payload": payload,
                }
            }
            if isinstance(reasoning, str) and reasoning.strip():
                metadata["reasoning"] = reasoning

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                metadata=metadata,
                usage=self._build_usage_dict(usage),
                gen_time=gen_time
            )

        except AttributeError as e:
            # Handle None type errors specifically
            if "'NoneType'" in str(e):
                raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} not properly initialized: {str(e)}")
            else:
                raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} configuration error: {str(e)}")
        except Exception as e:
            error_str = str(e).lower()
            if ("not found" in error_str) and ("model" in error_str):
                self._raise_model_not_found()
            raise

    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response"""
        request_url = f"{self.base_url}/chat/completions"

        with self.client.stream(
            "POST",
            request_url,
            json=payload,
            headers=self._get_headers()
        ) as response:
            status0 = getattr(response, "status_code", None)
            if status0 is not None and int(status0) >= 400:
                try:
                    response.read()  # buffer the error body so detail extraction can parse it
                except Exception:
                    pass
                if self._is_prompt_cache_key_rejection(response, payload):
                    self._mark_prompt_cache_key_unsupported()
                    retry_payload = {k: v for k, v in payload.items() if k != "prompt_cache_key"}
                    yield from self._stream_generate(retry_payload)
                    return
                if self._is_stream_options_rejection(response, payload):
                    self._mark_stream_options_unsupported()
                    retry_payload = {k: v for k, v in payload.items() if k != "stream_options"}
                    yield from self._stream_generate(retry_payload)
                    return
            self._raise_for_status(response, request_url=request_url)

            for line in response.iter_lines():
                if line:
                    # Decode bytes to string if necessary
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    line = line.strip()

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)

                            # In-stream ERROR events must be LOUD. Servers can
                            # fail mid-generation (live find: LM Studio evicted
                            # the model mid-stream and sent
                            # `data: {"error": {"message": "Model unloaded."}}`);
                            # silently dropping the event ends the stream looking
                            # like a normal stop — the consumer keeps a TRUNCATED
                            # answer with no signal anything went wrong.
                            self._raise_on_stream_error_event(chunk)

                            # Usage may ride the last content chunk (LM Studio style)
                            # or a final chunk with EMPTY choices (OpenAI
                            # stream_options style) — capture both.
                            chunk_usage = chunk.get("usage")
                            usage_dict = (
                                self._build_usage_dict(chunk_usage)
                                if isinstance(chunk_usage, dict) and chunk_usage
                                else None
                            )

                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                if not isinstance(delta, dict):
                                    delta = {}
                                content = delta.get("content", "")
                                reasoning = extract_reasoning_from_message(
                                    delta,
                                    architecture_format=self.architecture_config,
                                    model_capabilities=self.model_capabilities,
                                )
                                tool_calls = delta.get("tool_calls") or choice.get("tool_calls")
                                finish_reason = choice.get("finish_reason")

                                metadata = {}
                                if isinstance(reasoning, str) and reasoning.strip():
                                    metadata["reasoning"] = reasoning

                                yield GenerateResponse(
                                    content=content,
                                    model=self.model,
                                    finish_reason=finish_reason,
                                    tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                                    metadata=metadata or None,
                                    usage=usage_dict,
                                    raw_response=chunk
                                )
                            elif usage_dict:
                                # Usage-only final chunk: no content, but the
                                # accounting must reach the consumer.
                                yield GenerateResponse(
                                    content="",
                                    model=self.model,
                                    finish_reason=None,
                                    usage=usage_dict,
                                    raw_response=chunk
                                )

                        except json.JSONDecodeError:
                            continue

    def _raise_on_stream_error_event(self, chunk: Any) -> None:
        """Raise ProviderAPIError for an SSE data event that carries an error.

        Mid-stream server failures (model eviction, backend crash) arrive as
        `{"error": ...}` events with no choices and no usage. They classify as
        ProviderAPIError — the transient class RetryManager resamples — never
        as a silent end-of-stream.
        """
        if not isinstance(chunk, dict):
            return
        err = chunk.get("error")
        if not err:
            return
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("error") or err)
        else:
            message = str(err)
        raise ProviderAPIError(
            f"{self.__class__.__name__.replace('Provider', '')} stream failed mid-generation: {message}"
        )

    async def _agenerate_internal(self,
                                   prompt: str,
                                   messages: Optional[List[Dict[str, str]]] = None,
                                   system_prompt: Optional[str] = None,
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   media: Optional[List['MediaContent']] = None,
                                   stream: bool = False,
                                   response_model: Optional[Type[BaseModel]] = None,
                                   execute_tools: Optional[bool] = None,
                                   tool_call_tags: Optional[str] = None,
                                   **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Native async implementation using httpx.AsyncClient - 3-10x faster for batch operations."""

        # Build messages for chat completions with tool support (same logic as sync)
        chat_messages = []

        # Add tools to system prompt if provided
        final_system_prompt = system_prompt
        # Prefer native tools when available; only inject prompted tool syntax as fallback.
        if tools and self.tool_handler.supports_prompted and not self.tool_handler.supports_native:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt += f"\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Add system message if provided
        if final_system_prompt:
            chat_messages.append({
                "role": "system",
                "content": final_system_prompt
            })

        # Add conversation history
        if messages:
            chat_messages.extend(messages)

        # Handle media content
        if media:
            user_message_text = prompt.strip() if prompt else ""
            if not user_message_text and chat_messages:
                for msg in reversed(chat_messages):
                    if msg.get("role") == "user" and msg.get("content"):
                        user_message_text = msg["content"]
                        break
            try:
                processed_media = self._process_media_content(media)
                media_handler = self._get_media_handler_for_model(self.model)
                multimodal_message = media_handler.create_multimodal_message(user_message_text, processed_media)

                if isinstance(multimodal_message, str):
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1]["content"] = multimodal_message
                    else:
                        chat_messages.append({"role": "user", "content": multimodal_message})
                else:
                    if chat_messages and chat_messages[-1].get("role") == "user":
                        chat_messages[-1] = multimodal_message
                    else:
                        chat_messages.append(multimodal_message)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                if user_message_text:
                    chat_messages.append({"role": "user", "content": user_message_text})
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                if user_message_text:
                    chat_messages.append({"role": "user", "content": user_message_text})

        # Add prompt as separate message if provided
        elif prompt and prompt.strip():
            chat_messages.append({"role": "user", "content": prompt})

        # Parity with the sync path (adversarial-review find: this fallback existed only in
        # sync): some servers' templates fail on system-only requests ("No user query found").
        if not any(isinstance(m, dict) and m.get("role") == "user" for m in chat_messages):
            fallback = str(prompt or "").strip() or "Continue."
            chat_messages.append({"role": "user", "content": fallback})

        # Strict-server system-message normalization (must run after ALL message building).
        chat_messages = self._normalize_system_messages_for_strict_servers(chat_messages)

        # Build request payload
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "max_tokens": max_output_tokens,
            "top_p": generation_kwargs.get("top_p", 0.9),
        }
        top_k_value = generation_kwargs.get("top_k")
        if top_k_value is not None and str(getattr(self, "provider", "") or "").strip().lower() == "lmstudio":
            payload["top_k"] = top_k_value

        # Streamed usage accounting (sync-path parity).
        if stream and not getattr(self, "_stream_options_unsupported", False):
            payload["stream_options"] = {"include_usage": True}

        # Prompt caching (best-effort) — sync-path parity (adversarial-review find:
        # the async builder silently dropped `prompt_cache_key`, so async callers
        # lost session cache identity). Same rejection-latch semantics as sync.
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if (
            isinstance(prompt_cache_key, str)
            and prompt_cache_key.strip()
            and not getattr(self, "_prompt_cache_key_unsupported", False)
        ):
            payload["prompt_cache_key"] = prompt_cache_key.strip()

        # Native tools (OpenAI-compatible): send structured tools/tool_choice when supported.
        if tools and self.tool_handler.supports_native:
            payload["tools"] = self.tool_handler.prepare_tools_for_native(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add additional parameters
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]

        # Add seed if provided
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            payload["seed"] = seed_value

        # Add structured output support
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            if isinstance(json_schema, dict) and json_schema:
                json_schema = _inline_json_schema_refs(json_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Registry-driven parameter constraints (unsupported_parameters / token_param_name)
        payload = self._apply_model_parameter_constraints(payload)

        # Provider-specific request extensions (vLLM extra_body, OpenRouter headers, etc.)
        payload = self._mutate_payload(payload, **kwargs)

        if stream:
            return self._async_stream_generate(payload)
        else:
            response = await self._async_single_generate(payload)

            # Execute tools if enabled
            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    async def _async_single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Native async single response generation."""
        try:
            # Track generation time
            start_time = time.time()
            request_url = f"{self.base_url}/chat/completions"
            response = await self.async_client.post(
                request_url,
                json=payload,
                headers=self._get_headers()
            )
            if self._is_prompt_cache_key_rejection(response, payload):
                self._mark_prompt_cache_key_unsupported()
                payload = {k: v for k, v in payload.items() if k != "prompt_cache_key"}
                response = await self.async_client.post(request_url, json=payload, headers=self._get_headers())
            self._raise_for_status(response, request_url=request_url)
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            # Extract response from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    message = {}

                content = message.get("content", "")
                reasoning = extract_reasoning_from_message(
                    message,
                    architecture_format=self.architecture_config,
                    model_capabilities=self.model_capabilities,
                )
                tool_calls = message.get("tool_calls")
                if tool_calls is None:
                    tool_calls = choice.get("tool_calls")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                reasoning = None
                tool_calls = None
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            metadata: Dict[str, Any] = {
                "_provider_request": {
                    "url": request_url,
                    "payload": payload,
                }
            }
            if isinstance(reasoning, str) and reasoning.strip():
                metadata["reasoning"] = reasoning

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                metadata=metadata,
                usage=self._build_usage_dict(usage),
                gen_time=gen_time
            )

        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            if ("not found" in error_str) and ("model" in error_str):
                self._raise_model_not_found()
            raise

    async def _async_stream_generate(self, payload: Dict[str, Any]) -> AsyncIterator[GenerateResponse]:
        """Native async streaming response generation."""
        request_url = f"{self.base_url}/chat/completions"

        async with self.async_client.stream(
            "POST",
            request_url,
            json=payload,
            headers=self._get_headers()
        ) as response:
            status0 = getattr(response, "status_code", None)
            if status0 is not None and int(status0) >= 400:
                try:
                    await response.aread()  # buffer the error body so detail extraction can parse it
                except Exception:
                    pass
                if self._is_prompt_cache_key_rejection(response, payload):
                    self._mark_prompt_cache_key_unsupported()
                    retry_payload = {k: v for k, v in payload.items() if k != "prompt_cache_key"}
                    async for chunk in self._async_stream_generate(retry_payload):
                        yield chunk
                    return
                if self._is_stream_options_rejection(response, payload):
                    self._mark_stream_options_unsupported()
                    retry_payload = {k: v for k, v in payload.items() if k != "stream_options"}
                    async for chunk in self._async_stream_generate(retry_payload):
                        yield chunk
                    return
            self._raise_for_status(response, request_url=request_url)

            async for line in response.aiter_lines():
                if line:
                    line = line.strip()

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)

                            # Mid-stream error events raise loudly (sync parity).
                            self._raise_on_stream_error_event(chunk)

                            # Usage on the last content chunk OR a final
                            # empty-choices chunk (stream_options) — sync parity.
                            chunk_usage = chunk.get("usage")
                            usage_dict = (
                                self._build_usage_dict(chunk_usage)
                                if isinstance(chunk_usage, dict) and chunk_usage
                                else None
                            )

                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                if not isinstance(delta, dict):
                                    delta = {}
                                content = delta.get("content", "")
                                reasoning = extract_reasoning_from_message(
                                    delta,
                                    architecture_format=self.architecture_config,
                                    model_capabilities=self.model_capabilities,
                                )
                                tool_calls = delta.get("tool_calls") or choice.get("tool_calls")
                                finish_reason = choice.get("finish_reason")

                                metadata = {}
                                if isinstance(reasoning, str) and reasoning.strip():
                                    metadata["reasoning"] = reasoning

                                yield GenerateResponse(
                                    content=content,
                                    model=self.model,
                                    finish_reason=finish_reason,
                                    tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                                    metadata=metadata or None,
                                    usage=usage_dict,
                                    raw_response=chunk
                                )
                            elif usage_dict:
                                yield GenerateResponse(
                                    content="",
                                    model=self.model,
                                    finish_reason=None,
                                    usage=usage_dict,
                                    raw_response=chunk
                                )

                        except json.JSONDecodeError:
                            continue

    def supports_prompt_cache(self) -> bool:
        """Best-effort: forward `prompt_cache_key` to OpenAI-compatible servers that support it."""
        return True

    def get_capabilities(self) -> List[str]:
        """Get OpenAI-compatible server capabilities"""
        return ["streaming", "chat", "tools"]

    def validate_config(self) -> bool:
        """Validate OpenAI-compatible server connection"""
        try:
            response = self.client.get(f"{self.base_url}/models", headers=self._get_headers())
            return response.status_code == 200
        except:
            return False

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for OpenAI-compatible API"""
        # For OpenAI-compatible servers, max_tokens is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout when timeout is changed."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                # Create new client with updated timeout
                self.client.close()

                # Get timeout value - None means unlimited timeout
                timeout_value = getattr(self, '_timeout', None)
                # Validate timeout if provided (None is allowed for unlimited)
                if timeout_value is not None and timeout_value <= 0:
                    timeout_value = None  # Invalid timeout becomes unlimited

                self.client = httpx.Client(timeout=timeout_value)
            except Exception as e:
                # Log error but don't fail - timeout update is not critical
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Failed to update HTTP client timeout: {e}")
                # Try to create a new client with default timeout
                try:
                    fallback_timeout = None
                    try:
                        from ..config.manager import get_config_manager

                        fallback_timeout = float(get_config_manager().get_default_timeout())
                    except Exception:
                        fallback_timeout = 7200.0
                    if isinstance(fallback_timeout, (int, float)) and float(fallback_timeout) <= 0:
                        fallback_timeout = None
                    self.client = httpx.Client(timeout=fallback_timeout)
                except Exception:
                    pass  # Best effort - don't fail the operation

    def _get_media_handler_for_model(self, model_name: str):
        """Get appropriate media handler based on model vision capabilities."""
        from ..media.handlers import OpenAIMediaHandler, LocalMediaHandler

        # Determine if model supports vision
        try:
            from ..architectures.detection import supports_vision
            use_vision_handler = supports_vision(model_name)
        except Exception as e:
            self.logger.debug(f"Vision detection failed: {e}, defaulting to LocalMediaHandler")
            use_vision_handler = False

        # Create appropriate handler
        if use_vision_handler:
            handler = OpenAIMediaHandler(self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using OpenAIMediaHandler for vision model: {model_name}")
        else:
            handler = LocalMediaHandler(self.provider, self.model_capabilities, model_name=model_name)
            self.logger.debug(f"Using LocalMediaHandler for model: {model_name}")

        return handler

    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models from OpenAI-compatible server.

        Args:
            **kwargs: Optional parameters including:
                - base_url: Server URL
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            # Use provided base_url or fall back to instance base_url
            base_url = kwargs.get('base_url', self.base_url)
            raise_on_error = bool(kwargs.get("raise_on_error", False))

            response = self.client.get(f"{base_url}/models", headers=self._get_headers(), timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                models = sorted(models)

                # Apply capability filtering if provided
                input_capabilities = kwargs.get('input_capabilities')
                output_capabilities = kwargs.get('output_capabilities')
                capability_routes = kwargs.get('capability_routes')

                if input_capabilities or output_capabilities or capability_routes:
                    models = filter_models_by_capabilities(
                        models,
                        input_capabilities=input_capabilities,
                        output_capabilities=output_capabilities,
                        capability_routes=capability_routes,
                    )

                return models
            else:
                if raise_on_error:
                    self._raise_for_status(response, request_url=f"{base_url}/models")

                detail = self._extract_error_detail(response)
                suffix = f": {detail}" if detail else ""
                self.logger.warning(f"{self.PROVIDER_DISPLAY_NAME} /models returned {response.status_code}{suffix}")
                return []
        except Exception as e:
            if bool(kwargs.get("raise_on_error", False)):
                raise
            self.logger.warning(f"Failed to list models from {self.PROVIDER_DISPLAY_NAME}: {e}")
            return []

    def embed(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings using OpenAI-compatible embedding API.

        Args:
            input_text: Single string or list of strings to embed
            **kwargs: Additional parameters (encoding_format, dimensions, user, etc.)

        Returns:
            Dict with embeddings in OpenAI-compatible format:
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
                "model": "model-name",
                "usage": {"prompt_tokens": N, "total_tokens": N}
            }
        """
        try:
            # Prepare request payload for OpenAI-compatible API
            payload = {
                "input": input_text,
                "model": self.model
            }

            # Add optional parameters if provided
            if "encoding_format" in kwargs:
                payload["encoding_format"] = kwargs["encoding_format"]
            if "dimensions" in kwargs and kwargs["dimensions"]:
                payload["dimensions"] = kwargs["dimensions"]
            if "user" in kwargs:
                payload["user"] = kwargs["user"]

            # Call server's embeddings API (OpenAI-compatible)
            response = self.client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self._get_headers()
            )
            self._raise_for_status(response, request_url=f"{self.base_url}/embeddings")

            # Server returns OpenAI-compatible format
            result = response.json()

            # Ensure the model field uses our provider-prefixed format
            result["model"] = self.model

            return result

        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} embedding error: {str(e)}")

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
        **kwargs: Any,
    ) -> Tuple[bytes, str]:
        """Transcribe audio through an OpenAI-compatible `/audio/transcriptions` endpoint.

        Returns raw response bytes and upstream content type so callers can
        preserve JSON and non-JSON transcription response formats.
        """
        try:
            data: Dict[str, Any] = {"model": self.model}
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            if response_format:
                data["response_format"] = response_format
            if temperature is not None:
                data["temperature"] = temperature
            for key in ("timestamp_granularities", "stream"):
                value = kwargs.get(key)
                if value is not None:
                    data[key] = value

            response = self.client.post(
                f"{self.base_url}/audio/transcriptions",
                data=data,
                files={"file": (filename, audio, content_type or "application/octet-stream")},
                headers=self._get_headers(),
            )
            self._raise_for_status(response, request_url=f"{self.base_url}/audio/transcriptions")
            return response.content, response.headers.get("content-type", "application/json")
        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {e}")
            raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} audio transcription error: {str(e)}")

    def synthesize_speech(
        self,
        input_text: str,
        *,
        voice: str,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[bytes, str]:
        """Generate speech through an OpenAI-compatible `/audio/speech` endpoint."""
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "input": input_text,
                "voice": voice,
            }
            if response_format:
                payload["response_format"] = response_format
            if speed is not None:
                payload["speed"] = speed
            if instructions:
                payload["instructions"] = instructions
            provider_options = kwargs.get("provider")
            if isinstance(provider_options, dict):
                payload["provider"] = provider_options

            response = self.client.post(
                f"{self.base_url}/audio/speech",
                json=payload,
                headers=self._get_headers(),
            )
            self._raise_for_status(response, request_url=f"{self.base_url}/audio/speech")
            return response.content, response.headers.get("content-type", "application/octet-stream")
        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to synthesize speech: {e}")
            raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} audio speech error: {str(e)}")

    def clone_voice(
        self,
        audio: bytes,
        *,
        filename: str = "reference.wav",
        content_type: str = "audio/wav",
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        clone_path: str = "/voice/clone",
        file_field: str = "file",
        validate: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a cloned voice through an OpenAI-compatible voice-clone endpoint.

        Voice cloning is an AbstractVoice-compatible extension, not part of the
        core OpenAI audio API. Compatible servers commonly expose
        `/v1/voice/clone` and return either `id` or `voice_id`.
        """
        try:
            fields: Dict[str, str] = {}
            if name:
                fields["name"] = str(name)
            if reference_text:
                fields["reference_text"] = str(reference_text)
            if validate is not None:
                fields["validate"] = "true" if bool(validate) else "false"
            for key in ("consent",):
                value = kwargs.get(key)
                if value is not None:
                    fields[key] = str(value)

            path = str(clone_path or "/voice/clone").strip()
            if not path.startswith("/"):
                path = "/" + path
            field_name = str(file_field or "file").strip() or "file"

            response = self.client.post(
                f"{self.base_url.rstrip('/')}{path}",
                data=fields,
                files={field_name: (filename, bytes(audio), content_type or "audio/wav")},
                headers=self._get_headers(),
            )
            self._raise_for_status(response, request_url=f"{self.base_url.rstrip('/')}{path}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderAPIError("Voice clone response was not a JSON object")
            return payload
        except (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError, ProviderAPIError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to clone voice: {e}")
            raise ProviderAPIError(f"{self.PROVIDER_DISPLAY_NAME} voice clone error: {str(e)}")
