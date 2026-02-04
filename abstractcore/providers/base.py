"""
Base provider with integrated telemetry, events, and exception handling.
"""

import time
import uuid
import asyncio
import warnings
import json
import re
import socket
import hashlib
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Type, TYPE_CHECKING, Tuple
from abc import ABC, abstractmethod

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

from ..core.interface import AbstractCoreInterface
from ..core.types import GenerateResponse
from ..events import EventType, Event
from datetime import datetime
from ..utils.structured_logging import get_logger
from ..utils.jsonish import loads_dict_like
from ..utils.truncation import preview_text
from ..exceptions import (
    ProviderAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    UnsupportedFeatureError,
    ModelNotFoundError
)
from ..architectures import detect_architecture, get_architecture_format, get_model_capabilities
from ..architectures.response_postprocessing import (
    normalize_assistant_text,
    strip_output_wrappers,
)
from ..tools import execute_tools
from ..core.retry import RetryManager, RetryConfig

if TYPE_CHECKING:  # pragma: no cover
    # Imported for type checking only to avoid hard dependencies in minimal installs.
    from ..media.types import MediaContent


@dataclass
class _PromptCacheEntry:
    value: Any
    created_at_s: float
    last_accessed_at_s: float
    ttl_s: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class PromptCacheStore:
    """Best-effort in-process prompt cache store (LRU + optional TTL).

    Providers can store arbitrary backend-specific cache objects keyed by a caller-provided string
    (`prompt_cache_key`). This is primarily useful for local inference backends (MLX, llama.cpp).

    Notes:
    - This store is intentionally simple and in-process only.
    - Callers should treat prompt caches as potentially sensitive (they contain user prompt state).
    """

    def __init__(self, *, max_entries: int = 32, default_ttl_s: Optional[float] = None):
        self._max_entries = int(max_entries) if max_entries and int(max_entries) > 0 else 32
        self._default_ttl_s = default_ttl_s if default_ttl_s is None else float(default_ttl_s)
        self._entries: "OrderedDict[str, _PromptCacheEntry]" = OrderedDict()

    def _is_expired(self, entry: _PromptCacheEntry) -> bool:
        ttl_s = entry.ttl_s if entry.ttl_s is not None else self._default_ttl_s
        if ttl_s is None:
            return False
        return (time.time() - entry.last_accessed_at_s) > float(ttl_s)

    def get(self, key: str) -> Optional[Any]:
        if not isinstance(key, str) or not key.strip():
            return None
        key = key.strip()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            self.delete(key)
            return None
        entry.last_accessed_at_s = time.time()
        self._entries.move_to_end(key)
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_s: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("prompt cache key must be a non-empty string")
        key = key.strip()
        now = time.time()
        self._entries[key] = _PromptCacheEntry(
            value=value,
            created_at_s=now,
            last_accessed_at_s=now,
            ttl_s=ttl_s,
            meta=dict(meta or {}),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def delete(self, key: str) -> bool:
        if not isinstance(key, str) or not key.strip():
            return False
        key = key.strip()
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> Dict[str, Any]:
        # Opportunistically purge expired entries.
        expired = []
        for k, v in self._entries.items():
            if self._is_expired(v):
                expired.append(k)
        for k in expired:
            self.delete(k)

        return {
            "entries": len(self._entries),
            "max_entries": self._max_entries,
            "default_ttl_s": self._default_ttl_s,
        }

    def keys(self) -> List[str]:
        return list(self._entries.keys())

    def meta(self, key: str) -> Optional[Dict[str, Any]]:
        if not isinstance(key, str) or not key.strip():
            return None
        entry = self._entries.get(key.strip())
        if entry is None:
            return None
        return dict(entry.meta or {})


@dataclass(frozen=True)
class PromptCacheModule:
    """A single cacheable module of prompt context.

    This is intentionally generic and JSON-serializable so higher-level layers (runtime/agent/memory)
    can express cache intent without hard-coding provider-specific prompt formats.
    """

    module_id: str
    system_prompt: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: bool = False
    scope: str = "private"  # "private" | "shared" (advisory; enforcement is host-dependent)
    meta: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "PromptCacheModule":
        module_id = str(self.module_id or "").strip()
        system_prompt = str(self.system_prompt).strip() if isinstance(self.system_prompt, str) and self.system_prompt else None
        prompt = str(self.prompt).strip() if isinstance(self.prompt, str) and self.prompt else None
        messages = None
        if isinstance(self.messages, list) and self.messages:
            out: List[Dict[str, Any]] = []
            for m in self.messages:
                if isinstance(m, dict):
                    out.append(dict(m))
            messages = out or None
        tools = None
        if isinstance(self.tools, list) and self.tools:
            out_tools: List[Dict[str, Any]] = []
            for t in self.tools:
                if isinstance(t, dict):
                    out_tools.append(dict(t))
            tools = out_tools or None
        add_generation_prompt = bool(self.add_generation_prompt)
        scope = str(self.scope or "private").strip().lower() or "private"
        if scope not in {"private", "shared"}:
            scope = "private"
        meta = dict(self.meta or {})
        return PromptCacheModule(
            module_id=module_id,
            system_prompt=system_prompt,
            prompt=prompt,
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            scope=scope,
            meta=meta,
        )

    def fingerprint(self, *, version: int = 1) -> str:
        """Stable module fingerprint for hierarchical cache keys (hex sha256)."""
        mod = self.normalized()
        payload = {
            "v": int(version),
            "module_id": mod.module_id,
            "system_prompt": mod.system_prompt,
            "prompt": mod.prompt,
            "messages": mod.messages,
            "tools": mod.tools,
            "add_generation_prompt": bool(mod.add_generation_prompt),
            "scope": mod.scope,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BaseProvider(AbstractCoreInterface, ABC):
    """
    Base provider class with integrated telemetry and events.
    All providers should inherit from this class.
    """

    def __init__(self, model: str, **kwargs):
        AbstractCoreInterface.__init__(self, model, **kwargs)
        self.provider = None

        # Setup structured logging
        self.logger = get_logger(self.__class__.__name__)

        # Detect architecture and get model capabilities
        self.architecture = detect_architecture(model)
        self.architecture_config = get_architecture_format(self.architecture)
        self.model_capabilities = get_model_capabilities(model)

        # #[WARNING:TIMEOUT]
        # Setup timeout configuration (centralized defaults).
        #
        # Semantics:
        # - If the caller passes `timeout=...`, we respect it (including `None` for unlimited).
        # - If the caller omits `timeout`, we use AbstractCore's global config default.
        # - Same logic for `tool_timeout`.
        timeout_provided = "timeout" in kwargs
        tool_timeout_provided = "tool_timeout" in kwargs

        timeout_value = kwargs.get("timeout", None) if timeout_provided else None
        tool_timeout_value = kwargs.get("tool_timeout", None) if tool_timeout_provided else None

        if not timeout_provided or not tool_timeout_provided:
            try:
                from ..config.manager import get_config_manager

                cfg = get_config_manager()
            except Exception:
                cfg = None

            if not timeout_provided:
                try:
                    timeout_value = float(cfg.get_default_timeout()) if cfg is not None else None
                except Exception:
                    timeout_value = None

            if not tool_timeout_provided:
                try:
                    tool_timeout_value = float(cfg.get_tool_timeout()) if cfg is not None else None
                except Exception:
                    tool_timeout_value = None

        # Validate timeouts: non-positive numbers become "unlimited" (None).
        try:
            if isinstance(timeout_value, (int, float)) and float(timeout_value) <= 0:
                timeout_value = None
        except Exception:
            pass
        try:
            if isinstance(tool_timeout_value, (int, float)) and float(tool_timeout_value) <= 0:
                tool_timeout_value = None
        except Exception:
            pass

        self._timeout = timeout_value  # None = unlimited HTTP requests
        self._tool_timeout = tool_timeout_value  # None = unlimited tool execution

        # Setup tool execution mode
        # execute_tools: True = AbstractCore executes tools (legacy mode)
        #                False = Pass-through mode (default - for API server / agentic CLI)
        self.execute_tools = kwargs.get('execute_tools', False)
        if self.execute_tools:
            warnings.warn(
                "execute_tools=True is deprecated. Prefer passing tools explicitly to generate() "
                "and executing tool calls in the host/runtime via a ToolExecutor.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Setup retry manager with optional configuration
        retry_config = kwargs.get('retry_config', None)
        if retry_config is None:
            # Use default retry configuration
            retry_config = RetryConfig()
        self.retry_manager = RetryManager(retry_config)

        # Create provider key for circuit breaker tracking
        self.provider_key = f"{self.__class__.__name__}:{self.model}"
        
        # Setup Glyph compression configuration
        self.glyph_config = kwargs.get('glyph_config', None)

        # Setup interaction tracing
        self.enable_tracing = kwargs.get('enable_tracing', False)
        self._traces = deque(maxlen=kwargs.get('max_traces', 100))  # Ring buffer for memory efficiency

        # Prompt caching (best-effort; provider-specific behavior).
        #
        # - Remote providers (OpenAI): supports `prompt_cache_key` pass-through (server-managed caching).
        # - Local runtimes (MLX / llama.cpp): can store KV/prefix caches in-process keyed by `prompt_cache_key`.
        self._default_prompt_cache_key: Optional[str] = None
        prompt_cache_max_entries = kwargs.get("prompt_cache_max_entries", kwargs.get("prompt_cache_max_items", 32))
        prompt_cache_ttl_s = kwargs.get("prompt_cache_ttl_s", None)
        self._prompt_cache_store = PromptCacheStore(
            max_entries=int(prompt_cache_max_entries) if prompt_cache_max_entries is not None else 32,
            default_ttl_s=prompt_cache_ttl_s,
        )

        # Provider created successfully - no event emission needed
        # (The simplified event system focuses on generation and tool events only)

        # Set default token limits if not provided
        self._initialize_token_limits()

    def __init_subclass__(cls, **kwargs):  # pragma: no cover
        super().__init_subclass__(**kwargs)
        # Enforce a single unload path: providers must implement `unload_model()` and must not define `unload()`.
        if "unload" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} defines unload(). "
                "Providers must implement unload_model(model_name) and must not provide any other unload entrypoint."
            )

    def _track_generation(self, prompt: str, response: Optional[GenerateResponse],
                         start_time: float, success: bool = True,
                         error: Optional[Exception] = None, stream: bool = False):
        """
        Track generation with telemetry and events.

        Args:
            prompt: Input prompt
            response: Generated response
            start_time: Generation start time
            success: Whether generation succeeded
            error: Error if failed
            stream: Whether this was a streaming generation
        """
        latency_ms = (time.time() - start_time) * 1000

        # Extract token and cost information from response
        tokens_input = None
        tokens_output = None
        cost_usd = None

        if response and response.usage:
            tokens_input = response.usage.get('prompt_tokens') or response.usage.get('input_tokens')
            tokens_output = response.usage.get('completion_tokens') or response.usage.get('output_tokens')
            # Calculate cost if possible (simplified - could be enhanced)
            total_tokens = response.usage.get('total_tokens', 0)
            if total_tokens > 0:
                # Very rough cost estimation - should be provider-specific
                cost_usd = total_tokens * 0.00002  # ~$0.02 per 1K tokens average

        # Emit comprehensive event with all data in one dict
        event_data = {
            "prompt": preview_text(prompt, max_chars=100),
            "success": success,
            "error": str(error) if error else None,
            "response_length": len(response.content) if response and response.content else 0,
            "stream": stream,
            "model": self.model,
            "provider": self.__class__.__name__,
            "duration_ms": latency_ms,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost_usd": cost_usd
        }

        from ..events import emit_global
        emit_global(EventType.GENERATION_COMPLETED, event_data, source=self.__class__.__name__)

        # Track with structured logging (using formatted strings)
        if error:
            # Only log debug info for model not found errors to avoid duplication
            if isinstance(error, ModelNotFoundError):
                self.logger.debug(f"Model not found: {self.model}")
            else:
                self.logger.error(f"Generation failed for {self.model}: {error} (latency: {latency_ms:.2f}ms)")
        else:
            tokens_info = ""
            if response and response.usage:
                tokens_info = f" (tokens: {response.usage.get('total_tokens', 0)})"

            self.logger.info(f"Generation completed for {self.model}: {latency_ms:.2f}ms{tokens_info}")

    def _track_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                        result: Optional[Any] = None, success: bool = True,
                        error: Optional[Exception] = None):
        """
        Track tool call with telemetry and events.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            success: Whether call succeeded
            error: Error if failed
        """
        # Emit comprehensive tool event
        event_type = EventType.TOOL_COMPLETED if success else EventType.ERROR
        event_data = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": preview_text(result, max_chars=100) if result else None,
            "error": str(error) if error else None,
            "success": success
        }

        # Add model and provider to event data
        event_data["model"] = self.model
        event_data["provider"] = self.__class__.__name__

        from ..events import emit_global
        emit_global(event_type, event_data, source=self.__class__.__name__)

        # Track with structured logging (using formatted strings)
        if error:
            self.logger.warning(f"Tool call failed: {tool_name} - {error}")
        else:
            result_info = f" (result length: {len(str(result))})" if result else ""
            self.logger.info(f"Tool call completed: {tool_name}{result_info}")

    def _capture_trace(self, prompt: str, messages: Optional[List[Dict[str, str]]],
                       system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]],
                       response: GenerateResponse, kwargs: Dict[str, Any]) -> str:
        """
        Capture interaction trace for observability.

        Args:
            prompt: Input prompt
            messages: Conversation history
            system_prompt: System prompt
            tools: Available tools
            response: Generated response
            kwargs: Additional generation parameters

        Returns:
            Trace ID (UUID string)
        """
        trace_id = str(uuid.uuid4())

        # If trace retention is disabled, still return a trace_id for correlation
        # without constructing/storing a full trace payload.
        maxlen = getattr(getattr(self, "_traces", None), "maxlen", None)
        if maxlen == 0:
            return trace_id

        # Extract generation parameters
        temperature = kwargs.get('temperature', self.temperature)
        if temperature is None:
            temperature = self.temperature
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        max_output_tokens = kwargs.get('max_output_tokens', self.max_output_tokens)
        seed = self._normalize_seed(kwargs.get('seed', self.seed))
        top_p = kwargs.get('top_p', getattr(self, 'top_p', None))
        top_k = kwargs.get('top_k', getattr(self, 'top_k', None))

        # Build parameters dict
        parameters = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'max_output_tokens': max_output_tokens,
        }
        if seed is not None:
            parameters['seed'] = seed
        if top_p is not None:
            parameters['top_p'] = top_p
        if top_k is not None:
            parameters['top_k'] = top_k

        # Create trace record
        trace = {
            'trace_id': trace_id,
            'timestamp': datetime.now().isoformat(),
            'provider': self.__class__.__name__,
            'model': self.model,
            'system_prompt': system_prompt,
            'prompt': prompt,
            'messages': messages,
            'tools': tools,
            'parameters': parameters,
            'response': {
                'content': response.content,
                'raw_response': None,  # Omit raw_response to save memory and avoid logging sensitive data
                'tool_calls': response.tool_calls,
                'finish_reason': response.finish_reason,
                'usage': response.usage,
                'generation_time_ms': response.gen_time,
            },
            'metadata': kwargs.get('trace_metadata', {})
        }

        # Store trace in ring buffer
        self._traces.append(trace)

        return trace_id

    def get_traces(self, trace_id: Optional[str] = None, last_n: Optional[int] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Retrieve interaction traces.

        Args:
            trace_id: Optional specific trace ID to retrieve
            last_n: Optional number of most recent traces to retrieve

        Returns:
            Single trace dict if trace_id provided, list of traces otherwise
        """
        if trace_id:
            # Find specific trace by ID
            for trace in self._traces:
                if trace['trace_id'] == trace_id:
                    return trace
            return None

        if last_n:
            # Return last N traces
            return list(self._traces)[-last_n:] if len(self._traces) >= last_n else list(self._traces)

        # Return all traces
        return list(self._traces)

    def _handle_api_error(self, error: Exception) -> Exception:
        """
        Convert API errors to custom exceptions.

        Args:
            error: Original exception

        Returns:
            Custom exception
        """
        def _provider_label() -> str:
            raw = getattr(self, "provider", None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            name = self.__class__.__name__
            return name[:-8] if name.endswith("Provider") else name

        def _configured_timeout_s() -> Optional[float]:
            v = getattr(self, "_timeout", None)
            if v is None:
                return None
            try:
                f = float(v)
            except Exception:
                return None
            return f if f > 0 else None

        def _looks_like_timeout(exc: Exception) -> bool:
            # Type-based (preferred)
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError, socket.timeout)):
                return True
            cls = exc.__class__
            name = (getattr(cls, "__name__", "") or "").lower()
            mod = (getattr(cls, "__module__", "") or "").lower()
            if "timeout" in name:
                return True
            if mod.startswith(("httpx", "requests", "aiohttp")) and ("timeout" in name):
                return True

            # String-based fallback (covers wrapped SDK exceptions)
            msg = str(exc or "").lower()
            return ("timed out" in msg) or ("timeout" in msg) or ("time out" in msg)

        def _has_explicit_duration(msg: str) -> bool:
            # e.g. "... after 300s" or "... after 300.0s"
            return bool(re.search(r"\bafter\s+\d+(?:\.\d+)?\s*s\b", msg))

        # Preserve typed custom exceptions, but allow ProviderAPIError timeout messages
        # to be normalized centrally (avoid per-provider inconsistencies).
        if isinstance(error, ProviderAPIError):
            msg = str(error)
            if _looks_like_timeout(error) and not _has_explicit_duration(msg):
                t = _configured_timeout_s()
                if t is not None:
                    return ProviderAPIError(
                        f"{_provider_label()} API error: timed out after {t}s "
                        "(configured timeout; set timeout=None or default_timeout=0 for unlimited)"
                    )
                return ProviderAPIError(f"{_provider_label()} API error: timed out")
            return error

        if isinstance(error, (ModelNotFoundError, AuthenticationError, RateLimitError, InvalidRequestError)):
            return error

        # Central timeout normalization for all providers (httpx/requests/SDKs).
        if _looks_like_timeout(error):
            t = _configured_timeout_s()
            if t is not None:
                return ProviderAPIError(
                    f"{_provider_label()} API error: timed out after {t}s "
                    "(configured timeout; set timeout=None or default_timeout=0 for unlimited)"
                )
            return ProviderAPIError(f"{_provider_label()} API error: timed out")

        error_str = str(error).lower()

        if "rate" in error_str and "limit" in error_str:
            return RateLimitError(f"Rate limit exceeded: {error}")
        elif "auth" in error_str or "api key" in error_str or "unauthorized" in error_str:
            return AuthenticationError(f"Authentication failed: {error}")
        elif "invalid" in error_str or "bad request" in error_str:
            return InvalidRequestError(f"Invalid request: {error}")
        else:
            return ProviderAPIError(f"API error: {error}")

    @staticmethod
    def _normalize_thinking_request(thinking: Optional[Union[bool, str]]) -> Tuple[Optional[bool], Optional[str]]:
        """Normalize `thinking=` into (enabled, level).

        - enabled: True/False/None (None == "auto")
        - level: Optional[str] in {"low","medium","high"} when requested
        """
        if thinking is None:
            return None, None

        if isinstance(thinking, bool):
            return thinking, None

        if isinstance(thinking, str):
            s = thinking.strip().lower()
            if not s or s == "auto":
                return None, None
            if s in {"on", "true", "yes"}:
                return True, None
            if s in {"off", "false", "no"}:
                return False, None
            if s in {"low", "medium", "high"}:
                return True, s

        raise ValueError('thinking must be one of: None, bool, "auto", "on", "off", "low", "medium", "high"')

    def _model_reasoning_levels(self) -> List[str]:
        levels = None
        for src in (self.model_capabilities, self.architecture_config):
            if not isinstance(src, dict):
                continue
            value = src.get("reasoning_levels")
            if isinstance(value, list) and value:
                levels = value
                break
        if not isinstance(levels, list):
            return []
        out: List[str] = []
        for x in levels:
            if isinstance(x, str) and x.strip():
                out.append(x.strip().lower())
        # Deduplicate while preserving order.
        seen: set[str] = set()
        uniq: List[str] = []
        for x in out:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    def _model_supports_thinking_control(self) -> bool:
        caps = self.model_capabilities if isinstance(self.model_capabilities, dict) else {}
        arch = self.architecture_config if isinstance(self.architecture_config, dict) else {}

        if caps.get("thinking_support") is True:
            return True
        if isinstance(caps.get("thinking_tags"), (list, tuple)) and len(caps.get("thinking_tags")) == 2:
            return True
        if isinstance(caps.get("thinking_output_field"), str) and caps.get("thinking_output_field").strip():
            return True
        if self._model_reasoning_levels():
            return True

        if isinstance(arch.get("thinking_tags"), (list, tuple)) and len(arch.get("thinking_tags")) == 2:
            return True
        if isinstance(arch.get("thinking_control"), str) and arch.get("thinking_control").strip():
            return True
        if arch.get("reasoning_support") is True:
            return True
        if isinstance(arch.get("reasoning_levels"), list) and arch.get("reasoning_levels"):
            return True

        return False

    def _apply_thinking_request(
        self,
        *,
        thinking: Optional[Union[bool, str]],
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Tuple[str, Optional[List[Dict[str, str]]], Optional[str], Dict[str, Any]]:
        """Apply unified thinking controls to the request."""
        enabled, level = self._normalize_thinking_request(thinking)
        if enabled is None and level is None:
            return prompt, messages, system_prompt, kwargs

        supports_control = self._model_supports_thinking_control()
        reasoning_levels = self._model_reasoning_levels()

        if level is not None and reasoning_levels and level not in reasoning_levels:
            warnings.warn(
                f"thinking level '{level}' requested but not supported for model '{self.model}' "
                f"(supported: {reasoning_levels}); falling back to thinking='on'.",
                RuntimeWarning,
                stacklevel=3,
            )
            level = None
            enabled = True

        if level is not None and not reasoning_levels:
            warnings.warn(
                f"thinking level '{level}' requested but model '{self.model}' has no configured reasoning_levels; "
                "falling back to thinking='on'.",
                RuntimeWarning,
                stacklevel=3,
            )
            level = None
            enabled = True

        handled_by_model_prompt = False

        # Harmony (GPT-OSS): control via system message `Reasoning: low|medium|high`.
        msg_fmt = str((self.architecture_config or {}).get("message_format") or "").strip().lower()
        resp_fmt = str((self.model_capabilities or {}).get("response_format") or "").strip().lower()
        is_harmony = msg_fmt == "harmony" or resp_fmt == "harmony"
        if is_harmony:
            target_level: Optional[str] = None
            if level is not None:
                target_level = level
            elif enabled is False:
                warnings.warn(
                    f"thinking='off' requested for Harmony model '{self.model}', but GPT-OSS reasoning traces "
                    "cannot be fully disabled; using Reasoning: low.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                target_level = "low"
            elif enabled is True:
                # Make the default explicit when the caller opts-in.
                target_level = "medium"

            if target_level:
                line = f"Reasoning: {target_level}"
                if isinstance(system_prompt, str) and system_prompt.strip():
                    # Replace any existing Reasoning line; otherwise prepend.
                    if re.search(r"(?mi)^\\s*Reasoning\\s*:\\s*(low|medium|high)\\s*$", system_prompt):
                        system_prompt = re.sub(
                            r"(?mi)^\\s*Reasoning\\s*:\\s*(low|medium|high)\\s*$",
                            line,
                            system_prompt,
                            count=1,
                        )
                    else:
                        system_prompt = f"{line}\n{system_prompt}"
                else:
                    system_prompt = line
                handled_by_model_prompt = True

        # Model-level control token for disabling thinking (e.g., GLM `/nothink`).
        thinking_control = None
        for src in (self.model_capabilities, self.architecture_config):
            if not isinstance(src, dict):
                continue
            token = src.get("thinking_control")
            if isinstance(token, str) and token.strip():
                thinking_control = token.strip()

        if enabled is False and thinking_control:
            handled_by_model_prompt = True

            def _append_control(text: str) -> str:
                if thinking_control in text:
                    return text
                return f"{text.rstrip()}\n{thinking_control}".strip()

            if isinstance(prompt, str) and prompt.strip():
                prompt = _append_control(prompt)
            elif isinstance(messages, list) and messages:
                # Append to the most recent user turn, if possible.
                new_messages: List[Dict[str, str]] = []
                appended = False
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    new_messages.append(dict(m))
                for m in reversed(new_messages):
                    if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
                        m["content"] = _append_control(m["content"])
                        appended = True
                        break
                messages = new_messages
                if not appended:
                    warnings.warn(
                        f"thinking='off' requested for model '{self.model}', but no user prompt was available "
                        f"to append thinking_control='{thinking_control}'.",
                        RuntimeWarning,
                        stacklevel=3,
                    )

        kwargs, handled_by_provider = self._apply_provider_thinking_kwargs(
            enabled=enabled,
            level=level,
            kwargs=kwargs,
        )

        if not supports_control and thinking is not None:
            warnings.warn(
                f"thinking={thinking!r} requested but model '{self.model}' is not marked as thinking-capable "
                "in model_capabilities.json; the request may be ignored.",
                RuntimeWarning,
                stacklevel=3,
            )

        if not handled_by_model_prompt and not handled_by_provider and (enabled is False or level is not None):
            warnings.warn(
                f"thinking={thinking!r} requested but provider '{self.provider or self.__class__.__name__}' "
                "does not implement a thinking control mapping for this model; the request may be ignored.",
                RuntimeWarning,
                stacklevel=3,
            )

        return prompt, messages, system_prompt, kwargs

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """Provider-specific thinking knob hook (default: unsupported)."""
        _ = (enabled, level)
        return kwargs, False

    def generate_with_telemetry(self,
                               prompt: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               system_prompt: Optional[str] = None,
                               tools: Optional[List] = None,  # Accept both ToolDefinition and Dict
                               media: Optional[List[Union[str, Dict[str, Any], 'MediaContent']]] = None,  # Media files
                               stream: bool = False,
                               response_model: Optional[Type[BaseModel]] = None,
                               retry_strategy=None,  # Custom retry strategy for structured output
                               tool_call_tags: Optional[str] = None,  # Tool call tag rewriting
                               execute_tools: Optional[bool] = None,  # Tool execution control
                               glyph_compression: Optional[str] = None,  # Glyph compression preference
                               thinking: Optional[Union[bool, str]] = None,  # Unified reasoning/thinking control
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse], BaseModel]:
        """
        Generate with integrated telemetry and error handling.
        Providers should override _generate_internal instead of generate.

        Args:
            prompt: The input prompt
            messages: Optional conversation history
            system_prompt: Optional system prompt
            tools: Optional list of available tools
            media: Optional list of media files (file paths, MediaContent objects, or dicts)
            stream: Whether to stream the response
            response_model: Optional Pydantic model for structured output
            retry_strategy: Optional retry strategy for structured output validation
            tool_call_tags: Optional tool call tag format for rewriting
            execute_tools: Whether to execute tools automatically (True) or let agent handle execution (False)
            glyph_compression: Glyph compression preference ("auto", "always", "never")
            thinking: Unified reasoning/thinking control (auto/on/off or low/medium/high when supported)
        """
        # Normalize token limit naming at the provider boundary.
        #
        # - OpenAI-style APIs use `max_tokens` for the output-token cap.
        # - AbstractCore's unified internal name is `max_output_tokens`.
        #
        # AbstractRuntime (and some hosts) may still emit `max_tokens` in effect payloads.
        # That translation is a provider integration concern, so keep it in AbstractCore.
        if "max_output_tokens" not in kwargs and "max_tokens" in kwargs and kwargs.get("max_tokens") is not None:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        # Prompt caching: apply a default `prompt_cache_key` if configured.
        self._apply_default_prompt_cache_key(kwargs)

        # Apply unified thinking controls (provider-agnostic + provider-specific mappings).
        prompt, messages, system_prompt, kwargs = self._apply_thinking_request(
            thinking=thinking,
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            kwargs=kwargs,
        )

        # Handle structured output request
        if response_model is not None:
            if not PYDANTIC_AVAILABLE:
                raise ImportError(
                    "Pydantic is required for structured outputs. "
                    "Install with: pip install pydantic>=2.0.0"
                )

            # Handle hybrid case: tools + structured output.
            #
            # NOTE: `tools=[]` should behave like "no tools". Treating an empty list as
            # "tools present" triggers the hybrid 2-pass flow (unstructured call + structured
            # follow-up) which is both slower and can cause provider-side timeouts/unloads.
            if isinstance(tools, list) and len(tools) > 0:
                return self._handle_tools_with_structured_output(
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    media=media,
                    response_model=response_model,
                    retry_strategy=retry_strategy,
                    tool_call_tags=tool_call_tags,
                    execute_tools=execute_tools,
                    stream=stream,
                    **kwargs
                )

            # Standard structured output (no tools)
            from ..structured import StructuredOutputHandler
            handler = StructuredOutputHandler(retry_strategy=retry_strategy)
            return handler.generate_structured(
                provider=self,
                prompt=prompt,
                response_model=response_model,
                messages=messages,
                system_prompt=system_prompt,
                tools=None,  # No tools in this path
                media=media,
                stream=stream,
                **kwargs
            )

        # Process media content if provided
        processed_media = None
        media_metadata = None
        media_enrichment = None
        if media:
            compression_pref = glyph_compression or kwargs.get('glyph_compression', 'auto')
            processed_media = self._process_media_content(media, compression_pref)
            
            # Extract metadata from processed media for response
            if processed_media:
                media_metadata = []
                for media_content in processed_media:
                    if hasattr(media_content, 'metadata') and media_content.metadata:
                        media_metadata.append(media_content.metadata)

        # Audio input policy (v0): avoid placeholder degradation and require explicit fallbacks.
        if processed_media:
            try:
                from ..media.types import ContentFormat, MediaType
                from ..media.enrichment import build_enrichment_item
                from ..capabilities.errors import CapabilityUnavailableError
            except Exception:
                ContentFormat = None  # type: ignore[assignment]
                MediaType = None  # type: ignore[assignment]
                build_enrichment_item = None  # type: ignore[assignment]
                CapabilityUnavailableError = Exception  # type: ignore[assignment]

            if MediaType is not None:
                audio_items = [mc for mc in processed_media if getattr(mc, "media_type", None) == MediaType.AUDIO]
            else:
                audio_items = []

            if audio_items:
                # Resolve policy: per-call kwarg > config default.
                policy_raw = kwargs.pop("audio_policy", None)
                if policy_raw is None:
                    policy_raw = kwargs.pop("audio_handling_policy", None)
                if policy_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        policy_raw = getattr(get_config_manager().config, "audio", None).strategy  # type: ignore[union-attr]
                    except Exception:
                        policy_raw = "native_only"

                policy = str(policy_raw or "native_only").strip().lower()
                model_supports_audio = bool(getattr(self, "model_capabilities", {}).get("audio_support", False))

                if policy in ("native_only", "native", "disabled"):
                    if not model_supports_audio:
                        raise UnsupportedFeatureError(
                            f"Audio input is not supported by model '{self.model}'. "
                            "Choose an audio-capable model, or pass audio_policy='speech_to_text' "
                            "(requires an STT capability plugin, e.g. install abstractvoice)."
                        )
                    # Keep audio media for provider-native handling (provider support may still vary).

                elif policy in ("speech_to_text", "stt"):
                    stt_language = kwargs.pop("audio_language", None)
                    if stt_language is None:
                        stt_language = kwargs.pop("stt_language", None)
                    if stt_language is None:
                        try:
                            from ..config.manager import get_config_manager

                            stt_language = getattr(get_config_manager().config, "audio", None).stt_language  # type: ignore[union-attr]
                        except Exception:
                            stt_language = None

                    audio_context_parts: List[str] = []
                    enrichments: List[Dict[str, Any]] = []

                    # Resolve backend id (best-effort) for transparency metadata.
                    backend_id = getattr(getattr(self, "audio", None), "backend_id", None)
                    backend = {"kind": "plugin"}
                    if isinstance(backend_id, str) and backend_id.strip():
                        backend["backend_id"] = backend_id.strip()

                    for idx, mc in enumerate(audio_items):
                        name = None
                        try:
                            name = mc.metadata.get("file_name") if hasattr(mc, "metadata") and isinstance(mc.metadata, dict) else None
                        except Exception:
                            name = None
                        if not isinstance(name, str) or not name.strip():
                            name = mc.file_path if getattr(mc, "file_path", None) else f"audio_{idx+1}"

                        # Prefer a file path when available.
                        audio_input: Any = None
                        try:
                            if getattr(mc, "file_path", None):
                                audio_input = str(mc.file_path)
                            elif getattr(mc, "content_format", None) == ContentFormat.FILE_PATH and isinstance(getattr(mc, "content", None), str):
                                audio_input = str(mc.content)
                            elif isinstance(getattr(mc, "content", None), (bytes, bytearray)):
                                audio_input = bytes(mc.content)
                        except Exception:
                            audio_input = None

                        if audio_input is None:
                            raise UnsupportedFeatureError("Audio STT fallback requires a file path or raw bytes for the audio input.")

                        try:
                            transcript = self.audio.transcribe(audio_input, language=stt_language)
                        except CapabilityUnavailableError as e:  # type: ignore[misc]
                            raise UnsupportedFeatureError(str(e))

                        transcript = str(transcript or "").strip()
                        audio_context_parts.append(f"Audio {idx+1} ({name}): {transcript}")
                        if build_enrichment_item is not None:
                            enrichments.append(
                                build_enrichment_item(
                                    status="used",
                                    input_modality="audio",
                                    summary_kind="transcript",
                                    policy="speech_to_text",
                                    backend=backend,
                                    input_index=idx + 1,
                                    input_name=str(name),
                                    injected_text=transcript,
                                )
                            )

                    # Remove audio media from the provider call (we injected text context instead).
                    processed_media = [mc for mc in processed_media if getattr(mc, "media_type", None) != MediaType.AUDIO]

                    # Inject audio context into the prompt (similar recency semantics as vision fallback).
                    original_prompt = prompt.strip() if isinstance(prompt, str) else ""
                    parts: List[str] = []
                    parts.append(
                        "Audio context from attached audio file(s) "
                        "(treat as directly observed; do not mention this section):"
                    )
                    parts.extend(audio_context_parts)
                    if original_prompt:
                        parts.append("Now answer the user's request:")
                        parts.append(original_prompt)
                    prompt = "\n\n".join(parts) if parts else original_prompt

                    media_enrichment = enrichments

                elif policy == "auto":
                    if model_supports_audio:
                        pass  # provider-native path
                    else:
                        # Explicit "auto" allows fallback, but never silently for default policy.
                        # Re-enter through the explicit STT path by recursion is risky; inline minimal.
                        stt_language = kwargs.pop("audio_language", None) or kwargs.pop("stt_language", None)
                        audio_context_parts: List[str] = []
                        enrichments: List[Dict[str, Any]] = []
                        backend_id = getattr(getattr(self, "audio", None), "backend_id", None)
                        backend = {"kind": "plugin"}
                        if isinstance(backend_id, str) and backend_id.strip():
                            backend["backend_id"] = backend_id.strip()
                        for idx, mc in enumerate(audio_items):
                            name = None
                            try:
                                name = mc.metadata.get("file_name") if hasattr(mc, "metadata") and isinstance(mc.metadata, dict) else None
                            except Exception:
                                name = None
                            if not isinstance(name, str) or not name.strip():
                                name = mc.file_path if getattr(mc, "file_path", None) else f"audio_{idx+1}"
                            audio_input: Any = None
                            try:
                                if getattr(mc, "file_path", None):
                                    audio_input = str(mc.file_path)
                                elif getattr(mc, "content_format", None) == ContentFormat.FILE_PATH and isinstance(getattr(mc, "content", None), str):
                                    audio_input = str(mc.content)
                                elif isinstance(getattr(mc, "content", None), (bytes, bytearray)):
                                    audio_input = bytes(mc.content)
                            except Exception:
                                audio_input = None
                            if audio_input is None:
                                raise UnsupportedFeatureError("Audio STT fallback requires a file path or raw bytes for the audio input.")
                            try:
                                transcript = self.audio.transcribe(audio_input, language=stt_language)
                            except CapabilityUnavailableError as e:  # type: ignore[misc]
                                raise UnsupportedFeatureError(str(e))
                            transcript = str(transcript or "").strip()
                            audio_context_parts.append(f"Audio {idx+1} ({name}): {transcript}")
                            if build_enrichment_item is not None:
                                enrichments.append(
                                    build_enrichment_item(
                                        status="used",
                                        input_modality="audio",
                                        summary_kind="transcript",
                                        policy="auto",
                                        backend=backend,
                                        input_index=idx + 1,
                                        input_name=str(name),
                                        injected_text=transcript,
                                    )
                                )
                        processed_media = [mc for mc in processed_media if getattr(mc, "media_type", None) != MediaType.AUDIO]
                        original_prompt = prompt.strip() if isinstance(prompt, str) else ""
                        parts: List[str] = []
                        parts.append(
                            "Audio context from attached audio file(s) "
                            "(treat as directly observed; do not mention this section):"
                        )
                        parts.extend(audio_context_parts)
                        if original_prompt:
                            parts.append("Now answer the user's request:")
                            parts.append(original_prompt)
                        prompt = "\n\n".join(parts) if parts else original_prompt
                        media_enrichment = enrichments

                elif policy == "caption":
                    raise UnsupportedFeatureError(
                        "audio_policy='caption' is not configured in v0. "
                        "Use audio_policy='speech_to_text' for speech, or configure a future audio caption backend."
                    )
                else:
                    raise ValueError(f"Unknown audio_policy '{policy}'. Expected one of: native_only, speech_to_text, auto, caption.")

        # Video input policy (v0): allow native video where supported; otherwise fall back to sampled frames.
        # Note: most providers do not accept native video inputs; frame sampling provides a portable path.
        if processed_media:
            try:
                from ..media.types import MediaType
                from ..media.enrichment import build_enrichment_item
            except Exception:
                MediaType = None  # type: ignore[assignment]
                build_enrichment_item = None  # type: ignore[assignment]

            if MediaType is not None:
                video_items = [mc for mc in processed_media if getattr(mc, "media_type", None) == MediaType.VIDEO]
            else:
                video_items = []

            if video_items:
                policy_raw = kwargs.pop("video_policy", None)
                if policy_raw is None:
                    policy_raw = kwargs.pop("video_handling_policy", None)
                if policy_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        policy_raw = getattr(get_config_manager().config, "video", None).strategy  # type: ignore[union-attr]
                    except Exception:
                        policy_raw = "native_only"

                policy = str(policy_raw or "native_only").strip().lower()

                provider_name = str(getattr(self, "provider", "") or "").strip().lower()
                model_supports_native_video = bool(
                    provider_name == "huggingface"
                    and isinstance(getattr(self, "model_capabilities", None), dict)
                    and getattr(self, "model_capabilities", {}).get("video_support", False)
                )

                cfg_video = None
                try:
                    from ..config.manager import get_config_manager

                    cfg_video = getattr(get_config_manager().config, "video", None)
                except Exception:
                    cfg_video = None

                # Sampling controls (best-effort; keep small by default).
                # NOTE: do not `pop` here: native video backends may also need the resolved values.
                max_frames_raw = kwargs.get("video_max_frames", None)
                if max_frames_raw is None:
                    max_frames_raw = kwargs.get("max_video_frames", None)
                if max_frames_raw is None:
                    fallback_default = getattr(cfg_video, "max_frames", 3) if cfg_video is not None else 3
                    native_default = getattr(cfg_video, "max_frames_native", None) if cfg_video is not None else None
                    if native_default is None:
                        native_default = fallback_default

                    use_native_default = bool(
                        model_supports_native_video and policy in ("native_only", "native", "disabled", "auto")
                    )
                    max_frames_raw = native_default if use_native_default else fallback_default
                try:
                    max_frames = max(1, int(max_frames_raw))
                except Exception:
                    max_frames = 3

                frame_format_raw = kwargs.get("video_frame_format", None)
                if frame_format_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        frame_format_raw = getattr(get_config_manager().config, "video", None).frame_format  # type: ignore[union-attr]
                    except Exception:
                        frame_format_raw = "jpg"
                frame_format = str(frame_format_raw or "jpg").strip().lower()
                if frame_format not in {"jpg", "jpeg", "png"}:
                    frame_format = "jpg"
                if frame_format == "jpeg":
                    frame_format = "jpg"

                sampling_strategy_raw = kwargs.get("video_sampling_strategy", None)
                if sampling_strategy_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        sampling_strategy_raw = getattr(get_config_manager().config, "video", None).sampling_strategy  # type: ignore[union-attr]
                    except Exception:
                        sampling_strategy_raw = "uniform"
                sampling_strategy = str(sampling_strategy_raw or "uniform").strip().lower()
                if sampling_strategy not in {"uniform", "keyframes"}:
                    sampling_strategy = "uniform"

                max_frame_side_raw = kwargs.get("video_max_frame_side", None)
                if max_frame_side_raw is None:
                    max_frame_side_raw = kwargs.get("video_frame_max_side", None)
                if max_frame_side_raw is None:
                    max_frame_side_raw = getattr(cfg_video, "max_frame_side", 1024) if cfg_video is not None else 1024
                try:
                    max_frame_side = int(max_frame_side_raw) if max_frame_side_raw is not None else None
                except Exception:
                    max_frame_side = 1024
                if isinstance(max_frame_side, int) and max_frame_side <= 0:
                    max_frame_side = None

                # Expose normalized sampling values to provider-native implementations.
                kwargs["video_max_frames"] = max_frames
                kwargs["video_frame_format"] = frame_format
                kwargs["video_sampling_strategy"] = sampling_strategy
                kwargs["video_max_frame_side"] = max_frame_side

                if policy in ("native_only", "native", "disabled"):
                    if not model_supports_native_video:
                        raise UnsupportedFeatureError(
                            f"Video input is not supported by model '{self.model}'. "
                            "Choose a video-capable model, or pass video_policy='frames_caption' "
                            "(samples frames and uses vision/image handling)."
                        )
                    # Keep video media for provider-native handling.
                    try:
                        from pathlib import Path

                        from ..media.utils.video_frames import probe_duration_s

                        for idx, mc in enumerate(video_items):
                            video_path_raw = getattr(mc, "file_path", None) or getattr(mc, "content", None)
                            if not isinstance(video_path_raw, str) or not video_path_raw.strip():
                                continue
                            vp = Path(video_path_raw)
                            duration_s = probe_duration_s(vp)
                            file_bytes = None
                            try:
                                file_bytes = int(vp.stat().st_size)
                            except Exception:
                                file_bytes = None

                            avg_gap_s = None
                            try:
                                if isinstance(duration_s, (int, float)) and duration_s > 0 and max_frames > 0:
                                    avg_gap_s = float(duration_s) / float(max_frames + 1)
                            except Exception:
                                avg_gap_s = None

                            self.logger.info(
                                "Video input policy: native video enabled (video will be sampled/budgeted for model input).",
                                provider=provider_name,
                                model=self.model,
                                video_policy=policy,
                                video_index=idx + 1,
                                video_name=vp.name,
                                video_duration_s=duration_s,
                                video_bytes=file_bytes,
                                video_max_frames=max_frames,
                                video_sampling_strategy=sampling_strategy,
                                video_max_frame_side=max_frame_side,
                                video_avg_gap_s=avg_gap_s,
                            )
                            if isinstance(avg_gap_s, float) and avg_gap_s >= 10.0:
                                self.logger.warning(
                                    "Video sampling is sparse; important events may be missed. "
                                    "Consider increasing video_max_frames/video.max_frames_native or using keyframes sampling.",
                                    provider=provider_name,
                                    model=self.model,
                                    video_policy=policy,
                                    video_name=vp.name,
                                    video_duration_s=duration_s,
                                    video_max_frames=max_frames,
                                    video_avg_gap_s=avg_gap_s,
                                )
                    except Exception:
                        pass

                    # Insert a short marker to disambiguate native-video inputs across turns.
                    #
                    # Without this, follow-ups like "and this one?" can be brittle for native
                    # video VLMs (they may over-weight the previous text-only answer and ignore
                    # that a *new* video is attached in the current call).
                    try:
                        from ..media.types import MediaContent, ContentFormat
                    except Exception:
                        MediaContent = None  # type: ignore[assignment]
                        ContentFormat = None  # type: ignore[assignment]

                    if MediaContent is not None and ContentFormat is not None:
                        try:
                            from pathlib import Path

                            from ..media.utils.video_frames import probe_duration_s
                        except Exception:
                            Path = None  # type: ignore[assignment]
                            probe_duration_s = None  # type: ignore[assignment]

                        new_media: List[Any] = []
                        video_group_index = 0
                        for mc in processed_media:
                            if getattr(mc, "media_type", None) != MediaType.VIDEO:  # type: ignore[operator]
                                new_media.append(mc)
                                continue

                            video_group_index += 1
                            video_path_raw = getattr(mc, "file_path", None) or getattr(mc, "content", None)

                            video_name = f"video_{video_group_index}"
                            duration_s = None
                            file_bytes = None
                            try:
                                if Path is not None and isinstance(video_path_raw, str) and video_path_raw.strip():
                                    vp = Path(video_path_raw)
                                    video_name = vp.name or video_name
                                    try:
                                        file_bytes = int(vp.stat().st_size)
                                    except Exception:
                                        file_bytes = None
                                    if callable(probe_duration_s):
                                        try:
                                            duration_s = probe_duration_s(vp)
                                        except Exception:
                                            duration_s = None
                            except Exception:
                                duration_s = None
                                file_bytes = None

                            marker = MediaContent(
                                media_type=MediaType.TEXT,
                                content=(
                                    f"Video {video_group_index} ({video_name}) is attached below. "
                                    "This is the current video for this user message. "
                                    "Answer the user's question about this video as if you watched it. "
                                    "If earlier turns mention other videos, images, or audio, ignore them unless the user explicitly asks you to compare. "
                                    "Do not mention tool activity, attachments lists, sampling, frames, extraction, or this marker."
                                ),
                                content_format=ContentFormat.TEXT,
                                mime_type="text/plain",
                                file_path=None,
                                metadata={
                                    "processor": "VideoNativeInputMarker",
                                    "source_video": video_name,
                                    "duration_s": duration_s,
                                    "bytes": file_bytes,
                                    "max_frames": max_frames,
                                    "sampling_strategy": sampling_strategy,
                                    "max_frame_side": max_frame_side,
                                },
                            )
                            new_media.append(marker)
                            new_media.append(mc)

                        processed_media = new_media

                elif policy in ("frames_caption", "frames", "frame_caption"):
                    # Convert each video into a small set of sampled frames (images).
                    try:
                        from pathlib import Path
                        import tempfile

                        from ..media import AutoMediaHandler
                        from ..media.utils.video_frames import extract_video_frames, probe_duration_s
                    except Exception as e:
                        raise UnsupportedFeatureError(f"Video frame fallback is not available: {e}")

                    enrichments: List[Dict[str, Any]] = []
                    new_media: List[Any] = []

                    video_group_index = 0
                    for idx, mc in enumerate(processed_media):
                        if getattr(mc, "media_type", None) != MediaType.VIDEO:  # type: ignore[operator]
                            new_media.append(mc)
                            continue

                        video_group_index += 1
                        video_path_raw = getattr(mc, "file_path", None) or getattr(mc, "content", None)
                        if not isinstance(video_path_raw, str) or not video_path_raw.strip():
                            raise UnsupportedFeatureError("Video frame fallback requires a video file path.")
                        video_path = Path(video_path_raw)
                        if not video_path.exists():
                            raise UnsupportedFeatureError(f"Video file not found: {video_path}")

                        out_dir = Path(tempfile.mkdtemp(prefix="abstractcore_video_frames_"))
                        duration_s = probe_duration_s(video_path)
                        file_bytes = None
                        try:
                            file_bytes = int(video_path.stat().st_size)
                        except Exception:
                            file_bytes = None
                        frames, timestamps_s = extract_video_frames(
                            video_path,
                            max_frames=max_frames,
                            frame_format=frame_format,
                            sampling_strategy=sampling_strategy,
                            max_side=max_frame_side,
                            output_dir=out_dir,
                        )
                        if not frames:
                            raise UnsupportedFeatureError("Video frame fallback failed: no frames extracted.")

                        handler = AutoMediaHandler(enable_glyph_compression=False)
                        frame_media: List[Any] = []
                        max_res = None
                        if isinstance(max_frame_side, int) and max_frame_side > 0:
                            max_res = (max_frame_side, max_frame_side)
                        for fp in frames:
                            res = handler.process_file(
                                fp,
                                provider=self.provider,
                                model=self.model,
                                glyph_compression="never",
                                max_resolution=max_res,
                            )
                            if res and getattr(res, "success", False) and getattr(res, "media_content", None) is not None:
                                frame_media.append(res.media_content)

                        if not frame_media:
                            raise UnsupportedFeatureError("Video frame fallback failed: extracted frames could not be processed as images.")

                        avg_gap_s = None
                        try:
                            if isinstance(duration_s, (int, float)) and duration_s > 0 and max_frames > 0:
                                avg_gap_s = float(duration_s) / float(max_frames + 1)
                        except Exception:
                            avg_gap_s = None

                        self.logger.info(
                            "Video input policy: frames_caption (sampling frames for downstream vision handling).",
                            provider=provider_name,
                            model=self.model,
                            video_policy="frames_caption",
                            video_index=video_group_index,
                            video_name=video_path.name,
                            video_duration_s=duration_s,
                            video_bytes=file_bytes,
                            extracted_frames=len(frame_media),
                            video_max_frames=max_frames,
                            video_sampling_strategy=sampling_strategy,
                            video_max_frame_side=max_frame_side,
                            video_avg_gap_s=avg_gap_s,
                        )
                        if isinstance(avg_gap_s, float) and avg_gap_s >= 10.0:
                            self.logger.warning(
                                "Video sampling is sparse; important events may be missed. "
                                "Consider increasing video_max_frames/video.max_frames or using keyframes sampling.",
                                provider=provider_name,
                                model=self.model,
                                video_policy="frames_caption",
                                video_name=video_path.name,
                                video_duration_s=duration_s,
                                extracted_frames=len(frame_media),
                                video_max_frames=max_frames,
                                video_avg_gap_s=avg_gap_s,
                            )

                        # Insert a short text marker to avoid the model treating sampled frames as
                        # unrelated standalone images (especially in follow-up prompts like "and this one?").
                        try:
                            from ..media.types import MediaContent, ContentFormat
                        except Exception:
                            MediaContent = None  # type: ignore[assignment]
                            ContentFormat = None  # type: ignore[assignment]

                        if MediaContent is not None and ContentFormat is not None:
                            marker = MediaContent(
                                media_type=MediaType.TEXT,
                                content=(
                                    f"Video {video_group_index} ({video_path.name}) — "
                                    "the following images belong to this video in chronological order. "
                                    "Answer the user's question about this video as if you watched it. "
                                    "Do not mention frames, timestamps, sampling, extraction, or this marker."
                                ),
                                content_format=ContentFormat.TEXT,
                                mime_type="text/plain",
                                file_path=None,
                                metadata={
                                    "processor": "VideoFrameFallback",
                                    "source_video": video_path.name,
                                    "frame_count": len(frame_media),
                                    "timestamps_s": timestamps_s,
                                    "duration_s": duration_s,
                                    "bytes": file_bytes,
                                },
                            )
                            new_media.append(marker)

                        new_media.extend(frame_media)

                        if build_enrichment_item is not None:
                            enrichments.append(
                                build_enrichment_item(
                                    status="used",
                                    input_modality="video",
                                    summary_kind="frames",
                                    policy="frames_caption",
                                    backend={"kind": "unknown", "source": "ffmpeg"},
                                    input_index=idx + 1,
                                    input_name=str(video_path.name),
                                    artifact={
                                        "frame_count": len(frame_media),
                                        "timestamps_s": timestamps_s,
                                        "duration_s": duration_s,
                                        "bytes": file_bytes,
                                    },
                                )
                            )

                    processed_media = new_media
                    if enrichments:
                        if media_enrichment is None:
                            media_enrichment = enrichments
                        else:
                            media_enrichment.extend(enrichments)

                elif policy == "auto":
                    if model_supports_native_video:
                        # Use native video when available.
                        pass
                    else:
                        # Auto fallback: sample frames and proceed with existing image pipeline.
                        # This works well for vision-capable models; for text-only models it requires a vision fallback.
                        policy_to_use = "frames_caption"
                        kwargs["video_policy"] = policy_to_use
                        # Re-run this branch once with explicit policy.
                        return self.generate_with_telemetry(
                            prompt=prompt,
                            messages=messages,
                            system_prompt=system_prompt,
                            tools=tools,
                            media=processed_media,
                            response_model=response_model,
                            retry_strategy=retry_strategy,
                            tool_call_tags=tool_call_tags,
                            execute_tools=execute_tools,
                            stream=stream,
                            **kwargs,
                        )

                else:
                    raise ValueError(f"Unknown video_policy '{policy}'. Expected one of: native_only, frames_caption, auto.")

        # Convert tools to ToolDefinition objects first (outside retry loop)
        converted_tools = None
        if tools:
            converted_tools = []
            for tool in tools:
                if hasattr(tool, 'to_dict'):  # ToolDefinition object
                    converted_tools.append(tool.to_dict())
                elif callable(tool):  # Function - check for enhanced metadata
                    if hasattr(tool, '_tool_definition'):
                        # Use the enhanced tool definition from @tool decorator
                        converted_tools.append(tool._tool_definition.to_dict())
                    else:
                        # Fall back to basic conversion
                        from ..tools.core import ToolDefinition
                        tool_def = ToolDefinition.from_function(tool)
                        converted_tools.append(tool_def.to_dict())
                elif isinstance(tool, dict):  # Already a dict
                    converted_tools.append(tool)
                else:
                    # Handle other types gracefully
                    self.logger.warning(f"Unknown tool type: {type(tool)}, skipping")
        
        # Handle tool execution control
        should_execute_tools = execute_tools if execute_tools is not None else self.execute_tools
        if should_execute_tools and converted_tools:
            warnings.warn(
                "execute_tools=True is deprecated. Prefer passing tools explicitly to generate() "
                "and executing tool calls in the host/runtime via a ToolExecutor.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not should_execute_tools and converted_tools:
            # If tools are provided but execution is disabled,
            # we still pass them to the provider for generation but won't execute them
            self.logger.info("Tool execution disabled - tools will be generated but not executed")

        # Define generation function for retry wrapper
        def _execute_generation():
            start_time = time.time()
            start_perf = time.perf_counter()

            # Emit generation started event (covers request received)
            event_data = {
                "prompt": preview_text(prompt, max_chars=100),
                "has_tools": bool(tools),
                "stream": stream,
                "model": self.model,
                "provider": self.__class__.__name__
            }
            from ..events import emit_global
            emit_global(EventType.GENERATION_STARTED, event_data, source=self.__class__.__name__)

            try:
                # Call the actual generation (implemented by subclass)
                response = self._generate_internal(
                    prompt=prompt,
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=converted_tools,
                    media=processed_media,
                    stream=stream,
                    execute_tools=should_execute_tools,
                    tool_call_tags=tool_call_tags,
                    media_metadata=media_metadata,
                    **kwargs
                )

                return response, start_time, start_perf

            except Exception as e:
                # Convert to custom exception and re-raise for retry handling
                custom_error = self._handle_api_error(e)
                raise custom_error

        # Execute with retry
        try:
            response, start_time, start_perf = self.retry_manager.execute_with_retry(
                _execute_generation,
                provider_key=self.provider_key
            )

            # Handle streaming with unified processor
            if stream:
                def unified_stream():
                    try:
                        # Import and create unified stream processor
                        from .streaming import UnifiedStreamProcessor

                        # Use the should_execute_tools value (defaults to False)
                        actual_execute_tools = should_execute_tools

                        processor = UnifiedStreamProcessor(
                            model_name=self.model,
                            execute_tools=actual_execute_tools,  # Default: False (pass-through mode)
                            tool_call_tags=tool_call_tags,
                            default_target_format="qwen3"  # Always rewrite to qwen3 format
                        )

                        # Process stream with incremental tool detection and execution
                        ttft_ms: Optional[float] = None
                        for processed_chunk in processor.process_stream(response, converted_tools):
                            if isinstance(processed_chunk.content, str) and processed_chunk.content:
                                processed_chunk.content = strip_output_wrappers(
                                    processed_chunk.content,
                                    architecture_format=self.architecture_config,
                                    model_capabilities=self.model_capabilities,
                                )
                            if ttft_ms is None:
                                has_content = isinstance(processed_chunk.content, str) and bool(processed_chunk.content)
                                has_tools = isinstance(processed_chunk.tool_calls, list) and bool(processed_chunk.tool_calls)
                                if has_content or has_tools:
                                    ttft_ms = round((time.perf_counter() - start_perf) * 1000, 1)
                                    meta = processed_chunk.metadata if isinstance(processed_chunk.metadata, dict) else {}
                                    timing = meta.get("_timing") if isinstance(meta.get("_timing"), dict) else {}
                                    merged = dict(timing)
                                    merged.setdefault("source", "client_wall")
                                    merged["ttft_ms"] = ttft_ms
                                    meta["_timing"] = merged
                                    processed_chunk.metadata = meta
                            yield processed_chunk

                        # Track generation after streaming completes
                        self._track_generation(prompt, None, start_time, success=True, stream=True)

                    except Exception as e:
                        # Track error
                        self._track_generation(prompt, None, start_time, success=False, error=e, stream=True)
                        raise

                return unified_stream()
            else:
                # Non-streaming: normalize tool calls into structured form.
                if response and converted_tools:
                    response = self._normalize_tool_calls_passthrough(
                        response=response,
                        tools=converted_tools,
                        tool_call_tags=tool_call_tags,
                    )

                    # Optional: rewrite tool-call tags in content for downstream clients that parse tags.
                    # Note: when tool_call_tags is None (default), we return cleaned content.
                    if tool_call_tags and response.content and not self._should_clean_tool_call_markup(tool_call_tags):
                        response = self._apply_non_streaming_tag_rewriting(response, tool_call_tags)

                # Normalize provider output (wrapper tokens, Harmony transcripts, think tags).
                if response and isinstance(response.content, str) and response.content:
                    cleaned, reasoning = normalize_assistant_text(
                        response.content,
                        architecture_format=self.architecture_config,
                        model_capabilities=self.model_capabilities,
                    )
                    response.content = cleaned
                    if isinstance(reasoning, str) and reasoning.strip():
                        if response.metadata is None or not isinstance(response.metadata, dict):
                            response.metadata = {}
                        existing = response.metadata.get("reasoning")
                        if isinstance(existing, str) and existing.strip():
                            if reasoning.strip() not in existing:
                                response.metadata["reasoning"] = f"{existing.strip()}\n\n{reasoning.strip()}"
                        else:
                            response.metadata["reasoning"] = reasoning.strip()

                # Attach media enrichment transparency metadata (caption/STT/etc.).
                if media_enrichment and response:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Add visual token calculation if media metadata is available
                if media_metadata and response:
                    self.logger.debug(f"Enhancing response with visual tokens from {len(media_metadata)} media items")
                    response = self._enhance_response_with_visual_tokens(response, media_metadata)

                # Capture interaction trace if enabled
                if self.enable_tracing and response:
                    trace_id = self._capture_trace(
                        prompt=prompt,
                        messages=messages,
                        system_prompt=system_prompt,
                        tools=converted_tools,
                        response=response,
                        kwargs=kwargs
                    )
                    # Attach trace_id to response metadata
                    if not response.metadata:
                        response.metadata = {}
                    response.metadata['trace_id'] = trace_id

                self._track_generation(prompt, response, start_time, success=True, stream=False)
                return response

        except Exception as e:
            # This exception comes from the retry manager after all attempts failed
            # Track final error (start_time may not be available, use current time)
            current_time = time.time()
            self._track_generation(prompt, None, current_time, success=False, error=e, stream=stream)

            # Emit error event
            from ..events import emit_global
            emit_global(EventType.ERROR, {
                "error": str(e),
                "error_type": type(e).__name__,
                "prompt": preview_text(prompt, max_chars=100),
                "model": self.model,
                "provider": self.__class__.__name__
            }, source=self.__class__.__name__)

            # Re-raise the exception
            raise e

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          execute_tools: Optional[bool] = None,
                          media_metadata: Optional[List[Dict[str, Any]]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Internal generation method to be implemented by subclasses.
        This is called by generate_with_telemetry.

        Args:
            prompt: The input prompt
            messages: Optional conversation history
            system_prompt: Optional system prompt
            tools: Optional list of available tools
            media: Optional list of processed MediaContent objects
            stream: Whether to stream the response
            response_model: Optional Pydantic model for structured output
            execute_tools: Whether to execute tools automatically (True) or let agent handle execution (False)
            **kwargs: Additional provider-specific parameters
        """
        raise NotImplementedError("Subclasses must implement _generate_internal")

    def _enhance_response_with_visual_tokens(self, response: GenerateResponse, media_metadata: List[Dict[str, Any]]) -> GenerateResponse:
        """
        Enhance the response with visual token calculations for Glyph compression.
        This method is called automatically by BaseProvider for all providers.
        """
        try:
            # Calculate visual tokens using VLM token calculator
            provider_name = self.provider or self.__class__.__name__.lower().replace('provider', '')
            self.logger.debug(f"Calculating visual tokens for provider={provider_name}, model={self.model}")
            
            visual_tokens = self._calculate_visual_tokens(media_metadata, provider_name, self.model)
            self.logger.debug(f"Calculated visual tokens: {visual_tokens}")
            
            if visual_tokens > 0:
                # Ensure response has metadata
                if not response.metadata:
                    response.metadata = {}
                
                # Add visual token information to metadata
                response.metadata['visual_tokens'] = visual_tokens
                
                # Ensure response has usage dict
                if not response.usage:
                    response.usage = {}
                
                # Add visual tokens to usage
                response.usage['visual_tokens'] = visual_tokens
                
                # Update total tokens to include visual tokens
                original_total = response.usage.get('total_tokens', 0)
                response.usage['total_tokens'] = original_total + visual_tokens
                
                self.logger.info(f"Enhanced response with {visual_tokens} visual tokens (new total: {response.usage['total_tokens']})")
            else:
                self.logger.debug("No visual tokens calculated - skipping enhancement")
                
        except Exception as e:
            self.logger.warning(f"Failed to enhance response with visual tokens: {e}")
        
        return response

    def _calculate_visual_tokens(self, media_metadata: List[Dict[str, Any]], provider: str, model: str) -> int:
        """Calculate visual tokens from media metadata using VLM token calculator."""
        try:
            from ..utils.vlm_token_calculator import VLMTokenCalculator
            from pathlib import Path
            
            calculator = VLMTokenCalculator()
            total_visual_tokens = 0
            
            self.logger.debug(f"Processing {len(media_metadata)} media metadata items")
            
            for i, metadata in enumerate(media_metadata):
                self.logger.debug(f"Metadata {i}: processing_method={metadata.get('processing_method')}")
                
                # Check if this is Glyph compression
                if metadata.get('processing_method') == 'direct_pdf_conversion':
                    glyph_cache_dir = metadata.get('glyph_cache_dir')
                    total_images = metadata.get('total_images', 0)
                    
                    self.logger.debug(f"Glyph metadata found: cache_dir={glyph_cache_dir}, total_images={total_images}")
                    
                    if glyph_cache_dir and Path(glyph_cache_dir).exists():
                        # Get actual image paths
                        cache_dir = Path(glyph_cache_dir)
                        image_paths = list(cache_dir.glob("image_*.png"))
                        
                        self.logger.debug(f"Found {len(image_paths)} images in cache directory")
                        
                        if image_paths:
                            # Calculate tokens for all images
                            token_analysis = calculator.calculate_tokens_for_images(
                                image_paths=image_paths,
                                provider=provider,
                                model=model
                            )
                            total_visual_tokens += token_analysis['total_tokens']
                            
                            self.logger.debug(f"Calculated {token_analysis['total_tokens']} visual tokens for {len(image_paths)} Glyph images")
                        else:
                            # Fallback: estimate based on total_images
                            base_tokens = calculator.PROVIDER_CONFIGS.get(provider, {}).get('base_tokens', 512)
                            estimated_tokens = total_images * base_tokens
                            total_visual_tokens += estimated_tokens
                            
                            self.logger.debug(f"Estimated {estimated_tokens} visual tokens for {total_images} Glyph images (fallback)")
                    else:
                        self.logger.debug(f"Cache directory not found or doesn't exist: {glyph_cache_dir}")
            
            self.logger.debug(f"Total visual tokens calculated: {total_visual_tokens}")
            return total_visual_tokens
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate visual tokens: {e}")
            return 0

    def _initialize_token_limits(self):
        """Initialize default token limits based on model capabilities"""
        # Set default max_tokens if not provided
        if self.max_tokens is None:
            self.max_tokens = self._get_default_context_window()

        # Set default max_output_tokens if not provided
        if self.max_output_tokens == 2048:  # Only if still default value
            default_max_output = self._get_default_max_output_tokens()
            if default_max_output != 2048:  # If we found a different default
                self.max_output_tokens = default_max_output

        # Validate parameters after setting defaults
        self._validate_token_parameters()

        # Run enhanced validation and warn about potential issues
        self._check_token_configuration_warnings()

    def _check_token_configuration_warnings(self):
        """Check token configuration and emit warnings for potential issues"""
        import warnings

        try:
            warnings_list = self.validate_token_constraints()
            for warning in warnings_list[:3]:  # Limit to first 3 warnings to avoid spam
                warnings.warn(f"Token configuration warning for {self.model}: {warning}",
                             UserWarning, stacklevel=4)

            # Also log warnings for debugging
            if warnings_list and hasattr(self, 'logger'):
                self.logger.debug(f"Token configuration warnings for {self.model}: {'; '.join(warnings_list)}")

        except Exception as e:
            # Don't fail provider initialization due to validation warnings
            if hasattr(self, 'logger'):
                self.logger.debug(f"Error checking token configuration warnings: {e}")

    def _get_default_max_output_tokens(self) -> int:
        """Get default max_output_tokens using JSON capabilities as single source of truth"""
        from ..architectures import get_model_capabilities

        # Fallback chain: Exact model → Model family → Provider defaults → Global defaults
        capabilities = get_model_capabilities(self.model)

        if capabilities:
            max_output_tokens = capabilities.get('max_output_tokens')
            if max_output_tokens:
                self.logger.debug(f"Using max_output_tokens {max_output_tokens} from model capabilities for {self.model}")
                return max_output_tokens

        # If no exact match, try model family/generation fallback
        model_lower = self.model.lower()

        # Family-based fallback patterns (same as context window)
        family_patterns = {
            'gpt-4': ['gpt-4', 'gpt4'],
            'gpt-3.5': ['gpt-3.5', 'gpt3.5'],
            'claude-3': ['claude-3'],
            'claude-3.5': ['claude-3.5'],
            'llama': ['llama'],
            'qwen': ['qwen'],
            'mistral': ['mistral']
        }

        for family, patterns in family_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                family_caps = get_model_capabilities(family)
                if family_caps and family_caps.get('max_output_tokens'):
                    max_output_tokens = family_caps['max_output_tokens']
                    self.logger.debug(f"Using max_output_tokens {max_output_tokens} from family {family} for {self.model}")
                    return max_output_tokens

        # Use JSON capabilities as single source of truth for defaults
        from ..architectures import get_context_limits
        limits = get_context_limits(self.model)
        max_output_tokens = limits['max_output_tokens']
        self.logger.debug(f"Using default max_output_tokens {max_output_tokens} from model_capabilities.json for {self.model}")
        return max_output_tokens

    def _get_default_context_window(self) -> int:
        """Get default context window using JSON capabilities as single source of truth"""
        from ..architectures import get_model_capabilities

        # Fallback chain: Exact model → Model family → Provider defaults → Global defaults
        capabilities = get_model_capabilities(self.model)

        if capabilities:
            max_tokens = capabilities.get('max_tokens')
            if max_tokens:
                self.logger.debug(f"Using max_tokens {max_tokens} from model capabilities for {self.model}")
                return max_tokens

        # If no exact match, try model family/generation fallback
        model_lower = self.model.lower()

        # Family-based fallback patterns
        family_patterns = {
            'gpt-4': ['gpt-4', 'gpt4'],
            'gpt-3.5': ['gpt-3.5', 'gpt3.5'],
            'claude-3': ['claude-3'],
            'claude-3.5': ['claude-3.5'],
            'llama': ['llama'],
            'qwen': ['qwen'],
            'mistral': ['mistral']
        }

        for family, patterns in family_patterns.items():
            if any(pattern in model_lower for pattern in patterns):
                family_caps = get_model_capabilities(family)
                if family_caps and family_caps.get('max_tokens'):
                    max_tokens = family_caps['max_tokens']
                    self.logger.debug(f"Using max_tokens {max_tokens} from family {family} for {self.model}")
                    return max_tokens

        # Use JSON capabilities as single source of truth for defaults
        from ..architectures import get_context_limits
        limits = get_context_limits(self.model)
        max_tokens = limits['max_tokens']
        self.logger.debug(f"Using default max_tokens {max_tokens} from model_capabilities.json for {self.model}")
        return max_tokens

    def _prepare_generation_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare generation kwargs by translating unified token parameters
        to provider-specific ones.

        Args:
            **kwargs: Generation parameters including unified token params

        Returns:
            Dictionary with provider-specific parameters
        """
        # Get effective token limits
        max_tokens, max_output_tokens, max_input_tokens = self._calculate_effective_token_limits()

        # Override max_output_tokens if provided in kwargs
        effective_max_output = kwargs.get("max_output_tokens", max_output_tokens)
        # Safety clamp: never exceed the provider/model's configured max_output_tokens.
        #
        # Upstream callers (runtimes/agents) may request large output budgets based on
        # stale capabilities or user configuration. Providers should not forward values
        # that violate the model's hard limits (Anthropic returns 400 for this).
        try:
            if effective_max_output is None:
                effective_max_output_i = int(max_output_tokens)
            else:
                effective_max_output_i = int(effective_max_output)
        except Exception:
            effective_max_output_i = int(max_output_tokens)
        if effective_max_output_i <= 0:
            effective_max_output_i = int(max_output_tokens)
        if effective_max_output_i > int(max_output_tokens):
            effective_max_output_i = int(max_output_tokens)

        # Return base kwargs with unified parameter
        result_kwargs = kwargs.copy()
        result_kwargs["max_output_tokens"] = effective_max_output_i

        # Add unified generation parameters with fallback hierarchy: kwargs → instance → defaults
        temperature = result_kwargs.get("temperature", self.temperature)
        if temperature is None:
            temperature = self.temperature
        result_kwargs["temperature"] = temperature

        seed_value = self._normalize_seed(result_kwargs.get("seed", self.seed))
        if seed_value is not None:
            result_kwargs["seed"] = seed_value
        else:
            # Do not forward seed when unset/random (None or negative sentinel like -1).
            result_kwargs.pop("seed", None)

        return result_kwargs

    @staticmethod
    def _normalize_seed(seed: Any) -> Optional[int]:
        """Normalize seed semantics across providers.

        - None or any negative value -> None (meaning: don't send a provider seed / random).
        - Non-bool numeric-ish values -> int(seed) if >= 0.
        """
        try:
            if seed is None:
                return None
            if isinstance(seed, bool):
                return None
            seed_i = int(seed)
            return seed_i if seed_i >= 0 else None
        except Exception:
            return None

    def _extract_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Extract generation parameters with consistent fallback hierarchy.
        
        Returns:
            Dict containing temperature, seed, and other generation parameters
        """
        params = {}
        
        # Temperature (always present)
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is None:
            temperature = self.temperature
        params["temperature"] = temperature
        
        # Seed (only if not None)
        seed_value = self._normalize_seed(kwargs.get("seed", self.seed))
        if seed_value is not None:
            params["seed"] = seed_value
            
        return params

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """
        Extract the appropriate max tokens parameter for this provider.
        This should be overridden by subclasses to return the provider-specific
        parameter name and value.

        Args:
            kwargs: Generation parameters

        Returns:
            Max output tokens for the provider's API
        """
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _handle_prompted_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]], execute_tools_param: bool = None) -> GenerateResponse:
        """Handle tool execution for prompted responses (shared implementation)"""
        if not response.content:
            return response

        # Parse tool calls from response content using UniversalToolHandler
        tool_call_response = self.tool_handler.parse_response(response.content, mode="prompted")
        tool_calls = tool_call_response.tool_calls

        if not tool_calls:
            return response

        # Execute with events and return result
        return self._execute_tools_with_events(response, tool_calls, execute_tools_param)

    def _execute_tools_with_events(self, response: GenerateResponse, tool_calls: List, execute_tools_param: bool = None) -> GenerateResponse:
        """Core tool execution with event emission (shared implementation)"""
        # Check if tool execution is enabled
        should_execute = execute_tools_param if execute_tools_param is not None else self.execute_tools
        
        if not should_execute:
            # Tool execution disabled - return response with tool calls but don't execute
            self.logger.info("Tool execution disabled - returning response with tool calls")
            return response
        
        # Emit tool started event
        event_data = {
            "tool_calls": [{
                "name": call.name,
                "arguments": call.arguments
            } for call in tool_calls],
            "tool_count": len(tool_calls),
            "model": self.model,
            "provider": self.__class__.__name__
        }

        from ..events import emit_global
        emit_global(EventType.TOOL_STARTED, event_data, source=self.__class__.__name__)

        # Execute tools
        tool_results = execute_tools(tool_calls)

        # Emit tool completed event
        after_event_data = {
            "tool_results": [{
                "name": call.name,
                "success": result.success,
                "error": str(result.error) if result.error else None
            } for call, result in zip(tool_calls, tool_results)],
            "successful_count": sum(1 for r in tool_results if r.success),
            "failed_count": sum(1 for r in tool_results if not r.success),
            "model": self.model,
            "provider": self.__class__.__name__
        }

        emit_global(EventType.TOOL_COMPLETED, after_event_data, source=self.__class__.__name__)

        # Track tool calls
        for call, result in zip(tool_calls, tool_results):
            self._track_tool_call(
                tool_name=call.name,
                arguments=call.arguments,
                success=result.success,
                error=result.error if not result.success else None
            )

        # Format tool results and append to response
        results_text = self._format_tool_results(tool_calls, tool_results)

        # Return updated response with tool results
        # Use the cleaned content from tool parsing
        tool_call_response = self.tool_handler.parse_response(response.content, mode="prompted")
        cleaned_content = tool_call_response.content or ""
        return GenerateResponse(
            content=cleaned_content + results_text,
            model=response.model,
            finish_reason=response.finish_reason,
            raw_response=response.raw_response,
            usage=response.usage,
            tool_calls=response.tool_calls,  # Keep original format
            metadata=response.metadata,
        )

    def _format_tool_results(self, tool_calls: List, tool_results: List) -> str:
        """Format tool results with tool transparency (shared implementation)"""
        results_text = "\n\nTool Results:\n"
        for call, result in zip(tool_calls, tool_results):
            # Format parameters for display (limit size)
            params_str = preview_text(str(call.arguments) if call.arguments else "{}", max_chars=100)

            # Show tool name and parameters for transparency
            results_text += f"🔧 Tool: {call.name}({params_str})\n"

            # Show result
            if result.success:
                results_text += f"- {result.output}\n"
            else:
                results_text += f"- Error: {result.error}\n"
            results_text += "\n"  # Add spacing between tool calls

        return results_text

    def _convert_native_tool_calls_to_standard(self, native_tool_calls: List[Dict[str, Any]]) -> List:
        """Convert native API tool calls to standard ToolCall objects (shared implementation)"""
        from ..tools.core import ToolCall
        import json

        tool_calls = []
        for call in native_tool_calls:
            arguments = call.get('arguments', {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(ToolCall(
                name=call.get('name', ''),
                arguments=arguments,
                call_id=call.get('id')
            ))
        return tool_calls

    # Timeout management methods
    def get_timeout(self) -> float:
        """Get the current HTTP request timeout in seconds."""
        return self._timeout

    def set_timeout(self, timeout: float) -> None:
        """Set the HTTP request timeout in seconds."""
        self._timeout = timeout
        # Update HTTP clients if they exist
        self._update_http_client_timeout()

    def get_recovery_timeout(self) -> float:
        """Get the current circuit breaker recovery timeout in seconds."""
        return self.retry_manager.config.recovery_timeout

    def set_recovery_timeout(self, timeout: float) -> None:
        """Set the circuit breaker recovery timeout in seconds."""
        self.retry_manager.config.recovery_timeout = timeout

    def get_tool_timeout(self) -> float:
        """Get the current tool execution timeout in seconds."""
        return self._tool_timeout

    def set_tool_timeout(self, timeout: float) -> None:
        """Set the tool execution timeout in seconds."""
        self._tool_timeout = timeout

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout if the provider has one. Override in subclasses."""
        pass

    # Prompt cache management methods
    def supports_prompt_cache(self) -> bool:
        """Return True if this provider supports best-effort prompt caching.

        Semantics differ by provider:
        - Remote providers (OpenAI): `prompt_cache_key` is forwarded; cache is managed server-side.
        - Local providers (MLX / llama.cpp): in-process KV/prefix caches can be retained across calls.
        """
        return False

    # Provider-specific prompt cache backend hooks (optional)
    #
    # Providers that implement in-process KV caching (MLX, llama.cpp, etc.) can override these to enable
    # `prompt_cache_update`, `prompt_cache_fork`, and `prompt_cache_prepare_modules`.
    def _prompt_cache_backend_create(self) -> Optional[Any]:
        return None

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        _ = cache_value
        return None

    def _prompt_cache_backend_append(
        self,
        cache_value: Any,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> bool:
        _ = (cache_value, prompt, messages, system_prompt, tools, add_generation_prompt, kwargs)
        return False

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        _ = cache_value
        return None

    def _normalize_prompt_cache_key(self, key: Any) -> Optional[str]:
        if not isinstance(key, str):
            return None
        key = key.strip()
        return key if key else None

    def _apply_default_prompt_cache_key(self, kwargs: Dict[str, Any]) -> None:
        # Explicit caller override wins (even if None / empty to disable).
        if "prompt_cache_key" in kwargs:
            kwargs["prompt_cache_key"] = self._normalize_prompt_cache_key(kwargs.get("prompt_cache_key"))
            return

        if self._default_prompt_cache_key and self.supports_prompt_cache():
            kwargs["prompt_cache_key"] = self._default_prompt_cache_key

    def get_prompt_cache_stats(self) -> Dict[str, Any]:
        """Return basic prompt cache stats (in-process store only)."""
        stats = self._prompt_cache_store.stats()
        stats["default_key"] = self._default_prompt_cache_key
        try:
            keys = self._prompt_cache_store.keys()
            if isinstance(keys, list):
                stats["keys"] = list(keys)
                meta_by_key: Dict[str, Any] = {}
                for k in keys:
                    meta = self._prompt_cache_store.meta(k)
                    if isinstance(meta, dict) and meta:
                        meta_by_key[str(k)] = dict(meta)
                if meta_by_key:
                    stats["meta_by_key"] = meta_by_key
        except Exception:
            pass
        return stats

    def prompt_cache_set(self, key: str, *, make_default: bool = True, **kwargs) -> bool:
        """Set the default prompt cache key for this provider instance.

        Provider-specific cache allocation/warming is implemented by subclasses when applicable.
        """
        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return False
        if not self.supports_prompt_cache():
            return False
        _ = kwargs
        # Best-effort: allocate backend cache if the provider supports it.
        if self._prompt_cache_store.get(normalized) is None:
            created = self._prompt_cache_backend_create()
            if created is not None:
                try:
                    self._prompt_cache_store.set(normalized, created, meta={"backend": "provider"})
                except Exception:
                    pass
        if make_default:
            self._default_prompt_cache_key = normalized
        return True

    def prompt_cache_update(
        self,
        key: str,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Append new prompt context into an existing cache key (best-effort).

        Semantics:
        - Local runtimes can implement true KV prefill updates (append-only).
        - Remote providers typically cannot be “pre-filled” explicitly; they may ignore this.

        Arguments are intentionally similar to `generate()` so higher-level code can reuse its own
        prompt/module construction logic.
        """
        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return False
        if not self.supports_prompt_cache():
            return False

        # Ensure the cache exists if the provider can allocate a backend cache object.
        cache_value = self._prompt_cache_store.get(normalized)
        if cache_value is None:
            if not self.prompt_cache_set(normalized, make_default=False):
                return False
            cache_value = self._prompt_cache_store.get(normalized)
            if cache_value is None:
                return False

        ok = self._prompt_cache_backend_append(
            cache_value,
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=bool(add_generation_prompt),
            **kwargs,
        )
        if not ok:
            return False

        # Update TTL/metadata best-effort.
        if ttl_s is not None:
            try:
                meta = self._prompt_cache_store.meta(normalized) or {}
                self._prompt_cache_store.set(normalized, cache_value, ttl_s=ttl_s, meta=meta)
            except Exception:
                pass
        return True

    def prompt_cache_fork(
        self,
        from_key: str,
        to_key: str,
        *,
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Create a new cache key by cloning another cache (best-effort).

        This is the primitive needed for hierarchical/module caches:
        - build stable shared prefixes (persona, memory blueprints, tool schemas)
        - fork them into per-session caches that can be appended/mutated safely.
        """
        _ = kwargs
        src = self._normalize_prompt_cache_key(from_key)
        dst = self._normalize_prompt_cache_key(to_key)
        if src is None or dst is None:
            return False
        if not self.supports_prompt_cache():
            return False

        src_value = self._prompt_cache_store.get(src)
        if src_value is None:
            return False

        cloned = self._prompt_cache_backend_clone(src_value)
        if cloned is None:
            return False

        try:
            meta = self._prompt_cache_store.meta(src) or {}
            meta = dict(meta)
            meta.setdefault("forked_from", src)
            self._prompt_cache_store.set(dst, cloned, ttl_s=ttl_s, meta=meta)
        except Exception:
            return False

        if make_default:
            self._default_prompt_cache_key = dst
        return True

    def prompt_cache_prepare_modules(
        self,
        *,
        namespace: str,
        modules: List[Union[PromptCacheModule, Dict[str, Any]]],
        make_default: bool = False,
        ttl_s: Optional[float] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        """Ensure hierarchical prefix caches exist for an ordered module list (best-effort).

        This builds immutable prefix caches (by derived keys) so callers can:
        - reuse stable sub-prefixes (persona, memory blueprints, etc.)
        - fork the final prefix into a per-session cache for incremental chat

        Returns a JSON-serializable dict containing per-module derived keys.
        """
        ns = str(namespace or "").strip()
        if not ns:
            return {"supported": False, "error": "namespace required"}
        if not self.supports_prompt_cache():
            return {"supported": False, "error": "provider does not support prompt caching"}

        normalized_modules: List[PromptCacheModule] = []
        for m in modules or []:
            if isinstance(m, PromptCacheModule):
                normalized_modules.append(m.normalized())
            elif isinstance(m, dict):
                try:
                    normalized_modules.append(PromptCacheModule(**m).normalized())
                except Exception:
                    continue

        if not normalized_modules:
            return {"supported": False, "error": "no modules provided"}

        # Derive deterministic prefix keys per module boundary.
        prefix_hash = hashlib.sha256(f"acore-prompt-cache:{int(version)}".encode("utf-8")).hexdigest()
        derived: List[Dict[str, Any]] = []
        keys: List[str] = []
        for mod in normalized_modules:
            prefix_hash = hashlib.sha256((prefix_hash + mod.fingerprint(version=version)).encode("utf-8")).hexdigest()
            key = f"{ns}:{prefix_hash[:16]}"
            keys.append(key)
            derived.append({"module_id": mod.module_id, "cache_key": key, "module_hash": mod.fingerprint(version=version)})

        # Find the longest existing prefix cache.
        start_idx = -1
        for i, key in enumerate(keys):
            if self._prompt_cache_store.get(key) is None:
                break
            start_idx = i

        # Start from existing prefix (clone to avoid mutating the stored snapshot).
        current_cache: Optional[Any] = None
        if start_idx >= 0:
            existing = self._prompt_cache_store.get(keys[start_idx])
            if existing is not None:
                current_cache = self._prompt_cache_backend_clone(existing) or None

        # If we have no starting cache, start from empty backend cache.
        if current_cache is None:
            current_cache = self._prompt_cache_backend_create()
            if current_cache is None:
                return {"supported": False, "error": "provider does not implement in-process cache backend"}

        # Build missing caches.
        for j in range(start_idx + 1, len(keys)):
            mod = normalized_modules[j]
            ok = self._prompt_cache_backend_append(
                current_cache,
                prompt=str(mod.prompt or ""),
                messages=mod.messages,
                system_prompt=mod.system_prompt,
                tools=mod.tools,
                add_generation_prompt=bool(mod.add_generation_prompt),
            )
            if not ok:
                return {"supported": False, "error": f"failed to append module '{mod.module_id}'"}

            snapshot = self._prompt_cache_backend_clone(current_cache) or None
            if snapshot is None:
                return {"supported": False, "error": "provider does not support cache cloning"}

            meta = {
                "namespace": ns,
                "module_id": mod.module_id,
                "module_hash": mod.fingerprint(version=version),
                "index": j,
                "backend": "provider",
            }
            tok = self._prompt_cache_backend_token_count(snapshot)
            if isinstance(tok, int) and tok >= 0:
                meta["token_count"] = tok

            self._prompt_cache_store.set(keys[j], snapshot, ttl_s=ttl_s, meta=meta)

        if make_default:
            self._default_prompt_cache_key = keys[-1]

        return {
            "supported": True,
            "namespace": ns,
            "version": int(version),
            "modules": derived,
            "final_cache_key": keys[-1],
        }

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        """Clear prompt caches for this provider instance (best-effort)."""
        normalized = self._normalize_prompt_cache_key(key) if key is not None else None
        if not self.supports_prompt_cache():
            return False

        if normalized is None:
            self._default_prompt_cache_key = None
            self._prompt_cache_store.clear()
            return True

        cleared = self._prompt_cache_store.delete(normalized)
        if self._default_prompt_cache_key == normalized:
            self._default_prompt_cache_key = None
        return cleared

    # Memory management methods
    @abstractmethod
    def unload_model(self, model_name: str) -> None:
        """
        Unload/cleanup resources for a specific model.

        This is the single canonical unload entrypoint across providers.
        Providers must implement this as a best-effort cleanup hook:

        - In-process providers (e.g. MLX, HuggingFace): free local model resources.
        - Some self-hosted servers (e.g. Ollama): may request server-side eviction/unload.
        - OpenAI-compatible servers (e.g. LMStudio, vLLM, openai-compatible): typically only close client
          connections; server-side model unloading may not be available and is controlled by the server (TTL/eviction).
        - Cloud APIs (e.g. OpenAI, Anthropic): usually a no-op (safe to call).
        """

    # Token configuration helpers - expose interface methods for user convenience
    def get_token_configuration_summary(self) -> str:
        """Get a human-readable summary of current token configuration"""
        return super().get_token_configuration_summary()

    def validate_token_constraints(self) -> List[str]:
        """Validate token configuration and return warnings/suggestions"""
        warnings_list = super().validate_token_constraints()

        # Embedding models are not text-generative: output token limits are irrelevant and can
        # legitimately be 0 (e.g. Nomic Embed). Suppress misleading output-token warnings.
        try:
            caps = getattr(self, "model_capabilities", None)
            model_type = caps.get("model_type") if isinstance(caps, dict) else None
            if isinstance(model_type, str) and model_type.strip().lower() == "embedding":
                warnings_list = [w for w in warnings_list if "max_output_tokens" not in str(w)]
        except Exception:
            pass

        return warnings_list

    def calculate_token_budget(self, input_text: str, desired_output_tokens: int,
                              safety_margin: float = 0.1) -> tuple[int, List[str]]:
        """Helper to estimate required max_tokens given input and desired output"""
        return super().calculate_token_budget(input_text, desired_output_tokens, safety_margin)

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count for given text"""
        return super().estimate_tokens(text)

    def _process_media_content(self, media: List[Union[str, Dict[str, Any], 'MediaContent']], 
                              glyph_compression: str = "auto") -> List['MediaContent']:
        """
        Process media content from various input formats into standardized MediaContent objects.

        Args:
            media: List of media inputs (file paths, MediaContent objects, or dicts)
            glyph_compression: Glyph compression preference (auto, always, never)

        Returns:
            List of processed MediaContent objects

        Raises:
            ImportError: If media processing dependencies are not available
            ValueError: If media input format is invalid
        """
        if not media:
            return []

        try:
            # Import media handler components
            from ..media import AutoMediaHandler
            from ..media.types import MediaContent
        except ImportError as e:
            raise ImportError(
                f"Media processing requires additional dependencies. "
                f"Install with: pip install \"abstractcore[media]\". Error: {e}"
            )

        processed_media = []

        for i, media_item in enumerate(media):
            try:
                if isinstance(media_item, str):
                    # File path - process with auto media handler
                    handler = AutoMediaHandler(
                        enable_glyph_compression=True,
                        glyph_config=getattr(self, 'glyph_config', None)
                    )
                    result = handler.process_file(
                        media_item,
                        provider=self.provider,
                        model=self.model,
                        glyph_compression=glyph_compression
                    )
                    if result.success:
                        processed_media.append(result.media_content)
                    else:
                        self.logger.warning(f"Failed to process media file {media_item}: {result.error_message}")
                        continue

                elif hasattr(media_item, 'media_type'):
                    # Already a MediaContent object
                    processed_media.append(media_item)

                elif isinstance(media_item, dict):
                    # Dictionary format - convert to MediaContent
                    try:
                        media_content = MediaContent.from_dict(media_item)
                        processed_media.append(media_content)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert media dict at index {i}: {e}")
                        continue

                else:
                    self.logger.warning(f"Unsupported media type at index {i}: {type(media_item)}")
                    continue

            except Exception as e:
                self.logger.warning(f"Failed to process media item at index {i}: {e}")
                continue

        if not processed_media and media:
            self.logger.warning("No media items were successfully processed")

        return processed_media

    @abstractmethod
    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models for this provider.

        Each provider must implement this method to return a list of available models.
        The server will use this method to aggregate models across all providers.

        Args:
            **kwargs: Provider-specific parameters including:
                - api_key: API key for authentication (if required)
                - base_url: Base URL for API endpoint (if applicable)
                - input_capabilities: Optional list of ModelInputCapability enums to filter by input capability
                    (e.g., [ModelInputCapability.IMAGE] for vision models)
                - output_capabilities: Optional list of ModelOutputCapability enums to filter by output capability
                    (e.g., [ModelOutputCapability.EMBEDDINGS] for embedding models)

        Returns:
            List of model names available for this provider, optionally filtered by capabilities

        Examples:
            >>> from abstractcore.providers import OpenAIProvider
            >>> from abstractcore.providers.model_capabilities import ModelInputCapability, ModelOutputCapability
            >>>
            >>> # Get all models
            >>> all_models = OpenAIProvider.list_available_models(api_key="...")
            >>>
            >>> # Get models that can analyze images
            >>> vision_models = OpenAIProvider.list_available_models(
            ...     api_key="...",
            ...     input_capabilities=[ModelInputCapability.IMAGE]
            ... )
            >>>
            >>> # Get embedding models
            >>> embedding_models = OpenAIProvider.list_available_models(
            ...     api_key="...",
            ...     output_capabilities=[ModelOutputCapability.EMBEDDINGS]
            ... )
            >>>
            >>> # Get vision models that generate text (most common case)
            >>> vision_text_models = OpenAIProvider.list_available_models(
            ...     api_key="...",
            ...     input_capabilities=[ModelInputCapability.TEXT, ModelInputCapability.IMAGE],
            ...     output_capabilities=[ModelOutputCapability.TEXT]
            ... )

        Note:
            This is an abstract method that MUST be implemented by all provider subclasses.
            Each provider should implement its own discovery logic (API calls, local scanning, etc.).
            Providers should apply the capability filters if provided in kwargs.
        """
        pass

    def health(self, timeout: Optional[float] = 5.0) -> Dict[str, Any]:
        """
        Check provider health and connectivity.

        This method tests if the provider is online and accessible by attempting
        to list available models. A successful model listing indicates the provider
        is healthy and ready to serve requests.

        Args:
            timeout: Maximum time in seconds to wait for health check (default: 5.0).
                    None means unlimited timeout (not recommended for health checks).

        Returns:
            Dict with health status information:
            {
                "status": bool,              # True if provider is healthy/online
                "provider": str,             # Provider class name (e.g., "OpenAIProvider")
                "models": List[str] | None,  # Available models if online, None if offline
                "model_count": int,          # Number of models available (0 if offline)
                "error": str | None,         # Error message if offline, None if healthy
                "latency_ms": float          # Time taken for health check in milliseconds
            }

        Example:
            >>> provider = OllamaProvider(model="llama2")
            >>> health = provider.health(timeout=3.0)
            >>> if health["status"]:
            >>>     print(f"Healthy! {health['model_count']} models available")
            >>> else:
            >>>     print(f"Offline: {health['error']}")

        Note:
            - This method never raises exceptions; errors are captured in the response
            - Uses list_available_models() as the connectivity test
            - Providers can override this method for custom health check logic
        """
        import time as time_module

        start_time = time_module.time()
        provider_name = self.__class__.__name__

        try:
            # Attempt to list models as connectivity test
            # Store original timeout if provider has HTTP client
            original_timeout = None
            timeout_changed = False

            if timeout is not None and hasattr(self, '_timeout'):
                original_timeout = self._timeout
                if original_timeout != timeout:
                    self.set_timeout(timeout)
                    timeout_changed = True

            try:
                models = self.list_available_models()

                # Restore original timeout if changed
                if timeout_changed and original_timeout is not None:
                    self.set_timeout(original_timeout)

                latency_ms = (time_module.time() - start_time) * 1000

                return {
                    "status": True,
                    "provider": provider_name,
                    "models": models,
                    "model_count": len(models) if models else 0,
                    "error": None,
                    "latency_ms": round(latency_ms, 2)
                }

            except Exception as e:
                # Restore original timeout on error
                if timeout_changed and original_timeout is not None:
                    try:
                        self.set_timeout(original_timeout)
                    except:
                        pass  # Best effort restoration
                raise  # Re-raise to be caught by outer handler

        except Exception as e:
            latency_ms = (time_module.time() - start_time) * 1000

            # Extract meaningful error message
            error_message = str(e)
            if not error_message:
                error_message = f"{type(e).__name__} occurred during health check"

            return {
                "status": False,
                "provider": provider_name,
                "models": None,
                "model_count": 0,
                "error": error_message,
                "latency_ms": round(latency_ms, 2)
            }

    def _needs_tag_rewriting(self, tool_call_tags) -> bool:
        """Check if tag rewriting is needed (tags are non-standard)"""
        try:
            from ..tools.tag_rewriter import ToolCallTags

            if isinstance(tool_call_tags, str):
                # String format - handle comma-separated format
                if ',' in tool_call_tags:
                    # Comma-separated format like '<function_call>,</function_call>'
                    parts = tool_call_tags.split(',')
                    if len(parts) == 2:
                        opening_tag = parts[0].strip()
                        closing_tag = parts[1].strip()
                        if opening_tag == "<function_call>" and closing_tag == "</function_call>":
                            return False
                else:
                    # Single tag format
                    if tool_call_tags in ["<function_call>", "</function_call>"]:
                        return False
            elif isinstance(tool_call_tags, ToolCallTags):
                # ToolCallTags object - check if it contains standard tags
                if (hasattr(tool_call_tags, 'start_tag') and hasattr(tool_call_tags, 'end_tag')):
                    # Only standard if exactly matches the standard format
                    if (tool_call_tags.start_tag == "<function_call>" and tool_call_tags.end_tag == "</function_call>"):
                        return False

            # Any other format or non-standard tags need rewriting
            return True

        except Exception:
            # If we can't determine, err on the side of applying rewriting
            return True

    def _apply_non_streaming_tag_rewriting(self, response: GenerateResponse, tool_call_tags: Optional[str] = None) -> GenerateResponse:
        """Apply tag rewriting to non-streaming response content."""
        try:
            from .streaming import UnifiedStreamProcessor

            # Create a temporary processor for tag rewriting
            processor = UnifiedStreamProcessor(
                model_name=self.model,
                execute_tools=False,  # No execution, just rewriting
                tool_call_tags=tool_call_tags,
                default_target_format="qwen3"  # Always rewrite to qwen3 format
            )

            # Apply tag rewriting to the content
            if processor.tag_rewriter and response.content:
                rewritten_content = processor._apply_tag_rewriting_direct(response.content)

                # Return new response with rewritten content
                return GenerateResponse(
                    content=rewritten_content,
                    model=response.model,
                    finish_reason=response.finish_reason,
                    raw_response=response.raw_response,
                    usage=response.usage,
                    tool_calls=response.tool_calls
                )

        except Exception as e:
            self.logger.debug(f"Non-streaming tag rewriting failed: {e}")

        # Return original response if rewriting fails
        return response

    def _normalize_tool_calls_passthrough(
        self,
        *,
        response: GenerateResponse,
        tools: List[Dict[str, Any]],
        tool_call_tags: Optional[str] = None,
    ) -> GenerateResponse:
        """Populate `response.tool_calls` (and usually clean `response.content`) in passthrough mode.

        Contract:
        - AbstractCore always returns structured `tool_calls` when tools are provided and the model emits tool syntax,
          even for prompted tool calling (tool calls embedded in `content`).
        - By default (`tool_call_tags is None`), tool-call markup is stripped from `content` for clean UX/history.
        - When `tool_call_tags` is set, we preserve `content` (for clients that parse tags) but still populate
          structured `tool_calls`.
        """

        # Only normalize when tools were actually provided.
        if not tools:
            return response

        allowed_names = self._get_allowed_tool_names(tools)

        # 1) If provider already returned tool_calls (native tools), normalize shape + args.
        normalized_existing = self._normalize_tool_calls_payload(
            response.tool_calls,
            allowed_tool_names=allowed_names,
        )
        if normalized_existing:
            response.tool_calls = normalized_existing

            # Clean any echoed tool syntax from content unless the caller explicitly requested tag passthrough.
            if self._should_clean_tool_call_markup(tool_call_tags) and isinstance(response.content, str) and response.content.strip():
                cleaned = self._clean_content_using_tool_calls(response.content, normalized_existing)
                response.content = cleaned

            return response

        # 2) Prompted tools: parse tool calls embedded in content.
        content = response.content
        if not isinstance(content, str) or not content.strip():
            return response

        tool_handler = getattr(self, "tool_handler", None)
        if tool_handler is None:
            return response

        try:
            parsed = tool_handler.parse_response(content, mode="prompted")
        except Exception:
            return response

        parsed_calls = getattr(parsed, "tool_calls", None)
        if not isinstance(parsed_calls, list) or not parsed_calls:
            return response

        normalized_parsed = self._normalize_tool_calls_payload(
            parsed_calls,
            allowed_tool_names=allowed_names,
        )
        if normalized_parsed:
            response.tool_calls = normalized_parsed

        # Always use the cleaned content from AbstractCore parsing when we are not explicitly preserving tags.
        if self._should_clean_tool_call_markup(tool_call_tags):
            cleaned_content = getattr(parsed, "content", None)
            if isinstance(cleaned_content, str):
                response.content = cleaned_content

        return response

    def _should_clean_tool_call_markup(self, tool_call_tags: Optional[str]) -> bool:
        """Return True when we should strip tool-call markup from assistant content."""
        if tool_call_tags is None:
            return True
        # OpenAI/Codex formats carry tool calls in structured fields, not in content.
        value = str(tool_call_tags).strip().lower()
        return value in {"openai", "codex"}

    def _get_allowed_tool_names(self, tools: List[Dict[str, Any]]) -> set[str]:
        """Extract allowed tool names from provider-normalized tool definitions."""
        names: set[str] = set()
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
                continue
            func = tool.get("function") if isinstance(tool.get("function"), dict) else None
            fname = func.get("name") if isinstance(func, dict) else None
            if isinstance(fname, str) and fname.strip():
                names.add(fname.strip())
        return names

    def _normalize_tool_calls_payload(
        self,
        tool_calls: Any,
        *,
        allowed_tool_names: Optional[set[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool call shapes into a canonical dict form.

        Canonical shape:
            {"name": str, "arguments": dict, "call_id": Optional[str]}
        """
        if tool_calls is None or not isinstance(tool_calls, list):
            return None

        def _unwrap_arguments(arguments: Any, *, expected_tool_name: Optional[str]) -> Any:
            """Unwrap common wrapper payloads to get tool kwargs.

            Some providers/models emit nested wrappers like:
              {"name":"tool","arguments":{...},"call_id": "..."}
            inside the tool call `arguments` field (or even multiple times).

            We unwrap when the object looks like a wrapper (only wrapper keys) OR when
            it includes wrapper metadata fields (e.g. "name"/"call_id") and an inner
            "arguments" dict. When wrapper fields and tool kwargs are partially mixed,
            we merge the outer kwargs into the inner dict (inner takes precedence).
            """
            if not isinstance(arguments, dict):
                return arguments

            wrapper_keys = {"name", "arguments", "call_id", "id"}
            current = arguments
            for _ in range(4):
                if not isinstance(current, dict):
                    break
                keys = set(current.keys())
                if "arguments" not in current:
                    break
                inner = current.get("arguments")
                if isinstance(inner, dict) or isinstance(inner, str):
                    inner_dict: Any = inner
                    if isinstance(inner, str):
                        parsed = loads_dict_like(inner)
                        inner_dict = parsed if isinstance(parsed, dict) else None
                    if not isinstance(inner_dict, dict):
                        break

                    name_matches = False
                    raw_name = current.get("name")
                    if isinstance(raw_name, str) and expected_tool_name and raw_name.strip() == expected_tool_name:
                        name_matches = True

                    wrapperish = keys.issubset(wrapper_keys) or name_matches or bool(keys & {"call_id", "id"})
                    if not wrapperish:
                        break

                    # Merge any outer kwargs that were accidentally placed alongside wrapper fields.
                    extras = {k: v for k, v in current.items() if k not in wrapper_keys}
                    if extras:
                        merged = dict(inner_dict)
                        for k, v in extras.items():
                            merged.setdefault(k, v)
                        current = merged
                    else:
                        current = inner_dict
                    continue
                break

            return current

        def _map_wrapped_name_to_allowed(raw: str, allowed: set[str]) -> Optional[str]:
            """Best-effort mapping when a provider returns a wrapped tool name.

            Some OpenAI-compatible servers/models occasionally return tool names wrapped in
            extra tokens/text (e.g. "{function-name: write_file}"). If we can confidently
            detect an allowed tool name as a standalone token within the raw string, map it
            back to the exact allowed name so tool execution can proceed.
            """
            s = str(raw or "").strip()
            if not s:
                return None
            if s in allowed:
                return s

            try:
                import re

                # Prefer exact token-boundary matches (tool names are usually snake_case).
                candidates: List[str] = []
                for name in allowed:
                    if not isinstance(name, str) or not name:
                        continue
                    pat = r"(^|[^\w])" + re.escape(name) + r"([^\w]|$)"
                    if re.search(pat, s):
                        candidates.append(name)
                if candidates:
                    # Prefer the most specific (longest) match deterministically.
                    return max(candidates, key=lambda n: (len(n), n))
            except Exception:
                return None

            return None

        normalized: List[Dict[str, Any]] = []

        for tc in tool_calls:
            name: Optional[str] = None
            arguments: Any = None
            call_id: Any = None

            if isinstance(tc, dict):
                call_id = tc.get("call_id", None)
                if call_id is None:
                    call_id = tc.get("id", None)

                raw_name = tc.get("name")
                raw_args = tc.get("arguments")

                func = tc.get("function") if isinstance(tc.get("function"), dict) else None
                if func and (not isinstance(raw_name, str) or not raw_name.strip()):
                    raw_name = func.get("name")
                if func and raw_args is None:
                    raw_args = func.get("arguments")

                if isinstance(raw_name, str) and raw_name.strip():
                    name = raw_name.strip()
                arguments = raw_args if raw_args is not None else {}
            else:
                raw_name = getattr(tc, "name", None)
                raw_args = getattr(tc, "arguments", None)
                call_id = getattr(tc, "call_id", None)
                if isinstance(raw_name, str) and raw_name.strip():
                    name = raw_name.strip()
                arguments = raw_args if raw_args is not None else {}

            if not isinstance(name, str) or not name:
                continue
            if isinstance(allowed_tool_names, set) and allowed_tool_names and name not in allowed_tool_names:
                mapped = _map_wrapped_name_to_allowed(name, allowed_tool_names)
                if not isinstance(mapped, str) or not mapped:
                    continue
                name = mapped

            if isinstance(arguments, str):
                parsed = loads_dict_like(arguments)
                arguments = parsed if isinstance(parsed, dict) else {}

            # Recover tool kwargs from nested wrapper payloads when present.
            if isinstance(arguments, dict) and call_id is None:
                wrapper_id = arguments.get("call_id") or arguments.get("id")
                if isinstance(wrapper_id, str) and wrapper_id.strip():
                    call_id = wrapper_id.strip()
            arguments = _unwrap_arguments(arguments, expected_tool_name=name)
            if not isinstance(arguments, dict):
                arguments = {}

            try:
                from ..tools.arg_canonicalizer import canonicalize_tool_arguments

                arguments = canonicalize_tool_arguments(name, arguments)
            except Exception:
                pass

            normalized.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "call_id": str(call_id) if call_id is not None else None,
                }
            )

        if not normalized:
            return None

        # Defense-in-depth: remove accidental duplicates introduced by overlapping parsing paths.
        unique: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for tc in normalized:
            try:
                args_key = json.dumps(tc.get("arguments", {}), sort_keys=True, ensure_ascii=False)
            except Exception:
                args_key = str(tc.get("arguments", {}))
            key = (str(tc.get("name") or ""), args_key)
            if key in seen:
                continue
            seen.add(key)
            unique.append(tc)

        return unique or None

    def _clean_content_using_tool_calls(self, content: str, tool_calls: List[Dict[str, Any]]) -> str:
        """Strip tool-call markup from assistant content using known tool calls."""
        try:
            from ..tools.core import ToolCall as CoreToolCall
            from ..tools.parser import clean_tool_syntax

            core_calls: List[CoreToolCall] = []
            for tc in tool_calls or []:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                args = tc.get("arguments")
                args_dict = dict(args) if isinstance(args, dict) else {}
                core_calls.append(CoreToolCall(name=name.strip(), arguments=args_dict, call_id=tc.get("call_id")))

            if not core_calls:
                return content
            return clean_tool_syntax(content, core_calls)
        except Exception:
            return content

    def _handle_tools_with_structured_output(self,
                                           prompt: str,
                                           messages: Optional[List[Dict[str, str]]] = None,
                                           system_prompt: Optional[str] = None,
                                           tools: Optional[List] = None,
                                           response_model: Optional[Type[BaseModel]] = None,
                                           retry_strategy=None,
                                           tool_call_tags: Optional[str] = None,
                                           execute_tools: Optional[bool] = None,
                                           stream: bool = False,
                                           **kwargs) -> BaseModel:
        """
        Handle the hybrid case: tools + structured output.
        
        Strategy: Sequential execution
        1. First, generate response with tools (may include tool calls)
        2. If tool calls are generated, execute them
        3. Then generate structured output using tool results as context
        
        Args:
            prompt: Input prompt
            messages: Optional message history
            system_prompt: Optional system prompt
            tools: List of available tools
            response_model: Pydantic model for structured output
            retry_strategy: Optional retry strategy for structured output
            tool_call_tags: Optional tool call tag format
            execute_tools: Whether to execute tools automatically
            stream: Whether to use streaming (not supported for hybrid mode)
            **kwargs: Additional parameters
            
        Returns:
            Validated instance of response_model
            
        Raises:
            ValueError: If streaming is requested (not supported for hybrid mode)
        """
        if stream:
            raise ValueError(
                "Streaming is not supported when combining tools with structured output. "
                "Please use either stream=True OR response_model, but not both."
            )
            
        # Step 1: Generate response with tools (normal tool execution flow)
        self.logger.info("Hybrid mode: Executing tools first, then structured output",
                        model=self.model,
                        response_model=response_model.__name__,
                        num_tools=len(tools) if tools else 0)
        
        # Force tool execution for hybrid mode
        should_execute_tools = execute_tools if execute_tools is not None else True
        
        # Generate response with tools using the normal flow (without response_model)
        tool_response = self.generate_with_telemetry(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            stream=False,  # Never stream in hybrid mode
            response_model=None,  # No structured output in first pass
            tool_call_tags=tool_call_tags,
            execute_tools=should_execute_tools,
            **kwargs
        )
        
        # Step 2: Generate structured output using tool results as context
        # Create enhanced prompt with tool execution context
        if hasattr(tool_response, 'content') and tool_response.content:
            enhanced_prompt = f"""{prompt}

Based on the following tool execution results:
{tool_response.content}

Please provide a structured response."""
        else:
            enhanced_prompt = prompt
            
        self.logger.info("Hybrid mode: Generating structured output with tool context",
                        model=self.model,
                        response_model=response_model.__name__,
                        has_tool_context=bool(hasattr(tool_response, 'content') and tool_response.content))
        
        # Generate structured output using the enhanced prompt
        from ..structured import StructuredOutputHandler
        handler = StructuredOutputHandler(retry_strategy=retry_strategy)
        
        structured_result = handler.generate_structured(
            provider=self,
            prompt=enhanced_prompt,
            response_model=response_model,
            messages=messages,
            system_prompt=system_prompt,
            tools=None,  # No tools in structured output pass
            stream=False,
            **kwargs
        )
        
        self.logger.info("Hybrid mode: Successfully completed tools + structured output",
                        model=self.model,
                        response_model=response_model.__name__,
                        success=True)
        
        return structured_result

    def generate(self,
                prompt: str,
                messages: Optional[List[Dict[str, str]]] = None,
                system_prompt: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                stream: bool = False,
                **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse], BaseModel]:
        """
        Generate response from the LLM.

        This method implements the AbstractCoreInterface and delegates to generate_with_telemetry.

        Args:
            prompt: The input prompt
            messages: Optional conversation history
            system_prompt: Optional system prompt
            tools: Optional list of available tools
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters (including response_model)

        Returns:
            GenerateResponse, iterator of GenerateResponse for streaming, or BaseModel for structured output
        """
        return self.generate_with_telemetry(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            stream=stream,
            **kwargs
        )

    async def agenerate(self,
                       prompt: str = "",
                       messages: Optional[List[Dict]] = None,
                       system_prompt: Optional[str] = None,
                       tools: Optional[List] = None,
                       media: Optional[List] = None,
                       stream: bool = False,
                       **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse], BaseModel]:
        """
        Async generation - works with all providers.

        Calls _agenerate_internal() which can be overridden for native async.
        Default implementation uses asyncio.to_thread() fallback.

        Args:
            prompt: Text prompt
            messages: Conversation history
            system_prompt: System instructions
            tools: Available tools
            media: Media attachments
            stream: Enable streaming
            **kwargs: Additional generation parameters (including response_model)

        Returns:
            GenerateResponse, AsyncIterator[GenerateResponse] for streaming, or BaseModel for structured output
        """
        self._apply_default_prompt_cache_key(kwargs)
        response = await self._agenerate_internal(
            prompt, messages, system_prompt, tools, media, stream, **kwargs
        )

        # Capture interaction trace if enabled (match sync generate_with_telemetry behavior)
        # Only for non-streaming responses that are GenerateResponse objects
        if not stream and self.enable_tracing and response and isinstance(response, GenerateResponse):
            trace_id = self._capture_trace(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                response=response,
                kwargs=kwargs
            )
            # Attach trace_id to response metadata
            if not response.metadata:
                response.metadata = {}
            response.metadata['trace_id'] = trace_id

        return response

    async def _agenerate_internal(self,
                                   prompt: str,
                                   messages: Optional[List[Dict]],
                                   system_prompt: Optional[str],
                                   tools: Optional[List],
                                   media: Optional[List],
                                   stream: bool,
                                   **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse], BaseModel]:
        """
        Internal async generation method.

        Default implementation: Uses asyncio.to_thread() to run sync generate().
        Providers override this for native async (3-10x faster for batch operations).

        Args:
            prompt: Text prompt
            messages: Conversation history
            system_prompt: System instructions
            tools: Available tools
            media: Media attachments
            stream: Enable streaming
            **kwargs: Additional generation parameters

        Returns:
            GenerateResponse, AsyncIterator[GenerateResponse] for streaming, or BaseModel for structured output
        """
        if stream:
            # Return async iterator for streaming
            return self._async_stream_generate(
                prompt, messages, system_prompt, tools, media, **kwargs
            )
        else:
            # Run sync generate in thread pool (fallback)
            return await asyncio.to_thread(
                self.generate,
                prompt, messages, system_prompt, tools, stream, **kwargs
            )

    async def _async_stream_generate(self,
                                     prompt: str,
                                     messages: Optional[List[Dict]],
                                     system_prompt: Optional[str],
                                     tools: Optional[List],
                                     media: Optional[List],
                                     **kwargs) -> AsyncIterator[GenerateResponse]:
        """
        Async streaming generator.

        Wraps sync streaming in async iterator, yielding control to event loop.
        """
        # Get sync generator in thread pool
        def get_sync_stream():
            return self.generate(
                prompt, messages, system_prompt, tools,
                stream=True, **kwargs
            )

        sync_gen = await asyncio.to_thread(get_sync_stream)

        # Yield chunks asynchronously
        for chunk in sync_gen:
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop
