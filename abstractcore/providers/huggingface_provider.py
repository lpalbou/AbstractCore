"""
HuggingFace provider implementation with GGUF support.
Supports both transformers models and GGUF models via llama-cpp-python.
"""

from __future__ import annotations

import importlib.util
import os
import copy
import json
import platform
import sys
import threading
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Type, TYPE_CHECKING

# Import config manager to respect offline-first settings
from ..config.manager import get_config_manager

# Get config instance and set offline environment variables if needed
_config = get_config_manager()
if _config.is_offline_first():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


TRANSFORMERS_AVAILABLE = _module_available("transformers")
LLAMACPP_AVAILABLE = _module_available("llama_cpp")
OUTLINES_AVAILABLE = _module_available("outlines")

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider, PromptCacheCapabilities, ThinkingControlHandling
from ..core.types import GenerateResponse
from ..exceptions import ModelNotFoundError, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

if TYPE_CHECKING:
    import torch
    from ..media.types import MediaContent


_MPS_GENERATION_LOCK = threading.Lock()
_AUTO_GROWING_LLAMA_RAM_CACHE_CLS = None

# We no longer download models - cache-only approach
# huggingface_hub not required for basic operation


def _get_local_model_path(model_name: str) -> Optional[str]:
    """Get local cache path for a HuggingFace model if it exists."""
    # Use centralized configuration for cache directory
    config = _config
    hf_cache_dir = Path(config.config.cache.huggingface_cache_dir).expanduser()

    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = hf_cache_dir / "hub" / model_cache_name / "snapshots"

    if model_cache_path.exists():
        snapshot_dirs = [d for d in model_cache_path.iterdir() if d.is_dir()]
        if snapshot_dirs:
            return str(snapshot_dirs[0])  # Return first snapshot
    return None


@dataclass
class _GGUFPromptCacheValue:
    cache: Any
    capacity_bytes: int
    system_prompt_parts: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: bool = False
    prompt_text: str = ""
    prompt_tokens: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class _TransformersPromptCacheValue:
    """Best-effort cache state for HuggingFace transformers KV reuse.

    `cache` is expected to be a `transformers.cache_utils.Cache` (typically `DynamicCache`).
    `prompt_tokens` tracks the token ids that have been prefetched into `cache` so the provider
    can build attention masks and compute delta lengths.
    """

    cache: Any
    prompt_tokens: tuple[int, ...] = field(default_factory=tuple)
    system_prompt_parts: List[str] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    add_generation_prompt: bool = False


class HuggingFaceProvider(BaseProvider):
    """HuggingFace provider with dual support for transformers and GGUF models"""

    @staticmethod
    def _resolve_requested_device(device: Optional[str]) -> Optional[str]:
        """Resolve the requested device from explicit arg or env override.

        Supported env var: ABSTRACTCORE_HF_DEVICE=cpu|mps|cuda|auto
        """
        if isinstance(device, str) and device.strip():
            val = device.strip().lower()
            return None if val == "auto" else val

        env_device = os.environ.get("ABSTRACTCORE_HF_DEVICE")
        if isinstance(env_device, str) and env_device.strip():
            val = env_device.strip().lower()
            if val in {"auto", "cpu", "mps", "cuda"}:
                return None if val == "auto" else val
        return None

    def __init__(self, model: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                 device: Optional[str] = None,
                 n_gpu_layers: Optional[int] = None,
                 structured_output_method: str = "auto",
                 **kwargs):

        # Handle legacy context_size parameter with deprecation warning
        context_size = kwargs.pop("context_size", None)
        if context_size is not None:
            import warnings
            warnings.warn(
                "The 'context_size' parameter is deprecated. Use 'max_tokens' instead. "
                "context_size will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = context_size

        user_provided_max_tokens = "max_tokens" in kwargs and kwargs.get("max_tokens") is not None

        super().__init__(model, **kwargs)
        self.provider = "huggingface"
        self._user_provided_max_tokens = bool(user_provided_max_tokens)

        # Handle timeout parameter for local models
        self._handle_timeout_parameter(kwargs)

        # Structured output method: "auto", "native_outlines", "prompted"
        # auto: Use Outlines if available (for transformers), otherwise prompted (default)
        # native_outlines: Force Outlines (error if unavailable)
        # prompted: Always use prompted fallback (fastest for transformers, still 100% success)
        # Note: GGUF models always use llama-cpp-python native support regardless of this setting
        self.structured_output_method = structured_output_method

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Store provider-specific configuration
        self.n_gpu_layers = n_gpu_layers
        self.model_type = None  # Will be "transformers" or "gguf"
        self.device = self._resolve_requested_device(device)

        # Store transformers-specific parameters
        self.transformers_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['trust_remote_code', 'torch_dtype', 'device_map', 'load_in_8bit', 'load_in_4bit', 'attn_implementation']
        }

        # Store device preference for custom models
        self.preferred_device = kwargs.get('device_map', 'auto')

        # Model instances
        self.tokenizer = None
        self.processor = None  # For vision models
        self.model_instance = None
        self.pipeline = None
        self.llm = None  # For GGUF models
        self._gguf_prompt_cache_lock = threading.Lock()
        self._gguf_prompt_cache_default_capacity_bytes = self._coerce_gguf_prompt_cache_capacity_bytes(
            kwargs.get("prompt_cache_capacity_bytes", None)
        )
        self._gguf_prompt_cache_pending_capacity_bytes: Optional[int] = None

        # Detect model type and load accordingly
        is_gguf = self._is_gguf_model(model)

        # LM Studio Hub aliases (e.g. "qwen/qwen3.5-35b-a3b") may not contain "gguf" in the name,
        # but resolve to a GGUF dependency stored in LM Studio's cache. If a local Hub manifest
        # exists and we can resolve a GGUF file from caches, treat this as a GGUF model.
        if not is_gguf:
            try:
                from ..utils.model_cache import resolve_lmstudio_hub_manifest

                if resolve_lmstudio_hub_manifest(model) is not None and self._find_gguf_in_cache(model):
                    is_gguf = True
            except Exception:
                pass

        if is_gguf:
            if not LLAMACPP_AVAILABLE:
                raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.model_type = "gguf"
            self._setup_device_gguf()
            self._load_gguf_model()
        else:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not installed. Install with: pip install transformers torch")
            self.model_type = "transformers"
            self._setup_device_transformers()
            self._load_transformers_model()

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        # llama-cpp-python (GGUF) does not currently expose chat-template kwargs such as
        # `enable_thinking` / `thinking_budget`. For Qwen3/Qwen3.5/Qwen3.6 GGUF models, we treat
        # thinking levels as a best-effort "enable thinking" request and rely on BaseProvider's
        # Qwen hard-switch marker for disabling thinking.
        #
        # TODO(upstream): Add an explicit `chat_template_kwargs` (or at least `enable_thinking`)
        # parameter to llama-cpp-python's `Llama.create_chat_completion()` and forward it into the
        # chat handler / Jinja template renderer. Once available, map our unified `thinking=...`
        # directly to `enable_thinking` instead of relying on the `<think>\n\n</think>\n\n` marker.
        # See: `docs/backlog/planned/2026-03-30_llama-cpp-python_expose_chat_template_kwargs.md`.
        model_type = str(getattr(self, "model_type", "") or "").strip().lower()
        if model_type == "gguf" and self.architecture in {"qwen3", "qwen3_5", "qwen3_6"}:
            if enabled is True or level is not None:
                return kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=False)
        return kwargs, ThinkingControlHandling()

    def unload_model(self, model_name: str) -> None:
        """
        Unload the model from memory.

        For GGUF models, calls llm.close() to free llama.cpp resources.
        For transformers models, clears model and tokenizer references.
        """
        import gc
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # Try to properly close the Llama object (GGUF models)
                if hasattr(self.llm, 'close'):
                    self.llm.close()
                # Clear the reference
                self.llm = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.tokenizer = None

            if hasattr(self, 'processor') and self.processor is not None:
                self.processor = None

            if hasattr(self, 'model') and hasattr(self, 'model') and self.model is not None:
                # For transformers models, clear the model
                self.model = None

            # Force garbage collection to free memory immediately
            gc.collect()
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def _coerce_gguf_prompt_cache_capacity_bytes(self, value: Any) -> int:
        try:
            cap = int(value)
        except Exception:
            cap = 0
        # `0` means "auto": the cache backend may resize to fit large prompts.
        return cap if cap > 0 else 0

    def _gguf_prompt_cache_chat_format(self) -> str:
        llm = getattr(self, "llm", None)
        chat_format = str(getattr(llm, "chat_format", "") or "").strip().lower()
        if chat_format:
            return chat_format
        model_lower = str(getattr(self, "model", "") or "").lower()
        if "qwen" in model_lower or "coder" in model_lower:
            return "chatml-function-calling"
        if "llama-3" in model_lower or "llama3" in model_lower:
            return "llama-3"
        return ""

    def _gguf_prompt_cache_control_plane_chat_format(self) -> str:
        chat_format = self._gguf_prompt_cache_chat_format()
        aliases = {
            "chatml-function-calling": "chatml-function-calling",
            "llama-3": "llama-3",
            "llama3": "llama-3",
        }
        return aliases.get(chat_format, "")

    def _gguf_prompt_cache_supports_local_control_plane(self) -> bool:
        if getattr(self, "model_type", None) != "gguf":
            return False
        return bool(self._gguf_prompt_cache_control_plane_chat_format())

    def _transformers_prompt_cache_supported(self) -> bool:
        if getattr(self, "model_type", None) != "transformers":
            return False
        if not TRANSFORMERS_AVAILABLE:
            return False
        if getattr(self, "tokenizer", None) is None or getattr(self, "model_instance", None) is None:
            return False
        if not hasattr(self.model_instance, "generate"):
            return False
        # Avoid claiming prompt-cache support for vision/custom models that do not follow the
        # decoder-only chat caching semantics.
        if getattr(self, "pipeline", None) is None:
            return False
        return True

    def supports_prompt_cache(self) -> bool:
        """Return True if this provider can retain an in-process prompt cache keyed by `prompt_cache_key`."""
        model_type = getattr(self, "model_type", None)
        if model_type == "gguf":
            return True
        return self._transformers_prompt_cache_supported()

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        """Return True when this provider can treat the prompt cache as the context source-of-truth."""
        return self._transformers_prompt_cache_supported()

    def get_prompt_cache_capabilities(self) -> PromptCacheCapabilities:
        if not self.supports_prompt_cache():
            return PromptCacheCapabilities()

        if getattr(self, "model_type", None) == "transformers":
            return PromptCacheCapabilities(
                supported=True,
                mode="local_control_plane",
                supports_set=True,
                supports_clear=True,
                supports_update=True,
                supports_fork=True,
                supports_prepare_modules=True,
                supports_stats=True,
                supports_save=True,
                supports_load=True,
                supports_ttl=True,
                notes=(
                    "Transformers prompt caching uses cross-call KV reuse (past_key_values / Cache).",
                    "Supports KV source-of-truth mode (delta-only prompts) via CachedSession.",
                    "cache_implementation=dynamic",
                ),
            )

        if self._gguf_prompt_cache_supports_local_control_plane():
            chat_format = self._gguf_prompt_cache_control_plane_chat_format()
            return PromptCacheCapabilities(
                supported=True,
                mode="local_control_plane",
                supports_set=True,
                supports_clear=True,
                supports_update=True,
                supports_fork=True,
                supports_prepare_modules=True,
                supports_stats=True,
                supports_save=True,
                supports_load=True,
                supports_ttl=True,
                notes=(
                    "GGUF prompt caching uses llama.cpp state snapshots plus keyed prefix reuse.",
                    f"exact_prompt_renderer={chat_format}",
                ),
            )

        chat_format = self._gguf_prompt_cache_chat_format() or "unknown"
        return PromptCacheCapabilities(
            supported=True,
            mode="keyed",
            supports_set=True,
            supports_clear=True,
            supports_update=False,
            supports_fork=False,
            supports_prepare_modules=False,
            supports_stats=True,
            supports_save=True,
            supports_load=True,
            supports_ttl=True,
            notes=(
                "GGUF prompt caching supports keyed cache selection for this model.",
                "Local control-plane parity currently requires an exact cached prompt renderer.",
                f"supported_renderers=chatml-function-calling,llama-3 current_chat_format={chat_format}",
            ),
        )

    def get_prompt_cache_stats(self) -> Dict[str, Any]:
        """Return prompt cache stats, including GGUF cache sizing (best-effort)."""
        stats = super().get_prompt_cache_stats()

        if str(getattr(self, "model_type", "") or "").strip().lower() != "gguf":
            return stats

        keys = stats.get("keys") if isinstance(stats, dict) else None
        if not isinstance(keys, list):
            return stats

        per_key: Dict[str, Any] = {}
        for key in keys:
            key_s = str(key)
            try:
                cache_value = self._prompt_cache_store.get(key_s)
            except Exception:
                continue

            state = self._gguf_prompt_cache_state(cache_value)
            if state is None:
                continue

            cache_obj = self._gguf_prompt_cache_unwrap(state)
            if cache_obj is None:
                continue

            cap_bytes = None
            try:
                cap_bytes = int(getattr(cache_obj, "capacity_bytes", None) or state.capacity_bytes)
            except Exception:
                try:
                    cap_bytes = int(state.capacity_bytes)
                except Exception:
                    cap_bytes = None

            cache_state = getattr(cache_obj, "cache_state", None)
            cache_entries: Optional[int] = None
            total_state_bytes: Optional[int] = None
            max_state_bytes: Optional[int] = None
            if hasattr(cache_state, "items"):
                total = 0
                max_b = 0
                count = 0
                try:
                    for _k, llama_state in cache_state.items():
                        count += 1
                        try:
                            size = int(getattr(llama_state, "llama_state_size", 0) or 0)
                        except Exception:
                            size = 0
                        if size > 0:
                            total += size
                            if size > max_b:
                                max_b = size
                except Exception:
                    count = 0
                    total = 0
                    max_b = 0
                cache_entries = int(count)
                total_state_bytes = int(total) if total > 0 else None
                max_state_bytes = int(max_b) if max_b > 0 else None

            per_key[key_s] = {
                "capacity_bytes": cap_bytes,
                "cache_state_entries": cache_entries,
                "cache_state_total_bytes": total_state_bytes,
                "cache_state_max_bytes": max_state_bytes,
                "prompt_tokens": int(len(state.prompt_tokens or ())),
                "prompt_text_chars": int(len(state.prompt_text or "")),
            }

        stats["gguf"] = {
            "control_plane_chat_format": (
                self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
            ),
            "keys": per_key,
        }
        return stats

    def _gguf_prompt_cache_export_state(self, cache_value: Any) -> Dict[str, Any]:
        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return {}
        cap_val = getattr(state.cache, "capacity_bytes", state.capacity_bytes)
        try:
            cap_i = int(cap_val)
        except Exception:
            cap_i = int(state.capacity_bytes)
        return {
            "capacity_bytes": cap_i,
            "system_prompt_parts": copy.deepcopy(state.system_prompt_parts),
            "messages": copy.deepcopy(state.messages),
            "tools": copy.deepcopy(state.tools),
            "add_generation_prompt": bool(state.add_generation_prompt),
            "prompt_text": str(state.prompt_text or ""),
            "prompt_tokens": [int(tok) for tok in state.prompt_tokens],
        }

    def _gguf_prompt_cache_import_state(self, cache_obj: Any, meta: Optional[Dict[str, Any]] = None) -> _GGUFPromptCacheValue:
        cap = getattr(cache_obj, "capacity_bytes", None)
        state = _GGUFPromptCacheValue(
            cache=cache_obj,
            capacity_bytes=self._coerce_gguf_prompt_cache_capacity_bytes(cap),
        )
        payload = dict(meta or {})
        raw_parts = payload.get("system_prompt_parts")
        if isinstance(raw_parts, list):
            state.system_prompt_parts = [str(part) for part in raw_parts if isinstance(part, str) and part]
        raw_messages = payload.get("messages")
        if isinstance(raw_messages, list):
            state.messages = [copy.deepcopy(msg) for msg in raw_messages if isinstance(msg, dict)]
        raw_tools = payload.get("tools")
        if isinstance(raw_tools, list):
            state.tools = [copy.deepcopy(tool) for tool in raw_tools if isinstance(tool, dict)]
        state.add_generation_prompt = bool(payload.get("add_generation_prompt"))
        if isinstance(payload.get("prompt_text"), str):
            state.prompt_text = str(payload.get("prompt_text") or "")
        raw_tokens = payload.get("prompt_tokens")
        if isinstance(raw_tokens, list):
            toks: List[int] = []
            for tok in raw_tokens:
                try:
                    toks.append(int(tok))
                except Exception:
                    continue
            state.prompt_tokens = tuple(toks)
        if not state.prompt_tokens:
            state.prompt_tokens = self._gguf_prompt_cache_longest_prefix_tokens(cache_obj)
        return state

    def _gguf_prompt_cache_state(self, cache_value: Any) -> Optional[_GGUFPromptCacheValue]:
        if isinstance(cache_value, _GGUFPromptCacheValue):
            return cache_value
        cache_obj = self._gguf_prompt_cache_unwrap(cache_value)
        if cache_obj is None:
            return None
        return self._gguf_prompt_cache_import_state(cache_obj, None)

    def _gguf_prompt_cache_unwrap(self, cache_value: Any) -> Optional[Any]:
        if isinstance(cache_value, _GGUFPromptCacheValue):
            return cache_value.cache
        if cache_value is None:
            return None
        try:
            from llama_cpp.llama_cache import LlamaRAMCache
        except Exception:
            return None
        return cache_value if isinstance(cache_value, LlamaRAMCache) else None

    def _gguf_prompt_cache_longest_prefix_tokens(self, cache_obj: Any) -> tuple[int, ...]:
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "keys"):
            return ()
        best: tuple[int, ...] = ()
        for key in state_map.keys():
            try:
                normalized = tuple(int(tok) for tok in key)
            except Exception:
                continue
            if len(normalized) > len(best):
                best = normalized
        return best

    def _gguf_clone_llama_state(self, state: Any) -> Optional[Any]:
        try:
            import numpy as np
            from llama_cpp.llama import LlamaState
        except Exception:
            return None
        try:
            return LlamaState(
                input_ids=np.asarray(getattr(state, "input_ids"), dtype=np.intc).copy(),
                scores=np.asarray(getattr(state, "scores"), dtype=np.single).copy(),
                n_tokens=int(getattr(state, "n_tokens", 0) or 0),
                llama_state=bytes(getattr(state, "llama_state", b"")),
                llama_state_size=int(getattr(state, "llama_state_size", 0) or 0),
                seed=int(getattr(state, "seed", 0) or 0),
            )
        except Exception:
            return None

    def _gguf_clone_llama_cache(self, cache_obj: Any, *, capacity_bytes: int) -> Optional[Any]:
        try:
            from llama_cpp.llama_cache import LlamaRAMCache
        except Exception:
            return None

        try:
            cap_i = int(capacity_bytes)
        except Exception:
            cap_i = 0
        if cap_i < 0:
            cap_i = 0

        # Preserve the concrete cache implementation (auto-growing vs fixed-capacity).
        cache_cls = cache_obj.__class__ if isinstance(cache_obj, LlamaRAMCache) else LlamaRAMCache
        try:
            cloned = cache_cls(capacity_bytes=int(cap_i))
        except Exception:
            cloned = LlamaRAMCache(capacity_bytes=int(cap_i))
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "items"):
            return cloned
        for key, state in state_map.items():
            cloned_state = self._gguf_clone_llama_state(state)
            if cloned_state is None:
                return None
            try:
                cloned[tuple(int(tok) for tok in key)] = cloned_state
            except Exception:
                return None
        return cloned

    def _gguf_build_chat_messages(
        self,
        *,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        user_message_content: Any = None,
    ) -> List[Dict[str, Any]]:
        chat_messages: List[Dict[str, Any]] = []

        if isinstance(system_prompt, str) and system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        if tools and self.tool_handler.supports_prompted:
            system_text = (
                chat_messages[0].get("content", "")
                if chat_messages and chat_messages[0].get("role") == "system"
                else ""
            )
            include_tool_list = "## Tools (session)" not in str(system_text)
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if tool_prompt:
                if self._gguf_prompt_cache_supports_local_control_plane():
                    # Keep tools in a stable prefix position so prefix/KV caches remain effective.
                    insert_at = 1 if chat_messages and chat_messages[0].get("role") == "system" else 0
                    chat_messages.insert(insert_at, {"role": "system", "content": tool_prompt})
                elif chat_messages and chat_messages[0].get("role") == "system":
                    chat_messages[0]["content"] = f"{chat_messages[0].get('content', '')}\n\n{tool_prompt}"
                else:
                    chat_messages.insert(0, {"role": "system", "content": tool_prompt})

        if isinstance(messages, list) and messages:
            chat_messages.extend(copy.deepcopy(messages))

        if user_message_content is not None:
            # Allow "messages-only" calls (prompt="") without appending an empty user turn.
            if isinstance(user_message_content, str):
                if user_message_content.strip():
                    chat_messages.append({"role": "user", "content": user_message_content})
            else:
                chat_messages.append({"role": "user", "content": user_message_content})

        return chat_messages

    def _gguf_prompt_cache_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)
        if content is None:
            return ""
        return str(content)

    def _gguf_prompt_cache_tool_call_text(self, tool_call: Any) -> str:
        if not isinstance(tool_call, dict):
            return ""
        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(fn.get("name") or tool_call.get("name") or "").strip()
        if not name:
            return ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, (dict, list)):
            try:
                args_text = json.dumps(raw_args, ensure_ascii=False)
            except Exception:
                args_text = str(raw_args)
        elif raw_args is None:
            args_text = ""
        else:
            args_text = str(raw_args)
        return f"functions.{name}:\n{args_text}"

    def _gguf_render_chatml_prompt(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role not in {"system", "user", "assistant"}:
                continue
            parts.append(f"<|im_start|>{role}\n")
            if role in {"system", "user"}:
                parts.append(self._gguf_prompt_cache_message_text(message.get("content")))
                parts.append("<|im_end|>\n")
                continue

            content = self._gguf_prompt_cache_message_text(message.get("content"))
            if content:
                parts.append(content)
                parts.append("<|im_end|>\n")

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for tool_call in tool_calls:
                    rendered = self._gguf_prompt_cache_tool_call_text(tool_call)
                    if rendered:
                        parts.append(rendered)
                parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _gguf_render_llama3_prompt(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
    ) -> str:
        role_map = {
            "system": "<|start_header_id|>system<|end_header_id|>\n\n",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        }
        sep = "<|eot_id|>"
        parts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            prefix = role_map.get(role)
            if not prefix:
                continue
            content = self._gguf_prompt_cache_message_text(message.get("content"))
            if content:
                parts.append(prefix)
                parts.append(content)
                parts.append(sep)
            else:
                parts.append(prefix)
        if add_generation_prompt:
            parts.append(role_map["assistant"])
        return "".join(parts)

    def _gguf_tokenize_completion_prompt(self, prompt_text: str) -> List[int]:
        if getattr(self, "llm", None) is None:
            return []
        bos_token_id = int(self.llm.token_bos())
        cls_token_id = int(self.llm._model.token_cls())
        bos_tokens: List[int] = [cls_token_id if cls_token_id != -1 else bos_token_id]

        if not self.llm._model.add_bos_token() or bos_tokens[:1] == [-1]:
            bos_tokens = []

        prefix_tokens = (
            self.llm.tokenize(
                prompt_text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
            if prompt_text != ""
            else []
        )
        # For modular prompt-cache prefixes we intentionally omit the terminal EOS that
        # llama.cpp adds for string prompts. Keeping EOS in the stored prefix would make
        # the key non-extendable (`prefix + eos` is not a prefix of `prefix + delta + eos`).
        return list(bos_tokens + prefix_tokens)

    def _gguf_render_prompt_tokens(
        self,
        *,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool,
    ) -> tuple[str, tuple[int, ...]]:
        chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
        if chat_format == "llama-3":
            prompt_text = self._gguf_render_llama3_prompt(
                messages=messages,
                add_generation_prompt=bool(add_generation_prompt),
            )
        else:
            prompt_text = self._gguf_render_chatml_prompt(
                messages=messages,
                add_generation_prompt=bool(add_generation_prompt),
            )
        if chat_format == "chatml":
            prompt_tokens = tuple(
                int(tok)
                for tok in self.llm.tokenize(
                    prompt_text.encode("utf-8"),
                    add_bos=True,
                    special=True,
                )
            )
        else:
            prompt_tokens = tuple(int(tok) for tok in self._gguf_tokenize_completion_prompt(prompt_text))
        return prompt_text, prompt_tokens

    def _gguf_prompt_cache_prefix_state(self, cache_obj: Any, prompt_tokens: tuple[int, ...]) -> tuple[int, Optional[Any]]:
        state_map = getattr(cache_obj, "cache_state", None)
        if not hasattr(state_map, "items") or not prompt_tokens:
            return 0, None

        llm = getattr(self, "llm", None)
        longest_prefix_fn = getattr(llm, "longest_token_prefix", None)
        if not callable(longest_prefix_fn):
            longest_prefix_fn = getattr(getattr(llm, "__class__", None), "longest_token_prefix", None)
        if not callable(longest_prefix_fn):
            try:
                from llama_cpp.llama import Llama as _Llama  # type: ignore

                longest_prefix_fn = getattr(_Llama, "longest_token_prefix", None)
            except Exception:
                longest_prefix_fn = None

        best_len = 0
        best_state = None
        for key, state in state_map.items():
            try:
                normalized = tuple(int(tok) for tok in key)
            except Exception:
                continue
            try:
                prefix_len = int(longest_prefix_fn(normalized, prompt_tokens)) if callable(longest_prefix_fn) else 0
            except Exception:
                prefix_len = 0
            if prefix_len != len(normalized):
                continue
            if len(normalized) > best_len:
                best_len = len(normalized)
                best_state = state
        return best_len, best_state

    def _gguf_prefill_prompt_cache(
        self,
        cache_obj: Any,
        prompt_tokens: tuple[int, ...],
        *,
        save_state: bool = True,
        set_cache: bool = True,
    ) -> bool:
        llm = getattr(self, "llm", None)
        if llm is None:
            return False
        try:
            llm.reset()
        except Exception:
            return False

        prefix_len, prefix_state = self._gguf_prompt_cache_prefix_state(cache_obj, prompt_tokens)
        if prefix_state is not None and prefix_len > 0:
            try:
                llm.load_state(prefix_state)
            except Exception:
                try:
                    llm.reset()
                except Exception:
                    pass
                prefix_len = 0

        remaining = list(prompt_tokens[prefix_len:])
        try:
            if remaining:
                llm.eval(remaining)
            if save_state and prompt_tokens:
                saved_state = llm.save_state()
                cloned_state = self._gguf_clone_llama_state(saved_state)
                cache_obj[prompt_tokens] = cloned_state if cloned_state is not None else saved_state
            if set_cache and hasattr(llm, "set_cache"):
                llm.set_cache(cache_obj)
        except Exception:
            return False
        return True

    def _transformers_prompt_cache_state(self, cache_value: Any) -> Optional[_TransformersPromptCacheValue]:
        return cache_value if isinstance(cache_value, _TransformersPromptCacheValue) else None

    def _transformers_cache_device(self) -> Optional["torch.device"]:
        try:
            import torch  # type: ignore
        except Exception:
            return None

        model = getattr(self, "model_instance", None)
        if model is not None:
            try:
                param = next(model.parameters(), None)
                if param is not None:
                    return param.device
            except Exception:
                pass
        dev = str(getattr(self, "device", "") or "").strip().lower()
        if dev in {"cuda", "mps", "cpu"}:
            try:
                return torch.device(dev)
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")

    def _transformers_cache_device_str(self) -> str:
        dev = self._transformers_cache_device()
        if dev is None:
            return "cpu"
        s = str(dev)
        if s.startswith("cuda"):
            return "cuda"
        if s.startswith("mps"):
            return "mps"
        return "cpu"

    def _transformers_arch_prefix_suffix(self, role: str) -> tuple[str, str]:
        cfg = getattr(self, "architecture_config", None)
        if not isinstance(cfg, dict):
            cfg = {}
        r = str(role or "").strip().lower()
        if r == "system":
            return str(cfg.get("system_prefix") or ""), str(cfg.get("system_suffix") or "")
        if r == "user":
            return str(cfg.get("user_prefix") or ""), str(cfg.get("user_suffix") or "")
        if r == "assistant":
            return str(cfg.get("assistant_prefix") or ""), str(cfg.get("assistant_suffix") or "")
        # Fallback: simple conversational format.
        return "", "\n"

    def _transformers_render_message(self, role: str, content: str, *, close: bool = True) -> str:
        prefix, suffix = self._transformers_arch_prefix_suffix(role)
        if prefix or suffix:
            out = f"{prefix}{content}"
            if close:
                out += suffix
            return out
        label = str(role or "user").strip().capitalize() or "User"
        out = f"{label}: {content}\n"
        return out if close else out.rstrip("\n")

    def _transformers_build_prompt_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[Union[List[str], tuple[str, ...]]] = None,
    ) -> str:
        """Build a prompt fragment that can be appended to an existing KV cache."""

        prefilled: set[str] = set()
        if prefilled_modules:
            for item in prefilled_modules:
                try:
                    norm = str(item or "").strip().lower()
                except Exception:
                    norm = ""
                if norm:
                    prefilled.add(norm)

        def _as_text(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        parts: List[str] = []

        base_system_prompt = str(system_prompt or "").strip() if system_prompt is not None else ""
        if base_system_prompt and "system" not in prefilled:
            parts.append(self._transformers_render_message("system", base_system_prompt, close=True))

        if tools is not None and getattr(self, "tool_handler", None) is not None:
            if getattr(self.tool_handler, "supports_prompted", False) and "tools" not in prefilled:
                include_tool_list = True
                if base_system_prompt and "## Tools (session)" in base_system_prompt:
                    include_tool_list = False
                try:
                    tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
                except Exception:
                    tool_prompt = ""
                tool_prompt = str(tool_prompt or "").strip()
                if tool_prompt:
                    parts.append(self._transformers_render_message("system", tool_prompt, close=True))

        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "user").strip().lower() or "user"
                if role in {"tool", "function"}:
                    role = "assistant"
                content = _as_text(msg.get("content"))
                parts.append(self._transformers_render_message(role, content, close=True))

        if isinstance(prompt, str) and prompt:
            parts.append(self._transformers_render_message("user", str(prompt), close=True))

        if add_generation_prompt:
            parts.append(self._transformers_render_message("assistant", "", close=False))

        return "".join(parts)

    def _transformers_tokenize_fragment(self, fragment: str, *, add_bos_if_empty: bool) -> List[int]:
        tok = getattr(self, "tokenizer", None)
        if tok is None:
            return []

        text = str(fragment or "")
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            try:
                ids = list(tok(text, add_special_tokens=False)["input_ids"])
            except Exception:
                return []

        out = [int(i) for i in ids] if ids else []
        if add_bos_if_empty:
            bos = getattr(tok, "bos_token_id", None)
            add_bos = getattr(tok, "add_bos_token", None)
            if isinstance(add_bos, bool) and not add_bos:
                return out
            try:
                bos_i = int(bos) if bos is not None else None
            except Exception:
                bos_i = None
            if bos_i is not None and bos_i >= 0:
                if not out or out[0] != bos_i:
                    out.insert(0, bos_i)
        return out

    def _transformers_prefill_cache(self, state: _TransformersPromptCacheValue, token_ids: List[int]) -> bool:
        if not token_ids:
            return True
        if getattr(self, "model_instance", None) is None:
            return False
        try:
            import torch  # type: ignore
        except Exception:
            return False

        device = self._transformers_cache_device() or torch.device("cpu")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        past_len = len(state.prompt_tokens)
        attention_mask = torch.ones((1, past_len + len(token_ids)), dtype=torch.long, device=device)

        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": state.cache,
        }

        try:
            with torch.inference_mode():
                use_mps_lock = str(device).startswith("mps") or str(getattr(self, "device", "") or "").strip().lower() == "mps"
                if use_mps_lock:
                    with _MPS_GENERATION_LOCK:
                        outputs = self.model_instance(**kwargs)
                else:
                    outputs = self.model_instance(**kwargs)
        except Exception:
            return False

        new_cache = getattr(outputs, "past_key_values", None)
        if new_cache is not None:
            state.cache = new_cache
        state.prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens + tuple(token_ids)))
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        if not self.supports_prompt_cache():
            return None

        model_type = getattr(self, "model_type", None)
        if model_type == "transformers":
            try:
                from transformers.cache_utils import DynamicCache  # type: ignore
            except Exception:
                return None
            return _TransformersPromptCacheValue(cache=DynamicCache())

        try:
            from llama_cpp.llama_cache import LlamaRAMCache
        except Exception:
            return None

        cap = getattr(self, "_gguf_prompt_cache_pending_capacity_bytes", None)
        cap = self._coerce_gguf_prompt_cache_capacity_bytes(
            cap if cap is not None else getattr(self, "_gguf_prompt_cache_default_capacity_bytes", None)
        )
        if cap > 0:
            cache_obj = LlamaRAMCache(capacity_bytes=int(cap))
            cap_effective = int(getattr(cache_obj, "capacity_bytes", cap) or cap)
        else:
            # Default is "auto": do not silently disable caching for large prompts.
            # When a single LlamaState exceeds a fixed capacity, llama-cpp-python evicts it
            # immediately (the cache stays empty). Auto-grow ensures at least the most recent
            # prefix KV snapshot can be retained for subsequent turns.
            global _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            cache_cls = _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            if cache_cls is None:
                class _AutoGrowingLlamaRAMCache(LlamaRAMCache):  # type: ignore[misc]
                    def __setitem__(self, key, value):  # type: ignore[override]
                        try:
                            state_size = int(getattr(value, "llama_state_size", 0) or 0)
                        except Exception:
                            state_size = 0
                        if state_size > 0:
                            try:
                                cap_now = int(getattr(self, "capacity_bytes", 0) or 0)
                            except Exception:
                                cap_now = 0
                            if state_size > cap_now:
                                # Grow just enough to retain this state (the base class eviction policy
                                # will still drop older entries as needed).
                                self.capacity_bytes = int(state_size)
                        return super().__setitem__(key, value)

                cache_cls = _AutoGrowingLlamaRAMCache
                _AUTO_GROWING_LLAMA_RAM_CACHE_CLS = cache_cls

            cache_obj = cache_cls(capacity_bytes=0)
            cap_effective = int(getattr(cache_obj, "capacity_bytes", 0) or 0)

        return _GGUFPromptCacheValue(
            cache=cache_obj,
            capacity_bytes=cap_effective,
        )

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        transformers_state = self._transformers_prompt_cache_state(cache_value)
        if transformers_state is not None:
            try:
                from transformers.cache_utils import DynamicCache  # type: ignore
            except Exception:
                return None
            src_cache = transformers_state.cache
            if not isinstance(src_cache, DynamicCache):
                return None

            cloned_cache = DynamicCache()
            for idx, layer in enumerate(getattr(src_cache, "layers", []) or []):
                try:
                    if not bool(getattr(layer, "is_initialized", False)):
                        continue
                    keys = getattr(layer, "keys", None)
                    values = getattr(layer, "values", None)
                    if keys is None or values is None:
                        continue
                    cloned_cache.update(keys, values, layer_idx=int(idx))
                except Exception:
                    return None

            return _TransformersPromptCacheValue(
                cache=cloned_cache,
                prompt_tokens=tuple(int(tok) for tok in transformers_state.prompt_tokens),
                system_prompt_parts=copy.deepcopy(transformers_state.system_prompt_parts),
                messages=copy.deepcopy(transformers_state.messages),
                tools=copy.deepcopy(transformers_state.tools),
                add_generation_prompt=bool(transformers_state.add_generation_prompt),
            )

        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return None
        cloned_cache = self._gguf_clone_llama_cache(state.cache, capacity_bytes=state.capacity_bytes)
        if cloned_cache is None:
            return None
        return _GGUFPromptCacheValue(
            cache=cloned_cache,
            capacity_bytes=int(state.capacity_bytes),
            system_prompt_parts=copy.deepcopy(state.system_prompt_parts),
            messages=copy.deepcopy(state.messages),
            tools=copy.deepcopy(state.tools),
            add_generation_prompt=bool(state.add_generation_prompt),
            prompt_text=str(state.prompt_text or ""),
            prompt_tokens=tuple(int(tok) for tok in state.prompt_tokens),
        )

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
        _ = kwargs

        transformers_state = self._transformers_prompt_cache_state(cache_value)
        if transformers_state is not None:
            prev_add_generation_prompt = bool(transformers_state.add_generation_prompt)
            prev_prompt_tokens = tuple(int(tok) for tok in (transformers_state.prompt_tokens or ()))

            # Mutate state first; we may rebuild or append depending on what changed.
            if system_prompt is not None:
                text = str(system_prompt or "").strip()
                if text:
                    transformers_state.system_prompt_parts.append(text)
            if tools is not None:
                transformers_state.tools = [copy.deepcopy(tool) for tool in tools if isinstance(tool, dict)] or None

            delta_messages: List[Dict[str, Any]] = []
            if isinstance(messages, list) and messages:
                for msg in messages:
                    if isinstance(msg, dict):
                        copied = copy.deepcopy(msg)
                        delta_messages.append(copied)
                        transformers_state.messages.append(copied)
            if isinstance(prompt, str) and prompt:
                user_msg = {"role": "user", "content": prompt}
                delta_messages.append(copy.deepcopy(user_msg))
                transformers_state.messages.append(user_msg)

            new_add_generation_prompt = bool(add_generation_prompt)
            transformers_state.add_generation_prompt = new_add_generation_prompt

            needs_rebuild = bool(system_prompt is not None or tools is not None or not prev_prompt_tokens)
            # Changing add_generation_prompt from True -> False is a structural edit; rebuild.
            if prev_add_generation_prompt and not new_add_generation_prompt:
                needs_rebuild = True
            # Appending after a generation prompt is ambiguous; rebuild for safety.
            if prev_add_generation_prompt and (delta_messages or system_prompt is not None or tools is not None):
                needs_rebuild = True

            if needs_rebuild:
                system_text = "\n\n".join(
                    part for part in transformers_state.system_prompt_parts if isinstance(part, str) and part
                )
                full_text = self._transformers_build_prompt_fragment(
                    prompt="",
                    messages=transformers_state.messages,
                    system_prompt=system_text or None,
                    tools=transformers_state.tools,
                    add_generation_prompt=transformers_state.add_generation_prompt,
                )

                token_ids = self._transformers_tokenize_fragment(full_text, add_bos_if_empty=True)
                try:
                    from transformers.cache_utils import DynamicCache  # type: ignore
                except Exception:
                    return False
                transformers_state.cache = DynamicCache()
                transformers_state.prompt_tokens = ()
                return self._transformers_prefill_cache(transformers_state, token_ids)

            # Incremental: append-only messages or toggling add_generation_prompt to True.
            delta_add_gen = bool(new_add_generation_prompt and not prev_add_generation_prompt)
            delta_text = self._transformers_build_prompt_fragment(
                prompt="",
                messages=delta_messages,
                system_prompt=None,
                tools=None,
                add_generation_prompt=delta_add_gen,
            )
            token_ids = self._transformers_tokenize_fragment(delta_text, add_bos_if_empty=False)
            return self._transformers_prefill_cache(transformers_state, token_ids)

        state = self._gguf_prompt_cache_state(cache_value)
        if state is None or getattr(self, "llm", None) is None:
            return False

        prev_add_generation_prompt = bool(state.add_generation_prompt)
        prev_prompt_text = str(state.prompt_text or "")
        prev_prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens or ()))

        if system_prompt is not None:
            text = str(system_prompt or "").strip()
            if text:
                state.system_prompt_parts.append(text)
        if tools is not None:
            state.tools = [copy.deepcopy(tool) for tool in tools if isinstance(tool, dict)] or None
        delta_messages: List[Dict[str, Any]] = []
        if isinstance(messages, list) and messages:
            for msg in messages:
                if isinstance(msg, dict):
                    copied = copy.deepcopy(msg)
                    delta_messages.append(copied)
                    state.messages.append(copied)
        if isinstance(prompt, str) and prompt:
            user_msg = {"role": "user", "content": prompt}
            delta_messages.append(copy.deepcopy(user_msg))
            state.messages.append(user_msg)
        new_add_generation_prompt = bool(add_generation_prompt)
        state.add_generation_prompt = new_add_generation_prompt

        if not self._gguf_prompt_cache_supports_local_control_plane():
            # Keyed-only GGUF caches still keep the in-process cache object, but they do not
            # advertise modular update/fork support to higher layers.
            return False

        # Fast path: append-only updates (no system/tools changes) can reuse the serialized prompt.
        can_incremental = (
            system_prompt is None
            and tools is None
            and prev_prompt_tokens
            and (not prev_add_generation_prompt)
            and (not new_add_generation_prompt)
        )
        if can_incremental and delta_messages:
            # Reject non-append semantics: new system messages belong in the prefix, not the tail.
            has_system_delta = any(
                str(m.get("role") or "").strip().lower() == "system" for m in delta_messages if isinstance(m, dict)
            )
            if not has_system_delta:
                chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
                if chat_format == "llama-3":
                    delta_text = self._gguf_render_llama3_prompt(messages=delta_messages, add_generation_prompt=False)
                else:
                    delta_text = self._gguf_render_chatml_prompt(messages=delta_messages, add_generation_prompt=False)

                try:
                    delta_tokens = (
                        tuple(
                            int(tok)
                            for tok in self.llm.tokenize(
                                delta_text.encode("utf-8"),
                                add_bos=False,
                                special=True,
                            )
                        )
                        if delta_text
                        else ()
                    )
                except Exception:
                    delta_tokens = ()

                if delta_tokens:
                    prompt_tokens = tuple(int(tok) for tok in (prev_prompt_tokens + delta_tokens))
                    with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
                        ok = self._gguf_prefill_prompt_cache(state.cache, prompt_tokens)
                    if not ok:
                        return False
                    state.prompt_text = prev_prompt_text + delta_text
                    state.prompt_tokens = prompt_tokens
                    return True

        system_text = "\n\n".join(part for part in state.system_prompt_parts if isinstance(part, str) and part)
        chat_messages = self._gguf_build_chat_messages(
            system_prompt=system_text or None,
            messages=state.messages,
            tools=state.tools,
            user_message_content=None,
        )
        prompt_text, prompt_tokens = self._gguf_render_prompt_tokens(
            messages=chat_messages,
            add_generation_prompt=state.add_generation_prompt,
        )

        with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
            ok = self._gguf_prefill_prompt_cache(state.cache, prompt_tokens)
        if not ok:
            return False

        state.prompt_text = prompt_text
        state.prompt_tokens = tuple(int(tok) for tok in prompt_tokens)
        return True

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        transformers_state = self._transformers_prompt_cache_state(cache_value)
        if transformers_state is not None:
            if transformers_state.prompt_tokens:
                return len(transformers_state.prompt_tokens)
            cache_obj = transformers_state.cache
            try:
                tok = cache_obj.get_seq_length() if cache_obj is not None else None
            except Exception:
                tok = None
            if isinstance(tok, int) and tok >= 0:
                return tok
            return None

        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return None
        longest = self._gguf_prompt_cache_longest_prefix_tokens(state.cache)
        if state.prompt_tokens:
            return max(len(state.prompt_tokens), len(longest))
        return len(longest)

    def prompt_cache_set(
        self,
        key: str,
        *,
        make_default: bool = True,
        ttl_s: Optional[float] = None,
        capacity_bytes: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Create/reset a prompt cache for the given key (best-effort)."""
        _ = kwargs
        if not self.supports_prompt_cache():
            return False

        if getattr(self, "model_type", None) != "gguf":
            ok = super().prompt_cache_set(key, make_default=make_default)
            if not ok:
                return False
            normalized = self._normalize_prompt_cache_key(key)
            if normalized is None:
                return False
            cache_value = self._prompt_cache_store.get(normalized)
            state = self._transformers_prompt_cache_state(cache_value)
            if state is None:
                return False
            try:
                self._prompt_cache_store.set(
                    normalized,
                    state,
                    ttl_s=ttl_s,
                    meta={"backend": "transformers"},
                )
            except Exception:
                return False
            return True

        self._gguf_prompt_cache_pending_capacity_bytes = self._coerce_gguf_prompt_cache_capacity_bytes(capacity_bytes)
        try:
            ok = super().prompt_cache_set(key, make_default=make_default)
        finally:
            self._gguf_prompt_cache_pending_capacity_bytes = None
        if not ok:
            return False

        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return False
        cache_value = self._prompt_cache_store.get(normalized)
        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            return False
        try:
            self._prompt_cache_store.set(
                normalized,
                state,
                ttl_s=ttl_s,
                meta={
                    "backend": "llama_cpp",
                    "capacity_bytes": int(getattr(state.cache, "capacity_bytes", state.capacity_bytes) or state.capacity_bytes),
                },
            )
        except Exception:
            return False

        try:
            if getattr(self, "llm", None) is not None and hasattr(self.llm, "set_cache"):
                self.llm.set_cache(state.cache)
        except Exception:
            pass

        return True

    def prompt_cache_clear(self, key: Optional[str] = None) -> bool:
        """Clear llama.cpp prompt caches (GGUF only; best-effort)."""
        cleared = super().prompt_cache_clear(key)
        llm = getattr(self, "llm", None)
        try:
            if llm is not None and hasattr(llm, "set_cache"):
                llm.set_cache(None)
        except Exception:
            pass
        # llama.cpp can still reuse in-process KV state via prefix matching even when no cache
        # object is configured. When clearing *all* caches, reset the runtime context as well so
        # "cache cleared" is observable in long-running processes (CLI/REPL).
        if key is None and str(getattr(self, "model_type", "") or "").strip().lower() == "gguf":
            try:
                if llm is not None and hasattr(llm, "reset"):
                    llm.reset()
            except Exception:
                pass
        return cleared

    def prompt_cache_save(self, key: str, filename: str, **kwargs: Any) -> Dict[str, Any]:
        """Save a GGUF llama.cpp prompt cache snapshot to disk (best-effort).

        This persists the provider-side cache metadata plus the *single* longest-prefix llama.cpp
        state snapshot for the key. It is sufficient to warm-start large chats without rebuilding
        the prefix, but it does not attempt to persist every intermediate prefix in the RAM cache.
        """
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        if getattr(self, "model_type", None) != "gguf":
            try:
                import torch  # type: ignore
                from safetensors.torch import save_file  # type: ignore
                from transformers.cache_utils import DynamicCache  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Transformers prompt cache saving requires `torch`, `transformers`, and `safetensors`."
                ) from e

            normalized = self._normalize_prompt_cache_key(key)
            if normalized is None:
                raise ValueError("prompt cache key must be a non-empty string")

            cache_value = self._prompt_cache_store.get(normalized)
            state = self._transformers_prompt_cache_state(cache_value)
            if state is None:
                raise ValueError(f"prompt cache key '{normalized}' does not exist")
            if not isinstance(state.cache, DynamicCache):
                raise ValueError("prompt cache key does not reference a supported transformers cache object")

            tensors: Dict[str, torch.Tensor] = {}
            prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens or ()))
            tensors["prompt_tokens"] = torch.tensor(prompt_tokens, dtype=torch.int32, device="cpu")

            layers = getattr(state.cache, "layers", []) or []
            for idx, layer in enumerate(layers):
                if not bool(getattr(layer, "is_initialized", False)):
                    continue
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    continue
                tensors[f"layer_{idx}_keys"] = keys.detach().to("cpu")
                tensors[f"layer_{idx}_values"] = values.detach().to("cpu")

            meta: Dict[str, str] = {
                "format": "abstractcore-transformers-prompt-cache/v1",
                "provider": str(getattr(self, "provider", "huggingface")),
                "model": str(getattr(self, "model", "")),
                "saved_at": datetime.now().isoformat(),
                "token_count": str(len(prompt_tokens)),
                "cache_implementation": "dynamic",
            }

            save_file(tensors, str(filename), metadata=meta)

            return {
                "supported": True,
                "operation": "save",
                "provider": str(getattr(self, "provider", "huggingface")),
                "model": str(getattr(self, "model", "")),
                "key": normalized,
                "filename": str(filename),
                "meta": meta,
            }

        try:
            import numpy as np
            from llama_cpp.llama import LlamaState
        except Exception as e:
            raise ImportError("GGUF prompt cache saving requires `llama-cpp-python` and `numpy`.") from e

        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            raise ValueError("prompt cache key must be a non-empty string")

        cache_value = self._prompt_cache_store.get(normalized)
        state = self._gguf_prompt_cache_state(cache_value)
        if state is None:
            raise ValueError(f"prompt cache key '{normalized}' does not exist")

        cache_obj = self._gguf_prompt_cache_unwrap(state)
        if cache_obj is None:
            raise ValueError("prompt cache key does not reference a llama.cpp cache object")

        prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens or self._gguf_prompt_cache_longest_prefix_tokens(cache_obj)))
        if not prompt_tokens:
            raise ValueError("prompt cache has no stored prefix tokens to save")

        # Ensure a concrete state exists for the stored prefix tokens.
        state_map = getattr(cache_obj, "cache_state", None)
        llama_state = None
        if hasattr(state_map, "get"):
            llama_state = state_map.get(prompt_tokens)
        if llama_state is None:
            with getattr(self, "_gguf_prompt_cache_lock", _MPS_GENERATION_LOCK):
                if not self._gguf_prefill_prompt_cache(cache_obj, prompt_tokens):
                    raise RuntimeError("failed to prefill prompt cache prior to saving")
            state_map = getattr(cache_obj, "cache_state", None)
            if hasattr(state_map, "get"):
                llama_state = state_map.get(prompt_tokens)

        if not isinstance(llama_state, LlamaState):
            raise RuntimeError("could not retrieve a llama.cpp state snapshot for the prompt cache key")

        exported_state = self._gguf_prompt_cache_export_state(state)
        meta: Dict[str, Any] = {
            "format": "abstractcore-gguf-prompt-cache/v1",
            "provider": str(getattr(self, "provider", "huggingface")),
            "model": str(getattr(self, "model", "")),
            "saved_at": datetime.now().isoformat(),
            "cache_state": exported_state,
        }

        np.savez_compressed(
            str(filename),
            meta_json=np.array(json.dumps(meta, ensure_ascii=False), dtype=np.str_),
            prompt_tokens=np.asarray(prompt_tokens, dtype=np.intc),
            input_ids=np.asarray(getattr(llama_state, "input_ids"), dtype=np.intc),
            scores=np.asarray(getattr(llama_state, "scores"), dtype=np.single),
            n_tokens=np.asarray(int(getattr(llama_state, "n_tokens", 0) or 0), dtype=np.int64),
            llama_state=np.frombuffer(bytes(getattr(llama_state, "llama_state", b"")), dtype=np.uint8),
            llama_state_size=np.asarray(int(getattr(llama_state, "llama_state_size", 0) or 0), dtype=np.int64),
            seed=np.asarray(int(getattr(llama_state, "seed", 0) or 0), dtype=np.int64),
        )

        return {
            "supported": True,
            "operation": "save",
            "provider": str(getattr(self, "provider", "huggingface")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": meta,
        }

    def prompt_cache_load(
        self,
        filename: str,
        *,
        key: Optional[str] = None,
        make_default: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load a GGUF llama.cpp prompt cache snapshot from disk (best-effort)."""
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        if getattr(self, "model_type", None) != "gguf":
            try:
                import torch  # type: ignore
                from safetensors import safe_open  # type: ignore
                from safetensors.torch import load_file  # type: ignore
                from transformers.cache_utils import DynamicCache, DynamicLayer  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Transformers prompt cache loading requires `torch`, `transformers`, and `safetensors`."
                ) from e

            device_str = self._transformers_cache_device_str()
            meta: Dict[str, Any] = {}
            try:
                with safe_open(str(filename), framework="pt", device="cpu") as f:
                    raw_meta = f.metadata() or {}
                    meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
            except Exception:
                meta = {}

            fmt = meta.get("format")
            if fmt and str(fmt) != "abstractcore-transformers-prompt-cache/v1":
                raise ValueError(f"Unsupported transformers prompt cache format: {fmt}")

            required_model = meta.get("model")
            current_model = str(getattr(self, "model", "") or "")
            if isinstance(required_model, str) and required_model.strip() and required_model.strip() != current_model:
                raise ValueError(
                    f"Prompt cache model mismatch: cache expects '{required_model.strip()}', current model is '{current_model}'."
                )

            tensors = load_file(str(filename), device=device_str)
            prompt_tok_tensor = tensors.get("prompt_tokens")
            if prompt_tok_tensor is None:
                raise ValueError("Invalid transformers prompt cache file (missing prompt_tokens)")
            prompt_tokens = tuple(int(tok) for tok in prompt_tok_tensor.to("cpu").tolist())

            layer_indices: List[int] = []
            for name in tensors.keys():
                if name.startswith("layer_") and name.endswith("_keys"):
                    try:
                        layer_indices.append(int(name.split("_", 2)[1]))
                    except Exception:
                        continue
            max_idx = max(layer_indices) if layer_indices else -1

            cache = DynamicCache()
            if max_idx >= 0:
                cache.layers = [DynamicLayer() for _ in range(max_idx + 1)]
                for idx in range(max_idx + 1):
                    keys = tensors.get(f"layer_{idx}_keys")
                    values = tensors.get(f"layer_{idx}_values")
                    if keys is None or values is None:
                        continue
                    layer = cache.layers[idx]
                    layer.keys = keys
                    layer.values = values
                    layer.is_initialized = True

            imported_state = _TransformersPromptCacheValue(cache=cache, prompt_tokens=prompt_tokens)

            normalized = self._normalize_prompt_cache_key(key) if key is not None else None
            if normalized is None:
                normalized = f"cache:{uuid.uuid4().hex[:12]}"

            store_meta: Dict[str, Any] = {
                "backend": "transformers",
                "loaded_from": str(filename),
            }
            store_meta.update(meta)
            try:
                store_meta.setdefault("token_count", len(prompt_tokens))
            except Exception:
                pass

            self._prompt_cache_store.set(normalized, imported_state, meta=store_meta)
            if make_default:
                self._default_prompt_cache_key = normalized

            return {
                "supported": True,
                "operation": "load",
                "provider": str(getattr(self, "provider", "huggingface")),
                "model": str(getattr(self, "model", "")),
                "key": normalized,
                "filename": str(filename),
                "meta": store_meta,
            }

        try:
            import numpy as np
            from llama_cpp.llama_cache import LlamaRAMCache
            from llama_cpp.llama import LlamaState
        except Exception as e:
            raise ImportError("GGUF prompt cache loading requires `llama-cpp-python` and `numpy`.") from e

        with np.load(str(filename), allow_pickle=False) as data:
            meta_json = data.get("meta_json")
            meta_raw = str(meta_json.tolist()) if meta_json is not None else ""
            try:
                meta: Dict[str, Any] = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = {}

            fmt = meta.get("format")
            if fmt and str(fmt) != "abstractcore-gguf-prompt-cache/v1":
                raise ValueError(f"Unsupported GGUF prompt cache format: {fmt}")

            required_model = meta.get("model") if isinstance(meta, dict) else None
            current_model = str(getattr(self, "model", "") or "")
            if isinstance(required_model, str) and required_model.strip() and required_model.strip() != current_model:
                raise ValueError(
                    f"Prompt cache model mismatch: cache expects '{required_model.strip()}', current model is '{current_model}'."
                )

            prompt_tokens_arr = data.get("prompt_tokens")
            if prompt_tokens_arr is None:
                raise ValueError("Invalid GGUF prompt cache file (missing prompt_tokens)")
            prompt_tokens = tuple(int(tok) for tok in prompt_tokens_arr.tolist())

            input_ids = data.get("input_ids")
            scores = data.get("scores")
            n_tokens = int(data.get("n_tokens").tolist()) if data.get("n_tokens") is not None else 0
            llama_state_u8 = data.get("llama_state")
            llama_state_size = int(data.get("llama_state_size").tolist()) if data.get("llama_state_size") is not None else 0
            seed = int(data.get("seed").tolist()) if data.get("seed") is not None else 0

        if not prompt_tokens or input_ids is None or scores is None or llama_state_u8 is None:
            raise ValueError("Invalid GGUF prompt cache file (missing required arrays)")

        llama_state = LlamaState(
            input_ids=np.asarray(input_ids, dtype=np.intc).copy(),
            scores=np.asarray(scores, dtype=np.single).copy(),
            n_tokens=int(n_tokens),
            llama_state=bytes(np.asarray(llama_state_u8, dtype=np.uint8).tobytes()),
            llama_state_size=int(llama_state_size),
            seed=int(seed),
        )

        cache_state = meta.get("cache_state") if isinstance(meta, dict) else None
        cap = None
        if isinstance(cache_state, dict):
            cap = cache_state.get("capacity_bytes")
        cap_i = self._coerce_gguf_prompt_cache_capacity_bytes(cap)
        try:
            state_size_i = int(getattr(llama_state, "llama_state_size", 0) or 0)
        except Exception:
            state_size_i = 0

        # If the saved state is larger than the declared capacity, fall back to the auto-growing
        # cache so we don't evict the single snapshot during load.
        if cap_i > 0 and state_size_i > 0 and state_size_i <= cap_i:
            cache_obj = LlamaRAMCache(capacity_bytes=int(cap_i))
        else:
            global _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            cache_cls = _AUTO_GROWING_LLAMA_RAM_CACHE_CLS
            if cache_cls is None:
                class _AutoGrowingLlamaRAMCache(LlamaRAMCache):  # type: ignore[misc]
                    def __setitem__(self, key, value):  # type: ignore[override]
                        try:
                            state_size = int(getattr(value, "llama_state_size", 0) or 0)
                        except Exception:
                            state_size = 0
                        if state_size > 0:
                            try:
                                cap_now = int(getattr(self, "capacity_bytes", 0) or 0)
                            except Exception:
                                cap_now = 0
                            if state_size > cap_now:
                                self.capacity_bytes = int(state_size)
                        return super().__setitem__(key, value)

                cache_cls = _AutoGrowingLlamaRAMCache
                _AUTO_GROWING_LLAMA_RAM_CACHE_CLS = cache_cls
            cache_obj = cache_cls(capacity_bytes=0)
        cache_obj[prompt_tokens] = llama_state

        imported_state = self._gguf_prompt_cache_import_state(cache_obj, cache_state if isinstance(cache_state, dict) else None)
        # Ensure the loaded state knows the saved prompt_tokens.
        imported_state.prompt_tokens = prompt_tokens

        normalized = self._normalize_prompt_cache_key(key) if key is not None else None
        if normalized is None:
            normalized = f"cache:{uuid.uuid4().hex[:12]}"

        store_meta: Dict[str, Any] = {"backend": "llama_cpp", "loaded_from": str(filename)}
        if isinstance(meta, dict):
            store_meta.update({k: v for k, v in meta.items() if k != "cache_state"})
        try:
            store_meta.setdefault("token_count", len(prompt_tokens))
        except Exception:
            pass

        self._prompt_cache_store.set(normalized, imported_state, meta=store_meta)
        if make_default:
            self._default_prompt_cache_key = normalized

        try:
            if getattr(self, "llm", None) is not None and hasattr(self.llm, "set_cache"):
                self.llm.set_cache(cache_obj)
        except Exception:
            pass

        return {
            "supported": True,
            "operation": "load",
            "provider": str(getattr(self, "provider", "huggingface")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": store_meta,
        }

    def _is_gguf_model(self, model: str) -> bool:
        """Detect if the model is a GGUF model"""
        # Check if it's a .gguf file path
        if model.endswith('.gguf'):
            return True

        # Check if local file exists with .gguf extension
        model_path = Path(model)
        if model_path.exists() and model_path.suffix == '.gguf':
            return True

        # Check if it's a HF repo with GGUF in the name (various formats)
        model_lower = model.lower()
        if 'gguf' in model_lower:
            # Handle formats like:
            # - "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
            # - "unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF" (cache format)
            # - "repo/model-GGUF"
            return True

        return False

    def _is_vision_model(self, model: str) -> bool:
        """Detect if the model is a vision model that requires special handling"""
        model_lower = model.lower()

        # Known vision models that require AutoModelForImageTextToText
        vision_models = [
            'glyph',           # zai-org/Glyph
            'glm-4.1v',        # GLM-4.1V variants
            'glm4v',           # GLM4V architecture
            'qwen-vl',         # Qwen-VL models
            'qwen2-vl',        # Qwen2-VL models
            'qwen2.5-vl',      # Qwen2.5-VL models
            'llava',           # LLaVA models
            'instructblip',    # InstructBLIP models
            'blip2',           # BLIP2 models
            'flamingo',        # Flamingo models
        ]

        return any(vision_keyword in model_lower for vision_keyword in vision_models)

    def _setup_device_transformers(self):
        """Setup device for transformers models (best-effort).

        We validate explicit device requests even when Transformers isn't available,
        since Torch availability (MPS/CUDA) may still matter for downstream behavior.
        """
        try:
            import torch  # type: ignore
        except Exception:
            self.device = "cpu"
            return

        requested = str(self.device or "").strip().lower() if isinstance(self.device, str) else ""
        if requested and requested != "auto":
            # Respect explicit user/env request, but fall back safely if unavailable.
            if requested == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and not torch.backends.mps.is_available():
                    self.logger.warning(
                        "HuggingFaceProvider requested device=mps but MPS is not available. "
                        "This usually means the process cannot see Metal devices (sandboxed execution). "
                        "Falling back to CPU. To silence this, set ABSTRACTCORE_HF_DEVICE=cpu."
                    )
                    self.device = "cpu"
                else:
                    self.device = "mps"
                    # Enable MPS fallback for unsupported ops (notably some vision pipelines).
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            elif requested == "cuda":
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.logger.warning(
                        "HuggingFaceProvider requested device=cuda but CUDA is not available; falling back to CPU."
                    )
                    self.device = "cpu"
            else:
                self.device = "cpu"
            return

        if not TRANSFORMERS_AVAILABLE:
            # Without transformers, default to CPU for safety.
            self.device = "cpu"
            return

        # Auto device selection.
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Apple Silicon: MPS built but unavailable is usually a sandbox / Metal visibility issue.
        try:
            import platform

            if (
                self.device == "cpu"
                and platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and not torch.backends.mps.is_available()
            ):
                self.logger.warning(
                    "PyTorch was built with MPS support, but MPS is not available. "
                    "This often indicates the process cannot access Metal devices (sandboxed execution). "
                    "Run outside the sandbox or force CPU via ABSTRACTCORE_HF_DEVICE=cpu."
                )
        except Exception:
            pass

    def _setup_device_gguf(self):
        """Setup device for GGUF models"""
        if self.n_gpu_layers is not None:
            return

        requested = str(self.device or "").strip().lower() if isinstance(self.device, str) else ""
        if requested == "cpu":
            self.n_gpu_layers = 0
            return

        is_metal_platform = platform.system().lower() == "darwin" and platform.machine().lower() == "arm64"
        wants_metal_request = requested == "mps" or (not requested and is_metal_platform)

        # Safety guard: on macOS, importing PyTorch/transformers in-process can hard-crash
        # llama.cpp when using Metal offload. Avoid SIGABRT by forcing CPU in that scenario,
        # unless the user explicitly opts into the unsafe path.
        llama_cpp_preimported_for_metal = False
        if "llama_cpp" in sys.modules:
            try:
                import llama_cpp  # type: ignore

                llama_cpp_preimported_for_metal = bool(
                    getattr(llama_cpp, "__abstractcore_preimported_for_metal", False)
                )
            except Exception:
                llama_cpp_preimported_for_metal = False

        if (
            wants_metal_request
            and os.environ.get("ABSTRACTCORE_GGUF_METAL_UNSAFE", "").strip().lower() not in {"1", "true", "yes"}
            and ("torch" in sys.modules or "transformers" in sys.modules)
            and not llama_cpp_preimported_for_metal
        ):
            import warnings

            warnings.warn(
                "GGUF Metal offload disabled because PyTorch/transformers is already imported in this process "
                "(llama-cpp-python Metal offload can SIGABRT). "
                "Start a fresh Python process (import GGUF first) to use Metal, "
                "or ensure `llama_cpp` is imported before PyTorch, "
                "or set ABSTRACTCORE_GGUF_METAL_UNSAFE=1 to force Metal offload.",
                RuntimeWarning,
                stacklevel=3,
            )
            self.n_gpu_layers = 0
            return

        # Prefer GPU offload when available. Use llama.cpp's own capability probe so we
        # don't need to import PyTorch.
        supports_gpu_offload = False
        try:
            import llama_cpp  # type: ignore

            probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
            supports_gpu_offload = bool(probe() if callable(probe) else probe)
        except Exception:
            supports_gpu_offload = False

        wants_metal = requested == "mps" or (not requested and is_metal_platform and supports_gpu_offload)
        wants_cuda = requested == "cuda" or (not requested and not is_metal_platform and supports_gpu_offload)

        self.n_gpu_layers = int(-1 if (wants_metal or wants_cuda) else 0)

    def _load_transformers_model(self):
        """Load standard HuggingFace transformers model"""
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModel,
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
        except Exception as e:
            raise ImportError("Transformers + PyTorch are required for HuggingFace (transformers) models.") from e

        try:
            # Check if this is a vision model that requires special handling
            if self._is_vision_model(self.model):
                return self._load_vision_model()

            # Load tokenizer with transformers-specific parameters
            tokenizer_kwargs = {k: v for k, v in self.transformers_kwargs.items() 
                              if k in ['trust_remote_code']}
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                tokenizer_kwargs['local_files_only'] = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, **tokenizer_kwargs)

            # Load model with all transformers-specific parameters
            # Try AutoModelForCausalLM first, fall back to AutoModel for custom models
            model_kwargs = self.transformers_kwargs.copy()
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                model_kwargs['local_files_only'] = True

            try:
                self.model_instance = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
            except ValueError as e:
                if "Unrecognized configuration class" in str(e) or "glm4v" in str(e).lower():
                    # Fall back to AutoModel for custom models like DeepSeek-OCR
                    self.model_instance = AutoModel.from_pretrained(self.model, **model_kwargs)
                else:
                    raise

            # Move to device (only if not using device_map)
            if self.device in ["cuda", "mps"] and 'device_map' not in self.transformers_kwargs:
                self.model_instance = self.model_instance.to(self.device)

            # Create pipeline - handle custom models that don't support text-generation
            device_arg = 0 if self.device == "cuda" else -1
            if self.device == "mps":
                device_arg = -1

            try:
                # Don't pass device argument if using device_map (accelerate)
                if 'device_map' in self.transformers_kwargs:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_instance,
                        tokenizer=self.tokenizer
                    )
                else:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_instance,
                        tokenizer=self.tokenizer,
                        device=device_arg
                    )
            except ValueError as e:
                if "not supported for text-generation" in str(e) or "accelerate" in str(e):
                    # For custom models like DeepSeek-OCR, skip pipeline creation
                    # We'll handle generation directly through the model
                    self.pipeline = None
                else:
                    raise

        except Exception as e:
            error_str = str(e).lower()
            if ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                available_models = self.list_available_models()
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model}: {str(e)}")

    def _load_vision_model(self):
        """Load vision model using AutoModelForImageTextToText and AutoProcessor"""
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

            # Suppress progress bars during model loading unless in debug mode
            import os
            from transformers.utils import logging as transformers_logging

            if not self.debug:
                # Disable transformers progress bars
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                transformers_logging.set_verbosity_error()
                # Disable tqdm progress bars
                os.environ['DISABLE_TQDM'] = '1'

            # Load processor for vision models (handles both text and images)
            processor_kwargs = {k: v for k, v in self.transformers_kwargs.items() 
                              if k in ['trust_remote_code']}
            # Enable trust_remote_code for custom architectures like GLM4V
            processor_kwargs['trust_remote_code'] = True
            # Set use_fast=True to avoid the slow processor warning
            processor_kwargs['use_fast'] = True
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                processor_kwargs['local_files_only'] = True

            # Use local cache path if offline mode is enabled and model is cached
            model_path = self.model
            if _config.should_force_local_files_only():
                local_path = _get_local_model_path(self.model)
                if local_path:
                    model_path = local_path
                    processor_kwargs.pop('local_files_only', None)  # Remove since we're using local path
                    self.logger.debug(f"Loading processor from local cache: {local_path}")

            self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

            # Load vision model using AutoModelForImageTextToText with trust_remote_code
            vision_kwargs = self.transformers_kwargs.copy()
            vision_kwargs['trust_remote_code'] = True
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                vision_kwargs['local_files_only'] = True

            # Safer defaults on GPU backends: float16 unless caller provided torch_dtype.
            try:
                if self.device in {"mps", "cuda"} and "torch_dtype" not in vision_kwargs:
                    import torch as _torch

                    vision_kwargs["torch_dtype"] = _torch.float16
            except Exception:
                pass

            # Use local cache path if offline mode is enabled and model is cached
            model_path = self.model
            if _config.should_force_local_files_only():
                local_path = _get_local_model_path(self.model)
                if local_path:
                    model_path = local_path
                    vision_kwargs.pop('local_files_only', None)  # Remove since we're using local path
                    self.logger.debug(f"Loading model from local cache: {local_path}")

            self.model_instance = AutoModelForImageTextToText.from_pretrained(model_path, **vision_kwargs)

            # Restore logging levels if they were suppressed
            if not self.debug:
                # Restore transformers logging
                transformers_logging.set_verbosity_warning()
                # Remove tqdm suppression
                if 'DISABLE_TQDM' in os.environ:
                    del os.environ['DISABLE_TQDM']

            # Move to device (only if not using device_map)
            if self.device in ["cuda", "mps"] and 'device_map' not in self.transformers_kwargs:
                self.model_instance = self.model_instance.to(self.device)

            try:
                self.model_instance.eval()
            except Exception:
                pass

            # For vision models, we don't use the standard pipeline
            self.pipeline = None

            self.logger.info(f"Successfully loaded vision model {self.model} using AutoModelForImageTextToText")

        except Exception as e:
            error_str = str(e).lower()

            # Check for transformers version issues
            if 'glm4v' in error_str and 'does not recognize this architecture' in error_str:
                import transformers
                current_version = transformers.__version__
                raise RuntimeError(
                    f"GLM4V architecture requires transformers>=4.57.1, but you have {current_version}. "
                    f"Please upgrade: pip install transformers>=4.57.1"
                )
            elif ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                available_models = self.list_available_models()
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace vision model {self.model}: {str(e)}")

    def _find_gguf_in_cache(self, model_name: str, *, _seen: Optional[set[str]] = None) -> Optional[str]:
        """Find GGUF model in local caches (HuggingFace hub / LM Studio; cache-only, no downloading)."""

        if _seen is None:
            _seen = set()
        key = str(model_name or "").strip()
        if not key or key in _seen:
            return None
        _seen.add(key)

        def _pick_preferred_gguf(gguf_files: list[Path]) -> Optional[str]:
            if not gguf_files:
                return None
            gguf_files = sorted(gguf_files, key=lambda p: p.name)
            explicit_selector = None
            selector_source = str(model_name or "").strip()
            if ":" in selector_source:
                explicit_selector = selector_source.split(":", 1)[1].strip().strip("/")
                explicit_selector = explicit_selector or None
            if explicit_selector:
                selector_upper = explicit_selector.upper()
                for gguf_file in gguf_files:
                    file_name_upper = gguf_file.name.upper()
                    file_path_upper = str(gguf_file).upper()
                    if selector_upper == file_name_upper or selector_upper in file_name_upper or selector_upper in file_path_upper:
                        return str(gguf_file)
            preferred_quants = ['Q4_K_M', 'Q5_K_M', 'Q4_0', 'Q4_1', 'Q5_0', 'Q8_0']
            for quant in preferred_quants:
                for gguf_file in gguf_files:
                    if quant in gguf_file.name.upper():
                        return str(gguf_file)
            return str(gguf_files[0])

        def _to_repo_id(raw: str) -> Optional[str]:
            s = str(raw or "").strip()
            if not s:
                return None
            if ":" in s:
                s = s.split(":", 1)[0].strip()
            if not s:
                return None
            if s.startswith("models--"):
                parts = s.replace("models--", "").split("--", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return f"{parts[0]}/{parts[1]}"
            if "--" in s and "/" not in s:
                parts = s.split("--", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    return f"{parts[0]}/{parts[1]}"
            if "/" in s:
                return s.strip().strip("/")
            return None

        # Normalize model name to cache format
        # Convert "unsloth/model" or "unsloth--model" to "models--unsloth--model"
        cache_name = self._normalize_to_cache_format(model_name)

        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_base / cache_name

        if not model_cache_dir.exists():
            model_cache_dir = None

        # Look for GGUF files in HuggingFace snapshots
        if model_cache_dir is not None:
            snapshots_dir = model_cache_dir / "snapshots"
            if snapshots_dir.exists():
                # Find the latest snapshot (most recent directory)
                try:
                    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        # Use the most recent snapshot
                        latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)

                        # Look for GGUF files in the snapshot
                        gguf_files = list(latest_snapshot.glob("*.gguf"))
                        picked = _pick_preferred_gguf(gguf_files)
                        if picked:
                            return picked

                except Exception:
                    pass

        # Fallback: LM Studio model cache (~/.lmstudio/models) often stores GGUF files directly.
        try:
            from ..utils.model_cache import default_lmstudio_model_dirs, resolve_lmstudio_model_dir

            repo_id = _to_repo_id(model_name)
            lm_dir = resolve_lmstudio_model_dir(repo_id, base_dirs=default_lmstudio_model_dirs()) if repo_id else None
            if lm_dir is not None and lm_dir.is_dir():
                gguf_files = list(lm_dir.glob("*.gguf"))
                picked = _pick_preferred_gguf(gguf_files)
                if picked:
                    return picked
        except Exception:
            pass

        # LM Studio Hub alias support: resolve org/model manifest to its GGUF dependency.
        try:
            from ..utils.model_cache import resolve_lmstudio_hub_manifest

            repo_id = _to_repo_id(model_name)
            if repo_id:
                manifest_path = resolve_lmstudio_hub_manifest(repo_id)
            else:
                manifest_path = None
            if manifest_path is not None:
                try:
                    raw = manifest_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    raw = ""
                manifest = json.loads(raw) if raw.strip() else {}
                deps = manifest.get("dependencies") if isinstance(manifest, dict) else None
                if isinstance(deps, list):
                    candidates: list[str] = []
                    for dep in deps:
                        if not isinstance(dep, dict):
                            continue
                        for src in dep.get("sources") or []:
                            if not isinstance(src, dict):
                                continue
                            if str(src.get("type") or "").strip().lower() != "huggingface":
                                continue
                            user = str(src.get("user") or "").strip()
                            repo = str(src.get("repo") or "").strip()
                            if user and repo:
                                candidates.append(f"{user}/{repo}")
                        for mk in dep.get("modelKeys") or []:
                            if isinstance(mk, str) and mk.strip():
                                candidates.append(mk.strip())

                    seen: set[str] = set()
                    ordered: list[str] = []
                    for c in candidates:
                        c2 = str(c or "").strip().strip("/")
                        if not c2 or c2 in seen:
                            continue
                        seen.add(c2)
                        ordered.append(c2)

                    for cand in ordered:
                        resolved = self._find_gguf_in_cache(cand, _seen=_seen)
                        if resolved:
                            return resolved
        except Exception:
            pass

        return None

    def _normalize_to_cache_format(self, model_name: str) -> str:
        """Convert model name to HuggingFace cache directory format"""
        # Remove any ":filename" suffix
        if ':' in model_name:
            model_name = model_name.split(':', 1)[0]

        # Handle different input formats:
        if model_name.startswith('models--'):
            # Already in cache format
            return model_name
        elif '/' in model_name:
            # Standard format: "unsloth/model" -> "models--unsloth--model"
            return f"models--{model_name.replace('/', '--')}"
        elif '--' in model_name and not model_name.startswith('models--'):
            # Cache format without prefix: "unsloth--model" -> "models--unsloth--model"
            return f"models--{model_name}"
        else:
            # Single name, assume it's just the model part
            return f"models--{model_name}"

    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python (cache-only, no downloading)"""
        import os
        try:
            llama_cls = globals().get("Llama")
            if llama_cls is None:
                from llama_cpp import Llama as llama_cls  # type: ignore

            # llama-cpp-python 0.3.x can throw an "Exception ignored in: __del__" when model
            # initialization fails early (missing `sampler` attribute). Patch defensively to
            # keep failures clean and actionable.
            try:  # pragma: no cover
                import llama_cpp._internals as _llama_internals  # type: ignore

                if hasattr(_llama_internals, "LlamaModel") and not hasattr(_llama_internals.LlamaModel, "sampler"):
                    setattr(_llama_internals.LlamaModel, "sampler", None)
            except Exception:
                pass

            model_path = None

            # First, try as a direct file path
            if Path(self.model).exists() and self.model.endswith('.gguf'):
                model_path = self.model
            else:
                # Try to find in HuggingFace cache
                model_path = self._find_gguf_in_cache(self.model)

            if not model_path:
                # Model not found in cache - provide graceful fallback
                self._handle_gguf_not_found()
                return

            # Verify file exists and is accessible
            if not Path(model_path).exists():
                raise FileNotFoundError(f"GGUF model file not found: {model_path}")

            gguf_arch: str | None = None
            try:
                from ..utils.model_cache import read_gguf_architecture

                gguf_arch = read_gguf_architecture(Path(model_path))
            except Exception:
                gguf_arch = None

            model_lower = self.model.lower()

            if "mtp" in model_lower:
                self.logger.warning(
                    "Loading an MTP GGUF through llama-cpp-python. The model can be used as a regular GGUF, "
                    "but current public llama-cpp-python bindings do not expose native MTP acceleration in-process. "
                    "Use an external llama.cpp server/runtime with native MTP support if you need the speedup."
                )

            # Determine chat format for function calling
            chat_format = None
            if 'qwen' in model_lower or 'coder' in model_lower:
                # Qwen models often support function calling
                chat_format = "chatml-function-calling"
            elif 'functionary' in model_lower:
                chat_format = "functionary-v2"

            # IMPORTANT (macOS): when loading GGUFs from LM Studio's cache, llama-cpp-python can
            # segfault with `use_mmap=True` (even on supported architectures). Disable mmap for
            # LM Studio paths to keep loads stable.
            use_mmap = True
            try:
                import platform

                if platform.system().lower() == "darwin":
                    from ..utils.model_cache import default_lmstudio_model_dirs

                    model_real = Path(model_path).resolve()
                    for base in default_lmstudio_model_dirs():
                        try:
                            base_real = base.resolve()
                        except Exception:
                            base_real = base
                        try:
                            if model_real.is_relative_to(base_real):
                                use_mmap = False
                                break
                        except Exception:
                            # Python <3.9 fallback (or odd path types).
                            if str(model_real).startswith(str(base_real) + os.sep):
                                use_mmap = False
                                break
            except Exception:
                pass

            # Initialize llama-cpp-python with stderr redirected to our logger.
            #
            # `self.max_tokens` is AbstractCore's unified "context window budget" and defaults
            # to the model's `max_tokens` from `assets/model_capabilities.json`.
            #
            # For GGUF/llama.cpp we must allocate a concrete KV cache (`n_ctx`). When callers do
            # not pass `max_tokens=...` explicitly and the advertised context is too large for
            # the local machine, we retry with smaller windows (best-effort) instead of using a
            # hidden env var.
            requested_n_ctx = self.max_tokens if self.max_tokens is not None else 16384
            try:
                requested_n_ctx_i = int(requested_n_ctx)
            except Exception:
                requested_n_ctx_i = 16384
            if requested_n_ctx_i <= 0:
                requested_n_ctx_i = 16384

            if getattr(self, "_user_provided_max_tokens", False):
                candidate_ctxs = [requested_n_ctx_i]
            else:
                candidate_ctxs = [requested_n_ctx_i]
                for fallback in (131072, 65536, 32768, 16384, 8192, 4096):
                    if fallback < requested_n_ctx_i:
                        candidate_ctxs.append(int(fallback))

            last_error: Exception | None = None
            chosen_n_ctx: int | None = None
            for n_ctx_i in candidate_ctxs:
                llama_kwargs = {
                    "model_path": model_path,
                    "n_ctx": int(n_ctx_i),
                    "n_gpu_layers": self.n_gpu_layers,
                    "chat_format": chat_format,
                    "verbose": self.debug,  # Use debug flag for verbose output
                    "n_threads": os.cpu_count() // 2 if os.cpu_count() else 4,
                    # Additional performance settings
                    "n_batch": 512,
                    "use_mmap": use_mmap,
                    "use_mlock": False,
                }

                try:
                    self.llm = llama_cls(**llama_kwargs)
                    chosen_n_ctx = int(n_ctx_i)
                    break
                except Exception as e:
                    # Common on macOS: Metal backend unavailable for the current process. Retry on CPU.
                    if isinstance(self.n_gpu_layers, int) and self.n_gpu_layers != 0:
                        try:
                            self.logger.warning(
                                f"GGUF load failed with n_gpu_layers={self.n_gpu_layers}; retrying with CPU (n_gpu_layers=0). Error: {e}"
                            )
                            llama_kwargs_cpu = llama_kwargs.copy()
                            llama_kwargs_cpu["n_gpu_layers"] = 0
                            # Avoid any GPU KV offload in the retry.
                            llama_kwargs_cpu["offload_kqv"] = False
                            self.llm = llama_cls(**llama_kwargs_cpu)
                            self.n_gpu_layers = 0
                            chosen_n_ctx = int(n_ctx_i)
                            break
                        except Exception as e_cpu:
                            e = e_cpu

                    last_error = e

                    # If caller explicitly requested a context window, fail fast with a clear message.
                    if getattr(self, "_user_provided_max_tokens", False):
                        raise RuntimeError(
                            f"Failed to load GGUF model {self.model} with n_ctx={n_ctx_i}. "
                            "Try lowering max_tokens=... when constructing HuggingFaceProvider(). "
                            f"Underlying error: {e}"
                        ) from e

                    # Best-effort retry with smaller context windows for local stability.
                    is_last = (n_ctx_i == candidate_ctxs[-1])
                    if not is_last:
                        self.logger.warning(
                            f"GGUF load failed for {self.model} with n_ctx={n_ctx_i}; retrying with a smaller context window. Error: {e}"
                        )
                        continue

                    # No more fallbacks available.
                    if gguf_arch == "qwen35moe":
                        raise RuntimeError(
                            "This GGUF uses architecture 'qwen35moe', which is not supported by the "
                            "llama.cpp version bundled with your installed llama-cpp-python. "
                            "LM Studio's Metal llama.cpp backend can load it; use `--provider lmstudio` "
                            "for this model, or upgrade to a llama-cpp-python build that includes "
                            "qwen35moe support."
                        ) from e
                    if gguf_arch:
                        raise RuntimeError(
                            f"Failed to load GGUF model (architecture: {gguf_arch}): {e}"
                        ) from e
                    raise

            if self.llm is None or chosen_n_ctx is None:
                raise RuntimeError(f"Failed to load GGUF model {self.model}: {last_error}")

            # Keep AbstractCore's token budget in sync with the actual llama.cpp context window.
            #
            # `model_capabilities.json` stores advertised limits, but GGUF loads require a
            # concrete `n_ctx` allocation. After successful load (including fallbacks),
            # treat `self.max_tokens` as the runtime context window budget.
            self.max_tokens = int(chosen_n_ctx)

            # Ensure output reservation never exceeds the runtime context window.
            #
            # Many models (e.g. Qwen3.5) advertise very large output limits, but GGUF runtime
            # context windows are often smaller locally. Keep invariants consistent to avoid
            # negative/invalid derived `max_input_tokens` and provider-side errors.
            try:
                if (
                    isinstance(self.max_output_tokens, int)
                    and int(self.max_output_tokens) > int(self.max_tokens)
                ):
                    self.logger.warning(
                        (
                            f"Clamping max_output_tokens for {self.model}: configured "
                            f"max_output_tokens={self.max_output_tokens} exceeds GGUF n_ctx={self.max_tokens}."
                        )
                    )
                    self.max_output_tokens = int(self.max_tokens)
            except Exception:
                pass

        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model {self.model}: {str(e)}")

    def _handle_gguf_not_found(self):
        """Handle GGUF model not found with graceful fallback like other providers"""
        # Suggest the correct repo format
        suggested_repo = self._suggest_correct_repo_format(self.model)

        # List any similar models in cache
        similar_models = self._find_similar_gguf_models()

        error_parts = [
            f"❌ GGUF model '{self.model}' not found in local caches (HuggingFace hub / LM Studio).",
            "",
            "💡 To download this model, run:",
            f"   huggingface-cli download {suggested_repo}",
            "",
            "🔍 Suggested formats:",
            f"   • Correct: '{suggested_repo}'",
            f"   • Your input: '{self.model}'",
        ]

        if similar_models:
            error_parts.extend([
                "",
                "📂 Similar GGUF models found in cache:",
            ])
            for model in similar_models[:5]:  # Show max 5
                error_parts.append(f"   • {model}")

        error_parts.extend([
            "",
            "📖 For more info: https://huggingface.co/docs/hub/en/gguf",
            "🔧 AbstractCore only uses cached models - we never download automatically."
        ])

        error_message = "\n".join(error_parts)
        raise ModelNotFoundError(error_message)

    def _suggest_correct_repo_format(self, model_name: str) -> str:
        """Suggest the correct repository format"""
        # Handle various input formats and suggest the standard format
        if model_name.startswith('models--'):
            # "models--unsloth--model" -> "unsloth/model"
            parts = model_name.replace('models--', '').split('--', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"

        elif '--' in model_name and not '/' in model_name:
            # "unsloth--model" -> "unsloth/model"
            parts = model_name.split('--', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"

        # Return as-is if already in correct format or unknown format
        return model_name

    def _find_similar_gguf_models(self) -> List[str]:
        """Find similar GGUF models in cache"""
        similar: set[str] = set()

        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_base.exists():
            try:
                for cache_dir in cache_base.iterdir():
                    if cache_dir.is_dir() and 'gguf' in cache_dir.name.lower():
                        if cache_dir.name.startswith('models--'):
                            repo_name = cache_dir.name.replace('models--', '').replace('--', '/', 1)
                            similar.add(repo_name)
            except Exception:
                pass

        # Also include GGUF models stored in LM Studio's model cache.
        try:
            from ..utils.model_cache import default_lmstudio_model_dirs

            for base in default_lmstudio_model_dirs():
                try:
                    for org_dir in base.iterdir():
                        if not org_dir.is_dir():
                            continue
                        for model_dir in org_dir.iterdir():
                            if not model_dir.is_dir():
                                continue
                            try:
                                if any(p.suffix.lower() == ".gguf" for p in model_dir.iterdir()):
                                    similar.add(f"{org_dir.name}/{model_dir.name}")
                            except Exception:
                                continue
                except Exception:
                    continue
        except Exception:
            pass

        return sorted(similar)

    def _handle_timeout_parameter(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle timeout parameter for HuggingFace provider.

        Since HuggingFace models run locally (both transformers and GGUF),
        timeout parameters don't apply. If a non-None timeout is provided,
        issue a warning and treat it as None (infinity).

        Args:
            kwargs: Initialization kwargs that may contain timeout
        """
        timeout_value = kwargs.get('timeout')
        if timeout_value is not None:
            import warnings
            warnings.warn(
                f"HuggingFace provider runs models locally and does not support timeout parameters. "
                f"Provided timeout={timeout_value} will be ignored and treated as None (unlimited).",
                UserWarning,
                stacklevel=3
            )
            # Force timeout to None for local models
            self._timeout = None
        else:
            # Keep None value (unlimited timeout is appropriate for local models)
            self._timeout = None

    def _update_http_client_timeout(self) -> None:
        """
        HuggingFace provider doesn't use HTTP clients for model inference.
        Local models (transformers and GGUF) don't have timeout constraints.
        """
        # No-op for local models - they don't use HTTP clients
        pass

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
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using appropriate backend"""

        if self.model_type == "gguf":
            return self._generate_gguf(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
        else:
            return self._generate_transformers(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)

    def _generate_transformers(self,
                               prompt: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               system_prompt: Optional[str] = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               media: Optional[List['MediaContent']] = None,
                               stream: bool = False,
                               response_model: Optional[Type[BaseModel]] = None,
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using transformers backend with optional Outlines native structured output"""

        if not self.pipeline:
            # Handle vision models that use processor instead of pipeline
            if self.processor and hasattr(self.model_instance, 'generate'):
                return self._generate_vision_model(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
            # Handle custom models like DeepSeek-OCR that don't support standard pipelines
            elif hasattr(self.model_instance, 'infer'):
                return self._generate_custom_model(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
            else:
                return GenerateResponse(
                    content="Error: Transformers model not loaded or doesn't support generation",
                    model=self.model,
                    finish_reason="error"
                )

        # Native structured output via Outlines (if configured and available)
        should_use_outlines = (
            response_model and
            PYDANTIC_AVAILABLE and
            not stream and
            self.structured_output_method != "prompted"  # Skip if explicitly prompted
        )

        if should_use_outlines:
            # Check if Outlines is required but unavailable
            if self.structured_output_method == "native_outlines" and not OUTLINES_AVAILABLE:
                return GenerateResponse(
                    content="Error: structured_output_method='native_outlines' requires Outlines library. Install with: pip install \"abstractcore[huggingface]\"",
                    model=self.model,
                    finish_reason="error"
                )

            # Try Outlines if available (auto or native_outlines mode)
            if OUTLINES_AVAILABLE:
                try:
                    import outlines  # type: ignore

                    # Cache Outlines model wrapper to avoid re-initialization
                    if not hasattr(self, '_outlines_model') or self._outlines_model is None:
                        self.logger.debug("Creating Outlines model wrapper for native structured output")
                        self._outlines_model = outlines.from_transformers(
                            self.model_instance,
                            self.tokenizer
                        )

                    # Build input text (same as normal generation)
                    input_text = self._build_input_text_transformers(prompt, messages, system_prompt, tools)

                    generation_kwargs = self._prepare_generation_kwargs(**kwargs)
                    max_new_tokens = self._get_provider_max_tokens_param(generation_kwargs)

                    # Create constrained generator with JSON schema
                    self.logger.debug(f"Using Outlines native structured output for {response_model.__name__}")
                    generator = self._outlines_model(
                        input_text,
                        outlines.json_schema(response_model),
                        max_tokens=max_new_tokens,
                    )

                    # Validate and return
                    validated_obj = response_model.model_validate(generator)

                    return GenerateResponse(
                        content=validated_obj.model_dump_json(),
                        model=self.model,
                        finish_reason="stop",
                        validated_object=validated_obj
                    )
                except Exception as e:
                    # If native_outlines was explicitly requested, don't fall back
                    if self.structured_output_method == "native_outlines":
                        return GenerateResponse(
                            content=f"Error: Outlines native structured output failed: {str(e)}",
                            model=self.model,
                            finish_reason="error"
                        )
                    # Otherwise fall back to prompted approach
                    self.logger.debug(f"Outlines generation failed, falling back to prompted: {e}")
                    # Continue with normal generation below

        # Build input text with tool and media support
        # Handle media content first if present
        media_enrichment = None
        if media:
            try:
                from ..media.handlers import LocalMediaHandler
                media_handler = LocalMediaHandler("huggingface", self.model_capabilities, model_name=self.model)

                # Create multimodal message combining text and media
                multimodal_message = media_handler.create_multimodal_message(prompt, media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                # For local providers, we get text-embedded content
                if isinstance(multimodal_message, str):
                    prompt = multimodal_message
                else:
                    # If we get a structured message, extract the content
                    if isinstance(multimodal_message, dict) and "content" in multimodal_message:
                        if isinstance(multimodal_message["content"], list):
                            # Find text content in the structured message
                            text_content = ""
                            for item in multimodal_message["content"]:
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                    break
                            prompt = text_content or prompt
                        else:
                            prompt = str(multimodal_message["content"])
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        # Generation parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_new_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        temperature = generation_kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", 0.9)
        seed_value = generation_kwargs.get("seed")

        prompt_cache_key = kwargs.get("prompt_cache_key")
        prefilled_modules = kwargs.get("prompt_cache_prefilled_modules")
        if (
            isinstance(prompt_cache_key, str)
            and prompt_cache_key.strip()
            and self._transformers_prompt_cache_supported()
        ):
            try:
                cached = self._single_generate_transformers_cached(
                    prompt=str(prompt or ""),
                    prompt_cache_key=prompt_cache_key.strip(),
                    messages=messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    prefilled_modules=prefilled_modules,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature) if temperature is not None else None,
                    top_p=float(top_p) if top_p is not None else 0.9,
                    seed=seed_value,
                )
            except Exception as e:
                return GenerateResponse(
                    content=f"Error generating response with prompt cache: {str(e)}",
                    model=self.model,
                    finish_reason="error",
                )

            if stream:
                def _stream_cached() -> Iterator[GenerateResponse]:
                    # Simulated streaming: yield word chunks, then run tool execution if requested.
                    content = cached.content or ""
                    tool_call_tags = kwargs.get("tool_call_tags")
                    if tool_call_tags and content:
                        try:
                            from ..tools.tag_rewriter import create_tag_rewriter
                            rewriter = create_tag_rewriter(tool_call_tags)
                            content = rewriter.rewrite_text(content)
                        except Exception:
                            pass

                    words = content.split()
                    collected = ""
                    if not words:
                        yield GenerateResponse(content="", model=self.model, finish_reason="stop")
                        return
                    for i, word in enumerate(words):
                        chunk_content = word + (" " if i < len(words) - 1 else "")
                        collected += chunk_content
                        yield GenerateResponse(
                            content=chunk_content,
                            model=self.model,
                            finish_reason="stop" if i == len(words) - 1 else None,
                        )

                    # Tool execution (prompted) happens after streaming in this provider.
                    if tools and getattr(self.tool_handler, "supports_prompted", False) and collected:
                        complete = GenerateResponse(
                            content=collected,
                            model=self.model,
                            finish_reason="stop",
                        )
                        final = self._handle_prompted_tool_execution(complete, tools)
                        if final.content and final.content != collected:
                            suffix = final.content[len(collected):]
                            if suffix:
                                yield GenerateResponse(
                                    content=suffix,
                                    model=self.model,
                                    finish_reason="stop",
                                )

                return _stream_cached()

            response = cached
            if media_enrichment:
                from ..media.enrichment import merge_enrichment_metadata

                response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

            # Handle tool execution for prompted models
            if tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools)

            return response

        input_text = self._build_input_text_transformers(prompt, messages, system_prompt, tools)

        try:
            if stream:
                return self._stream_generate_transformers_with_tools(input_text, max_new_tokens, temperature, top_p, tools, kwargs.get('tool_call_tags'), seed_value)
            else:
                response = self._single_generate_transformers(input_text, max_new_tokens, temperature, top_p, seed_value)
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    response = self._handle_prompted_tool_execution(response, tools)

                return response

        except Exception as e:
            return GenerateResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _generate_custom_model(self,
                              prompt: str,
                              messages: Optional[List[Dict[str, str]]] = None,
                              system_prompt: Optional[str] = None,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              media: Optional[List['MediaContent']] = None,
                              stream: bool = False,
                              response_model: Optional[Type[BaseModel]] = None,
                              **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using custom model methods (e.g., DeepSeek-OCR's infer method)"""

        import time
        import tempfile
        import os
        start_time = time.time()

        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore[assignment]

        try:
            # Handle media content for vision models like DeepSeek-OCR
            if media and len(media) > 0:
                # Use the first image for OCR
                media_item = media[0]

                # DeepSeek-OCR expects image file path
                if hasattr(media_item, 'file_path') and media_item.file_path:
                    image_file = str(media_item.file_path)
                else:
                    # If no file path, save media content to temp file
                    from PIL import Image

                    if hasattr(media_item, 'content') and media_item.content:
                        # Handle base64 content
                        if media_item.content_format == 'BASE64':
                            import base64
                            image_data = base64.b64decode(media_item.content)
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_file.write(image_data)
                            temp_file.close()
                            image_file = temp_file.name
                        else:
                            return GenerateResponse(
                                content="Error: Unsupported media format for DeepSeek-OCR",
                                model=self.model,
                                finish_reason="error"
                            )
                    else:
                        return GenerateResponse(
                            content="Error: No valid image content found",
                            model=self.model,
                            finish_reason="error"
                        )

                # Use DeepSeek-OCR's infer method
                try:
                    # Create temporary output directory for DeepSeek-OCR
                    temp_output_dir = tempfile.mkdtemp()

                    # Patch DeepSeek-OCR for MPS/CPU compatibility if needed
                    if torch is not None and (
                        self.device == "mps"
                        or (self.device is None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                    ):
                        self._patch_deepseek_for_mps()

                    result = self.model_instance.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path=temp_output_dir,  # DeepSeek-OCR requires output path
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False
                    )

                    # Clean up temp output directory
                    import shutil
                    shutil.rmtree(temp_output_dir, ignore_errors=True)

                    # Clean up temp file if created
                    if 'temp_file' in locals() and os.path.exists(image_file):
                        os.unlink(image_file)

                    # Calculate generation time
                    gen_time = (time.time() - start_time) * 1000

                    return GenerateResponse(
                        content=result if isinstance(result, str) else str(result),
                        model=self.model,
                        finish_reason="stop",
                        input_tokens=len(prompt.split()),  # Rough estimate
                        output_tokens=len(str(result).split()) if result else 0,
                        gen_time=gen_time
                    )

                except Exception as e:
                    return GenerateResponse(
                        content=f"Error during DeepSeek-OCR inference: {str(e)}",
                        model=self.model,
                        finish_reason="error"
                    )
            else:
                return GenerateResponse(
                    content="Error: DeepSeek-OCR requires image input",
                    model=self.model,
                    finish_reason="error"
                )

        except Exception as e:
            return GenerateResponse(
                content=f"Error in custom model generation: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _generate_vision_model(self,
                              prompt: str,
                              messages: Optional[List[Dict[str, str]]] = None,
                              system_prompt: Optional[str] = None,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              media: Optional[List['MediaContent']] = None,
                              stream: bool = False,
                              response_model: Optional[Type[BaseModel]] = None,
                              **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using vision model (Glyph, GLM-4.1V, etc.)"""

        import time
        start_time = time.time()

        # Import torch safely
        try:
            import torch
        except ImportError:
            return GenerateResponse(
                content="Error: PyTorch not available for vision model generation",
                model=self.model,
                finish_reason="error",
                gen_time=0.0
            )

        try:
            # Server/gateway sometimes call providers with prompt="" + messages=[...] + media=[...].
            # For multimodal models, the user text and the media must live in the SAME user turn.
            # Best-effort: if prompt is empty, lift the last user message text into the prompt and
            # remove that message from the history to avoid duplication.
            prompt_text = prompt
            messages_for_context = list(messages) if isinstance(messages, list) else None
            if (not isinstance(prompt_text, str) or not prompt_text.strip()) and media and messages_for_context:
                for i in range(len(messages_for_context) - 1, -1, -1):
                    msg = messages_for_context[i] or {}
                    role = str(msg.get("role", "") or "").strip().lower()
                    if role != "user":
                        continue
                    content = msg.get("content", "")
                    lifted = None
                    if isinstance(content, str) and content.strip():
                        lifted = content.strip()
                    elif isinstance(content, list):
                        # OpenAI-style list content: [{"type":"text","text":"..."}, ...]
                        for item in content:
                            if not isinstance(item, dict):
                                continue
                            if str(item.get("type", "") or "").strip().lower() == "text":
                                text_val = item.get("text")
                                if isinstance(text_val, str) and text_val.strip():
                                    lifted = text_val.strip()
                                    break
                    if lifted:
                        prompt_text = lifted
                        del messages_for_context[i]
                    break

            # Build messages for vision model
            chat_messages = []

            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})

            if messages_for_context:
                chat_messages.extend(messages_for_context)

            # Build user message with media content
            user_content = []

            # Add text content
            if isinstance(prompt_text, str) and prompt_text.strip():
                user_content.append({"type": "text", "text": prompt_text.strip()})

            # Add media content (images, video)
            has_video = False
            try:
                from ..media.types import MediaType, ContentFormat
            except Exception:
                MediaType = None  # type: ignore[assignment]
                ContentFormat = None  # type: ignore[assignment]

            if media:
                for media_item in media:
                    media_type = getattr(media_item, "media_type", None)

                    # Text markers (e.g. provenance / policy annotations) should be preserved for the model.
                    if MediaType is not None and media_type == MediaType.TEXT:
                        txt = getattr(media_item, "content", None)
                        if isinstance(txt, str) and txt.strip():
                            user_content.append({"type": "text", "text": txt.strip()})
                        continue

                    # Video inputs
                    if MediaType is not None and media_type == MediaType.VIDEO:
                        has_video = True
                        # The actual video content is provided to the processor via `videos=...`;
                        # the chat template only needs a `<video>` placeholder token.
                        user_content.append({"type": "video"})
                        continue

                    # Image inputs
                    if MediaType is None or media_type == MediaType.IMAGE:
                        if getattr(media_item, "file_path", None):
                            user_content.append({"type": "image", "url": str(media_item.file_path)})
                            continue

                        content = getattr(media_item, "content", None)
                        if not content:
                            continue

                        content_format = getattr(media_item, "content_format", None)
                        is_base64 = False
                        if ContentFormat is not None and content_format == ContentFormat.BASE64:
                            is_base64 = True
                        elif isinstance(content_format, str) and content_format.strip().lower() == "base64":
                            is_base64 = True

                        if is_base64:
                            mime_type = getattr(media_item, "mime_type", "image/png")
                            data_url = f"data:{mime_type};base64,{content}"
                            user_content.append({"type": "image", "url": data_url})

            # Add user message
            chat_messages.append({
                "role": "user",
                "content": user_content
            })

            # Process messages using the processor.
            #
            # Some multimodal processors (e.g. LlavaNextVideoProcessor) return a *string*
            # from apply_chat_template; for those we must call the processor separately
            # with explicit images/videos tensors and keep video frame counts bounded.
            if has_video:
                # Resolve max frames for video sampling (keep small to avoid huge context).
                max_frames_raw = kwargs.get("video_max_frames", None)
                if max_frames_raw is None:
                    try:
                        from ..config.manager import get_config_manager

                        cfg_video = getattr(get_config_manager().config, "video", None)
                        max_frames_raw = getattr(cfg_video, "max_frames_native", None) if cfg_video is not None else None
                        if max_frames_raw is None:
                            max_frames_raw = getattr(cfg_video, "max_frames", None) if cfg_video is not None else None
                    except Exception:
                        max_frames_raw = 3
                try:
                    max_video_frames = max(1, int(max_frames_raw))
                except Exception:
                    max_video_frames = 3

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
                    try:
                        from ..config.manager import get_config_manager

                        max_frame_side_raw = getattr(get_config_manager().config, "video", None).max_frame_side  # type: ignore[union-attr]
                    except Exception:
                        max_frame_side_raw = 1024
                try:
                    max_frame_side = int(max_frame_side_raw) if max_frame_side_raw is not None else None
                except Exception:
                    max_frame_side = 1024
                if isinstance(max_frame_side, int) and max_frame_side <= 0:
                    max_frame_side = None

                # Build multimodal-typed messages for chat_template renderers that expect list content.
                # NOTE: Many HF native-video VLMs are brittle in multi-turn mode if prior turns
                # referenced media but we only retained text history (no `<video>` placeholders).
                # This can cause follow-ups like "and this one?" to over-weight the previous
                # text-only answer and ignore the newly attached video.
                #
                # To make follow-ups robust, collapse prior USER/ASSISTANT turns into a single
                # text block inside the current user message, and keep exactly one `<video>`
                # placeholder (the current attachment) in the chat template input.
                history_lines = []
                if messages_for_context:
                    for msg in messages_for_context:
                        role = str(msg.get("role", "user") or "").strip().lower()
                        if role not in {"user", "assistant"}:
                            continue
                        content = msg.get("content", "")
                        text = ""
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            # OpenAI-style list content: [{"type":"text","text":"..."}, ...]
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                if str(item.get("type", "") or "").strip().lower() != "text":
                                    continue
                                v = item.get("text")
                                if isinstance(v, str) and v.strip():
                                    text = v
                                    break
                        else:
                            text = str(content)

                        text = str(text or "").strip()
                        if not text:
                            continue
                        prefix = "USER" if role == "user" else "ASSISTANT"
                        history_lines.append(f"{prefix}: {text}")

                if history_lines:
                    history_block = "Prior chat context (text-only):\n" + "\n".join(history_lines) + "\n\n"
                    # Cap to avoid pathological prompt growth; keep the most recent tail.
                    if len(history_block) > 8_000:
                        history_block = "Prior chat context (text-only; truncated):\n…\n" + history_block[-7_800:]
                    user_content = [{"type": "text", "text": history_block}] + list(user_content)

                mm_messages = []
                if system_prompt:
                    mm_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
                mm_messages.append({"role": "user", "content": user_content})

                prompt_text = self.processor.apply_chat_template(mm_messages, add_generation_prompt=True)

                # Prepare explicit video inputs for the processor.
                #
                # Prefer ffmpeg-sampled frames (our own extraction) over relying on torchvision/torchcodec
                # decoding inside Transformers, which can vary by platform/codec support (notably for .mov).
                video_paths = []
                image_inputs = []
                for media_item in (media or []):
                    if MediaType is not None and getattr(media_item, "media_type", None) == MediaType.VIDEO:
                        video_path = getattr(media_item, "file_path", None) or getattr(media_item, "content", None)
                        if not isinstance(video_path, str) or not video_path.strip():
                            raise ValueError("Video MediaContent must provide file_path for HuggingFace video models.")
                        video_paths.append(video_path)
                    elif MediaType is not None and getattr(media_item, "media_type", None) == MediaType.IMAGE:
                        fp = getattr(media_item, "file_path", None)
                        if isinstance(fp, str) and fp.strip():
                            try:
                                from PIL import Image as PILImage
                            except ImportError as e:
                                raise RuntimeError(f"PIL is required for HuggingFace image inputs: {e}")
                            image_inputs.append(PILImage.open(fp).convert("RGB"))

                processor_call: Dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
                if image_inputs:
                    processor_call["images"] = image_inputs if len(image_inputs) > 1 else image_inputs[0]
                if video_paths:
                    # Try ffmpeg frame sampling first.
                    video_frame_inputs = []
                    temp_dirs = []
                    try:
                        from pathlib import Path
                        import tempfile

                        from ..media.utils.video_frames import extract_video_frames
                        from PIL import Image as PILImage

                        for vp in video_paths:
                            out_dir = Path(tempfile.mkdtemp(prefix="abstractcore_hf_video_frames_"))
                            temp_dirs.append(out_dir)
                            frames, _timestamps_s = extract_video_frames(
                                Path(vp),
                                max_frames=max_video_frames,
                                frame_format="jpg",
                                sampling_strategy=sampling_strategy,
                                max_side=max_frame_side,
                                output_dir=out_dir,
                            )
                            if not frames:
                                raise RuntimeError("No frames extracted")
                            video_frame_inputs.append([PILImage.open(p).convert("RGB") for p in frames])

                        # Single video -> pass list[PIL]; multiple videos -> list[list[PIL]]
                        processor_call["videos"] = (
                            video_frame_inputs[0]
                            if len(video_frame_inputs) == 1
                            else video_frame_inputs
                        )
                    except Exception:
                        # If anything goes wrong with ffmpeg sampling, fall back to transformers decode.
                        processor_call["videos"] = video_paths if len(video_paths) > 1 else video_paths[0]
                        processor_call["videos_kwargs"] = {"do_sample_frames": True, "num_frames": max_video_frames}
                    finally:
                        # Cleanup extracted frames directories (frames are already loaded into memory as PIL).
                        for d in temp_dirs:
                            try:
                                import shutil

                                shutil.rmtree(d, ignore_errors=True)
                            except Exception:
                                pass

                inputs = self.processor(**processor_call)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.model_instance.device)
            else:
                templated = self.processor.apply_chat_template(
                    chat_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if isinstance(templated, str):
                    # Processor returned a prompt string; fall back to explicit processor call.
                    image_inputs = []
                    for media_item in (media or []):
                        if MediaType is not None and getattr(media_item, "media_type", None) == MediaType.IMAGE:
                            fp = getattr(media_item, "file_path", None)
                            if isinstance(fp, str) and fp.strip():
                                try:
                                    from PIL import Image as PILImage
                                except ImportError as e:
                                    raise RuntimeError(f"PIL is required for HuggingFace image inputs: {e}")
                                image_inputs.append(PILImage.open(fp).convert("RGB"))

                    processor_call: Dict[str, Any] = {"text": templated, "return_tensors": "pt"}
                    if image_inputs:
                        processor_call["images"] = image_inputs if len(image_inputs) > 1 else image_inputs[0]
                    inputs = self.processor(**processor_call)
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.model_instance.device)
                else:
                    inputs = templated.to(self.model_instance.device)

            temperature_value = kwargs.get("temperature", self.temperature)
            # For HF multimodal video models, default to greedy decoding unless the caller explicitly
            # provided a temperature. This avoids premature EOS producing unusably short answers.
            if has_video and ("temperature" in kwargs) and kwargs.get("temperature") is None:
                temperature_value = 0.0
            if temperature_value is None:
                temperature_value = self.temperature

            max_new_tokens_raw = kwargs.get("max_output_tokens", None)
            if max_new_tokens_raw is None:
                max_new_tokens_raw = kwargs.get("max_tokens", None)
            if max_new_tokens_raw is None:
                max_new_tokens_raw = self.max_output_tokens or 512
            try:
                max_new_tokens_value = max(1, int(max_new_tokens_raw))
            except Exception:
                max_new_tokens_value = int(self.max_output_tokens or 512)

            do_sample = True
            try:
                if temperature_value is None or float(temperature_value) <= 0:
                    do_sample = False
                    temperature_value = 0.0
            except Exception:
                do_sample = True

            generation_kwargs = {
                "max_new_tokens": max_new_tokens_value,
                "temperature": temperature_value,
                "do_sample": do_sample,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }

            # Add seed if provided
            seed_value = self._normalize_seed(kwargs.get("seed", self.seed))
            if seed_value is not None:
                torch.manual_seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)

            # Generate response
            generated_ids = None
            try:
                with torch.inference_mode():
                    use_mps_lock = str(getattr(self, "device", "") or "").strip().lower() == "mps"
                    if use_mps_lock:
                        with _MPS_GENERATION_LOCK:
                            generated_ids = self.model_instance.generate(**inputs, **generation_kwargs)
                    else:
                        generated_ids = self.model_instance.generate(**inputs, **generation_kwargs)
            except RuntimeError as e:
                if str(getattr(self, "device", "") or "").strip().lower() == "mps":
                    raise RuntimeError(
                        "HuggingFaceProvider vision/video generation failed on MPS. "
                        "If this persists, force CPU via ABSTRACTCORE_HF_DEVICE=cpu."
                    ) from e
                raise
            finally:
                # Best-effort: keep MPS memory pressure low between calls.
                try:
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            if hasattr(torch.mps, "synchronize"):
                                torch.mps.synchronize()
                            torch.mps.empty_cache()
                except Exception:
                    pass
                try:
                    import gc

                    gc.collect()
                except Exception:
                    pass

            # Decode response
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )

            # Calculate generation time
            gen_time = (time.time() - start_time) * 1000

            # Calculate token usage
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = len(generated_ids[0]) - input_tokens

            response = GenerateResponse(
                content=output_text.strip(),
                model=self.model,
                finish_reason="stop",
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens
                },
                gen_time=gen_time
            )
            if stream:
                def _single_chunk_stream() -> Iterator[GenerateResponse]:
                    yield response
                return _single_chunk_stream()
            return response

        except Exception as e:
            gen_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0
            error_resp = GenerateResponse(
                content=f"Error in vision model generation: {str(e)}",
                model=self.model,
                finish_reason="error",
                gen_time=gen_time
            )
            if stream:
                def _error_stream() -> Iterator[GenerateResponse]:
                    yield error_resp
                return _error_stream()
            return error_resp

    def _patch_deepseek_for_mps(self):
        """Patch DeepSeek-OCR model to work with MPS instead of CUDA"""
        import types

        def patched_infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
            """Patched infer method that uses MPS instead of CUDA"""
            import torch

            # Determine the best available device
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            # Call the original infer method but patch tensor.cuda() calls
            original_cuda = torch.Tensor.cuda

            def patched_cuda(tensor, device=None, non_blocking=False, **kwargs):
                """Redirect .cuda() calls to the appropriate device"""
                if device == 'mps' or (device is None and torch.backends.mps.is_available()):
                    return tensor.to('mps', non_blocking=non_blocking)
                elif torch.cuda.is_available():
                    return original_cuda(tensor, device, non_blocking, **kwargs)
                else:
                    return tensor.to('cpu', non_blocking=non_blocking)

            # Temporarily patch the cuda method
            torch.Tensor.cuda = patched_cuda

            try:
                # Move model to the appropriate device first
                self.to(device)

                # Call original infer with device patching
                return self._original_infer(tokenizer, prompt, image_file, output_path, base_size, image_size, crop_mode, test_compress, save_results, eval_mode)
            finally:
                # Restore original cuda method
                torch.Tensor.cuda = original_cuda

        # Only patch if not already patched
        if not hasattr(self.model_instance, '_original_infer'):
            self.model_instance._original_infer = self.model_instance.infer
            self.model_instance.infer = types.MethodType(patched_infer, self.model_instance)

    def _generate_gguf(self,
                       prompt: str,
                       messages: Optional[List[Dict[str, str]]] = None,
                       system_prompt: Optional[str] = None,
                       tools: Optional[List[Dict[str, Any]]] = None,
                       media: Optional[List['MediaContent']] = None,
                       stream: bool = False,
                       response_model: Optional[Type[BaseModel]] = None,
                       **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using GGUF backend with llama-cpp-python"""

        if not self.llm:
            return GenerateResponse(
                content="Error: GGUF model not loaded",
                model=self.model,
                finish_reason="error"
            )

        # Handle media content for the user message - use proper vision format for GGUF models
        media_enrichment = None
        if media:
            try:
                from ..architectures.detection import supports_vision

                # Check if this model supports vision natively
                if supports_vision(self.model):
                    # Use HuggingFace multimodal format for vision-capable GGUF models
                    user_message_content = []

                    # Add text content
                    user_message_content.append({"type": "text", "text": prompt})

                    # Add media content
                    for media_item in media:
                        if hasattr(media_item, 'file_path') and media_item.file_path:
                            # Use file:// URL format as specified in HuggingFace docs
                            file_path = str(media_item.file_path)
                            if not file_path.startswith('file://'):
                                file_path = f"file://{file_path}"
                            user_message_content.append({
                                "type": "image",
                                "image": file_path
                            })
                        elif hasattr(media_item, 'content') and media_item.content:
                            # For base64 or other content, we might need to save to temp file
                            import tempfile
                            import base64
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                                if isinstance(media_item.content, str) and media_item.content.startswith('data:'):
                                    # Handle base64 data URLs
                                    header, data = media_item.content.split(',', 1)
                                    decoded_data = base64.b64decode(data)
                                    tmp_file.write(decoded_data)
                                else:
                                    tmp_file.write(media_item.content)
                                tmp_file.flush()
                                user_message_content.append({
                                    "type": "image",
                                    "image": f"file://{tmp_file.name}"
                                })
                else:
                    # Fallback to text-based media handling for non-vision models
                    from ..media.handlers import LocalMediaHandler
                    media_handler = LocalMediaHandler("huggingface", self.model_capabilities, model_name=self.model)
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    media_enrichment = getattr(media_handler, "media_enrichment", None)
                    user_message_content = multimodal_message if isinstance(multimodal_message, str) else prompt

            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                user_message_content = prompt
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                user_message_content = prompt
        else:
            user_message_content = prompt

        chat_messages = self._gguf_build_chat_messages(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            user_message_content=user_message_content,
        )

        # Prompt caching (GGUF/llama.cpp): best-effort per-key cache selection.
        cache_obj = None
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            key = prompt_cache_key.strip()
            cache_value = self._prompt_cache_store.get(key)
            if cache_value is None:
                self.prompt_cache_set(key, make_default=False)
                cache_value = self._prompt_cache_store.get(key)
            cache_obj = self._gguf_prompt_cache_unwrap(cache_value)
            try:
                if cache_obj is not None and hasattr(self.llm, "set_cache"):
                    self.llm.set_cache(cache_obj)
            except Exception:
                pass
        else:
            # Disable cache for this request when no key is provided.
            try:
                if hasattr(self.llm, "set_cache"):
                    self.llm.set_cache(None)
            except Exception:
                pass

        # Prepare parameters using unified system
        unified_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(unified_kwargs)

        generation_kwargs = {
            "messages": chat_messages,
            "max_tokens": max_output_tokens,  # This is max_output_tokens for llama-cpp
            "temperature": unified_kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": stream
        }

        # Add seed if provided (GGUF/llama-cpp supports seed)
        seed_value = unified_kwargs.get("seed")
        if seed_value is not None:
            generation_kwargs["seed"] = seed_value

        # Add native structured output support (llama-cpp-python format)
        # llama-cpp-python supports native structured outputs using the response_format parameter
        # This provides server-side guaranteed schema compliance
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            generation_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Handle tools - both native and prompted support
        has_native_tools = False
        if tools:
            # Check if model supports native tools - but fall back to prompted for now
            # TODO: Re-enable native tools once parameter default handling is fixed
            if False and self.llm.chat_format in ["chatml-function-calling", "functionary-v2"]:
                # Use unified tool handler for consistent formatting
                openai_tools = self.tool_handler.prepare_tools_for_native(tools)
                generation_kwargs["tools"] = openai_tools

                # Debug: Print what we're sending to the model
                print(f"DEBUG: Sending tools to HuggingFace model (unified handler):")
                import json
                print(json.dumps(openai_tools, indent=2))

                # Don't use auto for streaming (limitation of llama-cpp-python)
                if not stream:
                    generation_kwargs["tool_choice"] = "auto"
                has_native_tools = True

        try:
            # GGUF local control-plane generation: use cached state snapshots + `llm.generate(reset=False)`
            # to avoid llama-cpp-python's `create_chat_completion()` resetting and re-evaluating the full prompt.
            control_plane_enabled = (
                cache_obj is not None
                and self._gguf_prompt_cache_supports_local_control_plane()
                and os.environ.get("ABSTRACTCORE_GGUF_CONTROL_PLANE", "1").strip().lower() not in {"0", "false", "no", "off"}
                and response_model is None
                and not has_native_tools
                and self._gguf_control_plane_can_stream(chat_messages)
            )
            if control_plane_enabled:
                return self._gguf_control_plane_generate(
                    chat_messages=chat_messages,
                    cache_obj=cache_obj,
                    max_output_tokens=int(max_output_tokens),
                    temperature=float(generation_kwargs.get("temperature") or 0.2),
                    top_p=float(generation_kwargs.get("top_p") or 0.95),
                    top_k=int(kwargs.get("top_k", 40) or 40),
                    min_p=float(kwargs.get("min_p", 0.05) or 0.05),
                    typical_p=float(kwargs.get("typical_p", 1.0) or 1.0),
                    repeat_penalty=float(kwargs.get("repeat_penalty", 1.1) or 1.1),
                    presence_penalty=float(kwargs.get("presence_penalty", 0.0) or 0.0),
                    frequency_penalty=float(kwargs.get("frequency_penalty", 0.0) or 0.0),
                    tfs_z=float(kwargs.get("tfs_z", 1.0) or 1.0),
                    mirostat_mode=int(kwargs.get("mirostat_mode", 0) or 0),
                    mirostat_tau=float(kwargs.get("mirostat_tau", 5.0) or 5.0),
                    mirostat_eta=float(kwargs.get("mirostat_eta", 0.1) or 0.1),
                    seed=seed_value,
                    stream=bool(stream),
                )

            if stream:
                return self._stream_generate_gguf_with_tools(generation_kwargs, tools, has_native_tools, kwargs.get('tool_call_tags'))
            else:
                response = self._single_generate_gguf(generation_kwargs)
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Handle tool execution for both native and prompted responses
                if tools and (response.has_tool_calls() or
                             (self.tool_handler.supports_prompted and response.content)):
                    response = self._handle_tool_execution_gguf(response, tools, has_native_tools)

                return response

        except Exception as e:
            error_message = str(e)
            if stream:
                # Return error as a generator
                def error_generator():
                    yield GenerateResponse(
                        content=f"Error: {error_message}",
                        model=self.model,
                        finish_reason="error"
                    )
                return error_generator()
            else:
                return GenerateResponse(
                    content=f"Error: {error_message}",
                    model=self.model,
                    finish_reason="error"
                )

    def _single_generate_gguf(self, kwargs: Dict[str, Any]) -> GenerateResponse:
        """Generate single response using GGUF"""
        response = self.llm.create_chat_completion(**kwargs)

        choice = response['choices'][0]
        message = choice['message']

        # Extract tool calls if present
        tool_calls = None
        if 'tool_calls' in message:
            tool_calls = []
            for tc in message['tool_calls']:
                tool_calls.append({
                    "id": tc.get('id'),
                    "type": tc.get('type', 'function'),
                    "name": tc['function']['name'],
                    "arguments": tc['function']['arguments']
                })

        # Extract usage
        usage = None
        if 'usage' in response:
            usage = {
                "prompt_tokens": response['usage'].get('prompt_tokens', 0),
                "completion_tokens": response['usage'].get('completion_tokens', 0),
                "total_tokens": response['usage'].get('total_tokens', 0)
            }

        # Fix HTML escaping in llama-cpp-python responses
        content = message.get('content', '')
        if content:
            import html
            content = html.unescape(content)

        return GenerateResponse(
            content=content,
            raw_response=response,
            model=self.model,
            finish_reason=choice.get('finish_reason', 'stop'),
            usage=usage,
            tool_calls=tool_calls
        )

    def _gguf_control_plane_stop_strings(self) -> List[str]:
        """Return stop strings for GGUF local control-plane generation."""
        chat_format = self._gguf_prompt_cache_control_plane_chat_format() or self._gguf_prompt_cache_chat_format()
        fmt = str(chat_format or "").strip().lower()
        if fmt == "llama-3":
            return ["<|eot_id|>"]
        # ChatML and chatml-function-calling.
        return ["<|im_end|>"]

    def _gguf_control_plane_can_stream(self, chat_messages: List[Dict[str, Any]]) -> bool:
        """Return True when control-plane streaming can safely handle the message payloads."""
        # Control-plane renderer/tokenizer only supports text content (strings / JSON-serializable).
        for msg in chat_messages or []:
            if not isinstance(msg, dict):
                return False
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"system", "user", "assistant"}:
                return False
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                continue
            # For now, fall back to llama-cpp-python's chat completion for multimodal payloads.
            return False
        return True

    def _gguf_control_plane_stream_generate(
        self,
        *,
        chat_messages: List[Dict[str, Any]],
        cache_obj: Any,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        typical_p: float,
        repeat_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        tfs_z: float,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        seed: Optional[int],
    ) -> Iterator[GenerateResponse]:
        """Generate GGUF text by prefilling cached KV state and sampling from it.

        This bypasses llama-cpp-python's `create_chat_completion()` so we can benefit from
        cached state snapshots even when llama.cpp does not support incremental KV trimming.
        """
        llm = getattr(self, "llm", None)
        if llm is None:
            yield GenerateResponse(
                content="Error: GGUF model not loaded",
                model=self.model,
                finish_reason="error",
            )
            return

        stop_strs = [s for s in (self._gguf_control_plane_stop_strings() or []) if isinstance(s, str) and s]
        flush_threshold = 160

        prompt_text, prompt_tokens = self._gguf_render_prompt_tokens(
            messages=chat_messages,
            add_generation_prompt=True,
        )

        # Prefill prompt KV from cache (do not persist the generation-prompt state; key mode
        # maintains transcript-aligned caches via prompt_cache_update).
        ok = self._gguf_prefill_prompt_cache(cache_obj, prompt_tokens, save_state=False, set_cache=False)
        if not ok:
            yield GenerateResponse(
                content="Error: failed to prefill GGUF prompt cache",
                model=self.model,
                finish_reason="error",
            )
            return

        # Best-effort determinism.
        if seed is not None:
            try:
                llm.set_seed(int(seed))
            except Exception:
                pass

        # Prefer stop detection by token id because special tokens (e.g. `<|im_end|>`) often
        # detokenize to empty bytes.
        stop_token_seqs: List[tuple[int, ...]] = []
        for s in stop_strs:
            try:
                toks = llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
                seq = tuple(int(t) for t in toks)
                if seq:
                    stop_token_seqs.append(seq)
            except Exception:
                continue

        max_stop_seq_len = max((len(seq) for seq in stop_token_seqs), default=0)
        recent_tokens: List[int] = []

        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")()
        pending = ""
        output_tokens = 0
        finish_reason = "stop"

        try:
            for tok in llm.generate(
                [],
                top_k=int(top_k),
                top_p=float(top_p),
                min_p=float(min_p),
                typical_p=float(typical_p),
                temp=float(temperature),
                repeat_penalty=float(repeat_penalty),
                frequency_penalty=float(frequency_penalty),
                presence_penalty=float(presence_penalty),
                tfs_z=float(tfs_z),
                mirostat_mode=int(mirostat_mode),
                mirostat_tau=float(mirostat_tau),
                mirostat_eta=float(mirostat_eta),
                reset=False,
            ):
                tok_i = int(tok)

                # Stop token detection (token-id based).
                if stop_token_seqs:
                    recent_tokens.append(tok_i)
                    if max_stop_seq_len and len(recent_tokens) > max_stop_seq_len:
                        recent_tokens = recent_tokens[-max_stop_seq_len:]
                    if any(
                        len(seq) <= len(recent_tokens) and tuple(recent_tokens[-len(seq) :]) == seq
                        for seq in stop_token_seqs
                    ):
                        finish_reason = "stop"
                        break

                output_tokens += 1
                if isinstance(max_output_tokens, int) and max_output_tokens > 0 and output_tokens > int(max_output_tokens):
                    finish_reason = "length"
                    break

                try:
                    token_bytes = llm.detokenize([tok_i])
                except Exception:
                    token_bytes = b""

                if token_bytes:
                    pending += decoder.decode(token_bytes)

                if len(pending) > flush_threshold:
                    yield GenerateResponse(content=pending, model=self.model)
                    pending = ""

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error",
            )
            return

        # Flush decoder and any remaining buffered content.
        try:
            pending += decoder.decode(b"", final=True) or ""
        except Exception:
            pass

        if pending:
            yield GenerateResponse(content=pending, model=self.model)

        completion_tokens = int(output_tokens)
        if finish_reason == "length" and isinstance(max_output_tokens, int) and max_output_tokens > 0:
            completion_tokens = int(max_output_tokens)

        usage = {
            "prompt_tokens": int(len(prompt_tokens)),
            "completion_tokens": completion_tokens,
            "total_tokens": int(len(prompt_tokens) + completion_tokens),
        }

        yield GenerateResponse(
            content="",
            model=self.model,
            finish_reason=finish_reason,
            usage=usage,
            metadata={
                "_acore_backend": "gguf_control_plane",
            },
        )

    def _gguf_control_plane_generate(
        self,
        *,
        chat_messages: List[Dict[str, Any]],
        cache_obj: Any,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        typical_p: float,
        repeat_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        tfs_z: float,
        mirostat_mode: int,
        mirostat_tau: float,
        mirostat_eta: float,
        seed: Optional[int],
        stream: bool,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        if stream:
            return self._gguf_control_plane_stream_generate(
                chat_messages=chat_messages,
                cache_obj=cache_obj,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                seed=seed,
            )

        collected = ""
        last: Optional[GenerateResponse] = None
        for chunk in self._gguf_control_plane_stream_generate(
            chat_messages=chat_messages,
            cache_obj=cache_obj,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            seed=seed,
        ):
            last = chunk
            if isinstance(chunk.content, str) and chunk.content:
                collected += chunk.content

        return GenerateResponse(
            content=collected,
            model=self.model,
            finish_reason=getattr(last, "finish_reason", None) if last is not None else "stop",
            usage=getattr(last, "usage", None) if last is not None else None,
            metadata=getattr(last, "metadata", None) if last is not None else None,
        )

    def _stream_generate_gguf(self, kwargs: Dict[str, Any], tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream response using GGUF with tool tag rewriting support"""
        stream = self.llm.create_chat_completion(**kwargs)

        current_tool_call = None
        accumulated_arguments = ""

        # Initialize tool tag rewriter if needed
        rewriter = None
        buffer = ""
        if tool_call_tags:
            try:
                from ..tools.tag_rewriter import create_tag_rewriter
                rewriter = create_tag_rewriter(tool_call_tags)
            except ImportError:
                pass

        for chunk in stream:
            if 'choices' not in chunk or not chunk['choices']:
                continue

            choice = chunk['choices'][0]
            delta = choice.get('delta', {})

            # Handle text content
            if 'content' in delta and delta['content']:
                # Fix HTML escaping in streaming content
                content = delta['content']
                if content:
                    import html
                    content = html.unescape(content)

                    # Apply tool tag rewriting if enabled
                    if rewriter:
                        rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                        content = rewritten_content

                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason=choice.get('finish_reason')
                )

            # Handle tool calls
            if 'tool_calls' in delta:
                for tc in delta['tool_calls']:
                    if 'function' in tc:
                        if tc.get('id'):  # New tool call
                            if current_tool_call and accumulated_arguments:
                                # Yield the previous tool call
                                current_tool_call['arguments'] = accumulated_arguments
                                yield GenerateResponse(
                                    content="",
                                    model=self.model,
                                    tool_calls=[current_tool_call]
                                )

                            # Start new tool call
                            current_tool_call = {
                                "id": tc.get('id'),
                                "type": tc.get('type', 'function'),
                                "name": tc['function'].get('name'),
                                "arguments": ""
                            }
                            accumulated_arguments = tc['function'].get('arguments', '')
                        else:
                            # Accumulate arguments
                            if current_tool_call:
                                accumulated_arguments += tc['function'].get('arguments', '')

            # Handle finish reason
            if choice.get('finish_reason'):
                # Yield any pending tool call
                if current_tool_call and accumulated_arguments:
                    current_tool_call['arguments'] = accumulated_arguments
                    yield GenerateResponse(
                        content="",
                        model=self.model,
                        finish_reason=choice['finish_reason'],
                        tool_calls=[current_tool_call]
                    )
                else:
                    yield GenerateResponse(
                        content="",
                        model=self.model,
                        finish_reason=choice['finish_reason']
                    )

    def _single_generate_transformers_cached(
        self,
        *,
        prompt: str,
        prompt_cache_key: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Any]],
        prefilled_modules: Any,
        max_new_tokens: int,
        temperature: Optional[float],
        top_p: float,
        seed: Optional[int] = None,
    ) -> GenerateResponse:
        """Generate a single response using a transformers KV cache keyed by `prompt_cache_key`."""

        if not isinstance(prompt_cache_key, str) or not prompt_cache_key.strip():
            raise ValueError("prompt_cache_key must be a non-empty string")

        if not self._transformers_prompt_cache_supported():
            raise ValueError("Transformers prompt caching is not available for this model/provider instance.")

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ImportError("Transformers prompt caching requires `torch`.") from e

        if getattr(self, "model_instance", None) is None or getattr(self, "tokenizer", None) is None:
            raise RuntimeError("Transformers model/tokenizer not loaded")

        key = prompt_cache_key.strip()
        cache_value = self._prompt_cache_store.get(key)
        if cache_value is None:
            self.prompt_cache_set(key, make_default=False)
            cache_value = self._prompt_cache_store.get(key)

        state = self._transformers_prompt_cache_state(cache_value)
        if state is None:
            raise RuntimeError("prompt cache key does not reference a transformers cache state")

        # Best-effort first-call prefill when callers pass system/tools/messages alongside the key.
        if not state.prompt_tokens and (system_prompt is not None or messages or tools):
            tools_for_cache = None
            if isinstance(tools, list) and tools and all(isinstance(t, dict) for t in tools):
                tools_for_cache = tools  # type: ignore[assignment]
            self.prompt_cache_update(
                key,
                system_prompt=system_prompt,
                tools=tools_for_cache,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                add_generation_prompt=False,
            )
            cache_value = self._prompt_cache_store.get(key)
            state = self._transformers_prompt_cache_state(cache_value) or state

        # Seed for determinism (best-effort).
        if seed is not None:
            try:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass

        start_time = time.time()

        # Delta-only fragment: user message + assistant generation prefix.
        delta_text = self._transformers_build_prompt_fragment(
            prompt=str(prompt or ""),
            messages=None,
            system_prompt=None,
            tools=None,
            add_generation_prompt=True,
            prefilled_modules=prefilled_modules,
        )
        delta_ids = self._transformers_tokenize_fragment(delta_text, add_bos_if_empty=not bool(state.prompt_tokens))
        if not delta_ids:
            return GenerateResponse(
                content="",
                model=self.model,
                finish_reason="stop",
                usage={"prompt_tokens": len(state.prompt_tokens), "completion_tokens": 0, "total_tokens": len(state.prompt_tokens)},
                gen_time=round((time.time() - start_time) * 1000, 1),
            )

        device = self._transformers_cache_device() or torch.device("cpu")
        past_len = len(state.prompt_tokens)
        input_ids = torch.tensor([delta_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, past_len + len(delta_ids)), dtype=torch.long, device=device)

        do_sample = True
        temp_val: float = float(temperature) if temperature is not None else float(getattr(self, "temperature", 0.7) or 0.7)
        try:
            if temp_val <= 0:
                do_sample = False
                temp_val = 0.0
        except Exception:
            do_sample = True

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        try:
            pad_i = int(pad_token_id) if pad_token_id is not None else None
        except Exception:
            pad_i = None
        try:
            eos_i = int(eos_token_id) if eos_token_id is not None else None
        except Exception:
            eos_i = None
        if pad_i is None and eos_i is not None:
            pad_i = eos_i

        generate_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "temperature": temp_val,
            "top_p": float(top_p),
            "do_sample": bool(do_sample),
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": pad_i,
            "cache_implementation": "dynamic",
        }
        if eos_i is not None:
            generate_kwargs["eos_token_id"] = eos_i

        # Prefer updating the existing cache object in-place for speed.
        generate_kwargs["past_key_values"] = state.cache

        output = None
        try:
            with torch.inference_mode():
                use_mps_lock = str(device).startswith("mps") or str(getattr(self, "device", "") or "").strip().lower() == "mps"
                if use_mps_lock:
                    with _MPS_GENERATION_LOCK:
                        output = self.model_instance.generate(**generate_kwargs)
                else:
                    output = self.model_instance.generate(**generate_kwargs)
        except Exception:
            # Fallback: some models may reject an empty cache object on the first call.
            try:
                generate_kwargs["past_key_values"] = None
                with torch.inference_mode():
                    output = self.model_instance.generate(**generate_kwargs)
            except Exception as e:
                raise RuntimeError(f"Transformers cached generation failed: {e}") from e

        sequences = getattr(output, "sequences", None)
        if sequences is None:
            # Some generate paths return a raw tensor; treat it like sequences.
            sequences = output

        seq0 = sequences[0].tolist() if hasattr(sequences, "__getitem__") else []
        gen_ids = [int(tok) for tok in seq0[len(delta_ids):]] if seq0 else []

        decoded = ""
        try:
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip() if gen_ids else ""
        except Exception:
            decoded = ""

        # Update KV cache + token tracker.
        new_cache = getattr(output, "past_key_values", None)
        if new_cache is not None:
            state.cache = new_cache
        state.prompt_tokens = tuple(int(tok) for tok in (state.prompt_tokens + tuple(delta_ids) + tuple(gen_ids)))
        state.add_generation_prompt = False

        if prompt:
            try:
                state.messages.append({"role": "user", "content": str(prompt)})
            except Exception:
                pass
        if decoded:
            try:
                state.messages.append({"role": "assistant", "content": decoded})
            except Exception:
                pass

        try:
            meta = self._prompt_cache_store.meta(key) or {}
            meta = dict(meta)
            meta["token_count"] = len(state.prompt_tokens)
            self._prompt_cache_store.set(key, state, meta=meta)
        except Exception:
            pass

        gen_time = round((time.time() - start_time) * 1000, 1)
        usage = {
            "prompt_tokens": past_len + len(delta_ids),
            "completion_tokens": len(gen_ids),
            "total_tokens": past_len + len(delta_ids) + len(gen_ids),
            "input_tokens": past_len + len(delta_ids),
            "output_tokens": len(gen_ids),
        }

        return GenerateResponse(
            content=decoded,
            model=self.model,
            finish_reason="stop",
            usage=usage,
            gen_time=gen_time,
        )

    def _single_generate_transformers(self, input_text: str, max_new_tokens: int,
                                     temperature: float, top_p: float, seed: Optional[int] = None) -> GenerateResponse:
        """Generate single response using transformers (original implementation)"""
        try:
            # Set seed for deterministic generation if provided
            if seed is not None:
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except ImportError:
                    pass  # Skip seeding if torch not available

            # Track generation time
            start_time = time.time()

            outputs = self.pipeline(
                input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                truncation=True,
                return_full_text=False
            )

            gen_time = round((time.time() - start_time) * 1000, 1)

            if outputs and len(outputs) > 0:
                response_text = outputs[0]['generated_text'].strip()

                # Calculate token usage using centralized utilities
                usage = self._calculate_usage(input_text, response_text)

                return GenerateResponse(
                    content=response_text,
                    model=self.model,
                    finish_reason="stop",
                    usage=usage,
                    gen_time=gen_time
                )
            else:
                return GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop",
                    gen_time=gen_time
                )

        except Exception as e:
            gen_time = round((time.time() - start_time) * 1000, 1) if 'start_time' in locals() else 0.0
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error",
                gen_time=gen_time
            )

    def _calculate_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate token usage using centralized token utilities."""
        from ..utils.token_utils import TokenUtils

        input_tokens = TokenUtils.estimate_tokens(prompt, self.model)
        output_tokens = TokenUtils.estimate_tokens(response, self.model)
        total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Keep legacy keys for backward compatibility
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens
        }

    def _stream_generate_transformers(self, input_text: str, max_new_tokens: int,
                                     temperature: float, top_p: float, tool_call_tags: Optional[str] = None, seed: Optional[int] = None) -> Iterator[GenerateResponse]:
        """Stream response using transformers (simulated, original implementation) with tool tag rewriting support"""
        try:
            # HuggingFace doesn't have native streaming, so we simulate it
            full_response = self._single_generate_transformers(input_text, max_new_tokens, temperature, top_p, seed)

            if full_response.content:
                # Apply tool tag rewriting if enabled
                content = full_response.content
                if tool_call_tags:
                    try:
                        from ..tools.tag_rewriter import create_tag_rewriter
                        rewriter = create_tag_rewriter(tool_call_tags)
                        content = rewriter.rewrite_text(content)
                    except ImportError:
                        pass

                words = content.split()
                for i, word in enumerate(words):
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    yield GenerateResponse(
                        content=chunk_content,
                        model=self.model,
                        finish_reason="stop" if i == len(words) - 1 else None
                    )
            else:
                yield GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop"
                )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_input_text_transformers(self, prompt: str, messages: Optional[List[Dict[str, str]]],
                                      system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build input text for transformers model with tool support"""

        # Add tools to system prompt if provided
        final_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt += f"\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Check if model has chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template if available
            chat_messages = []

            if final_system_prompt:
                chat_messages.append({"role": "system", "content": final_system_prompt})

            if messages:
                chat_messages.extend(messages)

            chat_messages.append({"role": "user", "content": prompt})

            try:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback if chat template fails
                pass

        # Build simple conversational format
        text_parts = []

        if system_prompt:
            text_parts.append(f"System: {system_prompt}\n")

        if messages:
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                text_parts.append(f"{role}: {content}\n")

        text_parts.append(f"User: {prompt}\n")
        text_parts.append("Assistant:")

        return "".join(text_parts)

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = ["chat", "streaming"]

        if self.model_type == "gguf":
            capabilities.append("gguf")
            if self.llm and self.llm.chat_format:
                capabilities.append("tools")
        else:
            # Check for specific model capabilities
            model_lower = self.model.lower()

            if "gpt2" in model_lower or "dialogpt" in model_lower:
                capabilities.append("dialogue")

            if "codegen" in model_lower or "starcoder" in model_lower or "coder" in model_lower:
                capabilities.append("code")

        return capabilities

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if self.model_type == "gguf":
            return self.llm is not None
        else:
            return self.pipeline is not None


    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter appropriate for the model type"""
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)

        if self.model_type == "gguf":
            # For GGUF models, this is the generation limit
            return max_output_tokens
        else:
            # For transformers, this is max_new_tokens
            return max_output_tokens


    def _stream_generate_transformers_with_tools(self, input_text: str, max_new_tokens: int,
                                               temperature: float, top_p: float,
                                               tools: Optional[List[Dict[str, Any]]] = None,
                                               tool_call_tags: Optional[str] = None, seed: Optional[int] = None) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate_transformers(input_text, max_new_tokens, temperature, top_p, tool_call_tags, seed):
            collected_content += chunk.content
            yield chunk

        # Handle tool execution if we have tools and content
        if tools and self.tool_handler.supports_prompted and collected_content:
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop"
            )

            # Handle tool execution using base method
            final_response = self._handle_prompted_tool_execution(complete_response, tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )

    def _handle_tool_execution_gguf(self, response: GenerateResponse, tools: List[Dict[str, Any]], has_native_tools: bool) -> GenerateResponse:
        """Handle tool execution for GGUF responses - both native and prompted"""
        if has_native_tools and response.has_tool_calls():
            # Handle native tool calls using base method
            tool_calls = self._convert_native_tool_calls_to_standard(response.tool_calls)
            return self._execute_tools_with_events(response, tool_calls)
        elif self.tool_handler.supports_prompted and response.content:
            # Handle prompted tool calls using base method
            return self._handle_prompted_tool_execution(response, tools)

        return response

    def _stream_generate_gguf_with_tools(self, generation_kwargs: Dict[str, Any],
                                       tools: Optional[List[Dict[str, Any]]] = None,
                                       has_native_tools: bool = False,
                                       tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream generate GGUF with tool execution at the end"""
        collected_content = ""
        collected_tool_calls = []

        # Stream the response content
        for chunk in self._stream_generate_gguf(generation_kwargs, tool_call_tags):
            collected_content += chunk.content
            if chunk.tool_calls:
                collected_tool_calls.extend(chunk.tool_calls)
            yield chunk

        # Handle tool execution if we have tools and content/calls
        if tools and (collected_tool_calls or
                     (self.tool_handler.supports_prompted and collected_content)):
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop",
                tool_calls=collected_tool_calls
            )

            # Handle tool execution using simplified method
            final_response = self._handle_tool_execution_gguf(complete_response, tools, has_native_tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """
        List available HuggingFace models from local cache (excluding MLX models).

        Args:
            **kwargs: Optional parameters including:
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if not hf_cache.exists():
                return []

            models = []
            for item in hf_cache.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Convert models--microsoft--DialoGPT-medium to microsoft/DialoGPT-medium
                    model_name = item.name.replace("models--", "").replace("--", "/")

                    # CRITICAL: Exclude MLX models from HuggingFace list
                    # Any model with "mlx" in the name should be classified as MLX, not HuggingFace
                    if "mlx" not in model_name.lower():
                        models.append(model_name)

            models = sorted(models)

            # Apply new capability filtering if provided
            input_capabilities = kwargs.get('input_capabilities')
            output_capabilities = kwargs.get('output_capabilities')

            if input_capabilities or output_capabilities:
                models = filter_models_by_capabilities(
                    models, 
                    input_capabilities=input_capabilities,
                    output_capabilities=output_capabilities
                )


            return models

        except Exception:
            return []
