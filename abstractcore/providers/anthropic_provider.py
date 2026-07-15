"""
Anthropic provider implementation.
"""

import os
import json
import time
import warnings
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Type, TYPE_CHECKING

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider, ThinkingControlHandling
from ..core.types import GenerateResponse
from ..exceptions import AuthenticationError, ProviderAPIError, ModelNotFoundError, format_model_error, format_auth_error
from ..tools import UniversalToolHandler, execute_tools, merge_tools_into_system
from ..events import EventType

if TYPE_CHECKING:
    from ..media.types import MediaContent

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider with full integration"""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "anthropic"

        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

        # Get API key from param or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        # Get base URL from param or environment
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        # Initialize client with timeout and optional base_url
        client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self._async_client = None  # Lazy-loaded async client

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Store provider-specific configuration
        self.top_p = kwargs.get("top_p", getattr(self, "top_p", 1.0))
        self.top_k = kwargs.get("top_k", getattr(self, "top_k", None))

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        # Anthropic Messages API exposes thinking controls via a `thinking` object.
        #
        # As of early 2026, newer Claude models recommend "adaptive" thinking with a separate
        # `output_config.effort` knob, while manual `budget_tokens` is deprecated but still
        # accepted for extended thinking on some models.
        if enabled is None and level is None:
            return kwargs, ThinkingControlHandling()

        caps = self.model_capabilities if isinstance(self.model_capabilities, dict) else {}
        mode = str(caps.get("thinking_control_mode") or "").strip().lower()
        adaptive_supported = mode == "adaptive"
        max_effort_supported = bool(caps.get("max_effort_supported")) if "max_effort_supported" in caps else False
        if not mode:
            # Backward-compatible heuristic fallback for models missing capability metadata.
            model_s = str(getattr(self, "model", "") or "").strip().lower()
            adaptive_supported = ("opus-4-6" in model_s) or ("sonnet-4-6" in model_s) or ("4.6" in model_s)
            max_effort_supported = ("opus-4-6" in model_s) or ("4.6" in model_s and "opus" in model_s)

        def _level_to_effort(lvl: Optional[str]) -> str:
            if lvl in {"low", "medium", "high"}:
                return lvl
            if lvl == "xhigh":
                return "max" if max_effort_supported else "high"
            return "medium"

        def _level_to_budget_tokens(lvl: Optional[str]) -> int:
            budget_map = {"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384}
            return int(budget_map.get(str(lvl or "").strip().lower(), 4096))

        new_kwargs = dict(kwargs)

        if enabled is False:
            new_kwargs["thinking"] = {"type": "disabled"}
            return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=False)

        if adaptive_supported:
            new_kwargs["thinking"] = {"type": "adaptive"}
            effort = _level_to_effort(level)
            if level == "xhigh" and effort != "max":
                warnings.warn(
                    f"thinking='xhigh' requested for Anthropic model '{self.model}', but effort='max' is not "
                    "supported; using effort='high'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            output_config = new_kwargs.get("output_config")
            output_config_dict: Dict[str, Any] = dict(output_config) if isinstance(output_config, dict) else {}
            output_config_dict["effort"] = effort
            new_kwargs["output_config"] = output_config_dict
            return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)

        # Manual budget fallback (deprecated on newest models but still best-effort for older ones).
        budget_tokens = _level_to_budget_tokens(level)
        max_out = new_kwargs.get("max_output_tokens")
        try:
            max_out_i = int(max_out) if max_out is not None else None
        except Exception:
            max_out_i = None
        if isinstance(max_out_i, int) and max_out_i > 0:
            budget_tokens = max(0, min(int(budget_tokens), int(max_out_i)))

        new_kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(budget_tokens)}
        return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)

    @property
    def async_client(self):
        """Lazy-load AsyncAnthropic client for native async operations."""
        if self._async_client is None:
            client_kwargs = {"api_key": self.api_key, "timeout": self._timeout}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._async_client = anthropic.AsyncAnthropic(**client_kwargs)
        return self._async_client

    # Explicit framing for system instructions that must be delivered inside the user
    # turn stream (Anthropic's Messages API accepts only user/assistant roles in
    # `messages`; `system` is a top-level parameter). XML-style tags follow Anthropic's
    # own prompting conventions for injected context.
    _SYSTEM_WRAP_OPEN = "<system_instruction>"
    _SYSTEM_WRAP_CLOSE = "</system_instruction>"

    def _build_anthropic_history(
        self,
        messages: Optional[List[Dict[str, Any]]],
    ) -> tuple[List[Dict[str, Any]], List[str], int]:
        """Convert AbstractCore chat history into Anthropic Messages API form.

        `role:"system"` entries are converted instead of silently dropped:
        - a LEADING contiguous run of system messages is returned as
          ``leading_system_parts`` for the caller to merge into the top-level ``system``
          parameter (its native surface — covers clients that send the system prompt as
          ``messages[0]``);
        - NON-LEADING system messages are converted in place into user messages wrapped
          in ``<system_instruction>`` tags, preserving their position (tail-placed hints
          stay tail-anchored). Emission is deferred past contiguous ``tool`` runs so a
          converted message never lands between an assistant tool_use turn and its
          tool_result (Anthropic placement rule).

        Returns ``(api_messages, leading_system_parts, wrapped_count)``.
        """
        api_messages: List[Dict[str, Any]] = []
        leading_system_parts: List[str] = []
        pending_wrapped: List[Dict[str, Any]] = []
        wrapped_count = 0
        seen_non_system = False

        def _flush_pending() -> None:
            nonlocal pending_wrapped
            if pending_wrapped:
                api_messages.extend(pending_wrapped)
                pending_wrapped = []

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip()
            if not role:
                continue
            content = msg.get("content")

            if role == "system":
                text = self._message_content_to_text(content).strip()
                if not text:
                    continue
                if not seen_non_system:
                    leading_system_parts.append(text)
                else:
                    pending_wrapped.append(
                        {
                            "role": "user",
                            "content": f"{self._SYSTEM_WRAP_OPEN}\n{text}\n{self._SYSTEM_WRAP_CLOSE}",
                        }
                    )
                    wrapped_count += 1
                continue

            seen_non_system = True

            if role == "assistant":
                _flush_pending()
                api_messages.append({"role": "assistant", "content": "" if content is None else content})
            elif role == "tool":
                # Anthropic Messages API represents tool outputs as `tool_result` content
                # blocks inside a USER message (there is no `role="tool"`). Tool results are
                # emitted BEFORE any pending wrapped system message so the tool_result stays
                # adjacent to its assistant tool_use turn.
                meta = msg.get("metadata") if isinstance(msg.get("metadata"), dict) else {}
                tool_use_id = meta.get("call_id") or meta.get("tool_use_id") or meta.get("id")
                tool_text = "" if content is None else str(content)

                if isinstance(tool_use_id, str) and tool_use_id.strip():
                    api_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id.strip(),
                                    "content": tool_text,
                                }
                            ],
                        }
                    )
                else:
                    # Fallback: preserve as plain user text when no tool_use_id is available.
                    api_messages.append({"role": "user", "content": tool_text})
            else:
                # Anthropic accepts only user/assistant roles; other roles are delivered as
                # user content (consecutive user turns are merged server-side).
                _flush_pending()
                api_messages.append({"role": "user", "content": "" if content is None else content})

        _flush_pending()
        return api_messages, leading_system_parts, wrapped_count

    @staticmethod
    def _merge_system_parts(system_prompt: Optional[str], leading_system_parts: List[str]) -> Optional[str]:
        """Merge the explicit system_prompt with leading in-`messages` system content."""
        parts: List[str] = []
        if isinstance(system_prompt, str) and system_prompt:
            parts.append(system_prompt)
        parts.extend(p for p in leading_system_parts if p)
        return "\n\n".join(parts) if parts else None

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with Anthropic API"""

        # Build messages array (system entries are converted, never dropped).
        api_messages, leading_system_parts, wrapped_system_count = self._build_anthropic_history(messages)

        # Add current prompt as user message
        if prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            # Handle multimodal message with media content
            if media:
                try:
                    from ..media.handlers import AnthropicMediaHandler
                    media_handler = AnthropicMediaHandler(self.model_capabilities)

                    # Create multimodal message combining text and media
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    api_messages.append(multimodal_message)
                except ImportError:
                    self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                    api_messages.append({"role": "user", "content": prompt})
                except Exception as e:
                    self.logger.warning(f"Failed to process media content: {e}")
                    api_messages.append({"role": "user", "content": prompt})
            else:
                api_messages.append({"role": "user", "content": prompt})

        # If media is present but no multimodal message was created (common when prompt="" and the
        # caller provided the request in `messages`), attach media to the last plain user message.
        if media:
            try:
                has_image = False
                for m in api_messages:
                    if not isinstance(m, dict):
                        continue
                    if m.get("role") != "user":
                        continue
                    c = m.get("content")
                    if not isinstance(c, list):
                        continue
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "image":
                            has_image = True
                            break
                    if has_image:
                        break

                if not has_image:
                    from ..media.handlers import AnthropicMediaHandler

                    media_handler = AnthropicMediaHandler(self.model_capabilities)
                    idx: Optional[int] = None
                    for i in range(len(api_messages) - 1, -1, -1):
                        m = api_messages[i]
                        if not isinstance(m, dict):
                            continue
                        if m.get("role") != "user":
                            continue
                        if isinstance(m.get("content"), str):
                            idx = i
                            break
                    if idx is None:
                        api_messages.append(media_handler.create_multimodal_message("", media))
                    else:
                        text0 = str(api_messages[idx].get("content") or "")
                        api_messages[idx] = media_handler.create_multimodal_message(text0, media)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        # Prepare API call parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_output_tokens,  # This is max_output_tokens for Anthropic
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "stream": stream
        }

        # Prompt caching (Anthropic): explicit per-block breakpoints when a cache key is
        # provided (Anthropic does not use our key value; it signals caching intent).
        # The previous top-level `cache_control` request param marks only the LAST cacheable
        # block — the end of the message transcript. Live-verified (2026-07-08, haiku-4.5):
        # in the agent-loop shape (volatile trailing message) that paid the 1.25x cache-WRITE
        # premium on the full prompt EVERY call and never produced a read (7,042/7,043-token
        # writes, 0 reads); below the model's minimum cacheable size it was a silent no-op.
        # The breakpoint now goes on the last system text block, caching the tools+system
        # static head (server prompt order is tools -> system -> messages): live-verified
        # write 6,302 on call 1 -> read 6,302 on call 2.
        prompt_cache_key = kwargs.get("prompt_cache_key")
        cache_enabled = (
            isinstance(prompt_cache_key, str) and prompt_cache_key.strip() and self.supports_prompt_cache()
        )
        cache_control: Optional[Dict[str, Any]] = None
        if cache_enabled:
            cache_control = {"type": "ephemeral"}
            ttl = kwargs.get("prompt_cache_ttl")
            if isinstance(ttl, str) and ttl.strip():
                cache_control["ttl"] = ttl.strip()

        thinking_cfg = kwargs.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg:
            call_params["thinking"] = thinking_cfg

        output_config = kwargs.get("output_config")
        if isinstance(output_config, dict) and output_config:
            call_params["output_config"] = output_config

        # Add system prompt if provided. Leading system messages inside `messages`
        # (e.g. server-mediated clients sending the system prompt as messages[0])
        # merge into the same top-level parameter — their native Anthropic surface.
        merged_system = self._merge_system_parts(system_prompt, leading_system_parts)
        if merged_system:
            call_params["system"] = merged_system

        # Add top_p if specified
        top_p_value = generation_kwargs.get("top_p", self.top_p)
        top_p_is_requested = (
            "top_p" in kwargs
            or "top_p" in getattr(self, "_explicit_generation_params", frozenset())
            or self._metadata_generation_default("top_p") is not None
        )
        if top_p_value is not None and (top_p_is_requested or top_p_value < 1.0):
            call_params["top_p"] = top_p_value

        # Add top_k if specified
        top_k_value = generation_kwargs.get("top_k", self.top_k)
        top_k_is_requested = (
            "top_k" in kwargs
            or "top_k" in getattr(self, "_explicit_generation_params", frozenset())
            or self._metadata_generation_default("top_k") is not None
        )
        if top_k_value is not None and top_k_is_requested:
            call_params["top_k"] = top_k_value

        # Handle seed parameter (Anthropic doesn't support seed natively)
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            import warnings
            warnings.warn(
                f"Seed parameter ({seed_value}) is not supported by Anthropic Claude API. "
                f"For deterministic outputs, use temperature=0.0 which may provide more consistent results, "
                f"though true determinism is not guaranteed.",
                UserWarning,
                stacklevel=3
            )
            self.logger.warning(f"Seed {seed_value} requested but not supported by Anthropic API")

        # Handle structured output using the "tool trick"
        structured_tool_name = None
        if response_model and PYDANTIC_AVAILABLE:
            # Create a synthetic tool for structured output
            structured_tool = self._create_structured_output_tool(response_model)

            # Add to existing tools or create new tools list
            if tools:
                tools = list(tools) + [structured_tool]
            else:
                tools = [structured_tool]

            structured_tool_name = structured_tool["name"]

            # Modify the prompt to instruct the model to use the structured tool
            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] += f"\n\nPlease use the {structured_tool_name} tool to provide your response."

        # Add tools if provided (convert to native format)
        if tools:
            if self.tool_handler.supports_native:
                # Use Anthropic-specific tool formatting instead of universal handler
                call_params["tools"] = self._format_tools_for_anthropic(tools)

                # Force tool use for structured output
                if structured_tool_name:
                    call_params["tool_choice"] = {"type": "tool", "name": structured_tool_name}
                elif kwargs.get("tool_choice"):
                    call_params["tool_choice"] = {"type": kwargs.get("tool_choice", "auto")}
            else:
                # Add tools as system prompt for prompted models (shared policy).
                system_text = call_params.get("system") if isinstance(call_params.get("system"), str) else ""
                merged = merge_tools_into_system(self.tool_handler, system_text, tools)
                if merged:
                    call_params["system"] = merged

        # Apply the prompt-cache breakpoint AFTER tools/system folding so the marked block
        # is genuinely the end of the static head (tools -> system in Anthropic's prompt order).
        self._apply_prompt_cache_breakpoints(call_params, cache_control)

        # Make API call with proper exception handling
        try:
            if stream:
                return self._stream_response(call_params, tools)
            else:
                # Track generation time
                start_time = time.time()
                response = self.client.messages.create(**call_params)
                gen_time = round((time.time() - start_time) * 1000, 1)

                formatted = self._format_response(response)
                # Add generation time to response
                formatted.gen_time = gen_time
                formatted.metadata = dict(formatted.metadata or {})
                formatted.metadata["_provider_request"] = {"call_params": call_params}
                if wrapped_system_count:
                    # Observability: non-leading system messages were delivered as
                    # <system_instruction>-wrapped user turns (Anthropic has no system role
                    # in `messages`); count the conversions instead of warning per call.
                    formatted.metadata["system_role_user_wrapped"] = wrapped_system_count

                # Handle tool execution for Anthropic responses
                if tools and (formatted.has_tool_calls() or
                             (self.tool_handler.supports_prompted and formatted.content)):
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception as e:
            # Use proper exception handling from base
            error_str = str(e).lower()

            if 'api_key' in error_str or 'authentication' in error_str:
                raise AuthenticationError(format_auth_error("anthropic", str(e)))
            elif ('not_found_error' in error_str and 'model:' in error_str) or '404' in error_str:
                # Model not found - show available models
                available_models = self.list_available_models(api_key=self.api_key)
                error_message = format_model_error("Anthropic", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise

    async def _agenerate_internal(self,
                                   prompt: str,
                                   messages: Optional[List[Dict[str, str]]] = None,
                                   system_prompt: Optional[str] = None,
                                   tools: Optional[List[Dict[str, Any]]] = None,
                                   media: Optional[List['MediaContent']] = None,
                                   stream: bool = False,
                                   response_model: Optional[Type[BaseModel]] = None,
                                   **kwargs) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Native async implementation using AsyncAnthropic - 3-10x faster for batch operations."""

        # Build messages array (same logic as sync; system entries converted, never dropped).
        api_messages, leading_system_parts, wrapped_system_count = self._build_anthropic_history(messages)

        # Add current prompt as user message
        if prompt and prompt not in [msg.get("content") for msg in (messages or [])]:
            # Handle multimodal message with media content
            if media:
                try:
                    from ..media.handlers import AnthropicMediaHandler
                    media_handler = AnthropicMediaHandler(self.model_capabilities)

                    # Create multimodal message combining text and media
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    api_messages.append(multimodal_message)
                except ImportError:
                    self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                    api_messages.append({"role": "user", "content": prompt})
                except Exception as e:
                    self.logger.warning(f"Failed to process media content: {e}")
                    api_messages.append({"role": "user", "content": prompt})
            else:
                api_messages.append({"role": "user", "content": prompt})

        # If media is present but no multimodal message was created (common when prompt="" and the
        # caller provided the request in `messages`), attach media to the last plain user message.
        if media:
            try:
                has_image = False
                for m in api_messages:
                    if not isinstance(m, dict):
                        continue
                    if m.get("role") != "user":
                        continue
                    c = m.get("content")
                    if not isinstance(c, list):
                        continue
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "image":
                            has_image = True
                            break
                    if has_image:
                        break

                if not has_image:
                    from ..media.handlers import AnthropicMediaHandler

                    media_handler = AnthropicMediaHandler(self.model_capabilities)
                    idx: Optional[int] = None
                    for i in range(len(api_messages) - 1, -1, -1):
                        m = api_messages[i]
                        if not isinstance(m, dict):
                            continue
                        if m.get("role") != "user":
                            continue
                        if isinstance(m.get("content"), str):
                            idx = i
                            break
                    if idx is None:
                        api_messages.append(media_handler.create_multimodal_message("", media))
                    else:
                        text0 = str(api_messages[idx].get("content") or "")
                        api_messages[idx] = media_handler.create_multimodal_message(text0, media)
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        # Prepare API call parameters (same logic as sync)
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        call_params = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_output_tokens,
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "stream": stream
        }

        prompt_cache_key = kwargs.get("prompt_cache_key")
        cache_enabled = (
            isinstance(prompt_cache_key, str) and prompt_cache_key.strip() and self.supports_prompt_cache()
        )
        cache_control: Optional[Dict[str, Any]] = None
        if cache_enabled:
            cache_control = {"type": "ephemeral"}
            ttl = kwargs.get("prompt_cache_ttl")
            if isinstance(ttl, str) and ttl.strip():
                cache_control["ttl"] = ttl.strip()

        thinking_cfg = kwargs.get("thinking")
        if isinstance(thinking_cfg, dict) and thinking_cfg:
            call_params["thinking"] = thinking_cfg

        output_config = kwargs.get("output_config")
        if isinstance(output_config, dict) and output_config:
            call_params["output_config"] = output_config

        # Add system prompt if provided (Anthropic-specific: separate parameter).
        # Leading system messages inside `messages` merge into the same parameter.
        merged_system = self._merge_system_parts(system_prompt, leading_system_parts)
        if merged_system:
            call_params["system"] = merged_system

        # Add top_p if specified
        top_p_value = generation_kwargs.get("top_p", self.top_p)
        top_p_is_requested = (
            "top_p" in kwargs
            or "top_p" in getattr(self, "_explicit_generation_params", frozenset())
            or self._metadata_generation_default("top_p") is not None
        )
        if top_p_value is not None and (top_p_is_requested or top_p_value < 1.0):
            call_params["top_p"] = top_p_value

        # Add top_k if specified
        top_k_value = generation_kwargs.get("top_k", self.top_k)
        top_k_is_requested = (
            "top_k" in kwargs
            or "top_k" in getattr(self, "_explicit_generation_params", frozenset())
            or self._metadata_generation_default("top_k") is not None
        )
        if top_k_value is not None and top_k_is_requested:
            call_params["top_k"] = top_k_value

        # Handle seed parameter (Anthropic doesn't support seed natively)
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            import warnings
            warnings.warn(
                f"Seed parameter ({seed_value}) is not supported by Anthropic Claude API. "
                f"For deterministic outputs, use temperature=0.0 which may provide more consistent results, "
                f"though true determinism is not guaranteed.",
                UserWarning,
                stacklevel=3
            )
            self.logger.warning(f"Seed {seed_value} requested but not supported by Anthropic API")

        # Handle structured output using the "tool trick"
        structured_tool_name = None
        if response_model and PYDANTIC_AVAILABLE:
            structured_tool = self._create_structured_output_tool(response_model)

            if tools:
                tools = list(tools) + [structured_tool]
            else:
                tools = [structured_tool]

            structured_tool_name = structured_tool["name"]

            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] += f"\n\nPlease use the {structured_tool_name} tool to provide your response."

        # Add tools if provided
        if tools:
            if self.tool_handler.supports_native:
                call_params["tools"] = self._format_tools_for_anthropic(tools)

                if structured_tool_name:
                    call_params["tool_choice"] = {"type": "tool", "name": structured_tool_name}
                elif kwargs.get("tool_choice"):
                    call_params["tool_choice"] = {"type": kwargs.get("tool_choice", "auto")}
            else:
                system_text = call_params.get("system") if isinstance(call_params.get("system"), str) else ""
                merged = merge_tools_into_system(self.tool_handler, system_text, tools)
                if merged:
                    call_params["system"] = merged

        # Apply the prompt-cache breakpoint AFTER tools/system folding (see sync path).
        self._apply_prompt_cache_breakpoints(call_params, cache_control)

        # Make async API call
        try:
            if stream:
                return self._async_stream_response(call_params, tools)
            else:
                start_time = time.time()
                response = await self.async_client.messages.create(**call_params)
                gen_time = round((time.time() - start_time) * 1000, 1)

                formatted = self._format_response(response)
                formatted.gen_time = gen_time
                formatted.metadata = dict(formatted.metadata or {})
                formatted.metadata["_provider_request"] = {"call_params": call_params}
                if wrapped_system_count:
                    formatted.metadata["system_role_user_wrapped"] = wrapped_system_count

                if tools and (formatted.has_tool_calls() or
                             (self.tool_handler.supports_prompted and formatted.content)):
                    formatted = self._handle_tool_execution(formatted, tools)

                return formatted
        except Exception as e:
            error_str = str(e).lower()

            if 'api_key' in error_str or 'authentication' in error_str:
                raise AuthenticationError(format_auth_error("anthropic", str(e)))
            elif ('not_found_error' in error_str and 'model:' in error_str) or '404' in error_str:
                available_models = self.list_available_models(api_key=self.api_key)
                error_message = format_model_error("Anthropic", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise

    async def _async_stream_response(self, call_params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[GenerateResponse]:
        """Native async streaming with Anthropic's context manager pattern."""
        stream_params = {k: v for k, v in call_params.items() if k != 'stream'}

        try:
            async with self.async_client.messages.stream(**stream_params) as stream:
                async for chunk in stream:
                    yield GenerateResponse(
                        content=getattr(chunk, 'content', ''),
                        model=self.model,
                        finish_reason=getattr(chunk, 'finish_reason', None),
                        raw_response=chunk
                    )
        except Exception as e:
            raise

    def unload_model(self, model_name: str) -> None:
        """Close async client if it was created."""
        if self._async_client is not None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.close())
            except RuntimeError:
                import asyncio
                asyncio.run(self._async_client.close())

    def _format_tools_for_anthropic(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API format"""
        formatted_tools = []
        for tool in tools:
            # Anthropic expects `input_schema` to be a JSON Schema object:
            # https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
            #
            # Our internal tool representation typically uses:
            #   tool["parameters"] = { "arg": {"type": "...", "default": ...?}, ... }
            # or, less commonly:
            #   tool["parameters"] = {"type":"object","properties":{...},"required":[...]}
            params = tool.get("parameters", {})

            properties: Dict[str, Any] = {}
            required: List[str] = []

            if isinstance(params, dict) and "properties" in params:
                # Treat as already-schema-like.
                raw_props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
                properties = dict(raw_props)
                raw_required = params.get("required")
                if isinstance(raw_required, list):
                    required = [str(x) for x in raw_required if isinstance(x, (str, int))]
            elif isinstance(params, dict):
                # Treat as compact parameter dict; infer required args by absence of `default`.
                properties = dict(params)
                for k, v in params.items():
                    if isinstance(v, dict) and "default" not in v:
                        required.append(str(k))

            input_schema: Dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                input_schema["required"] = required

            # Wire-safe alias: Anthropic enforces the same strict tool-name
            # contract as OpenAI — namespaced names (mcp::server::tool) 400
            # the whole request. Safe names pass through byte-identical; the
            # response normalizer maps aliases back (tools.wire_naming).
            from ..tools.wire_naming import wire_safe_tool_name

            formatted_tool = {
                "name": wire_safe_tool_name(str(tool.get("name") or "")),
                "description": tool.get("description", ""),
                "input_schema": input_schema
            }
            formatted_tools.append(formatted_tool)
        return formatted_tools

    def _format_response(self, response) -> GenerateResponse:
        """Format Anthropic response to GenerateResponse"""

        # Extract content from response
        content = ""
        tool_calls = None
        reasoning_parts: List[str] = []

        # Handle different content types
        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type in {"thinking", "redacted_thinking"}:
                thinking_text = getattr(content_block, "thinking", None)
                if thinking_text is None:
                    thinking_text = getattr(content_block, "text", None)
                if thinking_text is None:
                    thinking_text = getattr(content_block, "content", None)
                if thinking_text is not None:
                    reasoning_parts.append(str(thinking_text))
            elif content_block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": content_block.id,
                    "type": "tool_use",
                    "name": content_block.name,
                    "arguments": json.dumps(content_block.input) if not isinstance(content_block.input, str) else content_block.input
                })

        # Build usage dict
        usage = None
        if hasattr(response, 'usage'):
            usage = self._build_usage_dict(response.usage)

        metadata: Optional[Dict[str, Any]] = None
        reasoning = "\n".join([p for p in reasoning_parts if isinstance(p, str) and p.strip()]).strip()
        if reasoning:
            metadata = {"reasoning": reasoning}

        return GenerateResponse(
            content=content,
            raw_response=response,
            model=response.model,
            finish_reason=response.stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            metadata=metadata,
        )

    def _handle_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]]) -> GenerateResponse:
        """Handle tool execution for Anthropic responses"""
        # Check for native tool calls first
        if response.has_tool_calls():
            # Convert Anthropic tool calls to standard format using base method
            tool_calls = self._convert_native_tool_calls_to_standard(response.tool_calls)
            # Execute with events using base method
            return self._execute_tools_with_events(response, tool_calls)
        elif self.tool_handler.supports_prompted and response.content:
            # Handle prompted tool calls using base method
            return self._handle_prompted_tool_execution(response, tools)

        return response

    def _stream_response(self, call_params: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """Stream responses from Anthropic"""
        # Remove stream parameter for streaming API
        stream_params = {k: v for k, v in call_params.items() if k != 'stream'}
        with self.client.messages.stream(**stream_params) as stream:
            current_tool_call = None
            accumulated_input = ""

            # For tool execution, collect complete response
            collected_content = ""
            collected_tool_calls = []

            for chunk in stream:
                # Handle different event types
                if chunk.type == "content_block_start":
                    # Start of a new content block (could be text or tool_use)
                    if hasattr(chunk, 'content_block') and chunk.content_block.type == "tool_use":
                        # Starting a tool call
                        current_tool_call = {
                            "id": chunk.content_block.id,
                            "type": "tool_use",
                            "name": chunk.content_block.name,
                            "arguments": ""
                        }
                        accumulated_input = ""

                elif chunk.type == "content_block_delta":
                    if hasattr(chunk.delta, 'text'):
                        # Text content
                        text_content = chunk.delta.text
                        collected_content += text_content
                        yield GenerateResponse(
                            content=text_content,
                            raw_response=chunk,
                            model=call_params.get("model")
                        )
                    elif hasattr(chunk.delta, 'partial_json'):
                        # Tool call arguments coming in chunks
                        if current_tool_call:
                            accumulated_input += chunk.delta.partial_json
                            # Yield partial tool call
                            tool_call_partial = current_tool_call.copy()
                            tool_call_partial["arguments"] = accumulated_input
                            yield GenerateResponse(
                                content="",
                                raw_response=chunk,
                                model=call_params.get("model"),
                                tool_calls=[tool_call_partial]
                            )

                elif chunk.type == "content_block_stop":
                    # End of a content block
                    if current_tool_call and accumulated_input:
                        # Finalize the tool call with complete arguments
                        current_tool_call["arguments"] = accumulated_input
                        collected_tool_calls.append(current_tool_call)
                        yield GenerateResponse(
                            content="",
                            raw_response=chunk,
                            model=call_params.get("model"),
                            tool_calls=[current_tool_call]
                        )
                        current_tool_call = None
                        accumulated_input = ""

                elif chunk.type == "message_stop":
                    # Final chunk with usage info and tool execution
                    usage = None
                    if hasattr(stream, 'response') and hasattr(stream.response, 'usage'):
                        usage = self._build_usage_dict(stream.response.usage)

                    # Handle tool execution if we have tools and collected calls
                    if tools and (collected_tool_calls or
                                 (self.tool_handler.supports_prompted and collected_content)):
                        # Create complete response for tool processing
                        complete_response = GenerateResponse(
                            content=collected_content,
                            raw_response=chunk,
                            model=call_params.get("model"),
                            finish_reason="stop",
                            usage=usage,
                            tool_calls=collected_tool_calls
                        )

                        # Handle tool execution
                        final_response = self._handle_tool_execution(complete_response, tools)

                        # If tools were executed, yield the tool results as final chunk
                        if final_response.content != collected_content:
                            tool_results_content = final_response.content[len(collected_content):]
                            yield GenerateResponse(
                                content=tool_results_content,
                                raw_response=chunk,
                                model=call_params.get("model"),
                                finish_reason="stop",
                                usage=usage,
                                tool_calls=None
                            )

                    # Always yield final chunk
                    yield GenerateResponse(
                        content="",
                        raw_response=chunk,
                        model=call_params.get("model"),
                        finish_reason="stop",
                        usage=usage
                    )

    @staticmethod
    def _apply_prompt_cache_breakpoints(call_params: Dict[str, Any], cache_control: Optional[Dict[str, Any]]) -> None:
        """Place an explicit `cache_control` breakpoint at the end of the STATIC head.

        Anthropic caches the prompt prefix up to each marked block, in server prompt order
        tools -> system -> messages. Marking the LAST system block therefore caches
        tools + system — the byte-stable head of agent-loop requests — and costs one slot
        of the 4-breakpoint budget. Messages are deliberately NOT marked in v1: the
        agent-loop transcript ends in volatile per-call content, and a breakpoint there
        pays the 1.25x write premium for a block the next call cannot re-read.

        (The previous implementation passed a top-level `cache_control` request param,
        which marks only the last cacheable block = the volatile transcript tail.
        Live-verified 2026-07-08: full-prompt cache WRITE every call, zero reads — an
        active cost increase, not a no-op, whenever the prompt exceeded the model's
        minimum cacheable size. Explicit block placement is the correct surface.)
        """
        if not cache_control:
            return

        # Respect caller-placed breakpoints: Anthropic allows at most 4 explicit
        # cache_control blocks per request (a 5th is an API 400). If the caller already
        # marked ANY block (system, messages, or tools), they own the placement — adding
        # ours could exceed the budget or override their TTL choice.
        def _has_marker(blocks_like: Any) -> bool:
            if not isinstance(blocks_like, list):
                return False
            for item in blocks_like:
                if not isinstance(item, dict):
                    continue
                if item.get("cache_control"):
                    return True
                if _has_marker(item.get("content")):
                    return True
            return False

        if (
            _has_marker(call_params.get("system"))
            or _has_marker(call_params.get("messages"))
            or _has_marker(call_params.get("tools"))
        ):
            return

        system = call_params.get("system")
        blocks: list
        if isinstance(system, str) and system.strip():
            blocks = [{"type": "text", "text": system}]
        elif isinstance(system, list) and system:
            blocks = [dict(b) if isinstance(b, dict) else b for b in system]
        else:
            return  # no stable head to cache; skip rather than mark volatile content

        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            if isinstance(b, dict) and b.get("type") == "text":
                b = dict(b)
                b["cache_control"] = dict(cache_control)
                blocks[i] = b
                call_params["system"] = blocks
                return

    @staticmethod
    def _build_usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
        """Normalize Anthropic usage, including prompt-cache accounting.

        Anthropic reports cache traffic OUTSIDE `input_tokens`: the prompt actually
        processed is input_tokens + cache_read_input_tokens + cache_creation_input_tokens
        (billed at 1x / 0.1x / 1.25x respectively). Reporting the raw `input_tokens` as
        prompt size undercounts input whenever caching engages, so we report the INCLUSIVE
        sum (matching OpenAI semantics, where prompt_tokens includes cached tokens) and
        surface the split via normalized keys:
        - `cached_input_tokens`: input served from cache (read)
        - `cache_write_tokens`: input written to cache (creation premium)
        Absent != 0 is contractual: the keys appear only when the API reported the fields,
        so "cannot report" is distinguishable from "measured zero".
        """
        if usage is None:
            return None

        def _field(name: str) -> Optional[int]:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        base_in = _field("input_tokens") or 0
        out_toks = _field("output_tokens") or 0
        cache_read = _field("cache_read_input_tokens")
        cache_write = _field("cache_creation_input_tokens")
        total_in = base_in + (cache_read or 0) + (cache_write or 0)

        out: Dict[str, Any] = {
            "input_tokens": total_in,
            "output_tokens": out_toks,
            "total_tokens": total_in + out_toks,
            # Legacy keys for backward compatibility
            "prompt_tokens": total_in,
            "completion_tokens": out_toks,
        }
        if cache_read is not None:
            out["cached_input_tokens"] = cache_read
        if cache_write is not None:
            out["cache_write_tokens"] = cache_write
        return out

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = [
            "chat",
            "streaming",
            "system_prompt",
            "tools",
            "vision"  # All Claude 3 models support vision
        ]
        return capabilities


    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if not self.api_key:
            return False
        return True

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for Anthropic API"""
        # For Anthropic, max_tokens in the API is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _create_structured_output_tool(self, response_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Create a synthetic tool for structured output using Anthropic's tool calling.

        Args:
            response_model: Pydantic model to create tool for

        Returns:
            Tool definition dict for Anthropic API
        """
        schema = response_model.model_json_schema()
        tool_name = f"extract_{response_model.__name__.lower()}"

        return {
            "name": tool_name,
            "description": f"Extract structured data in {response_model.__name__} format",
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "additionalProperties": False
            }
        }

    def _update_http_client_timeout(self) -> None:
        """Update Anthropic client timeout when timeout is changed."""
        # Create new client with updated timeout
        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=self._timeout)

    def supports_prompt_cache(self) -> bool:
        """Anthropic supports prompt caching via `cache_control` (server-managed) on modern Claude models."""
        model_s = str(getattr(self, "model", "") or "").strip().lower()
        if not model_s:
            return False
        return any(
            token in model_s
            for token in (
                # Opus/Sonnet 4.x and Haiku 4.5+
                "opus-4",
                "sonnet-4",
                "haiku-4",
                # Sonnet 3.7 + Haiku 3.x
                "sonnet-3-7",
                "3.7-sonnet",
                "haiku-3",
                "3-haiku",
                "haiku-3-5",
                "3-5-haiku",
                "3.5-haiku",
            )
        )

    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models from Anthropic API.

        Args:
            **kwargs: Optional parameters including:
                - api_key: Anthropic API key
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            import httpx
            from .model_capabilities import filter_models_by_capabilities

            # Use provided API key or instance API key
            api_key = kwargs.get('api_key', self.api_key)
            if not api_key:
                self.logger.debug("No Anthropic API key available for model listing")
                return []

            base_url = kwargs.get('base_url') or self.base_url or os.getenv("ANTHROPIC_BASE_URL")
            models_url = "https://api.anthropic.com/v1/models"
            if isinstance(base_url, str) and base_url.strip():
                models_url = f"{base_url.strip().rstrip('/')}/models"

            # Make API call to list models
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }

            response = httpx.get(
                models_url,
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                self.logger.debug(f"Retrieved {len(models)} models from Anthropic API")
                models = sorted(models, reverse=True)  # Latest models first

                # Apply new capability filtering if provided
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
                self.logger.warning(f"Anthropic API returned status {response.status_code}")
                return []

        except Exception as e:
            self.logger.debug(f"Failed to fetch Anthropic models from API: {e}")
            return []
