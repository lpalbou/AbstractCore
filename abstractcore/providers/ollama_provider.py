"""
Ollama provider implementation.
"""

import json
import os
import httpx
import time
import warnings
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider, ThinkingControlHandling
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error, format_provider_error
from ..tools import UniversalToolHandler, ToolDefinition, execute_tools
from ..events import EventType


class OllamaProvider(BaseProvider):
    """Ollama provider for local models with full integration"""

    def __init__(self, model: str = "qwen3:4b-instruct-2507-q4_K_M", base_url: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "ollama"

        # Base URL priority: parameter > OLLAMA_BASE_URL > OLLAMA_HOST > default
        self.base_url = (
            base_url or
            os.getenv("OLLAMA_BASE_URL") or
            os.getenv("OLLAMA_HOST") or
            "http://localhost:11434"
        ).rstrip('/')
        self.client = httpx.Client(timeout=self._timeout)
        self._async_client = None  # Lazy-loaded async client

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

    @property
    def async_client(self):
        """Lazy-load async HTTP client for native async operations."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout
            )
        return self._async_client

    def unload_model(self, model_name: str) -> None:
        """
        Unload the model from Ollama server memory.

        Sends a request with keep_alive=0 to immediately unload the model
        from the Ollama server, freeing server-side memory.
        """
        try:
            target_model = model_name.strip() if isinstance(model_name, str) and model_name.strip() else self.model

            # Send a minimal generate request with keep_alive=0 to unload
            payload = {
                "model": target_model,
                "prompt": "",  # Minimal prompt
                "stream": False,
                "keep_alive": 0  # Immediately unload after this request
            }

            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()

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
                    # No running loop, close synchronously
                    import asyncio
                    asyncio.run(self._async_client.aclose())

        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

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
        # Ollama exposes "thinking" via the request field `think`.
        #
        # Observed semantics:
        # - Many models accept a boolean (`think: true|false`).
        # - Some models/serving stacks also accept effort strings (`think: "low"|"medium"|"high"`).
        #
        # IMPORTANT: `thinking="off"` should translate to a *real disable* (`think: false`) whenever
        # possible. Only models that cannot disable traces (e.g. GPT-OSS/Harmony) should degrade to
        # a low-effort mode.
        if enabled is None and level is None:
            return kwargs, ThinkingControlHandling()

        new_kwargs = dict(kwargs)

        def _is_harmony_model() -> bool:
            msg_fmt = str((self.architecture_config or {}).get("message_format") or "").strip().lower()
            resp_fmt = str((self.model_capabilities or {}).get("response_format") or "").strip().lower()
            return msg_fmt == "harmony" or resp_fmt == "harmony"

        if _is_harmony_model():
            # Harmony (GPT-OSS): supports string effort levels, but cannot fully disable traces.
            if level is not None:
                new_kwargs["think"] = level
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)
            if enabled is False:
                new_kwargs["think"] = "low"
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)
            if enabled is True:
                new_kwargs["think"] = "medium"
                return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=True)
            return kwargs, ThinkingControlHandling()

        if enabled is False:
            new_kwargs["think"] = False
            new_kwargs.pop("think_level", None)
            return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=False)

        # Default mode (most models): boolean thinking enable/disable.
        #
        # Ollama's official "thinking" docs document boolean control for most models, and reserve
        # effort strings for GPT-OSS. Preserve our unified `thinking="low|medium|high"` API by:
        # - enabling thinking (think=true)
        # - carrying the requested level in a private kwarg for output-budget heuristics
        if enabled is True or level is not None:
            new_kwargs["think"] = True
            if level is not None:
                new_kwargs["think_level"] = level
            return new_kwargs, ThinkingControlHandling(handled_enable_disable=True, handled_level=False)

        return kwargs, ThinkingControlHandling()

    def _convert_messages_for_ollama(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI messages to Ollama-compatible format

        Ollama only supports roles: ["system", "user", "assistant"]
        - Converts role: "tool" to role: "user" with markers
        - Removes tool_calls from assistant messages
        - Preserves all other message types
        """
        if not messages:
            return []

        converted = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role")

            if role == "tool":
                # Convert tool message to user message with clear markers
                tool_content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id", "unknown")
                converted.append({
                    "role": "user",
                    "content": f"[TOOL RESULT {tool_call_id}]: {tool_content}"
                })
            elif role == "assistant" and msg.get("tool_calls"):
                # Remove tool_calls from assistant messages (Ollama doesn't support them)
                converted.append({
                    "role": "assistant",
                    "content": msg.get("content", "")
                })
            else:
                # Keep supported roles as-is (system, user, assistant without tool_calls)
                converted.append(msg.copy())

        return converted

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          media_metadata: Optional[List[Dict[str, Any]]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with Ollama"""

        # Handle tools for prompted models
        final_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt = f"{final_system_prompt}\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Build request payload using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        # Some reasoning templates (notably Qwen3.x on Ollama) can emit substantial "thinking"
        # output before producing any final answer tokens. When the caller enables thinking but
        # also sets a very small `max_output_tokens`, the model may exhaust the token budget in
        # the reasoning channel and return an empty final `content`.
        #
        # To keep `thinking=` usable out-of-the-box, enforce a small minimum `num_predict` when
        # thinking is enabled. This is best-effort and only triggers for *very* small budgets.
        think_value = generation_kwargs.get("think")
        think_level = generation_kwargs.get("think_level")
        num_predict = int(max_output_tokens)
        if think_value not in (None, False):
            # NOTE: Qwen3.5 (and similar hybrid reasoning templates) can spend a lot of tokens in the
            # reasoning channel before producing any final answer tokens. Empirically, some sizes
            # (notably 4B) can exhaust a 1024-token budget entirely in thinking for "low", yielding
            # empty final content.
            #
            # Keep `thinking=` usable with small `max_output_tokens` by enforcing a slightly larger
            # minimum `num_predict` when thinking is enabled.
            min_map = {"minimal": 1024, "low": 2048, "medium": 4096, "high": 8192}
            requested_level = None
            if isinstance(think_level, str) and think_level.strip():
                requested_level = think_level.strip().lower()
            elif isinstance(think_value, str) and think_value.strip():
                requested_level = think_value.strip().lower()
            min_predict = min_map.get(requested_level, 2048)
            if num_predict < int(min_predict):
                warnings.warn(
                    f"max_output_tokens={num_predict} is likely too small when thinking is enabled for Ollama; "
                    f"bumping options.num_predict to {min_predict} to avoid empty final content.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                num_predict = int(min_predict)

        payload = {
            "model": self.model,
            "stream": stream,
            "options": {
                "temperature": generation_kwargs.get("temperature", self.temperature),
                "num_predict": num_predict,  # Ollama uses num_predict for max output tokens
            }
        }

        # Add seed if provided (Ollama supports seed for deterministic outputs)
        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            payload["options"]["seed"] = seed_value

        # Unified thinking/reasoning control (Ollama-native).
        if think_value is not None:
            payload["think"] = think_value

        # Add structured output support (Ollama native JSON schema)
        # Ollama accepts the full JSON schema in the "format" parameter
        # This provides server-side guaranteed schema compliance
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["format"] = json_schema  # Pass the full schema, not just "json"

        # Use chat format by default (recommended by Ollama docs), especially when tools are present.
        #
        # Qwen3.x reasoning templates also behave more predictably on `/api/chat` when thinking is enabled
        # (some Ollama versions will emit a populated `thinking` field but an empty `response`/`content`
        # in `/api/generate` for the same request).
        use_chat_format = (
            tools is not None
            or messages is not None
            or think_value is not None
        )

        # Media enrichment metadata (populated only when we route media through LocalMediaHandler).
        media_enrichment = None

        if use_chat_format:
            payload["messages"] = []

            # Add system message if provided
            if final_system_prompt:
                payload["messages"].append({
                    "role": "system",
                    "content": final_system_prompt
                })

            # Add conversation history (converted to Ollama-compatible format)
            if messages:
                converted_messages = self._convert_messages_for_ollama(messages)
                payload["messages"].extend(converted_messages)

            # Handle media content regardless of prompt (media can be used with messages too)
            if media:
                # Get the text to combine with media
                user_message_text = prompt.strip() if prompt else ""
                try:
                    from ..media.handlers import LocalMediaHandler
                    media_handler = LocalMediaHandler("ollama", self.model_capabilities, model_name=self.model)

                    # Create multimodal message combining text and media
                    multimodal_message = media_handler.create_multimodal_message(user_message_text, media)
                    media_enrichment = getattr(media_handler, "media_enrichment", None)

                    # For local providers, we might get a string (embedded text) or dict (structured)
                    if isinstance(multimodal_message, str):
                        payload["messages"].append({
                            "role": "user",
                            "content": multimodal_message
                        })
                    else:
                        payload["messages"].append(multimodal_message)
                except ImportError:
                    self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
                    if user_message_text:
                        payload["messages"].append({
                            "role": "user",
                            "content": user_message_text
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to process media content: {e}")
                    if user_message_text:
                        payload["messages"].append({
                            "role": "user",
                            "content": user_message_text
                        })

            # Add prompt as separate message if provided (for backward compatibility)
            elif prompt and prompt.strip():
                payload["messages"].append({
                    "role": "user",
                    "content": prompt
                })

            endpoint = "/api/chat"
        else:
            # Use generate format for single prompt (legacy fallback)
            full_prompt = prompt
            if final_system_prompt:
                full_prompt = f"{final_system_prompt}\n\n{prompt}"

            payload["prompt"] = full_prompt
            endpoint = "/api/generate"

        if stream:
            return self._stream_generate(endpoint, payload, tools, kwargs.get('tool_call_tags'))
        else:
            response = self._single_generate(endpoint, payload, tools, media_metadata)
            if media_enrichment:
                from ..media.enrichment import merge_enrichment_metadata

                response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)
            return response

    def _single_generate(self, endpoint: str, payload: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None, media_metadata: Optional[List[Dict[str, Any]]] = None) -> GenerateResponse:
        """Generate single response"""
        try:
            # Track generation time
            start_time = time.time()
            response = self.client.post(
                f"{self.base_url}{endpoint}",
                json=payload
            )
            response.raise_for_status()
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            # Extract content based on endpoint
            if endpoint == "/api/chat":
                content = result.get("message", {}).get("content", "")
            else:
                content = result.get("response", "")

            # Create initial response
            generate_response = GenerateResponse(
                content=content,
                model=self.model,
                finish_reason="stop",
                raw_response=result,
                usage={
                    "input_tokens": result.get("prompt_eval_count", 0),
                    "output_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                    # Keep legacy keys for backward compatibility
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0)
                },
                gen_time=gen_time
            )

            # Runtime observability: capture the exact HTTP JSON payload we sent to Ollama.
            if not generate_response.metadata:
                generate_response.metadata = {}
            generate_response.metadata["_provider_request"] = {
                "url": f"{self.base_url}{endpoint}",
                "payload": payload,
            }

            # Capture Ollama thinking output (if present) into canonical metadata["reasoning"].
            thinking_text = None
            try:
                if endpoint == "/api/chat":
                    msg = result.get("message") if isinstance(result, dict) else None
                    msg = msg if isinstance(msg, dict) else {}
                    thinking_text = msg.get("thinking") or msg.get("reasoning")
                else:
                    thinking_text = result.get("thinking") or result.get("reasoning")
            except Exception:
                thinking_text = None
            if isinstance(thinking_text, str) and thinking_text.strip():
                generate_response.metadata.setdefault("reasoning", thinking_text.strip())
            
            # Attach media metadata if available
            if media_metadata:
                if not generate_response.metadata:
                    generate_response.metadata = {}
                generate_response.metadata['media_metadata'] = media_metadata

            # Execute tools if enabled and tools are present
            if self.execute_tools and tools and self.tool_handler.supports_prompted and content:
                return self._handle_tool_execution(generate_response, tools)

            return generate_response

        except Exception as e:
            # Check for model not found errors
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model not found' in error_str or
                'pull model' in error_str or 'no such model' in error_str):
                # Model not found - provide helpful error
                available_models = self.list_available_models(base_url=self.base_url)
                error_message = format_model_error("Ollama", self.model, available_models)
                raise ModelNotFoundError(error_message)
            # Let BaseProvider normalize (timeouts/connectivity/etc.) consistently.
            raise

    def _stream_generate(self, endpoint: str, payload: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None, tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Generate streaming response with tool tag rewriting support"""
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}{endpoint}",
                json=payload
            ) as response:
                response.raise_for_status()

                # Collect full response for tool processing
                full_content = ""
                
                # Initialize tool tag rewriter if needed
                rewriter = None
                buffer = ""
                if tool_call_tags:
                    try:
                        from ..tools.tag_rewriter import create_tag_rewriter
                        rewriter = create_tag_rewriter(tool_call_tags)
                    except ImportError:
                        pass

                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)

                            # Extract content based on endpoint
                            if endpoint == "/api/chat":
                                content = chunk.get("message", {}).get("content", "")
                            else:
                                content = chunk.get("response", "")

                            done = chunk.get("done", False)
                            full_content += content

                            # Apply tool tag rewriting if enabled
                            if rewriter and content:
                                rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                                content = rewritten_content

                            metadata: Dict[str, Any] = {
                                "_provider_request": {
                                    "url": f"{self.base_url}{endpoint}",
                                    "payload": payload,
                                }
                            }
                            # Capture incremental thinking output when present.
                            try:
                                if endpoint == "/api/chat":
                                    msg = chunk.get("message") if isinstance(chunk, dict) else None
                                    msg = msg if isinstance(msg, dict) else {}
                                    thinking_text = msg.get("thinking") or msg.get("reasoning")
                                else:
                                    thinking_text = chunk.get("thinking") or chunk.get("reasoning")
                            except Exception:
                                thinking_text = None
                            if isinstance(thinking_text, str) and thinking_text.strip():
                                metadata.setdefault("reasoning", thinking_text.strip())

                            chunk_response = GenerateResponse(
                                content=content,
                                model=self.model,
                                finish_reason="stop" if done else None,
                                raw_response=chunk,
                                metadata=metadata,
                            )

                            yield chunk_response

                            if done:
                                break

                        except json.JSONDecodeError:
                            continue

                # Execute tools if enabled and we have collected content
                if self.execute_tools and tools and self.tool_handler.supports_prompted and full_content:
                    # Create complete response for tool processing
                    complete_response = GenerateResponse(
                        content=full_content,
                        model=self.model,
                        finish_reason="stop"
                    )

                    # Execute tools and yield results
                    final_response = self._handle_tool_execution(complete_response, tools)

                    # If tools were executed, yield the tool results
                    if final_response.content != full_content:
                        tool_results_content = final_response.content[len(full_content):]
                        yield GenerateResponse(
                            content=tool_results_content,
                            model=self.model,
                            finish_reason="stop"
                        )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    async def _agenerate_internal(self,
                                   prompt: str,
                                   messages: Optional[List[Dict]],
                                   system_prompt: Optional[str],
                                   tools: Optional[List],
                                   media: Optional[List],
                                   stream: bool,
                                   **kwargs):
        """Native async implementation using httpx.AsyncClient - 3-10x faster for batch operations."""
        # Handle tools for prompted models
        final_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt = f"{final_system_prompt}\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Build request payload (same logic as sync)
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        response_model = kwargs.get('response_model')

        payload = {
            "model": self.model,
            "stream": stream,
            "options": {
                "temperature": generation_kwargs.get("temperature", self.temperature),
                "num_predict": max_output_tokens,
            }
        }

        seed_value = generation_kwargs.get("seed")
        if seed_value is not None:
            payload["options"]["seed"] = seed_value

        # Add structured output support
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["format"] = json_schema

        # Use chat format
        use_chat_format = tools is not None or messages is not None

        if use_chat_format:
            payload["messages"] = []

            if final_system_prompt:
                payload["messages"].append({
                    "role": "system",
                    "content": final_system_prompt
                })

            if messages:
                converted_messages = self._convert_messages_for_ollama(messages)
                payload["messages"].extend(converted_messages)

            if media:
                user_message_text = prompt.strip() if prompt else ""
                try:
                    from ..media.handlers import LocalMediaHandler
                    media_handler = LocalMediaHandler("ollama", self.model_capabilities, model_name=self.model)
                    multimodal_message = media_handler.create_multimodal_message(user_message_text, media)

                    if isinstance(multimodal_message, str):
                        payload["messages"].append({"role": "user", "content": multimodal_message})
                    else:
                        payload["messages"].append(multimodal_message)
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Failed to process media: {e}")
                    if user_message_text:
                        payload["messages"].append({"role": "user", "content": user_message_text})

            elif prompt and prompt.strip():
                payload["messages"].append({"role": "user", "content": prompt})

            endpoint = "/api/chat"
        else:
            full_prompt = prompt
            if final_system_prompt:
                full_prompt = f"{final_system_prompt}\n\n{prompt}"
            payload["prompt"] = full_prompt
            endpoint = "/api/generate"

        if stream:
            return self._async_stream_generate(endpoint, payload, tools, kwargs.get('tool_call_tags'))
        else:
            return await self._async_single_generate(endpoint, payload, tools, kwargs.get('media_metadata'))

    async def _async_single_generate(self, endpoint: str, payload: Dict[str, Any],
                                      tools: Optional[List[Dict[str, Any]]] = None,
                                      media_metadata: Optional[List[Dict[str, Any]]] = None) -> GenerateResponse:
        """Native async single response generation."""
        try:
            start_time = time.time()
            response = await self.async_client.post(endpoint, json=payload)
            response.raise_for_status()
            gen_time = round((time.time() - start_time) * 1000, 1)

            result = response.json()

            if endpoint == "/api/chat":
                content = result.get("message", {}).get("content", "")
            else:
                content = result.get("response", "")

            generate_response = GenerateResponse(
                content=content,
                model=self.model,
                finish_reason="stop",
                raw_response=result,
                usage={
                    "input_tokens": result.get("prompt_eval_count", 0),
                    "output_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0)
                },
                gen_time=gen_time
            )

            if media_metadata:
                if not generate_response.metadata:
                    generate_response.metadata = {}
                generate_response.metadata['media_metadata'] = media_metadata

            if self.execute_tools and tools and self.tool_handler.supports_prompted and content:
                return self._handle_tool_execution(generate_response, tools)

            return generate_response

        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str):
                available_models = self.list_available_models(base_url=self.base_url)
                error_message = format_model_error("Ollama", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                return GenerateResponse(
                    content=f"Error: {str(e)}",
                    model=self.model,
                    finish_reason="error"
                )

    async def _async_stream_generate(self, endpoint: str, payload: Dict[str, Any],
                                      tools: Optional[List[Dict[str, Any]]] = None,
                                      tool_call_tags: Optional[str] = None):
        """Native async streaming response generation."""
        try:
            async with self.async_client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()

                full_content = ""
                rewriter = None
                buffer = ""
                if tool_call_tags:
                    try:
                        from ..tools.tag_rewriter import create_tag_rewriter
                        rewriter = create_tag_rewriter(tool_call_tags)
                    except ImportError:
                        pass

                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)

                            if endpoint == "/api/chat":
                                content = chunk.get("message", {}).get("content", "")
                            else:
                                content = chunk.get("response", "")

                            done = chunk.get("done", False)
                            full_content += content

                            if rewriter and content:
                                rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                                content = rewritten_content

                            chunk_response = GenerateResponse(
                                content=content,
                                model=self.model,
                                finish_reason="stop" if done else None,
                                raw_response=chunk
                            )

                            yield chunk_response

                            if done:
                                break

                        except json.JSONDecodeError:
                            continue

                # Execute tools if enabled
                if self.execute_tools and tools and self.tool_handler.supports_prompted and full_content:
                    complete_response = GenerateResponse(
                        content=full_content,
                        model=self.model,
                        finish_reason="stop"
                    )

                    final_response = self._handle_tool_execution(complete_response, tools)

                    if final_response.content != full_content:
                        tool_results_content = final_response.content[len(full_content):]
                        yield GenerateResponse(
                            content=tool_results_content,
                            model=self.model,
                            finish_reason="stop"
                        )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _handle_tool_execution(self, response: GenerateResponse, tools: List[Dict[str, Any]]) -> GenerateResponse:
        """Handle tool execution for prompted models"""
        # Parse tool calls from response
        tool_call_response = self.tool_handler.parse_response(response.content, mode="prompted")

        if not tool_call_response.has_tool_calls():
            return response

        # Emit tool started event
        from ..events import emit_global
        event_data = {
            "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in tool_call_response.tool_calls],
            "model": self.model,
            "provider": self.__class__.__name__
        }
        emit_global(EventType.TOOL_STARTED, event_data, source=self.__class__.__name__)

        # Execute tools
        tool_results = execute_tools(tool_call_response.tool_calls)

        # Emit tool completed event
        emit_global(EventType.TOOL_COMPLETED, {
            "tool_results": [{"name": call.name, "success": result.success, "error": str(result.error) if result.error else None}
                           for call, result in zip(tool_call_response.tool_calls, tool_results)],
            "model": self.model,
            "provider": self.__class__.__name__
        }, source=self.__class__.__name__)

        # Format tool results and append to response
        results_text = "\n\nTool Results:\n"
        for result in tool_results:
            if result.success:
                results_text += f"- {result.output}\n"
            else:
                results_text += f"- Error: {result.error}\n"

        # Return updated response with tool results
        return GenerateResponse(
            content=tool_call_response.content + results_text,
            model=response.model,
            finish_reason=response.finish_reason,
            raw_response=response.raw_response,
            usage=response.usage,
            tool_calls=tool_call_response.tool_calls,
            metadata=response.metadata,
        )

    def get_capabilities(self) -> List[str]:
        """Get Ollama capabilities"""
        capabilities = ["streaming", "chat"]
        if self.tool_handler.supports_prompted:
            capabilities.append("tools")
        return capabilities

    def validate_config(self) -> bool:
        """Validate Ollama connection"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for Ollama API"""
        # For Ollama, num_predict is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout when timeout is changed."""
        if hasattr(self, 'client'):
            # Create new client with updated timeout
            self.client.close()
            self.client = httpx.Client(timeout=self._timeout)

    def list_available_models(self, **kwargs) -> List[str]:
        """
        List available models from Ollama server.

        Args:
            **kwargs: Optional parameters including:
                - base_url: Ollama server URL
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            # Use provided base_url or fall back to instance base_url
            base_url = kwargs.get('base_url', self.base_url)

            response = self.client.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
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
            else:
                self.logger.warning(f"Ollama API returned status {response.status_code}")
                return []
        except Exception as e:
            self.logger.warning(f"Failed to list Ollama models: {e}")
            return []

    def embed(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings using Ollama's embedding API.
        
        Args:
            input_text: Single string or list of strings to embed
            **kwargs: Additional parameters (currently unused)
            
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
            # Convert single string to list for uniform processing
            texts = [input_text] if isinstance(input_text, str) else input_text
            
            embeddings_data = []
            total_tokens = 0
            
            for idx, text in enumerate(texts):
                # Call Ollama's embeddings API
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                
                response = self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("embedding", [])
                
                # Use centralized token estimation for accuracy
                from ..utils.token_utils import TokenUtils
                estimated_tokens = TokenUtils.estimate_tokens(text, self.model)
                total_tokens += estimated_tokens
                
                embeddings_data.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": idx
                })
            
            return {
                "object": "list",
                "data": embeddings_data,
                "model": self.model,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProviderAPIError(f"Ollama embedding error: {str(e)}")
