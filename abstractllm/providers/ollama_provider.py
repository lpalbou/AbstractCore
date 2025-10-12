"""
Ollama provider implementation.
"""

import json
import httpx
import time
from typing import List, Dict, Any, Optional, Union, Iterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error
from ..tools import UniversalToolHandler, ToolDefinition, execute_tools
from ..events import EventType


class OllamaProvider(BaseProvider):
    """Ollama provider for local models with full integration"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=self._timeout)

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

    def unload(self) -> None:
        """
        Unload the model from Ollama server memory.

        Sends a request with keep_alive=0 to immediately unload the model
        from the Ollama server, freeing server-side memory.
        """
        try:
            # Send a minimal generate request with keep_alive=0 to unload
            payload = {
                "model": self.model,
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

        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

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
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with Ollama"""

        # Handle tools for prompted models
        effective_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            tool_prompt = self.tool_handler.format_tools_prompt(tools)
            if effective_system_prompt:
                effective_system_prompt = f"{effective_system_prompt}\n\n{tool_prompt}"
            else:
                effective_system_prompt = tool_prompt

        # Build request payload using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        payload = {
            "model": self.model,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": max_output_tokens,  # Ollama uses num_predict for max output tokens
            }
        }

        # Add structured output support (Ollama native JSON schema)
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            payload["format"] = json_schema

        # Use chat format if messages provided
        if messages:
            payload["messages"] = []

            # Add system message if provided
            if effective_system_prompt:
                payload["messages"].append({
                    "role": "system",
                    "content": effective_system_prompt
                })

            # Add conversation history (converted to Ollama-compatible format)
            converted_messages = self._convert_messages_for_ollama(messages)
            payload["messages"].extend(converted_messages)

            # Add current prompt as user message (only if non-empty)
            # When using messages array, prompt should be empty or already in messages
            if prompt and prompt.strip():
                payload["messages"].append({
                    "role": "user",
                    "content": prompt
                })

            endpoint = "/api/chat"
        else:
            # Use generate format for single prompt
            full_prompt = prompt
            if effective_system_prompt:
                full_prompt = f"{effective_system_prompt}\n\n{prompt}"

            payload["prompt"] = full_prompt
            endpoint = "/api/generate"

        if stream:
            return self._stream_generate(endpoint, payload, tools, kwargs.get('tool_call_tags'))
        else:
            return self._single_generate(endpoint, payload, tools)

    def _single_generate(self, endpoint: str, payload: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> GenerateResponse:
        """Generate single response"""
        try:
            response = self.client.post(
                f"{self.base_url}{endpoint}",
                json=payload
            )
            response.raise_for_status()

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
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            )

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
            else:
                return GenerateResponse(
                    content=f"Error: {str(e)}",
                    model=self.model,
                    finish_reason="error"
                )

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
            tool_calls=tool_call_response.tool_calls
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
        """List available models from Ollama server."""
        try:
            # Use provided base_url or fall back to instance base_url
            base_url = kwargs.get('base_url', self.base_url)

            response = self.client.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return sorted(models)
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
                
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = max(1, len(text) // 4)
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