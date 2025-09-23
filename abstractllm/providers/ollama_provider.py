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
from ..exceptions import ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error
from ..tools import UniversalToolHandler, ToolDefinition, execute_tools
from ..events import EventType


class OllamaProvider(BaseProvider):
    """Ollama provider for local models with full integration"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

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

            # Add conversation history
            payload["messages"].extend(messages)

            # Add current prompt as user message
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
            return self._stream_generate(endpoint, payload, tools)
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

            # Handle tool execution for prompted models
            if tools and self.tool_handler.supports_prompted and content:
                generate_response = self._handle_tool_execution(generate_response, tools)

            return generate_response

        except Exception as e:
            # Check for model not found errors
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model not found' in error_str or
                'pull model' in error_str or 'no such model' in error_str):
                # Model not found - provide helpful error
                available_models = get_available_models("ollama", base_url=self.base_url)
                error_message = format_model_error("Ollama", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                return GenerateResponse(
                    content=f"Error: {str(e)}",
                    model=self.model,
                    finish_reason="error"
                )

    def _stream_generate(self, endpoint: str, payload: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None) -> Iterator[GenerateResponse]:
        """Generate streaming response"""
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}{endpoint}",
                json=payload
            ) as response:
                response.raise_for_status()

                # Collect full response for tool processing
                full_content = ""

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

                            chunk_response = GenerateResponse(
                                content=content,
                                model=self.model,
                                finish_reason="stop" if done else None,
                                raw_response=chunk
                            )

                            # For streaming, only handle tools on the final chunk
                            if done and tools and self.tool_handler.supports_prompted and full_content:
                                # Create a complete response for tool processing
                                complete_response = GenerateResponse(
                                    content=full_content,
                                    model=self.model,
                                    finish_reason="stop",
                                    raw_response=chunk
                                )

                                # Handle tool execution and yield tool results as additional chunks
                                final_response = self._handle_tool_execution(complete_response, tools)

                                # If tools were executed, yield the tool results as final chunk
                                if final_response.content != full_content:
                                    yield GenerateResponse(
                                        content=final_response.content[len(full_content):],
                                        model=self.model,
                                        finish_reason="stop",
                                        raw_response=chunk
                                    )
                                else:
                                    yield chunk_response
                            else:
                                yield chunk_response

                            if done:
                                break

                        except json.JSONDecodeError:
                            continue

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
            content=response.content + results_text,
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