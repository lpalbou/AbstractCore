"""
LM Studio provider implementation (OpenAI-compatible API).
"""

import httpx
import json
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
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType


class LMStudioProvider(BaseProvider):
    """LM Studio provider using OpenAI-compatible API"""

    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1", **kwargs):
        super().__init__(model, **kwargs)

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        self.base_url = base_url.rstrip('/')

        # Ensure timeout is properly set and not None
        timeout_value = getattr(self, '_timeout', None)
        if timeout_value is None:
            timeout_value = 300.0

        try:
            self.client = httpx.Client(timeout=timeout_value)
        except Exception as e:
            # Fallback with default timeout if client creation fails
            try:
                self.client = httpx.Client(timeout=300.0)
            except Exception:
                raise RuntimeError(f"Failed to create HTTP client for LMStudio: {e}")

        # Validate model exists in LMStudio
        self._validate_model()

    def _validate_model(self):
        """Validate that the model exists in LMStudio"""
        try:
            # Use base_url as-is (should include /v1) for model discovery
            available_models = self.list_available_models(base_url=self.base_url)
            if available_models and self.model not in available_models:
                error_message = format_model_error("LMStudio", self.model, available_models)
                raise ModelNotFoundError(error_message)
        except httpx.ConnectError:
            # LMStudio not running - will fail later when trying to generate
            if hasattr(self, 'logger'):
                self.logger.debug(f"LMStudio server not accessible at {self.base_url} - model validation skipped")
            pass
        except ModelNotFoundError:
            # Re-raise model not found errors
            raise
        except Exception as e:
            # Other errors (like timeout, None type errors) - continue, will fail later if needed
            if hasattr(self, 'logger'):
                self.logger.debug(f"Model validation failed with error: {e} - continuing anyway")
            pass

    def unload(self) -> None:
        """
        Close HTTP client connection.

        Note: LMStudio manages model memory automatically using TTL (time-to-live)
        and auto-evict features. There is no explicit API to unload models.
        Models will be automatically unloaded after the configured TTL expires.

        This method only closes the HTTP client connection for cleanup.
        """
        try:
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

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          execute_tools: Optional[bool] = None,
                          tool_call_tags: Optional[str] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using LM Studio"""

        # Build messages for chat completions with tool support
        chat_messages = []

        # Add tools to system prompt if provided
        enhanced_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            tool_prompt = self.tool_handler.format_tools_prompt(tools)
            if enhanced_system_prompt:
                enhanced_system_prompt += f"\n\n{tool_prompt}"
            else:
                enhanced_system_prompt = tool_prompt

        # Add system message if provided
        if enhanced_system_prompt:
            chat_messages.append({
                "role": "system",
                "content": enhanced_system_prompt
            })

        # Add conversation history
        if messages:
            chat_messages.extend(messages)

        # Add current prompt
        chat_messages.append({
            "role": "user",
            "content": prompt
        })

        # Build request payload using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(generation_kwargs)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": max_output_tokens,  # LMStudio uses max_tokens for output tokens
            "top_p": kwargs.get("top_p", 0.9),
        }

        if stream:
            # Return streaming response - BaseProvider will handle tag rewriting via UnifiedStreamProcessor
            return self._stream_generate(payload)
        else:
            response = self._single_generate(payload)

            # Execute tools if enabled and tools are present
            if self.execute_tools and tools and self.tool_handler.supports_prompted and response.content:
                response = self._handle_prompted_tool_execution(response, tools, execute_tools)

            return response

    def _single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Generate single response"""
        try:
            # Ensure client is available
            if not hasattr(self, 'client') or self.client is None:
                raise ProviderAPIError("HTTP client not initialized")

            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()

            # Extract response from OpenAI format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason", "stop")
            else:
                content = "No response generated"
                finish_reason = "error"

            # Extract usage info
            usage = result.get("usage", {})

            return GenerateResponse(
                content=content,
                model=self.model,
                finish_reason=finish_reason,
                raw_response=result,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            )

        except AttributeError as e:
            # Handle None type errors specifically
            if "'NoneType'" in str(e):
                raise ProviderAPIError(f"LMStudio provider not properly initialized: {str(e)}")
            else:
                raise ProviderAPIError(f"LMStudio configuration error: {str(e)}")
        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model' in error_str) and ('not found' in error_str):
                # Model not found - show available models
                try:
                    available_models = self.list_available_models(base_url=self.base_url)
                    error_message = format_model_error("LMStudio", self.model, available_models)
                    raise ModelNotFoundError(error_message)
                except Exception:
                    # If model discovery also fails, provide a generic error
                    raise ModelNotFoundError(f"Model '{self.model}' not found in LMStudio and could not fetch available models")
            else:
                raise ProviderAPIError(f"LMStudio API error: {str(e)}")

    def _stream_generate(self, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response"""
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()

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

                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    choice = chunk["choices"][0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    finish_reason = choice.get("finish_reason")

                                    yield GenerateResponse(
                                        content=content,
                                        model=self.model,
                                        finish_reason=finish_reason,
                                        raw_response=chunk
                                    )

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def get_capabilities(self) -> List[str]:
        """Get LM Studio capabilities"""
        return ["streaming", "chat", "tools"]

    def validate_config(self) -> bool:
        """Validate LM Studio connection"""
        try:
            response = self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except:
            return False

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for LMStudio API"""
        # For LMStudio (OpenAI-compatible), max_tokens is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)

    def _update_http_client_timeout(self) -> None:
        """Update HTTP client timeout when timeout is changed."""
        if hasattr(self, 'client') and self.client is not None:
            try:
                # Create new client with updated timeout
                self.client.close()

                # Ensure timeout is not None
                timeout_value = getattr(self, '_timeout', None)
                if timeout_value is None:
                    timeout_value = 300.0

                self.client = httpx.Client(timeout=timeout_value)
            except Exception as e:
                # Log error but don't fail - timeout update is not critical
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Failed to update HTTP client timeout: {e}")
                # Try to create a new client with default timeout
                try:
                    self.client = httpx.Client(timeout=300.0)
                except Exception:
                    pass  # Best effort - don't fail the operation



    def list_available_models(self, **kwargs) -> List[str]:
        """List available models from LMStudio server."""
        try:
            # Use provided base_url or fall back to instance base_url
            base_url = kwargs.get('base_url', self.base_url)

            response = self.client.get(f"{base_url}/models", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data.get("data", [])]
                return sorted(models)
            else:
                self.logger.warning(f"LMStudio API returned status {response.status_code}")
                return []
        except Exception as e:
            self.logger.warning(f"Failed to list LMStudio models: {e}")
            return []

    def embed(self, input_text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings using LMStudio's OpenAI-compatible embedding API.
        
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
            
            # Call LMStudio's embeddings API (OpenAI-compatible)
            response = self.client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # LMStudio returns OpenAI-compatible format, so we can return it directly
            result = response.json()
            
            # Ensure the model field uses our provider-prefixed format
            result["model"] = self.model
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise ProviderAPIError(f"LMStudio embedding error: {str(e)}")
