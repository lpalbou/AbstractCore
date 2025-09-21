"""
LM Studio provider implementation (OpenAI-compatible API).
"""

import httpx
import json
from typing import List, Dict, Any, Optional, Union, Iterator
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error


class LMStudioProvider(BaseProvider):
    """LM Studio provider using OpenAI-compatible API"""

    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=120.0)

        # Validate model exists in LMStudio
        self._validate_model()

    def _validate_model(self):
        """Validate that the model exists in LMStudio"""
        try:
            # Remove /v1 from base_url for model discovery since it adds /v1/models
            discovery_base_url = self.base_url.replace("/v1", "")
            available_models = get_available_models("lmstudio", base_url=discovery_base_url)
            if available_models and self.model not in available_models:
                error_message = format_model_error("LMStudio", self.model, available_models)
                raise ModelNotFoundError(error_message)
        except httpx.ConnectError:
            # LMStudio not running - will fail later when trying to generate
            pass
        except ModelNotFoundError:
            # Re-raise model not found errors
            raise
        except Exception:
            # Other errors (like timeout) - continue, will fail later if needed
            pass

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using LM Studio"""

        # Build messages for chat completions
        chat_messages = []

        # Add system message if provided
        if system_prompt:
            chat_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add conversation history
        if messages:
            chat_messages.extend(messages)

        # Add current prompt
        chat_messages.append({
            "role": "user",
            "content": prompt
        })

        # Build request payload
        payload = {
            "model": self.model,
            "messages": chat_messages,
            "stream": stream,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.9),
        }

        if stream:
            return self._stream_generate(payload)
        else:
            return self._single_generate(payload)

    def _single_generate(self, payload: Dict[str, Any]) -> GenerateResponse:
        """Generate single response"""
        try:
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

        except Exception as e:
            error_str = str(e).lower()
            if ('404' in error_str or 'not found' in error_str or 'model' in error_str) and ('not found' in error_str):
                # Model not found - show available models
                available_models = get_available_models("lmstudio", base_url=self.base_url)
                error_message = format_model_error("LMStudio", self.model, available_models)
                raise ModelNotFoundError(error_message)
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