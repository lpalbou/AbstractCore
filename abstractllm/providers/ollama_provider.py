"""
Ollama provider implementation.
"""

import json
import httpx
import time
from typing import List, Dict, Any, Optional, Union, Iterator
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error


class OllamaProvider(BaseProvider):
    """Ollama provider for local models with full integration"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)

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
        """Internal generation with Ollama"""

        # Build request payload
        payload = {
            "model": self.model,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 2048),
            }
        }

        # Use chat format if messages provided
        if messages:
            payload["messages"] = []

            # Add system message if provided
            if system_prompt:
                payload["messages"].append({
                    "role": "system",
                    "content": system_prompt
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
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload["prompt"] = full_prompt
            endpoint = "/api/generate"

        if stream:
            return self._stream_generate(endpoint, payload)
        else:
            return self._single_generate(endpoint, payload)

    def _single_generate(self, endpoint: str, payload: Dict[str, Any]) -> GenerateResponse:
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

            return GenerateResponse(
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

    def _stream_generate(self, endpoint: str, payload: Dict[str, Any]) -> Iterator[GenerateResponse]:
        """Generate streaming response"""
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}{endpoint}",
                json=payload
            ) as response:
                response.raise_for_status()

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

                            yield GenerateResponse(
                                content=content,
                                model=self.model,
                                finish_reason="stop" if done else None,
                                raw_response=chunk
                            )

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

    def get_capabilities(self) -> List[str]:
        """Get Ollama capabilities"""
        return ["streaming", "chat"]

    def validate_config(self) -> bool:
        """Validate Ollama connection"""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False