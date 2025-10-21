"""
Mock provider for testing purposes.
"""

from typing import List, Dict, Any, Optional, Union, Iterator, Type

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

from .base import BaseProvider
from ..core.types import GenerateResponse


class MockProvider(BaseProvider):
    """Simple mock provider for testing core functionality."""

    def __init__(self, model: str = "mock-model", **kwargs):
        super().__init__(model, **kwargs)
        
        # Handle timeout parameter for mock provider
        self._handle_timeout_parameter(kwargs)
        
        # Mock provider uses prompted strategy for structured output
        self.model_capabilities = {"structured_output": "prompted"}

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
        """Mock generation implementation"""

        if stream:
            return self._stream_response(prompt)
        else:
            return self._single_response(prompt, response_model)

    def _single_response(self, prompt: str, response_model: Optional[Type[BaseModel]] = None) -> GenerateResponse:
        """Generate single mock response"""
        import time
        
        # Simulate generation time (10-100ms for mock)
        start_time = time.time()
        time.sleep(0.01 + (len(prompt) % 10) * 0.01)  # 10-100ms based on prompt length
        gen_time = round((time.time() - start_time) * 1000, 1)

        if response_model and PYDANTIC_AVAILABLE:
            # Generate valid JSON for structured output
            content = self._generate_mock_json(response_model)
        else:
            content = f"Mock response to: {prompt}"

        return GenerateResponse(
            content=content,
            model=self.model,
            finish_reason="stop",
            usage=self._calculate_mock_usage(prompt, content),
            gen_time=gen_time
        )

    def _calculate_mock_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate mock token usage using centralized token utilities."""
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

    def _stream_response(self, prompt: str) -> Iterator[GenerateResponse]:
        """Generate streaming mock responses"""
        words = f"Mock response to: {prompt}".split()
        for i, word in enumerate(words):
            yield GenerateResponse(
                content=word + (" " if i < len(words) - 1 else ""),
                model=self.model,
                finish_reason="stop" if i == len(words) - 1 else None
            )

    def _generate_mock_json(self, model_class: Type[BaseModel]) -> str:
        """Generate valid JSON for Pydantic model"""
        import json

        # Create mock data based on field types
        mock_data = {}
        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation

            # Handle basic types
            if field_type == str:
                mock_data[field_name] = f"mock_{field_name}"
            elif field_type == int:
                mock_data[field_name] = 42
            elif field_type == float:
                mock_data[field_name] = 3.14
            elif field_type == bool:
                mock_data[field_name] = True
            else:
                # For complex types, provide reasonable defaults
                mock_data[field_name] = f"mock_{field_name}"

        return json.dumps(mock_data)

    def _handle_timeout_parameter(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle timeout parameter for Mock provider.
        
        Mock provider simulates responses instantly, so timeout parameters
        don't apply. If a non-None timeout is provided, it's accepted but
        has no effect on mock generation.
        
        Args:
            kwargs: Initialization kwargs that may contain timeout
        """
        timeout_value = kwargs.get('timeout')
        if timeout_value is not None:
            # For mock provider, we accept timeout but it has no effect
            # No warning needed since this is for testing
            self._timeout = timeout_value
        else:
            # Keep None value 
            self._timeout = None

    def _update_http_client_timeout(self) -> None:
        """
        Mock provider doesn't use HTTP clients.
        Timeout changes have no effect on mock responses.
        """
        # No-op for mock provider - no HTTP clients used
        pass

    def get_capabilities(self) -> List[str]:
        """Get mock capabilities"""
        return ["tools", "streaming", "vision"]

    def validate_config(self) -> bool:
        """Validate mock provider configuration"""
        return True

    def list_available_models(self, **kwargs) -> List[str]:
        """List available mock models for testing."""
        return [
            "mock-model",
            "mock-chat-model",
            "mock-streaming-model",
            "mock-structured-model",
            "mock-vision-model"
        ]