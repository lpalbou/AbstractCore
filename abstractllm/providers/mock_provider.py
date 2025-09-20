"""
Mock provider for testing.
"""

from typing import List, Dict, Any, Optional, Union, Iterator
from ..core.interface import AbstractLLMInterface
from ..core.types import GenerateResponse


class MockProvider(AbstractLLMInterface):
    """Mock provider for testing purposes"""

    def __init__(self, model: str = "mock-model", **kwargs):
        super().__init__(model, **kwargs)

    def generate(self,
                prompt: str,
                messages: Optional[List[Dict[str, str]]] = None,
                system_prompt: Optional[str] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
                stream: bool = False,
                **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Mock generation"""

        if stream:
            return self._stream_response(prompt)
        else:
            return self._single_response(prompt)

    def _single_response(self, prompt: str) -> GenerateResponse:
        """Generate single mock response"""
        return GenerateResponse(
            content=f"Mock response to: {prompt}",
            model=self.model,
            finish_reason="stop",
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "total_tokens": len(prompt.split()) + 10}
        )

    def _stream_response(self, prompt: str) -> Iterator[GenerateResponse]:
        """Generate streaming mock responses"""
        words = f"Mock response to: {prompt}".split()
        for i, word in enumerate(words):
            yield GenerateResponse(
                content=word + (" " if i < len(words) - 1 else ""),
                model=self.model,
                finish_reason="stop" if i == len(words) - 1 else None
            )

    def get_capabilities(self) -> List[str]:
        """Get mock capabilities"""
        return ["tools", "streaming", "vision"]