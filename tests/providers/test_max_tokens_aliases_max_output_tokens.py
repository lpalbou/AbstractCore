from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _DummyProvider(BaseProvider):
    """Minimal provider stub to validate BaseProvider argument normalization."""

    def __init__(self, model: str) -> None:
        super().__init__(model=model, enable_tracing=False)
        self.last_kwargs: Optional[Dict[str, Any]] = None

    def get_capabilities(self) -> List[str]:
        return []

    def list_available_models(self, api_key: Optional[str] = None) -> List[str]:
        return [self.model]

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        response_model: Any = None,
        execute_tools: Optional[bool] = None,
        media_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(content="ok", tool_calls=None, model=self.model, finish_reason="stop")


def test_base_provider_maps_max_tokens_to_max_output_tokens_when_unified_key_missing() -> None:
    provider = _DummyProvider(model="claude-haiku-4-5")
    _ = provider.generate("hi", max_tokens=123, temperature=0.0)

    assert provider.last_kwargs is not None
    assert provider.last_kwargs.get("max_output_tokens") == 123
    assert "max_tokens" not in provider.last_kwargs


