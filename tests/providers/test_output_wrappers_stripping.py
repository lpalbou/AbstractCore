from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Type, Union

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _DummyProvider(BaseProvider):
    def __init__(self, *, model: str, content: str):
        super().__init__(model)
        self.provider = "dummy"
        self._content = content

    def get_capabilities(self) -> List[str]:
        return []

    def list_available_models(self, **kwargs) -> List[str]:
        return []

    def unload_model(self, model_name: str) -> None:
        return None

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        response_model: Optional[Type[Any]] = None,
        execute_tools: Optional[bool] = None,
        media_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        return GenerateResponse(content=self._content, model=self.model, finish_reason="stop")


def test_strips_glm_box_wrappers_when_configured() -> None:
    provider = _DummyProvider(model="glm-4.6v", content="\n<|begin_of_box|>pong<|end_of_box|>")
    resp = provider.generate_with_telemetry(prompt="test", stream=False)
    assert resp.content == "pong"


def test_does_not_strip_when_not_configured() -> None:
    provider = _DummyProvider(model="llama-3.1", content="\n<|begin_of_box|>pong<|end_of_box|>")
    resp = provider.generate_with_telemetry(prompt="test", stream=False)
    assert resp.content == "\n<|begin_of_box|>pong<|end_of_box|>"
