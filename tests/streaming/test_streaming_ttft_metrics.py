from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _TTFTProvider(BaseProvider):
    def get_capabilities(self) -> list[str]:
        return ["streaming"]

    def list_available_models(self, **kwargs) -> list[str]:
        return [self.model]

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
        response_model: Optional[Any] = None,
        execute_tools: Optional[bool] = None,
        media_metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        if not stream:
            return GenerateResponse(content="ok", model=self.model, finish_reason="stop", gen_time=0.0)

        def _gen() -> Iterator[GenerateResponse]:
            yield GenerateResponse(content="hello", model=self.model)

        return _gen()


def test_streaming_first_chunk_includes_ttft_ms(monkeypatch) -> None:
    import abstractcore.providers.base as base_mod

    ticks = iter([1.0, 1.1234])
    monkeypatch.setattr(base_mod.time, "perf_counter", lambda: next(ticks))

    llm = _TTFTProvider(model="unit-test")
    stream = llm.generate(prompt="hi", stream=True)

    first = next(stream)
    assert isinstance(first.metadata, dict)
    timing = first.metadata.get("_timing")
    assert isinstance(timing, dict)
    assert timing.get("ttft_ms") == 123.4
