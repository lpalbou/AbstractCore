"""Streamed channel-separated reasoning: per-chunk deltas + trailing aggregate.

Providers that separate reasoning from content at the wire level (Ollama `thinking`,
LM Studio native `reasoning.delta`, OpenAI-compatible `reasoning_content`, Anthropic
`thinking_delta`) emit incremental `metadata["reasoning_delta"]` chunks. BaseProvider
must join those deltas and guarantee the stream ends with a trailing chunk carrying
the COMPLETE `metadata["reasoning"]` — that final aggregate is what stream folds
(runtime rehydration) persist. Without it, consumers only ever saw one delta.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _ReasoningDeltaStreamProvider(BaseProvider):
    """Stub provider emitting channel-separated reasoning deltas."""

    chunks: List[GenerateResponse] = []

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
            return GenerateResponse(content="Final", model=self.model, finish_reason="stop")

        def _gen() -> Iterator[GenerateResponse]:
            for chunk in type(self).chunks:
                yield chunk

        return _gen()


def _make_provider(chunks: List[GenerateResponse]) -> _ReasoningDeltaStreamProvider:
    _ReasoningDeltaStreamProvider.chunks = chunks
    provider = _ReasoningDeltaStreamProvider(model="unit-test")
    provider.model_capabilities = {"thinking_support": True}
    return provider


def test_streaming_reasoning_deltas_aggregate_on_trailing_chunk() -> None:
    provider = _make_provider(
        [
            GenerateResponse(
                content="",
                model="unit-test",
                metadata={"reasoning_delta": "Let's think"},
            ),
            GenerateResponse(
                content="",
                model="unit-test",
                # Leading whitespace must survive the join (deltas are verbatim).
                metadata={"reasoning_delta": " about it."},
            ),
            GenerateResponse(
                content="Final",
                model="unit-test",
                finish_reason="stop",
                usage={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
            ),
        ]
    )

    chunks = list(provider.generate(prompt="hi", stream=True))

    visible = "".join(c.content or "" for c in chunks)
    assert visible == "Final"

    # Per-chunk deltas remain visible for live tails.
    deltas = [
        c.metadata.get("reasoning_delta")
        for c in chunks
        if isinstance(c.metadata, dict) and c.metadata.get("reasoning_delta")
    ]
    assert deltas == ["Let's think", " about it."]

    # The trailing chunk carries the COMPLETE reasoning (whitespace-faithful join)
    # and re-carries the last seen usage.
    final = chunks[-1]
    assert isinstance(final.metadata, dict)
    assert final.metadata.get("reasoning") == "Let's think about it."
    assert final.reasoning == "Let's think about it."
    assert final.usage == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


def test_streaming_legacy_reasoning_key_is_tolerated_as_delta() -> None:
    # Out-of-tree providers may still emit per-chunk `reasoning` for deltas; the
    # aggregate must not lose them.
    provider = _make_provider(
        [
            GenerateResponse(content="", model="unit-test", metadata={"reasoning": "a"}),
            GenerateResponse(content="", model="unit-test", metadata={"reasoning": "b"}),
            GenerateResponse(content="Done", model="unit-test", finish_reason="stop"),
        ]
    )

    chunks = list(provider.generate(prompt="hi", stream=True))
    assert "".join(c.content or "" for c in chunks) == "Done"
    final = chunks[-1]
    assert isinstance(final.metadata, dict)
    assert final.metadata.get("reasoning") == "ab"


def test_streaming_without_reasoning_emits_no_trailing_reasoning_chunk() -> None:
    provider = _make_provider(
        [
            GenerateResponse(content="Hello", model="unit-test"),
            GenerateResponse(content=" world", model="unit-test", finish_reason="stop"),
        ]
    )

    chunks = list(provider.generate(prompt="hi", stream=True))
    assert "".join(c.content or "" for c in chunks) == "Hello world"
    for c in chunks:
        if isinstance(c.metadata, dict):
            assert "reasoning" not in c.metadata
            assert "reasoning_delta" not in c.metadata
