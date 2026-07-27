"""Reasoning must survive into session history (metadata-only, never replayed).

The assistant message stores `metadata["reasoning"]` for both non-streamed and
streamed generations; `_provider_message_dicts` keeps provider replay role/content
only, so persisted reasoning never re-enters the wire.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from abstractcore.core.session import BasicSession
from abstractcore.core.types import GenerateResponse


class _FakeProvider:
    def __init__(self, response: Any):
        self._response = response

    def generate(self, **_kwargs) -> Any:
        return self._response


def _last_assistant_message(session: BasicSession):
    for message in reversed(session.messages):
        if message.role == "assistant":
            return message
    raise AssertionError("no assistant message recorded")


def test_non_stream_reasoning_persists_into_history() -> None:
    provider = _FakeProvider(
        GenerateResponse(
            content="Final",
            model="m",
            finish_reason="stop",
            metadata={"reasoning": "Because of X."},
        )
    )
    session = BasicSession(provider=provider)
    session.generate("hi")

    message = _last_assistant_message(session)
    assert message.content == "Final"
    assert isinstance(message.metadata, dict)
    assert message.metadata.get("reasoning") == "Because of X."


def test_stream_reasoning_persists_from_trailing_aggregate() -> None:
    def _chunks() -> Iterator[GenerateResponse]:
        yield GenerateResponse(content="", model="m", metadata={"reasoning_delta": "step "})
        yield GenerateResponse(content="", model="m", metadata={"reasoning_delta": "one"})
        yield GenerateResponse(content="Final", model="m", finish_reason="stop")
        yield GenerateResponse(content="", model="m", metadata={"reasoning": "step one"})

    provider = _FakeProvider(_chunks())
    session = BasicSession(provider=provider)
    consumed = list(session.generate("hi", stream=True))
    assert consumed  # stream fully drained

    message = _last_assistant_message(session)
    assert message.content == "Final"
    assert isinstance(message.metadata, dict)
    assert message.metadata.get("reasoning") == "step one"


def test_stream_reasoning_falls_back_to_joined_deltas() -> None:
    # Defensive: if no trailing aggregate arrives (older provider), join deltas verbatim.
    def _chunks() -> Iterator[GenerateResponse]:
        yield GenerateResponse(content="Final", model="m", metadata={"reasoning_delta": "a "})
        yield GenerateResponse(content="", model="m", metadata={"reasoning_delta": "b"}, finish_reason="stop")

    provider = _FakeProvider(_chunks())
    session = BasicSession(provider=provider)
    list(session.generate("hi", stream=True))

    message = _last_assistant_message(session)
    assert message.metadata.get("reasoning") == "a b"


def test_reasoning_is_not_replayed_to_providers() -> None:
    provider = _FakeProvider(
        GenerateResponse(content="Final", model="m", metadata={"reasoning": "secret plan"})
    )
    session = BasicSession(provider=provider)
    session.generate("hi")

    replay = session._provider_message_dicts(session.messages)
    for wire_message in replay:
        assert set(wire_message.keys()) == {"role", "content"}
        assert "secret plan" not in wire_message["content"]
