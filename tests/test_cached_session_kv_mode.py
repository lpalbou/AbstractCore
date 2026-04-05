from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import pytest

from abstractcore.core.cached_session import CachedSession
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


@dataclass
class _FakeCache:
    chunks: List[Dict[str, Any]] = field(default_factory=list)


class _StubKVProvider(BaseProvider):
    def __init__(self, model: str = "stub", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.provider = "stubkv"
        self.last_call: Dict[str, Any] = {}
        self.last_kwargs: Dict[str, Any] = {}

    def supports_prompt_cache(self) -> bool:
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        return _FakeCache()

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        if not isinstance(cache_value, _FakeCache):
            return None
        return _FakeCache(chunks=list(cache_value.chunks))

    def _prompt_cache_backend_append(
        self,
        cache_value: Any,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> bool:
        _ = kwargs
        if not isinstance(cache_value, _FakeCache):
            return False
        cache_value.chunks.append(
            {
                "prompt": str(prompt or ""),
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "add_generation_prompt": bool(add_generation_prompt),
            }
        )
        return True

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        self.last_call = {
            "prompt": prompt,
            "messages": messages,
            "system_prompt": system_prompt,
            "tools": tools,
            "media": media,
            "stream": stream,
        }
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub"]


def _noop_tool(text: str) -> str:
    """Return the input."""

    return text


def test_cached_session_kv_mode_passes_tracing_and_prefilled_modules() -> None:
    llm = _StubKVProvider()
    session = CachedSession(
        provider=llm,
        system_prompt="You are helpful.",
        tools=[_noop_tool],
        prompt_cache_strategy="kv",
        enable_tracing=True,
    )

    assert session.prompt_cache_mode == "kv"
    assert isinstance(session.prompt_cache_key, str) and session.prompt_cache_key

    _ = session.generate("hi")

    assert llm.last_call["messages"] is None
    assert llm.last_call["system_prompt"] is None
    assert llm.last_call["tools"]  # session tools are passed for tool parsing/execution
    assert llm.last_kwargs.get("prompt_cache_prefilled_modules") == ("system", "tools")

    trace_metadata = llm.last_kwargs.get("trace_metadata")
    assert isinstance(trace_metadata, dict)
    assert trace_metadata.get("session_id") == session.id
    assert trace_metadata.get("step_type") == "chat"
    assert trace_metadata.get("attempt_number") == 1


def test_cached_session_rebuild_prompt_cache_replays_transcript() -> None:
    llm = _StubKVProvider()
    session = CachedSession(
        provider=llm,
        system_prompt="You are helpful.",
        tools=[_noop_tool],
        prompt_cache_strategy="kv",
    )

    _ = session.generate("Say OK")
    old_key = session.prompt_cache_key
    assert isinstance(old_key, str) and old_key

    assert session.rebuild_prompt_cache() is True
    assert session.prompt_cache_key != old_key

    cache_value = llm._prompt_cache_store.get(session.prompt_cache_key)
    assert isinstance(cache_value, _FakeCache)

    replay_chunks = [c for c in cache_value.chunks if c.get("messages")]
    assert replay_chunks
    replay_messages = replay_chunks[-1]["messages"]
    assert isinstance(replay_messages, list)
    assert replay_messages[0]["role"] == "user"
    assert replay_messages[1]["role"] == "assistant"


def test_cached_session_kv_mode_warns_on_per_call_overrides() -> None:
    llm = _StubKVProvider()
    session = CachedSession(
        provider=llm,
        system_prompt="You are helpful.",
        tools=[_noop_tool],
        prompt_cache_strategy="kv",
    )

    with pytest.warns(RuntimeWarning, match="ignores per-call `tools=`"):
        _ = session.generate("hi", tools=[])

