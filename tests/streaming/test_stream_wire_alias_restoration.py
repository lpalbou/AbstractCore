"""Backlog 0856: native tool aliases are restored in STREAMING responses too.

Native declarations alias names a strict endpoint would reject
(`mcp::RAGnarok::query` -> `mcp_RAGnarok_query_<sha1[:8]>`, see
`abstractcore.tools.wire_naming`), so the model answers with the alias. The
non-streaming passthrough always mapped it back to the declared name; streamed
native calls reached the host with the alias, and a gateway run refused
`mcp_RAGnarok_query_b6ff8949` as "not allowed for this node or run" while the
same request without streaming worked.

These tests drive the public `generate`/`agenerate` of stub providers that emit
deterministic native calls (no live model):
- complete calls on the terminal chunk (OpenAI provider shape),
- partial deltas (OpenAI-compatible shape: name on the first delta only),
- the native async lane (providers that override `_agenerate_internal` and
  never pass through the sync `UnifiedStreamProcessor`).
The allowlist contract is unchanged: a name that is not the alias of a tool
declared on THIS request is never rewritten.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.tools.wire_naming import wire_safe_tool_name

MCP = "mcp::RAGnarok::query"
WIRE = wire_safe_tool_name(MCP)
ARGS = {"workspace": "epar", "mode": "quick"}
ARGS_JSON = json.dumps(ARGS)
CALL_ID = "call_d8d0ee3fa3fe4d86cb2802250e52848"


def _tool(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": f"tool {name}",
        "parameters": {"type": "object", "properties": {"workspace": {"type": "string"}}},
    }


TOOLS = [_tool(MCP), _tool("fetch_url")]


def _complete_call(name: str = WIRE, call_id: str = CALL_ID) -> Dict[str, Any]:
    # openai_provider stream assembly: complete calls on the terminal chunk.
    return {"id": call_id, "type": "function", "name": name, "arguments": ARGS_JSON}


def _raw_call(name: str = WIRE, call_id: str = CALL_ID) -> Dict[str, Any]:
    # Chat-completions message shape (non-streaming / native async lanes).
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": ARGS_JSON}}


def _complete_chunks(name: str = WIRE) -> List[GenerateResponse]:
    return [
        GenerateResponse(content="", model="stub"),
        GenerateResponse(
            content="", model="stub", finish_reason="tool_calls", tool_calls=[_complete_call(name)]
        ),
    ]


def _delta_chunks(name: str = WIRE) -> List[GenerateResponse]:
    # openai_compatible_provider stream: raw deltas, the name only on the first.
    half = len(ARGS_JSON) // 2
    return [
        GenerateResponse(
            content="",
            model="stub",
            tool_calls=[
                {
                    "index": 0,
                    "id": CALL_ID,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
            ],
        ),
        GenerateResponse(
            content="",
            model="stub",
            tool_calls=[{"index": 0, "function": {"arguments": ARGS_JSON[:half]}}],
        ),
        GenerateResponse(
            content="",
            model="stub",
            tool_calls=[{"index": 0, "function": {"arguments": ARGS_JSON[half:]}}],
        ),
        GenerateResponse(content="", model="stub", finish_reason="tool_calls"),
    ]


class _NativeStubProvider(BaseProvider):
    """Emits native calls with deterministic wire names; sync lane only."""

    def __init__(self, chunks: List[GenerateResponse], *, name: str = WIRE, **kwargs: Any):
        super().__init__(model="stub", **kwargs)
        self._chunks = chunks
        self._name = name

    def get_capabilities(self) -> list[str]:
        return ["streaming", "tools"]

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
            return GenerateResponse(
                content="", model="stub", finish_reason="tool_calls", tool_calls=[_raw_call(self._name)]
            )
        return iter([GenerateResponse(**c.__dict__) for c in self._chunks])


class _NativeAsyncStubProvider(_NativeStubProvider):
    """Overrides `_agenerate_internal` like the OpenAI/compatible/Anthropic/Ollama providers."""

    async def _agenerate_internal(self, prompt, messages, system_prompt, tools, media, stream, **kwargs):
        if not stream:
            return GenerateResponse(
                content="", model="stub", finish_reason="tool_calls", tool_calls=[_raw_call(self._name)]
            )

        async def _agen() -> AsyncIterator[GenerateResponse]:
            for c in self._chunks:
                await asyncio.sleep(0)
                yield GenerateResponse(**c.__dict__)

        return _agen()


def _names(calls: List[Any]) -> List[str]:
    out = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not name and isinstance(call.get("function"), dict):
            name = call["function"].get("name")
        if name:
            out.append(name)
    return out


def _stream_calls(chunks) -> List[Any]:
    return [call for c in chunks for call in (c.tool_calls or [])]


def _sync_stream(provider, tools=TOOLS) -> List[Any]:
    return _stream_calls(list(provider.generate(prompt="summarize gazyvaro", tools=tools, stream=True)))


def _async_stream(provider, tools=TOOLS) -> List[Any]:
    async def _run():
        stream = await provider.agenerate(prompt="summarize gazyvaro", tools=tools, stream=True)
        return [c async for c in stream]

    return _stream_calls(asyncio.run(_run()))


def _assembled_args(calls: List[Any]) -> Dict[str, Any]:
    text = ""
    for call in calls:
        raw = call.get("arguments")
        if raw is None and isinstance(call.get("function"), dict):
            raw = call["function"].get("arguments")
        if isinstance(raw, dict):
            return raw
        text += raw or ""
    return json.loads(text)


def _call_ids(calls: List[Any]) -> List[str]:
    return [c.get("id") or c.get("call_id") for c in calls if c.get("id") or c.get("call_id")]


def test_alias_is_not_already_the_declared_name():
    # The regression only means something if the wire name differs.
    assert WIRE != MCP and WIRE.startswith("mcp_RAGnarok_query_")


def test_non_streaming_sync_is_canonical_baseline():
    response = _NativeStubProvider(_complete_chunks()).generate(prompt="x", tools=TOOLS)
    assert [c["name"] for c in response.tool_calls] == [MCP]
    assert response.tool_calls[0]["call_id"] == CALL_ID
    assert response.tool_calls[0]["arguments"] == ARGS


@pytest.mark.parametrize("lane", ["sync", "async-default", "async-native"])
def test_streamed_complete_native_call_is_restored(lane):
    if lane == "sync":
        calls = _sync_stream(_NativeStubProvider(_complete_chunks()))
    elif lane == "async-default":
        calls = _async_stream(_NativeStubProvider(_complete_chunks()))
    else:
        calls = _async_stream(_NativeAsyncStubProvider(_complete_chunks()))
    assert _names(calls) == [MCP]
    assert _call_ids(calls) == [CALL_ID]
    assert _assembled_args(calls) == ARGS


@pytest.mark.parametrize("lane", ["sync", "async-default", "async-native"])
def test_streamed_partial_deltas_restore_the_name_and_keep_the_fragments(lane):
    chunks = _delta_chunks()
    if lane == "sync":
        calls = _sync_stream(_NativeStubProvider(chunks))
    elif lane == "async-default":
        calls = _async_stream(_NativeStubProvider(chunks))
    else:
        calls = _async_stream(_NativeAsyncStubProvider(chunks))
    assert _names(calls) == [MCP]
    # Fragments are not parsed (a partial JSON never becomes `{}`), and none is dropped.
    assert len(calls) == 3
    assert [c["function"]["arguments"] for c in calls[1:]] == [
        ARGS_JSON[: len(ARGS_JSON) // 2],
        ARGS_JSON[len(ARGS_JSON) // 2 :],
    ]
    assert _assembled_args(calls) == ARGS
    assert calls[0]["id"] == CALL_ID and calls[0]["index"] == 0


def test_async_native_non_streaming_is_restored():
    async def _run():
        return await _NativeAsyncStubProvider(_complete_chunks()).agenerate(prompt="x", tools=TOOLS)

    response = asyncio.run(_run())
    assert _names(response.tool_calls) == [MCP]
    assert _call_ids(response.tool_calls) == [CALL_ID]


@pytest.mark.parametrize("lane", ["sync", "async-native"])
def test_alias_of_an_undeclared_tool_is_never_rewritten(lane):
    # Allowlist: the request declares only fetch_url, so the MCP alias must
    # reach the host verbatim (and be refused there), never be "restored".
    tools = [_tool("fetch_url")]
    provider_cls = _NativeStubProvider if lane == "sync" else _NativeAsyncStubProvider
    run = _sync_stream if lane == "sync" else _async_stream
    calls = run(provider_cls(_complete_chunks()), tools=tools)
    assert _names(calls) == [WIRE]


def test_unknown_lookalike_alias_is_not_rewritten():
    forged = "mcp_RAGnarok_query_deadbeef"
    assert forged != WIRE
    calls = _sync_stream(_NativeStubProvider(_complete_chunks(forged), name=forged))
    assert _names(calls) == [forged]


def test_safe_names_and_input_chunks_are_unchanged():
    chunks = _complete_chunks("fetch_url")
    calls = _sync_stream(_NativeStubProvider(chunks, name="fetch_url"))
    assert _names(calls) == ["fetch_url"]
    # The provider's own chunk objects are not mutated by restoration.
    aliased = _complete_chunks()
    provider = _NativeAsyncStubProvider(aliased)
    _async_stream(provider)
    assert aliased[1].tool_calls[0]["name"] == WIRE


def test_concurrent_requests_do_not_share_alias_maps():
    # Request A declares the MCP tool; request B (same provider class, same
    # wire name) declares only fetch_url. Interleave their streams.
    a = iter(_NativeStubProvider(_complete_chunks()).generate(prompt="a", tools=TOOLS, stream=True))
    b = iter(
        _NativeStubProvider(_complete_chunks()).generate(prompt="b", tools=[_tool("fetch_url")], stream=True)
    )
    a_chunks, b_chunks = [], []
    for _ in range(10):
        for it, sink in ((a, a_chunks), (b, b_chunks)):
            try:
                sink.append(next(it))
            except StopIteration:
                pass
    assert _names(_stream_calls(a_chunks)) == [MCP]
    assert _names(_stream_calls(b_chunks)) == [WIRE]


@pytest.mark.parametrize("lane", ["sync", "async-native"])
def test_unresolved_native_name_is_kept_and_warned(lane):
    forged = "mcp_RAGnarok_query_deadbeef"
    provider_cls = _NativeStubProvider if lane == "sync" else _NativeAsyncStubProvider
    provider = provider_cls(_complete_chunks(forged), name=forged)
    if lane == "sync":
        chunks = list(provider.generate(prompt="x", tools=TOOLS, stream=True))
    else:

        async def _run():
            stream = await provider.agenerate(prompt="x", tools=TOOLS, stream=True)
            return [c async for c in stream]

        chunks = asyncio.run(_run())
    warnings = [
        w for c in chunks if isinstance(c.metadata, dict) for w in (c.metadata.get("warnings") or [])
    ]
    assert any(forged in w and "does not match any available tool" in w for w in warnings)


def test_repeated_terminal_envelopes_are_each_restored():
    chunks = [
        GenerateResponse(content="", model="stub", tool_calls=[_complete_call()]),
        GenerateResponse(content="", model="stub", finish_reason="tool_calls", tool_calls=[_complete_call()]),
    ]
    calls = _sync_stream(_NativeStubProvider(chunks))
    assert _names(calls) == [MCP, MCP]
    assert set(_call_ids(calls)) == {CALL_ID}


def test_restore_helper_shapes():
    from dataclasses import dataclass

    from abstractcore.tools.wire_naming import restore_tool_call_names

    @dataclass
    class _Call:
        name: str
        arguments: dict

    both = {"id": "c1", "name": WIRE, "function": {"name": WIRE, "arguments": "{"}}
    nameless = {"index": 0, "function": {"arguments": "}"}}
    obj = _Call(name=WIRE, arguments={})
    out, unresolved = restore_tool_call_names([both, nameless, obj], [MCP])
    assert out[0] == {"id": "c1", "name": MCP, "function": {"name": MCP, "arguments": "{"}}
    assert out[1] is nameless and out[2].name == MCP and unresolved == []
    assert both["name"] == WIRE and both["function"]["name"] == WIRE  # inputs untouched
    same, unresolved = restore_tool_call_names([{"name": "nope"}], [MCP])
    assert same == [{"name": "nope"}] and unresolved == ["nope"]
    untouched = [{"name": WIRE}]
    assert restore_tool_call_names(untouched, [])[0] is untouched


@pytest.mark.parametrize("lane", ["sync", "async-native"])
def test_name_repeated_on_every_partial_is_restored_and_warned_once(lane):
    # Anthropic-style partials repeat the full name on every delta.
    def partials(name):
        return [
            GenerateResponse(
                content="", model="stub", tool_calls=[{"id": CALL_ID, "name": name, "arguments": frag}]
            )
            for frag in ('{"workspace": ', '"epar"}')
        ] + [GenerateResponse(content="", model="stub", finish_reason="tool_calls")]

    provider_cls = _NativeStubProvider if lane == "sync" else _NativeAsyncStubProvider
    run = _sync_stream if lane == "sync" else _async_stream
    assert _names(run(provider_cls(partials(WIRE)))) == [MCP, MCP]

    forged = "mcp_RAGnarok_query_deadbeef"
    provider = provider_cls(partials(forged), name=forged)
    if lane == "sync":
        chunks = list(provider.generate(prompt="x", tools=TOOLS, stream=True))
    else:

        async def _run():
            stream = await provider.agenerate(prompt="x", tools=TOOLS, stream=True)
            return [c async for c in stream]

        chunks = asyncio.run(_run())
    warnings = [
        w for c in chunks if isinstance(c.metadata, dict) for w in (c.metadata.get("warnings") or [])
    ]
    assert len([w for w in warnings if forged in w]) == 1
