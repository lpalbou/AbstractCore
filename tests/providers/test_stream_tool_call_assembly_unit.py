"""Streamed tool calls arrive complete.

An OpenAI-compatible stream splits a tool call over many chunks (a named first delta, then
`function.arguments` string fragments addressed by `index`). The provider joins them and
publishes the complete calls once, with the finish reason, so a streamed call carries the same
name and arguments as the non-streamed one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.stream_tool_calls import StreamedToolCallAssembler

# ------------------------------------------------------------------ assembler


def _assemble(*chunks: List[Dict[str, Any]]):
    assembler = StreamedToolCallAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    return assembler.drain()


def _fragments(index: int, *pieces: str) -> List[List[Dict[str, Any]]]:
    return [[{"index": index, "function": {"arguments": piece}}] for piece in pieces]


def test_fragments_join_by_index_and_identical_pieces_are_kept() -> None:
    # vLLM shape: the first delta has no `arguments` key; '"' arrives twice.
    calls = _assemble(
        [{"id": "call-1", "type": "function", "index": 0, "function": {"name": "web_search"}}],
        *_fragments(0, '{"query":', ' "', "a", '"', "}"),
    )
    assert calls == [{"id": "call-1", "type": "function", "name": "web_search", "arguments": '{"query": "a"}'}]
    assert json.loads(calls[0]["arguments"]) == {"query": "a"}


def test_parallel_calls_keep_their_own_arguments() -> None:
    calls = _assemble(
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": ""}}],
        [{"id": "c1", "index": 1, "function": {"name": "b", "arguments": ""}}],
        [{"index": 1, "function": {"arguments": '{"y": 2}'}}],
        [{"index": 0, "function": {"arguments": '{"x": 1}'}}],
    )
    assert [(c["id"], c["name"], c["arguments"]) for c in calls] == [("c0", "a", '{"x": 1}'), ("c1", "b", '{"y": 2}')]


def test_a_new_id_on_a_reused_index_starts_a_new_call() -> None:
    calls = _assemble(
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": "{}"}}],
        [{"id": "c1", "index": 0, "function": {"name": "b", "arguments": '{"y": 2}'}}],
    )
    assert [(c["id"], c["arguments"]) for c in calls] == [("c0", "{}"), ("c1", '{"y": 2}')]


def test_restated_calls_keep_the_latest_snapshot() -> None:
    # Servers that re-send the whole call on every chunk send the arguments so far.
    calls = _assemble(
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": '{"x":'}}],
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": '{"x": 1}'}}],
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": '{"x": 1}'}}],
    )
    assert calls == [{"id": "c0", "type": "function", "name": "a", "arguments": '{"x": 1}'}]


def test_object_arguments_zero_argument_calls_and_truncated_calls() -> None:
    assert _assemble([{"id": "c0", "name": "a", "arguments": {"x": 1}}])[0]["arguments"] == '{"x": 1}'
    assert _assemble([{"id": "c0", "index": 0, "function": {"name": "noop"}}])[0]["arguments"] == "{}"
    # A call cut off mid-arguments keeps what was received.
    cut = _assemble([{"id": "c0", "index": 0, "function": {"name": "a"}}], *_fragments(0, '{"x": "unfin'))
    assert cut[0]["arguments"] == '{"x": "unfin'


def test_sdk_objects_and_index_less_fragments() -> None:
    first = SimpleNamespace(id="c0", index=0, type="function", function=SimpleNamespace(name="a", arguments=""))
    rest = SimpleNamespace(id=None, index=0, type=None, function=SimpleNamespace(name=None, arguments='{"x": 1}'))
    assert _assemble([first], [rest]) == [{"id": "c0", "type": "function", "name": "a", "arguments": '{"x": 1}'}]
    # No index and no id: the fragment continues the latest call.
    calls = _assemble([{"id": "c0", "function": {"name": "a"}}], [{"function": {"arguments": "{}"}}])
    assert calls[0]["arguments"] == "{}"


def test_drain_resets_and_reports_no_calls_as_none() -> None:
    assembler = StreamedToolCallAssembler()
    assert assembler.drain() is None
    assembler.feed(None)
    assembler.feed("not a list")
    assert not assembler
    assembler.feed([{"id": "c0", "index": 0, "function": {"name": "a", "arguments": "{}"}}])
    assert assembler and assembler.drain() and assembler.drain() is None


# ------------------------------------------------------------------ provider stream


class _Resp:
    status_code = 200

    def __init__(self, lines: List[str]):
        self._lines = lines

    def iter_lines(self):
        yield from self._lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _Client:
    def __init__(self, lines: List[str]):
        self._lines = lines

    @contextlib.contextmanager
    def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002 - httpx signature
        yield _Resp(self._lines)


class _AsyncClient:
    def __init__(self, lines: List[str]):
        self._lines = lines

    @contextlib.asynccontextmanager
    async def stream(self, method, url, json=None, headers=None, **kwargs):  # noqa: A002 - httpx signature
        yield _Resp(self._lines)


def _sse(delta: Dict[str, Any], finish: Any = None) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


def _tool_stream(finish: Any = "tool_calls") -> List[str]:
    lines = [
        _sse({"role": "assistant", "content": ""}),
        _sse({"reasoning": "search the web"}),
        _sse({"tool_calls": [{"id": "call-1", "type": "function", "index": 0, "function": {"name": "web_search"}}]}),
    ]
    for piece in ('{"query":', ' "', "latest", " v", "LLM", '"', "}"):
        lines.append(_sse({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}))
    if finish:
        lines.append(_sse({}, finish=finish))
    lines.append('data: {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}}')
    lines.append("data: [DONE]")
    return lines


def _provider() -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.base_url = "http://stub/v1"
    p.model = "stub-model"
    p._get_headers = lambda: {}
    p._httpx_timeout = lambda streaming=False: None
    p.architecture_config = None
    p.model_capabilities = {}
    return p


EXPECTED = [{"id": "call-1", "type": "function", "name": "web_search", "arguments": '{"query": "latest vLLM"}'}]
PAYLOAD = {"model": "stub-model", "messages": [], "stream": True}


def test_stream_emits_one_complete_call_with_the_finish_reason() -> None:
    p = _provider()
    p.client = _Client(_tool_stream())
    chunks = list(p._stream_generate(dict(PAYLOAD)))

    with_calls = [c for c in chunks if c.tool_calls]
    assert len(with_calls) == 1
    assert with_calls[0].tool_calls == EXPECTED
    assert with_calls[0].finish_reason == "tool_calls"
    # Usage still closes the stream.
    assert chunks[-1].usage and chunks[-1].usage["total_tokens"] == 16


def test_stream_without_a_finish_reason_still_delivers_the_call() -> None:
    p = _provider()
    p.client = _Client(_tool_stream(finish=None))
    chunks = list(p._stream_generate(dict(PAYLOAD)))
    assert [c.tool_calls for c in chunks if c.tool_calls] == [EXPECTED]


def test_async_stream_emits_the_same_complete_call() -> None:
    p = _provider()
    p._async_client = _AsyncClient(_tool_stream())

    async def collect():
        return [c async for c in p._async_stream_generate(dict(PAYLOAD))]

    chunks = asyncio.run(collect())
    assert [c.tool_calls for c in chunks if c.tool_calls] == [EXPECTED]


def test_plain_text_stream_carries_no_tool_calls() -> None:
    p = _provider()
    p.client = _Client([_sse({"content": "hel"}), _sse({"content": "lo"}, finish="stop"), "data: [DONE]"])
    chunks = list(p._stream_generate(dict(PAYLOAD)))
    assert "".join(c.content or "" for c in chunks) == "hello"
    assert all(c.tool_calls is None for c in chunks)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_openai_provider_keys_fragments_by_index(mode: str) -> None:
    """The OpenAI SDK sends the id on the first delta only; later deltas carry the index."""
    from abstractcore.providers.openai_provider import OpenAIProvider

    def delta(**kw):
        return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(**kw)])

    def chunk(d, finish=None):
        return SimpleNamespace(model="m", choices=[SimpleNamespace(delta=d, finish_reason=finish)])

    events = [
        chunk(delta(id="call-1", index=0, type="function", function=SimpleNamespace(name="web_search", arguments=""))),
        chunk(delta(id=None, index=0, type=None, function=SimpleNamespace(name=None, arguments='{"query":'))),
        chunk(delta(id=None, index=0, type=None, function=SimpleNamespace(name=None, arguments=' "x"}'))),
        chunk(SimpleNamespace(content=None, tool_calls=None), finish="tool_calls"),
    ]

    p = OpenAIProvider.__new__(OpenAIProvider)
    p._handle_tool_execution = lambda response, tools: response
    tools = [{"name": "web_search"}]

    if mode == "sync":
        p.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(events))))
        chunks = list(p._stream_response({}, tools))
    else:

        async def create(**kw):
            async def gen():
                for e in events:
                    yield e

            return gen()

        p._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        async def collect():
            return [c async for c in p._async_stream_response({}, tools)]

        chunks = asyncio.run(collect())

    calls = [c.tool_calls for c in chunks if c.tool_calls]
    assert calls == [[{"id": "call-1", "type": "function", "name": "web_search", "arguments": '{"query": "x"}'}]]
