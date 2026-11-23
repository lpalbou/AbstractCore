"""Streamed native tool calls arrive once, complete.

An OpenAI-compatible stream splits a tool call over many chunks (a named first delta, then
`function.arguments` string fragments addressed by `index`). The providers join them and publish
each call once, with the finish reason. A call the model did not finish is reported in the chunk
metadata instead of being delivered with missing arguments.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from abstractcore.core.retry import RetryConfig
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.native_tool_calls import StreamedToolCallAssembler, incomplete_tool_call_metadata

# ------------------------------------------------------------------ assembler


def _assemble(*chunks: List[Any]):
    assembler = StreamedToolCallAssembler()
    for chunk in chunks:
        assembler.feed(chunk)
    return assembler.drain()


def _complete(*chunks: List[Any]):
    complete, incomplete = _assemble(*chunks)
    assert incomplete == []
    return complete


def _fragments(index: int, *pieces: str) -> List[List[Dict[str, Any]]]:
    return [[{"index": index, "function": {"arguments": piece}}] for piece in pieces]


def test_fragments_join_by_index_and_identical_pieces_are_kept() -> None:
    # vLLM shape: the first delta has no `arguments` key; '"' arrives twice.
    calls = _complete(
        [{"id": "call-1", "type": "function", "index": 0, "function": {"name": "web_search"}}],
        *_fragments(0, '{"query":', ' "', "a", '"', "}"),
    )
    assert calls == [{"id": "call-1", "type": "function", "name": "web_search", "arguments": '{"query": "a"}'}]


def test_parallel_calls_keep_their_own_arguments() -> None:
    calls = _complete(
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": ""}}],
        [{"id": "c1", "index": 1, "function": {"name": "b", "arguments": ""}}],
        [{"index": 1, "function": {"arguments": '{"y": 2}'}}],
        [{"index": "0", "function": {"arguments": '{"x": 1}'}}],  # an index sent as a string
    )
    assert [(c["id"], c["name"], c["arguments"]) for c in calls] == [("c0", "a", '{"x": 1}'), ("c1", "b", '{"y": 2}')]


def test_a_named_delta_under_a_new_id_starts_the_next_call() -> None:
    calls = _complete(
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": "{}"}}],
        [{"id": "c1", "index": 0, "function": {"name": "b", "arguments": '{"y": 2}'}}],
    )
    assert [(c["id"], c["name"], c["arguments"]) for c in calls] == [("c0", "a", "{}"), ("c1", "b", '{"y": 2}')]


def test_an_id_that_changes_on_every_delta_still_makes_one_call() -> None:
    calls = _complete(
        [{"id": "x1", "index": 0, "function": {"name": "a", "arguments": '{"x":'}}],
        [{"id": "x2", "index": 0, "function": {"arguments": " 1"}}],
        [{"id": "x3", "function": {"arguments": "}"}}],  # no index either: continues the latest call
    )
    assert calls == [{"id": "x1", "type": "function", "name": "a", "arguments": '{"x": 1}'}]


def test_a_restated_call_keeps_the_latest_version() -> None:
    # Servers that send the whole call again send the arguments so far, with or without an id.
    with_id = _complete(
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": '{"x":'}}],
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": '{"x": 1}'}}],
        [{"id": "c0", "index": 0, "function": {"name": "a", "arguments": '{"x": 1}'}}],
    )
    without_id = _complete(
        [{"index": 0, "function": {"name": "a", "arguments": '{"x": 1}'}}],
        [{"index": 0, "function": {"name": "a", "arguments": '{"x": 1}'}}],
    )
    assert [c["arguments"] for c in with_id] == ['{"x": 1}']
    assert [c["arguments"] for c in without_id] == ['{"x": 1}']


def test_a_name_split_across_deltas_is_joined() -> None:
    calls = _complete(
        [{"id": "c0", "index": 0, "function": {"name": "get_", "arguments": ""}}],
        [{"index": 0, "function": {"name": "weather", "arguments": "{}"}}],
    )
    assert calls[0]["name"] == "get_weather"


def test_object_arguments_zero_argument_calls_and_sdk_objects() -> None:
    assert _complete([{"id": "c0", "name": "a", "arguments": {"x": 1}}])[0]["arguments"] == '{"x": 1}'
    assert _complete([{"id": "c0", "index": 0, "function": {"name": "noop"}}])[0]["arguments"] == "{}"
    first = SimpleNamespace(id="c0", index=0, type="function", function=SimpleNamespace(name="a", arguments=""))
    rest = SimpleNamespace(id=None, index=0, type=None, function=SimpleNamespace(name=None, arguments='{"x": 1}'))
    assert _complete([first], [rest]) == [{"id": "c0", "type": "function", "name": "a", "arguments": '{"x": 1}'}]


def test_a_call_the_model_did_not_finish_is_incomplete_and_keeps_every_byte() -> None:
    complete, incomplete = _assemble(
        [{"id": "c0", "index": 0, "function": {"name": "done", "arguments": '{"x": 1}'}}],
        [{"id": "c1", "index": 1, "function": {"name": "write_file"}}],
        *_fragments(1, '{"path": "a.txt", "content": "unfin'),
    )
    assert [c["name"] for c in complete] == ["done"]
    assert incomplete == [
        {"id": "c1", "type": "function", "name": "write_file", "arguments": '{"path": "a.txt", "content": "unfin'}
    ]

    meta = incomplete_tool_call_metadata(
        {"k": 1}, incomplete, finish_reason="length", available_tools=["write_file", "done"]
    )
    assert meta["k"] == 1
    assert meta["unparsed_tool_call"] == {
        "format": "native",
        "text": 'write_file {"path": "a.txt", "content": "unfin',
        "reason": "unclosed",
    }
    assert meta["tool_call_error"] == {"code": "invalid_tool_syntax", "available_tools": ["done", "write_file"]}
    assert len(meta["warnings"]) == 1
    assert incomplete_tool_call_metadata({"k": 1}, [], finish_reason="stop") == {"k": 1}


def test_nameless_pieces_are_not_calls_and_drain_resets() -> None:
    assembler = StreamedToolCallAssembler()
    assert assembler.drain() == (None, [])
    assembler.feed(None)
    assembler.feed("not a list")
    assembler.feed([{"index": 3, "function": {"arguments": '{"x": 1}'}}])  # no name ever arrives
    assert assembler.drain() == (None, [])
    assembler.feed([{"id": "c0", "index": 0, "function": {"name": "a", "arguments": "{}"}}])
    assert assembler and assembler.drain()[0] and assembler.drain() == (None, [])


# ------------------------------------------------------------------ OpenAI-compatible stream


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


def _tool_stream(finish: Any = "tool_calls", pieces=('{"query":', ' "', "latest", " v", "LLM", '"', "}")) -> List[str]:
    lines = [
        _sse({"role": "assistant", "content": ""}),
        _sse({"reasoning": "search the web"}),
        _sse({"tool_calls": [{"id": "call-1", "type": "function", "index": 0, "function": {"name": "web_search"}}]}),
    ]
    for piece in pieces:
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
TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {}}}}]
PAYLOAD = {"model": "stub-model", "messages": [], "stream": True, "tools": TOOLS}


def test_stream_emits_one_complete_call_with_the_finish_reason() -> None:
    p = _provider()
    p.client = _Client(_tool_stream())
    chunks = list(p._stream_generate(dict(PAYLOAD)))

    with_calls = [c for c in chunks if c.tool_calls]
    assert len(with_calls) == 1
    assert with_calls[0].tool_calls == EXPECTED
    assert with_calls[0].finish_reason == "tool_calls"
    # The chunks that carried the call's pieces are marked; usage still closes the stream.
    assert sum(1 for c in chunks if (c.metadata or {}).get("tool_call_delta")) == 8
    assert chunks[-1].usage and chunks[-1].usage["total_tokens"] == 16


def test_stream_without_a_finish_reason_still_delivers_the_call() -> None:
    p = _provider()
    p.client = _Client(_tool_stream(finish=None))
    chunks = list(p._stream_generate(dict(PAYLOAD)))
    assert [c.tool_calls for c in chunks if c.tool_calls] == [EXPECTED]


def test_async_stream_emits_the_same_complete_call_with_or_without_a_finish_reason() -> None:
    async def collect(lines):
        p = _provider()
        p._async_client = _AsyncClient(lines)
        return [c async for c in p._async_stream_generate(dict(PAYLOAD))]

    for finish in ("tool_calls", None):
        chunks = asyncio.run(collect(_tool_stream(finish=finish)))
        assert [c.tool_calls for c in chunks if c.tool_calls] == [EXPECTED]


def test_a_call_cut_at_the_output_limit_is_reported_not_delivered() -> None:
    p = _provider()
    p.client = _Client(_tool_stream(finish="length", pieces=('{"query":', ' "lat')))
    chunks = list(p._stream_generate(dict(PAYLOAD)))

    assert all(c.tool_calls is None for c in chunks)
    reported = [c for c in chunks if (c.metadata or {}).get("unparsed_tool_call")]
    assert len(reported) == 1 and reported[0].finish_reason == "length"
    assert reported[0].metadata["unparsed_tool_call"] == {
        "format": "native",
        "text": 'web_search {"query": "lat',
        "reason": "unclosed",
    }
    assert reported[0].metadata["tool_call_error"]["available_tools"] == ["web_search"]


def test_plain_text_stream_carries_no_tool_calls() -> None:
    p = _provider()
    p.client = _Client([_sse({"content": "hel"}), _sse({"content": "lo"}, finish="stop"), "data: [DONE]"])
    chunks = list(p._stream_generate(dict(PAYLOAD)))
    assert "".join(c.content or "" for c in chunks) == "hello"
    assert all(c.tool_calls is None and not c.metadata for c in chunks)


def test_generate_stream_times_the_first_token_at_the_first_piece_of_the_call() -> None:
    """Through the public API: the call arrives once and TTFT does not wait for the finish chunk."""
    provider = OpenAICompatibleProvider(
        model="stub-model",
        base_url="http://127.0.0.1:9/v1",  # unreachable on purpose
        api_key="x",
        validate_model=False,
        retry_config=RetryConfig(max_attempts=1, initial_delay=0.0, use_jitter=False),
    )
    provider.client = _Client(_tool_stream())
    tools = [{"name": "web_search", "description": "Search.", "parameters": {"query": {"type": "string"}}}]

    chunks = list(provider.generate("latest vLLM?", tools=tools, stream=True))

    calls = [call for c in chunks if c.tool_calls for call in c.tool_calls]
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [("web_search", {"query": "latest vLLM"})]
    timed = [i for i, c in enumerate(chunks) if isinstance(c.metadata, dict) and "_timing" in c.metadata]
    first_call_chunk = next(i for i, c in enumerate(chunks) if c.tool_calls)
    assert timed and timed[0] < first_call_chunk


# ------------------------------------------------------------------ non-streamed responses


class _JsonResp:
    status_code = 200
    text = ""

    def __init__(self, body):
        self._body = body

    def json(self):
        return dict(self._body)

    def read(self):
        return b""


class _PostClient:
    def __init__(self, body):
        self._body = body

    def post(self, url, json=None, headers=None):  # noqa: A002 - httpx signature
        return _JsonResp(self._body)


def _non_streamed(arguments: str, finish: str):
    provider = OpenAICompatibleProvider(
        model="stub-model",
        base_url="http://127.0.0.1:9/v1",  # unreachable on purpose
        api_key="x",
        validate_model=False,
        retry_config=RetryConfig(max_attempts=1, initial_delay=0.0, use_jitter=False),
    )
    call = {"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}
    provider.client = _PostClient(
        {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
        }
    )
    tools = [
        {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {"path": {"type": "string"}, "content": {"type": "string"}},
        }
    ]
    return provider.generate("write it", tools=tools)


def test_non_streamed_call_cut_at_the_output_limit_is_reported_not_delivered() -> None:
    response = _non_streamed('{"path": "a.txt", "content": "unfin', "length")
    assert not response.tool_calls
    assert response.metadata["unparsed_tool_call"] == {
        "format": "native",
        "text": 'write_file {"path": "a.txt", "content": "unfin',
        "reason": "unclosed",
    }
    assert response.metadata["tool_call_error"] == {"code": "invalid_tool_syntax", "available_tools": ["write_file"]}


def test_non_streamed_complete_and_argument_less_calls_are_delivered() -> None:
    complete = _non_streamed('{"path": "a.txt", "content": "done"}', "tool_calls")
    assert complete.tool_calls == [
        {"name": "write_file", "arguments": {"path": "a.txt", "content": "done"}, "call_id": "call-1"}
    ]
    assert "unparsed_tool_call" not in (complete.metadata or {})
    for empty in ("", "null", "{}"):
        assert _non_streamed(empty, "tool_calls").tool_calls[0]["arguments"] == {}

# ------------------------------------------------------------------ OpenAI and Anthropic SDK streams


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_openai_provider_keys_fragments_by_index(mode: str) -> None:
    """The OpenAI SDK sends the id on the first delta only; later deltas carry the index."""
    from abstractcore.providers.openai_provider import OpenAIProvider

    def delta(**kw):
        return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(**kw)])

    def chunk(d, finish=None):
        return SimpleNamespace(model="m", choices=[SimpleNamespace(delta=d, finish_reason=finish)])

    events = [
        chunk(SimpleNamespace(content="I'll search.", tool_calls=None)),
        chunk(delta(id="call-1", index=0, type="function", function=SimpleNamespace(name="web_search", arguments=""))),
        chunk(delta(id=None, index=0, type=None, function=SimpleNamespace(name=None, arguments='{"query":'))),
        chunk(delta(id=None, index=0, type=None, function=SimpleNamespace(name=None, arguments=' "x"}'))),
        chunk(SimpleNamespace(content=None, tool_calls=None), finish="tool_calls"),
    ]

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.execute_tools = False
    p.logger = logging.getLogger(__name__)
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

    # Text streamed before the call is delivered once.
    assert "".join(c.content or "" for c in chunks) == "I'll search."
    calls = [c.tool_calls for c in chunks if c.tool_calls]
    assert calls == [[{"id": "call-1", "type": "function", "name": "web_search", "arguments": '{"query": "x"}'}]]
    assert sum(1 for c in chunks if (c.metadata or {}).get("tool_call_delta")) == 3


class _AnthropicStream:
    def __init__(self, events):
        self._events = list(events)

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *_exc):
        return False


def _anthropic_chunks(events):
    from abstractcore.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")
    provider.client.messages.stream = lambda **_kw: _AnthropicStream(events)
    tools = [
        {"name": "web_search", "description": "Search.", "parameters": {"q": {"type": "string"}}},
        {"name": "now", "description": "Current time.", "parameters": {}},
    ]
    return list(provider.generate(prompt="hi", tools=tools, stream=True, max_output_tokens=64))


def test_anthropic_stream_publishes_each_call_once_and_complete() -> None:
    """Anthropic sends a call's input as `partial_json` pieces; a tool without arguments sends none."""
    ns = SimpleNamespace
    chunks = _anthropic_chunks(
        [
            ns(type="content_block_start", content_block=ns(type="text")),
            ns(type="content_block_delta", delta=ns(text="Searching.")),
            ns(type="content_block_stop"),
            ns(type="content_block_start", content_block=ns(type="tool_use", id="toolu_1", name="web_search")),
            ns(type="content_block_delta", delta=ns(partial_json='{"q')),
            ns(type="content_block_delta", delta=ns(partial_json='": "x"}')),
            ns(type="content_block_stop"),
            ns(type="content_block_start", content_block=ns(type="tool_use", id="toolu_2", name="now")),
            ns(type="content_block_stop"),
            ns(type="message_stop"),
        ]
    )

    assert "".join(c.content or "" for c in chunks) == "Searching."
    calls = [call for c in chunks if c.tool_calls for call in c.tool_calls]
    assert [(c["id"], c["name"], c["arguments"]) for c in calls] == [
        ("toolu_1", "web_search", '{"q": "x"}'),
        ("toolu_2", "now", "{}"),
    ]
    assert sum(1 for c in chunks if (c.metadata or {}).get("tool_call_delta")) == 2


def test_anthropic_call_cut_before_its_input_completes_is_reported() -> None:
    ns = SimpleNamespace
    chunks = _anthropic_chunks(
        [
            ns(type="content_block_start", content_block=ns(type="tool_use", id="toolu_1", name="web_search")),
            ns(type="content_block_delta", delta=ns(partial_json='{"q": "unfin')),
            ns(type="content_block_stop"),
            ns(type="message_stop"),
        ]
    )
    assert all(not c.tool_calls for c in chunks)
    reported = [c.metadata["unparsed_tool_call"] for c in chunks if (c.metadata or {}).get("unparsed_tool_call")]
    assert reported == [{"format": "native", "text": 'web_search {"q": "unfin', "reason": "unclosed"}]
