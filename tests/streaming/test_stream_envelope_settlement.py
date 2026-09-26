"""UnifiedStreamProcessor: how the stream ENDS.

Three contracts (mission S-core, 2026-09-26):

1. An envelope that opens and never closes (or closes around a body no parser
   accepts) is reported as `metadata["unparsed_tool_call"]` on the final chunk,
   with a warning, and NEVER printed as content (it used to be re-emitted as
   content, together with a duplicate of the text before it).
2. A ```json fenced block whose body is a tool call (`{"name", "arguments"}`,
   `{"tool_calls": [...]}`, a list of calls, the OpenAI `function` shape) is a
   tool-call envelope when the request carries tools: withheld from content and
   parsed. Any other ```json block is released verbatim; without tools, every
   ```json block is content.
3. The provider's terminal chunk (finish_reason / usage / prompt-cache
   telemetry) stays the LAST chunk: held-back content is settled before it and
   its finish_reason is not overwritten by a synthetic "stop".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.streaming import UnifiedStreamProcessor

TOOLS = [
    {"name": "get_weather", "description": "weather", "parameters": {"type": "object"}},
    {"name": "search", "description": "search", "parameters": {"type": "object"}},
]


def _run(
    parts: List[str],
    *,
    tools: Optional[List[Dict[str, Any]]] = TOOLS,
    terminal: Optional[GenerateResponse] = None,
    model: str = "qwen3-4b",
) -> List[GenerateResponse]:
    chunks = [GenerateResponse(content=p, model="m") for p in parts]
    if terminal is not None:
        chunks.append(terminal)
    processor = UnifiedStreamProcessor(model_name=model)
    return list(processor.process_stream(iter(chunks), tools))


def _content(out: List[GenerateResponse]) -> str:
    return "".join(c.content or "" for c in out)


def _calls(out: List[GenerateResponse]) -> List[Dict[str, Any]]:
    return [call for c in out for call in (c.tool_calls or [])]


def _meta(out: List[GenerateResponse], key: str) -> List[Any]:
    return [c.metadata[key] for c in out if isinstance(c.metadata, dict) and key in c.metadata]


def _char_split(text: str) -> List[str]:
    return list(text)


# --- 1. unclosed / unparsable envelopes -------------------------------------


@pytest.mark.parametrize(
    "parts",
    [
        ["Hello <tool", '_call>{"name": "search", "argu'],
        _char_split('Hello <tool_call>{"name": "search", "argu'),
        ["Hello <|tool_call|>", "not json at all"],
    ],
    ids=["split-marker", "char-by-char", "qwen-garbage"],
)
def test_unclosed_envelope_is_reported_not_printed(parts, caplog):
    with caplog.at_level(logging.WARNING):
        out = _run(parts)
    assert _content(out) == "Hello "
    unparsed = _meta(out, "unparsed_tool_call")
    assert len(unparsed) == 1
    assert unparsed[0]["reason"] == "unclosed"
    assert unparsed[0]["text"] == "".join(parts)[len("Hello ") :]
    assert out[-1].metadata["unparsed_tool_call"] == unparsed[0]
    assert any("Unparsed tool call" in w for w in out[-1].metadata["warnings"])
    assert not _calls(out)


def test_unclosed_envelope_rides_the_provider_terminal_chunk():
    terminal = GenerateResponse(
        content="", model="m", finish_reason="length", usage={"output_tokens": 9},
        metadata={"prompt_cache": {"mode": "key", "key": "k"}},
    )
    out = _run(["Answer: <tool_call>", '{"name": "search", "arguments": {"q"'], terminal=terminal)
    assert _content(out) == "Answer: "
    last = out[-1]
    assert last.finish_reason == "length"
    assert last.usage == {"output_tokens": 9}
    assert last.metadata["prompt_cache"] == {"mode": "key", "key": "k"}
    assert last.metadata["unparsed_tool_call"]["text"].startswith("<tool_call>")
    assert [c.finish_reason for c in out[:-1]] == [None] * (len(out) - 1)


def test_truncated_but_complete_json_in_unclosed_envelope_still_parses():
    out = _run(["<tool_call>", '{"name": "search", "arguments": {"q": "x"}}'])
    assert [c["name"] for c in _calls(out)] == ["search"]
    assert not _meta(out, "unparsed_tool_call")
    assert _content(out) == ""


# --- 2. ```json fenced tool calls --------------------------------------------

FENCED = 'Let me check.\n```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```\nDone.'


@pytest.mark.parametrize(
    "parts",
    [
        [FENCED],
        _char_split(FENCED),
        ["Let me check.\n``", "`js", "on\n{\"name\": \"get_weather\", ", "\"arguments\": {\"city\": \"Paris\"}}\n`", "``\nDone."],
    ],
    ids=["one-chunk", "char-by-char", "split-fences"],
)
def test_json_fenced_tool_call_is_withheld_and_parsed(parts):
    out = _run(parts)
    assert _calls(out) == [{"name": "get_weather", "arguments": {"city": "Paris"}, "call_id": None}]
    text = _content(out)
    assert "```" not in text and "get_weather" not in text
    assert text == "Let me check.\n\nDone."


@pytest.mark.parametrize(
    "body, names",
    [
        ('{"tool_calls": [{"name": "search", "arguments": {"q": "a"}}, {"name": "get_weather", "arguments": {}}]}', ["search", "get_weather"]),
        ('[{"name": "search", "arguments": {"q": "a"}}]', ["search"]),
        ('{"type": "function", "function": {"name": "search", "arguments": "{\\"q\\": \\"a\\"}"}}', ["search"]),
        ('{"name": "search", "parameters": {"q": "a"}}', ["search"]),
    ],
    ids=["tool_calls-array", "bare-list", "openai-function", "parameters-key"],
)
def test_json_fence_accepts_the_call_shapes(body, names):
    out = _run(["```json\n", body, "\n```"])
    assert [c["name"] for c in _calls(out)] == names
    assert _content(out) == ""


@pytest.mark.parametrize(
    "body",
    ['{"name": "Alice", "age": 3}', '[1, 2, 3]', '{"city": "Paris"}', "not json"],
)
def test_json_fence_that_is_not_a_tool_call_is_released_verbatim(body):
    text = f"Here:\n```json\n{body}\n```\nbye"
    out = _run(_char_split(text))
    assert _content(out) == text
    assert not _calls(out)
    assert not _meta(out, "unparsed_tool_call")


@pytest.mark.parametrize(
    "body",
    [
        '{"name": "alice", "parameters": {"age": 3}}',
        '{"name": "alice", "arguments": {}}',
        '{"tool_calls": [{"name": "search", "arguments": {}}, {"name": "delete_everything", "arguments": {}}]}',
    ],
    ids=["answer-shaped-like-a-call", "unknown-tool", "one-unknown-in-array"],
)
def test_json_fence_naming_a_tool_that_was_not_offered_is_content_verbatim(body):
    """A model asked to return JSON must never lose its answer to the tool parser."""
    text = f"Here you go:\n```json\n{body}\n```\n"
    for parts in ([text], _char_split(text)):
        out = _run(parts)
        assert _content(out) == text
        assert not _calls(out)
        assert not _meta(out, "unparsed_tool_call")


def test_json_fence_accepts_a_namespaced_spelling_of_an_offered_tool():
    out = _run(['```json\n{"name": "functions.search", "arguments": {"q": "a"}}\n```'])
    assert [c["name"] for c in _calls(out)] == ["search"]
    assert _content(out) == ""


def test_unclosed_json_fence_naming_an_unknown_tool_stays_content():
    out = _run(["```json\n", '{"name": "alice", "argu'])
    assert _content(out) == '```json\n{"name": "alice", "argu'
    assert not _meta(out, "unparsed_tool_call")


def test_json_fence_is_content_when_the_request_has_no_tools():
    out = _run([FENCED], tools=None)
    assert _content(out) == FENCED
    assert not _calls(out)


def test_unclosed_json_fence_answer_stays_content_but_broken_call_is_reported():
    answer = _run(["```json\n", '{"city": "Paris", "temp"'])
    assert _content(answer) == '```json\n{"city": "Paris", "temp"'
    assert not _meta(answer, "unparsed_tool_call")

    broken = _run(["```json\n", '{"name": "search", "argu'])
    assert _content(broken) == ""
    assert _meta(broken, "unparsed_tool_call")[0]["format"] == "json_fence"


# --- 3. terminal chunk stays last --------------------------------------------


def test_held_back_content_is_settled_before_the_terminal_chunk():
    terminal = GenerateResponse(
        content="", model="m", finish_reason="length", usage={"output_tokens": 3},
        metadata={"prompt_cache": {"mode": "key", "key": "k", "outcome": "cold"}},
    )
    out = _run(["Hi there <b>x", "yz"], terminal=terminal)
    assert _content(out) == "Hi there <b>xyz"
    assert out[-1] is terminal or out[-1].finish_reason == "length"
    assert out[-1].usage == {"output_tokens": 3}
    assert out[-1].metadata["prompt_cache"]["outcome"] == "cold"
    assert all(c.finish_reason is None for c in out[:-1])


def test_terminal_chunk_with_content_is_split_when_the_detector_holds_text():
    terminal = GenerateResponse(
        content=" and <", model="m", finish_reason="stop", usage={"output_tokens": 2},
        metadata={"prompt_cache": {"key": "k"}},
    )
    out = _run(["a", "b"], terminal=terminal)
    assert _content(out) == "ab and <"
    assert out[-1].content == "" and out[-1].finish_reason == "stop"
    assert out[-1].metadata["prompt_cache"] == {"key": "k"}
    assert all(c.finish_reason is None for c in out[:-1])


# --- 4. BaseProvider's trailing finalize chunk re-carries prompt_cache --------


def test_base_trailing_reasoning_chunk_recarries_prompt_cache():
    """The reasoning-aggregate chunk BaseProvider appends is built from scratch and
    becomes the stream's last chunk; it must still carry the provider's
    per-request prompt-cache record (as it does usage and media delivery)."""
    from typing import Iterator, Union

    from abstractcore.providers.base import BaseProvider

    class _Stub(BaseProvider):
        def get_capabilities(self):
            return ["streaming"]

        def list_available_models(self, **kwargs):
            return [self.model]

        def unload_model(self, model_name):
            return None

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None,
                               media=None, stream=False, response_model=None, execute_tools=None,
                               media_metadata=None, **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
            def _gen():
                yield GenerateResponse(content="", model=self.model, metadata={"reasoning_delta": "hmm"})
                yield GenerateResponse(content="Answer", model=self.model)
                yield GenerateResponse(
                    content="", model=self.model, finish_reason="stop", usage={"output_tokens": 2},
                    metadata={"prompt_cache": {"mode": "key", "key": "k", "outcome": "cold"}},
                )
            return _gen()

    provider = _Stub(model="unit-test")
    provider.model_capabilities = {"thinking_support": True}
    chunks = list(provider.generate("q", stream=True))
    last = chunks[-1]
    assert last.metadata["reasoning"] == "hmm"
    assert last.metadata["prompt_cache"] == {"mode": "key", "key": "k", "outcome": "cold"}
    assert last.usage == {"output_tokens": 2}
