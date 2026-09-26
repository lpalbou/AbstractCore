"""UnifiedStreamProcessor: raw Harmony (GPT-OSS) transcripts are split per channel WHILE streaming.

Before 2026-09-26 the processor knew `<|channel|>` only as the start of a
`to=<tool>` call, so a plain `<|channel|>final<|message|>answer` was withheld
until the stream ended. Contract: `analysis` -> reasoning (`reasoning_delta`),
`final` and recipient-less `commentary` -> content, `to=` -> tool call, framing
tokens never emitted, tokens cut across chunks handled, and the streamed final
text equals the non-streamed split (`split_harmony_response_text`).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

import pytest

from abstractcore.architectures.response_postprocessing import split_harmony_response_text
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.streaming import UnifiedStreamProcessor

MODEL = "openai/gpt-oss-20b"
TOOLS = [{"name": "get_weather"}, {"name": "search"}, {"name": "list_files"}]
CUTS = [1, 3, 7, 1000]

ANALYSIS_FINAL = (
    "<|channel|>analysis<|message|>User asks 2+2. Simple.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>The answer is 4.<|return|>"
)
PREAMBLE_AND_CALL = (
    "<|channel|>analysis<|message|>Need the weather.<|end|>"
    "<|start|>assistant<|channel|>commentary<|message|>Checking now.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
    '<|message|>{"city": "Paris"}<|call|>'
)


def _cut(text: str, n: int) -> List[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


def _run(parts: List[str], terminal: Optional[GenerateResponse] = None, tools=TOOLS) -> List[GenerateResponse]:
    chunks = [GenerateResponse(content=p, model="m") for p in parts]
    if terminal is not None:
        chunks.append(terminal)
    return list(UnifiedStreamProcessor(model_name=MODEL).process_stream(iter(chunks), tools))


def _content(out) -> str:
    return "".join(c.content or "" for c in out)


def _reasoning(out) -> str:
    return "".join((c.metadata or {}).get("reasoning_delta", "") for c in out)


def _calls(out) -> List[Dict[str, Any]]:
    return [call for c in out for call in (c.tool_calls or [])]


@pytest.mark.parametrize("n", CUTS)
def test_analysis_is_reasoning_and_final_is_content(n):
    # Backends stop at `<|return|>` (an EOS token) and usually drop it; that is
    # the shape the non-streamed split sees, and the parity oracle.
    transcript = ANALYSIS_FINAL[: -len("<|return|>")]
    out = _run(_cut(transcript, n))
    final, reasoning = split_harmony_response_text(transcript)
    assert _content(out) == final == "The answer is 4."
    assert _reasoning(out) == reasoning == "User asks 2+2. Simple."
    assert not _calls(out)
    # A backend that DOES emit the terminator: it is framing, never content.
    assert _content(_run(_cut(ANALYSIS_FINAL, n))) == "The answer is 4."


@pytest.mark.parametrize("n", CUTS)
def test_final_answer_streams_before_the_stream_ends(n):
    """The answer is released as it arrives, not withheld until the end."""
    text = "<|channel|>final<|message|>" + "word " * 40
    processor = UnifiedStreamProcessor(model_name=MODEL)
    released = ""

    def source() -> Iterator[GenerateResponse]:
        nonlocal released
        for part in _cut(text, n):
            yield GenerateResponse(content=part, model="m")
        # Everything but a possible trailing token fragment is out BEFORE the end.
        assert len(released) >= len("word " * 40) - 13

    for c in processor.process_stream(source(), TOOLS):
        released += c.content or ""
    assert released == "word " * 40


@pytest.mark.parametrize("n", CUTS)
def test_preamble_is_content_and_recipient_is_a_tool_call(n):
    out = _run(_cut(PREAMBLE_AND_CALL, n))
    assert _content(out) == "Checking now."
    assert _reasoning(out) == "Need the weather."
    assert _calls(out) == [{"name": "get_weather", "arguments": {"city": "Paris"}, "call_id": None}]


@pytest.mark.parametrize("n", CUTS)
def test_recipient_in_the_role_header(n):
    text = '<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{"q": "x"}<|call|>'
    out = _run(_cut(text, n))
    assert [c["name"] for c in _calls(out)] == ["search"]
    assert _content(out) == ""


@pytest.mark.parametrize("n", CUTS)
def test_framing_tokens_never_reach_content_or_reasoning(n):
    for text in (ANALYSIS_FINAL, PREAMBLE_AND_CALL):
        out = _run(_cut(text, n))
        assert "<|" not in _content(out) and "|>" not in _content(out)
        assert "<|" not in _reasoning(out)


@pytest.mark.parametrize("n", CUTS)
def test_plain_text_without_harmony_framing_is_content(n):
    text = "Just a plain answer, with a < sign and <b>tags</b>."
    assert _content(_run(_cut(text, n))) == text


@pytest.mark.parametrize("n", CUTS)
def test_legacy_inline_call_then_text(n):
    text = 'Let me check. <|channel|>commentary to=list_files <|constrain|>json<|message|>{"directory_path":".","recursive":true}\nDone.'
    out = _run(_cut(text, n))
    assert _calls(out) == [{"name": "list_files", "arguments": {"directory_path": ".", "recursive": True}, "call_id": None}]
    assert _content(out) == "Let me check. \nDone."


@pytest.mark.parametrize("n", CUTS)
def test_unclosed_tool_message_is_reported_not_printed(n):
    text = '<|channel|>commentary to=functions.search <|constrain|>json<|message|>{"q": "x'
    out = _run(_cut(text, n))
    assert _content(out) == ""
    unparsed = [c.metadata["unparsed_tool_call"] for c in out if "unparsed_tool_call" in (c.metadata or {})]
    assert len(unparsed) == 1 and unparsed[0]["format"] == "harmony"
    assert out[-1].metadata["unparsed_tool_call"] == unparsed[0]


@pytest.mark.parametrize("n", CUTS)
def test_terminal_chunk_stays_last_and_keeps_its_accounting(n):
    text = "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>Done <"
    terminal = GenerateResponse(content="", model="m", finish_reason="length", usage={"output_tokens": 9},
                                metadata={"prompt_cache": {"key": "k"}})
    out = _run(_cut(text, n), terminal=terminal)
    assert _content(out) == "Done <"
    assert out[-1].finish_reason == "length"
    assert out[-1].usage == {"output_tokens": 9}
    assert out[-1].metadata["prompt_cache"] == {"key": "k"}
    assert all(c.finish_reason is None for c in out[:-1])


def test_non_harmony_models_are_untouched():
    text = "<|channel|>final<|message|>hi"
    out = list(UnifiedStreamProcessor(model_name="qwen3-4b").process_stream(
        iter([GenerateResponse(content=text, model="m")]), TOOLS))
    assert _content(out) == text


def test_base_provider_aggregates_harmony_reasoning_on_the_trailing_chunk():
    from abstractcore.providers.base import BaseProvider

    class _Stub(BaseProvider):
        def get_capabilities(self):
            return ["streaming"]

        def list_available_models(self, **kwargs):
            return [self.model]

        def unload_model(self, model_name):
            return None

        def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None,
                               stream=False, response_model=None, execute_tools=None, media_metadata=None,
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
            def _gen():
                for part in _cut(ANALYSIS_FINAL, 5):
                    yield GenerateResponse(content=part, model=self.model)
                yield GenerateResponse(content="", model=self.model, finish_reason="stop")
            return _gen()

    chunks = list(_Stub(model=MODEL).generate("2+2?", stream=True))
    assert "".join(c.content or "" for c in chunks) == "The answer is 4."
    reasoning = [c.metadata["reasoning"] for c in chunks if "reasoning" in (c.metadata or {})]
    assert reasoning[-1] == "User asks 2+2. Simple."
