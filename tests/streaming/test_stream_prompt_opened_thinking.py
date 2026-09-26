"""Inline thinking streams AS IT IS GENERATED.

Live finding (gateway, Qwen3.8-27B on the MLX native lane): 26k characters of
reasoning were never streamed and the first delta arrived after 654 s, because
(1) the chat template ends the prompt with `<think>\\n`, so the model writes
only the CLOSING tag, and the incremental stripper held everything until that
tag arrived ("closing-only" case), and (2) the stripper only ever returned the
reasoning as the final aggregate, never per chunk.

Contract: a provider whose rendered prompt opened the block says so on a leading
empty chunk (`THINKING_OPENED_BY_PROMPT`); reasoning then streams as
`metadata["reasoning_delta"]` from the first token, the answer as content after
the closing tag, and the final record equals the non-streamed one.
"""

from __future__ import annotations

from typing import Iterator, List

import pytest

from abstractcore.architectures.response_postprocessing import (
    THINKING_OPENED_BY_PROMPT,
    IncrementalThinkingTagStripper,
    prompt_opens_thinking,
)
from abstractcore.core.types import GenerateResponse

MODEL = "qwen3-4b"  # an architecture with <think>...</think> tags
THINKING = "Let me count. s-t-r-a-w-b-e-r-r-y has three r letters."
ANSWER = "There are 3 r's in strawberry."
RAW = f"{THINKING}\n</think>\n\n{ANSWER}"


def _cut(text: str, n: int) -> List[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


@pytest.mark.parametrize("n", [1, 3, 7, 1000])
def test_stripper_streams_reasoning_deltas_once_the_prompt_opened_thinking(n):
    s = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")
    s.open_thinking()
    visible, deltas = [], []
    for part in _cut(RAW, n):
        visible.append(s.process(part))
        deltas.append(s.take_reasoning_delta())
    tail, reasoning = s.finalize()
    visible.append(tail)
    assert "".join(visible) == ANSWER  # no leading blank lines, like the sync path
    assert "".join(deltas).strip() == THINKING == reasoning
    # The first delta arrives with the first chunk, not after `</think>`.
    assert deltas[0]


def test_stripper_streams_explicit_think_blocks_too():
    s = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")
    out = [s.process("<think>abc"), s.take_reasoning_delta(), s.process("def</think>ok"), s.take_reasoning_delta()]
    assert out == ["", "abc", "ok", "def"]


def test_without_the_hint_the_closing_only_case_is_unchanged():
    s = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")
    assert s.process(THINKING) == ""  # ambiguous: held
    assert s.take_reasoning_delta() == ""
    assert s.process("</think>\n\n" + ANSWER) == ANSWER
    assert s.take_reasoning_delta() == THINKING


def test_prompt_opens_thinking_reads_the_rendered_prompt():
    from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities

    fmt = get_architecture_format(detect_architecture(MODEL))
    caps = get_model_capabilities(MODEL)
    assert prompt_opens_thinking("<|im_start|>assistant\n<think>\n", architecture_format=fmt, model_capabilities=caps)
    assert not prompt_opens_thinking("<|im_start|>assistant\n", architecture_format=fmt, model_capabilities=caps)
    assert not prompt_opens_thinking(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n", architecture_format=fmt, model_capabilities=caps
    )
    assert not prompt_opens_thinking([1, 2, 3], architecture_format=fmt, model_capabilities=caps)


def _stub(n: int, log: List[str]):
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
                               **kwargs):
            if not stream:
                return GenerateResponse(content=RAW, model=self.model, finish_reason="stop",
                                        usage={"output_tokens": 20})

            def _gen() -> Iterator[GenerateResponse]:
                yield GenerateResponse(content="", model=self.model, metadata={THINKING_OPENED_BY_PROMPT: True})
                for part in _cut(RAW, n):
                    log.append(f"produced:{part}")
                    yield GenerateResponse(content=part, model=self.model)
                yield GenerateResponse(content="", model=self.model, finish_reason="stop",
                                       usage={"output_tokens": 20})
            return _gen()

    return _Stub(model=MODEL)


@pytest.mark.parametrize("n", [1, 3, 7, 1000])
def test_provider_stream_shows_reasoning_before_the_closing_tag_and_matches_sync(n):
    log: List[str] = []
    chunks = []
    first_reasoning_at = None
    for c in _stub(n, log).generate("q", stream=True, thinking="on"):
        chunks.append(c)
        if first_reasoning_at is None and (c.metadata or {}).get("reasoning_delta"):
            first_reasoning_at = len(log)
            assert "</think>" not in "".join(p[len("produced:"):] for p in log) or n >= len(RAW)
    assert first_reasoning_at == 1 or n >= len(RAW)
    assert all(THINKING_OPENED_BY_PROMPT not in (c.metadata or {}) for c in chunks)
    streamed_reasoning = [c.metadata["reasoning"] for c in chunks if "reasoning" in (c.metadata or {})][-1]
    sync = _stub(n, []).generate("q", thinking="on")
    assert "".join(c.content or "" for c in chunks) == sync.content == ANSWER
    assert streamed_reasoning == sync.metadata["reasoning"] == THINKING
    assert "".join((c.metadata or {}).get("reasoning_delta", "") for c in chunks).strip() == THINKING


# --- review 23: parity of the streamed stripper with the non-streamed split ---------

from abstractcore.architectures.response_postprocessing import strip_thinking_tags  # noqa: E402

TAGS = dict(start_tag="<think>", end_tag="</think>")


def _fmt():
    from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities

    return dict(architecture_format=get_architecture_format(detect_architecture(MODEL)),
                model_capabilities=get_model_capabilities(MODEL))


def _stream_split(text: str, n: int, opened: bool = False):
    s = IncrementalThinkingTagStripper(**TAGS)
    if opened:
        s.open_thinking()
    visible = "".join(s.process(p) for p in _cut(text, n))
    tail, reasoning = s.finalize()
    return visible + tail, reasoning


@pytest.mark.parametrize("n", [1, 3, 7, 1000])
@pytest.mark.parametrize("text, opened", [
    ("Answer: <think>r</think> 42", False),                         # whitespace mid-answer is kept
    ("<think>r1</think>A1\n<think>r2</think>\n\nB2", False),        # a second block keeps the blank line
    ("<think>r</think>\nFinal", False),                             # the ANSWER start drops it
    ("reasoning cut off by the output lim", True),                  # truncated prompt-opened reply
    ("r1</think>\n\nA1 <think>r2</think> B2", True),                # closing-only + a later block
    ("<think>abc</think>\n\nx", True),                              # the model repeats the opened tag
])
def test_streamed_split_equals_the_non_streamed_split(n, text, opened):
    streamed = _stream_split(text, n, opened)
    whole = strip_thinking_tags(text, opened_by_prompt=opened, **_fmt())
    assert streamed == whole, (streamed, whole)
    assert "<think>" not in streamed[0] + (streamed[1] or "")
    assert "</think>" not in streamed[0] + (streamed[1] or "")


def test_truncated_prompt_opened_reply_is_reasoning_marked_truncated():
    content, reasoning = strip_thinking_tags("half a thought", opened_by_prompt=True, **_fmt())
    assert content == "" and reasoning == "half a thought (...)"
