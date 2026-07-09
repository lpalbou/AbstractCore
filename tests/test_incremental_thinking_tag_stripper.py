from __future__ import annotations

from abstractcore.architectures.response_postprocessing import IncrementalThinkingTagStripper


def test_incremental_thinking_tag_stripper_explicit_block_across_chunks() -> None:
    stripper = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")

    out = ""
    out += stripper.process("<thi")
    out += stripper.process("nk>hello")
    out += stripper.process("</th")
    out += stripper.process("ink>\nFinal")

    tail, reasoning = stripper.finalize()
    out += tail

    assert "<think>" not in out
    assert "</think>" not in out
    assert out == "\nFinal"
    assert reasoning == "hello"


def test_incremental_thinking_tag_stripper_closing_only_mode() -> None:
    stripper = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")

    out = ""
    out += stripper.process("reasoning ")
    out += stripper.process("text</think>Answer")
    tail, reasoning = stripper.finalize()
    out += tail

    assert out == "Answer"
    assert reasoning == "reasoning text"


def test_incremental_thinking_tag_stripper_auto_closes_unclosed_block() -> None:
    # Truncated streams (e.g. finish_reason=length) auto-close the thinking block and
    # capture it as reasoning with a truncation marker instead of leaking it as content.
    stripper = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")

    out = ""
    out += stripper.process("Hello <think>unfinished")
    tail, reasoning = stripper.finalize()
    out += tail

    assert out == "Hello "
    assert reasoning == "unfinished (...)"


def test_incremental_thinking_tag_stripper_assume_visible_start_streams_immediately() -> None:
    # With thinking effectively off, tagless streamed content must flow through
    # incrementally instead of being buffered until finalize().
    stripper = IncrementalThinkingTagStripper(
        start_tag="<think>", end_tag="</think>", assume_visible_start=True
    )

    first = stripper.process("Hello ")
    second = stripper.process("world")
    tail, reasoning = stripper.finalize()

    assert first == "Hello "
    assert second == "world"
    assert tail == ""
    assert reasoning is None


def test_incremental_thinking_tag_stripper_assume_visible_start_still_strips_full_blocks() -> None:
    stripper = IncrementalThinkingTagStripper(
        start_tag="<think>", end_tag="</think>", assume_visible_start=True
    )

    out = ""
    out += stripper.process("Answer: <think>internal")
    out += stripper.process(" reasoning</think> 42")
    tail, reasoning = stripper.finalize()
    out += tail

    assert out == "Answer:  42"
    assert reasoning == "internal reasoning"

