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


def test_incremental_thinking_tag_stripper_rolls_back_unclosed_block() -> None:
    stripper = IncrementalThinkingTagStripper(start_tag="<think>", end_tag="</think>")

    out = ""
    out += stripper.process("Hello <think>unfinished")
    tail, reasoning = stripper.finalize()
    out += tail

    assert out == "Hello <think>unfinished"
    assert reasoning is None

