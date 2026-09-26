from abstractcore.core.types import GenerateResponse
from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import (
    maybe_extract_harmony_final_text,
    split_harmony_response_text,
)


def test_split_harmony_response_extracts_final_and_reasoning():
    text = (
        "<|channel|>analysis<|message|>reasoning text<|end|>"
        "<|start|>assistant<|channel|>final<|message|>final text<|end|>"
    )
    final_text, reasoning = split_harmony_response_text(text)
    assert final_text == "final text"
    assert reasoning == "reasoning text"


def test_split_harmony_response_returns_reasoning_when_truncated_before_final():
    text = "<|channel|>analysis<|message|>partial reasoning"
    final_text, reasoning = split_harmony_response_text(text)
    assert final_text is None
    assert reasoning == "partial reasoning"


def test_maybe_extract_harmony_final_cleans_content_and_sets_metadata_reasoning():
    resp = GenerateResponse(
        content="<|channel|>analysis<|message|>r<|end|><|start|>assistant<|channel|>final<|message|>f<|end|>",
        model="gpt-oss-20b",
        finish_reason="stop",
    )
    arch_fmt = get_architecture_format(detect_architecture("gpt-oss-20b"))
    caps = get_model_capabilities("gpt-oss-20b")
    cleaned, reasoning = maybe_extract_harmony_final_text(
        resp.content or "",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == "f"
    assert reasoning == "r"


def test_maybe_extract_harmony_truncated_before_final_is_reasoning_not_the_answer():
    resp = GenerateResponse(
        content="<|channel|>analysis<|message|>partial",
        model="gpt-oss-20b",
        finish_reason="length",
    )
    arch_fmt = get_architecture_format(detect_architecture("gpt-oss-20b"))
    caps = get_model_capabilities("gpt-oss-20b")
    cleaned, reasoning = maybe_extract_harmony_final_text(
        resp.content or "",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    # The model's private, mid-sentence thinking is never presented as its reply.
    assert cleaned == ""
    assert reasoning == "partial (...)"


def test_split_harmony_response_never_keeps_return_or_call_tokens():
    for token in ("<|return|>", "<|call|>"):
        text = f"<|channel|>analysis<|message|>r<|end|><|start|>assistant<|channel|>final<|message|>Hello there.{token}"
        assert split_harmony_response_text(text) == ("Hello there.", "r")
