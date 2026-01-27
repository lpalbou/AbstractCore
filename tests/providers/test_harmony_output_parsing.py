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


def test_maybe_extract_harmony_final_strips_wrapper_tokens_when_only_analysis_present():
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
    assert cleaned == "partial"
    assert reasoning == "partial"
