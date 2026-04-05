from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import normalize_assistant_text


def test_gemma4_architecture_detected_for_common_variants() -> None:
    variants = [
        ("google/gemma-4-31B-it", 262144, False),
        ("gemma-4-31B-it", 262144, False),
        ("models--google--gemma-4-26B-A4B-it", 262144, False),
        ("google/gemma-4-E2B-it", 131072, True),
        ("google/gemma-4-E4B", 131072, True),
    ]

    for model, expected_max_tokens, expected_audio in variants:
        assert detect_architecture(model) == "gemma4"
        caps = get_model_capabilities(model)
        assert caps.get("architecture") == "gemma4"
        assert caps.get("max_tokens") == expected_max_tokens
        assert caps.get("tool_support") == "native"
        assert caps.get("vision_support") is True
        assert caps.get("audio_support") is expected_audio
        assert caps.get("video_support") is True


def test_gemma4_thinking_tags_are_stripped_and_reasoning_returned() -> None:
    model = "google/gemma-4-31B-it"
    arch_fmt = get_architecture_format(detect_architecture(model))
    caps = get_model_capabilities(model)

    cleaned, reasoning = normalize_assistant_text(
        "<|channel>thought\nreasoning text<channel|>Final answer.",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == "Final answer."
    assert reasoning == "reasoning text"

    cleaned, reasoning = normalize_assistant_text(
        "<|channel>thought\n<channel|>Final answer.",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == "Final answer."
    assert reasoning is None

