from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import normalize_assistant_text


def test_sera_architecture_detected_for_common_variants() -> None:
    variants = [
        "allenai/SERA-32B",
        "huggingface/allenai/SERA-32B",
        "SERA-32B",
        "sera-32b",
        "sera32b",
        "sera_32b",
        "models--allenai--SERA-32B",
        "allenai/SERA-32B-GA",
        "huggingface/allenai/SERA-32B-GA",
        "SERA-32B-GA",
        "models--allenai--SERA-32B-GA",
        "allenai/SERA-8B",
        "huggingface/allenai/SERA-8B",
        "SERA-8B",
        "models--allenai--SERA-8B",
        "allenai/SERA-8B-GA",
        "huggingface/allenai/SERA-8B-GA",
        "SERA-8B-GA",
        "models--allenai--SERA-8B-GA",
    ]

    for model in variants:
        assert detect_architecture(model) == "sera"
        caps = get_model_capabilities(model)
        assert caps.get("architecture") == "sera"
        assert caps.get("max_tokens") == 32768
        assert caps.get("tool_support") == "prompted"


def test_sera_thinking_tags_are_stripped_and_reasoning_returned() -> None:
    model = "allenai/SERA-32B"
    arch_fmt = get_architecture_format(detect_architecture(model))
    caps = get_model_capabilities(model)

    cleaned, reasoning = normalize_assistant_text(
        "<think>reasoning</think>\n\nFinal answer.",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == "Final answer."
    assert reasoning == "reasoning"


def test_sera_thinking_end_tag_only_is_handled() -> None:
    model = "allenai/SERA-32B"
    arch_fmt = get_architecture_format(detect_architecture(model))
    caps = get_model_capabilities(model)

    cleaned, reasoning = normalize_assistant_text(
        "reasoning only</think>\n\nFinal answer.",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == "Final answer."
    assert reasoning == "reasoning only"
