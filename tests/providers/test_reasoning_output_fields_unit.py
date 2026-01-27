from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import extract_reasoning_from_message


def test_glm47_flash_reasoning_content_is_extracted_via_architecture_config():
    model = "zai-org/glm-4.7-flash"
    arch = detect_architecture(model)
    arch_fmt = get_architecture_format(arch)
    caps = get_model_capabilities(model)

    msg = {"role": "assistant", "content": "final", "reasoning_content": "r"}
    reasoning = extract_reasoning_from_message(
        msg,
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert reasoning == "r"


def test_qwen3_thinking_2507_reasoning_content_is_extracted_via_model_capabilities():
    model = "qwen/qwen3-4b-thinking-2507"
    arch = detect_architecture(model)
    arch_fmt = get_architecture_format(arch)
    caps = get_model_capabilities(model)

    msg = {"role": "assistant", "content": "final", "reasoning_content": "r"}
    reasoning = extract_reasoning_from_message(
        msg,
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert reasoning == "r"

