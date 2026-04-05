from abstractcore.architectures import detect_architecture, get_model_capabilities


def test_variant_suffix_normalization_resolves_qwen_liquid_and_gpt_families() -> None:
    qwen = get_model_capabilities("mlx-community/Qwen3.5-9B-4bit")
    assert qwen.get("architecture") == "qwen3_5"
    assert qwen.get("max_tokens") == 262144

    qwen_gguf = get_model_capabilities("unsloth/Qwen3.5-9B-Q4_K_M.gguf")
    assert qwen_gguf.get("architecture") == "qwen3_5"
    assert qwen_gguf.get("max_tokens") == 262144

    qwen_frontier = get_model_capabilities("Qwen/Qwen3.5-122B-A10B-fp16")
    assert qwen_frontier.get("architecture") == "qwen3_5"
    assert qwen_frontier.get("total_parameters") == "122B"

    liquid = get_model_capabilities("LiquidAI/LFM2-24B-A2B-GGUF")
    assert liquid.get("architecture") == "lfm2"
    assert liquid.get("max_tokens") == 32768
    assert liquid.get("tool_support") == "prompted"

    liquid_q8 = get_model_capabilities("LiquidAI/LFM2-24B-A2B-Q8_0")
    assert liquid_q8.get("architecture") == "lfm2"
    assert liquid_q8.get("max_tokens") == 32768

    liquid_mlx = get_model_capabilities("mlx-community/LFM2.5-1.2B-Instruct-8bit")
    assert liquid_mlx.get("architecture") == "lfm2"
    assert liquid_mlx.get("structured_output") == "prompted"

    gpt = get_model_capabilities("openai/gpt-5.4-pro-fp16")
    assert gpt.get("architecture") == "gpt"
    assert gpt.get("max_tokens") == 1050000
    assert gpt.get("structured_output") == "prompted"


def test_new_family_entries_are_resolved_with_expected_architectures() -> None:
    assert detect_architecture("MiniMaxAI/MiniMax-M2.5") == "minimax_m2"
    assert detect_architecture("zai-org/GLM-5") == "glm4_moe"
    assert detect_architecture("Qwen/Qwen3-Coder-Next") == "qwen3_next"
    assert detect_architecture("LiquidAI/LFM2-24B-A2B-GGUF") == "lfm2"


def test_new_model_entries_expose_verified_capabilities() -> None:
    minimax = get_model_capabilities("MiniMaxAI/MiniMax-M2.5")
    assert minimax.get("architecture") == "minimax_m2"
    assert minimax.get("max_tokens") == 196608
    assert minimax.get("max_output_tokens") == 8192
    assert minimax.get("total_parameters") == "480B"
    assert minimax.get("active_parameters") == "40B"
    assert minimax.get("tool_support") == "native"

    glm = get_model_capabilities("zai-org/GLM-5")
    assert glm.get("architecture") == "glm4_moe"
    assert glm.get("max_tokens") == 128000
    assert glm.get("max_output_tokens") == 96000
    assert glm.get("tool_support") == "prompted"

    coder_next = get_model_capabilities("qwen/qwen3-coder-next-gguf")
    assert coder_next.get("architecture") == "qwen3_next"
    assert coder_next.get("max_tokens") == 262144
    assert coder_next.get("tool_support") == "native"
