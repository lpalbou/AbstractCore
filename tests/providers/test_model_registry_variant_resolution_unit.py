from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities


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


def test_precision_and_packaging_variants_resolve_to_the_right_family_entries() -> None:
    nemotron_nvfp4 = get_model_capabilities("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    assert nemotron_nvfp4.get("architecture") == "nemotron_hybrid_moe"
    assert nemotron_nvfp4.get("quantization_method") == "NVFP4"
    assert nemotron_nvfp4.get("max_tokens") == 1000000

    nemotron_fp16 = get_model_capabilities("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP16")
    assert nemotron_fp16.get("architecture") == "nemotron_hybrid_moe"
    assert nemotron_fp16.get("max_tokens") == 1000000

    nemotron_fp8 = get_model_capabilities("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
    assert nemotron_fp8.get("architecture") == "nemotron_hybrid_moe"
    assert nemotron_fp8.get("max_tokens") == 1000000

    omnicoder_gguf = get_model_capabilities("Tesslate/OmniCoder-9B-Q4_K_M-GGUF")
    assert omnicoder_gguf.get("architecture") == "omnicoder"
    assert omnicoder_gguf.get("max_tokens") == 8192
    assert omnicoder_gguf.get("max_output_tokens") == 4096

    omnicoder_bf16 = get_model_capabilities("Tesslate/OmniCoder-9B-BF16")
    assert omnicoder_bf16.get("architecture") == "omnicoder"
    assert omnicoder_bf16.get("max_tokens") == 262144
    assert omnicoder_bf16.get("max_output_tokens") == 81920


def test_new_family_entries_are_resolved_with_expected_architectures() -> None:
    assert detect_architecture("MiniMaxAI/MiniMax-M2.5") == "minimax_m2"
    assert detect_architecture("zai-org/GLM-5") == "glm4_moe"
    assert detect_architecture("Qwen/Qwen3-Coder-Next") == "qwen3_next"
    assert detect_architecture("LiquidAI/LFM2-24B-A2B-GGUF") == "lfm2"
    assert detect_architecture("Tesslate/OmniCoder-9B") == "omnicoder"
    assert detect_architecture("Qwen/Qwen3.6-35B-A3B") == "qwen3_6"
    assert detect_architecture("moonshotai/Kimi-K2.6") == "kimi_k2"
    assert detect_architecture("deepseek-ai/DeepSeek-V4-Pro") == "deepseek_v4"


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

    minimax_m2 = get_model_capabilities("MiniMaxAI/MiniMax-M2")
    assert minimax_m2.get("architecture") == "minimax_m2"
    assert minimax_m2.get("max_tokens") == 208896
    assert minimax_m2.get("tool_support") == "native"

    minimax_m21 = get_model_capabilities("MiniMaxAI/MiniMax-M2.1")
    assert minimax_m21.get("architecture") == "minimax_m2_1"
    assert minimax_m21.get("max_tokens") == 204800
    assert minimax_m21.get("tool_support") == "native"

    omnicoder = get_model_capabilities("Tesslate/OmniCoder-9B")
    assert omnicoder.get("architecture") == "omnicoder"
    assert omnicoder.get("max_tokens") == 262144
    assert omnicoder.get("max_output_tokens") == 81920
    assert omnicoder.get("tool_support") == "prompted"
    assert omnicoder.get("thinking_support") is True
    assert omnicoder.get("vision_support") is True
    assert omnicoder.get("agentic_coding") is True


def test_recent_mistral_granite_and_nemotron_variants_resolve_to_expected_families() -> None:
    assert detect_architecture("mistralai/Mistral-Small-4-119B-2603") == "mistral3"
    assert detect_architecture("mistralai/Ministral-3-14B-Instruct-2512") == "ministral3"
    assert detect_architecture("ibm-granite/granite-4.0-micro") == "granitemoehybrid"
    assert detect_architecture("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4") == "nemotron_hybrid_moe"

    large3 = get_model_capabilities("mistralai/Mistral-Large-3-675B-Instruct-2512")
    assert large3.get("architecture") == "mistral_large"
    assert large3.get("max_tokens") == 262144
    assert large3.get("vision_support") is True

    ministral = get_model_capabilities("mistralai/Ministral-3-14B-Instruct-2512")
    assert ministral.get("architecture") == "ministral3"
    assert ministral.get("vision_support") is True
    assert ministral.get("tool_support") == "native"

    granite = get_model_capabilities("ibm-granite/granite-4.0-h-small")
    assert granite.get("architecture") == "granitemoehybrid"
    assert granite.get("max_tokens") == 131072

    nemotron = get_model_capabilities("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    assert nemotron.get("architecture") == "nemotron_hybrid_moe"
    assert nemotron.get("quantization_method") == "NVFP4"
    assert nemotron.get("max_tokens") == 1000000


def test_2026_frontier_model_entries_expose_expected_capabilities() -> None:
    qwen = get_model_capabilities("Qwen/Qwen3.6-35B-A3B")
    assert qwen.get("architecture") == "qwen3_6"
    assert qwen.get("max_tokens") == 262144
    assert qwen.get("vision_support") is True
    assert qwen.get("video_support") is True

    kimi = get_model_capabilities("moonshotai/Kimi-K2.6")
    assert kimi.get("architecture") == "kimi_k2"
    assert kimi.get("max_tokens") == 262144
    assert kimi.get("vision_support") is True
    assert kimi.get("thinking_output_field") == "reasoning"

    deepseek = get_model_capabilities("deepseek-ai/DeepSeek-V4-Pro")
    assert deepseek.get("architecture") == "deepseek_v4"
    assert deepseek.get("max_tokens") == 1048576
    assert deepseek.get("total_parameters") == "1.6T"

    granite = get_model_capabilities("ibm-granite/granite-4.1-8b")
    assert granite.get("architecture") == "granitemoehybrid"
    assert granite.get("max_tokens") == 131072
    assert granite.get("tool_support") == "native"

    omni = get_model_capabilities("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
    assert omni.get("architecture") == "nemotron_hybrid_moe"
    assert omni.get("vision_support") is True
    assert omni.get("audio_support") is True
    assert omni.get("video_support") is True


def test_granite_and_nemotron_architecture_formats_match_upstream_tool_transcripts() -> None:
    granite_format = get_architecture_format(detect_architecture("ibm-granite/granite-4.0-micro"))
    assert granite_format.get("tool_format") == "xml"
    assert granite_format.get("tool_prefix") == "<tool_call>"
    assert granite_format.get("system_prefix") == "<|start_of_role|>system<|end_of_role|>"

    nemotron_format = get_architecture_format(detect_architecture("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"))
    assert nemotron_format.get("tool_format") == "xml"
    assert nemotron_format.get("tool_prefix") == "<tool_call>"
    assert nemotron_format.get("default_tool_support") == "native"

    omnicoder_format = get_architecture_format(detect_architecture("Tesslate/OmniCoder-9B"))
    assert omnicoder_format.get("tool_format") == "xml"
    assert omnicoder_format.get("tool_prefix") == "<tool_call>"
    assert omnicoder_format.get("output_wrappers") == {"end": "<|im_end|>"}
