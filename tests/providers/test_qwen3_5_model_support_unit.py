from abstractcore.architectures import detect_architecture, get_model_capabilities


def test_qwen3_5_variants_detect_qwen3_5_architecture() -> None:
    variants = [
        "Qwen/Qwen3.5-9B",
        "qwen/qwen3.5-9b",
        "lmstudio/qwen/qwen3.5-9b",
        "models--Qwen--Qwen3.5-9B",
        "qwen3.5-9b",
        "Jackrong/Qwopus3.5-9B-v3-GGUF",
        "Jackrong/Qwopus3.5-27B-v3",
    ]

    for model in variants:
        assert detect_architecture(model) == "qwen3_5"


def test_qwen3_5_capabilities_resolve_from_aliases() -> None:
    caps = get_model_capabilities("lmstudio/qwen/qwen3.5-9b")

    assert caps.get("architecture") == "qwen3_5"
    assert caps.get("max_tokens") == 262144
    assert caps.get("max_output_tokens") == 81920
    assert caps.get("tool_support") == "native"
    assert caps.get("vision_support") is True

    community_9b = get_model_capabilities("Jackrong/Qwopus3.5-9B-v3-GGUF")
    assert community_9b.get("architecture") == "qwen3_5"
    assert community_9b.get("max_tokens") == 262144
    assert community_9b.get("max_output_tokens") == 81920
    assert community_9b.get("tool_support") == "native"

    community_27b = get_model_capabilities("Jackrong/Qwopus3.5-27B-v3")
    assert community_27b.get("architecture") == "qwen3_5"
    assert community_27b.get("max_tokens") == 262144
    assert community_27b.get("max_output_tokens") == 81920
    assert community_27b.get("tool_support") == "native"


def test_qwen3_5_family_capabilities_cover_dense_and_moe_variants() -> None:
    dense_ids = [
        "qwen/qwen3.5-0.8b",
        "qwen/qwen3.5-2b",
        "qwen/qwen3.5-4b",
        "qwen/qwen3.5-9b",
        "qwen/qwen3.5-27b",
    ]

    for model_id in dense_ids:
        caps = get_model_capabilities(model_id)
        assert caps.get("architecture") == "qwen3_5"
        assert caps.get("max_tokens") == 262144
        assert caps.get("tool_support") == "native"
        assert caps.get("structured_output") == "prompted"
        assert caps.get("vision_support") is True

    moe = get_model_capabilities("qwen/qwen3.5-35b-a3b")
    assert moe.get("architecture") == "qwen3_5"
    assert moe.get("total_parameters") == "35B"
    assert moe.get("active_parameters") == "3B"
    assert moe.get("experts") == 256
    assert moe.get("shared_experts") == 1
    assert moe.get("experts_activated") == 8

    flagship_moe = get_model_capabilities("qwen/qwen3.5-397b-a17b")
    assert flagship_moe.get("architecture") == "qwen3_5"
    assert flagship_moe.get("total_parameters") == "397B"
    assert flagship_moe.get("active_parameters") == "17B"
    assert flagship_moe.get("experts") == 512
    assert flagship_moe.get("shared_experts") == 1
    assert flagship_moe.get("experts_activated") == 10

    frontier_moe = get_model_capabilities("qwen/qwen3.5-122b-a10b")
    assert frontier_moe.get("architecture") == "qwen3_5"
    assert frontier_moe.get("max_tokens") == 262144
    assert frontier_moe.get("total_parameters") == "122B"
    assert frontier_moe.get("active_parameters") == "10B"
    assert frontier_moe.get("experts") == 256
    assert frontier_moe.get("shared_experts") == 1
    assert frontier_moe.get("experts_activated") == 8


def test_lfm2_family_capabilities_resolve_from_aliases() -> None:
    tiny = get_model_capabilities("LiquidAI/LFM2.5-350M")
    assert tiny.get("architecture") == "lfm2"
    assert tiny.get("max_tokens") == 32768
    assert tiny.get("max_output_tokens") == 4096
    assert tiny.get("tool_support") == "prompted"
    assert tiny.get("structured_output") == "prompted"
    assert tiny.get("total_parameters") == "350M"

    instruct = get_model_capabilities("liquidai/lfm2.5-1.2b-instruct")
    assert instruct.get("architecture") == "lfm2"
    assert instruct.get("max_tokens") == 32768
    assert instruct.get("tool_support") == "prompted"
    assert instruct.get("structured_output") == "prompted"
    assert instruct.get("vision_support") is False

    moe_24b = get_model_capabilities("liquidai/lfm2-24b-a2b")
    assert moe_24b.get("architecture") == "lfm2"
    assert moe_24b.get("max_tokens") == 32768
    assert moe_24b.get("tool_support") == "prompted"
    assert moe_24b.get("total_parameters") == "24B"
    assert moe_24b.get("active_parameters") == "2B"
    assert moe_24b.get("experts") == 64
    assert moe_24b.get("experts_activated") == 6

    moe_8b = get_model_capabilities("liquidai/lfm2-8b-a1b")
    assert moe_8b.get("architecture") == "lfm2"
    assert moe_8b.get("tool_support") == "prompted"
    assert moe_8b.get("total_parameters") == "8B"
    assert moe_8b.get("active_parameters") == "1B"
    assert moe_8b.get("experts") == 32
    assert moe_8b.get("experts_activated") == 6

    dense_26b = get_model_capabilities("liquidai/lfm2-2.6b")
    assert dense_26b.get("architecture") == "lfm2"
    assert dense_26b.get("tool_support") == "prompted"
    assert dense_26b.get("thinking_support") is True
    assert dense_26b.get("max_tokens") == 32768
    assert dense_26b.get("total_parameters") == "2.6B"

    dense_12b = get_model_capabilities("liquidai/lfm2-1.2b")
    assert dense_12b.get("architecture") == "lfm2"
    assert dense_12b.get("tool_support") == "prompted"
    assert dense_12b.get("max_tokens") == 32768
    assert dense_12b.get("total_parameters") == "1.2B"
