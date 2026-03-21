from abstractcore.architectures import detect_architecture, get_model_capabilities


def test_gpt5x_variants_use_existing_gpt_architecture() -> None:
    variants = [
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "openai/gpt-5.4-pro",
    ]

    for model in variants:
        assert detect_architecture(model) == "gpt"


def test_gpt5_2_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("gpt-5.2")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 400000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "native"
    assert caps.get("reasoning_levels") == ["none", "low", "medium", "high", "xhigh"]
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True


def test_gpt5_2_pro_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("gpt-5.2-pro")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 400000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "prompted"
    assert caps.get("reasoning_levels") == ["medium", "high", "xhigh"]
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True


def test_gpt5_3_codex_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("gpt-5.3-codex")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 400000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "native"
    assert caps.get("reasoning_levels") == ["low", "medium", "high", "xhigh"]
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True


def test_gpt5_4_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("gpt-5.4")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 1050000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "native"
    assert caps.get("reasoning_levels") == ["none", "low", "medium", "high", "xhigh"]
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True


def test_gpt5_4_pro_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("openai/gpt-5.4-pro")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 1050000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "prompted"
    assert caps.get("reasoning_levels") == ["medium", "high", "xhigh"]
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True


def test_gpt5_4_mini_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("gpt-5.4-mini")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 400000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "native"
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True


def test_gpt5_4_nano_capabilities_match_official_model_page() -> None:
    caps = get_model_capabilities("gpt-5.4-nano")

    assert caps.get("architecture") == "gpt"
    assert caps.get("max_tokens") == 400000
    assert caps.get("max_output_tokens") == 128000
    assert caps.get("tool_support") == "native"
    assert caps.get("structured_output") == "native"
    assert caps.get("web_browsing") is True
    assert caps.get("python_execution") is True
