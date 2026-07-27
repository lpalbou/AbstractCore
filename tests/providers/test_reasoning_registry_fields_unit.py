"""Registry pins for reasoning capability fields (model_capabilities.json).

Evidence trail (2026-07-26):
- OpenAI reasoning guide + openai-python SDK `ReasoningEffort`: vocabulary is
  none|minimal|low|medium|high|xhigh; o-series accepts low|medium|high ("minimal"
  arrived with the gpt-5 family; "xhigh" only exists on models after
  gpt-5.1-codex-max; o1-mini rejects reasoning_effort entirely).
- gpt-5 family (gpt-5, -mini, -nano): minimal|low|medium|high, no "none"
  (pre-5.1 models do not support "none").
- Anthropic extended-thinking + AWS Bedrock adaptive-thinking docs: Opus 4.7,
  Fable 5, and Mythos 5 are adaptive-only and reject `thinking: {"type":
  "disabled"}` with HTTP 400 (`thinking_disable_supported: false`).
"""

from __future__ import annotations

from abstractcore.architectures.detection import (
    get_model_capabilities,
    lookup_registry_model_capabilities,
)


def test_o_series_declares_effort_levels() -> None:
    for model in ("o1", "o3", "o3-mini", "o4-mini"):
        caps = get_model_capabilities(model)
        assert caps.get("thinking_support") is True, model
        assert caps.get("reasoning_levels") == ["low", "medium", "high"], model


def test_o1_mini_declares_no_effort_levels() -> None:
    # o1-mini rejects reasoning_effort; declaring levels would make the OpenAI
    # provider send a parameter the API refuses.
    caps = lookup_registry_model_capabilities("o1-mini")
    assert isinstance(caps, dict)
    assert "reasoning_levels" not in caps


def test_gpt5_family_supports_minimal() -> None:
    for model in ("gpt-5", "gpt-5-mini", "gpt-5-nano"):
        caps = get_model_capabilities(model)
        levels = caps.get("reasoning_levels")
        assert isinstance(levels, list), model
        assert levels[0] == "minimal", model
        assert "none" not in levels, model  # "none" arrived with gpt-5.1


def test_gpt51_supports_none() -> None:
    caps = get_model_capabilities("gpt-5.1")
    levels = caps.get("reasoning_levels")
    assert isinstance(levels, list)
    assert "none" in levels


def test_adaptive_only_claudes_reject_disabled() -> None:
    for model in ("claude-opus-4-7", "claude-fable-5", "claude-mythos-5"):
        caps = get_model_capabilities(model)
        assert caps.get("thinking_control_mode") == "adaptive", model
        assert caps.get("thinking_disable_supported") is False, model


def test_registry_lookup_returns_none_for_unknown_models() -> None:
    assert lookup_registry_model_capabilities("no-such-model-family-entirely") is None
