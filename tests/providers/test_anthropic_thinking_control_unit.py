from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstractcore.providers.anthropic_provider import AnthropicProvider


def _fake_response(*, blocks: list[SimpleNamespace], model: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        model=model,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_anthropic_opus_4_6_thinking_high_maps_to_adaptive_effort(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert call_params["output_config"]["effort"] == "high"


def test_anthropic_opus_4_6_thinking_xhigh_maps_to_max_effort(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="xhigh", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert call_params["output_config"]["effort"] == "max"


def test_anthropic_sonnet_4_6_thinking_xhigh_warns_and_maps_to_high(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    with pytest.warns(RuntimeWarning):
        provider.generate(prompt="hi", thinking="xhigh", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert call_params["output_config"]["effort"] == "high"


def test_anthropic_legacy_thinking_level_maps_to_budget_tokens(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=5000)

    call_params = captured["call_params"]
    assert call_params["thinking"]["type"] == "enabled"
    # High maps to 8192, but must be clamped to max_output_tokens.
    assert call_params["thinking"]["budget_tokens"] == 5000


def test_anthropic_thinking_off_sets_disabled(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="off", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "disabled"}
    assert "output_config" not in call_params


def test_anthropic_thinking_blocks_are_captured_as_reasoning(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    def fake_create(**_call_params):
        return _fake_response(
            blocks=[
                SimpleNamespace(type="thinking", thinking="r"),
                SimpleNamespace(type="text", text="final"),
            ],
            model=provider.model,
        )

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    resp = provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=16)
    assert resp.content == "final"
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("reasoning") == "r"

