from __future__ import annotations

from types import SimpleNamespace

from abstractcore.providers.anthropic_provider import AnthropicProvider


def _fake_anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_anthropic_prompt_cache_key_enables_cache_control(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(
        prompt="hello",
        max_output_tokens=16,
        temperature=0.0,
        prompt_cache_key="session-123",
    )

    call_params = captured.get("call_params") or {}
    assert call_params.get("cache_control") == {"type": "ephemeral"}

