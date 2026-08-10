"""`thinking` must reach the wire as OpenAI-standard `reasoning_effort` on the
openai-compatible provider — for models that declare `reasoning_levels` but no
chat-template thinking surface (the gpt-5.4-shape).

MEASURED (2026-08-03, relay inbound logs + two adversarial re-investigations):
before the mapping existed, every layer above reported medium while the wire
carried NO `reasoning_effort` at all — the requested level fell through
`_apply_provider_thinking_kwargs` untouched, and even a hook-set kwarg would
have been eaten by the payload builders' explicit allowlists. Five of eight
benchmark arms ran with reasoning OFF (upstream `reasoning_tokens=0` in
238/238 sampled absent-effort requests) while the run store reported medium.

Two parts, each inert alone, each pinned here:
1. MAP — `_apply_provider_thinking_kwargs` sets `kwargs["reasoning_effort"]`
   when the model declares `reasoning_levels` and no template surface;
2. EMIT — `_mutate_payload` copies it into the payload; it is called by BOTH
   payload builders (sync `_generate_internal` and async
   `_agenerate_internal`), so single, streaming, and async lanes are covered —
   these tests pin all three, because a refactor moving the copy into
   `_single_generate` would pass every non-stream test and silently strip
   streaming (the `test_streaming_payload_is_filtered_too` lesson).

Negative space matters as much: models with a template surface (Qwen3.6) keep
their exact prior behavior, and models declaring no `reasoning_levels` must
never receive a field their server may reject.
"""

import asyncio

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


def _provider(monkeypatch, model: str) -> OpenAICompatibleProvider:
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    return OpenAICompatibleProvider(model=model, base_url="http://127.0.0.1:9/v1")


def _capture_single(provider, monkeypatch):
    captured = {}

    def _cap(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _cap)
    return captured


def test_gpt54_thinking_medium_emits_reasoning_effort(monkeypatch):
    provider = _provider(monkeypatch, "gpt-5.4")
    # Preconditions of the mapping's gate, pinned so a registry edit that
    # invalidates them fails HERE and not as a silent no-emit downstream.
    assert "medium" in (provider._model_reasoning_levels() or [])
    surfaces = provider._thinking_control_surfaces()
    assert not surfaces.template_kwarg and not surfaces.budget_template_kwarg

    captured = _capture_single(provider, monkeypatch)
    resp = provider.generate("hi", thinking="medium", temperature=0)

    payload = captured["payload"]
    assert payload.get("reasoning_effort") == "medium"
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("thinking_requested") == "medium"
    assert resp.metadata.get("thinking_effective") == "medium"


def test_gpt54_thinking_medium_reaches_streaming_payload(monkeypatch):
    provider = _provider(monkeypatch, "gpt-5.4")
    captured = {}

    def _capture_stream_generate(payload):
        captured["payload"] = payload
        yield GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_stream_generate", _capture_stream_generate)

    for _chunk in provider.generate("hi", stream=True, thinking="medium", temperature=0):
        pass

    assert captured["payload"].get("reasoning_effort") == "medium"


def test_gpt54_thinking_medium_reaches_async_payload(monkeypatch):
    """Parity pin: the async builder calls `_mutate_payload` at its own site."""
    provider = _provider(monkeypatch, "gpt-5.4")
    captured = {}

    async def _capture_async_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_async_single_generate", _capture_async_single_generate)

    asyncio.run(provider.agenerate("hi", thinking="medium", temperature=0))

    assert captured["payload"].get("reasoning_effort") == "medium"


def test_gpt54_thinking_off_maps_to_none_level(monkeypatch):
    """gpt-5.4 declares "none" among its levels: disabling is expressible on
    the wire, and expressed (previously: nothing sent, relay default won)."""
    provider = _provider(monkeypatch, "gpt-5.4")
    captured = _capture_single(provider, monkeypatch)

    provider.generate("hi", thinking=False, temperature=0)

    assert captured["payload"].get("reasoning_effort") == "none"


def test_gpt54_without_thinking_sends_no_field(monkeypatch):
    """Absent stays absent: the provider must not invent an effort the caller
    never chose (the capability-defaults cascade upstream owns defaults)."""
    provider = _provider(monkeypatch, "gpt-5.4")
    captured = _capture_single(provider, monkeypatch)

    provider.generate("hi", temperature=0)

    assert "reasoning_effort" not in captured["payload"]


def test_unknown_model_without_levels_never_gets_the_field(monkeypatch):
    """A model that advertises no reasoning_levels must not receive
    `reasoning_effort` — strict third-party servers 400 on unknown fields."""
    provider = _provider(monkeypatch, "totally-unknown-model-xyz")
    assert not provider._model_reasoning_levels()
    captured = _capture_single(provider, monkeypatch)

    provider.generate("hi", thinking="medium", temperature=0)

    assert "reasoning_effort" not in captured["payload"]


def test_template_surface_model_keeps_prior_path(monkeypatch):
    """Qwen3.6 declares a chat-template surface: the mapping's gate must leave
    that family byte-identical (no reasoning_effort, no template kwargs unless
    opted in — the latter already pinned in test_thinking_mode_control_unit)."""
    provider = _provider(monkeypatch, "Qwen/Qwen3.6-27B")
    surfaces = provider._thinking_control_surfaces()
    assert surfaces.template_kwarg or surfaces.budget_template_kwarg

    captured = _capture_single(provider, monkeypatch)
    with pytest.warns(RuntimeWarning, match="cannot enforce effort scaling"):
        provider.generate("hi", thinking="high", temperature=0)

    assert "reasoning_effort" not in captured["payload"]
