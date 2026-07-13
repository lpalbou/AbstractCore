"""Registry-driven parameter filtering on the OpenAI-compatible wire path.

model_capabilities.json declares two per-model API constraints:

- ``unsupported_parameters``: generation parameters the model's API REJECTS
  (o-series / GPT-5-class reject temperature/top_p/penalties; they accept seed).
- ``token_param_name``: the API's output-cap key (``max_tokens`` vs
  ``max_completion_tokens``).

The OpenAI provider has honored both since the capability-filtering wave
(v2.13.0). The OpenAI-compatible provider built its payload UNCONDITIONALLY
(temperature/top_p/max_tokens always present), so a restricted model served
through an OpenAI-compatible endpoint — a LiteLLM-style proxy in front of an
o-series/GPT-5-class API, or a strict vLLM deployment — received parameters
its API rejects and failed with a 400 the registry existed to prevent.

These tests pin the fix: `_apply_model_parameter_constraints` runs at BOTH
payload sites (sync and async — the async path has drifted from sync before,
see the "No user query found" parity note in the provider), before
`_mutate_payload` so subclass hooks see the filtered payload. Models WITHOUT
declarations must see byte-identical payloads (backward-compat pin).
"""

import asyncio

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


def _provider(model: str = "test-model", **caps) -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider(
        model=model,
        base_url="http://127.0.0.1:9/v1",
        api_key="x",
        validate_model=False,
    )
    if caps:
        existing = dict(p.model_capabilities) if isinstance(p.model_capabilities, dict) else {}
        existing.update(caps)
        p.model_capabilities = existing
    return p


def _capture_sync(provider, monkeypatch):
    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)
    return captured


def test_unsupported_temperature_and_top_p_dropped(monkeypatch):
    provider = _provider(unsupported_parameters=["temperature", "top_p"])
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", temperature=0.2, top_p=0.5)

    payload = captured["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload
    # The output cap always survives (rename-only handling, never dropped).
    assert "max_tokens" in payload


def test_token_param_renamed_to_max_completion_tokens(monkeypatch):
    provider = _provider(token_param_name="max_completion_tokens")
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", max_output_tokens=321)

    payload = captured["payload"]
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 321


def test_no_declarations_payload_unchanged(monkeypatch):
    """Backward-compat pin: absent registry fields = zero behavior change."""
    provider = _provider()
    caps = provider.model_capabilities if isinstance(provider.model_capabilities, dict) else {}
    assert caps.get("unsupported_parameters") is None
    assert "token_param_name" not in caps

    captured = _capture_sync(provider, monkeypatch)
    provider.generate("hi", temperature=0.3, top_p=0.8, max_output_tokens=100, seed=7)

    payload = captured["payload"]
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 100
    assert payload["seed"] == 7


def test_explicit_penalties_and_seed_dropped_when_unsupported(monkeypatch):
    provider = _provider(
        unsupported_parameters=["frequency_penalty", "presence_penalty", "seed"]
    )
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", frequency_penalty=0.4, presence_penalty=0.2, seed=42)

    payload = captured["payload"]
    assert "frequency_penalty" not in payload
    assert "presence_penalty" not in payload
    assert "seed" not in payload
    # Params without a declaration stay.
    assert "temperature" in payload


def test_gpt5_class_registry_shape_produces_clean_payload(monkeypatch):
    """The real o-series/GPT-5 registry shape end-to-end: all rejected params
    filtered AND the cap renamed — the exact proxy-served scenario that 400'd."""
    provider = _provider(
        unsupported_parameters=[
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ],
        token_param_name="max_completion_tokens",
    )
    captured = _capture_sync(provider, monkeypatch)

    provider.generate(
        "hi", temperature=0.7, top_p=0.9, frequency_penalty=0.1, max_output_tokens=64
    )

    payload = captured["payload"]
    for rejected in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "max_tokens"):
        assert rejected not in payload
    assert payload["max_completion_tokens"] == 64
    # Request stays otherwise intact.
    assert payload["model"] == provider.model
    assert payload["messages"]


def test_async_path_filters_identically(monkeypatch):
    """Parity pin: the async payload site applies the same constraints."""
    provider = _provider(
        unsupported_parameters=["temperature", "top_p"],
        token_param_name="max_completion_tokens",
    )
    captured = {}

    async def _capture_async_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_async_single_generate", _capture_async_single_generate)

    asyncio.run(provider.agenerate("hi", temperature=0.2, max_output_tokens=55))

    payload = captured["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 55


def test_lmstudio_top_k_respects_registry(monkeypatch):
    """Subclass check: LMStudio's top_k passthrough rides the same filter."""
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None, raising=False)

    kept = LMStudioProvider(model="test-model", base_url="http://127.0.0.1:9/v1")
    captured_kept = _capture_sync(kept, monkeypatch)
    kept.generate("hi", top_k=40)
    assert captured_kept["payload"]["top_k"] == 40

    dropped = LMStudioProvider(model="test-model", base_url="http://127.0.0.1:9/v1")
    caps = dict(dropped.model_capabilities) if isinstance(dropped.model_capabilities, dict) else {}
    caps["unsupported_parameters"] = ["top_k"]
    dropped.model_capabilities = caps
    captured_dropped = _capture_sync(dropped, monkeypatch)
    dropped.generate("hi", top_k=40)
    assert "top_k" not in captured_dropped["payload"]


def test_filter_runs_before_mutate_payload(monkeypatch):
    """Subclass hooks must see the FILTERED payload (portkey composes on it)."""
    provider = _provider(unsupported_parameters=["temperature"])
    seen = {}

    original_mutate = provider._mutate_payload

    def _spy_mutate(payload, **kwargs):
        seen["temperature_present"] = "temperature" in payload
        return original_mutate(payload, **kwargs)

    monkeypatch.setattr(provider, "_mutate_payload", _spy_mutate)
    _capture_sync(provider, monkeypatch)

    provider.generate("hi", temperature=0.5)
    assert seen["temperature_present"] is False


def _portkey(monkeypatch, **caps):
    from abstractcore.providers.portkey_provider import PortkeyProvider

    p = PortkeyProvider(
        model="gpt-5",
        api_key="pk-x",
        portkey_provider="openai",
        provider_api_key="sk-x",
        validate_model=False,
    )
    if caps:
        existing = dict(p.model_capabilities) if isinstance(p.model_capabilities, dict) else {}
        existing.update(caps)
        p.model_capabilities = existing
    return p


def test_portkey_unsolicited_cap_stripped_even_after_base_rename(monkeypatch):
    """Composition pin: the base filter renames max_tokens BEFORE portkey's
    hook; portkey's unsolicited-default strip must recognize the renamed
    spelling too, or the default cap (never requested) leaks to the backend."""
    provider = _portkey(monkeypatch, token_param_name="max_completion_tokens")
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi")

    payload = captured["payload"]
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


def test_portkey_explicit_cap_survives_base_rename(monkeypatch):
    provider = _portkey(monkeypatch, token_param_name="max_completion_tokens")
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", max_output_tokens=99)

    payload = captured["payload"]
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 99


def test_streaming_payload_is_filtered_too(monkeypatch):
    """Adversary pin: a refactor moving the filter into `_single_generate` would
    pass every non-stream test and silently unfilter streaming."""
    provider = _provider(
        unsupported_parameters=["temperature", "top_p"],
        token_param_name="max_completion_tokens",
    )
    captured = {}

    def _capture_stream_generate(payload):
        captured["payload"] = payload
        yield GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_stream_generate", _capture_stream_generate)

    for _chunk in provider.generate("hi", stream=True, temperature=0.2, max_output_tokens=44):
        pass

    payload = captured["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 44


def test_real_registry_gpt5_end_to_end(monkeypatch):
    """No injected caps: the REAL model_capabilities.json entry for gpt-5 drives
    the filter (registry lookup path exercised end-to-end)."""
    provider = OpenAICompatibleProvider(
        model="gpt-5",
        base_url="http://127.0.0.1:9/v1",
        api_key="x",
        validate_model=False,
    )
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", temperature=0.7, max_output_tokens=128)

    payload = captured["payload"]
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 128


def test_local_model_with_registry_colliding_name_is_not_filtered(monkeypatch):
    """Adversary P1 (fuzzy-match regression): a LOCAL model whose name merely
    CONTAINS a restricted registry key ("o1" midfix) must keep its sampling
    params and its `max_tokens` cap — midfix substring inheritance is family
    inference, not identity. A renamed cap on a llama.cpp-class server that
    ignores unknown fields would mean UNCAPPED generation."""
    provider = OpenAICompatibleProvider(
        model="Skywork-o1-Open-Llama-3.1-8B",
        base_url="http://127.0.0.1:9/v1",
        api_key="x",
        validate_model=False,
    )
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", temperature=0.3, max_output_tokens=77)

    payload = captured["payload"]
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 77
    assert "max_completion_tokens" not in payload


def test_prefix_aligned_snapshot_name_still_filtered(monkeypatch):
    """Dated snapshots ARE the registry family: "gpt-5-2025-08-07" must inherit
    gpt-5's wire constraints through the prefix-aligned partial match."""
    provider = OpenAICompatibleProvider(
        model="gpt-5-2025-08-07",
        base_url="http://127.0.0.1:9/v1",
        api_key="x",
        validate_model=False,
    )
    captured = _capture_sync(provider, monkeypatch)

    provider.generate("hi", temperature=0.7, max_output_tokens=128)

    payload = captured["payload"]
    assert "temperature" not in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 128


def test_async_payload_carries_prompt_cache_key(monkeypatch):
    """Sync/async parity (adversarial-review find): the async builder dropped
    `prompt_cache_key` entirely — async callers lost session cache identity."""
    provider = _provider()
    captured = {}

    async def _capture_async_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_async_single_generate", _capture_async_single_generate)

    asyncio.run(provider.agenerate("hi", prompt_cache_key="session-abc"))

    assert captured["payload"]["prompt_cache_key"] == "session-abc"


def test_malformed_registry_shapes_fail_safe():
    """Adversary P2: a string `unsupported_parameters` must not become substring
    matching; a null/empty `token_param_name` must not rename the cap to None."""
    provider = _provider()
    caps = dict(provider.model_capabilities) if isinstance(provider.model_capabilities, dict) else {}
    caps["unsupported_parameters"] = "temperature, top_p"  # malformed: string, not list
    caps["token_param_name"] = None  # malformed: null
    provider.model_capabilities = caps

    assert provider._is_parameter_supported("temperature") is True
    assert provider._is_parameter_supported("top_p") is True
    assert provider._get_token_param_name() == "max_tokens"

    payload = provider._apply_model_parameter_constraints(
        {"model": "m", "temperature": 0.5, "max_tokens": 10}
    )
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 10
    assert None not in payload
