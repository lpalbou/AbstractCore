"""Reasoning precedence for one call, pinned at the REQUEST BODY (docs/reasoning-control.md).

Precedence, highest first:
  1. the caller's `thinking=`;
  2. a provider-native reasoning field the caller passed (`reasoning_effort=` ...), sent as given;
  3. the reasoning effort configured on the text route, ONLY when that row describes this
     instance (same provider, same model; a reasoning-only row applies everywhere);
  4. nothing (model/server default).

THE DEFECT (live, 2026-10-25): with `reasoning: minimal` configured on the
`endpoint:airelay / gpt-5.6-luna` text route,
`create_llm("openai-compatible", model="gpt-5.6-luna").generate(..., reasoning_effort="high")`
sent `reasoning_effort: "low"` (configured "minimal" adopted as thinking=, mapped to the
model's "low", overwriting the caller's field), and so did reasoning_effort="none" and a call
with no reasoning at all. Every test here is hermetic: the config file points at a path that
does not exist and the route rows are injected as scoped capability defaults.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import warnings
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.openrouter_provider import OpenRouterProvider
from abstractcore.providers.vllm_provider import VLLMProvider

LUNA = "gpt-5.6-luna"  # registry levels none|low|medium|high|xhigh (no "minimal")
QWEN38 = "qwen/qwen3.8-27b"  # registry levels low|medium|xhigh


def _hermetic(provider, route: Optional[Dict[str, Any]]):
    provider._abstractcore_config_file = os.path.join(
        tempfile.gettempdir(), f"acore-absent-config-{uuid.uuid4().hex}.json"
    )
    provider._abstractcore_capability_defaults = (
        {"input.text": {"key": "input.text", "source": "abstractcore.capability_defaults", **route}}
        if route is not None
        else {}
    )
    return provider


def _quiet(factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return factory()


def _install_fake_openai(monkeypatch) -> None:
    import abstractcore.providers.openai_provider as openai_provider_module

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_k: object()))
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    fake = SimpleNamespace(OpenAI=_FakeOpenAIClient, AsyncOpenAI=_FakeOpenAIClient)
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider_module, "openai", fake, raising=False)


def _openai(monkeypatch, model: str = LUNA) -> OpenAIProvider:
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None, raising=False)
    return _quiet(lambda: OpenAIProvider(model=model))


def _compat(monkeypatch, model: str = LUNA) -> OpenAICompatibleProvider:
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    return _quiet(lambda: OpenAICompatibleProvider(model=model, base_url="http://relay.test/v1", api_key="k"))


def _lmstudio(monkeypatch, model: str = QWEN38) -> LMStudioProvider:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    return _quiet(lambda: LMStudioProvider(model=model, base_url="http://localhost:1234/v1"))


def _vllm(monkeypatch, model: str = QWEN38) -> VLLMProvider:
    monkeypatch.setattr(VLLMProvider, "_validate_model", lambda self: None, raising=False)
    return _quiet(lambda: VLLMProvider(model=model, base_url="http://127.0.0.1:8000/v1"))


def _openrouter(monkeypatch, model: str = LUNA) -> OpenRouterProvider:
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None, raising=False)
    return _quiet(lambda: OpenRouterProvider(model=model, api_key="k"))


FACTORIES = {
    "openai": (_openai, "openai", LUNA),
    "openai-compatible": (_compat, "openai-compatible", LUNA),
    "lmstudio": (_lmstudio, "lmstudio", QWEN38),
    "vllm": (_vllm, "vllm", QWEN38),
    "openrouter": (_openrouter, "openrouter", LUNA),
}


def _send(monkeypatch, provider, **gen_kwargs):
    """generate() through the real payload builder; return (body, response, warnings)."""
    captured: Dict[str, Any] = {}
    if isinstance(provider, OpenAIProvider):

        def _create(**call_params):
            captured["body"] = call_params
            return object()

        monkeypatch.setattr(provider.client.chat.completions, "create", _create)
        monkeypatch.setattr(
            provider,
            "_format_response",
            lambda _r: GenerateResponse(content="ok", model=provider.model, finish_reason="stop"),
        )
    else:

        def _single(payload, **_kw):
            captured["body"] = payload
            return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

        monkeypatch.setattr(provider, "_single_generate", _single)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resp = provider.generate("Say OK.", **gen_kwargs)
    return captured["body"], resp, [str(w.message) for w in caught]


def _own_route(name: str) -> Dict[str, Any]:
    _factory, provider_id, model = FACTORIES[name]
    return {"provider": provider_id, "model": model, "reasoning": "minimal"}


def _no_thinking_artifact(body: Dict[str, Any]) -> bool:
    ctk = body.get("chat_template_kwargs") or {}
    return "reasoning" not in body and not any(
        k in ctk for k in ("enable_thinking", "reasoning_effort", "thinking")
    )


# ---------------------------------------------------------------------------
# 2 beats 3: an explicit reasoning_effort= wins over the configured default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FACTORIES))
@pytest.mark.parametrize("effort", ["high", "none"])
def test_explicit_reasoning_effort_beats_the_configured_default(monkeypatch, name, effort) -> None:
    factory = FACTORIES[name][0]
    provider = _hermetic(factory(monkeypatch), _own_route(name))

    body, resp, caught = _send(monkeypatch, provider, reasoning_effort=effort)

    assert body.get("reasoning_effort") == effort  # OLD CODE: "low" (configured minimal, mapped)
    assert _no_thinking_artifact(body)
    assert not any("'minimal'" in w for w in caught), caught
    meta = resp.metadata
    shown = "off" if effort == "none" else effort
    assert meta["thinking_source"] == "reasoning_effort"
    assert meta["thinking_requested"] == shown
    if name == "openrouter":
        # OpenRouter documents its `reasoning` object, not a top-level reasoning_effort.
        assert "thinking_effective" not in meta
    else:
        assert meta["thinking_effective"] == shown
    route = meta["_resolved_generate_route"]
    assert route["reasoning"] is None
    assert route["reasoning_source"] == "not_applied:explicit_reasoning_effort"


def test_explicit_reasoning_effort_beats_the_configured_default_async(monkeypatch) -> None:
    provider = _hermetic(_compat(monkeypatch), _own_route("openai-compatible"))
    captured: Dict[str, Any] = {}

    async def _single(payload, **_kw):
        captured["body"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_async_single_generate", _single)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resp = asyncio.run(provider.agenerate("Say OK.", reasoning_effort="high"))
    assert captured["body"].get("reasoning_effort") == "high"
    assert resp.metadata["thinking_source"] == "reasoning_effort"
    assert resp.metadata["thinking_effective"] == "high"


# ---------------------------------------------------------------------------
# 3: the configured default still applies when the caller said nothing
# ---------------------------------------------------------------------------


def test_configured_default_applies_to_its_own_instance(monkeypatch) -> None:
    provider = _hermetic(_compat(monkeypatch), _own_route("openai-compatible"))
    body, resp, caught = _send(monkeypatch, provider)

    assert body.get("reasoning_effort") == "low"  # "minimal" → nearest declared level, warned
    assert any("'minimal'" in w for w in caught)
    meta = resp.metadata
    assert meta["thinking_source"] == "configured_default"
    assert meta["thinking_requested"] == "minimal"
    assert meta["thinking_effective"] == "low"


def test_reasoning_only_row_applies_to_every_instance(monkeypatch) -> None:
    provider = _hermetic(_openai(monkeypatch), {"reasoning": "high"})
    body, resp, _ = _send(monkeypatch, provider)
    assert body.get("reasoning_effort") == "high"
    assert resp.metadata["thinking_source"] == "configured_default"


def test_endpoint_profile_instance_matches_its_virtual_provider(monkeypatch) -> None:
    provider = _compat(monkeypatch)
    provider._abstractcore_virtual_provider = "endpoint:airelay"
    provider._abstractcore_provider_family = "openai-compatible"
    _hermetic(provider, {"provider": "endpoint:airelay", "model": LUNA, "reasoning": "high"})
    body, resp, _ = _send(monkeypatch, provider)
    assert body.get("reasoning_effort") == "high"
    assert resp.metadata["thinking_source"] == "configured_default"


# ---------------------------------------------------------------------------
# 3 is scoped: a row configured for another backend is not this instance's default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FACTORIES))
def test_a_route_for_another_backend_does_not_set_this_instances_reasoning(monkeypatch, name) -> None:
    factory = FACTORIES[name][0]
    # The live configuration: the text route is endpoint:airelay / gpt-5.6-luna.
    provider = _hermetic(
        factory(monkeypatch), {"provider": "endpoint:airelay", "model": LUNA, "reasoning": "minimal"}
    )

    body, resp, caught = _send(monkeypatch, provider)

    assert "reasoning_effort" not in body  # OLD CODE: the airelay route's "minimal" was applied
    assert _no_thinking_artifact(body)
    assert not any("'minimal'" in w for w in caught), caught
    meta = resp.metadata
    assert "thinking_requested" not in meta and "thinking_source" not in meta
    route = meta["_resolved_generate_route"]
    assert route["reasoning"] is None
    assert route["reasoning_source"] == "not_applied:route_provider_differs"


def test_a_route_for_another_model_on_the_same_provider_does_not_apply(monkeypatch) -> None:
    provider = _hermetic(
        _compat(monkeypatch), {"provider": "openai-compatible", "model": "gpt-5.6-sol", "reasoning": "high"}
    )
    body, resp, _ = _send(monkeypatch, provider)
    assert "reasoning_effort" not in body
    assert resp.metadata["_resolved_generate_route"]["reasoning_source"] == "not_applied:route_model_differs"


def test_prompt_cache_planner_uses_the_same_scoping(monkeypatch) -> None:
    provider = _hermetic(
        _compat(monkeypatch), {"provider": "endpoint:airelay", "model": LUNA, "reasoning": "minimal"}
    )
    assert provider._prompt_cache_default_thinking() is None
    _hermetic(provider, _own_route("openai-compatible"))
    assert provider._prompt_cache_default_thinking() == "minimal"


# ---------------------------------------------------------------------------
# 1 beats everything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("thinking, sent", [(False, "none"), ("high", "high")])
def test_explicit_thinking_beats_the_configured_default(monkeypatch, thinking, sent) -> None:
    provider = _hermetic(_openai(monkeypatch), _own_route("openai"))
    body, resp, _ = _send(monkeypatch, provider, thinking=thinking)
    assert body.get("reasoning_effort") == sent
    assert resp.metadata["thinking_source"] == "thinking"
    assert resp.metadata["_resolved_generate_route"]["reasoning_source"] == "explicit"


def test_thinking_auto_means_no_control_and_no_configured_default(monkeypatch) -> None:
    provider = _hermetic(_openai(monkeypatch), _own_route("openai"))
    body, resp, _ = _send(monkeypatch, provider, thinking="auto")
    assert "reasoning_effort" not in body
    assert "thinking_requested" not in resp.metadata
