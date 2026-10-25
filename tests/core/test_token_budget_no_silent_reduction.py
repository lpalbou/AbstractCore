"""Numeric-budget regressions (adversarial budget audit, 2026-08-02).

Three defects, all of the same family: a default silently handing back LESS
than the caller or the model could have used (ADR-0026).
"""
from __future__ import annotations

import logging

import pytest

from abstractcore.architectures import get_context_limits
from abstractcore.core.interface import AbstractCoreInterface


class _Budget:
    """Minimal stand-in exercising the shared derivation."""

    def __init__(self, max_tokens, max_output_tokens, max_input_tokens=None):
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens

    _calculate_effective_token_limits = (
        AbstractCoreInterface._calculate_effective_token_limits
    )


@pytest.mark.parametrize(
    "ctx, out",
    [
        (262144, 262144),  # gemma-4-31b-it: no separate completion cap
        (131072, 128000),  # gpt-oss-120b
        (32768, 32768),    # codestral / mixtral-8x7b
        (2048, 2048),      # phi-2
        (8192, 8192),      # gemma-7b
    ],
)
def test_shared_window_models_never_derive_a_zero_input_budget(ctx, out):
    """`max_output_tokens >= max_tokens` must not mean "accepts no input".

    30 registry entries record the output ceiling AS the context window,
    because the vendor publishes no separate completion cap. The old
    derivation read that as a reservation and produced max_input <= 0, which
    collapsed every downstream input budget (the summarizer's chunking
    threshold fell to its 8000-token floor even on a 262K model).
    """
    _, _, max_input = _Budget(ctx, out)._calculate_effective_token_limits()
    assert max_input > 0
    assert max_input == ctx - min(out, ctx // 4)


@pytest.mark.parametrize(
    "ctx, out",
    [
        (40960, 38912),    # qwen3-1.7b, widened by this audit
        (200000, 128000),  # claude-3.7-sonnet
        (1000000, 128000), # claude-opus-4-6
        (128000, 4096),    # gpt-4 (unchanged by the floor)
        (262144, 16384),   # qwen3-4b-2507 (unchanged by the floor)
    ],
)
def test_widening_an_output_ceiling_never_shrinks_the_input_budget(ctx, out):
    """The caller-asks-more-gets-less inversion.

    `max_output_tokens` is a maximum generation LENGTH, not a slice reserved
    out of the window. Deriving `max_input = ctx - out` therefore turned every
    honest widening of an output ceiling into a silent shrink of the input
    budget. The derivation reserves at most a quarter of the window, so it is
    provably never below the old `ctx - out` for any pair.
    """
    _, _, max_input = _Budget(ctx, out)._calculate_effective_token_limits()
    assert max_input >= ctx - out


def test_explicit_max_input_tokens_is_never_overridden():
    _, _, max_input = _Budget(262144, 262144, 5000)._calculate_effective_token_limits()
    assert max_input == 5000


def test_null_output_cap_resolves_to_the_context_window_not_an_invented_4096():
    """`"max_output_tokens": null` means "no published completion cap".

    It used to resolve to 4096 — an arbitrary output cap applied by default
    (ADR-0026 §2), sent verbatim by every provider that requires a bound.
    """
    limits = get_context_limits("LiquidAI/LFM2.5-8B-A1B")
    assert limits["max_output_tokens"] == limits["max_tokens"] == 128000

    grok = get_context_limits("grok-4")
    assert grok["max_output_tokens"] == grok["max_tokens"] == 256000


def test_clamping_a_caller_requested_output_budget_is_never_silent(caplog):
    """ADR-0026 §1: a budget the caller asked for and will not get must warn.

    The usual cause is an understated model_capabilities.json entry, not a
    real model limit, and without the warning the only symptom is an
    unexplained short completion.
    """
    from abstractcore.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")
    cap = int(provider.max_output_tokens)

    with caplog.at_level(logging.WARNING):
        kwargs = provider._prepare_generation_kwargs(
            max_output_tokens=cap * 10, temperature=0
        )

    assert kwargs["max_output_tokens"] == cap
    assert any(
        "Clamping caller-requested max_output_tokens" in r.getMessage()
        and "model_capabilities.json" in r.getMessage()
        for r in caplog.records
    ), "clamp must be attributable to the registry, not silent"


def test_no_alias_is_claimed_by_two_registry_entries():
    """A duplicate alias silently routes a model to another model's budget.

    Found live: a `gpt-5.6` entry added by inference carried gpt-5.5's dated
    alias AND shadowed the wire-verified `gpt-5.6-sol`, so `gpt-5.6` reported
    a 1,050,000-token window against a wire-verified 400,000 ceiling.
    """
    import json
    from pathlib import Path

    import abstractcore

    path = Path(abstractcore.__file__).parent / "assets" / "model_capabilities.json"
    models = json.loads(path.read_text(encoding="utf-8"))["models"]

    owner: dict[str, str] = {}
    collisions: list[str] = []
    for name, entry in models.items():
        for alias in entry.get("aliases") or []:
            if alias in owner:
                collisions.append(f"{alias!r}: {owner[alias]} and {name}")
            owner[alias] = name
        if name in owner and owner[name] != name:
            collisions.append(f"{name!r}: top-level entry shadows alias of {owner[name]}")

    assert not collisions, "ambiguous registry names: " + "; ".join(collisions)


UNREGISTERED = "acme-unregistered-chat-x1"  # never in model_capabilities.json


def _compat_send(monkeypatch, model, **gen_kwargs):
    """generate() through the real OpenAI-compatible payload builder; return (body, warnings)."""
    import warnings

    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = OpenAICompatibleProvider(model=model, base_url="http://relay.test/v1", api_key="k")
    captured = {}

    def _single(payload, **_kw):
        captured["body"] = payload
        return GenerateResponse(content="{}", model=model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _single)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider.generate("Return JSON.", **gen_kwargs)
    return provider, captured["body"], [str(w.message) for w in caught]


def test_an_unregistered_models_guessed_ceiling_never_cuts_an_explicit_budget(monkeypatch):
    """Live defect 2026-10-25: max_tokens=12000 on an unregistered model (claude-haiku-5-5 /
    gpt-6-luna through an OpenAI-compatible proxy) went out as 4096 — the
    `default_capabilities` guess — and the JSON answer came back truncated.

    For a model the registry does not know, 4096 is not a fact about the model; the
    caller's budget is sent as given and the server enforces its real limit.
    """
    from abstractcore.architectures.detection import lookup_registry_model_capabilities

    assert lookup_registry_model_capabilities(UNREGISTERED) is None
    provider, body, caught = _compat_send(monkeypatch, UNREGISTERED, max_tokens=12000)

    assert int(provider.max_output_tokens) < 12000  # the guessed default is still smaller
    sent = body.get("max_tokens", body.get("max_completion_tokens"))
    assert sent == 12000
    assert not any("Clamping caller-requested" in w for w in caught), caught


def test_a_registered_models_ceiling_still_clamps_with_a_warning(monkeypatch):
    provider, body, caught = _compat_send(monkeypatch, "gpt-5.6-luna", max_tokens=10_000_000)
    sent = body.get("max_tokens", body.get("max_completion_tokens"))
    assert sent == int(provider.max_output_tokens) == 128000
    assert any(
        "Clamping caller-requested" in w and "model_capabilities.json" in w for w in caught
    ), caught


def test_a_constructor_cap_still_bounds_per_call_requests_on_an_unregistered_model(monkeypatch):
    import warnings

    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = OpenAICompatibleProvider(
            model=UNREGISTERED, base_url="http://relay.test/v1", api_key="k", max_output_tokens=3000
        )
        kwargs = provider._prepare_generation_kwargs(max_output_tokens=12000)
    assert kwargs["max_output_tokens"] == 3000
