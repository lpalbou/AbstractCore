"""Output-token cap must not impose a SILENT budget (ADR 0001, use-full-capability).

Before this fix, every OpenAI-compatible / OpenAI call shipped
`max_tokens = <registry max_output_tokens>` even when the caller imposed no
cap — a silent per-call output ceiling (e.g. 8192 for gpt-oss-120b whose
context is 128k). These tests pin the corrected contract:

- caller specified NO cap  -> the output-token param is OMITTED (full capability)
- caller specified a cap    -> it is sent verbatim (explicit intent honored)
- constructor-explicit cap  -> honored
- a truncated response (finish_reason=length) is annotated + warned, never silent
- providers that REQUIRE a bound (lmstudio native REST) still send the true max
"""
from __future__ import annotations

import warnings

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


def _oai_compatible(**kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        model="test-model", base_url="http://127.0.0.1:9/v1", api_key="x", validate_model=False, **kwargs
    )


def _capture(provider, monkeypatch) -> dict:
    captured: dict = {}

    def _single(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _single)
    return captured


def test_no_cap_omits_output_token_param(monkeypatch):
    provider = _oai_compatible()
    assert provider._max_output_tokens_explicit is False
    captured = _capture(provider, monkeypatch)
    provider.generate("hi")
    payload = captured["payload"]
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


def test_explicit_per_call_cap_is_sent(monkeypatch):
    provider = _oai_compatible()
    captured = _capture(provider, monkeypatch)
    provider.generate("hi", max_output_tokens=333)
    assert captured["payload"]["max_tokens"] == 333


def test_constructor_explicit_cap_is_sent_and_flagged(monkeypatch):
    provider = _oai_compatible(max_output_tokens=512)
    assert provider._max_output_tokens_explicit is True
    captured = _capture(provider, monkeypatch)
    provider.generate("hi")
    assert captured["payload"]["max_tokens"] == 512


def test_resolver_returns_none_when_unspecified():
    provider = _oai_compatible()
    assert provider._get_provider_max_tokens_param({}) is None
    assert provider._get_provider_max_tokens_param({"max_output_tokens": 100}) == 100


def test_lmstudio_requires_a_bound():
    # LM Studio's native REST payload int()s the cap and a local backend can run
    # away — it must send the model's true max even when the caller gave none.
    lm = LMStudioProvider(model="local-model", base_url="http://127.0.0.1:9/v1")
    assert lm._requires_output_cap() is True
    assert lm._get_provider_max_tokens_param({}) == lm.max_output_tokens
    # ...but an explicit caller cap still wins.
    assert lm._get_provider_max_tokens_param({"max_output_tokens": 42}) == 42


class _FakeSelf:
    model = "m"


def test_truncation_is_annotated_and_warned():
    resp = GenerateResponse(content="clipped", model="m", finish_reason="length", usage={"output_tokens": 8192})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BaseProvider._annotate_output_truncation(_FakeSelf(), resp)
    assert resp.metadata.get("output_truncated") is True
    assert any("Output truncated" in str(x.message) for x in w)


def test_stop_reason_is_not_flagged():
    resp = GenerateResponse(content="done", model="m", finish_reason="stop")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BaseProvider._annotate_output_truncation(_FakeSelf(), resp)
    assert not (resp.metadata or {}).get("output_truncated")
    assert not w


def test_annotation_is_idempotent_across_chunks():
    """Streaming annotates per-chunk (the helper is called on every processed
    chunk); a length-finish terminal chunk must warn EXACTLY once even if the
    helper is invoked on it more than once."""
    resp = GenerateResponse(content="", model="m", finish_reason="length", usage={"output_tokens": 64})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BaseProvider._annotate_output_truncation(_FakeSelf(), resp)
        BaseProvider._annotate_output_truncation(_FakeSelf(), resp)  # second pass: no-op
    assert resp.metadata.get("output_truncated") is True
    assert sum("Output truncated" in str(x.message) for x in w) == 1


def test_non_length_stream_chunks_are_not_flagged():
    """A mid-stream content chunk (finish_reason=None) must never be annotated —
    only the terminal length chunk is truncation."""
    mid = GenerateResponse(content="1\n2\n", model="m", finish_reason=None)
    BaseProvider._annotate_output_truncation(_FakeSelf(), mid)
    assert not (mid.metadata or {}).get("output_truncated")
