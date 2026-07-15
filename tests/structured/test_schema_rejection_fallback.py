"""Schema-rejection fallback for native structured output (airelay 422 incident, 2026-07-15).

Strict-schema backends (OpenAI strict mode and subscription relays in front of
it) refuse structured-output requests whose JSON schema violates strict rules
(e.g. a free-form `{"type": "object"}` dict without `properties`) with a
deterministic 4xx. The structured handler must detect that class, retry
through the prompted lane (#FALLBACK), and cache the decision per
(provider, base_url, model, schema fingerprint) — while auth/context-length/
rate-limit errors keep their existing fatal semantics.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from abstractcore.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
)
from abstractcore.structured.handler import StructuredOutputHandler
from abstractcore.structured.schema_compat import (
    SchemaRejectionRegistry,
    is_schema_rejection_error,
    schema_rejection_registry,
)


# The two error shapes observed LIVE against the subscription relay (127.0.0.1:8317).
OBSERVED_ADDITIONAL_PROPERTIES_422 = (
    "OpenAI-compatible server API error (422): The subscription backend requires every "
    "object schema to set `additionalProperties` to `false`; incompatible value at "
    "`properties.next_tool_calls.items.properties.arguments`. "
    "[http://127.0.0.1:8317/v1/chat/completions]"
)
OBSERVED_INVALID_JSON_SCHEMA_400 = (
    "OpenAI-compatible server API error (400): {'error': {'message': \"Invalid schema for "
    "response_format 'ReActVerifier': In context=('properties', 'next_tool_calls', 'items'), "
    "'required' is required to be supplied and to be an array including every key in "
    "properties. Extra required key 'arguments' supplied.\", 'type': 'invalid_request_error', "
    "'param': 'text.format.schema', 'code': 'invalid_json_schema'}}"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    schema_rejection_registry.clear()
    yield
    schema_rejection_registry.clear()


class _Verdict(BaseModel):
    complete: bool
    notes: str


# ---------------------------------------------------------------------------
# Detector: positives (observed evidence) and negatives (must never match)
# ---------------------------------------------------------------------------

def test_detector_matches_observed_additional_properties_422() -> None:
    err = ProviderAPIError(OBSERVED_ADDITIONAL_PROPERTIES_422, status_code=422)
    assert is_schema_rejection_error(err) is True


def test_detector_matches_observed_invalid_json_schema_400() -> None:
    err = InvalidRequestError(OBSERVED_INVALID_JSON_SCHEMA_400, status_code=400)
    assert is_schema_rejection_error(err) is True


def test_detector_matches_status_from_message_dialect_when_attribute_missing() -> None:
    # Some wrap layers lose the status attribute; our providers' message dialect
    # ("API error (NNN)") still carries it.
    err = RuntimeError(OBSERVED_ADDITIONAL_PROPERTIES_422)
    assert is_schema_rejection_error(err) is True


def test_detector_rejects_auth_errors() -> None:
    err = AuthenticationError(
        "OpenAI-compatible server API error (401): Unauthorized", status_code=401
    )
    assert is_schema_rejection_error(err) is False


def test_detector_rejects_context_length_400() -> None:
    err = InvalidRequestError(
        "OpenAI-compatible server API error (400): This model's maximum context length is "
        "8192 tokens. However, your messages resulted in 9000 tokens.",
        status_code=400,
    )
    assert is_schema_rejection_error(err) is False


def test_detector_rejects_rate_limit_even_with_schema_words() -> None:
    err = RateLimitError(
        "OpenAI-compatible server API error (429): rate limit on json_schema requests",
        status_code=429,
    )
    assert is_schema_rejection_error(err) is False


def test_detector_rejects_5xx_with_schema_words() -> None:
    err = ProviderAPIError(
        "OpenAI-compatible server API error (503): response_format temporarily unavailable",
        status_code=503,
    )
    assert is_schema_rejection_error(err) is False


def test_detector_rejects_generic_400_without_schema_signature() -> None:
    err = InvalidRequestError(
        "OpenAI-compatible server API error (400): model must be a string", status_code=400
    )
    assert is_schema_rejection_error(err) is False


def test_detector_rejects_errors_without_any_status_evidence() -> None:
    # Schema words alone are not enough — a 4xx status (attribute or message
    # dialect) is required.
    assert is_schema_rejection_error(RuntimeError("response_format json_schema broke")) is False
    assert is_schema_rejection_error(ValueError("boom")) is False


# ---------------------------------------------------------------------------
# Handler: prompted fallback on schema rejection + per-session caching
# ---------------------------------------------------------------------------

class _StrictBackendProvider:
    """Native-capable provider whose backend refuses the schema with a 422."""

    def __init__(self, model: str = "gpt-5.4", base_url: str = "http://relay.test/v1") -> None:
        self.model = model
        self.base_url = base_url
        self.model_capabilities = {"structured_output": "native"}
        self.native_calls = 0
        self.prompted_calls = 0

    def _generate_internal(self, *, prompt: str, **kwargs):
        if kwargs.get("response_model") is not None:
            self.native_calls += 1
            raise ProviderAPIError(OBSERVED_ADDITIONAL_PROPERTIES_422, status_code=422)
        self.prompted_calls += 1
        return SimpleNamespace(content='{"complete": true, "notes": "done"}', finish_reason="stop")


def test_schema_rejection_falls_back_to_prompted_lane() -> None:
    provider = _StrictBackendProvider()
    handler = StructuredOutputHandler()

    out = handler.generate_structured(provider=provider, prompt="verify", response_model=_Verdict)

    assert isinstance(out, _Verdict)
    assert out.complete is True
    assert provider.native_calls == 1  # exactly one doomed attempt, never retried
    assert provider.prompted_calls == 1


def test_schema_rejection_decision_is_cached_per_session() -> None:
    provider = _StrictBackendProvider()

    out1 = StructuredOutputHandler().generate_structured(
        provider=provider, prompt="verify one", response_model=_Verdict
    )
    out2 = StructuredOutputHandler().generate_structured(
        provider=provider, prompt="verify two", response_model=_Verdict
    )

    assert isinstance(out1, _Verdict) and isinstance(out2, _Verdict)
    # The second call must NOT re-hit the 4xx: one native attempt total.
    assert provider.native_calls == 1
    assert provider.prompted_calls == 2


def test_cache_is_scoped_per_schema() -> None:
    provider = _StrictBackendProvider()

    class _OtherModel(BaseModel):
        complete: bool
        notes: str

        model_config = {"title": "OtherVerdict"}

    StructuredOutputHandler().generate_structured(
        provider=provider, prompt="verify", response_model=_Verdict
    )
    StructuredOutputHandler().generate_structured(
        provider=provider, prompt="verify", response_model=_OtherModel
    )
    # A different schema fingerprint gets its own native attempt.
    assert provider.native_calls == 2


def test_cache_is_scoped_per_endpoint_and_model() -> None:
    a = _StrictBackendProvider(model="gpt-5.4", base_url="http://relay-a.test/v1")
    b = _StrictBackendProvider(model="gpt-5.4", base_url="http://relay-b.test/v1")

    StructuredOutputHandler().generate_structured(provider=a, prompt="p", response_model=_Verdict)
    StructuredOutputHandler().generate_structured(provider=b, prompt="p", response_model=_Verdict)
    assert a.native_calls == 1
    assert b.native_calls == 1  # b's endpoint decides for itself


def test_non_schema_4xx_still_raises() -> None:
    class _AuthFailingProvider(_StrictBackendProvider):
        def _generate_internal(self, *, prompt: str, **kwargs):
            if kwargs.get("response_model") is not None:
                self.native_calls += 1
                raise AuthenticationError(
                    "OpenAI-compatible server API error (401): Unauthorized", status_code=401
                )
            self.prompted_calls += 1
            return SimpleNamespace(content="{}", finish_reason="stop")

    provider = _AuthFailingProvider()
    with pytest.raises(AuthenticationError):
        StructuredOutputHandler().generate_structured(
            provider=provider, prompt="verify", response_model=_Verdict
        )
    assert provider.prompted_calls == 0  # auth failures never lane-switch
    assert schema_rejection_registry.rejection_reason(provider, _Verdict) is None


def test_native_success_never_touches_registry() -> None:
    class _HealthyProvider(_StrictBackendProvider):
        def _generate_internal(self, *, prompt: str, **kwargs):
            self.native_calls += 1
            return SimpleNamespace(
                content='{"complete": true, "notes": "native"}', finish_reason="stop"
            )

    provider = _HealthyProvider()
    out = StructuredOutputHandler().generate_structured(
        provider=provider, prompt="verify", response_model=_Verdict
    )
    assert out.notes == "native"
    assert schema_rejection_registry.rejection_reason(provider, _Verdict) is None


def test_registry_bounds_entries() -> None:
    registry = SchemaRejectionRegistry(max_entries=2)
    providers = [
        _StrictBackendProvider(model=f"m{i}", base_url="http://x.test/v1") for i in range(3)
    ]
    for p in providers:
        registry.mark_rejected(p, _Verdict, "err")
    # Oldest entry evicted, newest two retained.
    assert registry.rejection_reason(providers[0], _Verdict) is None
    assert registry.rejection_reason(providers[1], _Verdict) == "err"
    assert registry.rejection_reason(providers[2], _Verdict) == "err"


# ---------------------------------------------------------------------------
# Prompted-lane JSON extraction: reasoning noise (think blocks)
# ---------------------------------------------------------------------------

def test_extract_json_strips_think_blocks_before_extraction() -> None:
    """Reasoning models can emit `<think>` blocks whose braces defeat the
    first-object regex; the fallback lane must survive them."""
    handler = StructuredOutputHandler()
    handler.current_provider = SimpleNamespace(
        architecture_config={"thinking_tags": ["<think>", "</think>"]},
        model_capabilities={},
    )
    content = (
        "<think>The schema wants {\"complete\": bool}. I think {\"complete\": false} at "
        "first... no, everything is done.</think>\n"
        '{"complete": true, "notes": "after reasoning"}'
    )
    extracted = handler._extract_json(content)
    import json

    assert json.loads(extracted) == {"complete": True, "notes": "after reasoning"}


def test_extract_json_without_provider_context_is_unchanged() -> None:
    handler = StructuredOutputHandler()
    assert handler._extract_json('{"a": 1}') == '{"a": 1}'
