"""C2 regression harness (offline half) — Harmony generation-artifact classification.

gpt-oss models on vLLM (e.g. OVH AI Endpoints) sometimes emit output that violates
their own Harmony template (an unclosed `to=...` recipient header, primed by tool-ish
prompt content). The server's strict `openai-harmony` parser rejects the MODEL'S OWN
OUTPUT and surfaces it as HTTP 400 "unexpected tokens remaining in message header"
on a perfectly valid request (vllm#23567, openai/harmony#38/#80; upstream lenient
parser vllm#28303 not deployed everywhere). Live incident 2026-07-09: ~21 unattended
entity-loop ticks died in one night because the 400 was classified InvalidRequestError
(never retried).

These tests pin the shipped carve-out end to end:
- the 400 signature maps to ProviderAPIError (transient), never InvalidRequestError;
- unrelated 400s keep their InvalidRequestError classification (no blanket retry);
- the RetryManager chain actually resamples the transient class exactly once;
- a full generate() absorbs one artifact 400 into a successful retry (the incident's
  post-fix behavior: errors still occur, each absorbs into a successful call);
- the operator-declared fallback pair (laurent c157) stays registry-READY.

The live half (native tool-call round-trip on the real OVH path) lives in
tests/providers/test_gpt_oss_120b_ovh_live_regression.py, env-gated.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from abstractcore.core.retry import RetryConfig, RetryManager, RetryableErrorType
from abstractcore.exceptions import InvalidRequestError, ProviderAPIError
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


# The exact live signature from the OVH incident (2026-07-09).
HARMONY_HEADER_400 = 'unexpected tokens remaining in message header: Some("to=tool")'


class _Resp:
    def __init__(self, status_code: int, body: Dict[str, Any]):
        self.status_code = status_code
        self._body = body

    def json(self) -> Dict[str, Any]:
        return self._body

    @property
    def text(self) -> str:
        return json.dumps(self._body)

    def read(self) -> bytes:  # parity with httpx buffering
        return self.text.encode("utf-8")


def _error_resp(message: str) -> _Resp:
    return _Resp(400, {"error": {"message": message}})


_SUCCESS_BODY = {
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _bare_provider() -> OpenAICompatibleProvider:
    """Provider with the minimal surface for _raise_for_status (no network in __init__)."""
    p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    p.base_url = "http://127.0.0.1:9/v1"  # unreachable on purpose
    p.model = "gpt-oss-120b"
    return p


# ---------------------------------------------------------------------------
# Classification pins (_raise_for_status)
# ---------------------------------------------------------------------------

def test_harmony_header_artifact_400_is_transient_provider_api_error():
    p = _bare_provider()
    with pytest.raises(ProviderAPIError) as exc_info:
        p._raise_for_status(_error_resp(HARMONY_HEADER_400))
    # Not the never-retried class.
    assert not isinstance(exc_info.value, InvalidRequestError)
    msg = str(exc_info.value)
    assert "transient harmony generation artifact" in msg
    assert HARMONY_HEADER_400.split(":")[0] in msg  # detail preserved for operators


@pytest.mark.parametrize(
    "detail",
    [
        "HarmonyError: could not parse assistant message",
        "openai-harmony parser rejected the completion",
        "openai_harmony: invalid recipient header",
    ],
)
def test_harmony_sibling_signatures_are_transient(detail: str):
    p = _bare_provider()
    with pytest.raises(ProviderAPIError) as exc_info:
        p._raise_for_status(_error_resp(detail))
    assert not isinstance(exc_info.value, InvalidRequestError)


def test_plain_400_stays_invalid_request():
    """The carve-out must never widen into a blanket 400 retry (deterministic 4xx
    stays deterministic — e.g. the strict system-message incident shape)."""
    p = _bare_provider()
    with pytest.raises(InvalidRequestError):
        p._raise_for_status(_error_resp("System message must be at the beginning."))


# ---------------------------------------------------------------------------
# Retry-chain pins (RetryManager contract the carve-out depends on)
# ---------------------------------------------------------------------------

def test_retry_manager_resamples_harmony_artifact_once():
    manager = RetryManager(RetryConfig(max_attempts=3))
    err = ProviderAPIError(f"OpenAI-Compatible API error (400): {HARMONY_HEADER_400} "
                           "[transient harmony generation artifact - the model's sampled "
                           "output violated its template; a retry resamples]")

    assert manager.classify_error(err) is RetryableErrorType.API_ERROR
    assert manager.should_retry(err, attempt=1) is True   # one resample
    assert manager.should_retry(err, attempt=2) is False  # bounded, not unlimited


def test_retry_manager_never_retries_invalid_request():
    manager = RetryManager(RetryConfig(max_attempts=3))
    err = InvalidRequestError("OpenAI-Compatible API error (400): System message must be at the beginning.")
    assert manager.should_retry(err, attempt=1) is False


# ---------------------------------------------------------------------------
# Wire-path pin: generate() absorbs one artifact 400 into a successful retry
# ---------------------------------------------------------------------------

class _FlakyHarmonyClient:
    """400s with the Harmony artifact signature once, then succeeds — the live race shape."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []

    def post(self, url, json=None, headers=None):  # noqa: A002 - httpx signature
        self.requests.append(dict(json or {}))
        if len(self.requests) == 1:
            return _error_resp(HARMONY_HEADER_400)
        return _Resp(200, dict(_SUCCESS_BODY))


def test_wire_path_generate_absorbs_harmony_artifact_into_retry():
    provider = OpenAICompatibleProvider(
        model="gpt-oss-120b",
        base_url="http://127.0.0.1:9/v1",
        api_key="x",
        validate_model=False,
        retry_config=RetryConfig(initial_delay=0.0, max_delay=0.0, use_jitter=False),
    )
    provider.client = _FlakyHarmonyClient()

    response = provider.generate("hello", max_output_tokens=8)

    assert response.content == "ok"
    assert len(provider.client.requests) == 2  # 400 artifact absorbed, resample succeeded
    # The request itself was untouched between attempts (a resample, not a mutation).
    assert provider.client.requests[0]["messages"] == provider.client.requests[1]["messages"]


class _AlwaysInvalidClient:
    def __init__(self):
        self.requests: List[Dict[str, Any]] = []

    def post(self, url, json=None, headers=None):  # noqa: A002
        self.requests.append(dict(json or {}))
        return _error_resp("System message must be at the beginning.")


def test_wire_path_generate_does_not_retry_plain_400():
    provider = OpenAICompatibleProvider(
        model="gpt-oss-120b",
        base_url="http://127.0.0.1:9/v1",
        api_key="x",
        validate_model=False,
        retry_config=RetryConfig(initial_delay=0.0, max_delay=0.0, use_jitter=False),
    )
    provider.client = _AlwaysInvalidClient()

    with pytest.raises(InvalidRequestError):
        provider.generate("hello", max_output_tokens=8)

    assert len(provider.client.requests) == 1  # deterministic 4xx: exactly one attempt


# ---------------------------------------------------------------------------
# Fallback-pair readiness pin (consensus plan, core C2: laurent c157 / c163)
# ---------------------------------------------------------------------------

def test_operator_declared_fallback_pair_stays_registry_ready():
    """The operator-declared fallback substrate (LMStudio `qwen/qwen3.6-35b-a3b`) must
    stay registered native-tools so engaging it never needs a registry delta. The
    primary path's model is pinned alongside. Engagement itself is operator config
    (never a code default — 04:26 ruling); this only pins REGISTRY readiness."""
    from abstractcore.architectures.detection import get_model_capabilities

    fallback = get_model_capabilities("qwen/qwen3.6-35b-a3b")
    assert fallback.get("tool_support") == "native"
    assert int(fallback.get("max_tokens") or 0) >= 262144

    primary = get_model_capabilities("gpt-oss-120b")
    assert primary.get("tool_support") == "native"
