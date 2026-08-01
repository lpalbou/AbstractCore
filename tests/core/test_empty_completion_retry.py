"""Empty-completion transient (operator 2026-08-01): the entity relay answered
HTTP 200 with choices[0].message = {"content": null}, no tool_calls,
finish_reason "stop", usage null — a transient upstream failure dressed as a
valid completion. BaseProvider accepted it as a completed generation, so the
retry manager recorded SUCCESS and every consumer (entity visit turns, the
runtime llm_call effect, the own-time tick loop) failed after ONE attempt with
a silent empty reply.

Pins:
- an all-empty non-streaming completion raises EmptyCompletionError INSIDE the
  retried closure — full retries (max_attempts), not API_ERROR's single resample;
- exhaustion surfaces an honest label naming the attempt count, same type
  (status-code-first classifiers downstream keep seeing the class);
- a resample that recovers returns normally;
- tool-call-only and reasoning-bearing completions are legitimate answer
  shapes and are never classified empty.
"""

from __future__ import annotations

import pytest

from abstractcore.core.retry import RetryableErrorType, RetryConfig, RetryManager
from abstractcore.core.types import GenerateResponse
from abstractcore.exceptions import EmptyCompletionError, ProviderAPIError
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

from tests.provider_stubs import StaticProvider


def _empty_response() -> GenerateResponse:
    # The live incident's exact signature: content null, no tool_calls,
    # finish_reason "stop", usage null.
    return GenerateResponse(content=None, model="stub", finish_reason="stop", usage=None)


class ScriptedProvider(StaticProvider):
    """StaticProvider that replays a scripted list of responses per call."""

    def __init__(self, model: str, responses, **kwargs):
        super().__init__(model, **kwargs)
        self._responses = list(responses)
        self.calls = 0

    def _generate_internal(self, *args, **kwargs):
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _provider(responses, max_attempts: int = 3) -> ScriptedProvider:
    # Zero-delay, jitter-free retries: these tests pin classification and
    # attempt counts, not backoff arithmetic.
    return ScriptedProvider(
        "stub-model",
        responses,
        retry_config=RetryConfig(max_attempts=max_attempts, initial_delay=0.0, use_jitter=False),
    )


# ---------------------------------------------------------------------------
# The fix: all-empty completions are retried, then fail LOUD and labeled
# ---------------------------------------------------------------------------

def test_all_empty_completion_retries_to_exhaustion_and_labels_attempts():
    provider = _provider([_empty_response()], max_attempts=3)

    with pytest.raises(EmptyCompletionError) as exc_info:
        provider.generate(prompt="say something")

    assert provider.calls == 3  # full retries — the whole point of the class
    message = str(exc_info.value)
    assert "empty completion" in message
    assert "no content, no tool calls" in message
    assert "3 attempts" in message  # honest exhaustion label, never a silent ""


def test_empty_completion_recovers_on_resample():
    provider = _provider(
        [_empty_response(), GenerateResponse(content="hello", model="stub", finish_reason="stop")],
        max_attempts=3,
    )

    response = provider.generate(prompt="say something")

    assert provider.calls == 2  # one transient absorbed, then the real answer
    assert response.content == "hello"


# ---------------------------------------------------------------------------
# Legitimate no-prose completions stay untouched
# ---------------------------------------------------------------------------

def test_tool_call_only_completion_is_not_classified_empty():
    tool_response = GenerateResponse(
        content="",
        model="stub",
        finish_reason="tool_calls",
        tool_calls=[{"name": "diary_read", "arguments": {}}],
    )
    provider = _provider([tool_response])

    response = provider.generate(prompt="use your tools")

    assert provider.calls == 1
    assert response.tool_calls  # a native tool election legitimately has no prose


def test_reasoning_only_completion_is_not_classified_empty():
    # Reasoning-channel output with empty content (usually an output-budget
    # truncation): the model DID speak — a different root cause, deliberately
    # excluded from the transient class (resampling the most expensive
    # completions 3x would cost more than it cures).
    reasoning_response = GenerateResponse(
        content="",
        model="stub",
        finish_reason="length",
        metadata={"reasoning": "thinking out loud..."},
    )
    provider = _provider([reasoning_response])

    response = provider.generate(prompt="think hard")

    assert provider.calls == 1
    assert response.reasoning == "thinking out loud..."


def test_streaming_generator_is_exempt_from_the_empty_check():
    # A stream holds no verdict at generate() time — the check must not
    # consume or refuse the generator.
    def _chunks():
        yield GenerateResponse(content="partial", model="stub")

    provider = _provider([_chunks()])
    result = provider.generate(prompt="stream it", stream=True)
    chunks = list(result)
    assert provider.calls == 1
    assert any(c.content for c in chunks)


# ---------------------------------------------------------------------------
# Wire path: the no-choices disguise (openai-compatible parse)
# ---------------------------------------------------------------------------
# operator 2026-08-01: the second disguise of the same transient — a 200 whose
# body has NO choices at all. The parse branch used to fabricate
# content="No response generated", prose that sailed past the base
# empty-completion check as a "real" reply. Pinned here through the REAL
# OpenAICompatibleProvider parse (fake client, harmony-regression house style).


class _Resp:
    def __init__(self, body):
        self.status_code = 200
        self._body = dict(body)
        self.text = ""

    def json(self):
        return dict(self._body)

    def read(self):  # parity with httpx buffering
        return b""


_NO_CHOICES_BODY = {"id": "x", "object": "chat.completion", "choices": [], "usage": None}
_ANSWER_BODY = {
    "choices": [{"message": {"content": "here I am"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class _ScriptedClient:
    """Replays scripted 200 bodies per POST (last body repeats)."""

    def __init__(self, bodies):
        self._bodies = list(bodies)
        self.requests = 0

    def post(self, url, json=None, headers=None):  # noqa: A002 - httpx signature
        body = self._bodies[min(self.requests, len(self._bodies) - 1)]
        self.requests += 1
        return _Resp(body)


def _openai_compatible(bodies) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        model="gpt-5.6-sol",
        base_url="http://127.0.0.1:9/v1",  # unreachable on purpose
        api_key="x",
        validate_model=False,
        retry_config=RetryConfig(max_attempts=3, initial_delay=0.0, use_jitter=False),
    )
    provider.client = _ScriptedClient(bodies)
    return provider


def test_no_choices_completion_retries_and_labels_instead_of_fabricating():
    provider = _openai_compatible([_NO_CHOICES_BODY])

    with pytest.raises(EmptyCompletionError) as exc_info:
        provider.generate(prompt="say something")

    assert provider.client.requests == 3  # full retries, same as the null-content shape
    message = str(exc_info.value)
    assert "no choices" in message  # honest label naming the shape
    assert "3 attempts" in message
    assert "No response generated" not in message  # the fabrication is gone


def test_no_choices_completion_recovers_on_resample():
    provider = _openai_compatible([_NO_CHOICES_BODY, _ANSWER_BODY])

    response = provider.generate(prompt="say something")

    assert provider.client.requests == 2
    assert response.content == "here I am"


# ---------------------------------------------------------------------------
# Retry-manager classification (the layer that grants full attempts)
# ---------------------------------------------------------------------------

def test_retry_manager_classifies_empty_completion_for_full_retries():
    manager = RetryManager(RetryConfig(max_attempts=3, initial_delay=0.0, use_jitter=False))
    error = EmptyCompletionError("stub returned an empty completion (no content, no tool calls)")

    assert manager.classify_error(error) is RetryableErrorType.EMPTY_COMPLETION
    # Full retries: attempt 2 of 3 still retries — unlike plain
    # ProviderAPIError (API_ERROR), which stops after its single resample.
    assert manager.should_retry(error, attempt=2) is True
    assert manager.should_retry(ProviderAPIError("api error"), attempt=2) is False
    # Bounded: exhaustion at max_attempts.
    assert manager.should_retry(error, attempt=3) is False


def test_execute_with_retry_relabels_exhausted_empty_completion():
    manager = RetryManager(RetryConfig(max_attempts=2, initial_delay=0.0, use_jitter=False))
    calls = {"n": 0}

    def always_empty():
        calls["n"] += 1
        raise EmptyCompletionError("stub returned an empty completion (no content, no tool calls)")

    with pytest.raises(EmptyCompletionError) as exc_info:
        manager.execute_with_retry(always_empty, provider_key="stub:m")

    assert calls["n"] == 2
    assert "2 attempts" in str(exc_info.value)
    # Same type end to end: downstream status-code-first classifiers (the
    # runtime's _llm_error_is_retryable) must keep seeing the class.
    assert type(exc_info.value) is EmptyCompletionError
