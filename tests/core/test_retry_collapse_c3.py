"""C3 — retry collapse (entity-topology plan item 12): cancellable backoff,
Retry-After honoring, single_attempt preset, per-endpoint damping.

Design: docs/backlog/proposed/0816_retry_collapse_cancellable_backoff_and_herd_budgets.md
(both reviewer asks discharged: runtime c260, agency 0015/082837Z). Boundary pin:
these are PER-PROCESS primitives — cross-process fairness is item 11's admission
controller. Consumer pins tested here by name:
- runtime pin 1: the Harmony carve-out SURVIVES the collapse (single-attempt inner
  + outer resample still absorbs the 400-signature race);
- runtime pin 2: typed errors + status_code remain the only 4xx gate;
- item-11 ask 3: budget exhaustion = ordinary loud per-run failure (typed error,
  labeled message), never a queue.
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest

from abstractcore.core.endpoint_damping import (
    EndpointDampingRegistry,
    get_endpoint_damping_registry,
)
from abstractcore.core.retry import (
    RetryCancelledError,
    RetryConfig,
    RetryManager,
)
from abstractcore.exceptions import (
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# single_attempt() preset (double-stack collapse, runtime-confirmed shape)
# ---------------------------------------------------------------------------

def test_single_attempt_makes_exactly_one_call():
    manager = RetryManager(RetryConfig.single_attempt())
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise ProviderAPIError("transient", status_code=503)

    with pytest.raises(ProviderAPIError):
        manager.execute_with_retry(flaky, provider_key="ep:m")
    assert calls["n"] == 1  # no inner resample — the OUTER policy owns attempts


def test_single_attempt_keeps_breaker_active():
    config = RetryConfig.single_attempt(failure_threshold=2, recovery_timeout=60.0)
    manager = RetryManager(config)

    def failing():
        raise ProviderAPIError("down", status_code=503)

    for _ in range(2):
        with pytest.raises(ProviderAPIError):
            manager.execute_with_retry(failing, provider_key="ep:m")

    # Threshold reached: the breaker refuses BEFORE any further attempt.
    with pytest.raises(ProviderAPIError, match="Circuit breaker open"):
        manager.execute_with_retry(failing, provider_key="ep:m")


def test_collapse_mode_outer_resample_absorbs_harmony_artifact():
    """Runtime pin 1: single-attempt inner + an outer retry layer (simulated here by
    the caller's own loop) still absorbs the Harmony 400 race — the inner layer
    raises the retryable-typed error once, the outer resample succeeds."""
    manager = RetryManager(RetryConfig.single_attempt())
    calls = {"n": 0}

    def race_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderAPIError(
                'OpenAI-Compatible API error (400): unexpected tokens remaining in '
                'message header: Some("to=tool") [transient harmony generation artifact]',
                status_code=400,
            )
        return "ok"

    # Outer layer (runtime's RetryPolicy stand-in): retries on retryable-typed errors.
    result = None
    for _ in range(3):
        try:
            result = manager.execute_with_retry(race_once, provider_key="ep:m")
            break
        except ProviderAPIError:
            continue  # runtime's classifier: ProviderAPIError = retryable
    assert result == "ok"
    assert calls["n"] == 2  # exactly one resample, owned by the OUTER layer


def test_typed_errors_remain_the_4xx_gate():
    """Runtime pin 2: with inner retries gone, InvalidRequestError must stay
    never-retried by ANY RetryManager in the stack."""
    manager = RetryManager(RetryConfig(max_attempts=3))
    calls = {"n": 0}

    def invalid():
        calls["n"] += 1
        raise InvalidRequestError("System message must be at the beginning.", status_code=400)

    with pytest.raises(InvalidRequestError):
        manager.execute_with_retry(invalid, provider_key="ep:m")
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Cancellable backoff
# ---------------------------------------------------------------------------

def test_cancel_during_backoff_is_observed_fast_and_labeled():
    manager = RetryManager(RetryConfig(max_attempts=3, initial_delay=60.0,
                                       max_delay=60.0, use_jitter=False))
    cancel = threading.Event()

    def failing():
        raise ProviderAPIError("transient", status_code=503)

    def cancel_soon():
        time.sleep(0.3)
        cancel.set()

    threading.Thread(target=cancel_soon, daemon=True).start()
    start = time.monotonic()
    with pytest.raises(RetryCancelledError) as exc_info:
        manager.execute_with_retry(failing, provider_key="ep:m", cancel_event=cancel)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"cancel took {elapsed:.2f}s to observe (bound: ~1s slice + margin)"
    assert "[retry cancelled by host]" in str(exc_info.value)
    assert isinstance(exc_info.value.last_error, ProviderAPIError)


def test_no_cancel_event_preserves_wait_behavior():
    manager = RetryManager(RetryConfig(max_attempts=2, initial_delay=0.2,
                                       max_delay=0.2, use_jitter=False))
    calls = {"n": 0}

    def fail_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderAPIError("transient", status_code=503)
        return "ok"

    start = time.monotonic()
    assert manager.execute_with_retry(fail_once, provider_key="ep:m") == "ok"
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2  # the full backoff still happens without a cancel signal
    assert calls["n"] == 2


def test_pre_set_cancel_never_retried_class_unaffected():
    # RetryCancelledError must classify as non-retryable if it ever reaches
    # another RetryManager (fails safe — a cancel is never resampled).
    manager = RetryManager(RetryConfig(max_attempts=3))
    err = RetryCancelledError("[retry cancelled by host] x", last_error=None)
    assert manager.should_retry(err, attempt=1) is False


# ---------------------------------------------------------------------------
# Retry-After honoring
# ---------------------------------------------------------------------------

def test_retry_after_beats_jitter_and_caps_at_max_delay(monkeypatch):
    waits: List[float] = []
    manager = RetryManager(RetryConfig(max_attempts=2, initial_delay=0.01,
                                       max_delay=5.0, use_jitter=False))
    monkeypatch.setattr(RetryManager, "_wait_cancellable",
                        staticmethod(lambda delay, cancel_event: waits.append(delay)))
    calls = {"n": 0}

    def rate_limited():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("429", status_code=429, retry_after_s=3.0)
        return "ok"

    assert manager.execute_with_retry(rate_limited, provider_key="ep:m") == "ok"
    assert waits == [3.0]  # the server's wait, not the 0.01s jitter guess

    # And a hostile/huge header is capped by OUR max_delay.
    waits.clear()
    calls["n"] = 0

    def rate_limited_huge():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("429", status_code=429, retry_after_s=7200.0)
        return "ok"

    assert manager.execute_with_retry(rate_limited_huge, provider_key="ep:m") == "ok"
    assert waits == [5.0]


def test_absent_retry_after_unchanged(monkeypatch):
    waits: List[float] = []
    manager = RetryManager(RetryConfig(max_attempts=2, initial_delay=0.25,
                                       max_delay=0.25, use_jitter=False))
    monkeypatch.setattr(RetryManager, "_wait_cancellable",
                        staticmethod(lambda delay, cancel_event: waits.append(delay)))
    calls = {"n": 0}

    def transient():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderAPIError("503", status_code=503)  # no retry_after_s
        return "ok"

    assert manager.execute_with_retry(transient, provider_key="ep:m") == "ok"
    assert waits == [0.25]  # computed delay untouched


def test_retry_after_header_parsing_seconds_and_absent():
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    class _R:
        def __init__(self, headers):
            self.headers = headers

    assert OpenAICompatibleProvider._extract_retry_after_s(_R({"retry-after": "7"})) == 7.0
    assert OpenAICompatibleProvider._extract_retry_after_s(_R({"Retry-After": "2.5"})) == 2.5
    assert OpenAICompatibleProvider._extract_retry_after_s(_R({})) is None
    assert OpenAICompatibleProvider._extract_retry_after_s(_R({"retry-after": "garbage"})) is None
    assert OpenAICompatibleProvider._extract_retry_after_s(None) is None


# ---------------------------------------------------------------------------
# Piece 2b: status_code survives the native wrap sites
# ---------------------------------------------------------------------------

def test_wrap_sites_preserve_status_code_and_retry_after():
    from abstractcore.providers.base import BaseProvider

    class _SDKStatusError(Exception):
        """openai/anthropic APIStatusError shape."""
        def __init__(self, message, status_code, headers=None):
            super().__init__(message)
            self.status_code = status_code
            if headers is not None:
                class _Resp:
                    pass
                self.response = _Resp()
                self.response.headers = headers

    assert BaseProvider._status_code_from_exception(
        _SDKStatusError("rate limit exceeded", 429)) == 429
    assert BaseProvider._retry_after_from_exception(
        _SDKStatusError("rate limit exceeded", 429, headers={"retry-after": "11"})) == 11.0

    class _HttpxLikeError(Exception):
        """httpx.HTTPStatusError shape (status on .response only)."""
        def __init__(self, message, status_code):
            super().__init__(message)
            class _Resp:
                pass
            self.response = _Resp()
            self.response.status_code = status_code

    assert BaseProvider._status_code_from_exception(_HttpxLikeError("bad request", 400)) == 400
    assert BaseProvider._status_code_from_exception(ValueError("no http anywhere")) is None


# ---------------------------------------------------------------------------
# Endpoint damping: shared breaker + retry budget (the herd demo)
# ---------------------------------------------------------------------------

def _fresh_registry() -> EndpointDampingRegistry:
    registry = EndpointDampingRegistry()
    return registry


def test_shared_breaker_first_trip_stops_the_rest():
    """Herd core: N managers on ONE endpoint share failure state — after the
    breaker trips, remaining instances fail fast WITHOUT calling the endpoint."""
    registry = _fresh_registry()
    config = RetryConfig.single_attempt(failure_threshold=3, recovery_timeout=60.0)
    endpoint_calls = {"n": 0}

    def dead_endpoint():
        endpoint_calls["n"] += 1
        raise ProviderAPIError("503 service unavailable", status_code=503)

    managers = []
    for _ in range(8):
        m = RetryManager(config)
        m.damping_domain = registry.domain_for("http://one-endpoint:8000/v1", "m", config)
        managers.append(m)

    failures = 0
    for m in managers:
        try:
            m.execute_with_retry(dead_endpoint, provider_key="ep:m")
        except ProviderAPIError:
            failures += 1

    assert failures == 8
    # WITHOUT damping: 8 calls (single-attempt) or 8x3 with default retries.
    # WITH the shared breaker (threshold 3): exactly 3 real calls, 5 refused fast.
    assert endpoint_calls["n"] == 3


def test_without_damping_every_instance_probes_independently():
    """The counterfactual arm of the herd demo: per-instance breakers never share,
    so all 8 instances burn real calls."""
    config = RetryConfig.single_attempt(failure_threshold=3)
    endpoint_calls = {"n": 0}

    def dead_endpoint():
        endpoint_calls["n"] += 1
        raise ProviderAPIError("503 service unavailable", status_code=503)

    for _ in range(8):
        m = RetryManager(config)  # no damping_domain
        with pytest.raises(ProviderAPIError):
            m.execute_with_retry(dead_endpoint, provider_key="ep:m")

    assert endpoint_calls["n"] == 8


def test_retry_budget_exhaustion_fails_fast_typed_and_labeled():
    """Item-11 ask-3 shape: no slot -> the LAST typed error, labeled, immediately —
    never a queue, never a new exception class."""
    registry = _fresh_registry()
    config = RetryConfig(max_attempts=3, initial_delay=30.0, max_delay=30.0,
                         use_jitter=False, failure_threshold=100)
    domain = registry.domain_for("http://busy:8000/v1", "m", config, max_retry_waiters=1)

    # Occupy the single retry slot.
    assert domain.try_acquire_retry_slot() is True
    try:
        manager = RetryManager(config)
        manager.damping_domain = domain

        def rate_limited():
            raise RateLimitError("429 too many requests", status_code=429)

        start = time.monotonic()
        with pytest.raises(RateLimitError) as exc_info:
            manager.execute_with_retry(rate_limited, provider_key="ep:m")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0  # failed FAST — no 30s backoff wait
        assert "[retry budget exhausted for endpoint" in str(exc_info.value)
        assert exc_info.value.status_code == 429  # typed fact preserved
    finally:
        domain.release_retry_slot()


def test_registry_is_keyed_per_endpoint():
    registry = _fresh_registry()
    config = RetryConfig()
    a1 = registry.domain_for("http://a:8000/v1", "m1", config)
    a2 = registry.domain_for("http://a:8000/v1/", "m1", config)  # trailing slash normalized
    b = registry.domain_for("http://b:8000/v1", "m1", config)
    assert a1 is a2
    assert a1 is not b


def test_provider_opt_in_resolves_domain_lazily():
    """endpoint_damping=True at construction wires the shared domain at first
    generate; default (absent) leaves damping off."""
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    p = OpenAICompatibleProvider(
        model="m", base_url="http://127.0.0.1:9/v1", api_key="x",
        validate_model=False,
        retry_config=RetryConfig.single_attempt(),
        endpoint_damping=True,
    )
    assert p._endpoint_damping_requested is True
    assert p.retry_manager.damping_domain is None  # lazy — nothing resolved yet

    class _OkClient:
        def post(self, url, json=None, headers=None):
            class _R:
                status_code = 200
                @staticmethod
                def json():
                    return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            return _R()

    p.client = _OkClient()
    assert p.generate("hi", max_output_tokens=4).content == "ok"
    domain = p.retry_manager.damping_domain
    assert domain is not None
    assert domain.key == ("http://127.0.0.1:9/v1", "m")
    # Same endpoint+model from a second instance shares the SAME domain object.
    assert get_endpoint_damping_registry().domain_for(
        "http://127.0.0.1:9/v1", "m", p.retry_manager.config) is domain

    p_off = OpenAICompatibleProvider(
        model="m2", base_url="http://127.0.0.1:9/v1", api_key="x", validate_model=False)
    assert p_off._endpoint_damping_requested is False
