"""Retry wall-clock budget (2026-07-21 wedge incident).

Per-attempt timeouts STACK across retries: 3 attempts x a 600s client timeout
against a wedged endpoint is 30 minutes of wall clock with the caller stuck on
"Thinking...". `RetryConfig.max_total_wall_clock_s` bounds the SEQUENCE:

- No retry starts once the budget is exceeded (the last error raises loudly).
- The FIRST attempt is never budget-gated (legitimate long generations are
  not retries).
- Default None preserves historical behavior exactly.
- The provider kwarg `retry_wall_clock_budget_s` reaches the config without
  constructing a RetryConfig.
"""

from __future__ import annotations

import time

import pytest

from abstractcore.core.retry import RetryConfig, RetryManager


class _WedgedEndpoint:
    """Simulates a wedged endpoint: every call burns wall clock then times out."""

    def __init__(self, seconds_per_call: float):
        self.seconds_per_call = seconds_per_call
        self.calls = 0

    def __call__(self):
        self.calls += 1
        time.sleep(self.seconds_per_call)
        raise TimeoutError("simulated read timeout on wedged endpoint")


def test_budget_stops_retries_after_first_attempt() -> None:
    config = RetryConfig(max_attempts=3, initial_delay=0.01, max_delay=0.02,
                         max_total_wall_clock_s=0.05)
    manager = RetryManager(config)
    endpoint = _WedgedEndpoint(seconds_per_call=0.08)  # one call exceeds the budget

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        manager.execute_with_retry(endpoint, provider_key="wedged:test")
    elapsed = time.monotonic() - started

    assert endpoint.calls == 1, "budget must stop the sequence before attempt 2"
    assert elapsed < 0.5, f"sequence must not stack attempts ({elapsed:.2f}s)"


def test_first_attempt_is_never_budget_gated() -> None:
    config = RetryConfig(max_attempts=3, max_total_wall_clock_s=0.001)
    manager = RetryManager(config)
    calls = {"n": 0}

    def slow_success():
        calls["n"] += 1
        time.sleep(0.02)  # longer than the whole budget
        return "ok"

    assert manager.execute_with_retry(slow_success, provider_key="slow:test") == "ok"
    assert calls["n"] == 1


def test_default_none_preserves_full_attempt_stack() -> None:
    config = RetryConfig(max_attempts=3, initial_delay=0.01, max_delay=0.02)
    manager = RetryManager(config)
    endpoint = _WedgedEndpoint(seconds_per_call=0.01)

    with pytest.raises(TimeoutError):
        manager.execute_with_retry(endpoint, provider_key="wedged:default")
    assert endpoint.calls == 3, "no budget -> historical 3-attempt behavior"


def test_budget_allows_retries_inside_the_window() -> None:
    config = RetryConfig(max_attempts=3, initial_delay=0.01, max_delay=0.02,
                         max_total_wall_clock_s=10.0)
    manager = RetryManager(config)
    endpoint = _WedgedEndpoint(seconds_per_call=0.01)

    with pytest.raises(TimeoutError):
        manager.execute_with_retry(endpoint, provider_key="wedged:roomy")
    assert endpoint.calls == 3, "a roomy budget must not cut legitimate retries"


def test_provider_kwarg_reaches_retry_config() -> None:
    from abstractcore.providers.base import BaseProvider

    class _Stub(BaseProvider):
        def __init__(self, **kwargs):
            super().__init__(model="stub-model", **kwargs)

        def _generate_internal(self, *a, **k):  # pragma: no cover - not exercised
            raise NotImplementedError

        async def _agenerate_internal(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def get_capabilities(self):  # pragma: no cover - not exercised
            return []

        def list_available_models(self, **k):  # pragma: no cover
            return []

        def unload_model(self, *a, **k):  # pragma: no cover
            return None

    provider = _Stub(retry_wall_clock_budget_s=180)
    assert provider.retry_manager.config.max_total_wall_clock_s == 180.0

    default_provider = _Stub()
    assert default_provider.retry_manager.config.max_total_wall_clock_s is None
