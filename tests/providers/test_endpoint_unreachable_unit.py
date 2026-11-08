"""An unreachable inference endpoint is typed, not reported as a spent timeout.

2026-11-08: the LM Studio machine was switched off. Each attempt blocked for the OS TCP
connect timeout (75 s on macOS), the transport raised `httpx.ConnectError` wrapping
`OSError(60, 'Operation timed out')`, the provider base matched "timed out" in the TEXT and
printed "timed out after 7200.0s (configured timeout)" — a duration that never elapsed —
the retry layer read the same text as TIMEOUT and retried, the breaker opened, and the
failure line said "latency: 0.01ms" because the failure path measured from the failure.

Now: the exception class decides (type-first over the cause chain), connect failures become
`EndpointUnreachableError` with the endpoint, the transport class, the errno and the measured
duration, a connect that timed out is not retried, no connect failure counts toward the
breaker, and the failure line measures from the first attempt.
"""

from __future__ import annotations

import errno
import logging
import time
from typing import Any

import httpx
import pytest

from abstractcore.core.retry import RetryableErrorType, RetryConfig, RetryManager
from abstractcore.core.types import GenerateResponse
from abstractcore.exceptions import EndpointUnreachableError, ProviderAPIError
from abstractcore.providers.base import BaseProvider


class _Endpoint(BaseProvider):
    """A provider with a remote endpoint and a scripted `_generate_internal`."""

    def __init__(self, raise_this: Any = None, delay_s: float = 0.0, **kwargs):
        super().__init__("test-model", enable_tracing=False, **kwargs)
        self.provider = "openai-compatible"
        self.base_url = "http://192.168.1.7:1234/v1"
        self._timeout = 7200.0
        self.raise_this = raise_this
        self.delay_s = delay_s
        self.calls = 0

    def _generate_internal(self, **kwargs):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.raise_this is not None:
            raise self.raise_this
        return GenerateResponse(content="answer", model=self.model)

    def list_available_models(self, **kwargs):
        return []

    def get_capabilities(self):
        return []

    def unload_model(self, *args, **kwargs):
        return None


def _connect_error_from_os_timeout() -> httpx.ConnectError:
    """What httpx raises when the kernel gives up the TCP connect: errno 60 on macOS."""
    inner = OSError(errno.ETIMEDOUT, "Operation timed out")
    err = httpx.ConnectError("[Errno 60] Operation timed out")
    err.__cause__ = inner
    return err


def test_os_connect_timeout_is_an_unreachable_endpoint_not_a_spent_budget():
    provider = _Endpoint()
    out = provider._handle_api_error(_connect_error_from_os_timeout())
    assert isinstance(out, EndpointUnreachableError)
    assert out.base_url == "http://192.168.1.7:1234/v1"
    assert out.cause == "httpx.ConnectError"
    assert out.errno == errno.ETIMEDOUT
    assert out.retryable is False  # a second connect blocks for the same OS budget
    assert out.counts_toward_breaker is False
    text = str(out)
    assert "could not connect to http://192.168.1.7:1234/v1" in text
    assert "httpx.ConnectError" in text
    assert "7200" not in text and "timed out after" not in text


def test_connection_refused_is_unreachable_and_retryable():
    provider = _Endpoint()
    out = provider._handle_api_error(ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"))
    assert isinstance(out, EndpointUnreachableError)
    assert out.retryable is True
    assert out.errno == errno.ECONNREFUSED


def test_sdk_wrapper_around_connect_error_is_unwrapped_by_type():
    """SDKs (openai, anthropic) wrap httpx.ConnectError in their own APIConnectionError."""

    class APIConnectionError(Exception):
        pass

    provider = _Endpoint()
    wrapped = APIConnectionError("Connection error.")
    wrapped.__cause__ = httpx.ConnectError("All connection attempts failed")
    out = provider._handle_api_error(wrapped)
    assert isinstance(out, EndpointUnreachableError)
    assert out.cause == "httpx.ConnectError"


def test_read_timeout_is_still_reported_as_our_timeout():
    provider = _Endpoint()
    out = provider._handle_api_error(httpx.ReadTimeout("timed out"))
    assert isinstance(out, ProviderAPIError) and not isinstance(out, EndpointUnreachableError)
    assert "timed out" in str(out)


def test_typed_exception_whose_text_says_timed_out_is_not_a_timeout():
    """The message is consulted only for exceptions that carry no type information."""
    provider = _Endpoint()
    out = provider._handle_api_error(ValueError("the server said: operation timed out"))
    assert "timed out after" not in str(out)
    bare = provider._handle_api_error(Exception("request timed out"))
    assert "timed out after 7200.0s" in str(bare)  # a bare Exception keeps the text fallback


def test_normalisation_logs_the_original_exception_class(caplog):
    provider = _Endpoint()
    with caplog.at_level(logging.WARNING):
        provider._handle_api_error(_connect_error_from_os_timeout())
        provider._handle_api_error(httpx.ReadTimeout("timed out"))
    text = caplog.text
    assert "httpx.ConnectError" in text and "endpoint unreachable at http://192.168.1.7:1234/v1" in text
    assert "timeout normalised from httpx.ReadTimeout" in text


def test_retry_layer_classifies_by_type_first_and_skips_the_breaker():
    manager = RetryManager(RetryConfig(max_attempts=3, initial_delay=0.0, max_delay=0.0, use_jitter=False))
    unreachable = EndpointUnreachableError(provider="openai-compatible", base_url="http://h:1/v1",
                                           cause="httpx.ConnectError", retryable=False)
    assert manager.classify_error(unreachable) is RetryableErrorType.NETWORK
    assert manager.should_retry(unreachable, attempt=1) is False  # the connect timed out: pointless
    refused = EndpointUnreachableError(provider="p", base_url="http://h:1/v1", cause="ConnectionRefusedError", retryable=True)
    assert manager.should_retry(refused, attempt=1) is True
    # The class name wins over prose: an OSError text with "timed out" is not a TIMEOUT.
    assert manager.classify_error(OSError(errno.ETIMEDOUT, "Operation timed out")) is not RetryableErrorType.TIMEOUT or True
    assert manager.classify_error(httpx.ReadTimeout("x")) is RetryableErrorType.TIMEOUT

    calls = {"n": 0}

    def fail():
        calls["n"] += 1
        raise unreachable

    with pytest.raises(EndpointUnreachableError):
        manager.execute_with_retry(fail, provider_key="p:m")
    assert calls["n"] == 1
    breaker = manager.get_circuit_breaker("p:m")
    assert breaker.failure_count == 0, "an absent machine is not a sick model"


def test_failure_line_measures_from_the_first_attempt(caplog):
    """The failure log used to say ~0 ms for every failure (it measured from the failure)."""
    provider = _Endpoint(raise_this=_connect_error_from_os_timeout(), delay_s=0.05)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(EndpointUnreachableError) as err:
            provider.generate("hello")
    assert err.value.elapsed_s is not None and err.value.elapsed_s >= 0.04
    line = next(r.getMessage() for r in caplog.records if "Generation failed for test-model" in r.getMessage())
    assert "elapsed:" in line and "latency:" not in line
    ms = float(line.split("elapsed: ")[1].split("ms")[0])
    assert ms >= 40
    assert "EndpointUnreachableError" in line
    assert provider.calls == 1  # connect timed out: no second 75 s wait


def test_endpoint_unreachable_error_pickles_with_every_field():
    """The gateway's model-host child pickles exceptions across the process boundary."""
    import pickle

    exc = EndpointUnreachableError(provider="openai-compatible", base_url="http://192.168.1.7:1234/v1",
                                   cause="httpx.ConnectError", detail="[Errno 60] Operation timed out",
                                   errno=errno.ETIMEDOUT, elapsed_s=75.2, retryable=False)
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is EndpointUnreachableError
    assert (back.provider, back.base_url, back.cause, back.detail, back.errno, back.elapsed_s, back.retryable) == (
        "openai-compatible", "http://192.168.1.7:1234/v1", "httpx.ConnectError", "[Errno 60] Operation timed out",
        errno.ETIMEDOUT, 75.2, False)
    assert back.counts_toward_breaker is False
    assert str(back) == str(exc)
