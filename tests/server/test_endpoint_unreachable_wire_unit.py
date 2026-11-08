"""EndpointUnreachableError crosses the HTTP server with every field (503, code endpoint_unreachable)."""

from __future__ import annotations

import errno

from abstractcore.exceptions import EndpointUnreachableError
from abstractcore.server.app import _endpoint_unreachable_payload, _provider_exception_status


def test_payload_carries_every_field_and_a_503():
    exc = EndpointUnreachableError(
        provider="openai-compatible", base_url="http://192.168.1.7:1234/v1", cause="httpx.ConnectError",
        detail="[Errno 60] Operation timed out", errno=errno.ETIMEDOUT, elapsed_s=75.2, retryable=False,
    )
    assert _provider_exception_status(exc) == 503
    body = _endpoint_unreachable_payload(exc)["error"]
    assert body["type"] == "endpoint_unreachable" and body["code"] == "endpoint_unreachable"
    assert body["provider"] == "openai-compatible"
    assert body["base_url"] == "http://192.168.1.7:1234/v1"
    assert body["cause"] == "httpx.ConnectError"
    assert body["errno"] == errno.ETIMEDOUT
    assert body["elapsed_s"] == 75.2
    assert body["retryable"] is False
    assert "could not connect to http://192.168.1.7:1234/v1" in body["message"]
    assert "timed out after" not in body["message"]
