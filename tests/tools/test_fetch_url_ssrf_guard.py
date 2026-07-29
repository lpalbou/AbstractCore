"""SSRF guard for fetch_url — destination validation and header hygiene."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from abstractcore.tools.fetch_url_ssrf import (
    FetchUrlSSRFBlocked,
    SSRFGuardAdapter,
    fetch_url_guard_destination,
    fetch_url_strip_sensitive_headers,
    reset_fetch_url_allowlist_cache,
)


def _mock_getaddrinfo(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]


@pytest.fixture(autouse=True)
def _clear_allowlist(monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_FETCH_URL_ALLOW", raising=False)
    reset_fetch_url_allowlist_cache()
    yield
    reset_fetch_url_allowlist_cache()


class TestFetchUrlGuardDestination:
    def test_blocks_aws_metadata_ip(self):
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("169.254.169.254")):
            out = fetch_url_guard_destination("http://169.254.169.254/latest/meta-data/")
        assert out is not None
        assert out["error_class"] == "blocked_ssrf"
        assert "Metadata" in out["error"]

    def test_blocks_alibaba_cgnat_metadata_ip(self):
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("100.100.100.200")):
            out = fetch_url_guard_destination("http://100.100.100.200/latest/meta-data/")
        assert out is not None
        assert out["error_class"] == "blocked_ssrf"
        assert "100.100.100.200" in out["error"]

    def test_blocks_loopback_without_allowlist(self):
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("127.0.0.1")):
            out = fetch_url_guard_destination("http://127.0.0.1:8080/admin")
        assert out is not None
        assert "Non-public" in out["error"]

    def test_blocks_metadata_hostname(self):
        out = fetch_url_guard_destination("http://metadata.google.internal/computeMetadata/v1/")
        assert out is not None
        assert "Metadata hostname" in out["error"]

    def test_metadata_ip_blocked_even_when_allowlisted(self, monkeypatch):
        monkeypatch.setenv("ABSTRACTCORE_FETCH_URL_ALLOW", "169.254.169.254:80")
        reset_fetch_url_allowlist_cache()
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("169.254.169.254")):
            out = fetch_url_guard_destination("http://169.254.169.254/latest/meta-data/")
        assert out is not None
        assert "Metadata" in out["error"]

    def test_allowlist_unblocks_localhost_port(self, monkeypatch):
        monkeypatch.setenv("ABSTRACTCORE_FETCH_URL_ALLOW", "localhost:3000")
        reset_fetch_url_allowlist_cache()
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("127.0.0.1")):
            out = fetch_url_guard_destination("http://localhost:3000/health")
        assert out is None

    def test_allowlist_is_port_specific(self, monkeypatch):
        monkeypatch.setenv("ABSTRACTCORE_FETCH_URL_ALLOW", "127.0.0.1:3000")
        reset_fetch_url_allowlist_cache()
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("127.0.0.1")):
            blocked = fetch_url_guard_destination("http://127.0.0.1:8080/admin")
            allowed = fetch_url_guard_destination("http://127.0.0.1:3000/health")
        assert blocked is not None
        assert allowed is None

    def test_blocks_unsupported_scheme(self):
        out = fetch_url_guard_destination("file:///etc/passwd")
        assert out is not None
        assert "Unsupported scheme" in out["error"]

    def test_allows_public_ip(self):
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("93.184.216.34")):
            out = fetch_url_guard_destination("http://example.com/")
        assert out is None


class TestHeaderStrip:
    def test_strips_auth_cookie_and_proxy_headers(self):
        cleaned = fetch_url_strip_sensitive_headers(
            {
                "Authorization": "Bearer secret",
                "Cookie": "session=abc",
                "Proxy-Authorization": "Basic x",
                "Accept": "text/html",
            }
        )
        assert cleaned == {"Accept": "text/html"}


class TestSSRFGuardAdapter:
    def test_adapter_raises_on_blocked_destination(self):
        adapter = SSRFGuardAdapter()
        request = type("Req", (), {"url": "http://127.0.0.1:8080/"})()
        with patch("socket.getaddrinfo", return_value=_mock_getaddrinfo("127.0.0.1")):
            with pytest.raises(FetchUrlSSRFBlocked) as exc:
                adapter.send(request)
        assert exc.value.payload["error_class"] == "blocked_ssrf"
