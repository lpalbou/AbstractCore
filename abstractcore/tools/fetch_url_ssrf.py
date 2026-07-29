"""SSRF guard for fetch_url — per-hop destination validation."""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any, Dict, FrozenSet, Optional, Tuple
from urllib.parse import urlparse

from requests.adapters import HTTPAdapter

# Cloud metadata endpoints — deny always, never allowlistable.
_FETCH_URL_METADATA_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
    }
)
_FETCH_URL_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("100.100.100.200"),
        ipaddress.ip_address("fd00:ec2::254"),
    }
)

_ALLOWLIST_CACHE: Optional[FrozenSet[Tuple[str, Optional[int]]]] = None


class FetchUrlSSRFBlocked(Exception):
    """Raised by the guarded session adapter when a destination is refused."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(str(payload.get("error") or "blocked_ssrf"))


def _fetch_url_allowlist_entries() -> FrozenSet[Tuple[str, Optional[int]]]:
    global _ALLOWLIST_CACHE
    if _ALLOWLIST_CACHE is not None:
        return _ALLOWLIST_CACHE
    raw = os.environ.get("ABSTRACTCORE_FETCH_URL_ALLOW", "").strip()
    entries: set[Tuple[str, Optional[int]]] = set()
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" in piece and piece.count(":") >= 1:
            host_part, port_part = piece.rsplit(":", 1)
            try:
                port = int(port_part)
            except ValueError:
                continue
            entries.add((host_part.lower(), port))
        else:
            entries.add((piece.lower(), None))
    _ALLOWLIST_CACHE = frozenset(entries)
    return _ALLOWLIST_CACHE


def reset_fetch_url_allowlist_cache() -> None:
    """Test helper: re-read ABSTRACTCORE_FETCH_URL_ALLOW."""
    global _ALLOWLIST_CACHE
    _ALLOWLIST_CACHE = None


def _fetch_url_ip_is_metadata(ip: ipaddress._BaseAddress) -> bool:
    return ip in _FETCH_URL_METADATA_IPS


def _fetch_url_ip_is_non_public(ip: ipaddress._BaseAddress) -> bool:
    # Deny-by-default: anything not globally routable is blocked unless
    # allowlisted. `is_global is False` catches CGNAT 100.64/10 (e.g. Alibaba
    # metadata at 100.100.100.200) and future non-global allocations.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


def _fetch_url_allowlisted(host_l: str, port: int, ip_str: str) -> bool:
    entries = _fetch_url_allowlist_entries()
    if not entries:
        return False
    candidates = (
        (host_l, port),
        (host_l, None),
        (ip_str.lower(), port),
        (ip_str.lower(), None),
    )
    return any(item in entries for item in candidates)


def _fetch_url_ssrf_block(url: str, reason: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": reason,
        "error_class": "blocked_ssrf",
        "retryable": False,
        "url": str(url),
        "rendered": f"⛔ Blocked (SSRF guard): {reason}\nURL: {url}",
    }


def fetch_url_guard_destination(url: str) -> Optional[Dict[str, Any]]:
    """Return a block payload when url must not be fetched, else None."""
    parsed = urlparse(str(url or ""))
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return _fetch_url_ssrf_block(url, f"Unsupported scheme for fetch: {scheme or '(none)'}")

    host = parsed.hostname
    if not host:
        return _fetch_url_ssrf_block(url, "Missing hostname")

    host_l = host.lower().rstrip(".")
    if host_l in _FETCH_URL_METADATA_HOSTNAMES:
        return _fetch_url_ssrf_block(url, "Metadata hostname blocked")

    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80

    try:
        addrinfos = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        return _fetch_url_ssrf_block(url, f"Could not resolve hostname: {exc}")

    seen: set[str] = set()
    for _family, _socktype, _proto, _canonname, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if ip_str in seen:
            continue
        seen.add(ip_str)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        if _fetch_url_ip_is_metadata(ip):
            return _fetch_url_ssrf_block(url, f"Metadata service IP blocked: {ip_str}")

        if _fetch_url_allowlisted(host_l, port, ip_str):
            continue

        if _fetch_url_ip_is_non_public(ip):
            return _fetch_url_ssrf_block(url, f"Non-public destination blocked: {ip_str}")

    if not seen:
        return _fetch_url_ssrf_block(url, "Hostname did not resolve to any address")

    return None


def fetch_url_strip_sensitive_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not headers:
        return {}
    cleaned: Dict[str, str] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered in {"authorization", "cookie"} or lowered.startswith("proxy-"):
            continue
        cleaned[str(key)] = str(value)
    return cleaned


class SSRFGuardAdapter(HTTPAdapter):
    """Re-validates every outbound hop (redirects re-enter via requests)."""

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        blocked = fetch_url_guard_destination(str(request.url))
        if blocked is not None:
            raise FetchUrlSSRFBlocked(blocked)
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )
