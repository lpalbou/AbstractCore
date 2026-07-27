"""fetch_url 401 guidance must not invite a credential paste at local hosts.

code-tui 401-incident root-cause (commons c4978): the generic 'supply
credentials via headers' suggestion is a prompt-injection footgun for
loopback/control-plane targets — a model holding a token in context (e.g. the
gateway bearer) would paste it into a fetch_url aimed at the loopback API.
For a local/private host the 401 guidance must give safe advice instead.
This is the softer sub-fix that needs no operator ruling (the broader
same-origin GUARD on fetch_url is the operator's call against the standing
base64-exfil 'ONE protection' ruling — deliberately NOT changed here).
"""

from __future__ import annotations

import pytest

from abstractcore.tools.common_tools import (
    _classify_fetch_http_error,
    _is_loopback_or_private_host,
)


@pytest.mark.parametrize(
    "host,expected",
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("127.0.0.1", True),
        ("127.5.5.5", True),  # whole 127.0.0.0/8 loopback block
        ("::1", True),
        ("192.168.1.10", True),
        ("10.0.0.5", True),
        ("172.16.0.1", True),  # 172.16/12 private
        ("169.254.10.1", True),  # link-local
        ("gateway.local", True),  # mDNS
        ("api.internal.localhost", True),
        ("example.com", False),  # public hostname
        ("8.8.8.8", False),  # public IP
        ("172.32.0.1", False),  # just outside 172.16/12
        ("", False),
        (None, False),
    ],
)
def test_loopback_or_private_host_detection(host, expected):
    assert _is_loopback_or_private_host(host) is expected


def test_401_public_host_keeps_credentials_suggestion():
    cls, suggestions = _classify_fetch_http_error(401, {}, host="example.com")
    assert cls == "auth_required"
    assert any("supply credentials" in s for s in suggestions)


def test_401_loopback_host_never_invites_credential_paste():
    cls, suggestions = _classify_fetch_http_error(401, {}, host="127.0.0.1")
    assert cls == "auth_required"
    # The footgun hint must be ABSENT for a local target...
    assert not any("supply credentials" in s for s in suggestions)
    # ...replaced by safe guidance that explicitly warns against pasting tokens.
    assert any("do NOT paste" in s for s in suggestions)
    assert any("control plane" in s.lower() or "session" in s.lower() for s in suggestions)


def test_401_private_host_also_safe():
    _, suggestions = _classify_fetch_http_error(401, {}, host="192.168.0.5")
    assert not any("supply credentials" in s for s in suggestions)


def test_401_no_host_keeps_backward_compatible_default():
    # host omitted (older callers) → the original generic guidance (never a
    # silent behavior change for callers that don't pass a host).
    _, suggestions = _classify_fetch_http_error(401, {})
    assert any("supply credentials" in s for s in suggestions)
