"""Hermetic email fixtures: a throwaway CA, local IMAP / SMTP / OAuth servers, a keychain
guard and a scratch AbstractCore config file.

Nothing here reaches a real mail service or an OS keychain: servers bind 127.0.0.1 on free
ports, every address is under example.test, and `keyring` cannot even be imported during a test
(the `memory_keyring` guard fails the test if anything tries; credentials are sealed with the
key file `<config dir>/secrets/sealing.key`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

import pytest

from abstractcore.testing.mailserver import (
    FakeImapServer,
    FakeOAuthServer,
    FakeSmtpServer,
    TestCA,
    TokenRegistry,
    build_message,
)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from email_fixtures import ME, PASSWORD  # noqa: E402


class KeychainImportGuard:
    """Refuses `keyring` and records who asked: AbstractCore never uses an OS keychain (2.26)."""

    def __init__(self) -> None:
        self.attempts: list = []

    def find_spec(self, name, path=None, target=None):
        if name == "keyring" or name.startswith("keyring."):
            self.attempts.append(name)
            raise ImportError(f"{name}: AbstractCore must never import keyring (tests/email/conftest.py guard)")
        return None


@pytest.fixture(autouse=True)
def memory_keyring() -> Iterator[KeychainImportGuard]:
    """Historic name: there is no keychain at all now. A test during which anything imports
    `keyring` fails."""

    guard = KeychainImportGuard()
    stashed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "keyring" or k.startswith("keyring.")}
    sys.meta_path.insert(0, guard)
    try:
        yield guard
    finally:
        try:
            sys.meta_path.remove(guard)
        except ValueError:
            pass
        sys.modules.update(stashed)
    assert not guard.attempts, f"keyring was imported during the test: {guard.attempts}"


@pytest.fixture(autouse=True)
def no_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("ABSTRACT_EMAIL_") or name in {"EMAIL_PASSWORD", "DEFAULT_EMAIL_PASSWORD"}:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_external_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """OAuth HTTP may only reach the local fake: a request to any other host fails the test
    (the Google / Microsoft presets carry real endpoints; they are never contacted)."""

    import urllib.parse

    import httpx

    real_send = httpx.Client.send

    def guarded_send(self, request, *args, **kwargs):
        host = urllib.parse.urlsplit(str(request.url)).hostname or ""
        if host not in {"localhost", "127.0.0.1", "testserver"}:  # testserver: FastAPI TestClient, in process
            raise AssertionError(f"a test tried to reach {host!r}; email tests stay on localhost")
        return real_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", guarded_send)


@pytest.fixture(autouse=True)
def no_discovery_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Mail server discovery never reaches the network in tests: its default lookups refuse
    and are recorded, and any recorded attempt fails the test at teardown (discovery turns a
    step's exception into a `tried` row, so raising alone would pass silently). A test injects
    its own lookups (`http_get=`, `resolve_srv=`, `resolve_mx=`, or `discovery._net`)."""

    from abstractcore.comms.email import discovery

    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise OSError("network refused in tests")

    monkeypatch.setattr(discovery._net, "http_get", refuse)
    monkeypatch.setattr(discovery._net, "resolve_srv", refuse)
    monkeypatch.setattr(discovery._net, "resolve_mx", refuse)
    yield
    assert not attempts, f"mail server discovery tried the network in a test: {attempts!r}"


@pytest.fixture(autouse=True)
def reset_tool_resolver() -> Iterator[None]:
    from abstractcore.tools import comms_tools

    comms_tools.set_email_account_resolver(None)
    yield
    comms_tools.set_email_account_resolver(None)


@pytest.fixture(scope="session")
def ca(tmp_path_factory) -> TestCA:
    return TestCA.create(tmp_path_factory.mktemp("email-ca"))


@pytest.fixture
def tokens() -> TokenRegistry:
    return TokenRegistry()


@pytest.fixture
def imap(ca: TestCA, tokens: TokenRegistry) -> Iterator[FakeImapServer]:
    server = FakeImapServer(ca, users={ME: PASSWORD}, security="ssl", tokens=tokens)
    yield server
    server.close()


@pytest.fixture
def imap_starttls(ca: TestCA, tokens: TokenRegistry) -> Iterator[FakeImapServer]:
    server = FakeImapServer(ca, users={ME: PASSWORD}, security="starttls", tokens=tokens)
    yield server
    server.close()


@pytest.fixture
def smtp(ca: TestCA, tokens: TokenRegistry) -> Iterator[FakeSmtpServer]:
    server = FakeSmtpServer(ca, users={ME: PASSWORD}, security="starttls", tokens=tokens, refuse={"blocked@example.test": 550, "full@example.test": 552})
    yield server
    server.close()


@pytest.fixture
def smtps(ca: TestCA, tokens: TokenRegistry) -> Iterator[FakeSmtpServer]:
    server = FakeSmtpServer(ca, users={ME: PASSWORD}, security="ssl", tokens=tokens)
    yield server
    server.close()


@pytest.fixture
def oauth_server(ca: TestCA, tokens: TokenRegistry) -> Iterator[FakeOAuthServer]:
    server = FakeOAuthServer(ca, tokens, user=ME)
    yield server
    server.close()


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch AbstractCore config file, also the default the tools and CLI resolve."""

    path = tmp_path / "config" / "abstractcore.json"
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_FILE", str(path))
    return path
