"""Hermetic email fixtures: a throwaway CA, local IMAP / SMTP / OAuth servers, an in-memory
keychain and a scratch AbstractCore config file.

Nothing here reaches a real mail service or the OS keychain: servers bind 127.0.0.1 on free
ports, every address is under example.test, and `keyring` is pointed at an in-memory backend
for every test (the macOS Keychain does not depend on HOME, so moving HOME is not enough).
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


class MemoryKeyring:
    """A keyring backend kept in memory (priority 1 so the vault accepts it)."""

    priority = 1

    def __init__(self) -> None:
        self.items: Dict[tuple, str] = {}

    def get_password(self, service: str, username: str):
        return self.items.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.items[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.items.pop((service, username), None)


@pytest.fixture(autouse=True)
def memory_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryKeyring]:
    import keyring
    from keyring.backend import KeyringBackend

    class _Backend(KeyringBackend):
        priority = 1

        def __init__(self, store: MemoryKeyring) -> None:
            super().__init__()
            self._store = store

        def get_password(self, service, username):
            return self._store.get_password(service, username)

        def set_password(self, service, username, password):
            self._store.set_password(service, username, password)

        def delete_password(self, service, username):
            self._store.delete_password(service, username)

    mem = MemoryKeyring()
    previous = keyring.get_keyring()
    keyring.set_keyring(_Backend(mem))
    try:
        yield mem
    finally:
        keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def no_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("ABSTRACT_EMAIL_") or name in {"EMAIL_PASSWORD", "DEFAULT_EMAIL_PASSWORD"}:
            monkeypatch.delenv(name, raising=False)


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
