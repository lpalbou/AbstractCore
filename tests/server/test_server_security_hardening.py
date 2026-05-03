from __future__ import annotations

import importlib
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse


class _StubLLM:
    def __init__(self) -> None:
        self.last_generate_kwargs: Dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> GenerateResponse:
        self.last_generate_kwargs = dict(kwargs)
        return GenerateResponse(
            content="ok",
            model="stub",
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )


def test_server_blocks_non_loopback_base_url_by_default(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called for blocked base_url overrides")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "base_url": "https://example.com/v1",
        },
    )
    assert r.status_code == 403


def test_server_allows_loopback_base_url_override(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    created: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        created.append(dict(kwargs))
        return _StubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "base_url": "http://localhost:1234/v1",
        },
    )
    assert r.status_code == 200
    assert created and created[-1].get("base_url") == "http://localhost:1234/v1"


def test_server_prevents_env_key_exfiltration_via_base_url_override(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    # Allow this non-loopback base_url for the test.
    monkeypatch.setenv("ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST", "https://example.com")
    # Simulate a server operator key configured in env.
    monkeypatch.setenv("OPENAI_API_KEY", "TEST_SECRET_DO_NOT_USE")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called when api_key is required but missing")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "base_url": "https://example.com/v1",
        },
    )
    assert r.status_code == 403


def test_url_allowlist_url_entries_reject_host_confusion() -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    allowlist = ["https://example.com"]
    assert server_app._allowlist_matches_url("https://example.com/v1", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com.evil/v1", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com@evil.test/v1", allowlist)


def test_url_allowlist_url_entries_enforce_port_and_path_boundaries() -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    allowlist = ["https://example.com:8443/v1"]
    assert server_app._allowlist_matches_url("https://example.com:8443/v1", allowlist)
    assert server_app._allowlist_matches_url("https://example.com:8443/v1/models", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com:8444/v1", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com:8443/v10", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com:8443/v1evil", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com:8443/v1/../admin", allowlist)
    assert not server_app._allowlist_matches_url(
        "https://example.com:8443/v1/%252e%252e/admin",
        allowlist,
    )


def test_url_allowlist_url_entries_use_default_ports() -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    allowlist = ["https://example.com"]
    assert server_app._allowlist_matches_url("https://example.com:443/v1", allowlist)
    assert not server_app._allowlist_matches_url("https://example.com:444/v1", allowlist)


def test_url_allowlist_host_globs_still_match_canonical_hosts() -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    allowlist = ["*.example.com"]
    assert server_app._allowlist_matches_url("https://api.example.com/v1", allowlist)
    assert not server_app._allowlist_matches_url("https://api.example.com.evil/v1", allowlist)


def test_server_blocks_local_file_paths_in_http_requests_by_default(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called when local paths are rejected")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": "README.md"}},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 403


def test_server_blocks_private_url_fetches_by_default(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called when SSRF is blocked")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": "http://127.0.0.1/secret.png"}},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 403
