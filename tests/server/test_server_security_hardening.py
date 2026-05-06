from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse


@pytest.fixture(autouse=True)
def _clear_security_sensitive_env(monkeypatch) -> None:
    for name in (
        "ABSTRACTCORE_SERVER_API_KEY",
        "ABSTRACTCORE_SERVER_PROTECT_DOCS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "PORTKEY_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_server_api_key_blocks_unauthenticated_requests_before_provider_creation(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_API_KEY", "server-secret")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called before server auth succeeds")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)

    health = client.get("/health")
    assert health.status_code == 200

    r = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401

    models = client.get("/v1/models")
    assert models.status_code == 401


def test_swagger_docs_are_usable_with_server_auth_enabled(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_API_KEY", "server-secret")
    server_app.app.openapi_schema = None

    client = TestClient(server_app.app)

    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "SwaggerUIBundle" in docs.text
    assert "responseInterceptor" in docs.text
    assert "__abstractcoreLatestAudioPreviewUrl" in docs.text

    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    body = schema.json()
    bearer = body["components"]["securitySchemes"]["AbstractCoreBearerAuth"]
    assert bearer["type"] == "http"
    assert bearer["scheme"] == "bearer"
    assert body["paths"]["/health"]["get"]["security"] == []
    assert body["paths"]["/v1/chat/completions"]["post"]["security"] == [{"AbstractCoreBearerAuth": []}]

    protected = client.get("/v1/models")
    assert protected.status_code == 401


def test_docs_can_be_protected_for_locked_down_deployments(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_API_KEY", "server-secret")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_PROTECT_DOCS", "1")

    client = TestClient(server_app.app)

    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/openapi.json", headers={"Authorization": "Bearer server-secret"}).status_code == 200


def test_server_auth_is_required_by_default_when_key_is_not_configured(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called when server auth is not configured")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    assert client.get("/health").status_code == 200

    r = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503


def test_server_api_key_allows_authenticated_request_without_forwarding_server_key(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_API_KEY", "server-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-openai-key")

    created: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        created.append(dict(kwargs))
        return _StubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer server-secret"},
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert created
    assert "api_key" not in created[-1]


def test_server_api_key_allows_provider_override_header(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_API_KEY", "server-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server-key")

    created: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        created.append(dict(kwargs))
        return _StubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer server-secret",
            "X-AbstractCore-Provider-API-Key": "sk-ant-request-key",
        },
        json={"model": "anthropic/claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert created and created[-1].get("api_key") == "sk-ant-request-key"


def test_provider_api_key_body_field_is_disabled(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called when body api_key is rejected")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "api_key": "sk-should-not-be-used",
        },
    )
    assert r.status_code == 400


def test_provider_api_key_query_param_is_disabled(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    def fake_get_models_from_provider(*args: Any, **kwargs: Any) -> List[str]:
        raise AssertionError("model discovery should not run when query api_key is rejected")

    monkeypatch.setattr(server_app, "get_models_from_provider", fake_get_models_from_provider)

    client = TestClient(server_app.app)
    r = client.get("/v1/models?provider=openai&api_key=sk-should-not-be-logged")
    assert r.status_code == 400


def test_provider_api_key_authorization_header_is_forwarded_only_without_server_auth(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    created: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        created.append(dict(kwargs))
        return _StubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-provider-request-key"},
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert created and created[-1].get("api_key") == "sk-provider-request-key"


def test_provider_api_key_authorization_overrides_server_env_without_server_auth(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-env-key")

    created: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        created.append(dict(kwargs))
        return _StubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-provider-request-key"},
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert created and created[-1].get("api_key") == "sk-provider-request-key"


def test_provider_override_header_is_forwarded_without_server_auth(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    created: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        created.append(dict(kwargs))
        return _StubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-request-key"},
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert created and created[-1].get("api_key") == "sk-provider-request-key"


def test_unauthenticated_request_cannot_use_server_provider_env_key(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-server-env-key")

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        raise AssertionError("create_llm should not be called with unauthenticated server-held provider keys")

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_sensitive_values_are_redacted_from_structured_log_fields() -> None:
    server_app = importlib.import_module("abstractcore.server.app")

    assert server_app._redact_headers(
        {
            "Authorization": "Bearer server-secret",
            "Cookie": "sid=abc",
            "User-Agent": "test",
            "X-AbstractCore-Provider-API-Key": "sk-provider-key",
        }
    ) == {
        "Authorization": "[REDACTED]",
        "Cookie": "[REDACTED]",
        "User-Agent": "test",
        "X-AbstractCore-Provider-API-Key": "[REDACTED]",
    }

    assert (
        server_app._redact_url("http://localhost/v1/models?provider=openai&api_key=sk-secret&x=1")
        == "http://localhost/v1/models?provider=openai&api_key=%5BREDACTED%5D&x=1"
    )

    assert server_app._redact_sensitive_data({"api_key": "sk-secret", "nested": {"token": "abc"}}) == {
        "api_key": "[REDACTED]",
        "nested": {"token": "[REDACTED]"},
    }


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
    assert r.status_code == 401


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
