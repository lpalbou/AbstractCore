from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from abstractcore.exceptions import ProviderAPIError
from abstractcore.server.app import app


@pytest.fixture(autouse=True)
def _clear_server_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PORTKEY_API_KEY", raising=False)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _provider_auth_headers() -> Dict[str, str]:
    return {"X-AbstractCore-Provider-API-Key": "sk-request-provider-key"}


def test_openai_embeddings_route_forwards_remote_parameters(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.providers.openai_provider import OpenAIProvider

    captured: Dict[str, Any] = {}

    def fake_init(self, model: str, **kwargs: Any) -> None:
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_embed(self, input_text, **kwargs: Any):
        captured["embed"] = {"input_text": input_text, **kwargs}
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": self.model,
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

    monkeypatch.setattr(OpenAIProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAIProvider, "embed", fake_embed)

    resp = client.post(
        "/v1/embeddings",
        headers=_provider_auth_headers(),
        json={
            "model": "openai/text-embedding-3-small",
            "input": "hello world",
            "dimensions": 2,
            "encoding_format": "float",
            "user": "user-123",
        },
    )

    assert resp.status_code == 200
    assert captured["init"] == {"model": "text-embedding-3-small", "api_key": "sk-request-provider-key"}
    assert captured["embed"] == {
        "input_text": "hello world",
        "encoding_format": "float",
        "dimensions": 2,
        "user": "user-123",
    }
    assert resp.json()["model"] == "openai/text-embedding-3-small"


def test_openai_compatible_embeddings_accept_loopback_base_url(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    captured: Dict[str, Any] = {}

    def fake_init(self, model: str, **kwargs: Any) -> None:
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_embed(self, input_text, **kwargs: Any):
        captured["embed"] = {"input_text": input_text, **kwargs}
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.3], "index": 0}],
            "model": self.model,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatibleProvider, "embed", fake_embed)

    resp = client.post(
        "/v1/embeddings",
        headers=_provider_auth_headers(),
        json={
            "model": "openai-compatible/local-embedding-model",
            "input": ["alpha", "beta"],
            "base_url": "http://127.0.0.1:1234/v1",
        },
    )

    assert resp.status_code == 200
    assert captured["init"] == {
        "model": "local-embedding-model",
        "api_key": "sk-request-provider-key",
        "base_url": "http://127.0.0.1:1234/v1",
    }
    assert captured["embed"]["input_text"] == ["alpha", "beta"]


def test_anthropic_embeddings_are_rejected_with_clear_error(client: TestClient) -> None:
    resp = client.post(
        "/v1/embeddings",
        headers=_provider_auth_headers(),
        json={"model": "anthropic/claude-sonnet-4", "input": "hello"},
    )

    assert resp.status_code == 400
    assert "Anthropic" in resp.json()["error"]["message"]


def test_remote_embedding_failure_is_not_silently_zero_vector(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.providers.openai_provider import OpenAIProvider

    def fake_init(self, model: str, **kwargs: Any) -> None:
        self.model = model

    def fake_embed(self, input_text, **kwargs: Any):
        raise ProviderAPIError("upstream failed")

    monkeypatch.setattr(OpenAIProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAIProvider, "embed", fake_embed)

    resp = client.post(
        "/v1/embeddings",
        headers=_provider_auth_headers(),
        json={"model": "openai/text-embedding-3-small", "input": "hello"},
    )

    assert resp.status_code == 502
    body = resp.json()
    assert "data" not in body
    assert "upstream failed" in body["error"]["message"]


def test_local_embedding_manager_runs_strict_in_server_mode(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import abstractcore.embeddings.manager as manager_module

    captured: Dict[str, Any] = {}

    class FakeEmbeddingManager:
        def __init__(self, **kwargs: Any) -> None:
            captured["init"] = kwargs
            self.model_name = kwargs["model"]

        def embed_batch(self, inputs):
            captured["inputs"] = inputs
            return [[0.4, 0.5] for _ in inputs]

    monkeypatch.setattr(manager_module, "EmbeddingManager", FakeEmbeddingManager)

    resp = client.post(
        "/v1/embeddings",
        headers=_provider_auth_headers(),
        json={"model": "ollama/nomic-embed-text", "input": ["a", "b"], "dimensions": 2},
    )

    assert resp.status_code == 200
    assert captured["init"]["strict"] is True
    assert captured["init"]["output_dims"] == 2
    assert captured["inputs"] == ["a", "b"]
