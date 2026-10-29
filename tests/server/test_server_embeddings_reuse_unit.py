"""`/v1/embeddings` builds ONE EmbeddingManager per (provider, model, dimensions, settings) and reuses it
(operator rule 2026-10-29: a request on a loaded model reuses it, never reloads it).

Mutation check: without `_embedding_manager_for` (a manager per request) the first test counts 2
constructions and goes red; without the fingerprinted settings in the key the second test goes red."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class _FakeManager:
    built = []

    def __init__(self, *, model, provider, output_dims=None, strict=True, provider_kwargs=None):
        self.model_id = model
        self.provider = provider
        self.output_dims = output_dims
        self.provider_kwargs = dict(provider_kwargs or {})
        type(self).built.append(self)

    def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "unit-token-0123456789abcdef")
    from abstractcore.embeddings import manager as manager_mod
    from abstractcore.server import app as server_app

    monkeypatch.setattr(manager_mod, "EmbeddingManager", _FakeManager)
    monkeypatch.setattr(server_app, "_SUPPORTED_EMBEDDING_PROVIDERS", {"huggingface"})
    monkeypatch.setattr(server_app, "_DIRECT_EMBEDDING_PROVIDERS", set())
    monkeypatch.setattr(server_app, "_EMBEDDING_MANAGERS", {})
    _FakeManager.built = []
    return TestClient(server_app.app, headers={"Authorization": "Bearer unit-token-0123456789abcdef"})


def _post(client, **body):
    payload = {"model": "huggingface/sentence-transformers/all-MiniLM-L6-v2", "input": ["hello"]}
    payload.update(body)
    r = client.post("/v1/embeddings", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_two_requests_on_the_same_model_build_one_manager(client):
    _post(client)
    _post(client, input=["hello", "again"])
    assert len(_FakeManager.built) == 1, [m.model_id for m in _FakeManager.built]


def test_different_dimensions_are_different_managers(client):
    _post(client)
    _post(client, dimensions=64)
    assert len(_FakeManager.built) == 2
    assert {m.output_dims for m in _FakeManager.built} == {None, 64}
    _post(client, dimensions=64)
    assert len(_FakeManager.built) == 2


def test_the_key_never_carries_provider_settings_in_clear():
    from abstractcore.server.app import _embedding_manager_key

    key = _embedding_manager_key("huggingface", "m", None, {"api_key": "sk-secret-value", "base_url": "http://x"})
    assert "sk-secret-value" not in repr(key) and "http://x" not in repr(key)
    assert key != _embedding_manager_key("huggingface", "m", None, {"api_key": "other"})
