"""An unknown provider (or a model the provider reports missing) is the caller's
404 `model_not_found` in the OpenAI error envelope, never a 500."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from abstractcore.exceptions import ModelNotFoundError, UnknownProviderError


@pytest.fixture
def server(monkeypatch):
    for name in ("ABSTRACTCORE_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    return importlib.import_module("abstractcore.server.app")


def test_unknown_provider_is_a_typed_value_error():
    from abstractcore import create_llm

    with pytest.raises(UnknownProviderError) as err:
        create_llm("nosuch-provider", model="m")
    assert isinstance(err.value, ValueError) and isinstance(err.value, ModelNotFoundError)


def test_chat_with_unknown_provider_answers_404_model_not_found(server):
    resp = TestClient(server.app).post("/v1/chat/completions", json={
        "model": "nosuch-provider/some-model", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 404, resp.text
    err = resp.json()["error"]
    assert err["code"] == "model_not_found" and err["param"] == "model" and err["type"] == "invalid_request_error"


def test_chat_with_a_model_the_provider_reports_missing_answers_404(server, monkeypatch):
    def missing(*a, **kw):
        raise ModelNotFoundError("Model 'ghost' not found")
    monkeypatch.setattr(server, "create_llm", missing)
    resp = TestClient(server.app).post("/v1/chat/completions", json={
        "model": "ollama/ghost", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "model_not_found"
