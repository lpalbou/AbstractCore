"""POST /v1/systemone: request validation, model handle parsing, the System One
response shape and error mapping, with the decision runtime stubbed."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from abstractcore.decision.runtime import DecisionModel
from tests.decision.test_decision_core import FakeBackend, FakeTokenizer


class _StubProvider:
    model_type = "decision"

    def __init__(self, provider: str, model: str) -> None:
        self.provider, self.model = provider, model
        self._dm = DecisionModel(FakeBackend(), FakeTokenizer(), model_id=model, provider=provider)

    def decision_model(self) -> DecisionModel:
        return self._dm

    def unload_model(self, model_name: str) -> None:
        return None


@pytest.fixture
def server(monkeypatch):
    for name in ("ABSTRACTCORE_AUTH_TOKEN", "ABSTRACTCORE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    server_app = importlib.import_module("abstractcore.server.app")
    created = []

    def fake_create_llm(provider: str, model: str = None, **kwargs: Any) -> _StubProvider:
        created.append((provider, model))
        return _StubProvider(provider, model)

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)
    monkeypatch.setattr(server_app, "_resident_provider_for_request", lambda *a, **k: None)
    pool = importlib.import_module("abstractcore.decision.pool")
    pool._reset_for_tests()
    loader = importlib.import_module("abstractcore.decision.loader")
    monkeypatch.setattr(loader, "is_decision_model", lambda model, provider=None: "kev" in str(model).lower())
    monkeypatch.setattr(server_app, "_configured_decision_route", lambda: None)
    return server_app, TestClient(server_app.app), created


def _request(model="mlx/fake/kev-4b"):
    return {
        "model": model,
        "state": "Shoes arrived two weeks late and in the wrong size.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": None, "shipping and delivery": None}},
            "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
            "frustration": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Very very angry"]},
        },
    }


def test_systemone_returns_typed_answers(server):
    server_app, client, created = server
    r = client.post("/v1/decisions", json=_request())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "mlx/fake/kev-4b"
    assert body["answers"]["department"]["type"] == "choice"
    assert body["answers"]["department"]["choice"] == "shipping and delivery"
    assert set(body["answers"]["department"]) == {"type", "choice", "confidence", "probabilities"}
    assert body["answers"]["escalate"] == {"type": "noul", "noul": 0.5}
    assert body["answers"]["frustration"]["legend"] == {"0": "Calm", "1": "Very very angry"}
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert "latency_ms" in body
    assert r.headers["x-typesafe-request-id"].startswith("s1-")
    assert created == [("mlx", "fake/kev-4b")]
    # the alias reuses the loaded model; nothing new is created
    r2 = client.post("/v1/decisions", json=_request(model="kev-latest"))
    assert r2.status_code == 200 and created == [("mlx", "fake/kev-4b")]
    # provider-scoped twin: the body model is unprefixed
    r3 = client.post("/mlx/v1/decisions", json=_request(model="fake/kev-4b"))
    assert r3.status_code == 200 and r3.json()["model"] == "mlx/fake/kev-4b" and created == [("mlx", "fake/kev-4b")]


def test_systemone_parses_handles(server):
    server_app, client, created = server
    from abstractcore.decision.pool import parse_handle

    assert parse_handle("huggingface/ggml-org/Kev-4B-GGUF:Q4_K_M") == ("huggingface", "ggml-org/Kev-4B-GGUF:Q4_K_M")
    assert parse_handle("hf/jaredpalmer/kev-4b") == ("huggingface", "jaredpalmer/kev-4b")
    assert parse_handle("ggml-org/Kev-4B-GGUF:Q4_K_M") == ("huggingface", "ggml-org/Kev-4B-GGUF:Q4_K_M")
    assert parse_handle("typesafe/jev-latest") == ("typesafe", "jev-latest")
    assert parse_handle("systemone/typesafe-jev-1.13.0") == ("systemone", "typesafe-jev-1.13.0")
    provider, model = parse_handle("jaredpalmer/kev-4b")
    assert provider in {"mlx", "huggingface"} and model == "jaredpalmer/kev-4b"


def test_systemone_rejects_bad_requests(server):
    server_app, client, created = server
    r = client.post("/v1/decisions", json={"model": "kev-latest", "state": "x", "questions": {"q": {"type": "noul"}}})
    assert r.status_code == 400 and "provider/model" in r.text
    r = client.post("/v1/decisions", json={"model": "mlx/fake/kev-4b", "state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}})
    assert r.status_code == 400
    r = client.post("/v1/decisions", json={"model": "mlx/fake/kev-4b", "state": "x", "questions": {}})
    assert r.status_code in (400, 422)


def test_models_listing_annotates_capabilities_and_decision_endpoint():
    server_app = importlib.import_module("abstractcore.server.app")
    kev = server_app._model_capability_annotations("jaredpalmer/kev-4b")
    assert kev["capabilities"] == {"input": ["text"], "output": ["decision"]}
    assert kev["decision"]["endpoint"] == "/v1/decisions" and kev["decision"]["kind"] == "kev" and kev["decision"]["generates_text"] is False
    text = server_app._model_capability_annotations("gpt-4")
    assert text["capabilities"]["output"] == ["text"] and "decision" not in text
    assert "decision" not in server_app._model_capability_annotations("unknown-model-xyz")


def test_chat_completions_on_a_decision_model_is_a_400_pointing_at_systemone(server):
    server_app, client, created = server
    from abstractcore.exceptions import DecisionModelError

    class _Refusing:
        model_type = "decision"

        def __init__(self, provider, model):
            self.provider, self.model = provider, model

        def generate(self, **kwargs):
            raise DecisionModelError(self.model, self.provider)

        def unload_model(self, model_name):
            return None

    server_app.create_llm = lambda provider, model=None, **kw: _Refusing(provider, model)
    r = client.post("/v1/chat/completions", json={"model": "huggingface/ggml-org/Kev-4B-GGUF:Q4_K_M", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400, r.text
    body = r.json()["error"]
    assert body["code"] == "decision_model" and body["endpoint"] == "/v1/decisions" and "/v1/decisions" in body["message"]


def test_alias_falls_back_to_the_configured_output_decision_default(server, monkeypatch):
    server_app, client, created = server
    monkeypatch.setattr(server_app, "_configured_decision_route", lambda: ("mlx", "fake/kev-0.8b"))
    r = client.post("/v1/decisions", json=_request(model="kev-latest"))
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "mlx/fake/kev-0.8b"
    assert created == [("mlx", "fake/kev-0.8b")]


def test_shorthand_questions_are_accepted_on_the_wire(server):
    server_app, client, created = server
    body = {"model": "mlx/fake/kev-4b", "state": "Shoes arrived late.", "questions": {"urgent": "Is this urgent?", "team": ["returns", "shipping"]}}
    r = client.post("/v1/decisions", json=body)
    assert r.status_code == 200, r.text
    answers = r.json()["answers"]
    assert answers["urgent"]["type"] == "noul" and answers["team"]["type"] == "choice"


def test_decision_residency_load_list_unload_on_the_shared_pool(server):
    server_app, client, created = server
    r = client.post("/acore/models/load", json={"task": "decision", "model": "mlx/fake/kev-0.8b"})
    assert r.status_code == 200, r.text
    assert r.json()["loaded_new"] is True and r.json()["runtime"]["runtime_id"] == "process:decision:mlx:fake/kev-0.8b"
    assert created == [("mlx", "fake/kev-0.8b")]
    # loading again is a no-op; a decision request reuses the same resident provider
    assert client.post("/acore/models/load", json={"task": "decision", "provider": "mlx", "model": "fake/kev-0.8b"}).json()["loaded_new"] is False
    assert client.post("/v1/decisions", json=_request(model="kev-latest")).status_code == 200
    assert created == [("mlx", "fake/kev-0.8b")]
    rows = client.get("/acore/models/loaded", params={"task": "decision"}).json()["data"]
    assert [row["runtime_id"] for row in rows] == ["process:decision:mlx:fake/kev-0.8b"]
    r = client.post("/acore/models/unload", json={"runtime_id": "process:decision:mlx:fake/kev-0.8b"})
    assert r.status_code == 200 and r.json()["unloaded"] is True
    assert client.get("/acore/models/loaded", params={"task": "decision"}).json()["data"] == []
    assert client.post("/acore/models/unload", json={"task": "decision", "model": "mlx/fake/kev-0.8b"}).status_code == 404


def test_decision_residency_refuses_a_text_model_before_loading(server):
    server_app, client, created = server
    r = client.post("/acore/models/load", json={"task": "decision", "model": "mlx/mlx-community/Qwen3.5-9B-4bit"})
    assert r.status_code == 400, r.text
    assert "is not a decision model" in r.json()["error"]["message"]
    assert created == []  # nothing was built, nothing is resident
    assert client.get("/acore/models/loaded", params={"task": "decision"}).json()["data"] == []


def test_model_not_found_and_context_overflow_map_to_404_and_422(server, monkeypatch):
    server_app, client, created = server
    from abstractcore.decision import ContextOverflow
    from abstractcore.exceptions import ModelNotFoundError

    def not_downloaded(provider, model):
        raise ModelNotFoundError(f"{model} is not downloaded")

    monkeypatch.setattr(server_app, "_decision_factory", not_downloaded)
    r = client.post("/v1/decisions", json=_request(model="mlx/fake/kev-9b"))
    assert r.status_code == 404 and "is not downloaded" in r.json()["error"]["message"]

    monkeypatch.setattr(server_app, "_decision_factory", lambda p, m: _StubProvider(p, m))
    monkeypatch.setattr(DecisionModel, "decide", lambda self, **kw: (_ for _ in ()).throw(ContextOverflow("too long")))
    r = client.post("/v1/decisions", json=_request(model="mlx/fake/kev-4b"))
    assert r.status_code == 422 and "too long" in r.json()["error"]["message"]


def test_unload_while_a_request_uses_the_model_is_deferred(server):
    server_app, client, created = server
    from abstractcore.decision import pool

    client.post("/acore/models/load", json={"task": "decision", "model": "mlx/fake/kev-4b"})
    with pool.use("mlx", "fake/kev-4b"):
        r = client.post("/acore/models/unload", json={"task": "decision", "model": "mlx/fake/kev-4b"})
        assert r.status_code == 200 and r.json()["unloaded"] is False and r.json()["runtime"]["state"] == "unloading"
    assert pool.resident() == []
