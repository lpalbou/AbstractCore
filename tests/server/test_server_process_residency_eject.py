"""The AbstractCore server's unload / listing reach EVERY in-process holder
(mission M2, 2026-09-25).

Before: `POST /acore/models/unload` and `unload_after` freed only the
server's own registry instance, and `/acore/models/loaded` listed only the
registry. MLX weights shared with another instance, a HuggingFace copy, or an
embedder stayed resident, unlisted and unreachable. Now both go through
`abstractcore.providers.process_residency` (faked here: no MLX, no torch).
MUTANTS that turn these RED: drop the `_process_eject_after_unload` call in the
route / in `unload_after`; drop `_merge_process_residency` from the listing.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import abstractcore.providers.process_residency as pr
from tests.server.test_server_memory_and_residency_visibility import _StubCacheProvider


@pytest.fixture
def server(monkeypatch):
    app = importlib.import_module("abstractcore.server.app")
    app._GATEWAY_LOADED_RUNTIMES.clear()
    app._GATEWAY_RUNTIME_IDS.clear()
    monkeypatch.setattr(app, "create_llm", lambda provider, model, **kw: _StubCacheProvider(model=model))
    state: Dict[str, Any] = {"rows": [], "ejects": [], "residual": None}

    def fake_rows(backend=None):
        return [dict(r) for r in state["rows"] if backend is None or r["backend"] == backend]

    def fake_eject(backend, model, *, reason="eject"):
        state["ejects"].append((backend, model, reason))
        if state["residual"] is not None:
            return {"ok": False, "backend": backend, "model": model, "holders_found": 1,
                    "holders_unloaded": [], "holders_refused": [], "residual": state["residual"]}
        state["rows"] = [r for r in state["rows"] if not (r["backend"] == backend and model in r["models"])]
        return {"ok": True, "backend": backend, "model": model, "holders_found": 1,
                "holders_unloaded": [{"id": 1}], "holders_refused": [], "residual": None,
                "before": {"active_bytes": 10, "row": {"x": 1}}, "after": {"active_bytes": 0}}

    monkeypatch.setattr(pr, "resident_rows", fake_rows)
    monkeypatch.setattr(pr, "eject", fake_eject)
    yield app, TestClient(app.app), state
    app._GATEWAY_LOADED_RUNTIMES.clear()
    app._GATEWAY_RUNTIME_IDS.clear()


def _row(backend: str, model: str, held: int = 5_000, holders: int = 1, shared: bool = True) -> Dict[str, Any]:
    return {"backend": backend, "lane": "mlx_lm" if backend == "mlx" else backend, "models": [model],
            "holders": holders, "weights_bytes": held, "cache_bytes": 0, "held_bytes": held,
            "weights_alive": True, "shared_weights": shared}


def _by_id(body) -> Dict[str, Dict[str, Any]]:
    return {m["runtime_id"]: m for m in body["models"]}


def test_listing_names_models_held_outside_the_registry(server):
    app, client, state = server
    state["rows"] = [_row("mlx", "vendor/held-mlx", 17_000), _row("huggingface", "vendor/hf", 900, 2, shared=False),
                     _row("embeddings", "sentence-transformers/mini", 90)]
    body = client.get("/acore/models/loaded").json()
    rows = _by_id(body)
    mlx = rows["process:text_generation:mlx:vendor/held-mlx"]
    assert mlx["loaded"] is True and mlx["resident"] is True and mlx["held_bytes"] == 17_000
    assert mlx["provider_state"] == "resident_via_other_holders" and mlx["lockable"] is False
    hf = rows["process:text_generation:huggingface:vendor/hf"]
    assert hf["process_holders"] == 2 and any("full copy" in w for w in hf["warnings"])
    emb = rows["process:embedding:huggingface:sentence-transformers/mini"]
    assert emb["task"] == "embedding" and emb["held_bytes"] == 90

    only_emb = client.get("/acore/models/loaded", params={"task": "embedding"}).json()["models"]
    assert [m["task"] for m in only_emb] == ["embedding"]
    only_mlx = client.get("/acore/models/loaded", params={"provider": "mlx"}).json()["models"]
    assert [m["model"] for m in only_mlx] == ["vendor/held-mlx"]


def test_registry_record_is_annotated_not_duplicated(server):
    app, client, state = server
    assert client.post("/acore/models/load", json={"provider": "mlx", "model": "vendor/a"}).status_code == 200
    state["rows"] = [_row("mlx", "vendor/a", 1_000, holders=2)]
    models = client.get("/acore/models/loaded").json()["models"]
    assert len([m for m in models if m["model"] == "vendor/a"]) == 1
    assert next(m for m in models if m["model"] == "vendor/a")["process_holders"] == 2


def test_unload_of_a_managed_runtime_ejects_every_holder(server):
    app, client, state = server
    client.post("/acore/models/load", json={"provider": "mlx", "model": "vendor/a"})
    state["rows"] = [_row("mlx", "vendor/a")]
    r = client.post("/acore/models/unload", json={"provider": "mlx", "model": "vendor/a"})
    body = r.json()
    assert r.status_code == 200 and body["ok"] is True and body["unloaded"] is True
    assert state["ejects"] == [("mlx", "vendor/a", "acore_models_unload")]
    assert body["process_eject"]["holders_unloaded"] and "row" not in body["process_eject"]["before"]
    assert not [m for m in client.get("/acore/models/loaded").json()["models"] if m["model"] == "vendor/a"]


def test_unload_never_claims_success_over_residual_weights(server):
    app, client, state = server
    client.post("/acore/models/load", json={"provider": "mlx", "model": "vendor/a"})
    state["residual"] = {"holders": 1, "held_bytes": 4_000, "holder_rows": [{"id": 9}]}
    body = client.post("/acore/models/unload", json={"provider": "mlx", "model": "vendor/a"}).json()
    assert body["ok"] is False and body["unloaded"] is False
    assert "still resident" in body["error"] and "holder_rows" not in body["process_eject"]["residual"]


def test_unload_of_a_process_only_model_by_pair_and_by_listing_id(server):
    app, client, state = server
    state["rows"] = [_row("mlx", "vendor/held"), _row("huggingface", "vendor/hf")]
    r = client.post("/acore/models/unload", json={"provider": "mlx", "model": "vendor/held"})
    assert r.status_code == 200 and r.json()["ok"] is True
    r2 = client.post("/acore/models/unload", json={"runtime_id": "process:text_generation:huggingface:vendor/hf"})
    assert r2.status_code == 200 and r2.json()["ok"] is True
    assert [e[:2] for e in state["ejects"]] == [("mlx", "vendor/held"), ("huggingface", "vendor/hf")]
    missing = client.post("/acore/models/unload", json={"task": "text_generation", "provider": "mlx", "model": "vendor/none"})
    assert missing.status_code == 404 and len(state["ejects"]) == 2, "nothing held -> no eject, not found"


def test_embedding_unload_reaches_the_embedders(server):
    app, client, state = server
    state["rows"] = [_row("embeddings", "sentence-transformers/mini")]
    r = client.post("/acore/models/unload", json={"task": "embedding", "model": "sentence-transformers/mini"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert state["ejects"] == [("embeddings", "sentence-transformers/mini", "acore_models_unload")]
    state["rows"] = [_row("embeddings", "m/x")]
    r2 = client.post("/acore/models/unload", json={"runtime_id": "process:embedding:huggingface:m/x"})
    assert r2.status_code == 200 and state["ejects"][-1][:2] == ("embeddings", "m/x")


def test_unload_after_ejects_process_wide_unless_a_managed_runtime_still_serves(server):
    app, client, state = server
    client.post("/acore/models/load", json={"provider": "mlx", "model": "vendor/a"})
    runtime = next(iter(app._GATEWAY_LOADED_RUNTIMES.values()))
    app._best_effort_unload_loaded_gateway_runtime(runtime, request_id="r1")
    assert state["ejects"] == [("mlx", "vendor/a", "unload_after")]

    # A raw per-request provider unloaded while a managed runtime still serves
    # the pair: the managed runtime's residency is not this request's to end.
    client.post("/acore/models/load", json={"provider": "mlx", "model": "vendor/b"})
    app._best_effort_unload(_StubCacheProvider(model="vendor/b"), request_id="r2", provider="mlx", model="vendor/b")
    assert state["ejects"] == [("mlx", "vendor/a", "unload_after")]


def test_remote_providers_are_never_process_ejected(server):
    app, client, state = server
    client.post("/acore/models/load", json={"provider": "ollama", "model": "gemma3:1b"})
    body = client.post("/acore/models/unload", json={"provider": "ollama", "model": "gemma3:1b"}).json()
    assert body["ok"] is True and "process_eject" not in body and state["ejects"] == []
