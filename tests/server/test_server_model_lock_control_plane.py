"""Model-lock control plane: /acore/models/lock, /acore/models/unlock, the
409 unload guard, force-unload, unload_after skip, and truthful lock fields
on residency records.

Unit-level: gateway runtimes wrap stub providers (plus an OllamaProvider with
a faked HTTP client for the provider-side keep_alive mapping) — no live
servers or models involved.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.providers.ollama_provider import OllamaProvider


class _StubGatewayProvider(BaseProvider):
    """Warm in-process stub: residency always verified/resident."""

    provider_name = "mlx"

    def __init__(self, model: str = "stub-model", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.provider = type(self).provider_name
        self.unload_model_calls: List[str] = []
        self.load_model_calls: List[Dict[str, Any]] = []

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        _ = (prompt, messages, system_prompt, tools, media, stream, kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        _ = kwargs
        return {
            "task": task,
            "provider": self.provider,
            "model": str(model or self.model),
            "provider_residency_verified": True,
            "provider_resident": True,
            "loaded": True,
            "state": "loaded",
            "source": "abstractcore.provider.test",
        }

    def unload_model(self, model_name: str) -> None:
        self.unload_model_calls.append(str(model_name))

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub-model"]


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.ps_models: List[Dict[str, Any]] = []
        self.posts: List[Dict[str, Any]] = []

    def get(self, url: str):  # noqa: ANN001
        import httpx

        return httpx.Response(200, json={"models": self.ps_models}, request=httpx.Request("GET", url))

    def post(self, url: str, *, json=None):  # noqa: ANN001
        import httpx

        self.posts.append({"url": url, "json": json})
        return httpx.Response(
            200,
            json={"model": (json or {}).get("model"), "done": True, "done_reason": "load"},
            request=httpx.Request("POST", url),
        )

    def close(self) -> None:
        return None


def _fake_ollama_provider(model: str) -> OllamaProvider:
    provider = object.__new__(OllamaProvider)
    provider.provider = "ollama"
    provider.model = model
    provider.base_url = "http://fake-ollama:11434"
    provider.client = _FakeOllamaClient()
    provider._async_client = None
    return provider


def _fresh_server(monkeypatch, provider_factory=None):
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    factory = provider_factory or (lambda provider, model, **kwargs: _StubGatewayProvider(model=model))
    monkeypatch.setattr(server_app, "create_llm", factory)
    return server_app, TestClient(server_app.app)


def _loaded_records(client) -> List[Dict[str, Any]]:
    r = client.get("/acore/models/loaded")
    assert r.status_code == 200
    return r.json()["data"]


# ---------------------------------------------------------------------------
# Lock / unlock endpoints
# ---------------------------------------------------------------------------

def test_lock_warm_runtime_and_truthful_record_fields(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    runtime_id = load.json()["runtime"]["runtime_id"]

    # Truthful default: an unlocked runtime is NOT pinned (the old record
    # hardcoded pinned: true regardless of reality).
    before = _loaded_records(client)[0]
    assert before["locked"] is False
    assert before["pinned"] is False
    assert before["lockable"] is True
    assert "locked_at" not in before

    lock = client.post("/acore/models/lock", json={"provider": "mlx", "model": "stub-model"})
    assert lock.status_code == 200
    body = lock.json()
    assert body["ok"] is True
    assert body["locked"] is True
    assert body["runtime_id"] == runtime_id
    assert body["provider"] == "mlx"
    assert body["model"] == "stub-model"
    # No provider-side pin knob exists for mlx; the registry lock still holds.
    assert body["provider_side"] == {"supported": False, "applied": False}

    after = _loaded_records(client)[0]
    assert after["locked"] is True
    assert after["pinned"] is True  # compat alias, now truthful
    assert after["lockable"] is True
    assert isinstance(after["locked_at"], float)


def test_unlock_clears_lock_by_runtime_id(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    runtime_id = load.json()["runtime"]["runtime_id"]
    assert client.post("/acore/models/lock", json={"runtime_id": runtime_id}).status_code == 200

    unlock = client.post("/acore/models/unlock", json={"runtime_id": runtime_id})
    assert unlock.status_code == 200
    assert unlock.json()["ok"] is True
    assert unlock.json()["locked"] is False
    assert unlock.json()["provider_side"] == {"supported": False, "applied": False}

    record = _loaded_records(client)[0]
    assert record["locked"] is False and record["pinned"] is False
    # An unlocked runtime unloads normally again.
    assert client.post("/acore/models/unload", json={"runtime_id": runtime_id}).status_code == 200
    assert _loaded_records(client) == []


def test_lock_selector_validation_and_missing_runtime(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    # The server's global HTTPException handler flattens details into
    # {"error": {"message", "type": "http_error"}} — same as the unload route.
    no_selector = client.post("/acore/models/lock", json={})
    assert no_selector.status_code == 400
    assert "runtime_id or provider+model" in no_selector.json()["error"]["message"]

    missing = client.post("/acore/models/lock", json={"provider": "mlx", "model": "never-loaded"})
    assert missing.status_code == 404
    assert "not found" in missing.json()["error"]["message"].lower()

    # Same shapes for unlock.
    assert client.post("/acore/models/unlock", json={}).status_code == 400
    assert client.post("/acore/models/unlock", json={"runtime_id": "nope"}).status_code == 404


# ---------------------------------------------------------------------------
# Unload guard: 409 without force, force unlocks then unloads
# ---------------------------------------------------------------------------

def test_unload_locked_runtime_conflicts_with_409(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    runtime_id = load.json()["runtime"]["runtime_id"]
    assert client.post("/acore/models/lock", json={"runtime_id": runtime_id}).status_code == 200

    unload = client.post("/acore/models/unload", json={"runtime_id": runtime_id})

    assert unload.status_code == 409
    body = unload.json()
    assert body["ok"] is False
    assert body["error"] == "model_locked"
    assert body["runtime_id"] == runtime_id
    assert "force" in body["detail"]
    # Still loaded, still locked.
    record = _loaded_records(client)[0]
    assert record["locked"] is True


def test_unload_locked_runtime_with_force_unlocks_and_unloads(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    runtime_id = load.json()["runtime"]["runtime_id"]
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))
    assert client.post("/acore/models/lock", json={"runtime_id": runtime_id}).status_code == 200

    unload = client.post("/acore/models/unload", json={"runtime_id": runtime_id, "force": True})

    assert unload.status_code == 200
    assert unload.json()["unloaded"] is True
    assert unload.json()["runtime"]["locked"] is False
    assert unload.json()["runtime"]["pinned"] is False
    assert runtime.llm.unload_model_calls == ["stub-model"]
    assert _loaded_records(client) == []


def test_raw_unload_after_skips_model_locked_by_managed_runtime(monkeypatch, caplog) -> None:
    """A chat request with an explicit base_url misses the managed registry
    (key mismatch) and reaches the RAW `_best_effort_unload` with a fresh
    provider instance — it must still not evict a model a locked managed
    runtime pins (LM Studio unloads are real server-side evictions). Covers
    the ollama unsafe-gate path too: both funnel through the same helper."""
    import logging

    server_app, client = _fresh_server(monkeypatch)
    load = client.post("/acore/models/load", json={"provider": "lmstudio", "model": "qwen3-4b"})
    assert load.status_code == 200
    assert client.post("/acore/models/lock", json={"provider": "lmstudio", "model": "qwen3-4b"}).status_code == 200

    raw_llm = _StubGatewayProvider(model="qwen/qwen3-4b")

    # The raw request names an LM Studio variant of the locked model.
    with caplog.at_level(logging.INFO):
        server_app._best_effort_unload(
            raw_llm, request_id="test-req", provider="lmstudio", model="qwen/qwen3-4b"
        )
    assert raw_llm.unload_model_calls == []
    assert "Provider unload skipped" in caplog.text

    # Ollama `:latest` aliases match the same guard.
    ollama_load = client.post("/acore/models/load", json={"provider": "ollama", "model": "qwen3"})
    assert ollama_load.status_code == 200
    assert client.post("/acore/models/lock", json={"provider": "ollama", "model": "qwen3"}).status_code == 200
    raw_ollama = _StubGatewayProvider(model="qwen3:latest")
    server_app._best_effort_unload(raw_ollama, request_id="test-req", provider="ollama", model="qwen3:latest")
    assert raw_ollama.unload_model_calls == []

    # A different model — and the same model once unlocked — unloads normally.
    other = _StubGatewayProvider(model="other-model")
    server_app._best_effort_unload(other, request_id="test-req", provider="lmstudio", model="other-model")
    assert other.unload_model_calls == ["other-model"]

    assert client.post("/acore/models/unlock", json={"provider": "lmstudio", "model": "qwen3-4b"}).status_code == 200
    server_app._best_effort_unload(raw_llm, request_id="test-req", provider="lmstudio", model="qwen/qwen3-4b")
    assert raw_llm.unload_model_calls == ["qwen/qwen3-4b"]


def test_force_unload_failure_keeps_runtime_locked(monkeypatch) -> None:
    """A failed force-unload must leave the runtime registered AND locked —
    never resident-but-silently-unlocked."""

    class _BrokenUnloadProvider(_StubGatewayProvider):
        def unload_model(self, model_name: str) -> None:
            raise RuntimeError("provider unload exploded")

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _BrokenUnloadProvider(model=model)
    )
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    runtime_id = load.json()["runtime"]["runtime_id"]
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))
    assert client.post("/acore/models/lock", json={"runtime_id": runtime_id}).status_code == 200

    # raise_server_exceptions=False: assert the 500 the real server would send
    # instead of the TestClient re-raising the handler-processed exception.
    tolerant_client = TestClient(server_app.app, raise_server_exceptions=False)
    unload = tolerant_client.post("/acore/models/unload", json={"runtime_id": runtime_id, "force": True})

    assert unload.status_code == 500  # error surfaced, not swallowed
    assert runtime.runtime_id in server_app._GATEWAY_RUNTIME_IDS
    assert runtime.locked is True
    assert runtime.locked_at is not None
    # And the next plain unload still conflicts — the lock survived.
    assert client.post("/acore/models/unload", json={"runtime_id": runtime_id}).status_code == 409


def test_lock_refuses_runtime_that_vanished_from_registry(monkeypatch) -> None:
    """`_lock_gateway_runtime` re-verifies registration under the registry
    lock: a lock flag on an unregistered runtime would be unenforceable."""
    from fastapi import HTTPException
    import pytest as _pytest

    server_app, client = _fresh_server(monkeypatch)
    client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))
    server_app._drop_loaded_gateway_runtime(runtime)

    with _pytest.raises(HTTPException) as exc_info:
        server_app._lock_gateway_runtime(runtime)

    assert exc_info.value.status_code == 404
    assert runtime.locked is False


def test_lock_refuses_a_non_resident_runtime_with_409(monkeypatch) -> None:
    """LOCK RULE: lock requires provider-VERIFIED residency. A warm registry
    entry alone (the lmstudio 'constructed HTTP client counts as warm'
    symptom) is configuration, not memory — locking it must refuse instead of
    presenting a configured model as loaded."""

    class _NotResidentProvider(_StubGatewayProvider):
        def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            record = super().get_model_residency(task=task, model=model, **kwargs)
            record.update({"provider_resident": False, "loaded": False, "state": "not_loaded"})
            return record

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _NotResidentProvider(model=model)
    )
    load = client.post("/acore/models/load", json={"provider": "lmstudio", "model": "qwen3-35b"})
    assert load.status_code == 200  # warm registry entry exists...
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))

    lock = client.post("/acore/models/lock", json={"provider": "lmstudio", "model": "qwen3-35b"})

    assert lock.status_code == 409
    body = lock.json()
    assert body["ok"] is False
    assert body["error"] == "model_not_resident"
    assert "lock:true" in body["detail"]
    assert body["runtime_id"] == runtime.runtime_id
    # The registry flag never flipped; unload needs no force.
    assert runtime.locked is False
    record = _loaded_records(client)[0]
    assert record["locked"] is False and record["pinned"] is False
    assert client.post("/acore/models/unload", json={"provider": "lmstudio", "model": "qwen3-35b"}).status_code == 200


def test_load_with_lock_on_unverifiable_residency_reports_refusal_but_load_stays_ok(monkeypatch) -> None:
    """`load {lock:true}` on a provider that cannot verify residency: the load
    succeeds, the lock block reports the model_not_resident refusal additively."""

    class _UnverifiableProvider(_StubGatewayProvider):
        def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            record = super().get_model_residency(task=task, model=model, **kwargs)
            record.update({"provider_residency_verified": False, "provider_resident": None, "loaded": None})
            return record

        def load_model(self, model_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            self.load_model_calls.append({"model": model_name, "kwargs": dict(kwargs)})
            return {"ok": True}

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _UnverifiableProvider(model=model)
    )

    load = client.post("/acore/models/load", json={"provider": "lmstudio", "model": "m", "lock": True})

    assert load.status_code == 200
    body = load.json()
    assert body["ok"] is True
    assert body["lock"]["locked"] is False
    assert body["lock"]["error"] == "model_not_resident"
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))
    assert runtime.locked is False


def test_unlock_of_evicted_locked_ollama_runtime_never_loads_it_back(monkeypatch) -> None:
    """Unlock must reach a locked-but-since-evicted runtime (locks are never
    stranded) WITHOUT the keep_alive restore riding Ollama's native load
    request and re-loading the evicted model as a side effect."""
    providers: Dict[str, OllamaProvider] = {}

    def factory(provider: str, model: str, **kwargs: Any) -> OllamaProvider:
        instance = _fake_ollama_provider(model)
        instance.client.ps_models = [{"name": model, "model": model, "size": 1000}]
        providers[model] = instance
        return instance

    server_app, client = _fresh_server(monkeypatch, provider_factory=factory)
    assert client.post("/acore/models/load", json={"provider": "ollama", "model": "qwen3:4b"}).status_code == 200
    assert client.post("/acore/models/lock", json={"provider": "ollama", "model": "qwen3:4b"}).status_code == 200
    fake = providers["qwen3:4b"]
    posts_after_lock = list(fake.client.posts)

    # Server-side eviction: /api/ps no longer lists the model.
    fake.client.ps_models = []
    unlock = client.post("/acore/models/unlock", json={"provider": "ollama", "model": "qwen3:4b"})

    assert unlock.status_code == 200
    body = unlock.json()
    assert body["ok"] is True and body["locked"] is False
    assert body["provider_side"]["supported"] is True
    assert body["provider_side"]["applied"] is False
    assert "not resident" in body["provider_side"]["detail"]
    # No new /api/generate call: the evicted model was NOT loaded back.
    assert fake.client.posts == posts_after_lock


def test_unlock_with_unverifiable_residency_skips_restore_honestly(monkeypatch) -> None:
    """UNKNOWN residency is not evidence of eviction: a transient probe
    failure still skips the keep_alive restore (safe — the knob rides a load
    request), but the detail says "unverified", never "not resident"."""
    providers: Dict[str, OllamaProvider] = {}

    def factory(provider: str, model: str, **kwargs: Any) -> OllamaProvider:
        instance = _fake_ollama_provider(model)
        instance.client.ps_models = [{"name": model, "model": model, "size": 1000}]
        providers[model] = instance
        return instance

    server_app, client = _fresh_server(monkeypatch, provider_factory=factory)
    assert client.post("/acore/models/load", json={"provider": "ollama", "model": "qwen3:4b"}).status_code == 200
    assert client.post("/acore/models/lock", json={"provider": "ollama", "model": "qwen3:4b"}).status_code == 200
    fake = providers["qwen3:4b"]
    posts_after_lock = list(fake.client.posts)

    def _boom(url: str):  # noqa: ANN001 - transient transport failure
        raise RuntimeError("probe transport failed")

    fake.client.get = _boom  # residency now UNVERIFIABLE, not proven absent
    unlock = client.post("/acore/models/unlock", json={"provider": "ollama", "model": "qwen3:4b"})

    assert unlock.status_code == 200
    body = unlock.json()
    assert body["ok"] is True and body["locked"] is False
    assert body["provider_side"]["supported"] is True
    assert body["provider_side"]["applied"] is False
    assert "unverified" in body["provider_side"]["detail"]
    assert "not resident" not in body["provider_side"]["detail"]
    assert fake.client.posts == posts_after_lock  # no load side effect


def test_unload_after_best_effort_skips_locked_runtime(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))
    runtime.locked = True

    server_app._best_effort_unload_loaded_gateway_runtime(runtime, request_id="test-req")

    # Skipped: still registered, provider untouched, still locked.
    assert runtime.runtime_id in server_app._GATEWAY_RUNTIME_IDS
    assert runtime.llm.unload_model_calls == []
    assert runtime.locked is True

    runtime.locked = False
    server_app._best_effort_unload_loaded_gateway_runtime(runtime, request_id="test-req")
    assert runtime.runtime_id not in server_app._GATEWAY_RUNTIME_IDS
    assert runtime.llm.unload_model_calls == ["stub-model"]


# ---------------------------------------------------------------------------
# Provider-side knob mapping
# ---------------------------------------------------------------------------

def test_ollama_lock_unlock_map_keep_alive_via_native_rest(monkeypatch) -> None:
    providers: Dict[str, OllamaProvider] = {}

    def factory(provider: str, model: str, **kwargs: Any) -> OllamaProvider:
        instance = _fake_ollama_provider(model)
        # Lock requires provider-verified residency: the server's /api/ps
        # must actually hold the model.
        instance.client.ps_models = [{"name": model, "model": model, "size": 1000}]
        providers[model] = instance
        return instance

    server_app, client = _fresh_server(monkeypatch, provider_factory=factory)

    load = client.post("/acore/models/load", json={"provider": "ollama", "model": "qwen3:4b"})
    assert load.status_code == 200
    fake = providers["qwen3:4b"]

    lock = client.post("/acore/models/lock", json={"provider": "ollama", "model": "qwen3:4b"})
    assert lock.status_code == 200
    assert lock.json()["provider_side"] == {"supported": True, "applied": True}
    assert fake.client.posts[-1]["url"] == "http://fake-ollama:11434/api/generate"
    assert fake.client.posts[-1]["json"] == {
        "model": "qwen3:4b",
        "prompt": "",
        "stream": False,
        "keep_alive": -1,
    }

    unlock = client.post("/acore/models/unlock", json={"provider": "ollama", "model": "qwen3:4b"})
    assert unlock.status_code == 200
    assert unlock.json()["provider_side"] == {"supported": True, "applied": True}
    assert fake.client.posts[-1]["json"]["keep_alive"] == "5m"


def test_provider_side_failure_never_fails_the_lock(monkeypatch) -> None:
    class _BrokenKnobOllamaStub(_StubGatewayProvider):
        provider_name = "ollama"

        def load_model(self, model_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("keep_alive knob exploded")

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _BrokenKnobOllamaStub(model=model)
    )
    load = client.post("/acore/models/load", json={"provider": "ollama", "model": "qwen3:4b"})
    assert load.status_code == 200

    lock = client.post("/acore/models/lock", json={"provider": "ollama", "model": "qwen3:4b"})

    assert lock.status_code == 200
    body = lock.json()
    assert body["ok"] is True and body["locked"] is True
    assert body["provider_side"]["supported"] is True
    assert body["provider_side"]["applied"] is False
    assert "keep_alive knob exploded" in body["provider_side"]["detail"]
    # The registry lock is fully enforced regardless.
    assert client.post("/acore/models/unload", json={"provider": "ollama", "model": "qwen3:4b"}).status_code == 409


# ---------------------------------------------------------------------------
# Load-time lock flag + record truth
# ---------------------------------------------------------------------------

def test_load_with_lock_flag_locks_registry(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model", "lock": True})

    assert load.status_code == 200
    body = load.json()
    assert body["runtime"]["locked"] is True
    assert body["runtime"]["pinned"] is True
    assert body["lock"] == {"locked": True, "provider_side": {"supported": False, "applied": False}}
    assert client.post("/acore/models/unload", json={"provider": "mlx", "model": "stub-model"}).status_code == 409
    assert (
        client.post("/acore/models/unload", json={"provider": "mlx", "model": "stub-model", "force": True}).status_code
        == 200
    )


def test_managed_ollama_record_surfaces_expires_at_from_claim(monkeypatch) -> None:
    def factory(provider: str, model: str, **kwargs: Any) -> OllamaProvider:
        instance = _fake_ollama_provider(model)
        instance.client.ps_models = [
            {
                "name": model,
                "model": model,
                "expires_at": "2318-08-31T12:29:48+02:00",
                "size": 1000,
                "size_vram": 900,
            }
        ]
        return instance

    server_app, client = _fresh_server(monkeypatch, provider_factory=factory)
    load = client.post("/acore/models/load", json={"provider": "ollama", "model": "gemma3:1b"})
    assert load.status_code == 200

    record = _loaded_records(client)[0]
    assert record["expires_at"] == "2318-08-31T12:29:48+02:00"
    assert record["size_bytes"] == 1000
    assert record["size_vram_bytes"] == 900


def test_managed_record_carries_est_weights_and_cache_bytes(monkeypatch) -> None:
    """Per-model memory footprint on managed rows: `est_weights_bytes` flows
    from the provider claim; `cache_bytes` is computed from the instance's
    prompt-cache store stats (per-key bytes + MLX snapshot bytes)."""

    class _FootprintProvider(_StubGatewayProvider):
        def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            record = super().get_model_residency(task=task, model=model, **kwargs)
            record["est_weights_bytes"] = 3_100_000_000
            return record

        def get_prompt_cache_stats(self) -> Dict[str, Any]:
            return {
                "keys": ["a", "b"],
                "meta_by_key": {"a": {"bytes": 100}, "b": {"bytes": 23}},
                "snapshots": {"count": 1, "bytes": 1000},
            }

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _FootprintProvider(model=model)
    )
    assert client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"}).status_code == 200

    record = _loaded_records(client)[0]

    assert record["est_weights_bytes"] == 3_100_000_000
    assert record["cache_bytes"] == 1123


def test_managed_record_cache_bytes_absent_when_stats_unavailable(monkeypatch) -> None:
    class _BrokenStatsProvider(_StubGatewayProvider):
        def get_prompt_cache_stats(self) -> Dict[str, Any]:
            raise RuntimeError("stats exploded")

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _BrokenStatsProvider(model=model)
    )
    assert client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"}).status_code == 200

    record = _loaded_records(client)[0]

    assert "cache_bytes" not in record  # unknown stays unknown, never 0


def test_sweep_only_rows_are_lockable(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {"provider": "ollama", "model": "gemma3:1b", "resident": True, "loaded": True, "source": "provider_server"},
            {"provider": "lmstudio", "model": "qwen/qwen3-4b", "resident": True, "loaded": True, "source": "provider_server"},
        ],
    )

    records = _loaded_records(client)

    assert len(records) == 2
    # Sweep-resident rows are lockable: `POST /acore/models/lock` ADOPTS them
    # into the managed registry (client construction only) and then enforces
    # the lock like any managed runtime.
    assert all(record["lockable"] is True for record in records)
    assert all("locked" not in record for record in records)


# ---------------------------------------------------------------------------
# Lock ADOPTION of sweep-resident models (externally-loaded Ollama/LM Studio)
# ---------------------------------------------------------------------------


class _ResidentLMStudioStub(_StubGatewayProvider):
    """LM Studio-flavored stub: resident, records every load/generate call so
    adoption can pin that NO provider-side load is ever triggered."""

    provider_name = "lmstudio"

    def load_model(self, model_name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        self.load_model_calls.append({"model": model_name, "kwargs": dict(kwargs)})
        return {"ok": True}


def test_lock_adopts_a_sweep_resident_lmstudio_model_without_loading(monkeypatch) -> None:
    """Operator rule: every LOADED model is lockable — including one loaded by
    LM Studio itself. The lock ADOPTS it (client construction only; the fake
    pins that load_model/generate are NEVER called during adoption)."""
    constructed: List[_ResidentLMStudioStub] = []

    def factory(provider: str, model: str, **kwargs: Any) -> _ResidentLMStudioStub:
        instance = _ResidentLMStudioStub(model=model)
        constructed.append(instance)
        return instance

    server_app, client = _fresh_server(monkeypatch, provider_factory=factory)
    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {
                "provider": "lmstudio",
                "model": "qwen/qwen3-vl-4b",
                "provider_instance_ids": ["qwen/qwen3-vl-4b"],
                "resident": True,
                "loaded": True,
                "source": "provider_server",
            }
        ],
    )
    assert server_app._GATEWAY_LOADED_RUNTIMES == {}  # nothing managed yet

    lock = client.post("/acore/models/lock", json={"provider": "lmstudio", "model": "qwen/qwen3-vl-4b"})

    assert lock.status_code == 200
    body = lock.json()
    assert body["ok"] is True and body["locked"] is True
    assert body["adopted"] is True
    assert body["provider"] == "lmstudio"
    # LM Studio has no residency-pin knob; the honesty detail states the scope.
    assert body["provider_side"]["supported"] is False
    assert body["provider_side"]["applied"] is False
    assert "may still evict" in body["provider_side"]["detail"]
    # ADOPTION IS CONSTRUCTION ONLY: no provider-side load, no generate.
    assert len(constructed) == 1
    assert constructed[0].load_model_calls == []
    # The adopted runtime is managed: listed, locked, and unload follows the
    # normal 409/force rules.
    record = _loaded_records(client)[0]
    assert record["locked"] is True and record["lockable"] is True
    assert client.post("/acore/models/unload", json={"provider": "lmstudio", "model": "qwen/qwen3-vl-4b"}).status_code == 409
    assert (
        client.post(
            "/acore/models/unload", json={"provider": "lmstudio", "model": "qwen/qwen3-vl-4b", "force": True}
        ).status_code
        == 200
    )


def test_lock_adopts_a_sweep_resident_ollama_model_and_applies_keep_alive(monkeypatch) -> None:
    """Ollama adoption: the keep_alive=-1 knob applies (it rides a native
    request against an already-resident model — a refresh, not a load)."""
    providers: Dict[str, OllamaProvider] = {}

    def factory(provider: str, model: str, **kwargs: Any) -> OllamaProvider:
        instance = _fake_ollama_provider(model)
        instance.client.ps_models = [{"name": model, "model": model, "size": 1000}]
        providers[model] = instance
        return instance

    server_app, client = _fresh_server(monkeypatch, provider_factory=factory)
    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        # The sweep row carries Ollama's `:latest` alias; the lock names the bare model.
        lambda timeout_s=2.0: [
            {"provider": "ollama", "model": "qwen3:latest", "resident": True, "loaded": True, "source": "provider_server"}
        ],
    )

    lock = client.post("/acore/models/lock", json={"provider": "ollama", "model": "qwen3"})

    assert lock.status_code == 200
    body = lock.json()
    assert body["ok"] is True and body["locked"] is True
    assert body["adopted"] is True
    assert body["provider_side"] == {"supported": True, "applied": True}
    fake = providers["qwen3"]
    # Exactly ONE provider-side call: the keep_alive=-1 knob (never a plain load).
    assert [p["json"].get("keep_alive") for p in fake.client.posts] == [-1]


def test_lock_of_a_non_sweep_resident_model_refuses_with_409(monkeypatch) -> None:
    """Unmanaged + not sweep-resident -> the existing model_not_resident 409
    (nothing adopted: runtime_id is null and the registry stays empty)."""
    server_app, client = _fresh_server(monkeypatch)
    monkeypatch.setattr(server_app, "sweep_loaded_models", lambda timeout_s=2.0: [])

    lock = client.post("/acore/models/lock", json={"provider": "ollama", "model": "never-loaded"})

    assert lock.status_code == 409
    body = lock.json()
    assert body["ok"] is False
    assert body["error"] == "model_not_resident"
    assert body["runtime_id"] is None
    assert "lock:true" in body["detail"]
    assert server_app._GATEWAY_LOADED_RUNTIMES == {}


def test_adoption_refused_by_provider_probe_drops_the_adopted_runtime(monkeypatch) -> None:
    """The sweep said resident but the provider's own post-construction probe
    disagreed: the 409 stands and the just-adopted registry entry is dropped
    (no stray 'configured' row appears)."""

    class _NotResidentLMStudioStub(_ResidentLMStudioStub):
        def get_model_residency(self, *, task: str = "text_generation", model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
            record = super().get_model_residency(task=task, model=model, **kwargs)
            record.update({"provider_resident": False, "loaded": False, "state": "not_loaded"})
            return record

    server_app, client = _fresh_server(
        monkeypatch, provider_factory=lambda provider, model, **kwargs: _NotResidentLMStudioStub(model=model)
    )
    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {"provider": "lmstudio", "model": "phantom", "resident": True, "loaded": True, "source": "provider_server"}
        ],
    )

    lock = client.post("/acore/models/lock", json={"provider": "lmstudio", "model": "phantom"})

    assert lock.status_code == 409
    assert lock.json()["error"] == "model_not_resident"
    assert lock.json()["runtime_id"] is None
    assert server_app._GATEWAY_LOADED_RUNTIMES == {}


def test_lock_by_runtime_id_never_adopts(monkeypatch) -> None:
    """Adoption is a provider+model convenience; an unknown runtime_id stays
    the 404 (a runtime_id names a managed entry or nothing)."""
    server_app, client = _fresh_server(monkeypatch)
    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {"provider": "ollama", "model": "gemma3:1b", "resident": True, "loaded": True, "source": "provider_server"}
        ],
    )

    missing = client.post("/acore/models/lock", json={"runtime_id": "rid-unknown"})

    assert missing.status_code == 404
    assert server_app._GATEWAY_LOADED_RUNTIMES == {}
