"""Server visibility control plane: /acore/memory, the loaded-models sweep
merge, cross-runtime prompt-cache stats, and prompt-cache key metadata.

Unit-level: gateway runtimes are stubbed providers and the model-server sweep
is faked — no live Ollama/LM Studio/models involved.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _StubCacheProvider(BaseProvider):
    def __init__(self, model: str = "stub-model", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.provider = "mlx"
        self.unload_model_calls: List[str] = []

    def supports_prompt_cache(self) -> bool:
        return True

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        return {"chunks": []}

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


def _fresh_server(monkeypatch):
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    monkeypatch.setattr(
        server_app, "create_llm", lambda provider, model, **kwargs: _StubCacheProvider(model=model)
    )
    return server_app, TestClient(server_app.app)


def test_acore_memory_reports_snapshot() -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    r = client.get("/acore/memory")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert set(body["ram"].keys()) == {"total_bytes", "available_bytes", "used_bytes", "percent"}
    assert set(body["process"].keys()) == {"rss_bytes"}
    assert set(body["device"].keys()) == {
        "backend",
        "allocated_bytes",
        "total_bytes",
        "free_bytes",
        # host-wide accelerator truth (Metal; null elsewhere)
        "host_in_use_bytes",
        "wired_limit_bytes",
    }
    assert isinstance(body["ts"], float)


def test_acore_models_loaded_merges_provider_server_sweep(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "ollama", "model": "gemma3:1b"})
    assert load.status_code == 200
    runtime_id = load.json()["runtime"]["runtime_id"]

    load2 = client.post("/acore/models/load", json={"provider": "ollama", "model": "qwen3"})
    assert load2.status_code == 200

    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {
                "provider": "ollama",
                "model": "gemma3:1b",
                "resident": True,
                "loaded": True,
                "size_bytes": 111,
                "size_vram_bytes": 222,
                "source": "provider_server",
            },
            # `:latest` alias of the second registry runtime — must dedup.
            {
                "provider": "ollama",
                "model": "qwen3:latest",
                "resident": True,
                "loaded": True,
                "size_bytes": 444,
                "source": "provider_server",
            },
            # Sweep-only entry: no gateway runtime wraps it.
            {
                "provider": "lmstudio",
                "model": "qwen/qwen3-4b",
                "provider_instance_ids": ["inst-1"],
                "resident": True,
                "loaded": True,
                "size_bytes": 333,
                "source": "provider_server",
            },
        ],
    )

    loaded = client.get("/acore/models/loaded")
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["ok"] is True and body["success"] is True
    records = body["data"]
    assert body["models"] == records and body["affected_models"] == records

    by_model = {(r.get("provider"), r.get("model")): r for r in records}
    assert len(records) == 3

    # Registry records win and absorb the sweep's memory fields.
    registry_record = by_model[("ollama", "gemma3:1b")]
    assert registry_record["runtime_id"] == runtime_id
    assert registry_record["size_bytes"] == 111
    assert registry_record["size_vram_bytes"] == 222

    alias_record = by_model[("ollama", "qwen3")]
    assert "runtime_id" in alias_record
    assert alias_record["size_bytes"] == 444

    # Sweep-only entries appear as provider_server records — with NO task
    # label: the server enumerations cannot classify what is resident
    # (Ollama's /api/ps lists embedding models too).
    sweep_record = by_model[("lmstudio", "qwen/qwen3-4b")]
    assert sweep_record["source"] == "provider_server"
    assert sweep_record["loaded"] is True
    assert sweep_record["resident"] is True
    assert "task" not in sweep_record
    assert "runtime_id" not in sweep_record


def test_acore_models_loaded_model_filter_matches_sweep_aliases(monkeypatch) -> None:
    """`?provider=ollama&model=qwen3` must return the sweep-only `qwen3:latest`
    row — the filter uses the same normalization as dedup."""
    server_app, client = _fresh_server(monkeypatch)

    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {"provider": "ollama", "model": "qwen3:latest", "resident": True, "loaded": True, "source": "provider_server"}
        ],
    )

    loaded = client.get("/acore/models/loaded", params={"provider": "ollama", "model": "qwen3"})
    assert loaded.status_code == 200
    records = loaded.json()["data"]
    assert [r["model"] for r in records] == ["qwen3:latest"]
    assert records[0]["source"] == "provider_server"

    # Text-generation-filtered listings also carry the (untasked) sweep rows.
    loaded_text = client.get(
        "/acore/models/loaded", params={"task": "text_generation", "provider": "ollama"}
    )
    assert [r["model"] for r in loaded_text.json()["data"]] == ["qwen3:latest"]

    # A non-alias model filter still excludes it.
    other = client.get("/acore/models/loaded", params={"provider": "ollama", "model": "gemma3"})
    assert other.json()["data"] == []


def test_acore_models_loaded_lmstudio_variant_dedups_against_server_key(monkeypatch) -> None:
    """LM Studio resolves models by substring: a runtime named `qwen3-vl-4b`
    and the server key `qwen/qwen3-vl-4b` are ONE resident model — two rows
    would double-count its memory."""
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "lmstudio", "model": "qwen3-vl-4b"})
    assert load.status_code == 200

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
                "size_bytes": 3109915433,
                "source": "provider_server",
            }
        ],
    )

    loaded = client.get("/acore/models/loaded")
    assert loaded.status_code == 200
    records = loaded.json()["data"]
    assert len(records) == 1
    assert records[0]["model"] == "qwen3-vl-4b"
    assert records[0]["size_bytes"] == 3109915433


def test_acore_models_loaded_skips_sweep_for_non_sweep_providers(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    sweep_calls: List[int] = []

    def _recording_sweep(timeout_s: float = 2.0):
        sweep_calls.append(1)
        return []

    monkeypatch.setattr(server_app, "sweep_loaded_models", _recording_sweep)

    assert client.get("/acore/models/loaded", params={"provider": "mlx"}).status_code == 200
    assert sweep_calls == []

    assert client.get("/acore/models/loaded", params={"provider": "ollama"}).status_code == 200
    assert len(sweep_calls) == 1


def test_acore_models_loaded_sweep_failure_never_breaks_endpoint(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    def _boom(timeout_s: float = 2.0):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(server_app, "sweep_loaded_models", _boom)

    loaded = client.get("/acore/models/loaded")
    assert loaded.status_code == 200
    assert loaded.json()["ok"] is True
    assert loaded.json()["data"] == []


def test_acore_prompt_cache_stats_without_selector_enumerates_runtimes(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    runtime_id = load.json()["runtime"]["runtime_id"]

    set_key = client.post(
        "/acore/prompt_cache/set",
        json={"provider": "mlx", "model": "stub-model", "key": "k1"},
    )
    assert set_key.status_code == 200
    assert set_key.json()["ok"] is True

    stats = client.get("/acore/prompt_cache/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["ok"] is True
    assert len(body["runtimes"]) == 1
    row = body["runtimes"][0]
    assert row["runtime_id"] == runtime_id
    assert row["provider"] == "mlx"
    assert row["model"] == "stub-model"
    assert row["stats"]["entries"] == 1
    assert "k1" in row["stats"]["keys"]

    # The selector path is unchanged.
    selected = client.get(
        "/acore/prompt_cache/stats", params={"provider": "mlx", "model": "stub-model"}
    )
    assert selected.status_code == 200
    assert selected.json()["supported"] is True
    assert selected.json()["stats"]["entries"] == 1


def test_acore_prompt_cache_stats_enumeration_reports_busy_runtime(monkeypatch) -> None:
    """A runtime holding its lock (mid-generation) must yield a `busy` row —
    never stall the whole enumeration."""
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))

    assert runtime.lock.acquire(timeout=1.0)
    try:
        stats = client.get("/acore/prompt_cache/stats")
        assert stats.status_code == 200
        rows = stats.json()["runtimes"]
        assert len(rows) == 1
        assert rows[0]["stats"] is None
        assert rows[0]["error"] == "busy"
    finally:
        runtime.lock.release()


def test_acore_prompt_cache_stats_enumeration_does_not_wait_on_worker_thread(monkeypatch) -> None:
    """Stats are read off-worker: a runtime whose single provider-executor
    thread is busy (generation in flight) still enumerates promptly."""
    import threading

    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))

    release = threading.Event()
    started = threading.Event()

    def _block_worker() -> None:
        started.set()
        release.wait(timeout=30.0)

    future = runtime.provider_executor.submit(_block_worker)
    assert started.wait(timeout=5.0)
    try:
        stats = client.get("/acore/prompt_cache/stats")
        assert stats.status_code == 200
        rows = stats.json()["runtimes"]
        assert len(rows) == 1
        assert rows[0]["stats"] is not None
        assert "error" not in rows[0]
    finally:
        release.set()
        future.result(timeout=5.0)


def test_acore_prompt_cache_key_meta_stamps_session_attribution(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200

    set_key = client.post(
        "/acore/prompt_cache/set",
        json={"provider": "mlx", "model": "stub-model", "key": "k1"},
    )
    assert set_key.status_code == 200

    # The full attribution vocabulary a remote runtime stamps must pass —
    # including `namespace` and a field literally named `key`.
    r = client.post(
        "/acore/prompt_cache/key_meta",
        json={
            "provider": "mlx",
            "model": "stub-model",
            "key": "k1",
            "meta": {
                "session_id": "sess-1",
                "run_id": "run-9",
                "workflow_id": "wf-2",
                "node_id": "node-3",
                "namespace": "tenant-a",
                "key": "caller-key-field",
                "skipped": None,
            },
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["key"] == "k1"
    assert body["meta"]["session_id"] == "sess-1"
    assert body["meta"]["run_id"] == "run-9"
    assert body["meta"]["workflow_id"] == "wf-2"
    assert body["meta"]["node_id"] == "node-3"
    assert body["meta"]["namespace"] == "tenant-a"
    assert body["meta"]["key"] == "caller-key-field"
    assert "skipped" not in body["meta"]

    # The stamped attribution is visible through stats meta_by_key.
    stats = client.get(
        "/acore/prompt_cache/stats", params={"provider": "mlx", "model": "stub-model"}
    )
    assert stats.status_code == 200
    assert stats.json()["stats"]["meta_by_key"]["k1"]["session_id"] == "sess-1"


def test_acore_prompt_cache_key_meta_missing_key_and_selector_errors(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200

    missing = client.post(
        "/acore/prompt_cache/key_meta",
        json={"provider": "mlx", "model": "stub-model", "key": "nope", "meta": {"a": 1}},
    )
    assert missing.status_code == 200
    assert missing.json()["ok"] is False
    assert missing.json()["code"] == "prompt_cache_missing_key"

    no_selector = client.post(
        "/acore/prompt_cache/key_meta", json={"key": "k1", "meta": {"a": 1}}
    )
    assert no_selector.status_code == 200
    assert no_selector.json()["supported"] is False
    assert "base_url or provider+model" in no_selector.json()["error"]


def test_acore_prompt_cache_key_meta_rejects_reserved_internal_keys(monkeypatch) -> None:
    """`fed_token_ids` (and friends) are the providers' record of what the
    resident KV actually holds — a caller overwriting them would falsify
    cache composition. key_meta must refuse, loudly."""
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    set_key = client.post(
        "/acore/prompt_cache/set",
        json={"provider": "mlx", "model": "stub-model", "key": "k1"},
    )
    assert set_key.status_code == 200

    for reserved_key in ("fed_token_ids", "binding_id", "token_count", "backend", "model"):
        r = client.post(
            "/acore/prompt_cache/key_meta",
            json={
                "provider": "mlx",
                "model": "stub-model",
                "key": "k1",
                "meta": {"session_id": "sess-1", reserved_key: "forged"},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["code"] == "prompt_cache_meta_reserved_key"
        assert reserved_key in body["error"]

    # Nothing was merged — the key's meta is untouched.
    stats = client.get(
        "/acore/prompt_cache/stats", params={"provider": "mlx", "model": "stub-model"}
    )
    assert "session_id" not in (stats.json()["stats"].get("meta_by_key", {}).get("k1") or {})


def test_acore_prompt_cache_key_meta_caps_payload_size(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    set_key = client.post(
        "/acore/prompt_cache/set",
        json={"provider": "mlx", "model": "stub-model", "key": "k1"},
    )
    assert set_key.status_code == 200

    r = client.post(
        "/acore/prompt_cache/key_meta",
        json={
            "provider": "mlx",
            "model": "stub-model",
            "key": "k1",
            "meta": {"blob": "x" * (17 * 1024)},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["code"] == "prompt_cache_meta_too_large"
    assert "16384" in body["error"]
