"""Context-estimate endpoint + modality/host-identity stamping on residency
records. Unit-level: stub providers, faked sweep, no live servers.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterator, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.utils.hostinfo import get_host_identity


class _StubProvider(BaseProvider):
    residency_extra: Dict[str, Any] = {}

    def __init__(self, model: str = "stub-model", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.provider = "mlx"

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
            **type(self).residency_extra,
        }

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    @classmethod
    def list_available_models(cls, **kwargs: Any) -> List[str]:
        _ = kwargs
        return ["stub-model"]


def _fresh_server(monkeypatch):
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()
    _StubProvider.residency_extra = {}
    monkeypatch.setattr(server_app, "create_llm", lambda provider, model, **kwargs: _StubProvider(model=model))
    return server_app, TestClient(server_app.app)


def _loaded_records(client) -> List[Dict[str, Any]]:
    r = client.get("/acore/models/loaded")
    assert r.status_code == 200
    return r.json()["data"]


# ---------------------------------------------------------------------------
# /acore/models/context_estimate
# ---------------------------------------------------------------------------

def test_context_estimate_endpoint_passes_through_estimator(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    calls: List[tuple] = []

    def fake_estimate(provider: str, model: str, context_length: Optional[int] = None, base_url: Optional[str] = None):
        calls.append((provider, model, context_length))
        return {"ok": True, "provider": provider, "model": model, "confidence": "estimated", "predicted_max_context": 4096, "memory": {}, "notes": []}

    monkeypatch.setattr(server_app, "estimate_context_fit", fake_estimate)

    r = client.get(
        "/acore/models/context_estimate",
        params={"provider": "huggingface", "model": "some/model", "context_length": 8192},
    )

    assert r.status_code == 200
    assert calls == [("huggingface", "some/model", 8192)]
    assert r.json()["confidence"] == "estimated"
    assert r.json()["predicted_max_context"] == 4096


def test_context_estimate_endpoint_unknown_is_honest(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)

    r = client.get("/acore/models/context_estimate", params={"provider": "lmstudio", "model": "whatever"})

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["confidence"] == "unknown"
    assert body["predicted_max_context"] is None

    # provider/model are required query params.
    assert client.get("/acore/models/context_estimate").status_code == 422


def test_gguf_calibration_fields_flow_from_provider_claim(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    _StubProvider.residency_extra = {"context_calibrated": True, "calibrated_context_length": 8192}

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200

    record = _loaded_records(client)[0]
    assert record["context_calibrated"] is True
    assert record["calibrated_context_length"] == 8192


# ---------------------------------------------------------------------------
# Modalities stamping
# ---------------------------------------------------------------------------

def test_managed_record_stamps_registry_modalities(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    # gpt-4o is a registry vision model — a stable stand-in for "registry hit".
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "gpt-4o"})
    assert load.status_code == 200

    record = load.json()["runtime"]
    assert "input.text" in record["modalities"]
    assert "input.image" in record["modalities"]
    assert "modalities_note" not in record


def test_managed_record_omits_modalities_on_registry_miss(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "totally-unknown-model-zzz"})
    assert load.status_code == 200

    record = load.json()["runtime"]
    assert "modalities" not in record  # absent knowledge stays absent
    assert "modalities_note" not in record


def test_managed_record_mlx_vision_unusable_removes_image_modality(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "gpt-4o"})
    assert load.status_code == 200
    runtime = next(iter(server_app._GATEWAY_LOADED_RUNTIMES.values()))
    runtime.llm._vision_usable = False  # runtime truth beats declared truth

    record = _loaded_records(client)[0]
    assert "input.image" not in record["modalities"]
    assert "input.text" in record["modalities"]
    assert record["modalities_note"] == "vision_unusable"


def test_sweep_rows_stamp_modalities_by_model_name(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {"provider": "ollama", "model": "gpt-4o", "resident": True, "loaded": True, "source": "provider_server"},
            {"provider": "ollama", "model": "no-such-model-xyz", "resident": True, "loaded": True, "source": "provider_server"},
        ],
    )

    records = _loaded_records(client)
    by_model = {r["model"]: r for r in records}

    assert "input.image" in by_model["gpt-4o"]["modalities"]
    assert "modalities" not in by_model["no-such-model-xyz"]


# ---------------------------------------------------------------------------
# Host identity stamping
# ---------------------------------------------------------------------------

def test_records_and_memory_snapshot_carry_host_identity(monkeypatch) -> None:
    server_app, client = _fresh_server(monkeypatch)
    identity = get_host_identity()

    load = client.post("/acore/models/load", json={"provider": "mlx", "model": "stub-model"})
    assert load.status_code == 200
    assert load.json()["runtime"]["host_id"] == identity["host_id"]
    assert load.json()["runtime"]["host_name"] == identity["host_name"]

    monkeypatch.setattr(
        server_app,
        "sweep_loaded_models",
        lambda timeout_s=2.0: [
            {"provider": "lmstudio", "model": "qwen/qwen3-4b", "resident": True, "loaded": True, "source": "provider_server"}
        ],
    )
    for record in _loaded_records(client):
        assert record["host_id"] == identity["host_id"]
        assert record["host_name"] == identity["host_name"]

    memory = client.get("/acore/memory")
    assert memory.status_code == 200
    assert memory.json()["host"] == identity
