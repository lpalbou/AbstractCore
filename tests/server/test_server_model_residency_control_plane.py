from __future__ import annotations

import importlib
from typing import Any, Dict, List

from fastapi.testclient import TestClient


def test_acore_models_routes_image_residency_through_server_vision_cache(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()

    calls: List[tuple[str, Dict[str, Any]]] = []
    runtime = {
        "runtime_id": "diffusers/runwayml/stable-diffusion-v1-5",
        "load_id": "diffusers/runwayml/stable-diffusion-v1-5",
        "task": "image_generation",
        "provider": "huggingface",
        "backend_kind": "diffusers",
        "model": "runwayml/stable-diffusion-v1-5",
        "state": "loaded",
        "loaded": True,
    }

    def fake_load(payload: Dict[str, Any], *, api_key: str | None = None) -> Dict[str, Any]:
        calls.append(("load", dict(payload) | {"api_key": api_key}))
        return dict(runtime, loaded_new=True)

    def fake_list(filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        calls.append(("list", dict(filters or {})))
        return [dict(runtime)]

    def fake_unload(payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(("unload", dict(payload)))
        return dict(runtime, state="unloaded", loaded=False, unloaded=True)

    monkeypatch.setattr(vision_endpoints, "load_server_vision_loaded_model", fake_load)
    monkeypatch.setattr(vision_endpoints, "list_server_vision_loaded_models", fake_list)
    monkeypatch.setattr(vision_endpoints, "unload_server_vision_loaded_model", fake_unload)

    client = TestClient(server_app.app)
    load = client.post(
        "/acore/models/load",
        json={
            "task": "image_generation",
            "provider": "diffusers",
            "model": "runwayml/stable-diffusion-v1-5",
        },
    )
    assert load.status_code == 200
    assert load.json()["runtime"]["backend_kind"] == "diffusers"
    assert calls[0][0] == "load"
    assert calls[0][1]["task"] == "image_generation"

    loaded = client.get("/acore/models/loaded", params={"task": "image_generation", "provider": "diffusers"})
    assert loaded.status_code == 200
    assert loaded.json()["data"][0]["runtime_id"] == "diffusers/runwayml/stable-diffusion-v1-5"
    assert calls[1] == ("list", {"provider": "diffusers", "model": "", "task": "image_generation"})

    unload = client.post("/acore/models/unload", json={"runtime_id": runtime["runtime_id"]})
    assert unload.status_code == 200
    assert unload.json()["runtime"]["state"] == "unloaded"
    assert calls[-1][0] == "unload"


class _FakeVoiceResidency:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def load_resident_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("load", dict(payload)))
        return {
            "task": "tts",
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "state": "loaded",
            "loaded": True,
        }

    def list_loaded_models(self, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        self.calls.append(("list", dict(filters or {})))
        return [{"task": "tts", "provider": "cloned", "model": "omnivoice", "state": "loaded", "loaded": True}]

    def unload_resident_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("unload", dict(payload)))
        return {
            "task": "tts",
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "state": "unloaded",
            "loaded": False,
            "unloaded": True,
        }


class _FakeAudioResidency:
    def list_loaded_models(self, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        _ = filters
        return []


def test_acore_models_routes_tts_residency_through_shared_audio_core(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    audio_endpoints = importlib.import_module("abstractcore.server.audio_endpoints")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()

    voice = _FakeVoiceResidency()
    fake_core = type("_FakeCore", (), {"voice": voice, "audio": _FakeAudioResidency()})()
    monkeypatch.setattr(audio_endpoints, "_get_capability_core", lambda: fake_core)

    client = TestClient(server_app.app)
    load = client.post(
        "/acore/models/load",
        json={"task": "tts", "provider": "cloned", "model": "omnivoice", "options": {"voice": "clone-1"}},
    )
    assert load.status_code == 200
    assert load.json()["runtime"]["state"] == "loaded"
    assert load.json()["loaded_new"] is False
    assert voice.calls[0][1]["options"] == {"voice": "clone-1"}

    loaded = client.get("/acore/models/loaded", params={"task": "tts", "provider": "cloned"})
    assert loaded.status_code == 200
    assert loaded.json()["data"][0]["model"] == "omnivoice"

    unload = client.post("/acore/models/unload", json={"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert unload.status_code == 200
    assert unload.json()["runtime"]["state"] == "unloaded"
    assert voice.calls[-1][0] == "unload"


def test_acore_models_tts_loaded_new_uses_event_signal_not_resident_state(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    audio_endpoints = importlib.import_module("abstractcore.server.audio_endpoints")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()

    class _ExplicitVoiceResidency:
        def __init__(self, runtime: Dict[str, Any]) -> None:
            self.runtime = runtime

        def load_resident_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            _ = payload
            return dict(self.runtime)

    voice = _ExplicitVoiceResidency(
        {
            "task": "tts",
            "provider": "cloned",
            "model": "omnivoice",
            "state": "loaded",
            "loaded": True,
            "details": {"engine_cached": False},
        }
    )
    fake_core = type("_FakeCore", (), {"voice": voice, "audio": _FakeAudioResidency()})()
    monkeypatch.setattr(audio_endpoints, "_get_capability_core", lambda: fake_core)

    client = TestClient(server_app.app)
    loaded = client.post("/acore/models/load", json={"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert loaded.status_code == 200
    assert loaded.json()["loaded_new"] is True

    voice.runtime = {
        "task": "tts",
        "provider": "cloned",
        "model": "omnivoice",
        "state": "loaded",
        "loaded": True,
        "loaded_new": False,
    }
    already_loaded = client.post("/acore/models/load", json={"task": "tts", "provider": "cloned", "model": "omnivoice"})
    assert already_loaded.status_code == 200
    assert already_loaded.json()["loaded_new"] is False


def test_server_vision_residency_helpers_share_backend_cache(monkeypatch) -> None:
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    vision_endpoints._BACKEND_CACHE.clear()
    vision_endpoints._RESIDENCY_RECORDS.clear()

    class _FakeVisionBackend:
        def __init__(self) -> None:
            self.preloaded = 0
            self.unloaded = 0

        def preload(self) -> None:
            self.preloaded += 1

        def unload(self) -> None:
            self.unloaded += 1

    backend = _FakeVisionBackend()
    key = ("diffusers", "runwayml/stable-diffusion-v1-5", "auto", None, False, True)

    def fake_resolve(request_model: str | None, *, base_url: str | None = None, api_key: str | None = None):
        _ = request_model, base_url, api_key
        cached, call_lock = vision_endpoints._get_or_create_cached_backend(key, lambda: backend)
        return cached, call_lock, RuntimeError, object, object

    monkeypatch.setattr(vision_endpoints, "_resolve_backend", fake_resolve)

    loaded = vision_endpoints.load_server_vision_loaded_model(
        {
            "task": "image_generation",
            "provider": "diffusers",
            "model": "runwayml/stable-diffusion-v1-5",
        }
    )
    assert loaded["state"] == "loaded"
    assert loaded["runtime_id"] == "diffusers/runwayml/stable-diffusion-v1-5"
    assert backend.preloaded == 1

    listed = vision_endpoints.list_server_vision_loaded_models({"provider": "diffusers"})
    assert len(listed) == 1
    assert listed[0]["loaded"] is True

    unloaded = vision_endpoints.unload_server_vision_loaded_model({"runtime_id": loaded["runtime_id"]})
    assert unloaded["state"] == "unloaded"
    assert backend.unloaded == 1
    assert vision_endpoints.list_server_vision_loaded_models({}) == []


def test_server_vision_residency_eviction_clears_records_and_unloads(monkeypatch) -> None:
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    vision_endpoints._BACKEND_CACHE.clear()
    vision_endpoints._RESIDENCY_RECORDS.clear()
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND_CACHE_MAX", "1")

    class _FakeVisionBackend:
        def __init__(self) -> None:
            self.unloaded = 0

        def unload(self) -> None:
            self.unloaded += 1

    backends: Dict[str, _FakeVisionBackend] = {}
    created: List[_FakeVisionBackend] = []

    def fake_resolve(request_model: str | None, *, base_url: str | None = None, api_key: str | None = None):
        _ = base_url, api_key
        model_key = str(request_model or "default")
        backend = backends.get(model_key)
        if backend is None:
            backend = _FakeVisionBackend()
            backends[model_key] = backend
            created.append(backend)
        key = ("diffusers", model_key, "auto", None, False, True)
        cached, call_lock = vision_endpoints._get_or_create_cached_backend(key, lambda: backend)
        return cached, call_lock, RuntimeError, object, object

    monkeypatch.setattr(vision_endpoints, "_resolve_backend", fake_resolve)

    first = vision_endpoints.load_server_vision_loaded_model(
        {"task": "image_generation", "provider": "diffusers", "model": "model-one"}
    )
    second = vision_endpoints.load_server_vision_loaded_model(
        {"task": "image_generation", "provider": "diffusers", "model": "model-two"}
    )

    listed_ids = {
        str(item.get("load_id") or item.get("runtime_id") or "")
        for item in vision_endpoints.list_server_vision_loaded_models({})
    }
    assert str(first["load_id"]) not in listed_ids
    assert str(second["load_id"]) in listed_ids
    assert created[0].unloaded == 1
    assert created[1].unloaded == 0
