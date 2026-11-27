from __future__ import annotations

import importlib
from typing import Any, Dict, List

from abstractvision.backends.base_backend import VisionBackend
from fastapi.testclient import TestClient
import pytest


class _EngineFreeBackend(VisionBackend):
    """An AbstractVision backend whose engine is faked: it only records loads and unloads."""

    def __init__(self) -> None:
        self.preloaded = 0
        self.unloaded = 0
        self.holds_weights = False

    def preload(self) -> None:
        self.preloaded += 1
        self.holds_weights = True

    def unload(self) -> None:
        self.unloaded += 1
        self.holds_weights = False

    def has_loaded_model(self) -> bool:
        return self.holds_weights

    def generate_image(self, request):  # pragma: no cover - never run here
        raise NotImplementedError

    def edit_image(self, request):  # pragma: no cover
        raise NotImplementedError

    def generate_angles(self, request):  # pragma: no cover
        raise NotImplementedError

    def generate_video(self, request):  # pragma: no cover
        raise NotImplementedError

    def image_to_video(self, request):  # pragma: no cover
        raise NotImplementedError

    def get_capabilities(self):  # pragma: no cover
        return None


@pytest.fixture(autouse=True)
def _host_config_isolation(monkeypatch):
    """Keep this suite hermetic: the vision lanes are config-first (dm#177),
    so a real output.image/video route on the HOST machine would legitimately
    steer backend selection under these tests. Pin the not-configured shape;
    the config-wins precedence has its own suite (test_vision_config_precedence)."""
    from abstractcore.server import vision_endpoints as _ve

    monkeypatch.setattr(_ve, "_vision_route_defaults", lambda modality="image", task=None: {})
    # These are process globals and /acore/models/loaded merges their contents
    # with text runtimes. Fresh dictionaries keep a fake vision model from
    # leaking into the next test/module, while monkeypatch restores prior state.
    monkeypatch.setattr(_ve, "_BACKEND_CACHE", {})
    monkeypatch.setattr(_ve, "_RESIDENCY_RECORDS", {})
    _ve._VISION_ROUTE_WARNED.clear()




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


def test_acore_models_loaded_lists_shared_vision_cache_once_when_task_omitted(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()

    calls: List[Dict[str, Any]] = []
    runtime = {
        "runtime_id": "mlx-gen/AbstractFramework/qwen-image-edit-2511-4bit",
        "task": "image_generation",
        "tasks": ["image_generation", "text_to_image", "image_to_image"],
        "provider": "mlx-gen",
        "model": "AbstractFramework/qwen-image-edit-2511-4bit",
        "loaded": True,
        "state": "loaded",
    }

    def fake_list(filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        calls.append(dict(filters or {}))
        return [dict(runtime)]

    def fail_audio_residency(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("skip audio")

    monkeypatch.setattr(vision_endpoints, "list_server_vision_loaded_models", fake_list)
    monkeypatch.setattr(server_app, "_capability_residency_core_for_request", fail_audio_residency)

    client = TestClient(server_app.app)
    loaded = client.get("/acore/models/loaded")

    assert loaded.status_code == 200
    assert [item["runtime_id"] for item in loaded.json()["models"]] == [runtime["runtime_id"]]
    assert calls == [{"provider": "", "model": ""}]


def test_acore_models_routes_video_residency_through_server_vision_cache(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()

    calls: List[tuple[str, Dict[str, Any]]] = []
    runtime = {
        "runtime_id": "mlx-gen/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "load_id": "mlx-gen/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "task": "video_generation",
        "tasks": ["video_generation", "text_to_video", "image_to_video"],
        "provider": "mlx-gen",
        "backend_kind": "mlx-gen",
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "state": "loaded",
        "loaded": True,
    }

    def fake_load(payload: Dict[str, Any], *, api_key: str | None = None) -> Dict[str, Any]:
        calls.append(("load", dict(payload) | {"api_key": api_key}))
        return dict(runtime, loaded_new=True)

    def fake_list(filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        calls.append(("list", dict(filters or {})))
        return [dict(runtime)]

    monkeypatch.setattr(vision_endpoints, "load_server_vision_loaded_model", fake_load)
    monkeypatch.setattr(vision_endpoints, "list_server_vision_loaded_models", fake_list)

    client = TestClient(server_app.app)
    load = client.post(
        "/acore/models/load",
        json={
            "task": "t2v",
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        },
    )
    assert load.status_code == 200
    assert load.json()["runtime"]["task"] == "video_generation"
    assert calls[0][0] == "load"
    assert calls[0][1]["task"] == "text_to_video"

    loaded = client.get("/acore/models/loaded", params={"task": "i2v", "provider": "mlx-gen"})
    assert loaded.status_code == 200
    assert loaded.json()["data"][0]["tasks"] == ["video_generation", "text_to_video", "image_to_video"]
    assert calls[1] == ("list", {"provider": "mlx-gen", "model": "", "task": "image_to_video"})


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


class _FakeScene3dResidency:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def load_resident_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("load", dict(payload)))
        return {
            "task": "text_to_scene3d",
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "state": "loaded",
            "loaded": True,
        }

    def list_loaded_models(self, filters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        self.calls.append(("list", dict(filters or {})))
        return [{"task": "scene3d_generation", "provider": "triposr", "model": "stabilityai/TripoSR", "state": "loaded", "loaded": True}]

    def unload_resident_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(("unload", dict(payload)))
        return {
            "task": "scene3d_generation",
            "provider": payload.get("provider"),
            "model": payload.get("model"),
            "state": "unloaded",
            "loaded": False,
            "unloaded": True,
        }


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


def test_acore_models_routes_scene3d_residency_through_shared_core(monkeypatch) -> None:
    server_app = importlib.import_module("abstractcore.server.app")
    server_app._GATEWAY_LOADED_RUNTIMES.clear()
    server_app._GATEWAY_RUNTIME_IDS.clear()

    scene3d = _FakeScene3dResidency()
    fake_core = type("_FakeCore", (), {"scene3d": scene3d})()
    monkeypatch.setattr(server_app, "_capability_residency_core_for_request", lambda task, http_request: fake_core)

    client = TestClient(server_app.app)
    load = client.post(
        "/acore/models/load",
        json={"task": "t23d", "provider": "triposr", "model": "stabilityai/TripoSR"},
    )
    assert load.status_code == 200
    assert load.json()["runtime"]["state"] == "loaded"
    assert scene3d.calls[0][0] == "load"
    assert scene3d.calls[0][1]["task"] == "text_to_scene3d"

    loaded = client.get("/acore/models/loaded", params={"task": "t23d", "provider": "triposr"})
    assert loaded.status_code == 200
    assert loaded.json()["data"][0]["model"] == "stabilityai/TripoSR"

    unload = client.post(
        "/acore/models/unload",
        json={"task": "t23d", "provider": "triposr", "model": "stabilityai/TripoSR"},
    )
    assert unload.status_code == 200
    assert unload.json()["runtime"]["state"] == "unloaded"
    assert scene3d.calls[-1][0] == "unload"


def test_server_vision_residency_helpers_share_backend_cache(monkeypatch) -> None:
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    vision_endpoints._BACKEND_CACHE.clear()
    vision_endpoints._RESIDENCY_RECORDS.clear()

    backend = _EngineFreeBackend()
    key = ("diffusers", "runwayml/stable-diffusion-v1-5", "auto", None, False, True)

    def fake_resolve(
        request_model: str | None,
        *,
        modality: str = "image",
        task: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        _ = request_model, modality, task, base_url, api_key
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


def test_server_vision_residency_accepts_video_task_filters(monkeypatch) -> None:
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    vision_endpoints._BACKEND_CACHE.clear()
    vision_endpoints._RESIDENCY_RECORDS.clear()

    backend = _EngineFreeBackend()
    key = ("mlx-gen", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", None, None, False)

    def fake_resolve(
        request_model: str | None,
        *,
        modality: str = "image",
        task: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        _ = request_model, modality, task, base_url, api_key
        cached, call_lock = vision_endpoints._get_or_create_cached_backend(key, lambda: backend)
        return cached, call_lock, RuntimeError, object, object

    monkeypatch.setattr(vision_endpoints, "_resolve_backend", fake_resolve)

    loaded = vision_endpoints.load_server_vision_loaded_model(
        {
            "task": "text_to_video",
            "provider": "mlx-gen",
            "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        }
    )
    listed_t2v = vision_endpoints.list_server_vision_loaded_models({"task": "t2v"})
    listed_i2v = vision_endpoints.list_server_vision_loaded_models({"task": "image_to_video"})
    listed_t2i = vision_endpoints.list_server_vision_loaded_models({"task": "text_to_image"})

    assert loaded["task"] == "video_generation"
    assert loaded["tasks"] == ["video_generation", "text_to_video", "image_to_video"]
    assert len(listed_t2v) == 1
    assert len(listed_i2v) == 1
    assert listed_t2i == []


def test_server_vision_residency_eviction_unloads_request_loaded_models_and_keeps_pinned_ones(monkeypatch) -> None:
    vision_endpoints = importlib.import_module("abstractcore.server.vision_endpoints")
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND_CACHE_MAX", "1")
    backends: Dict[str, _EngineFreeBackend] = {}

    def key_of(model: str):
        return ("diffusers", model, "auto", None, False, True)

    def fake_resolve(
        request_model: str | None,
        *,
        modality: str = "image",
        task: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        _ = modality, task, base_url, api_key
        model = str(request_model or "default").split("/", 1)[-1]
        backend = backends.setdefault(model, _EngineFreeBackend())
        cached, call_lock = vision_endpoints._get_or_create_cached_backend(key_of(model), lambda: backend)
        return cached, call_lock, RuntimeError, object, object

    monkeypatch.setattr(vision_endpoints, "_resolve_backend", fake_resolve)

    # A model a generation request loaded (no pin): the cache may evict it.
    request_loaded = backends.setdefault("model-zero", _EngineFreeBackend())
    request_loaded.holds_weights = True
    vision_endpoints._keep_backend(key_of("model-zero"), request_loaded, task="image_generation")

    first = vision_endpoints.load_server_vision_loaded_model(
        {"task": "image_generation", "provider": "diffusers", "model": "model-one"}
    )
    assert request_loaded.unloaded == 1
    assert key_of("model-zero") not in vision_endpoints._BACKEND_CACHE

    # Explicit loads are pinned: a second one goes past the cache size instead of evicting the first.
    second = vision_endpoints.load_server_vision_loaded_model(
        {"task": "image_generation", "provider": "diffusers", "model": "model-two"}
    )
    listed_ids = {str(item["load_id"]) for item in vision_endpoints.list_server_vision_loaded_models({})}
    assert listed_ids == {str(first["load_id"]), str(second["load_id"])}
    assert backends["model-one"].unloaded == 0
    assert backends["model-two"].unloaded == 0
