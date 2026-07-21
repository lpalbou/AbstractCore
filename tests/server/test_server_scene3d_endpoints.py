import base64
import importlib.metadata

import pytest
from fastapi.testclient import TestClient

from abstractcore.server.app import app


class _FakeEntryPoint:
    def __init__(self, *, name: str, value: str, obj):
        self.name = name
        self.value = value
        self._obj = obj

    def load(self):
        return self._obj


class _EntryPoints:
    def __init__(self, eps):
        self._eps = list(eps)

    def select(self, *, group: str):
        if group == "abstractcore.capabilities_plugins":
            return list(self._eps)
        return []


def _make_fake_scene3d_plugin_ep(calls=None, *, backend_id="abstract3d:triposr", priority=10):
    def register(registry):
        class _Scene3D:
            def __init__(self):
                self.backend_id = backend_id

            def t23d(self, prompt: str, **kwargs):
                if calls is not None:
                    calls.append({"op": "t23d", "prompt": prompt, **kwargs})
                fmt = str(kwargs.get("format") or "glb").lower()
                return {
                    "data": f"{self.backend_id}:t23d:{fmt}".encode(),
                    "content_type": "model/gltf-binary" if fmt == "glb" else "model/obj",
                    "format": fmt,
                    "backend_id": self.backend_id,
                    "model_id": "stabilityai/TripoSR",
                    "metadata": {"model_id": "stabilityai/TripoSR"},
                }

            def i23d(self, image, *, prompt=None, **kwargs):
                if calls is not None:
                    calls.append({"op": "i23d", "image": image, "prompt": prompt, **kwargs})
                fmt = str(kwargs.get("format") or "glb").lower()
                return {
                    "data": f"{self.backend_id}:i23d:{fmt}".encode(),
                    "content_type": "model/gltf-binary",
                    "format": fmt,
                    "backend_id": self.backend_id,
                    "model_id": "stabilityai/TripoSR",
                    "metadata": {"model_id": "stabilityai/TripoSR"},
                }

        registry.register_scene3d_backend(
            backend_id=backend_id, factory=lambda _owner: _Scene3D(), priority=priority
        )

    return _FakeEntryPoint(name="fake", value="tests.fake_scene3d:register", obj=register)


def _make_fake_license_gated_scene3d_plugin_ep(*, marker_only: bool = False):
    def register(registry):
        class _LicenseRefusal(RuntimeError):
            # Mirrors abstract3d.errors.LicenseAcknowledgmentRequiredError's
            # stable machine-readable marker.
            error_class = "license_acknowledgment_required"

        class _Scene3D:
            backend_id = "abstract3d:hunyuan3d21-local"

            def t23d(self, prompt: str, **kwargs):
                _ = prompt, kwargs
                if marker_only:
                    # No message phrase at all: the typed marker alone must
                    # be enough for the 403 mapping.
                    raise _LicenseRefusal("Refused: set ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1.")
                raise RuntimeError(
                    "The Hunyuan3D-2.1 backend requires an explicit license acknowledgment: "
                    "opt in with ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1."
                )

        registry.register_scene3d_backend(
            backend_id="abstract3d:hunyuan3d21-local", factory=lambda _owner: _Scene3D(), priority=8
        )

    return _FakeEntryPoint(name="fake", value="tests.fake_gated_scene3d:register", obj=register)


@pytest.fixture()
def client():
    return TestClient(app)


def _reset_capability_core(monkeypatch):
    import abstractcore.server.audio_endpoints as audio_endpoints_module

    monkeypatch.setattr(audio_endpoints_module, "_CORE", None)


def test_scene3d_generations_returns_501_when_plugin_unavailable(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))
    _reset_capability_core(monkeypatch)

    resp = client.post("/v1/scene3d/generations", json={"prompt": "a teapot", "format": "glb"})
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert 'pip install "abstractcore[scene3d]"' in data["error"]["message"]


def test_scene3d_generations_t23d_happy_path(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep(calls)])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations",
        json={"prompt": "a glossy red cube", "format": "glb", "seed": 7, "device": "cpu"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("model/gltf-binary")
    assert resp.headers.get("x-abstractcore-backend-id") == "abstract3d:triposr"
    assert resp.headers.get("x-abstractcore-model") == "stabilityai/TripoSR"
    assert resp.headers.get("x-abstractcore-task") == "text_to_scene3d"
    assert resp.content == b"abstract3d:triposr:t23d:glb"
    assert calls[0]["op"] == "t23d"
    assert calls[0]["prompt"] == "a glossy red cube"
    assert calls[0]["seed"] == 7
    assert calls[0]["device"] == "cpu"


def test_scene3d_generations_i23d_decodes_image_b64(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep(calls)])
    )
    _reset_capability_core(monkeypatch)

    image_bytes = b"\x89PNG\r\n\x1a\nfakepng"
    resp = client.post(
        "/v1/scene3d/generations",
        json={
            "image_b64": base64.b64encode(image_bytes).decode(),
            "prompt": "make it a mesh",
            "task": "i23d",
            "format": "glb",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-abstractcore-task") == "image_to_scene3d"
    assert resp.content == b"abstract3d:triposr:i23d:glb"
    assert calls[0]["op"] == "i23d"
    assert calls[0]["image"] == image_bytes
    assert calls[0]["prompt"] == "make it a mesh"


def test_scene3d_generations_image_implies_i23d_without_task(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep(calls)])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations",
        json={"image_b64": base64.b64encode(b"img").decode(), "format": "glb"},
    )
    assert resp.status_code == 200
    assert calls[0]["op"] == "i23d"


def test_provider_scoped_scene3d_generations(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep(calls)])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/triposr/v1/scene3d/generations",
        json={"input": "a small owl statue", "format": "glb"},
    )
    assert resp.status_code == 200
    assert resp.content == b"abstract3d:triposr:t23d:glb"
    assert calls[0]["prompt"] == "a small owl statue"


def test_scene3d_generations_missing_prompt_and_image_is_422(client, monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep()])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post("/v1/scene3d/generations", json={"format": "glb"})
    assert resp.status_code == 422
    body = resp.json()
    message = body.get("detail") or (body.get("error") or {}).get("message") or ""
    assert "prompt" in message and "image_b64" in message


def test_scene3d_generations_unknown_field_is_422_with_supported_list(client, monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep()])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations", json={"prompt": "a cube", "texure_mode": "baked"}
    )
    assert resp.status_code == 422
    body = resp.json()
    message = body.get("detail") or (body.get("error") or {}).get("message") or ""
    assert "texure_mode" in message
    assert "texture_mode" in message  # supported list names the correct spelling


def test_scene3d_generations_bad_format_is_422(client, monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep()])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post("/v1/scene3d/generations", json={"prompt": "a cube", "format": "usdz"})
    assert resp.status_code == 422


def test_scene3d_generations_unknown_provider_is_501(client, monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep()])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations", json={"prompt": "a cube", "provider": "not-a-backend"}
    )
    assert resp.status_code == 501
    assert "not-a-backend" in resp.json()["error"]["message"]


def test_scene3d_generations_license_gate_maps_to_403(client, monkeypatch):
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_make_fake_license_gated_scene3d_plugin_ep()]),
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations", json={"prompt": "a cube", "provider": "hunyuan3d21"}
    )
    assert resp.status_code == 403
    assert "ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE" in resp.json()["error"]["message"]


def test_scene3d_generations_license_gate_typed_marker_alone_maps_to_403(client, monkeypatch):
    # The stable error_class marker must map to 403 even when the message
    # carries no recognizable phrase (message prose is only the fallback).
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_make_fake_license_gated_scene3d_plugin_ep(marker_only=True)]),
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations", json={"prompt": "a cube", "provider": "hunyuan3d21"}
    )
    assert resp.status_code == 403


def _make_strict_options_scene3d_plugin_ep():
    """Stub mirroring abstract3d's reject-unknown-options contract: its
    request-class errors are RuntimeError subclasses carrying `error_class`,
    invisible to abstractcore's exception taxonomy."""

    def register(registry):
        class _InvalidRequest(RuntimeError):
            error_class = "invalid_request"

        class _Scene3D:
            backend_id = "abstract3d:triposr"

            def t23d(self, prompt: str, **kwargs):
                supported = {"format", "device", "mc_resolution", "image_seed"}
                unknown = sorted(k for k in kwargs if k not in supported)
                if unknown:
                    raise _InvalidRequest(
                        f"abstract3d:triposr does not support these options: {', '.join(unknown)}."
                    )
                return {"data": b"glb-bytes", "content_type": "model/gltf-binary", "format": "glb"}

            i23d = None  # unused in these tests

        registry.register_scene3d_backend(
            backend_id="abstract3d:triposr", factory=lambda _owner: _Scene3D(), priority=10
        )

    return _FakeEntryPoint(name="fake", value="tests.fake_strict_scene3d:register", obj=register)


def test_scene3d_generations_backend_option_rejection_is_422_not_500(client, monkeypatch):
    # The real backends refuse unknown options with a RuntimeError subclass
    # carrying error_class="invalid_request" — the endpoint must map that to
    # 422, never 500 (adversarial finding: client errors surfaced as server
    # errors through _plugin_exception_status).
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_make_strict_options_scene3d_plugin_ep()]),
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations", json={"prompt": "a cube", "seed": 7}
    )
    assert resp.status_code == 422
    assert "seed" in resp.json()["error"]["message"]

    # And a clean request through the same strict stub still succeeds.
    resp = client.post("/v1/scene3d/generations", json={"prompt": "a cube", "mc_resolution": 256})
    assert resp.status_code == 200


def test_scene3d_generations_null_unknown_field_still_rejected(client, monkeypatch):
    # exclude_none must not let `"bogus": null` bypass the refuse-loudly
    # contract (adversarial finding).
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep()])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations", json={"prompt": "a cube", "bogus_option": None}
    )
    assert resp.status_code == 422
    assert "bogus_option" in resp.json()["error"]["message"]


def test_scene3d_generations_t23d_with_image_is_ambiguous_422(client, monkeypatch):
    # task=t23d ignores images; silently dropping the caller's image would be
    # a silent no-op on user input (adversarial finding).
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_scene3d_plugin_ep()])
    )
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations",
        json={
            "prompt": "a cube",
            "task": "t23d",
            "image_b64": base64.b64encode(b"img").decode(),
        },
    )
    assert resp.status_code == 422
    message = resp.json()["error"]["message"]
    assert "image_to_scene3d" in message


def test_scene3d_generations_unreadable_image_is_422(client, monkeypatch):
    # Valid base64 whose bytes are not an image raises PIL's
    # UnidentifiedImageError (an OSError) deep in the backend — the caller's
    # input, so 422, not 500 (adversarial finding).
    def register(registry):
        class _Scene3D:
            backend_id = "abstract3d:triposr"

            def i23d(self, image, **kwargs):
                from PIL import UnidentifiedImageError

                raise UnidentifiedImageError("cannot identify image file <_io.BytesIO>")

        registry.register_scene3d_backend(
            backend_id="abstract3d:triposr", factory=lambda _owner: _Scene3D(), priority=10
        )

    ep = _FakeEntryPoint(name="fake", value="tests.fake_unreadable_image:register", obj=register)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([ep]))
    _reset_capability_core(monkeypatch)

    resp = client.post(
        "/v1/scene3d/generations",
        json={"image_b64": base64.b64encode(b"not an image").decode()},
    )
    assert resp.status_code == 422
    assert "not a readable image" in resp.json()["error"]["message"]


def test_scene3d_generations_handlers_run_on_threadpool():
    # The handlers must be sync `def` (FastAPI threadpools them): an
    # `async def` around the synchronous multi-minute core.generate would
    # stall the whole event loop (adversarial finding; the music endpoint
    # carries this defect — deliberately not inherited).
    import asyncio

    from abstractcore.server.scene3d_endpoints import (
        provider_scene3d_generations,
        scene3d_generations,
    )

    assert not asyncio.iscoroutinefunction(scene3d_generations)
    assert not asyncio.iscoroutinefunction(provider_scene3d_generations)


@pytest.mark.parametrize("alias", ["hunyuan3d", "hunyuan3d21-local", "step1x-local", "trellis2"])
def test_scene3d_selector_aliases_resolve(alias):
    from abstractcore.capabilities.scene3d_selectors import resolve_scene3d_backend_id

    resolved = resolve_scene3d_backend_id(alias)
    assert resolved is not None and resolved.startswith("abstract3d:")
