"""Camera extension endpoints (/v1/camera/*).

Two lanes:
- A FAKE capability plugin exercising the route contracts hermetically
  (501-when-absent, error mapping, response shapes) — runs everywhere.
- The REAL abstractcamera plugin over its built-in simulator
  (ABSTRACTCAMERA_FAKE=1) — end-to-end through the actual capability
  registry; skips when abstractcamera is not installed.

Owning seat: camera drafted this module per core's ruling (agora commons
c3168, 2026-07-19); abstractcore owns review and merge.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from abstractcore.server.app import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def fresh_capability_core(monkeypatch):
    """Reset the shared server capability host so each test's registry
    reflects the entry points/monkeypatches active in THAT test."""
    from abstractcore.server import audio_endpoints

    monkeypatch.setattr(audio_endpoints, "_CORE", None)
    yield
    audio_endpoints._CORE = None


class _NoCameraEntryPoints:
    def select(self, *, group: str):
        return []


def test_camera_handlers_are_sync_def():
    """P0 regression (adversarial finding 2026-07-21): these handlers call
    BLOCKING capability ops (video capture holds up to 600s) — as
    `async def` they ran ON the event loop and serialized the whole server
    behind one camera call. Sync-def handlers run in FastAPI's threadpool
    (the audio_speech precedent). This pin fails if anyone "modernizes"
    a handler back to async without moving the blocking call off-loop."""
    import inspect

    from abstractcore.server import camera_endpoints

    for route in camera_endpoints.router.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            assert not inspect.iscoroutinefunction(endpoint), (
                f"{endpoint.__name__} must stay sync-def: it calls blocking "
                "camera ops and would wedge the event loop as async"
            )


def test_camera_routes_answer_501_when_no_plugin(client, fresh_capability_core, monkeypatch):
    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", lambda: _NoCameraEntryPoints())
    response = client.get("/v1/camera/devices")
    assert response.status_code == 501
    message = response.json()["error"]["message"]
    assert "camera" in message.lower()
    assert 'pip install "abstractcamera"' in message

    response = client.post("/v1/camera/photo", json={})
    assert response.status_code == 501


class _FakeCamera:
    backend_id = "fake:camera"

    def __init__(self):
        self.calls = []

    def list_cameras(self):
        self.calls.append("list")
        return [{"id": "fake:0", "name": "Fake Cam", "transport": "fake", "connected": False}]

    def open(self, camera_id=None, **kwargs):
        self.calls.append(("open", camera_id))
        return {"camera": "fake_cam", "status": {"connected": True}}

    def close(self, camera=None, **kwargs):
        return {"camera": camera or "fake_cam", "connected": False}

    def close_all(self):
        return {"connected": False}

    def status(self, camera=None):
        return {"active": "fake_cam", "cameras": {"fake_cam": {"connected": True}}}

    def capture_photo(self, camera=None, *, timeout_s=None, include_bytes=False, **kwargs):
        out = {"kind": "photo", "path": "/tmp/fake.jpg", "on_device": False}
        if include_bytes:
            out["data_b64"] = base64.b64encode(b"jpegbytes").decode("ascii")
            out["content_type"] = "image/jpeg"
        return out

    def capture_video(self, duration_s, camera=None, *, timeout_s=None, include_bytes=False, **kwargs):
        # Mirrors a body that keeps the movie on its own card.
        return {"kind": "video", "path": None, "delivered": False, "duration_s": duration_s, "note": "on camera"}

    def stop_recording(self, camera=None, *, timeout_s=None, **kwargs):
        raise RuntimeError("No video recording is running on this camera.")

    def preview_frame(self, camera=None, **kwargs):
        return b"\xff\xd8jpeg"

    def start_detection(self, camera=None, *, target="motion", action="photo", sensitivity=None, **kwargs):
        return {"detection_mode": "auto" if action != "monitor" else "monitor", "detection_target": target}

    def stop_detection(self, camera=None, **kwargs):
        return {"detection_mode": "off"}

    def detection_events(self, camera=None, *, since_id=0, kinds=None, limit=100, include_thumbnails=False):
        return {"events": [], "last_id": since_id, "truncated": False}


class _FakeEntryPoint:
    name = "fakecamera"
    value = "fake:register"

    def __init__(self, camera):
        self._camera = camera

    def load(self):
        camera = self._camera

        def register(registry):
            registry.register_backend(
                capability="camera",
                backend_id=camera.backend_id,
                factory=lambda owner: camera,
                priority=0,
            )

        return register


class _OnlyFakeCameraEntryPoints:
    def __init__(self, camera):
        self._ep = _FakeEntryPoint(camera)

    def select(self, *, group: str):
        return [self._ep] if group == "abstractcore.capabilities_plugins" else []


@pytest.fixture()
def fake_camera(fresh_capability_core, monkeypatch):
    import importlib.metadata as md

    camera = _FakeCamera()
    monkeypatch.setattr(md, "entry_points", lambda: _OnlyFakeCameraEntryPoints(camera))
    return camera


def test_camera_workflow_routes_over_fake_plugin(client, fake_camera):
    assert client.get("/v1/camera/devices").json()["devices"][0]["id"] == "fake:0"
    assert client.post("/v1/camera/open", json={}).json()["camera"] == "fake_cam"
    assert client.get("/v1/camera/status").json()["active"] == "fake_cam"

    preview = client.get("/v1/camera/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"

    photo = client.post("/v1/camera/photo", json={"response_format": "b64_json"})
    assert photo.status_code == 200
    body = photo.json()
    assert base64.b64decode(body["data"][0]["b64_json"]) == b"jpegbytes"
    assert body["camera_result"]["path"] == "/tmp/fake.jpg"
    assert "data_b64" not in body["camera_result"], "content must not ride twice"

    binary = client.post("/v1/camera/photo", json={"response_format": "binary"})
    assert binary.status_code == 200
    assert binary.content == b"jpegbytes"
    assert binary.headers["x-abstractcore-camera-path"] == "/tmp/fake.jpg"

    video = client.post("/v1/camera/video", json={"duration_s": 2.0})
    assert video.status_code == 200
    video_body = video.json()
    assert video_body["camera_result"]["delivered"] is False
    assert video_body["data"] == []

    assert client.post("/v1/camera/detection", json={"action": "monitor"}).json()["detection_mode"] == "monitor"
    assert client.post("/v1/camera/detection/stop", json={}).json()["detection_mode"] == "off"
    assert client.get("/v1/camera/events", params={"since_id": 0}).json()["events"] == []
    assert client.post("/v1/camera/close", json={}).json()["connected"] is False


def test_camera_state_refusals_map_to_409(client, fake_camera):
    response = client.post("/v1/camera/recording/stop", json={})
    assert response.status_code == 409
    assert "No video recording" in response.json()["error"]["message"]


def test_camera_photo_rejects_unknown_response_format(client, fake_camera):
    response = client.post("/v1/camera/photo", json={"response_format": "hologram"})
    assert response.status_code == 422


HAS_ABSTRACTCAMERA = importlib.util.find_spec("abstractcamera") is not None


@pytest.mark.skipif(not HAS_ABSTRACTCAMERA, reason="abstractcamera is not installed")
def test_camera_end_to_end_over_real_plugin_simulator(client, fresh_capability_core, monkeypatch):
    """The REAL abstractcamera plugin through the REAL entry-point scan,
    on its built-in simulator — no hardware."""
    monkeypatch.setenv("ABSTRACTCAMERA_FAKE", "1")
    capture_root = tempfile.mkdtemp(prefix="core_camera_route_")
    monkeypatch.setenv("ABSTRACTCAMERA_CAPTURE_ROOT", capture_root)
    from abstractcamera.service import reset_shared_service

    reset_shared_service()
    try:
        opened = client.post("/v1/camera/open", json={})
        assert opened.status_code == 200, opened.text
        uid = opened.json()["camera"]
        assert uid

        photo = client.post("/v1/camera/photo", json={"camera": uid, "response_format": "none"})
        assert photo.status_code == 200, photo.text
        result = photo.json()["camera_result"]
        assert result["path"] and result["path"].startswith(capture_root)
        assert os.path.exists(result["path"])

        events = client.get("/v1/camera/events", params={"since_id": 0})
        assert any(e["kind"] == "photo" for e in events.json()["events"])

        closed = client.post("/v1/camera/close", json={"camera": uid})
        assert closed.status_code == 200
    finally:
        reset_shared_service()
