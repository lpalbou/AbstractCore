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


def _make_fake_music_plugin_ep():
    def register(registry):
        class _Music:
            backend_id = "fake-music"

            def t2m(self, prompt: str, **kwargs):
                _ = prompt, kwargs
                return b"wav-bytes"

        registry.register_music_backend(backend_id="fake-music", factory=lambda _owner: _Music(), priority=0)

    return _FakeEntryPoint(name="fake", value="tests.fake_music:register", obj=register)


@pytest.fixture()
def client():
    return TestClient(app)


def _reset_audio_core(monkeypatch):
    import abstractcore.server.audio_endpoints as audio_endpoints_module

    monkeypatch.setattr(audio_endpoints_module, "_CORE", None)


def test_audio_music_returns_501_when_plugin_unavailable(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))
    _reset_audio_core(monkeypatch)

    resp = client.post("/v1/audio/music", json={"prompt": "hello", "format": "wav"})
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "pip install abstractmusic" in data["error"]["message"]


def test_audio_music_happy_path_with_stubbed_plugin(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_music_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp = client.post("/v1/audio/music", json={"prompt": "hello", "format": "wav", "duration_s": 1})
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/wav")
    assert resp.content == b"wav-bytes"

