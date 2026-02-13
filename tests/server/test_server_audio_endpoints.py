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


def _make_fake_voice_audio_plugin_ep():
    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def tts(self, text: str, **kwargs):
                _ = text, kwargs
                return b"wav-bytes"

            def stt(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

        class _Audio:
            backend_id = "fake-audio"

            def transcribe(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

        registry.register_voice_backend(backend_id="fake-voice", factory=lambda _owner: _Voice(), priority=0)
        registry.register_audio_backend(backend_id="fake-audio", factory=lambda _owner: _Audio(), priority=0)

    return _FakeEntryPoint(name="fake", value="tests.fake_voice_audio:register", obj=register)


@pytest.fixture()
def client():
    return TestClient(app)


def _reset_audio_core(monkeypatch):
    import abstractcore.server.audio_endpoints as audio_endpoints_module

    monkeypatch.setattr(audio_endpoints_module, "_CORE", None)


def test_audio_speech_returns_501_when_plugin_unavailable(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))
    _reset_audio_core(monkeypatch)

    resp = client.post("/v1/audio/speech", json={"input": "hello", "format": "wav"})
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "pip install abstractvoice" in data["error"]["message"]


def test_audio_transcriptions_returns_501_when_plugin_unavailable(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))
    _reset_audio_core(monkeypatch)

    files = {"file": ("audio.wav", b"abc", "audio/wav")}
    resp = client.post("/v1/audio/transcriptions", files=files)
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert "pip install abstractvoice" in data["error"]["message"]


def test_audio_endpoints_happy_path_with_stubbed_plugin(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_voice_audio_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp_tr = client.post("/v1/audio/translations", files={"file": ("audio.wav", b"abc", "audio/wav")})
    assert resp_tr.status_code == 501
    assert "audio/translations" in resp_tr.json()["error"]["message"]

    resp_tts = client.post("/v1/audio/speech", json={"input": "hello", "format": "wav"})
    assert resp_tts.status_code == 200
    assert resp_tts.headers.get("content-type", "").startswith("audio/wav")
    assert resp_tts.content == b"wav-bytes"

    files = {"file": ("audio.wav", b"abc", "audio/wav")}
    resp_stt = client.post("/v1/audio/transcriptions", files=files, data={"language": "en"})
    assert resp_stt.status_code == 200
    assert resp_stt.json() == {"text": "transcript"}

