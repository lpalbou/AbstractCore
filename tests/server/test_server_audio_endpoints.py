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


@pytest.fixture(autouse=True)
def _allow_audio_endpoint_tests_without_server_auth(monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")


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


def test_audio_endpoints_accept_local_abstractvoice_model_alias(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_voice_audio_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp_tts = client.post(
        "/v1/audio/speech",
        json={"model": "local/abstractvoice", "input": "hello", "format": "wav"},
    )
    assert resp_tts.status_code == 200
    assert resp_tts.headers.get("content-type", "").startswith("audio/wav")
    assert resp_tts.content == b"wav-bytes"

    files = {"file": ("audio.wav", b"abc", "audio/wav")}
    resp_stt = client.post(
        "/v1/audio/transcriptions",
        files=files,
        data={"model": "local/abstractvoice", "language": "en"},
    )
    assert resp_stt.status_code == 200
    assert resp_stt.json() == {"text": "transcript"}


def test_audio_speech_routes_to_openai_when_model_is_supplied(client, monkeypatch):
    from abstractcore.providers.openai_provider import OpenAIProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_speech(self, input_text: str, **kwargs):
        captured["speech"] = {"input_text": input_text, **kwargs}
        return b"mp3-bytes", "audio/mpeg"

    monkeypatch.setattr(OpenAIProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAIProvider, "synthesize_speech", fake_speech)

    resp = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer sk-provider-key"},
        json={
            "model": "openai/gpt-4o-mini-tts",
            "input": "hello",
            "voice": "coral",
            "response_format": "mp3",
            "instructions": "Speak clearly.",
        },
    )

    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/mpeg")
    assert resp.content == b"mp3-bytes"
    assert captured["init"] == {"model": "gpt-4o-mini-tts", "api_key": "sk-provider-key"}
    assert captured["speech"]["input_text"] == "hello"
    assert captured["speech"]["voice"] == "coral"
    assert captured["speech"]["response_format"] == "mp3"
    assert captured["speech"]["instructions"] == "Speak clearly."


def test_audio_transcriptions_routes_to_openai_when_model_is_supplied(client, monkeypatch):
    from abstractcore.providers.openai_provider import OpenAIProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_transcribe(self, audio: bytes, **kwargs):
        captured["transcribe"] = {"audio": audio, **kwargs}
        return b'{"text":"remote transcript"}', "application/json"

    monkeypatch.setattr(OpenAIProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAIProvider, "transcribe_audio", fake_transcribe)

    resp = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-provider-key"},
        files={"file": ("speech.wav", b"abc", "audio/wav")},
        data={"model": "openai/gpt-4o-mini-transcribe", "language": "en", "response_format": "json"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "remote transcript"}
    assert captured["init"] == {"model": "gpt-4o-mini-transcribe", "api_key": "sk-provider-key"}
    assert captured["transcribe"]["audio"] == b"abc"
    assert captured["transcribe"]["filename"] == "speech.wav"
    assert captured["transcribe"]["content_type"] == "audio/wav"
    assert captured["transcribe"]["language"] == "en"
    assert captured["transcribe"]["response_format"] == "json"


def test_audio_transcriptions_routes_to_openrouter_with_base64_provider_method(client, monkeypatch):
    from abstractcore.providers.openrouter_provider import OpenRouterProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_transcribe(self, audio: bytes, **kwargs):
        captured["transcribe"] = {"audio": audio, **kwargs}
        return b'{"text":"openrouter transcript","usage":{"total_tokens":1}}', "application/json"

    monkeypatch.setattr(OpenRouterProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenRouterProvider, "transcribe_audio", fake_transcribe)

    resp = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-provider-key"},
        files={"file": ("speech.mp3", b"abc", "audio/mpeg")},
        data={"model": "openrouter/openai/whisper-large-v3", "language": "en"},
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "openrouter transcript"
    assert captured["init"] == {
        "model": "openai/whisper-large-v3",
        "api_key": "sk-provider-key",
        "validate_model": False,
    }
    assert captured["transcribe"]["audio"] == b"abc"
    assert captured["transcribe"]["filename"] == "speech.mp3"
    assert captured["transcribe"]["content_type"] == "audio/mpeg"
