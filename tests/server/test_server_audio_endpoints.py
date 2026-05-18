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
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
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
    assert 'pip install "abstractcore[voice]"' in data["error"]["message"]


def test_audio_transcriptions_returns_501_when_plugin_unavailable(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))
    _reset_audio_core(monkeypatch)

    files = {"file": ("audio.wav", b"abc", "audio/wav")}
    resp = client.post("/v1/audio/transcriptions", files=files)
    assert resp.status_code == 501
    data = resp.json()
    assert "error" in data
    assert 'pip install "abstractcore[voice]"' in data["error"]["message"]


def test_audio_endpoints_happy_path_with_stubbed_plugin(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_voice_audio_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp_tr = client.post("/v1/audio/translations", files={"file": ("audio.wav", b"abc", "audio/wav")})
    assert resp_tr.status_code == 501
    assert "audio/translations" in resp_tr.json()["error"]["message"]

    resp_tts = client.post("/v1/audio/speech", json={"input": "hello", "format": "wav"})
    assert resp_tts.status_code == 200
    assert resp_tts.headers.get("content-type", "").startswith("audio/wav")
    assert resp_tts.headers.get("content-disposition") == 'inline; filename="abstractcore-speech.wav"'
    assert resp_tts.headers.get("x-content-type-options") == "nosniff"
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
        json={"model": "abstractvoice/default", "input": "hello", "format": "wav"},
    )
    assert resp_tts.status_code == 200
    assert resp_tts.headers.get("content-type", "").startswith("audio/wav")
    assert resp_tts.content == b"wav-bytes"

    files = {"file": ("audio.wav", b"abc", "audio/wav")}
    resp_stt = client.post(
        "/v1/audio/transcriptions",
        files=files,
        data={"model": "abstractvoice/default", "language": "en"},
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
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
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
    assert resp.headers.get("content-disposition") == 'inline; filename="abstractcore-speech.mp3"'
    assert resp.content == b"mp3-bytes"
    assert captured["init"] == {"model": "gpt-4o-mini-tts", "api_key": "sk-provider-key"}
    assert captured["speech"]["input_text"] == "hello"
    assert captured["speech"]["voice"] == "coral"
    assert captured["speech"]["response_format"] == "mp3"
    assert captured["speech"]["instructions"] == "Speak clearly."


def test_provider_scoped_audio_speech_routes_local_engine_to_capability_plugin(client, monkeypatch):
    captured = {}

    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def tts(self, text: str, **kwargs):
                captured["tts"] = {"text": text, **kwargs}
                return b"wav-bytes"

            def stt(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

        registry.register_voice_backend(backend_id="fake-voice", factory=lambda _owner: _Voice(), priority=0)

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_FakeEntryPoint(name="fake", value="tests.fake_voice:register", obj=register)]),
    )
    _reset_audio_core(monkeypatch)

    resp = client.post(
        "/supertonic/v1/audio/speech",
        json={"model": "supertonic-3", "input": "hello", "voice": "M1", "format": "wav"},
    )

    assert resp.status_code == 200
    assert resp.content == b"wav-bytes"
    assert captured["tts"]["provider"] == "supertonic"
    assert captured["tts"]["model"] == "supertonic-3"
    assert captured["tts"]["voice"] == "M1"


def test_provider_scoped_audio_speech_routes_remote_model_without_body_prefix(client, monkeypatch):
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
        "/openai/v1/audio/speech",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
        json={"model": "gpt-4o-mini-tts", "input": "hello", "voice": "coral", "response_format": "mp3"},
    )

    assert resp.status_code == 200
    assert resp.content == b"mp3-bytes"
    assert captured["init"] == {"model": "gpt-4o-mini-tts", "api_key": "sk-provider-key"}
    assert captured["speech"]["voice"] == "coral"


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
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
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
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
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


def test_audio_speech_routes_to_openai_compatible_with_request_base_url(client, monkeypatch):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_speech(self, input_text: str, **kwargs):
        captured["speech"] = {"input_text": input_text, **kwargs}
        return b"wav-bytes", "audio/wav"

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatibleProvider, "synthesize_speech", fake_speech)

    resp = client.post(
        "/v1/audio/speech",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
        json={
            "model": "openai-compatible/default",
            "base_url": "http://127.0.0.1:5000/v1",
            "input": "hello",
            "voice": "speaker-1",
            "response_format": "wav",
        },
    )

    assert resp.status_code == 200
    assert resp.content == b"wav-bytes"
    assert captured["init"] == {
        "model": "default",
        "api_key": "sk-provider-key",
        "base_url": "http://127.0.0.1:5000/v1",
    }
    assert captured["speech"]["input_text"] == "hello"
    assert captured["speech"]["voice"] == "speaker-1"


def test_audio_transcriptions_routes_to_openai_compatible_with_request_base_url(client, monkeypatch):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_transcribe(self, audio: bytes, **kwargs):
        captured["transcribe"] = {"audio": audio, **kwargs}
        return b'{"text":"local endpoint transcript"}', "application/json"

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatibleProvider, "transcribe_audio", fake_transcribe)

    resp = client.post(
        "/v1/audio/transcriptions",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
        files={"file": ("speech.wav", b"abc", "audio/wav")},
        data={
            "model": "openai-compatible/default",
            "base_url": "http://127.0.0.1:5000/v1",
            "language": "en",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "local endpoint transcript"}
    assert captured["init"] == {
        "model": "default",
        "api_key": "sk-provider-key",
        "base_url": "http://127.0.0.1:5000/v1",
    }
    assert captured["transcribe"]["audio"] == b"abc"
    assert captured["transcribe"]["filename"] == "speech.wav"


def test_provider_scoped_audio_transcriptions_prefixes_plain_model(client, monkeypatch):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_transcribe(self, audio: bytes, **kwargs):
        captured["transcribe"] = {"audio": audio, **kwargs}
        return b'{"text":"provider scoped transcript"}', "application/json"

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatibleProvider, "transcribe_audio", fake_transcribe)

    resp = client.post(
        "/openai-compatible/v1/audio/transcriptions",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
        files={"file": ("speech.wav", b"abc", "audio/wav")},
        data={
            "model": "default",
            "base_url": "http://127.0.0.1:5000/v1",
            "language": "en",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "provider scoped transcript"}
    assert captured["init"] == {
        "model": "default",
        "api_key": "sk-provider-key",
        "base_url": "http://127.0.0.1:5000/v1",
    }
    assert captured["transcribe"]["audio"] == b"abc"
    assert captured["transcribe"]["filename"] == "speech.wav"


def test_provider_scoped_audio_transcriptions_routes_local_engine_to_capability_plugin(client, monkeypatch):
    captured = {}

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
                captured["transcribe"] = {"audio": audio, **kwargs}
                return "provider scoped transcript"

        registry.register_voice_backend(backend_id="fake-voice", factory=lambda _owner: _Voice(), priority=0)
        registry.register_audio_backend(backend_id="fake-audio", factory=lambda _owner: _Audio(), priority=0)

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_FakeEntryPoint(name="fake", value="tests.fake_voice_audio:register", obj=register)]),
    )
    _reset_audio_core(monkeypatch)

    resp = client.post(
        "/faster-whisper/v1/audio/transcriptions",
        files={"file": ("speech.wav", b"abc", "audio/wav")},
        data={"model": "large-v3", "language": "en"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "provider scoped transcript"}
    assert captured["transcribe"]["language"] == "en"
    assert captured["transcribe"]["provider"] == "faster-whisper"
    assert captured["transcribe"]["model"] == "large-v3"


def test_voice_clone_routes_to_openai_compatible_endpoint(client, monkeypatch):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_clone(self, audio: bytes, **kwargs):
        captured["clone"] = {"audio": audio, **kwargs}
        return {"ok": True, "voice_id": "voice-123"}

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatibleProvider, "clone_voice", fake_clone)

    resp = client.post(
        "/v1/voice/clone",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
        files={"file": ("reference.wav", b"wavref", "audio/wav")},
        data={
            "model": "openai-compatible/default",
            "base_url": "http://127.0.0.1:5000/v1",
            "name": "my_voice",
            "reference_text": "hello there",
            "validate": "true",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "voice_id": "voice-123"}
    assert captured["init"] == {
        "model": "default",
        "api_key": "sk-provider-key",
        "base_url": "http://127.0.0.1:5000/v1",
    }
    assert captured["clone"]["audio"] == b"wavref"
    assert captured["clone"]["filename"] == "reference.wav"
    assert captured["clone"]["content_type"] == "audio/wav"
    assert captured["clone"]["name"] == "my_voice"
    assert captured["clone"]["reference_text"] == "hello there"
    assert captured["clone"]["validate"] is True
    assert captured["clone"]["clone_path"] == "/voice/clone"
    assert captured["clone"]["file_field"] == "file"


def test_provider_scoped_voice_clone_prefixes_plain_model(client, monkeypatch):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    captured = {}

    def fake_init(self, model: str, **kwargs):
        self.model = model
        captured["init"] = {"model": model, **kwargs}

    def fake_clone(self, audio: bytes, **kwargs):
        captured["clone"] = {"audio": audio, **kwargs}
        return {"ok": True, "voice_id": "voice-456"}

    monkeypatch.setattr(OpenAICompatibleProvider, "__init__", fake_init)
    monkeypatch.setattr(OpenAICompatibleProvider, "clone_voice", fake_clone)

    resp = client.post(
        "/openai-compatible/v1/voice/clone",
        headers={"X-AbstractCore-Provider-API-Key": "sk-provider-key"},
        files={"file": ("reference.wav", b"wavref", "audio/wav")},
        data={
            "base_url": "http://127.0.0.1:5000/v1",
            "name": "my_voice",
            "reference_text": "hello there",
            "validate": "true",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "voice_id": "voice-456"}
    assert captured["init"] == {
        "model": "default",
        "api_key": "sk-provider-key",
        "base_url": "http://127.0.0.1:5000/v1",
    }
    assert captured["clone"]["audio"] == b"wavref"
    assert captured["clone"]["filename"] == "reference.wav"
    assert captured["clone"]["name"] == "my_voice"


def test_provider_scoped_voice_clone_routes_local_engine_to_capability_plugin(client, monkeypatch):
    captured = {}

    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def tts(self, text: str, **kwargs):
                _ = text, kwargs
                return b"wav-bytes"

            def stt(self, audio, **kwargs):
                _ = audio, kwargs
                return "transcript"

            def clone(self, audio, **kwargs):
                captured["clone"] = {"audio": audio, **kwargs}
                return {"voice_id": "voice-local", "meta": {"engine": kwargs.get("cloning_engine")}}

        registry.register_voice_backend(backend_id="fake-voice", factory=lambda _owner: _Voice(), priority=0)

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_FakeEntryPoint(name="fake", value="tests.fake_voice:register", obj=register)]),
    )
    _reset_audio_core(monkeypatch)

    resp = client.post(
        "/omnivoice/v1/voice/clone",
        files={"file": ("reference.wav", b"wavref", "audio/wav")},
        data={"name": "my_voice", "reference_text": "hello there"},
    )

    assert resp.status_code == 200
    assert resp.json()["voice_id"] == "voice-local"
    assert captured["clone"]["name"] == "my_voice"
    assert captured["clone"]["reference_text"] == "hello there"
    assert captured["clone"]["cloning_engine"] == "omnivoice"
