from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from abstractcore.capabilities.errors import CapabilityUnavailableError
from abstractcore.server.app import app


class _FakeVoice:
    backend_id = "fake-voice"

    def voice_catalog(self):
        return {
            "kind": "tts",
            "engine_id": "fake",
            "active_profile": {"profile_id": "coral"},
            "active_model": "tts-test",
            "profiles": [{"profile_id": "coral"}],
            "tts_models": ["tts-test"],
        }

    def list_tts_models(self):
        return ["tts-test"]

    def list_stt_models(self):
        return ["stt-test"]


class _FakeVision:
    backend_id = "fake-vision"

    def list_provider_models(self, *, task=None):
        return [{"id": "image-test", "task": task}]


class _FakeCore:
    voice = _FakeVoice()
    vision = _FakeVision()


def _error_message(response) -> str:
    data = response.json()
    return str(data.get("error", {}).get("message") or data.get("detail") or "")


@pytest.fixture(autouse=True)
def clean_catalog_auth_env(monkeypatch):
    for key in (
        "ABSTRACTCORE_SERVER_API_KEY",
        "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED",
        "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST",
        "ABSTRACTVOICE_REMOTE_API_KEY",
        "ABSTRACTVOICE_OPENAI_API_KEY",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_audio_voice_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None: _FakeCore())

    response = TestClient(app).get("/v1/audio/voices")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["source"] == "abstractvoice"
    assert payload["backend_id"] == "fake-voice"
    assert payload["active_model"] == "tts-test"
    assert payload["profiles"][0]["profile_id"] == "coral"


def test_audio_speech_models_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None: _FakeCore())

    response = TestClient(app).get("/v1/audio/speech/models")
    assert response.status_code == 200
    assert response.json()["models"] == ["tts-test"]


def test_audio_transcription_models_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None: _FakeCore())

    response = TestClient(app).get("/v1/audio/transcriptions/models")
    assert response.status_code == 200
    assert response.json()["models"] == ["stt-test"]


def test_vision_provider_models_catalog_route(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)
    monkeypatch.setattr(vision_endpoints, "_vision_catalog_core", lambda request, *, base_url=None: _FakeCore())

    response = TestClient(app).get("/v1/vision/provider_models?task=text_to_image")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["source"] == "abstractvision"
    assert payload["backend_id"] == "fake-vision"
    assert payload["models"] == [{"id": "image-test", "task": "text_to_image"}]


def test_vision_provider_models_rejects_unknown_task(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)

    response = TestClient(app).get("/v1/vision/provider_models?task=unknown")
    assert response.status_code == 400


def test_catalog_routes_surface_missing_plugins_as_501(monkeypatch):
    from abstractcore.server import audio_endpoints, vision_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_SERVER_API_KEY", raising=False)

    def missing_audio(*_args, **_kwargs):
        raise CapabilityUnavailableError(capability="voice", reason="No backends registered")

    def missing_vision(*_args, **_kwargs):
        raise CapabilityUnavailableError(capability="vision", reason="No backends registered")

    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", missing_audio)
    monkeypatch.setattr(vision_endpoints, "_vision_catalog_core", missing_vision)

    client = TestClient(app)
    assert client.get("/v1/audio/voices").status_code == 501
    assert client.get("/v1/vision/provider_models").status_code == 501


@pytest.mark.parametrize("path", ["/v1/audio/voices", "/v1/audio/speech/models", "/v1/audio/transcriptions/models"])
def test_audio_catalog_routes_preserve_server_credential_auth_error(monkeypatch, path):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-held-key")

    response = TestClient(app).get(path)

    assert response.status_code == 401
    assert "server-held" in _error_message(response).lower()


@pytest.mark.parametrize("path", ["/v1/audio/voices", "/v1/audio/speech/models", "/v1/audio/transcriptions/models"])
def test_audio_catalog_routes_preserve_bad_base_url_error(monkeypatch, path):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    response = TestClient(app).get(f"{path}?base_url=not-a-url")

    assert response.status_code == 400
    assert "absolute http(s) url" in _error_message(response).lower()


@pytest.mark.parametrize("path", ["/v1/audio/voices", "/v1/audio/speech/models", "/v1/audio/transcriptions/models"])
def test_audio_catalog_routes_preserve_disallowed_base_url_error(monkeypatch, path):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    response = TestClient(app).get(f"{path}?base_url=https://example.invalid/v1")

    assert response.status_code == 403
    assert "restricted for security" in _error_message(response).lower()
