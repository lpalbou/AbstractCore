from __future__ import annotations

import json

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
            "active_tts_provider": "fake-tts",
            "active_stt_provider": "fake-stt",
            "active_cloning_provider": "fake-clone",
            "profiles": [{"profile_id": "coral"}],
            "tts_providers": ["fake-tts"],
            "stt_providers": ["fake-stt"],
            "cloning_providers": ["fake-clone"],
            "tts_models": ["tts-test"],
            "stt_models": ["stt-test"],
            "tts_models_by_provider": {"fake-tts": ["tts-test"]},
            "stt_models_by_provider": {"fake-stt": ["stt-test"]},
            "compatibility_catalog": {
                "providers": {
                    "cloning": {
                        "fake-clone": {
                            "models": {
                                "clone-test": {"surfaces": {"default": {"voice_clone": {"support": "native"}}}},
                                "*": {"surfaces": {"default": {"voice_clone": {"support": "native"}}}},
                            }
                        }
                    }
                }
            },
            "available_tts_providers": ["fake-tts"],
            "available_stt_providers": ["fake-stt"],
            "available_cloning_providers": ["fake-clone"],
        }

    def available_providers(self):
        return {
            "tts": ["fake-tts"],
            "stt": ["fake-stt"],
            "cloning": ["fake-clone"],
            "active_tts_provider": "fake-tts",
            "active_stt_provider": "fake-stt",
            "active_cloning_provider": "fake-clone",
        }

    def list_tts_models(self, provider=None):
        if provider and provider != "fake-tts":
            return []
        return ["tts-test"]

    def list_stt_models(self, provider=None):
        if provider and provider != "fake-stt":
            return []
        return ["stt-test"]

    def list_cloning_models(self, provider=None):
        if provider and provider != "fake-clone":
            return []
        return ["clone-test"]


class _FakeVision:
    backend_id = "fake-vision"

    def list_provider_models(self, *, task=None):
        return [{"id": "image-test", "model": "fake-vision/image-test", "provider": "fake-vision", "task": task}]


class _FakeCore:
    voice = _FakeVoice()
    vision = _FakeVision()


def _error_message(response) -> str:
    data = response.json()
    return str(data.get("error", {}).get("message") or data.get("detail") or "")


@pytest.fixture(autouse=True)
def clean_catalog_auth_env(monkeypatch):
    for key in (
        "ABSTRACTCORE_AUTH_TOKEN",
        "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED",
        "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_audio_voice_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())

    response = TestClient(app).get("/v1/audio/voices")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["source"] == "abstractvoice"
    assert payload["backend_id"] == "fake-voice"
    assert payload["active_model"] == "tts-test"
    assert payload["profiles"][0]["profile_id"] == "coral"


def test_audio_voice_catalog_route_filters_provider_and_model(monkeypatch):
    from abstractcore.server import audio_endpoints

    class _CatalogVoice:
        backend_id = "catalog-voice"

        def voice_catalog(self):
            return {
                "profiles": [
                    {"profile_id": "coral", "provider": "fake-tts", "params": {"model": "tts-test", "voice": "coral"}},
                    {
                        "profile_id": "M1",
                        "voice_id": "M1",
                        "provider": "supertonic",
                        "engine_id": "supertonic",
                        "params": {"model": "supertonic-3", "voice": "M1"},
                    },
                ],
                "voices": [
                    {"profile_id": "coral", "voice_id": "coral", "provider": "fake-tts", "params": {"model": "tts-test"}},
                    {
                        "profile_id": "M1",
                        "voice_id": "M1",
                        "provider": "supertonic",
                        "engine_id": "supertonic",
                        "params": {"model": "supertonic-3", "voice": "M1"},
                    },
                ],
                "tts_providers": ["fake-tts", "supertonic"],
                "stt_providers": ["fake-stt"],
                "tts_models": ["tts-test", "supertonic-3"],
                "tts_models_by_provider": {"fake-tts": ["tts-test"], "supertonic": ["supertonic-3"]},
                "tts_voices_by_provider": {"fake-tts": ["coral"], "supertonic": ["M1"]},
                "tts_profiles_by_provider": {"fake-tts": ["coral"], "supertonic": ["M1"]},
            }

    class _CatalogCore:
        voice = _CatalogVoice()

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _CatalogCore())

    response = TestClient(app).get("/v1/audio/voices?provider=supertonic&model=supertonic-3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"] == ["supertonic"]
    assert payload["tts_providers"] == ["supertonic"]
    assert payload["tts_models"] == ["supertonic-3"]
    assert payload["tts_voices_by_provider"] == {"supertonic": ["M1"]}
    assert [item["profile_id"] for item in payload["profiles"]] == ["M1"]
    assert [item["voice_id"] for item in payload["voices"]] == ["M1"]


def test_audio_voice_catalog_route_supports_providers_only(monkeypatch):
    from abstractcore.server import audio_endpoints

    class _CatalogVoice:
        backend_id = "catalog-voice"

        def voice_catalog(self):
            return {
                "profiles": [
                    {"profile_id": "coral", "provider": "fake-tts"},
                    {"profile_id": "M1", "provider": "supertonic", "params": {"model": "supertonic-3"}},
                ],
                "voices": [
                    {"voice_id": "coral", "provider": "fake-tts"},
                    {"voice_id": "M1", "provider": "supertonic", "params": {"model": "supertonic-3"}},
                ],
                "tts_providers": ["fake-tts", "supertonic"],
                "stt_providers": ["fake-stt"],
                "tts_models_by_provider": {"fake-tts": ["tts-test"], "supertonic": ["supertonic-3"]},
                "tts_voices_by_provider": {"fake-tts": ["coral"], "supertonic": ["M1"]},
            }

    class _CatalogCore:
        voice = _CatalogVoice()

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _CatalogCore())

    response = TestClient(app).get("/v1/audio/voices?provider=supertonic&providers_only=true")
    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"] == ["supertonic"]
    assert payload["tts_providers"] == ["supertonic"]
    assert payload["profiles"] == []
    assert payload["voices"] == []


def test_audio_speech_models_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())

    response = TestClient(app).get("/v1/audio/speech/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["tts-test"]
    assert payload["providers"] == ["fake-tts"]
    assert payload["active_provider"] == "fake-tts"
    assert payload["models_by_provider"] == {"fake-tts": ["tts-test"]}
    assert payload["provider_models"][0]["model"] == "fake-tts/tts-test"


def test_audio_provider_catalog_routes(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())

    client = TestClient(app)
    assert client.get("/v1/audio/speech/providers").json()["providers"] == ["fake-tts"]
    assert client.get("/v1/audio/transcriptions/providers").json()["providers"] == ["fake-stt"]
    assert client.get("/v1/voice/clone/providers").json()["providers"] == ["fake-clone"]


def test_voice_clone_models_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())

    response = TestClient(app).get("/v1/voice/clone/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["clone-test"]
    assert payload["providers"] == ["fake-clone"]
    assert payload["models_by_provider"] == {"fake-clone": ["clone-test"]}
    assert payload["provider_models"][0]["model"] == "fake-clone/clone-test"


def test_audio_voice_models_endpoint_removed(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    response = TestClient(app).get("/v1/audio/voices/models")

    assert response.status_code == 404


def test_audio_transcription_models_catalog_route(monkeypatch):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())

    response = TestClient(app).get("/v1/audio/transcriptions/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == ["stt-test"]
    assert payload["providers"] == ["fake-stt"]
    assert payload["active_provider"] == "fake-stt"
    assert payload["models_by_provider"] == {"fake-stt": ["stt-test"]}
    assert payload["provider_models"][0]["model"] == "fake-stt/stt-test"


def test_vision_provider_models_catalog_route(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(vision_endpoints, "_vision_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())
    monkeypatch.setattr(vision_endpoints, "_cached_vision_provider_model_entries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(vision_endpoints, "_configured_vision_provider_model_entries", lambda *_args, **_kwargs: [])

    response = TestClient(app).get("/v1/vision/provider_models?task=text_to_image")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["source"] == "abstractvision"
    assert payload["backend_id"] == "fake-vision"
    assert payload["providers"] == ["fake-vision"]
    assert payload["models_by_provider"] == {"fake-vision": ["fake-vision/image-test"]}
    assert payload["models"] == [{"id": "image-test", "model": "fake-vision/image-test", "provider": "fake-vision", "task": "text_to_image"}]


def test_vision_models_catalog_route_includes_provider_models(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(vision_endpoints, "_vision_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())
    calls = {"count": 0}

    def local_catalog():
        calls["count"] += 1
        return {"models": [], "registry_available": True, "cached_total": 0}

    monkeypatch.setattr("abstractcore.capabilities.vision_catalog.get_local_vision_cache_catalog", local_catalog)
    monkeypatch.setattr(vision_endpoints, "_active_state", lambda: {"has_backend": False})
    monkeypatch.setattr(vision_endpoints, "_configured_vision_provider_model_entries", lambda *_args, **_kwargs: [])

    response = TestClient(app).get("/v1/vision/models?task=text_to_image")

    assert response.status_code == 200
    payload = response.json()
    assert calls["count"] == 1
    assert payload["providers"] == ["fake-vision"]
    assert payload["models_by_provider"] == {"fake-vision": ["fake-vision/image-test"]}
    assert payload["models"][0]["provider"] == "fake-vision"
    assert payload["local_models"] == []
    assert payload["active"] == {"has_backend": False}


def test_vision_provider_models_rejects_unknown_task(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    response = TestClient(app).get("/v1/vision/providers/?task=unknown")
    assert response.status_code == 400


def test_vision_cached_diffusers_discovery_filters_non_image_pipelines(tmp_path):
    from abstractcore.capabilities import vision_catalog

    def write_model_index(model_id: str, class_name: str, extra: dict | None = None):
        folder = tmp_path / ("models--" + model_id.replace("/", "--")) / "snapshots" / "abc"
        folder.mkdir(parents=True)
        payload = {"_class_name": class_name}
        if extra:
            payload.update(extra)
        (folder / "model_index.json").write_text(json.dumps(payload), encoding="utf-8")

    write_model_index("black-forest-labs/FLUX.2-dev", "FluxPipeline")
    write_model_index(
        "ACE-Step/acestep-v15-xl-turbo-diffusers",
        "AceStepPipeline",
        {"vae": ["diffusers", "AutoencoderOobleck"], "transformer": ["diffusers", "AceStepTransformer1DModel"]},
    )

    assert vision_catalog._discover_cached_hf_diffusers_models([tmp_path]) == ["black-forest-labs/FLUX.2-dev"]
    assert not vision_catalog._is_hf_model_cached("ACE-Step/acestep-v15-xl-turbo-diffusers", [tmp_path])


def test_vision_provider_models_do_not_synthesize_openai_default(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_IMAGE_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert vision_endpoints._configured_vision_provider_model_entries("text_to_image") == []


def test_vision_catalog_prefers_openai_key_for_official_openai(monkeypatch):
    from abstractcore.server import vision_endpoints

    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert vision_endpoints._vision_catalog_config_from_env()["vision_api_key"] == "sk-real"

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:5000/v1")
    assert vision_endpoints._vision_catalog_config_from_env()["vision_api_key"] == "sk-real"


def test_catalog_routes_surface_missing_plugins_as_501(monkeypatch):
    from abstractcore.server import audio_endpoints, vision_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    def missing_audio(*_args, **_kwargs):
        raise CapabilityUnavailableError(capability="voice", reason="No backends registered")

    def missing_vision(*_args, **_kwargs):
        raise CapabilityUnavailableError(capability="vision", reason="No backends registered")

    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", missing_audio)
    monkeypatch.setattr(vision_endpoints, "_vision_catalog_core", missing_vision)
    monkeypatch.setattr(vision_endpoints, "_cached_vision_provider_model_entries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(vision_endpoints, "_configured_vision_provider_model_entries", lambda *_args, **_kwargs: [])

    client = TestClient(app)
    assert client.get("/v1/audio/voices").status_code == 501
    assert client.get("/v1/vision/providers/").status_code == 501


@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/voices",
        "/v1/audio/speech/models",
        "/v1/audio/speech/providers",
        "/v1/audio/transcriptions/models",
        "/v1/audio/transcriptions/providers",
        "/v1/voice/clone/providers",
        "/v1/voice/clone/models",
    ],
)
def test_audio_catalog_routes_allow_unauthenticated_with_server_credentials(monkeypatch, path):
    from abstractcore.server import audio_endpoints

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "server-held-key")
    monkeypatch.setattr(audio_endpoints, "_audio_catalog_core", lambda request, *, base_url=None, api_key=None: _FakeCore())

    response = TestClient(app).get(path)

    assert response.status_code == 200
    assert response.json()["available"] is True


@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/voices",
        "/v1/audio/speech/models",
        "/v1/audio/speech/providers",
        "/v1/audio/transcriptions/models",
        "/v1/audio/transcriptions/providers",
        "/v1/voice/clone/providers",
        "/v1/voice/clone/models",
    ],
)
def test_audio_catalog_routes_preserve_bad_base_url_error(monkeypatch, path):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    response = TestClient(app).get(f"{path}?base_url=not-a-url")

    assert response.status_code == 400
    assert "absolute http(s) url" in _error_message(response).lower()


@pytest.mark.parametrize(
    "path",
    [
        "/v1/audio/voices",
        "/v1/audio/speech/models",
        "/v1/audio/speech/providers",
        "/v1/audio/transcriptions/models",
        "/v1/audio/transcriptions/providers",
        "/v1/voice/clone/providers",
        "/v1/voice/clone/models",
    ],
)
def test_audio_catalog_routes_preserve_disallowed_base_url_error(monkeypatch, path):
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")

    response = TestClient(app).get(f"{path}?base_url=https://example.invalid/v1")

    assert response.status_code == 403
    assert "restricted for security" in _error_message(response).lower()
