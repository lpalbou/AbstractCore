"""A caller's own provider key reaches only the provider the discovery request names (backlog 0955).

`X-AbstractCore-Provider-API-Key` on a capability discovery route used to be handed to every
key slot the capability has (OpenAI and the OpenAI-compatible server for voice, AceMusic and
ElevenLabs for music; all of them on `/v1/capabilities`), so a key meant for one vendor could
reach another vendor's endpoint. Each test records the owner config the request's registry is
built with; no plugin, network or server key is involved.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from abstractcore.server import audio_endpoints as ae
from abstractcore.server import capability_generation
from abstractcore.server.app import app

CALLER = "sk-CALLER-ONE-VENDOR"
KEY_SLOTS = ("voice_openai_api_key", "voice_remote_api_key", "vision_api_key",
             "music_acemusic_api_key", "music_elevenlabs_api_key")


class _Registry:
    def __init__(self) -> None:
        self.music = self

    def list_models(self, *a: Any, **k: Any) -> List[Any]:
        return []

    def available_providers(self, *a: Any, **k: Any) -> List[Any]:
        return []

    def provider_details(self, *a: Any, **k: Any) -> List[Any]:
        return []

    def status(self) -> Dict[str, Any]:
        return {}

    def list_backend_infos(self) -> List[Any]:
        return []


@pytest.fixture()
def built(monkeypatch) -> List[Dict[str, Any]]:
    """The owner configs registries were built with for a caller key, in order."""
    configs: List[Dict[str, Any]] = []

    class _Core:
        capabilities = _Registry()

    def create(**config: Any) -> Any:
        configs.append(config)
        return _Core()

    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setattr(capability_generation, "create_capability_generation_core", create)
    monkeypatch.setattr(ae, "_capability_config", lambda: {"music_backend": "abstractmusic:elevenlabs-music"})
    remote = {"openai": ("openai", True), "compatible": ("openai-compatible", True), "piper": ("piper", False)}
    monkeypatch.setattr(ae, "_voice_engine_spends_server_keys", lambda engine: remote.get(engine, (engine, True)))
    return configs


def _get(path: str):
    return TestClient(app).get(path, headers={"X-AbstractCore-Provider-API-Key": CALLER})


def _slots_with_the_key(config: Dict[str, Any]) -> List[str]:
    return [slot for slot in KEY_SLOTS if config.get(slot) == CALLER]


def test_two_music_providers_the_key_reaches_only_the_named_one(built):
    response = _get("/v1/capabilities/music/models?provider=acemusic")
    assert response.status_code == 200, response.text
    assert _slots_with_the_key(built[-1]) == ["music_acemusic_api_key"]
    # The request's registry is pinned to the named backend, not the configured ElevenLabs one.
    assert built[-1]["music_backend"] == "abstractmusic:acemusic"

    assert _get("/v1/audio/music/models?provider=elevenlabs-music").status_code == 200
    assert _slots_with_the_key(built[-1]) == ["music_elevenlabs_api_key"]


def test_without_a_named_music_provider_the_key_reaches_only_the_configured_one(built):
    assert _get("/v1/audio/music/providers").status_code == 200
    assert _slots_with_the_key(built[-1]) == ["music_elevenlabs_api_key"]


def test_voice_the_key_reaches_only_the_named_remote_engine(built):
    assert _get("/v1/capabilities/voice/models?provider=openai").status_code == 200
    assert _slots_with_the_key(built[-1]) == ["voice_openai_api_key"]
    assert _get("/v1/capabilities/audio/models?provider=compatible").status_code == 200
    assert _slots_with_the_key(built[-1]) == ["voice_remote_api_key"]


def test_vision_has_one_destination(built):
    assert _get("/v1/capabilities/vision/models").status_code == 200
    assert _slots_with_the_key(built[-1]) == ["vision_api_key"]


@pytest.mark.parametrize("path", [
    "/v1/capabilities",                               # every capability: several vendors
    "/v1/capabilities/voice/providers",               # OpenAI and the compatible server
    "/v1/capabilities/voice/models",                  # no provider named
    "/v1/capabilities/voice/models?provider=piper",   # a local engine takes no key
    "/v1/capabilities/music/models?provider=nosuch",  # never falls back to the configured backend
])
def test_a_key_with_no_single_provider_named_is_refused_and_reaches_nobody(built, path):
    response = _get(path)
    assert response.status_code == 400, response.text
    assert "one provider only" in response.text
    assert built == []
