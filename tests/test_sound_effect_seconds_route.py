"""R10.1 (2026-10-04): the sound-effect route hands the music plugin the task, the route's model
and the requested length (`seconds` / `duration_s`), and the server answers a sound effect.

Before: the music facade could call `t2m` without the task, and the plugin then defaulted every
request to music behaviour (30 s on the configured music checkpoint). A plugin that exposes
`generate(prompt, task=...)` must receive the task; `seconds` and `model` must reach it.
"""

import importlib.metadata

import pytest
from fastapi.testclient import TestClient

from abstractcore.config import manager as capability_config_manager
from abstractcore.providers.capability_host import create_capability_host


class _FakeEntryPoint:
    name = "fake"
    value = "tests.fake_sound_plugin:register"

    def __init__(self, obj):
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


def _plugin(calls):
    def register(registry):
        class _Music:
            backend_id = "abstractmusic:stable-audio-3"

            def __init__(self, owner):
                self.owner = owner

            def generate(self, prompt, *, task=None, **kwargs):
                calls.append({"prompt": prompt, "task": task, **kwargs})
                return b"RIFF-fake-wav"

            def t2m(self, prompt, **kwargs):  # pragma: no cover - must not be the path taken
                calls.append({"prompt": prompt, "task": "<t2m without task>", **kwargs})
                return b"RIFF-fake-wav"

        registry.register_music_backend(
            backend_id="abstractmusic:stable-audio-3", factory=lambda owner: _Music(owner), priority=0
        )

    return _FakeEntryPoint(register)


@pytest.fixture()
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_plugin(recorded)]))

    class _NoDefaults:
        def get_capability_default(self, *args, **kwargs):
            return {"source": "not_configured"}

    monkeypatch.setattr(capability_config_manager, "get_config_manager", lambda: _NoDefaults())
    return recorded


SFX_MODEL = "stabilityai/stable-audio-3-small-sfx"


@pytest.mark.basic
def test_sound_output_passes_task_model_and_seconds_to_the_plugin(calls):
    host = create_capability_host()
    response = host.generate(
        "laser gunshot",
        output={"modality": "sound", "task": "text_to_audio", "format": "wav", "model": SFX_MODEL, "seconds": 3},
    )
    assert response.outputs["sound"][0].task == "sound_generation"
    call = calls[-1]
    assert call["task"] == "text_to_audio"
    assert call["model"] == SFX_MODEL
    assert call["seconds"] == 3
    assert call["prompt"] == "laser gunshot"


@pytest.mark.basic
def test_sound_output_passes_duration_s_too(calls):
    host = create_capability_host()
    host.generate(
        "door slam",
        output={"modality": "sound", "task": "text_to_audio", "format": "wav", "model": SFX_MODEL, "duration_s": 2.5},
    )
    assert calls[-1]["duration_s"] == 2.5
    assert calls[-1]["task"] == "text_to_audio"


@pytest.mark.basic
def test_sound_output_without_length_sends_none_so_the_plugin_default_applies(calls):
    host = create_capability_host()
    host.generate("door slam", output={"modality": "sound", "task": "text_to_audio", "format": "wav", "duration_s": None})
    assert "duration_s" not in calls[-1]
    assert "seconds" not in calls[-1]


@pytest.fixture()
def client(calls, monkeypatch):
    from abstractcore.server import audio_endpoints
    from abstractcore.server.app import app

    monkeypatch.setattr(audio_endpoints, "_CORE", None)
    for key in ("ACEMUSIC_API_KEY", "ELEVENLABS_API_KEY", "ABSTRACTCORE_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    return TestClient(app)


@pytest.mark.basic
def test_server_sound_effect_honours_seconds_and_answers_the_sound_output(client, calls):
    resp = client.post(
        "/v1/audio/music",
        json={"prompt": "laser gunshot", "task": "text_to_audio", "seconds": 3, "model": SFX_MODEL, "format": "wav"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.content == b"RIFF-fake-wav"
    assert calls[-1]["seconds"] == 3.0
    assert calls[-1]["task"] == "text_to_audio"


@pytest.mark.basic
@pytest.mark.parametrize("body", [{"seconds": 0}, {"seconds": -2}, {"duration_s": 0}, {"seconds": 3, "duration_s": 4}])
def test_server_refuses_a_bad_length_with_a_sentence(client, calls, body):
    resp = client.post("/v1/audio/music", json={"prompt": "laser gunshot", "task": "text_to_audio", **body})
    assert resp.status_code == 400
    assert "seconds" in resp.text
    assert calls == []
