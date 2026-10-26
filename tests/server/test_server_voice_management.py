"""Languages, voice navigation, cloned-voice management, voice design and one_shot.

The voice plugin is a fake that records what AbstractCore forwards to it, so these
tests pin the HTTP and facade contracts, not AbstractVoice's behavior.
"""
import importlib.metadata

import pytest
from fastapi.testclient import TestClient

from abstractcore.capabilities.errors import CapabilityUnavailableError
from abstractcore.server.app import app

from .test_server_audio_endpoints import _EntryPoints, _FakeEntryPoint, _reset_audio_core


class _ManagedVoice:
    backend_id = "fake-managed-voice"

    def __init__(self):
        self.calls = []
        self.records = {
            "v1": {"voice_id": "v1", "name": "Ada", "engine": "qwen3-tts", "reference_text": "hello", "meta": {}},
            "builtin-1": {"voice_id": "builtin-1", "name": "Camille", "engine": "qwen3-tts", "meta": {"builtin": True}},
        }

    # -- synthesis ----------------------------------------------------------
    def tts(self, text, **kwargs):
        self.calls.append(("tts", text, kwargs))
        if kwargs.get("one_shot") and len(text) > 20:
            raise ValueError("one_shot synthesis is limited to 20 characters")
        return b"wav-bytes"

    def tts_stream(self, text, **kwargs):
        self.calls.append(("tts_stream", text, kwargs))
        yield {"type": "done", "ok": True}

    def stt(self, audio, **kwargs):
        return "transcript"

    def clone(self, audio, **kwargs):
        self.calls.append(("clone", kwargs.get("name")))
        self.records["new"] = {"voice_id": "new", "name": kwargs.get("name"), "engine": "qwen3-tts", "meta": {}}
        return "new"

    # -- discovery ----------------------------------------------------------
    def list_languages(self, *, kind=None, provider=None):
        self.calls.append(("list_languages", kind, provider))
        if provider == "nope":
            raise KeyError("unknown provider 'nope'")
        if kind == "music":
            raise ValueError("kind must be one of tts, stt, cloning")
        return {"tts": {"qwen3-tts": {"provider": "qwen3-tts", "languages": ["en", "fr"], "open": False, "declared": True}}}

    def voice_catalog(self, provider=None, model=None, providers_only=False, language=None):
        self.calls.append(("voice_catalog", provider, model, language))
        voices = [
            {"voice_id": "siwis", "name": "siwis", "kind": "profile", "provider": "piper", "languages": ["fr"], "language": "fr"},
            {"voice_id": "amy", "name": "amy", "kind": "profile", "provider": "piper", "languages": ["en"], "language": "en"},
        ]
        if language:
            voices = [v for v in voices if language in v["languages"]]
        return {"voices": voices, "profiles": voices, "cloned_voices": [], "tts_providers": ["piper"]}

    def list_tts_voices(self, *, provider=None, model=None, include_clones=True, language=None):
        self.calls.append(("list_tts_voices", provider, model, include_clones, language))
        return [{"voice_id": "siwis", "kind": "profile", "provider": "piper", "languages": ["fr"], "language": "fr"}]

    def list_cloned_voices(self, *, provider=None, model=None, language=None):
        self.calls.append(("list_cloned_voices", provider, model, language))
        return [dict(r, kind="clone") for r in self.records.values()]

    def compatibility_catalog(self):
        return {"version": 1, "providers": {}}

    # -- management ---------------------------------------------------------
    def get_cloned_voice(self, voice_id):
        return dict(self.records[voice_id])

    def rename_cloned_voice(self, voice_id, new_name):
        if not new_name.strip():
            raise ValueError("new_name is required")
        self.records[voice_id]["name"] = new_name
        return {"ok": True, **self.records[voice_id]}

    def set_cloned_voice_reference_text(self, voice_id, reference_text):
        self.records[voice_id]["reference_text"] = reference_text
        return {"ok": True, **self.records[voice_id]}

    def delete_cloned_voice(self, voice_id):
        record = self.records[voice_id]
        if record["meta"].get("builtin"):
            raise ValueError(f"{voice_id!r} is a built-in voice and cannot be deleted")
        del self.records[voice_id]
        return {"ok": True, "deleted": True, "voice_id": voice_id, "name": record["name"], "engine": record["engine"]}

    def design_voice(self, description, *, name=None, language=None, text=None):
        self.calls.append(("design_voice", description, name, language, text))
        self.records["d1"] = {"voice_id": "d1", "name": name, "engine": "qwen3-tts", "meta": {"designed": True}}
        return {"ok": True, **self.records["d1"]}


class _BareVoice:
    """A voice backend with synthesis only (no languages, management or design)."""

    backend_id = "fake-bare-voice"

    def tts(self, text, **kwargs):
        return b"wav-bytes"

    def stt(self, audio, **kwargs):
        return "transcript"

    def clone(self, audio, **kwargs):
        return "bare-1"


def _install(monkeypatch, backend):
    def register(registry):
        registry.register_voice_backend(backend_id=backend.backend_id, factory=lambda _owner: backend, priority=0)

    ep = _FakeEntryPoint(name="fake-managed", value="tests.fake_managed_voice:register", obj=register)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([ep]))
    _reset_audio_core(monkeypatch)
    return backend


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _no_server_auth(monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_languages_route_forwards_filters_and_maps_errors(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    resp = client.get("/v1/audio/languages", params={"kind": "tts", "provider": "qwen3-tts"})
    assert resp.status_code == 200
    assert resp.json()["languages"]["tts"]["qwen3-tts"]["languages"] == ["en", "fr"]
    assert ("list_languages", "tts", "qwen3-tts") in voice.calls
    assert client.get("/v1/audio/languages", params={"provider": "nope"}).status_code == 404
    assert client.get("/v1/audio/languages", params={"kind": "music"}).status_code == 422


def test_voices_route_forwards_the_language_filter(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    body = client.get("/v1/audio/voices", params={"language": "fr"}).json()
    assert [v["voice_id"] for v in body["voices"]] == ["siwis"]
    assert ("voice_catalog", None, None, "fr") in voice.calls


def test_clone_management_round_trip(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    listed = client.get("/v1/voice/clones", params={"language": "fr"}).json()
    assert listed["count"] == 2 and ("list_cloned_voices", None, None, "fr") in voice.calls

    assert client.get("/v1/voice/clones/v1").json()["name"] == "Ada"
    updated = client.patch("/v1/voice/clones/v1", json={"name": "Grace", "reference_text": "good morning"}).json()
    assert updated["name"] == "Grace" and updated["reference_text"] == "good morning"
    assert client.patch("/v1/voice/clones/v1", json={}).status_code == 422
    assert client.patch("/v1/voice/clones/v1", json={"name": "  "}).status_code == 422

    assert client.delete("/v1/voice/clones/builtin-1").status_code == 422  # read-only
    assert client.delete("/v1/voice/clones/v1").json()["deleted"] is True
    assert client.get("/v1/voice/clones/v1").status_code == 404
    assert client.delete("/v1/voice/clones/v1").status_code == 404


def test_clone_response_carries_the_stored_record(client, monkeypatch):
    _install(monkeypatch, _ManagedVoice())
    resp = client.post(
        "/v1/voice/clone",
        files={"file": ("ref.wav", b"RIFF", "audio/wav")},
        data={"provider": "qwen3-tts", "name": "Mine"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["voice_id"] == "new" and body["voice"]["name"] == "Mine" and body["voice"]["engine"] == "qwen3-tts"


def test_clone_response_without_record_lookup_still_returns_the_id(client, monkeypatch):
    _install(monkeypatch, _BareVoice())
    resp = client.post("/v1/voice/clone", files={"file": ("ref.wav", b"RIFF", "audio/wav")}, data={"provider": "qwen3-tts"})
    assert resp.status_code == 200 and resp.json() == {"ok": True, "id": "bare-1", "voice_id": "bare-1", "voice": {}}


def test_design_route_creates_a_voice(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    resp = client.post("/v1/voice/design", json={"description": "A warm narrator", "name": "Mira", "language": "en"})
    assert resp.status_code == 200 and resp.json()["voice_id"] == "d1"
    assert ("design_voice", "A warm narrator", "Mira", "en", None) in voice.calls
    assert client.post("/v1/voice/design", json={"description": "   "}).status_code == 422
    assert client.post("/v1/voice/design", json={}).status_code == 422


def test_management_routes_answer_501_for_a_backend_without_them(client, monkeypatch):
    _install(monkeypatch, _BareVoice())
    assert client.get("/v1/audio/languages").status_code == 501
    assert client.get("/v1/voice/clones/x").status_code == 501
    assert client.delete("/v1/voice/clones/x").status_code == 501
    assert client.post("/v1/voice/design", json={"description": "a voice"}).status_code == 501


def test_one_shot_reaches_the_plugin_only_when_requested(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    assert client.post("/v1/audio/speech", json={"input": "hi", "provider": "qwen3-tts", "one_shot": True}).status_code == 200
    assert client.post("/v1/audio/speech", json={"input": "hi", "provider": "qwen3-tts"}).status_code == 200
    tts_calls = [kwargs for name, _text, kwargs in [c for c in voice.calls if c[0] == "tts"]]
    assert tts_calls[0].get("one_shot") is True and "one_shot" not in tts_calls[1]
    # The backend's refusal (text over its limit) is the caller's error, not the server's.
    long = client.post("/v1/audio/speech", json={"input": "x" * 40, "provider": "qwen3-tts", "one_shot": True})
    assert long.status_code == 422 and "limited to 20" in long.json()["error"]["message"]


def test_one_shot_is_refused_on_stream_and_remote_routes(client, monkeypatch):
    _install(monkeypatch, _ManagedVoice())
    assert client.post("/v1/audio/speech/stream", json={"input": "hi", "one_shot": True}).status_code == 422
    remote = client.post("/v1/audio/speech", json={"input": "hi", "model": "openai/tts-1", "one_shot": True})
    assert remote.status_code == 422 and "remote providers" in remote.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Facade (llm.voice / core.voice)
# ---------------------------------------------------------------------------


def _facade(monkeypatch, backend):
    from abstractcore.server.capability_generation import create_capability_generation_core

    _install(monkeypatch, backend)
    return create_capability_generation_core().voice


def test_facade_forwards_languages_voices_management_and_design(monkeypatch):
    backend = _ManagedVoice()
    voice = _facade(monkeypatch, backend)
    assert "qwen3-tts" in voice.list_languages(kind="tts")["tts"]
    assert voice.list_voices(language="fr")[0]["voice_id"] == "siwis"
    assert ("list_tts_voices", None, None, True, "fr") in backend.calls
    assert voice.voice_catalog(language="fr")["voices"][0]["voice_id"] == "siwis"
    assert len(voice.list_cloned_voices()) == 2
    assert voice.rename_cloned_voice("v1", "Grace")["name"] == "Grace"
    assert voice.set_cloned_voice_reference_text("v1", "hi")["reference_text"] == "hi"
    assert voice.get_cloned_voice("v1")["name"] == "Grace"
    assert voice.delete_cloned_voice("v1")["deleted"] is True
    assert voice.design_voice("A warm narrator", name="Mira")["voice_id"] == "d1"
    assert voice.compatibility_catalog() == {"version": 1, "providers": {}}


def test_facade_names_the_missing_backend_method(monkeypatch):
    voice = _facade(monkeypatch, _BareVoice())
    for call in (
        lambda: voice.list_languages(),
        lambda: voice.list_voices(),
        lambda: voice.get_cloned_voice("x"),
        lambda: voice.design_voice("a voice"),
        lambda: voice.voice_catalog(language="fr"),
    ):
        with pytest.raises(CapabilityUnavailableError):
            call()


# ---------------------------------------------------------------------------
# Hardening
# ---------------------------------------------------------------------------


def test_voice_store_changes_need_server_auth_not_a_provider_key(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    junk = {"X-AbstractCore-Provider-API-Key": "sk-junk"}  # lets a BYOK request past the middleware
    assert client.delete("/v1/voice/clones/v1", headers=junk).status_code == 401
    assert client.patch("/v1/voice/clones/v1", json={"name": "x"}, headers=junk).status_code == 401
    assert client.post("/v1/voice/design", json={"description": "a voice"}, headers=junk).status_code == 401
    assert "v1" in voice.records and voice.records["v1"]["name"] == "Ada"
    assert not any(c[0] == "design_voice" for c in voice.calls)


def test_voice_store_changes_work_with_server_auth(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "srv-token")
    auth = {"Authorization": "Bearer srv-token"}
    assert client.delete("/v1/voice/clones/v1", headers=auth).json()["deleted"] is True
    assert "v1" not in voice.records


def test_patch_checks_every_field_before_changing_anything(client, monkeypatch):
    voice = _install(monkeypatch, _ManagedVoice())
    resp = client.patch("/v1/voice/clones/v1", json={"name": "Grace", "reference_text": "   "})
    assert resp.status_code == 422 and voice.records["v1"]["name"] == "Ada"
    assert client.patch("/v1/voice/clones/nope", json={"name": "Grace"}).status_code == 404


def test_language_filter_is_never_silently_dropped(client, monkeypatch):
    _install(monkeypatch, _BareVoice())
    resp = client.get("/v1/audio/voices", params={"language": "fr"})
    assert resp.status_code == 501


def test_speech_value_errors_stay_server_errors_without_one_shot(client, monkeypatch):
    class _Broken(_ManagedVoice):
        def tts(self, text, **kwargs):
            raise ValueError("internal decoder state mismatch")

    _install(monkeypatch, _Broken())
    assert client.post("/v1/audio/speech", json={"input": "hi", "provider": "qwen3-tts"}).status_code == 500


def test_clone_response_survives_a_failed_record_lookup(client, monkeypatch):
    class _Flaky(_ManagedVoice):
        def get_cloned_voice(self, voice_id):
            raise KeyError(voice_id)

    _install(monkeypatch, _Flaky())
    resp = client.post("/v1/voice/clone", files={"file": ("ref.wav", b"RIFF", "audio/wav")}, data={"provider": "qwen3-tts"})
    assert resp.status_code == 200 and resp.json()["voice_id"] == "new" and resp.json()["voice"] == {}
