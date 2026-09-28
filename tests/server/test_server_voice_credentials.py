"""Server-held OpenAI credentials on the in-process voice lane (route level).

One rule for both audio lanes (`openai/...` model path and the AbstractVoice
lane): a request that is not server-authenticated never spends a key the
server holds — the one saved in the config (Providers) or OPENAI_API_KEY,
which AbstractVoice falls back to. An authenticated request gets the saved
key, read per request (a rotated key is used on the next call, a removed one
stops being sent); a caller key is the only key used when given. Local
engines never see a key. The shared capability core never carries one.
"""

from __future__ import annotations

import importlib.metadata
import io
import logging
import os
import wave

import pytest
from fastapi.testclient import TestClient

from abstractcore.config import manager as config_manager_module
from abstractcore.config.manager import ConfigurationManager
from abstractcore.server import audio_endpoints as ae
from abstractcore.server.app import app

SERVER_TOKEN = "srv-token-test"
KEY_A = "sk-SENTINEL-AAAA1111"
KEY_B = "sk-SENTINEL-BBBB2222"
CALLER = "sk-SENTINEL-CALLER33"

_REFUSAL = "Server-held OPENAI_API_KEY is configured, but inbound server auth is not"


class _EP:
    name = "fake"
    value = "tests.fake_voice:register"

    def __init__(self, register):
        self._register = register

    def load(self):
        return self._register


class _EntryPoints:
    def __init__(self, eps):
        self._eps = eps

    def select(self, *, group):
        return list(self._eps) if group == "abstractcore.capabilities_plugins" else []


@pytest.fixture()
def seen(monkeypatch):
    """Install a fake voice/audio plugin recording, per call, the key AbstractVoice
    would authenticate with: the host setting, else its OPENAI_API_KEY fallback."""
    calls: list = []

    def effective(owner):
        cfg = getattr(owner, "config", {}) or {}
        return {
            "voice_openai_api_key": cfg.get("voice_openai_api_key"),
            "voice_remote_api_key": cfg.get("voice_remote_api_key"),
            "env": os.environ.get("OPENAI_API_KEY"),
        }

    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def __init__(self, owner):
                self._owner = owner

            def tts(self, text, **kwargs):
                calls.append(("tts", kwargs.get("provider"), effective(self._owner)))
                return b"wav-bytes"

            def tts_stream(self, text, **kwargs):
                calls.append(("tts_stream", kwargs.get("provider"), effective(self._owner)))
                yield {"type": "audio", "content_type": "audio/wav", "format": "wav", "sequence": 0, "audio": b"x"}
                yield {"type": "done", "ok": True, "chunks": 1}

            def stt(self, audio, **kwargs):
                calls.append(("stt", kwargs.get("provider"), effective(self._owner)))
                return "transcript"

            def clone(self, audio, **kwargs):
                calls.append(("clone", kwargs.get("cloning_engine"), effective(self._owner)))
                return {"voice_id": "v1"}

            def available_providers(self):
                calls.append(("catalog", None, effective(self._owner)))
                return {"tts": ["openai"]}

        class _Audio:
            backend_id = "fake-audio"

            def __init__(self, owner):
                self._owner = owner

            def transcribe(self, audio, **kwargs):
                calls.append(("stt", kwargs.get("provider"), effective(self._owner)))
                return "transcript"

        registry.register_voice_backend(backend_id="fake-voice", factory=lambda owner: _Voice(owner), priority=0)
        registry.register_audio_backend(backend_id="fake-audio", factory=lambda owner: _Audio(owner), priority=0)

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_EP(register)]))
    monkeypatch.setattr(ae, "_CORE", None)
    return calls


@pytest.fixture()
def config(monkeypatch, tmp_path):
    """A real ConfigurationManager on a scratch file, as the process-global one."""
    for name in ("OPENAI_API_KEY", "ABSTRACTVOICE_TTS_ENGINE", "ABSTRACTVOICE_STT_ENGINE", "ABSTRACTVOICE_CLONING_ENGINE"):
        # setenv first: monkeypatch then restores the original state at teardown,
        # even though the manager writes OPENAI_API_KEY behind its back.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    mgr = ConfigurationManager(config_file=tmp_path / "abstractcore.json")
    # Speech through OpenAI (the fresh-config default is a local engine).
    mgr.set_capability_default("output", "voice", provider="openai", model="gpt-4o-mini-tts")
    monkeypatch.setattr(config_manager_module, "_config_manager", mgr)
    monkeypatch.setattr(config_manager_module, "get_config_manager", lambda: mgr)
    return mgr


@pytest.fixture()
def unauthenticated(monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    return TestClient(app)


@pytest.fixture()
def authenticated(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", SERVER_TOKEN)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {SERVER_TOKEN}"})
    return client


def _speech(client, **body):
    return client.post("/v1/audio/speech", json={"input": "hello", "format": "wav", **body})


def _stream(client, **body):
    return client.post("/v1/audio/speech/stream", json={"input": "hello", "format": "wav", **body})


def _stt(client, **data):
    return client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"abc", "audio/wav")}, data=data)


def _clone(client, **data):
    return client.post("/v1/voice/clone", files={"file": ("r.wav", b"abc", "audio/wav")}, data=data)


def _assert_refused(resp):
    assert resp.status_code == 401, resp.text
    assert _REFUSAL in resp.json()["error"]["message"]


# ------------------------------------------------------------ unauthenticated
def test_unauthenticated_requests_never_spend_the_saved_key(seen, config, unauthenticated, monkeypatch):
    config.set_api_key("openai", KEY_A)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # the manager exported it; the saved key alone is under test
    for resp in (_speech(unauthenticated), _stream(unauthenticated), _stt(unauthenticated)):
        _assert_refused(resp)
    assert seen == [], "the plugin must never run with the server's key for an unauthenticated caller"


def test_unauthenticated_requests_never_spend_the_env_key(seen, config, unauthenticated, monkeypatch):
    """AbstractVoice's own OPENAI_API_KEY fallback is closed by the same rule."""
    monkeypatch.setenv("OPENAI_API_KEY", KEY_A)
    monkeypatch.setenv("ABSTRACTVOICE_CLONING_ENGINE", "openai-compatible")
    for resp in (
        _speech(unauthenticated),
        _speech(unauthenticated, provider="openai-compatible"),
        _stream(unauthenticated),
        _stt(unauthenticated),
        _clone(unauthenticated),
    ):
        _assert_refused(resp)
    assert seen == []


def test_the_openai_model_path_applies_the_same_rule_to_the_saved_key(seen, config, unauthenticated, monkeypatch):
    config.set_api_key("openai", KEY_A)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _assert_refused(_speech(unauthenticated, model="openai/gpt-4o-mini-tts"))


def test_local_engines_run_unauthenticated_and_never_see_a_key(seen, config, unauthenticated, monkeypatch):
    config.set_api_key("openai", KEY_A)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # the manager exported it; only the host setting is under test
    assert _speech(unauthenticated, provider="piper").status_code == 200
    assert _stt(unauthenticated, provider="faster-whisper").status_code == 200
    assert _clone(unauthenticated, provider="omnivoice").status_code == 200
    assert [c[0] for c in seen] == ["tts", "stt", "clone"]
    assert all(c[2]["voice_openai_api_key"] is None for c in seen)


def test_a_caller_key_is_the_only_key_used(seen, config, unauthenticated):
    config.set_api_key("openai", KEY_A)
    headers = {"X-AbstractCore-Provider-API-Key": CALLER}
    assert unauthenticated.post("/v1/audio/speech", json={"input": "hi", "format": "wav"}, headers=headers).status_code == 200
    assert seen[-1][2]["voice_openai_api_key"] == CALLER
    assert seen[-1][2]["voice_remote_api_key"] == CALLER


# -------------------------------------------------------------- authenticated
def test_authenticated_requests_get_the_saved_key(seen, config, authenticated):
    config.set_api_key("openai", KEY_A)
    assert _speech(authenticated).status_code == 200
    assert _stream(authenticated).status_code == 200
    assert _stt(authenticated).status_code == 200
    assert [c[0] for c in seen] == ["tts", "tts_stream", "stt"]
    assert all(c[2]["voice_openai_api_key"] == KEY_A for c in seen)


def test_a_rotated_key_is_used_on_the_next_request_and_a_removed_key_stops(seen, config, authenticated):
    config.set_api_key("openai", KEY_A)
    assert _speech(authenticated).status_code == 200
    config.set_api_key("openai", KEY_B)
    assert _speech(authenticated).status_code == 200
    config.set_api_key("openai", "")
    assert _speech(authenticated).status_code == 200
    keys = [(c[2]["voice_openai_api_key"], c[2]["env"]) for c in seen]
    assert keys[0] == (KEY_A, KEY_A)
    assert keys[1] == (KEY_B, KEY_B), "the second request must send the rotated key"
    assert keys[2] == (None, None), "a removed key must stop being sent (host setting AND env fallback)"


def test_the_shared_core_never_carries_a_key(seen, config, authenticated):
    config.set_api_key("openai", KEY_A)
    assert _speech(authenticated).status_code == 200
    assert _speech(authenticated, provider="piper").status_code == 200  # builds the shared core
    assert ae._CORE is not None
    _ = ae._CORE.capabilities  # would self-seed the saved key if the server core were allowed to
    assert "voice_openai_api_key" not in ae._CORE.config
    assert "voice_openai_api_key" not in ae._capability_config()


def test_keys_never_reach_logs(seen, config, authenticated, caplog):
    caplog.set_level(logging.DEBUG)
    config.set_api_key("openai", KEY_A)
    assert _speech(authenticated).status_code == 200
    assert _stt(authenticated).status_code == 200
    assert KEY_A not in caplog.text


def test_the_catalog_gets_the_saved_key_only_for_authenticated_requests(seen, config, authenticated, monkeypatch):
    config.set_api_key("openai", KEY_A)
    assert authenticated.get("/v1/audio/speech/providers").status_code == 200
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    assert TestClient(app).get("/v1/audio/speech/providers").status_code == 200
    assert [c[0] for c in seen] == ["catalog", "catalog"]
    assert seen[0][2]["voice_openai_api_key"] == KEY_A
    assert seen[1][2]["voice_openai_api_key"] is None


def test_clearing_a_configured_key_restores_the_env_value_it_shadowed(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-own")
    mgr = ConfigurationManager(config_file=tmp_path / "abstractcore.json")
    mgr.set_api_key("openai", KEY_A)
    assert os.environ["OPENAI_API_KEY"] == KEY_A
    mgr.set_api_key("openai", "")
    assert os.environ["OPENAI_API_KEY"] == "sk-env-own"


# ------------------------------------------------ real AbstractVoice, real wire
def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 160)
    return buf.getvalue()


def test_real_abstractvoice_sends_the_rotated_key_upstream(config, authenticated, monkeypatch):
    """End to end through AbstractVoice (>= 0.13): the Authorization header on
    the outbound OpenAI request follows the saved key per request."""
    av = pytest.importorskip("abstractvoice")
    from packaging.version import Version

    if Version(av.__version__) < Version("0.13.0"):
        pytest.skip("AbstractVoice >= 0.13.0 reads voice_openai_api_key")
    import requests

    sent: list = []

    class _Resp:
        status_code = 200
        ok = True
        headers = {"content-type": "audio/wav"}
        text = ""
        content = _wav()

        def json(self):
            raise ValueError("binary")

        def raise_for_status(self):
            return None

    def fake_request(self, method, url, **kwargs):
        headers = {**dict(self.headers), **dict(kwargs.get("headers") or {})}
        sent.append((url, headers.get("Authorization", "")))
        return _Resp()

    monkeypatch.setattr(requests.Session, "request", fake_request)
    monkeypatch.setattr(ae, "_CORE", None)

    config.set_api_key("openai", KEY_A)
    assert _speech(authenticated).status_code == 200
    config.set_api_key("openai", KEY_B)
    assert _speech(authenticated).status_code == 200
    config.set_api_key("openai", "")
    third = _speech(authenticated)

    assert [a for _u, a in sent[:2]] == [f"Bearer {KEY_A}", f"Bearer {KEY_B}"]
    assert all("api.openai.com" in u for u, _a in sent)
    assert len(sent) == 2, "with the key removed nothing is sent"
    assert third.status_code >= 400
    assert KEY_A not in third.text and KEY_B not in third.text
