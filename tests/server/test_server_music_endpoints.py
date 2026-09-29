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


def _make_fake_music_plugin_ep(calls=None):
    def register(registry):
        class _Music:
            backend_id = "fake-music"

            def t2m(self, prompt: str, **kwargs):
                if calls is not None:
                    calls.append({"prompt": prompt, **kwargs})
                return b"wav-bytes"

        registry.register_music_backend(backend_id="fake-music", factory=lambda _owner: _Music(), priority=0)

    return _FakeEntryPoint(name="fake", value="tests.fake_music:register", obj=register)


def _make_fake_multi_music_plugin_ep(calls=None):
    def register(registry):
        class _Music:
            def __init__(self, backend_id: str):
                self.backend_id = backend_id

            def t2m(self, prompt: str, **kwargs):
                if calls is not None:
                    calls.append({"backend_id": self.backend_id, "prompt": prompt, **kwargs})
                fmt = str(kwargs.get("format") or "wav").lower()
                content_type = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
                return {"data": f"{self.backend_id}:{fmt}".encode(), "mime_type": content_type}

        registry.register_music_backend(
            backend_id="abstractmusic:diffusers",
            factory=lambda _owner: _Music("abstractmusic:diffusers"),
            priority=20,
        )
        registry.register_music_backend(
            backend_id="abstractmusic:acemusic",
            factory=lambda _owner: _Music("abstractmusic:acemusic"),
            priority=0,
        )

    return _FakeEntryPoint(name="fake", value="tests.fake_music_multi:register", obj=register)


def _make_fake_unconfigured_ace_music_plugin_ep():
    def register(registry):
        class _Music:
            backend_id = "abstractmusic:acemusic"

            def t2m(self, prompt: str, **kwargs):
                _ = prompt, kwargs
                raise RuntimeError("Missing ACE Music API key. Set ACEMUSIC_API_KEY.")

        registry.register_music_backend(backend_id="abstractmusic:acemusic", factory=lambda _owner: _Music(), priority=50)

    return _FakeEntryPoint(name="fake", value="tests.fake_unconfigured_ace_music:register", obj=register)


def _make_fake_timeout_ace_music_plugin_ep():
    def register(registry):
        class _Music:
            backend_id = "abstractmusic:acemusic"

            def t2m(self, prompt: str, **kwargs):
                _ = prompt, kwargs
                raise RuntimeError("ACE Music API request failed with HTTP 504: gateway time-out")

        registry.register_music_backend(backend_id="abstractmusic:acemusic", factory=lambda _owner: _Music(), priority=50)

    return _FakeEntryPoint(name="fake", value="tests.fake_timeout_ace_music:register", obj=register)


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
    assert 'pip install -U abstractcore' in data["error"]["message"]


def test_audio_music_happy_path_with_stubbed_plugin(client, monkeypatch):
    calls = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_music_plugin_ep(calls)]))
    _reset_audio_core(monkeypatch)

    resp = client.post(
        "/v1/audio/music",
        json={
            "prompt": "hello",
            "format": "wav",
            "duration_s": 1,
            "model": "ACE-Step/acestep-v15-xl-turbo-diffusers",
            "seed": 7,
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/wav")
    assert resp.content == b"wav-bytes"
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["duration_s"] == 1.0
    assert calls[0]["model"] == "ACE-Step/acestep-v15-xl-turbo-diffusers"
    assert calls[0]["seed"] == 7


def test_provider_scoped_audio_music_selects_backend_and_forwards_music_params(client, monkeypatch):
    calls = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_multi_music_plugin_ep(calls)]))
    _reset_audio_core(monkeypatch)

    resp = client.post(
        "/diffusers/v1/audio/music",
        json={"input": "ambient pulse", "format": "wav", "num_inference_steps": 12, "guidance_scale": 4.5},
    )

    assert resp.status_code == 200
    assert resp.content == b"abstractmusic:diffusers:wav"
    assert calls[0]["backend_id"] == "abstractmusic:diffusers"
    assert calls[0]["prompt"] == "ambient pulse"
    assert calls[0]["num_inference_steps"] == 12
    assert calls[0]["guidance_scale"] == 4.5


@pytest.mark.parametrize(
    ("url", "body"),
    [
        ("/v1/audio/music", {"prompt": "remote pulse", "provider": "acemusic", "format": "mp3"}),
        ("/acemusic/v1/audio/music", {"prompt": "remote pulse", "format": "mp3"}),
    ],
)
def test_audio_music_provider_selects_acemusic_and_allow_mp3(client, monkeypatch, url, body):
    calls = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_multi_music_plugin_ep(calls)]))
    _reset_audio_core(monkeypatch)

    resp = client.post(url, json=body)

    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/mpeg")
    assert resp.content == b"abstractmusic:acemusic:mp3"
    assert calls[0]["backend_id"] == "abstractmusic:acemusic"
    assert calls[0]["format"] == "mp3"


def test_audio_music_rejects_legacy_backend_fields(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_music_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp = client.post("/v1/audio/music", json={"prompt": "hello", "backend": "acemusic", "format": "wav"})

    assert resp.status_code == 422
    body = resp.json()
    message = body.get("detail") or (body.get("error") or {}).get("message") or ""
    assert "provider" in message


def test_audio_music_missing_ace_key_is_service_configuration_error(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_unconfigured_ace_music_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp = client.post("/v1/audio/music", json={"prompt": "remote pulse", "provider": "acemusic", "format": "wav"})

    assert resp.status_code == 503
    assert "ACEMUSIC_API_KEY" in resp.json()["error"]["message"]


def test_audio_music_upstream_timeout_preserves_gateway_status(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_timeout_ace_music_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp = client.post("/v1/audio/music", json={"prompt": "remote pulse", "provider": "acemusic", "format": "wav"})

    assert resp.status_code == 504
    assert "HTTP 504" in resp.json()["error"]["message"]


def _make_fake_detailed_music_plugin_ep(details_payload):
    def register(registry):
        class _Music:
            backend_id = "fake-music"

            def available_providers(self, task=None):
                return [{"provider_id": "acestep", "tasks": [task], "local": True}]

            def provider_details(self, *, task=None):
                _ = task
                return details_payload

            def t2m(self, prompt: str, **kwargs):
                _ = prompt, kwargs
                return b"wav-bytes"

        registry.register_music_backend(backend_id="fake-music", factory=lambda _owner: _Music(), priority=0)

    return _FakeEntryPoint(name="fake", value="tests.fake_detailed_music:register", obj=register)


def test_music_provider_details_route_reports_unusable_providers_with_reasons(client, monkeypatch):
    details_payload = [
        {"provider_id": "acestep", "usable": True, "metadata": {"reason": "", "cached_models": ["ACE-Step/acestep-v15-xl-turbo-diffusers"]}},
        {"provider_id": "acemusic", "usable": False, "metadata": {"reason": "no API key configured", "cached_models": []}},
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _EntryPoints([_make_fake_detailed_music_plugin_ep(details_payload)]),
    )
    _reset_audio_core(monkeypatch)

    resp = client.get("/v1/audio/music/provider-details")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["operation"] == "music_provider_details"
    assert body["capability"] == "music"
    assert body["task"] == "text_to_music"
    assert body["provider_details"] == details_payload


def test_music_provider_details_route_501_when_backend_lacks_method(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_music_plugin_ep()]))
    _reset_audio_core(monkeypatch)

    resp = client.get("/v1/audio/music/provider-details")

    assert resp.status_code == 501
    body = resp.json()
    assert body["ok"] is False
    assert "provider_details" in body["error"]
    assert "abstractmusic >= 0.1.14" in body["error"]


def test_music_provider_details_route_501_when_plugin_unavailable(client, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))
    _reset_audio_core(monkeypatch)

    resp = client.get("/v1/audio/music/provider-details")

    assert resp.status_code == 501
    body = resp.json()
    assert body["ok"] is False
    assert 'pip install -U abstractcore' in body["error"]


# ------------------------------------------------ server-held music keys (interim)
_MUSIC_REFUSAL = "Server-held music ("


def _owner_seeing_plugin(seen: list):
    """A plugin recording the key each call would authenticate with: the
    owner config key AbstractMusic reads first, else its env fallback."""
    import os

    def register(registry):
        class _Music:
            backend_id = "abstractmusic:acemusic"

            def __init__(self, owner):
                self._owner = owner

            def t2m(self, prompt: str, **kwargs):
                cfg = getattr(self._owner, "config", {}) or {}
                seen.append(cfg.get("music_acemusic_api_key") or os.environ.get("ACEMUSIC_API_KEY"))
                return b"wav-bytes"

        registry.register_music_backend(backend_id="abstractmusic:acemusic", factory=lambda owner: _Music(owner), priority=0)

    return _FakeEntryPoint(name="fake", value="tests.fake_music_owner:register", obj=register)


@pytest.fixture()
def unauthenticated_music(monkeypatch):
    import abstractcore.server.audio_endpoints as ae

    for name in ("ABSTRACTCORE_AUTH_TOKEN", "ACEMUSIC_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr(ae, "_CORE", None)
    seen: list = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_owner_seeing_plugin(seen)]))
    return TestClient(app), seen


@pytest.mark.parametrize("env_var", ["ACEMUSIC_API_KEY", "ELEVENLABS_API_KEY"])
@pytest.mark.parametrize("path", ["/v1/audio/music", "/acemusic/v1/audio/music", "/diffusers/v1/audio/music"])
def test_unauthenticated_music_never_spends_a_server_held_key(unauthenticated_music, monkeypatch, env_var, path):
    """Fail closed whatever backend is named (interim until AbstractMusic says
    which backends are remote), also with ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1."""
    client, seen = unauthenticated_music
    monkeypatch.setenv(env_var, "sk-SENTINEL-MUSIC")
    resp = client.post(path, json={"prompt": "x"})
    assert resp.status_code == 401, resp.text[:200]
    assert _MUSIC_REFUSAL in resp.json()["error"]["message"] and env_var in resp.json()["error"]["message"]
    assert seen == []


def test_a_music_key_held_in_the_capability_config_counts(unauthenticated_music, monkeypatch):
    import abstractcore.server.audio_endpoints as ae

    client, seen = unauthenticated_music
    real = ae._capability_config
    monkeypatch.setattr(ae, "_capability_config", lambda: {**real(), "music_elevenlabs_api_key": "sk-SENTINEL-MUSIC"})
    assert client.post("/v1/audio/music", json={"prompt": "x"}).status_code == 401
    assert seen == []


def test_local_music_without_server_held_keys_runs_unauthenticated(unauthenticated_music):
    client, seen = unauthenticated_music
    assert client.post("/v1/audio/music", json={"prompt": "x"}).status_code == 200
    assert seen == [None]


def test_a_caller_key_is_spent_instead_of_the_servers_music_key(unauthenticated_music, monkeypatch):
    client, seen = unauthenticated_music
    monkeypatch.setenv("ACEMUSIC_API_KEY", "sk-SENTINEL-MUSIC")
    resp = client.post("/v1/audio/music", json={"prompt": "x"}, headers={"X-AbstractCore-Provider-API-Key": "sk-caller-own"})
    assert resp.status_code == 200, resp.text[:200]
    assert seen == ["sk-caller-own"]


def test_authenticated_music_may_spend_the_server_key(unauthenticated_music, monkeypatch):
    client, seen = unauthenticated_music
    monkeypatch.setenv("ACEMUSIC_API_KEY", "sk-SENTINEL-MUSIC")
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "srv-token-test")
    resp = TestClient(app).post("/v1/audio/music", json={"prompt": "x"}, headers={"Authorization": "Bearer srv-token-test"})
    assert resp.status_code == 200, resp.text[:200]
    assert seen == ["sk-SENTINEL-MUSIC"]


def test_music_server_keys_match_what_abstractmusic_reads():
    """The constant is AbstractMusic's complete set of provider keys (when its source is here)."""
    import pathlib
    import re

    from abstractcore.server.credentials import MUSIC_SERVER_KEYS

    am = pytest.importorskip("abstractmusic")
    # The keys compared are those of the AbstractMusic this release requires
    # (pyproject: abstractmusic>=0.1.15); an older install declares fewer and
    # the set mismatch would read as a code bug. Name the real cause instead.
    from packaging.version import Version

    installed = importlib.metadata.version("abstractmusic")
    assert Version(installed) >= Version("0.1.15"), (
        f"abstractmusic {installed} is installed; this AbstractCore requires >=0.1.15 "
        "(upgrade the test environment: pip install -U 'abstractmusic>=0.1.15')"
    )
    root = pathlib.Path(am.__file__).parent
    found = set()
    for path in root.rglob("*.py"):
        found |= set(re.findall(r'"([A-Z0-9_]+_API_KEY)"', path.read_text(encoding="utf-8")))
    assert found == set(MUSIC_SERVER_KEYS)


def test_a_selected_backend_that_is_not_registered_never_falls_back(client, monkeypatch):
    """Only ACE Music is registered: `/diffusers/...` must not run it instead."""
    calls = []

    def register(registry):
        class _Music:
            backend_id = "abstractmusic:acemusic"

            def t2m(self, prompt, **kwargs):
                calls.append(kwargs)
                return b"remote"

        registry.register_music_backend(backend_id="abstractmusic:acemusic", factory=lambda _o: _Music(), priority=50)

    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(name="f", value="t:r", obj=register)])
    )
    _reset_audio_core(monkeypatch)
    resp = client.post("/diffusers/v1/audio/music", json={"prompt": "x"}, headers={"X-AbstractCore-Provider-API-Key": "sk-caller-own"})
    assert resp.status_code == 501, resp.text[:200]
    assert calls == []


@pytest.mark.parametrize("path, body", [
    ("/elevenlabs/v1/audio/music", {"prompt": "x"}),
    ("/v1/audio/music", {"prompt": "x", "provider": "elevenlabs"}),
    ("/v1/audio/music", {"prompt": "x", "provider": "ace-music"}),
])
def test_an_unknown_music_selector_is_refused_never_rerouted(unauthenticated_music, path, body):
    """AbstractMusic's ElevenLabs selector is `elevenlabs-music`; `elevenlabs` is unknown."""
    client, seen = unauthenticated_music
    resp = client.post(path, json=body, headers={"X-AbstractCore-Provider-API-Key": "sk-caller-own"})
    assert resp.status_code == 400, resp.text[:200]
    assert "Unknown music backend selector" in resp.json()["error"]["message"]
    assert seen == []


@pytest.mark.parametrize("path, body, reaches", [
    ("/acemusic/v1/audio/music", {"prompt": "x"}, {"music_acemusic_api_key"}),
    ("/v1/audio/music", {"prompt": "x", "provider": "elevenlabs-music"}, {"music_elevenlabs_api_key"}),
    ("/diffusers/v1/audio/music", {"prompt": "x"}, set()),
])
def test_a_caller_key_reaches_only_the_selected_backend(monkeypatch, path, body, reaches):
    import abstractcore.server.audio_endpoints as ae

    captured = {}

    def register(registry):
        class _Music:
            def __init__(self, owner, backend_id):
                self._owner, self.backend_id = owner, backend_id

            def t2m(self, prompt, **kwargs):
                cfg = getattr(self._owner, "config", {}) or {}
                captured["keys"] = {k for k in ("music_acemusic_api_key", "music_elevenlabs_api_key") if cfg.get(k) == "sk-caller-own"}
                return b"ok"

        for bid in ("abstractmusic:acemusic", "abstractmusic:elevenlabs-music", "abstractmusic:diffusers"):
            registry.register_music_backend(backend_id=bid, factory=lambda o, b=bid: _Music(o, b), priority=10)

    for name in ("ABSTRACTCORE_AUTH_TOKEN", "ACEMUSIC_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(name="f", value="t:r", obj=register)])
    )
    monkeypatch.setattr(ae, "_CORE", None)
    resp = TestClient(app).post(path, json=body, headers={"X-AbstractCore-Provider-API-Key": "sk-caller-own"})
    assert resp.status_code == 200, resp.text[:200]
    assert captured["keys"] == reaches
