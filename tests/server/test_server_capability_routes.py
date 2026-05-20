import importlib.metadata

from fastapi.testclient import TestClient


class _FakeEntryPoint:
    name = "fake-music"
    value = "tests.fake_music:register"

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


def test_server_capability_routes_use_music_env_backend(monkeypatch):
    def register(registry):
        class _Music:
            backend_id = "abstractmusic:diffusers"

            def t2m(self, prompt: str, **kwargs):
                _ = prompt, kwargs
                return b"wav"

        registry.register_music_backend(
            backend_id="abstractmusic:acestep-diffusers",
            factory=lambda _owner: _Music(),
            priority=30,
        )
        registry.register_music_backend(
            backend_id="abstractmusic:diffusers",
            factory=lambda _owner: _Music(),
            priority=0,
        )

    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setenv("ABSTRACTMUSIC_BACKEND", "diffusers")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(register)]))

    import abstractcore.server.audio_endpoints as audio_endpoints
    from abstractcore.server.app import app

    monkeypatch.setattr(audio_endpoints, "_CORE", None)
    response = TestClient(app).get("/v1/capabilities")

    assert response.status_code == 200
    assert response.json()["status"]["capabilities"]["music"]["selected_backend"] == "abstractmusic:diffusers"
