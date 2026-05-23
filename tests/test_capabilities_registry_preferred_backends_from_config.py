import importlib.metadata
from types import SimpleNamespace

import pytest

from abstractcore.core.interface import AbstractCoreInterface
from abstractcore.core.types import GenerateResponse


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


class _DummyProvider(AbstractCoreInterface):
    def generate(self, prompt: str, **kwargs):
        _ = kwargs
        return GenerateResponse(content=str(prompt))

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        _ = model_name
        return None


def _make_multi_audio_backend_plugin_ep():
    def register(registry):
        class _AudioA:
            backend_id = "a"

            def transcribe(self, audio, **kwargs):
                _ = audio, kwargs
                return "a"

        class _AudioB:
            backend_id = "b"

            def transcribe(self, audio, **kwargs):
                _ = audio, kwargs
                return "b"

        # Higher priority default.
        registry.register_audio_backend(backend_id="a", factory=lambda _owner: _AudioA(), priority=0)
        registry.register_audio_backend(backend_id="b", factory=lambda _owner: _AudioB(), priority=10)

    return _FakeEntryPoint(name="fake-multi-audio", value="tests.fake_multi_audio:register", obj=register)


def _make_multi_music_backend_plugin_ep():
    def register(registry):
        class _MusicAceStep:
            backend_id = "abstractmusic:acestep"

            def t2m(self, prompt, **kwargs):
                _ = prompt, kwargs
                return b"acestep"

        class _MusicRemote:
            backend_id = "abstractmusic:acemusic"

            def t2m(self, prompt, **kwargs):
                _ = prompt, kwargs
                return b"remote"

        registry.register_music_backend(
            backend_id="abstractmusic:acestep",
            factory=lambda _owner: _MusicAceStep(),
            priority=20,
        )
        registry.register_music_backend(
            backend_id="abstractmusic:acemusic",
            factory=lambda _owner: _MusicRemote(),
            priority=0,
        )

    return _FakeEntryPoint(name="fake-multi-music", value="tests.fake_multi_music:register", obj=register)


@pytest.mark.basic
def test_config_audio_stt_backend_id_selects_audio_backend(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_audio_backend_plugin_ep()]))

    import abstractcore.config.manager as config_manager_module

    monkeypatch.setattr(
        config_manager_module,
        "get_config_manager",
        lambda: SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(stt_backend_id="a"))),
    )

    llm = _DummyProvider(model="dummy")
    assert llm.capabilities.status()["capabilities"]["audio"]["selected_backend"] == "a"
    assert llm.audio.transcribe(b"123") == "a"


@pytest.mark.basic
def test_explicit_preferred_backend_overrides_config_default(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_audio_backend_plugin_ep()]))

    import abstractcore.config.manager as config_manager_module

    monkeypatch.setattr(
        config_manager_module,
        "get_config_manager",
        lambda: SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(stt_backend_id="a"))),
    )

    llm = _DummyProvider(model="dummy", capabilities_preferred_backends={"audio": "b"})
    assert llm.capabilities.status()["capabilities"]["audio"]["selected_backend"] == "b"
    assert llm.audio.transcribe(b"123") == "b"


@pytest.mark.basic
def test_music_backend_acestep_name_selects_internal_backend(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_music_backend_plugin_ep()]))

    import abstractcore.config.manager as config_manager_module

    monkeypatch.setattr(
        config_manager_module,
        "get_config_manager",
        lambda: SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(stt_backend_id=None))),
    )

    llm = _DummyProvider(model="dummy", music_backend="acestep")
    assert llm.capabilities.status()["capabilities"]["music"]["selected_backend"] == "abstractmusic:acestep"
    assert llm.music.t2m("bright melodic synth loop") == b"acestep"


@pytest.mark.basic
@pytest.mark.parametrize("selector", ["acemusic", "abstractmusic:acemusic"])
def test_music_backend_acemusic_exact_selector_selects_remote_backend(monkeypatch, selector):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_music_backend_plugin_ep()]))

    import abstractcore.config.manager as config_manager_module

    monkeypatch.setattr(
        config_manager_module,
        "get_config_manager",
        lambda: SimpleNamespace(config=SimpleNamespace(audio=SimpleNamespace(stt_backend_id=None))),
    )

    llm = _DummyProvider(model="dummy", music_backend=selector)
    assert llm.capabilities.status()["capabilities"]["music"]["selected_backend"] == "abstractmusic:acemusic"
    assert llm.music.t2m("bright melodic synth loop") == b"remote"
