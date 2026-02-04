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

