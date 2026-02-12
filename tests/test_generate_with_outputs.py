import importlib.metadata

import pytest

from abstractcore.core.interface import AbstractCoreInterface
from abstractcore.core.types import GenerateResponse


class _FakeEntryPoint:
    def __init__(self, *, obj):
        self.name = "fake"
        self.value = "tests.fake_plugin_outputs:register"
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
        return GenerateResponse(content="LLM says: hello")

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        return None


@pytest.mark.basic
def test_generate_with_outputs_tts(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    res = llm.generate_with_outputs("hi", outputs={"tts": {"format": "wav"}})
    assert res.response.content == "LLM says: hello"
    assert res.outputs["tts"] == b"audio"


@pytest.mark.basic
def test_generate_with_outputs_t2i(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    res = llm.generate_with_outputs("hi", outputs={"t2i": {"width": 64, "height": 64}})
    assert res.outputs["t2i"] == b"png"


@pytest.mark.basic
def test_generate_with_outputs_t2m(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    res = llm.generate_with_outputs("hi", outputs={"t2m": {"format": "mp3"}})
    assert res.outputs["t2m"] == b"mp3"


@pytest.mark.basic
def test_generate_with_outputs_stream_not_supported(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    with pytest.raises(ValueError):
        llm.generate_with_outputs("hi", outputs={"tts": {}}, stream=True)


def _make_plugin_ep():
    def register(registry):
        class _Voice:
            backend_id = "voice"

            def tts(self, text: str, **kwargs):
                return b"audio"

            def stt(self, audio, **kwargs):
                return "text"

        class _Vision:
            backend_id = "vision"

            def t2i(self, prompt: str, **kwargs):
                return b"png"

            def i2i(self, prompt: str, image, **kwargs):
                return b"png"

            def t2v(self, prompt: str, **kwargs):
                return b"mp4"

            def i2v(self, image, **kwargs):
                return b"mp4"

        class _Music:
            backend_id = "music"

            def t2m(self, prompt: str, **kwargs):
                return b"mp3"

        registry.register_voice_backend(backend_id="voice", factory=lambda _owner: _Voice())
        registry.register_vision_backend(backend_id="vision", factory=lambda _owner: _Vision())
        registry.register_music_backend(backend_id="music", factory=lambda _owner: _Music())

    return _FakeEntryPoint(obj=register)

