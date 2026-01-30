import importlib.metadata

import pytest

from abstractcore.core.interface import AbstractCoreInterface
from abstractcore.core.types import GenerateResponse
from abstractcore.capabilities.errors import CapabilityUnavailableError


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
        return GenerateResponse(content=str(prompt))

    def get_capabilities(self):
        return []

    def unload_model(self, model_name: str) -> None:
        return None


@pytest.mark.basic
def test_capabilities_status_empty(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    status = llm.capabilities.status()
    assert status["plugins_loaded"] is True
    assert status["capabilities"]["voice"]["available"] is False
    assert status["capabilities"]["voice"]["install_hint"] == "pip install abstractvoice"
    assert status["capabilities"]["vision"]["available"] is False
    assert status["capabilities"]["vision"]["install_hint"] == "pip install abstractvision"


@pytest.mark.basic
def test_missing_capability_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    with pytest.raises(CapabilityUnavailableError) as e:
        llm.voice.tts("hello")
    assert "voice:" in str(e.value)
    assert "pip install abstractvoice" in str(e.value)


@pytest.mark.basic
def test_plugin_registration_and_backend_resolution(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert llm.capabilities.status()["capabilities"]["voice"]["available"] is True
    assert llm.voice.tts("hello") == b"wav-bytes"
    assert llm.audio.transcribe(b"123") == "transcript"


@pytest.mark.basic
def test_preferred_backend_beats_priority(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_backend_plugin_ep()]))

    llm_default = _DummyProvider(model="dummy")
    assert llm_default.capabilities.status()["capabilities"]["voice"]["selected_backend"] == "b"

    llm_prefer_a = _DummyProvider(model="dummy", capabilities_preferred_backends={"voice": "a"})
    assert llm_prefer_a.capabilities.status()["capabilities"]["voice"]["selected_backend"] == "a"
    assert llm_prefer_a.voice.tts("hello") == b"a"


def _make_fake_plugin_ep():
    def register(registry):
        class _Voice:
            backend_id = "fake-voice"

            def tts(self, text: str, **kwargs):
                return b"wav-bytes"

            def stt(self, audio, **kwargs):
                return "transcript"

        class _Audio:
            backend_id = "fake-audio"

            def transcribe(self, audio, **kwargs):
                return "transcript"

        registry.register_voice_backend(
            backend_id="fake-voice",
            factory=lambda _owner: _Voice(),
            priority=0,
            description="Fake voice backend for tests",
        )
        registry.register_audio_backend(
            backend_id="fake-audio",
            factory=lambda _owner: _Audio(),
            priority=0,
            description="Fake audio backend for tests",
        )

    return _FakeEntryPoint(name="fake", value="tests.fake_plugin:register", obj=register)


def _make_multi_backend_plugin_ep():
    def register(registry):
        class _VoiceA:
            backend_id = "a"

            def tts(self, text: str, **kwargs):
                return b"a"

            def stt(self, audio, **kwargs):
                return "a"

        class _VoiceB:
            backend_id = "b"

            def tts(self, text: str, **kwargs):
                return b"b"

            def stt(self, audio, **kwargs):
                return "b"

        # Lower priority, but explicitly preferred in one test.
        registry.register_voice_backend(backend_id="a", factory=lambda _owner: _VoiceA(), priority=0)
        registry.register_voice_backend(backend_id="b", factory=lambda _owner: _VoiceB(), priority=10)

    return _FakeEntryPoint(name="fake-multi", value="tests.fake_plugin_multi:register", obj=register)

