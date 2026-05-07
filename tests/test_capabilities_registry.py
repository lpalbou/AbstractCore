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
    assert status["capabilities"]["voice"]["install_hint"] == 'pip install "abstractcore[voice]"'
    assert status["capabilities"]["vision"]["available"] is False
    assert status["capabilities"]["vision"]["install_hint"] == 'pip install "abstractcore[vision]"'


@pytest.mark.basic
def test_missing_capability_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    with pytest.raises(CapabilityUnavailableError) as e:
        llm.voice.tts("hello")
    assert "voice:" in str(e.value)
    assert 'pip install "abstractcore[voice]"' in str(e.value)


@pytest.mark.basic
def test_missing_music_capability_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([]))

    llm = _DummyProvider(model="dummy")
    with pytest.raises(CapabilityUnavailableError) as e:
        llm.music.t2m("hello")
    assert "music:" in str(e.value)
    assert "pip install abstractmusic" in str(e.value)


@pytest.mark.basic
def test_plugin_registration_and_backend_resolution(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_fake_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert llm.capabilities.status()["capabilities"]["voice"]["available"] is True
    assert llm.voice.tts("hello") == b"wav-bytes"
    assert llm.audio.transcribe(b"123") == "transcript"
    assert llm.music.t2m("hello") == b"mp3-bytes"


@pytest.mark.basic
def test_preferred_backend_beats_priority(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_multi_backend_plugin_ep()]))

    llm_default = _DummyProvider(model="dummy")
    assert llm_default.capabilities.status()["capabilities"]["voice"]["selected_backend"] == "b"

    llm_prefer_a = _DummyProvider(model="dummy", capabilities_preferred_backends={"voice": "a"})
    assert llm_prefer_a.capabilities.status()["capabilities"]["voice"]["selected_backend"] == "a"
    assert llm_prefer_a.voice.tts("hello") == b"a"


@pytest.mark.basic
def test_voice_facade_uses_remote_adapter_voice_without_treating_it_as_clone(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_make_remote_voice_plugin_ep()]))

    llm = _DummyProvider(model="dummy")
    assert llm.voice.tts("hello", voice="coral", format="wav", speed=1.1) == b"remote:coral:wav:1.1"


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

        class _Music:
            backend_id = "fake-music"

            def t2m(self, prompt: str, **kwargs):
                return b"mp3-bytes"

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
        registry.register_music_backend(
            backend_id="fake-music",
            factory=lambda _owner: _Music(),
            priority=0,
            description="Fake music backend for tests",
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


def _make_remote_voice_plugin_ep():
    def register(registry):
        class _Adapter:
            def is_available(self):
                return True

            def synthesize_to_bytes_with_voice(self, text: str, *, format: str, voice: str, speed=None, instructions=None):
                _ = (text, instructions)
                return f"remote:{voice}:{format}:{speed}".encode()

        class _VM:
            tts_adapter = _Adapter()

            def get_cloned_voice(self, voice_id: str):
                _ = voice_id
                return None

        class _Voice:
            backend_id = "remote-voice"

            def _get_vm(self):
                return _VM()

            def tts(self, text: str, **kwargs):
                raise AssertionError("built-in remote voices should use the adapter path")

            def stt(self, audio, **kwargs):
                return "transcript"

        registry.register_voice_backend(
            backend_id="remote-voice",
            factory=lambda _owner: _Voice(),
            priority=0,
        )

    return _FakeEntryPoint(name="fake-remote-voice", value="tests.fake_remote_voice:register", obj=register)
