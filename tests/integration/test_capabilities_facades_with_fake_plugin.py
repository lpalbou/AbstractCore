import importlib.metadata

import pytest

from abstractcore.core.interface import AbstractCoreInterface
from abstractcore.core.types import GenerateResponse


class _FakeEntryPoint:
    def __init__(self, *, obj):
        self.name = "fake-integration"
        self.value = "tests.fake_plugin_integration:register"
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


@pytest.mark.integration
def test_lazy_backend_instantiation(monkeypatch):
    calls = {"factory": 0}

    def register(registry):
        class _Voice:
            backend_id = "voice"

            def tts(self, text: str, **kwargs):
                return b"ok"

            def stt(self, audio, **kwargs):
                return "ok"

        def factory(_owner):
            calls["factory"] += 1
            return _Voice()

        registry.register_voice_backend(backend_id="voice", factory=factory, priority=0)

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints([_FakeEntryPoint(obj=register)]))

    llm = _DummyProvider(model="dummy")
    # Status loads plugins but should not instantiate the backend.
    status = llm.capabilities.status()
    assert status["capabilities"]["voice"]["available"] is True
    assert calls["factory"] == 0

    assert llm.voice.tts("hello") == b"ok"
    assert calls["factory"] == 1

