"""`verify_inference` verdicts, with a fake provider (the real run is the slow test)."""

from __future__ import annotations

import types

import pytest

from abstractcore.config import model_materializer as mm
from abstractcore.config import model_verify


class _LLM:
    def __init__(self, content, speculation):
        self.content, self.speculation, self.unloaded = content, speculation, False

    def generate(self, prompt, **kw):
        return types.SimpleNamespace(content=self.content, metadata={"speculation": self.speculation})

    def unload_model(self, name):
        self.unloaded = True


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(mm, "probe", lambda p, a, **k: mm.ModelPresence(p, a, mm.PRESENCE_INSTALLED, location="/x"))


def _run(monkeypatch, content, speculation, artifact="mlx-works/Qwen3.5-9B-oQ4e-mtp"):
    llm = _LLM(content, speculation)
    import abstractcore

    monkeypatch.setattr(abstractcore, "create_llm", lambda provider, model: llm)
    return model_verify.verify_inference("mlx", artifact), llm


def test_sensible_answer_with_mtp_passes(monkeypatch, installed):
    report, llm = _run(monkeypatch, "It boils at 100 degrees Celsius.", {"used": True})
    assert report["ok"] and llm.unloaded


def test_garbage_fails(monkeypatch, installed):
    report, _ = _run(monkeypatch, "��格 с", {"used": True})
    assert not report["ok"]
    assert [c["name"] for c in report["checks"] if not c["ok"]] == ["answer"]


def test_companion_model_without_mtp_fails_and_says_why(monkeypatch, installed):
    report, _ = _run(monkeypatch, "100 degrees Celsius.", {"used": False, "message": "MTP acceleration off: companion X not downloaded"})
    assert not report["ok"]
    failed = [c for c in report["checks"] if not c["ok"]]
    assert failed[0]["name"] == "mtp_used" and "companion X not downloaded" in failed[0]["detail"]


def test_a_model_without_companion_does_not_require_mtp(monkeypatch, installed):
    report, _ = _run(monkeypatch, "100 degrees Celsius.", None, artifact="mlx-community/Llama-3.2-1B-Instruct-4bit")
    assert report["ok"]


def test_not_installed_never_loads(monkeypatch):
    monkeypatch.setattr(mm, "probe", lambda p, a, **k: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT, detail="gone"))
    import abstractcore

    monkeypatch.setattr(abstractcore, "create_llm", lambda *a, **k: pytest.fail("must not load"))
    report = model_verify.verify_inference("mlx", "org/x")
    assert not report["ok"] and report["summary"].startswith("not installed")
