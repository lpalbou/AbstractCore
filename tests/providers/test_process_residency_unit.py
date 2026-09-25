"""`abstractcore.providers.process_residency`: one door to the MLX, HuggingFace
and embeddings residency backends (mission M2). Fakes only."""
from __future__ import annotations

import sys
import types

import pytest

import abstractcore.providers.process_residency as pr


def test_backend_for_maps_in_process_providers_only():
    assert pr.backend_for("mlx") == "mlx"
    assert pr.backend_for("HuggingFace") == "huggingface"
    assert pr.backend_for("ollama") is None and pr.backend_for("lmstudio") is None
    assert pr.backend_for("huggingface", "embedding") == "embeddings"
    assert pr.backend_for(None, "embeddings") == "embeddings"


def test_a_backend_never_imported_is_skipped_and_ejects_nothing(monkeypatch):
    monkeypatch.delitem(sys.modules, "abstractcore.embeddings.manager", raising=False)
    assert pr.resident_rows("embeddings") == []
    report = pr.eject("embeddings", "m")
    assert report["ok"] is True and report["holders_found"] == 0
    assert "abstractcore.embeddings.manager" not in sys.modules, "a listing/eject must not import the backend"


def test_embeddings_rows_and_eject_dispatch_to_the_manager_module(monkeypatch):
    calls = []
    fake = types.ModuleType("abstractcore.embeddings.manager")
    fake.resident_embedding_models = lambda: [{"models": ["m"], "holders": 1, "held_bytes": 9, "weights_alive": True,
                                              "holder_rows": [{"id": 1}], "backend": "embeddings"}]
    fake.eject_embedding_models = lambda model, reason="eject": calls.append((model, reason)) or {"ok": True}
    monkeypatch.setitem(sys.modules, "abstractcore.embeddings.manager", fake)
    rows = pr.resident_rows("embeddings")
    assert rows == [{"models": ["m"], "holders": 1, "held_bytes": 9, "weights_alive": True, "backend": "embeddings"}]
    assert pr.eject("embeddings", "m", reason="x") == {"ok": True, "backend": "embeddings"}
    assert calls == [("m", "x")]


def test_unknown_backend_fails_loudly():
    with pytest.raises(ValueError):
        pr.resident_rows("vllm")
