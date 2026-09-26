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


# -- claims (review follow-up 2026-09-26) -------------------------------------------
class _Owner:
    def __init__(self, claims):
        self.claims = claims

    def residency_claims(self):
        if isinstance(self.claims, Exception):
            raise self.claims
        return list(self.claims)


def test_eject_unclaimed_skips_a_model_another_owner_claims_in_any_spelling(monkeypatch):
    ejects = []
    monkeypatch.setattr(pr, "eject", lambda b, m, reason="eject": ejects.append((b, m)) or {"ok": True})
    owner = _Owner([{"provider": "mlx", "model": "mlx-community/Qwen3-4B-4bit", "locked": True, "kind": "pool"}])
    pr.register_claimant(owner)
    try:
        for spelling in ("MLX-Community/qwen3-4b-4bit", "/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/x"):
            out = pr.eject_unclaimed("mlx", spelling, reason="t")
            assert out["skipped"] is True and out["locked"] is True, spelling
        assert ejects == []
        # another backend's claim does not block
        assert pr.eject_unclaimed("huggingface", "mlx-community/Qwen3-4B-4bit")["ok"] is True
        assert ejects == [("huggingface", "mlx-community/Qwen3-4B-4bit")]
    finally:
        pr.unregister_claimant(owner)
    assert pr.eject_unclaimed("mlx", "MLX-Community/qwen3-4b-4bit")["ok"] is True


def test_a_claimant_that_cannot_answer_counts_as_a_claim(monkeypatch):
    monkeypatch.setattr(pr, "eject", lambda *a, **k: pytest.fail("must not eject"))
    owner = _Owner(RuntimeError("boom"))
    pr.register_claimant(owner)
    try:
        out = pr.eject_unclaimed("mlx", "vendor/x")
        assert out["skipped"] is True and out["claims"][0]["kind"] == "claimant_error"
    finally:
        pr.unregister_claimant(owner)


def test_claimants_are_held_weakly_and_must_answer():
    import gc

    owner = _Owner([{"provider": "mlx", "model": "vendor/x", "locked": False, "kind": "pool"}])
    pr.register_claimant(owner)
    assert pr.claims_for("mlx", "vendor/x")
    del owner
    gc.collect()
    assert pr.claims_for("mlx", "vendor/x") == []
    with pytest.raises(TypeError):
        pr.register_claimant(object())
