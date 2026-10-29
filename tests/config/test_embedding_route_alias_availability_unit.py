"""An embedding route that names a catalog alias is probed (and downloaded) as the
Hugging Face repo the embedding manager loads, and a not-downloaded model is not an
ERROR log line.

Linux test server (2026-10-29): `embedding.text = huggingface/qwen3-embedding`; every
boot logged "[ERROR] ... Failed to load embedding model Qwen/Qwen3-Embedding-0.6B: ... not
in the local Hugging Face cache" once per account, while the console's Defaults grid read
"not checked" ("'qwen3-embedding' is not an <org>/<repo> Hugging Face reference") with no
Download action.
"""
from __future__ import annotations

import logging

import pytest

from abstractcore.config import model_materializer as mm
from abstractcore.exceptions import ModelNotFoundError


def test_alias_route_is_probed_as_its_repo(monkeypatch):
    seen = []

    def fake_probe(provider, artifact, *, base_url=None):
        seen.append((provider, artifact))
        return mm.ModelPresence(provider, artifact, mm.PRESENCE_ABSENT, evidence="test", downloadable=True)

    monkeypatch.setattr(mm, "probe", fake_probe)
    rows = mm.annotate_route_availability([
        {"key": "embedding.text", "provider": "huggingface", "model": "qwen3-embedding"},
    ])
    assert ("huggingface", "Qwen/Qwen3-Embedding-0.6B") in seen, f"MUTANT: alias probed as-is -> unknown -> RED ({seen})"
    assert rows[0]["download_artifact"] == "Qwen/Qwen3-Embedding-0.6B"
    assert rows[0]["availability"]["status"] == mm.PRESENCE_ABSENT and rows[0]["availability"]["downloadable"]


def test_a_repo_id_route_is_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr(mm, "probe", lambda p, a, base_url=None: (seen.append(a), mm.ModelPresence(p, a, mm.PRESENCE_INSTALLED, evidence="t"))[1])
    mm.annotate_route_availability([{"key": "embedding.text", "provider": "huggingface", "model": "org/emb"}])
    assert seen == ["org/emb"]


def test_not_downloaded_embedding_is_not_an_error_log(monkeypatch, caplog):
    """MUTANT: log ERROR before re-raising -> one ERROR per account per boot -> RED."""
    from abstractcore.embeddings import manager as em

    def not_cached(self):
        raise ModelNotFoundError("Embedding model 'Qwen/Qwen3-Embedding-0.6B' is not in the local Hugging Face cache; download it first.")

    monkeypatch.setattr(em, "sentence_transformers", object())
    monkeypatch.setattr(em.EmbeddingManager, "_sentence_transformers_source", not_cached)
    mgr = em.EmbeddingManager.__new__(em.EmbeddingManager)
    mgr.model_id = "Qwen/Qwen3-Embedding-0.6B"
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ModelNotFoundError):
        mgr._load_model()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], [r.getMessage() for r in caplog.records]
