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


# -- the installed GGUF build of the same model serves the route ------------------------
def _gguf_snapshot(tmp_path, monkeypatch, *, present: bool):
    from abstractcore.utils import model_cache

    import importlib.util as _iu
    _real = _iu.find_spec
    monkeypatch.setattr(_iu, "find_spec", lambda name, *a: object() if name == "llama_cpp" else _real(name, *a))
    snap = tmp_path / "models--Qwen--Qwen3-Embedding-0.6B-GGUF" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "Qwen3-Embedding-0.6B-Q8_0.gguf").write_bytes(b"GGUF")
    (snap / "Qwen3-Embedding-0.6B-f16.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(model_cache, "resolve_hf_load_snapshot",
                        lambda repo, cache_dirs=None: snap if present and repo == "Qwen/Qwen3-Embedding-0.6B-GGUF" else None)
    return snap


def test_route_is_installed_when_only_the_gguf_build_is(tmp_path, monkeypatch):
    """The coordinator's server fact: only Qwen/Qwen3-Embedding-0.6B-GGUF is there.
    MUTANT: no sibling check -> "not downloaded" for a model that runs -> RED."""
    _gguf_snapshot(tmp_path, monkeypatch, present=True)
    monkeypatch.setattr(mm, "probe", lambda p, a, base_url=None: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT, evidence="t", downloadable=True))
    row = mm.annotate_route_availability([{"key": "embedding.text", "provider": "huggingface", "model": "qwen3-embedding"}])[0]
    assert row["availability"]["status"] == mm.PRESENCE_INSTALLED
    assert "Qwen/Qwen3-Embedding-0.6B-GGUF" in row["availability"]["detail"]


def test_manager_loads_the_gguf_build_when_the_repo_is_missing(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from abstractcore.embeddings import manager as em

    snap = _gguf_snapshot(tmp_path, monkeypatch, present=True)
    built = []

    class FakeLlama:
        def __init__(self, model_path, **kw):
            built.append((model_path, kw))

        def embed(self, text, normalize=True):
            return [0.6, 0.8]

        def close(self):
            built.append("closed")

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr(em, "sentence_transformers", object())

    def not_cached(self):
        raise ModelNotFoundError("not in the local Hugging Face cache")

    monkeypatch.setattr(em.EmbeddingManager, "_sentence_transformers_source", not_cached)
    mgr = em.EmbeddingManager.__new__(em.EmbeddingManager)
    mgr.model_id = "Qwen/Qwen3-Embedding-0.6B"
    mgr._load_model()
    assert built[0][0] == str(snap / "Qwen3-Embedding-0.6B-f16.gguf"), "the most faithful build is chosen"
    assert built[0][1]["embedding"] is True
    assert built[0][1]["n_batch"] == built[0][1]["n_ctx"], "MUTANT: default n_batch -> texts cut to 512 tokens -> RED"
    assert built[0][1]["offload_kqv"] is False and built[0][1]["n_gpu_layers"] == 0, (
        "MUTANT: KV / compute buffers offloaded -> 4.9 GiB of GPU for a 0.6B embedder -> RED")
    assert built[0][1]["n_ctx"] <= 4096
    assert mgr.model_id == "Qwen/Qwen3-Embedding-0.6B-GGUF", "another build is another embedding space"
    assert list(mgr.model.encode(["a", "b"]).shape) == [2, 2] and mgr.model.get_sentence_embedding_dimension() == 2
    mgr.model = None
    mgr._load_model()  # a reload after unload() comes back to the same build
    assert len([b for b in built if b != "closed"]) == 2 and mgr.model_id == "Qwen/Qwen3-Embedding-0.6B-GGUF"


def test_without_the_gguf_build_it_is_still_not_downloaded(tmp_path, monkeypatch):
    from abstractcore.embeddings import manager as em

    _gguf_snapshot(tmp_path, monkeypatch, present=False)
    assert em.gguf_sibling_snapshot("Qwen/Qwen3-Embedding-0.6B") is None


def test_a_route_of_another_provider_is_never_served_by_a_hf_gguf(tmp_path, monkeypatch):
    _gguf_snapshot(tmp_path, monkeypatch, present=True)
    monkeypatch.setattr(mm, "probe", lambda p, a, base_url=None: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT, evidence="t"))
    row = mm.annotate_route_availability([{"key": "embedding.text", "provider": "vllm", "model": "Qwen/Qwen3-Embedding-0.6B"}])[0]
    assert row["availability"]["status"] == mm.PRESENCE_ABSENT
