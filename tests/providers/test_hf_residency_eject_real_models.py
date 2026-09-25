"""Real-model proof for mission MEM2: load -> use -> a second holder -> eject
through `hf_residency.eject_model` -> live bytes back to baseline.

Needs tiny models from the Hub (a few MB): marked `network`, run with
`pytest --allow-network`. Downloads land in the per-test HOME the conftest
isolates. Measures torch's MPS allocator when MPS is available (this is where
the weights live on a Mac), else the object-level truth (no live provider,
no live engine, weights collected) plus the registry.
"""
from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import pytest

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_GGUF_REPO, TINY_GGUF_FILE = "ggml-org/models", "tinyllamas/stories260K.gguf"
TINY_ST = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def _hub_cache() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _mps_allocated():
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.synchronize()
            return int(torch.mps.current_allocated_memory())
    except Exception:
        pass
    return None


def _live(type_name: str, predicate=lambda o: True) -> int:
    return sum(1 for o in gc.get_objects() if type(o).__name__ == type_name and predicate(o))


@pytest.mark.network("downloads hf-internal-testing/tiny-random-LlamaForCausalLM (~2 MB)")
def test_transformers_two_holders_eject_to_baseline():
    from huggingface_hub import snapshot_download

    snapshot_download(TINY_LLAMA, cache_dir=str(_hub_cache()))
    from abstractcore import create_llm
    from abstractcore.providers import hf_residency

    before_rows = len(hf_residency.resident_models())
    pool = create_llm("huggingface", model=TINY_LLAMA, max_tokens=256)
    sibling = create_llm("huggingface", model=TINY_LLAMA, max_tokens=256)   # the summarizer's copy
    baseline_mps = None
    for holder in (pool, sibling):
        holder.generate("hello", max_output_tokens=3)
    row = hf_residency.process_residency_for(TINY_LLAMA)
    assert row is not None and row["holders"] == 2 and row["copies"] == 2
    assert row["weights_bytes"] and row["weights_bytes"] > 0
    weights_ref = weakref.ref(pool.model_instance)

    result = hf_residency.eject_model(TINY_LLAMA)
    assert result["ok"] is True and len(result["holders_unloaded"]) == 2, result
    assert result["residual"] is None
    assert hf_residency.process_residency_for(TINY_LLAMA) is None
    assert len(hf_residency.resident_models()) == before_rows
    del row
    gc.collect()
    assert weights_ref() is None, "the model object itself must be collected"
    assert _live("HuggingFaceProvider", lambda o: o.model_instance is not None or o.pipeline is not None) == 0
    mps = _mps_allocated()
    if mps is not None:
        # Tiny model: after the eject nothing of it may remain allocated. torch keeps a few
        # KB of scratch per stream; the two copies were >> 1 MB apiece.
        assert mps < 1 << 20, f"torch MPS still holds {mps} bytes after the eject"


@pytest.mark.network("downloads ggml-org/models tinyllamas/stories260K.gguf (~1 MB)")
def test_gguf_two_engines_eject_to_baseline():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TINY_GGUF_REPO, TINY_GGUF_FILE, cache_dir=str(_hub_cache()))
    from abstractcore import create_llm
    from abstractcore.providers import hf_residency

    pool = create_llm("huggingface", model=path, max_tokens=256)
    sibling = create_llm("huggingface", model=path, max_tokens=256)
    for holder in (pool, sibling):
        holder.generate("Once upon a time", max_output_tokens=4)
    assert pool.model_type == "gguf" and pool.llm is not None
    row = hf_residency.process_residency_for(path)
    assert row is not None and row["lane"] == "gguf" and row["holders"] == 2
    assert row["weights_bytes"] and row["weights_bytes"] > 0
    engine_ref = weakref.ref(pool.llm)

    result = hf_residency.eject_model(path)
    assert result["ok"] is True and len(result["holders_unloaded"]) == 2, result
    assert hf_residency.process_residency_for(path) is None
    del row
    gc.collect()
    assert engine_ref() is None, "the llama.cpp engine must be collected"
    live_engines = _live("Llama", lambda o: getattr(getattr(o, "_ctx", None), "ctx", None) is not None)
    assert live_engines == 0, f"{live_engines} llama.cpp context(s) still alive after the eject"
    assert hf_residency.hf_memory_report()["llama_cpp_bytes"] == 0


@pytest.mark.network("downloads sentence-transformers-testing/stsb-bert-tiny-safetensors (~35 MB)")
def test_embedding_manager_unload_frees_weights(tmp_path):
    from huggingface_hub import snapshot_download

    snapshot_download(TINY_ST, cache_dir=str(_hub_cache()))
    from abstractcore.embeddings import manager as manager_mod

    m = manager_mod.EmbeddingManager(provider="huggingface", model=TINY_ST, cache_dir=tmp_path / "emb")
    m.embed("a sentence")
    m.embed_batch(["one", "two"])
    rows = manager_mod.resident_embedding_models()
    assert any(TINY_ST in r["models"] for r in rows)
    weights_ref = weakref.ref(m.model)
    mps_loaded = _mps_allocated()

    report = m.unload()
    assert report["unloaded"] is True and report["residual_weights_alive"] is False, report
    gc.collect()
    assert weights_ref() is None
    assert not any(TINY_ST in r["models"] for r in manager_mod.resident_embedding_models())
    mps = _mps_allocated()
    if mps is not None and mps_loaded:
        assert mps < mps_loaded and mps < 1 << 20, f"torch MPS still holds {mps} bytes after the unload"
    wm = weakref.ref(m)
    del m
    gc.collect()
    assert wm() is None, "the manager is collectable (no atexit / lru_cache pin)"
