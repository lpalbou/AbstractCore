"""OVH 8080 pass four (2026-10-29):
R2 -- /v1/embeddings with "huggingface/Qwen/Qwen3-Embedding-0.6B-GGUF" reached sentence-transformers and
      answered 500 "Unrecognized model": a GGUF repo id goes to the llama.cpp lane.
R5 -- the GGUF embedder reserved n_batch = n_ubatch = 8,192 and held 6.6 GB of GPU memory, with
      device_bytes None: it now reserves the largest input that FITS (5 % buffer), shows it on the
      row, honours a set value exactly (or refuses), and measures device_bytes."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tests.config.test_embedding_route_alias_availability_unit import _gguf_snapshot  # noqa: E402

GIB = 1 << 30


def _fake_llama(built, monkeypatch, *, free_after=None):
    class FakeLlama:
        def __init__(self, model_path, n_ctx=512, n_batch=512, n_ubatch=512, flash_attn=False, **kw):
            built.append(dict(kw, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, flash_attn=flash_attn))

        def embed(self, text, normalize=True, truncate=True):
            return [0.6, 0.8]

        def tokenize(self, b, add_bos=True):
            return list(range(len(b.split())))

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama, llama_supports_gpu_offload=lambda: True))


def _geometry(monkeypatch):
    from abstractcore.providers import hf_residency
    # Qwen3-Embedding-0.6B: 28 layers, 16 heads, 8 kv heads x 128, embedding 1024.
    geom = {"block_count": 28, "head_count": 16, "head_count_kv": 8, "embedding_length": 1024,
            "key_length": 128, "value_length": 128, "context_length": 32768}
    monkeypatch.setattr(hf_residency, "gguf_file_geometry", lambda p: geom)


def test_a_gguf_repo_id_is_served_by_llama_cpp_never_sentence_transformers(tmp_path, monkeypatch):
    """R2. MUTANT: no '-GGUF' check -> sentence-transformers is asked -> RED."""
    from abstractcore.embeddings import manager as em

    _gguf_snapshot(tmp_path, monkeypatch, present=True)
    built = []
    _fake_llama(built, monkeypatch)

    def st_source(self):
        raise AssertionError("sentence-transformers was asked to read a GGUF build")

    monkeypatch.setattr(em, "sentence_transformers", object())
    monkeypatch.setattr(em.EmbeddingManager, "_sentence_transformers_source", st_source)
    monkeypatch.setattr(em, "_cuda_free_bytes", lambda: None)
    mgr = em.EmbeddingManager.__new__(em.EmbeddingManager)
    mgr.model_id = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    mgr._load_model_unshared()
    assert built and built[0]["embedding"] is True
    assert mgr.model_id == "Qwen/Qwen3-Embedding-0.6B-GGUF"
    assert list(mgr.model.encode(["a"]).shape) == [1, 2]


def _encoder(tmp_path, monkeypatch, *, free_gib, pinned=None, device=None, causal=True):
    from abstractcore.embeddings import manager as em
    from abstractcore.utils import model_cache

    monkeypatch.setattr(model_cache, "read_gguf_embedder_facts",
                        lambda p: {"architecture": "qwen3" if causal else "bert", "causal": causal, "vocab_size": 151669})

    path = tmp_path / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    path.write_bytes(b"G" * 1024)
    built = []
    _fake_llama(built, monkeypatch)
    _geometry(monkeypatch)
    readings = iter([int(free_gib * GIB), int((free_gib - 1.5) * GIB)])
    monkeypatch.setattr(em, "_cuda_free_bytes", lambda: next(readings))
    monkeypatch.setattr(em, "_memory_buffer", lambda: 0.05)
    enc = em._LlamaCppSentenceEncoder("Qwen/Qwen3-Embedding-0.6B-GGUF", path, n_ctx=8192, device=device, pinned=pinned)
    return enc, built


def test_a_causal_embedder_reads_its_full_input_in_small_micro_batches_with_flash_attention(tmp_path, monkeypatch):
    """The cause of the 6.6 GB (operator, 2026-10-29): n_ubatch = 8,192 reserved a 5.1 GB compute
    buffer. A causal embedder keeps n_ctx = n_batch = 8,192 (its full input) with 512-token
    micro-batches and flash attention. MUTANT: n_ubatch = n_ctx, or no flash attention -> RED."""
    enc, built = _encoder(tmp_path, monkeypatch, free_gib=2.0)
    assert enc.n_ctx == 8192 and enc.context_source is None, "the model's full input fits once the scratch is small"
    assert built[0]["n_ctx"] == built[0]["n_batch"] == 8192 and built[0]["n_ubatch"] == 512
    assert built[0]["flash_attn"] is True and enc.flash_attn is True


def test_a_bidirectional_encoder_keeps_the_whole_input_in_one_micro_batch(tmp_path, monkeypatch):
    enc, built = _encoder(tmp_path, monkeypatch, free_gib=40.0, causal=False)
    assert built[0]["n_ubatch"] == built[0]["n_ctx"] == 8192


def test_the_embedder_reserves_the_largest_input_that_fits(tmp_path, monkeypatch):
    """R5 safety net. 1.0 GiB free: 8,192 tokens (KV ~0.9 GB + compute) does not fit, a smaller
    multiple of 256 does. MUTANT: n_ctx = 8,192 regardless -> RED."""
    enc, built = _encoder(tmp_path, monkeypatch, free_gib=1.0)
    assert 512 <= enc.n_ctx < 8192 and enc.n_ctx % 256 == 0
    assert built[0]["n_ctx"] == built[0]["n_batch"] == enc.n_ctx and built[0]["n_ubatch"] == 512
    assert enc.context_source == "memory" and enc.context_reason == "limited by memory"
    assert enc.advertised_context_length == 8192 and enc.device == "gpu"
    assert enc.device_bytes == int(1.5 * GIB), "device_bytes measured (free before - after)"


def test_plenty_of_room_keeps_the_models_maximum(tmp_path, monkeypatch):
    enc, built = _encoder(tmp_path, monkeypatch, free_gib=40.0)
    assert enc.n_ctx == 8192 and enc.context_source is None


def test_a_set_maximum_input_is_used_exactly_or_refused(tmp_path, monkeypatch):
    """A value an admin set is a pin: used as is when it fits; on a GPU-only device that cannot hold
    it, refused in words (never shrunk)."""
    from abstractcore.embeddings.manager import EmbeddingDeviceUnavailable

    enc, built = _encoder(tmp_path, monkeypatch, free_gib=40.0, pinned=1024)
    assert enc.n_ctx == 1024 and built[0]["n_batch"] == 1024 and enc.context_source == "configured"
    with pytest.raises(EmbeddingDeviceUnavailable, match="is set to read 8,192 tokens"):
        _encoder(tmp_path, monkeypatch, free_gib=1.0, pinned=8192, device="gpu")


def test_a_text_longer_than_the_reserved_input_is_refused_never_cut(tmp_path, monkeypatch):
    from abstractcore.exceptions import ContextWindowExceededError

    enc, _ = _encoder(tmp_path, monkeypatch, free_gib=40.0, pinned=512)
    with pytest.raises(ContextWindowExceededError, match="split it before embedding it"):
        enc.encode(" ".join(["w"] * 600))


def test_the_row_shows_the_reserved_input(tmp_path, monkeypatch):
    from abstractcore.embeddings import manager as em

    enc, _ = _encoder(tmp_path, monkeypatch, free_gib=1.0)
    mgr = em.EmbeddingManager.__new__(em.EmbeddingManager)
    mgr.model_id, mgr.provider, mgr.model, mgr.device_preference = "Qwen/Qwen3-Embedding-0.6B-GGUF", "huggingface", enc, None
    claim = mgr.get_residency()
    assert claim["context_length"] == enc.n_ctx and claim["advertised_context_length"] == 8192
    assert claim["context_source"] == "memory" and claim["memory_buffer"] == 0.05
    assert claim["device_bytes"] == int(1.5 * GIB)
    assert claim["quant"] == "Q8_0" and claim["model_file"] == "Qwen3-Embedding-0.6B-Q8_0.gguf"
    assert claim["flash_attn"] is True and claim["micro_batch"] == 512
