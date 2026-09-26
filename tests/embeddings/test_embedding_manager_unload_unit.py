"""Mission MEM2 (2026-09-25): an in-process embedding model could never leave
memory, and nothing listed it.

Two pins in `EmbeddingManager.__init__`: `atexit.register(self._method)` (a
strong reference for the life of the process) and a class-level
`@lru_cache` on `embed` whose keys carry `self`. A gateway that rebuilt its
embedder kept every previous SentenceTransformer on MPS (86 MB each for
MiniLM, measured), invisible to every listing and unreachable by any eject.

Now: weak atexit registration, a per-instance memo, `unload()` as the eject,
a process registry (`resident_embedding_models` / `eject_embedding_models`)
in the shared residency row shape. Fakes only (no sentence-transformers, no
torch). Mutants that turn these RED: restore the bound-method atexit or the
class-level lru_cache (manager never collectable); skip `empty_cache` in
`unload()` (pool stays).
"""
from __future__ import annotations

import gc
import sys
import types
import weakref
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fake_torch(monkeypatch):
    state = {"allocated": 0, "pool": 0, "empties": 0}

    class Storage:
        def __init__(self, nbytes):
            self._n = int(nbytes)

        def data_ptr(self):
            return id(self)

        def nbytes(self):
            return self._n

    class Tensor:
        def __init__(self, nbytes):
            self._storage = Storage(nbytes)
            self.device = "mps:0"
            state["allocated"] += int(nbytes)

        def untyped_storage(self):
            return self._storage

        def numel(self):
            return self._storage._n

        def element_size(self):
            return 1

        def __del__(self):
            n = self._storage._n
            state["allocated"] -= n
            state["pool"] += n

    def empty_cache():
        state["empties"] += 1
        state["pool"] = 0

    torch = types.ModuleType("torch")
    torch.Tensor = Tensor
    torch.mps = SimpleNamespace(current_allocated_memory=lambda: state["allocated"],
                                driver_allocated_memory=lambda: state["allocated"] + state["pool"],
                                empty_cache=empty_cache, synchronize=lambda: None)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return state, Tensor


@pytest.fixture
def fake_sentence_transformers(monkeypatch, fake_torch):
    state, Tensor = fake_torch
    import abstractcore.embeddings.manager as manager_mod
    from abstractcore.embeddings.models import EmbeddingBackend

    class SentenceTransformer:
        def __init__(self, source, **kwargs):
            self.model_name_or_path = source
            self._weights = [Tensor(90_000)]
            self.device = "mps:0"

        def parameters(self):
            return iter(self._weights)

        def buffers(self):
            return iter(())

        def encode(self, text, **kwargs):
            return np.full(8, float(len(text)), dtype=np.float32)

        def get_sentence_embedding_dimension(self):
            return 8

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = SentenceTransformer
    monkeypatch.setattr(manager_mod, "sentence_transformers", fake)
    monkeypatch.setattr(manager_mod.EmbeddingManager, "_sentence_transformers_source", lambda self: ("fake/model", {}))
    monkeypatch.setattr(manager_mod.EmbeddingManager, "_select_backend", lambda self: EmbeddingBackend.PYTORCH)
    monkeypatch.setattr(manager_mod, "_EMBEDDING_MANAGERS", weakref.WeakSet())
    return manager_mod, state


def _manager(manager_mod, tmp_path, name="fake/model"):
    return manager_mod.EmbeddingManager(provider="huggingface", model=name, cache_dir=tmp_path / "emb")


def test_manager_is_collectable_after_use(fake_sentence_transformers, tmp_path):
    """MUTANT guard: `atexit.register(self._safe_save_*)` or the class-level
    `@lru_cache` keep the manager (and its weights) alive forever."""
    manager_mod, state = fake_sentence_transformers
    m = _manager(manager_mod, tmp_path)
    for text in ("alpha", "beta", "alpha"):
        m.embed(text)
    assert m.embed.cache_info().hits == 1, "the per-instance memo still memoizes"
    wm = weakref.ref(m)
    wmodel = weakref.ref(m.model)
    assert state["allocated"] == 90_000
    del m
    gc.collect()
    assert wm() is None, "the manager must be collectable once dropped (atexit / lru_cache pins)"
    assert wmodel() is None
    assert state["allocated"] == 0, "its SentenceTransformer weights went with it"


def test_unload_frees_weights_returns_pool_and_is_listed_truthfully(fake_sentence_transformers, tmp_path):
    manager_mod, state = fake_sentence_transformers
    m = _manager(manager_mod, tmp_path)
    m.embed("hello")
    rows = manager_mod.resident_embedding_models()
    assert len(rows) == 1 and rows[0]["models"] == ["fake/model"] and rows[0]["weights_bytes"] == 90_000
    assert rows[0]["lane"] == "embeddings" and rows[0]["holders"] == 1
    assert manager_mod.embeddings_memory_report()["held_bytes"] == 90_000

    report = m.unload()
    assert report["unloaded"] is True and report["loaded"] is False
    assert report["freed_weights_bytes"] == 90_000 and report["residual_weights_alive"] is False
    assert m.get_residency()["loaded"] is False
    assert manager_mod.resident_embedding_models() == []
    assert state["allocated"] == 0
    assert report["cache_cleared"] is True and state["empties"] >= 1
    assert state["pool"] == 0, "torch's MPS pool returned to the OS (MUTANT: skip empty_cache -> RED)"
    assert m.embed.cache_info().currsize == 0
    # idempotent
    again = m.unload()
    assert again["unloaded"] is False and again["loaded"] is False


def test_unload_reports_residual_when_weights_are_still_referenced(fake_sentence_transformers, tmp_path):
    manager_mod, state = fake_sentence_transformers
    m = _manager(manager_mod, tmp_path)
    keep = m.model                      # someone else holds the weights
    report = m.unload()
    assert report["unloaded"] is False and report["residual_weights_alive"] is True
    assert report["residual_weights_bytes"] == 90_000 and report.get("warnings")
    assert state["allocated"] == 90_000
    del keep
    gc.collect()
    assert state["allocated"] == 0


def test_eject_embedding_models_unloads_every_manager(fake_sentence_transformers, tmp_path):
    manager_mod, state = fake_sentence_transformers
    a = _manager(manager_mod, tmp_path)
    b = _manager(manager_mod, tmp_path)   # a rebuilt embedder: second full copy
    assert manager_mod.resident_embedding_models()[0]["holders"] == 2
    assert state["allocated"] == 180_000
    result = manager_mod.eject_embedding_models("fake/model")
    assert result["ok"] is True and len(result["holders_unloaded"]) == 2
    assert result["residual"] is None
    assert state["allocated"] == 0 and state["pool"] == 0
    assert a.model is None and b.model is None


def test_next_embed_after_unload_reloads_the_model_transparently(fake_sentence_transformers, tmp_path):
    """An eject is not a kill: the next NEW embedding loads the weights again
    (MUTANT: drop `_ensure_local_model` -> `None.encode` -> RED under strict)."""
    manager_mod, state = fake_sentence_transformers
    m = manager_mod.EmbeddingManager(provider="huggingface", model="fake/model", cache_dir=tmp_path / "emb", strict=True)
    m.embed("before")
    m.unload()
    assert m.model is None and state["allocated"] == 0
    vec = m.embed("a new text after the eject")
    assert len(vec) == 8 and m.model is not None and state["allocated"] == 90_000
    assert manager_mod.resident_embedding_models()[0]["holders"] == 1
    m.unload()
    assert len(m.embed_batch(["x", "yy"])) == 2 and m.model is not None
    m.unload()
    assert m.get_dimension() == 8 and m.model is not None


def test_unload_of_a_server_backed_embedder_keeps_its_client(monkeypatch, tmp_path):
    """Ollama / LM Studio / OpenAI-compatible embedders hold nothing in this
    process: `unload()` must not null their client (it broke every later
    embedding)."""
    import abstractcore.embeddings.manager as manager_mod

    class _Remote:
        def __init__(self, model, **kwargs):
            self.model = model

        def embed(self, input_text, **kwargs):
            items = input_text if isinstance(input_text, list) else [input_text]
            return {"data": [{"embedding": [0.5, 0.5]} for _ in items], "model": self.model}

    import abstractcore.providers.ollama_provider as ollama_mod
    monkeypatch.setattr(ollama_mod, "OllamaProvider", _Remote)
    m = manager_mod.EmbeddingManager(provider="ollama", model="nomic-embed-text", cache_dir=tmp_path / "emb", strict=True)
    client = m._provider_instance
    report = m.unload()
    assert report["unloaded"] is False and report["in_process"] is False
    assert m._provider_instance is client
    assert m.embed("still works") == [0.5, 0.5]
    assert manager_mod.eject_embedding_models()["holders_found"] == 0


# -- review S4 (2026-09-26): unload vs an embedding in flight ----------------------
def _blocking_manager(manager_mod, tmp_path, monkeypatch):
    import threading as _t

    gate, started = _t.Event(), _t.Event()
    m = manager_mod.EmbeddingManager(provider="huggingface", model="fake/model", cache_dir=tmp_path / "emb", strict=True)
    real_encode = type(m.model).encode

    def slow_encode(self, text, **kwargs):
        started.set()
        assert gate.wait(10)
        return real_encode(self, text, **kwargs)

    monkeypatch.setattr(type(m.model), "encode", slow_encode)
    return m, gate, started


def test_unload_during_an_embedding_reports_the_call_in_flight_and_frees_nothing(fake_sentence_transformers, tmp_path, monkeypatch):
    """MUTANT: no in-use count -> the unload drops the model under the running
    call and blames 'referenced elsewhere' -> RED."""
    import threading as _t

    manager_mod, state = fake_sentence_transformers
    m, gate, started = _blocking_manager(manager_mod, tmp_path, monkeypatch)
    out = {}
    worker = _t.Thread(target=lambda: out.setdefault("v", m.embed("in flight")))
    worker.start()
    assert started.wait(5)
    report = m.unload(drain_timeout_s=0.2)
    assert report["unloaded"] is False and report["in_flight"] == 1 and "still running" in report["reason"]
    assert m.model is not None, "nothing freed under the running call"
    gate.set()
    worker.join(5)
    assert len(out["v"]) == 8
    assert m.unload()["unloaded"] is True and state["allocated"] == 0


def test_an_unload_waits_for_the_running_embedding_then_frees(fake_sentence_transformers, tmp_path, monkeypatch):
    import threading as _t

    manager_mod, state = fake_sentence_transformers
    m, gate, started = _blocking_manager(manager_mod, tmp_path, monkeypatch)
    out = {}
    worker = _t.Thread(target=lambda: out.setdefault("v", m.embed("in flight")))
    worker.start()
    assert started.wait(5)
    unloader = _t.Thread(target=lambda: out.setdefault("report", m.unload(drain_timeout_s=10)))
    unloader.start()
    unloader.join(0.3)
    assert unloader.is_alive(), "the unload waits for the running call"
    gate.set()
    worker.join(5)
    unloader.join(5)
    assert len(out["v"]) == 8, "the running embed completed with its model"
    assert out["report"]["unloaded"] is True and m.model is None and state["allocated"] == 0
