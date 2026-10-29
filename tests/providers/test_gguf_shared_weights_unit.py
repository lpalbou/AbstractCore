"""ONE llama.cpp weights instance per GGUF in a process (ADR 0011; residency audit R1, 2026-10-29).

A fake `llama_cpp._internals` stands in for llama-cpp-python: its `Llama` builds the model through
`internals.LlamaModel(...)` exactly as the real one does, and counts the loads.
MUTANT: `shared_llama` returning `llama_cls` unchanged -> N loads -> RED.
"""
from __future__ import annotations

import gc
import logging
import sys
import threading
import types

import pytest

from abstractcore.exceptions import InvalidRequestError


@pytest.fixture()
def fake_llama(monkeypatch):
    internals = types.ModuleType("llama_cpp._internals")
    loads = []

    class LlamaModel:
        def __init__(self, path_model, params=None, verbose=False):
            self.path = path_model
            self.model = object()  # the C pointer stand-in
            self.closed = False
            loads.append(self)

        def close(self):
            self.closed = True

    internals.LlamaModel = LlamaModel
    pkg = types.ModuleType("llama_cpp")
    pkg._internals = internals

    class Llama:
        def __init__(self, model_path, n_ctx=512, n_gpu_layers=-1, use_mmap=True, **_):
            self.model_path = model_path
            self._n_ctx = n_ctx
            self._model = sys.modules["llama_cpp._internals"].LlamaModel(path_model=model_path)
            self.closed = False

        def n_ctx(self):
            return self._n_ctx

        def close(self):
            if not self.closed:
                self.closed = True
                self._model.close()

        def __del__(self):  # as llama-cpp-python's Llama
            self.close()

    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp._internals", internals)
    from abstractcore.providers import gguf_shared

    monkeypatch.setattr(gguf_shared, "_INSTALLED", False)
    monkeypatch.setattr(gguf_shared, "_WEIGHTS", {})
    return gguf_shared, Llama, loads


def test_n_engines_share_one_weights_instance(fake_llama, tmp_path):
    gguf_shared, Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    build = gguf_shared.shared_llama(Llama)
    engines = [build(model_path=str(path), n_ctx=4096, n_gpu_layers=-1) for _ in range(4)]
    assert len(loads) == 1, "one LlamaModel for four contexts"
    assert len({gguf_shared.weights_identity(e) for e in engines}) == 1
    assert [e._abstractcore_weights_borrowed for e in engines] == [False, True, True, True]
    assert gguf_shared.report() == [{"key": gguf_shared.key_for_kwargs({"model_path": str(path), "n_gpu_layers": -1}), "handles": 4}]
    for e in engines[:3]:
        e.close()
    assert not loads[0].closed, "the weights stay while one context lives"
    engines[3].close()
    assert loads[0].closed and gguf_shared.report() == []


def test_other_model_parameters_are_other_weights(fake_llama, tmp_path):
    gguf_shared, Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    build = gguf_shared.shared_llama(Llama)
    a = build(model_path=str(path), n_gpu_layers=-1)
    b = build(model_path=str(path), n_gpu_layers=0)  # the CPU fallback is another weights instance
    c = build(model_path=str(path), n_gpu_layers=-1, n_ctx=8192)  # another context size is NOT
    assert len(loads) == 2
    assert gguf_shared.weights_identity(a) == gguf_shared.weights_identity(c) != gguf_shared.weights_identity(b)


def test_a_dropped_context_returns_its_handle(fake_llama, tmp_path):
    gguf_shared, Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    build = gguf_shared.shared_llama(Llama)
    a = build(model_path=str(path))
    b = build(model_path=str(path))
    del a
    gc.collect()
    assert gguf_shared.report() == [{"key": gguf_shared.key_for_kwargs({"model_path": str(path)}), "handles": 1}]
    assert not loads[0].closed
    b.close()
    assert loads[0].closed and gguf_shared.report() == []


def test_a_plain_llama_outside_shared_llama_is_untouched(fake_llama, tmp_path):
    gguf_shared, Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    gguf_shared.shared_llama(Llama)(model_path=str(path))  # installs the hook
    Llama(model_path=str(path))  # e.g. the embedder: its own model, not borrowed
    assert len(loads) == 2


@pytest.fixture()
def provider_load(fake_llama, monkeypatch):
    """Actual provider admission/slot creation, with only native allocation and fit faked."""
    from abstractcore.providers import gguf_slots, huggingface_provider as hp

    _shared, Llama, _loads = fake_llama
    monkeypatch.setattr(hp, "Llama", Llama, raising=False)
    monkeypatch.setattr(gguf_slots, "_SCHEDULERS", {})
    providers = []

    def load(path, *, context=4096, layers=-1, explicit=True):
        p = hp.HuggingFaceProvider.__new__(hp.HuggingFaceProvider)
        p.model = str(path)
        p.logger = logging.getLogger("test_provider_admission")
        p.n_gpu_layers = layers
        p.max_tokens = context
        p.max_output_tokens = 16
        p.debug = False
        p._user_provided_max_tokens = True
        p._n_gpu_layers_explicit = explicit
        p._gguf_prompt_cache_lock = threading.Lock()
        p._generation_worker = object()
        p._gguf_ensure_fit = lambda *args: {}
        p._gguf_probe_decode = lambda *args: None
        p._gguf_configured_concurrency = lambda: 1
        p._load_gguf_model()
        providers.append(p)
        return p

    yield load
    for p in providers:
        p._gguf_detach()


@pytest.mark.parametrize("alias", [False, True])
def test_provider_refuses_conflicting_placement_across_pinned_contexts(
    fake_llama, provider_load, tmp_path, alias
):
    shared, _Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    first = provider_load(path)
    other = tmp_path / "alias.gguf"
    if alias:
        other.symlink_to(path)
    else:
        other = path
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        provider_load(other, context=8192, layers=0)
    assert len(loads) == 1 and shared.weights_identity(first.llm) is not None


def test_provider_other_context_shares_weights_and_adopts_implicit_placement(fake_llama, provider_load, tmp_path):
    shared, _Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    first = provider_load(path)
    second = provider_load(path, context=8192, layers=0, explicit=False)
    assert len(loads) == 1
    assert first.llm is not second.llm and first.max_tokens == 4096 and second.max_tokens == 8192
    assert shared.weights_identity(first.llm) == shared.weights_identity(second.llm)
    assert second.n_gpu_layers == -1 and second._adopted_gpu_layers_from_resident_pool


def test_provider_concurrent_first_loads_do_not_allocate_conflicting_copies(fake_llama, provider_load, tmp_path):
    _shared, _Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    gate = threading.Barrier(2)
    results = []

    def load(context, layers):
        gate.wait()
        try:
            results.append(provider_load(path, context=context, layers=layers))
        except InvalidRequestError as exc:
            results.append(exc)

    threads = [threading.Thread(target=load, args=pair) for pair in ((4096, -1), (8192, 0))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(loads) == 1
    assert len([r for r in results if isinstance(r, Exception)]) == 1
    assert any("second copy is never loaded" in str(r) for r in results)


def test_provider_after_unload_can_choose_new_placement_and_other_artifact(fake_llama, provider_load, tmp_path):
    shared, _Llama, loads = fake_llama
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    first = provider_load(path)
    first._gguf_detach()
    assert shared.report() == []
    second = provider_load(path, context=8192, layers=0)
    other = tmp_path / "another-quant.gguf"
    other.write_bytes(b"y")
    third = provider_load(other)
    assert len(loads) == 3 and second.n_gpu_layers == 0
    assert shared.weights_identity(second.llm) != shared.weights_identity(third.llm)


def test_gguf_first_load_while_embedding_module_is_importing(fake_llama, provider_load, monkeypatch, tmp_path):
    name = "abstractcore.embeddings.manager"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x")
    held = provider_load(path)
    assert held is not None and len(fake_llama[2]) == 1


@pytest.fixture()
def embedding_load(fake_llama, monkeypatch):
    """Actual GGUF encoder/manager admission with the same fake native allocator as chat."""
    import weakref
    from abstractcore.embeddings import manager as em

    _shared, Llama, _loads = fake_llama
    pkg = sys.modules["llama_cpp"]
    pkg.Llama = Llama
    pkg.llama_supports_gpu_offload = lambda: False
    monkeypatch.setattr(em, "_SHARED_EMBEDDING_MODELS", weakref.WeakValueDictionary())
    monkeypatch.setattr("abstractcore.utils.model_cache.read_gguf_embedder_facts", lambda path: {}, raising=False)
    managers = []
    def load(path):
        manager = em.EmbeddingManager.__new__(em.EmbeddingManager)
        manager.model_id = str(path)
        manager.backend = None
        manager.trust_remote_code = False
        manager.device_preference = "cpu"
        manager.model = None
        manager._load_model_unshared = lambda: setattr(
            manager, "model", em._LlamaCppSentenceEncoder(str(path), path, device="cpu"))
        manager._load_model()
        managers.append(manager)
        return manager
    yield load
    for manager in managers:
        model = manager.model
        manager._release_shared_model()
        manager.model = None
        if model is not None:
            model._llm.close()


@pytest.mark.parametrize("embedding_first", [False, True])
def test_chat_embedding_same_gguf_refuses_second_lane_before_allocation(
    fake_llama, provider_load, embedding_load, tmp_path, embedding_first
):
    _shared, _Llama, loads = fake_llama
    path = tmp_path / "same.gguf"
    path.write_bytes(b"x")
    alias = tmp_path / "alias.gguf"
    alias.symlink_to(path)
    first, second = (embedding_load, provider_load) if embedding_first else (provider_load, embedding_load)
    resident = first(path)
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        second(alias)
    assert len(loads) == 1
    if embedding_first:
        resident.model._llm.close()
        resident._release_shared_model()
        resident.model = None
    else:
        resident._gguf_detach()
    replacement = second(alias)
    assert replacement is not None and len(loads) == 2


def test_chat_embedding_concurrent_first_loads_allocate_once(
    fake_llama, provider_load, embedding_load, tmp_path
):
    _shared, _Llama, loads = fake_llama
    path = tmp_path / "same.gguf"
    path.write_bytes(b"x")
    gate = threading.Barrier(2)
    results = []
    def run(load):
        gate.wait()
        try:
            results.append(load(path))
        except InvalidRequestError as exc:
            results.append(exc)
    threads = [threading.Thread(target=run, args=(load,)) for load in (provider_load, embedding_load)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(loads) == 1 and sum(isinstance(r, InvalidRequestError) for r in results) == 1
