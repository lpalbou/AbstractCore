"""ONE llama.cpp weights instance per GGUF in a process (ADR 0011; residency audit R1, 2026-10-29).

A fake `llama_cpp._internals` stands in for llama-cpp-python: its `Llama` builds the model through
`internals.LlamaModel(...)` exactly as the real one does, and counts the loads.
MUTANT: `shared_llama` returning `llama_cls` unchanged -> N loads -> RED.
"""
from __future__ import annotations

import gc
import sys
import types

import pytest


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
