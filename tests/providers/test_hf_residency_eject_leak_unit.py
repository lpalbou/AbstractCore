"""CPU-only regression for mission MEM2 (2026-09-25): the HuggingFace provider's
copy of the MLX "No models loaded" over held memory lie.

Root: HuggingFace provider instances do NOT share weights. A boot-time chat
summarizer, a per-request override client or an old runtime that built its
own `HuggingFaceProvider` for the same model holds a SECOND FULL COPY
(transformers: another set of MPS tensors; GGUF: another llama.cpp context).
The runtime pool's eject freed only its own instance and reported
`unloaded: true` while the other copy stayed resident, unlisted and
unreachable (measured live: 251 MB of SmolLM2 weights and ~1 GB of llama.cpp
buffers surviving a "successful" eject).

`abstractcore.providers.hf_residency` is the process-level truth: every
instance registers itself at construction, `resident_models()` counts every
copy, and `eject_model` unloads EVERY holder then returns torch's MPS pool to
the OS. These tests fake torch with a CPU allocator (no weights, no Metal):
  * the report counts every copy and every cache; the eject frees to baseline;
  * a refusing holder yields `ok: False` + a named residual;
  * the GGUF lane counts engine state and LlamaState snapshots;
  * the provider registers itself (an unregistered holder is invisible);
  * mutants that turn these RED: skip `empty_cache` in the eject (pool stays),
    drop the registration in `HuggingFaceProvider.__init__` (holder unseen).
"""
from __future__ import annotations

import gc
import sys
import types
import weakref
from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_torch(monkeypatch):
    """A CPU stand-in for torch whose MPS counters behave like the real
    allocator: a freed tensor's bytes move from `allocated` to the `pool`;
    `empty_cache` returns the pool to the system."""
    state = {"allocated": 0, "pool": 0, "empties": 0}

    class Storage:
        def __init__(self, nbytes: int):
            self._n = int(nbytes)

        def data_ptr(self):
            return id(self)

        def nbytes(self):
            return self._n

    class Tensor:
        def __init__(self, nbytes: int):
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
    torch.mps = SimpleNamespace(
        current_allocated_memory=lambda: state["allocated"],
        driver_allocated_memory=lambda: state["allocated"] + state["pool"],
        empty_cache=empty_cache,
        synchronize=lambda: None,
    )
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return state, Tensor


@pytest.fixture
def registry(monkeypatch):
    from abstractcore.providers import hf_residency

    fresh: "weakref.WeakSet" = weakref.WeakSet()
    monkeypatch.setattr(hf_residency, "_HF_PROVIDERS", fresh)
    return hf_residency


class _FakeModel:
    def __init__(self, *tensors):
        self._tensors = list(tensors)

    def parameters(self):
        return iter(self._tensors)

    def buffers(self):
        return iter(())


class _FakeHolder:
    """Weakref-able stand-in for a HuggingFaceProvider (transformers lane)."""

    provider = "huggingface"

    def __init__(self, model: str, weights, *, refuse: bool = False):
        from abstractcore.providers.base import PromptCacheStore

        self.model = model
        self.model_instance = _FakeModel(weights)
        self.llm = None
        self.pipeline = None
        self.tokenizer = object()
        self.device = "mps"
        self._transformers_snapshots = {}
        self._prompt_cache_store = PromptCacheStore(max_entries=4)
        self._refuse = refuse

    def unload_model(self, _name):
        if self._refuse:
            raise RuntimeError("Cannot unload during generation")
        self.model_instance = None
        self.pipeline = None
        self.tokenizer = None
        self._transformers_snapshots.clear()
        self._prompt_cache_store._entries.clear()


def test_report_counts_every_copy_and_eject_frees_to_baseline(fake_torch, registry):
    state, Tensor = fake_torch
    hf = registry
    a = _FakeHolder("fake/model", Tensor(1_000_000))
    b = _FakeHolder("fake/model", Tensor(1_000_000))          # the summarizer's own copy
    a._transformers_snapshots["s"] = {"cache": SimpleNamespace(keys=[Tensor(200_000)]), "ids": [1]}
    a._prompt_cache_store._entries["session:x"] = SimpleNamespace(value=SimpleNamespace(cache=Tensor(100_000)))
    hf.register_provider(a)
    hf.register_provider(b)

    rows = hf.resident_models()
    assert len(rows) == 1
    row = rows[0]
    assert row["holders"] == 2 and row["copies"] == 2 and row["shared_weights"] is False
    assert row["weights_bytes"] == 2_000_000, "every copy counts -- holders do not share weights"
    assert row["cache_bytes"] == 300_000 and row["held_bytes"] == 2_300_000
    report = hf.hf_memory_report()
    assert report["resident_models"] == 1 and report["holders"] == 2
    assert report["torch_mps_allocated_bytes"] == 2_300_000
    assert hf.process_residency_for("fake/model")["held_bytes"] == 2_300_000

    result = hf.eject_model("fake/model")
    assert result["ok"] is True
    assert len(result["holders_unloaded"]) == 2 and not result["holders_refused"]
    assert hf.process_residency_for("fake/model") is None
    assert state["allocated"] == 0, "weights of EVERY copy are gone"
    assert result["cache_cleared"] is True and state["empties"] >= 1
    assert state["pool"] == 0, "torch's MPS pool returned to the OS (MUTANT: skip empty_cache -> RED)"
    assert result["after"]["driver_bytes"] == 0
    assert hf.hf_memory_report()["held_bytes"] == 0


def test_eject_reports_residual_when_a_holder_refuses(fake_torch, registry):
    state, Tensor = fake_torch
    hf = registry
    ok_holder = _FakeHolder("busy/model", Tensor(400_000))
    busy = _FakeHolder("busy/model", Tensor(500_000), refuse=True)
    hf.register_provider(ok_holder)
    hf.register_provider(busy)

    result = hf.eject_model("busy/model")
    assert result["ok"] is False
    assert len(result["holders_unloaded"]) == 1
    assert result["holders_refused"] and "generation" in result["holders_refused"][0]["error"]
    assert result["residual"] is not None
    assert result["residual"]["holders"] == 1 and result["residual"]["held_bytes"] == 500_000
    assert state["allocated"] == 500_000


def test_unload_that_leaves_the_instance_loaded_is_a_refusal(fake_torch, registry):
    """A holder whose unload_model returns without releasing must not be
    reported as unloaded (the eject never claims success over retained memory)."""
    state, Tensor = fake_torch
    hf = registry
    sticky = _FakeHolder("sticky/model", Tensor(100))
    sticky.unload_model = lambda _name: None          # keeps model_instance
    hf.register_provider(sticky)
    result = hf.eject_model("sticky/model")
    assert result["ok"] is False and result["holders_refused"]
    assert "still reports the model loaded" in result["holders_refused"][0]["error"]
    assert result["residual"]["held_bytes"] == 100


def test_gguf_lane_counts_engine_and_llama_state_snapshots(registry, monkeypatch):
    hf = registry
    # No real llama_cpp needed: the accounting falls back to `nbytes` for the
    # engine and reads `LlamaState.llama_state_size` for snapshots.
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    class LlamaState:  # name is what the walker keys on
        llama_state_size = 4096
        scores = SimpleNamespace(nbytes=16)
        input_ids = SimpleNamespace(nbytes=8)

    class _Engine:
        nbytes = 300_000
        model_path = "/models/tiny-Q4_K_M.gguf"

        def n_ctx(self):
            return 2048

        def close(self):
            self.nbytes = 0

    class _GGUFHolder:
        provider = "huggingface"

        def __init__(self):
            from abstractcore.providers.base import PromptCacheStore

            self.model = "vendor/tiny-GGUF"
            self.llm = _Engine()
            self.model_instance = None
            self.pipeline = None
            self.device = None
            self._transformers_snapshots = {}
            self._prompt_cache_store = PromptCacheStore(max_entries=4)
            self._prompt_cache_store._entries["k"] = SimpleNamespace(
                value=SimpleNamespace(cache=SimpleNamespace(cache_state={(1, 2): LlamaState()}))
            )

        def unload_model(self, _name):
            self.llm.close()
            self.llm = None
            self._prompt_cache_store._entries.clear()

    holder = _GGUFHolder()
    hf.register_provider(holder)
    row = hf.process_residency_for("vendor/tiny-GGUF")
    assert row is not None and row["lane"] == "gguf"
    assert row["weights_bytes"] == 300_000
    assert row["cache_bytes"] == 4096 + 16 + 8, "LlamaState snapshots are memory too"
    assert "kv_bytes_estimated" not in row, "no geometry -> nothing estimated"
    assert row["model_path"] == "/models/tiny-Q4_K_M.gguf"
    assert hf.hf_memory_report()["llama_cpp_bytes"] == 300_000 + 4096 + 16 + 8

    result = hf.eject_model("vendor/tiny-GGUF")
    assert result["ok"] is True and holder.llm is None
    assert hf.process_residency_for("vendor/tiny-GGUF") is None
    assert hf.hf_memory_report()["llama_cpp_bytes"] == 0


def test_huggingface_provider_registers_itself_and_its_unload_is_seen(fake_torch, registry, monkeypatch):
    """The seam that makes the whole thing work: `HuggingFaceProvider.__init__`
    registers the instance. MUTANT (drop the registration): the provider is
    invisible to the listing and the eject -> RED."""
    state, Tensor = fake_torch
    hf = registry
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    def _fake_load(self):
        self.model_instance = _FakeModel(Tensor(4096))
        self.pipeline = None

    monkeypatch.setattr(HuggingFaceProvider, "_is_gguf_model", lambda self, model: False)
    monkeypatch.setattr(HuggingFaceProvider, "_reject_silent_gguf_substitution", lambda self, model: None)
    monkeypatch.setattr(HuggingFaceProvider, "_setup_device_transformers", lambda self: setattr(self, "device", "cpu"))
    monkeypatch.setattr(HuggingFaceProvider, "_load_transformers_model", _fake_load)
    monkeypatch.setattr("abstractcore.providers.huggingface_provider.TRANSFORMERS_AVAILABLE", True)

    provider = HuggingFaceProvider(model="fake/registered-model")
    assert provider in hf.registered_providers(), "construction must register the instance"
    row = hf.process_residency_for("fake/registered-model")
    assert row is not None and row["holders"] == 1 and row["weights_bytes"] == 4096

    result = hf.eject_model("fake/registered-model")
    assert result["ok"] is True and result["holders_unloaded"]
    assert provider.model_instance is None
    assert hf.process_residency_for("fake/registered-model") is None
    assert state["allocated"] == 0 and state["pool"] == 0


def test_gguf_row_counts_the_kv_allocation_for_n_ctx(registry, monkeypatch):
    """`llama_state_get_size` measures tokens in use; the engine ALLOCATES the
    whole n_ctx at construction (measured 2026-09-25: 0.05 GB of state beside
    a 4.7 GB allocation). The row must carry the allocation."""
    from abstractcore.providers.hf_residency import llama_kv_alloc_bytes

    meta = {"general.architecture": "qwen3", "qwen3.block_count": "28", "qwen3.attention.head_count": "16",
            "qwen3.attention.head_count_kv": "8", "qwen3.attention.key_length": "128"}
    assert llama_kv_alloc_bytes(meta, 40960) == 40960 * 28 * 2 * 8 * 128 * 2   # 4.7 GB
    # head_dim from embedding_length / head_count when key_length is absent
    meta2 = {"general.architecture": "llama", "llama.block_count": "2", "llama.attention.head_count": "4",
             "llama.embedding_length": "64"}
    assert llama_kv_alloc_bytes(meta2, 100) == 100 * 2 * 2 * 4 * 16 * 2
    assert llama_kv_alloc_bytes({}, 100) is None and llama_kv_alloc_bytes(meta, None) is None

    hf = registry
    monkeypatch.setitem(sys.modules, "llama_cpp", None)

    class _Engine:
        nbytes = 1_000
        model_path = "/models/q.gguf"
        metadata = meta

        def n_ctx(self):
            return 1024

    class _Holder:  # a SimpleNamespace is not weakref-able; the registry is a WeakSet
        provider = "huggingface"

        def __init__(self):
            self.model = "vendor/q-GGUF"
            self.llm = _Engine()
            self.model_instance = None
            self.pipeline = None
            self.device = None
            self._transformers_snapshots = {}
            self._prompt_cache_store = None

    holder = _Holder()
    hf.register_provider(holder)
    row = hf.process_residency_for("vendor/q-GGUF")
    expected_kv = 1024 * 28 * 2 * 8 * 128 * 2
    assert row["holder_rows"][0]["kv_alloc_bytes"] == expected_kv
    assert row["cache_bytes"] == expected_kv and row["held_bytes"] == 1_000 + expected_kv
    # f16 geometry estimate, not a measurement: the row says so.
    assert row["kv_bytes_estimated"] is True and row["holder_rows"][0]["kv_bytes_estimated"] is True
