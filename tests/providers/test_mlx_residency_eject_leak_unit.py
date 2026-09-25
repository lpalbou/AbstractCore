"""CPU-only regression for the 2026-09-25 MLX memory leak / "No models loaded"
lie: a shared model whose weights outlive the eject, and a memory report that
did not say so.

Root of the incident: MLX weights are SHARED between provider instances
(`_SharedMLXModel`, `NativeSession`). `MLXProvider.unload_model` frees only the
calling instance's claim, and `_unload_model_unlocked` USED to skip
`mx.clear_cache()` whenever any sibling still held the model. When that sibling
was itself garbage-collected, its freed buffers sat in MLX's allocator cache
with nothing left to clear them, so the process kept ~90 GB while
`mx.get_active_memory()` read ~0 and every residency listing said "not loaded".

`abstractcore.providers.mlx_residency` is the process-level truth: it enumerates
every MLX model whose weights are alive (whoever holds them), reports the held
bytes, and `eject_model` unloads EVERY holder then clears the allocator cache.

These tests fake MLX with a CPU allocator (no weights, no Metal) and assert:
  * eject frees the shared weights and clears the allocator cache to baseline;
  * the report never says 0 held while a buffer is alive;
  * three mutants of the production code turn the leak test RED:
      1. clear_cache skipped on shared unload,
      2. prompt-cache store kept across unload,
      3. drafter kept across unload.
"""
from __future__ import annotations

import sys
import types
import weakref
from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_mlx(monkeypatch):
    """A CPU stand-in for `mlx.core` whose active/cache/peak counters and
    `clear_cache` behave like MLX's allocator: freeing an array moves its bytes
    from active to cache; `clear_cache` returns the cache to the system."""
    state = {"active": 0, "cache": 0, "peak": 0, "clears": 0}

    class Array:
        def __init__(self, nbytes: int):
            self._nbytes = int(nbytes)
            state["active"] += self._nbytes
            state["peak"] = max(state["peak"], state["active"])

        @property
        def nbytes(self) -> int:
            return self._nbytes

        def __del__(self):
            state["active"] -= self._nbytes
            state["cache"] += self._nbytes

    core = types.ModuleType("mlx.core")
    core.array = Array
    core.get_active_memory = lambda: state["active"]
    core.get_cache_memory = lambda: state["cache"]
    core.get_peak_memory = lambda: state["peak"]

    def clear_cache():
        state["clears"] += 1
        state["cache"] = 0

    core.clear_cache = clear_cache
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return state, Array


class _FakeModule:
    """A stand-in nn.Module: `params_bytes`/reachability see it via nbytes."""

    def __init__(self, array):
        self.weight = array

    @property
    def nbytes(self):
        return self.weight.nbytes


def _make_shared(mp, Array, *, key: str, weight_bytes: int, cache_bytes: int = 0):
    """Register a `_SharedMLXModel` with `holders` and a prompt-cache store that
    holds `cache_bytes`, mirroring a model resident in the process."""
    from abstractcore.providers.base import PromptCacheStore

    llm = _FakeModule(Array(weight_bytes))
    store = PromptCacheStore(max_entries=8)
    if cache_bytes:
        store._entries["session:x"] = SimpleNamespace(value=Array(cache_bytes))
    shared = mp._SharedMLXModel(key=key, llm=llm, tokenizer=object(), prompt_cache_store=store)
    return shared, llm


class _FakeHolder:
    """A minimal weakref-able stand-in for an MLXProvider (WeakSet needs it):
    `model`, `llm`, and an `unload_model` that drops the shared claim and, when
    it is the LAST holder, tears the shared model down exactly as production
    `_release_shared_model` does (pop the registry, drop weights + caches) so
    the arrays become collectable and the residency truth flips to not-loaded."""

    def __init__(self, mp, model, shared, key, unload_side_effect=None):
        self._mp = mp
        self.model = model
        self.llm = shared.llm
        self._shared = shared
        self._key = key
        self._side = unload_side_effect

    def unload_model(self, _name):
        if self._side is not None:
            self._side()  # a refusing holder raises here, before releasing
        self._shared.holders.discard(self)
        self.llm = None
        if not list(self._shared.holders):
            self._mp._SHARED_MLX_MODELS.pop(self._key, None)
            self._shared.llm = None
            try:
                self._shared.prompt_cache_store._entries.clear()
            except Exception:
                pass


def _holder(mp, model: str, shared, *, key=None, unload_side_effect=None):
    holder = _FakeHolder(mp, model, shared, key or model, unload_side_effect)
    shared.holders.add(holder)
    return holder


def test_report_and_eject_are_truthful_and_free_to_baseline(fake_mlx, monkeypatch):
    state, Array = fake_mlx
    import importlib

    import abstractcore.providers.mlx_provider as mp
    from abstractcore.providers import mlx_residency

    importlib.reload(mlx_residency)
    monkeypatch.setattr(mp._SHARED_MLX_MODELS, "clear", lambda: None, raising=False)

    baseline_active = state["active"]
    shared, llm = _make_shared(mp, Array, key="fake/model", weight_bytes=1_000_000, cache_bytes=200_000)
    mp._SHARED_MLX_MODELS["fake/model"] = shared
    holder = _holder(mp, "fake/model", shared, key="fake/model")

    # The report must SEE the resident model and its bytes, never say 0.
    report = mlx_residency.mlx_memory_report()
    assert report["resident_models"] == 1
    row = mlx_residency.process_residency_for("fake/model")
    assert row is not None and row["weights_bytes"] == 1_000_000
    assert row["prompt_cache_bytes"] == 200_000 and row["holders"] == 1
    assert state["active"] >= baseline_active + 1_200_000

    # Eject: every holder unloaded, weights + cache freed, allocator cleared.
    del llm  # only the shared store + holder hold the arrays now
    result = mlx_residency.eject_model("fake/model")
    assert result["ok"] is True
    assert result["holders_unloaded"] and not result["holders_refused"]
    assert result["cache_cleared"] is True
    assert mlx_residency.process_residency_for("fake/model") is None
    assert state["active"] == baseline_active     # weights gone
    assert state["cache"] == 0                     # allocator returned to system
    assert state["clears"] >= 1                     # MUTANT 1 (skip clear) -> cache != 0 -> RED


def test_eject_reports_residual_when_a_holder_refuses(fake_mlx, monkeypatch):
    """A holder mid-generation refuses to unload: the eject must NOT claim
    success, and the residual bytes must be named (never a silent 'freed')."""
    state, Array = fake_mlx
    import importlib

    import abstractcore.providers.mlx_provider as mp
    from abstractcore.providers import mlx_residency

    importlib.reload(mlx_residency)

    shared, llm = _make_shared(mp, Array, key="busy/model", weight_bytes=500_000)
    mp._SHARED_MLX_MODELS["busy/model"] = shared

    def refuse():
        raise RuntimeError("Cannot unload during generation")

    holder = _holder(mp, "busy/model", shared, key="busy/model", unload_side_effect=refuse)
    assert holder is not None  # keep the WeakSet member alive
    result = mlx_residency.eject_model("busy/model")
    assert result["ok"] is False
    assert result["holders_refused"] and "generation" in result["holders_refused"][0]["error"]
    assert result["residual"] is not None and result["residual"]["holders"] == 1
    mp._SHARED_MLX_MODELS.pop("busy/model", None)


def test_provider_unload_clears_allocator_even_with_sibling(fake_mlx):
    """MUTANT 1 guard at the provider seam: `_unload_model_unlocked` must call
    `clear_cache` on EVERY unload, including when a sibling still holds the
    weights. The pre-fix code guarded it behind `if not shared_still_used`."""
    state, Array = fake_mlx
    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    provider._release_shared_model = lambda: True   # a sibling still uses it
    provider._native_shared_still_used = True
    provider._clear_prompt_caches_for_unload = lambda: None
    provider.llm = _FakeModule(Array(300_000))
    provider.tokenizer = object()
    provider._vision_addon = None
    provider._outlines_model = None
    provider._mtp_drafter = _FakeModule(Array(50_000))   # MUTANT 3: keep this -> stays active
    provider._mtp_processor = None
    provider._mtp_last_result = None
    provider._native_qwen4 = None

    clears_before = state["clears"]
    provider._unload_model_unlocked("fake/model")
    assert state["clears"] > clears_before, "clear_cache must run even when a sibling remains (MUTANT 1)"
    assert provider._mtp_drafter is None, "drafter must be dropped on unload (MUTANT 3)"


def test_provider_unload_drops_prompt_cache_store(fake_mlx):
    """MUTANT 2 guard: `_unload_model_unlocked` must drop this instance's
    prompt-cache store (its KV entries are only usable with the resident
    weights and are the memory hogs). The store's arrays must be freed."""
    state, Array = fake_mlx
    from abstractcore.providers.base import PromptCacheStore
    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    provider._release_shared_model = lambda: False   # this instance owns the model
    provider._native_shared_still_used = False
    provider.llm = _FakeModule(Array(400_000))
    provider.tokenizer = object()
    provider._vision_addon = None
    provider._outlines_model = None
    provider._mtp_drafter = None
    provider._mtp_processor = None
    provider._mtp_last_result = None
    provider._native_qwen4 = None
    # A store holding one KV entry (a fake array). Force the wrapper's own
    # fallback clear so the test does not depend on the full capability setup.
    store = PromptCacheStore(max_entries=8)
    store._entries["session:live"] = SimpleNamespace(value=Array(600_000))
    provider._prompt_cache_store = store
    provider.prompt_cache_clear = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("force fallback"))

    active_with_cache = state["active"]
    assert active_with_cache >= 1_000_000
    provider._unload_model_unlocked("fake/model")
    assert len(store._entries) == 0, "prompt-cache store must be emptied on unload (MUTANT 2)"
    assert state["active"] == 0 and state["cache"] == 0, "weights + KV cache freed and allocator cleared"
