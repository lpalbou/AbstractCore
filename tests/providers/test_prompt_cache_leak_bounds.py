"""Prompt-cache memory-bound regression tests (2026-08-03 leak audit).

Context: the HF transformers benchmark lane peaked at 164.5 GB on a 128 GB host.
The device-side component (MPS allocator pooling every freed KV buffer because
`torch.mps.empty_cache()` was only ever called by the vision lane) cannot be
tested on CPU — but the same audit found a Python-object leak class that CAN:
cache-holding structures keyed by `prompt_cache_key` whose entries survive the
key's eviction from `PromptCacheStore` (LRU capacity / TTL expiry, which fired
no callback), plus unbounded per-provider side dicts (MLX `_hybrid_snapshots`,
red-team P1).

These tests are CPU-only, model-free, and count RESIDENT OBJECTS via gc plus
liveness via weakref — the same instruments as the leak repro. Run against the
pre-fix code they FAIL (that is the point: a leak test that passes on broken
code is worthless); against the fixed code they pass.
"""

from __future__ import annotations

import gc
import inspect
import weakref

import pytest

from abstractcore.providers.base import PromptCacheStore


class _Payload:
    """Weakref-able stand-in for a KV cache (deepcopy-sized in real life)."""

    def __init__(self, tag: str):
        self.tag = tag


def _make_store(max_entries: int, provider=None) -> PromptCacheStore:
    """Build the store the way BaseProvider does on THIS code version.

    On post-fix code the store gets the provider eviction hook; on pre-fix code
    (no `on_evict` parameter) it is built plain — faithful to what each version
    actually wires, so the assertions below measure behavior, not API shape.
    """
    kwargs = {}
    params = inspect.signature(PromptCacheStore.__init__).parameters
    if "on_evict" in params and provider is not None and hasattr(provider, "_prompt_cache_store_evicted"):
        kwargs["on_evict"] = provider._prompt_cache_store_evicted
    return PromptCacheStore(max_entries=max_entries, **kwargs)


# ---------------------------------------------------------------------------
# Store-level: eviction must RELEASE (no secondary refs, callback fires)
# ---------------------------------------------------------------------------

def test_store_lru_eviction_releases_evicted_values():
    """N unique keys through a bounded store: at most `max_entries` values may
    remain alive. Evicted values must be garbage immediately (no secondary
    refs inside the store)."""
    store = PromptCacheStore(max_entries=4)
    refs = []
    for i in range(12):
        payload = _Payload(f"k{i}")
        refs.append(weakref.ref(payload))
        store.set(f"k{i}", payload)
        del payload
    gc.collect()
    alive = [r for r in refs if r() is not None]
    assert len(store.keys()) == 4
    assert len(alive) == 4, (
        f"{len(alive)} values alive with max_entries=4 — evicted entries are "
        f"being retained"
    )


def test_store_lru_eviction_fires_release_callback():
    """Silent eviction is the leak: per-key side state (MLX hybrid snapshots,
    device pools) can only be released if the owner HEARS about the eviction.
    Pre-fix `PromptCacheStore` had no eviction signal at all — this test fails
    there by construction."""
    params = inspect.signature(PromptCacheStore.__init__).parameters
    assert "on_evict" in params, (
        "PromptCacheStore has no eviction callback: LRU/TTL evictions are "
        "silent, so per-key provider state (e.g. MLX _hybrid_snapshots) "
        "leaks for the provider's lifetime"
    )
    evicted = []
    store = PromptCacheStore(max_entries=3, on_evict=lambda k, v: evicted.append(k))
    for i in range(8):
        store.set(f"k{i}", _Payload(f"k{i}"))
    assert evicted == [f"k{i}" for i in range(5)]
    # Explicit delete/clear must NOT fire it (callers handle their own teardown).
    evicted.clear()
    store.delete("k7")
    store.clear()
    assert evicted == []


def test_store_ttl_expiry_fires_release_callback():
    evicted = []
    store = PromptCacheStore(max_entries=8, on_evict=lambda k, v: evicted.append(k))
    store.set("t1", _Payload("t1"), ttl_s=0.0)
    import time

    time.sleep(0.01)
    assert store.get("t1") is None
    assert evicted == ["t1"]


# ---------------------------------------------------------------------------
# MLX: `_hybrid_snapshots` must stay bounded under key churn
# ---------------------------------------------------------------------------

def _mlx_snapshot_provider(max_entries: int):
    from abstractcore.providers.mlx_provider import MLXProvider

    p = MLXProvider.__new__(MLXProvider)  # no model load; repo test precedent
    p._ensure_hybrid_snapshot_state()
    p._prompt_cache_store = _make_store(max_entries, provider=p)
    return p


def test_mlx_hybrid_snapshots_bounded_under_key_churn():
    """Agent churn shape: many distinct `prompt_cache_key`s over one provider
    (per-session keys, per-rep bench keys). Each key stores a full-KV deepcopy
    snapshot (~1 GB at 30k on Qwen3.5-4B). The store LRU-evicts at
    `max_entries`; the snapshots MUST follow, or every evicted key leaks its
    deepcopy for the provider's lifetime (red-team P1, formally requested)."""
    bound = 8
    p = _mlx_snapshot_provider(bound)
    refs = []
    for i in range(25):
        key = f"session-{i}"
        payload = _Payload(key)
        refs.append(weakref.ref(payload))
        p._prompt_cache_store.set(key, {"state": key})
        p._store_hybrid_snapshot(key, payload, [1, 2, 3])
        del payload
    gc.collect()
    alive = [r for r in refs if r() is not None]
    n_snaps = len(p._hybrid_snapshots)
    assert n_snaps <= bound, (
        f"{n_snaps} hybrid snapshots resident with store max_entries={bound} — "
        f"snapshots outlive their evicted keys (each is a full KV deepcopy)"
    )
    assert len(alive) <= bound, (
        f"{len(alive)} snapshot cache deepcopies still alive with store "
        f"max_entries={bound}"
    )
    # And the survivors must be exactly (a subset of) the keys still in the store.
    assert set(p._hybrid_snapshots.keys()) <= set(p._prompt_cache_store.keys())


def test_mlx_hybrid_snapshots_hard_bound_without_store_wiring():
    """Belt for the belt: even with NO store present (unit-test construction,
    future wiring regressions), the snapshot dict itself must refuse to grow
    without bound."""
    from abstractcore.providers.mlx_provider import MLXProvider

    p = MLXProvider.__new__(MLXProvider)
    p._ensure_hybrid_snapshot_state()
    refs = []
    for i in range(64):
        payload = _Payload(f"k{i}")
        refs.append(weakref.ref(payload))
        p._store_hybrid_snapshot(f"k{i}", payload, [i])
        del payload
    gc.collect()
    n = len(p._hybrid_snapshots)
    alive = sum(1 for r in refs if r() is not None)
    assert n <= 32, f"{n} hybrid snapshots resident — unbounded snapshot dict"
    assert alive <= 32, f"{alive} snapshot deepcopies alive — unbounded snapshot dict"


def test_mlx_snapshot_clear_and_drop_still_work():
    """The bound must not break the existing lifecycle: same-key replacement
    keeps one snapshot; drop removes it."""
    p = _mlx_snapshot_provider(8)
    p._prompt_cache_store.set("k", {"state": "k"})
    p._store_hybrid_snapshot("k", _Payload("a"), [1])
    p._store_hybrid_snapshot("k", _Payload("b"), [1, 2])
    assert len(p._hybrid_snapshots) == 1
    assert p._get_hybrid_snapshot("k")["ids"] == [1, 2]
    p._drop_hybrid_snapshot("k")
    assert p._get_hybrid_snapshot("k") is None


# ---------------------------------------------------------------------------
# HF transformers: rebuild-per-turn must not accumulate Cache objects
# ---------------------------------------------------------------------------

def _resident_dynamic_caches():
    from transformers.cache_utils import Cache

    gc.collect()
    return [o for o in gc.get_objects() if isinstance(o, Cache)]


def test_transformers_rebuild_per_turn_flat_cache_object_count():
    """The non-croppable-hybrid warm path rebuilds a fresh Cache each turn
    (`state.cache = <fresh>`; pre-fix crop-accept mutated in place, post-fix
    refusal rebuilds). Rebuilding N turns must keep the resident transformers
    `Cache` object count FLAT: the store holds one live state per key and the
    replaced caches must die with their replacement."""
    torch = pytest.importorskip("torch")
    from transformers.cache_utils import DynamicCache

    from abstractcore.providers.huggingface_provider import _TransformersPromptCacheValue

    baseline = len(_resident_dynamic_caches())
    store = PromptCacheStore(max_entries=4)
    state = _TransformersPromptCacheValue(cache=DynamicCache())
    store.set("agent", state)
    refs = []
    counts = []
    for turn in range(25):
        # the rebuild lane's exact object lifecycle (hf provider, rebuild path)
        state.cache = None
        fresh = DynamicCache()
        # give the cache a real tensor so retention would be memory, not noise
        fresh.update(torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4), 0)
        refs.append(weakref.ref(fresh))
        state.cache = fresh
        store.set("agent", state)
        del fresh
        counts.append(len(_resident_dynamic_caches()) - baseline)
    alive = sum(1 for r in refs if r() is not None)
    assert alive == 1, f"{alive} rebuilt caches alive after 25 turns (want 1: the live one)"
    assert max(counts) <= 2, f"cache object count grew across turns: {counts}"


def test_transformers_states_bounded_by_store_capacity():
    """N unique keys with real DynamicCache payloads: resident transformers
    Cache objects ≤ store max_entries."""
    torch = pytest.importorskip("torch")
    from transformers.cache_utils import DynamicCache

    from abstractcore.providers.huggingface_provider import _TransformersPromptCacheValue

    baseline = len(_resident_dynamic_caches())
    store = PromptCacheStore(max_entries=6)
    for i in range(20):
        cache = DynamicCache()
        cache.update(torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 8, 4), 0)
        store.set(f"key-{i}", _TransformersPromptCacheValue(cache=cache))
        del cache
    resident = len(_resident_dynamic_caches()) - baseline
    assert resident <= 6, (
        f"{resident} transformers Cache objects resident with max_entries=6 — "
        f"evicted states retained"
    )


# ---------------------------------------------------------------------------
# GGUF: `_AutoGrowingLlamaRAMCache` must stay bounded within a key
# ---------------------------------------------------------------------------

class _FakeLlamaState:
    def __init__(self, size: int):
        self.llama_state_size = size


def test_gguf_autogrow_cache_bounded_within_key():
    """Auto-grow raises capacity to fit the LARGEST single state; it must not
    accumulate states without bound across turns/prefixes under one key."""
    pytest.importorskip("llama_cpp")
    from abstractcore.providers import huggingface_provider as hf

    # Materialize the lazily-defined class the way the provider does.
    provider = hf.HuggingFaceProvider.__new__(hf.HuggingFaceProvider)
    provider.model_type = "gguf"
    provider._gguf_prompt_cache_pending_capacity_bytes = None
    provider._gguf_prompt_cache_default_capacity_bytes = None
    value = hf.HuggingFaceProvider._prompt_cache_backend_create(provider)
    assert value is not None
    cache = value.cache

    state_size = 100 * 1024 * 1024  # pretend 100 MB LlamaStates (no real alloc)
    refs = []
    for i in range(10):
        st = _FakeLlamaState(state_size)
        refs.append(weakref.ref(st))
        cache[(1, 2, 3, i)] = st
        del st
    gc.collect()
    alive = sum(1 for r in refs if r() is not None)
    total = sum(s.llama_state_size for s in cache.cache_state.values())
    assert cache.capacity_bytes == state_size, "capacity must grow to max single state only"
    assert total <= cache.capacity_bytes, (
        f"cache_size {total} exceeds capacity {cache.capacity_bytes} — "
        f"unbounded accumulation across keys within one cache object"
    )
    assert alive == len(cache.cache_state), "evicted LlamaStates retained"
