"""Pins for the MLX snapshot/restore lane for UNTRIMMABLE architectures.

Gated-DeltaNet hybrids (Qwen3.5/3.6/Ornith) and pure-SSM models hold a
recurrent state that cannot be trimmed (rewound), so the delta lane used to
rebuild-fresh every warm turn (zero savings). The snapshot lane instead keeps
ONE deepcopy snapshot per key at the last prefill boundary, keyed by the exact
tokens it holds, and restores it when the next full-context prompt extends it —
forward-only, no rewind (the discipline llama.cpp's GGUF lane and mlx_lm's own
server use). These tests pin the DECISION logic with a fake prefill (no model
load); byte-exact correctness + speedup are proven live in
scripts/verify_prompt_cache_families.py and the 2026-07-15 parity report.
"""

from typing import List

import pytest

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.mlx_provider import MLXProvider


class _FakeLayer:
    def __init__(self, offset: int = 0):
        self.offset = int(offset)

    def empty(self) -> bool:
        return self.offset == 0


class _FakeTokenizer:
    """Whitespace tokenizer with STABLE ids (a fixed vocab map, not hash())
    so prefixes are genuine token prefixes across calls."""

    def __init__(self):
        self._vocab = {}

    def encode(self, text: str) -> List[int]:
        out = []
        for w in str(text).split():
            out.append(self._vocab.setdefault(w, len(self._vocab) + 1))
        return out


class _FakeLogger:
    def warning(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


def _provider() -> MLXProvider:
    import threading

    p = MLXProvider.__new__(MLXProvider)
    p.tokenizer = _FakeTokenizer()
    p.logger = _FakeLogger()
    p._prompt_cache_store = PromptCacheStore()
    p._delta_feed_warned_keys = set()
    p._append_stash_lock = threading.RLock()
    p._hybrid_snapshot_lock = threading.RLock()
    p._hybrid_snapshots = {}
    # Untrimmable: trim always refuses → the hybrid lane is taken.
    p._trim_prompt_cache_tokens = lambda cache, n: False  # type: ignore[method-assign]
    # Fresh cache = a countable KV-shaped layer (so cache_len is a number and
    # the trim-refusal branch, not the uncountable branch, is exercised).
    p._prompt_cache_backend_create = lambda: [_FakeLayer(0)]  # type: ignore[method-assign]

    # Fake prefill: bump the layer offset by the number of tokens "prefilled",
    # so cache_len reflects what the cache holds (mirrors the real model pass).
    def _fake_prefill(cache_value, token_ids):
        for layer in cache_value:
            layer.offset += len(token_ids)
        return True

    p._prefill_tokens_into_cache = _fake_prefill  # type: ignore[method-assign]
    return p


def _warm_untrimmable(p: MLXProvider, key: str, fed_prompt: str, extra_generated: int = 0):
    ids = p.tokenizer.encode(fed_prompt)
    cache = [_FakeLayer(offset=len(ids) + extra_generated)]
    p._prompt_cache_store.set(key, cache, meta={"backend": "mlx", "fed_token_ids": list(ids)})
    return cache, ids


def test_cold_turn_creates_snapshot_and_seeds_decode():
    p = _provider()
    cache, _ = _warm_untrimmable(p, "k", "a b c d e", extra_generated=3)
    new_ids = p.tokenizer.encode("a b c d e f g")

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d e f g", full_context=True
    )

    # No snapshot yet → fresh working cache, prefilled to new_ids[:-1], and the
    # trailing token is the decode seed. Record is the full logical prompt.
    assert record == new_ids
    assert feed == [new_ids[-1]]
    # A snapshot now exists at the new_ids[:-1] boundary for the next turn.
    snap = p._get_hybrid_snapshot("k")
    assert snap is not None
    assert snap["ids"] == new_ids[:-1]


def _seed_snapshot(p: MLXProvider, key: str, snap_prompt: str):
    """Install a snapshot boundary directly (mirrors what a prior turn stored),
    plus a warm untrimmable live cache with a matching fed record so the next
    call reaches the trim-refusal → snapshot lane."""
    snap_ids = p.tokenizer.encode(snap_prompt)
    p._store_hybrid_snapshot(key, [_FakeLayer(offset=len(snap_ids))], snap_ids)
    # Live cache: holds snap + a generated reply (so trim is NEEDED and refused).
    live = [_FakeLayer(offset=len(snap_ids) + 4)]
    p._prompt_cache_store.set(key, live, meta={"backend": "mlx", "fed_token_ids": list(snap_ids)})
    return live, snap_ids


def test_warm_turn_restores_snapshot_and_feeds_only_suffix():
    p = _provider()
    live, snap_ids = _seed_snapshot(p, "k", "a b c d e f")
    new_ids = p.tokenizer.encode("a b c d e f g h i")   # true extension of the snapshot

    cache_tel = {}
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", live, "a b c d e f g h i", full_context=True, telemetry=cache_tel
    )

    assert cache_tel["outcome"] == "hit_restore"
    assert cache_tel["cached_tokens"] == len(snap_ids)   # reused the snapshot boundary
    assert record == new_ids
    assert feed == [new_ids[-1]]                          # only the trailing seed feeds
    # The snapshot advanced to the new boundary (one per key: evicted the old).
    assert p._get_hybrid_snapshot("k")["ids"] == new_ids[:-1]


def test_divergent_prompt_does_not_restore_stale_snapshot():
    p = _provider()
    live, _ = _seed_snapshot(p, "k", "a b c d e f")

    # A prompt that DIVERGES from the snapshot's recorded prefix must not reuse it.
    cache_tel = {}
    _out, _feed, record = p._prepare_cache_delta_feed(
        "k", live, "x y z totally different", full_context=True, telemetry=cache_tel
    )
    assert cache_tel["outcome"] == "rebuilt"          # fresh, not a false restore
    assert cache_tel["cached_tokens"] == 0
    assert record == p.tokenizer.encode("x y z totally different")


def test_prompt_cache_clear_drops_snapshot():
    p = _provider()
    p._default_prompt_cache_key = None  # attr prompt_cache_clear/super() need
    _seed_snapshot(p, "k", "a b c d e f")
    assert p._get_hybrid_snapshot("k") is not None

    p.prompt_cache_clear("k")
    assert p._get_hybrid_snapshot("k") is None


def test_clone_is_independent_of_parent_deepcopy_not_aliased():
    """The clone-aliasing bug: a from_state clone shares ArraysCache's list, so
    mutating the clone corrupts the parent. deepcopy must isolate them."""
    p = _provider()
    parent = [_FakeLayer(offset=5)]
    clone = p._prompt_cache_backend_clone(parent)
    assert clone is not None
    clone[0].offset = 999
    assert parent[0].offset == 5   # parent untouched → not aliased
