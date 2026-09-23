"""The APC shape keeps a conversation's lineage alive across a tool loop (mission A3).

Model-free, but against the REAL mlx-vlm `APCManager` / `APCCoordinator`: the
failure this guards was an interaction between two mlx-vlm rules, so a mock
would prove nothing. `APCCoordinator.checkpoint_lengths` caps the snapshots ONE
prompt emits at `checkpoint_entries`, and `APCManager.store_exact_cache` caps the
WHOLE store at the same number (LRU, tenant-blind). With the previous 256 x 4,
one prompt filled the store; after a tool loop every resident snapshot sat in the
loop's discarded branch, and the next turn (which still extends the run's FIRST
prompt exactly) went cold — 51 s of prefill on the operator's live model.
"""
from types import SimpleNamespace

import pytest

apc = pytest.importorskip("mlx_vlm.apc")
coordinator_mod = pytest.importorskip("mlx_vlm.apc_coordinator")

from abstractcore.providers.mlx_native_session import (  # noqa: E402
    CHECKPOINT_ENTRIES,
    CHECKPOINT_GUARD_TOKENS,
    CHECKPOINT_INTERVAL_TOKENS,
)


def _coordinator():
    manager = apc.APCManager(
        num_blocks=8,
        block_size=256,
        overrides={
            "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
            "checkpoint_entries": CHECKPOINT_ENTRIES,
            "checkpoint_guard_tokens": CHECKPOINT_GUARD_TOKENS,
        },
    )
    c = coordinator_mod.APCCoordinator.__new__(coordinator_mod.APCCoordinator)
    c.manager, c.model = manager, None
    c.plan = SimpleNamespace(restorable=True, strategy="checkpoint", legacy_mode=None)
    return manager, c


@pytest.mark.parametrize("prompt_len", [4752, 12563, 28533, 120_000])
def test_each_call_stores_exactly_one_snapshot_just_before_its_end(prompt_len):
    _, c = _coordinator()
    assert c.checkpoint_lengths(list(range(prompt_len)), set()) == [prompt_len - CHECKPOINT_GUARD_TOKENS]


def test_the_first_prompt_of_a_run_survives_a_tool_loop_and_restores_the_next_turn():
    """A run's first call, three growing loop iterations and the final call
    store 5 snapshots; the next turn's first prompt extends ONLY the first one,
    and must still find it. Each snapshot is a tiny real recurrent `ArraysCache`
    (the non-trimmable hybrid case: only an EXACT stored prefix restores)."""
    mx = pytest.importorskip("mlx.core")
    from mlx_vlm.models.cache import ArraysCache

    def snapshot():
        cache = ArraysCache(size=2)
        cache[0] = mx.zeros((1, 4))
        cache[1] = mx.zeros((1, 4))
        return [cache]

    manager, c = _coordinator()
    first = list(range(5000))
    loop, prompts = list(first), [first]
    for it in range(4):
        loop = loop + [100_000 + it] * 7000  # the loop's own branch
        prompts.append(list(loop))
    for p in prompts:
        (final,) = c.checkpoint_lengths(p, set())
        assert manager.store_exact_cache(p[:final], snapshot(), extra_hash=7)
    next_turn = first + [900_000] * 300  # history + the new question, NOT the loop
    cache, restored = manager.lookup_exact_cache(next_turn, extra_hash=7)
    assert cache is not None
    assert restored == len(first) - CHECKPOINT_GUARD_TOKENS
