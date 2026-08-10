"""The GGUF lane must not destroy llama.cpp's own prefix reuse.

MEASURED REGRESSION (2026-08-03, `unsloth/Qwen3-4B-Instruct-2507-GGUF`, same model,
same process, same growing prompts; the ONLY difference between arms was whether a
`prompt_cache_key` was passed):

    turn | shared prefix | WITH key: prefilled / time | NO key: prefilled / time
      2  |     9,993     |    10,016 / 14.971s        |     24 / 0.310s
      3  |     9,996     |    10,019 / 15.116s        |     24 / 0.270s
      4  |     9,999     |    10,022 / 15.207s        |     24 / 0.283s

Attaching the cache made the lane ~48x SLOWER and fed 417x more tokens, because
`_gguf_prefill_prompt_cache` opened with an unconditional `llm.reset()` — which
erases `n_tokens`/`_input_ids`, the exact state `Llama.generate` reads to trim its
resident context to the longest common prefix. `load_state` fired ZERO times across
the run; `save_state` was paid on every call at 1.41 GB.

THE ASSERTION SHAPE THAT CATCHES IT IS A TOKEN COUNT, NOT A TIMING. With a cache key
and a growing transcript, the tokens fed on turn N+1 must be O(suffix), not O(prompt).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import pytest

from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _FakeCtx:
    """Only the KV-removal primitive `Llama.generate` probes before reusing."""

    def __init__(self, supports_partial_removal: bool = True):
        self.supports_partial_removal = supports_partial_removal
        self.removals: List[int] = []

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        if not self.supports_partial_removal:
            return False
        self.removals.append(int(p0))
        return True


class _FakeState:
    """Stands in for `llama_cpp.llama.LlamaState`."""

    def __init__(self, tokens: Sequence[int], size: int = 1_410_000_000):
        self.input_ids = list(int(t) for t in tokens)
        self.llama_state_size = int(size)


class _FakeLlama:
    """A llama.cpp stand-in that records exactly what it was asked to evaluate.

    Mirrors the two behaviours the fix depends on: `eval` appends at `n_tokens`
    (dropping any KV past it), and the resident `_input_ids` survive between calls
    unless something resets them.
    """

    def __init__(self, supports_partial_removal: bool = True):
        self.n_tokens = 0
        self._input_ids: List[int] = []
        self._ctx = _FakeCtx(supports_partial_removal)
        self.eval_batches: List[int] = []
        self.save_state_calls = 0
        self.load_state_calls = 0
        self.reset_calls = 0
        self.cache: Any = None

    def reset(self) -> None:
        self.reset_calls += 1
        self.n_tokens = 0
        self._input_ids = []

    def eval(self, tokens: Sequence[int]) -> None:
        toks = [int(t) for t in tokens]
        self.eval_batches.append(len(toks))
        self._input_ids = self._input_ids[: self.n_tokens] + toks
        self.n_tokens += len(toks)

    def save_state(self) -> _FakeState:
        self.save_state_calls += 1
        return _FakeState(self._input_ids[: self.n_tokens])

    def load_state(self, state: _FakeState) -> None:
        self.load_state_calls += 1
        self._input_ids = list(state.input_ids)
        self.n_tokens = len(self._input_ids)

    def set_cache(self, cache: Any) -> None:
        self.cache = cache

    @staticmethod
    def longest_token_prefix(a: Sequence[int], b: Sequence[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if int(x) != int(y):
                break
            n += 1
        return n


class _FakeRAMCache:
    """`LlamaRAMCache`-shaped: a token-tuple -> state map with `cache_state`."""

    def __init__(self):
        self.cache_state: dict = {}

    def __setitem__(self, key: Sequence[int], value: Any) -> None:
        self.cache_state[tuple(int(t) for t in key)] = value


def _provider(llm: _FakeLlama) -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.llm = llm
    return p


def _prompt(n_turns: int, base: int = 10_000, grow: int = 24) -> Tuple[int, ...]:
    """A growing transcript: turn N is a TRUE token prefix of turn N+1."""
    return tuple(range(1, base + 1 + grow * (n_turns - 1)))


def test_growing_turn_feeds_the_suffix_not_the_whole_prompt():
    """THE PIN. Turn 2 of a keyed session must evaluate the grown suffix only."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()

    t1 = _prompt(1)
    assert p._gguf_prefill_prompt_cache(cache, t1)
    assert llm.eval_batches == [len(t1)]          # cold turn: the whole prompt

    # The reply lands in the resident context, exactly as generation would leave it.
    llm.eval([99_001, 99_002, 99_003])
    llm.eval_batches.clear()

    t2 = _prompt(2)
    assert p._gguf_prefill_prompt_cache(cache, t2)

    fed = sum(llm.eval_batches)
    grown = len(t2) - len(t1)
    assert fed <= grown + 8, (
        f"turn 2 fed {fed} tokens for a {grown}-token growth — the resident "
        f"context was discarded (this is the 10,016-vs-24 regression)"
    )
    assert fed < len(t2) / 10                     # O(suffix), emphatically not O(prompt)
    # ...and it must come from the RESIDENT context, not a multi-GB snapshot
    # restore. On hardware `load_state` fired zero times across the whole run, so a
    # token count alone would pass for the wrong reason on a rig where the snapshot
    # path happens to work; pin the mechanism too.
    assert llm.load_state_calls == 0, "reuse came from a snapshot, not the live context"
    assert llm.reset_calls == 1, "the resident context was reset away and rebuilt"


def test_resident_context_is_not_reset_when_it_is_a_usable_prefix():
    """The proximate cause, pinned directly: no `llm.reset()` on a growing turn."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()

    p._gguf_prefill_prompt_cache(cache, _prompt(1))
    resets_after_cold = llm.reset_calls
    llm.eval([99_001])

    p._gguf_prefill_prompt_cache(cache, _prompt(2))
    assert llm.reset_calls == resets_after_cold, "the resident KV was thrown away"
    # Reuse is the in-place KV trim, not a snapshot restore.
    assert llm._ctx.removals, "kv_cache_seq_rm was never used to trim the context"
    assert llm.load_state_calls == 0


def test_snapshot_is_not_paid_for_when_the_resident_context_already_served():
    """`save_state` measured 1.41 GB per call while `load_state` never fired. It must
    not be paid on turns the resident context already covered."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()

    p._gguf_prefill_prompt_cache(cache, _prompt(1), save_state_on_live_reuse=False)
    assert llm.save_state_calls == 1              # cold turn: worth snapshotting
    llm.eval([99_001])

    p._gguf_prefill_prompt_cache(cache, _prompt(2), save_state_on_live_reuse=False)
    assert llm.save_state_calls == 1              # warm turn: snapshot adds nothing


def test_callers_that_exist_to_persist_a_snapshot_still_do():
    """`prompt_cache_update` builds durable states on purpose — the default must
    keep snapshotting even when the resident context served the prefill."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()

    p._gguf_prefill_prompt_cache(cache, _prompt(1))
    llm.eval([99_001])
    p._gguf_prefill_prompt_cache(cache, _prompt(2))     # default save_state_on_live_reuse=True

    assert llm.save_state_calls == 2
    assert len(cache.cache_state) == 2


def test_stored_state_wins_when_it_beats_the_resident_context():
    """llama.cpp's own policy (`_create_completion`): load a snapshot only when it
    offers a LONGER prefix than what is already resident."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()

    long_prompt = _prompt(1)
    cache[long_prompt] = _FakeState(long_prompt)

    # Resident context shares only a short prefix with the next prompt.
    llm.eval(list(long_prompt[:50]) + [777_777])

    target = long_prompt + tuple(range(90_001, 90_025))
    llm.eval_batches.clear()
    assert p._gguf_prefill_prompt_cache(cache, target)

    assert llm.load_state_calls == 1
    assert sum(llm.eval_batches) == len(target) - len(long_prompt)


def test_unsupported_partial_kv_removal_falls_back_to_a_full_prefill():
    """`Llama.generate` probes `kv_cache_seq_rm` before committing to reuse. A
    backend that refuses partial removal must get a clean reset and full prefill,
    never an eval against stale KV. With no snapshot to fall back to, the whole
    prompt is the honest cost."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    cache = _FakeRAMCache()

    p._gguf_prefill_prompt_cache(cache, _prompt(1), save_state=False)
    llm.eval([99_001])
    llm.eval_batches.clear()
    before_resets = llm.reset_calls

    t2 = _prompt(2)
    assert p._gguf_prefill_prompt_cache(cache, t2, save_state=False)
    assert llm.reset_calls == before_resets + 1
    assert sum(llm.eval_batches) == len(t2)       # correct, just not cheap
    assert llm.n_tokens == len(t2)


def test_unsupported_partial_kv_removal_still_uses_a_snapshot():
    """Same backend limitation, but a snapshot exists: it must be used, so the
    fallback costs the suffix rather than the whole prompt."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    cache = _FakeRAMCache()

    t1 = _prompt(1)
    p._gguf_prefill_prompt_cache(cache, t1)       # snapshots t1
    llm.eval([99_001])
    llm.eval_batches.clear()

    t2 = _prompt(2)
    assert p._gguf_prefill_prompt_cache(cache, t2)
    assert llm.load_state_calls == 1
    assert sum(llm.eval_batches) == len(t2) - len(t1)
    assert llm.n_tokens == len(t2)


def test_cold_first_turn_still_prefills_everything():
    """No resident context, no snapshot: the whole prompt is the honest cost."""
    llm = _FakeLlama()
    p = _provider(llm)
    t1 = _prompt(1)

    assert _provider(llm)._gguf_prefill_prompt_cache(_FakeRAMCache(), t1)
    assert llm.eval_batches == [len(t1)]
