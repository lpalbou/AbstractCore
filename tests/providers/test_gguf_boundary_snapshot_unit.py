"""Two GGUF prompt-cache defects, pinned as token counts and sampling positions.

DEFECT 1 — STALE LOGITS ON AN IDENTICAL RESEND (correctness).
`_gguf_prefill_prompt_cache` restored a saved state and set `prefix_len =
state_prefix_len` with no cap, while the cache stored its states keyed by the
FULL prompt. On a byte-identical resend the restored state therefore covered the
whole prompt, `remaining` was empty, `llm.eval` never ran — and llama.cpp writes
no logits when nothing is decoded (`Llama.eval`'s explicit `else: pass` under
`logits_all=False`), while `LlamaState` restores KV/`input_ids`/`n_tokens` but
NOT the context's output-logits buffer. The sampler then read the PREVIOUS call's
last decoded position.

Measured on `unsloth/Qwen3-4B-Instruct-2507-GGUF` (CPU, temp 0, fixed seed), same
prompt twice:

    cold, keyed                       first token 785  'The survey team catalogued…'
    warm, keyed (identical resend)    first token 304  ' in one sentence, what did…'
    what the stale logits predict     first token 304   <- predicted BEFORE measuring

The sibling live-context path already carries the bound and documents it
(`_gguf_live_context_prefix_len`, zipping against `prompt_tokens[:-1]`); it is
llama.cpp's own bound in `Llama.generate`.

DEFECT 2 — SNAPSHOT TAKEN AT THE VOLATILE TAIL (performance, and it made the lane
SLOWER than no cache). A chat render puts `<|im_end|>\\n<|im_start|>assistant\\n`
AFTER the user content, so turn *i*'s full prompt is not a prefix of turn *i+1*'s
and the forward-only restore is refused. Dense models hide it because the live
context can be trimmed; on Gated-DeltaNet hybrids llama.cpp REFUSES a partial
`kv_cache_seq_rm`, so there is no trim to fall back on and every grown turn
re-prefills the whole prompt. Measured before the fix: Ornith-9B@10k x0.94 and
35B-A3B@30k x0.93 versus no cache.

Both are pinned here as TOKEN COUNTS and SAMPLING POSITIONS, never timings.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import pytest

from abstractcore.providers.huggingface_provider import (
    HuggingFaceProvider,
    _GGUFPromptCacheValue,
)


# --------------------------------------------------------------------------
# llama.cpp stand-ins. The two behaviours the defects live in are modelled
# exactly: `eval` is the ONLY thing that produces logits, and partial
# `kv_cache_seq_rm` is refused on recurrent state.
# --------------------------------------------------------------------------


class _FakeCtx:
    def __init__(self, supports_partial_removal: bool = True):
        self.supports_partial_removal = supports_partial_removal
        self.removals: List[int] = []

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> bool:
        if int(p0) > 0 and not self.supports_partial_removal:
            return False  # recurrent state cannot be rewound to an arbitrary position
        self.removals.append(int(p0))
        return True


class _FakeState:
    def __init__(self, tokens: Sequence[int], size: int = 1_000_000):
        self.input_ids = [int(t) for t in tokens]
        self.n_tokens = len(self.input_ids)
        self.scores: List[float] = []
        self.llama_state = b""
        self.llama_state_size = int(size)
        self.seed = 0


class _FakeLlama:
    """Records what it was asked to evaluate, and where the sampler would read.

    `logits_pos` is the position the output-logits buffer describes. Only `eval`
    moves it — `load_state` and `reset` deliberately do NOT, which is precisely
    the llama.cpp behaviour defect 1 fell through.
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
        self.logits_pos: Optional[int] = None

    def reset(self) -> None:
        self.reset_calls += 1
        self.n_tokens = 0
        self._input_ids = []

    def eval(self, tokens: Sequence[int]) -> None:
        toks = [int(t) for t in tokens]
        self.eval_batches.append(len(toks))
        self._input_ids = self._input_ids[: self.n_tokens] + toks
        self.n_tokens += len(toks)
        if toks:
            self.logits_pos = self.n_tokens - 1

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
    """`LlamaRAMCache`-shaped: token-tuple -> state, insertion ordered."""

    def __init__(self):
        self.cache_state: dict = {}

    def __setitem__(self, key: Sequence[int], value: Any) -> None:
        k = tuple(int(t) for t in key)
        self.cache_state.pop(k, None)
        self.cache_state[k] = value


def _provider(llm: _FakeLlama) -> HuggingFaceProvider:
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.llm = llm
    return p


HEAD = 1_200  # stable transcript head, as in the hardware probe
SCAFFOLD = (90_001, 90_002, 90_003, 90_004, 90_005)  # the generation prompt


def _turn(i: int) -> Tuple[int, ...]:
    """Turn *i* of a growing conversation whose VOLATILE TAIL is rewritten.

    `head + grown user text + generation scaffolding`. Turn *i* is deliberately
    NOT a prefix of turn *i+1* — that is the whole shape of the defect, and it is
    what a chat template produces for every real agent loop.
    """
    body = tuple(range(1, HEAD + 1)) + tuple(range(50_000, 50_000 + 4 * i))
    return body + SCAFFOLD


def _generation_boundary(i: int) -> int:
    return len(_turn(i)) - len(SCAFFOLD)


# --------------------------------------------------------------------------
# DEFECT 1 — never sample from logits this prompt did not produce
# --------------------------------------------------------------------------


def test_identical_resend_evaluates_at_least_one_token():
    """THE PIN for defect 1. A state covering the whole prompt must still leave a
    token to feed, or the sampler reads the previous call's logits."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()

    prompt = _turn(1)
    cache[prompt] = _FakeState(prompt)  # what every pre-fix turn stored

    # A previous call left its own logits behind, at a different position.
    llm.eval([7, 7, 7])
    stale_pos = llm.logits_pos
    llm.eval_batches.clear()

    assert p._gguf_prefill_prompt_cache(cache, prompt)

    fed = sum(llm.eval_batches)
    assert fed >= 1, "zero tokens evaluated — the sampler would read stale logits"
    assert llm.logits_pos == len(prompt) - 1, (
        f"logits describe position {llm.logits_pos}, not this prompt's last "
        f"position {len(prompt) - 1} (stale position was {stale_pos})"
    )
    assert llm.n_tokens == len(prompt), "the context no longer holds exactly the prompt"
    assert fed == 1, f"one forward pass is the whole cost; fed {fed}"


def test_identical_resend_is_correct_even_when_the_rollback_is_refused():
    """Recurrent state cannot be rewound one token, so the honest answer is a full
    re-prefill. Correctness is not optional; the speed of this path is."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    cache = _FakeRAMCache()

    prompt = _turn(1)
    cache[prompt] = _FakeState(prompt)
    llm.eval([7, 7, 7])
    llm.eval_batches.clear()

    assert p._gguf_prefill_prompt_cache(cache, prompt)
    assert sum(llm.eval_batches) == len(prompt)
    assert llm.logits_pos == len(prompt) - 1
    assert llm.n_tokens == len(prompt)


def test_a_boundary_snapshot_makes_the_resend_cheap_and_correct_at_once():
    """With the boundary policy the identical resend never needs a rollback: the
    stored boundary stops before the volatile tail, so the tail is the feed."""
    llm = _FakeLlama(supports_partial_removal=False)  # hybrid
    p = _provider(llm)
    cache = _FakeRAMCache()

    prompt = _turn(1)
    assert p._gguf_prefill_prompt_cache(
        cache, prompt, snapshot_at_boundary=True,
        generation_boundary=_generation_boundary(1),
    )
    llm.eval([99_001, 99_002])  # the reply
    llm.eval_batches.clear()

    assert p._gguf_prefill_prompt_cache(
        cache, prompt, snapshot_at_boundary=True,
        prev_prompt_tokens=prompt,
    )
    assert llm.load_state_calls == 1
    assert sum(llm.eval_batches) == len(SCAFFOLD)
    assert llm.logits_pos == len(prompt) - 1
    assert llm.n_tokens == len(prompt)


def test_empty_prompt_is_refused_rather_than_sampled():
    """No prompt means no logits; failing loudly beats sampling whatever was left."""
    llm = _FakeLlama()
    assert _provider(llm)._gguf_prefill_prompt_cache(_FakeRAMCache(), ()) is False


def test_every_lane_leaves_fresh_logits():
    """Sweep the reachable (backend, cache, residency) combinations and assert the
    invariant once, so a future path cannot reintroduce an empty feed quietly."""
    prompt = _turn(2)
    for partial in (True, False):
        for preload in (None, "full", "boundary"):
            for resident in ([], [7, 7, 7], list(prompt[:-1])):
                llm = _FakeLlama(supports_partial_removal=partial)
                p = _provider(llm)
                cache = _FakeRAMCache()
                if preload == "full":
                    cache[prompt] = _FakeState(prompt)
                elif preload == "boundary":
                    cache[prompt[:-6]] = _FakeState(prompt[:-6])
                if resident:
                    llm.eval(resident)
                assert p._gguf_prefill_prompt_cache(cache, prompt)
                assert llm.n_tokens == len(prompt)
                assert llm.logits_pos == len(prompt) - 1, (
                    f"stale logits for partial={partial} preload={preload} "
                    f"resident={len(resident)}"
                )


# --------------------------------------------------------------------------
# DEFECT 2 — snapshot before the volatile tail
# --------------------------------------------------------------------------


def _run_loop(llm: _FakeLlama, p: HuggingFaceProvider, cache: Any, turns: int,
              *, boundary: bool) -> List[int]:
    """Drive `turns` grown turns the way the generate lane does, returning the
    tokens actually evaluated per turn."""
    fed_per_turn: List[int] = []
    prev: Tuple[int, ...] = ()
    for i in range(1, turns + 1):
        prompt = _turn(i)
        llm.eval_batches.clear()
        kwargs: dict = {"save_state_on_live_reuse": False}
        if boundary:
            kwargs.update(
                snapshot_at_boundary=True,
                prev_prompt_tokens=prev,
                generation_boundary=_generation_boundary(i),
            )
        assert p._gguf_prefill_prompt_cache(cache, prompt, **kwargs)
        fed_per_turn.append(sum(llm.eval_batches))
        llm.eval([99_000 + i])  # the reply, left resident as generation would
        prev = prompt
    return fed_per_turn


def test_grown_turn_on_a_recurrent_backend_feeds_only_the_suffix():
    """THE PIN for defect 2, on the architecture that fails: Gated-DeltaNet, where
    partial `kv_cache_seq_rm` is refused so a snapshot is the only reuse there is.

    Hardware probe this mirrors: 1216, 1220, 1224, 1228 evaluated tokens before,
    1216, 20, 24, 28 after (turn 4: 43.9x).

    Steady state here is a CONSTANT 13 = this turn's 4 grown tokens + the 5
    rewritten scaffolding tokens + the 4 tokens of the previous turn's growth that
    the boundary necessarily lags by (a boundary can only be placed where two
    consecutive prompts have been observed to agree)."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    fed = _run_loop(llm, p, _FakeRAMCache(), 7, boundary=True)

    assert fed[0] == len(_turn(1)), "turn 1 is an honest cold prefill"
    for i in range(1, 7):
        prompt = _turn(i + 1)
        assert fed[i] < len(prompt) / 10, (
            f"turn {i + 1} fed {fed[i]} of {len(prompt)} tokens — the whole prompt "
            f"was re-prefilled (the x0.94-vs-no-cache regression)"
        )
    assert fed[1] == 4 + len(SCAFFOLD), fed
    assert fed[2:] == [4 + 4 + len(SCAFFOLD)] * 5, fed
    assert max(fed[1:]) <= fed[1] + 4, "per-turn cost must be flat, not accumulating"


def test_grown_turns_regress_without_the_boundary_policy():
    """The control that proves the test above is measuring the boundary and not
    something incidental: with the full-prompt snapshot, every grown turn pays for
    the whole prompt on this backend."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    fed = _run_loop(llm, p, _FakeRAMCache(), 7, boundary=False)
    for i in range(1, 7):
        assert fed[i] == len(_turn(i + 1))


def test_dense_backend_is_untouched_by_the_boundary_policy():
    """Plain attention models were healthy (14 tokens/turn measured) and must stay
    on the in-place live-context trim, paying no snapshot at all."""
    for boundary in (False, True):
        llm = _FakeLlama(supports_partial_removal=True)
        p = _provider(llm)
        fed = _run_loop(llm, p, _FakeRAMCache(), 4, boundary=boundary)
        assert fed[0] == len(_turn(1))
        for i in range(1, 4):
            # growth + rewritten scaffolding + the resident reply token
            assert fed[i] <= 4 + len(SCAFFOLD) + 1, f"boundary={boundary} fed {fed[i]}"
        assert llm.load_state_calls == 0, "a dense grow must not need a snapshot"


def test_turn_one_boundary_is_the_renderer_supplied_position():
    """Turn 1 has no previous prompt to difference against, so the holdback comes
    from the renderer. The stored key must stop exactly at the scaffolding."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    cache = _FakeRAMCache()
    prompt = _turn(1)

    assert p._gguf_prefill_prompt_cache(
        cache, prompt, snapshot_at_boundary=True,
        generation_boundary=_generation_boundary(1),
    )
    keys = list(cache.cache_state.keys())
    assert keys == [tuple(prompt[: _generation_boundary(1)])], keys
    assert llm.n_tokens == len(prompt), "the context must still hold the FULL prompt"


def test_without_a_renderer_boundary_the_snapshot_still_leaves_a_token_to_feed():
    """No holdback available is not a licence to store the full prompt: the
    `len - 1` floor is what keeps defect 1 structurally impossible."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()
    prompt = _turn(1)
    assert p._gguf_prefill_prompt_cache(cache, prompt, snapshot_at_boundary=True)
    assert list(cache.cache_state.keys()) == [tuple(prompt[:-1])]


def test_boundary_snapshots_are_bounded():
    """A `LlamaState` is the whole serialized context (multi-GB at bench sizes).
    One per turn of a long loop is a leak."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    cache = _FakeRAMCache()
    _run_loop(llm, p, cache, 12, boundary=True)
    assert len(cache.cache_state) <= p._gguf_snapshot_bound(), (
        f"{len(cache.cache_state)} states retained for one key"
    )


def test_durable_prefix_is_never_pruned():
    """`prompt_cache_save` looks its state up verbatim by the bloc's own tokens."""
    llm = _FakeLlama(supports_partial_removal=False)
    p = _provider(llm)
    cache = _FakeRAMCache()
    bloc = tuple(range(1, 401))
    cache[bloc] = _FakeState(bloc)

    prev: Tuple[int, ...] = ()
    for i in range(1, 8):
        prompt = _turn(i)
        assert p._gguf_prefill_prompt_cache(
            cache, prompt, snapshot_at_boundary=True,
            prev_prompt_tokens=prev, generation_boundary=_generation_boundary(i),
            protect_snapshot_key=bloc, save_state_on_live_reuse=False,
        )
        llm.eval([99_000 + i])
        prev = prompt
    assert bloc in cache.cache_state


def test_non_generate_callers_keep_the_full_prompt_boundary():
    """`prompt_cache_update` / `prompt_cache_save` build durable states on purpose
    and index them by the FULL prompt; the default must not move."""
    llm = _FakeLlama()
    p = _provider(llm)
    cache = _FakeRAMCache()
    prompt = _turn(1)
    assert p._gguf_prefill_prompt_cache(cache, prompt)  # defaults
    assert tuple(prompt) in cache.cache_state


# --------------------------------------------------------------------------
# The boundary lattice itself, and where the boundary comes from
# --------------------------------------------------------------------------


def test_boundary_lattice():
    llm = _FakeLlama()
    p = _provider(llm)
    cur = _turn(3)

    # No record, no renderer answer: the len-1 floor.
    assert p._gguf_snapshot_boundary(cur, prefix_len=0) == len(cur) - 1
    # No record, renderer answer: hold the scaffolding back.
    assert p._gguf_snapshot_boundary(
        cur, prefix_len=0, generation_boundary=_generation_boundary(3)
    ) == _generation_boundary(3)
    # Record whose tail was rewritten: the first divergence.
    prev = _turn(2)
    shared = HEAD + 8
    assert p._gguf_snapshot_boundary(cur, prefix_len=0, prev_prompt_tokens=prev) == shared
    # Record that IS a prefix (append-only caller): nothing to hold back.
    assert p._gguf_snapshot_boundary(
        cur, prefix_len=0, prev_prompt_tokens=cur[:100]
    ) == len(cur) - 1
    # A restored boundary is a floor, never regressed below.
    assert p._gguf_snapshot_boundary(
        cur, prefix_len=len(cur) - 2, prev_prompt_tokens=prev
    ) == len(cur) - 2


def test_generation_prompt_boundary_is_derived_from_the_renderer():
    """Not a literal list: the position is obtained by asking the SAME renderer for
    the same messages without the generation prompt. A hardcoded tail list was
    previously inert for 8 of 10 template families."""
    llm = _FakeLlama()
    p = _provider(llm)
    full = _turn(2)
    head = full[: _generation_boundary(2)]

    def _render(*, messages, add_generation_prompt, enable_thinking=None, reasoning_effort=None):
        return ("<text>", full if add_generation_prompt else head)

    p._gguf_render_prompt_tokens = _render  # type: ignore[assignment]
    assert p._gguf_generation_prompt_boundary(
        messages=[{"role": "user", "content": "x"}], prompt_tokens=full
    ) == len(head)


def test_generation_prompt_boundary_declines_rather_than_guesses():
    llm = _FakeLlama()
    p = _provider(llm)
    full = _turn(2)

    def _raises(**_kwargs):
        raise ValueError("this template refuses this conversation shape")

    p._gguf_render_prompt_tokens = _raises  # type: ignore[assignment]
    assert p._gguf_generation_prompt_boundary(messages=[], prompt_tokens=full) is None

    # A renderer that emits no generation prompt has nothing volatile to hold back.
    p._gguf_render_prompt_tokens = (  # type: ignore[assignment]
        lambda **_k: ("<text>", full)
    )
    assert p._gguf_generation_prompt_boundary(messages=[], prompt_tokens=full) is None

    # A template that does more than append can only cost reuse, never correctness:
    # the LCP is a prefix of the prompt by construction.
    p._gguf_render_prompt_tokens = (  # type: ignore[assignment]
        lambda **_k: ("<text>", full[:50] + (999_999,) * 20)
    )
    assert p._gguf_generation_prompt_boundary(messages=[], prompt_tokens=full) == 50


def test_fed_record_is_not_the_durable_bloc_prefix():
    """`prompt_tokens` is the durable-bloc prefix that
    `_gguf_compose_cached_prompt_tokens` CONCATENATES a suffix onto. The per-turn
    boundary record must live somewhere else or a divergent turn would compose
    previous-prompt + new-prompt into a garbage prompt."""
    v = _GGUFPromptCacheValue(cache=None, capacity_bytes=0)
    assert v.prompt_tokens == ()
    assert v.fed_prompt_tokens == ()
    v.fed_prompt_tokens = (1, 2, 3)
    assert v.prompt_tokens == ()
