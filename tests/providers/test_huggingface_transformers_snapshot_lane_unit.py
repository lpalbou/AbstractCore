"""Snapshot/restore lane for untrimmable transformers architectures (2026-08-04).

CONTEXT. `Cache.crop` is an explicit no-op on linear-attention layers, so
`_transformers_crop_cache` refuses Gated-DeltaNet hybrids (Qwen3.5/3.6,
Ornith) and — before this lane — every warm full-context call rebuilt fresh:
correct output, ZERO savings (measured x0.963 vs no-cache at 10k on
`Qwen/Qwen3.5-4B` bf16/MPS, `results/bench_b/hf_bf16_10000_postfix.json`).
The MLX provider solved the identical problem with snapshot-before-decode:
deepcopy the cache at a clean boundary BEFORE generation mutates it, restore
forward-only when the stored ids are a TRUE PREFIX of the new prompt, feed
only the suffix, never roll back. This file pins the transformers port.

These tests run the REAL cached lane (`_single_generate_transformers_cached`)
with the REAL tokenizer and renderer; only the model forward/generate are
stubbed (they record every fed batch and grow the fake cache). On pre-port
code the hybrid suffix-feeding tests FAIL (every warm turn feeds the full
prompt) — that is the point; a port test that passes on the old code is
worthless. The dense-path tests assert the croppable lane still behaves
exactly as before and never touches snapshot machinery.
"""

from __future__ import annotations

import copy
import gc
import os
import weakref
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

torch = pytest.importorskip("torch")

from abstractcore.architectures import get_architecture_format
from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.huggingface_provider import (
    HuggingFaceProvider,
    _TransformersPromptCacheValue,
)

MODEL_ID = "Qwen/Qwen3.5-4B"
ANSWER_TOKEN = 9  # arbitrary fixed generated token id


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except Exception:
        return None


_TOKENIZER = _load_tokenizer()

pytestmark = pytest.mark.skipif(
    _TOKENIZER is None,
    reason=f"{MODEL_ID} tokenizer not in the local HF cache (offline unit test)",
)


# ---------------------------------------------------------------- fake caches
class _FullLayer:
    """DynamicLayer stand-in: croppable, reports a length."""

    def __init__(self, seq_len: int = 0):
        self.seq_len = int(seq_len)

    def crop(self, max_length: int) -> None:
        if self.seq_len > int(max_length):
            self.seq_len = int(max_length)

    def get_seq_length(self) -> int:
        return self.seq_len


class _LinearLayer:
    """Linear-attention/recurrent layer: crop is a no-op, no length to report
    — `_transformers_uncroppable_layers` must flag it (same predicate the
    crop-refusal guard uses)."""

    def __init__(self):
        self.recurrent_tokens = 0

    def crop(self, max_length: int) -> None:
        return None


class _FakeCache:
    def __init__(self, layers: List[Any]):
        self.layers = layers

    def grow(self, n: int) -> None:
        for layer in self.layers:
            if isinstance(layer, _FullLayer):
                layer.seq_len += int(n)
            else:
                layer.recurrent_tokens += int(n)

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        for layer in self.layers[layer_idx:]:
            fn = getattr(layer, "get_seq_length", None)
            if callable(fn):
                return int(fn())
        return 0


def _hybrid_cache() -> _FakeCache:
    """Qwen3.5-4B's shape: 24 linear / 8 full of 32, linear-first."""
    return _FakeCache(
        [_FullLayer(0) if i % 4 == 3 else _LinearLayer() for i in range(32)]
    )


def _dense_cache() -> _FakeCache:
    return _FakeCache([_FullLayer(0) for _ in range(8)])


# ---------------------------------------------------------------- stub model
class _StubModel:
    """Records every fed batch; grows the cache like a real forward/generate."""

    def __init__(self, trace: List[Any]):
        self.trace = trace

    def parameters(self):
        return iter(())

    def __call__(self, input_ids=None, attention_mask=None, past_key_values=None, **kw):
        n = int(input_ids.shape[-1])
        self.trace.append(("forward", n))
        if past_key_values is not None:
            past_key_values.grow(n)
        return SimpleNamespace(past_key_values=past_key_values)

    def generate(self, input_ids=None, attention_mask=None, past_key_values=None,
                 max_new_tokens=1, **kw):
        n = int(input_ids.shape[-1])
        self.trace.append(("generate", n))
        if past_key_values is not None:
            past_key_values.grow(n + int(max_new_tokens))
        seq = input_ids[0].tolist() + [ANSWER_TOKEN] * int(max_new_tokens)
        return SimpleNamespace(
            sequences=[torch.tensor(seq, dtype=torch.long)],
            past_key_values=past_key_values,
        )


class _Log:
    def __init__(self):
        self.warnings: List[str] = []

    def warning(self, msg: str, *a: Any, **k: Any) -> None:
        self.warnings.append(str(msg))

    def debug(self, *a: Any, **k: Any) -> None:
        pass

    info = error = debug


# ---------------------------------------------------------------- provider rig
def _provider(cache_factory, *, max_entries: int = 32):
    trace: List[Any] = []
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.logger = _Log()
    p.model = MODEL_ID
    p.model_type = "transformers"
    p.device = "cpu"
    p.architecture = "qwen3_5"
    p.architecture_config = get_architecture_format("qwen3_5")
    p.tool_handler = None
    p.tokenizer = _TOKENIZER
    p.model_instance = _StubModel(trace)
    p.temperature = 0.0
    p._transformers_logits_to_keep_supported = False
    p._transformers_prompt_cache_supported = lambda: True  # type: ignore[method-assign]
    p._transformers_empty_native_cache = cache_factory  # type: ignore[method-assign]
    p._default_prompt_cache_key = None
    p._prompt_cache_store = PromptCacheStore(
        max_entries=max_entries, on_evict=p._prompt_cache_store_evicted
    )
    return p, trace


def _seed_key(p, key: str, cache_factory) -> None:
    p._prompt_cache_store.set(
        key,
        _TransformersPromptCacheValue(cache=cache_factory()),
        meta={"backend": "transformers"},
    )


_DOC = " ".join(
    f"Fact {i}: the study of item {i} shows a measured value of {i * 7} units."
    for i in range(60)
)
_Q = "What does the middle of the document say? Answer briefly."


def _call(p, key: str, messages: List[Dict[str, str]]):
    return p._single_generate_transformers_cached(
        prompt="",
        prompt_cache_key=key,
        messages=messages,
        system_prompt=_DOC,
        tools=None,
        prefilled_modules=None,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=1234,
        enable_thinking=False,
    )


def _agent_loop(p, trace, key: str, turns: int = 4):
    """Full-context caller, monotonically growing transcript (the agent-loop
    shape). Returns per-turn dicts with the prompt length, tokens actually fed
    to the model this turn, and the lane's telemetry."""
    hist = [{"role": "user", "content": _Q}]
    out = []
    for t in range(turns):
        full_text = p._transformers_build_prompt_fragment(
            prompt="", messages=list(hist), system_prompt=_DOC, tools=None,
            add_generation_prompt=True, prefilled_modules=None, enable_thinking=False,
        )
        full_len = len(p._transformers_tokenize_fragment(full_text, add_bos_if_empty=True))
        before = len(trace)
        resp = _call(p, key, list(hist))
        fed = sum(n for _op, n in trace[before:])
        meta = (getattr(resp, "metadata", None) or {}).get("prompt_cache") or {}
        out.append({"turn": t + 1, "full_len": full_len, "fed": fed,
                    "telemetry": meta, "content": resp.content})
        hist = list(hist) + [
            {"role": "assistant", "content": f"Noted observation {t}."},
            {"role": "user", "content": f"Follow-up {t}: {_Q}"},
        ]
    return out


# ===================================================================== hybrid
def test_hybrid_warm_turns_feed_only_the_suffix():
    """THE PIN (fails on pre-port code, where every warm turn re-fed the full
    prompt). Warm turns must feed strictly less than half the prompt — in
    practice a few dozen tokens: the previous turn's rewritten tail plus the
    new user turn."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "loop", _hybrid_cache)

    turns = _agent_loop(p, trace, "loop", turns=4)

    assert turns[0]["fed"] >= turns[0]["full_len"] - 1  # turn 1 is a real cold prefill
    for t in turns[1:]:
        assert t["fed"] < 0.5 * t["full_len"], (
            f"turn {t['turn']} fed {t['fed']} of {t['full_len']} tokens — "
            f"the snapshot lane is not feeding suffix-only"
        )


def test_hybrid_turn2_restores_thanks_to_the_generation_boundary_holdback():
    """The MLX edit-5 lesson, pinned for transformers: turn 1's snapshot
    boundary must exclude the generation scaffolding (assistant header +
    thinking-disabled block, DERIVED from the renderer), otherwise turn 2's
    forward-only restore is refused and the loop rebuilds. Turn 2 must
    already be `hit_restore` with a majority of the prompt served cached."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "t2", _hybrid_cache)

    turns = _agent_loop(p, trace, "t2", turns=4)

    # Turn 1 on a freshly-seeded (empty) key reports `cold`, not `rebuilt`
    # (parity relabel, 2026-08-07): nothing existed to discard, and the MLX
    # snapshot lane and the GGUF lane both report `cold` in this exact state.
    assert turns[0]["telemetry"].get("outcome") == "cold"
    for t in turns[1:]:
        tele = t["telemetry"]
        assert tele.get("outcome") == "hit_restore", (
            f"turn {t['turn']}: {tele} — expected a forward-only snapshot restore"
        )
        assert tele.get("cached_tokens", 0) > 0.5 * t["full_len"]
        assert tele.get("fed_tokens") == t["fed"]  # telemetry matches the hook truth


def test_hybrid_snapshot_is_taken_before_generation_mutates_the_cache():
    """The stored snapshot must describe a PRE-DECODE boundary: its ids must be
    a true prefix of the next prompt (never containing generated tokens), and
    the snapshot cache's own token count must equal the boundary length."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "b", _hybrid_cache)

    _agent_loop(p, trace, "b", turns=2)

    snap = p._transformers_snapshots.get("b")
    assert snap is not None
    ids = snap["ids"]
    assert ANSWER_TOKEN not in ids  # generated tokens never enter a boundary
    counts = {l.recurrent_tokens for l in snap["cache"].layers if isinstance(l, _LinearLayer)}
    assert counts == {len(ids)}, "snapshot cache holds a different token count than its ids"


def test_hybrid_identical_resend_is_served_from_the_snapshot():
    """The benchmark C_warm_same shape: identical full-context resends on one
    key. Every warm rep must restore and feed only the held-back scaffolding
    tail (single-digit-to-tens of tokens), never the full prompt."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "warm", _hybrid_cache)
    msgs = [{"role": "user", "content": _Q}]

    _call(p, "warm", msgs)  # cold
    for rep in range(3):
        before = len(trace)
        resp = _call(p, "warm", msgs)
        fed = sum(n for _op, n in trace[before:])
        tele = (resp.metadata or {}).get("prompt_cache") or {}
        assert tele.get("outcome") == "hit_restore", f"rep {rep}: {tele}"
        assert fed <= 64, f"rep {rep} fed {fed} tokens on an identical resend"


def test_hybrid_divergent_prompt_rebuilds_never_stale():
    """A prompt that diverges from everything recorded must pay one honest
    cold prefill (never serve a stale boundary): outcome=rebuilt, cached 0,
    full feed."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "d", _hybrid_cache)
    _agent_loop(p, trace, "d", turns=2)

    divergent = [{"role": "user", "content": "Entirely different question about nothing."}]
    before = len(trace)
    resp = _call(p, "d", divergent)
    fed = sum(n for _op, n in trace[before:])
    tele = (resp.metadata or {}).get("prompt_cache") or {}

    assert tele.get("outcome") == "rebuilt"
    assert tele.get("cached_tokens") == 0
    assert fed >= tele.get("fed_tokens", 0) > 0


def test_hybrid_crop_is_never_called():
    """The snapshot lane is forward-only by construction: the refused `crop`
    must not even be attempted on the untrimmable architecture (the refusal
    guard itself stays for the non-hybrid refusal cases and is covered by
    test_huggingface_transformers_crop_refusal_unit.py)."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "nc", _hybrid_cache)
    calls = {"crop": 0}
    orig = HuggingFaceProvider._transformers_crop_cache

    def spy(self, state, keep):
        calls["crop"] += 1
        return orig(self, state, keep)

    p._transformers_crop_cache = spy.__get__(p)  # type: ignore[method-assign]
    _agent_loop(p, trace, "nc", turns=3)

    assert calls["crop"] == 0


# ====================================================================== dense
def test_dense_path_untouched_no_snapshots_and_crop_still_runs():
    """Pure-attention models must keep the existing crop/delta lane: crops
    happen, suffixes are fed, and NO snapshot machinery is ever engaged (the
    port's routing predicate must not leak onto the dense path)."""
    p, trace = _provider(_dense_cache)
    _seed_key(p, "dense", _dense_cache)

    turns = _agent_loop(p, trace, "dense", turns=4)

    assert getattr(p, "_transformers_snapshots", {}) == {}

    # TELEMETRY CONTRACT CHANGED DELIBERATELY (parity, 2026-08-07).
    # This assertion used to be `t["telemetry"] == {}`. That was written to
    # prove the snapshot machinery does not leak onto the dense path — a good
    # intent — but it pinned the crop lane to reporting NOTHING AT ALL, which
    # made it the only local cache lane a user could not observe. MLX and GGUF
    # both report every call; the crop lane now does too, in the same vocabulary.
    # The no-leak intent is preserved and made sharper below: the lane label must
    # be "crop", never "snapshot".
    for t in turns:
        tel = t["telemetry"]
        assert tel, f"dense turn {t['turn']} reported no prompt-cache telemetry"
        assert tel["lane"] == "crop", f"snapshot machinery leaked onto the dense path: {tel}"
        assert tel["mode"] == "key" and tel["key"] == "dense"
        # `fed_tokens` is REAL, not a label: the crop lane's defining identity is
        # that the reused prefix plus the fed suffix is exactly the whole prompt.
        assert tel["cached_tokens"] + tel["fed_tokens"] == t["full_len"], (
            f"dense turn {t['turn']} telemetry {tel} does not account for "
            f"full_len={t['full_len']}"
        )
        assert 0 < tel["fed_tokens"] <= t["fed"]

    assert turns[0]["telemetry"]["outcome"] == "cold"
    assert turns[0]["telemetry"]["cached_tokens"] == 0
    assert turns[0]["fed"] >= turns[0]["full_len"] - 1
    for t in turns[1:]:
        # warm dense turns crop generated drift and feed only the suffix
        assert t["telemetry"]["outcome"] == "hit_extend"
        assert t["telemetry"]["cached_tokens"] > 0
        assert t["fed"] < 0.5 * t["full_len"], (
            f"dense turn {t['turn']} fed {t['fed']} of {t['full_len']}"
        )
        assert t["telemetry"]["fed_tokens"] < 0.5 * t["full_len"]


def test_dense_trace_matches_golden_when_provided():
    """Byte-identity instrument for the A/B run: the ordered (op, n) trace of
    the dense loop. The catches-it runner records this on PRE-PORT code into
    a golden JSON and asserts the POST-PORT trace equals it exactly. Without
    the env var the comparison is skipped (the invariants above still run)."""
    golden_path = os.environ.get("ABSTRACTCORE_DENSE_TRACE_GOLDEN")
    p, trace = _provider(_dense_cache)
    _seed_key(p, "golden", _dense_cache)

    _agent_loop(p, trace, "golden", turns=4)

    if not golden_path:
        pytest.skip("no golden trace provided (set ABSTRACTCORE_DENSE_TRACE_GOLDEN)")
    import json

    if os.environ.get("ABSTRACTCORE_DENSE_TRACE_RECORD") == "1":
        with open(golden_path, "w") as fh:
            json.dump([[op, n] for op, n in trace], fh)
    with open(golden_path) as fh:
        golden = [tuple(x) for x in json.load(fh)]
    assert [tuple(x) for x in trace] == golden, "dense trace changed across the port"


# ============================================================== memory bounds
def test_snapshot_count_is_hard_bounded_lru():
    """N distinct keys must never hold more than the bound in snapshots, and
    the evicted deepcopies must actually be collectable (no secondary refs)."""
    os.environ["ABSTRACTCORE_TRANSFORMERS_SNAPSHOT_BOUND"] = "4"
    try:
        p, trace = _provider(_hybrid_cache)
        refs = []
        for i in range(9):
            key = f"k{i}"
            _seed_key(p, key, _hybrid_cache)
            _call(p, key, [{"role": "user", "content": _Q}])
            snap = p._transformers_snapshots.get(key)
            if snap is not None:
                refs.append(weakref.ref(snap["cache"]))
        assert len(p._transformers_snapshots) <= 4
        gc.collect()
        alive = sum(1 for r in refs if r() is not None)
        assert alive <= 4, f"{alive} snapshot caches alive with bound 4 — leaked references"
    finally:
        os.environ.pop("ABSTRACTCORE_TRANSFORMERS_SNAPSHOT_BOUND", None)


def test_store_eviction_drops_the_keys_snapshot():
    """PromptCacheStore LRU eviction fires `_prompt_cache_store_evicted`; the
    evicted key's snapshot must go with it (the MLX leak fix's exact
    contract, ported)."""
    p, trace = _provider(_hybrid_cache, max_entries=2)
    for key in ("a", "b", "c"):
        _seed_key(p, key, _hybrid_cache)  # seeding c evicts a (max_entries=2)
        _call(p, key, [{"role": "user", "content": _Q}])

    assert "a" not in p._transformers_snapshots
    assert set(p._transformers_snapshots) <= {"b", "c"}


def test_prompt_cache_clear_drops_snapshots():
    p, trace = _provider(_hybrid_cache)
    for key in ("x", "y"):
        _seed_key(p, key, _hybrid_cache)
        _call(p, key, [{"role": "user", "content": _Q}])
    assert set(p._transformers_snapshots) == {"x", "y"}

    p.prompt_cache_clear("x")
    assert set(p._transformers_snapshots) == {"y"}

    p.prompt_cache_clear(None)
    assert p._transformers_snapshots == {}


def test_one_key_keeps_exactly_one_snapshot_across_a_loop():
    """The growing boundary must REPLACE its predecessor in place — a 6-turn
    loop on one key leaves exactly one resident snapshot."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "one", _hybrid_cache)

    _agent_loop(p, trace, "one", turns=6)

    assert list(p._transformers_snapshots) == ["one"]


# ============================================================ chunked prefill
# A one-shot long prefill materializes an [heads, L, L] fp32 score transient
# when torch's SDPA falls back to math — on MPS at 30k that is a single
# 107.15 GiB MTLBuffer and Metal ABORTS THE PROCESS (measured twice:
# `Failed to allocate private MTLBuffer for size 115054126208` = 32 x
# 29981^2 x 4 exactly). The lanes must therefore never feed more than the
# chunk step to a single forward. The Metal abort itself cannot be
# reproduced on CPU; what CAN be pinned is the batch-size contract.

def _with_step(step: int):
    os.environ["ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP"] = str(step)


def _pop_step():
    os.environ.pop("ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP", None)


def test_long_cold_prefill_is_chunked_on_the_cached_lane():
    """No single forward/generate batch may exceed the chunk step on a cold
    full-context prefill (dense AND hybrid) — the exact shape that aborted
    the dense 30k cell twice."""
    _with_step(64)
    try:
        for factory in (_dense_cache, _hybrid_cache):
            p, trace = _provider(factory)
            _seed_key(p, "chunk", factory)
            _call(p, "chunk", [{"role": "user", "content": _Q}])
            biggest = max(n for _op, n in trace)
            assert biggest <= 64, (
                f"{factory.__name__}: a single batch of {biggest} tokens was fed — "
                f"one-shot prefill aborts the process at 30k on MPS"
            )
            total = sum(n for _op, n in trace)
            full = trace and total
            assert total > 64  # the whole prompt still went through
    finally:
        _pop_step()


def test_small_deltas_keep_the_single_shot_path():
    """Below the step nothing changes: the golden-trace test already pins the
    dense case; this pins that a small delta is one generate() call."""
    p, trace = _provider(_dense_cache)
    _seed_key(p, "small", _dense_cache)
    _call(p, "small", [{"role": "user", "content": _Q}])

    gens = [n for op, n in trace if op == "generate"]
    assert len(gens) == 1 and gens[0] > 64  # one-shot, whole prompt


def test_exception_mid_generate_resets_state_and_next_turn_restores_clean():
    """ADVERSARY FINDING 1 pin. generate() can raise AFTER mutating the cache
    (MPS OOM mid-decode): bookkeeping then misdescribes the physical KV, and
    without the except-path reset the next turn live-seeds over phantom
    tokens and stores a POISONED snapshot (ids misdescribing its cache) —
    silent wrong attention, self-propagating. The reset makes the next turn
    restore from the still-clean pre-decode snapshot."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "boom", _hybrid_cache)
    inner = p.model_instance

    class _Exploder:
        explode = False

        def parameters(self):
            return iter(())

        def __call__(self, *a, **kw):
            return inner(*a, **kw)

        def generate(self, input_ids=None, past_key_values=None, **kw):
            if _Exploder.explode:
                _Exploder.explode = False
                if past_key_values is not None:
                    past_key_values.grow(int(input_ids.shape[-1]) + 1)  # mutate, then die
                raise RuntimeError("synthetic MPS OOM mid-decode")
            return inner.generate(input_ids=input_ids, past_key_values=past_key_values, **kw)

    p.model_instance = _Exploder()
    msgs = [{"role": "user", "content": _Q}]
    _call(p, "boom", msgs)  # turn 1: cold, snapshot stored

    msgs2 = msgs + [{"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": f"Follow-up: {_Q}"}]
    _Exploder.explode = True
    with pytest.raises(RuntimeError):
        _call(p, "boom", msgs2)

    state = p._transformers_prompt_cache_state(p._prompt_cache_store.get("boom"))
    assert state.prompt_tokens == () and state.cache is None  # reset, not poisoned

    msgs3 = msgs2 + [{"role": "assistant", "content": "Noted again."},
                     {"role": "user", "content": f"Follow-up 2: {_Q}"}]
    resp = _call(p, "boom", msgs3)
    tele = (resp.metadata or {}).get("prompt_cache") or {}
    assert tele.get("outcome") == "hit_restore"  # served from the clean snapshot

    snap = p._transformers_snapshots.get("boom")
    counts = {l.recurrent_tokens for l in snap["cache"].layers if isinstance(l, _LinearLayer)}
    assert counts == {len(snap["ids"])}  # stored boundary describes its cache
    state3 = p._transformers_prompt_cache_state(p._prompt_cache_store.get("boom"))
    phys = {l.recurrent_tokens for l in state3.cache.layers if isinstance(l, _LinearLayer)}
    assert phys == {len(state3.prompt_tokens)}  # live KV matches the mask arithmetic


def test_unload_model_releases_snapshots_and_routing_flag():
    """ADVERSARY FINDING 2 pin. Snapshots are the largest tensors the provider
    holds; an unload that strands them frees almost nothing, and a stale
    routing flag would mis-route the next model loaded onto this instance."""
    p, trace = _provider(_hybrid_cache)
    _seed_key(p, "u", _hybrid_cache)
    _call(p, "u", [{"role": "user", "content": _Q}])
    assert p._transformers_snapshots and hasattr(p, "_transformers_snapshot_lane_flag")

    p.unload_model(MODEL_ID)

    assert p._transformers_snapshots == {}
    assert not hasattr(p, "_transformers_snapshot_lane_flag")


def test_uncached_long_prompt_takes_the_chunked_manual_path():
    """The uncached lane (pipeline) one-shots too; long prompts must divert to
    the chunked manual path and short ones must return None (pipeline as
    before)."""
    _with_step(64)
    try:
        p, trace = _provider(_dense_cache)
        long_text = _DOC + " " + _DOC
        resp = p._transformers_generate_uncached_chunked(long_text, 1, 0.0, 1.0)
        assert resp is not None
        assert max(n for _op, n in trace) <= 64
        assert resp.finish_reason == "stop"

        trace.clear()
        short = p._transformers_generate_uncached_chunked("Say ok.", 1, 0.0, 1.0)
        assert short is None and trace == []
    finally:
        _pop_step()
