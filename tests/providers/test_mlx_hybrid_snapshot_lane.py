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

    def is_trimmable(self) -> bool:
        # A Gated-DeltaNet / SSM recurrent layer (`ArraysCache`): the state
        # cannot be rewound, at ANY fill level. mlx_lm's `can_trim_prompt_cache`
        # reads exactly this predicate, so the provider's real
        # `_cache_is_trimmable` runs here instead of falling through its
        # exception guard — which is what routes this fake onto the snapshot lane
        # for the right reason.
        return False


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
    # The snapshot boundary DERIVES the generation-prompt tail by re-rendering
    # through `_build_prompt_fragment`, so a provider used here must be able to
    # render. Without these three attributes the render raises, no literal is
    # derived, and the holdback is silently inert — which is precisely the
    # failure mode being guarded against, so it must not be the test's default.
    p.tool_handler = _ToolHandler()
    p.model = "vendor/testmodel-4b"
    p.architecture_config = {"message_format": "im_start_end"}
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


def test_snapshot_boundary_survives_a_rewritten_ephemeral_tail():
    """2026-08-02 regression: agent loops end EVERY prompt in per-call ephemeral bytes
    (a `[loop] iteration N of M.` tail, a fresh-timestamp grounding envelope, the
    generation prompt). Snapshotting the full prompt boundary therefore stored ids that
    the NEXT prompt never extends — the forward-only restore was refused and the lane
    paid a full cold prefill on every single turn (`rebuilt`, cached=0).

    The boundary must fall back to what the two consecutive prompts actually share.
    """
    p = _provider()

    # Turn N-1 fed: stable transcript P, then an ephemeral tail.
    prev_prompt = "sys tools task obs1 TAIL1 GEN"
    live, prev_ids = _warm_untrimmable(p, "k", prev_prompt, extra_generated=4)

    # Turn N: same P, one NEW observation, a DIFFERENT tail. Not an extension of prev.
    cur_prompt = "sys tools task obs1 obs2 TAIL2 GEN"
    cur_ids = p.tokenizer.encode(cur_prompt)

    tel_a = {}
    p._prepare_cache_delta_feed("k", live, cur_prompt, full_context=True, telemetry=tel_a)

    snap = p._get_hybrid_snapshot("k")
    assert snap is not None
    shared = p._token_lcp_len(prev_ids, cur_ids)
    # Boundary held back to the shared stable transcript, NOT cur_ids[:-1].
    assert snap["ids"] == cur_ids[:shared]
    assert snap["ids"] != cur_ids[:-1]

    # Turn N+1: P + obs2 is preserved, so the snapshot IS restorable now.
    next_prompt = "sys tools task obs1 obs2 obs3 TAIL3 GEN"
    live2 = p._prompt_cache_store.get("k")
    p._prompt_cache_store.set(
        "k", live2, meta={"backend": "mlx", "fed_token_ids": list(cur_ids)}
    )
    tel_b = {}
    p._prepare_cache_delta_feed("k", live2, next_prompt, full_context=True, telemetry=tel_b)

    assert tel_b["outcome"] == "hit_restore"
    assert tel_b["cached_tokens"] == shared > 0


# ---------------------------------------------------------------------------
# Sessions FORKED from a bloc-chain prefix (the shape the runtime actually uses)
# ---------------------------------------------------------------------------


def _fork_bloc_prefix(p: MLXProvider, key: str, prefix_prompt: str) -> List[int]:
    """Install what `prompt_cache_fork` leaves behind: a cache holding EXACTLY the
    bloc chain's tokens, its fed-token record, and the `forked_from` provenance
    (`BaseProvider.prompt_cache_fork` copies the source meta)."""
    prefix_ids = p.tokenizer.encode(prefix_prompt)
    p._prompt_cache_store.set(
        key,
        [_FakeLayer(offset=len(prefix_ids))],
        meta={
            "backend": "mlx",
            "fed_token_ids": list(prefix_ids),
            "forked_from": "abstractcode:bloc-chain-final",
        },
    )
    return prefix_ids


def _drive_session(p: MLXProvider, key: str, prompts: List[str], *, reply_tokens: int = 4):
    """Drive turns the way `generate()` does: ask the delta feed what to do, feed
    exactly that, let the model append its reply, then record the fed ids."""
    telemetry = []
    for prompt in prompts:
        cache = p._prompt_cache_store.get(key)
        tel: dict = {}
        used, feed, record = p._prepare_cache_delta_feed(
            key, cache, prompt, full_context=True, telemetry=tel
        )
        if used is not None:
            fed_n = len(feed) if isinstance(feed, list) else len(p.tokenizer.encode(feed))
            for layer in used:
                layer.offset += fed_n + reply_tokens
            meta = dict(p._prompt_cache_store.meta(key) or {})
            if record:
                meta["fed_token_ids"] = list(record)
            p._prompt_cache_store.set(key, used, meta=meta)
        telemetry.append(tel)
    return telemetry


def _turns(prefix: str, n: int, *, ephemeral: bool = True) -> List[str]:
    """A faithful growing transcript. Turn N ends with the generation prompt, and
    turn N+1 carries that SAME generation prompt at the SAME position followed by
    the reply produced there — so an append-only prompt is a true token prefix of
    its successor, and the per-call ephemeral envelope (dropped and re-appended
    every turn, just before the generation prompt) is the ONLY thing that breaks
    that relation."""
    out = []
    for turn in range(1, n + 1):
        parts = [prefix]
        for i in range(1, turn + 1):
            parts.append(f"u{i}")
            if i == turn and ephemeral:
                parts.append(f"EPHEM{turn}")
            parts.append("GEN")
            if i < turn:
                parts.append(f"a{i}")
        out.append(" ".join(parts))
    return out


def test_forked_bloc_session_does_not_rebuild_on_turn_two():
    """THE 2026-08-03 REGRESSION PIN.

    A session forked from a `(system, tools)` bloc chain is WARM on turn 1 and its
    cache exactly matches its fed-token record, so `trim_needed` is 0. The
    architecture check used to live only inside the `cold_empty` branch, so this
    shape never discovered the cache was untrimmable: turn 1 completed on the trim
    lane as `hit_extend` and stored NO snapshot, and turn 2 — the first call that
    genuinely needs a trim — was refused with nothing to restore and paid a full
    cold prefill (`rebuilt`, cached=0). That single rebuild was the whole gap
    between ~47% and ~96% steady-state reuse, on every locally available model
    above 4B (they are all Gated-DeltaNet hybrids on this lane).

    No turn of a growing session may rebuild.
    """
    p = _provider()
    prefix = "sys1 sys2 sys3 tool1 tool2"
    prefix_ids = _fork_bloc_prefix(p, "k", prefix)

    tel = _drive_session(p, "k", _turns(prefix, 3))
    outcomes = [t["outcome"] for t in tel]

    assert "rebuilt" not in outcomes, outcomes
    assert outcomes == ["hit_restore", "hit_restore", "hit_restore"]
    # Turn 1 reuses the forked bloc itself rather than re-prefilling it...
    assert tel[0]["cached_tokens"] == len(prefix_ids)
    # ...and, critically, leaves a boundary behind for turn 2.
    assert all(t["cached_tokens"] > 0 for t in tel)


def test_forked_bloc_session_turn_one_leaves_a_restorable_snapshot():
    """The precise mechanism: turn 1 must WRITE a snapshot, and that snapshot must
    stop before the per-call ephemeral tail. A boundary that includes the tail is
    not a prefix of turn 2 and is therefore worth nothing."""
    p = _provider()
    prefix = "sys1 sys2 sys3 tool1 tool2"
    prefix_ids = _fork_bloc_prefix(p, "k", prefix)
    turn1 = _turns(prefix, 1)[0]

    _drive_session(p, "k", [turn1])

    snap = p._get_hybrid_snapshot("k")
    assert snap is not None                      # the missing snapshot WAS the bug
    assert snap["ids"] == list(prefix_ids)        # held back before EPHEM1/GEN
    turn2 = _turns(prefix, 2)[1]
    assert p.tokenizer.encode(turn2)[: len(snap["ids"])] == snap["ids"]  # restorable


def test_untrimmable_arch_takes_the_snapshot_lane_even_when_no_trim_is_needed():
    """The gate itself: an untrimmable cache must route on the ARCHITECTURE, not on
    the fill state. `trim_needed == 0` must not be read as 'the trim lane works
    here' — that inference is what silently skipped the snapshot."""
    p = _provider()
    prefix_ids = _fork_bloc_prefix(p, "k", "a b c d e")
    cache = p._prompt_cache_store.get("k")
    # Cache holds EXACTLY its record → effective_prefix == cache_len → trim_needed 0.
    assert p._prompt_cache_backend_token_count(cache) == len(prefix_ids)

    tel: dict = {}
    p._prepare_cache_delta_feed("k", cache, "a b c d e f g h", full_context=True, telemetry=tel)

    assert tel["outcome"] != "hit_extend"        # the trim lane must NOT claim this
    assert tel["outcome"] == "hit_restore"
    assert p._get_hybrid_snapshot("k") is not None


def test_append_only_session_keeps_the_full_boundary():
    """Guard against over-correction. With no ephemeral tail the previous prompt IS
    a true prefix of this one, so nothing is volatile and the boundary must NOT be
    held back — otherwise every turn re-feeds the previous turn's growth and the
    fix would be a slowdown dressed as a fix."""
    p = _provider()
    prefix = "sys1 sys2 sys3 tool1 tool2"
    prompts = _turns(prefix, 3, ephemeral=False)
    _fork_bloc_prefix(p, "k", prefix)

    tel = _drive_session(p, "k", prompts)
    assert [t["outcome"] for t in tel] == ["hit_restore"] * 3

    # Steady state: the snapshot sits at the FULL boundary (new_ids[:-1]) of the
    # turn that wrote it, so the next turn feeds only its own growth.
    last_ids = p.tokenizer.encode(prompts[-1])
    assert p._get_hybrid_snapshot("k")["ids"] == last_ids[:-1]


# ---------------------------------------------------------------------------
# Turn 1 on a FRESH key: the generation-prompt scaffolding is volatile
# ---------------------------------------------------------------------------

# What `_build_prompt_fragment` appends for ChatML with thinking disabled. On the
# real Qwen3.5 tokenizer `<think>\n\n</think>\n\n` is exactly the 4 tokens
# ['<think>', '\n\n', '</think>', '\n\n']; under this file's whitespace tokenizer
# the whole literal is 3 tokens. Either way it is the trailing scaffolding.
_GEN_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class _ToolHandler:
    supports_prompted = True

    def format_tools_prompt(self, tools, include_tool_list=True):
        return "## Tools (session)\nSCHEMA"


# Every `message_format` in assets/architecture_formats.json. The renderer has
# THREE branches, so eight of these fall through to the `assistant:` fallback —
# which is exactly the set a hand-written literal list missed, and it includes
# `granitemoehybrid`'s family: an UNTRIMMABLE HYBRID, i.e. the architecture this
# whole lane exists for.
_ALL_MESSAGE_FORMATS = [
    "im_start_end", "basic", "inst", "openai_chat", "llama3_header",
    "special_tokens", "glm_special_tokens", "gemma_turn", "human_assistant",
    "harmony",
]


def _render_provider(message_format: str, model: str = "vendor/testmodel-4b"):
    """A provider whose renderer branch is chosen ONLY by `message_format`.

    The model id must not contain "qwen": `_build_prompt_fragment` sets
    `is_chatml = (msg_fmt == "im_start_end") or ("qwen" in model.lower())`, so a
    Qwen id makes EVERY family render ChatML and any per-family assertion becomes
    vacuous. That defect made the first version of this test pass while covering
    one family.
    """
    p = _provider()
    p.tool_handler = _ToolHandler()
    p.model = model
    p.architecture_config = {"message_format": message_format}
    return p


@pytest.mark.parametrize("message_format", _ALL_MESSAGE_FORMATS)
@pytest.mark.parametrize("thinking", [False, None])
def test_generation_prompt_literal_is_exactly_what_the_renderer_appends(message_format, thinking):
    """The derivation, pinned per family: rendering with every content input empty
    must yield precisely the block `add_generation_prompt` adds, so that

        render(gen=True) == render(gen=False) + literal

    holds for every template family. This is what makes the boundary derived
    rather than mirrored — a renderer change cannot silently desynchronise it."""
    p = _render_provider(message_format)

    with_gen = p._build_prompt_fragment(
        prompt="hi", system_prompt="SYS", add_generation_prompt=True, enable_thinking=thinking
    )
    without_gen = p._build_prompt_fragment(
        prompt="hi", system_prompt="SYS", add_generation_prompt=False, enable_thinking=thinking
    )
    literals = p._generation_prompt_literals()

    assert literals, f"{message_format}: renderer produced no generation prompt at all"
    match = next((t for t in literals if with_gen.endswith(t)), None)
    assert match is not None, (
        f"{message_format}/thinking={thinking!r}: derived literals {literals!r} do not "
        f"match the rendered tail {with_gen[len(without_gen):]!r}"
    )
    assert with_gen == without_gen + with_gen[len(without_gen):]
    assert with_gen[len(without_gen):] in literals


@pytest.mark.parametrize("message_format", _ALL_MESSAGE_FORMATS)
def test_generation_prompt_boundary_lands_before_the_scaffolding(message_format):
    """The boundary is a POSITION, and it must be the first token of the
    scaffolding — for every family, including the eight that use the fallback."""
    p = _render_provider(message_format)

    with_gen = p._build_prompt_fragment(
        prompt="hi there", system_prompt="SYS", add_generation_prompt=True, enable_thinking=False
    )
    without_gen = p._build_prompt_fragment(
        prompt="hi there", system_prompt="SYS", add_generation_prompt=False, enable_thinking=False
    )
    ids = p.tokenizer.encode(with_gen)

    at = p._generation_prompt_boundary(with_gen, ids)
    assert at is not None, f"{message_format}: no boundary derived — holdback would be inert"
    assert at == p._token_lcp_len(p.tokenizer.encode(without_gen), ids)
    assert 0 < at < len(ids)
    # Everything from the boundary on is scaffolding, so the boundary is a true
    # prefix of any continuation that replaces it.
    assert ids[:at] == p.tokenizer.encode(without_gen)[:at]


def test_generation_prompt_boundary_is_none_when_the_prompt_has_no_generation_prompt():
    """No scaffolding → no holdback. An unrecognised shape must inherit the old
    full boundary, never a guessed one."""
    p = _render_provider("im_start_end")
    body = p._build_prompt_fragment(
        prompt="hi", system_prompt="SYS", add_generation_prompt=False
    )
    assert p._generation_prompt_boundary(body, p.tokenizer.encode(body)) is None


def test_fresh_key_turn_one_snapshot_excludes_the_generation_scaffolding():
    """THE 2026-08-03 HARDWARE FINDING (agent 4, Qwen3.5-4B-MLX-4bit, 10k loop):
    turn 1 `rebuilt`, **turn 2 `rebuilt` again** (a full 10,118-token prefill),
    turn 3 `hit_restore`.

    Turn 1 of a session on a fresh key has NO fed-token record, so the LCP holdback
    has nothing to compare against and the boundary defaulted to the whole prompt —
    including `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`, which the next
    turn replaces with the assistant turn that actually happened. The turn-1
    boundary was therefore a prefix of nothing, turn 2's restore was refused, and
    only turn 3 — the first turn that HAS a record — could hold the boundary back
    correctly. An off-by-one-turn boundary bug, not a lane-selection bug.

    Turn 1's boundary must stop before the scaffolding.
    """
    p = _provider()
    stable = "sys tools u1"
    # A bare fresh key: exactly what generate() creates via prompt_cache_set.
    p._prompt_cache_store.set("k", [_FakeLayer(offset=0)], meta={"backend": "mlx"})

    turn1 = f"{stable} {_GEN_TAIL}"
    tel1: dict = {}
    used, feed, rec = p._prepare_cache_delta_feed(
        "k", p._prompt_cache_store.get("k"), turn1, full_context=True, telemetry=tel1
    )
    assert tel1["outcome"] == "cold"             # genuinely cold, and now says so
    for layer in used:
        layer.offset += len(feed) + 3
    p._prompt_cache_store.set("k", used, meta={"backend": "mlx", "fed_token_ids": list(rec)})

    snap = p._get_hybrid_snapshot("k")
    assert snap is not None
    turn1_ids = p.tokenizer.encode(turn1)
    assert snap["ids"] != turn1_ids[:-1]          # the old, unusable boundary
    assert snap["ids"] == turn1_ids[: len(p.tokenizer.encode(stable))]

    # Turn 2 carries the assistant turn that actually happened. The turn-1 boundary
    # must still be a true prefix of it, so this restores instead of rebuilding.
    turn2 = f"{stable} <|im_start|>assistant\n reply1 <|im_end|> u2 {_GEN_TAIL}"
    assert p.tokenizer.encode(turn2)[: len(snap["ids"])] == snap["ids"]

    tel2: dict = {}
    p._prepare_cache_delta_feed(
        "k", p._prompt_cache_store.get("k"), turn2, full_context=True, telemetry=tel2
    )
    assert tel2["outcome"] == "hit_restore", tel2
    assert tel2["cached_tokens"] == len(snap["ids"]) > 0


def test_fresh_key_session_rebuilds_only_on_turn_one():
    """End to end on the agent-loop shape agent 4 ran: fresh key, no bloc fork,
    monotonically growing transcript, generation scaffolding on every call. Turn 1
    is a genuine cold prefill; NO LATER TURN may rebuild."""
    p = _provider()
    p._prompt_cache_store.set("k", [_FakeLayer(offset=0)], meta={"backend": "mlx"})

    prompts = []
    for n in range(1, 5):
        parts = ["sys tools"]
        for i in range(1, n + 1):
            parts.append(f"u{i}a u{i}b u{i}c")
            if i < n:
                parts.append(f"<|im_start|>assistant\n a{i}x a{i}y <|im_end|>")
        parts.append(_GEN_TAIL)
        prompts.append(" ".join(parts))

    outcomes = [t["outcome"] for t in _drive_session(p, "k", prompts)]
    # Turn 1 reports `cold`, not `rebuilt` (parity relabel, 2026-08-07): a fresh
    # key had no prior cache to discard, and the dense/trim lane and the GGUF
    # lane both already called this state `cold`. The invariant this test exists
    # for is unchanged and now asserted more strictly — turn 1 is the ONLY full
    # prefill, so neither full-prefill outcome may appear again.
    assert outcomes[0] == "cold"
    assert "rebuilt" not in outcomes[1:], outcomes
    assert "cold" not in outcomes[1:], outcomes


# ---------------------------------------------------------------------------
# Artifact protection: the artifact, yes; a private copy of it, no
# ---------------------------------------------------------------------------

_ARTIFACT_META = {
    "loaded_from": "/blocs/persona.safetensors",
    "binding_id": "bind-1",
    "artifact_sha256": "deadbeef",
}


def _artifact_turns(prefix: str, n: int) -> List[str]:
    out = []
    for turn in range(1, n + 1):
        parts = [prefix]
        for i in range(1, turn + 1):
            parts.append(f"u{i}a u{i}b")
            if i < turn:
                parts.append(f"<|im_start|>assistant\n a{i} <|im_end|>")
        parts.append(_GEN_TAIL)
        out.append(" ".join(parts))
    return out


def test_session_forked_from_an_artifact_bloc_still_reuses():
    """`prompt_cache_fork` copies the source meta wholesale, so a session forked
    from a durable bloc KV artifact inherited `loaded_from`/`binding_id`/
    `artifact_sha256` — and every branch that flag guards is "bypass rather than
    modify". From turn 2 on, the session cache was never used at all: measured
    `hit_extend, bypassed, bypassed, bypassed` on BOTH lanes, so any benchmark of
    bloc-artifact reuse was measuring turn 1 only.

    A fork is a private copy. The artifact it came from is untouched.
    """
    p = _provider()
    prefix = "sys1 sys2 sys3 tool1 tool2"
    prefix_ids = p.tokenizer.encode(prefix)
    p._prompt_cache_store.set(
        "k",
        [_FakeLayer(offset=len(prefix_ids))],
        meta={
            "backend": "mlx",
            "fed_token_ids": list(prefix_ids),
            "forked_from": "bloc:stable",
            **_ARTIFACT_META,
        },
    )

    outcomes = [t["outcome"] for t in _drive_session(p, "k", _artifact_turns(prefix, 4))]
    assert "bypassed" not in outcomes, outcomes
    assert outcomes == ["hit_restore"] * 4


def test_a_genuine_artifact_key_keeps_its_protection():
    """The other half: an artifact key that was LOADED, not forked, must still
    refuse to be degraded by a divergent caller. Dropping the protection wholesale
    would let one caller trim a shared verified bloc down to a stub."""
    p = _provider()
    body = "sys1 sys2 sys3 tool1 tool2 u1a u1b"
    body_ids = p.tokenizer.encode(body)
    p._prompt_cache_store.set(
        "k",
        [_FakeLayer(offset=len(body_ids))],
        meta={"backend": "mlx", "fed_token_ids": list(body_ids), **_ARTIFACT_META},
    )

    tel: dict = {}
    used, _feed, _rec = p._prepare_cache_delta_feed(
        "k", p._prompt_cache_store.get("k"), "totally different prompt entirely",
        full_context=True, telemetry=tel,
    )
    assert tel["outcome"] == "bypassed"
    assert used is None                      # the artifact is left whole
    assert p._prompt_cache_store.meta("k")["artifact_sha256"] == "deadbeef"


def test_loading_an_artifact_clears_an_inherited_forked_from():
    """A cache forked and THEN saved carries `forked_from` inside the artifact
    file. On load it is the artifact again, so that stale field must not survive
    to strip the key of its protection."""
    p = _provider()
    store_meta = {"backend": "mlx", "loaded_from": "/blocs/x.safetensors"}
    store_meta.update({"forked_from": "bloc:ancestor", "artifact_sha256": "cafe"})
    store_meta.pop("forked_from", None)      # mirrors prompt_cache_load
    p._prompt_cache_store.set("k", [_FakeLayer(offset=3)], meta=store_meta)

    meta = p.prompt_cache_key_meta("k")
    assert "forked_from" not in meta
    assert meta["artifact_sha256"] == "cafe"
