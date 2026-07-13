"""Pins for the MLX delta-generation path over warm KV caches.

Adversarial find B2 (2026-07-12, code seat + fable5): mlx_lm has no
common-prefix dedup, and the MLX provider fed the FULL rendered prompt on top
of a warm cache — caching ON cost ~2x the prefill of caching OFF and
duplicated the transcript in-context. The fix ports the HuggingFace
provider's delta pattern to the token level: track fed token ids per cache,
LCP the new prompt against them, trim the cache to the shared prefix, feed
ONLY the suffix.

These tests exercise the decision logic without loading a model: the provider
is built via __new__ with a fake tokenizer, a real PromptCacheStore, and fake
cache objects carrying `.offset` (what the real token counter reads).

Caller-shape discriminator (found attacking the first fix): delta discipline
applies only to FULL-CONTEXT callers (messages= present — they re-send the
whole logical context every call). Prompt-only callers (CachedSession KV mode,
direct accumulators) send FRAGMENTS over a cache that IS the context — LCP
arithmetic there would trim away the whole session. They keep append
semantics.
"""

from typing import List, Optional

import pytest

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.mlx_provider import MLXProvider


class _FakeLayer:
    """KVCache-shaped: carries `offset` and `empty()` like mlx_lm caches."""

    def __init__(self, offset: int = 0):
        self.offset = int(offset)

    def empty(self) -> bool:
        return self.offset == 0


class _FakeArraysLayer:
    """ArraysCache-shaped: NO size()/offset; only empty() (recurrent state)."""

    def __init__(self, warm: bool = True):
        self._warm = bool(warm)

    def empty(self) -> bool:
        return not self._warm


class _FakeTokenizer:
    """Deterministic word-level tokenizer: one int per whitespace token."""

    def encode(self, text: str) -> List[int]:
        return [hash(w) % 100_000 for w in str(text).split()]


class _FakeLogger:
    def __init__(self):
        self.warnings: List[str] = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def debug(self, msg, *a, **k):
        pass


def _provider(trim_result: bool = True) -> MLXProvider:
    import threading

    p = MLXProvider.__new__(MLXProvider)
    p.tokenizer = _FakeTokenizer()
    p.logger = _FakeLogger()
    p._prompt_cache_store = PromptCacheStore()
    p._delta_feed_warned_keys = set()
    p._pending_append_fragment = None
    p._pending_append_precount = 0
    p._append_stash_lock = threading.RLock()
    trims: List[int] = []

    def _trim(cache_value, n):
        if not trim_result:
            return False
        trims.append(int(n))
        for layer in cache_value:
            layer.offset = max(0, layer.offset - int(n))
        return True

    p._trim_prompt_cache_tokens = _trim  # type: ignore[method-assign]
    p._recorded_trims = trims  # test-visible
    return p


def _warm_cache(p: MLXProvider, key: str, fed_prompt: str, extra_generated: int = 0):
    """Install a warm cache whose KV covers fed_prompt (+ generated tokens)."""
    ids = p.tokenizer.encode(fed_prompt)
    cache = [_FakeLayer(offset=len(ids) + extra_generated)]
    p._prompt_cache_store.set(key, cache, meta={"backend": "mlx", "fed_token_ids": list(ids)})
    return cache, ids


def test_cold_cache_feeds_full_prompt_and_starts_tracking():
    p = _provider()
    cache = [_FakeLayer(offset=0)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "hello world how are you", full_context=True
    )

    assert out_cache is cache
    assert feed == "hello world how are you"          # legacy string feed on cold
    assert record == p.tokenizer.encode("hello world how are you")


def test_warm_extension_feeds_only_suffix():
    """The ReAct shape: cycle N+1's prompt = cycle N's prompt + new tail."""
    p = _provider()
    base = "sys tools task assistant tool result"
    cache, fed_ids = _warm_cache(p, "k", base, extra_generated=3)  # 3 stale generated tokens

    new_prompt = base + " assistant2 tool2 result2 tail"
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, new_prompt, full_context=True
    )

    new_ids = p.tokenizer.encode(new_prompt)
    assert out_cache is cache
    # Trimmed exactly the stale generated tokens (cache_len - lcp).
    assert p._recorded_trims == [3]
    assert feed == new_ids[len(fed_ids):]              # ONLY the suffix tokens
    assert record == new_ids


def test_warm_divergence_trims_to_lcp_then_feeds_suffix():
    """Mid-history divergence (edited transcript): trim back to the shared
    prefix and feed the new tail — the prefix KV is preserved. Safe because
    _trim_prompt_cache_tokens treats a PARTIAL trim as failure."""
    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "a b c d e")

    new_prompt = "a b c X Y Z"                          # diverges after 3 tokens
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, new_prompt, full_context=True
    )

    new_ids = p.tokenizer.encode(new_prompt)
    lcp = 3
    assert out_cache is cache
    assert p._recorded_trims == [len(fed_ids) - lcp]    # trimmed back to the LCP
    assert feed == new_ids[lcp:]
    assert record == new_ids


def test_identical_prompt_keeps_one_token_to_step():
    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "a b c d e")

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d e", full_context=True
    )

    assert feed == fed_ids[-1:]                         # never an empty feed
    assert p._recorded_trims == [1]


def test_prompt_only_caller_keeps_append_semantics():
    """CachedSession KV mode / direct accumulators: the cache IS the context
    and the prompt is the NEXT FRAGMENT — LCP arithmetic would trim away the
    whole session. Feed as-is, extend the id record."""
    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "sys history so far")

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "next user turn", full_context=False
    )

    assert out_cache is cache
    assert feed == "next user turn"                     # fragment fed untouched
    assert record == fed_ids + p.tokenizer.encode("next user turn")
    assert p._recorded_trims == []                      # never trims


def test_prompt_only_caller_record_freezes_once_generated_tokens_intervene():
    """Once the cache holds generated tokens beyond the record, extending the
    record with the next fragment would misdescribe the cache (fed + GAP +
    fragment). The old record must stand — still a true prefix."""
    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "sys history", extra_generated=4)

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "next turn", full_context=False
    )

    assert out_cache is cache
    assert feed == "next turn"
    assert record is None                               # frozen, not extended
    assert p._fed_token_ids_for_key("k") == fed_ids     # old truth stands


def test_prompt_only_caller_with_unknown_head_stays_unrecorded():
    p = _provider()
    cache = [_FakeLayer(offset=50)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})  # no fed ids

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "next turn", full_context=False
    )

    assert out_cache is cache
    assert feed == "next turn"
    assert record is None                               # unknown head stays unknown
    assert p.logger.warnings == []                      # append lane is not a fallback


def test_warm_cache_without_ids_under_full_context_rebuilds_fresh():
    """Unknown composition (pre-fix cache, loaded artifact) + a caller that
    re-sends everything: a fresh cache is CORRECT and kills the double
    prefill; the key becomes delta-tracked from this call on."""
    p = _provider()
    cache = [_FakeLayer(offset=50)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})  # no fed ids
    fresh = [_FakeLayer(offset=0)]
    p._prompt_cache_backend_create = lambda: fresh  # type: ignore[method-assign]

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c", full_context=True
    )
    assert out_cache is fresh
    assert feed == "a b c"
    assert record == p.tokenizer.encode("a b c")        # tracked from now on
    assert len(p.logger.warnings) == 1
    assert "#FALLBACK" in p.logger.warnings[0]

    p._prompt_cache_store.set("k2", [_FakeLayer(offset=9)], meta={"backend": "mlx"})
    p._prepare_cache_delta_feed("k2", p._prompt_cache_store.get("k2"), "a b", full_context=True)
    assert len(p.logger.warnings) == 2                  # warned once per key


def test_trim_failure_falls_back_to_fresh_cache_never_double_prefill():
    p = _provider(trim_result=False)
    # Stale generated tokens force a trim, which the cache type refuses.
    cache, _ = _warm_cache(p, "k", "a b c d e", extra_generated=3)
    fresh = [_FakeLayer(offset=0)]
    p._prompt_cache_backend_create = lambda: fresh  # type: ignore[method-assign]

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d e f g", full_context=True
    )

    assert out_cache is fresh                           # replaced, one cold prefill
    assert feed == "a b c d e f g"
    assert record == p.tokenizer.encode("a b c d e f g")
    assert p._prompt_cache_store.get("k") is fresh


def test_trim_failure_without_fresh_cache_bypasses_cache():
    p = _provider(trim_result=False)
    cache, _ = _warm_cache(p, "k", "a b c d e", extra_generated=3)
    p._prompt_cache_backend_create = lambda: None  # type: ignore[method-assign]

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d e f g", full_context=True
    )

    assert out_cache is None                            # uncached > double-prefill
    assert feed == "a b c d e f g"
    assert record is None


def test_tokenize_failure_on_warm_cache_bypasses_cache():
    p = _provider()
    cache, _ = _warm_cache(p, "k", "a b c")

    class _BrokenTok:
        def encode(self, text):
            raise RuntimeError("no tokenizer")

    p.tokenizer = _BrokenTok()
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d", full_context=True
    )

    assert out_cache is None
    assert feed == "a b c d"
    assert record is None


def test_tokenize_failure_on_prompt_only_caller_keeps_cache():
    """Append lane: an untokenizable fragment still feeds over the warm cache
    (that was always the legacy behavior); only the record is skipped."""
    p = _provider()
    cache, _ = _warm_cache(p, "k", "a b c")

    class _BrokenTok:
        def encode(self, text):
            raise RuntimeError("no tokenizer")

    p.tokenizer = _BrokenTok()
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d", full_context=False
    )

    assert out_cache is cache
    assert feed == "a b c d"
    assert record is None


def test_prompt_cache_update_extends_fed_id_record(monkeypatch):
    """The control-plane lane (the runtime's per-call cache prepare) must keep
    the id record true, or every generate would fall to the legacy lane."""
    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "a b c")

    def _fake_base_update(self, key, **kwargs):
        # Simulate the base flow: backend append stashed the fragment it fed.
        self._pending_append_fragment = "d e"
        self._pending_append_precount = len(fed_ids)
        for layer in cache:
            layer.offset += 2
        return True

    monkeypatch.setattr(
        "abstractcore.providers.base.BaseProvider.prompt_cache_update", _fake_base_update
    )

    assert p.prompt_cache_update("k", prompt="d e") is True
    assert p._fed_token_ids_for_key("k") == fed_ids + p.tokenizer.encode("d e")


def test_prompt_cache_update_never_records_over_unknown_head(monkeypatch):
    p = _provider()
    cache = [_FakeLayer(offset=40)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})  # unknown head

    def _fake_base_update(self, key, **kwargs):
        self._pending_append_fragment = "d e"
        self._pending_append_precount = 40
        return True

    monkeypatch.setattr(
        "abstractcore.providers.base.BaseProvider.prompt_cache_update", _fake_base_update
    )

    assert p.prompt_cache_update("k", prompt="d e") is True
    assert p._fed_token_ids_for_key("k") is None        # partial record refused


def test_lcp_helper():
    assert MLXProvider._token_lcp_len([1, 2, 3], [1, 2, 4]) == 2
    assert MLXProvider._token_lcp_len([], [1]) == 0
    assert MLXProvider._token_lcp_len([1, 2], [1, 2]) == 2


# --- adversary wave 2 (P0-1, P1-2..P1-7, P2-8/9) -----------------------------


def test_encode_mirrors_mlx_lm_bos_inference():
    """P0-1: mlx_lm infers add_special_tokens from whether the text STARTS
    WITH the BOS literal. A record encoded with a different rule runs one
    token long for BOS-templated architectures (gemma-turn) — a silent
    per-call off-by-one in every trim."""
    p = _provider()

    class _BosTokenizer:
        bos_token = "<bos>"

        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens=True):
            self.calls.append((text, add_special_tokens))
            ids = [hash(w) % 100_000 for w in text.replace("<bos>", " <bos> ").split()]
            if add_special_tokens:
                ids = [7] + ids
            return ids

    p.tokenizer = _BosTokenizer()

    with_bos = p._encode_prompt_token_ids("<bos>user hello")
    assert p.tokenizer.calls[-1][1] is False            # starts with BOS: not added again
    assert with_bos[0] == hash("<bos>") % 100_000

    without_bos = p._encode_prompt_token_ids("user hello")
    assert p.tokenizer.calls[-1][1] is True             # no BOS literal: tokenizer adds it
    assert without_bos[0] == 7


def test_warm_uncountable_cache_never_reads_cold():
    """P1-2: pure-SSM/CacheList architectures expose no size/offset — a warm
    cache must read as UNKNOWN (fresh rebuild lane), never as cold (which
    would revive the double prefill and record a false composition)."""
    p = _provider()
    cache = [_FakeArraysLayer(warm=True)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})
    fresh = [_FakeLayer(offset=0)]
    p._prompt_cache_backend_create = lambda: fresh  # type: ignore[method-assign]

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c", full_context=True
    )

    assert out_cache is fresh                           # rebuilt, not fed-on-top
    assert feed == "a b c"
    assert any("#FALLBACK" in w for w in p.logger.warnings)

    # Cold ArraysCache-only (all empty) still reads as genuinely cold.
    cold = [_FakeArraysLayer(warm=False)]
    assert p._prompt_cache_backend_token_count(cold) == 0
    assert p._prompt_cache_backend_token_count([_FakeArraysLayer(warm=True)]) is None


def test_prompt_only_uncountable_cache_feeds_legacy_without_record():
    """P1-2 append half: an uncountable warm cache under a prompt-only caller
    keeps legacy append behavior but can never verify prefix-truth — no
    record may be written."""
    p = _provider()
    cache = [_FakeArraysLayer(warm=True)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "next turn", full_context=False
    )

    assert out_cache is cache
    assert feed == "next turn"
    assert record is None


def test_prompt_cache_update_refuses_record_across_generated_gap(monkeypatch):
    """P1-3: precount includes generated tokens from a prior generate on the
    key; extending `prior + fragment` over `prior + gap + fragment` would
    write a false record. The old record must stand."""
    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "a b c", extra_generated=5)

    def _fake_base_update(self, key, **kwargs):
        self._pending_append_fragment = "d e"
        self._pending_append_precount = len(fed_ids) + 5   # gap of 5 generated tokens
        return True

    monkeypatch.setattr(
        "abstractcore.providers.base.BaseProvider.prompt_cache_update", _fake_base_update
    )

    assert p.prompt_cache_update("k", prompt="d e") is True
    assert p._fed_token_ids_for_key("k") == fed_ids     # frozen, not misdescribed


def test_fresh_rebuild_preserves_meta_and_artifact_caches_are_bypassed():
    """P1-5: the fresh cache is the same LOGICAL key — entry meta (minus the
    stale id record) must survive; loaded/bound artifacts must never be
    destroyed to save a prefill (bypass instead)."""
    p = _provider(trim_result=False)
    cache, _ = _warm_cache(p, "k", "a b c", extra_generated=2)
    p._prompt_cache_store.set(
        "k", cache, meta={"backend": "mlx", "fed_token_ids": p.tokenizer.encode("a b c"),
                          "token_count": 5, "custom": "keep-me"},
    )
    fresh = [_FakeLayer(offset=0)]
    p._prompt_cache_backend_create = lambda: fresh  # type: ignore[method-assign]

    out_cache, feed, record = p._prepare_cache_delta_feed(
        "k", cache, "a b c d e", full_context=True
    )
    assert out_cache is fresh
    kept = p._prompt_cache_store.meta("k")
    assert kept["custom"] == "keep-me"                  # meta survived the rebuild
    assert "fed_token_ids" not in kept                  # stale record dropped

    # Artifact-backed key: bypass, never replace.
    art_cache = [_FakeLayer(offset=40)]
    p._prompt_cache_store.set("art", art_cache, meta={"backend": "mlx", "loaded_from": "/x.safetensors"})
    out_cache2, feed2, record2 = p._prepare_cache_delta_feed(
        "art", art_cache, "a b c", full_context=True
    )
    assert out_cache2 is None                           # bypassed for this call
    assert record2 is None
    assert p._prompt_cache_store.get("art") is art_cache  # artifact untouched


def test_stats_surface_exposes_count_never_ids():
    """P1-4: fed_token_ids decode back to the full prompt text with the
    model's own tokenizer, and HTTP servers return stats verbatim — the
    stats boundary must ship the count only."""
    from abstractcore.providers.base import BaseProvider

    p = _provider()
    cache, fed_ids = _warm_cache(p, "k", "secret system prompt words")
    p._require_prompt_cache_operation = lambda op: type(
        "C", (), {"to_dict": lambda self: {}}
    )()
    p._default_prompt_cache_key = "k"

    stats = BaseProvider.get_prompt_cache_stats(p)

    meta = stats["meta_by_key"]["k"]
    assert "fed_token_ids" not in meta
    assert meta["fed_token_count"] == len(fed_ids)


def test_generate_wiring_feeds_suffix_and_skips_record_on_error():
    """P1-7: the only test that exercises _generate_internal's cache block —
    mutations like passing full_prompt to _single_generate (the original
    bug), flipping the discriminator, or recording on error must fail here."""
    from abstractcore.core.types import GenerateResponse

    p = _provider()
    p.llm = object()
    p.model = "test-model"
    p.model_capabilities = {}
    p.max_tokens = 64
    p.temperature = 0.7
    p.structured_output_method = "prompted"
    transcript = {"text": "sys task turn1"}
    p._build_prompt = lambda *a, **k: transcript["text"]  # type: ignore[method-assign]
    p._prepare_generation_kwargs = lambda **kw: dict(kw)  # type: ignore[method-assign]
    p._get_provider_max_tokens_param = lambda kw: 64  # type: ignore[method-assign]

    calls = []
    outcome = {"finish_reason": "stop"}

    def _fake_single_generate(prompt, *a, usage_prompt=None, **k):
        calls.append({"prompt": prompt, "usage_prompt": usage_prompt})
        return GenerateResponse(content="ok", model="test-model", finish_reason=outcome["finish_reason"])

    p._single_generate = _fake_single_generate  # type: ignore[method-assign]
    p.tool_handler = type("T", (), {"supports_prompted": False, "supports_native": False})()

    warm = [_FakeLayer(offset=0)]
    p._prompt_cache_store.set("sess", warm, meta={"backend": "mlx"})

    # Call 1 (cold): string feed, record starts.
    r1 = p._generate_internal("q", messages=[{"role": "user", "content": "q"}], prompt_cache_key="sess")
    assert r1.finish_reason == "stop"
    assert calls[0]["prompt"] == "sys task turn1"
    assert calls[0]["usage_prompt"] == "sys task turn1"
    ids1 = p.tokenizer.encode("sys task turn1")
    assert p._fed_token_ids_for_key("sess") == ids1
    warm[0].offset = len(ids1)                          # simulate the prefill

    # Call 2 (warm extension): ONLY the suffix ids reach the backend.
    transcript["text"] = "sys task turn1 turn2 tail"
    r2 = p._generate_internal("q2", messages=[{"role": "user", "content": "q2"}], prompt_cache_key="sess")
    ids2 = p.tokenizer.encode(transcript["text"])
    assert calls[1]["prompt"] == ids2[len(ids1):]       # token-list suffix, not full_prompt
    assert calls[1]["usage_prompt"] == transcript["text"]
    assert p._fed_token_ids_for_key("sess") == ids2
    warm[0].offset = len(ids2)

    # Call 3 (error): the record must NOT advance.
    outcome["finish_reason"] = "error"
    transcript["text"] = "sys task turn1 turn2 tail turn3"
    p._generate_internal("q3", messages=[{"role": "user", "content": "q3"}], prompt_cache_key="sess")
    assert p._fed_token_ids_for_key("sess") == ids2     # unchanged after error


def test_generate_wiring_empty_messages_is_full_context():
    """P2-8: messages=[] means "full context, empty so far" (key-mode turn
    one) — it must take the delta lane, not the append lane."""
    from abstractcore.core.types import GenerateResponse

    p = _provider()
    p.llm = object()
    p.model = "test-model"
    p.model_capabilities = {}
    p.max_tokens = 64
    p.temperature = 0.7
    p.structured_output_method = "prompted"
    p._build_prompt = lambda *a, **k: "sys prompt body"  # type: ignore[method-assign]
    p._prepare_generation_kwargs = lambda **kw: dict(kw)  # type: ignore[method-assign]
    p._get_provider_max_tokens_param = lambda kw: 64  # type: ignore[method-assign]
    calls = []

    def _fake_single_generate(prompt, *a, usage_prompt=None, **k):
        calls.append(prompt)
        return GenerateResponse(content="ok", model="test-model", finish_reason="stop")

    p._single_generate = _fake_single_generate  # type: ignore[method-assign]
    p.tool_handler = type("T", (), {"supports_prompted": False, "supports_native": False})()

    # Warm cache prefilled with the system prompt (unknown composition) —
    # append lane would feed the full prompt ON TOP (B2 residue); the delta
    # lane rebuilds fresh instead.
    warm = [_FakeLayer(offset=3)]
    p._prompt_cache_store.set("sess", warm, meta={"backend": "mlx"})
    fresh = [_FakeLayer(offset=0)]
    p._prompt_cache_backend_create = lambda: fresh  # type: ignore[method-assign]

    p._generate_internal("q", messages=[], prompt_cache_key="sess")

    assert p._prompt_cache_store.get("sess") is fresh   # delta lane engaged


def test_normalized_module_preserves_tool_order():
    """The normalized module is BOTH the fingerprint input and the rendered
    content. Sorting tools here made prepare_modules render a different byte
    stream than generate for the same list — the runtime lane's warm calls
    re-prefilled everything after the second tool (live find, 2026-07-12)."""
    from abstractcore.providers.base import PromptCacheModule

    tools = [
        {"name": "zeta", "description": "z"},
        {"name": "alpha", "description": "a"},
        {"name": "mike", "description": "m"},
    ]
    mod = PromptCacheModule(module_id="tools", tools=tools).normalized()
    assert [t["name"] for t in mod.tools] == ["zeta", "alpha", "mike"]


def test_append_record_meta_chains_module_records():
    """prepare_modules' per-module hook: the new module cache's record is
    prior + the exact fragment the backend append fed — so the FINAL prefix
    key (what sessions fork from) has a known composition and generate-side
    delta feeds can engage instead of fresh-rebuilding."""
    p = _provider()

    # Module 1: chain starts empty; append stashed fragment at precount 0.
    p._pending_append_fragment = "sys rules here"
    p._pending_append_precount = 0
    meta1 = p._prompt_cache_append_record_meta(None)
    ids1 = p.tokenizer.encode("sys rules here")
    assert meta1 == {"fed_token_ids": ids1}

    # Module 2: prior record exactly describes the pre-append cache.
    p._pending_append_fragment = "tool schemas"
    p._pending_append_precount = len(ids1)
    meta2 = p._prompt_cache_append_record_meta(meta1)
    assert meta2 == {"fed_token_ids": ids1 + p.tokenizer.encode("tool schemas")}

    # Unknown/stale head: refuse to describe (no record beats a wrong one).
    p._pending_append_fragment = "tail"
    p._pending_append_precount = 99
    assert p._prompt_cache_append_record_meta(meta1) is None


def test_real_trim_helper_treats_partial_trim_as_failure(monkeypatch):
    """The real _trim_prompt_cache_tokens (not the test fake): mlx_lm returns
    the count actually trimmed; a short count must read as failure or the
    delta arithmetic silently corrupts."""
    p = MLXProvider.__new__(MLXProvider)

    import mlx_lm.models.cache as mlx_cache

    monkeypatch.setattr(mlx_cache, "can_trim_prompt_cache", lambda c: True)
    monkeypatch.setattr(mlx_cache, "trim_prompt_cache", lambda c, n: n - 1)  # partial
    assert p._trim_prompt_cache_tokens([object()], 5) is False

    monkeypatch.setattr(mlx_cache, "trim_prompt_cache", lambda c, n: n)      # full
    assert p._trim_prompt_cache_tokens([object()], 5) is True

    monkeypatch.setattr(mlx_cache, "can_trim_prompt_cache", lambda c: False)
    assert p._trim_prompt_cache_tokens([object()], 5) is False               # refused
