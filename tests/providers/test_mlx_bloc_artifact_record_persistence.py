"""Pins for 0819: KV artifacts persist the fed-token-id record + delta admission.

Adversary P0-1 (bloc-composability review, 2026-07-13): the fed-token-id
record — the bookkeeping the whole MLX delta lane rides on — was store-meta
only and NEVER written into the saved artifact. Every artifact-backed cache
therefore loaded as "warm-unknown" and the full-context lane BYPASSED it:
load cost + RAM, then a full re-prefill — negative value under the runtime's
calling convention (messages= every call).

The fix, three parts, pinned here:
1. SAVE persists the record into the artifact metadata (safetensors is
   string-keyed/valued — the record rides as a JSON string).
2. LOAD parses it back to a real int list and VERIFIES admission: a record
   longer than the loaded cache cannot be a true token-prefix and is dropped
   loudly (the artifact keeps the protective bypass); shorter is legitimate
   (freeze invariant: generated tokens beyond the record stay unrecorded).
3. The delta lattice ADMITS artifact-backed keys with a true record
   (LCP -> trim -> suffix feed) — with one artifact-only protection: a
   DIVERGENT full-context prompt bypasses instead of trimming, so one
   divergent call can never degrade a shared stable bloc cache.

Plus the runtime seam's non-negotiable rider: the `prompt_cache` telemetry
struct (mode/key, outcome, MEASURED cached/fed tokens, binding shas,
degraded_reason) — the decision must be explainable from the ledger, not
from a log line.
"""

import threading
from typing import Any, Dict, List, Optional

import pytest

from abstractcore.providers.base import PromptCacheStore
from abstractcore.providers.mlx_provider import MLXProvider


class _FakeLayer:
    def __init__(self, offset: int = 0):
        self.offset = int(offset)

    def empty(self) -> bool:
        return self.offset == 0


class _FakeTokenizer:
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
    p._recorded_trims = trims
    # Working backend-create so the REBUILT lane is exercisable (without it,
    # fresh creation fails and the honest outcome degrades to a bypass).
    p._prompt_cache_backend_create = lambda: [_FakeLayer(offset=0)]  # type: ignore[method-assign]
    return p


def _artifact_cache(
    p: MLXProvider,
    key: str,
    fed_prompt: str,
    *,
    extra_generated: int = 0,
    record: Optional[List[int]] = None,
):
    """Install an artifact-backed warm cache (loaded_from + binding identity)."""
    ids = p.tokenizer.encode(fed_prompt)
    cache = [_FakeLayer(offset=len(ids) + extra_generated)]
    meta: Dict[str, Any] = {
        "backend": "mlx",
        "loaded_from": "/tmp/bloc.safetensors",
        "artifact_sha256": "a" * 64,
        "bloc_sha256": "b" * 64,
        "binding_id": "bind-123",
    }
    if record is not None:
        meta["fed_token_ids"] = list(record)
    p._prompt_cache_store.set(key, cache, meta=meta)
    return cache, ids


# ---------------------------------------------------------------------------
# 1. Save persists the record
# ---------------------------------------------------------------------------


def test_save_meta_includes_fed_token_id_record(monkeypatch):
    """prompt_cache_save must carry the store record into artifact metadata
    (JSON string under the safetensors string-value constraint)."""
    import json as _json

    p = _provider()
    ids = [11, 22, 33]
    cache = [_FakeLayer(offset=3)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx", "fed_token_ids": list(ids)})

    captured: Dict[str, Any] = {}

    def _fake_save(filename, cache_obj, metadata=None):
        captured["filename"] = filename
        captured["metadata"] = dict(metadata or {})

    import sys
    import types

    fake_cache_mod = types.ModuleType("mlx_lm.models.cache")
    fake_cache_mod.save_prompt_cache = _fake_save
    fake_models = types.ModuleType("mlx_lm.models")
    fake_models.cache = fake_cache_mod
    fake_root = types.ModuleType("mlx_lm")
    fake_root.models = fake_models
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_root)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache_mod)

    p.supports_prompt_cache = lambda: True  # type: ignore[method-assign]
    p.provider = "mlx"
    p.model = "test-model"

    result = p.prompt_cache_save("k", "/tmp/out.safetensors")
    assert result["operation"] == "save"
    meta = captured["metadata"]
    assert "fed_token_ids" in meta, "record dropped at the save boundary (P0-1 regression)"
    assert _json.loads(meta["fed_token_ids"]) == ids


def test_save_without_record_omits_the_field(monkeypatch):
    """A cache with no record (legacy/unknown composition) must not invent one."""
    p = _provider()
    cache = [_FakeLayer(offset=5)]
    p._prompt_cache_store.set("k", cache, meta={"backend": "mlx"})

    captured: Dict[str, Any] = {}

    def _fake_save(filename, cache_obj, metadata=None):
        captured["metadata"] = dict(metadata or {})

    import sys
    import types

    fake_cache_mod = types.ModuleType("mlx_lm.models.cache")
    fake_cache_mod.save_prompt_cache = _fake_save
    fake_models = types.ModuleType("mlx_lm.models")
    fake_models.cache = fake_cache_mod
    fake_root = types.ModuleType("mlx_lm")
    fake_root.models = fake_models
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_root)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache_mod)

    p.supports_prompt_cache = lambda: True  # type: ignore[method-assign]
    p.provider = "mlx"
    p.model = "test-model"

    p.prompt_cache_save("k", "/tmp/out.safetensors")
    assert "fed_token_ids" not in captured["metadata"]


# ---------------------------------------------------------------------------
# 2. Load parses + verifies the record
# ---------------------------------------------------------------------------


def _fake_load_modules(monkeypatch, loaded_cache, meta: Dict[str, Any]):
    import sys
    import types

    def _fake_load(filename, return_metadata=False):
        if return_metadata:
            return loaded_cache, dict(meta)
        return loaded_cache

    fake_cache_mod = types.ModuleType("mlx_lm.models.cache")
    fake_cache_mod.load_prompt_cache = _fake_load
    fake_models = types.ModuleType("mlx_lm.models")
    fake_models.cache = fake_cache_mod
    fake_root = types.ModuleType("mlx_lm")
    fake_root.models = fake_models
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_root)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache_mod)


def test_load_reconstructs_record_from_json_string(monkeypatch):
    """The round-trip: a JSON-string record in artifact metadata becomes a real
    int list in store meta — the delta lane can read it again."""
    p = _provider()
    p.supports_prompt_cache = lambda: True  # type: ignore[method-assign]
    p.provider = "mlx"
    p.model = "test-model"

    loaded = [_FakeLayer(offset=3)]
    _fake_load_modules(
        monkeypatch, loaded,
        {"model": "test-model", "fed_token_ids": "[11, 22, 33]"},
    )

    result = p.prompt_cache_load("/tmp/bloc.safetensors", key="bloc-key", make_default=False)
    assert result["operation"] == "load"
    record = p._fed_token_ids_for_key("bloc-key")
    assert record == [11, 22, 33]


def test_load_drops_record_longer_than_cache_loudly(monkeypatch):
    """A record LONGER than the loaded cache misdescribes it (cannot be a true
    token-prefix): dropped with #FALLBACK, protective bypass preserved."""
    p = _provider()
    p.supports_prompt_cache = lambda: True  # type: ignore[method-assign]
    p.provider = "mlx"
    p.model = "test-model"

    loaded = [_FakeLayer(offset=2)]  # cache holds 2 tokens
    _fake_load_modules(
        monkeypatch, loaded,
        {"model": "test-model", "fed_token_ids": "[11, 22, 33]"},  # record claims 3
    )

    p.prompt_cache_load("/tmp/bloc.safetensors", key="bloc-key", make_default=False)
    assert p._fed_token_ids_for_key("bloc-key") is None
    assert any("#FALLBACK" in w and "longer than the loaded cache" in w for w in p.logger.warnings)


def test_load_keeps_record_shorter_than_cache(monkeypatch):
    """Record shorter than cache = frozen record over a generated tail — a TRUE
    prefix by the freeze invariant; kept (the trim arithmetic handles the tail)."""
    p = _provider()
    p.supports_prompt_cache = lambda: True  # type: ignore[method-assign]
    p.provider = "mlx"
    p.model = "test-model"

    loaded = [_FakeLayer(offset=5)]  # 3 recorded + 2 generated
    _fake_load_modules(
        monkeypatch, loaded,
        {"model": "test-model", "fed_token_ids": "[11, 22, 33]"},
    )

    p.prompt_cache_load("/tmp/bloc.safetensors", key="bloc-key", make_default=False)
    assert p._fed_token_ids_for_key("bloc-key") == [11, 22, 33]


def test_load_garbage_record_is_dropped_silently_to_bypass(monkeypatch):
    """Unparseable record text is never mistaken for cache truth."""
    p = _provider()
    p.supports_prompt_cache = lambda: True  # type: ignore[method-assign]
    p.provider = "mlx"
    p.model = "test-model"

    loaded = [_FakeLayer(offset=3)]
    _fake_load_modules(
        monkeypatch, loaded,
        {"model": "test-model", "fed_token_ids": "not json at all"},
    )

    p.prompt_cache_load("/tmp/bloc.safetensors", key="bloc-key", make_default=False)
    assert p._fed_token_ids_for_key("bloc-key") is None


def test_parse_persisted_record_shapes():
    parse = MLXProvider._parse_persisted_fed_token_ids
    assert parse("[1, 2, 3]") == [1, 2, 3]
    assert parse([1, 2, 3]) == [1, 2, 3]
    assert parse("") is None
    assert parse("   ") is None
    assert parse("[]") is None
    assert parse("{\"a\": 1}") is None
    assert parse('["x", "y"]') is None
    assert parse(None) is None
    assert parse(42) is None


# ---------------------------------------------------------------------------
# 3. Delta-lattice admission for artifact-backed caches
# ---------------------------------------------------------------------------


def test_artifact_with_record_joins_delta_lattice_suffix_feed():
    """THE 0819 outcome: a full-context call over a recorded artifact feeds
    ONLY the suffix (the question), never the whole bloc again."""
    p = _provider()
    bloc_text = "doc alpha beta gamma delta epsilon"
    cache, bloc_ids = _artifact_cache(p, "bloc", bloc_text, record=p.tokenizer.encode(bloc_text))

    full_prompt = bloc_text + " question one two"
    telemetry: Dict[str, Any] = {}
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "bloc", cache, full_prompt, full_context=True, telemetry=telemetry,
    )

    assert out_cache is cache, "artifact cache must be USED, not bypassed"
    assert isinstance(feed, list)
    assert len(feed) == 3, f"expected 3 suffix tokens, got {len(feed)}"
    assert record == p.tokenizer.encode(full_prompt)
    assert telemetry["outcome"] == "hit_extend"
    assert telemetry["cached_tokens"] == len(bloc_ids)
    assert telemetry["fed_tokens"] == 3
    assert p._recorded_trims == [], "pure extension over a fresh bloc must trim nothing"


def test_artifact_divergent_prompt_bypasses_and_preserves_cache():
    """Divergence over an artifact bypasses (no trim): one divergent call must
    never degrade a shared stable bloc cache."""
    p = _provider()
    bloc_text = "doc alpha beta gamma delta epsilon"
    cache, bloc_ids = _artifact_cache(p, "bloc", bloc_text, record=p.tokenizer.encode(bloc_text))
    before_offset = cache[0].offset

    telemetry: Dict[str, Any] = {}
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "bloc", cache, "different head entirely then doc alpha", full_context=True,
        telemetry=telemetry,
    )

    assert out_cache is None, "divergent artifact call must bypass"
    assert feed == "different head entirely then doc alpha"
    assert record is None
    assert cache[0].offset == before_offset, "artifact cache was mutated by a divergent call"
    assert telemetry["outcome"] == "bypassed"
    assert "diverges" in telemetry["degraded_reason"]
    assert any("#FALLBACK" in w and "diverges" in w for w in p.logger.warnings)


def test_artifact_without_record_still_bypasses():
    """Legacy artifacts (no persisted record) keep the protective bypass —
    the backfill posture is honest degradation, never a guessed record."""
    p = _provider()
    cache, _ = _artifact_cache(p, "bloc", "doc alpha beta", record=None)

    telemetry: Dict[str, Any] = {}
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "bloc", cache, "doc alpha beta question", full_context=True, telemetry=telemetry,
    )

    assert out_cache is None
    assert record is None
    assert telemetry["outcome"] == "bypassed"
    assert "without a fed-token record" in telemetry["degraded_reason"]


def test_artifact_binding_meta_untouched_by_delta_use():
    """Delta use must not disturb binding/provenance meta (P1-5 lineage)."""
    p = _provider()
    bloc_text = "doc alpha beta gamma"
    cache, _ = _artifact_cache(p, "bloc", bloc_text, record=p.tokenizer.encode(bloc_text))

    p._prepare_cache_delta_feed(
        "bloc", cache, bloc_text + " question", full_context=True,
    )

    meta = p.prompt_cache_key_meta("bloc") or {}
    assert meta.get("binding_id") == "bind-123"
    assert meta.get("artifact_sha256") == "a" * 64
    assert meta.get("bloc_sha256") == "b" * 64
    assert meta.get("loaded_from") == "/tmp/bloc.safetensors"


def test_artifact_untrimmable_architecture_still_bypasses():
    """Hybrid architectures whose caches refuse trim: artifact-backed keys
    bypass (never rebuilt) even WITH a record."""
    p = _provider(trim_result=False)
    bloc_text = "doc alpha beta gamma"
    # Generated tail forces a trim on the extension path.
    cache, _ = _artifact_cache(
        p, "bloc", bloc_text, extra_generated=2, record=p.tokenizer.encode(bloc_text)
    )

    telemetry: Dict[str, Any] = {}
    out_cache, feed, record = p._prepare_cache_delta_feed(
        "bloc", cache, bloc_text + " question", full_context=True, telemetry=telemetry,
    )

    assert out_cache is None
    assert feed == bloc_text + " question"
    assert telemetry["outcome"] == "bypassed"
    assert "not trimmable" in telemetry["degraded_reason"]


# ---------------------------------------------------------------------------
# 4. Telemetry struct (runtime seam condition)
# ---------------------------------------------------------------------------


def test_telemetry_outcomes_cover_the_lattice():
    p = _provider()

    # cold
    cold_cache = [_FakeLayer(offset=0)]
    p._prompt_cache_store.set("cold", cold_cache, meta={"backend": "mlx"})
    t_cold: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("cold", cold_cache, "a b c", full_context=True, telemetry=t_cold)
    assert t_cold["outcome"] == "cold"
    assert t_cold["cached_tokens"] == 0
    assert t_cold["fed_tokens"] == 3

    # hit_extend
    warm_text = "a b c"
    warm_ids = p.tokenizer.encode(warm_text)
    warm_cache = [_FakeLayer(offset=len(warm_ids))]
    p._prompt_cache_store.set(
        "warm", warm_cache, meta={"backend": "mlx", "fed_token_ids": list(warm_ids)}
    )
    t_ext: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("warm", warm_cache, "a b c d e", full_context=True, telemetry=t_ext)
    assert t_ext["outcome"] == "hit_extend"
    assert t_ext["cached_tokens"] == 3
    assert t_ext["fed_tokens"] == 2

    # hit_full (identical prompt keeps one token to step generation)
    t_full: Dict[str, Any] = {}
    ident_cache = [_FakeLayer(offset=len(warm_ids))]
    p._prompt_cache_store.set(
        "ident", ident_cache, meta={"backend": "mlx", "fed_token_ids": list(warm_ids)}
    )
    p._prepare_cache_delta_feed("ident", ident_cache, warm_text, full_context=True, telemetry=t_full)
    assert t_full["outcome"] == "hit_full"
    assert t_full["fed_tokens"] == 1

    # append (prompt-only caller)
    app_cache = [_FakeLayer(offset=4)]
    p._prompt_cache_store.set("app", app_cache, meta={"backend": "mlx"})
    t_app: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("app", app_cache, "next fragment", full_context=False, telemetry=t_app)
    assert t_app["outcome"] == "append"

    # rebuilt (warm-unknown, non-artifact)
    unk_cache = [_FakeLayer(offset=7)]
    p._prompt_cache_store.set("unk", unk_cache, meta={"backend": "mlx"})
    t_reb: Dict[str, Any] = {}
    p._prepare_cache_delta_feed("unk", unk_cache, "x y z", full_context=True, telemetry=t_reb)
    assert t_reb["outcome"] == "rebuilt"
    assert "#FALLBACK" in t_reb["degraded_reason"]


def test_generate_internal_attaches_prompt_cache_telemetry():
    """The sync generate path must attach the struct (with binding identity)
    to response.metadata — the runtime ledger's explanation surface."""
    from abstractcore.core.types import GenerateResponse

    p = _provider()
    p.llm = object()
    p.model = "test-model"
    p.model_capabilities = {}
    p.structured_output_method = "prompted"
    p.max_output_tokens = 64
    p.temperature = 0.0

    bloc_text = "doc alpha beta gamma delta"
    cache, bloc_ids = _artifact_cache(p, "bloc", bloc_text, record=p.tokenizer.encode(bloc_text))

    p._build_prompt = (  # type: ignore[method-assign]
        lambda prompt, messages, system_prompt, tools, **kw: bloc_text + " question one"
    )
    p._prepare_generation_kwargs = lambda **kw: dict(kw)  # type: ignore[method-assign]
    p._get_provider_max_tokens_param = lambda gk: 64  # type: ignore[method-assign]

    captured: Dict[str, Any] = {}

    def _fake_single_generate(prompt_to_feed, *a, **kw):
        captured["fed"] = prompt_to_feed
        return GenerateResponse(content="answer", model="test-model", finish_reason="stop")

    p._single_generate = _fake_single_generate  # type: ignore[method-assign]

    response = p._generate_internal(
        "question one", messages=[], prompt_cache_key="bloc",
    )

    telemetry = (response.metadata or {}).get("prompt_cache")
    assert isinstance(telemetry, dict), "prompt_cache telemetry missing from response metadata"
    assert telemetry["mode"] == "key"
    assert telemetry["key"] == "bloc"
    assert telemetry["outcome"] == "hit_extend"
    assert telemetry["cached_tokens"] == len(bloc_ids)
    assert telemetry["fed_tokens"] == 2
    assert telemetry["binding_id"] == "bind-123"
    assert telemetry["artifact_sha256"] == "a" * 64
    assert telemetry["bloc_sha256"] == "b" * 64
    assert isinstance(captured["fed"], list) and len(captured["fed"]) == 2


def test_no_cache_key_means_no_telemetry_struct():
    """Absence is the honest signal for cache-off calls — no fake struct."""
    from abstractcore.core.types import GenerateResponse

    p = _provider()
    p.llm = object()
    p.model = "test-model"
    p.model_capabilities = {}
    p.structured_output_method = "prompted"
    p.max_output_tokens = 64
    p.temperature = 0.0
    p._build_prompt = (  # type: ignore[method-assign]
        lambda prompt, messages, system_prompt, tools, **kw: "plain prompt"
    )
    p._prepare_generation_kwargs = lambda **kw: dict(kw)  # type: ignore[method-assign]
    p._get_provider_max_tokens_param = lambda gk: 64  # type: ignore[method-assign]
    p._single_generate = (  # type: ignore[method-assign]
        lambda *a, **kw: GenerateResponse(content="ok", model="test-model", finish_reason="stop")
    )

    response = p._generate_internal("plain prompt", messages=[])
    assert "prompt_cache" not in (response.metadata or {})
