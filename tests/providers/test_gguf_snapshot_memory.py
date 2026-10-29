"""A GGUF prompt-cache snapshot holds only what a restore uses, every byte it holds is counted,
and the snapshots of all keys together stay under a byte budget (8080 pass four, 2026-10-29: the
model host child grew 1.65 GB per basic-agent run; 0.5 GB of each snapshot was a `scores` array
that `logits_all=False` never writes or reads, and nothing bounded the bytes across keys).
CPU-only, model-free; `llama_cpp` is needed for the real `LlamaState`/`LlamaRAMCache` classes."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from abstractcore.providers.base import PromptCacheStore

llama_cpp = pytest.importorskip("llama_cpp")
from llama_cpp.llama import LlamaState  # noqa: E402
from llama_cpp.llama_cache import LlamaRAMCache  # noqa: E402

from abstractcore.providers import huggingface_provider as hf  # noqa: E402

N_BATCH, N_VOCAB, N_CTX = 64, 1000, 4096


def _provider(store: PromptCacheStore | None = None) -> hf.HuggingFaceProvider:
    p = hf.HuggingFaceProvider.__new__(hf.HuggingFaceProvider)
    p.model_type = "gguf"
    p.model = "test/gguf"
    p.logger = logging.getLogger("test_gguf_snapshot_memory")
    p.__dict__["_llm_own"] = None
    p._default_prompt_cache_key = None
    p._prompt_cache_store = store if store is not None else PromptCacheStore(max_entries=32)
    if store is not None:
        store.set_on_evict(p._prompt_cache_store_evicted)
    return p


def _saved_state(n_tokens: int, kv_bytes: int) -> LlamaState:
    """What `Llama.save_state` returns with logits_all=False: scores[:n_tokens] of (n_batch, n_vocab)."""
    return LlamaState(
        input_ids=np.zeros((N_CTX,), dtype=np.intc),
        scores=np.ones((min(n_tokens, N_BATCH), N_VOCAB), dtype=np.single),
        n_tokens=n_tokens,
        llama_state=b"\0" * kv_bytes,
        llama_state_size=kv_bytes,
        seed=1,
    )


def test_a_stored_snapshot_keeps_one_scores_row():
    p = _provider()
    stored = p._gguf_clone_llama_state(_saved_state(3000, 1000))
    assert stored.scores.shape == (1, N_VOCAB)
    assert not stored.scores.any()
    assert stored.n_tokens == 3000 and stored.llama_state_size == 1000
    assert stored.input_ids.shape == (N_CTX,)


def test_logits_all_keeps_the_scores():
    """With logits_all=True (logprobs) the rows are real logits: kept whole."""
    p = _provider()
    saved = _saved_state(40, 10)
    stored = p._gguf_clone_llama_state(saved, compact_scores=False)
    assert stored.scores.shape == saved.scores.shape and stored.scores.all()


@pytest.mark.parametrize("n_tokens", [0, 1, 17, N_BATCH, 3000])
def test_load_state_accepts_the_compacted_row(n_tokens):
    """`Llama.load_state` does `self.scores[:n_tokens, :] = state.scores.copy()` on a
    (n_batch, n_vocab) array: the one row must broadcast for every prompt length."""
    live = np.full((N_BATCH, N_VOCAB), 5.0, dtype=np.single)
    stored = _provider()._gguf_clone_llama_state(_saved_state(n_tokens, 10))
    live[:n_tokens, :] = stored.scores.copy()
    assert not live[: min(n_tokens, N_BATCH)].any()


def test_the_size_counts_the_arrays():
    state = _saved_state(3000, 1000)
    assert hf.HuggingFaceProvider._gguf_llama_state_nbytes(state) == 1000 + state.scores.nbytes + state.input_ids.nbytes


def _cache_with(p: hf.HuggingFaceProvider, key: str, kv_bytes: int) -> LlamaRAMCache:
    cache = LlamaRAMCache(capacity_bytes=1 << 40)
    cache[(1, 2, len(key))] = p._gguf_clone_llama_state(_saved_state(3000, kv_bytes))
    p._prompt_cache_store.set(key, cache)
    return cache


def test_keys_past_the_budget_are_evicted_oldest_first(monkeypatch):
    mb = 1 << 20
    monkeypatch.setenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", str(3.5 * mb / (1 << 30)))
    store = PromptCacheStore(max_entries=32)
    p = _provider(store)
    caches = [_cache_with(p, f"run:{i}", mb) for i in range(6)]
    evicted = p._gguf_enforce_prompt_cache_budget(keep=caches[-1])
    assert evicted == ["run:0", "run:1", "run:2"]
    assert store.keys() == ["run:3", "run:4", "run:5"]
    assert p._gguf_prompt_cache_total_bytes() <= p._gguf_prompt_cache_budget_bytes()
    # The eviction hook ran: the evicted cache's snapshots are freed, not just unlisted.
    assert len(caches[0].cache_state) == 0 and len(caches[5].cache_state) == 1
    stats = p.get_prompt_cache_stats()
    assert stats["gguf"]["prompt_cache_total_bytes"] == p._gguf_prompt_cache_total_bytes()
    assert stats["gguf"]["prompt_cache_budget_bytes"] == int(3.5 * mb)
    assert stats["meta_by_key"]["run:5"]["bytes"] == p._gguf_cache_nbytes(caches[5])


def test_the_cache_just_written_survives_a_budget_smaller_than_it(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", str(0.5 / 1024))  # 0.5 MiB
    p = _provider(PromptCacheStore(max_entries=32))
    old = _cache_with(p, "old", 1 << 20)
    new = _cache_with(p, "new", 1 << 20)
    assert p._gguf_enforce_prompt_cache_budget(keep=new) == ["old"]
    assert p._prompt_cache_store.keys() == ["new"] and len(new.cache_state) == 1 and len(old.cache_state) == 0


def test_default_budget_is_a_quarter_of_ram_capped_at_8_gib(monkeypatch):
    monkeypatch.delenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", raising=False)
    budget = _provider()._gguf_prompt_cache_budget_bytes()
    phys = __import__("os").sysconf("SC_PAGE_SIZE") * __import__("os").sysconf("SC_PHYS_PAGES")
    assert budget == min(8 << 30, phys // 4)


@pytest.mark.parametrize("raw", ["0", "-1", "lots"])
def test_a_bad_budget_is_refused(monkeypatch, raw):
    monkeypatch.setenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", raw)
    with pytest.raises(ValueError, match="ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB"):
        _provider()._gguf_prompt_cache_budget_bytes()


def test_store_evict_fires_the_callback_and_delete_does_not():
    seen = []
    store = PromptCacheStore(max_entries=4, on_evict=lambda k, v: seen.append(k))
    store.set("a", 1)
    store.set("b", 2)
    assert store.evict("a") is True and store.evict("a") is False
    assert store.delete("b") is True
    assert seen == ["a"]


class _FakeLlama:
    """Enough of `llama_cpp.Llama` for `_gguf_prefill_prompt_cache`: eval, save/load_state."""

    n_batch = 512

    def __init__(self):
        self.n_tokens = 0
        self._input_ids: list = []
        self._logits_all = False
        self._ctx = type("Ctx", (), {"kv_cache_seq_rm": lambda self, *a: True})()

    def reset(self):
        self.n_tokens, self._input_ids = 0, []

    def eval(self, tokens):
        self._input_ids = self._input_ids[: self.n_tokens] + [int(t) for t in tokens]
        self.n_tokens = len(self._input_ids)

    def save_state(self):
        return LlamaState(input_ids=np.zeros((N_CTX,), dtype=np.intc), scores=np.ones((N_BATCH, N_VOCAB), dtype=np.single),
                          n_tokens=self.n_tokens, llama_state=b"\0" * 1024, llama_state_size=1024, seed=0)

    def load_state(self, state):
        self.n_tokens = int(state.n_tokens)
        self._input_ids = [int(t) for t in state.input_ids[: self.n_tokens]]

    def set_cache(self, cache):
        pass

    @staticmethod
    def longest_token_prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if int(x) != int(y):
                break
            n += 1
        return n


def test_the_prefill_snapshot_is_compacted_and_the_budget_enforced(monkeypatch):
    """Through the real save site: the stored snapshot keeps one scores row, and the keys of
    earlier runs past the budget are evicted at once (red if either step is removed)."""
    store = PromptCacheStore(max_entries=32)
    p = _provider(store)
    llm = _FakeLlama()
    p.llm = llm
    one = p._gguf_cache_nbytes(_cache_with(p, "run:0", 0))
    _cache_with(p, "run:1", 0)
    monkeypatch.setenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", str(1.5 * one / (1 << 30)))
    cache = LlamaRAMCache(capacity_bytes=1 << 40)
    store.set("run:2", cache)
    prompt = tuple(range(1, 200))
    assert p._gguf_prefill_prompt_cache(cache, prompt, generation_boundary=150)
    (stored,) = list(cache.cache_state.values())
    assert stored.scores.shape == (1, N_VOCAB)
    assert store.keys()[-1] == "run:2" and "run:0" not in store.keys()
    assert p._gguf_prompt_cache_total_bytes() <= p._gguf_prompt_cache_budget_bytes()


def test_every_insert_path_is_held_to_the_budget(monkeypatch):
    """A key that enters by fork/load/clone (`store.set`, not the prefill save site) is held to the
    budget too: the store's after-set hook runs the enforcement."""
    mb = 1 << 20
    monkeypatch.setenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", str(2.5 * mb / (1 << 30)))
    store = PromptCacheStore(max_entries=32)
    p = _provider(store)
    store.set_on_set(p._prompt_cache_store_after_set)
    for i in range(5):
        _cache_with(p, f"fork:{i}", mb)  # each one only store.set()
    assert store.keys() == ["fork:3", "fork:4"]
    assert p._gguf_prompt_cache_total_bytes() <= p._gguf_prompt_cache_budget_bytes()


def test_a_provider_store_is_wired_with_the_after_set_hook():
    import inspect

    src = inspect.getsource(hf.HuggingFaceProvider._gguf_detach)
    assert "on_set=self._prompt_cache_store_after_set" in src
    from abstractcore.providers import base

    assert "on_set=self._prompt_cache_store_after_set" in inspect.getsource(base.BaseProvider.__init__)


def test_a_detaching_holder_gets_its_own_hooked_store_and_leaves_the_shared_one_alone(monkeypatch):
    """A holder leaving a pool other holders still use gets a fresh private store wired with its own
    hooks; the pool's shared store keeps the hooks it had."""
    from abstractcore.providers import gguf_slots

    shared = PromptCacheStore(max_entries=32)
    first = _provider(shared)  # the pool's first holder: its hooks are on the shared store
    first.__dict__["_gguf_scheduler"] = object()
    shared.set_on_set(first._prompt_cache_store_after_set)
    hooks = (shared._on_evict, shared._on_set)
    leaving = _provider()
    leaving._prompt_cache_store = shared
    leaving.__dict__["_gguf_scheduler"] = object()
    monkeypatch.setattr(gguf_slots, "detach", lambda sched, holder: True)
    assert leaving._gguf_detach() is True
    own = leaving._prompt_cache_store
    assert own is not shared
    assert own._on_evict == leaving._prompt_cache_store_evicted and own._on_set == leaving._prompt_cache_store_after_set
    assert (shared._on_evict, shared._on_set) == hooks


def test_the_budget_comes_from_the_config_and_the_environment_overrides_it(monkeypatch, tmp_path):
    cfg = tmp_path / "abstractcore.json"
    cfg.write_text('{"gguf_prompt_cache_max_gb": 2}', encoding="utf-8")
    monkeypatch.delenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", raising=False)
    p = _provider()
    p._abstractcore_config_file = str(cfg)
    assert p._gguf_prompt_cache_budget_bytes() == 2 << 30
    monkeypatch.setenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB", "0.5")
    q = _provider()
    q._abstractcore_config_file = str(cfg)
    assert q._gguf_prompt_cache_budget_bytes() == 1 << 29
    cfg.write_text('{"gguf_prompt_cache_max_gb": "lots"}', encoding="utf-8")
    monkeypatch.delenv("ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB")
    r = _provider()
    r._abstractcore_config_file = str(cfg)
    with pytest.raises(ValueError, match="gguf_prompt_cache_max_gb in the AbstractCore config"):
        r._gguf_prompt_cache_budget_bytes()


def test_a_restore_from_a_compacted_snapshot_re_evaluates_the_last_prompt_token():
    """The sampler must never read logits this prompt did not produce. A compacted snapshot has a
    zero scores row, so this holds only because the provider never restores to the full prompt:
    `feed_floor = n_prompt - 1` and the zero-feed guard in `_gguf_prefill_prompt_cache` roll a
    whole-prompt state back one token, which `eval` then decodes. Red if that guard goes."""
    llm = _FakeLlama()
    evals = []
    real_eval = llm.eval
    llm.eval = lambda toks: (evals.append(list(toks)), real_eval(toks))[1]
    loads = []
    real_load = llm.load_state
    llm.load_state = lambda st: (loads.append(st.scores.shape), real_load(st))[1]
    llm._ctx = type("Ctx", (), {"kv_cache_seq_rm": lambda self, *a: True})()
    p = _provider(PromptCacheStore(max_entries=32))
    p.llm = llm
    prompt = tuple(range(1, 200))
    cache = LlamaRAMCache(capacity_bytes=1 << 40)
    saved = _saved_state(len(prompt), 64)
    saved.input_ids[: len(prompt)] = prompt  # the state holds exactly this prompt
    whole = p._gguf_clone_llama_state(saved)
    assert whole.scores.shape == (1, N_VOCAB)
    cache[prompt] = whole  # a snapshot covering the WHOLE prompt (a durable bloc stores this)
    llm.eval([7, 7, 7])  # a previous call left another context resident
    evals.clear()
    assert p._gguf_prefill_prompt_cache(cache, prompt, save_state=False)
    fed = [t for batch in evals for t in batch]
    assert fed and fed[-1] == prompt[-1], f"the last prompt token was not decoded after the restore: {fed[-5:]}"
    assert llm.n_tokens == len(prompt)
    assert loads == [(1, N_VOCAB)], "the compacted snapshot was not the one restored"
