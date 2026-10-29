"""The residency failsafe (core ADR 0010): fit before load, evict unlocked LRU,
refuse with a typed error and smaller suggestions. Fakes only: no weights."""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, List

import pytest

import abstractcore.providers.huggingface_provider  # noqa: F401 - the HF backend is "loaded"
from abstractcore.exceptions import ModelDoesNotFitError
from abstractcore.providers import hf_residency, process_residency as pr
from abstractcore.providers.generation_cancel import InflightGenerations

GB = 1000 ** 3


class _Holder:
    def __init__(self, model: str, size: int, last_used: float):
        self.model = model
        self.llm = object()
        self.size = size
        self._reg = InflightGenerations()
        self._reg.last_used_at = last_used

    def _inflight_generations(self):
        return self._reg


class _Card:
    """A device whose free memory grows when a holder is ejected."""

    def __init__(self, total: int, holders: List[_Holder]):
        self.total = total
        self.holders = holders

    def free(self) -> int:
        return self.total - sum(h.size for h in self.holders if h.llm is not None)


class _Locker:
    def __init__(self, model: str, label: str):
        self.model, self.label = model, label

    def residency_claims(self):
        return [{"provider": "huggingface", "model": self.model, "locked": True, "kind": "lock",
                 "owner": "x", "owner_label": self.label}]


@pytest.fixture
def card(monkeypatch):
    gc.collect()
    holders: List[_Holder] = []
    c = _Card(15 * GB, holders)
    monkeypatch.setattr(hf_residency, "registered_providers", lambda: list(holders))
    monkeypatch.setattr(hf_residency, "_holder_alive", lambda h: h.llm is not None)
    monkeypatch.setattr(hf_residency, "_holder_row", lambda h: {"held_bytes": h.size})
    ejected: List[str] = []

    def fake_eject(backend, model, reason="eject"):
        for h in holders:
            if h.model == model:
                h.llm = None
        ejected.append(model)
        return {"ok": True}

    monkeypatch.setattr(pr, "eject", fake_eject)
    c.ejected = ejected
    return c


def test_evicts_least_recently_used_first_and_only_what_is_needed(card):
    now = time.time()
    card.holders += [_Holder("old/A", 4 * GB, now - 300), _Holder("recent/B", 4 * GB, now - 10),
                     _Holder("older/C", 4 * GB, now - 600)]
    out = pr.ensure_fit("new/D", 6 * GB, card.free)       # free 3 GB -> one eviction of 4 GB is enough
    assert card.ejected == ["older/C"], f"MUTANT: not LRU -> RED ({card.ejected})"
    assert out["fits"] and [e["model"] for e in out["evicted"]] == ["older/C"]


def test_locked_and_in_use_models_are_never_evicted(card):
    now = time.time()
    a, b = _Holder("locked/A", 6 * GB, now - 900), _Holder("busy/B", 6 * GB, now - 800)
    card.holders += [a, b]
    lock = _Locker("locked/A", "the runtime of account default/admin")
    pr.register_claimant(lock)
    running = type("C", (), {})()                        # a generation running on B (held: a WeakSet)
    b._reg._calls.add(running)
    try:
        with pytest.raises(ModelDoesNotFitError) as err:
            pr.ensure_fit("big/E", 8 * GB, card.free)
    finally:
        pr.unregister_claimant(lock)
    assert card.ejected == [], "MUTANT: lock / in-flight ignored -> evicted -> RED"
    e = err.value
    assert e.needed_bytes == 8 * GB and e.free_bytes == 3 * GB and e.device == "GPU"
    msg = str(e)
    assert "big/E needs 8.0 GB but only 3.0 GB is free on the GPU" in msg
    assert "locked/A (locked by the runtime of account default/admin)" in msg and "busy/B (in use)" in msg
    assert "unlock" in msg


def test_refusal_suggests_smaller_artifacts_of_the_same_model(card, monkeypatch):
    import abstractcore.config.model_catalog as mc

    monkeypatch.setattr(mc, "load_seed", lambda: {"rows": [{
        "id": "qwen3.5-9b", "family": "qwen3.5", "display_name": "Qwen3.5 9B", "artifacts": [
            {"provider": "huggingface", "artifact": "unsloth/Qwen3.5-9B-GGUF:Q8_0", "quant": "q8_0", "download_bytes": 9 * GB},
            {"provider": "huggingface", "artifact": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M", "quant": "q4_k_m", "download_bytes": int(5.6 * GB)},
            {"provider": "mlx", "artifact": "mlx/x", "quant": "4bit", "download_bytes": GB}]},
        {"id": "qwen3.5-4b", "family": "qwen3.5", "display_name": "Qwen3.5 4B", "artifacts": [
            {"provider": "huggingface", "artifact": "unsloth/Qwen3.5-4B-GGUF:Q4_K_M", "quant": "q4_k_m", "download_bytes": int(2.7 * GB)}]}]})
    card.holders.append(_Holder("pinned/A", 7 * GB, time.time()))
    lock = _Locker("pinned/A", "the runtime of account default/admin")
    pr.register_claimant(lock)
    try:
        with pytest.raises(ModelDoesNotFitError) as err:
            pr.ensure_fit("unsloth/Qwen3.5-9B-GGUF:Q8_0", 11 * GB, card.free)
    finally:
        pr.unregister_claimant(lock)
    arts = [s["artifact"] for s in err.value.suggestions]
    assert arts == ["unsloth/Qwen3.5-9B-GGUF:Q4_K_M", "unsloth/Qwen3.5-4B-GGUF:Q4_K_M"], arts
    assert "unsloth/Qwen3.5-9B-GGUF:Q4_K_M (5.6 GB)" in str(err.value)


def test_the_model_being_loaded_is_never_its_own_victim(card):
    card.holders.append(_Holder("same/M", 12 * GB, 0.0))
    with pytest.raises(ModelDoesNotFitError):
        pr.ensure_fit("same/M", 14 * GB, card.free)
    assert card.ejected == []


def test_the_gguf_load_runs_the_fit_check_before_allocating(monkeypatch):
    """MUTANT: no call in `_load_gguf_model` -> llama.cpp allocates first -> RED."""
    import inspect
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    src = inspect.getsource(HuggingFaceProvider._load_gguf_model_unlocked)
    assert src.index("self._gguf_ensure_fit(") < src.index("llama_cls(**llama_kwargs)")
    assert "self._gguf_ensure_cpu_fit(" in src
    # one load at a time: the check and the allocation it admits under one lock (review)
    assert "with load_lock():" in inspect.getsource(HuggingFaceProvider._load_gguf_model)


def test_suggestions_use_the_fit_reserve(monkeypatch):
    import abstractcore.config.model_catalog as mc

    monkeypatch.setattr(mc, "load_seed", lambda: {"rows": [{"id": "fam-x", "family": "famx", "artifacts": [
        {"provider": "huggingface", "artifact": "org/fam-x:Q4", "download_bytes": int(4.3 * GB)}]}]})
    assert pr.smaller_suggestions("org/fam-x:Q8", int(6.1 * GB), reserve_bytes=int(1.6 * GB)) == [], (
        "MUTANT: a 1 GiB reserve -> suggests a model the check itself refuses -> RED")
    assert pr.smaller_suggestions("org/fam-x:Q8", int(7 * GB), reserve_bytes=int(1.6 * GB))
