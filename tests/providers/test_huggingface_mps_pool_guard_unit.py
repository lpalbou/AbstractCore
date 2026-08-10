"""In-call MPS pool guard for the transformers text lanes (2026-08-07).

CONTEXT. `_transformers_maybe_release_device_pool` is CALL-scoped: every text-lane
call site invokes it in a `finally`, i.e. after `generate()` has already returned.
The ratchet it bounds is STEP-scoped — `DynamicCache` grows by `torch.cat` on every
decode step, so step *t* asks the allocator for a buffer sized for *t* tokens and
frees the *t-1* one, and those sizes never repeat and never shrink. Nothing inside
`generate()` ever yielded to the guard, so one long call ratcheted without limit.

MEASURED on `Qwen/Qwen3.5-4B` bf16/MPS with a 12,718-token UNCACHED prompt, one
process, one model load (`untracked/prompt-cache-bench/oom/results/`):

    max_output_tokens=512   ->  driver  10.26 GiB, pool slack   1.89 GiB
    max_output_tokens=4096  ->  driver 113.26 GiB, pool slack 104.78 GiB
                                (`current_allocated_memory` 8.48 GiB — the
                                computation needed ~8.5 GiB and the allocator
                                held 104.8 GiB of FREED buffers; host
                                free+inactive fell 111 -> 18 GB inside ONE call)

With the guard, the same 4096-token arm completed at 15.25 GiB driver / 7.87 GiB
slack, full 4096 tokens, no truncation, for 2.34 s of release time in 270 s.

These tests stub the allocator counters so the policy is pinned without a GPU: the
guard must fire from INSIDE the generate call, must respect the threshold, and must
be switchable off. On pre-fix code `test_guard_releases_during_generate` fails
because no release happens until the call returns — that is the point.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _FakeModel(torch.nn.Module):
    """A module whose `forward` stands in for one decode step."""

    def forward(self, *args, **kwargs):  # pragma: no cover - never called directly
        return None


def _provider(monkeypatch, *, device="mps", pooled_series: List[int], threshold_gib=4.0):
    """Provider with no model load, stubbed MPS counters and a fake model.

    `pooled_series` is consumed one entry per counter READ, so a test controls
    exactly what the guard sees on each check.
    """
    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.device = device
    p.model_instance = _FakeModel()
    p._transformers_pool_release_threshold = int(threshold_gib * 1073741824)

    reads = {"n": 0}
    calls: Dict[str, int] = {"empty_cache": 0, "synchronize": 0}

    def _current():
        return 0

    def _driver():
        i = min(reads["n"], len(pooled_series) - 1)
        reads["n"] += 1
        return pooled_series[i]

    monkeypatch.setattr(torch.mps, "current_allocated_memory", _current, raising=False)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", _driver, raising=False)
    monkeypatch.setattr(torch.mps, "empty_cache",
                        lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
                        raising=False)
    monkeypatch.setattr(torch.mps, "synchronize",
                        lambda: calls.__setitem__("synchronize", calls["synchronize"] + 1),
                        raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    return p, calls


GIB = 1073741824


def test_guard_releases_during_generate(monkeypatch):
    """The release must happen WHILE the call is in flight, not after it."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "1")
    # Pool sits under the bound, then crosses it partway through the "decode".
    series = [1 * GIB] * 5 + [9 * GIB] * 5
    p, calls = _provider(monkeypatch, pooled_series=series)

    with p._transformers_decode_pool_guard() as stats:
        assert stats["enabled"] is True
        for _ in range(10):
            p.model_instance(torch.zeros(1))
            # The whole point: releases accumulate BEFORE the block exits.
        assert calls["empty_cache"] == 5, "guard did not release inside the call"

    assert stats["checks"] == 10
    assert stats["releases"] == 5
    assert stats["peak_pooled_bytes"] == 9 * GIB
    assert calls["synchronize"] == 5


def test_guard_is_noop_below_threshold(monkeypatch):
    """Healthy loops keep full pool reuse — the pool is never dropped while it
    is doing its job."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "1")
    p, calls = _provider(monkeypatch, pooled_series=[3 * GIB])

    with p._transformers_decode_pool_guard() as stats:
        for _ in range(20):
            p.model_instance(torch.zeros(1))

    assert stats["checks"] == 20
    assert stats["releases"] == 0
    assert calls["empty_cache"] == 0


def test_guard_respects_stride(monkeypatch):
    """Counters are read once per `stride` forwards, not on every token."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "8")
    p, calls = _provider(monkeypatch, pooled_series=[9 * GIB])

    with p._transformers_decode_pool_guard() as stats:
        for _ in range(32):
            p.model_instance(torch.zeros(1))

    assert stats["checks"] == 4  # 32 forwards / stride 8
    assert stats["releases"] == 4


def test_guard_kill_switch_disables(monkeypatch):
    """`<= 0` disables, matching ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP and
    ABSTRACTCORE_MPS_POOL_RELEASE_GB. This is the A/B control arm."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "0")
    p, calls = _provider(monkeypatch, pooled_series=[99 * GIB])

    with p._transformers_decode_pool_guard() as stats:
        for _ in range(20):
            p.model_instance(torch.zeros(1))

    assert stats["enabled"] is False
    assert stats["releases"] == 0
    assert calls["empty_cache"] == 0


def test_guard_noop_off_mps(monkeypatch):
    """CUDA/CPU never pay for this."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "1")
    p, calls = _provider(monkeypatch, device="cpu", pooled_series=[99 * GIB])

    with p._transformers_decode_pool_guard() as stats:
        for _ in range(10):
            p.model_instance(torch.zeros(1))

    assert stats["enabled"] is False
    assert calls["empty_cache"] == 0


def test_guard_removes_its_hook(monkeypatch):
    """The hook must not outlive the call — a leaked pre-hook would run the
    allocator check on every later forward, including the vision lane's."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "1")
    p, calls = _provider(monkeypatch, pooled_series=[9 * GIB])

    with p._transformers_decode_pool_guard():
        p.model_instance(torch.zeros(1))
    fired_inside = calls["empty_cache"]

    for _ in range(10):
        p.model_instance(torch.zeros(1))

    assert fired_inside == 1
    assert calls["empty_cache"] == 1, "pre-hook survived the context manager"
    assert not p.model_instance._forward_pre_hooks


def test_guard_never_breaks_generation_on_counter_failure(monkeypatch):
    """An allocator that raises must not take the generation down with it."""
    monkeypatch.setenv("ABSTRACTCORE_MPS_POOL_GUARD_STRIDE", "1")
    p, calls = _provider(monkeypatch, pooled_series=[1 * GIB])

    def _boom():
        raise RuntimeError("allocator counter unavailable")

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", _boom, raising=False)

    with p._transformers_decode_pool_guard() as stats:
        for _ in range(5):
            p.model_instance(torch.zeros(1))  # must not raise

    assert stats["releases"] == 0
