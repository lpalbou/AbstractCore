"""CPU-only guards for the native MTP lane's prefix-cache budget and telemetry.

Background (2026-09-22): the native MTP lane handed mlx-vlm's APC a flat
0.5 GiB memory budget. mlx-vlm refuses to RETAIN any single prefix snapshot
larger than that budget (`APCManager.store_exact_cache`), a hybrid model's
snapshot grows ~27-50 KB per prompt token, and the skip is silent. Past
~10-19k prompt tokens every turn re-prefilled the whole conversation with
`cached_tokens == 0` and nothing in the ledger saying why. Neither the
constant nor the silence may come back.
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from abstractcore.providers.mlx_native_session import NativeSession
from abstractcore.providers.mlx_provider import MLXProvider


@pytest.fixture
def apc_factory(monkeypatch):
    package = types.ModuleType("mlx_vlm")
    package.__path__ = []
    module = types.ModuleType("mlx_vlm.apc")
    factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(kwargs=kwargs))
    module.APCManager = factory
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    monkeypatch.setitem(sys.modules, "mlx_vlm.apc", module)
    return factory


def test_session_budget_is_upstream_auto_sizing_unless_the_operator_chose(apc_factory, monkeypatch):
    monkeypatch.delenv("APC_CHECKPOINT_INTERVAL_TOKENS", raising=False)
    monkeypatch.delenv("APC_CHECKPOINT_ENTRIES", raising=False)
    session = NativeSession()
    assert session.prompt_cache() is session.prompt_cache()
    overrides = apc_factory.call_args.kwargs["overrides"]
    for forced in ("memory_max_gb", "memory_reserve_gb"):
        assert forced not in overrides, f"{forced} forced to a constant nobody sized for the model"
    NativeSession().prompt_cache(memory_max_gb=3)
    assert apc_factory.call_args.kwargs["overrides"]["memory_max_gb"] == 3.0


def test_session_snapshots_are_one_per_call_with_lineage_depth(apc_factory, monkeypatch):
    """Re-sized 2026-09-22 (mission A3); supersedes the 256 x 4 pin.

    mlx-vlm caps a single prompt's intermediates AND the whole snapshot store
    with the same `checkpoint_entries`, so 256 x 4 made every prompt fill the
    store and no earlier lineage survived a tool loop (the next turn after any
    tool-using turn re-prefilled the whole conversation). One snapshot per call
    (interval past any context window), eight distinct prompts restorable, and
    the snapshot placed 8 tokens before the end so a merge that rewrites the
    previous last message's closing still restores all but those 8.
    """
    for env in ("APC_CHECKPOINT_INTERVAL_TOKENS", "APC_CHECKPOINT_ENTRIES", "APC_CHECKPOINT_GUARD_TOKENS"):
        monkeypatch.delenv(env, raising=False)
    NativeSession().prompt_cache()
    overrides = apc_factory.call_args.kwargs["overrides"]
    assert overrides["checkpoint_interval_tokens"] >= 1_000_000
    assert overrides["checkpoint_entries"] == 8
    assert overrides["checkpoint_guard_tokens"] == 8
    # The operator's environment still wins: the session must not shadow it.
    monkeypatch.setenv("APC_CHECKPOINT_INTERVAL_TOKENS", "512")
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "6")
    monkeypatch.setenv("APC_CHECKPOINT_GUARD_TOKENS", "1")
    NativeSession().prompt_cache()
    overrides = apc_factory.call_args.kwargs["overrides"]
    assert "checkpoint_interval_tokens" not in overrides
    assert "checkpoint_entries" not in overrides
    assert "checkpoint_guard_tokens" not in overrides


def _provider(**cache_options):
    p = MLXProvider.__new__(MLXProvider)
    p.logger = Mock()
    p._mtp_processor = object()
    # No `_mtp_drafter` -> `_mtp_active` reads False and `_mtp_kwargs` adds
    # nothing; only the APC kwargs are under test here.
    p._mtp_prompt_cache_warned = True
    p._mlx_cache_options = {"disk_path": None, "disk_max_gb": 4.0, **cache_options}
    p._native_cache_key = "conversation"
    p._native_session = SimpleNamespace(prompt_cache=Mock(return_value=object()))
    return p


@pytest.mark.parametrize("budget", [3.0, None])
def test_provider_forwards_exactly_its_memory_budget_option(budget):
    p = _provider(memory_max_gb=budget)
    out = p._mtp_call_kwargs({"max_tokens": 4})
    p._native_session.prompt_cache.assert_called_once_with(memory_max_gb=budget)
    assert out["apc_manager"] is p._native_session.prompt_cache.return_value


class _FakeApc:
    def __init__(self, **counters):
        self.counters = counters

    def stats_snapshot(self):
        return dict(self.counters)


def _telemetry_provider(apc):
    p = MLXProvider.__new__(MLXProvider)
    p.logger = Mock()
    p._native_session = SimpleNamespace(apc=apc)
    return p


def test_skipped_store_is_reported_not_disguised_as_a_plain_miss():
    apc = _FakeApc(exact_stores=0, memory_skips=2, memory_max_bytes=512 << 20, resident_bytes=0)
    p = _telemetry_provider(apc)
    before = p._native_apc_counters()
    apc.counters["memory_skips"] = 4  # this call tried twice and could retain nothing
    p._mtp_last_apc = p._native_apc_delta(before)
    telemetry = {"mode": "key", "key": "conversation", "backend": "mlx_vlm_apc"}
    p._native_apc_telemetry(telemetry, SimpleNamespace(cached_tokens=0, prompt_tokens=15394))
    assert telemetry["outcome"] == "cold"
    assert (telemetry["cached_tokens"], telemetry["fed_tokens"]) == (0, 15394)
    assert telemetry["apc"]["memory_skips"] == 2 and telemetry["apc"]["exact_stores"] == 0
    assert telemetry["apc"]["memory_max_bytes"] == 512 << 20
    assert telemetry["degraded_reason"].startswith("#FALLBACK native_apc_store_skipped")
    assert "15394-token" in telemetry["degraded_reason"] and "512 MiB" in telemetry["degraded_reason"]
    p.logger.warning.assert_called_once()
    # Once per key: the next turn on the same key is reported, not re-logged.
    again = {"mode": "key", "key": "conversation", "backend": "mlx_vlm_apc"}
    p._native_apc_telemetry(again, SimpleNamespace(cached_tokens=0, prompt_tokens=15400))
    assert "degraded_reason" in again
    p.logger.warning.assert_called_once()


def test_hit_with_a_retained_store_carries_no_degraded_reason():
    apc = _FakeApc(exact_stores=2, exact_hits=0, memory_skips=0, memory_max_bytes=8 << 30, resident_bytes=1)
    p = _telemetry_provider(apc)
    before = p._native_apc_counters()
    apc.counters.update(exact_stores=3, exact_hits=1)
    p._mtp_last_apc = p._native_apc_delta(before)
    telemetry = {"mode": "key", "key": "conversation", "backend": "mlx_vlm_apc"}
    p._native_apc_telemetry(telemetry, SimpleNamespace(cached_tokens=12033, prompt_tokens=12054))
    assert telemetry["outcome"] == "hit_restore"
    assert (telemetry["cached_tokens"], telemetry["fed_tokens"]) == (12033, 21)
    assert telemetry["apc"]["exact_stores"] == 1 and telemetry["apc"]["exact_hits"] == 1
    assert "degraded_reason" not in telemetry
    p.logger.warning.assert_not_called()


def test_scheduled_lane_reports_only_what_its_runtime_measured():
    p = MLXProvider.__new__(MLXProvider)
    p.logger = Mock()
    p._native_runtime = object()
    p._native_session = SimpleNamespace(apc=_FakeApc(memory_skips=9))
    assert p._native_apc_counters() is None, "worker-owned manager must not be read from the caller thread"
    p._mtp_last_apc = p._native_apc_delta(None)
    telemetry = {"mode": "key", "key": "k", "backend": "mlx_vlm_apc"}
    p._native_apc_telemetry(telemetry, SimpleNamespace(cached_tokens=5, prompt_tokens=9))
    assert telemetry == {"mode": "key", "key": "k", "backend": "mlx_vlm_apc",
                         "cached_tokens": 5, "fed_tokens": 4, "outcome": "hit_restore"}
