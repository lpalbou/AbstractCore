"""Unit tests for `abstractcore.utils.memory.get_memory_snapshot`.

The snapshot is pure observation: fixed shape, unknown values are None, and it
NEVER raises — even when psutil or the device backends are broken/missing.
"""

from __future__ import annotations

import sys

import pytest

from abstractcore.utils import memory as memory_mod


def test_memory_snapshot_shape() -> None:
    snap = memory_mod.get_memory_snapshot()

    assert set(snap.keys()) == {"ts", "ram", "process", "device", "host"}
    assert isinstance(snap["ts"], float)
    assert set(snap["ram"].keys()) == {"total_bytes", "available_bytes", "used_bytes", "percent"}
    assert set(snap["process"].keys()) == {"rss_bytes"}
    assert set(snap["device"].keys()) == {
        "backend",
        "allocated_bytes",
        "total_bytes",
        "free_bytes",
        "host_in_use_bytes",
        "wired_limit_bytes",
    }
    assert snap["device"]["backend"] in {"metal", "cuda", "mps", None}
    assert set(snap["host"].keys()) == {"host_id", "host_name", "kind"}
    assert snap["host"]["kind"] == "local"


def test_memory_snapshot_reports_real_ram_on_this_host() -> None:
    snap = memory_mod.get_memory_snapshot()

    # psutil is a declared dependency; on a working host RAM values are known.
    assert isinstance(snap["ram"]["total_bytes"], int) and snap["ram"]["total_bytes"] > 0
    assert isinstance(snap["ram"]["available_bytes"], int)
    assert isinstance(snap["process"]["rss_bytes"], int) and snap["process"]["rss_bytes"] > 0


def test_memory_snapshot_never_raises_when_psutil_broken(monkeypatch) -> None:
    import psutil

    def _boom(*args, **kwargs):
        raise RuntimeError("psutil broken")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    monkeypatch.setattr(psutil, "Process", _boom)

    snap = memory_mod.get_memory_snapshot()

    assert snap["ram"] == {
        "total_bytes": None,
        "available_bytes": None,
        "used_bytes": None,
        "percent": None,
    }
    assert snap["process"] == {"rss_bytes": None}


def test_memory_snapshot_never_raises_without_device_backends(monkeypatch) -> None:
    # None entries in sys.modules make the guarded imports raise ImportError.
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    snap = memory_mod.get_memory_snapshot()

    assert snap["device"] == {
        "backend": None,
        "allocated_bytes": None,
        "total_bytes": None,
        "free_bytes": None,
        "host_in_use_bytes": None,
        "wired_limit_bytes": None,
    }


def test_memory_snapshot_never_raises_when_device_probe_errors(monkeypatch) -> None:
    import types

    class _BrokenMx:
        metal = None

        @staticmethod
        def get_active_memory():
            raise RuntimeError("probe failed")

        @staticmethod
        def device_info():
            raise RuntimeError("probe failed")

    broken = _BrokenMx()
    # Both entries so `import mlx.core as mx` binds the broken stand-in even if
    # the real mlx was imported earlier in the process.
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=broken))
    monkeypatch.setitem(sys.modules, "mlx.core", broken)
    monkeypatch.setitem(sys.modules, "torch", None)
    # Keep the metal-branch host probes hermetic (no real ioreg/sysctl runs).
    monkeypatch.setattr(memory_mod, "_ioreg_accelerator_in_use_bytes", lambda timeout_s=1.0: None)
    monkeypatch.setattr(memory_mod, "metal_wired_limit_bytes", lambda: None)
    monkeypatch.setattr(memory_mod, "metal_recommended_working_set_bytes", lambda: None)

    snap = memory_mod.get_memory_snapshot()

    # The backend probe claimed "metal" but every value stayed unknown — and
    # nothing raised.
    assert snap["device"]["backend"] in {"metal", None}
    assert snap["device"]["allocated_bytes"] is None
    assert snap["device"]["free_bytes"] is None
    assert snap["device"]["host_in_use_bytes"] is None
    assert snap["device"]["wired_limit_bytes"] is None


# ---------------------------------------------------------------------------
# Host-wide accelerator truth (metal): ioreg parse + wired-limit ceiling
# ---------------------------------------------------------------------------

# Real single-line ioreg -r -c IOAccelerator -l shape from an Apple-silicon
# host (values shortened): the exact-key discipline matters — "In use system
# memory (driver)" is a DIFFERENT statistic and must not be counted.
_IOREG_FIXTURE = (
    '+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X, id 0x100000482>\n'
    '    "PerformanceStatistics" = {"In use system memory (driver)"=0,'
    '"Alloc system memory"=49865850880,"Tiler Utilization %"=7,'
    '"Renderer Utilization %"=30,"Device Utilization %"=30,'
    '"In use system memory"=8461860864}\n'
)


def test_parse_ioreg_accelerator_in_use_bytes_exact_key() -> None:
    assert memory_mod.parse_ioreg_accelerator_in_use_bytes(_IOREG_FIXTURE) == 8_461_860_864


def test_parse_ioreg_accelerator_in_use_bytes_sums_blocks_and_tolerates_junk() -> None:
    two_gpus = _IOREG_FIXTURE + (
        '+-o OtherAccel  <class OtherAccel, id 0x1>\n'
        '    "PerformanceStatistics" = {"In use system memory"=1000,"Device Utilization %"=1}\n'
    )
    assert memory_mod.parse_ioreg_accelerator_in_use_bytes(two_gpus) == 8_461_860_864 + 1000

    # No PerformanceStatistics / no key / junk values -> None, never a guess.
    assert memory_mod.parse_ioreg_accelerator_in_use_bytes("") is None
    assert memory_mod.parse_ioreg_accelerator_in_use_bytes("+-o Foo\n") is None
    assert (
        memory_mod.parse_ioreg_accelerator_in_use_bytes(
            '"PerformanceStatistics" = {"In use system memory (driver)"=123}'
        )
        is None
    )
    assert (
        memory_mod.parse_ioreg_accelerator_in_use_bytes(
            '"PerformanceStatistics" = {"In use system memory"=not-a-number}'
        )
        is None
    )


def test_metal_snapshot_carries_host_in_use_and_wired_limit(monkeypatch) -> None:
    import types

    class _WorkingMx:
        metal = None

        @staticmethod
        def get_active_memory():
            return 5_000

        @staticmethod
        def device_info():
            return {"memory_size": 137_438_953_472}

    working = _WorkingMx()
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=working))
    monkeypatch.setitem(sys.modules, "mlx.core", working)
    monkeypatch.setattr(memory_mod, "_ioreg_accelerator_in_use_bytes", lambda timeout_s=1.0: 8_461_860_864)
    monkeypatch.setattr(memory_mod, "metal_wired_limit_bytes", lambda: 115_343_360_000)

    device = memory_mod.get_memory_snapshot()["device"]

    assert device["backend"] == "metal"
    assert device["allocated_bytes"] == 5_000  # process-local truth unchanged
    assert device["host_in_use_bytes"] == 8_461_860_864  # HOST-wide truth
    assert device["wired_limit_bytes"] == 115_343_360_000

    # Sysctl unset -> Metal's recommended working set is the ceiling fallback.
    monkeypatch.setattr(memory_mod, "metal_wired_limit_bytes", lambda: None)
    monkeypatch.setattr(memory_mod, "metal_recommended_working_set_bytes", lambda: 99_000)
    assert memory_mod.get_memory_snapshot()["device"]["wired_limit_bytes"] == 99_000


def test_context_estimator_reuses_the_memory_ceiling_helpers() -> None:
    """ONE ceiling truth: the estimator's budget probes ARE the memory
    module's helpers (extract-and-share, not duplicate)."""
    from abstractcore.utils import context_estimate as est_mod

    assert est_mod._metal_wired_limit_bytes is memory_mod.metal_wired_limit_bytes
    assert est_mod._metal_recommended_working_set_bytes is memory_mod.metal_recommended_working_set_bytes


# ---------------------------------------------------------------------------
# prompt_cache_store_bytes (per-model cache footprint helper)
# ---------------------------------------------------------------------------


def test_prompt_cache_store_bytes_sums_key_and_snapshot_bytes() -> None:
    stats = {
        "keys": ["a", "b", "c"],
        "meta_by_key": {
            "a": {"bytes": 100},
            "b": {"token_count": 5},  # no byte figure -> not counted
            "c": {"bytes": 23},
        },
        "snapshots": {"count": 2, "bytes": 1000},
    }
    assert memory_mod.prompt_cache_store_bytes(stats) == 1123


def test_prompt_cache_store_bytes_empty_store_is_a_known_zero() -> None:
    assert memory_mod.prompt_cache_store_bytes({"keys": []}) == 0
    assert memory_mod.prompt_cache_store_bytes({"keys": [], "snapshots": {"count": 0, "bytes": None}}) == 0


def test_prompt_cache_store_bytes_unknown_stays_none() -> None:
    # Keys with no byte figures anywhere -> unknown, never 0.
    assert memory_mod.prompt_cache_store_bytes({"keys": ["a"], "meta_by_key": {"a": {"token_count": 5}}}) is None
    assert memory_mod.prompt_cache_store_bytes(None) is None
    assert memory_mod.prompt_cache_store_bytes("junk") is None
    assert memory_mod.prompt_cache_store_bytes({}) is None
    # Bools are not byte counts.
    assert memory_mod.prompt_cache_store_bytes({"keys": ["a"], "meta_by_key": {"a": {"bytes": True}}}) is None
