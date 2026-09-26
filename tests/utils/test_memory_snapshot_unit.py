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

    assert set(snap.keys()) == {"ts", "ram", "process", "device", "held", "resident", "host"}
    assert isinstance(snap["ts"], float)
    assert set(snap["ram"].keys()) == {"total_bytes", "available_bytes", "used_bytes", "percent"}
    assert set(snap["process"].keys()) == {"rss_bytes", "footprint_bytes"}
    assert set(snap["device"].keys()) == {
        "backend",
        "allocated_bytes",
        "total_bytes",
        "free_bytes",
        "host_in_use_bytes",
        "wired_limit_bytes",
        "mlx_active_bytes",
        "mlx_cache_bytes",
        "mlx_peak_bytes",
        "mlx_held_bytes",
        # MEM2 (2026-09-25): every in-process allocator, one process figure.
        "torch_mps_allocated_bytes",
        "torch_mps_driver_bytes",
        "llama_cpp_bytes",
        "metal_process_allocated_bytes",
        "process_held_bytes",
        "process_held_basis",
        "torch_cuda_reserved_bytes",
    }
    # `held` is the process-level MLX residency block (None without MLX).
    held = snap["held"]
    assert held is None or isinstance(held, dict)
    if isinstance(held, dict) and "error" not in held:
        assert {"backend", "active_bytes", "cache_bytes", "peak_bytes", "held_bytes", "models", "holders", "resident_models"} <= set(held)
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
    # rss comes from psutil (now broken -> None); footprint is a separate,
    # psutil-free macOS probe (proc_pid_rusage), so it may still be known.
    assert snap["process"]["rss_bytes"] is None
    assert snap["process"]["footprint_bytes"] is None or isinstance(snap["process"]["footprint_bytes"], int)
    assert set(snap["process"].keys()) == {"rss_bytes", "footprint_bytes"}


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
        "mlx_active_bytes": None,
        "mlx_cache_bytes": None,
        "mlx_peak_bytes": None,
        "mlx_held_bytes": None,
        # MEM2: no torch -> unknown; no llama.cpp engine -> a KNOWN 0; the
        # process figure is then a known 0 too (nothing can be held).
        "torch_mps_allocated_bytes": None,
        "torch_mps_driver_bytes": None,
        "llama_cpp_bytes": 0,
        "metal_process_allocated_bytes": None,
        "process_held_bytes": 0,
        "process_held_basis": "sum:llama_cpp_bytes",
        "torch_cuda_reserved_bytes": None,
    }
    assert snap["held"] is None


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


# ---------------------------------------------------------------------------
# Mission MEM2 (2026-09-25): one process figure across every backend, and a
# report that never imports a backend.
# ---------------------------------------------------------------------------
def test_process_held_bytes_sums_every_in_process_allocator(monkeypatch) -> None:
    from abstractcore.providers import hf_residency

    monkeypatch.setattr(memory_mod, "_device_snapshot_backend", lambda: {
        "backend": "metal", "allocated_bytes": 7, "total_bytes": None, "free_bytes": None,
        "host_in_use_bytes": None, "wired_limit_bytes": None, "mlx_active_bytes": 7,
        "mlx_cache_bytes": 3, "mlx_peak_bytes": 10, "mlx_held_bytes": 10,
    })
    monkeypatch.setattr(hf_residency, "hf_memory_report", lambda: {
        "backend": "huggingface", "torch_mps_allocated_bytes": 40, "torch_mps_driver_bytes": 100,
        "llama_cpp_bytes": 50, "held_bytes": 90, "models": [], "holders": 0, "resident_models": 0,
    })
    dev = memory_mod._device_snapshot()
    assert dev["torch_mps_allocated_bytes"] == 40 and dev["torch_mps_driver_bytes"] == 100
    assert dev["llama_cpp_bytes"] == 50
    # torch present: its driver counter is the device's allocation for the
    # process -- MLX and llama.cpp buffers are INSIDE it (measured 2026-09-25),
    # so the per-backend figures attribute it and are never added on top.
    assert dev["metal_process_allocated_bytes"] == 100
    assert dev["process_held_bytes"] == 100, "the device counter, not a sum that double-counts"
    assert dev["process_held_basis"] == "metal_device_counter"

    # torch absent: nothing else can allocate on the device, so the
    # per-backend figures ARE addends.
    monkeypatch.setattr(hf_residency, "hf_memory_report", lambda: {
        "backend": "huggingface", "torch_mps_allocated_bytes": None, "torch_mps_driver_bytes": None,
        "llama_cpp_bytes": 50, "held_bytes": 50, "models": [], "holders": 0, "resident_models": 0,
    })
    dev = memory_mod._device_snapshot()
    assert dev["metal_process_allocated_bytes"] is None
    assert dev["process_held_bytes"] == 10 + 50, "MLX held + llama.cpp when torch is absent"
    assert dev["process_held_basis"] == "sum:mlx_held_bytes+llama_cpp_bytes(estimated)"


def test_process_held_bytes_on_cuda_uses_torch_reserved_plus_llama_cpp(monkeypatch) -> None:
    """Review S5: on the GPU profile the Metal branch never fires; the figure
    fell to MLX + llama.cpp = 0 while torch held GBs on the GPU (MUTANT: drop
    the CUDA branch -> RED)."""
    import sys as _sys
    import types
    from types import SimpleNamespace

    from abstractcore.providers import hf_residency

    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, device_count=lambda: 2,
                                 memory_reserved=lambda i: [3_000, 1_000][i])
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(_sys.modules, "torch", torch)
    monkeypatch.setattr(memory_mod, "_device_snapshot_backend", lambda: {
        "backend": "cuda", "allocated_bytes": 2_000, "total_bytes": None, "free_bytes": None,
        "host_in_use_bytes": None, "wired_limit_bytes": None})
    report = {"backend": "huggingface", "torch_mps_allocated_bytes": None, "torch_mps_driver_bytes": None,
              "llama_cpp_bytes": 0, "held_bytes": 0, "models": [], "holders": 0, "resident_models": 0}
    monkeypatch.setattr(hf_residency, "hf_memory_report", lambda: dict(report))
    dev = memory_mod._device_snapshot()
    assert dev["torch_cuda_reserved_bytes"] == 4_000
    assert dev["process_held_bytes"] == 4_000 and dev["process_held_basis"] == "cuda_device_counter"
    report["llama_cpp_bytes"] = 500
    dev = memory_mod._device_snapshot()
    assert dev["process_held_bytes"] == 4_500
    assert dev["process_held_basis"] == "cuda_device_counter+llama_cpp_bytes(estimated)"
    # CUDA not initialized yet: never initialize it just to report
    torch.cuda.is_initialized = lambda: False
    assert memory_mod._device_snapshot()["torch_cuda_reserved_bytes"] is None


def test_resident_block_folds_every_backend_and_sums_bytes(monkeypatch) -> None:
    import types

    from abstractcore.providers import hf_residency, mlx_residency

    monkeypatch.setattr(mlx_residency, "mlx_memory_report", lambda: {
        "backend": "mlx", "active_bytes": 7, "cache_bytes": 0, "peak_bytes": 7, "held_bytes": 7,
        "models": [{"lane": "mlx_lm", "models": ["a/mlx"], "holders": 1, "held_bytes": 7, "weights_alive": True, "holder_rows": [{"id": 1}]}],
        "holders": 1, "resident_models": 1,
    })
    monkeypatch.setattr(hf_residency, "hf_memory_report", lambda: {
        "backend": "huggingface", "torch_mps_allocated_bytes": None, "torch_mps_driver_bytes": None,
        "llama_cpp_bytes": 0, "held_bytes": 5,
        "models": [{"lane": "transformers", "backend": "huggingface", "models": ["b/hf"], "holders": 2, "held_bytes": 5, "weights_alive": True, "holder_rows": [{"id": 2}]}],
        "holders": 2, "resident_models": 1,
    })
    fake_manager = types.ModuleType("abstractcore.embeddings.manager")
    fake_manager.embeddings_memory_report = lambda: {
        "backend": "embeddings", "held_bytes": 3,
        "models": [{"lane": "embeddings", "backend": "embeddings", "models": ["c/emb"], "holders": 1, "held_bytes": 3, "weights_alive": True}],
        "holders": 1, "resident_models": 1,
    }
    monkeypatch.setitem(sys.modules, "abstractcore.embeddings.manager", fake_manager)
    # make MLX "importable" for this test regardless of the host
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=types.ModuleType("mlx.core")))
    monkeypatch.setitem(sys.modules, "mlx.core", types.ModuleType("mlx.core"))

    resident = memory_mod._resident_snapshot()
    assert set(resident["backends"]) == {"mlx", "huggingface", "embeddings"}
    assert [r["models"][0] for r in resident["models"]] == ["a/mlx", "b/hf", "c/emb"]
    assert all("holder_rows" not in r for r in resident["models"]), "object ids stay out of the JSON"
    assert resident["models"][0]["backend"] == "mlx"
    assert resident["total_held_bytes"] == 7 + 5 + 3
    assert resident["holders"] == 4 and resident["resident_models"] == 3


def test_report_never_imports_the_embeddings_backend(monkeypatch) -> None:
    """The manager module imports sentence-transformers (and torch) eagerly; a
    memory report must read it only when it is already loaded."""
    monkeypatch.setitem(sys.modules, "abstractcore.embeddings.manager", None)  # absent -> import would raise
    resident = memory_mod._resident_snapshot()
    assert resident["backends"]["embeddings"] is None
    assert sys.modules.get("abstractcore.embeddings.manager") is None
