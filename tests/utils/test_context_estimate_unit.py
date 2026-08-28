"""Unit tests for `abstractcore.utils.context_estimate.estimate_context_fit`
and the GGUF geometry header walker.

Every geometry source is faked/local: a real (tiny) GGUF header written to
disk, snapshot dirs with config.json, a faked Ollama /api/show. No weights,
no live servers, and the estimator must NEVER raise.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from abstractcore.utils import memory as memory_mod
from abstractcore.utils.context_estimate import estimate_context_fit
from abstractcore.utils.model_cache import read_gguf_geometry

GIB = 1024**3


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path: Path, kvs: Dict[str, Any]) -> Path:
    """Minimal valid GGUF header: string values type 8, int values u32."""
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    for key, value in kvs.items():
        blob += _gguf_string(key)
        if isinstance(value, str):
            blob += struct.pack("<I", 8) + _gguf_string(value)
        else:
            blob += struct.pack("<I", 4) + struct.pack("<I", int(value))
    path.write_bytes(blob)
    return path


_LLAMA_KVS = {
    "general.architecture": "llama",
    "general.name": "test model",
    "llama.block_count": 32,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
    "llama.embedding_length": 4096,
    "llama.context_length": 131072,
}
# 2 tensors (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes (f16)
_LLAMA_KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2


@pytest.fixture(autouse=True)
def _isolated_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_CALIBRATION_DIR", str(tmp_path / "cal"))
    yield


@pytest.fixture(autouse=True)
def _hermetic_ceiling_probes(monkeypatch):
    """The budget basis probes read the REAL host (sysctl, mlx) — pin them to
    "unknown" so every test computes from the mocked snapshot only; tests that
    exercise the real-ceiling lanes override these explicitly."""
    from abstractcore.utils import context_estimate as est_mod

    monkeypatch.setattr(est_mod, "_metal_wired_limit_bytes", lambda: None)
    monkeypatch.setattr(est_mod, "_metal_recommended_working_set_bytes", lambda: None)
    yield


# Fallback-lane budget for the `_fixed_memory` snapshot below: no wired-limit
# sysctl / Metal working-set info -> basis = 75% of device_total, reserve
# max(2 GiB, 5%), allocated deducted.
_FIXED_BUDGET = int(0.75 * 8 * GIB) - max(2 * GIB, int(0.05 * int(0.75 * 8 * GIB))) - 2 * GIB


@pytest.fixture()
def _fixed_memory(monkeypatch):
    snapshot = {
        "ts": 1.0,
        "ram": {"total_bytes": 32 * GIB, "available_bytes": 16 * GIB, "used_bytes": 16 * GIB, "percent": 50.0},
        "process": {"rss_bytes": 1},
        "device": {"backend": "metal", "allocated_bytes": 2 * GIB, "total_bytes": 8 * GIB, "free_bytes": None},
        "host": {"host_id": "abc", "host_name": "x", "kind": "local"},
    }
    monkeypatch.setattr(memory_mod, "get_memory_snapshot", lambda: snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# GGUF header walker
# ---------------------------------------------------------------------------

def test_read_gguf_geometry_reads_header_keys(tmp_path) -> None:
    path = _write_gguf(tmp_path / "m.gguf", _LLAMA_KVS)

    geometry = read_gguf_geometry(path)

    assert geometry == {
        "architecture": "llama",
        "block_count": 32,
        "head_count": 32,
        "head_count_kv": 8,
        "key_length": 128,
        "value_length": 128,
        "embedding_length": 4096,
        "context_length": 131072,
    }


def test_read_gguf_geometry_returns_none_for_non_gguf(tmp_path) -> None:
    path = tmp_path / "not.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    assert read_gguf_geometry(path) is None
    assert read_gguf_geometry(tmp_path / "missing.gguf") is None


# ---------------------------------------------------------------------------
# Estimator lanes
# ---------------------------------------------------------------------------

def test_estimate_gguf_direct_path(tmp_path, _fixed_memory) -> None:
    path = _write_gguf(tmp_path / "m.gguf", _LLAMA_KVS)

    out = estimate_context_fit("huggingface", str(path), context_length=8192)

    assert out["ok"] is True
    assert out["confidence"] == "estimated"
    assert out["kv_bytes_per_token"] == _LLAMA_KV_BYTES_PER_TOKEN
    assert out["geometry"]["n_layers"] == 32
    assert out["geometry"]["n_kv_heads"] == 8
    assert out["geometry"]["head_dim"] == 128
    assert out["geometry"]["max_position_embeddings"] == 131072
    assert out["requested_context_length"] == 8192
    assert out["est_kv_bytes"] == 8192 * _LLAMA_KV_BYTES_PER_TOKEN
    assert out["est_weights_bytes"] == path.stat().st_size
    # Fallback budget basis (75% of device_total) minus max(2 GiB, 5%) reserve
    # and the allocated bytes; predicted = context that fits BESIDE the
    # weights, clamped to model max.
    assert out["budget_bytes"] == _FIXED_BUDGET
    assert out["fits_weights"] is True
    assert out["fits_requested_context"] is True  # tiny weights + 8192 * kv < budget
    assert out["predicted_max_context"] == min(
        (_FIXED_BUDGET - path.stat().st_size) // _LLAMA_KV_BYTES_PER_TOKEN, 131072
    )
    assert out["memory"] == {
        "ram_available_bytes": 16 * GIB,
        "device_total_bytes": 8 * GIB,
        "device_allocated_bytes": 2 * GIB,
    }
    assert any("f16" in note for note in out["notes"])
    # The notes state basis + reserve so the budget is auditable.
    assert any("budget basis" in note and "reserve" in note for note in out["notes"])
    assert any("beside the weights" in note for note in out["notes"])


def test_estimate_hf_transformers_snapshot_config(tmp_path, _fixed_memory) -> None:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "hidden_size": 2048,
                "max_position_embeddings": 32768,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"x" * 1234)

    out = estimate_context_fit("huggingface", str(snapshot))

    assert out["confidence"] == "estimated"
    assert out["geometry"] == {
        "n_layers": 24,
        "n_kv_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "max_position_embeddings": 32768,
    }
    assert out["kv_bytes_per_token"] == 2 * 24 * 4 * 128 * 2
    assert out["est_weights_bytes"] == 1234
    assert "requested_context_length" not in out
    assert "est_kv_bytes" not in out


def test_estimate_mlx_snapshot_config(tmp_path, _fixed_memory) -> None:
    snapshot = tmp_path / "mlx-snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"num_hidden_layers": 12, "num_attention_heads": 8, "hidden_size": 1024}),
        encoding="utf-8",
    )

    out = estimate_context_fit("mlx", str(snapshot))

    assert out["confidence"] == "estimated"
    # No num_key_value_heads -> falls back to head count; head_dim derived.
    assert out["geometry"]["n_kv_heads"] == 8
    assert out["geometry"]["head_dim"] == 128
    assert out["kv_bytes_per_token"] == 2 * 12 * 8 * 128 * 2
    # No weight files in the snapshot -> weights unknown: fits_weights stays
    # null (never guessed) and the budget is NOT weight-reduced (noted).
    assert out["fits_weights"] is None
    assert out["fits_requested_context"] is None
    # No max_position_embeddings -> no clamp, floor math only.
    assert out["predicted_max_context"] == _FIXED_BUDGET // (2 * 12 * 8 * 128 * 2)
    assert any("weight size unknown" in note for note in out["notes"])


def test_estimate_ollama_api_show(monkeypatch, _fixed_memory) -> None:
    posts = []

    def fake_post(url: str, *, json: Any = None, timeout: Any = None) -> httpx.Response:
        posts.append({"url": url, "json": json})
        return httpx.Response(
            200,
            json={
                "model_info": {
                    "general.architecture": "qwen3",
                    "qwen3.block_count": 40,
                    "qwen3.attention.head_count": 32,
                    "qwen3.attention.head_count_kv": 8,
                    "qwen3.attention.key_length": 128,
                    "qwen3.embedding_length": 4096,
                    "qwen3.context_length": 262144,
                }
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    out = estimate_context_fit("ollama", "qwen3:30b")

    assert posts == [{"url": "http://localhost:11434/api/show", "json": {"model": "qwen3:30b"}}]
    assert out["confidence"] == "estimated"
    assert out["geometry"]["n_layers"] == 40
    assert out["geometry"]["n_kv_heads"] == 8
    assert out["geometry"]["max_position_embeddings"] == 262144
    assert out["kv_bytes_per_token"] == 2 * 40 * 8 * 128 * 2


def test_estimate_ollama_unreachable_is_unknown(monkeypatch, _fixed_memory) -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)

    out = estimate_context_fit("ollama", "qwen3:30b")

    assert out["confidence"] == "unknown"
    assert out["predicted_max_context"] is None
    assert "geometry" not in out


def test_estimate_lmstudio_and_unknown_provider_are_unknown(_fixed_memory) -> None:
    for provider in ("lmstudio", "openai"):
        out = estimate_context_fit(provider, "whatever")
        assert out["ok"] is True
        assert out["confidence"] == "unknown"
        assert out["predicted_max_context"] is None
        assert "kv_bytes_per_token" not in out


def test_calibration_beats_estimation(tmp_path, _fixed_memory) -> None:
    from abstractcore.utils.context_calibration import record_context_calibration

    path = _write_gguf(tmp_path / "m.gguf", _LLAMA_KVS)
    record_context_calibration(
        {
            "provider": "huggingface",
            "model_id": path.name,
            "requested_context": 131072,
            "settled_context": 16384,
            "device_total_bytes": 8 * GIB,
            "ram_total_bytes": 32 * GIB,
            "rungs_tried": [131072, 65536, 32768, 16384],
        }
    )

    out = estimate_context_fit("huggingface", str(path))

    assert out["confidence"] == "calibrated"
    assert out["calibrated_context_length"] == 16384
    assert out["predicted_max_context"] == 16384
    # Geometry is still reported alongside the calibrated verdict.
    assert out["kv_bytes_per_token"] == _LLAMA_KV_BYTES_PER_TOKEN


def test_calibration_hits_hub_model_id_via_resolved_gguf_basename(tmp_path, monkeypatch, _fixed_memory) -> None:
    """The ladder records under the RESOLVED artifact id (gguf basename /
    quant-dir-qualified name) — a hub repo id must still hit that entry."""
    from abstractcore.utils.context_calibration import record_context_calibration

    hub = tmp_path / "hub"
    snap = hub / "models--test-org--test-repo" / "snapshots" / "rev0"
    snap.mkdir(parents=True)
    _write_gguf(snap / "m.gguf", _LLAMA_KVS)
    nested_snap = hub / "models--test-org--test-repo2" / "snapshots" / "rev0" / "UD-Q3_K_XL"
    nested_snap.mkdir(parents=True)
    _write_gguf(nested_snap / "m2.gguf", _LLAMA_KVS)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    base = {
        "provider": "huggingface",
        "requested_context": 131072,
        "device_total_bytes": 8 * GIB,
        "ram_total_bytes": 32 * GIB,
        "rungs_tried": [131072, 32768],
    }
    # Ladder-style ids: bare basename at snapshot root, quant-dir-qualified nested.
    record_context_calibration(dict(base, model_id="m.gguf", settled_context=32768))
    record_context_calibration(dict(base, model_id="UD-Q3_K_XL/m2.gguf", settled_context=8192))

    out = estimate_context_fit("huggingface", "test-org/test-repo")
    assert out["confidence"] == "calibrated"
    assert out["calibrated_context_length"] == 32768

    out_nested = estimate_context_fit("huggingface", "test-org/test-repo2")
    assert out_nested["confidence"] == "calibrated"
    assert out_nested["calibrated_context_length"] == 8192


def test_budget_uses_the_wired_limit_ceiling_for_large_weights(tmp_path, monkeypatch) -> None:
    """Measured on the reference 128 GB M5 Max: `sysctl iogpu.wired_limit_mb`
    = 110000 (115,343,360,000 bytes; mlx max_recommended_working_set_size
    reports the same). A 93 GB quant MUST read fits_weights: true with a
    reduced predicted_max_context — never "doesn't fit"; a 120 GB quant reads
    false. Sparse files supply the sizes; nothing large is ever loaded."""
    from abstractcore.utils import context_estimate as est_mod

    wired_limit = 110000 * 1024 * 1024  # 115,343,360,000
    snapshot = {
        "ts": 1.0,
        "ram": {"total_bytes": 137438953472, "available_bytes": 100 * GIB, "used_bytes": 28 * GIB, "percent": 21.0},
        "process": {"rss_bytes": 1},
        "device": {"backend": "metal", "allocated_bytes": 0, "total_bytes": 137438953472, "free_bytes": None},
        "host": {"host_id": "abc", "host_name": "x", "kind": "local"},
    }
    monkeypatch.setattr(memory_mod, "get_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(est_mod, "_metal_wired_limit_bytes", lambda: wired_limit)

    reserve = max(2 * GIB, int(0.05 * wired_limit))
    budget = wired_limit - reserve  # 109,576,192,000 bytes

    path_93 = _write_gguf(tmp_path / "big93.gguf", _LLAMA_KVS)
    import os

    os.truncate(path_93, 93_000_000_000)  # sparse: size without disk usage
    out = estimate_context_fit("huggingface", str(path_93))
    assert out["budget_bytes"] == budget
    assert out["fits_weights"] is True
    assert out["confidence"] == "estimated"
    assert out["predicted_max_context"] == min((budget - 93_000_000_000) // _LLAMA_KV_BYTES_PER_TOKEN, 131072)
    assert out["predicted_max_context"] > 0  # reduced context, not "doesn't fit"
    assert any("iogpu.wired_limit_mb" in note for note in out["notes"])

    path_120 = _write_gguf(tmp_path / "big120.gguf", _LLAMA_KVS)
    os.truncate(path_120, 120_000_000_000)
    out_120 = estimate_context_fit("huggingface", str(path_120))
    assert out_120["fits_weights"] is False
    assert out_120["predicted_max_context"] is None
    assert any("no context fits beside the weights" in note for note in out_120["notes"])


def test_already_resident_weights_are_not_double_counted(tmp_path, monkeypatch) -> None:
    """Review probe: a 93 GB model ALREADY loaded in this process shows up in
    `device_allocated_bytes` (93 GB), which the budget already deducts —
    subtracting est_weights again reported the RUNNING model as not fitting.
    When the allocation covers the weights and is mostly them, the weights
    count once: fits_weights true, predicted = budget // kv."""
    import os

    from abstractcore.utils import context_estimate as est_mod

    wired_limit = 110000 * 1024 * 1024  # basis 115,343,360,000
    snapshot = {
        "ts": 1.0,
        "ram": {"total_bytes": 137438953472, "available_bytes": 30 * GIB, "used_bytes": 98 * GIB, "percent": 77.0},
        "process": {"rss_bytes": 1},
        "device": {"backend": "metal", "allocated_bytes": 93_000_000_000, "total_bytes": 137438953472, "free_bytes": None},
        "host": {"host_id": "abc", "host_name": "x", "kind": "local"},
    }
    monkeypatch.setattr(memory_mod, "get_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(est_mod, "_metal_wired_limit_bytes", lambda: wired_limit)

    path = _write_gguf(tmp_path / "big93.gguf", _LLAMA_KVS)
    os.truncate(path, 93_000_000_000)
    out = estimate_context_fit("huggingface", str(path), context_length=8192)

    reserve = max(2 * GIB, int(0.05 * wired_limit))
    budget = wired_limit - reserve - 93_000_000_000  # 16,576,192,000
    assert out["budget_bytes"] == budget
    assert out["fits_weights"] is True, "a RUNNING model must never read as not fitting"
    # KV for the requested context rides the remaining budget alone.
    assert out["fits_requested_context"] is (8192 * _LLAMA_KV_BYTES_PER_TOKEN <= budget)
    assert out["predicted_max_context"] == min(budget // _LLAMA_KV_BYTES_PER_TOKEN, 131072)
    assert any("already resident" in note and "counted once" in note for note in out["notes"])

    # A SMALL model beside a big unrelated allocation is NOT "resident": the
    # allocation is not mostly its weights, so its weights still deduct.
    small = _write_gguf(tmp_path / "small.gguf", _LLAMA_KVS)
    os.truncate(small, 5_000_000_000)
    out_small = estimate_context_fit("huggingface", str(small))
    assert not any("already resident" in note for note in out_small["notes"])
    assert out_small["fits_weights"] is (5_000_000_000 <= budget)
    assert out_small["predicted_max_context"] == min(
        (budget - 5_000_000_000) // _LLAMA_KV_BYTES_PER_TOKEN, 131072
    )


def test_budget_falls_back_to_metal_recommended_working_set(tmp_path, monkeypatch) -> None:
    """No wired-limit sysctl -> Metal's own max_recommended_working_set_size
    is the basis (the value mlx device_info reports)."""
    from abstractcore.utils import context_estimate as est_mod

    recommended = 115343360000
    snapshot = {
        "ts": 1.0,
        "ram": {"total_bytes": 137438953472, "available_bytes": 100 * GIB, "used_bytes": 28 * GIB, "percent": 21.0},
        "process": {"rss_bytes": 1},
        "device": {"backend": "metal", "allocated_bytes": 3 * GIB, "total_bytes": 137438953472, "free_bytes": None},
        "host": {"host_id": "abc", "host_name": "x", "kind": "local"},
    }
    monkeypatch.setattr(memory_mod, "get_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(est_mod, "_metal_wired_limit_bytes", lambda: None)
    monkeypatch.setattr(est_mod, "_metal_recommended_working_set_bytes", lambda: recommended)

    path = _write_gguf(tmp_path / "m.gguf", _LLAMA_KVS)
    out = estimate_context_fit("huggingface", str(path))

    reserve = max(2 * GIB, int(0.05 * recommended))
    assert out["budget_bytes"] == recommended - reserve - 3 * GIB
    assert any("max_recommended_working_set_size" in note for note in out["notes"])


def test_budget_cuda_uses_mem_get_info_free_bytes(tmp_path, monkeypatch) -> None:
    """CUDA lane: mem_get_info free bytes (already excludes allocations) minus
    the same small reserve — allocated is NOT deducted twice."""
    snapshot = {
        "ts": 1.0,
        "ram": {"total_bytes": 64 * GIB, "available_bytes": 32 * GIB, "used_bytes": 32 * GIB, "percent": 50.0},
        "process": {"rss_bytes": 1},
        "device": {"backend": "cuda", "allocated_bytes": 6 * GIB, "total_bytes": 24 * GIB, "free_bytes": 18 * GIB},
        "host": {"host_id": "abc", "host_name": "x", "kind": "local"},
    }
    monkeypatch.setattr(memory_mod, "get_memory_snapshot", lambda: snapshot)

    path = _write_gguf(tmp_path / "m.gguf", _LLAMA_KVS)
    out = estimate_context_fit("huggingface", str(path))

    reserve = max(2 * GIB, int(0.05 * 18 * GIB))
    assert out["budget_bytes"] == 18 * GIB - reserve
    assert out["fits_weights"] is True
    assert any("cuda mem_get_info free bytes" in note for note in out["notes"])


def test_estimate_is_advisory_only_no_load_path_references_it() -> None:
    """The estimator is a HINT: no load/unload path in core gates on it. The
    only server call site is the read-only /acore/models/context_estimate
    endpoint; the provider load ladders never import it."""
    import inspect

    import abstractcore.server.app as server_app

    assert "estimate_context_fit" not in inspect.getsource(server_app.acore_models_load)
    assert "estimate_context_fit" not in inspect.getsource(server_app.acore_models_unload)
    assert "estimate_context_fit" not in inspect.getsource(server_app._lock_gateway_runtime)
    assert "estimate_context_fit" in inspect.getsource(server_app.acore_models_context_estimate)

    from abstractcore.providers import huggingface_provider, mlx_provider

    assert "estimate_context_fit" not in inspect.getsource(huggingface_provider)
    assert "estimate_context_fit" not in inspect.getsource(mlx_provider)


def test_estimator_never_raises(monkeypatch) -> None:
    out = estimate_context_fit(None, None)  # type: ignore[arg-type]
    assert out["ok"] is True
    assert out["confidence"] == "unknown"

    # Even an internal failure surfaces as an unknown verdict, not an exception.
    from abstractcore.utils import context_estimate as est_mod

    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("internal explosion")

    monkeypatch.setattr(est_mod, "_estimate_context_fit", _boom)
    out2 = estimate_context_fit("huggingface", "m")
    assert out2["ok"] is True
    assert out2["confidence"] == "unknown"
    assert any("internal explosion" in note for note in out2["notes"])
    # The fallback's memory block carries the SAME key set as the normal path.
    assert set(out2["memory"].keys()) == {
        "ram_available_bytes",
        "device_total_bytes",
        "device_allocated_bytes",
    }
