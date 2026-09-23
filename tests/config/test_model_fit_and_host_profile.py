"""Contract A (host profile) and the fit estimator (explore-models section 7).

The three reference machines are synthetic `host_profile_v1` dicts, so the
verdicts are the same on every CI runner: an Apple M-series with 128 GB of
unified memory, an RTX 4090 box (24 GB VRAM + 64 GB RAM), and a 16 GB CPU box.
"""

from __future__ import annotations

import pytest

from abstractcore.utils import host_profile as hp
from abstractcore.utils import model_fit as mf
from tests.models_engines_fakes import isolate_host, synthetic_host

GiB = 1024**3


# ---------------------------------------------------------------------------
# quant -> bits, parameter parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,bits",
    [
        ("Q4_K_M", 4.85),
        ("q4_k_m", 4.85),
        ("4bit", 4.5),
        ("4-bit", 4.5),
        ("8bit", 8.5),
        ("q8_0", 8.5),
        ("Q5_K_M", 5.7),
        ("q6_k", 6.6),
        ("bf16", 16.0),
        ("F16", 16.0),
        ("fp8", 8.0),
        ("MXFP4", 4.25),
        ("UD-Q4_K_XL", 4.95),
        ("iq4_xs", 4.25),
    ],
)
def test_quant_labels_map_to_effective_bits(label, bits):
    assert mf.bits_for_quant(label) == pytest.approx(bits)


def test_unknown_quant_is_none_not_a_guess():
    assert mf.bits_for_quant("banana") is None
    assert mf.bits_for_quant(None) is None


@pytest.mark.parametrize(
    "name,total,active",
    [
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", 30e9, 3e9),
        ("qwen3:8b", 8e9, None),
        ("mlx-community/Qwen3.5-9B-4bit", 9e9, None),
        ("Qwen/Qwen3-Embedding-0.6B", 0.6e9, None),
        ("mlx-community/Llama-3.2-1B-Instruct-4bit", 1e9, None),
        ("mistralai/Mixtral-8x7B-Instruct", 56e9, None),
        ("qwen/qwen3.6-35b-a3b", 35e9, 3e9),
        ("all-MiniLM-L6-v2", None, None),
        # `4bit` is a quant, `e4b` is Gemma's effective size: neither is a count.
        ("gemma3n:e4b", None, None),
    ],
)
def test_parameter_counts_are_read_off_names(name, total, active):
    got_total, got_active = mf.parse_params_from_name(name)
    assert got_total == (int(total) if total else None)
    assert got_active == (int(active) if active else None)


def test_param_count_strings():
    assert mf.parse_param_count("8.2B") == 8_200_000_000
    assert mf.parse_param_count("22M") == 22_000_000
    assert mf.parse_param_count("35B-A3B") == 35_000_000_000
    assert mf.parse_param_count("") is None


# ---------------------------------------------------------------------------
# estimate_fit on the three reference machines
# ---------------------------------------------------------------------------


def test_8b_q4_fits_on_gpus_and_is_tight_on_a_16gb_cpu_box():
    expected = {"metal128": "fits", "cuda24": "fits", "cpu16": "tight"}
    for kind, verdict in expected.items():
        fit = mf.estimate_fit(host=synthetic_host(kind), params_total=8_200_000_000, quant="q4_k_m", max_tokens=131072)
        # cpu16: 5.1 GB weights + 4.4 GB rough KV (8k ctx) + 0.5 GiB > 0.8 x 10 GiB usable.
        assert fit["verdict"] == verdict, (kind, fit)
        assert fit["confidence"] == "rough"  # no geometry: KV by the rough rule
        assert fit["need_bytes"] > fit["weight_bytes"]
        assert fit["max_context"] and fit["max_context"] <= 131072


def test_70b_q4_is_tight_on_metal_partial_on_cuda_too_large_on_cpu():
    kwargs = dict(params_total=70_600_000_000, quant="q4_k_m", max_tokens=128000)
    metal = mf.estimate_fit(host=synthetic_host("metal128"), **kwargs)
    cuda = mf.estimate_fit(host=synthetic_host("cuda24"), **kwargs)
    cpu = mf.estimate_fit(host=synthetic_host("cpu16"), **kwargs)
    assert metal["verdict"] in ("fits", "tight")
    assert cuda["verdict"] == "partial_offload"
    assert cpu["verdict"] == "too_large"


def test_huge_moe_is_too_large_even_for_cuda_offload():
    fit = mf.estimate_fit(host=synthetic_host("cuda24"), params_total=235_000_000_000, quant="q4_k_m")
    assert fit["verdict"] == "too_large"


def test_exact_size_and_geometry_give_exact_confidence_and_real_kv():
    geometry = {"n_layers": 36, "n_kv_heads": 8, "head_dim": 128}
    fit = mf.estimate_fit(
        host=synthetic_host("metal128"),
        weight_bytes=5_000_000_000,
        geometry=geometry,
        context=8192,
        max_tokens=40960,
    )
    assert fit["confidence"] == "exact"
    assert fit["kv_bytes"] == 8192 * 2 * 36 * 8 * 128 * 2
    assert fit["max_context"] == 40960  # clamped to the model window


def test_fits_now_uses_free_memory_and_says_why():
    host = synthetic_host("metal128")
    host["free_now_bytes"] = 4 * GiB
    fit = mf.estimate_fit(host=host, params_total=8_200_000_000, quant="4bit")
    assert fit["verdict"] == "fits"
    assert fit["fits_now"] is False
    assert any("unload" in n for n in fit["notes"])


def test_disk_check_needs_five_gib_headroom():
    host = synthetic_host("metal128", disk_free=10 * 10**9)
    ok = mf.estimate_fit(host=host, weight_bytes=4 * 10**9, disk_free_bytes=10 * 10**9)
    bad = mf.estimate_fit(host=host, weight_bytes=6 * 10**9, disk_free_bytes=10 * 10**9)
    assert ok["disk_ok"] is True
    assert bad["disk_ok"] is False


def test_missing_inputs_are_unknown_never_a_guess():
    assert mf.estimate_fit(host=synthetic_host("metal128"))["verdict"] == "unknown"
    no_ceiling = synthetic_host("cpu16")
    no_ceiling["ceiling_bytes"] = None
    fit = mf.estimate_fit(host=no_ceiling, params_total=8e9, quant="q4_k_m")
    assert fit["verdict"] == "unknown"
    assert fit["weight_bytes"] is not None  # what IS known is still reported


def test_name_derived_params_are_labelled_rough():
    fit = mf.estimate_fit(host=synthetic_host("metal128"), params_total=8e9, params_source="name", quant="q4_k_m",
                          geometry={"n_layers": 36, "n_kv_heads": 8, "head_dim": 128})
    assert fit["confidence"] == "rough"


# ---------------------------------------------------------------------------
# host_profile (contract A)
# ---------------------------------------------------------------------------

_CONTRACT_A_KEYS = {
    "schema", "os", "arch", "accelerator", "gpu_name", "unified_memory", "ram_bytes", "vram_bytes",
    "ceiling_bytes", "ceiling_source", "free_now_bytes", "disk", "python", "generated_at",
}


def test_host_profile_has_every_contract_field_and_never_raises(tmp_path, monkeypatch):
    stores = isolate_host(tmp_path, monkeypatch)
    profile = hp.host_profile(refresh=True)
    assert _CONTRACT_A_KEYS <= set(profile)
    assert profile["schema"] == "host_profile_v1"
    assert profile["os"] in {"darwin", "linux", "windows"}
    assert profile["arch"] in {"arm64", "x86_64"}
    assert profile["accelerator"] in {"metal", "cuda", "rocm", "none"}
    assert profile["ceiling_source"] in {"metal_wired_limit", "metal_recommended", "cuda_total", "ram_75pct", None}
    assert set(profile["disk"]) == {"hf_cache", "lmstudio", "ollama"}
    # The stores resolve to the isolated tmp dirs, and free space is measured.
    assert profile["disk"]["hf_cache"]["abs_path"] == str(stores["hf"])
    assert profile["disk"]["ollama"]["abs_path"] == str(stores["ollama_models"])
    assert all(isinstance(v["free_bytes"], int) for v in profile["disk"].values())
    assert profile["generated_at"].endswith("Z")


def test_host_profile_ceiling_falls_back_to_75_percent_of_ram(monkeypatch):
    monkeypatch.setattr(hp, "normalize_os", lambda value=None: "linux")
    monkeypatch.setattr(hp, "normalize_arch", lambda value=None: "x86_64")
    monkeypatch.setattr(hp, "_nvidia_gpus", lambda: [])
    monkeypatch.setattr(hp, "_torch_cuda", lambda: None)
    monkeypatch.setattr(hp, "_rocm_present", lambda: False)
    monkeypatch.setattr(hp, "_ram_total_and_available", lambda: (16 * GiB, 9 * GiB))
    profile = hp.host_profile(refresh=True)
    assert profile["accelerator"] == "none"
    assert profile["ceiling_bytes"] == int(0.75 * 16 * GiB)
    assert profile["ceiling_source"] == "ram_75pct"
    assert profile["free_now_bytes"] == 9 * GiB


def test_host_profile_reads_cuda_from_nvidia_smi(monkeypatch):
    monkeypatch.setattr(hp, "normalize_os", lambda value=None: "linux")
    monkeypatch.setattr(hp, "normalize_arch", lambda value=None: "x86_64")
    monkeypatch.setattr(
        hp, "_nvidia_gpus", lambda: [{"name": "NVIDIA RTX 4090", "total_bytes": 24 * GiB, "free_bytes": 23 * GiB}]
    )
    monkeypatch.setattr(hp, "_ram_total_and_available", lambda: (64 * GiB, 40 * GiB))
    profile = hp.host_profile(refresh=True)
    assert profile["accelerator"] == "cuda"
    assert profile["gpu_name"] == "NVIDIA RTX 4090"
    assert profile["vram_bytes"] == 24 * GiB
    assert profile["ceiling_source"] == "cuda_total"
    assert profile["free_now_bytes"] == 23 * GiB
    assert profile["unified_memory"] is False
