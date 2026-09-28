"""Backlog 0947: the fit estimate knows the Mac's GPU wired limit.

On a 128 GiB Mac the recommended text model (Qwen3.8 Flash-Next 4-bit, ~109 GiB
with its cache) is larger than Metal's default working set (~102 GiB usable),
yet it is the tier and the operator runs it -- after raising
`iogpu.wired_limit_mb`. The contract:
  - `iogpu.wired_limit_mb` > 0 IS the ceiling (full AND light host reading),
    and the fit notes say so;
  - with the default limit, a model that fits under a limit macOS can grant
    (RAM - max(4 GiB, 12.5% of RAM), operator ruling 2026-09-28) is
    `needs_gpu_limit` with the exact command and value, never a bare
    `too_large` next to a recommendation;
  - running that command makes the verdict what `verdict_with_limit` promised.

Operator measurement 2026-09-28 (Mac mini, 24 GB; the operator's "GB" are
GiB here: 20480 MB is "20 GB", 21504 MB "21 GB"): macOS's default GPU limit is
~17.8 GB (~75% of RAM); Qwen3.8 27B 4-bit runs out of the box with a small
context; `sudo sysctl iogpu.wired_limit_mb=20480` is safe and gives ~30k
tokens; beyond ~21 GB (40k tokens) apps may crash; a 32 GB Mac reaches ~120k.
"""

from __future__ import annotations

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.utils import host_profile as hp
from abstractcore.utils import memory
from abstractcore.utils.model_fit import METAL_WORKING_RESERVE_BYTES, estimate_fit, gpu_limit_max_bytes
from tests.models_engines_fakes import isolate_host, synthetic_host

GIB = 1024**3
MIB = 1024**2
DEFAULT_128 = dict(synthetic_host("metal128"), ceiling_bytes=int(107.52 * GIB), ceiling_source="metal_recommended")


def _raised(host: dict, mb: int) -> dict:
    return dict(host, ceiling_bytes=mb * MIB, ceiling_source="metal_wired_limit")


def test_default_limit_says_needs_gpu_limit_with_the_exact_command():
    pick = mc.recommended_text_model(DEFAULT_128, mtp=False)
    fit = pick["fit"]
    assert fit["verdict"] == "needs_gpu_limit"
    gl = fit["gpu_limit"]
    assert gl["sysctl"] == "iogpu.wired_limit_mb" and gl["current_mb"] == 0
    assert gl["command"] == f"sudo sysctl iogpu.wired_limit_mb={gl['required_mb']}"
    assert gl["required_mb"] % 1024 == 0, "a whole number of GiB"
    assert gl["needs_admin"] is True and gl["resets_at_restart"] is True
    # The smallest whole GiB that fits: one GiB less would not.
    need = fit["need_bytes"]
    usable = lambda c: c - METAL_WORKING_RESERVE_BYTES  # noqa: E731 (estimate_fit's Metal reserve)
    assert usable(gl["required_mb"] * MIB) >= need > usable((gl["required_mb"] - 1024) * MIB)
    assert gl["required_mb"] * MIB <= gpu_limit_max_bytes(128 * GIB)
    # Recomputed with the ruling's reserve: 112 GiB, no longer 117760 MB.
    assert gl["command"] == "sudo sysctl iogpu.wired_limit_mb=114688"
    assert pick["fits"] is False and gl["command"] in pick["warning"]
    assert "asks for your password; lasts until the Mac restarts" in pick["warning"]


def test_running_the_command_gives_the_promised_verdict_with_that_basis():
    gl = mc.recommended_text_model(DEFAULT_128, mtp=False)["fit"]["gpu_limit"]
    after = mc.recommended_text_model(_raised(DEFAULT_128, gl["required_mb"]), mtp=False)
    assert after["fit"]["verdict"] == gl["verdict_with_limit"] == "tight"
    assert after["fits"] is True and "gpu_limit" not in after["fit"]
    assert after["fit"]["ceiling_source"] == "metal_wired_limit"
    assert any(f"iogpu.wired_limit_mb = {gl['required_mb']} MB" in n for n in after["fit"]["notes"])


def test_a_generous_raised_limit_fits_outright():
    fit = mc.recommended_text_model(_raised(DEFAULT_128, 124 * 1024), mtp=False)["fit"]
    assert fit["verdict"] in ("fits", "tight") and fit["ceiling_source"] == "metal_wired_limit"


def test_a_raised_limit_that_is_still_too_small_says_how_much_more():
    fit = mc.recommended_text_model(_raised(DEFAULT_128, 100 * 1024), mtp=False)["fit"]
    assert fit["verdict"] == "needs_gpu_limit" and fit["gpu_limit"]["current_mb"] == 100 * 1024


def test_beyond_what_macos_can_grant_stays_too_large():
    # Flash-Next 4-bit on a 64 GiB Mac: the limit it would need exceeds
    # RAM - max(4 GiB, 12.5%) = 56 GiB.
    row, art = mc._seed_row_and_artifact("qwen3.8-flash-next", "mlx", "mlx-community/Qwen3.8-Flash-Next-4bit")
    fit = mc._fit_for_seed_artifact(row, art, synthetic_host("metal64"))
    assert fit["verdict"] == "too_large" and "gpu_limit" not in fit


# ---------------------------------------------------------------------------
# The operator's measurement (Mac mini, 24 GB, 2026-09-28)
# ---------------------------------------------------------------------------


def test_the_default_limit_is_75_percent_of_ram_as_measured_on_a_24gb_mac():
    """macOS's default GPU limit on a 24 GB Mac: ~17.8 GB measured; the model's
    fallback (no Metal reading) is 75% of RAM = 18 GiB. The fit sentence says
    that limit, never only what is left after the working buffers."""

    mac24 = synthetic_host("metal24")
    assert abs(mac24["ceiling_bytes"] / GIB - 17.8) <= 0.25
    row, art = mc._seed_row_and_artifact("qwen3.8-27b", "mlx", "mlx-community/Qwen3.8-27B-4bit")
    fit = mc._fit_for_seed_artifact(row, art, mac24)
    assert fit["ceiling_bytes"] == 18 * GIB and fit["usable_bytes"] == 16 * GIB
    warning = mc._fit_warning(row, fit)
    assert "macOS's GPU memory limit on this Mac is 18.0 GiB" in warning
    assert "can give a model about 16" not in warning


@pytest.mark.parametrize("gib,mb", [(24, 20480), (32, 28672), (64, 57344), (128, 114688), (8, 4096), (16, 12288)])
def test_the_raised_limit_keeps_max_4gib_or_12_5_percent_for_macos(gib, mb):
    assert gpu_limit_max_bytes(gib * GIB) == mb * MIB


def test_27b_on_a_24gb_mac_runs_with_a_small_context_and_says_the_command_for_30k():
    row, art = mc._seed_row_and_artifact("qwen3.8-27b", "mlx", "mlx-community/Qwen3.8-27B-4bit")
    fit = mc._fit_for_seed_artifact(row, art, synthetic_host("metal24"))
    assert fit["verdict"] == "tight" and fit["small_context"] is True
    assert 0 < fit["max_context"] < 8192, "runs out of the box, with a small context"
    raised = fit["raised_limit"]
    assert raised["command"] == "sudo sysctl iogpu.wired_limit_mb=20480"
    assert 25_000 <= raised["max_context"] <= 40_000, "the measurement: ~30k tokens at 20480 MB"
    assert "gpu_limit" not in fit, "it runs without the command: not needs_gpu_limit"
    after = mc._fit_for_seed_artifact(row, art, _raised(synthetic_host("metal24"), 20480))
    assert after["verdict"] in ("fits", "tight") and after["small_context"] is False
    assert "raised_limit" not in after, "already at the highest safe limit"


def test_27b_on_a_32gb_mac_fits_with_a_long_context():
    row, art = mc._seed_row_and_artifact("qwen3.8-27b", "mlx", "mlx-community/Qwen3.8-27B-4bit")
    fit = mc._fit_for_seed_artifact(row, art, synthetic_host("metal32"))
    assert fit["verdict"] == "fits" and fit["max_context"] >= 90_000  # measured ~120k


def test_a_model_whose_weights_do_not_fit_is_not_small_context():
    fit = estimate_fit(host=synthetic_host("metal8"), weight_bytes=5 * GIB, geometry={"n_layers": 8, "n_kv_heads": 4, "head_dim": 128})
    assert fit["verdict"] == "too_large" and fit["small_context"] is False


def test_only_apple_silicon_has_a_gpu_limit_to_raise():
    host = dict(synthetic_host("cuda24"), ceiling_bytes=24 * GIB)
    fit = estimate_fit(host=host, weight_bytes=30 * GIB, context=1)
    assert fit["verdict"] in ("too_large", "partial_offload") and "gpu_limit" not in fit


def test_the_fits_filter_keeps_a_model_that_needs_the_limit(tmp_path, monkeypatch):
    isolate_host(tmp_path, monkeypatch)
    payload = mc.catalog(host=DEFAULT_128, fits=True)
    arts = [a for r in payload["rows"] for a in r["artifacts"]]
    assert any(a["artifact"] == "mlx-community/Qwen3.8-Flash-Next-4bit" for a in arts)
    assert "needs_gpu_limit" in mc.FITS_FILTER_VERDICTS


def test_a_fit_gated_route_that_needs_the_limit_says_the_command(monkeypatch):
    # The fit gate (output.video) never writes a route that needs an admin
    # command first -- and its reason carries that command, not "too large".
    def fake(provider, artifact, host):
        return {"row": {"display_name": "Wan"}, "fit": {
            "verdict": "needs_gpu_limit", "need_bytes": 60 * GIB, "usable_bytes": 44 * GIB,
            "gpu_limit": {"required_mb": 65 * 1024, "command": "sudo sysctl iogpu.wired_limit_mb=66560"}}}

    monkeypatch.setattr(mc, "recommended_artifact_fit", fake)
    reason = cd._fit_gate_reason("output.video", {"provider": "mlx-gen", "artifact": "x"}, synthetic_host("metal64"))
    assert "sudo sysctl iogpu.wired_limit_mb=66560" in reason
    assert "more unified memory" not in reason


# ---------------------------------------------------------------------------
# The host reading: the sysctl IS the ceiling, light probe included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wired", [None, 120 * 1024 * MIB])
def test_the_light_reading_uses_a_raised_limit(monkeypatch, wired):
    monkeypatch.setattr(hp, "normalize_os", lambda *_a: "darwin")
    monkeypatch.setattr(hp, "normalize_arch", lambda *_a: "arm64")
    monkeypatch.setattr(hp, "_ram_total_and_available", lambda: (128 * GIB, 64 * GIB))
    monkeypatch.setattr(memory, "metal_wired_limit_bytes", lambda: wired)
    light = hp._build_light_profile()
    if wired:
        assert (light["ceiling_bytes"], light["ceiling_source"]) == (wired, "metal_wired_limit")
    else:
        assert "ceiling_bytes" not in light, "no limit set: the fit keeps its 75%-of-RAM basis"


def test_the_full_reading_prefers_the_raised_limit(monkeypatch):
    monkeypatch.setattr(memory, "metal_wired_limit_bytes", lambda: 110 * GIB)
    monkeypatch.setattr(memory, "metal_recommended_working_set_bytes", lambda: 100 * GIB)
    assert hp._metal_ceiling() == (110 * GIB, "metal_wired_limit")
    monkeypatch.setattr(memory, "metal_wired_limit_bytes", lambda: None)
    assert hp._metal_ceiling() == (100 * GIB, "metal_recommended")


def test_the_sysctl_reader_treats_zero_as_unset(monkeypatch):
    import subprocess

    class P:
        def __init__(self, out):
            self.returncode, self.stdout = 0, out

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: P("0\n"))
    assert memory.metal_wired_limit_bytes() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: P("117760\n"))
    assert memory.metal_wired_limit_bytes() == 117760 * MIB
