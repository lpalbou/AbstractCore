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

Operator measurement 2026-09-28 (Mac mini, 24 GB): macOS reported a GPU limit
of 17.8 GB (17.8e9 bytes = 16.6 GiB, ~69% of RAM; AbstractCore's fallback when
it cannot read the limit is 75% = 18 GiB); Qwen3.8 27B 4-bit (16.2 GB of
weights) runs out of the box with a small context;
`sudo sysctl iogpu.wired_limit_mb=20480` is safe and gives ~30k tokens (also
measured); beyond 21504 MB apps may crash. Every real 24 GB reading (17.8e9
bytes, 16 GiB, 17.8 GiB) and the fallback must read "runs, small context",
never needs_gpu_limit, and the only command ever printed is the ruling's one
safe value (20480 on 24 GB, 114688 on 128 GB).
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
DEFAULT_128 = synthetic_host("metal128")  # 75% fallback: 96 GiB
# The operator's M5 Max probe (2026-09-24): Metal's recommended working set 107.52 GiB.
M5MAX_128 = dict(synthetic_host("metal128"), ceiling_bytes=int(107.52 * GIB), ceiling_source="metal_recommended")


def _raised(host: dict, mb: int) -> dict:
    return dict(host, ceiling_bytes=mb * MIB, ceiling_source="metal_wired_limit")


def test_default_limit_says_needs_gpu_limit_with_the_exact_command():
    pick = mc.recommended_text_model(DEFAULT_128, mtp=False)
    fit = pick["fit"]
    assert fit["verdict"] == "needs_gpu_limit"
    gl = fit["gpu_limit"]
    assert gl["sysctl"] == "iogpu.wired_limit_mb" and gl["current_mb"] == 0
    assert gl["command"] == f"sudo sysctl iogpu.wired_limit_mb={gl['required_mb']}"
    assert gl["needs_admin"] is True and gl["resets_at_restart"] is True
    # ONE value: the highest safe limit (RAM - max(4 GiB, 12.5%)), never a
    # second "just enough" number; the need fits under it.
    assert gl["required_mb"] * MIB == gpu_limit_max_bytes(128 * GIB)
    assert fit["need_bytes"] <= gl["required_mb"] * MIB - METAL_WORKING_RESERVE_BYTES
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


MAC24_READINGS = {
    "measured 17.8 GB": int(17.8e9),  # 16.6 GiB, the operator's Mac mini
    "16 GiB": 16 * GIB,
    "17.8 GiB": int(17.8 * GIB),
    "fallback 75%": None,  # 18 GiB
}


def _27b(host: dict) -> tuple:
    row, art = mc._seed_row_and_artifact("qwen3.8-27b", "mlx", "mlx-community/Qwen3.8-27B-4bit")
    fit = mc._fit_for_seed_artifact(row, art, host)
    return row, fit


def test_the_measured_24gb_limit_is_16_6_gib_not_75_percent():
    """17.8 GB is 17.8e9 bytes = 16.6 GiB, ~69% of a 24 GB Mac's RAM; the
    fallback without a reading is 75% (18 GiB). Keep the units apart."""

    assert round(17.8e9 / GIB, 1) == 16.6 and round(17.8e9 / (24 * GIB), 2) == 0.69
    assert synthetic_host("metal24")["ceiling_bytes"] == 18 * GIB


@pytest.mark.parametrize("reading", list(MAC24_READINGS), ids=list(MAC24_READINGS))
def test_every_24gb_reading_runs_the_27b_with_a_small_context_and_one_command(reading):
    host = synthetic_host("metal24")
    if MAC24_READINGS[reading] is not None:
        host = dict(host, ceiling_bytes=MAC24_READINGS[reading], ceiling_source="metal_recommended")
    row, fit = _27b(host)
    assert fit["verdict"] == "tight" and fit["small_context"] is True, "runs out of the box (the measurement)"
    assert "gpu_limit" not in fit, "never needs_gpu_limit on a 24 GB Mac"
    assert fit["raised_limit"]["command"] == "sudo sysctl iogpu.wired_limit_mb=20480"
    assert fit["max_context"] is None and "max_context" not in fit["raised_limit"], "no estimated token count"
    warning = mc._fit_warning(row, fit)
    assert "Tight: it runs with a small context by default; close other apps first." in warning
    assert "about 30k tokens, measured on a 24 GB Mac mini" in warning
    assert "sudo sysctl iogpu.wired_limit_mb=20480" in warning and "19456" not in warning
    assert "can give a model about" not in warning


@pytest.mark.parametrize("gib,mb", [(24, 20480), (32, 28672), (64, 57344), (128, 114688), (8, 4096), (16, 12288)])
def test_the_raised_limit_keeps_max_4gib_or_12_5_percent_for_macos(gib, mb):
    assert gpu_limit_max_bytes(gib * GIB) == mb * MIB


def test_at_20480_the_27b_fits_its_context_and_no_second_command_is_offered():
    _row, after = _27b(_raised(synthetic_host("metal24"), 20480))
    assert after["verdict"] in ("fits", "tight") and after["small_context"] is False
    assert "raised_limit" not in after and "gpu_limit" not in after, "already at the highest safe limit"


def test_27b_on_a_32gb_mac_fits():
    _row, fit = _27b(synthetic_host("metal32"))
    assert fit["verdict"] == "fits" and "raised_limit" not in fit


def test_a_model_whose_weights_do_not_fit_is_not_small_context():
    fit = estimate_fit(host=synthetic_host("metal8"), weight_bytes=int(6.5 * GIB), geometry={"n_layers": 8, "n_kv_heads": 4, "head_dim": 128})
    assert fit["verdict"] == "too_large" and fit["small_context"] is False


def test_the_8gb_mac_9b_is_tight_never_the_safe_choice_sentence():
    """Operator ruling 2026-09-28: the 8 GB Mac keeps Qwen3.5 9B, "tight: runs
    with a small context; close other apps first"."""
    pick = mc.recommended_text_model(synthetic_host("metal8"), mtp=False)
    assert pick["artifact"] == "mlx-community/Qwen3.5-9B-MLX-4bit"
    assert pick["fit"]["verdict"] == "tight" and pick["fit"]["small_context"] is True
    assert "Tight: it runs with a small context by default; close other apps first." in pick["warning"]
    assert "safe choice" not in pick["warning"] and "sysctl" not in pick["warning"]


def test_the_m5_max_reading_runs_flash_next_with_a_small_context():
    """At Metal's 107.52 GiB working set the Flash-Next weights (103.9 GiB)
    fit the limit: it runs with a small context, and the one safe value gives
    it more."""
    fit = mc.recommended_text_model(M5MAX_128, mtp=False)["fit"]
    assert fit["verdict"] == "tight" and fit["small_context"] is True
    assert fit["raised_limit"]["command"] == "sudo sysctl iogpu.wired_limit_mb=114688"


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
