"""Backlog 0947: the fit estimate knows the Mac's GPU wired limit.

On a 128 GiB Mac the recommended text model (Qwen3.8 Flash-Next 4-bit, ~109 GiB
with its cache) is larger than Metal's default working set (~102 GiB usable),
yet it is the tier and the operator runs it -- after raising
`iogpu.wired_limit_mb`. The contract:
  - `iogpu.wired_limit_mb` > 0 IS the ceiling (full AND light host reading),
    and the fit notes say so;
  - with the default limit, a model that fits under a limit macOS can grant
    (RAM - 8 GiB) is `needs_gpu_limit` with the exact command and value, never
    a bare `too_large` next to a recommendation;
  - running that command makes the verdict what `verdict_with_limit` promised.
"""

from __future__ import annotations

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.utils import host_profile as hp
from abstractcore.utils import memory
from abstractcore.utils.model_fit import GPU_LIMIT_OS_HEADROOM_BYTES, estimate_fit
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
    usable = lambda c: c - max(2 * GIB, int(0.05 * c))  # noqa: E731 (estimate_fit's reserve)
    assert usable(gl["required_mb"] * MIB) >= need > usable((gl["required_mb"] - 1024) * MIB)
    assert gl["required_mb"] * MIB <= 128 * GIB - GPU_LIMIT_OS_HEADROOM_BYTES
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
    # 27B 4-bit on a 24 GiB Mac: the limit it would need exceeds RAM - 8 GiB.
    row, art = mc._seed_row_and_artifact("qwen3.8-27b", "mlx", "mlx-community/Qwen3.8-27B-4bit")
    fit = mc._fit_for_seed_artifact(row, art, synthetic_host("metal24"))
    assert fit["verdict"] == "too_large" and "gpu_limit" not in fit


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
