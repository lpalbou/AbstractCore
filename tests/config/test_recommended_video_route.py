"""The VIDEO category: curated Wan2.2 rows and a host-aware `output.video` row.

Operator request (2026-09-27): the installer must propose a video generation
model where the computer can run one, and say why not elsewhere.

What AbstractFramework can really run (checked in code, 2026-09-27):
  - AbstractVision serves video ONLY through MLX-Gen (Apple silicon): its
    registry marks Wan2.2 TI2V-5B / T2V-A14B / I2V-A14B `backend: mlx-gen` and
    every other video model `not_supported`; its Diffusers text-to-video is
    disabled for its one model and stable-diffusion.cpp raises for video.
  - Video needs far more memory than its file size, so the seed records the
    MEASURED run-time memory (`resident`) and the fit verdict uses it.

The rule, per host:
  - off Apple silicon: `output.video` is `unavailable` (MLX-Gen needs MLX),
    with the next step (an OpenAI-compatible video endpoint);
  - on Apple silicon: recommended (route + download + starter) only where the
    catalog's fit estimate says Wan2.2 TI2V-5B fits (`fits`/`tight`, the model
    browser's own filter): >= ~96 GiB of unified memory; below that
    `unavailable` with the two numbers the verdict compared.
"""

from __future__ import annotations

import copy

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.config.manager import ConfigurationManager
from tests.models_engines_fakes import isolate_host, synthetic_host

TI2V = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
T2V = "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
I2V = "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"
VIDEO_ROUTE = {"provider": "mlx-gen", "model": TI2V}
VIDEO_DOWNLOAD = {"provider": "mlx-gen", "artifact": TI2V}


def _host(name: str) -> dict:
    cpu = synthetic_host("cpu16")
    if name.startswith("metal"):
        return synthetic_host(name)
    return {
        "linux_x86_64_cpu": cpu,
        "linux_x86_64_cuda": synthetic_host("cuda24"),
        "linux_x86_64_rocm": synthetic_host("rocm32"),
        "linux_arm64": dict(cpu, os="linux", arch="arm64"),
        "windows_x86_64": dict(cpu, os="windows", arch="x86_64"),
        "windows_arm64": dict(cpu, os="windows", arch="arm64"),
        "intel_mac": dict(cpu, os="darwin", arch="x86_64", accelerator="none", unified_memory=False),
    }[name]


def _light(host: dict) -> dict:
    """What the import-time seed sees: os / arch / accelerator / RAM only."""
    return {k: host[k] for k in ("os", "arch", "accelerator", "unified_memory", "ram_bytes")}


NON_APPLE = [
    "linux_x86_64_cpu",
    "linux_x86_64_cuda",
    "linux_x86_64_rocm",
    "linux_arm64",
    "windows_x86_64",
    "windows_arm64",
    "intel_mac",
]
MAC_TOO_SMALL = ["metal16", "metal24", "metal32", "metal48", "metal64"]
MAC_FITS = ["metal96", "metal128", "metal192"]


# ---------------------------------------------------------------------------
# The per-host decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NON_APPLE)
def test_off_apple_silicon_video_is_unavailable_and_says_what_to_do(name):
    host = _host(name)
    assert "output.video" not in cd.recommended_capability_default_routes(host)
    assert "output.video" not in cd.recommended_model_downloads(host)
    row = cd.recommended_unavailable_routes(host)["output.video"]
    assert (row["provider"], row["model"]) == ("mlx-gen", TI2V)
    assert row["reason"].startswith("MLX-Gen video generation needs MLX")
    assert "Apple Silicon" in row["reason"]
    assert "OpenAI-compatible video endpoint" in row["reason"], "the reason names the remaining option"


@pytest.mark.parametrize("name", MAC_TOO_SMALL)
def test_a_mac_whose_memory_the_video_model_does_not_fit_gets_the_numbers(name):
    host = _host(name)
    assert "output.video" not in cd.recommended_capability_default_routes(host)
    assert "output.video" not in cd.recommended_model_downloads(host)
    reason = cd.recommended_unavailable_routes(host)["output.video"]["reason"]
    fit = mc.recommended_artifact_fit("mlx-gen", TI2V, host)["fit"]
    assert fit["verdict"] == "too_large"
    # The sentence carries the two amounts the verdict compared.
    assert f"needs about {fit['need_bytes'] / 1024**3:.1f} GiB" in reason
    assert f"about {fit['usable_bytes'] / 1024**3:.1f} GiB of it left for a model" in reason
    assert f"GPU memory limit on this Mac is about {fit['ceiling_bytes'] / 1024**3:.1f} GiB" in reason
    assert "MLX" not in reason.split(";")[0], "the engine runs here: memory is the reason"
    # The image row still runs on every Apple silicon Mac.
    assert "output.image" in cd.recommended_capability_default_routes(host)


@pytest.mark.parametrize("name", MAC_FITS)
def test_a_mac_the_video_model_fits_gets_route_download_and_plan(name):
    host = _host(name)
    assert cd.recommended_capability_default_routes(host)["output.video"].to_dict() == VIDEO_ROUTE
    assert cd.recommended_model_downloads(host)["output.video"] == VIDEO_DOWNLOAD
    assert cd.recommended_unavailable_routes(host) == {}
    plan = {p["key"]: p for p in cd.plan_recommended_capability_defaults({}, host=host)}
    video = plan["output.video"]
    assert video["action"] == "apply" and video["selector"] == "video"
    assert video["after"] == VIDEO_ROUTE and video["download"] == VIDEO_DOWNLOAD


@pytest.mark.parametrize("name", ["metal64", "metal96", "metal128"])
def test_the_light_import_time_reading_decides_like_the_full_probe(name):
    # No ceiling in the light reading: the gate uses the probe's own 75%
    # fallback, which is what the synthetic full profiles carry too.
    full = _host(name)
    assert set(cd.recommended_unavailable_routes(_light(full))) == set(cd.recommended_unavailable_routes(full))
    assert cd.recommended_model_downloads(_light(full)) == cd.recommended_model_downloads(full)


def test_a_mac_with_no_memory_reading_writes_no_video_route():
    host = dict(_light(synthetic_host("metal128")), ram_bytes=None)
    assert "output.video" not in cd.recommended_capability_default_routes(host)
    assert "could not be measured" in cd.recommended_unavailable_routes(host)["output.video"]["reason"]


def test_the_word_video_selects_only_the_video_row():
    assert cd.RECOMMENDED_SELECTORS["video"] == "output.video"
    plan = cd.plan_recommended_capability_defaults({}, only=["video"], host=_host("metal128"))
    assert [p["key"] for p in plan] == ["output.video"]


def test_video_is_never_written_where_unavailable_even_with_force():
    mine = {"output.video": cd.CapabilityRouteDefault(provider="openai-compatible", model="my-video")}
    plan = cd.plan_recommended_capability_defaults(mine, only=["video"], force=True, host=_host("linux_x86_64_cuda"))
    assert plan[0]["action"] == "unavailable" and plan[0]["after"] == mine["output.video"].to_dict()


# ---------------------------------------------------------------------------
# Through the real entry points (the light host probe patched)
# ---------------------------------------------------------------------------


@pytest.fixture()
def pin_host(monkeypatch):
    from abstractcore.utils import host_profile as hp

    def pin(host: dict) -> None:
        monkeypatch.setattr(hp, "host_profile", lambda **_k: dict(host))

    return pin


def test_a_fresh_install_on_a_128_gib_mac_seeds_the_video_route(tmp_path, pin_host):
    pin_host(_light(synthetic_host("metal128")))
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    video = manager.config.capability_defaults.routes["output.video"]
    assert (video.provider, video.model) == ("mlx-gen", TI2V)


def test_a_fresh_install_on_a_64_gib_mac_leaves_video_unset_and_the_grid_says_why(tmp_path, pin_host):
    pin_host(_light(synthetic_host("metal64")))
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    assert not manager.config.capability_defaults.routes.get("output.video", cd.CapabilityRouteDefault()).configured()
    rows = {row["key"]: row for row in manager.list_capability_defaults()}
    assert "needs about" in rows["output.video"]["recommendation_unavailable"]["reason"]
    assert "recommendation_unavailable" not in rows["output.image"]
    report = manager.apply_recommended_capability_defaults(only=["video"], force=True)
    assert report["unavailable"] == 1 and report["changed"] == 0


# ---------------------------------------------------------------------------
# The curated catalog: the VIDEO category with real fit data
# ---------------------------------------------------------------------------


def _rows(tmp_path, monkeypatch, host: dict, **kw) -> dict:
    isolate_host(tmp_path, monkeypatch)  # no presence probe reaches a live engine
    return {r["id"]: r for r in mc.catalog(host=host, **kw)["rows"]}


def test_the_catalog_has_the_three_wan_video_rows(tmp_path, monkeypatch):
    rows = _rows(tmp_path, monkeypatch, synthetic_host("metal128"), tags=["video"])
    assert set(rows) == {"wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.2-i2v-a14b"}
    tasks = {rid: (r["capabilities"]["text_to_video"], r["capabilities"]["image_to_video"]) for rid, r in rows.items()}
    assert tasks == {"wan2.2-ti2v-5b": (True, True), "wan2.2-t2v-a14b": (True, False), "wan2.2-i2v-a14b": (False, True)}
    for row in rows.values():
        assert row["capabilities"]["video_generation"] is True and row["capabilities"]["text"] is False
        (art,) = row["artifacts"]
        assert art["provider"] == "mlx-gen" and art["engine"] == "mlx"


def test_the_video_fit_uses_the_measured_memory_not_the_file_size(tmp_path, monkeypatch):
    seed = {a["artifact"]: a for r in mc.load_seed()["rows"] for a in r["artifacts"]}
    rows = _rows(tmp_path, monkeypatch, synthetic_host("metal128"), tags=["video"])
    art = rows["wan2.2-ti2v-5b"]["artifacts"][0]
    assert art["download_bytes"] == seed[TI2V]["download_bytes"]
    assert art["resident_bytes"] == seed[TI2V]["resident"]["bytes"]
    assert art["fit"]["weight_bytes"] == seed[TI2V]["resident"]["bytes"]
    assert any(n.startswith("memory need is measured") for n in art["fit"]["notes"])
    # The A14B 8-bit packages are measured at AbstractVision's default canvas
    # (1280x720, 81 frames): ~72 GiB each, not their 39.7 GiB files (backlog 0948).
    for rid, artifact in (("wan2.2-t2v-a14b", T2V), ("wan2.2-i2v-a14b", I2V)):
        resident = seed[artifact]["resident"]
        assert "1280x720" in resident["source"] and "81 frames" in resident["source"] and "mx.get_peak_memory" in resident["source"]
        art = rows[rid]["artifacts"][0]
        assert art["resident_bytes"] == resident["bytes"] > seed[artifact]["download_bytes"]
        assert art["fit"]["weight_bytes"] == resident["bytes"]
    # TI2V-5B: the file is 16.9 GiB, the run needs ~58 GiB. A file-size fit
    # would call it `fits` on a 64 GiB Mac; the measured one does not.
    mac64 = _rows(tmp_path / "b", monkeypatch, synthetic_host("metal64"), tags=["video"])
    assert mac64["wan2.2-ti2v-5b"]["artifacts"][0]["fit"]["verdict"] == "too_large"


@pytest.mark.parametrize("artifact", [T2V, I2V])
def test_a14b_8bit_per_memory_band_at_the_default_canvas(artifact):
    """Measured ~72 GiB at the default canvas: too large for a 64 GiB Mac (so
    it is NOT the 64-95 GiB video recommendation), fits a 96 GiB Mac once the
    GPU limit is raised (the command is said), fits 128 GiB."""

    fit = lambda name: mc.recommended_artifact_fit("mlx-gen", artifact, synthetic_host(name))["fit"]  # noqa: E731
    assert fit("metal64")["verdict"] == "too_large" and "gpu_limit" not in fit("metal64")
    f96 = fit("metal96")
    assert f96["verdict"] == "needs_gpu_limit"
    # need + 2 GiB of working buffers, whole GiB (78 GiB), under 96 - 12 GiB.
    assert f96["gpu_limit"]["command"] == "sudo sysctl iogpu.wired_limit_mb=79872"
    assert fit("metal128")["verdict"] in ("fits", "tight")
    # The recommendation per band is unchanged: TI2V-5B from ~96 GiB, and no
    # A14B route anywhere (it does not fit 64-95 GiB at the default canvas).
    for name in ("metal64", "metal96", "metal128"):
        routes = cd.recommended_capability_default_routes(synthetic_host(name))
        assert all(r.model not in (T2V, I2V) for r in routes.values())


def test_the_fits_filter_hides_video_where_it_cannot_run(tmp_path, monkeypatch):
    linux = _rows(tmp_path, monkeypatch, synthetic_host("cuda24"), tags=["video"])
    for row in linux.values():
        (art,) = row["artifacts"]
        assert art["supported_on_host"] is False and art["downloadable"] is False
        assert art["recommended"] is False
        assert "the mlx engine does not run on this host" in art["fit"]["notes"]
    assert _rows(tmp_path / "b", monkeypatch, synthetic_host("cuda24"), tags=["video"], fits=True) == {}
    assert set(_rows(tmp_path / "c", monkeypatch, synthetic_host("metal128"), tags=["video"], fits=True)) == {
        "wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.2-i2v-a14b"
    }


@pytest.mark.parametrize("kind, starter", [("cuda24", False), ("metal64", False), ("metal128", True)])
def test_the_video_starter_follows_the_host(tmp_path, monkeypatch, kind, starter):
    rows = _rows(tmp_path, monkeypatch, synthetic_host(kind), tags=["video"])
    assert rows["wan2.2-ti2v-5b"]["starter"] is starter
    assert rows["wan2.2-t2v-a14b"]["starter"] is rows["wan2.2-i2v-a14b"]["starter"] is False


# ---------------------------------------------------------------------------
# Seed contract
# ---------------------------------------------------------------------------


def _seed_with(mutate) -> list:
    seed = copy.deepcopy(mc.load_seed())
    art = next(r for r in seed["rows"] if r["id"] == "wan2.2-ti2v-5b")["artifacts"][0]
    mutate(art)
    return mc.validate_catalog(seed)


def test_the_shipped_seed_validates():
    assert mc.validate_catalog(mc.load_seed()) == []


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: a["resident"].pop("source"), "resident.source"),
        (lambda a: a["resident"].update(bytes=0), "resident.bytes"),
        (lambda a: a["resident"].update(guess=True), "unknown field 'guess'"),
        (lambda a: a.update(resident=123), "resident must be an object"),
    ],
)
def test_a_resident_figure_must_be_measured_and_sourced(mutate, message):
    errors = _seed_with(mutate)
    assert any(message in e for e in errors), errors


def test_a_recommendation_naming_an_artifact_outside_the_catalog_fails_loudly():
    with pytest.raises(LookupError):
        mc.recommended_artifact_fit("mlx-gen", "nobody/not-a-video-model", synthetic_host("metal128"))
