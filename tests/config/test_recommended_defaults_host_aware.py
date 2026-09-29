"""The recommended capability defaults are chosen PER HOST, for every route.

Field report (2026-09-27, headless Ubuntu x86_64 VPS, light profile): the
fresh-install seed wrote `output.image: mlx-gen/...flux.2-klein-4b-8bit` -- an
MLX engine, Apple silicon only -- and "apply recommended" re-applied it. Only
the text row was host-aware.

The rule now, per recommended route and host:
  - a route is written only when its engine runs on that host;
  - a route whose recommended engine cannot run is left UNSET and reported
    with its reason (grid: `recommendation_unavailable`; apply: `unavailable`);
  - voice stays Supertonic everywhere (ONNX Runtime on CPU, every desktop OS);
  - text keeps its owner (`recommended_text_model`), which now also skips LM
    Studio where it has no build (Intel Macs) for the same model on Ollama;
  - Apple silicon is BYTE-IDENTICAL to before (golden values below), plus
    the video row (2026-09-27, `output.video`, MLX-Gen Wan2.2 TI2V-5B), which
    is written only where its measured memory at AbstractVision's default
    canvas (832x480) fits (>= 32 GiB of unified memory) and reported
    unavailable everywhere else
    (tests/config/test_recommended_video_route.py owns that matrix).
"""

from __future__ import annotations

import json

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.config.manager import ConfigurationManager
from tests.models_engines_fakes import isolate_host, synthetic_host

MTP_OPTIONS = {"speculation": {"mode": "native_mtp", "num_draft_tokens": 2, "require_acceleration": False}}


def _host(name: str) -> dict:
    cpu = synthetic_host("cpu16")
    return {
        "linux_x86_64_cpu": cpu,
        "linux_x86_64_cuda": synthetic_host("cuda24"),
        "linux_x86_64_rocm": synthetic_host("rocm32"),
        "linux_arm64": dict(cpu, os="linux", arch="arm64"),
        "windows_x86_64": dict(cpu, os="windows", arch="x86_64"),
        "windows_arm64": dict(cpu, os="windows", arch="arm64"),
        "intel_mac": dict(cpu, os="darwin", arch="x86_64", accelerator="none", unified_memory=False),
    }[name]


NON_APPLE = [
    "linux_x86_64_cpu",
    "linux_x86_64_cuda",
    "linux_x86_64_rocm",
    "linux_arm64",
    "windows_x86_64",
    "windows_arm64",
    "intel_mac",
]
TEXT_BY_HOST = {name: ("lmstudio", "qwen/qwen3.5-9b", "qwen/qwen3.5-9b@4bit") for name in NON_APPLE}
TEXT_BY_HOST["intel_mac"] = ("ollama", "qwen3.5:9b", "qwen3.5:9b")


@pytest.fixture()
def pin_host(monkeypatch):
    from abstractcore.utils import host_profile as hp

    def pin(host: dict) -> None:
        monkeypatch.setattr(hp, "host_profile", lambda **_k: dict(host))

    return pin


# ---------------------------------------------------------------------------
# Every non-Apple host: nothing Apple-only, image unset with a reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NON_APPLE)
def test_no_route_or_download_names_an_engine_the_host_cannot_run(name):
    host = _host(name)
    routes = cd.recommended_capability_default_routes(host)
    downloads = cd.recommended_model_downloads(host)
    assert set(routes) == set(downloads) == {"input.text", "output.voice"}
    for key, route in routes.items():
        assert route.provider not in {"mlx", "mlx-gen"}, (name, key)
        assert cd.recommended_route_unavailable_reason(route.provider, host) is None
    provider, model, artifact = TEXT_BY_HOST[name]
    assert (routes["input.text"].provider, routes["input.text"].model) == (provider, model)
    assert routes["input.text"].options == MTP_OPTIONS
    assert downloads["input.text"] == {"provider": provider, "artifact": artifact}
    # Supertonic runs on CPU on every desktop OS: kept.
    assert (routes["output.voice"].provider, routes["output.voice"].model) == ("supertonic", "supertonic-3")
    assert downloads["output.voice"] == {"provider": "supertonic", "artifact": "supertonic-3"}


@pytest.mark.parametrize("name", NON_APPLE)
def test_the_image_route_is_reported_unavailable_with_its_reason(name):
    unavailable = cd.recommended_unavailable_routes(_host(name))
    assert set(unavailable) == {"output.image", "output.video"}
    row = unavailable["output.image"]
    assert (row["provider"], row["model"]) == ("mlx-gen", "AbstractFramework/flux.2-klein-4b-8bit")
    assert "Apple Silicon" in row["reason"]
    assert "diffusers" in row["reason"] and "sdcpp" in row["reason"], "the reason names what to do instead"


@pytest.mark.parametrize("name", NON_APPLE)
def test_the_seed_never_writes_the_image_route(name):
    seeded = cd.seed_recommended_capability_defaults(cd.CapabilityDefaultsConfig(), host=_host(name))
    assert set(seeded.routes) == {"input.text", "output.voice"}
    assert seeded.seeded == cd.RECOMMENDED_SEED_VERSION


@pytest.mark.parametrize("name", NON_APPLE)
def test_apply_reports_image_unavailable_and_writes_the_rest(name):
    plan = {p["key"]: p for p in cd.plan_recommended_capability_defaults({}, host=_host(name), force=True)}
    assert list(plan) == ["input.text", "output.voice", "output.image", "output.video"]
    assert plan["input.text"]["action"] == plan["output.voice"]["action"] == "apply"
    for key in ("output.image", "output.video"):
        row = plan[key]
        assert row["action"] == "unavailable" and row["changed"] is False
        assert row["after"] == row["before"] == {} and row["download"] == {}
        assert "Apple Silicon" in row["reason"]


def test_an_intel_mac_text_pick_is_the_ollama_build_of_the_portable_model():
    pick = mc.recommended_text_model(_host("intel_mac"), fit=False)
    assert (pick["provider"], pick["artifact"], pick["model"]) == ("ollama", "qwen3.5:9b", "qwen3.5:9b")
    assert pick["catalog_id"] == "qwen3.5-9b"
    assert pick["basis"] == "portable_engine_fallback"
    assert "LM Studio" in pick["tier"]
    assert pick["options"] == MTP_OPTIONS


def test_a_host_with_no_supported_engine_gets_no_route_at_all():
    host = dict(synthetic_host("cpu16"), os="freebsd")
    assert cd.recommended_capability_default_routes(host) == {}
    unavailable = cd.recommended_unavailable_routes(host)
    assert set(unavailable) == {"input.text", "input.image", "output.voice", "output.image", "output.video"}
    # Image input is read by the text model: unavailable for the same reason.
    assert unavailable["input.image"]["reason"] == unavailable["input.text"]["reason"]


def test_an_unknown_recommended_provider_fails_loudly():
    with pytest.raises(ValueError, match="no host-support rule"):
        cd.recommended_route_unavailable_reason("someengine", synthetic_host("cpu16"))


# ---------------------------------------------------------------------------
# Through the real entry points (the light host probe patched)
# ---------------------------------------------------------------------------


def test_a_fresh_linux_install_leaves_image_unset_and_the_grid_says_why(tmp_path, pin_host):
    pin_host(_host("linux_x86_64_cpu"))
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    stored = {k for k, r in manager.config.capability_defaults.routes.items() if r.configured()}
    assert stored == {"input.text", "output.voice"}
    rows = {row["key"]: row for row in manager.list_capability_defaults()}
    image = rows["output.image"]
    assert image["configured"] is False and image["source"] == "not_configured"
    assert "Apple Silicon" in image["recommendation_unavailable"]["reason"]
    assert "recommendation_unavailable" not in rows["output.voice"]
    assert "recommendation_unavailable" not in rows["output.image.text_to_image"]

    report = manager.apply_recommended_capability_defaults(force=True)
    assert report["unavailable"] == 2 and report["changed"] == 0
    assert "output.image" not in {k for k, r in manager.config.capability_defaults.routes.items() if r.configured()}


def test_an_operator_image_route_on_linux_carries_no_unavailable_note(tmp_path, pin_host):
    pin_host(_host("linux_x86_64_cuda"))
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    manager.set_capability_default("output", "image", provider="diffusers", model="black-forest-labs/FLUX.2-klein-4B")
    image = next(row for row in manager.list_capability_defaults() if row["key"] == "output.image")
    assert image["provider"] == "diffusers"
    assert "recommendation_unavailable" not in image


def test_the_catalog_starter_kit_follows_the_host(tmp_path, monkeypatch):
    isolate_host(tmp_path, monkeypatch)  # no presence probe reaches a live engine
    linux = {r["id"]: r for r in mc.catalog(host=_host("linux_x86_64_cpu"))["rows"]}
    assert linux["flux.2-klein-4b"]["starter"] is False, "an Apple-only model is not a Linux starter"
    assert linux["supertonic-3"]["starter"] is True
    assert linux["qwen3.5-9b"]["starter"] is True

    intel = {r["id"]: r for r in mc.catalog(host=_host("intel_mac"))["rows"]}
    picked = [(a["provider"], a["artifact"]) for a in intel["qwen3.5-9b"]["artifacts"] if a["recommended"]]
    assert picked == [("ollama", "qwen3.5:9b")], "the catalog pre-selects the same build the route stores"

    mac = {r["id"]: r for r in mc.catalog(host=synthetic_host("metal64"))["rows"]}
    assert mac["flux.2-klein-4b"]["starter"] is True


# ---------------------------------------------------------------------------
# Apple silicon: byte-identical to the pre-fix behaviour
# ---------------------------------------------------------------------------

APPLE_TEXT = {
    "metal16": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "metal64": "mlx-community/Qwen3.8-27B-4bit",
    "metal128": "mlx-community/Qwen3.8-Flash-Next-4bit",
}


# Where the recommended video model fits (measured memory at its 832x480
# default canvas; >= 32 GiB).
VIDEO_FITS = {"metal16": False, "metal64": True, "metal128": True}
VIDEO = {"provider": "mlx-gen", "model": "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"}


def _golden_apple_seed(text_model: str, video: bool = False) -> str:
    # The exact JSON the pre-fix seed wrote on Apple silicon; the video row is
    # the one addition, only where the video model fits.
    routes = {
        "input.text": {"provider": "mlx", "model": text_model, "options": MTP_OPTIONS},
        "output.image": {"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-4b-8bit"},
        "output.voice": {"provider": "supertonic", "model": "supertonic-3"},
    }
    if video:
        routes = dict(sorted({**routes, "output.video": dict(VIDEO)}.items()))
    return json.dumps({"version": 1, "routes": routes, "seeded": "recommended-v1"}, sort_keys=False)


@pytest.mark.parametrize("kind", sorted(APPLE_TEXT))
def test_apple_silicon_seed_is_byte_identical(kind, monkeypatch):
    monkeypatch.setattr(mc, "MTP_RECOMMENDED", False)
    host = synthetic_host(kind)
    seeded = cd.seed_recommended_capability_defaults(cd.CapabilityDefaultsConfig(), host=host)
    assert json.dumps(seeded.to_dict(), sort_keys=False) == _golden_apple_seed(APPLE_TEXT[kind], VIDEO_FITS[kind])
    assert set(cd.recommended_unavailable_routes(host)) == (set() if VIDEO_FITS[kind] else {"output.video"})


@pytest.mark.parametrize("kind", sorted(APPLE_TEXT))
def test_apple_silicon_routes_downloads_and_plan_are_unchanged(kind, monkeypatch):
    monkeypatch.setattr(mc, "MTP_RECOMMENDED", False)
    # Every engine installed (`engine_missing` would otherwise add its key:
    # test_route_engine_missing.py owns that state).
    from abstractcore.config import route_engines

    monkeypatch.setattr(route_engines, "route_engine_missing", lambda *a, **k: None)
    host = synthetic_host(kind)
    text = APPLE_TEXT[kind]
    video = VIDEO_FITS[kind]
    routes = cd.recommended_capability_default_routes(host)
    assert list(routes) == ["input.text", "output.voice", "output.image"] + (["output.video"] if video else [])
    assert {k: v.to_dict() for k, v in routes.items()} == {
        "input.text": {"provider": "mlx", "model": text, "options": MTP_OPTIONS},
        "output.voice": {"provider": "supertonic", "model": "supertonic-3"},
        "output.image": {"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-4b-8bit"},
        **({"output.video": dict(VIDEO)} if video else {}),
    }
    assert cd.recommended_model_downloads(host) == {
        "input.text": {"provider": "mlx", "artifact": text},
        "output.voice": {"provider": "supertonic", "artifact": "supertonic-3"},
        "output.image": {"provider": "mlx-gen", "artifact": "AbstractFramework/flux.2-klein-4b-8bit"},
        **({"output.video": {"provider": VIDEO["provider"], "artifact": VIDEO["model"]}} if video else {}),
    }
    plan = cd.plan_recommended_capability_defaults({}, host=host)
    assert [p["key"] for p in plan] == ["input.text", "output.voice", "output.image", "output.video"]
    for entry in list(plan[:3]) + (list(plan[3:]) if video else []):
        # The pre-fix entry shape, exactly: no `reason`, no new keys.
        assert set(entry) == {"key", "selector", "action", "changed", "recommended", "before", "after", "download"}
        assert entry["action"] == "apply"
    assert plan[2]["after"] == {"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-4b-8bit"}
    if not video:
        assert plan[3]["action"] == "unavailable" and plan[3]["after"] == {}


def test_apple_silicon_grid_carries_no_unavailable_note(tmp_path, pin_host):
    # 128 GiB: every recommended row runs, the video one included (a smaller
    # Mac notes only the video row: test_recommended_video_route.py).
    pin_host(synthetic_host("metal128"))
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    manager.clear_capability_default("output", "image")
    rows = manager.list_capability_defaults()
    assert not any("recommendation_unavailable" in row for row in rows)
    report = manager.apply_recommended_capability_defaults(dry_run=True)
    assert report["unavailable"] == 0
