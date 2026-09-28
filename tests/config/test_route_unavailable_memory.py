"""A CONFIGURED route whose model does not fit this computer is `route_unavailable`.

Gateway-worker finding (2026-09-28, before the 2.18.1 tag):
`configured_routes_unavailable` judged engine support only. An 8 GB Mac whose
SAVED routes hold the FLUX.2 klein 4B image route (~8.5 GiB, the route the
recommendation itself stopped writing there) read as fine and ran out of
memory at first use.

The contract:
  - a configured route on a memory-gated capability (image, video, music)
    whose catalog fit verdict for this host is "does not fit" is flagged, with
    the reason wording the recommendations use (ONE fit rule:
    `model_catalog.recommended_artifact_fit`, the one the seed and the
    recommendations read);
  - `tight` and `needs_gpu_limit` (fits after the `sysctl`) are NOT unavailable;
  - the saved config is never changed by the check.

Also here (same wave): `input.image` on a host whose recommended text model
does not read images reports why (`recommended_unavailable_routes`), and a
text tier whose model has no MTP build is not written with an MTP
`speculation` policy.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.config import recommendations as rec
from abstractcore.config.manager import ConfigurationManager
from tests.models_engines_fakes import synthetic_host

FLUX = "AbstractFramework/flux.2-klein-4b-8bit"
WAN_A14B = "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
MAC8 = synthetic_host("metal8")
MAC64 = synthetic_host("metal64")


@pytest.fixture()
def pin_host(monkeypatch):
    from abstractcore.utils import host_profile as hp

    def pin(host: dict) -> None:
        monkeypatch.setattr(hp, "host_profile", lambda **_k: dict(host))

    return pin


def _store(tmp_path, routes: dict) -> ConfigurationManager:
    cfg = tmp_path / "abstractcore.json"
    cfg.write_text(
        json.dumps({"capability_defaults": {"version": 1, "routes": routes, "seeded": "recommended-v1"}}),
        encoding="utf-8",
    )
    return ConfigurationManager(config_file=cfg, apply_env=False)


def _stored_route(manager, key: str) -> dict:
    raw = json.loads(manager.config_file.read_text(encoding="utf-8"))
    route = raw["capability_defaults"]["routes"].get(key) or {}
    return {k: route[k] for k in ("provider", "model") if k in route}


def _rows(manager) -> dict:
    return {row["key"]: row for row in manager.list_capability_defaults()}


# ---------------------------------------------------------------------------
# The memory gate on configured routes
# ---------------------------------------------------------------------------


def test_a_saved_image_route_on_an_8gb_mac_is_flagged_with_the_recommendations_reason():
    assert mc.recommended_artifact_fit("mlx-gen", FLUX, MAC8)["fit"]["verdict"] == "too_large"
    broken = cd.configured_routes_unavailable({"output.image": {"provider": "mlx-gen", "model": FLUX}}, MAC8)
    assert set(broken) == {"output.image"}
    flag = broken["output.image"]
    assert (flag["provider"], flag["model"]) == ("mlx-gen", FLUX)
    # The same sentence the recommendation gives for not writing this route.
    assert flag["reason"] == cd.recommended_unavailable_routes(MAC8)["output.image"]["reason"]
    assert "FLUX.2" in flag["reason"] and "of memory" in flag["reason"]


def test_a_task_row_is_judged_by_its_modality(tmp_path):
    broken = cd.configured_routes_unavailable(
        {"output.image.text_to_image": {"provider": "mlx-gen", "model": FLUX}}, MAC8
    )
    assert "FLUX.2" in broken["output.image.text_to_image"]["reason"]


def test_the_grid_flags_it_and_never_changes_the_saved_config(tmp_path, pin_host):
    pin_host(MAC8)
    manager = _store(tmp_path, {"output.image": {"provider": "mlx-gen", "model": FLUX}})
    before = manager.config_file.read_bytes()
    image = _rows(manager)["output.image"]
    assert image["configured"] is True
    assert "FLUX.2" in image["route_unavailable"]["reason"]
    report = manager.apply_recommended_capability_defaults(dry_run=True)
    entry = next(row for row in report["routes"] if row["key"] == "output.image")
    assert entry["route_unavailable"]["model"] == FLUX
    assert manager.config_file.read_bytes() == before, "reporting never rewrites the store"


def test_a_64gb_mac_does_not_flag_the_image_route(tmp_path, pin_host):
    assert cd.configured_routes_unavailable({"output.image": {"provider": "mlx-gen", "model": FLUX}}, MAC64) == {}
    pin_host(MAC64)
    manager = _store(tmp_path, {"output.image": {"provider": "mlx-gen", "model": FLUX}})
    assert "route_unavailable" not in _rows(manager)["output.image"]


def test_needs_sysctl_is_not_unavailable():
    """A 96 GB Mac runs Wan2.2 T2V-A14B once the GPU memory limit is raised:
    that is advice (the sysctl), never "cannot run here"."""

    mac96 = synthetic_host("metal96")
    assert mc.recommended_artifact_fit("mlx-gen", WAN_A14B, mac96)["fit"]["verdict"] == "needs_gpu_limit"
    route = {"output.video.text_to_video": {"provider": "mlx-gen", "model": WAN_A14B}}
    assert cd.configured_routes_unavailable(route, mac96) == {}


def test_tight_is_not_unavailable():
    mac16 = synthetic_host("metal16")
    assert mc.recommended_artifact_fit("mlx-gen", FLUX, mac16)["fit"]["verdict"] == "tight"
    assert cd.configured_routes_unavailable({"output.image": {"provider": "mlx-gen", "model": FLUX}}, mac16) == {}


def test_a_model_outside_the_catalog_is_not_judged_by_memory():
    route = {"output.image": {"provider": "mlx-gen", "model": "someone/unknown-image-model"}}
    assert cd.configured_routes_unavailable(route, MAC8) == {}


# ---------------------------------------------------------------------------
# input.image says why on a host whose text model does not read images
# ---------------------------------------------------------------------------


@pytest.fixture()
def text_reads_no_images(monkeypatch):
    """A host whose recommended text model does not read images (the 2.18.1
    staging 8 GB tier was one; no tier is today): the one rule decides it."""
    from abstractcore.config import manager as mgr

    real = mgr.model_supports_input
    monkeypatch.setattr(mgr, "model_supports_input", lambda model, modality: False if modality == "image" else real(model, modality))


def test_image_input_reports_the_recommendations_reason_where_text_reads_no_images(tmp_path, pin_host, text_reads_no_images):
    mac16 = synthetic_host("metal16")
    unavailable = cd.recommended_unavailable_routes(mac16)
    vision = rec.recommended_models(mac16)["vision"]
    assert vision["status"] == "unavailable"
    assert unavailable["input.image"]["reason"] == vision["reason"]
    assert "does not read images" in vision["reason"]
    pin_host(mac16)
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    row = _rows(manager)["input.image"]
    assert row["configured"] is False
    assert row["recommendation_unavailable"]["reason"] == vision["reason"]


@pytest.mark.parametrize("kind", ["metal8", "metal16", "cuda24"])
def test_image_input_is_not_unavailable_where_the_text_model_reads_images(kind):
    assert "input.image" not in cd.recommended_unavailable_routes(synthetic_host(kind))


# ---------------------------------------------------------------------------
# No MTP policy for a text model without an MTP drafter
# ---------------------------------------------------------------------------


def test_a_tier_without_an_mtp_build_carries_no_speculation_policy(monkeypatch):
    tiers = list(mc.APPLE_TEXT_TIERS)
    tiers[0] = dict(tiers[0], mtp=None)
    monkeypatch.setattr(mc, "APPLE_TEXT_TIERS", tuple(tiers))
    pick = mc.recommended_text_model(MAC8, fit=False)
    assert "speculation" not in pick["options"]
    assert "speculation" not in cd.recommended_capability_default_routes(MAC8)["input.text"].options


@pytest.mark.parametrize("gib", [8, 16, 32, 128])
def test_tiers_with_an_mtp_build_keep_the_policy(gib):
    pick = mc.recommended_text_model(synthetic_host(f"metal{gib}"), fit=False)
    assert pick["options"]["speculation"]["mode"] == "native_mtp"


# ---------------------------------------------------------------------------
# A saved video route our own reason told a 64 GB user to set is never flagged
# ---------------------------------------------------------------------------

TI2V = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
T2V_A14B = "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"


@pytest.mark.parametrize("key,model,canvas", [
    ("output.video", TI2V, "832x480"),
    ("output.video.text_to_video", T2V_A14B, "640x352"),
])
def test_a_video_route_that_runs_at_a_measured_smaller_canvas_is_not_flagged(tmp_path, pin_host, key, model, canvas):
    mac64 = synthetic_host("metal64")
    assert mc.recommended_artifact_fit("mlx-gen", model, mac64)["fit"]["verdict"] == "too_large"
    assert mc.smaller_canvas_fit("mlx-gen", model, mac64)["canvas"].startswith(canvas)
    if key == "output.video":
        # The recommendation's own reason tells this Mac to set exactly this route.
        assert f"set output.video to mlx-gen/{model} yourself" in cd.recommended_unavailable_routes(mac64)[key]["reason"]
    assert cd.configured_routes_unavailable({key: {"provider": "mlx-gen", "model": model}}, mac64) == {}
    pin_host(mac64)
    manager = _store(tmp_path, {key: {"provider": "mlx-gen", "model": model}})
    report = manager.apply_recommended_capability_defaults(force=True)
    assert report["cleared"] == 0, "--force never clears the route the reason told the user to set"
    assert _stored_route(manager, key) == {"provider": "mlx-gen", "model": model}


def test_a_video_route_with_no_fitting_canvas_is_still_flagged():
    mac32 = synthetic_host("metal32")
    assert mc.smaller_canvas_fit("mlx-gen", TI2V, mac32) is None
    assert "output.video" in cd.configured_routes_unavailable({"output.video": {"provider": "mlx-gen", "model": TI2V}}, mac32)
