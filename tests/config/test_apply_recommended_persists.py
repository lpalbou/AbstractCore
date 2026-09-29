"""apply-recommended plans against the STORE ON DISK and always persists.

Framework rehearsal 0.6.3 (Linux + NVIDIA, fresh store): a grid read ran the
full host probe (cached for 5 s); an apply within those 5 s built a manager
whose in-memory fresh-install seed read the light profile, got the CACHED FULL
profile (CUDA) and so already held the Diffusers image route. Planned against
memory, apply reported the image "already set" and saved nothing; the next read
(no file, light reading, no CUDA) showed image missing.

These tests drive the REAL `host_profile` cache with a frozen clock: only the
probes (`_build_profile`, `_build_light_profile`) are synthetic.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.config.manager import ConfigurationManager
from abstractcore.utils import host_profile as hp
from tests.models_engines_fakes import synthetic_host

CUDA_IMAGE = {"provider": "diffusers", "model": "black-forest-labs/FLUX.2-klein-4B"}


@pytest.fixture
def clock(monkeypatch):
    """Frozen `time.monotonic` for the host-profile cache, advanced by hand."""

    now = {"t": 10_000.0}
    monkeypatch.setattr(hp.time, "monotonic", lambda: now["t"])
    full = synthetic_host("cuda24")
    light = dict(full, accelerator="none", vram_bytes=None, ceiling_bytes=None, ceiling_source=None, light=True)
    monkeypatch.setattr(hp, "_build_profile", lambda: dict(full))
    monkeypatch.setattr(hp, "_build_light_profile", lambda: dict(light))
    monkeypatch.setitem(hp._cache, "value", None)
    monkeypatch.setitem(hp._cache, "at", 0.0)
    return now


def _manager(tmp_path) -> ConfigurationManager:
    return ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)


def _stored_routes(tmp_path) -> dict:
    return json.loads((tmp_path / "abstractcore.json").read_text(encoding="utf-8"))["capability_defaults"]["routes"]


def _row(report: dict, key: str) -> dict:
    return next(row for row in report["routes"] if row["key"] == key)


def test_apply_within_the_probe_cache_window_persists_the_image_route(tmp_path, clock):
    # A grid read: the full probe runs and is cached (clock frozen from here).
    assert hp.host_profile()["accelerator"] == "cuda"

    manager = _manager(tmp_path)
    assert not (tmp_path / "abstractcore.json").exists()
    seeded = manager.config.capability_defaults.routes["output.image"]
    assert {"provider": seeded.provider, "model": seeded.model} == CUDA_IMAGE, (
        "precondition: within the cache window the in-memory seed already holds the image route"
    )

    report = manager.apply_recommended_capability_defaults()

    image = _row(report, "output.image")
    assert image["action"] == "apply" and image["changed"] is True, image
    assert image["before"] == {}
    stored = _stored_routes(tmp_path)
    assert {k: stored["output.image"][k] for k in ("provider", "model")} == CUDA_IMAGE

    # Past the cache window a fresh reader (light reading, no CUDA) still sees it.
    clock["t"] += 60.0
    reread = _manager(tmp_path).config.capability_defaults.routes["output.image"]
    assert {"provider": reread.provider, "model": reread.model} == CUDA_IMAGE


def test_already_means_persisted_and_apply_always_saves(tmp_path, clock):
    manager = _manager(tmp_path)
    manager.apply_recommended_capability_defaults()
    before = _stored_routes(tmp_path)

    again = _manager(tmp_path).apply_recommended_capability_defaults()

    assert again["changed"] == 0
    for key in ("input.text", "output.voice", "input.voice", "output.image"):
        assert _row(again, key)["action"] == "already", key
    assert _stored_routes(tmp_path) == before


def test_a_seed_matching_the_recommendation_is_written_by_apply(tmp_path, clock):
    # Outside any cache window: the seed reads the light profile (no CUDA) and
    # holds text, voice output and speech input only -- none of it on disk yet.
    manager = _manager(tmp_path)
    assert not (tmp_path / "abstractcore.json").exists()

    report = manager.apply_recommended_capability_defaults()

    stored = _stored_routes(tmp_path)
    for key in ("input.text", "output.voice", "input.voice", "output.image"):
        assert _row(report, key)["action"] == "apply", key
        assert stored[key]["provider"], key


def test_dry_run_writes_nothing(tmp_path, clock):
    report = _manager(tmp_path).apply_recommended_capability_defaults(dry_run=True)
    assert _row(report, "output.image")["action"] == "apply"
    assert not (tmp_path / "abstractcore.json").exists()
