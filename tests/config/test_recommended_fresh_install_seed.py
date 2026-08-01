"""Fresh-install recommended defaults (operator ruling 2026-08-01).

A brand-new install works out of the box on the recommended stack — text
`lmstudio/qwen/qwen3.5-9b`, voice `supertonic/supertonic-3`, image
`mlx-gen/AbstractFramework/flux.2-klein-4b-8bit` — seeded exactly once, when
no config file has ever existed. The seed writes ordinary rows: visible in
every grid, overridable and clearable from either entry point, always beaten
by request pins. The `seeded` marker is provenance only; FILE EXISTENCE gates
re-seeding, so a cleared route can never resurrect.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractcore.config.capability_defaults import (
    RECOMMENDED_CAPABILITY_DEFAULT_ROUTES,
    RECOMMENDED_SEED_VERSION,
)
from abstractcore.config.manager import ConfigurationManager


@pytest.fixture()
def fresh_config(tmp_path: Path) -> Path:
    return tmp_path / "abstractcore.json"


def _configured_keys(manager: ConfigurationManager) -> set[str]:
    return {
        key
        for key, route in manager.config.capability_defaults.routes.items()
        if route.configured()
    }


def test_fresh_install_seeds_the_three_recommended_routes(fresh_config: Path) -> None:
    manager = ConfigurationManager(config_file=fresh_config)
    routes = manager.config.capability_defaults.routes
    assert routes["input.text"].provider == "lmstudio"
    assert routes["input.text"].model == "qwen/qwen3.5-9b"
    assert routes["output.voice"].provider == "supertonic"
    assert routes["output.voice"].model == "supertonic-3"
    assert routes["output.image"].provider == "mlx-gen"
    assert routes["output.image"].model == "AbstractFramework/flux.2-klein-4b-8bit"
    assert manager.config.capability_defaults.seeded == RECOMMENDED_SEED_VERSION


def test_the_derived_text_route_reads_the_seed(fresh_config: Path) -> None:
    manager = ConfigurationManager(config_file=fresh_config)
    row = manager.get_capability_default("output", "text")
    assert row.get("provider") == "lmstudio"
    assert row.get("model") == "qwen/qwen3.5-9b"


def test_the_marker_survives_persist_and_reload(fresh_config: Path) -> None:
    manager = ConfigurationManager(config_file=fresh_config)
    manager._save_config()
    reloaded = ConfigurationManager(config_file=fresh_config)
    assert reloaded.config.capability_defaults.seeded == RECOMMENDED_SEED_VERSION
    assert _configured_keys(reloaded) >= set(RECOMMENDED_CAPABILITY_DEFAULT_ROUTES)


def test_a_cleared_recommended_route_never_resurrects(fresh_config: Path) -> None:
    manager = ConfigurationManager(config_file=fresh_config)
    manager._save_config()
    assert manager.clear_capability_default("output", "image")
    reloaded = ConfigurationManager(config_file=fresh_config)
    assert "output.image" not in _configured_keys(reloaded)


def test_an_operator_override_beats_the_seed_and_persists(fresh_config: Path) -> None:
    manager = ConfigurationManager(config_file=fresh_config)
    manager._save_config()
    assert manager.update_capability_default("input.text", provider="lmstudio", model="qwen3-0.6b")
    reloaded = ConfigurationManager(config_file=fresh_config)
    assert reloaded.config.capability_defaults.routes["input.text"].model == "qwen3-0.6b"


def test_an_existing_store_is_never_seeded_even_when_empty(fresh_config: Path) -> None:
    fresh_config.write_text(
        json.dumps({"version": "1.0", "capability_defaults": {"version": 1, "routes": {}}})
    )
    manager = ConfigurationManager(config_file=fresh_config)
    assert _configured_keys(manager) == set()
    assert manager.config.capability_defaults.seeded is None


def test_the_corrupt_file_fallback_is_never_seeded(fresh_config: Path) -> None:
    # An unparseable store falls back to defaults but the operator's settings
    # are RECOVERABLE from the backup — recommendations must not shadow them.
    fresh_config.write_text("{ this is not json")
    manager = ConfigurationManager(config_file=fresh_config)
    assert _configured_keys(manager) == set()
    assert manager.config.capability_defaults.seeded is None
