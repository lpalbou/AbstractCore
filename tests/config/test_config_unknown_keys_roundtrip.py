"""A settings save must keep the keys this version does not know.

Defect D-B (upgrade-safety review, 2026-09-30): `_dict_to_config` filters every
section down to the dataclass fields this version knows, and `_save_config`
published the known sections only. The three-way merge then read each unknown
key as "in my baseline, absent from mine, unchanged on disk" -- a DELETE -- so
the first save after any restart removed a user-added top-level key, a
`vision.user_note`, or every key a NEWER core had written (a downgrade). Same
in 2.19.0.

The rule now: known keys are validated as before; everything else in the
document this manager loaded round-trips unchanged in value.
"""

from __future__ import annotations

import json
from pathlib import Path

from abstractcore.config.manager import ConfigurationManager, merge_store_documents


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _base_document() -> dict:
    return {
        "vision": {"strategy": "disabled", "user_note": {"why": "kept by hand", "n": [1, 2]}},
        "capability_defaults": {
            "version": 1,
            "routes": {"input.text": {"provider": "lmstudio", "model": "old-model"}},
        },
        "user_custom_key": "mine, not yours",
    }


def test_a_setter_save_keeps_a_top_level_custom_key_and_a_nested_section_key(tmp_path: Path) -> None:
    cfg = tmp_path / "abstractcore.json"
    _write(cfg, _base_document())

    manager = ConfigurationManager(config_file=cfg, apply_env=False)
    manager.update_capability_default("output.text", provider="lmstudio", model="new-model")
    manager.set_default_timeout(123.0)

    on_disk = _read(cfg)
    assert on_disk["user_custom_key"] == "mine, not yours"
    assert on_disk["vision"]["user_note"] == {"why": "kept by hand", "n": [1, 2]}
    # The known keys the setters wrote are there too.
    # The text route is stored under `input.text`.
    assert on_disk["capability_defaults"]["routes"]["input.text"]["model"] == "new-model"
    assert on_disk["timeouts"]["default_timeout"] == 123.0

    # A restart and another save keep them as well.
    again = ConfigurationManager(config_file=cfg, apply_env=False)
    again.set_default_timeout(45.0)
    on_disk = _read(cfg)
    assert on_disk["user_custom_key"] == "mine, not yours"
    assert on_disk["vision"]["user_note"] == {"why": "kept by hand", "n": [1, 2]}
    assert on_disk["timeouts"]["default_timeout"] == 45.0


def test_keys_only_a_newer_version_knows_survive_a_downgrade_save(tmp_path: Path) -> None:
    cfg = tmp_path / "abstractcore.json"
    document = _base_document()
    document["future_section"] = {"enabled": True, "nested": {"level": 3}}
    document["server"] = {"port": 8123, "future_server_field": "v9"}
    document["capability_defaults"]["future_meta"] = {"schema": 9}
    document["capability_defaults"]["routes"]["output.hologram"] = {"provider": "abstract4d", "model": "h1"}
    document["provider_profiles"] = {
        "profiles": {
            "work": {
                "id": "work",
                "provider_family": "openai-compatible",
                "base_url": "http://127.0.0.1:9999/v1",
                "future_column": {"tier": "gold"},
            }
        },
        "future_profiles_meta": 2,
    }
    _write(cfg, document)

    manager = ConfigurationManager(config_file=cfg, apply_env=False)
    # The unknown route is not a route here; the known sections still validate.
    assert "output.hologram" not in manager.config.capability_defaults.routes
    assert manager.config.server.port == 8123
    manager.update_capability_default("output.text", provider="lmstudio", model="new-model")

    on_disk = _read(cfg)
    assert on_disk["future_section"] == {"enabled": True, "nested": {"level": 3}}
    assert on_disk["server"]["future_server_field"] == "v9"
    assert on_disk["server"]["port"] == 8123
    assert on_disk["capability_defaults"]["future_meta"] == {"schema": 9}
    assert on_disk["capability_defaults"]["routes"]["output.hologram"] == {"provider": "abstract4d", "model": "h1"}
    assert on_disk["provider_profiles"]["profiles"]["work"]["future_column"] == {"tier": "gold"}
    assert on_disk["provider_profiles"]["future_profiles_meta"] == 2


def test_a_known_key_cleared_by_a_setter_stays_cleared(tmp_path: Path) -> None:
    """Carrying unknown keys must not resurrect a known row an operator deleted."""
    cfg = tmp_path / "abstractcore.json"
    document = _base_document()
    document["capability_defaults"]["routes"]["output.image"] = {"provider": "mflux", "model": "m"}
    document["capability_defaults"]["seeded"] = "recommended-v2"
    _write(cfg, document)

    manager = ConfigurationManager(config_file=cfg, apply_env=False)
    manager.clear_capability_default("output.image")

    on_disk = _read(cfg)
    assert "output.image" not in on_disk["capability_defaults"]["routes"]
    assert on_disk["user_custom_key"] == "mine, not yours"


def test_a_save_merged_with_another_writer_keeps_unknown_keys_from_both(tmp_path: Path) -> None:
    """The store merge preserves the unknown keys of every document it merges.

    The manager loaded `user_custom_key` / `vision.user_note` (its baseline);
    another writer then added its own unknown keys and changed a carried one.
    The merged publish keeps all of them and lets the other writer's change win.
    """
    cfg = tmp_path / "abstractcore.json"
    _write(cfg, _base_document())
    manager = ConfigurationManager(config_file=cfg, apply_env=False)

    other = _read(cfg)
    other["other_writer_key"] = {"from": "gateway"}
    other["vision"]["other_note"] = "added later"
    other["vision"]["user_note"] = {"why": "edited by the other writer"}
    _write(cfg, other)

    manager.set_default_timeout(77.0)
    on_disk = _read(cfg)
    assert on_disk["user_custom_key"] == "mine, not yours"
    assert on_disk["other_writer_key"] == {"from": "gateway"}
    assert on_disk["vision"]["other_note"] == "added later"
    assert on_disk["vision"]["user_note"] == {"why": "edited by the other writer"}
    assert on_disk["timeouts"]["default_timeout"] == 77.0


def test_merge_store_documents_keeps_unknown_keys_of_disk_and_mine() -> None:
    baseline = {"known": 1, "vision": {"strategy": "a", "note": "x"}, "custom": "c"}
    mine = {"known": 2, "vision": {"strategy": "b", "note": "x"}, "custom": "c"}
    disk = {
        "known": 1,
        "vision": {"strategy": "a", "note": "x", "disk_only": True},
        "custom": "c",
        "future": {"k": "v"},
    }
    merged = merge_store_documents(baseline, mine, disk)
    assert merged == {
        "known": 2,
        "vision": {"strategy": "b", "note": "x", "disk_only": True},
        "custom": "c",
        "future": {"k": "v"},
    }
