"""A provider profile whose family this Core does not know is kept as stored and said, never fatal.

2026-11-03: abstractcore 2.31 (no `typesafe` family) read a config written by 2.32 with a TypeSafe
profile. The row's ValueError made the whole file read as corrupt: a `.corrupt-<stamp>.bak` copy on
every import and DEFAULTS for the session, so every other profile and setting was gone too."""
from __future__ import annotations

import json
import logging

from abstractcore.config.manager import ConfigurationManager

FUTURE_ROW = {
    "id": "future-key",
    "display_name": "Future (API key)",
    "provider_family": "a-family-from-a-newer-core",
    "base_url": "",
    "api_key": "sk-future",
    "enabled": True,
    "created_at": "2026-11-01T23:20:40Z",
    "updated_at": "2026-11-01T23:27:16Z",
    "a_newer_column": {"kept": True},
}
KNOWN_ROW = {
    "id": "ovh-provider",
    "provider_family": "openai-compatible",
    "base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    "api_key": "sk-ovh",
}


def _write(tmp_path):
    config_file = tmp_path / "abstractcore.json"
    config_file.write_text(json.dumps({"provider_profiles": {"profiles": {
        "future-key": FUTURE_ROW, "ovh-provider": KNOWN_ROW}}}, indent=2), encoding="utf-8")
    return config_file


def test_a_config_with_an_unknown_family_loads_writes_nothing_and_says_so(tmp_path, caplog) -> None:
    config_file = _write(tmp_path)
    before = config_file.read_bytes()
    with caplog.at_level(logging.WARNING):
        manager = ConfigurationManager(config_file=config_file, apply_env=False)

    assert [p["id"] for p in manager.list_provider_profiles()] == ["ovh-provider"]  # not DEFAULTS
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abstractcore.json"]  # no .corrupt-*.bak
    assert config_file.read_bytes() == before
    said = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("'future-key' is not loaded" in m and "a-family-from-a-newer-core" in m for m in said), said


def test_a_save_keeps_the_unknown_row_exactly_as_stored(tmp_path) -> None:
    config_file = _write(tmp_path)
    manager = ConfigurationManager(config_file=config_file, apply_env=False)
    manager.set_provider_profile("local-vllm", provider_family="vllm", base_url="http://127.0.0.1:8000/v1")

    rows = json.loads(config_file.read_text(encoding="utf-8"))["provider_profiles"]["profiles"]
    assert rows["future-key"] == FUTURE_ROW
    assert sorted(rows) == ["future-key", "local-vllm", "ovh-provider"]
    assert sorted(p["id"] for p in manager.list_provider_profiles()) == ["local-vllm", "ovh-provider"]


def test_the_decision_model_providers_are_profile_families(tmp_path) -> None:
    """`typesafe` (hosted Jev) and `systemone` (any System One endpoint) are registered providers, so a
    connection to them loads like any other (2.31's hard-coded family list did not have them)."""
    config_file = tmp_path / "abstractcore.json"
    config_file.write_text(json.dumps({"provider_profiles": {"profiles": {
        "typesafe-key": {"id": "typesafe-key", "provider_family": "typesafe", "api_key": "ts-not-a-real-key"},
        "my-systemone": {"id": "my-systemone", "provider_family": "systemone", "base_url": "https://s1.example/v1"},
    }}}), encoding="utf-8")
    manager = ConfigurationManager(config_file=config_file, apply_env=False)

    assert sorted(p["id"] for p in manager.list_provider_profiles()) == ["my-systemone", "typesafe-key"]
    assert manager.get_provider_config("endpoint:typesafe-key")["provider_family"] == "typesafe"
    assert manager.get_provider_config("endpoint:my-systemone")["provider_family"] == "systemone"
