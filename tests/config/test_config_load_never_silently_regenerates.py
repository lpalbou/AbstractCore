"""Config data-safety: an existing settings file is never silently replaced by defaults.

A file that cannot be parsed or read stops the load with the path and the error, exactly as it
is on disk; a valid file loads untouched; only a missing file is a fresh install.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import pytest

from abstractcore.config.manager import ConfigParseError, ConfigurationManager


def test_corrupt_config_stops_the_load_and_is_left_exactly_as_it_is() -> None:
    d = Path(tempfile.mkdtemp())
    bad = d / "abstractcore.json"
    original = '{ not valid json ,,, "embeddings": {"model": "text-embedding-qwen3-embedding-0.6b"}'
    bad.write_bytes(original.encode("utf-8"))

    with pytest.raises(ConfigParseError) as info:
        ConfigurationManager(config_file=bad, apply_env=False)

    assert str(bad) in str(info.value), "the operator must be told which file"
    assert "nothing was changed" in str(info.value)
    assert bad.read_bytes() == original.encode("utf-8"), "the file is left as it is"
    assert sorted(p.name for p in d.iterdir()) == ["abstractcore.json"], "no copy, no backup, no rewrite"


def test_valid_config_loads_untouched_no_backup(caplog: pytest.LogCaptureFixture) -> None:
    d = Path(tempfile.mkdtemp())
    good = d / "abstractcore.json"
    good.write_text(
        json.dumps(
            {"embeddings": {"provider": "lmstudio", "model": "text-embedding-qwen3-embedding-0.6b"}}
        )
    )
    with caplog.at_level(logging.WARNING):
        cm = ConfigurationManager(config_file=good, apply_env=False)

    assert cm.config.embeddings.model == "text-embedding-qwen3-embedding-0.6b"
    assert cm.config.embeddings.provider == "lmstudio"
    assert sorted(p.name for p in d.iterdir()) == ["abstractcore.json"], "a valid config must never spawn a copy"


def test_missing_config_is_defaults_no_backup() -> None:
    d = Path(tempfile.mkdtemp())
    cm = ConfigurationManager(config_file=d / "abstractcore.json", apply_env=False)
    assert cm.config.embeddings.model  # default, non-empty
    assert not list(d.glob("*.bak")), "a missing config is not an error and needs no backup"


def test_unreadable_config_raises_instead_of_falling_back_to_defaults() -> None:
    """An I/O failure (a full disk, an EIO, an unreadable mount) is not evidence the file is bad:
    the load stops, the file stays, nothing is copied."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "abstractcore.json"
        config_file.write_text(json.dumps({"capability_defaults": {"routes": {}}}), encoding="utf-8")
        os.chmod(config_file, 0o000)
        try:
            with pytest.raises(OSError) as excinfo:
                ConfigurationManager(config_file=config_file)
            assert "Nothing was changed" in str(excinfo.value)
            assert sorted(p.name for p in Path(tmpdir).iterdir()) == ["abstractcore.json"]
        finally:
            os.chmod(config_file, 0o600)
