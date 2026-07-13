"""Config data-safety: an unparseable config is preserved, never silently
regenerated to defaults (incident 2026-07-11).

The old `_load_config` returned all-defaults on ANY read error, and the next
`_save_config` overwrote the recoverable file with defaults — silently
discarding operator settings (provider/model, embedding model, capability
routes) and reasserting the stale framework embedding default. These tests pin
the fix: a corrupt file is backed up (raw bytes) with a loud warning, and a
valid file loads untouched with no spurious backup.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from abstractcore.config.manager import ConfigurationManager


def test_corrupt_config_preserved_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    d = Path(tempfile.mkdtemp())
    bad = d / "abstractcore.json"
    original = '{ not valid json ,,, "embeddings": {"model": "text-embedding-qwen3-embedding-0.6b"}'
    bad.write_bytes(original.encode("utf-8"))

    with caplog.at_level(logging.WARNING):
        cm = ConfigurationManager(config_file=bad, apply_env=False)

    # A backup with the ORIGINAL bytes exists (nothing lost).
    backups = list(d.glob("abstractcore.json.corrupt-*.bak"))
    assert len(backups) == 1, "the unreadable config must be backed up exactly once"
    assert backups[0].read_bytes() == original.encode("utf-8"), "backup must preserve raw bytes"

    # The degradation is LOUD (never a silent defaults regeneration).
    warned = [r for r in caplog.records if "could not be parsed" in r.getMessage()]
    assert warned, "an unparseable config must warn loudly"
    assert "#FALLBACK" in warned[0].getMessage()

    # This session falls back to defaults (expected) — but the on-disk file the
    # operator can recover from is the backup, not a silent overwrite.
    assert cm.config.embeddings.model  # defaulted, non-empty


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
    assert not list(d.glob("*.bak")), "a valid config must never spawn a backup"
    assert not [r for r in caplog.records if "could not be parsed" in r.getMessage()]


def test_missing_config_is_defaults_no_backup() -> None:
    d = Path(tempfile.mkdtemp())
    cm = ConfigurationManager(config_file=d / "abstractcore.json", apply_env=False)
    assert cm.config.embeddings.model  # default, non-empty
    assert not list(d.glob("*.bak")), "a missing config is not an error and needs no backup"
