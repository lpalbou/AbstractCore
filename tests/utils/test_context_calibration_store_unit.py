"""Unit tests for the context-calibration store (`abstractcore.utils.context_calibration`).

The store is a rebuildable cache: atomic writes, exact-key lookups, tolerant
of corruption, capped, and raise-free.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.utils import context_calibration as cal_mod
from abstractcore.utils.context_calibration import (
    calibration_path,
    lookup_context_calibration,
    record_context_calibration,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_CALIBRATION_DIR", str(tmp_path / "calibration"))
    yield


def _entry(**overrides):
    entry = {
        "provider": "huggingface",
        "model_id": "model.gguf",
        "requested_context": 131072,
        "settled_context": 32768,
        "device_total_bytes": 137438953472,
        "ram_total_bytes": 137438953472,
        "rungs_tried": [131072, 65536, 32768],
    }
    entry.update(overrides)
    return entry


def test_calibration_path_respects_env_override(tmp_path) -> None:
    assert calibration_path() == tmp_path / "calibration" / "context_calibration.json"


def test_record_and_lookup_roundtrip() -> None:
    record_context_calibration(_entry())

    hit = lookup_context_calibration("huggingface", "model.gguf", 137438953472, 137438953472)

    assert hit is not None
    assert hit["settled_context"] == 32768
    assert hit["requested_context"] == 131072
    assert hit["rungs_tried"] == [131072, 65536, 32768]
    assert isinstance(hit["ts"], float)

    # Exact-key matching: a different hardware key is a MISS, never a guess.
    assert lookup_context_calibration("huggingface", "model.gguf", 1, 137438953472) is None
    assert lookup_context_calibration("huggingface", "other.gguf", 137438953472, 137438953472) is None
    assert lookup_context_calibration("mlx", "model.gguf", 137438953472, 137438953472) is None


def test_record_replaces_same_key_entry() -> None:
    record_context_calibration(_entry(settled_context=32768))
    record_context_calibration(_entry(settled_context=16384))

    hit = lookup_context_calibration("huggingface", "model.gguf", 137438953472, 137438953472)
    assert hit is not None and hit["settled_context"] == 16384

    data = json.loads(calibration_path().read_text(encoding="utf-8"))
    assert len(data["entries"]) == 1


def test_saved_file_is_valid_json_with_no_tmp_leftovers() -> None:
    record_context_calibration(_entry())

    path = calibration_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert isinstance(data["entries"], list)
    leftovers = [p for p in path.parent.iterdir() if ".tmp." in p.name]
    assert leftovers == []


def test_corrupt_file_starts_fresh_and_never_raises() -> None:
    path = calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json !!!", encoding="utf-8")

    assert lookup_context_calibration("huggingface", "model.gguf", 1, 1) is None
    record_context_calibration(_entry())

    hit = lookup_context_calibration("huggingface", "model.gguf", 137438953472, 137438953472)
    assert hit is not None and hit["settled_context"] == 32768


def test_record_rejects_incomplete_entries_silently() -> None:
    record_context_calibration({"provider": "huggingface"})  # no model_id/settled
    record_context_calibration(_entry(settled_context=None))
    record_context_calibration(_entry(settled_context=0))
    record_context_calibration("not-a-dict")  # type: ignore[arg-type]

    assert not calibration_path().exists()


def test_store_caps_entries_dropping_oldest(monkeypatch) -> None:
    monkeypatch.setattr(cal_mod, "_MAX_ENTRIES", 5)
    fake_now = [1000.0]
    monkeypatch.setattr(cal_mod.time, "time", lambda: fake_now[0])

    for i in range(8):
        fake_now[0] += 1.0
        record_context_calibration(_entry(model_id=f"model-{i}.gguf"))

    data = json.loads(calibration_path().read_text(encoding="utf-8"))
    kept = sorted(e["model_id"] for e in data["entries"])
    assert kept == [f"model-{i}.gguf" for i in range(3, 8)]


def test_gguf_calibration_model_id_derivation(tmp_path) -> None:
    from abstractcore.utils.context_calibration import gguf_calibration_model_id

    # Direct paths and snapshot-ROOT files: bare basename (a revision hash or
    # arbitrary parent dir in the id would fracture keys across re-downloads).
    assert gguf_calibration_model_id(tmp_path / "model.gguf") == "model.gguf"
    root = tmp_path / "hub" / "models--org--repo" / "snapshots" / "abc123" / "model.gguf"
    assert gguf_calibration_model_id(root) == "model.gguf"

    # Files nested one level under the snapshot (quant-anonymous multi-part
    # layouts) include the quant dir — bare basenames would collide.
    nested = tmp_path / "hub" / "models--org--repo" / "snapshots" / "abc123" / "UD-Q3_K_XL" / "model-00001-of-00003.gguf"
    assert gguf_calibration_model_id(nested) == "UD-Q3_K_XL/model-00001-of-00003.gguf"

    # Never raises on junk.
    assert isinstance(gguf_calibration_model_id(None), str)


def test_lookup_never_raises_on_unreadable_dir(monkeypatch) -> None:
    def _boom() -> None:
        raise RuntimeError("no path")

    monkeypatch.setattr(cal_mod, "calibration_path", _boom)

    assert lookup_context_calibration("huggingface", "model.gguf", 1, 1) is None
    record_context_calibration(_entry())  # must not raise either
