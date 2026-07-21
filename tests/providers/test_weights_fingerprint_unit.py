"""0817 axis 4: weights-identity fingerprint (unit contract).

The fingerprint must be CHEAP (never full-content hashing of multi-GB
weights), move when the weights' identity moves (revision swap, re-quant,
re-shard, header rewrite), and reserve "" for "not observable" (validators
abstain on it, never match).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from abstractcore.providers.weights_fingerprint import (
    check_weights_fingerprint,
    weights_fingerprint_for_dir,
    weights_fingerprint_for_file,
    weights_fingerprint_for_revision,
)


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Minimal safetensors writer: 8-byte little-endian header length + JSON
    header + zero-filled data region (content bytes irrelevant to the tier)."""
    offset = 0
    header = {}
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * offset)


def test_revision_tier_accepts_hub_shas_only() -> None:
    assert weights_fingerprint_for_revision("a" * 40) == f"weights-revision:{'a' * 40}"
    assert weights_fingerprint_for_revision("AB12cd3") == "weights-revision:ab12cd3"
    assert weights_fingerprint_for_revision("main") == ""
    assert weights_fingerprint_for_revision("") == ""
    assert weights_fingerprint_for_revision(None) == ""


def test_snapshot_dir_yields_revision_tier(tmp_path: Path) -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    snap = tmp_path / "models--org--name" / "snapshots" / sha
    snap.mkdir(parents=True)
    _write_safetensors(snap / "model.safetensors", {"w": ("F16", [2, 2], 8)})
    assert weights_fingerprint_for_dir(snap) == f"weights-revision:{sha}"


def test_fileset_tier_moves_on_header_rewrite_same_size(tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    _write_safetensors(model_dir / "model.safetensors", {"w": ("F16", [2, 2], 8)})
    base = weights_fingerprint_for_dir(model_dir)
    assert base.startswith("weights-fileset:sha256:")

    # Re-quantization rewrites the header (dtype) while sizes can collide:
    # same tensor byte count, different dtype label — the header digest moves.
    _write_safetensors(model_dir / "model.safetensors", {"w": ("Q8_0", [2, 2], 8)})
    requantized = weights_fingerprint_for_dir(model_dir)
    assert requantized != base, "same-size header rewrite must move the fingerprint"


def test_fileset_tier_moves_on_reshard(tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    _write_safetensors(model_dir / "model.safetensors", {"w": ("F16", [2, 2], 8)})
    base = weights_fingerprint_for_dir(model_dir)
    (model_dir / "model.safetensors").unlink()
    _write_safetensors(model_dir / "model-00001-of-00002.safetensors", {"w1": ("F16", [2], 4)})
    _write_safetensors(model_dir / "model-00002-of-00002.safetensors", {"w2": ("F16", [2], 4)})
    assert weights_fingerprint_for_dir(model_dir) != base


def test_fileset_tier_stable_across_calls(tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    _write_safetensors(model_dir / "model.safetensors", {"w": ("F16", [4], 8)})
    (model_dir / "config.json").write_text("{}")  # non-weight files ignored
    a = weights_fingerprint_for_dir(model_dir)
    (model_dir / "README.md").write_text("docs change, weights identity does not")
    b = weights_fingerprint_for_dir(model_dir)
    assert a == b


def test_gguf_tier_moves_on_header_edit(tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"metadata-region-A" + b"\x00" * 64)
    base = weights_fingerprint_for_file(gguf)
    assert base.startswith("weights-gguf:")
    gguf.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"metadata-region-B" + b"\x00" * 64)
    assert weights_fingerprint_for_file(gguf) != base


def test_unobservable_weights_are_empty(tmp_path: Path) -> None:
    assert weights_fingerprint_for_dir(tmp_path / "missing") == ""
    assert weights_fingerprint_for_file(tmp_path / "missing.gguf") == ""
    empty_dir = tmp_path / "no-weights"
    empty_dir.mkdir()
    (empty_dir / "config.json").write_text("{}")
    assert weights_fingerprint_for_dir(empty_dir) == "", "a dir with no weight files is unverifiable"


def test_verdict_contract() -> None:
    assert check_weights_fingerprint("", "") == "unverified_stored"
    assert check_weights_fingerprint("", "weights-revision:aa") == "unverified_stored"
    assert check_weights_fingerprint("weights-revision:aa", "") == "unverified_current"
    assert check_weights_fingerprint("weights-revision:aa", "weights-revision:aa") == "ok"
    assert check_weights_fingerprint("weights-revision:aa", "weights-fileset:sha256:bb") == "mismatch"
