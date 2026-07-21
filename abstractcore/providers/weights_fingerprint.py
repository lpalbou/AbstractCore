"""Weights identity for KV-artifact validity (backlog 0817, axis 4).

A saved KV artifact holds K/V tensors COMPUTED BY one set of weights. A
checkpoint swap under the SAME model id — a force-pushed HF revision, a
re-quantized GGUF, edited safetensors shards — can leave the rendered text,
the tokenizer, and the config all IDENTICAL while the projections that
produced the cached tensors are gone (runtime c1734: swapped checkpoint under
the same id was accepted). This axis records a CHEAP identity of the weights
at compile and refuses reuse when it moves.

Cheapness is a hard constraint: weight files are multi-GB, so full-content
hashing on the hot path is forbidden. Tiered subjects, strongest cheap signal
first, with the tier encoded in the prefix (cross-tier compares are unequal →
fail-safe mismatch → one recompile records the stronger tier):

1. ``weights-revision:`` — the HF hub commit sha when observable (the
   ``snapshots/<sha>`` path segment of a cache-resolved model, or
   transformers' ``config._commit_hash``). Content-addressed by the hub;
   zero I/O.
2. ``weights-fileset:`` — for a local model directory: sorted
   (relative name, size) of every weight file, plus the sha256 of each
   safetensors HEADER (bounded read: the header carries tensor names,
   shapes, dtypes and offsets, so re-quantization and re-sharding move it
   even when file sizes collide). Honest limit: a same-size, same-header
   pure value edit is invisible — catching it requires full-content
   hashing, which this axis deliberately does not do.
3. ``weights-gguf:`` — for a single GGUF file: file size + sha256 of the
   first bounded slice (the GGUF header + metadata region: architecture,
   quantization layout, tensor table — a re-quant rewrites it).

"" is reserved for "weights not observable" (model not loaded, no local
path): validators must treat it as "cannot verify", never as "matches".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from .tokenizer_fingerprint import check_tokenizer_fingerprint as _shared_verdict

__all__ = [
    "weights_fingerprint_for_dir",
    "weights_fingerprint_for_file",
    "weights_fingerprint_for_revision",
    "check_weights_fingerprint",
]

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".npz", ".pt", ".pth")
# Above this many weight files, skip per-file header hashing (names+sizes
# stay); keeps the fileset tier bounded on exotic many-shard layouts.
_HEADER_HASH_MAX_FILES = 64
# safetensors headers are small (tensor table JSON); refuse absurd lengths so
# a corrupt length field can never trigger a huge read.
_SAFETENSORS_HEADER_MAX = 32 * 1024 * 1024
# GGUF identity slice: magic + version + tensor table + metadata live at the
# front of the file; 1 MiB covers them for every model in the wild.
_GGUF_SLICE = 1024 * 1024

_SNAPSHOT_SHA_RE = re.compile(r"/snapshots/([0-9a-f]{7,64})(?:/|$)")


def _safetensors_header_digest(path: Path) -> Optional[str]:
    """sha256 of a safetensors file's header (bounded read), None on any miss."""
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                return None
            header_len = int.from_bytes(raw_len, "little")
            if not (0 < header_len <= _SAFETENSORS_HEADER_MAX):
                return None
            header = fh.read(header_len)
            if len(header) != header_len:
                return None
            return hashlib.sha256(header).hexdigest()[:16]
    except OSError:
        return None


def weights_fingerprint_for_revision(revision: Optional[str]) -> str:
    """Tier 1: a hub commit sha (content-addressed upstream)."""
    text = str(revision or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,64}", text):
        return ""
    return f"weights-revision:{text}"


def weights_fingerprint_for_file(path) -> str:
    """Tier 3: single weight file (GGUF and friends) — size + header slice."""
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return ""
        size = p.stat().st_size
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            digest.update(fh.read(_GGUF_SLICE))
        return f"weights-gguf:{size}:{digest.hexdigest()[:24]}"
    except OSError:
        return ""


def weights_fingerprint_for_dir(path) -> str:
    """Tier 1 when the directory IS a hub snapshot, else tier 2 (fileset).

    The ``snapshots/<sha>`` path segment of an HF-cache-resolved model names
    the hub commit — content-addressed identity for free. Any other local
    directory fingerprints its weight-file set (sorted relative name + size,
    plus safetensors header digests when the file count is bounded).
    """
    try:
        p = Path(path).expanduser()
        if not p.is_dir():
            return ""
    except OSError:
        return ""

    match = _SNAPSHOT_SHA_RE.search(str(p.resolve()).replace("\\", "/") + "/")
    if match:
        return weights_fingerprint_for_revision(match.group(1))

    try:
        files = sorted(
            f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in _WEIGHT_SUFFIXES
        )
    except OSError:
        return ""
    if not files:
        return ""

    digest = hashlib.sha256()
    hash_headers = len(files) <= _HEADER_HASH_MAX_FILES
    for f in files:
        try:
            rel = f.relative_to(p).as_posix()
            size = f.stat().st_size
        except (OSError, ValueError):
            continue
        digest.update(rel.encode("utf-8", errors="replace"))
        digest.update(b"\x01")
        digest.update(str(size).encode("ascii"))
        if hash_headers and f.suffix.lower() == ".safetensors":
            header = _safetensors_header_digest(f)
            if header:
                digest.update(b"\x02")
                digest.update(header.encode("ascii"))
        digest.update(b"\x00")
    return f"weights-fileset:sha256:{digest.hexdigest()[:24]}"


def check_weights_fingerprint(stored, current) -> str:
    """Three-way verdict shared with axes 2/3 (identical contract):

    - "mismatch": both known and different — the cached tensors were computed
      by weights that are gone; refuse (recompile or raise), never reload.
    - "unverified_stored": pre-axis artifact — reuse LABELED, never silent.
    - "unverified_current": weights not observable right now — abstain; a
      later gate that can see them re-checks.
    - "ok": both known and equal.
    """
    return _shared_verdict(stored, current)
