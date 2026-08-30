"""Synthesize tiny, VALID GGUF v3 files for adversarial MTP-detection tests.

The point of these fixtures is to divorce the *name* of a GGUF from its
*contents*, so a detector that sniffs the filename can be caught:

  * ``build_gguf`` with ``nextn=True``  + an innocuous filename  -> must be MTP
  * ``build_gguf`` with ``nextn=False`` + a filename full of "MTP" -> must NOT be

Layout follows the GGUF v3 spec, and was cross-checked byte-for-byte against
the real ``Qwen3.8-27B-Q4_K_M.gguf`` on this machine (866 tensors, 39 KV pairs,
``qwen35.nextn_predict_layers=1``, ``blk.64.nextn.{eh_proj,enorm,hnorm,
shared_head_norm}.weight``).

No model weights, no network, a few KB on disk.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# GGUF value type tags (subset we emit).
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# ggml tensor dtype tags.
GGML_TYPE_F32 = 0

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32

# The four nextn tensors the real Qwen3.8-27B GGUF carries, on its last block.
REAL_NEXTN_TENSOR_NAMES: Tuple[str, ...] = (
    "blk.64.nextn.eh_proj.weight",
    "blk.64.nextn.enorm.weight",
    "blk.64.nextn.hnorm.weight",
    "blk.64.nextn.shared_head_norm.weight",
)

# The metadata key the real file carries alongside them.
REAL_NEXTN_METADATA_KEY = "qwen35.nextn_predict_layers"


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _u64(len(raw)) + raw


def _kv(key: str, value: Any) -> bytes:
    """Encode one key/value pair. Supports str, int, and list-of-str."""
    out = _gguf_string(key)
    if isinstance(value, str):
        return out + _u32(GGUF_TYPE_STRING) + _gguf_string(value)
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        raise TypeError("bool KV values are not needed by these fixtures")
    if isinstance(value, int):
        return out + _u32(GGUF_TYPE_UINT32) + _u32(value)
    if isinstance(value, (list, tuple)):
        body = _u32(GGUF_TYPE_ARRAY) + _u32(GGUF_TYPE_STRING) + _u64(len(value))
        for item in value:
            body += _gguf_string(str(item))
        return out + body
    raise TypeError(f"unsupported KV value type: {type(value)!r}")


def build_gguf(
    path: Path,
    *,
    architecture: str = "qwen35",
    nextn: bool = False,
    nextn_predict_layers: int = 1,
    extra_metadata: Optional[Dict[str, Any]] = None,
    extra_tensors: Iterable[str] = (),
    include_nextn_metadata: bool = True,
    include_nextn_tensors: bool = True,
) -> Path:
    """Write a minimal but structurally valid GGUF file to ``path``.

    Args:
        architecture: value for ``general.architecture``.
        nextn: when True, add MTP evidence (metadata key and/or tensors).
        nextn_predict_layers: value for ``<arch>.nextn_predict_layers``.
        extra_metadata: additional KV pairs to embed.
        extra_tensors: additional tensor names to declare.
        include_nextn_metadata: with ``nextn=True``, emit the metadata key.
        include_nextn_tensors: with ``nextn=True``, emit the nextn tensors.
            Setting one of these False builds a "half-evidence" file, which is
            how we probe whether a detector needs BOTH signals or either one.

    Returns:
        The path written.
    """
    metadata: Dict[str, Any] = {
        "general.architecture": architecture,
        "general.name": path.stem,
        # Geometry keys the repo's read_gguf_geometry looks for, so these
        # fixtures stay usable by the existing reader too.
        f"{architecture}.block_count": 4,
        f"{architecture}.embedding_length": 64,
        f"{architecture}.context_length": 2048,
        f"{architecture}.attention.head_count": 4,
        f"{architecture}.attention.head_count_kv": 2,
    }
    if nextn and include_nextn_metadata:
        metadata[f"{architecture}.nextn_predict_layers"] = nextn_predict_layers
    if extra_metadata:
        metadata.update(extra_metadata)

    tensor_names: List[str] = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "output_norm.weight",
    ]
    if nextn and include_nextn_tensors:
        tensor_names.extend(REAL_NEXTN_TENSOR_NAMES)
    tensor_names.extend(extra_tensors)

    # Every tensor is a tiny 1-D F32 vector; 8 floats = 32 bytes = one
    # alignment unit, so offsets stay aligned without extra padding.
    elems = 8
    tensor_nbytes = elems * 4

    header = bytearray()
    header += GGUF_MAGIC
    header += _u32(GGUF_VERSION)
    header += _u64(len(tensor_names))
    header += _u64(len(metadata))
    for key, value in metadata.items():
        header += _kv(key, value)

    offset = 0
    for name in tensor_names:
        header += _gguf_string(name)
        header += _u32(1)  # n_dims
        header += _u64(elems)  # dims[0]
        header += _u32(GGML_TYPE_F32)
        header += _u64(offset)
        offset += tensor_nbytes

    # Pad the header out to the alignment boundary, then append tensor data.
    pad = (-len(header)) % GGUF_ALIGNMENT
    header += b"\x00" * pad
    body = b"\x00" * (tensor_nbytes * len(tensor_names))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + body)
    return path


def build_trap_pair(directory: Path) -> Tuple[Path, Path]:
    """Build the two files that break a filename-based MTP detector.

    Returns ``(evidenced, decoy)``:

      * ``evidenced`` -- named ``Qwen3.8-27B-Q4_K_M.gguf`` (NO "mtp" anywhere in
        the name, exactly like the real local file), but carrying the nextn
        metadata key and the four nextn tensors. A filename sniffer says "no
        MTP" and is WRONG.
      * ``decoy`` -- named ``Qwen3.6-27B-MTP-Q4_K_M.gguf`` (screams MTP) with no
        nextn metadata and no nextn tensors. A filename sniffer says "MTP" and
        is WRONG. This mirrors the real local Qwen3.6 GGUFs, which report zero
        nextn tensors.
    """
    evidenced = build_gguf(
        directory / "Qwen3.8-27B-Q4_K_M.gguf",
        architecture="qwen35",
        nextn=True,
    )
    decoy = build_gguf(
        directory / "Qwen3.6-27B-MTP-Q4_K_M.gguf",
        architecture="qwen35",
        nextn=False,
    )
    return evidenced, decoy
