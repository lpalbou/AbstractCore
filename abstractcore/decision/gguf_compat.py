"""COMPATIBILITY SHIM — delete when abetlen/llama-cpp-python#2398 ships.

A pointer-head decision GGUF (``ggml-org/Kev-4B-GGUF``) carries two tensors a
pre-decision llama.cpp does not know (``cls.output.weight`` / ``cls.output.bias``,
the ``[q | k]`` projection of the head) plus ``<arch>.decision.*`` metadata.
llama.cpp learned them in ggml-org/llama.cpp#29818 (merged 2026-10-02); every
llama-cpp-python release up to 0.3.36 bundles an older llama.cpp and refuses the
file ("wrong number of tensors"). The bindings bump is abetlen/llama-cpp-python#2398.

Until that release is installed, this module derives — once, cached — a backbone
copy of the file without the head tensors and metadata (every other tensor is
copied byte for byte, quantization untouched) and stores the head as
``head.safetensors`` + ``decision.json`` next to it. The backbone loads on any
llama.cpp that knows the architecture; ``GGUFDecisionBackend`` then reads the
final-norm hidden states through the embeddings output and applies the head in
numpy. Same numbers as the in-graph head, measured against the b12032 build.

Removal: when ``llamacpp_has_decision_support()`` is true everywhere this package
supports, drop this module, the ``compat`` branch of ``backends/gguf.py`` and the
``gguf`` dependency it needs; the official file then loads as is.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..utils.model_cache import _llama_library_bytes
from .head import CONFIG_FILE, HEAD_FILE, PointerHead

UPSTREAM_PR = "abetlen/llama-cpp-python#2398 (llama.cpp >= b12032, ggml-org/llama.cpp#29818)"
_DECISION_MARKER = b".decision.block_count"   # an LLM_KV name only a post-#29818 libllama carries
_support_cache: Dict[str, bool] = {}


def llamacpp_has_decision_support() -> Optional[bool]:
    """Whether the installed llama-cpp-python bundles a llama.cpp that knows decision models.
    None when llama-cpp-python is not installed."""
    if "value" in _support_cache:
        return _support_cache["value"]
    blob = _llama_library_bytes()
    if blob is None:
        return None
    value = _DECISION_MARKER in blob
    _support_cache["value"] = value
    return value


def gguf_has_pointer_head(path: Path) -> bool:
    """Whether the file carries the ``cls.output.*`` head tensors (needs gguf-py)."""
    try:
        from gguf import GGUFReader
    except ImportError:
        return False
    try:
        reader = GGUFReader(str(path))
    except Exception:
        return False
    return any(t.name == "cls.output.weight" for t in reader.tensors)


def compat_home() -> Path:
    """Where derived backbone files live: a registered AbstractCore data home."""
    home = Path(os.path.expanduser("~/.abstractcore/models/decision-compat"))
    try:
        from ..utils.data_registry import DATA_HOME_KINDS, ensure_data_home_registered

        kind = "model-cache" if "model-cache" in DATA_HOME_KINDS else DATA_HOME_KINDS[0]
        ensure_data_home_registered("abstractcore-decision-compat", path=str(home), kind=kind, owner="abstractcore",
                                    safe_to_purge=True, description="Backbone copies of decision GGUFs for a pre-#29818 llama.cpp (re-derived on demand)")
    except Exception:
        pass
    home.mkdir(parents=True, exist_ok=True)
    return home


def _source_tag(path: Path) -> str:
    st = path.stat()
    return hashlib.sha256(f"{path.resolve()}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]


def derive_backbone(path: Path, out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """-> (backbone .gguf, directory holding head.safetensors + decision.json). Cached by source file identity."""
    try:
        from gguf import GGUFReader, GGUFWriter, GGUFValueType
        from gguf.quants import dequantize
    except ImportError as e:
        raise ImportError(
            "running a pointer-head decision GGUF on this llama-cpp-python needs the `gguf` package "
            f"(pip install gguf) until {UPSTREAM_PR} ships"
        ) from e

    path = Path(path)
    base = (out_dir or compat_home()) / f"{path.stem}-{_source_tag(path)}"
    backbone = base / f"{path.stem}.backbone.gguf"
    marker = base / "derived.json"
    if backbone.is_file() and (base / HEAD_FILE).is_file() and marker.is_file():
        return backbone, base
    base.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(str(path))
    arch = reader.fields["general.architecture"].contents()
    head = {t.name: t for t in reader.tensors if t.name.startswith("cls.")}
    if "cls.output.weight" not in head or "cls.output.bias" not in head:
        raise ValueError(f"{path} carries no cls.output.* head tensors")

    tmp = backbone.with_suffix(".tmp")
    writer = GGUFWriter(str(tmp), arch)
    try:
        for key, field in reader.fields.items():
            if key == "general.architecture" or key.startswith("GGUF."):
                continue
            if f"{arch}.decision." in key or key.endswith(".embedding_length_out") or key == "tokenizer.chat_template.systemone":
                continue
            vtype = field.types[0]
            sub = field.types[1] if vtype == GGUFValueType.ARRAY and len(field.types) > 1 else None
            writer.add_key_value(key, field.contents(), vtype, sub_type=sub)
        for tensor in reader.tensors:
            if tensor.name in head:
                continue
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
    finally:
        writer.close()
    os.replace(tmp, backbone)

    weight = dequantize(head["cls.output.weight"].data, head["cls.output.weight"].tensor_type).astype(np.float32)
    bias = np.asarray(head["cls.output.bias"].data, dtype=np.float32).reshape(-1)
    if weight.shape[0] != bias.shape[0] or weight.shape[0] % 2:
        raise ValueError(f"unexpected head shapes {weight.shape} / {bias.shape}")
    dp = weight.shape[0] // 2
    temperatures = {}
    for kind in ("choice", "noul", "score"):
        field = reader.fields.get(f"{arch}.decision.temperature.{kind}")
        if field is not None:
            temperatures[kind] = float(field.contents())
    temperature = temperatures.get("choice", 1.0)
    pointer = PointerHead(weight[:dp], bias[:dp], weight[dp:], bias[dp:], temperature=temperature)
    pointer.save(str(base), {"kind": "kev", "runtime": "gguf-compat", "source": {"gguf": str(path), "tag": _source_tag(path)},
                             "temperatures": temperatures, "upstream_fix": UPSTREAM_PR})
    marker.write_text(json.dumps({"source": str(path), "tag": _source_tag(path), "backbone": backbone.name, "upstream_fix": UPSTREAM_PR}, indent=2))
    return backbone, base


__all__ = ["llamacpp_has_decision_support", "gguf_has_pointer_head", "derive_backbone", "compat_home", "UPSTREAM_PR"]
