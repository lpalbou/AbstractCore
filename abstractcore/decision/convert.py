"""Convert a Kev checkpoint (LoRA adapter + head.pt on a Qwen3.5 base) into
self-contained decision checkpoints AbstractCore loads directly:

- ``transformers``: the text backbone with the adapter folded in (bf16),
  ``head.safetensors`` + ``decision.json``, tokenizer files.
- ``mlx``: the same merged backbone in mlx-lm layout, bf16 or quantized
  (4/8 bit, optionally mixed per layer), with the same head and config.

    python -m abstractcore.decision.convert jaredpalmer/kev-4b --runtime mlx --bits 4 --out ./kev-4b-mlx-4bit
    python -m abstractcore.decision.convert jaredpalmer/kev-4b --runtime transformers --out ./kev-4b

Every converted folder carries ``decision.json`` (format, kind, base, revision,
temperature, head_dim, quantization recipe, source checkpoint sha) so a loader
and a reader can tell what it holds without opening the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .head import CONFIG_FILE, PointerHead, read_head_meta
from .loader import describe_decision_model, resolve_hf_snapshot_dir

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_info(adapter_dir: Path, base_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"adapter_dir": str(adapter_dir), "base_dir": str(base_dir)}
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if adapter_file.is_file():
        info["adapter_sha256"] = _sha256(adapter_file)
    head_file = adapter_dir / "head.pt"
    if head_file.is_file():
        info["head_sha256"] = _sha256(head_file)
    for d in (adapter_dir, base_dir):
        parts = d.parts
        if "snapshots" in parts:
            i = parts.index("snapshots")
            repo = parts[i - 1].replace("models--", "").replace("--", "/")
            info.setdefault("repos", {})[repo] = parts[i + 1]
    return info


def _copy_tokenizer(src: Path, dst: Path) -> None:
    for name in TOKENIZER_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)


def resolve_sources(model: str, allow_download: bool = False):
    """-> (adapter_dir, base_dir, head meta) for a Kev adapter checkpoint."""
    info = describe_decision_model(model)
    if info is None or info.layout != "adapter":
        if allow_download and info is None:
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(model)
            info = describe_decision_model(model)
        if info is None or info.layout != "adapter":
            raise ValueError(f"{model!r} is not a Kev adapter checkpoint (adapter_config.json + head.pt)")
    meta = read_head_meta(os.path.join(info.path, "head.pt"))
    base, revision = meta.get("base") or info.base, meta.get("base_revision") or info.base_revision
    base_dir = resolve_hf_snapshot_dir(base, revision=revision) or resolve_hf_snapshot_dir(base)
    if base_dir is None:
        if not allow_download:
            raise ValueError(f"base model {base!r} (revision {revision}) is not downloaded; pass allow_download=True")
        from huggingface_hub import snapshot_download  # type: ignore

        base_dir = Path(snapshot_download(base, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    return Path(info.path), Path(base_dir), meta


def _decision_config(meta: Dict[str, Any], head: PointerHead, runtime: str, adapter_dir: Path, base_dir: Path, extra: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {
        "runtime": runtime,
        "base": meta.get("base"),
        "base_revision": meta.get("base_revision"),
        "lora_rank": meta.get("lora"),
        "backbone": "merged",
        "delimiters": ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"],
        "question_types": ["choice", "noul", "score"],
        "max_state_tokens": 65536,
        "max_row_tokens": 65536 + 8192,
        "source": _source_info(adapter_dir, base_dir),
        "license": "apache-2.0",
    }
    cfg.update(extra)
    return cfg


# --- transformers -------------------------------------------------------------

def convert_transformers(model: str, out: Union[str, Path], *, allow_download: bool = False, dtype: str = "bf16") -> Path:
    """Merge the adapter into the base text backbone and save a transformers checkpoint."""
    import torch  # type: ignore
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

    from .backends.hf import HFDecisionBackend

    adapter_dir, base_dir, meta = resolve_sources(model, allow_download)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[dtype]
    config = AutoConfig.from_pretrained(str(base_dir), local_files_only=True)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    full = AutoModelForCausalLM.from_pretrained(str(base_dir), config=config, dtype=torch_dtype, local_files_only=True, attn_implementation="eager")
    backend = HFDecisionBackend.__new__(HFDecisionBackend)
    backend.torch = torch
    backend.lm = _text_model_of(full)
    merged = backend._merge_adapter(str(adapter_dir))
    # Save the text-only causal LM: drop vision towers so the checkpoint is a plain text backbone.
    text_lm = _text_causal_lm(full, text_config)
    text_lm.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True).save_pretrained(str(out))
    head = PointerHead.from_head_pt(str(adapter_dir / "head.pt"))
    head.save(str(out), _decision_config(meta, head, "transformers", adapter_dir, base_dir, {"dtype": dtype, "merged_tensors": merged}))
    return out


def _text_model_of(full: Any) -> Any:
    from .backends.hf import _text_model

    return _text_model(full)


def _text_causal_lm(full: Any, text_config: Any) -> Any:
    """The causal LM to save: the text model when the wrapper is multimodal (vision tower dropped)."""
    import torch  # type: ignore

    if not hasattr(full, "model") or not hasattr(full.model, "language_model"):
        return full
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING  # type: ignore

        cls = None
        for cfg_cls, model_cls in MODEL_FOR_CAUSAL_LM_MAPPING.items():
            if cfg_cls.__name__ == type(text_config).__name__ or getattr(cfg_cls, "model_type", None) == getattr(text_config, "model_type", None):
                cls = model_cls
                break
        if cls is None:
            return full
        text_lm = cls(text_config)
        text_lm.model.load_state_dict(full.model.language_model.state_dict(), strict=True)
        if hasattr(full, "lm_head") and hasattr(text_lm, "lm_head"):
            text_lm.lm_head.load_state_dict(full.lm_head.state_dict(), strict=True)
        return text_lm.to(next(full.parameters()).dtype)
    except Exception:
        return full


# --- mlx ------------------------------------------------------------------------

def mixed_predicate_factory(bits: int, group_size: int, upgrade: Optional[Dict[str, int]] = None) -> Callable[[str, Any], Union[bool, dict]]:
    """A quant predicate: everything at ``bits`` except modules whose path contains one of
    ``upgrade``'s keys, which get that many bits (8 for a sensitive projection, say)."""
    upgrade = upgrade or {}

    def predicate(path: str, module: Any) -> Union[bool, dict]:
        if not hasattr(module, "to_quantized"):
            return False
        for needle, b in upgrade.items():
            if needle in path:
                if b >= 16:
                    return False
                return {"bits": int(b), "group_size": int(group_size)}
        return {"bits": int(bits), "group_size": int(group_size)}

    return predicate


def convert_mlx(
    model: str,
    out: Union[str, Path],
    *,
    allow_download: bool = False,
    bits: Optional[int] = None,
    group_size: int = 64,
    upgrade: Optional[Dict[str, int]] = None,
    quant_predicate: Optional[Callable[[str, Any], Union[bool, dict]]] = None,
) -> Path:
    """Merge the adapter into the base on mlx-lm, optionally quantize, save."""
    import mlx.core as mx  # type: ignore
    from mlx.utils import tree_flatten  # type: ignore
    from mlx_lm.utils import load_model, quantize_model, save_config, save_model  # type: ignore

    from .backends.mlx import MLXDecisionBackend

    adapter_dir, base_dir, meta = resolve_sources(model, allow_download)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(1 << 30)
    mlx_model, config = load_model(base_dir, lazy=False)
    backend = MLXDecisionBackend.__new__(MLXDecisionBackend)
    backend.mx = mx
    backend.model = mlx_model
    merged = backend.merge_adapter(str(adapter_dir))
    # vision tower: never loaded by mlx-lm's qwen3_5 text model; drop its config so the folder is a text model
    config = dict(config)
    recipe: Dict[str, Any] = {"bits": None}
    if bits:
        predicate = quant_predicate or mixed_predicate_factory(bits, group_size, upgrade)
        mlx_model, config = quantize_model(mlx_model, config, group_size, bits, quant_predicate=predicate)
        recipe = {"bits": bits, "group_size": group_size, "upgrade": upgrade or {}}
    weights = dict(tree_flatten(mlx_model.parameters()))
    save_model(out, mlx_model, donate_model=True)
    save_config(config, config_path=out / "config.json")
    _copy_tokenizer(base_dir, out)
    head = PointerHead.from_head_pt(str(adapter_dir / "head.pt"))
    head.save(str(out), _decision_config(meta, head, "mlx", adapter_dir, base_dir, {"quantization_recipe": recipe, "merged_tensors": merged, "tensor_count": len(weights)}))
    mx.clear_cache()
    return out


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Kev decision checkpoint for AbstractCore.")
    parser.add_argument("model", help="adapter checkpoint (repo id or directory), e.g. jaredpalmer/kev-4b")
    parser.add_argument("--runtime", choices=["transformers", "mlx"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bits", type=int, default=None, help="mlx: quantize to this many bits (4 or 8); default bf16")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--upgrade", default=None, help='mlx: JSON {"path substring": bits} for modules kept at higher precision')
    parser.add_argument("--dtype", default="bf16", help="transformers: bf16 | fp16 | fp32")
    parser.add_argument("--download", action="store_true", help="download missing snapshots first")
    args = parser.parse_args(argv)
    upgrade = json.loads(args.upgrade) if args.upgrade else None
    if args.runtime == "mlx":
        out = convert_mlx(args.model, args.out, allow_download=args.download, bits=args.bits, group_size=args.group_size, upgrade=upgrade)
    else:
        out = convert_transformers(args.model, args.out, allow_download=args.download, dtype=args.dtype)
    print(f"wrote {out}")
    with open(out / CONFIG_FILE, encoding="utf-8") as f:
        print(f.read())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
