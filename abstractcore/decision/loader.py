"""Finding and loading decision models.

A decision model comes in three layouts:

- ``adapter``: the original Kev release — a LoRA adapter plus ``head.pt`` on a
  Qwen3.5 base (``jaredpalmer/kev-4b``). Loading needs the base snapshot too;
  the adapter is folded into the base at load.
- ``merged``: an AbstractCore conversion — the backbone with the adapter folded
  in (bf16 or quantized, transformers or mlx-lm layout), ``head.safetensors``
  and ``decision.json`` next to it. Loads directly.
- ``gguf``: llama.cpp's conversion (``ggml-org/Kev-4B-GGUF``) — the merged
  backbone with the pointer head inside the file.

Detection is file-based (cache-only, no network) with the model-capabilities
catalog as a fallback for a handle that is not downloaded yet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..exceptions import ModelNotFoundError
from ..utils.model_cache import (
    _GGUF_MAGIC,
    _GGUF_TYPE_STRING,
    _read_exact,
    _read_gguf_scalar,
    _read_gguf_string,
    _read_u32,
    _read_u64,
    _skip_gguf_value,
    is_gguf_model_ref,
    resolve_hf_snapshot_dir,
)
from .head import CONFIG_FILE, HEAD_FILE, PointerHead
from .runtime import DecisionModel

HF_PROVIDERS = {"huggingface", "hf", "transformers"}
GGUF_PROVIDERS = {"gguf", "llamacpp", "llama.cpp", "llama_cpp"}
MLX_PROVIDERS = {"mlx"}


@dataclass
class DecisionModelInfo:
    model_id: str
    layout: str                      # adapter | merged | gguf
    path: str                        # snapshot dir, local dir or .gguf file
    kind: str = "kev"
    base: Optional[str] = None       # adapter layout: the base repo id
    base_revision: Optional[str] = None
    base_path: Optional[str] = None  # adapter layout: the base snapshot (when present locally)
    runtime: Optional[str] = None    # merged layout: "transformers" | "mlx"
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return os.path.exists(self.path)


# --- GGUF metadata ---------------------------------------------------------

_GGUF_TYPE_FLOAT32 = 6


def read_gguf_metadata(path: Path, keys: Optional[set] = None, max_keys: int = 400) -> Dict[str, Any]:
    """Decision-relevant metadata of a GGUF header (best effort, cache-only): the
    architecture, every ``<arch>.decision.*`` key (strings and float32 temperatures),
    the ``systemone`` chat template and ``embedding_length_out``; ``keys`` adds more."""
    out: Dict[str, Any] = {}
    wanted = {"general.architecture", "tokenizer.chat_template.systemone"} | set(keys or ())

    def interesting(key: str) -> bool:
        return key in wanted or ".decision." in key or key.endswith(".embedding_length_out")

    try:
        with Path(path).open("rb") as f:
            if _read_exact(f, 4) != _GGUF_MAGIC:
                return out
            _read_u32(f)
            _read_u64(f)
            kv_count = _read_u64(f)
            for _ in range(min(kv_count, max_keys)):
                key = _read_gguf_string(f)
                value_type = _read_u32(f)
                if interesting(key) and value_type == _GGUF_TYPE_STRING:
                    out[key] = _read_gguf_string(f)
                elif interesting(key) and value_type in (4, 5, 6, 10, 11, 12):   # uint32, int32, float32, uint64, int64, float64
                    out[key] = _read_gguf_scalar(f, value_type)
                else:
                    _skip_gguf_value(f, value_type)
    except Exception:
        return out
    return out


def gguf_decision_type(path: Path) -> Optional[str]:
    meta = read_gguf_metadata(path, keys={"general.architecture"})
    arch = meta.get("general.architecture")
    if not arch:
        return None
    return meta.get(f"{arch}.decision.type")


# --- resolution ----------------------------------------------------------------

def _resolve_gguf_file(model: str) -> Optional[Path]:
    head, _, selector = model.partition(":")
    p = Path(head).expanduser()
    if p.is_file() and p.suffix.lower() == ".gguf":
        return p
    candidates = []
    if p.is_dir():
        candidates = sorted(p.rglob("*.gguf"))
    else:
        snapshot = resolve_hf_snapshot_dir(head)
        if snapshot is not None:
            candidates = sorted(snapshot.rglob("*.gguf"))
    if not candidates:
        return None
    if selector:
        sel = selector.strip().upper()
        for match in (lambda c: sel == c.name.upper(), lambda c: sel in c.name.upper()):
            picked = [c for c in candidates if match(c)]
            if picked:
                return picked[0]
        return None
    # Prefer a 4-bit file when the repo offers several: smallest useful default.
    for pref in ("Q4_K_M", "Q4_K", "Q8_0", "BF16", "F16"):
        picked = [c for c in candidates if pref in c.name.upper()]
        if picked:
            return picked[0]
    return candidates[0]


def _resolve_dir(model: str) -> Optional[Path]:
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    return resolve_hf_snapshot_dir(model)


def _adapter_base(snapshot: Path) -> tuple:
    with (snapshot / "adapter_config.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    revision = cfg.get("revision")
    training = snapshot / "training_config.json"
    if revision is None and training.is_file():
        try:
            with training.open(encoding="utf-8") as f:
                revision = json.load(f).get("args", {}).get("base_revision")
        except Exception:
            revision = None
    return base, revision


def describe_decision_model(model: str, provider: Optional[str] = None) -> Optional[DecisionModelInfo]:
    """What ``model`` is, from local files (no network). None when it is not a decision model."""
    model = str(model or "").strip()
    if not model:
        return None
    if is_gguf_model_ref(model):
        path = _resolve_gguf_file(model)
        if path is None:
            return _from_catalog(model, layout="gguf")
        meta = read_gguf_metadata(path)
        arch = meta.get("general.architecture")
        dtype = meta.get(f"{arch}.decision.type") if arch else None
        if dtype is None:
            return None
        return DecisionModelInfo(model_id=model, layout="gguf", path=str(path), kind=str(dtype), config=meta)
    directory = _resolve_dir(model)
    if directory is None:
        return _from_catalog(model, layout="adapter")
    config_path = directory / CONFIG_FILE
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        runtime = config.get("runtime") or ("mlx" if (directory / "config.json").is_file() and _is_mlx_dir(directory) else "transformers")
        return DecisionModelInfo(model_id=model, layout="merged", path=str(directory), kind=config.get("kind", "kev"),
                                 base=config.get("base"), base_revision=config.get("base_revision"), runtime=runtime, config=config)
    if (directory / "head.pt").is_file() and (directory / "adapter_config.json").is_file():
        base, revision = _adapter_base(directory)
        base_dir = resolve_hf_snapshot_dir(base, revision=revision) if base else None
        return DecisionModelInfo(model_id=model, layout="adapter", path=str(directory), base=base, base_revision=revision,
                                 base_path=str(base_dir) if base_dir else None)
    if (directory / "lev_release.json").is_file() and (directory / "adapter_config.json").is_file():
        with (directory / "lev_release.json").open(encoding="utf-8") as f:
            manifest = json.load(f)
        base = manifest.get("base_model") or _adapter_base(directory)[0]
        base_dir = resolve_hf_snapshot_dir(base) if base else None
        return DecisionModelInfo(model_id=model, layout="adapter", path=str(directory), kind="lev", base=base,
                                 base_path=str(base_dir) if base_dir else None, config=manifest)
    if (directory / "rl_agent_config.json").is_file() and (directory / "encoder").is_dir():
        return DecisionModelInfo(model_id=model, layout="merged", path=str(directory), kind="laya", runtime="transformers")
    registry_kind = _registry_kind(model)
    if registry_kind:
        runtime = "mlx" if _is_mlx_dir(directory) else "transformers"
        return DecisionModelInfo(model_id=model, layout="merged", path=str(directory), kind=registry_kind, runtime=runtime)
    if (directory / "helper" / "shim.py").is_file() and (directory / "config.json").is_file():
        return DecisionModelInfo(model_id=model, layout="merged", path=str(directory), kind="openjev", runtime="transformers")
    return None


def _registry_kind(model: str) -> Optional[str]:
    """The decision kind the capabilities registry declares for a handle (``decision.kind``), if any."""
    try:
        from ..architectures.detection import lookup_registry_model_capabilities

        caps = lookup_registry_model_capabilities(model) or {}
    except Exception:
        return None
    block = caps.get("decision") if isinstance(caps, dict) else None
    kind = block.get("kind") if isinstance(block, dict) else None
    return str(kind) if kind else None


def _is_mlx_dir(directory: Path) -> bool:
    try:
        with (directory / "config.json").open(encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return False
    quant = cfg.get("quantization")
    return isinstance(quant, dict) or "mlx" in str(directory).lower()


def _from_catalog(model: str, layout: str) -> Optional[DecisionModelInfo]:
    """A handle that is not on disk: the capabilities catalog may still know it is a decision model."""
    try:
        from ..providers.model_capabilities import get_model_capability_routes

        routes = get_model_capability_routes(model)
    except Exception:
        return None
    if "output.decision" not in routes:
        return None
    return DecisionModelInfo(model_id=model, layout=layout, path=model, kind=_registry_kind(model) or "kev")


def is_decision_model(model: str, provider: Optional[str] = None) -> bool:
    try:
        return describe_decision_model(model, provider) is not None
    except Exception:
        return False


# --- loading ----------------------------------------------------------------------

def _download(repo: str, revision: Optional[str] = None, patterns: Optional[list] = None) -> str:
    from huggingface_hub import snapshot_download  # type: ignore

    return snapshot_download(repo, revision=revision, allow_patterns=patterns)


def load_decision_model(
    provider: str,
    model: str,
    *,
    allow_download: bool = False,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    quantization_config: Any = None,
    temperature: Optional[float] = None,
    n_ctx: int = 16_384,
    n_gpu_layers: int = -1,
    prefix_cache_size: int = 4,
    truncate_states: bool = False,
    **kwargs: Any,
) -> DecisionModel:
    """Load ``model`` for ``provider`` ("huggingface" for transformers or GGUF, "mlx").

    Cache-only by default: a handle that is not downloaded raises
    ModelNotFoundError naming what to download. ``allow_download=True`` fetches
    the snapshot(s) first (the base model too, for an adapter).
    """
    provider_key = str(provider or "").strip().lower()
    info = describe_decision_model(model, provider_key)
    if info is None or not info.is_local:
        if not allow_download:
            raise ModelNotFoundError(
                f"{model!r} is not a downloaded decision model. Download it first "
                f"(abstractcore.download_model(provider={provider_key!r}, model={model!r})) "
                "or pass allow_download=True.",
            )
        if is_gguf_model_ref(model):
            repo, _, selector = model.partition(":")
            _download(repo, patterns=[f"*{selector}*" if selector else "*Q4_K_M*", "*.json", "*.md"])
        else:
            _download(model)
        info = describe_decision_model(model, provider_key)
        if info is None:
            raise ModelNotFoundError(f"{model!r} was downloaded but carries no decision model (no decision.json, head.pt or decision GGUF metadata)")
    if info.kind in ("lev", "openjev"):
        return _load_label_logit_model(provider_key, model, info, allow_download=allow_download, device=device, dtype=dtype,
                                       quantization_config=quantization_config, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **kwargs)
    if info.kind == "laya":
        if info.layout == "gguf":
            raise ValueError(f"{model!r}: Laya GGUFs are not supported yet; use the transformers checkpoint convaiinnovations/laya on the huggingface provider")
        if provider_key in MLX_PROVIDERS:
            raise ValueError(f"{model!r}: Laya runs on the huggingface provider (transformers); mlx-lm has no ModernBERT encoder")
        from .engines.laya import LayaDecisionModel

        return LayaDecisionModel(info.path, device=device, dtype=dtype or "fp32")
    if info.kind != "kev":
        raise ValueError(f"{model!r} is a {info.kind!r} decision model, which AbstractCore does not run yet (kinds: kev, lev, openjev, laya, systemone)")

    if info.layout == "gguf":
        from .backends.gguf import GGUFDecisionBackend

        backend = GGUFDecisionBackend(info.path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, temperature=temperature,
                                      max_parallel_questions=int(kwargs.get("max_parallel_questions", 8)))
        return DecisionModel(backend, backend.tokenizer, model_id=model, provider=provider_key or "huggingface",
                             prefix_cache_size=1, truncate_states=truncate_states)

    if info.layout == "adapter":
        if info.base_path is None:
            if not allow_download:
                raise ModelNotFoundError(
                    f"{model!r} is a LoRA adapter on {info.base!r}, which is not downloaded. Download the base "
                    f"(abstractcore.download_model(provider={provider_key!r}, model={info.base!r})) or pass allow_download=True."
                )
            info.base_path = _download(info.base, revision=info.base_revision, patterns=["*.json", "*.safetensors", "*.txt"])
        head = PointerHead.from_head_pt(os.path.join(info.path, "head.pt"), temperature=temperature)
        backbone_path, adapter_path, tokenizer_dir = info.base_path, info.path, info.base_path
    else:
        head = PointerHead.load(info.path, temperature=temperature)
        backbone_path, adapter_path, tokenizer_dir = info.path, None, info.path

    from .tokenizers import load_hf_tokenizer

    tokenizer = load_hf_tokenizer(tokenizer_dir, local_files_only=True)

    if provider_key in MLX_PROVIDERS or (not provider_key and info.runtime == "mlx"):
        if info.layout == "merged" and info.runtime == "transformers":
            raise ValueError(f"{model!r} is a transformers checkpoint; the MLX provider needs an mlx-lm conversion (see abstractcore.decision.convert)")
        from .backends.mlx import MLXDecisionBackend

        backend = MLXDecisionBackend(backbone_path, head, adapter_path=adapter_path, pad_id=tokenizer.pad_id)
        return DecisionModel(backend, tokenizer, model_id=model, provider="mlx",
                             prefix_cache_size=prefix_cache_size, truncate_states=truncate_states)

    if provider_key in HF_PROVIDERS or not provider_key:
        if info.layout == "merged" and info.runtime == "mlx":
            raise ValueError(f"{model!r} is an mlx-lm checkpoint; use provider='mlx' or a transformers conversion")
        from .backends.hf import HFDecisionBackend

        backend = HFDecisionBackend(backbone_path, head, adapter_path=adapter_path, device=device, dtype=dtype,
                                    attn_implementation=kwargs.get("attn_implementation"),
                                    quantization_config=quantization_config, pad_id=tokenizer.pad_id)
        return DecisionModel(backend, tokenizer, model_id=model, provider="huggingface",
                             prefix_cache_size=prefix_cache_size, truncate_states=truncate_states)

    raise ValueError(f"provider {provider!r} cannot run decision models; use 'huggingface' (transformers or GGUF) or 'mlx'")


__all__ = ["DecisionModelInfo", "describe_decision_model", "is_decision_model", "load_decision_model", "read_gguf_metadata", "gguf_decision_type"]


def _label_spec_for(info: DecisionModelInfo):
    from .kinds import lev_spec, openjev_spec

    if info.layout == "gguf":
        meta = info.config or {}
        arch = meta.get("general.architecture", "")
        temps = {}
        prefix = f"{arch}.decision.temperature."
        for key, value in meta.items():
            if key.startswith(prefix):
                try:
                    temps[key[len(prefix):]] = float(value)
                except (TypeError, ValueError):
                    pass
        spec = lev_spec() if info.kind == "lev" else openjev_spec()
        if temps:
            spec.temperatures = temps
        template = meta.get("tokenizer.chat_template.systemone")
        if isinstance(template, str) and template.strip():
            spec.template = template
        return spec
    if info.kind == "lev":
        calibration = {}
        path = os.path.join(info.path, "calibration.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                calibration = json.load(f).get("temperatures", {})
        manifest = info.config or {}
        return lev_spec(calibration, adapter_scale=float(manifest.get("adapter_scale", 0.25)), base=info.base)
    return openjev_spec()


def _load_label_logit_model(provider_key: str, model: str, info: DecisionModelInfo, *, allow_download: bool, device, dtype,
                            quantization_config, n_ctx: int, n_gpu_layers: int, **kwargs: Any):
    from .engines.labels import LabelLogitDecisionModel
    from .tokenizers import load_hf_tokenizer

    spec = _label_spec_for(info)
    if kwargs.get("adapter_scale") is not None:
        # The Lev release manifest applies its adapter at 0.25 (the author's benchmarks); llama.cpp's
        # GGUF conversion merges it at full strength. Callers can pick either for the adapter layout.
        spec.adapter_scale = float(kwargs["adapter_scale"])
    if info.layout == "gguf":
        from .backends.logits import GGUFLogitsBackend

        backend = GGUFLogitsBackend(info.path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        return LabelLogitDecisionModel(backend, backend.tokenizer, spec, model_id=model, provider="huggingface")
    adapter_path = None
    if info.layout == "adapter":
        if info.base_path is None:
            if not allow_download:
                raise ModelNotFoundError(f"{model!r} is a LoRA adapter on {info.base!r}, which is not downloaded. Download the base first or pass allow_download=True.")
            info.base_path = _download(info.base, patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        backbone_path, adapter_path = info.base_path, info.path
    else:
        backbone_path = info.path
    tokenizer = load_hf_tokenizer(backbone_path, local_files_only=True)
    if provider_key in MLX_PROVIDERS or (not provider_key and info.runtime == "mlx"):
        from .backends.logits import MLXLogitsBackend

        backend = MLXLogitsBackend(backbone_path, adapter_path=adapter_path, adapter_scale=spec.adapter_scale, pad_id=tokenizer.pad_id)
        return LabelLogitDecisionModel(backend, tokenizer, spec, model_id=model, provider="mlx")
    from .backends.logits import HFLogitsBackend

    backend = HFLogitsBackend(backbone_path, adapter_path=adapter_path, adapter_scale=spec.adapter_scale, device=device, dtype=dtype,
                              quantization_config=quantization_config, attn_implementation=kwargs.get("attn_implementation"), pad_id=tokenizer.pad_id)
    return LabelLogitDecisionModel(backend, tokenizer, spec, model_id=model, provider="huggingface")
