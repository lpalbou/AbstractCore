"""Native Qwen4-Exp loading for MLX, including embedded MTP and PLE mmap.

No server, alternate inference framework, remote code, or copied model is
involved. Upstream mlx-vlm owns the model graph and speculative cache rollback;
AbstractCore owns artifact validation, lifecycle and the provider contract.
Imports are lazy so other providers do not require Apple Silicon packages.
"""

from __future__ import annotations

import json
import tempfile
import threading
import weakref
from pathlib import Path
from typing import Any
from .mlx_native_session import NativeSession

_SESSIONS = weakref.WeakValueDictionary()
_LOAD_LOCK = threading.RLock()


def checkpoint_config(path: str) -> dict:
    config = Path(path) / "config.json"
    return json.loads(config.read_text()) if config.is_file() else {}


def is_qwen4_checkpoint(path: str) -> bool:
    return checkpoint_config(path).get("model_type") == "qwen4_exp"


def embedded_mtp_keys(path: str) -> dict[str, str]:
    """Tensor-index evidence, never the model name or mtp_num_hidden_layers."""
    root = Path(path)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return {}
    document = json.loads(index.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("weight_map"), dict):
        raise ValueError("Invalid MTP tensor index: weight_map must be an object")
    weights = document["weight_map"]
    selected = {k: v for k, v in weights.items() if k.startswith("mtp.")}
    for filename in selected.values():
        if not isinstance(filename, str) or Path(filename).name != filename or not (root / filename).is_file():
            raise ValueError(f"Invalid/missing embedded MTP shard: {filename!r}")
    return selected


class Qwen4Session(NativeSession):
    """Shared weights, processor and exclusive generation ownership."""

    def __init__(self):
        super().__init__()
        self.storage: Any = None
        self.ple_offload = False


def _ScaledPLEEmbedding(embedding, scale):
    """Preserve a checkpoint's shared FP8-source scale after row dequantization.

    oQ conversions of the FP8 source quantize *unscaled* lookup rows to affine
    Q4. The shared scale is still essential, even though the rows are no longer
    FP8. Upstream's generic FP8 converter does not recognize this hybrid layout.
    """

    import mlx.core as mx
    import mlx.nn as nn

    class ScaledEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            self.scale = scale

        def __call__(self, indices):
            return self.embedding(indices).astype(mx.bfloat16) * self.scale

    return ScaledEmbedding()


def prepare_images(model, processor, prompt, media):
    """Native pixel tensors and measured delivery records; no temporary files."""
    import base64
    import io
    import mlx.core as mx
    from PIL import Image
    from mlx_vlm.utils import prepare_inputs

    images, records = [], []
    for index, part in media:
        path = getattr(part, "file_path", None)
        raw = Path(path).read_bytes() if path else part.content
        if isinstance(raw, str):
            raw = base64.b64decode(raw, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
        records.append({"index": index, "kind": "image", "content": raw,
                        "transport": "mlx_vlm_native"})
    inputs = dict(prepare_inputs(processor, images=images, prompts=prompt, add_special_tokens=False))
    grid = inputs.get("image_grid_thw")
    pixels = inputs.get("pixel_values")
    if grid is None or len(grid) != len(images) or pixels is None or pixels.size == 0:
        raise ValueError("Native Qwen4 image processor did not produce matching pixel grids")
    merge = int(model.config.vision_config.spatial_merge_size)
    if merge < 1 or any(t < 1 or h < 1 or w < 1 or h % merge or w % merge for t, h, w in grid.tolist()):
        raise ValueError("Native Qwen4 image grid dimensions are invalid")
    counts = [int(t * h * w) // (merge * merge) for t, h, w in grid.tolist()]
    actual = int(mx.sum(inputs["input_ids"] == model.config.image_token_id).item())
    if actual != sum(counts) or any(n <= 1 for n in counts):
        raise ValueError("Native Qwen4 image placeholders were not expanded correctly")
    for record, count in zip(records, counts):
        record["tokens"] = count
    if "attention_mask" in inputs:
        inputs["mask"] = inputs.pop("attention_mask")
    return inputs, records


def _qwen4_mtp_drafter_class(base):
    """Keep QSA indexer rows aligned when an MTP cohort loses a request.

    mlx-vlm 0.7.1's generic autoregressive MTP filter compacts KV and sampler
    state, but omits Qwen4's auxiliary indexer arrays. Use a native-owned
    subclass, not an upstream/global patch. If a later upstream implementation
    already replaces or invalidates an auxiliary array, preserve that result.
    """
    class NativeQwen4MTPDraftModel(base):
        def filter_batch(self, keep):
            import mlx.core as mx

            if not isinstance(keep, mx.array):
                keep = mx.array(keep, dtype=mx.int32)
            auxiliary = [
                (cache, name, getattr(cache, name, None))
                for cache in self._cache
                for name in ("index_keys", "index_position_ids", "index_block_keys")
            ]
            # Upstream remains the authority for KV, seed and position state.
            # Calling cache.filter() again here would double-filter those KV
            # rows (and break non-prefix subsets such as [1, 3]).
            super().filter_batch(keep)
            for cache, name, previous in auxiliary:
                if previous is not None and getattr(cache, name, None) is previous:
                    axis = 1 if name == "index_position_ids" and previous.ndim == 3 else 0
                    # Scalar drafter positions are shared across the cohort:
                    # [1,T] text or [3,1,T] MRoPE, not one position row per
                    # cache row. Keep that broadcast axis singleton so the next
                    # scalar-cursor forward can append its own [1,1] positions.
                    if name == "index_position_ids" and previous.shape[axis] == 1:
                        continue
                    setattr(cache, name, previous[:, keep] if axis == 1 else previous[keep])

    return NativeQwen4MTPDraftModel


def _load_embedded_drafter(source: Path, config: dict):
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors import safe_open
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import Qwen4ExpMTPDraftModel

    by_file: dict[str, list[str]] = {}
    for key, filename in embedded_mtp_keys(str(source)).items():
        by_file.setdefault(filename, []).append(key)
    if not by_file:
        raise ValueError("The checkpoint has no indexed embedded MTP tensors")
    weights = {}
    for filename, keys in by_file.items():
        try:
            with safe_open(source / filename, framework="mlx") as shard:
                selected = {key: shard.get_tensor(key) for key in keys}
        except TypeError:
            # Some safetensors builds cannot expose BF16 through numpy. MLX's
            # loader is lazy: only the selected head tensors are evaluated.
            shard_weights = mx.load(str(source / filename))
            selected = {key: shard_weights[key] for key in keys}
        weights.update(selected)
    drafter = _qwen4_mtp_drafter_class(Qwen4ExpMTPDraftModel)(Qwen4ExpMTPConfig.from_dict({
        "text_config": config["text_config"], "block_size": 2,
    }))
    weights = drafter.sanitize(weights)
    quantization = config.get("quantization") or {}
    if any(key.endswith(".scales") for key in weights):
        if not all(key in quantization for key in ("bits", "group_size")):
            raise ValueError("Quantized embedded MTP weights require quantization metadata")

        def predicate(path, module):
            if f"{path}.scales" not in weights:
                return False
            return quantization.get("mtp." + path, {
                "bits": quantization["bits"],
                "group_size": quantization["group_size"],
                "mode": quantization.get("mode", "affine"),
            })

        nn.quantize(drafter, class_predicate=predicate)
    drafter.load_weights(list(weights.items()), strict=True)
    drafter.eval()
    mx.eval(drafter.parameters())
    # This is an instance setting, not a patch of upstream global classes.
    # The caller's fixed depth must not become an adaptive ceiling silently.
    drafter.prefer_requested_block_size = True
    return drafter


def load_qwen4_session(path: str, *, mtp: bool, ple_offload: bool = True) -> Qwen4Session:
    """Load the existing checkpoint in-process without a second weight copy."""
    key = (str(Path(path).resolve()), ple_offload)
    with _LOAD_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            session = _create_qwen4_session(path, mtp=mtp, ple_offload=ple_offload)
            _SESSIONS[key] = session
        elif mtp:
            # The provider binds its immutable worker/head under config_lock.
            # Keep the runtime check and late-head publication in that same
            # critical section, including the potentially slow head load.
            # Lock order: registry -> config -> nonblocking direct-generation
            # lock. Never wait for generation while holding configuration.
            with session.config_lock:
                if session.drafter is None:
                    if session.runtime is not None:
                        raise RuntimeError("Cannot attach MTP to an existing target-only scheduled session; unload its providers first")
                    if not session.lock.acquire(blocking=False):
                        raise RuntimeError("Cannot attach MTP while native MLX generation is active; close its stream first")
                    try:
                        session.drafter = _load_embedded_drafter(Path(path), checkpoint_config(path))
                        session.draft_kind = "mtp"
                    finally:
                        session.lock.release()
        return session


def _create_qwen4_session(path: str, *, mtp: bool, ple_offload: bool) -> Qwen4Session:
    try:
        from mlx_vlm import load
        from mlx_vlm.models.qwen4_exp.ple_storage import build_quantized_ple_manifest
    except ImportError as exc:
        raise ImportError(
            "Qwen3.8-Flash-Next native MLX requires mlx-vlm>=0.7.1 and mlx>=0.32.2; "
            "install the updated abstractcore[mlx] extra in this interpreter"
        ) from exc
    source = Path(path).resolve()
    config = checkpoint_config(str(source))
    session = Qwen4Session()
    try:
        # Load and strictly validate the small head before allocating the target.
        if mtp:
            session.drafter = _load_embedded_drafter(source, config)
            session.draft_kind = "mtp"
        load_path = source
        if ple_offload:
            session.storage = tempfile.TemporaryDirectory(prefix="abstractcore-qwen4-")
            load_path = Path(session.storage.name)
            build_quantized_ple_manifest(source, load_path / "ple-store.json", cache_rows=4096)
            for item in source.iterdir():
                if item.is_file() and item.name != "config.json":
                    (load_path / item.name).symlink_to(item)
            config["text_config"]["ple_storage"] = {
                "manifest": "ple-store.json", "cache_rows": 4096,
            }
            (load_path / "config.json").write_text(json.dumps(config))
            session.ple_offload = True
        session.model, session.processor = _load_target(load_path)
        if session.drafter is not None:
            from mlx_vlm.speculative.drafters import validate_drafter_compatibility
            validate_drafter_compatibility(session.model, session.drafter, "mtp")
        return session
    except BaseException:
        session.model = session.processor = session.drafter = None
        if session.storage is not None:
            session.storage.cleanup()
            session.storage = None
        raise


def _load_target(path: Path):
    """Strict native loader preserving FP8-source scales on quantized PLE rows.

    A shared FP8-source scale on affine Q4 tables must be applied AFTER row
    dequantization. The generic loader mistakes this layout for raw FP8. Keep
    the scale separately, then delegate model/key sanitization to upstream.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp import Model, ModelConfig, VisionModel, LanguageModel
    from mlx_vlm.utils import load_config, load_processor, load_image_processor, sanitize_weights
    from mlx_vlm.models.qwen3_5.qwen3_5 import sanitize_key

    config = load_config(path)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = {}
    for filename in sorted(set(index["weight_map"].values())):
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid target tensor shard: {filename!r}")
        weights.update(mx.load(str(path / filename)))
    ple = config["text_config"].get("ple_storage")
    ple_scales = {}
    for key in list(weights):
        if key.endswith(".ngram_embedding.weight_scale"):
            prefix = key.removesuffix("weight_scale")
            if prefix + "shards.0.scales" in weights:
                ple_scales[sanitize_key(key.removesuffix(".weight_scale"))] = weights.pop(key)
    if ple:
        ple["manifest"] = str(path / ple["manifest"])
        weights = {k: v for k, v in weights.items() if ".ple.ple_embedding.ngram_embedding." not in k}
    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)
    weights = sanitize_weights(model, weights)
    weights = sanitize_weights(VisionModel, weights, model_config.vision_config)
    weights = sanitize_weights(LanguageModel, weights, model_config.text_config)
    quantization = model_config.quantization
    if quantization:
        def predicate(name, module):
            if name + ".scales" not in weights:
                return False
            return quantization.get(name, True)

        nn.quantize(model, group_size=quantization["group_size"], bits=quantization["bits"],
                    mode=quantization.get("mode", "affine"), class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    for name, scale in ple_scales.items():
        mx.eval(scale)
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], _ScaledPLEEmbedding(getattr(parent, parts[-1]), scale))
    mx.eval(model.parameters())
    model.eval()
    model.model_path = path
    processor = load_processor(path, True, eos_token_ids=model_config.eos_token_id, trust_remote_code=False)
    image_processor = load_image_processor(path, trust_remote_code=False)
    if image_processor is not None:
        processor.image_processor = image_processor
    return model, processor
