"""mlx-lm backend for decision models on Apple silicon.

Runs the Qwen3.5 hybrid text backbone through mlx-lm (Metal kernels for both
the attention and the Gated DeltaNet layers), either from a converted merged
checkpoint (bf16 or quantized, with ``head.safetensors`` next to it) or from
the base model with a Kev LoRA adapter folded in at load.

The state is prefilled once into an mlx-lm prompt cache; the question rows run
as one right-padded batch on a replicated copy of that cache, so the prefix
stays pristine and can serve the next request with the same state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from ..encoding import EncodedQuestion, rows_per_pass
from ..head import PointerHead

CACHE_LIMIT_BYTES = 1 << 30
PREFILL_CHUNK = 1024


@dataclass
class MLXPrefix:
    state_ids: List[int]
    cache: Any


class MLXDecisionBackend:
    name = "mlx"

    def __init__(
        self,
        model_path: str,
        head: PointerHead,
        *,
        adapter_path: Optional[str] = None,
        pad_id: int = 0,
        pass_token_budget: int = 16_384,
        lazy: bool = False,
    ):
        try:
            import mlx.core as mx  # type: ignore
            from mlx_lm.utils import load_model  # type: ignore
        except ImportError as e:
            raise ImportError('MLX decision models need mlx-lm: pip install "abstractcore[apple]"') from e
        self.mx = mx
        self.model_path = model_path
        self.head = head
        self.pad_id = int(pad_id)
        self.pass_token_budget = int(pass_token_budget)
        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        model, config = load_model(Path(model_path), lazy=lazy)
        self.model = model
        self.config = config
        if adapter_path:
            self.merge_adapter(adapter_path)
        self.text = model.language_model.model if hasattr(model, "language_model") else model.model
        if not hasattr(self.text, "layers") or not hasattr(self.text, "embed_tokens"):
            raise ValueError(f"{model_path}: not a Qwen3.5-style text backbone mlx-lm can run for decisions")
        self.precision = _precision_name(config)
        mx.eval(model.parameters())

    # --- loading helpers ------------------------------------------------------

    def merge_adapter(self, adapter_path: str, scale: float = 1.0) -> int:
        """Fold a PEFT LoRA adapter into the backbone: W + (B @ A) * alpha / r, computed in
        float32 on the CPU stream (exact) and rounded once to the stored dtype."""
        mx = self.mx
        from mlx.utils import tree_flatten  # type: ignore

        with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("trainable_token_indices") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            raise ValueError(f"{adapter_path}: only a plain LoRA adapter can be merged on the MLX backend")
        alpha = float(cfg["lora_alpha"]) / (float(cfg["r"]) ** 0.5 if cfg.get("use_rslora") else float(cfg["r"]))
        weights = mx.load(os.path.join(adapter_path, "adapter_model.safetensors"))
        params = dict(tree_flatten(self.model.parameters()))
        merged = {}
        with mx.stream(mx.cpu):
            for name, a in weights.items():
                if not name.endswith(".lora_A.weight"):
                    continue
                stem = name[: -len(".lora_A.weight")]
                suffix = stem.split("base_model.model.", 1)[-1] + ".weight"
                target = next((k for k in params if k == suffix or k.endswith("." + suffix)), None)
                if target is None:
                    raise ValueError(f"adapter tensor {stem} has no weight in the mlx-lm model")
                base = params[target]
                if "scales" in target or base.ndim != 2:
                    raise ValueError(f"{target} is quantized; merge the adapter before quantizing")
                delta = (weights[stem + ".lora_B.weight"].astype(mx.float32) @ a.astype(mx.float32)) * (alpha * scale)
                merged[target] = (base.astype(mx.float32) + delta).astype(base.dtype)
                mx.eval(merged[target])
        self.model.load_weights(list(merged.items()), strict=False)
        mx.eval(self.model.parameters())
        del params, weights
        mx.clear_cache()
        return len(merged)

    # --- scoring -------------------------------------------------------------

    def _hidden(self, rows: Sequence[Sequence[int]], cache: Any):
        mx = self.mx
        longest = max(len(r) for r in rows)
        ids = mx.array([list(r) + [self.pad_id] * (longest - len(r)) for r in rows], dtype=mx.int32)
        h = self.text(ids, cache=cache)
        mx.eval(h)
        return h

    def prefix(self, state_ids: Sequence[int]) -> MLXPrefix:
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache  # type: ignore

        state_ids = [int(t) for t in state_ids]
        cache = make_prompt_cache(self.model)
        for start in range(0, len(state_ids), PREFILL_CHUNK):
            self.text(mx.array([state_ids[start:start + PREFILL_CHUNK]], dtype=mx.int32), cache=cache)
            mx.eval([c.state for c in cache])
        return MLXPrefix(state_ids=state_ids, cache=cache)

    def question_logits(self, prefix: MLXPrefix, questions: Sequence[EncodedQuestion], kinds: Optional[Sequence[str]] = None) -> List[np.ndarray]:
        mx = self.mx
        n_state = len(prefix.state_ids)
        out: List[np.ndarray] = []
        chunk = rows_per_pass([q.ids for q in questions], n_state, self.pass_token_budget)
        for start in range(0, len(questions), chunk):
            part = questions[start:start + chunk]
            batch = [type(c).merge([c] * len(part)) for c in prefix.cache]   # merge copies: the prefix stays pristine
            hidden = self._hidden([q.ids for q in part], batch)
            for i, q in enumerate(part):
                picks = mx.array([q.decide, *q.option_ends], dtype=mx.int32)
                picked = np.asarray(hidden[i][picks].astype(mx.float32))
                out.append(self.head.logits(picked[0], picked[1:]))
            del batch, hidden
        return out

    def release(self, prefix: Any) -> None:
        return None

    def unload(self) -> None:
        self.model = None
        self.text = None
        self.mx.clear_cache()


def _precision_name(config: dict) -> str:
    quant = config.get("quantization") or config.get("quantization_config")
    if isinstance(quant, dict) and quant.get("bits"):
        mixed = any(isinstance(v, dict) and v.get("bits") not in (None, quant.get("bits")) for v in quant.values())
        return f"{quant['bits']}bit" + ("-mixed" if mixed else "")
    return str(config.get("dtype") or config.get("torch_dtype") or "bf16").replace("bfloat16", "bf16").replace("float16", "fp16").replace("float32", "fp32")


__all__ = ["MLXDecisionBackend", "MLXPrefix"]
