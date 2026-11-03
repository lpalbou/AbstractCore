"""transformers backend for decision models (CUDA, MPS or CPU).

Loads the text backbone of a Qwen3.5-family checkpoint — a converted merged
checkpoint, or the base model with a Kev LoRA adapter folded in at load — and
returns post-norm hidden states for the pointer head.

The state runs once into a ``DynamicCache`` (attention keys/values plus the
Gated DeltaNet conv/recurrent states); every question is a causal row that
continues from a replica of that cache, so the state is never recomputed and
questions never see each other.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..encoding import EncodedQuestion, rows_per_pass
from ..head import PointerHead


@dataclass
class HFPrefix:
    state_ids: List[int]
    cache: Any


class HFDecisionBackend:
    name = "hf"

    def __init__(
        self,
        model_path: str,
        head: PointerHead,
        *,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        quantization_config: Any = None,
        local_files_only: bool = True,
        pad_id: int = 0,
        pass_token_budget: int = 16_384,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache  # type: ignore
        except ImportError as e:
            raise ImportError("transformers decision models need torch and transformers") from e
        self.torch = torch
        self.DynamicCache = DynamicCache
        self.model_path = model_path
        self.head = head
        self.pad_id = int(pad_id)
        self.pass_token_budget = int(pass_token_budget)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = _torch_dtype(torch, dtype) if dtype else (torch.float32 if self.device == "cpu" else torch.bfloat16)
        if adapter_path and quantization_config is not None:
            raise ValueError("a LoRA adapter cannot be folded into quantized weights; convert the adapter into a merged checkpoint first")
        load_kwargs: Dict[str, Any] = {"dtype": torch_dtype, "local_files_only": local_files_only}
        load_kwargs["attn_implementation"] = attn_implementation or ("sdpa" if self.device == "cuda" else "eager")
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = {"": self.device}
        config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        full = AutoModelForCausalLM.from_pretrained(model_path, config=config, **load_kwargs)
        self.lm = _text_model(full)
        self.config = self.lm.config
        if adapter_path:
            self._merge_adapter(adapter_path)
        if quantization_config is None:
            self.lm.to(self.device)
        self.lm.eval()
        self.precision = _precision_name(torch_dtype, quantization_config)
        self._quantized = quantization_config is not None

    # --- loading helpers ------------------------------------------------------

    def _merge_adapter(self, adapter_path: str) -> int:
        """W += (alpha / r) * B @ A for every adapter tensor, computed in float32 and
        rounded once to the backbone dtype. Returns the number of merged weights."""
        torch = self.torch
        from safetensors.torch import load_file

        with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("peft_type") != "LORA" or cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("lora_bias") or cfg.get("rank_pattern") or cfg.get("alpha_pattern") or cfg.get("trainable_token_indices"):
            raise ValueError(f"{adapter_path}: only a plain LoRA adapter (no DoRA/rsLoRA/bias/rank patterns/token training) can be merged")
        scale = float(cfg["lora_alpha"]) / float(cfg["r"])
        weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
        params = dict(self.lm.named_parameters())
        merged = 0
        with torch.no_grad():
            for name, a in weights.items():
                if not name.endswith(".lora_A.weight"):
                    continue
                stem = name[: -len(".lora_A.weight")]
                b = weights[stem + ".lora_B.weight"]
                target = _adapter_target(stem, params)
                w = params[target]
                delta = (b.float() @ a.float()) * scale
                if delta.shape != w.shape:
                    raise ValueError(f"adapter tensor {stem} has shape {tuple(delta.shape)}, weight {target} has {tuple(w.shape)}")
                w.copy_((w.float() + delta.to(w.device)).to(w.dtype))
                merged += 1
        if merged == 0:
            raise ValueError(f"{adapter_path}: no LoRA tensors matched the backbone")
        return merged

    # --- scoring -------------------------------------------------------------

    def _pad_rows(self, rows: Sequence[Sequence[int]], pos0: int):
        torch = self.torch
        longest = max(len(r) for r in rows)
        ids = torch.full((len(rows), longest), self.pad_id, dtype=torch.long, device=self.device)
        pos = torch.zeros((len(rows), longest), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), longest), dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(list(row), device=self.device)
            pos[i, : len(row)] = torch.arange(pos0, pos0 + len(row), device=self.device)
            att[i, : len(row)] = 1
        return ids, pos, att

    def _replicate_cache(self, cache: Any, batch: int) -> Any:
        """A batch-``batch`` copy of a batch-1 cache; the original stays untouched."""
        torch = self.torch
        from transformers.cache_utils import LinearAttentionCacheLayerMixin  # type: ignore

        replica = copy.copy(cache)
        replica.layers = [copy.copy(layer) for layer in cache.layers]
        for source, target in zip(cache.layers, replica.layers):
            if isinstance(source, LinearAttentionCacheLayerMixin):
                target.conv_states = dict(source.conv_states)
                target.recurrent_states = dict(source.recurrent_states)
                target.is_conv_states_initialized = dict(source.is_conv_states_initialized)
                target.is_recurrent_states_initialized = dict(source.is_recurrent_states_initialized)
                target.has_previous_state = dict(source.has_previous_state)
                target.conv_kernel_size = dict(source.conv_kernel_size)
        replica.reorder_cache(torch.zeros(batch, dtype=torch.long, device=self.device))
        return replica

    def prefix(self, state_ids: Sequence[int]) -> HFPrefix:
        torch = self.torch
        state_ids = [int(t) for t in state_ids]
        with torch.no_grad():
            ids = torch.tensor([state_ids], device=self.device)
            pos = torch.arange(len(state_ids), device=self.device)[None]
            cache = self.DynamicCache(config=self.config)
            self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True)
        return HFPrefix(state_ids=state_ids, cache=cache)

    def question_logits(self, prefix: HFPrefix, questions: Sequence[EncodedQuestion], kinds: Optional[Sequence[str]] = None) -> List[np.ndarray]:
        torch = self.torch
        n_state = len(prefix.state_ids)
        out: List[np.ndarray] = []
        chunk = rows_per_pass([q.ids for q in questions], n_state, self.pass_token_budget)
        with torch.no_grad():
            for start in range(0, len(questions), chunk):
                part = questions[start:start + chunk]
                ids, pos, att = self._pad_rows([q.ids for q in part], n_state)
                att = torch.cat([torch.ones((len(part), n_state), dtype=torch.long, device=self.device), att], 1)
                cache = self._replicate_cache(prefix.cache, len(part))
                hidden = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, past_key_values=cache, use_cache=True).last_hidden_state.float()
                for i, q in enumerate(part):
                    h = hidden[i]
                    h_decide = h[q.decide].cpu().numpy()
                    h_opts = h[torch.tensor(q.option_ends, device=self.device)].cpu().numpy()
                    out.append(self.head.logits(h_decide, h_opts))
                del cache, hidden
        return out

    def release(self, prefix: Any) -> None:
        return None

    def unload(self) -> None:
        self.lm = None
        try:
            if self.device == "mps":
                self.torch.mps.empty_cache()
            elif self.device == "cuda":
                self.torch.cuda.empty_cache()
        except Exception:
            pass


def _text_model(full: Any) -> Any:
    """The text backbone (embed_tokens -> layers -> norm) of a causal-LM wrapper."""
    for candidate in (getattr(full, "model", None), full):
        if candidate is None:
            continue
        for sub in (candidate, getattr(candidate, "language_model", None)):
            if sub is not None and hasattr(sub, "layers") and hasattr(sub, "embed_tokens"):
                return sub
    raise ValueError(f"cannot find the text backbone inside {type(full).__name__}")


def _adapter_target(stem: str, params: Dict[str, Any]) -> str:
    """peft names the wrapped model ``base_model.model.<module>``; find that module's weight."""
    suffix = stem.split("base_model.model.", 1)[-1] + ".weight"
    if suffix in params:
        return suffix
    matches = [name for name in params if name.endswith("." + suffix) or name == suffix]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"adapter tensor {stem} has no unique weight in the backbone (candidates: {matches[:3]})")


def _torch_dtype(torch: Any, name: str) -> Any:
    table = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
             "fp32": torch.float32, "float32": torch.float32}
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; use bf16, fp16 or fp32")
    return table[name]


def _precision_name(torch_dtype: Any, quantization_config: Any) -> str:
    if quantization_config is not None:
        cfg = quantization_config.to_dict() if hasattr(quantization_config, "to_dict") else dict(quantization_config)
        if cfg.get("load_in_4bit"):
            return f"bnb-4bit-{cfg.get('bnb_4bit_quant_type', 'nf4')}"
        if cfg.get("load_in_8bit"):
            return "bnb-8bit"
        return str(cfg.get("quant_method", "quantized"))
    return str(torch_dtype).replace("torch.", "").replace("bfloat16", "bf16").replace("float32", "fp32").replace("float16", "fp16")


__all__ = ["HFDecisionBackend", "HFPrefix"]
