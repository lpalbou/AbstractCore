"""Backends that return next-token logits at the end of a prompt (label-logit engines),
with reuse of a shared prompt prefix.

Each exposes ``prefix(ids) -> handle`` (run the shared prefix once into a cache) and
``suffix_logits(handle, suffixes, candidate_ids) -> [np.ndarray]`` (one logit vector per
suffix, each suffix continuing from a copy of the cached prefix), plus the one-shot
``last_token_logits(ids, candidate_ids)``, ``name``, ``precision`` and ``unload()``.
The prefix is what every prompt of one document shares (system text and the state), so a
document's questions and option orders pay for the state once.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _lora_pairs(adapter_path: str):
    """(stem, A, B, scale) for every LoRA pair of a plain PEFT adapter (torch tensors)."""
    from safetensors.torch import load_file

    with open(os.path.join(adapter_path, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("peft_type", "LORA") != "LORA" or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern") or cfg.get("trainable_token_indices"):
        raise ValueError(f"{adapter_path}: only a plain LoRA adapter can be folded in")
    alpha = float(cfg["lora_alpha"]) / (float(cfg["r"]) ** 0.5 if cfg.get("use_rslora") else float(cfg["r"]))
    weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    for name, a in weights.items():
        if name.endswith(".lora_A.weight"):
            stem = name[: -len(".lora_A.weight")]
            yield stem, a, weights[stem + ".lora_B.weight"], alpha


def merge_lora_torch(module: Any, adapter_path: str, scale: float = 1.0) -> int:
    """W += scale * alpha/r * B @ A on a torch module (float32 math, one rounding)."""
    import torch

    params = dict(module.named_parameters())
    merged = 0
    with torch.no_grad():
        for stem, a, b, alpha in _lora_pairs(adapter_path):
            suffix = stem.split("base_model.model.", 1)[-1] + ".weight"
            matches = [n for n in params if n == suffix or n.endswith("." + suffix)]
            if len(matches) != 1:
                raise ValueError(f"adapter tensor {stem} has no unique weight in the backbone ({matches[:3]})")
            w = params[matches[0]]
            delta = (b.float() @ a.float()) * (alpha * scale)
            w.copy_((w.float() + delta.to(w.device)).to(w.dtype))
            merged += 1
    if not merged:
        raise ValueError(f"{adapter_path}: no LoRA tensor matched the backbone")
    return merged


@dataclass
class LogitsPrefix:
    ids: List[int]
    cache: Any


class HFLogitsBackend:
    name = "hf"

    def __init__(self, model_path: str, *, adapter_path: Optional[str] = None, adapter_scale: float = 1.0,
                 device: Optional[str] = None, dtype: Optional[str] = None, quantization_config: Any = None,
                 attn_implementation: Optional[str] = None, local_files_only: bool = True, pad_id: int = 0):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

        from .hf import _precision_name, _torch_dtype

        self.torch = torch
        self.DynamicCache = DynamicCache
        self.model_path = model_path
        self.pad_id = int(pad_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = _torch_dtype(torch, dtype) if dtype else (torch.float32 if self.device == "cpu" else torch.bfloat16)
        if adapter_path and quantization_config is not None:
            raise ValueError("a LoRA adapter cannot be folded into quantized weights; convert first")
        kwargs: Dict[str, Any] = {"dtype": torch_dtype, "local_files_only": local_files_only,
                                  "attn_implementation": attn_implementation or ("sdpa" if self.device == "cuda" else "eager")}
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
            kwargs["device_map"] = {"": self.device}
        config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)
        if adapter_path:
            merge_lora_torch(self.lm, adapter_path, adapter_scale)
        if quantization_config is None:
            self.lm.to(self.device)
        self.lm.eval()
        self.config = self.lm.config
        self.precision = _precision_name(torch_dtype, quantization_config)

    def last_token_logits(self, ids: Sequence[int], candidate_ids: Sequence[int]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            out = self.lm(input_ids=torch.tensor([list(ids)], device=self.device), use_cache=False)
            logits = out.logits[0, -1].float()
            return logits[torch.tensor(list(candidate_ids), device=self.device)].cpu().numpy()

    def prefix(self, ids: Sequence[int]) -> LogitsPrefix:
        torch = self.torch
        ids = [int(t) for t in ids]
        with torch.no_grad():
            cache = self.DynamicCache(config=self.config)
            self.lm(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True)
        return LogitsPrefix(ids=ids, cache=cache)

    def _replicate(self, cache: Any, batch: int) -> Any:
        from .hf import HFDecisionBackend

        return HFDecisionBackend._replicate_cache(self, cache, batch)   # same cache classes, same copy rules

    def suffix_logits(self, prefix: LogitsPrefix, suffixes: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> List[np.ndarray]:
        torch = self.torch
        n = len(prefix.ids)
        out: List[np.ndarray] = []
        cand = torch.tensor(list(candidate_ids), device=self.device)
        with torch.no_grad():
            for start in range(0, len(suffixes), 4):
                part = [list(s) for s in suffixes[start:start + 4]]
                longest = max(len(s) for s in part)
                ids = torch.full((len(part), longest), self.pad_id, dtype=torch.long, device=self.device)
                att = torch.zeros((len(part), n + longest), dtype=torch.long, device=self.device)
                att[:, :n] = 1
                pos = torch.zeros((len(part), longest), dtype=torch.long, device=self.device)
                for i, s in enumerate(part):
                    ids[i, : len(s)] = torch.tensor(s, device=self.device); att[i, n: n + len(s)] = 1
                    pos[i, : len(s)] = torch.arange(n, n + len(s), device=self.device)
                cache = self._replicate(prefix.cache, len(part))
                logits = self.lm(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=cache, use_cache=True).logits
                for i, s in enumerate(part):
                    out.append(logits[i, len(s) - 1].float()[cand].cpu().numpy())
                del cache, logits
        return out

    def release(self, prefix: Any) -> None:
        return None

    def unload(self) -> None:
        self.lm = None


class MLXLogitsBackend:
    name = "mlx"

    def __init__(self, model_path: str, *, adapter_path: Optional[str] = None, adapter_scale: float = 1.0, pad_id: int = 0):
        import mlx.core as mx
        from mlx_lm.utils import load_model

        from .mlx import MLXDecisionBackend, _precision_name

        self.mx = mx
        self.model_path = model_path
        self.pad_id = int(pad_id)
        mx.set_cache_limit(1 << 30)
        self.model, config = load_model(Path(model_path), lazy=False)
        if adapter_path:
            helper = MLXDecisionBackend.__new__(MLXDecisionBackend)
            helper.mx, helper.model = mx, self.model
            helper.merge_adapter(adapter_path, scale=adapter_scale)
        self.precision = _precision_name(config)
        mx.eval(self.model.parameters())

    def last_token_logits(self, ids: Sequence[int], candidate_ids: Sequence[int]) -> np.ndarray:
        mx = self.mx
        logits = self.model(mx.array([list(ids)], dtype=mx.int32))[0, -1]
        picked = logits[mx.array(list(candidate_ids), dtype=mx.int32)].astype(mx.float32)
        mx.eval(picked)
        return np.asarray(picked)

    def prefix(self, ids: Sequence[int]) -> LogitsPrefix:
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = [int(t) for t in ids]
        cache = make_prompt_cache(self.model)
        for start in range(0, len(ids), 1024):
            self.model(mx.array([ids[start:start + 1024]], dtype=mx.int32), cache=cache)
            mx.eval([c.state for c in cache])
        return LogitsPrefix(ids=ids, cache=cache)

    def suffix_logits(self, prefix: LogitsPrefix, suffixes: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> List[np.ndarray]:
        mx = self.mx
        out: List[np.ndarray] = []
        cand = mx.array(list(candidate_ids), dtype=mx.int32)
        for start in range(0, len(suffixes), 4):
            part = [list(s) for s in suffixes[start:start + 4]]
            longest = max(len(s) for s in part)
            ids = mx.array([s + [self.pad_id] * (longest - len(s)) for s in part], dtype=mx.int32)
            batch = [type(c).merge([c] * len(part)) for c in prefix.cache]
            logits = self.model(ids, cache=batch)
            for i, s in enumerate(part):
                picked = logits[i, len(s) - 1][cand].astype(mx.float32)
                mx.eval(picked)
                out.append(np.asarray(picked))
            del batch, logits
        return out

    def release(self, prefix: Any) -> None:
        return None

    def unload(self) -> None:
        self.model = None
        self.mx.clear_cache()


class GGUFLogitsBackend:
    """llama.cpp logits with prefix reuse: the prefix lives in sequence 0, every suffix runs on a
    copy of it (``llama_memory_seq_cp``, attention KV and recurrent state alike). Plain logits,
    so it works with every released llama-cpp-python."""

    name = "gguf"

    def __init__(self, model_path: str, *, n_ctx: int = 8192, n_batch: int = 2048, n_gpu_layers: int = -1,
                 max_parallel: int = 4, verbose: bool = False):
        import llama_cpp
        from llama_cpp import Llama

        from .gguf import _quant_from_filename

        self.llama_cpp = llama_cpp
        self.model_path = model_path
        self.n_batch = int(n_batch)
        self.max_parallel = max(1, int(max_parallel))
        # The Llama wrapper only sets n_seq_max / kv_unified in embedding mode, so it keeps a tiny
        # context of its own (tokenizer, vocab) and the decoding context is built here with the
        # sequence count the prefix copies need and one unified KV buffer (a copy shares cells).
        self.llm = Llama(model_path=model_path, n_ctx=64, n_batch=64, n_gpu_layers=int(n_gpu_layers), logits_all=False, verbose=verbose)
        params = llama_cpp.llama_context_default_params()
        params.n_ctx = int(n_ctx)
        params.n_batch = self.n_batch
        params.n_ubatch = self.n_batch
        params.n_seq_max = self.max_parallel + 1
        params.kv_unified = True
        params.embeddings = False
        params.n_threads = self.llm.context_params.n_threads
        params.n_threads_batch = self.llm.context_params.n_threads_batch
        self.ctx = llama_cpp.llama_init_from_model(self.llm._model.model, params)
        if not self.ctx:
            raise RuntimeError(f"llama.cpp could not create a {n_ctx}-token context with {self.max_parallel + 1} sequences for {model_path}")
        self.mem = llama_cpp.llama_get_memory(self.ctx)
        self.n_vocab = int(self.llm._n_vocab)
        self.n_ctx = int(llama_cpp.llama_n_ctx(self.ctx))
        self.precision = _quant_from_filename(model_path)
        self._resident: Optional[LogitsPrefix] = None

    @property
    def tokenizer(self):
        from ..tokenizers import LlamaTokenizer

        if not hasattr(self, "_tokenizer"):
            self._tokenizer = LlamaTokenizer(self.llm)
        return self._tokenizer

    def _decode(self, items: Sequence[Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """Decode runs on their sequences, chunked by n_batch; returns {seq: logits of its last token}
        for every item whose ``read`` flag is set."""
        lc = self.llama_cpp
        results: Dict[int, np.ndarray] = {}
        for item in items:
            tokens = item["tokens"]
            for start in range(0, len(tokens), self.n_batch):
                chunk = tokens[start:start + self.n_batch]
                batch = lc.llama_batch_init(len(chunk), 0, 1)
                try:
                    batch.n_tokens = len(chunk)
                    last = start + len(chunk) == len(tokens)
                    for j, tok in enumerate(chunk):
                        batch.token[j] = int(tok); batch.pos[j] = int(item["pos0"] + start + j)
                        batch.n_seq_id[j] = 1; batch.seq_id[j][0] = int(item["seq"])
                        batch.logits[j] = 1 if (last and item["read"] and j == len(chunk) - 1) else 0
                    rc = lc.llama_decode(self.ctx, batch)
                    if rc != 0:
                        raise RuntimeError(f"llama_decode failed with code {rc} (context {self.n_ctx} tokens; raise n_ctx?)")
                    if last and item["read"]:
                        ptr = lc.llama_get_logits_ith(self.ctx, len(chunk) - 1)
                        if not ptr:
                            raise RuntimeError("llama.cpp returned no logits for the last token")
                        results[item["seq"]] = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float32).copy()
                finally:
                    lc.llama_batch_free(batch)
        return results

    def last_token_logits(self, ids: Sequence[int], candidate_ids: Sequence[int]) -> np.ndarray:
        lc = self.llama_cpp
        lc.llama_memory_clear(self.mem, True); self._resident = None
        row = self._decode([{"seq": 0, "tokens": [int(t) for t in ids], "pos0": 0, "read": True}])[0]
        lc.llama_memory_clear(self.mem, True)
        return row[np.asarray(list(candidate_ids), dtype=np.int64)].copy()

    def prefix(self, ids: Sequence[int]) -> LogitsPrefix:
        ids = [int(t) for t in ids]
        if self._resident is not None and self._resident.ids == ids:
            return self._resident
        lc = self.llama_cpp
        lc.llama_memory_clear(self.mem, True)
        if len(ids) >= self.n_ctx:
            raise ValueError(f"prompt prefix of {len(ids):,} tokens exceeds this context ({self.n_ctx:,}); raise n_ctx")
        self._decode([{"seq": 0, "tokens": ids, "pos0": 0, "read": False}])
        self._resident = LogitsPrefix(ids=ids, cache=None)
        return self._resident

    def suffix_logits(self, prefix: LogitsPrefix, suffixes: Sequence[Sequence[int]], candidate_ids: Sequence[int]) -> List[np.ndarray]:
        if self._resident is not prefix:
            self.prefix(prefix.ids)
        lc = self.llama_cpp
        n = len(prefix.ids)
        cand = np.asarray(list(candidate_ids), dtype=np.int64)
        out: List[Optional[np.ndarray]] = [None] * len(suffixes)
        for start in range(0, len(suffixes), self.max_parallel):
            group = list(enumerate(suffixes))[start:start + self.max_parallel]
            if n + sum(len(s) for _, s in group) > self.n_ctx:
                raise ValueError(f"prefix plus suffixes need more than this context ({self.n_ctx:,} tokens); raise n_ctx")
            items = []
            for slot, (qi, s) in enumerate(group, start=1):
                lc.llama_memory_seq_cp(self.mem, 0, slot, -1, -1)
                items.append({"seq": slot, "tokens": [int(t) for t in s], "pos0": n, "read": True})
            rows = self._decode(items)
            for slot, (qi, s) in enumerate(group, start=1):
                out[qi] = rows[slot][cand].copy()
                lc.llama_memory_seq_rm(self.mem, slot, -1, -1)
        return [r for r in out if r is not None]

    def release(self, prefix: Any) -> None:
        return None

    def unload(self) -> None:
        try:
            if getattr(self, "ctx", None):
                self.llama_cpp.llama_free(self.ctx)
                self.ctx = None
            self.llm.close()
        except Exception:
            pass
        self.llm = None


__all__ = ["HFLogitsBackend", "MLXLogitsBackend", "GGUFLogitsBackend", "LogitsPrefix", "merge_lora_torch"]
