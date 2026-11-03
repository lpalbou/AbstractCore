"""The pointer head: scores each option against the question's decide token.

Two linear projections map hidden states (``d`` wide) to a pointer space
(``head_dim`` wide): ``q`` for the ``<decide>`` token and ``k`` for every
option's ``</opt>`` token. The logit of option ``i`` is
``k(h_i) . q(h_decide) / sqrt(head_dim) / temperature`` and a softmax over the
options of one question gives its probabilities. ``temperature`` is the single
calibration scalar fitted on the checkpoint's development data; it never
changes which option ranks first.

Weights are tiny (2 x 256 x 2560 for Kev-4B) and are applied in float32 numpy,
whatever precision the backbone runs in.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

HEAD_FILE = "head.safetensors"
CONFIG_FILE = "decision.json"


@dataclass
class PointerHead:
    q_weight: np.ndarray   # [head_dim, d]
    q_bias: np.ndarray     # [head_dim]
    k_weight: np.ndarray   # [head_dim, d]
    k_bias: np.ndarray     # [head_dim]
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.q_weight = np.ascontiguousarray(self.q_weight, dtype=np.float32)
        self.q_bias = np.ascontiguousarray(self.q_bias, dtype=np.float32)
        self.k_weight = np.ascontiguousarray(self.k_weight, dtype=np.float32)
        self.k_bias = np.ascontiguousarray(self.k_bias, dtype=np.float32)
        if self.q_weight.shape != self.k_weight.shape or self.q_weight.ndim != 2:
            raise ValueError(f"pointer head projections must share one [head_dim, d] shape, got {self.q_weight.shape} and {self.k_weight.shape}")
        if float(self.temperature) <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(self.temperature)

    @property
    def head_dim(self) -> int:
        return int(self.q_weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.q_weight.shape[1])

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    # --- from hidden states -------------------------------------------------

    def project(self, hidden: np.ndarray) -> np.ndarray:
        """[..., d] hidden states -> [..., 2*head_dim] = [q(h) | k(h)] (the layout the
        GGUF conversion stores as ``classifier.out_proj``)."""
        h = np.asarray(hidden, dtype=np.float32)
        q = h @ self.q_weight.T + self.q_bias
        k = h @ self.k_weight.T + self.k_bias
        return np.concatenate([q, k], axis=-1)

    def logits(self, h_decide: np.ndarray, h_options: np.ndarray) -> np.ndarray:
        """[d], [K, d] -> calibrated logits [K]."""
        q = np.asarray(h_decide, dtype=np.float32) @ self.q_weight.T + self.q_bias
        k = np.asarray(h_options, dtype=np.float32) @ self.k_weight.T + self.k_bias
        return (k @ q) * self.scale / self.temperature

    # --- from already projected vectors (GGUF path) --------------------------

    def logits_from_projected(self, p_decide: np.ndarray, p_options: np.ndarray) -> np.ndarray:
        """[2*head_dim], [K, 2*head_dim] -> calibrated logits [K]."""
        dp = self.head_dim
        q = np.asarray(p_decide, dtype=np.float32)[:dp]
        k = np.asarray(p_options, dtype=np.float32)[:, dp:]
        return (k @ q) * self.scale / self.temperature

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max()
        e = np.exp(z)
        return (e / e.sum()).astype(np.float32)

    def probs(self, h_decide: np.ndarray, h_options: np.ndarray) -> np.ndarray:
        return self.softmax(self.logits(h_decide, h_options))

    # --- persistence ----------------------------------------------------------

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"q.weight": self.q_weight, "q.bias": self.q_bias, "k.weight": self.k_weight, "k.bias": self.k_bias}

    def save(self, directory: str, extra_config: Optional[Dict[str, Any]] = None) -> None:
        """Write ``head.safetensors`` and ``decision.json`` into ``directory``."""
        from safetensors.numpy import save_file

        os.makedirs(directory, exist_ok=True)
        save_file(self.state_dict(), os.path.join(directory, HEAD_FILE), metadata={"format": "np"})
        config = {"format": "abstractcore-decision/1", "kind": "kev", "head_dim": self.head_dim,
                  "hidden_size": self.hidden_size, "temperature": self.temperature, "head_file": HEAD_FILE}
        if extra_config:
            config.update(extra_config)
        with open(os.path.join(directory, CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str, temperature: Optional[float] = None) -> "PointerHead":
        """From a converted checkpoint (``head.safetensors`` + ``decision.json``) or an
        original Kev checkpoint (``head.pt``, which needs torch to read)."""
        config_path = os.path.join(directory, CONFIG_FILE)
        head_path = os.path.join(directory, HEAD_FILE)
        if os.path.isfile(head_path):
            from safetensors.numpy import load_file

            tensors = load_file(head_path)
            config = {}
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
            t = temperature if temperature is not None else float(config.get("temperature", 1.0))
            return cls(tensors["q.weight"], tensors["q.bias"], tensors["k.weight"], tensors["k.bias"], temperature=t)
        pt_path = os.path.join(directory, "head.pt")
        if os.path.isfile(pt_path):
            return cls.from_head_pt(pt_path, temperature=temperature)
        raise FileNotFoundError(f"no pointer head in {directory}: expected {HEAD_FILE} or head.pt")

    @classmethod
    def from_head_pt(cls, path: str, temperature: Optional[float] = None) -> "PointerHead":
        """Read an original ``head.pt`` (a torch pickle holding the head's state dict and its temperature)."""
        try:
            import torch
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError("reading head.pt needs torch; use a converted checkpoint (head.safetensors) instead") from e
        meta = torch.load(path, map_location="cpu", weights_only=False)
        head = meta["head"]
        t = temperature if temperature is not None else float(meta.get("temperature", 1.0))
        return cls(
            head["q.weight"].float().numpy(), head["q.bias"].float().numpy(),
            head["k.weight"].float().numpy(), head["k.bias"].float().numpy(), temperature=t,
        )


def read_head_meta(path: str) -> Dict[str, Any]:
    """Everything else ``head.pt`` carries (base model, revision, LoRA rank, head_dim...)."""
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in meta.items() if k != "head"}


__all__ = ["PointerHead", "HEAD_FILE", "CONFIG_FILE", "read_head_meta"]
