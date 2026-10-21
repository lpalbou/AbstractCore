# Copyright 2026 Laurent-Philippe Albou / AbstractFramework. Apache-2.0.
#
# HY V3 (Tencent Hunyuan V3, model_type "hy_v3") for mlx-lm.
#
# Why this file exists: mlx-lm (<= 0.32.2 at the time of writing) ships no
# `mlx_lm.models.hy_v3` module (upstream PR ml-explore/mlx-lm#1211 was closed
# unmerged), so `mlx_lm.load` cannot build Tencent Hy-Memory-30B-A3B or Hy3.
#
# The same file serves two roles:
#   1. inside AbstractCore it is `abstractcore.providers.mlx_hy_v3`, registered
#      as `mlx_lm.models.hy_v3` (see `register_with_mlx_lm`) only when the
#      installed mlx-lm has no module of that name, so a future upstream module
#      always wins;
#   2. copied verbatim as `hy_v3.py` next to converted weights, with
#      `"model_file": "hy_v3.py"` in config.json, it lets PLAIN mlx-lm load the
#      checkpoint (`mlx_lm.utils.load_model` honours `model_file`).
# It therefore imports mlx-lm only through absolute `mlx_lm.models.*` paths.
#
# Reference for every numeric choice: transformers' modeling_hy_v3.py (5.8.0):
#   * attention: GQA, per-head RMSNorm on q and k BEFORE RoPE, default RoPE
#     (rotate-half), scale head_dim ** -0.5, no bias;
#   * layer 0 is a dense SwiGLU MLP (intermediate_size); every "sparse" layer is
#     a MoE: router logits = x @ W.T computed in FLOAT32, sigmoid scores, the
#     expert-score correction bias is added ONLY to choose the top-k experts,
#     the combine weights are the UNBIASED sigmoid scores of the chosen experts,
#     normalised to sum 1 (route_norm, +1e-20) and multiplied by
#     router_scaling_factor; one always-on shared expert is added, in float32
#     when enable_moe_fp32_combine;
#   * Hy-Memory-30B-A3B ships routed AND shared experts 768 wide
#     (config "expert_hidden_dim") while "moe_intermediate_size" says 1536; the
#     tensors are the truth, so expert_hidden_dim wins when present.
# Two deliberate deviations from transformers, both toward MORE precision (so a
# small bf16 parity drift against transformers is expected, not a regression):
#   * the routed experts are summed in float32 (transformers accumulates them in
#     the model dtype with index_add_ and uses float32 only for routed + shared);
#   * enable_lm_head_fp32 is honoured for an unquantized lm_head (transformers
#     ignores the flag).
# Dense layers when config has no mlp_layer_types: the first
# `first_k_dense_replace` layers (the checkpoint's own field, as in mlx-lm PR
# #1211); transformers' HYV3Config builds exactly one. Equal for every released
# config (Hy-Memory-30B-A3B, Hy3, Hy3-preview: 1).
# Parameter names follow the mlx-lm MoE convention used by
# mlx-community/Hy3-preview-* and the PR #1211 head: `mlp.switch_mlp.*`,
# `mlp.router.gate`, `mlp.router.expert_bias`, `mlp.shared_mlp.*`. `sanitize`
# maps the tencent per-expert layout, the transformers stacked layout and this
# module's own earlier names (`mlp.experts.*`, `mlp.expert_bias`) onto them.
# Cross-checked against hermitdave/Hy-Memory-30B-A3B-MLX-8bit and
# Duckets/Hy-Memory-30B-A3B-MLX-4bit (community hy_v3.py) and the PR #1211 head
# (router matmul in fp32, router + correction bias kept unquantized).

# NO `from __future__ import annotations`: mlx-lm loads a bundled model_file with
# importlib.util.module_from_spec WITHOUT registering it in sys.modules, and
# dataclasses resolves string annotations through sys.modules[cls.__module__]
# (AttributeError: 'NoneType' object has no attribute '__dict__').
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.switch_layers import SwitchGLU

MODEL_TYPE = "hy_v3"
_UPSTREAM_MODULE = "mlx_lm.models.hy_v3"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = MODEL_TYPE
    vocab_size: int = 120832
    hidden_size: int = 2048
    intermediate_size: int = 6912
    moe_intermediate_size: int = 1536
    expert_hidden_dim: Optional[int] = None
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: Optional[int] = 128
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    rope_theta: Optional[float] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    num_experts: Optional[int] = 128
    num_local_experts: Optional[int] = None
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    router_scaling_factor: float = 2.826
    route_norm: bool = True
    moe_router_use_sigmoid: bool = True
    mlp_layer_types: Optional[List[str]] = None
    first_k_dense_replace: int = 1
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    enable_moe_fp32_combine: bool = True
    enable_lm_head_fp32: bool = False
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        if self.num_experts is None:
            self.num_experts = self.num_local_experts
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        rope = dict(self.rope_parameters or {})
        if self.rope_theta is None:
            self.rope_theta = float(rope.get("rope_theta", 11158840.0))
        if self.rope_scaling is None and rope.get("rope_type", "default") not in (None, "default"):
            self.rope_scaling = rope
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]
        if not self.moe_router_use_sigmoid:
            raise ValueError("hy_v3: only the sigmoid router (moe_router_use_sigmoid=true) is implemented")
        if self.hidden_act != "silu":
            raise ValueError(f"hy_v3: unsupported hidden_act {self.hidden_act!r} (expected 'silu')")

    @property
    def expert_dim(self) -> int:
        """Routed/shared expert width: the checkpoint's `expert_hidden_dim` when present."""
        return int(self.expert_hidden_dim or self.moe_intermediate_size)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        head_dim = int(args.head_dim)
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        if args.rope_scaling:
            from mlx_lm.models.rope_utils import initialize_rope

            self.rope = initialize_rope(
                head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )
        else:
            self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """`mlp.router.gate.weight` + `mlp.router.expert_bias` (mlx-lm MoE convention, as in
    mlx-community/Hy3-preview-* and mlx-lm PR #1211). Logits in float32 (reference)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, int(args.num_experts), bias=False)
        self.expert_bias = mx.zeros((int(args.num_experts),), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate
        if isinstance(gate, nn.Linear) and not isinstance(gate, nn.QuantizedLinear):
            return x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
        # A third-party conversion that quantized the router: best available.
        return gate(x).astype(mx.float32)


def route(
    logits: mx.array,
    expert_bias: mx.array,
    top_k: int,
    scaling_factor: float,
    route_norm: bool = True,
):
    """HYV3TopKRouter: (expert indices, float32 combine weights).

    The correction bias only CHOOSES the experts; the weights come from the
    unbiased sigmoid scores.
    """
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice = scores + expert_bias.astype(mx.float32)
    inds = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if route_norm:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, weights * scaling_factor


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = int(args.num_experts_per_tok)
        self.scaling_factor = float(args.router_scaling_factor)
        self.route_norm = bool(args.route_norm)
        self.fp32_combine = bool(args.enable_moe_fp32_combine)
        self.router = Router(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.expert_dim, int(args.num_experts), bias=args.mlp_bias)
        self.shared_mlp = MLP(args.hidden_size, args.expert_dim * int(args.num_shared_experts), bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = route(self.router(x), self.router.expert_bias, self.top_k, self.scaling_factor, self.route_norm)
        routed = self.switch_mlp(x, inds)
        shared = self.shared_mlp(x)
        if self.fp32_combine:
            y = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2)
            return (y + shared.astype(mx.float32)).astype(x.dtype)
        y = (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)
        return y + shared


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)
        sparse = args.mlp_layer_types[layer_idx] == "sparse"
        self.mlp = MoE(args) if sparse else MLP(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class HYV3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


# Per-expert checkpoint tensors: model.layers.L.mlp.experts.E.{gate,up,down}_proj[.weight|.scales|.biases]
_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
# transformers-layout renames (HYV3 conversion mapping) -> checkpoint layout used here.
_RENAMES = (
    (re.compile(r"\.mlp\.gate\.weight$"), ".mlp.router.gate.weight"),
    (re.compile(r"\.mlp\.e_score_correction_bias$"), ".mlp.router.expert_bias"),
    (re.compile(r"\.mlp\.expert_bias$"), ".mlp.router.expert_bias"),
    (re.compile(r"\.mlp\.shared_experts\."), ".mlp.shared_mlp."),
    # stacked experts written by this module before it adopted the convention
    (re.compile(r"\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"), r".mlp.switch_mlp.\1.\2"),
)


def _is_router_or_bias(path: str) -> bool:
    return path.endswith("mlp.router.gate") or path.endswith("expert_bias")


def quant_predicate(path: str, module: Any = None, config: Any = None) -> Union[bool, Dict[str, Any]]:
    """Keep the MoE router (and its correction bias) in full precision.

    Expert SELECTION is a discrete top-k over sigmoid scores whose margins the
    correction bias deliberately tightens; quantizing the gate perturbs which
    experts run. Norms and the bias are not Linear layers and are never
    quantized; everything else follows the requested bits/group size.
    """
    return not _is_router_or_bias(path)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HYV3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: Optional[mx.array] = None) -> mx.array:
        h = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        head = self.lm_head
        if (
            self.args.enable_lm_head_fp32
            and isinstance(head, nn.Linear)
            and not isinstance(head, nn.QuantizedLinear)
        ):
            return h.astype(mx.float32) @ head.weight.astype(mx.float32).T
        return head(h)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        return quant_predicate

    @property
    def cast_predicate(self):
        # `mlx_lm.convert --dtype` must not round the correction bias to bf16.
        return lambda k: not k.endswith("expert_bias")

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        out: Dict[str, mx.array] = {}
        experts: Dict[tuple, Dict[int, mx.array]] = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key or key.startswith("mtp."):
                continue
            layer = re.match(r"^model\.layers\.(\d+)\.", key)
            if layer and int(layer.group(1)) >= n_layers:
                continue  # MTP / next-n layers (Hy3 layer 80): no AR role
            for pattern, repl in _RENAMES:
                key = pattern.sub(repl, key)
            m = _EXPERT_KEY_RE.match(key)
            if m:
                experts.setdefault((m.group(1), m.group(3), m.group(4)), {})[int(m.group(2))] = value
                continue
            if key.endswith(".mlp.experts.gate_up_proj"):
                # transformers layout: [E, 2*I, H] -> SwitchGLU gate/up
                prefix = key[: -len("experts.gate_up_proj")] + "switch_mlp."
                gate, up = mx.split(value, 2, axis=-2)
                out[prefix + "gate_proj.weight"] = gate
                out[prefix + "up_proj.weight"] = up
                continue
            if key.endswith(".mlp.experts.down_proj"):
                out[key[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = value
                continue
            if key.endswith(".mlp.router.expert_bias"):
                value = value.astype(mx.float32)
            out[key] = value
        n_experts = int(self.args.num_experts)
        for (layer_idx, proj, suffix), by_index in experts.items():
            if sorted(by_index) != list(range(n_experts)):
                raise ValueError(
                    f"hy_v3: layer {layer_idx} {proj}.{suffix} has experts {sorted(by_index)[:4]}... "
                    f"({len(by_index)}), expected 0..{n_experts - 1}"
                )
            out[f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.{suffix}"] = mx.stack(
                [by_index[i] for i in range(n_experts)], axis=0
            )
        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        return out


def describe_weight_layout(path: str) -> str:
    """Name the HY V3 weight layout of a local checkpoint from its tensor names."""
    import json
    import struct
    from pathlib import Path

    root = Path(path)
    names: List[str] = []
    index = root / "model.safetensors.index.json"
    try:
        if index.is_file():
            names = list(json.loads(index.read_text()).get("weight_map", {}))
        else:
            for shard in sorted(root.glob("*.safetensors")):
                with open(shard, "rb") as fh:
                    size = struct.unpack("<Q", fh.read(8))[0]
                    names.extend(k for k in json.loads(fh.read(size)) if k != "__metadata__")
    except Exception as exc:  # the description must never mask the real error
        return f"unreadable ({type(exc).__name__}: {exc})"
    joined = "\n".join(names)
    if ".mlp.switch_mlp." in joined:
        return "mlx-lm convention (mlp.switch_mlp.*, mlp.router.expert_bias)"
    if re.search(r"\.mlp\.experts\.\d+\.", joined):
        return "tencent per-expert (mlp.experts.N.*, mlp.expert_bias)"
    if ".mlp.experts.gate_up_proj" in joined:
        return "transformers stacked (mlp.experts.gate_up_proj, mlp.e_score_correction_bias)"
    if ".mlp.experts." in joined:
        return "early abstractcore/community stacked (mlp.experts.{gate,up,down}_proj, mlp.expert_bias)"
    return "unrecognised"


def register_with_mlx_lm() -> bool:
    """Expose this module as `mlx_lm.models.hy_v3` when mlx-lm has none.

    Returns True when the shim is (now or already) installed, False when the
    installed mlx-lm ships its own hy_v3 module — upstream always wins.
    """
    import importlib.util
    import sys

    this = sys.modules.get(__name__)
    if this is None:  # loaded as a bundled model_file: nothing to register
        return False
    current = sys.modules.get(_UPSTREAM_MODULE)
    if current is this:
        return True
    if current is not None:
        return False
    try:
        upstream = importlib.util.find_spec(_UPSTREAM_MODULE)
    except (ImportError, ValueError):
        upstream = None
    if upstream is not None:
        return False
    sys.modules[_UPSTREAM_MODULE] = this
    return True


__all__ = ["Model", "ModelArgs", "describe_weight_layout", "quant_predicate", "register_with_mlx_lm", "route"]
