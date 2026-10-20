"""abstractcore.providers.mlx_hy_v3 against transformers' HYV3ForCausalLM (the reference).

A tiny random HY V3 (dense layer 0 + sparse MoE layers, GQA, QK-norm, sigmoid
router with a NON-zero correction bias, shared expert) is built in transformers
fp32, its weights are fed to the MLX port in both layouts it must accept (the
transformers fused layout and the released checkpoint's per-expert layout), and
the logits must agree. Weights are re-drawn with a large std so expert choice
is decisive: with transformers' default 0.006 init every sigmoid score is ~0.5
and a parity test would only measure tie-breaking.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None
    or importlib.util.find_spec("mlx_lm") is None
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="needs mlx, mlx-lm, torch and transformers",
)


def _tiny_reference(seed: int = 0):
    import torch
    from transformers import HYV3Config, HYV3ForCausalLM

    cfg = HYV3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=32,
        max_position_embeddings=512,
        router_scaling_factor=2.826,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    torch.manual_seed(seed)
    model = HYV3ForCausalLM(cfg).eval()
    # The default grouped_mm experts kernel has no CPU implementation.
    model.set_experts_implementation("eager")
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=gen))
            else:
                param.copy_(0.15 * torch.randn(param.shape, generator=gen))
        for name, buf in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buf.copy_(0.3 * torch.randn(buf.shape, generator=gen))
    config = cfg.to_dict()
    config["model_type"] = "hy_v3"
    return model, config


def _per_expert_layout(state: dict) -> dict:
    """The released checkpoint's names: router.gate, expert_bias, shared_mlp, experts.N.*"""
    out = {}
    for key, value in state.items():
        if key.endswith("mlp.experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            half = value.shape[1] // 2
            for e in range(value.shape[0]):
                out[f"{prefix}{e}.gate_proj.weight"] = value[e, :half]
                out[f"{prefix}{e}.up_proj.weight"] = value[e, half:]
        elif key.endswith("mlp.experts.down_proj"):
            prefix = key[: -len("down_proj")]
            for e in range(value.shape[0]):
                out[f"{prefix}{e}.down_proj.weight"] = value[e]
        else:
            key = key.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
            key = key.replace(".mlp.e_score_correction_bias", ".mlp.expert_bias")
            key = key.replace(".mlp.shared_experts.", ".mlp.shared_mlp.")
            out[key] = value
    return out


def _mlx_model(config: dict, state: dict):
    import mlx.core as mx

    from abstractcore.providers.mlx_hy_v3 import Model, ModelArgs

    model = Model(ModelArgs.from_dict(config))
    weights = {k: mx.array(np.asarray(v.detach().float().numpy())) for k, v in state.items()}
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    mx.eval(model.parameters())
    return model


TOKENS = [[3, 17, 99, 4, 250, 31, 7, 7, 128, 64, 1, 200]]


@pytest.fixture(autouse=True)
def _mlx_cpu_stream():
    """Exact-arithmetic parity runs on the MLX CPU device.

    Measured: on the CPU device the port matches the fp32 reference to 4e-6;
    the Metal fp32 matmul kernels drift by ~1e-2 on these deliberately large
    random weights, which would force tolerances loose enough to hide a real bug.
    """
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("layout", ["transformers", "checkpoint"])
def test_logits_match_the_transformers_reference(layout: str) -> None:
    import mlx.core as mx
    import torch

    ref, config = _tiny_reference()
    state = dict(ref.state_dict())
    if layout == "checkpoint":
        state = _per_expert_layout(state)
    model = _mlx_model(config, state)

    with torch.no_grad():
        expected = ref(torch.tensor(TOKENS)).logits.float().numpy()
    got = np.asarray(model(mx.array(TOKENS)).astype(mx.float32))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
    assert (got.argmax(-1) == expected.argmax(-1)).all()


def test_kv_cache_decoding_matches_a_full_forward() -> None:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    ref, config = _tiny_reference(seed=3)
    model = _mlx_model(config, dict(ref.state_dict()))
    full = np.asarray(model(mx.array(TOKENS)).astype(mx.float32))[0]

    cache = make_prompt_cache(model)
    prefix = np.asarray(model(mx.array([TOKENS[0][:7]]), cache=cache).astype(mx.float32))[0]
    steps = [np.asarray(model(mx.array([[t]]), cache=cache).astype(mx.float32))[0, -1] for t in TOKENS[0][7:]]
    np.testing.assert_allclose(prefix, full[:7], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.stack(steps), full[7:], atol=1e-4, rtol=1e-4)


def test_correction_bias_chooses_but_does_not_weight() -> None:
    import mlx.core as mx

    from abstractcore.providers.mlx_hy_v3 import route

    logits = mx.array([[0.0, 0.1, 0.2, 3.0]])
    bias = mx.array([5.0, 0.0, 0.0, 0.0])  # forces expert 0 into the top-2
    inds, weights = route(logits, bias, top_k=2, scaling_factor=2.0)
    chosen = sorted(np.asarray(inds)[0].tolist())
    assert chosen == [0, 3]
    sig = 1.0 / (1.0 + np.exp(-np.array([0.0, 3.0])))
    expected = 2.0 * sig / sig.sum()
    got = dict(zip(np.asarray(inds)[0].tolist(), np.asarray(weights)[0].tolist()))
    np.testing.assert_allclose([got[0], got[3]], expected, rtol=1e-6)


def test_expert_hidden_dim_sizes_routed_and_shared_experts() -> None:
    from abstractcore.providers.mlx_hy_v3 import Model, ModelArgs

    args = ModelArgs.from_dict(
        {"model_type": "hy_v3", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96, "num_experts": 4,
         "num_experts_per_tok": 2, "moe_intermediate_size": 64, "expert_hidden_dim": 24, "vocab_size": 32}
    )
    moe = Model(args).model.layers[1].mlp
    assert moe.experts.gate_proj.weight.shape == (4, 24, 64)
    assert moe.shared_mlp.gate_proj.weight.shape == (24, 64)


def test_quant_predicate_keeps_the_router_full_precision() -> None:
    import mlx.nn as nn
    from mlx_lm.utils import quantize_model

    from abstractcore.providers.mlx_hy_v3 import Model, ModelArgs

    config = {"model_type": "hy_v3", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 128, "num_experts": 4,
              "num_experts_per_tok": 2, "moe_intermediate_size": 64, "vocab_size": 64}
    model = Model(ModelArgs.from_dict(config))
    model, qconfig = quantize_model(model, dict(config), 64, 4)  # model.quant_predicate is the default
    moe = model.model.layers[1].mlp
    assert type(moe.router.gate) is nn.Linear
    assert isinstance(moe.shared_mlp.gate_proj, nn.QuantizedLinear)
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert str(moe.expert_bias.dtype) == "mlx.core.float32"


def test_mtp_and_rotary_tensors_are_dropped() -> None:
    import mlx.core as mx

    from abstractcore.providers.mlx_hy_v3 import Model, ModelArgs

    model = Model(ModelArgs.from_dict({"model_type": "hy_v3", "hidden_size": 64, "num_hidden_layers": 1,
                                       "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                                       "intermediate_size": 32, "vocab_size": 16}))
    out = model.sanitize({
        "model.layers.1.self_attn.q_proj.weight": mx.zeros((1,)),
        "mtp.layers.0.weight": mx.zeros((1,)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((1,)),
        "model.norm.weight": mx.zeros((1,)),
    })
    assert list(out) == ["model.norm.weight"]


def test_register_shim_defers_to_an_upstream_module(monkeypatch) -> None:
    import types

    import abstractcore.providers.mlx_hy_v3 as hy

    monkeypatch.delitem(sys.modules, "mlx_lm.models.hy_v3", raising=False)
    upstream = types.ModuleType("mlx_lm.models.hy_v3")
    monkeypatch.setitem(sys.modules, "mlx_lm.models.hy_v3", upstream)
    assert hy.register_with_mlx_lm() is False
    assert sys.modules["mlx_lm.models.hy_v3"] is upstream

    monkeypatch.delitem(sys.modules, "mlx_lm.models.hy_v3")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "mlx_lm.models.hy_v3" else object())
    assert hy.register_with_mlx_lm() is True
    assert sys.modules["mlx_lm.models.hy_v3"] is hy
    from mlx_lm.utils import _get_classes

    assert _get_classes({"model_type": "hy_v3"}) == (hy.Model, hy.ModelArgs)
