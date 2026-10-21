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


def _mlx_convention_layout(state: dict) -> dict:
    """mlx-lm PR #1211 / mlx-community Hy3-preview names: switch_mlp + router.expert_bias."""
    out = {}
    for key, value in _transformers_to_stacked(state).items():
        key = key.replace(".mlp.experts.", ".mlp.switch_mlp.")
        key = key.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
        key = key.replace(".mlp.e_score_correction_bias", ".mlp.router.expert_bias")
        key = key.replace(".mlp.shared_experts.", ".mlp.shared_mlp.")
        out[key] = value
    return out


def _interim_layout(state: dict) -> dict:
    """This module's own pre-convention names (mlp.experts.{proj}.weight, mlp.expert_bias)."""
    out = {}
    for key, value in _transformers_to_stacked(state).items():
        key = key.replace(".mlp.gate.weight", ".mlp.router.gate.weight")
        key = key.replace(".mlp.e_score_correction_bias", ".mlp.expert_bias")
        key = key.replace(".mlp.shared_experts.", ".mlp.shared_mlp.")
        out[key] = value
    return out


def _transformers_to_stacked(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if key.endswith("mlp.experts.gate_up_proj"):
            prefix = key[: -len("gate_up_proj")]
            half = value.shape[1] // 2
            out[prefix + "gate_proj.weight"] = value[:, :half]
            out[prefix + "up_proj.weight"] = value[:, half:]
        elif key.endswith("mlp.experts.down_proj"):
            out[key + ".weight"] = value
        else:
            out[key] = value
    return out


@pytest.mark.parametrize("layout", ["transformers", "checkpoint", "mlx_convention", "interim"])
def test_logits_match_the_transformers_reference(layout: str) -> None:
    import mlx.core as mx
    import torch

    ref, config = _tiny_reference()
    state = dict(ref.state_dict())
    if layout == "checkpoint":
        state = _per_expert_layout(state)
    elif layout == "mlx_convention":
        state = _mlx_convention_layout(state)
    elif layout == "interim":
        state = _interim_layout(state)
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
    assert moe.switch_mlp.gate_proj.weight.shape == (4, 24, 64)
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
    assert str(moe.router.expert_bias.dtype) == "mlx.core.float32"


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


def test_bundled_copy_loads_the_way_mlx_lm_loads_a_model_file(tmp_path) -> None:
    """`mlx_lm.utils.load_model` honours config `model_file` with
    spec_from_file_location + module_from_spec and NEVER registers the module in
    sys.modules; the published quants ship this exact file as hy_v3.py."""
    import shutil

    import abstractcore.providers.mlx_hy_v3 as hy

    bundled = tmp_path / "hy_v3.py"
    shutil.copyfile(hy.__file__, bundled)
    spec = importlib.util.spec_from_file_location("custom_model", bundled)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "custom_model" not in sys.modules
    args = module.ModelArgs.from_dict({"model_type": "hy_v3", "hidden_size": 64, "num_hidden_layers": 2,
                                       "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                                       "intermediate_size": 32, "num_experts": 4, "num_experts_per_tok": 2,
                                       "moe_intermediate_size": 16, "vocab_size": 32})
    model = module.Model(args)
    assert model.layers[1].mlp.switch_mlp.gate_proj.weight.shape == (4, 16, 64)
    assert module.register_with_mlx_lm() is False  # a bundled copy never shims mlx_lm


def test_parameter_names_follow_the_mlx_lm_moe_convention() -> None:
    """mlx-community/Hy3-preview-* and mlx-lm PR #1211 store mlp.switch_mlp.* and
    mlp.router.expert_bias; a strict load of that layout must work (review B1), and the
    model's own parameter tree must use exactly those names (what convert saves)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from abstractcore.providers.mlx_hy_v3 import Model, ModelArgs

    cfg = {"model_type": "hy_v3", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
           "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96, "num_experts": 4,
           "num_experts_per_tok": 2, "moe_intermediate_size": 32, "vocab_size": 32}
    names = {k for k, _ in tree_flatten(Model(ModelArgs.from_dict(cfg)).parameters())}
    assert "model.layers.1.mlp.switch_mlp.gate_proj.weight" in names
    assert "model.layers.1.mlp.router.expert_bias" in names
    assert "model.layers.1.mlp.router.gate.weight" in names
    assert not any(".mlp.experts." in n or n.endswith(".mlp.expert_bias") for n in names)
    pr1211 = {k: mx.zeros(v.shape, dtype=v.dtype) for k, v in tree_flatten(Model(ModelArgs.from_dict(cfg)).parameters())}
    m = Model(ModelArgs.from_dict(cfg))
    m.load_weights(list(m.sanitize(pr1211).items()), strict=True)


def test_dense_layers_without_mlp_layer_types_follow_first_k_dense_replace() -> None:
    from abstractcore.providers.mlx_hy_v3 import MLP, Model, ModelArgs, MoE

    base = {"model_type": "hy_v3", "hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96, "num_experts": 4,
            "num_experts_per_tok": 2, "moe_intermediate_size": 32, "vocab_size": 32}
    layers = Model(ModelArgs.from_dict({**base, "first_k_dense_replace": 1})).layers
    assert [type(layer.mlp) for layer in layers] == [MLP, MoE, MoE]
    explicit = Model(ModelArgs.from_dict({**base, "mlp_layer_types": ["dense", "dense", "sparse"]})).layers
    assert [type(layer.mlp) for layer in explicit] == [MLP, MLP, MoE]


def test_register_shim_defers_to_an_installed_but_unimported_upstream(monkeypatch) -> None:
    import abstractcore.providers.mlx_hy_v3 as hy

    monkeypatch.delitem(sys.modules, "mlx_lm.models.hy_v3", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: object() if name == "mlx_lm.models.hy_v3" else real(name, *a, **k))
    assert hy.register_with_mlx_lm() is False
    assert "mlx_lm.models.hy_v3" not in sys.modules


def test_mlx_provider_never_runs_the_bundled_model_file_and_installs_the_shim(monkeypatch) -> None:
    import logging

    import abstractcore.providers.mlx_hy_v3 as hy
    from abstractcore.providers.mlx_provider import MLXProvider

    monkeypatch.delitem(sys.modules, "mlx_lm.models.hy_v3", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "mlx_lm.models.hy_v3" else real(name, *a, **k))
    p = MLXProvider.__new__(MLXProvider)
    p.logger = logging.getLogger("test")
    assert p._hy_v3_load_kwargs() == {"model_config": {"model_file": None}}
    assert sys.modules["mlx_lm.models.hy_v3"] is hy
    assert p._hy_v3_class_source == "abstractcore.providers.mlx_hy_v3"


@pytest.mark.parametrize(
    "names,expected",
    [
        (["model.layers.1.mlp.switch_mlp.gate_proj.weight", "model.layers.1.mlp.router.expert_bias"], "mlx-lm convention"),
        (["model.layers.1.mlp.experts.0.gate_proj.weight", "model.layers.1.mlp.expert_bias"], "tencent per-expert"),
        (["model.layers.1.mlp.experts.gate_up_proj"], "transformers stacked"),
        (["model.layers.1.mlp.experts.gate_proj.weight", "model.layers.1.mlp.expert_bias"], "early abstractcore"),
    ],
)
def test_weight_layout_is_named_for_the_load_error(tmp_path, names, expected) -> None:
    import json

    from abstractcore.providers.mlx_hy_v3 import describe_weight_layout

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "m.safetensors" for n in names}}))
    assert describe_weight_layout(str(tmp_path)).startswith(expected)
