"""0817 axis 3: model-config KV-geometry fingerprint (unit contract).

The fingerprint must move exactly when the cached tensors' MEANING moves
(rope/window/position keys) and must NOT move on irrelevant config churn
(the false-invalidation direction that would recompile the corpus for
nothing). "" is reserved for "config unreachable" — validators abstain on
it, never match.
"""

from __future__ import annotations

from abstractcore.providers.model_config_fingerprint import (
    check_model_config_fingerprint,
    model_config_fingerprint_for,
)


class _ConfigObject:
    """PretrainedConfig-like duck: exposes to_dict()."""

    def __init__(self, data):
        self._data = dict(data)

    def to_dict(self):
        return dict(self._data)


class _ArgsObject:
    """mlx_lm ModelArgs-like duck: plain attributes, vars() view."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


_BASE = {
    "model_type": "qwen3",
    "rope_theta": 1000000.0,
    "rope_scaling": None,
    "sliding_window": None,
    "max_position_embeddings": 40960,
    "transformers_version": "5.6.0",
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}


def test_geometry_edit_changes_fingerprint() -> None:
    base = model_config_fingerprint_for(dict(_BASE))
    retuned = model_config_fingerprint_for({**_BASE, "rope_theta": 5000000.0})
    assert base and retuned
    assert base != retuned, "a rope_theta retune must invalidate"


def test_adding_rope_scaling_changes_fingerprint() -> None:
    base = model_config_fingerprint_for(dict(_BASE))
    longrope = model_config_fingerprint_for(
        {**_BASE, "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
    )
    assert base != longrope, "adding a scaling block must invalidate"


def test_window_layout_edit_changes_fingerprint() -> None:
    base = model_config_fingerprint_for(dict(_BASE))
    windowed = model_config_fingerprint_for({**_BASE, "sliding_window": 4096})
    assert base != windowed, "a sliding-window change must invalidate"


def test_irrelevant_churn_keeps_fingerprint() -> None:
    base = model_config_fingerprint_for(dict(_BASE))
    churned = model_config_fingerprint_for(
        {**_BASE, "transformers_version": "5.9.0", "torch_dtype": "float16", "_name_or_path": "/elsewhere"}
    )
    assert base == churned, "version/dtype/name churn must NOT invalidate the corpus"


def test_rope_scaling_dict_key_order_is_canonical() -> None:
    a = model_config_fingerprint_for({**_BASE, "rope_scaling": {"type": "yarn", "factor": 4.0}})
    b = model_config_fingerprint_for({**_BASE, "rope_scaling": {"factor": 4.0, "type": "yarn"}})
    assert a == b


def test_nested_text_config_geometry_is_seen() -> None:
    multimodal_base = {
        "model_type": "gemma3",
        "text_config": {"model_type": "gemma3_text", "rope_theta": 1000000.0, "sliding_window": 1024},
    }
    base = model_config_fingerprint_for(dict(multimodal_base))
    edited = model_config_fingerprint_for(
        {
            "model_type": "gemma3",
            "text_config": {"model_type": "gemma3_text", "rope_theta": 1000000.0, "sliding_window": 512},
        }
    )
    assert base and edited
    assert base != edited, "geometry edits under text_config must invalidate"


def test_config_object_and_dict_agree() -> None:
    as_dict = model_config_fingerprint_for(dict(_BASE))
    as_object = model_config_fingerprint_for(_ConfigObject(_BASE))
    assert as_dict == as_object


def test_model_args_object_fingerprints() -> None:
    args = _ArgsObject(model_type="qwen3", rope_theta=1000000.0, sliding_window=None, hidden_size=4096)
    fingerprint = model_config_fingerprint_for(args)
    assert fingerprint.startswith("model-config:sha256:")
    retuned = _ArgsObject(model_type="qwen3", rope_theta=5000000.0, sliding_window=None, hidden_size=4096)
    assert model_config_fingerprint_for(retuned) != fingerprint


def test_unreachable_config_is_empty_but_keyless_config_is_stable() -> None:
    assert model_config_fingerprint_for(None) == ""
    assert model_config_fingerprint_for(42) == ""
    keyless_a = model_config_fingerprint_for({"vocab_size": 100})
    keyless_b = model_config_fingerprint_for({"hidden_size": 64})
    assert keyless_a and keyless_a == keyless_b, "zero geometry keys = stable empty-geometry constant"
    with_geometry = model_config_fingerprint_for({"rope_theta": 10000.0})
    assert with_geometry != keyless_a, "a geometry key appearing must change the value"


def test_verdict_contract() -> None:
    assert check_model_config_fingerprint("", "") == "unverified_stored"
    assert check_model_config_fingerprint("", "model-config:sha256:aa") == "unverified_stored"
    assert check_model_config_fingerprint("model-config:sha256:aa", "") == "unverified_current"
    assert check_model_config_fingerprint("model-config:sha256:aa", "model-config:sha256:aa") == "ok"
    assert check_model_config_fingerprint("model-config:sha256:aa", "model-config:sha256:bb") == "mismatch"
