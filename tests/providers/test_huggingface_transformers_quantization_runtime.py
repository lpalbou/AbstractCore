import pytest

import abstractcore.providers.huggingface_provider as hf_provider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


def _provider(model: str = "example/model") -> HuggingFaceProvider:
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = model
    return provider


def test_transformers_compressed_tensors_quantization_requires_runtime(monkeypatch):
    monkeypatch.setattr(hf_provider, "_module_available", lambda name: False)
    provider = _provider("trusted/example-compressed")

    with pytest.raises(ImportError) as exc:
        provider._validate_transformers_quantization_runtime(
            {"quant_method": "compressed-tensors", "format": "pack-quantized"}
        )

    message = str(exc.value)
    assert "compressed-tensors quantization" in message
    assert "`compressed-tensors` package is not installed" in message


def test_transformers_awq_quantization_requires_runtime(monkeypatch):
    monkeypatch.setattr(hf_provider, "_module_available", lambda name: False)
    provider = _provider("trusted/example-awq")

    with pytest.raises(ImportError) as exc:
        provider._validate_transformers_quantization_runtime({"quant_method": "awq"})

    assert "uses AWQ quantization" in str(exc.value)


def test_transformers_mlx_quantized_checkpoint_points_to_mlx_provider():
    provider = _provider("mlx-community/example-4bit")

    with pytest.raises(ImportError) as exc:
        provider._validate_transformers_quantization_runtime(
            {"bits": 4, "group_size": 64, "mode": "affine"}
        )

    message = str(exc.value)
    assert "MLX-format quantized checkpoint" in message
    assert "create_llm('mlx'" in message


def test_quantized_weight_load_rejects_missing_and_unexpected_keys():
    provider = _provider("trusted/example-awq")

    with pytest.raises(RuntimeError) as exc:
        provider._validate_transformers_weight_load(
            {
                "missing_keys": ["model.layers.0.mlp.up_proj.weight"],
                "unexpected_keys": ["model.layers.0.mlp.up_proj.weight_packed"],
            },
            {"quant_method": "awq"},
        )

    message = str(exc.value)
    assert "did not load its quantized weights cleanly" in message
    assert "model/runtime compatibility issue" in message
    assert "missing_keys=" in message
    assert "unexpected_keys=" in message
