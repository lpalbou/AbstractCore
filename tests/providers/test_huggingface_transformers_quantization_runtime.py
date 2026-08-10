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


def test_transformers_fp8_quantization_requires_cuda_or_xpu(monkeypatch):
    class TorchStub:
        class cuda:
            @staticmethod
            def is_available():
                return False

    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return TorchStub
        if name == "transformers.utils" and "is_torch_xpu_available" in fromlist:
            class UtilsStub:
                @staticmethod
                def is_torch_xpu_available():
                    return False

            return UtilsStub
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    provider = _provider("Qwen/Qwen3.6-27B-FP8")

    with pytest.raises(ImportError) as exc:
        provider._validate_transformers_quantization_runtime({"quant_method": "fp8", "fmt": "e4m3"})

    assert "uses FP8 quantization" in str(exc.value)
    assert "requires a CUDA/XPU runtime" in str(exc.value)


# ---------------------------------------------------------------------------
# Caller-requested quantization (added 2026-08-06).
#
# Before this, the provider's transformers kwarg allowlist forwarded the LEGACY
# `load_in_4bit`/`load_in_8bit` flags — which transformers 5.9 REMOVED from
# `from_pretrained` — and did not forward `quantization_config` at all, so there
# was no way to ask abstractcore for a bitsandbytes-quantized transformers model.
# ---------------------------------------------------------------------------

_BUILD = HuggingFaceProvider._build_transformers_quantization_config


def test_no_quantization_kwargs_is_inert():
    assert _BUILD({}) is None
    assert _BUILD({"trust_remote_code": True, "device_map": "auto"}) is None
    assert _BUILD({"load_in_4bit": False, "load_in_8bit": False}) is None


def test_explicit_quantization_config_is_passed_through_unchanged():
    sentinel = object()
    assert _BUILD({"quantization_config": sentinel}) is sentinel
    # ...and wins over the legacy flags rather than being merged with them.
    assert _BUILD({"quantization_config": sentinel, "load_in_4bit": True}) is sentinel


def test_legacy_4bit_and_8bit_are_mutually_exclusive():
    with pytest.raises(ValueError) as exc:
        _BUILD({"load_in_4bit": True, "load_in_8bit": True})
    assert "mutually exclusive" in str(exc.value)


def test_legacy_flags_require_bitsandbytes(monkeypatch):
    monkeypatch.setattr(hf_provider, "_module_available", lambda name: False)
    with pytest.raises(ImportError) as exc:
        _BUILD({"load_in_4bit": True})
    assert "`bitsandbytes` package" in str(exc.value)


def test_legacy_flags_translate_to_bitsandbytesconfig(monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.setattr(hf_provider, "_module_available", lambda name: True)
    torch = pytest.importorskip("torch")

    cfg = _BUILD({
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",   # a dtype NAME, not a torch object
        "device_map": "auto",                   # unrelated kwarg, must be ignored
    })

    from transformers import BitsAndBytesConfig
    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.load_in_4bit is True and cfg.load_in_8bit is False
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype is torch.bfloat16


def test_bad_compute_dtype_name_is_rejected(monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.setattr(hf_provider, "_module_available", lambda name: True)
    with pytest.raises(ValueError) as exc:
        _BUILD({"load_in_4bit": True, "bnb_4bit_compute_dtype": "not_a_dtype"})
    assert "is not a torch dtype name" in str(exc.value)
