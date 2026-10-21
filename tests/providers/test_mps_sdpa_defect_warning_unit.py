"""`_warn_if_mps_sdpa_defective` names the torch builds measured broken on MPS.

torch 2.8.0: scaled_dot_product_attention(is_causal=True) in bf16/fp16 is wrong for
prompts longer than ~20 tokens (measured 2026-10-21, Hy-Memory-30B-A3B bf16 unusable).
torch 2.10.x: long-KV decode fault (pytorch#163597). Other versions: no claim.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

import abstractcore.providers.huggingface_provider as hf


class _Model:
    def __init__(self, dtype):
        self._p = torch.zeros(1, dtype=dtype)

    def parameters(self):
        yield self._p


@pytest.mark.parametrize(
    "version,dtype,device,expected",
    [
        ("2.8.0", torch.bfloat16, "mps", True),
        ("2.8.1", torch.float16, "mps", True),
        ("2.10.0", torch.bfloat16, "mps", True),
        ("2.8.0", torch.float32, "mps", False),
        ("2.8.0", torch.bfloat16, "cpu", False),
        ("2.9.1", torch.bfloat16, "mps", False),
        ("2.11.0", torch.bfloat16, "mps", False),
    ],
)
def test_warning_gate(monkeypatch, version, dtype, device, expected) -> None:
    monkeypatch.setattr(hf, "_MPS_SDPA_DEFECT_WARNED", False)
    monkeypatch.setattr(torch, "__version__", version)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert hf._warn_if_mps_sdpa_defective(device, _Model(dtype)) is expected
    assert bool(caught) is expected
    if expected and version.startswith("2.8"):
        assert "ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL=eager" in str(caught[0].message)
