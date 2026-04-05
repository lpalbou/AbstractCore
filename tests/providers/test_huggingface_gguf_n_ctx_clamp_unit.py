from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractcore.providers import huggingface_provider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


class _DummyLlama:
    def __init__(self, **kwargs: Any):
        self._kwargs: Dict[str, Any] = dict(kwargs)

    def n_ctx(self) -> int:
        return int(self._kwargs.get("n_ctx") or 0)

class _FailOnLargeCtxDummyLlama(_DummyLlama):
    def __init__(self, **kwargs: Any):
        n_ctx = int(kwargs.get("n_ctx") or 0)
        if n_ctx > 8192:
            raise RuntimeError(f"simulated OOM for n_ctx={n_ctx}")
        super().__init__(**kwargs)


def test_huggingface_gguf_uses_capabilities_max_tokens_as_default_n_ctx(tmp_path, monkeypatch) -> None:
    # Create a tiny placeholder file so HuggingFaceProvider treats it as a direct GGUF path.
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _DummyLlama, raising=False)

    llm = HuggingFaceProvider(
        model=str(gguf_path),
        device="cpu",
    )

    assert llm.model_type == "gguf"
    # Unknown models use the default capabilities context window (16384) unless overridden.
    assert llm.llm.n_ctx() == 16384
    assert llm.max_tokens == 16384


def test_huggingface_gguf_falls_back_to_smaller_n_ctx_on_load_failure(tmp_path, monkeypatch) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _FailOnLargeCtxDummyLlama, raising=False)

    llm = HuggingFaceProvider(
        model=str(gguf_path),
        device="cpu",
    )

    assert llm.model_type == "gguf"
    assert llm.llm.n_ctx() == 8192
    assert llm.max_tokens == 8192


def test_huggingface_gguf_respects_explicit_max_tokens_as_runtime_n_ctx(tmp_path, monkeypatch) -> None:
    gguf_path = tmp_path / "dummy.gguf"
    gguf_path.write_bytes(b"GGUF")

    monkeypatch.setattr(huggingface_provider, "LLAMACPP_AVAILABLE", True, raising=False)
    monkeypatch.setattr(huggingface_provider, "Llama", _DummyLlama, raising=False)

    with pytest.warns(UserWarning):
        llm = HuggingFaceProvider(
            model=str(gguf_path),
            device="cpu",
            max_tokens=4096,
            max_output_tokens=5000,
        )

    assert llm.model_type == "gguf"
    assert llm.llm.n_ctx() == 4096
    assert llm.max_tokens == 4096
    assert llm.max_output_tokens == 4096
