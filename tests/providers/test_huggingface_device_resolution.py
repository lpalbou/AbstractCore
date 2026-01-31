from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


@pytest.mark.basic
def test_resolve_requested_device_prefers_explicit_arg(monkeypatch) -> None:
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    monkeypatch.setenv("ABSTRACTCORE_HF_DEVICE", "cpu")
    assert HuggingFaceProvider._resolve_requested_device("mps") == "mps"


@pytest.mark.basic
def test_resolve_requested_device_uses_env_override(monkeypatch) -> None:
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    monkeypatch.setenv("ABSTRACTCORE_HF_DEVICE", "cpu")
    assert HuggingFaceProvider._resolve_requested_device(None) == "cpu"


@pytest.mark.basic
def test_setup_device_transformers_warns_and_falls_back_when_mps_unavailable(monkeypatch) -> None:
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed")

    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.device = "mps"
    provider.logger = SimpleNamespace(warning=lambda *a, **k: None)

    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    provider._setup_device_transformers()
    assert provider.device == "cpu"

