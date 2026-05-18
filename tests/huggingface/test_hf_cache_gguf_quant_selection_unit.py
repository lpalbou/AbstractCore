from pathlib import Path

import abstractcore.providers.huggingface_provider as huggingface_provider_module
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


def test_find_gguf_in_cache_prefers_q4_k_m_case_insensitive(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    # Hugging Face hub cache layout: hub/models--ORG--REPO/snapshots/<hash>/*.gguf
    snapshot_dir = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Tesslate--OmniCoder-9B-GGUF"
        / "snapshots"
        / "snapshot123"
    )
    snapshot_dir.mkdir(parents=True)

    # Ensure we prefer a quantized model even when the filename uses lowercase quant naming.
    (snapshot_dir / "omnicoder-9b-bf16.gguf").write_bytes(b"GGUF")
    expected = snapshot_dir / "omnicoder-9b-q4_k_m.gguf"
    expected.write_bytes(b"GGUF")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    found = provider._find_gguf_in_cache("Tesslate/OmniCoder-9B-GGUF")

    assert found == str(expected)


def test_find_gguf_in_cache_honors_explicit_quant_selector(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    snapshot_dir = (
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF"
        / "snapshots"
        / "snapshot123"
    )
    snapshot_dir.mkdir(parents=True)

    default_pick = snapshot_dir / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    explicit_pick = snapshot_dir / "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
    default_pick.write_bytes(b"GGUF")
    explicit_pick.write_bytes(b"GGUF")

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    found = provider._find_gguf_in_cache("unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q5_K_M")

    assert found == str(explicit_pick)


def test_load_gguf_model_mtp_path_warns_without_name_error(
    monkeypatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf"
    model_path.write_bytes(b"GGUF")

    warnings: list[str] = []

    class _Logger:
        def warning(self, message: str) -> None:
            warnings.append(str(message))

        def debug(self, _message: str) -> None:
            return None

    class _FakeLlama:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(huggingface_provider_module, "Llama", _FakeLlama, raising=False)

    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = str(model_path)
    provider.max_tokens = 4096
    provider.max_output_tokens = 1024
    provider.n_gpu_layers = 0
    provider.debug = False
    provider._user_provided_max_tokens = False
    provider.llm = None
    provider.logger = _Logger()

    provider._load_gguf_model()

    assert isinstance(provider.llm, _FakeLlama)
    assert any("MTP GGUF" in message for message in warnings)
