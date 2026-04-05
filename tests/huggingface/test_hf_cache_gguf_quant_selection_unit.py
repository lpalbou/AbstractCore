from pathlib import Path

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
