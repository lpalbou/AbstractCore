"""Absolute/relative filesystem paths to a local GGUF model must resolve.

Regression for the operator-reported bug (2026-07-15): launching with
`--model /abs/path/to/Model-GGUF` failed ModelNotFoundError while the HF
repo-id form (`org/Model-GGUF`) worked — the resolver mis-parsed an absolute
path as a repo id and searched caches instead of using the path in hand. A
user pointing at an on-disk model (the LM Studio layout is the common one)
must be recognized as GGUF and loaded directly.
"""

from pathlib import Path

from abstractcore.providers.huggingface_provider import HuggingFaceProvider


def _provider() -> HuggingFaceProvider:
    return HuggingFaceProvider.__new__(HuggingFaceProvider)


def test_absolute_directory_path_resolves_to_gguf(tmp_path: Path) -> None:
    model_dir = tmp_path / "org" / "My-Model-35B-GGUF"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "my-model-35b-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")

    p = _provider()
    assert p._is_gguf_model(str(model_dir)) is True
    assert p._find_gguf_in_cache(str(model_dir)) == str(gguf)


def test_absolute_file_path_resolves_directly(tmp_path: Path) -> None:
    gguf = tmp_path / "org" / "My-Model-GGUF" / "my-model-Q5_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")

    p = _provider()
    assert p._is_gguf_model(str(gguf)) is True
    assert p._find_gguf_in_cache(str(gguf)) == str(gguf)


def test_directory_with_quant_selector(tmp_path: Path) -> None:
    model_dir = tmp_path / "org" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    picked = model_dir / "model-Q5_K_M.gguf"
    picked.write_bytes(b"GGUF")

    p = _provider()
    ref = f"{model_dir}:Q5_K_M"
    assert p._is_gguf_model(ref) is True
    assert p._find_gguf_in_cache(ref) == str(picked)


def test_directory_prefers_q4_k_m_when_multiple(tmp_path: Path) -> None:
    model_dir = tmp_path / "Model-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "model-bf16.gguf").write_bytes(b"GGUF")
    preferred = model_dir / "model-Q4_K_M.gguf"
    preferred.write_bytes(b"GGUF")

    p = _provider()
    assert p._find_gguf_in_cache(str(model_dir)) == str(preferred)


def test_nonexistent_path_returns_none_and_not_gguf(tmp_path: Path) -> None:
    missing = tmp_path / "does" / "not" / "exist-GGUF"
    p = _provider()
    # No .gguf on disk -> resolver finds nothing (and a bare non-'gguf' path
    # would not be treated as gguf; this one has the token, which is fine —
    # the point is the resolver returns None rather than a wrong path).
    assert p._find_gguf_in_cache(str(missing)) is None


def test_plain_hub_id_without_gguf_is_not_treated_as_path(tmp_path: Path) -> None:
    # A normal transformers repo id must NOT be misclassified as GGUF just
    # because the path test runs — it isn't a file/dir and carries no token.
    p = _provider()
    assert p._is_gguf_model("meta-llama/Llama-3.2-3B-Instruct") is False


def test_directory_without_model_binaries_is_not_recognized(tmp_path: Path) -> None:
    # A directory that exists but holds no .gguf must not be claimed as GGUF.
    # NOTE: the dir name (and this test's name) must avoid the "gguf" token —
    # the pre-existing `'gguf' in name` rule would otherwise match the path
    # string itself, which is correct behavior but not what THIS case probes.
    empty = tmp_path / "plain-model"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    assert "gguf" not in str(empty).lower(), "test path must not carry the token"
    p = _provider()
    assert p._is_gguf_model(str(empty)) is False
    assert p._find_gguf_in_cache(str(empty)) is None
