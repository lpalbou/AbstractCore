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


def _gguf_provider(model_path: Path, monkeypatch, warnings: list) -> HuggingFaceProvider:
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
    return provider


def test_load_gguf_model_warns_when_the_FILE_carries_an_mtp_head(
    monkeypatch, tmp_path: Path
) -> None:
    """The warning is driven by header evidence, not by the filename.

    Note the name says nothing about MTP -- this mirrors the real world, where
    `Qwen3.8-27B-Q4_K_M.gguf` carries `qwen35.nextn_predict_layers=1` with no
    "mtp" anywhere in its name.
    """
    from tests.providers.fixtures.mtp_adv_gguf_builder import build_gguf

    model_path = build_gguf(tmp_path / "Qwen3.8-27B-Q4_K_M.gguf", nextn=True)

    warnings: list[str] = []
    provider = _gguf_provider(model_path, monkeypatch, warnings)
    provider._load_gguf_model()

    assert provider.llm is not None
    mtp_warnings = [w for w in warnings if "multi-token-prediction head" in w]
    assert mtp_warnings, f"no MTP warning emitted; got {warnings}"
    # It must name the two runtimes that CAN execute the head, or the warning
    # tells the user they have a problem without telling them the way out.
    assert "--spec-type draft-mtp" in mtp_warnings[0]
    assert "--speculative-draft-mtp" in mtp_warnings[0]


def test_load_gguf_model_does_not_warn_for_a_file_merely_NAMED_mtp(
    monkeypatch, tmp_path: Path
) -> None:
    """The old heuristic was `"mtp" in self.model.lower()` -- a false positive.

    unsloth puts "MTP" in the REPO name, so a plain quant downloaded from a
    neighbouring path could inherit it and claim a head it does not have.
    """
    from tests.providers.fixtures.mtp_adv_gguf_builder import build_gguf

    model_path = build_gguf(
        tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf", nextn=False
    )

    warnings: list[str] = []
    provider = _gguf_provider(model_path, monkeypatch, warnings)
    provider._load_gguf_model()

    assert provider.llm is not None
    assert not [w for w in warnings if "multi-token-prediction head" in w], (
        "a file with MTP only in its NAME was reported as carrying a head: "
        f"{warnings}"
    )


def test_load_gguf_model_mtp_path_completes_without_name_error(
    monkeypatch, tmp_path: Path
) -> None:
    """The MTP branch must survive a file it cannot parse.

    Originally this pinned the filename-triggered warning; the trigger is now
    header evidence (see the two tests above). What is still worth pinning is
    the branch running at all on a stub file -- the header reader swallows
    every exception, so a NameError inside it would silently return None and go
    unnoticed. That exact bug happened once during development.
    """
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
    # A 4-byte stub carries no header evidence, so no MTP claim may be made.
    assert not [w for w in warnings if "multi-token-prediction head" in w]
