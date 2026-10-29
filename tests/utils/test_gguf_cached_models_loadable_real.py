"""Every GGUF in THIS computer's Hugging Face cache that the installed llama.cpp loads is never
refused by the architecture gate (merge desk, 2026-10-29). Real files: runs with
ABSTRACTCORE_RUN_GGUF_TESTS=1. The files are found in the account's real hub cache (the test conftest
isolates HOME, so a model id cannot resolve inside a test). Each file is opened with `vocab_only=True` (header
and vocabulary: the architecture check, no weights) -- the ground truth the gate must agree with.
MUTANT: refuse every architecture -> RED for each loadable file."""
from __future__ import annotations

import glob
import os

import pytest

def _real_hub() -> str:
    # The account's real home (the conftest points HOME / HF_HUB_CACHE at a temporary one).
    override = os.environ.get("ABSTRACTCORE_REAL_HF_HUB_CACHE")
    if override:
        return override
    try:
        import pwd

        home = pwd.getpwuid(os.getuid()).pw_dir
    except Exception:  # noqa: BLE001 - Windows
        home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "huggingface", "hub")


_HUB = _real_hub()
_FILES = sorted(
    p for p in glob.glob(os.path.join(_HUB, "models--*", "snapshots", "*", "**", "*.gguf"), recursive=True)
    if "mmproj" not in os.path.basename(p).lower() and "-of-0000" not in os.path.basename(p)
)

pytestmark = pytest.mark.skipif(os.environ.get("ABSTRACTCORE_RUN_GGUF_TESTS") != "1" or not _FILES,
                                reason="real cached GGUFs: set ABSTRACTCORE_RUN_GGUF_TESTS=1")


@pytest.mark.parametrize("path", _FILES, ids=[p.split("models--", 1)[-1].split("/snapshots/")[0] + "/" + os.path.basename(p) for p in _FILES])
def test_a_loadable_cached_gguf_is_not_refused(path):
    from abstractcore.utils.model_cache import gguf_unloadable_reason, read_gguf_architecture

    import llama_cpp

    from pathlib import Path

    arch = read_gguf_architecture(Path(path))
    assert arch, f"the header of {path} names no architecture (the gate would have nothing to check)"
    try:
        llama_cpp.Llama(model_path=path, vocab_only=True, verbose=False).close()
        loads = True
    except Exception:
        loads = False
    reason = gguf_unloadable_reason(arch)
    print(f"\n{os.path.basename(path)}: architecture {arch}, installed llama.cpp loads it: {loads}, gate: {reason or 'ok'}")
    if loads:
        assert reason is None, f"{path} loads on the installed llama.cpp but the gate refuses it: {reason}"
    else:
        assert reason, f"{path} does not load ({arch}) but the gate says nothing"
