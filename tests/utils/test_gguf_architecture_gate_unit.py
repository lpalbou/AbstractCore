"""The GGUF architecture gate answers from the INSTALLED llama.cpp only, and only for an
architecture (merge desk, 2026-10-29: a model id passed as the architecture read as "No llama.cpp
release supports the unsloth/Qwen3.5-4B-GGUF architecture yet")."""
from __future__ import annotations

import pytest

from abstractcore.utils import model_cache as mc


@pytest.fixture()
def lib_knows(monkeypatch):
    table = b"\x00".join([b"llama", b"qwen2", b"qwen35", b"gemma4", b"lfm2", b""])
    monkeypatch.setattr(mc, "_llama_library_bytes", lambda: b"junk\x00" + table)
    monkeypatch.setattr(mc, "newest_llamacpp_architectures",
                        lambda: {"llama_cpp_python": "0.3.36", "architectures": ["llama", "qwen35", "spark2_5"]})


@pytest.mark.parametrize("not_an_arch", ["unsloth/Qwen3.5-4B-GGUF", "unsloth/gemma-4-E4B-it-GGUF",
                                         "LiquidAI/LFM2.5-1.2B-GGUF", "Qwen3.8-27B-Q4_K_M.gguf", "Qwen3"])
def test_a_model_id_is_never_refused_as_an_architecture(lib_knows, not_an_arch):
    """MUTANT: no token check -> the id is "not in libllama" -> refused -> RED."""
    assert mc.llamacpp_knows_architecture(not_an_arch) is None
    assert mc.gguf_unloadable_reason(not_an_arch) is None


def test_the_installed_library_decides(lib_knows):
    for arch in ("qwen35", "gemma4", "lfm2"):
        assert mc.llamacpp_knows_architecture(arch) is True and mc.gguf_unloadable_reason(arch) is None
    # Not in the installed library, known to a newer release: "update llama.cpp".
    assert "update llama.cpp" in mc.gguf_unloadable_reason("spark2_5")
    # Known nowhere: the honest sentence.
    assert "No llama.cpp release supports the madeup9 architecture" in mc.gguf_unloadable_reason("madeup9")
