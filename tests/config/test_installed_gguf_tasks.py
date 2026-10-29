"""A GGUF-only repo in the Hugging Face cache (no config, no card) says what it is from its own
header (`general.architecture`) and its name (operator QA 2026-10-28 (c): a model outside the
catalog read as an unknown model and was never offered as the default text model)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from abstractcore.config import model_materializer as mm


def _gguf(path: Path, arch: str) -> None:
    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
                     + s("general.architecture") + struct.pack("<I", 8) + s(arch))


def _repo(tmp_path: Path, repo: str, files: dict) -> Path:
    root = tmp_path / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / "abc"
    for name, arch in files.items():
        _gguf(snap / name, arch)
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text("abc")
    return root


@pytest.mark.parametrize("repo,files,kind,tasks", [
    ("mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF", {"Qwen3.5-9B-Kimi.Q4_K_M.gguf": "qwen35"}, "model", ["text_generation"]),
    ("unsloth/gemma-4-E4B-it-GGUF", {"gemma-4-E4B-it-Q4_K_M.gguf": "gemma4"}, "model", ["text_generation"]),
    # An architecture AbstractCore does not know stays unknown (never called a text model).
    ("XHToken/Spark-X2.6-4B-GGUF", {"Spark-X2.6-4B-Q4_K_M.gguf": "spark2_5"}, "model", []),
    ("someone/Foo-GGUF", {"foo.gguf": "foo"}, "model", []),
    ("Qwen/Qwen3-Embedding-0.6B-GGUF", {"Qwen3-Embedding-0.6B-Q8_0.gguf": "qwen3"}, "embedding", ["text_embedding"]),
    ("Serveurperso/Qwen3-TTS-GGUF", {"qwen-talker-0.6b-base-Q4_K_M.gguf": "qwen3-tts"}, "model", ["text_to_speech"]),
    ("unsloth/FLUX.2-klein-4B-GGUF", {"flux-2-klein-4b-Q4_K_M.gguf": "flux"}, "model", ["text_to_image"]),
])
def test_a_gguf_only_repo_says_what_it_is(tmp_path, repo, files, kind, tasks) -> None:
    got = mm._hf_local_kind_tasks(_repo(tmp_path, repo, files))
    assert {k: got[k] for k in ("kind", "tasks", "tasks_source")} == {"kind": kind, "tasks": tasks, "tasks_source": "gguf"}


def test_a_repo_with_its_own_config_keeps_the_config_answer(tmp_path) -> None:
    root = _repo(tmp_path, "org/Model-GGUF", {"m.gguf": "flux"})
    (root / "snapshots" / "abc" / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    assert mm._hf_local_kind_tasks(root)["tasks"] == ["text_generation"]


def test_every_seed_gguf_architecture_is_a_known_text_architecture() -> None:
    """The allow-list holds what the catalog's GGUF repos ship (2026-10-29, Hugging Face API);
    the embedding one (`nomic-bert`) is named by its repo, never listed as text."""
    assert "nomic-bert" not in mm._GGUF_TEXT_ARCH
    for arch in ("qwen35", "gemma4", "llama", "lfm2", "gpt-oss", "deepseek2", "mistral3"):
        assert arch in mm._GGUF_TEXT_ARCH


def test_a_gguf_this_llamacpp_cannot_load_says_so(tmp_path, monkeypatch) -> None:
    """Batch item (f): XHToken/Spark-X2.6-4B (`spark2_5`) failed only at its first call. The
    installed row now says it cannot be loaded here, before it is chosen."""
    from abstractcore.utils import model_cache

    lib = b"\x00llama\x00qwen2\x00qwen35\x00gemma4\x00qwen3tts\x00"
    monkeypatch.setattr(model_cache, "_llama_library_bytes", lambda: lib)
    spark = mm._hf_local_kind_tasks(_repo(tmp_path, "XHToken/Spark-X2.6-4B-GGUF", {"Spark-X2.6-4B-Q4_K_M.gguf": "spark2_5"}))
    assert spark["architecture"] == "spark2_5" and spark["loadable"] is False
    assert "cannot load the spark2_5 architecture" in spark["unloadable_reason"]
    qwen = mm._hf_local_kind_tasks(_repo(tmp_path / "b", "unsloth/Qwen3.5-9B-GGUF", {"Qwen3.5-9B-Q4_K_M.gguf": "qwen35"}))
    assert qwen["tasks"] == ["text_generation"] and "loadable" not in qwen
    # No llama-cpp-python here: nothing is claimed either way.
    monkeypatch.setattr(model_cache, "_llama_library_bytes", lambda: None)
    assert "loadable" not in mm._hf_local_kind_tasks(_repo(tmp_path / "c", "XHToken/Spark-X2.6-4B-GGUF", {"s.gguf": "spark2_5"}))


def test_the_real_llama_library_knows_its_own_architectures() -> None:
    from abstractcore.utils.model_cache import llamacpp_knows_architecture

    known = llamacpp_knows_architecture("llama")
    if known is None:
        pytest.skip("llama-cpp-python is not installed here")
    assert known is True and llamacpp_knows_architecture("no_such_arch_zz") is False


def test_a_library_without_the_table_claims_nothing(tmp_path, monkeypatch) -> None:
    """A libllama found but laid out differently (stripped, a stub): every architecture would read
    unknown to it. The sentinel names are missing, so nothing is claimed (merge desk review)."""
    from abstractcore.utils import model_cache

    monkeypatch.setattr(model_cache, "_llama_library_bytes", lambda: b"\x00qwen35\x00gemma4\x00")  # no llama / qwen2
    row = mm._hf_local_kind_tasks(_repo(tmp_path, "XHToken/Spark-X2.6-4B-GGUF", {"Spark-X2.6-4B-Q4_K_M.gguf": "spark2_5"}))
    assert "loadable" not in row and "unloadable_reason" not in row
    assert model_cache.llamacpp_knows_architecture("qwen35") is None


def test_libllama_is_preferred_over_another_llama_file(tmp_path, monkeypatch) -> None:
    import importlib.util
    from types import SimpleNamespace

    from abstractcore.utils import model_cache

    pkg = tmp_path / "llama_cpp"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "llama.cfg").write_bytes(b"not a library")
    (pkg / "lib" / "libllama.so").write_bytes(b"\x00llama\x00qwen2\x00qwen35\x00")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(pkg)]))
    model_cache._LLAMA_LIB_CACHE.clear()
    assert model_cache.llamacpp_knows_architecture("qwen35") is True
    assert model_cache.llamacpp_knows_architecture("spark2_5") is False
    model_cache._LLAMA_LIB_CACHE.clear()
