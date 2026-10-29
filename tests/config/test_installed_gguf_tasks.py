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
    ("XHToken/Spark-X2.6-4B-GGUF", {"Spark-X2.6-4B-Q4_K_M.gguf": "spark2_5"}, "model", ["text_generation"]),
    ("Qwen/Qwen3-Embedding-0.6B-GGUF", {"Qwen3-Embedding-0.6B-Q8_0.gguf": "qwen3"}, "embedding", ["text_embedding"]),
    ("Serveurperso/Qwen3-TTS-GGUF", {"qwen-talker-0.6b-base-Q4_K_M.gguf": "qwen3-tts"}, "model", ["text_to_speech"]),
    ("unsloth/FLUX.2-klein-4B-GGUF", {"flux-2-klein-4b-Q4_K_M.gguf": "flux"}, "model", ["text_to_image"]),
])
def test_a_gguf_only_repo_says_what_it_is(tmp_path, repo, files, kind, tasks) -> None:
    got = mm._hf_local_kind_tasks(_repo(tmp_path, repo, files))
    assert got == {"kind": kind, "tasks": tasks, "tasks_source": "gguf"}


def test_a_repo_with_its_own_config_keeps_the_config_answer(tmp_path) -> None:
    root = _repo(tmp_path, "org/Model-GGUF", {"m.gguf": "flux"})
    (root / "snapshots" / "abc" / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    assert mm._hf_local_kind_tasks(root)["tasks"] == ["text_generation"]
