"""Installed-model rows say what the artifact IS (`kind`) and can DO (`tasks`).

Local evidence only -- the cached snapshot's files and the engine's own
listing; never a card fetch. What the files cannot say stays `None` / `[]`
with `tasks_source: None`, so a consumer can tell "unknown" from "none".
"""

from __future__ import annotations

import json
import socket

import pytest

from abstractcore.config import model_materializer as mm
from tests.models_engines_fakes import install_fake_lms, isolate_host, make_hf_repo


@pytest.fixture
def host(tmp_path, monkeypatch):
    h = isolate_host(tmp_path, monkeypatch)

    def _no_network(*args, **kwargs):
        raise AssertionError(f"installed-model listing opened a connection: {args[:2]}")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    return h


def _card(pipeline=None, tags=(), library=None):
    lines = ["---", "license: apache-2.0"]
    if library:
        lines.append(f"library_name: {library}")
    if pipeline:
        lines.append(f"pipeline_tag: {pipeline}")
    if tags:
        lines.append("tags:")
        lines += [f"  - {t}" for t in tags]
    lines += ["---", "", "# Model"]
    return "\n".join(lines).encode()


def _cfg(**kw):
    return json.dumps(kw).encode()


def _rows(provider=None):
    payload = mm.list_installed(provider, include_loaded=False)
    return {r["artifact"]: r for r in payload["rows"]}


def _kt(row):
    return row["kind"], row["tasks"], row["tasks_source"]


def test_hf_cache_rows_carry_kind_and_tasks_from_local_files(host):
    hf = host["hf"]
    make_hf_repo(hf, "mlx-community/Tiny-VL-4bit", {
        "config.json": _cfg(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"], vision_config={"depth": 2}),
        "model.safetensors": b"w",
    })
    make_hf_repo(hf, "mlx-community/Tiny-Chat-4bit", {
        "config.json": _cfg(model_type="qwen2", architectures=["Qwen2ForCausalLM"]),
        "model.safetensors": b"w",
    })
    make_hf_repo(hf, "org/text-lora", {
        "adapter_config.json": json.dumps({"task_type": "CAUSAL_LM", "base_model_name_or_path": "org/base"}).encode(),
        "adapter_model.safetensors": b"a",
    })
    make_hf_repo(hf, "org/storybook-lora", {"README.md": _card(tags=["lora", "text-to-image"])})
    make_hf_repo(hf, "org/mini-embed", {
        "config.json": _cfg(model_type="bert", architectures=["BertModel"]),
        "modules.json": b"[]",
        "model.safetensors": b"w",
    })
    make_hf_repo(hf, "org/dino", {"config.json": _cfg(model_type="dinov2", architectures=["Dinov2Model"])})
    make_hf_repo(hf, "org/asr", {"README.md": _card(pipeline="automatic-speech-recognition"), "config.json": _cfg(model_type="x", architectures=["XModel"])})
    make_hf_repo(hf, "org/whisper-mini", {"config.json": _cfg(model_type="whisper", architectures=["WhisperForConditionalGeneration"])})
    make_hf_repo(hf, "org/opaque", {"weights.ckpt": b"?"})

    rows = _rows()
    assert _kt(rows["mlx-community/Tiny-VL-4bit"]) == ("model", ["text_generation", "image_to_text"], "config")
    assert _kt(rows["mlx-community/Tiny-Chat-4bit"]) == ("model", ["text_generation"], "config")
    assert _kt(rows["org/text-lora"]) == ("adapter", ["text_generation"], "adapter_config")
    assert _kt(rows["org/storybook-lora"]) == ("adapter", ["text_to_image"], "model_card")
    assert _kt(rows["org/mini-embed"]) == ("embedding", ["text_embedding"], "files")
    assert _kt(rows["org/dino"]) == ("encoder", [], None)
    # A generative pipeline tag outranks an encoder-looking top-level config.
    assert _kt(rows["org/asr"]) == ("model", ["speech_to_text"], "model_card")
    assert _kt(rows["org/whisper-mini"]) == ("model", ["speech_to_text"], "config")
    # Nothing local says what it is: unknown, not guessed.
    assert _kt(rows["org/opaque"]) == (None, [], None)


def test_lmstudio_rows_take_kind_and_tasks_from_lms_ls(host, monkeypatch):
    install_fake_lms(host["fakebin"], monkeypatch, [
        {"type": "llm", "modelKey": "qwen/qwen3-vl-4b", "path": "qwen/Qwen3-VL-4B-GGUF/q.gguf", "vision": True, "sizeBytes": 10},
        {"type": "llm", "modelKey": "qwen/qwen3-4b", "path": "qwen/Qwen3-4B-GGUF/q.gguf", "vision": False, "sizeBytes": 10},
        {"type": "embedding", "modelKey": "nomic-embed", "path": "nomic/nomic-embed-GGUF/e.gguf", "sizeBytes": 10},
    ])
    rows = list(_rows("lmstudio").values())
    by_key = {r.get("model_key"): r for r in rows}
    assert _kt(by_key["qwen/qwen3-vl-4b"]) == ("model", ["text_generation", "image_to_text"], "engine")
    assert _kt(by_key["qwen/qwen3-4b"]) == ("model", ["text_generation"], "engine")
    assert _kt(by_key["nomic-embed"]) == ("embedding", ["text_embedding"], "engine")


def test_every_row_has_the_fields_even_when_unknown():
    row = mm._installed_row("ollama", "qwen3:4b")
    assert (row["kind"], row["tasks"], row["tasks_source"]) == (None, [], None)
