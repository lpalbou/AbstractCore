"""The one rule that splits local repos between the MLX and HuggingFace lists.

The regression these tests exist for: `Jundot/Qwen3.8-27B-oQ4e-mtp` and
`Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` are MLX safetensors with no "mlx" in the
handle, so the old substring test offered them under `huggingface` -- a
provider whose loader refuses them by design -- and hid them from `mlx`, the
only provider that can run them.
"""

import json

import pytest

from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.mlx_model_rules import (
    ENV_FORCE_MLX_PATTERNS,
    ENV_FORCE_NON_MLX_PATTERNS,
    explain_mlx_model,
    is_mlx_model,
    is_mlx_quantization_config,
)

JUNDOT_MODELS = (
    "Jundot/Qwen3.8-27B-oQ4e-mtp",
    "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
)


@pytest.mark.parametrize("model", JUNDOT_MODELS)
def test_jundot_omlx_repos_are_mlx_by_name_alone(model):
    # Name alone has to be enough: a hub entry that was resolved but never
    # downloaded has only `refs/`, so there is no config.json to inspect.
    assert is_mlx_model(model) is True


def test_omlx_quantizer_tag_is_mlx_for_any_publisher():
    assert explain_mlx_model("someone/Model-oQ6-mtp") == (True, "name:oq-quantizer-tag")


def test_publisher_rule_covers_future_jundot_repos():
    assert explain_mlx_model("Jundot/Some-New-Model") == (True, "publisher")


@pytest.mark.parametrize(
    "model",
    [
        "mlx-community/Qwen3.8-27B-4bit",
        "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        "prism-ml/Bonsai-8B-mlx-1bit",
    ],
)
def test_mlx_in_the_name_still_wins(model):
    assert is_mlx_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        # llama.cpp containers: MLX cannot load these, whoever published them.
        "lmstudio-community/Qwen3.8-27B-GGUF",
        "unsloth/Qwen3-4B-GGUF",
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "local/model.gguf",
        # A GGUF quant tag must not be mistaken for oMLX's oQ tag.
        "someone/Llama-3-8B-Q4_K_M",
        "someone/Llama-3-8B-q8_0",
        # Ordinary Transformers repos.
        "microsoft/DialoGPT-medium",
        "Qwen/Qwen3.6-27B",
        "",
    ],
)
def test_non_mlx_repos_stay_out(model):
    assert is_mlx_model(model) is False


def test_mlx_quantization_config_signature():
    # What mlx_lm/mlx_vlm write.
    assert is_mlx_quantization_config({"bits": 4, "group_size": 64, "mode": "affine"}) is True
    # What every Transformers quantizer writes: it names itself.
    assert is_mlx_quantization_config(
        {"quant_method": "awq", "bits": 4, "group_size": 128}
    ) is False
    assert is_mlx_quantization_config({"quant_method": "bitsandbytes", "load_in_4bit": True}) is False
    assert is_mlx_quantization_config({}) is False
    assert is_mlx_quantization_config(None) is False


def _write_repo(root, name, *, config=None, card=None, weights=True):
    """Create a hub-cache entry `models--org--name` with one snapshot.

    `weights` defaults to True because a DOWNLOADED repo is the normal case and
    the listing requires weight files — a repo without them is a resolved-only
    cache entry, which `test_a_resolved_but_undownloaded_repo_is_not_offered`
    builds deliberately.
    """
    entry = root / f"models--{name.replace('/', '--')}"
    snapshot = entry / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (entry / "refs").mkdir(parents=True, exist_ok=True)
    (entry / "refs" / "main").write_text("deadbeef", encoding="utf-8")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"\x00")
    if config is not None:
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if card is not None:
        (snapshot / "README.md").write_text(card, encoding="utf-8")
    return entry


def test_repo_signature_detects_a_neutrally_named_mlx_repo(tmp_path):
    entry = _write_repo(
        tmp_path,
        "acme/neutral-name",
        config={"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}},
    )
    assert is_mlx_model("acme/neutral-name") is False
    assert explain_mlx_model("acme/neutral-name", local_path=entry) == (True, "repo-signature")


def test_repo_signature_reads_the_model_card_library(tmp_path):
    entry = _write_repo(
        tmp_path, "acme/carded", card="---\nlicense: apache-2.0\nlibrary_name: mlx\ntags:\n- mlx\n---\n# hi\n"
    )
    assert is_mlx_model("acme/carded", local_path=entry) is True


def test_mlx_gen_media_repos_are_not_llm_models(tmp_path):
    # MLX format, but an image/video backend owns it; it must not show up in
    # the LLM picker just because its card tags `mlx`.
    entry = _write_repo(
        tmp_path,
        "AbstractFramework/flux.2-klein-9b-8bit",
        card="---\npipeline_tag: text-to-image\nlibrary_name: mlx-gen\ntags:\n- mlx\n- mlx-gen\n---\n",
    )
    assert is_mlx_model("AbstractFramework/flux.2-klein-9b-8bit", local_path=entry) is False


def test_repo_with_no_materialized_snapshot_falls_back_to_name(tmp_path):
    entry = tmp_path / "models--Jundot--Qwen3.8-27B-oQ4e-mtp"
    (entry / "refs").mkdir(parents=True)
    (entry / "refs" / "main").write_text("04dc5509", encoding="utf-8")
    assert is_mlx_model("Jundot/Qwen3.8-27B-oQ4e-mtp", local_path=entry) is True


def test_operator_overrides(monkeypatch):
    monkeypatch.setenv(ENV_FORCE_MLX_PATTERNS, "acme/*-custom,someorg/exact-model")
    assert is_mlx_model("acme/thing-custom") is True  # glob pattern
    assert is_mlx_model("SomeOrg/Exact-Model") is True  # plain substring, case-insensitive
    assert is_mlx_model("acme/other") is False

    monkeypatch.setenv(ENV_FORCE_NON_MLX_PATTERNS, "mlx-community/broken-*")
    assert is_mlx_model("mlx-community/broken-repo") is False
    assert is_mlx_model("mlx-community/other-repo") is True


def _fake_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    hub = tmp_path / ".cache" / "huggingface" / "hub"
    hub.mkdir(parents=True)
    # Pin every cache the listings read to this hub. huggingface_hub freezes
    # HF_HUB_CACHE at its first import: a test that imported it under another
    # (still existing) tmp cache would otherwise leak its repos into this one.
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)
    try:
        import huggingface_hub.constants as hf_constants
    except Exception:
        hf_constants = None
    if hf_constants is not None:
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(hub))
    return hub


def test_the_two_listings_are_complements(monkeypatch, tmp_path):
    hub = _fake_home(monkeypatch, tmp_path)
    for name in JUNDOT_MODELS:
        _write_repo(hub, name)
    _write_repo(hub, "mlx-community/Qwen3.8-27B-4bit")
    _write_repo(hub, "microsoft/DialoGPT-medium")
    _write_repo(hub, "unsloth/Qwen3-4B-GGUF")

    mlx_models = set(MLXProvider.list_available_models())
    hf_models = set(HuggingFaceProvider.list_available_models())

    assert set(JUNDOT_MODELS) <= mlx_models
    assert not (set(JUNDOT_MODELS) & hf_models)
    assert "mlx-community/Qwen3.8-27B-4bit" in mlx_models
    assert {"microsoft/DialoGPT-medium", "unsloth/Qwen3-4B-GGUF"} <= hf_models
    # No repo may be offered by both providers, and none may vanish from both.
    assert not (mlx_models & hf_models)
    assert mlx_models | hf_models == {
        *JUNDOT_MODELS,
        "mlx-community/Qwen3.8-27B-4bit",
        "microsoft/DialoGPT-medium",
        "unsloth/Qwen3-4B-GGUF",
    }


def test_lmstudio_gguf_is_not_offered_as_mlx(monkeypatch, tmp_path):
    _fake_home(monkeypatch, tmp_path)
    store = tmp_path / ".lmstudio" / "models"
    (store / "lmstudio-community" / "Qwen3.8-27B-GGUF").mkdir(parents=True)
    (store / "lmstudio-community" / "Qwen3-VL-4B-Instruct-MLX-4bit").mkdir(parents=True)
    (store / "mlx-community" / "gemma-3-1b-it-qat-4bit").mkdir(parents=True)

    models = set(MLXProvider.list_available_models())

    assert "lmstudio-community/Qwen3.8-27B-GGUF" not in models
    assert "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit" in models
    assert "mlx-community/gemma-3-1b-it-qat-4bit" in models


def test_a_resolved_but_undownloaded_repo_is_not_offered(tmp_path, monkeypatch):
    """A cache entry is not a model.

    `models--Jundot--Qwen3.8-27B-oQ4e-mtp` was 4KB of `refs/` and nothing else
    — the repo had been resolved, never downloaded. Listing it put a model in
    the picker that `_load_model` refuses with "Model not found for MLX
    provider", while printing that same model in its own list of available
    models. An assistant pinned to it failed every single turn.
    """
    hub = _fake_home(monkeypatch, tmp_path)

    # Resolved only: refs, no snapshot, no weights.
    stub = hub / "models--Jundot--Qwen3.8-27B-oQ4e-mtp"
    (stub / "refs").mkdir(parents=True)
    (stub / "refs" / "main").write_text("04dc5509", encoding="utf-8")

    # Downloaded: a snapshot carrying real weight files.
    real = _write_repo(hub, "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp", config={"model_type": "qwen4_exp"})
    (real / "snapshots" / "deadbeef" / "model-00001-of-00002.safetensors").write_bytes(b"\x00")

    models = set(MLXProvider.list_available_models())

    assert "Jundot/Qwen3.8-27B-oQ4e-mtp" not in models, "offered a model with no weights"
    assert "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp" in models


def test_weight_detection_accepts_component_subdirectories(tmp_path, monkeypatch):
    """Diffusers-style repos keep weights under transformer/, vae/, ... ."""
    from abstractcore.providers.mlx_model_rules import has_local_weights

    hub = _fake_home(monkeypatch, tmp_path)
    entry = _write_repo(hub, "acme/split-weights", config={"model_type": "x"})
    component = entry / "snapshots" / "deadbeef" / "transformer"
    component.mkdir()
    (component / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")

    assert has_local_weights(entry) is True
