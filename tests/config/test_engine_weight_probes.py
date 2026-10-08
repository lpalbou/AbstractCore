"""Every provider a capability route can name has a local-weights answer.

Operator review 2026-10-07: the gateway console's Voice Input row
(faster-whisper / large-v3) read "not checked -- AbstractCore has no
local-weights probe for provider 'faster-whisper'" on a machine that had
`Systran/faster-whisper-large-v3` in its Hugging Face cache. Output sound and
music (stable-audio-3) said the same. These tests pin, per provider, that the
probe maps the route's model id to the storage the ENGINE uses (with the
engine's own table), answers installed / absent / remote with the evidence
path and one short `summary` sentence, and never touches the network or runs a
download tool while doing so.

Fixtures are cache LAYOUTS built under tmp_path (`huggingface_hub`'s
`models--org--name/{blobs,snapshots,refs}`, Piper's voice folder,
AbstractVoice's cloning folders); HOME points at tmp_path, so nothing on the
machine running the tests is read or written.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
from pathlib import Path

import pytest

from abstractcore.config import model_materializer as mm


def _hf_repo(root: Path, repo_id: str, files=("model.safetensors",)) -> Path:
    repo = root / ("models--" + repo_id.replace("/", "--"))
    blobs = repo / "blobs"
    snap = repo / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    for index, name in enumerate(files):
        blob = blobs / f"sha{index}"
        blob.write_bytes(b"\x00" * 16)
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).symlink_to(blob)
    return repo


class _NetworkUsed(AssertionError):
    pass


@pytest.fixture()
def offline(monkeypatch, tmp_path):
    """A scratch HOME + HF cache, and a machine with NO network and NO subprocess:
    any socket connect, hub call or child process fails the test."""

    home = tmp_path / "home"
    home.mkdir()
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(mm, "_hf_cache_dirs", lambda: [cache])

    def no_net(*_a, **_k):
        raise _NetworkUsed("a weights probe opened a network connection")

    def no_proc(*_a, **_k):
        raise _NetworkUsed("a weights probe started a subprocess")

    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.setattr(subprocess, "Popen", no_proc)
    monkeypatch.setattr(subprocess, "run", no_proc)
    try:
        import huggingface_hub

        for name in ("snapshot_download", "hf_hub_download", "model_info", "list_repo_files"):
            if hasattr(huggingface_hub, name):
                monkeypatch.setattr(huggingface_hub, name, no_net)
    except Exception:
        pass
    monkeypatch.setattr(mm, "_download_huggingface", no_net)
    return {"home": home, "cache": cache}


def _needs(module: str):
    return pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} is not installed here")


# (provider as a route stores it, model as a route stores it, the HF repo the engine loads)
HF_ENGINE_CASES = [
    pytest.param("faster-whisper", "large-v3", "Systran/faster-whisper-large-v3", marks=_needs("faster_whisper"), id="faster-whisper"),
    pytest.param("faster-whisper", "large", "Systran/faster-whisper-large-v3", marks=_needs("faster_whisper"), id="faster-whisper-alias"),
    pytest.param("whisper", "base", "Systran/faster-whisper-base", marks=_needs("faster_whisper"), id="whisper-alias"),
    pytest.param("faster-whisper", "Systran/faster-distil-whisper-large-v3", "Systran/faster-distil-whisper-large-v3", id="faster-whisper-repo"),
    pytest.param("transformers-asr", "whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", marks=_needs("abstractvoice"), id="transformers-asr-alias"),
    pytest.param("transformers-asr", "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B", id="transformers-asr"),
    pytest.param("audiodit", "meituan-longcat/LongCat-AudioDiT-1B", "meituan-longcat/LongCat-AudioDiT-1B", id="audiodit"),
    pytest.param("omnivoice", "k2-fsa/OmniVoice", "k2-fsa/OmniVoice", id="omnivoice"),
    pytest.param("qwen3-tts", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", id="qwen3-tts"),
    pytest.param("stable-audio-3", "stabilityai/stable-audio-3-medium", "stabilityai/stable-audio-3-medium", id="stable-audio-3"),
    pytest.param("stable-audio", "stabilityai/stable-audio-open-small", "stabilityai/stable-audio-open-small", id="stable-audio"),
    pytest.param("acestep", "ACE-Step/acestep-v15-xl-turbo-diffusers", "ACE-Step/acestep-v15-xl-turbo-diffusers", id="acestep"),
    pytest.param("triposr", "stabilityai/TripoSR", "stabilityai/TripoSR", id="triposr"),
    pytest.param("trellis2", "microsoft/TRELLIS.2-4B", "microsoft/TRELLIS.2-4B", id="trellis2"),
    pytest.param("abstract3d:step1x-local", "stepfun-ai/Step1X-3D", "stepfun-ai/Step1X-3D", id="step1x"),
    pytest.param("hunyuan3d", "tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2.1", id="hunyuan3d"),
    pytest.param("sdcpp", "FLUX.2-dev", "city96/FLUX.2-dev-gguf", marks=_needs("abstractvision"), id="sdcpp-preset"),
    pytest.param("mlx-gen", "klein-4b", "AbstractFramework/flux.2-klein-4b-4bit", marks=_needs("abstractvision"), id="mlx-gen-preset"),
]


@pytest.mark.parametrize("provider,model,repo", HF_ENGINE_CASES)
def test_an_engine_model_id_is_probed_in_the_repo_the_engine_loads(offline, provider, model, repo):
    absent = mm.probe(provider, model)
    assert absent.status == mm.PRESENCE_ABSENT, absent
    assert absent.evidence == "hf cache scan"
    assert absent.summary_text() == "Not in the Hugging Face cache."
    assert repo in absent.detail, "the detail names the repository that was looked for"
    assert absent.instruction and absent.instruction.endswith(f" {model}"), "the fix names the route's own model id"
    assert absent.downloadable is True

    _hf_repo(offline["cache"], repo, files=("model.bin" if provider in ("faster-whisper", "whisper") else "model.safetensors",))
    present = mm.probe(provider, model)
    assert present.status == mm.PRESENCE_INSTALLED, present
    assert present.location and ("models--" + repo.replace("/", "--")) in present.location, "evidence path"
    assert present.to_dict()["summary"] == "In the Hugging Face cache."
    assert present.artifact == model


def test_the_operators_voice_input_row_reads_installed_with_its_snapshot(offline):
    """The reported row, end to end through the grid annotation."""

    _hf_repo(offline["cache"], "Systran/faster-whisper-large-v3", files=("model.bin", "config.json", "tokenizer.json"))
    (row,) = mm.annotate_route_availability([{"key": "input.voice", "provider": "faster-whisper", "model": "large-v3"}])
    availability = row["availability"]
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("faster-whisper is not installed: its short names cannot be resolved here")
    assert availability["status"] == mm.PRESENCE_INSTALLED
    assert availability["summary"] == "In the Hugging Face cache."
    assert "models--Systran--faster-whisper-large-v3" in availability["location"]
    assert "no local-weights probe" not in str(availability)


def test_an_unknown_faster_whisper_name_says_so_with_the_engines_names(offline):
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("faster-whisper is not installed here")
    presence = mm.probe("faster-whisper", "huge-v9")
    assert presence.status == mm.PRESENCE_UNKNOWN
    assert presence.summary_text() == "Unknown model id."
    assert "large-v3" in presence.detail and "huge-v9" in presence.detail


def test_faster_whisper_names_resolve_without_importing_the_engine(offline, monkeypatch):
    """`import faster_whisper` loads CTranslate2 and torch; the probe reads the
    engine's table from its source instead."""

    import sys

    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("faster-whisper is not installed here")
    for name in [m for m in sys.modules if m == "faster_whisper" or m.startswith("faster_whisper.")]:
        monkeypatch.delitem(sys.modules, name)
    mm._FW_TABLE.clear()
    mm.probe("faster-whisper", "large-v3")
    assert "faster_whisper" not in sys.modules, "the probe imported the faster-whisper engine"


@_needs("abstractvoice")
def test_piper_voice_in_its_voice_folder(offline):
    folder = offline["home"] / ".piper" / "models"
    presence = mm.probe("piper", "en_US-amy-medium")
    assert presence.status == mm.PRESENCE_ABSENT
    assert presence.instruction == "abstractvoice-prefetch --piper en"
    assert presence.downloadable is False
    folder.mkdir(parents=True)
    (folder / "en_US-amy-medium.onnx").write_bytes(b"\x00")
    (folder / "en_US-amy-medium.onnx.json").write_text("{}")
    presence = mm.probe("piper", "en_US-amy-medium")
    assert presence.status == mm.PRESENCE_INSTALLED
    assert presence.location == str(folder / "en_US-amy-medium.onnx")
    assert presence.summary_text() == "In Piper's voice folder."
    assert mm.probe("piper", "xx_XX-nobody-low").status == mm.PRESENCE_UNKNOWN


def _importable(module: str) -> bool:
    """True when `module` imports here (its own dependencies included)."""
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _needs_import(module: str):
    return pytest.mark.skipif(not _importable(module), reason=f"{module} does not import here")


@pytest.mark.parametrize(
    "provider,folder,files,flag",
    [
        pytest.param("f5_tts", "openf5", ("cfg/model.yaml", "model.pt", "vocab.txt"), "--openf5",
                     marks=_needs_import("abstractvoice.cloning.engine_f5"), id="f5_tts"),
        pytest.param("chroma", "chroma", ("config.json", "model.safetensors.index.json"), "--chroma",
                     marks=_needs_import("abstractvoice.cloning.engine_chroma"), id="chroma"),
    ],
)
def test_cloning_engines_answer_from_their_own_folder(offline, provider, folder, files, flag):
    presence = mm.probe(provider, "default")
    assert presence.status == mm.PRESENCE_ABSENT
    assert presence.instruction == f"abstractvoice-prefetch {flag}"
    root = offline["home"] / ".cache" / "abstractvoice" / folder
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x")
    presence = mm.probe(provider, "default")
    assert presence.status == mm.PRESENCE_INSTALLED
    assert presence.location == str(root)


@pytest.mark.parametrize("provider", ["elevenlabs-music", "acemusic", "abstractmusic:elevenlabs-music"])
def test_remote_music_services_are_remote(offline, provider):
    presence = mm.probe(provider, "elevenlabs/music_v1")
    assert presence.status == mm.PRESENCE_NOT_APPLICABLE
    assert presence.summary_text() == "Served remotely."


def test_a_local_checkpoint_path_is_answered_from_disk(offline, tmp_path):
    ckpt = tmp_path / "my-whisper"
    presence = mm.probe("faster-whisper", str(ckpt))
    assert presence.status == mm.PRESENCE_ABSENT and presence.summary_text() == "Path not found."
    ckpt.mkdir()
    (ckpt / "model.bin").write_bytes(b"\x00")
    presence = mm.probe("faster-whisper", str(ckpt))
    assert presence.status == mm.PRESENCE_INSTALLED and presence.location == str(ckpt)


def test_only_a_truly_unknown_provider_keeps_the_no_probe_answer(offline):
    presence = mm.probe("mystery-engine", "some/model")
    assert presence.status == mm.PRESENCE_UNKNOWN
    assert presence.evidence == "no materializer"
    assert "mystery-engine" in presence.detail and "mystery-engine" in presence.summary_text()
    assert "Check mystery-engine's own model list" in (presence.instruction or "")


def _route_provider_inventory():
    """Every provider id a capability route can name, from the owners' own lists:
    AbstractVoice's engines, core's music and scene3d selectors, the vision
    and text/embedding providers the gateway's discovery offers."""

    ids = set()
    try:
        from abstractvoice.engine_runtime import known_engines

        ids.update(known_engines())
    except Exception:
        ids.update({"openai", "openai-compatible", "supertonic", "piper", "audiodit", "qwen3-tts", "omnivoice", "f5_tts", "chroma", "faster-whisper", "transformers-asr"})
    from abstractcore.capabilities.music_selectors import MUSIC_BACKEND_NAMES
    from abstractcore.capabilities.scene3d_selectors import SCENE3D_BACKEND_ALIASES, SCENE3D_BACKEND_IDS

    ids.update(MUSIC_BACKEND_NAMES)
    ids.update(SCENE3D_BACKEND_IDS)
    ids.update(SCENE3D_BACKEND_ALIASES)
    ids.update({"openai", "openai-compatible", "huggingface", "mlx-gen", "mflux", "sdcpp", "diffusers"})  # image / video
    ids.update({"lmstudio", "ollama", "mlx", "mlx-vlm", "huggingface", "vllm", "openrouter", "portkey", "anthropic"})  # text, embedding, rerank
    return sorted(ids)


@pytest.mark.parametrize("provider", _route_provider_inventory())
def test_no_route_provider_falls_back_to_the_no_probe_answer(offline, monkeypatch, provider):
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(mm, "_lms_cli", lambda: None)
    monkeypatch.setattr(mm, "_http_json", lambda *_a, **_k: (None, "refused (test)"))
    monkeypatch.setattr(mm.shutil, "which", lambda *_a, **_k: None)
    presence = mm.probe(provider, "acme/some-model")
    assert presence.evidence != "no materializer", f"{provider} has no local-weights probe: {presence}"
    assert presence.status in mm.PRESENCE_STATES
    assert presence.to_dict()["summary"], "every answer carries its short sentence"


def test_every_state_has_a_short_sentence():
    for status in mm.PRESENCE_STATES:
        text = mm.ModelPresence("p", "m", status).to_dict()["summary"]
        assert text and len(text) <= 40 and text.endswith(".")


def test_downloading_an_engine_id_fetches_the_engines_repo(offline):
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("faster-whisper is not installed here")
    outcome = mm.download("faster-whisper", "large-v3", dry_run=True)
    assert outcome.status == "planned", outcome
    assert outcome.command[:2] == ["huggingface_hub.snapshot_download", "Systran/faster-whisper-large-v3"]


# ---------------------------------------------------------------------------
# Route aliases: the ids a voice route accepts besides engine ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,provider,model,repo",
    [
        pytest.param("input.voice", "local", "large-v3", "Systran/faster-whisper-large-v3", marks=_needs("faster_whisper"), id="local"),
        pytest.param("input.voice", "hf", "whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", marks=_needs("abstractvoice"), id="hf"),
        pytest.param("input.voice", "faster_whisper", "base", "Systran/faster-whisper-base", marks=_needs("faster_whisper"), id="faster_whisper"),
    ],
)
def test_a_voice_route_alias_is_probed_as_the_engine_it_runs(offline, key, provider, model, repo):
    """`local` / `hf` on input.voice are faster-whisper / transformers-asr (core's
    route_engines.voice_engine_id): the weights are those of the engine the route runs."""

    (row,) = mm.annotate_route_availability([{"key": key, "provider": provider, "model": model}])
    assert row["availability"]["status"] == mm.PRESENCE_ABSENT, row
    assert row["availability"]["evidence"] != "no materializer"
    _hf_repo(offline["cache"], repo, files=("model.bin",))
    (row,) = mm.annotate_route_availability([{"key": key, "provider": provider, "model": model}])
    assert row["availability"]["status"] == mm.PRESENCE_INSTALLED, row
    assert ("models--" + repo.replace("/", "--")) in row["availability"]["location"]


@_needs("abstractvoice")
@pytest.mark.parametrize("provider", ["remote", "compatible", "proxy"])
def test_abstractvoices_relay_aliases_are_remote_on_a_voice_route(offline, provider):
    """AbstractVoice's own alias table (engine_runtime.normalize_engine_id) names these
    openai-compatible: a relay, so the weights are 'served remotely'."""

    for key in ("output.voice", "input.voice"):
        (row,) = mm.annotate_route_availability([{"key": key, "provider": provider, "model": "tts-1"}])
        assert row["availability"]["status"] == mm.PRESENCE_NOT_APPLICABLE, (key, row)
        assert row["availability"]["summary"] == "Served remotely."


def test_route_aliases_only_apply_on_voice_routes(offline):
    """`local` on a text route is not faster-whisper."""

    (row,) = mm.annotate_route_availability([{"key": "input.text", "provider": "local", "model": "large-v3"}])
    assert row["availability"]["evidence"] == "no materializer"
