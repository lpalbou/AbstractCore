"""`engine_missing`: a route whose in-process engine is NOT INSTALLED says so.

Three states, never folded together (route_engines.py):
  route_unavailable  this host cannot run the engine at all (MLX on Linux)
  engine_missing     the host can run it, its software is not installed here
  weights            the engine is here, the model is "not downloaded"

Every test pins which Python packages "exist" (`_importable` / `_distributed`)
so the answer does not depend on the machine running the suite, and voice uses
a stand-in for abstractvoice's public `engine_runtime` module (the real one
ships with abstractvoice >= ABSTRACTVOICE_ENGINE_RUNTIME_FLOOR).
"""

from __future__ import annotations

import json
import shlex
import sys
import types
from dataclasses import dataclass
from typing import Optional

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import engines
from abstractcore.config import route_engines as re_mod
from abstractcore.config.manager import ConfigurationManager
from tests.models_engines_fakes import synthetic_host

MAC = synthetic_host("metal64")
LINUX = synthetic_host("cpu16")
FLUX = "AbstractFramework/flux.2-klein-4b-8bit"
MLX_TEXT = "mlx-community/Qwen3.5-9B-MLX-4bit"


@pytest.fixture()
def pin_host(monkeypatch):
    from abstractcore.utils import host_profile as hp

    def pin(host: dict) -> None:
        monkeypatch.setattr(hp, "host_profile", lambda **_k: dict(host))

    return pin


@pytest.fixture()
def packages(monkeypatch):
    """`packages({"mlx_lm", ...}, dists={"abstractvision", ...})`: exactly
    these modules / distributions exist; everything else is absent."""

    def pin(modules=(), dists=()) -> None:
        mods, ds = set(modules), set(dists)
        monkeypatch.setattr(re_mod, "_importable", lambda name: name in mods)
        monkeypatch.setattr(re_mod, "_distributed", lambda name: name in ds)

    pin()
    return pin


@dataclass(frozen=True)
class _Status:
    engine: str
    label: str
    installed: bool
    install_command: Optional[str]
    reason: Optional[str]
    missing_modules: tuple = ()


@pytest.fixture()
def voice_api(monkeypatch):
    """A stand-in `abstractvoice.engine_runtime` with the public contract
    (`engine_runtime_status(engine)` -> installed / reason / install_command,
    ValueError for an unknown engine). `installed` = the engines that are."""

    def install(installed=()) -> list:
        calls: list = []

        def engine_runtime_status(engine, *, kind=None):
            calls.append(engine)
            if engine not in {"supertonic", "faster-whisper", "mlx-whisper", "piper", "transformers-asr"}:
                raise ValueError(f"unknown AbstractVoice engine {engine!r}; known engines: supertonic, faster-whisper, mlx-whisper, piper, transformers-asr")
            ok = engine in installed
            cmd = {"supertonic": 'pip install "abstractvoice[supertonic]"', "faster-whisper": 'pip install "abstractvoice[stt]"'}.get(
                engine, f'pip install "abstractvoice[{engine}]"'
            )
            return _Status(
                engine=engine,
                label={"supertonic": "Supertonic", "faster-whisper": "faster-whisper"}.get(engine, engine),
                installed=ok,
                install_command=cmd,
                reason=None if ok else f"{engine} is not installed. Install it with: {cmd}",
                missing_modules=() if ok else (
                    {"supertonic": ("onnxruntime",), "faster-whisper": ("faster_whisper",), "mlx-whisper": ("mlx_whisper",)}.get(engine, (engine,))
                ),
            )

        def known_engines(kind=None):
            table = {"tts": ("openai", "supertonic", "piper"), "stt": ("openai", "faster-whisper", "mlx-whisper", "transformers-asr")}
            return table[kind] if kind else ("openai", "supertonic", "piper", "faster-whisper", "mlx-whisper", "transformers-asr")

        pkg = types.ModuleType("abstractvoice")
        mod = types.ModuleType("abstractvoice.engine_runtime")
        mod.engine_runtime_status = engine_runtime_status
        mod.known_engines = known_engines
        pkg.engine_runtime = mod
        monkeypatch.setitem(sys.modules, "abstractvoice", pkg)
        monkeypatch.setitem(sys.modules, "abstractvoice.engine_runtime", mod)
        return calls

    return install


def _store(tmp_path, routes: dict) -> ConfigurationManager:
    cfg = tmp_path / "abstractcore.json"
    cfg.write_text(
        # The CURRENT seed version: an older one would gain the rows later
        # seeds added (`upgrade_recommended_seed`), which is not under test here.
        json.dumps({"capability_defaults": {"version": 1, "routes": routes, "seeded": cd.RECOMMENDED_SEED_VERSION}}),
        encoding="utf-8",
    )
    return ConfigurationManager(config_file=cfg, apply_env=False)


def _rows(manager) -> dict:
    return {row["key"]: row for row in manager.list_capability_defaults()}


# ---------------------------------------------------------------------------
# One answer per in-process engine
# ---------------------------------------------------------------------------


def test_mlx_without_mlx_lm_names_the_engines_screen_command(packages):
    flag = re_mod.route_engine_missing("mlx", MLX_TEXT, "input.text")
    assert flag["engine"] == "mlx" and flag["engine_row"] == "mlx"
    # The ONE install allowlist: the Engines screen's own plan, verbatim. Off Apple silicon the
    # plan has no command (MLX is in no setting there), so the flag carries none either.
    argv = engines.engine_install_plan("mlx")["argv"]
    expected = shlex.join(argv) if argv else None
    assert flag["install"] == expected
    assert "mlx_lm missing" in flag["reason"]
    if expected:
        assert expected in flag["reason"]
    packages({"mlx_lm"})
    assert re_mod.route_engine_missing("mlx", MLX_TEXT, "input.text") is None


@pytest.mark.parametrize(
    "model, engine, modules",
    [
        ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "llamacpp", {"llama_cpp"}),
        ("/models/qwen3-0.6b-q4_k_m.gguf", "llamacpp", {"llama_cpp"}),
        ("Qwen/Qwen3-0.6B", "huggingface", {"transformers", "torch"}),
    ],
)
def test_the_huggingface_lane_follows_the_providers_gguf_rule(packages, model, engine, modules):
    flag = re_mod.route_engine_missing("huggingface", model, "input.text")
    assert flag["engine"] == engine and flag["engine_row"] == engine
    assert flag["install"] == shlex.join(engines.engine_install_plan(engine)["argv"])
    packages(modules)
    assert re_mod.route_engine_missing("huggingface", model, "input.text") is None


def test_transformers_needs_torch_too(packages):
    packages({"transformers"})
    flag = re_mod.route_engine_missing("huggingface", "Qwen/Qwen3-0.6B", "input.text")
    assert "torch missing" in flag["reason"] and "transformers" not in flag["reason"].split("(")[1].split(")")[0]


def _pin_setting(monkeypatch, setting):
    from abstractcore.utils import install_settings

    monkeypatch.setattr(install_settings, "local_engines_setting", lambda: setting)


def test_mlx_gen_needs_abstractvision_and_the_mlx_gen_runtime(packages, monkeypatch):
    _pin_setting(monkeypatch, "apple")
    flag = re_mod.route_engine_missing("mlx-gen", FLUX, "output.image")
    assert flag["engine"] == "mlx-gen"
    # One of the three settings, never AbstractVision's own `abstractvision[mlx-gen]` extra.
    assert flag["install"] == engines.pip_install_command("abstractcore[apple]")
    assert sys.executable in flag["install"], "installs into THIS interpreter, like the engine rows"
    assert "abstractvision, mlx-gen missing" in flag["reason"]
    packages(dists={"abstractvision"})
    video = re_mod.route_engine_missing("mlx-gen", FLUX, "output.video")
    assert "(mlx-gen missing)" in video["reason"] and video["install"] == engines.pip_install_command("abstractcore[apple]")
    # AbstractVision alone missing: it is part of light, so the install is broken.
    packages(dists={"mlx-gen"})
    assert re_mod.route_engine_missing("mlx-gen", FLUX, "output.image")["install"] == engines.pip_install_command(
        "-U", "abstractcore"
    )
    # No setting on this host (Intel Mac, Windows on ARM): no command, a plain sentence.
    _pin_setting(monkeypatch, None)
    packages(dists={"abstractvision"})
    flag = re_mod.route_engine_missing("mlx-gen", FLUX, "output.image")
    assert flag["install"] is None and "not available on this machine" in flag["reason"]
    assert "pip install" not in flag["reason"]
    packages(dists={"abstractvision", "mlx-gen"})
    assert re_mod.route_engine_missing("mlx-gen", FLUX, "output.image") is None


def test_voice_asks_abstractvoices_public_probe(packages, voice_api, monkeypatch):
    _pin_setting(monkeypatch, "apple")
    packages(dists={"abstractvoice"})
    calls = voice_api(installed={"faster-whisper"})
    flag = re_mod.route_engine_missing("supertonic", "supertonic-3", "output.voice")
    install = engines.pip_install_command("abstractcore[apple]")
    # AbstractVoice answers WHETHER; the install is AbstractCore's setting, never
    # AbstractVoice's standalone `abstractvoice[supertonic]` extra.
    assert flag == {
        "engine": "supertonic",
        "name": "Supertonic",
        "reason": f"Supertonic is not installed in this Python environment (onnxruntime missing). Install it with: {install}",
        "install": install,
    }
    assert re_mod.route_engine_missing("faster-whisper", "base", "input.voice") is None
    assert calls == ["supertonic", "faster-whisper"], "every local voice answer is AbstractVoice's own"


def test_an_engine_abstractvoice_does_not_have_is_reported_in_its_words(packages, voice_api):
    packages(dists={"abstractvoice"})
    voice_api()
    flag = re_mod.route_engine_missing("voxtral", "x", "output.voice")
    assert flag["install"] is None
    assert flag["reason"] == "'voxtral' is not a voice engine AbstractVoice has (it has: openai, supertonic, piper). Pick a voice engine for speech output (Multimodal page in the gateway console, Routes in the core console)."


def test_without_abstractvoice_every_local_voice_route_is_missing_it(packages):
    flag = re_mod.route_engine_missing("supertonic", "supertonic-3", "output.voice")
    assert flag["name"] == "AbstractVoice" and flag["install"] == engines.pip_install_command("-U", "abstractcore")
    assert sys.executable in flag["install"]


def _old_abstractvoice(monkeypatch):
    """abstractvoice 0.12: installed, but without `abstractvoice.engine_runtime`."""
    monkeypatch.setitem(sys.modules, "abstractvoice", types.ModuleType("abstractvoice"))
    monkeypatch.setitem(sys.modules, "abstractvoice.engine_runtime", None)  # import raises
    monkeypatch.setattr(re_mod, "_dist_version", lambda dist: "0.12.0")


def test_an_abstractvoice_without_the_probe_is_the_missing_engine(packages, monkeypatch):
    packages(dists={"abstractvoice"})
    _old_abstractvoice(monkeypatch)
    flag = re_mod.route_engine_missing("supertonic", "supertonic-3", "output.voice")
    floor = re_mod.ABSTRACTVOICE_ENGINE_RUNTIME_FLOOR
    install = engines.pip_install_command("-U", "abstractcore")
    assert flag == {
        "engine": "supertonic",
        "name": "AbstractVoice",
        "reason": (
            "abstractvoice 0.12.0 has no public engine runtime probe (abstractvoice.engine_runtime); "
            f"AbstractCore needs abstractvoice>={floor}. Install it with: {install}"
        ),
        "install": install,
    }
    assert sys.executable in install


def test_an_old_abstractvoice_never_takes_the_grid_down(tmp_path, pin_host, packages, monkeypatch):
    pin_host(MAC)
    packages(dists={"abstractvoice"})
    _old_abstractvoice(monkeypatch)
    manager = _store(
        tmp_path,
        {
            "output.voice": {"provider": "supertonic", "model": "supertonic-3"},
            "input.voice": {"provider": "faster-whisper", "model": "base"},
            "input.text": {"provider": "mlx", "model": MLX_TEXT},
        },
    )
    rows = _rows(manager)
    for key in ("output.voice", "input.voice"):
        assert "abstractvoice.engine_runtime" in rows[key]["engine_missing"]["reason"]
    assert rows["input.text"]["engine_missing"]["engine"] == "mlx", "the other rows are still judged"


def test_pip_install_command_targets_this_interpreter(monkeypatch):
    """One helper for every Python install hint: `uv pip install --python <this
    interpreter>` in a pip-less uv venv, else `<this interpreter> -m pip install`."""
    import importlib.util as ilu

    real_find_spec = ilu.find_spec
    monkeypatch.setattr(engines.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(engines.importlib.util, "find_spec", lambda name, *a: None if name == "pip" else real_find_spec(name, *a))
    assert engines.pip_install_command("abstractvoice>=0.13.0") == shlex.join(
        ["uv", "pip", "install", "--python", sys.executable, "abstractvoice>=0.13.0"]
    )
    monkeypatch.setattr(engines.importlib.util, "find_spec", lambda name, *a: object() if name == "pip" else real_find_spec(name, *a))
    assert engines.pip_install_command("abstractvoice>=0.13.0") == shlex.join(
        [sys.executable, "-m", "pip", "install", "abstractvoice>=0.13.0"]
    )


@pytest.mark.parametrize(
    "provider, key",
    [("openai", "output.voice"), ("openai-compatible", "input.voice"), ("lmstudio", "input.text"),
     ("ollama", "input.text"), ("vllm", "input.text"), ("anthropic", "input.text"), ("diffusers", "output.image")],
)
def test_servers_and_clouds_are_never_judged(packages, provider, key):
    assert re_mod.route_engine_missing(provider, "m", key) is None


# ---------------------------------------------------------------------------
# The grid, the apply plan, the recommended plan, the CLI
# ---------------------------------------------------------------------------


def test_the_grid_flags_the_row_and_its_derived_mirror(tmp_path, pin_host, packages):
    pin_host(MAC)
    manager = _store(tmp_path, {"input.text": {"provider": "mlx", "model": MLX_TEXT}})
    rows = _rows(manager)
    flag = rows["input.text"]["engine_missing"]
    assert flag["engine"] == "mlx"
    assert rows["output.text"]["engine_missing"] == flag, "output.text IS input.text"
    assert "route_unavailable" not in rows["input.text"]
    packages({"mlx_lm"})
    assert not any("engine_missing" in row for row in manager.list_capability_defaults())


def test_a_host_that_cannot_run_it_says_route_unavailable_never_both(tmp_path, pin_host, packages):
    pin_host(LINUX)
    manager = _store(tmp_path, {"input.text": {"provider": "mlx", "model": MLX_TEXT}})
    row = _rows(manager)["input.text"]
    assert "route_unavailable" in row and "engine_missing" not in row


def test_apply_recommended_says_the_written_route_still_needs_its_engine(tmp_path, pin_host, packages, voice_api, monkeypatch):
    pin_host(MAC)
    _pin_setting(monkeypatch, "apple")
    packages(dists={"abstractvoice"})
    voice_api()
    manager = ConfigurationManager(config_file=tmp_path / "fresh.json", apply_env=False)
    for key in ("input.text", "output.voice", "output.image", "output.video"):
        manager.clear_capability_default(key)
    report = manager.apply_recommended_capability_defaults(dry_run=True)
    by_key = {row["key"]: row for row in report["routes"]}
    assert by_key["output.voice"]["action"] == "apply"
    assert by_key["output.voice"]["engine_missing"]["install"] == engines.pip_install_command("abstractcore[apple]")
    assert by_key["input.text"]["engine_missing"]["engine"] == "mlx"
    assert by_key["output.image"]["engine_missing"]["engine"] == "mlx-gen"


def test_the_recommended_plan_rows_carry_it(pin_host, packages, voice_api, monkeypatch, tmp_path):
    from abstractcore.config import model_materializer as mm
    from tests.models_engines_fakes import isolate_host

    isolate_host(tmp_path, monkeypatch)
    pin_host(MAC)
    packages(dists={"abstractvoice"})
    voice_api(installed={"supertonic"})
    plan = mm.recommended_plan()
    by_route = {row["route"]: row for row in plan["recommended"]}
    assert "engine_missing" not in by_route["output.voice"]
    assert by_route["output.image"]["engine_missing"]["engine"] == "mlx-gen"
    # Speech input: the download is the Hugging Face repo, the ENGINE is the
    # route's (round 16 on Apple silicon: mlx-whisper), never AbstractVoice asked for "huggingface".
    voice = by_route["input.voice"]
    assert (voice["provider"], voice["artifact"]) == ("huggingface", "mlx-community/whisper-large-v3-mlx")
    assert (voice["route_provider"], voice["route_model"]) == ("mlx-whisper", "large-v3")
    assert voice["engine_missing"]["engine"] == "mlx-whisper"
    assert "mlx_whisper missing" in voice["engine_missing"]["reason"]


def test_the_recommended_plan_judges_the_routes_engine_not_the_download_provider(pin_host, packages, voice_api, monkeypatch, tmp_path):
    """Round 2 item 1, reproduced on a hermetic gateway (/models/availability):
    input.voice carried engine_missing "unknown AbstractVoice engine 'huggingface'"."""

    from abstractcore.config import model_materializer as mm
    from tests.models_engines_fakes import isolate_host

    isolate_host(tmp_path, monkeypatch)
    pin_host(MAC)
    packages(dists={"abstractvoice"})
    calls = voice_api(installed={"supertonic", "faster-whisper", "mlx-whisper"})
    by_route = {row["route"]: row for row in mm.recommended_plan()["recommended"]}
    assert "engine_missing" not in by_route["input.voice"]
    assert "huggingface" not in calls


def test_speech_input_aliases_resolve_to_abstractvoices_engines(packages, voice_api):
    """The plugin's STT aliases (abstractcore_plugin.py `_norm_compat_provider_id`)."""

    packages(dists={"abstractvoice"})
    calls = voice_api(installed={"faster-whisper", "transformers-asr"})
    for alias in ("whisper", "local", "faster_whisper", "Faster-Whisper"):
        assert re_mod.route_engine_missing(alias, "base", "input.voice") is None, alias
    for alias in ("hf", "hf-asr", "transformers", "transformers_asr"):
        assert re_mod.route_engine_missing(alias, "openai/whisper-small", "input.voice") is None, alias
    assert set(calls) == {"faster-whisper", "transformers-asr"}
    # Output voice keeps AbstractVoice's own ids: "local" is not a TTS engine.
    assert re_mod.voice_engine_id("local", "output.voice") == "local"


def test_a_download_source_on_the_transcription_route_says_what_to_do(packages, voice_api):
    packages(dists={"abstractvoice"})
    voice_api(installed={"faster-whisper"})
    flag = re_mod.route_engine_missing("huggingface", "Systran/faster-whisper-base", "input.voice")
    assert flag["install"] is None and flag["engine"] == "huggingface"
    assert flag["reason"] == (
        "'huggingface' is not a transcription engine AbstractVoice has (it has: openai, faster-whisper, mlx-whisper, "
        "transformers-asr). Pick a transcription engine for speech input (Multimodal page in the gateway console, "
        "Routes in the core console)."
    )


# ---------------------------------------------------------------------------
# The repair of a route stored with the download pair
# ---------------------------------------------------------------------------


def _raw_store(tmp_path, routes: dict):
    cfg = tmp_path / "abstractcore.json"
    raw = json.dumps({"capability_defaults": {"version": 1, "routes": routes, "seeded": cd.RECOMMENDED_SEED_VERSION}})
    cfg.write_text(raw, encoding="utf-8")
    return cfg, raw


def test_a_voice_route_holding_the_download_pair_is_repaired_on_disk_with_a_backup(tmp_path, packages, voice_api):
    packages(dists={"abstractvoice"})
    voice_api(installed={"faster-whisper"})
    cfg, raw = _raw_store(tmp_path, {
        "input.voice": {"provider": "huggingface", "model": "Systran/faster-whisper-base", "options": {"language": "fr"}},
        "output.voice": {"provider": "supertonic", "model": "supertonic-3"},
    })
    manager = ConfigurationManager(config_file=cfg, apply_env=False)
    stored = json.loads(cfg.read_text())["capability_defaults"]["routes"]
    assert stored["input.voice"] == {"provider": "faster-whisper", "model": "base", "options": {"language": "fr"}}
    assert stored["output.voice"] == {"provider": "supertonic", "model": "supertonic-3"}
    backups = sorted(tmp_path.glob("abstractcore.json.route-repair-*.bak"))
    assert len(backups) == 1 and backups[0].read_text() == raw
    assert manager._download_pair_repairs[0]["before"] == {"provider": "huggingface", "model": "Systran/faster-whisper-base"}
    rows = _rows(manager)
    assert (rows["input.voice"]["provider"], rows["input.voice"]["model"]) == ("faster-whisper", "base")
    assert "engine_missing" not in rows["input.voice"]
    # Idempotent: a second load has nothing to repair and writes no backup.
    before = cfg.read_bytes()
    again = ConfigurationManager(config_file=cfg, apply_env=False)
    assert again._download_pair_repairs == [] and cfg.read_bytes() == before
    assert len(list(tmp_path.glob("abstractcore.json.route-repair-*.bak"))) == 1


def test_an_unknown_pair_is_left_alone_and_the_grid_says_what_to_do(tmp_path, packages, voice_api):
    packages(dists={"abstractvoice"})
    voice_api(installed={"faster-whisper"})
    cfg, raw = _raw_store(tmp_path, {"input.voice": {"provider": "huggingface", "model": "someone/whisper-custom"}})
    manager = ConfigurationManager(config_file=cfg, apply_env=False)
    assert cfg.read_text() == raw and not list(tmp_path.glob("*.route-repair-*.bak"))
    flag = _rows(manager)["input.voice"]["engine_missing"]
    assert "Pick a transcription engine for speech input" in flag["reason"]


def test_the_repair_only_follows_the_catalog_route_key():
    doc = {"capability_defaults": {"routes": {
        # The same download pair on a route the catalog does not name for it: untouched.
        "output.voice": {"provider": "huggingface", "model": "Systran/faster-whisper-base"},
        "input.voice": {"provider": "huggingface", "model": "Systran/faster-whisper-base"},
    }}}
    changes = cd.repair_download_pair_routes(doc)
    assert [c["key"] for c in changes] == ["input.voice"]
    assert doc["capability_defaults"]["routes"]["output.voice"]["provider"] == "huggingface"
    assert cd.repair_download_pair_routes(doc) == []


def test_every_recommendation_run_by_another_engine_is_in_the_catalog():
    """The catalog artifact's `route` and RECOMMENDED_MODELS never disagree: a
    recommended route whose provider is not its download provider is exactly
    the catalog's `route` for that download."""

    from abstractcore.config.model_catalog import route_for_download

    picks = []
    for key, rec in cd.RECOMMENDED_MODELS.items():
        picks.append((key, rec.route, rec.download))
        picks.extend((key, alt.route, alt.download) for alt in rec.by_accelerator.values())
    checked = 0
    for key, route, download in picks:
        if route.provider == download["provider"]:
            continue
        checked += 1
        assert route_for_download(download["provider"], download["artifact"]) == {
            "key": key, "provider": route.provider, "model": route.model,
        }, key
    assert checked >= 1


def test_the_cli_grid_and_models_status_print_it(tmp_path, pin_host, packages, capsys):
    from abstractcore.config import main as cli

    pin_host(MAC)
    manager = _store(tmp_path, {"input.text": {"provider": "mlx", "model": MLX_TEXT}})
    rows = manager.list_capability_defaults()
    cli._print_models_status({"config_file": "x", "routes": rows, "recommended": {}, "show_all": False})
    out = capsys.readouterr().out
    assert "⚠️ engine not installed: MLX (mlx-lm) is not installed" in out
    cli._print_capability_defaults({"config_file": "x", "routes": rows})
    out = capsys.readouterr().out
    assert f"- input.text: mlx/{MLX_TEXT} (" in out and "⚠️ engine not installed: MLX (mlx-lm)" in out


def test_the_transcription_routes_weights_are_probed_where_the_download_puts_them(pin_host, monkeypatch, tmp_path):
    """mlx-whisper `large-v3` (the Apple silicon recommendation since round 16) is the
    Hugging Face repo mlx-community/whisper-large-v3-mlx (catalog `route`): the grid
    probes that repo, not a provider named mlx-whisper that has no materializer."""

    from abstractcore.config import model_materializer as mm
    from tests.models_engines_fakes import isolate_host

    isolate_host(tmp_path, monkeypatch)
    pin_host(MAC)
    probed = []

    def fake_probe(provider, artifact, base_url=None):
        probed.append((provider, artifact))
        return mm.ModelPresence(provider, artifact, mm.PRESENCE_INSTALLED, evidence="test")

    monkeypatch.setattr(mm, "probe", fake_probe)
    rows = mm.annotate_route_availability([{"key": "input.voice", "provider": "mlx-whisper", "model": "large-v3"}])
    assert probed == [("huggingface", "mlx-community/whisper-large-v3-mlx")]
    assert rows[0]["availability"]["status"] == mm.PRESENCE_INSTALLED
    assert rows[0]["download_provider"] == "huggingface"
    assert rows[0]["download_artifact"] == "mlx-community/whisper-large-v3-mlx"
