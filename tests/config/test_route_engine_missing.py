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


@pytest.fixture()
def voice_api(monkeypatch):
    """A stand-in `abstractvoice.engine_runtime` with the public contract
    (`engine_runtime_status(engine)` -> installed / reason / install_command,
    ValueError for an unknown engine). `installed` = the engines that are."""

    def install(installed=()) -> list:
        calls: list = []

        def engine_runtime_status(engine, *, kind=None):
            calls.append(engine)
            if engine not in {"supertonic", "faster-whisper", "piper"}:
                raise ValueError(f"unknown AbstractVoice engine {engine!r}; known engines: supertonic, faster-whisper, piper")
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
            )

        pkg = types.ModuleType("abstractvoice")
        mod = types.ModuleType("abstractvoice.engine_runtime")
        mod.engine_runtime_status = engine_runtime_status
        pkg.engine_runtime = mod
        monkeypatch.setitem(sys.modules, "abstractvoice", pkg)
        monkeypatch.setitem(sys.modules, "abstractvoice.engine_runtime", mod)
        return calls

    return install


def _store(tmp_path, routes: dict) -> ConfigurationManager:
    cfg = tmp_path / "abstractcore.json"
    cfg.write_text(
        json.dumps({"capability_defaults": {"version": 1, "routes": routes, "seeded": "recommended-v1"}}),
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
    # The ONE install allowlist: the Engines screen's own plan, verbatim.
    assert flag["install"] == shlex.join(engines.engine_install_plan("mlx")["argv"])
    assert "mlx_lm missing" in flag["reason"] and flag["install"] in flag["reason"]
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


def test_mlx_gen_needs_abstractvision_and_the_mlx_gen_runtime(packages):
    flag = re_mod.route_engine_missing("mlx-gen", FLUX, "output.image")
    assert flag["engine"] == "mlx-gen"
    assert flag["install"] == engines.pip_install_command("abstractvision[mlx-gen]")
    assert sys.executable in flag["install"], "installs into THIS interpreter, like the engine rows"
    assert "abstractvision, mlx-gen missing" in flag["reason"]
    packages(dists={"abstractvision"})
    assert "(mlx-gen missing)" in re_mod.route_engine_missing("mlx-gen", FLUX, "output.video")["reason"]
    packages(dists={"abstractvision", "mlx-gen"})
    assert re_mod.route_engine_missing("mlx-gen", FLUX, "output.image") is None


def test_voice_asks_abstractvoices_public_probe(packages, voice_api):
    packages(dists={"abstractvoice"})
    calls = voice_api(installed={"faster-whisper"})
    flag = re_mod.route_engine_missing("supertonic", "supertonic-3", "output.voice")
    assert flag == {
        "engine": "supertonic",
        "name": "Supertonic",
        "reason": 'supertonic is not installed. Install it with: pip install "abstractvoice[supertonic]"',
        "install": 'pip install "abstractvoice[supertonic]"',
    }
    assert re_mod.route_engine_missing("faster-whisper", "base", "input.voice") is None
    assert calls == ["supertonic", "faster-whisper"], "every local voice answer is AbstractVoice's own"


def test_an_engine_abstractvoice_does_not_have_is_reported_in_its_words(packages, voice_api):
    packages(dists={"abstractvoice"})
    voice_api()
    flag = re_mod.route_engine_missing("voxtral", "x", "output.voice")
    assert flag["install"] is None and "unknown AbstractVoice engine 'voxtral'" in flag["reason"]


def test_without_abstractvoice_every_local_voice_route_is_missing_it(packages):
    flag = re_mod.route_engine_missing("supertonic", "supertonic-3", "output.voice")
    assert flag["name"] == "AbstractVoice" and flag["install"] == engines.pip_install_command("abstractcore[voice]")
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
    install = engines.pip_install_command(f"abstractvoice>={floor}")
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


def test_apply_recommended_says_the_written_route_still_needs_its_engine(tmp_path, pin_host, packages, voice_api):
    pin_host(MAC)
    packages(dists={"abstractvoice"})
    voice_api()
    manager = ConfigurationManager(config_file=tmp_path / "fresh.json", apply_env=False)
    for key in ("input.text", "output.voice", "output.image", "output.video"):
        manager.clear_capability_default(key)
    report = manager.apply_recommended_capability_defaults(dry_run=True)
    by_key = {row["key"]: row for row in report["routes"]}
    assert by_key["output.voice"]["action"] == "apply"
    assert by_key["output.voice"]["engine_missing"]["install"] == 'pip install "abstractvoice[supertonic]"'
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
