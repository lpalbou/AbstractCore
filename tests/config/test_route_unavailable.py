"""`route_unavailable`: a CONFIGURED route this host cannot run is never shown as fine.

Review finding M8 (2026-09-27): Linux, Windows and Intel-Mac installs seeded
before the host-aware recommendation still hold `output.image:
mlx-gen/...flux.2-klein-4b-8bit`. The grid flagged only UNSET rows
(`recommendation_unavailable`), and `apply-recommended --force` kept the broken
route because the image recommendation is `unavailable` on those hosts.

The contract (lead, REVIEW-1.md):
  - every grid row whose configured provider cannot run here carries
    `route_unavailable: {provider, model, reason}` (the shape of
    `recommendation_unavailable`); absent otherwise;
  - apply-recommended without `force` never touches it, and flags it;
  - with `force` it is replaced by the host's recommendation, or -- where
    nothing recommended runs here -- removed (`cleared`) with the reason;
  - the CLI grid prints it.
Only in-process providers are judged (mlx, mlx-gen, supertonic): a server or
cloud route may run from any host.

Also here: the host-support matrix now checks the ARCHITECTURE for LM Studio,
Ollama and Supertonic's ONNX Runtime, and a host with no text engine at all
(FreeBSD, 32-bit ARM) gets one consistent answer everywhere.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import engines
from abstractcore.config import model_catalog as mc
from abstractcore.config.manager import ConfigurationManager
from tests.models_engines_fakes import isolate_host, synthetic_host

LINUX = synthetic_host("cpu16")
ARMV7 = dict(LINUX, arch="armv7l")
FREEBSD = dict(LINUX, os="freebsd")
FLUX = "AbstractFramework/flux.2-klein-4b-8bit"


@pytest.fixture()
def pin_host(monkeypatch):
    from abstractcore.utils import host_profile as hp

    def pin(host: dict) -> None:
        monkeypatch.setattr(hp, "host_profile", lambda **_k: dict(host))

    return pin


def _stale_store(tmp_path, routes: dict):
    """A store written BEFORE the host-aware recommendation (file exists, so
    no seed runs): exactly what an old Linux install carries."""

    cfg = tmp_path / "abstractcore.json"
    cfg.write_text(
        json.dumps({"capability_defaults": {"version": 1, "routes": routes, "seeded": "recommended-v1"}}),
        encoding="utf-8",
    )
    return ConfigurationManager(config_file=cfg, apply_env=False)


def _stored(manager) -> dict:
    raw = json.loads(manager.config_file.read_text(encoding="utf-8"))
    return raw["capability_defaults"]["routes"]


def _rows(manager) -> dict:
    return {row["key"]: row for row in manager.list_capability_defaults()}


def _entry(report: dict, key: str) -> dict:
    return next(row for row in report["routes"] if row["key"] == key)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_a_stale_mlx_gen_image_route_on_linux_is_flagged_in_the_grid(tmp_path, pin_host):
    pin_host(LINUX)
    manager = _stale_store(tmp_path, {"output.image": {"provider": "mlx-gen", "model": FLUX}})
    image = _rows(manager)["output.image"]
    assert image["configured"] is True and image["provider"] == "mlx-gen"
    flag = image["route_unavailable"]
    assert set(flag) == {"provider", "model", "reason"}
    assert (flag["provider"], flag["model"]) == ("mlx-gen", FLUX)
    assert "Apple Silicon" in flag["reason"]
    assert "diffusers" in flag["reason"], "the reason names what runs here instead"
    # A configured row is not "unset with no recommendation".
    assert "recommendation_unavailable" not in image


def test_task_rows_and_the_derived_text_row_are_flagged_too(tmp_path, pin_host):
    pin_host(LINUX)
    manager = _stale_store(
        tmp_path,
        {
            "input.text": {"provider": "mlx", "model": "mlx-community/Qwen3.5-9B-MLX-4bit"},
            "output.image.image_to_image": {"provider": "mlx-gen", "model": FLUX},
            "output.video.text_to_video": {"provider": "mlx-gen", "model": "x/wan"},
        },
    )
    rows = _rows(manager)
    text = rows["input.text"]["route_unavailable"]
    assert "MLX runs only on Apple Silicon" in text["reason"]
    assert "recommended route is lmstudio/qwen/qwen3.5-9b" in text["reason"], "names this host's pick"
    assert rows["output.text"]["route_unavailable"] == text, "output.text IS input.text"
    assert "image generation" in rows["output.image.image_to_image"]["route_unavailable"]["reason"]
    video = rows["output.video.text_to_video"]["route_unavailable"]["reason"]
    assert video.startswith("MLX-Gen video generation needs MLX"), "worded by the row's own modality"


@pytest.mark.parametrize(
    "route",
    [
        {"provider": "diffusers", "model": "black-forest-labs/FLUX.2-klein-4B"},
        {"provider": "openai", "model": "gpt-image-1"},
        # A server provider may be on another machine: never judged by this host.
        {"provider": "lmstudio", "model": "qwen/qwen3.5-9b", "base_url": "http://gpu-box:1234/v1"},
    ],
)
def test_a_route_that_can_run_carries_no_flag(tmp_path, pin_host, route):
    pin_host(ARMV7)  # no LM Studio build here, yet the lmstudio route is a remote server
    manager = _stale_store(tmp_path, {"output.image": route})
    assert "route_unavailable" not in _rows(manager)["output.image"]


def test_apple_silicon_grid_has_no_flag_and_an_unchanged_report_shape(tmp_path, pin_host):
    pin_host(synthetic_host("metal64"))
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    manager.set_capability_default("output", "image", task="image_to_image", provider="mlx-gen", model=FLUX)
    assert not any("route_unavailable" in row for row in manager.list_capability_defaults())
    report = manager.apply_recommended_capability_defaults(force=True, dry_run=True)
    assert report["cleared"] == 0
    assert not any("route_unavailable" in entry for entry in report["routes"])


# ---------------------------------------------------------------------------
# apply-recommended
# ---------------------------------------------------------------------------


def test_without_force_a_broken_route_is_kept_and_flagged(tmp_path, pin_host):
    pin_host(LINUX)
    manager = _stale_store(
        tmp_path,
        {
            "input.text": {"provider": "mlx", "model": "m/q"},
            "output.image": {"provider": "mlx-gen", "model": FLUX, "options": {"steps": 4}},
        },
    )
    report = manager.apply_recommended_capability_defaults()

    image = _entry(report, "output.image")
    assert image["action"] == "unavailable" and image["changed"] is False
    assert image["after"] == image["before"]
    assert image["route_unavailable"]["provider"] == "mlx-gen"
    text = _entry(report, "input.text")
    assert text["action"] == "kept" and text["route_unavailable"]["model"] == "m/q"
    assert report["cleared"] == 0
    stored = _stored(manager)
    assert stored["input.text"] == {"provider": "mlx", "model": "m/q"}, "untouched without force"
    assert stored["output.image"] == {"provider": "mlx-gen", "model": FLUX, "options": {"steps": 4}}


def test_force_replaces_or_clears_a_broken_route(tmp_path, pin_host):
    pin_host(LINUX)
    manager = _stale_store(
        tmp_path,
        {
            "input.text": {"provider": "mlx", "model": "m/q"},
            "output.image": {"provider": "mlx-gen", "model": FLUX, "options": {"steps": 4}},
            "output.voice": {"provider": "supertonic", "model": "supertonic-3"},
        },
    )
    report = manager.apply_recommended_capability_defaults(force=True)

    image = _entry(report, "output.image")
    assert image["action"] == "cleared" and image["changed"] is True
    assert image["after"] == {} and "Apple Silicon" in image["reason"]
    assert image["route_unavailable"]["model"] == FLUX
    text = _entry(report, "input.text")
    assert text["action"] == "overwrite" and text["route_unavailable"]["provider"] == "mlx"
    assert report["cleared"] == 1

    stored = _stored(manager)
    assert "output.image" not in stored, "removed whole, options included"
    assert stored["input.text"]["provider"] == "lmstudio"
    rows = _rows(manager)
    assert not any("route_unavailable" in row for row in rows.values())
    assert "Apple Silicon" in rows["output.image"]["recommendation_unavailable"]["reason"]


def test_force_never_clears_a_working_route_where_nothing_is_recommended(tmp_path, pin_host):
    pin_host(LINUX)
    manager = _stale_store(tmp_path, {"output.image": {"provider": "diffusers", "model": "my/sdxl"}})
    report = manager.apply_recommended_capability_defaults(force=True)
    assert _entry(report, "output.image")["action"] == "unavailable"
    assert "route_unavailable" not in _entry(report, "output.image")
    assert _stored(manager)["output.image"]["provider"] == "diffusers"


def test_the_cli_grid_and_apply_output_say_it(tmp_path, pin_host, capsys):
    from abstractcore.config import main as config_main

    pin_host(LINUX)
    manager = _stale_store(tmp_path, {"output.image": {"provider": "mlx-gen", "model": FLUX}})
    cfg = str(manager.config_file)

    assert config_main.main(["config", "--config-file", cfg, "defaults"]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("- output.image:"))
    assert "cannot run on this computer" in line and "Apple Silicon" in line

    assert config_main.main(["config", "--config-file", cfg, "apply-recommended", "--dry-run"]) == 0
    out = capsys.readouterr().out
    image = next(l for l in out.splitlines() if "output.image" in l)
    assert "yours cannot run on this computer" in image and "left as mlx-gen/" in image
    assert "--force replaces each" in out

    assert config_main.main(["config", "--config-file", cfg, "apply-recommended", "--force"]) == 0
    image = next(l for l in capsys.readouterr().out.splitlines() if "output.image" in l)
    assert f"removed mlx-gen/{FLUX}" in image
    assert "output.image" not in _stored(manager)


# ---------------------------------------------------------------------------
# The host-support matrix: architectures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "os_id", "arch", "ok"),
    [
        ("lmstudio", "linux", "x86_64", True),
        ("lmstudio", "linux", "arm64", True),
        ("lmstudio", "windows", "arm64", True),
        ("lmstudio", "linux", "armv7l", False),
        ("lmstudio", "windows", "x86", False),
        ("lmstudio", "darwin", "x86_64", False),
        ("ollama", "linux", "riscv64", False),
        ("ollama", "darwin", "x86_64", True),
        ("onnxruntime", "linux", "armv7l", False),
        ("onnxruntime", "linux", "arm64", True),
        ("onnxruntime", "darwin", "x86_64", True),
        ("onnxruntime", "freebsd", "x86_64", False),
    ],
)
def test_the_support_matrix_checks_the_architecture(engine, os_id, arch, ok):
    supported, reason = engines._support(engine, os_id, arch, None)
    assert supported is ok
    assert (reason is None) is ok


def test_lm_studio_install_plan_refuses_an_arch_it_has_no_build_for():
    plan = engines.engine_install_plan("lmstudio", "linux", "armv7l", tools={"brew": False, "winget": False})
    assert plan["available"] is False
    assert "armv7l" in plan["notes"]


def test_supertonic_follows_onnx_runtime_builds():
    assert cd.recommended_route_unavailable_reason("supertonic", dict(LINUX, arch="arm64")) is None
    reason = cd.recommended_route_unavailable_reason("supertonic", ARMV7)
    assert reason.startswith("Supertonic voice runs on ONNX Runtime") and "armv7l" in reason


def test_a_configured_supertonic_route_on_armv7_is_flagged(tmp_path, pin_host):
    pin_host(ARMV7)
    manager = _stale_store(tmp_path, {"output.voice": {"provider": "supertonic", "model": "supertonic-3"}})
    assert "ONNX Runtime" in _rows(manager)["output.voice"]["route_unavailable"]["reason"]


# ---------------------------------------------------------------------------
# A host with no text engine: one answer everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [FREEBSD, ARMV7], ids=["freebsd", "linux-armv7"])
def test_no_text_engine_gives_one_consistent_answer(host, tmp_path, monkeypatch):
    pick = mc.recommended_text_model(host, fit=False)
    # Never an Ollama fallback that cannot run either.
    assert pick["basis"] == "no_supported_engine" and pick["provider"] == "lmstudio"
    assert "LM Studio" in pick["tier"] and "Ollama" in pick["tier"]

    unavailable = cd.recommended_unavailable_routes(host)
    assert set(unavailable) == {"input.text", "input.image", "output.voice", "output.image", "output.video"}
    assert unavailable["input.image"]["reason"] == unavailable["input.text"]["reason"]
    text = unavailable["input.text"]
    assert text["provider"] == "lmstudio" and text["reason"].startswith("LM Studio has no")
    assert "cloud provider" in text["reason"], "names the way out"
    assert cd.recommended_capability_default_routes(host) == {}
    assert cd.recommended_model_downloads(host) == {}

    isolate_host(tmp_path, monkeypatch)
    rows = {r["id"]: r for r in mc.catalog(host=host)["rows"]}
    assert not any(r["starter"] for r in rows.values()), "no starter this host cannot run"
