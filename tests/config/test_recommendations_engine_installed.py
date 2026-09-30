"""Recommendations never name an in-process engine this install does not have (core 2.20.2).

The 0.7.0 end-to-end proof on a 128 GB Mac with the LIGHT install profile: the fresh-install
seed wrote `mlx/mlx-community/Qwen3.8-Flash-Next-4bit` as the text default although the light
profile has no MLX, and every call (voice included, through the runtime) failed with "MLX
dependencies not installed".

Now every host profile carries `engines_installed` (lookups only) and the recommendation reads
it: without MLX the Apple silicon tier's model is picked on LM Studio (a server route, nothing
to install in this Python), else the nearest smaller tier's; MLX-Gen / PyTorch rows are left
unset with the honest reason. The operator's tier ruling (<24 GiB 9B, 24-<128 27B, 128+
Flash-Next) holds unchanged where MLX is installed. A synthetic host without the key is not
judged (every earlier test host keeps its result).

Also: a route write that moves a route to another provider and names no options drops the
stored options (an LM Studio text route kept the MLX tier's `speculation: native_mtp`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.config.manager import ConfigurationManager
from tests.models_engines_fakes import synthetic_host

GiB = 1024**3
LIGHT = {"mlx": False, "mlx-gen": False, "diffusers": False, "acestep": False}
APPLE = {"mlx": True, "mlx-gen": True, "diffusers": True, "acestep": True}


def _mac(gib: int, engines=None) -> dict:
    host = dict(synthetic_host("metal128"), ram_bytes=gib * GiB)
    if engines is not None:
        host["engines_installed"] = dict(engines)
    return host


@pytest.mark.parametrize(
    ("gib", "model", "row"),
    [(16, "qwen/qwen3.5-9b", "qwen3.5-9b"), (64, "qwen/qwen3.8-27b", "qwen3.8-27b"), (128, "qwen/qwen3.8-27b", "qwen3.8-27b")],
)
def test_without_mlx_the_tier_runs_on_lm_studio(gib: int, model: str, row: str) -> None:
    pick = mc.recommended_text_model(_mac(gib, LIGHT), fit=False)
    assert (pick["provider"], pick["model"], pick["catalog_id"]) == ("lmstudio", model, row)
    assert pick["artifact"] == f"{model}@q4_k_m" and pick["basis"] == "apple_silicon_engine_fallback"
    assert "cannot run in this install" in pick["tier"]
    if gib >= 128:
        assert "qwen3.8-flash-next has no LM Studio build" in pick["tier"]
    routes = cd.recommended_capability_default_routes(_mac(gib, LIGHT))
    assert (routes["input.text"].provider, routes["input.text"].model) == ("lmstudio", model)
    assert all(r.provider not in ("mlx", "mlx-gen", "diffusers", "acestep") for r in routes.values())


@pytest.mark.parametrize(("gib", "artifact"), [(16, "mlx-community/Qwen3.5-9B-MLX-4bit"),
                                               (64, "mlx-community/Qwen3.8-27B-4bit"),
                                               (128, "mlx-community/Qwen3.8-Flash-Next-4bit")])
def test_with_mlx_or_unjudged_the_operator_tiers_hold(gib: int, artifact: str) -> None:
    for host in (_mac(gib, APPLE), _mac(gib)):
        pick = mc.recommended_text_model(host, fit=False)
        assert (pick["provider"], pick["artifact"], pick["basis"]) == ("mlx", artifact, "apple_silicon_tiers")


def test_mlx_gen_rows_are_left_unset_with_the_reason() -> None:
    unavailable = cd.recommended_unavailable_routes(_mac(64, LIGHT))
    for key in ("output.image", "output.video"):
        assert "not installed in this Python environment" in unavailable[key]["reason"] or "not installed" in unavailable[key]["reason"]
    assert "output.image" not in cd.recommended_capability_default_routes(_mac(64, LIGHT))
    # With the engines, the image row is recommended again (64 GiB fits FLUX.2 klein).
    assert "output.image" in cd.recommended_capability_default_routes(_mac(64, APPLE))


def test_the_fresh_install_seed_writes_the_lm_studio_route_on_a_light_mac() -> None:
    config = cd.seed_recommended_capability_defaults(cd.CapabilityDefaultsConfig(), host=_mac(128, LIGHT))
    text = config.routes["input.text"]
    assert (text.provider, text.model) == ("lmstudio", "qwen/qwen3.8-27b")
    assert not any(r.provider in ("mlx", "mlx-gen") for r in config.routes.values())


def test_every_host_profile_carries_the_installed_engines() -> None:
    from abstractcore.config.route_engines import provider_engine_installed
    from abstractcore.utils.host_profile import _build_light_profile, host_profile

    for profile in (_build_light_profile(), host_profile(refresh=True)):
        installed = profile["engines_installed"]
        assert set(installed) == {"mlx", "mlx-gen", "diffusers", "acestep"}
        assert installed["mlx"] is provider_engine_installed("mlx")
    assert provider_engine_installed("lmstudio") is None  # a server route is never judged


def test_moving_a_route_to_another_provider_drops_the_old_engine_s_options(tmp_path: Path) -> None:
    m = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    mtp = {"speculation": {"mode": "native_mtp", "num_draft_tokens": 2, "require_acceleration": False}}
    m.set_capability_default("output.text", provider="mlx", model="mlx-community/Qwen3.8-Flash-Next-4bit", options=mtp)
    # Same provider, no options named: kept.
    m.update_capability_default("output.text", model="mlx-community/Qwen3.8-27B-4bit")
    assert m.stored_capability_default("output.text")["options"] == mtp
    # Another provider, no options named: dropped (they were MLX's).
    m.update_capability_default("output.text", provider="lmstudio", model="llama-3.2-1b-instruct")
    row = m.stored_capability_default("output.text")
    assert (row["provider"], row["model"]) == ("lmstudio", "llama-3.2-1b-instruct") and not row.get("options")
    # Options named with the change: those win.
    m.update_capability_default("output.text", provider="ollama", model="qwen3.5:9b", options={"keep_alive": "5m"})
    assert m.stored_capability_default("output.text")["options"] == {"keep_alive": "5m"}
