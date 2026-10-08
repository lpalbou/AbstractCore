"""Speech input on the GPU (round 16): large-v3 everywhere, mlx-whisper on Apple silicon.

- the recommendation: Apple silicon -> mlx-whisper/large-v3 where mlx-whisper is installed,
  faster-whisper/large-v3 elsewhere (and on the light profile, which lacks mlx-whisper);
- the weights probe resolves mlx-whisper ids with AbstractVoice's own table;
- the served hint for a STORED faster-whisper route (never a migration).
"""

from __future__ import annotations

import sys
import types

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_materializer as mm
from abstractcore.config import recommendations as rec
from tests.models_engines_fakes import synthetic_host


def _mac(**installed):
    return dict(synthetic_host("metal64"), engines_installed={"mlx": True, "mlx-gen": True, **installed})


def test_apple_silicon_with_mlx_whisper_recommends_it_with_large_v3() -> None:
    routes = cd.recommended_capability_default_routes(_mac(**{"mlx-whisper": True}))
    downloads = cd.recommended_model_downloads(_mac(**{"mlx-whisper": True}))
    assert (routes["input.voice"].provider, routes["input.voice"].model) == ("mlx-whisper", "large-v3")
    assert downloads["input.voice"] == {"provider": "huggingface", "artifact": "mlx-community/whisper-large-v3-mlx"}


def test_apple_silicon_without_mlx_whisper_keeps_faster_whisper_never_an_unavailable_row() -> None:
    host = _mac(**{"mlx-whisper": False})
    routes = cd.recommended_capability_default_routes(host)
    assert (routes["input.voice"].provider, routes["input.voice"].model) == ("faster-whisper", "large-v3")
    assert "input.voice" not in cd.recommended_unavailable_routes(host)


@pytest.mark.parametrize("name", ["cuda24", "cpu16"])
def test_every_other_host_gets_faster_whisper_large_v3(name) -> None:
    route = cd.recommended_capability_default_routes(synthetic_host(name))["input.voice"]
    assert (route.provider, route.model) == ("faster-whisper", "large-v3")


def test_the_image_rows_cuda_pick_is_not_affected_by_the_installed_filter() -> None:
    # `only_if_installed` is the voice pick's opt-in; the image row's Diffusers pick keeps its rule.
    host = dict(synthetic_host("cuda24"), engines_installed={"diffusers": False})
    assert cd.RECOMMENDED_MODELS["output.image"].pick_for(host)[0].provider == "diffusers"


def test_the_fresh_seed_on_a_light_mac_writes_faster_whisper_and_on_an_apple_mac_mlx_whisper() -> None:
    light = cd.seed_recommended_capability_defaults(cd.CapabilityDefaultsConfig(), host=_mac(**{"mlx-whisper": False}))
    apple = cd.seed_recommended_capability_defaults(cd.CapabilityDefaultsConfig(), host=_mac(**{"mlx-whisper": True}))
    assert light.routes["input.voice"].provider == "faster-whisper"
    assert apple.routes["input.voice"].provider == "mlx-whisper"


def test_a_stored_route_is_never_migrated_by_the_seed_upgrade() -> None:
    config = cd.CapabilityDefaultsConfig()
    config.routes["input.voice"] = cd.CapabilityRouteDefault(provider="faster-whisper", model="large-v3")
    config.seeded = "recommended-v1"
    cd.upgrade_recommended_seed(config, host=_mac(**{"mlx-whisper": True}))
    assert config.routes["input.voice"].provider == "faster-whisper"


# --- weights probe ---------------------------------------------------------------------------


@pytest.fixture()
def mlx_table(monkeypatch):
    module = types.ModuleType("abstractvoice.adapters.stt_mlx_whisper")

    class MLXWhisperAdapter:
        MODEL_REPOS = {"large-v3": "mlx-community/whisper-large-v3-mlx", "large-v3-turbo": "mlx-community/whisper-large-v3-turbo"}
        _MODEL_ALIASES = {"large": "large-v3", "turbo": "large-v3-turbo"}

        @classmethod
        def resolve_repo(cls, model_id):
            key = cls._MODEL_ALIASES.get(str(model_id).lower(), str(model_id).lower())
            return cls.MODEL_REPOS.get(key, str(model_id))

        @classmethod
        def selectable_model_ids(cls):
            return [*cls.MODEL_REPOS, *cls._MODEL_ALIASES]

    module.MLXWhisperAdapter = MLXWhisperAdapter
    monkeypatch.setitem(sys.modules, "abstractvoice.adapters.stt_mlx_whisper", module)
    return MLXWhisperAdapter


def test_mlx_whisper_ids_resolve_with_abstractvoices_table(mlx_table) -> None:
    assert mm._hf_repo_for("mlx-whisper", "large-v3") == (
        "mlx-community/whisper-large-v3-mlx", "AbstractVoice's mlx-whisper model table"
    )
    assert mm._hf_repo_for("mlx-whisper", "turbo")[0] == "mlx-community/whisper-large-v3-turbo"
    assert mm._hf_repo_for("mlx-whisper", "someone/custom")[0] == "someone/custom"
    repo, why = mm._hf_repo_for("mlx-whisper", "tinyish")
    assert repo is None and "mlx-whisper has no model named 'tinyish'" in why


def test_the_probe_reports_mlx_whisper_weights_from_the_hugging_face_cache(mlx_table, monkeypatch, tmp_path) -> None:
    snap = tmp_path / "models--mlx-community--whisper-large-v3-mlx" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "weights.npz").write_bytes(b"x" * 16)
    (snap / "config.json").write_text("{}")
    (tmp_path / "models--mlx-community--whisper-large-v3-mlx" / "refs").mkdir()
    (tmp_path / "models--mlx-community--whisper-large-v3-mlx" / "refs" / "main").write_text("abc")
    monkeypatch.setattr(mm, "_hf_cache_dirs", lambda: [tmp_path])
    presence = mm.probe("mlx-whisper", "large-v3")
    assert presence.status == mm.PRESENCE_INSTALLED, presence
    absent = mm.probe("mlx-whisper", "large-v3-turbo")
    assert absent.status == mm.PRESENCE_ABSENT, absent


def test_the_engine_capabilities_table_probes_and_downloads_mlx_whisper() -> None:
    assert mm.supported_providers()["mlx-whisper"] == {"probe": True, "download": True, "tool": "huggingface_hub"}


# --- the served hint for a stored route -------------------------------------------------------


def test_apple_mac_with_mlx_whisper_gets_the_switch_hint_keeping_the_model(mlx_table) -> None:
    hint = rec.voice_input_hint({"provider": "faster-whisper", "model": "large-v3"}, host=_mac(**{"mlx-whisper": True}))
    assert hint["code"] == "apple_gpu_engine"
    assert hint["route"] == {"key": "input.voice", "provider": "mlx-whisper", "model": "large-v3"}
    assert hint["sentence"] == (
        "Runs on the processor: faster-whisper has no Apple GPU backend. mlx-whisper runs large-v3 on this Mac's "
        "GPU, about 15 times faster: about 1.4 s instead of about 20 s for a 17 s clip on an M5 Max."
    )
    # The aliases of the route's provider are the same engine.
    assert rec.voice_input_hint({"provider": "whisper", "model": "large"}, host=_mac(**{"mlx-whisper": True}))["route"]["model"] == "large"


def test_apple_mac_without_mlx_whisper_gets_the_setting_not_a_route() -> None:
    hint = rec.voice_input_hint({"provider": "faster-whisper", "model": "large-v3"}, host=_mac(**{"mlx-whisper": False}))
    assert hint["code"] == "apple_gpu_engine_not_installed" and hint["route"] is None
    assert "abstractcore[" in hint["sentence"]


def test_processor_only_host_on_large_v3_gets_the_turbo_sentence() -> None:
    hint = rec.voice_input_hint({"provider": "faster-whisper", "model": "large-v3"}, host=synthetic_host("cpu16"))
    assert hint == {"code": "processor_turbo_faster", "sentence": rec.FASTER_WHISPER_CPU_NOTE, "route": None}


@pytest.mark.parametrize(
    "route,host",
    [
        ({"provider": "mlx-whisper", "model": "large-v3"}, "metal64"),  # already on the GPU
        ({"provider": "faster-whisper", "model": "large-v3"}, "cuda24"),  # CUDA
        ({"provider": "faster-whisper", "model": "large-v3-turbo"}, "cpu16"),  # already the fast one
        ({"provider": "openai", "model": "whisper-1"}, "metal64"),
        ({"provider": "faster-whisper", "model": ""}, "metal64"),
    ],
)
def test_no_hint_where_there_is_nothing_to_say(route, host, mlx_table) -> None:
    h = synthetic_host(host)
    if host == "metal64":
        h = _mac(**{"mlx-whisper": True})
    assert rec.voice_input_hint(route, host=h) is None


def test_route_engines_reports_whether_mlx_whisper_is_installed(monkeypatch) -> None:
    from abstractcore.config import route_engines

    monkeypatch.setattr(route_engines, "_importable", lambda m: m == "mlx_whisper")
    assert route_engines.provider_engine_installed("mlx-whisper") is True
    assert route_engines.provider_engines_installed()["mlx-whisper"] is True
    assert route_engines.voice_engine_id("mlx_whisper", "input.voice") == "mlx-whisper"
