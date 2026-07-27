"""Vision behavior config: CONFIG WINS, env is #FALLBACK (operator dm#177).

Vision was the un-repaired voice twin (consolidated env-conflict report,
angle A finding 4): the console PUTs output.image routes into the core
server's config, which /images/generations never read — env on the host
decided, so the console could swear diffusers while OPENAI_BASE_URL presence
flipped generation to the proxy. These pins hold the ported repair:

- the configured route's provider/model/base_url override env for the
  backend-kind, per-backend model defaults, and upstream base URL;
- the route row is ONE backend identity: its model never leaks into another
  backend's lane (the M1-to-piper class, image edition);
- with NO config configured, env behavior is verbatim-unchanged (compat);
- overrides warn ONCE naming the env var, never per-request spam.
"""

from __future__ import annotations

import logging

import pytest

from abstractcore.server import vision_endpoints as ve


_ALL_ENV = [
    "ABSTRACTCORE_VISION_BACKEND",
    "ABSTRACTVISION_BACKEND",
    "ABSTRACTCORE_VISION_MODEL_ID",
    "ABSTRACTVISION_DIFFUSERS_MODEL_ID",
    "ABSTRACTVISION_MODEL_ID",
    "ABSTRACTCORE_VISION_MFLUX_MODEL",
    "ABSTRACTVISION_MFLUX_MODEL",
    "ABSTRACTCORE_VISION_UPSTREAM_MODEL_ID",
    "OPENAI_BASE_URL",
    "ABSTRACTCORE_VISION_SDCPP_MODEL",
    "ABSTRACTVISION_SDCPP_MODEL",
    "ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL",
    "ABSTRACTVISION_SDCPP_DIFFUSION_MODEL",
    "ABSTRACTCORE_VISION_SDCPP_VAE",
    "ABSTRACTVISION_SDCPP_VAE",
    # Route-option fan-out lanes (backlog 0826).
    "ABSTRACTCORE_VISION_DEVICE",
    "ABSTRACTVISION_DIFFUSERS_DEVICE",
    "ABSTRACTCORE_VISION_TORCH_DTYPE",
    "ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE",
    "ABSTRACTCORE_VISION_ALLOW_DOWNLOAD",
    "ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD",
    "ABSTRACTCORE_VISION_AUTO_RETRY_FP32",
    "ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32",
    "ABSTRACTCORE_VISION_MFLUX_BASE_MODEL",
    "ABSTRACTVISION_MFLUX_BASE_MODEL",
    "ABSTRACTCORE_VISION_MODEL_DIR",
    "ABSTRACTVISION_MODEL_DIR",
    "ABSTRACTCORE_VISION_MFLUX_ALLOW_DOWNLOAD",
    "ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD",
    "ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH",
    "ABSTRACTVISION_IMAGES_GENERATIONS_PATH",
    "ABSTRACTCORE_VISION_IMAGE_TO_VIDEO_MODE",
    "ABSTRACTVISION_IMAGE_TO_VIDEO_MODE",
]


@pytest.fixture()
def clean_env(monkeypatch):
    for name in _ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    ve._VISION_ROUTE_WARNED.clear()
    return monkeypatch


def _patch_image_route(monkeypatch, route: dict) -> None:
    """Make get_capability_default("output","image") return the given route.

    Task rows read through this double return the SAME route (it ignores
    `task`) — fine for broad-route tests; task-precedence tests use
    _patch_routes below.
    """

    class _Mgr:
        def get_capability_default(self, kind, modality=None, task=None):
            if kind == "output" and modality == "image":
                return dict(route)
            return {"source": "not_configured"}

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: _Mgr())


def _patch_routes(monkeypatch, routes: dict) -> None:
    """Route table keyed by (modality, task) — task=None is the broad row."""

    class _Mgr:
        def get_capability_default(self, kind, modality=None, task=None):
            if kind != "output":
                return {"source": "not_configured"}
            row = routes.get((modality, task))
            return dict(row) if row is not None else {"source": "not_configured"}

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: _Mgr())


def test_config_backend_wins_over_env_backend(clean_env, monkeypatch, caplog):
    # The incident shape: env exports one backend, config names another.
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "openai")
    _patch_image_route(monkeypatch, {"provider": "diffusers", "model": "stabilityai/sdxl-turbo"})
    with caplog.at_level(logging.WARNING):
        kind = ve._vision_backend_kind()
    assert kind == "diffusers", "configured backend must win over the exported env var"
    assert any(
        "ABSTRACTCORE_VISION_BACKEND" in r.getMessage() and "IGNORED" in r.getMessage() for r in caplog.records
    ), "override warning must name the env var to unset"


def test_configured_model_never_leaks_across_backends(clean_env, monkeypatch):
    # Route row is ONE backend identity (the M1-to-piper class): an mflux
    # route's model must not become the diffusers/upstream/sdcpp default.
    _patch_image_route(monkeypatch, {"provider": "mflux", "model": "flux2-klein-9b"})
    assert ve._mflux_model_default() == "flux2-klein-9b"
    assert ve._diffusers_model_default() is None
    assert ve._upstream_model_default() is None
    assert ve._route_model_for("sdcpp") is None


def test_config_model_wins_over_env_model_same_backend(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_MODEL", "old-env-model")
    _patch_image_route(monkeypatch, {"provider": "mflux", "model": "flux2-klein-9b"})
    with caplog.at_level(logging.WARNING):
        assert ve._mflux_model_default() == "flux2-klein-9b"
    assert any("ABSTRACTCORE_VISION_MFLUX_MODEL" in r.getMessage() for r in caplog.records)


def test_env_only_behavior_unchanged_without_config(clean_env, monkeypatch):
    # Compat: nothing configured -> env chain verbatim.
    _patch_image_route(monkeypatch, {"source": "not_configured"})
    monkeypatch.setenv("ABSTRACTVISION_BACKEND", "sdcpp")
    monkeypatch.setenv("ABSTRACTCORE_VISION_SDCPP_MODEL", "/models/sd.gguf")
    assert ve._vision_backend_kind() == "sdcpp"
    assert ve._sdcpp_setting("MODEL") == "/models/sd.gguf"


def test_openai_base_url_presence_no_longer_flips_configured_local(clean_env, monkeypatch):
    # Report finding 1: mere PRESENCE of OPENAI_BASE_URL flipped image
    # backend selection to the proxy. A configured local route must win.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    _patch_image_route(monkeypatch, {"provider": "diffusers", "model": "stabilityai/sdxl-turbo"})
    assert ve._effective_backend_kind(None) == "diffusers"
    # And with NO config, presence still selects the proxy (env compat).
    _patch_image_route(monkeypatch, {"source": "not_configured"})
    assert ve._effective_backend_kind(None) == "openai_compatible_proxy"


def test_providerless_configured_model_attributed_by_shape(clean_env, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    # HF repo id -> diffusers even though the proxy env is exported.
    _patch_image_route(monkeypatch, {"model": "stabilityai/sdxl-turbo"})
    assert ve._effective_backend_kind(None) == "diffusers"
    # Local path -> sdcpp.
    _patch_image_route(monkeypatch, {"model": "/models/sd.gguf"})
    assert ve._effective_backend_kind(None) == "sdcpp"
    # Unattributable bare name: withheld from per-backend defaults (never guessed).
    _patch_image_route(monkeypatch, {"model": "flux2-klein-9b"})
    assert ve._diffusers_model_default() is None
    assert ve._route_model_for("mlx-gen") is None


def test_route_base_url_wins_for_proxy_lane(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    _patch_image_route(
        monkeypatch,
        {"provider": "openai-compatible", "model": "gpt-image-1", "base_url": "http://localhost:8033/v1"},
    )
    with caplog.at_level(logging.WARNING):
        assert ve._upstream_base_url_default() == "http://localhost:8033/v1"
    assert any("OPENAI_BASE_URL" in r.getMessage() for r in caplog.records)
    assert ve._upstream_model_default() == "gpt-image-1"


def test_route_base_url_ignored_for_non_proxy_backend(clean_env, monkeypatch):
    # base_url next to a local backend identity means nothing to the proxy lane.
    _patch_image_route(monkeypatch, {"provider": "diffusers", "model": "x/y", "base_url": "http://h:1/v1"})
    assert ve._upstream_base_url_default() is None


def test_sdcpp_route_options_supply_components_env_fallback_below(clean_env, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_VISION_SDCPP_VAE", "/env/vae.safetensors")
    _patch_image_route(
        monkeypatch,
        {
            "provider": "sdcpp",
            "model": "/models/sd.gguf",
            "options": {"vae": "/cfg/vae.safetensors", "diffusion_model": "/cfg/dm.gguf"},
        },
    )
    assert ve._sdcpp_setting("VAE") == "/cfg/vae.safetensors", "route option must win over env"
    assert ve._sdcpp_setting("DIFFUSION_MODEL") == "/cfg/dm.gguf"
    # Env still answers for components config does not carry.
    assert ve._sdcpp_setting("LLM") is None
    monkeypatch.setenv("ABSTRACTVISION_SDCPP_LLM", "/env/llm.gguf")
    assert ve._sdcpp_setting("LLM") == "/env/llm.gguf"
    # Options never serve another backend's lane: behind a diffusers identity
    # the sdcpp lane sees only env, and nothing once env is unset.
    _patch_image_route(monkeypatch, {"provider": "diffusers", "options": {"vae": "/cfg/vae.safetensors"}})
    assert ve._sdcpp_setting("VAE") == "/env/vae.safetensors"
    monkeypatch.delenv("ABSTRACTCORE_VISION_SDCPP_VAE", raising=False)
    assert ve._sdcpp_setting("VAE") is None, "options behind a non-sdcpp identity must not leak"


def test_catalog_builder_seeds_from_config_env_fills_holes(clean_env, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "openai")
    monkeypatch.setenv("ABSTRACTCORE_VISION_TIMEOUT_S", "120")
    _patch_image_route(monkeypatch, {"provider": "mflux", "model": "flux2-klein-9b"})
    config = ve._vision_catalog_config_from_env()
    assert config["vision_backend"] == "mlx-gen", "advertising must match config-first execution"
    assert config["vision_mflux_model"] == "flux2-klein-9b"
    assert config["vision_timeout_s"] == "120", "env still fills keys config leaves empty"


def test_override_warning_fires_once_not_per_call(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "openai")
    _patch_image_route(monkeypatch, {"provider": "diffusers", "model": "x/y"})
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            ve._vision_backend_kind()
    hits = [r for r in caplog.records if "ABSTRACTCORE_VISION_BACKEND" in r.getMessage()]
    assert len(hits) == 1, "per-request handlers must not spam the override warning"


def test_unreadable_config_degrades_to_env_with_label(clean_env, monkeypatch, caplog):
    def _raise():
        raise RuntimeError("config store unavailable")

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", _raise)
    monkeypatch.setenv("ABSTRACTVISION_BACKEND", "diffusers")
    with caplog.at_level(logging.WARNING):
        assert ve._vision_backend_kind() == "diffusers"
    assert any("#FALLBACK" in r.getMessage() for r in caplog.records)


def test_video_lane_never_inherits_image_route(clean_env, monkeypatch):
    # Adversary P1-1: an Image Output route must not steer /v1/videos/* — a
    # video request with no model must stay honestly unconfigured, not
    # attempt video generation on the configured image model.
    def _image_only(modality="image", task=None):
        if modality == "image":
            return {"backend": "mlx-gen", "model": "flux2-klein-9b"}
        return {}

    monkeypatch.setattr(ve, "_vision_route_defaults", _image_only)
    assert ve._effective_backend_kind(None, "image") == "mlx-gen"
    assert ve._effective_backend_kind(None, "video") == "auto_unconfigured"
    assert ve._route_model_for("mlx-gen", "video") is None


def test_video_lane_reads_output_video_route(clean_env, monkeypatch):
    # The console's Video Output route is the video lanes' config home.
    def _routes(modality="image", task=None):
        if modality == "video":
            return {"backend": "openai_compatible_proxy", "model": "video-gen-1", "base_url": "http://vid:9000/v1"}
        return {}

    monkeypatch.setattr(ve, "_vision_route_defaults", _routes)
    assert ve._effective_backend_kind(None, "video") == "openai_compatible_proxy"
    assert ve._upstream_base_url_default("video") == "http://vid:9000/v1"
    assert ve._upstream_model_default("video") == "video-gen-1"
    # And the image lane is untouched by the video route.
    assert ve._effective_backend_kind(None, "image") == "auto_unconfigured"


def test_advertising_seeds_from_configured_proxy_route(clean_env, monkeypatch):
    # Adversary P1-2: a config-only proxy route executed but advertised [].
    _patch_image_route(
        monkeypatch,
        {"provider": "openai-compatible", "model": "gpt-image-1", "base_url": "http://localhost:8033/v1"},
    )
    monkeypatch.setattr(ve, "_vision_provider_model_capabilities_for_task", lambda task: {"text_to_image": True})
    entries = ve._configured_vision_provider_model_entries("text_to_image")
    routed = [e.get("model") for e in entries]
    assert any("gpt-image-1" in str(m) for m in routed), f"configured proxy route must be advertised, got {routed}"


def test_env_backend_alias_abstractvision_stays_verbatim(clean_env, monkeypatch):
    # P2-6: the package-hint alias is CONFIG-only; env keeps historical
    # behavior (unknown spelling passes through and fails loudly downstream).
    _patch_image_route(monkeypatch, {"source": "not_configured"})
    monkeypatch.setenv("ABSTRACTCORE_VISION_BACKEND", "abstractvision")
    assert ve._vision_backend_kind() == "abstractvision"
    _patch_image_route(monkeypatch, {"provider": "abstractvision", "model": "stabilityai/sdxl-turbo"})
    monkeypatch.delenv("ABSTRACTCORE_VISION_BACKEND", raising=False)
    # Config-sourced hint means auto: shape attribution picks diffusers.
    assert ve._effective_backend_kind(None, "image") == "diffusers"


def test_clone_engine_call_site_uses_capability_config():
    # Structural pin: the clone handler must not read the engine from env
    # directly (that bypassed the dm#177 repair) — it goes through
    # _capability_config(), whose precedence the audio suite pins.
    import inspect

    from abstractcore.server import audio_endpoints as ae

    src = inspect.getsource(ae)
    assert 'os.getenv("ABSTRACTVOICE_CLONING_ENGINE")' not in src.replace(
        '"ABSTRACTVOICE_CLONING_ENGINE": "voice_cloning_engine"', ""
    ), "clone engine must resolve through _capability_config, not a direct env read"


def test_image_request_parts_shape_by_image_route_not_video(clean_env, monkeypatch):
    # Regression (security adversary, 2026-07-23): _image_generation_request_parts
    # passed modality="video" to _effective_backend_kind, so image size-vs-
    # width/height shaping was decided by the VIDEO route — the cross-modality
    # bleed the config-wins work set out to kill, mirrored. With image=local
    # diffusers + video=proxy, an image request MUST keep width/height (the
    # local backend needs them; only a proxy image route size-shapes).
    def _routes(modality="image", task=None):
        if modality == "image":
            return {"backend": "diffusers", "model": "stabilityai/sdxl-turbo"}
        if modality == "video":
            return {"backend": "openai_compatible_proxy", "model": "sora", "base_url": "http://vid:9000/v1"}
        return {}

    monkeypatch.setattr(ve, "_vision_route_defaults", _routes)
    width, height, extra = ve._image_generation_request_parts({"width": 512, "height": 512})
    assert width == 512 and height == 512, "local image route must keep width/height"
    assert "size" not in extra


def test_image_request_parts_size_shape_when_image_route_is_proxy(clean_env, monkeypatch):
    # The mirror: image=proxy + video=local. The image request MUST size-shape
    # (drop width/height, set size) driven by the IMAGE route — never held
    # local by the video route.
    def _routes(modality="image", task=None):
        if modality == "image":
            return {"backend": "openai_compatible_proxy", "model": "gpt-image-1", "base_url": "http://img:8000/v1"}
        if modality == "video":
            return {"backend": "diffusers", "model": "org/local-vid"}
        return {}

    monkeypatch.setattr(ve, "_vision_route_defaults", _routes)
    width, height, extra = ve._image_generation_request_parts({"width": 512, "height": 512})
    assert width is None and height is None, "proxy image route must drop width/height"
    assert extra.get("size") == "512x512"


# ---------------------------------------------------------------------------
# Task-specific routes (backlog 0826): output.image.text_to_image etc. win
# over the broad output.image row for THEIR task; the broad row serves when
# the task row is absent. A configured task row wins WHOLESALE (one backend
# identity — never field-merged with the broad row).
# ---------------------------------------------------------------------------


def test_task_route_overrides_broad_route_for_its_task(clean_env, monkeypatch):
    _patch_routes(
        monkeypatch,
        {
            ("image", None): {"provider": "diffusers", "model": "stabilityai/sdxl-turbo"},
            ("image", "text_to_image"): {"provider": "mflux", "model": "flux2-klein-9b"},
        },
    )
    # The task the row names follows the task row...
    assert ve._effective_backend_kind(None, "image", "text_to_image") == "mlx-gen"
    assert ve._mflux_model_default("image", "text_to_image") == "flux2-klein-9b"
    # ...other tasks (and task-less reads) keep following the broad row.
    assert ve._effective_backend_kind(None, "image", "image_to_image") == "diffusers"
    assert ve._effective_backend_kind(None, "image") == "diffusers"
    assert ve._diffusers_model_default("image", "image_to_image") == "stabilityai/sdxl-turbo"


def test_task_route_falls_back_to_broad_when_absent(clean_env, monkeypatch):
    _patch_routes(monkeypatch, {("image", None): {"provider": "diffusers", "model": "stabilityai/sdxl-turbo"}})
    assert ve._vision_route_defaults("image", "text_to_image") == {
        "backend": "diffusers",
        "model": "stabilityai/sdxl-turbo",
    }
    assert ve._diffusers_model_default("image", "text_to_image") == "stabilityai/sdxl-turbo"


def test_task_route_wins_wholesale_not_field_merged(clean_env, monkeypatch):
    # The task row is ONE backend identity: a provider-only task row must not
    # inherit the broad row's model (that would mint a pairing the operator
    # never configured — the M1-to-piper class, task edition).
    _patch_routes(
        monkeypatch,
        {
            ("image", None): {"provider": "diffusers", "model": "stabilityai/sdxl-turbo"},
            ("image", "text_to_image"): {"provider": "mflux"},
        },
    )
    row = ve._vision_route_defaults("image", "text_to_image")
    assert row.get("backend") == "mlx-gen"
    assert "model" not in row, "task row must win wholesale, never field-merge the broad model in"
    assert ve._mflux_model_default("image", "text_to_image") is None
    assert ve._diffusers_model_default("image", "text_to_image") is None


def test_video_task_route_scoped_to_video_modality(clean_env, monkeypatch):
    _patch_routes(
        monkeypatch,
        {
            ("video", "image_to_video"): {"provider": "mflux", "model": "AbstractFramework/wan2.2-i2v"},
        },
    )
    assert ve._effective_backend_kind(None, "video", "image_to_video") == "mlx-gen"
    # The image lane and the OTHER video task stay unconfigured.
    assert ve._effective_backend_kind(None, "image", "text_to_image") == "auto_unconfigured"
    assert ve._effective_backend_kind(None, "video", "text_to_video") == "auto_unconfigured"


def test_upscale_defaults_read_task_route_only(clean_env, monkeypatch):
    # /v1/images/upscale seeds from output.image.image_upscale when nothing
    # explicit is passed...
    _patch_routes(
        monkeypatch,
        {("image", "image_upscale"): {"provider": "mflux", "model": "AbstractFramework/seedvr2-7b"}},
    )
    assert ve._image_upscale_route_defaults(provider=None, model=None, base_url=None) == (
        "mlx-gen",
        "AbstractFramework/seedvr2-7b",
    )
    # ...but a broad GENERATION route must NOT become the upscale default
    # (a text-to-image model is not an upscaler): built-in default preserved.
    _patch_routes(monkeypatch, {("image", None): {"provider": "diffusers", "model": "stabilityai/sdxl-turbo"}})
    assert ve._image_upscale_route_defaults(provider=None, model=None, base_url=None) == (
        ve._DEFAULT_IMAGE_UPSCALE_PROVIDER,
        ve._DEFAULT_IMAGE_UPSCALE_MODEL,
    )
    # Explicit request identity always wins over the task row.
    _patch_routes(
        monkeypatch,
        {("image", "image_upscale"): {"provider": "mflux", "model": "AbstractFramework/seedvr2-7b"}},
    )
    assert ve._image_upscale_route_defaults(provider="diffusers", model="org/x", base_url=None) == (
        "diffusers",
        "org/x",
    )


# ---------------------------------------------------------------------------
# Route-option fan-out (backlog 0826): options used to reach only the sdcpp
# lane — diffusers/mflux/proxy options were dropped SILENTLY. Config wins,
# env is labeled #FALLBACK, unknown keys warn (never silently dropped).
# ---------------------------------------------------------------------------


def test_diffusers_route_options_win_over_env(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ABSTRACTCORE_VISION_DEVICE", "cpu")
    _patch_image_route(
        monkeypatch,
        {
            "provider": "diffusers",
            "model": "stabilityai/sdxl-turbo",
            "options": {"device": "mps", "torch_dtype": "float16", "allow_download": True},
        },
    )
    with caplog.at_level(logging.WARNING):
        settings = ve._diffusers_backend_settings()
    assert settings["device"] == "mps", "configured route option must win over the exported env var"
    assert settings["torch_dtype"] == "float16"
    assert settings["allow_download"] is True
    assert settings["auto_retry_fp32"] is True, "unset option keeps the env/default value"
    assert any(
        "ABSTRACTCORE_VISION_DEVICE" in r.getMessage() and "IGNORED" in r.getMessage() for r in caplog.records
    ), "override warning must name the env var that is actually set"


def test_diffusers_env_only_settings_byte_parity(clean_env, monkeypatch):
    _patch_image_route(monkeypatch, {"source": "not_configured"})
    monkeypatch.setenv("ABSTRACTCORE_VISION_DEVICE", "cpu")
    monkeypatch.setenv("ABSTRACTCORE_VISION_TORCH_DTYPE", "bfloat16")
    monkeypatch.setenv("ABSTRACTCORE_VISION_ALLOW_DOWNLOAD", "1")
    monkeypatch.setenv("ABSTRACTCORE_VISION_AUTO_RETRY_FP32", "0")
    settings = ve._diffusers_backend_settings()
    assert settings == {
        "device": "cpu",
        "torch_dtype": "bfloat16",
        "allow_download": True,
        "auto_retry_fp32": False,
    }


def test_mflux_base_model_config_first_env_fallback(clean_env, monkeypatch, caplog):
    # base_model is a model CHOICE (behavior): the mflux route's options are
    # its config home; the env var survives as labeled #FALLBACK below it.
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_BASE_MODEL", "env-base-model")
    _patch_image_route(
        monkeypatch,
        {
            "provider": "mflux",
            "model": "flux2-klein-9b",
            "options": {"base_model": "cfg-base-model", "model_dir": "/cfg/models"},
        },
    )
    with caplog.at_level(logging.WARNING):
        settings = ve._mflux_backend_settings()
    assert settings["base_model"] == "cfg-base-model"
    assert settings["model_dir"] == "/cfg/models"
    assert settings["allow_download"] is False
    assert any("ABSTRACTCORE_VISION_MFLUX_BASE_MODEL" in r.getMessage() for r in caplog.records)
    # And the catalog/advertising builder seeds the SAME configured values.
    config = ve._vision_catalog_config_from_env()
    assert config["vision_mflux_base_model"] == "cfg-base-model"
    assert config["vision_model_dir"] == "/cfg/models"


def test_mflux_env_only_settings_byte_parity(clean_env, monkeypatch):
    _patch_image_route(monkeypatch, {"source": "not_configured"})
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_BASE_MODEL", "env-base-model")
    monkeypatch.setenv("ABSTRACTCORE_VISION_MODEL_DIR", "/env/models")
    monkeypatch.setenv("ABSTRACTCORE_VISION_MFLUX_ALLOW_DOWNLOAD", "true")
    settings = ve._mflux_backend_settings()
    assert settings == {"base_model": "env-base-model", "model_dir": "/env/models", "allow_download": True}
    config = ve._vision_catalog_config_from_env()
    assert config["vision_mflux_base_model"] == "env-base-model", "env-only catalog seeding unchanged"


def test_proxy_route_options_override_upstream_paths(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH", "/env/gen")
    _patch_image_route(
        monkeypatch,
        {
            "provider": "openai-compatible",
            "model": "gpt-image-1",
            "base_url": "http://localhost:8033/v1",
            "options": {"image_generations_path": "/cfg/gen", "image_to_video_mode": "json"},
        },
    )
    with caplog.at_level(logging.WARNING):
        settings = ve._proxy_upstream_settings()
    assert settings["image_generations_path"] == "/cfg/gen"
    assert settings["image_to_video_mode"] == "json"
    assert settings["image_edits_path"] == "/images/edits", "unset options keep the defaults"
    assert any("ABSTRACTCORE_VISION_UPSTREAM_IMAGES_GENERATIONS_PATH" in r.getMessage() for r in caplog.records)
    # Env-only parity.
    _patch_image_route(monkeypatch, {"source": "not_configured"})
    settings = ve._proxy_upstream_settings()
    assert settings["image_generations_path"] == "/env/gen"
    assert settings["image_to_video_mode"] == "multipart"


def test_unknown_route_option_warns_once_never_silent(clean_env, monkeypatch, caplog):
    _patch_image_route(
        monkeypatch,
        {
            "provider": "diffusers",
            "model": "stabilityai/sdxl-turbo",
            "options": {"device": "mps", "funky_knob": 1},
        },
    )
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            settings = ve._diffusers_backend_settings()
    assert settings["device"] == "mps", "known keys still apply next to an unknown one"
    hits = [r for r in caplog.records if "funky_knob" in r.getMessage()]
    assert len(hits) == 1, "unknown option keys must warn exactly once, never silently drop"
    assert "diffusers" in hits[0].getMessage(), "the warning must name the lane that refused the key"


def test_request_level_options_warn_as_request_layer_not_dropped(clean_env, monkeypatch, caplog):
    # The SHIPPED task-route example: an mflux image_upscale row carrying
    # resolution/softness. Those are REQUEST-level parameters the
    # generate-contract lane folds into output specs — the backend-settings
    # warning must say "left to the request layer", never falsely claim they
    # were dropped.
    _patch_routes(
        monkeypatch,
        {
            ("image", "image_upscale"): {
                "provider": "mflux",
                "model": "AbstractFramework/seedvr2-3b-8bit",
                "options": {"resolution": "2x", "softness": 0.25},
            }
        },
    )
    with caplog.at_level(logging.WARNING):
        settings = ve._mflux_backend_settings("image", "image_upscale")
    assert settings["base_model"] is None
    hits = [r for r in caplog.records if "resolution" in r.getMessage()]
    assert len(hits) == 1
    assert "request layer" in hits[0].getMessage()
    assert "NOT applied" not in hits[0].getMessage()


def test_sdcpp_unknown_route_option_warns(clean_env, monkeypatch, caplog):
    _patch_image_route(
        monkeypatch,
        {
            "provider": "sdcpp",
            "model": "/models/sd.gguf",
            "options": {"vae": "/cfg/vae.safetensors", "mystery": "x"},
        },
    )
    with caplog.at_level(logging.WARNING):
        assert ve._sdcpp_setting("VAE") == "/cfg/vae.safetensors"
    assert any("mystery" in r.getMessage() for r in caplog.records)


def test_route_options_never_leak_across_backends(clean_env, monkeypatch):
    # Options behind a diffusers identity mean nothing to the mflux/proxy
    # lanes (the sdcpp rule, generalized).
    _patch_image_route(
        monkeypatch,
        {"provider": "diffusers", "model": "stabilityai/sdxl-turbo", "options": {"device": "mps"}},
    )
    assert ve._route_options_for("mlx-gen") == {}
    assert ve._route_options_for("openai_compatible_proxy") == {}
    assert ve._mflux_backend_settings()["base_model"] is None
    assert ve._proxy_upstream_settings()["image_generations_path"] == "/images/generations"


def test_providerless_route_options_warned_not_applied(clean_env, monkeypatch, caplog):
    # Options are backend-specific facts: on a provider-less row no lane can
    # claim them — they must not apply, and must not vanish silently either.
    _patch_image_route(
        monkeypatch,
        {"model": "stabilityai/sdxl-turbo", "options": {"device": "mps"}},
    )
    with caplog.at_level(logging.WARNING):
        settings = ve._diffusers_backend_settings()
    assert settings["device"] == "auto", "provider-less options must not steer any lane"
    assert any(
        "without a route provider" in r.getMessage() and "no backend lane" in r.getMessage() for r in caplog.records
    )
