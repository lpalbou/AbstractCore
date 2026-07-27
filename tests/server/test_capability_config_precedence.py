"""Voice/music behavior config: CONFIG WINS, env is #FALLBACK (operator dm#177).

An exported ABSTRACTVOICE_*/ABSTRACTMUSIC_* var must never silently flip a
value configured via `abstractcore --config` (the ABSTRACTVOICE_TTS_ENGINE
incident). These pins hold the precedence + the no-silent-flip guarantee:

- centralized config route defaults override env for overlapping keys;
- env-only keys survive (nothing breaks mid-migration) but are flagged;
- with NO config configured, env still works verbatim (compat).
"""

from __future__ import annotations

import logging

import pytest

from abstractcore.server import audio_endpoints as ae


@pytest.fixture()
def clean_env(monkeypatch):
    for name in list(ae._CAPABILITY_ENV_MAP) + [
        "ABSTRACTVOICE_ALLOW_DOWNLOADS",
        "ABSTRACTVOICE_CLONED_TTS_STREAMING",
        "ABSTRACTVOICE_DEBUG",
        "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(name, raising=False)
    # Reset the warn-once memo so each test observes its own warnings.
    ae._CAPABILITY_WARNED.clear()
    return monkeypatch


def _patch_config_routes(monkeypatch, routes: dict) -> None:
    """Make get_capability_default(kind, modality) return the given routes."""

    class _Mgr:
        def get_capability_default(self, kind, modality=None, task=None):
            return dict(routes.get(f"{kind}.{modality}", {}))

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: _Mgr())


def test_config_wins_over_env_for_tts_engine(clean_env, monkeypatch, caplog):
    # The exact incident: env exports one engine, config names another.
    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "piper")
    _patch_config_routes(monkeypatch, {"output.voice": {"provider": "supertonic", "model": "M2"}})
    with caplog.at_level(logging.WARNING):
        cfg = ae._capability_config()
    assert cfg["voice_tts_engine"] == "supertonic", "config must win over the exported env var"
    assert cfg["voice_tts_model"] == "M2"
    # The warning names the ENV VAR to unset, not the plugin kwarg (F10).
    assert any("ABSTRACTVOICE_TTS_ENGINE" in r.getMessage() and "IGNORED" in r.getMessage() for r in caplog.records)


def test_precedence_warning_fires_once_not_per_call(clean_env, monkeypatch, caplog):
    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "piper")
    _patch_config_routes(monkeypatch, {"output.voice": {"provider": "supertonic", "model": "M2"}})
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            ae._capability_config()
    overrides = [r for r in caplog.records if "IGNORED" in r.getMessage()]
    assert len(overrides) == 1, "per-request builders must not spam the override warning (F9)"


def test_input_voice_model_fans_out_to_whisper(clean_env, monkeypatch):
    _patch_config_routes(monkeypatch, {"input.voice": {"provider": "whisper", "model": "small"}})
    cfg = ae._capability_config()
    assert cfg["voice_stt_model"] == "small"
    assert cfg["voice_whisper_model"] == "small", "local whisper reads voice_whisper_model (F4)"


def test_known_route_option_is_translated(clean_env, monkeypatch):
    _patch_config_routes(monkeypatch, {"output.voice": {"provider": "supertonic", "options": {"language": "fr"}}})
    cfg = ae._capability_config()
    assert cfg["voice_language"] == "fr", "the 'language' route option must map to voice_language (F3)"
    assert "language" not in cfg, "the raw option name must not leak as an inert key"


def test_option_never_clobbers_route_identity(clean_env, monkeypatch):
    _patch_config_routes(
        monkeypatch,
        {"output.voice": {"provider": "supertonic", "options": {"voice_tts_engine": "piper"}}},
    )
    cfg = ae._capability_config()
    assert cfg["voice_tts_engine"] == "supertonic", "a route option must not override the route's own provider (F5)"


def test_call_sites_use_config_wins_not_raw_env(clean_env):
    """F11: pin the CALL SITES, not just _capability_config(). Re-pointing any
    builder back to _capability_config_from_env() (env-wins) must fail here."""
    import inspect

    for fn in (ae._get_capability_core, ae._music_capability_core_for_request, ae._audio_catalog_core):
        src = inspect.getsource(fn)
        assert "_capability_config()" in src, f"{fn.__name__} must build from _capability_config() (config-wins)"
        assert "_capability_config_from_env()" not in src, (
            f"{fn.__name__} must NOT call the env-only source directly — that reopens the incident"
        )


def test_env_only_survives_when_no_config(clean_env, monkeypatch):
    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "piper")
    _patch_config_routes(monkeypatch, {})  # nothing configured
    cfg = ae._capability_config()
    assert cfg["voice_tts_engine"] == "piper", "env-only compat must hold when config is empty"


def test_env_only_key_flagged_when_config_present(clean_env, monkeypatch, caplog):
    # config sets the STT route; env sets a music tuning key with no config home.
    monkeypatch.setenv("ABSTRACTMUSIC_GUIDANCE_SCALE", "7.5")
    _patch_config_routes(monkeypatch, {"input.voice": {"provider": "whisper", "model": "base"}})
    with caplog.at_level(logging.WARNING):
        cfg = ae._capability_config()
    assert cfg["voice_stt_engine"] == "whisper"
    assert cfg["music_guidance_scale"] == "7.5", "env-only key survives (no silent break)"
    assert any("still sourced from env" in r.getMessage() for r in caplog.records)


def test_route_options_pass_through_as_plugin_kwargs(clean_env, monkeypatch):
    _patch_config_routes(
        monkeypatch,
        {"output.voice": {"provider": "supertonic", "model": "M1", "options": {"voice_tts_delivery_mode": "stream"}}},
    )
    cfg = ae._capability_config()
    assert cfg["voice_tts_delivery_mode"] == "stream", "route options must reach the plugin kwargs"


def test_music_route_maps_backend_and_model(clean_env, monkeypatch):
    _patch_config_routes(monkeypatch, {"output.music": {"provider": "acemusic", "model": "ace-v15"}})
    cfg = ae._capability_config()
    assert cfg["music_backend"] == "acemusic"
    assert cfg["music_model_id"] == "ace-v15"


def test_config_read_failure_degrades_to_env(clean_env, monkeypatch):
    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "piper")

    def _boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", _boom)
    cfg = ae._capability_config()
    assert cfg["voice_tts_engine"] == "piper", "a config read failure must not lose the env compat path"
