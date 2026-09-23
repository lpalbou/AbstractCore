"""MTP policy is a Core setting, never an app guess or an implicit download."""
from types import SimpleNamespace
from unittest.mock import Mock

import importlib.util

import pytest

from abstractcore.config.manager import ConfigurationManager
from abstractcore.config.capability_defaults import capability_default_speculation
from abstractcore.providers import speculation as spec
from abstractcore.providers.mlx_provider import MLXProvider


_requires_mlx_stack = pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("mlx", "mlx_lm", "mlx_vlm")),
    reason="requires the optional MLX stack (pip install \"abstractcore[mlx]\")",
)


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch, tmp_path):
    import abstractcore.config.manager as config_module
    monkeypatch.setattr(config_module, "_config_manager", None)
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_FILE", str(tmp_path / "core.json"))
    monkeypatch.setenv("ABSTRACTCORE_DATA_REGISTRY_PATH", str(tmp_path / "data.json"))
    spec._read_configured_speculation.cache_clear()


def test_fresh_seed_is_depth_two_and_existing_empty_store_is_not_reseeded(tmp_path):
    fresh = ConfigurationManager(apply_env=False)
    assert spec.configured_speculation_default() == {
        "mode": "native_mtp", "num_draft_tokens": 2, "require_acceleration": False,
    }
    fresh.config_file.write_text("{}")
    existing = ConfigurationManager(apply_env=False)
    assert capability_default_speculation({row["key"]: row for row in existing.list_capability_defaults()}) is None
    assert spec.configured_speculation_default() is None


def test_config_off_depth_clear_and_external_changes_preserve_other_fields():
    manager = ConfigurationManager(apply_env=False)
    manager.update_capability_default("output.text", options={"other": 4, "speculation": False})
    assert spec.configured_speculation_default() is False
    manager.update_capability_default("output.text", provider="mlx", model="model")
    assert spec.configured_speculation_default() is False
    assert manager.get_capability_default("output.text")["options"]["other"] == 4
    second = ConfigurationManager(apply_env=False)
    second.update_capability_default("output.text", options={"other": 4, "speculation": {"num_draft_tokens": 5}})
    assert spec.configured_speculation_default() == {"num_draft_tokens": 5}
    second.update_capability_default("output.text", options={"other": 4, "speculation": None})
    assert spec.configured_speculation_default() is None
    assert second.get_capability_default("output.text")["options"] == {"other": 4}


@pytest.mark.parametrize("value", [None, False, True, {}, {"num_draft_tokens": 2}, {"require_acceleration": False}])
def test_json_normalizer_preserves_partial_controls(value):
    assert spec.normalize_speculation_value(value) == value


@pytest.mark.parametrize("value", [{"num_draft_tokens": True}, {"num_draft_tokens": 0}, {"depth": 3}, {"require_acceleration": "false"}])
def test_invalid_config_fails_without_writing(value):
    manager = ConfigurationManager(apply_env=False)
    manager._save_config()
    before = manager.config_file.read_bytes()
    with pytest.raises(ValueError):
        manager.update_capability_default("output.text", options={"speculation": value})
    assert manager.config_file.read_bytes() == before


@_requires_mlx_stack
def test_discovery_is_provider_specific_and_does_not_create_config_or_load(monkeypatch, tmp_path):
    monkeypatch.setattr(spec, "_local_model_directory", lambda model: None)
    hf = spec.get_execution_capabilities("mlx-community/Qwen3.8-27B-4bit", provider="huggingface")
    mlx = spec.get_execution_capabilities("mlx-community/Qwen3.8-27B-4bit", provider="mlx")
    assert hf["speculation"]["supported"] is False
    assert hf["speculation"]["supported_depths"] == []
    assert mlx["speculation"]["supported"] is True
    assert mlx["speculation"]["ready"] is None
    assert mlx["concurrency"]["supported"] is None
    assert not (tmp_path / "core.json").exists()


@_requires_mlx_stack
def test_loaded_instance_facts_do_not_use_last_call_status(monkeypatch):
    monkeypatch.setattr(spec, "_local_model_directory", lambda model: None)
    loaded = SimpleNamespace(_mtp_active=True, model_capabilities={}, supports_concurrent_generation=lambda: True)
    caps = spec.get_execution_capabilities("exact-loaded-model", provider="mlx", instance=loaded)
    assert caps["speculation"]["ready"] is True
    assert caps["speculation"]["requires_reload"] is False
    assert caps["concurrency"]["supported"] is True
    loaded._mtp_active = False
    assert spec.get_execution_capabilities("exact-loaded-model", provider="mlx", instance=loaded)["speculation"]["ready"] is False


@pytest.mark.parametrize("value", [False, {"num_draft_tokens": 5}])
def test_instance_constructor_default_is_distinct_from_configured_policy(monkeypatch, value):
    monkeypatch.setattr(spec, "_local_model_directory", lambda model: None)
    loaded = SimpleNamespace(_mtp_active=True, _speculation_inherits_config=False,
                             _speculation_request=spec.normalize_speculation_request(value), model_capabilities={})
    result = spec.describe_speculation_capabilities("loaded", "mlx", loaded)
    assert result["default"]["num_draft_tokens"] == 2
    if value is False:
        assert result["effective_default"] is False
    else:
        assert result["effective_default"]["num_draft_tokens"] == 5


def test_inherited_policy_changes_per_request_without_mutating_session(monkeypatch):
    provider = MLXProvider.__new__(MLXProvider)
    provider.logger = Mock()
    provider._speculation_request = spec.SpeculationRequest(mode="native_mtp", num_draft_tokens=2)
    provider._speculation_inherits_config = True
    provider._speculation_default_supported = True
    provider._mtp_drafter = object()
    provider._mtp_block_size = 2
    provider._mtp_drafter_id = "head"
    monkeypatch.setattr(spec, "configured_speculation_default", lambda **kw: {"num_draft_tokens": 4})
    provider._apply_per_call_speculation(None)
    assert provider._mtp_call_block_size == 4
    provider._apply_per_call_speculation(False)
    assert provider._mtp_call_disabled is True
    provider._apply_per_call_speculation({"num_draft_tokens": 5})
    assert provider._mtp_call_block_size == 5
    assert provider._speculation_request.num_draft_tokens == 2
    monkeypatch.setattr(spec, "configured_speculation_default", lambda **kw: None)
    provider._apply_per_call_speculation(None)
    assert provider._mtp_call_disabled is True


def test_unsupported_model_does_not_inherit_enabled_default(monkeypatch):
    provider = MLXProvider.__new__(MLXProvider)
    provider._speculation_inherits_config = True
    provider._speculation_default_supported = False
    provider._mtp_drafter = None
    monkeypatch.setattr(spec, "configured_speculation_default", lambda **kw: {"num_draft_tokens": 2})
    provider._apply_per_call_speculation(None)
    assert not provider._mtp_call_request.enabled


@_requires_mlx_stack
def test_default_missing_head_never_enters_upstream_downloader(monkeypatch):
    provider = MLXProvider.__new__(MLXProvider)
    provider.logger = Mock()
    provider.model = "mlx-community/Qwen3.8-27B-4bit"
    provider.model_capabilities = {"speculation": {"native_mtp": True, "runtimes": {"mlx": {"drafter": "absent/head"}}}}
    provider._speculation_request = spec.SpeculationRequest(mode="native_mtp", num_draft_tokens=2)
    provider._speculation_inherits_config = True
    monkeypatch.setattr(spec, "_local_model_directory", lambda model: None)
    assert provider._plan_mtp_lane("target") is None
    assert provider._mtp_outcome_at_load.reason == "mtp_head_not_cached"


@pytest.mark.parametrize("provider", ["huggingface", "ollama", "openai"])
def test_explicit_unsupported_provider_controls_are_never_silent(provider):
    instance = SimpleNamespace(provider=provider, logger=Mock())
    kwargs = {"speculation": {"num_draft_tokens": 2, "require_acceleration": True}}
    with pytest.raises(spec.SpeculationUnavailableError, match="no native MTP adapter"):
        spec.prepare_provider_speculation(instance, kwargs)
    kwargs = {"speculation": {"num_draft_tokens": 2}}
    outcome = spec.prepare_provider_speculation(instance, kwargs)
    assert outcome.used is False
    assert outcome.reason == "native_mtp_backend_unavailable"
    assert "speculation" not in kwargs
    assert spec.prepare_provider_speculation(instance, {"speculation": False}).requested is False


def test_core_cli_default_selector_preserves_other_options():
    from abstractcore.config.main import main
    manager = ConfigurationManager(apply_env=False)
    manager.update_capability_default("output.text", options={"other": 9})
    for value, expected in [("2", 2), ("5", 5), ("off", False), ("inherit", None)]:
        assert main(["config", "--config-file", str(manager.config_file), "set-default", "output.text", "--speculation", value]) == 0
        policy = spec.configured_speculation_default()
        assert (policy.get("num_draft_tokens") if isinstance(policy, dict) else policy) == expected
        assert ConfigurationManager(apply_env=False).get_capability_default("output.text")["options"]["other"] == 9


def test_scoped_routes_do_not_fall_back_to_global_policy():
    manager = ConfigurationManager(apply_env=False)
    manager._save_config()
    assert spec.configured_speculation_default()["num_draft_tokens"] == 2
    assert spec.configured_speculation_default(capability_defaults={}) is None
    assert spec.configured_speculation_default(capability_defaults={"output.text": {"options": {"speculation": False}}}) is False


def test_snapshot_identity_uses_exact_hf_repository(tmp_path):
    checkpoint = tmp_path / "models--mlx-community--Qwen3.8-27B-4bit" / "snapshots" / "revision"
    checkpoint.mkdir(parents=True)
    assert spec.mlx_speculation_artifact(str(checkpoint))["drafter"] == "mlx-community/Qwen3.8-27B-MTP-4bit"
    impostor = tmp_path / "prefix-qwen3.8-27b-suffix"
    impostor.mkdir()
    assert spec.mlx_speculation_artifact(str(impostor)) is None


@pytest.mark.parametrize("stream", [False, True])
def test_base_provider_attaches_refusal_metadata_and_strict_fails_before_generation(stream):
    from abstractcore.providers.base import BaseProvider
    from abstractcore.core.types import GenerateResponse

    class Unsupported(BaseProvider):
        def __init__(self):
            super().__init__("test-model", enable_tracing=False)
            self.provider = "huggingface"
            self.calls = 0

        def _generate_internal(self, **kwargs):
            self.calls += 1
            assert "speculation" not in kwargs
            response = GenerateResponse(content="answer", model=self.model)
            return iter([response]) if kwargs.get("stream") else response

        def list_available_models(self, **kwargs):
            return []

        def get_capabilities(self):
            return []

        def unload_model(self, *args, **kwargs):
            return None

    provider = Unsupported()
    with pytest.raises(spec.SpeculationUnavailableError):
        provider.generate("hello", speculation={"num_draft_tokens": 2, "require_acceleration": True}, stream=stream)
    assert provider.calls == 0
    result = provider.generate("hello", speculation={"num_draft_tokens": 2}, stream=stream)
    rows = list(result) if stream else [result]
    assert rows and all(row.metadata["speculation"]["used"] is False for row in rows)
    assert all(row.metadata["speculation"]["reason"] == "native_mtp_backend_unavailable" for row in rows)
