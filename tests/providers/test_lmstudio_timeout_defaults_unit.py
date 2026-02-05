import pytest


@pytest.mark.basic
def test_lmstudio_provider_defaults_to_unlimited_timeout(monkeypatch):
    """ADR-0027: LMStudioProvider should not inherit low global default timeouts."""

    class DummyCfgMgr:
        def __init__(self):
            from abstractcore.config.manager import AbstractCoreConfig

            self.config = AbstractCoreConfig.default()

        def is_offline_first(self):
            return False

        def is_network_allowed(self):
            return True

        def should_force_local_files_only(self):
            return False

        def get_default_timeout(self):
            return 1200.0

        def get_tool_timeout(self):
            return 600.0

    # BaseProvider consults the config manager when `timeout` is omitted. We patch it to a low
    # value to ensure LMStudioProvider still forces `timeout=None` by default.
    monkeypatch.setattr("abstractcore.config.manager.get_config_manager", lambda: DummyCfgMgr())

    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)

    from abstractcore.providers.lmstudio_provider import LMStudioProvider

    provider = LMStudioProvider(model="qwen/qwen3-next-80b", base_url="http://localhost:1234/v1")
    assert provider._timeout is None


@pytest.mark.basic
def test_lmstudio_provider_respects_explicit_timeout(monkeypatch):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)

    from abstractcore.providers.lmstudio_provider import LMStudioProvider

    provider = LMStudioProvider(
        model="qwen/qwen3-next-80b",
        base_url="http://localhost:1234/v1",
        timeout=123.0,
    )
    assert provider._timeout == 123.0
