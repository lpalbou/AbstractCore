"""Tests for the interactive config flow."""
import builtins

from abstractcore.config import main as config_main


class StubConfigManager:
    """Capture interactive-config calls without touching disk."""

    def __init__(self) -> None:
        self.default_model = None
        self.vision_calls = []
        self.api_keys = {}
        self.server_auth_token = None
        self.server_allow_unauthenticated = None
        self.server_base_url_allowlist = None
        self.server_url_fetch_allowlist = None
        self.server_media_root = None
        self.server_allow_local_files = None
        self.server_bind_calls = []
        self.audio_strategies = []
        self.video_strategies = []
        self.embeddings_models = []
        self.console_levels = []

    def set_default_model(self, model: str) -> None:
        self.default_model = model

    def set_vision_provider(self, provider: str, model: str) -> None:
        self.vision_calls.append((provider, model))

    def set_api_key(self, provider: str, key: str) -> None:
        self.api_keys[provider] = key

    def set_server_auth_token(self, key: str) -> None:
        self.server_auth_token = key

    def set_server_allow_unauthenticated(self, enabled: bool) -> None:
        self.server_allow_unauthenticated = enabled

    def set_server_base_url_allowlist(self, allowlist: str) -> None:
        self.server_base_url_allowlist = allowlist

    def set_server_url_fetch_allowlist(self, allowlist: str) -> None:
        self.server_url_fetch_allowlist = allowlist

    def set_server_media_root(self, path: str) -> None:
        self.server_media_root = path

    def set_server_allow_local_files(self, enabled: bool) -> None:
        self.server_allow_local_files = enabled

    def set_server_bind(self, host=None, port=None) -> bool:
        self.server_bind_calls.append((host, port))
        return True

    def set_console_log_level(self, level: str) -> None:
        self.console_levels.append(level)

    def set_audio_strategy(self, strategy: str) -> None:
        self.audio_strategies.append(strategy)

    def set_video_strategy(self, strategy: str) -> None:
        self.video_strategies.append(strategy)

    def set_embeddings_model(self, model: str) -> None:
        self.embeddings_models.append(model)


def test_interactive_configure_accepts_any_vision_provider(monkeypatch) -> None:
    """Interactive config should accept any provider/model pair for vision fallback."""
    stub = StubConfigManager()
    monkeypatch.setattr(config_main, "get_config_manager", lambda: stub)

    inputs = iter(
        [
            "n",  # default model
            "y",  # vision fallback
            "openai-compatible",
            "my-vision-model",
            "n",  # api keys
            "n",  # server auth/hardening
            "",   # audio strategy (default)
            "",   # video strategy (default)
            "n",  # embeddings
            "",   # console logging (default)
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    config_main.interactive_configure()

    assert stub.vision_calls == [("openai-compatible", "my-vision-model")]


def test_interactive_configure_accepts_provider_model_combo(monkeypatch) -> None:
    """Interactive config should accept provider/model in a single prompt."""
    stub = StubConfigManager()
    monkeypatch.setattr(config_main, "get_config_manager", lambda: stub)

    inputs = iter(
        [
            "n",  # default model
            "y",  # vision fallback
            "lmstudio/qwen/qwen2.5-vl-7b",
            "",   # model prompt (provider/model already provided)
            "n",  # api keys
            "n",  # server auth/hardening
            "",   # audio strategy (default)
            "",   # video strategy (default)
            "n",  # embeddings
            "",   # console logging (default)
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    config_main.interactive_configure()

    assert stub.vision_calls == [("lmstudio", "qwen/qwen2.5-vl-7b")]


def test_interactive_configure_defaults_console_level_to_error(monkeypatch) -> None:
    """Interactive config should default console logging to ERROR."""
    stub = StubConfigManager()
    monkeypatch.setattr(config_main, "get_config_manager", lambda: stub)

    inputs = iter(
        [
            "n",  # default model
            "n",  # vision fallback
            "n",  # api keys
            "n",  # server auth/hardening
            "",   # audio strategy (default)
            "",   # video strategy (default)
            "n",  # embeddings
            "",   # console logging (default)
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    config_main.interactive_configure()

    assert stub.console_levels == ["ERROR"]


def test_interactive_configure_can_set_server_auth(monkeypatch) -> None:
    """Interactive config should expose the hardened HTTP server auth settings."""
    stub = StubConfigManager()
    monkeypatch.setattr(config_main, "get_config_manager", lambda: stub)

    inputs = iter(
        [
            "n",  # default model
            "n",  # vision fallback
            "n",  # api keys
            "y",  # server auth/hardening
            "server-secret",
            "n",  # allow unauthenticated
            "https://example.com/v1",
            "https://files.example.com",
            "/srv/abstractcore-media",
            "n",  # unrestricted local files
            "127.0.0.1",
            "8787",
            "",   # audio strategy (default)
            "",   # video strategy (default)
            "n",  # embeddings
            "",   # console logging (default)
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    config_main.interactive_configure()

    assert stub.server_auth_token == "server-secret"
    assert stub.server_allow_unauthenticated is False
    assert stub.server_base_url_allowlist == "https://example.com/v1"
    assert stub.server_url_fetch_allowlist == "https://files.example.com"
    assert stub.server_media_root == "/srv/abstractcore-media"
    assert stub.server_allow_local_files is False
    assert stub.server_bind_calls == [("127.0.0.1", None), (None, 8787)]
