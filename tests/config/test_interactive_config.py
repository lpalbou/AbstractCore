"""Tests for the interactive config flow."""
import builtins

from abstractcore.config import main as config_main


class StubConfigManager:
    """Capture interactive-config calls without touching disk."""

    def __init__(self) -> None:
        self.default_model = None
        self.vision_calls = []
        self.api_keys = {}
        self.console_levels = []

    def set_default_model(self, model: str) -> None:
        self.default_model = model

    def set_vision_provider(self, provider: str, model: str) -> None:
        self.vision_calls.append((provider, model))

    def set_api_key(self, provider: str, key: str) -> None:
        self.api_keys[provider] = key

    def set_console_log_level(self, level: str) -> None:
        self.console_levels.append(level)


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
            "",   # console logging (default)
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    config_main.interactive_configure()

    assert stub.console_levels == ["ERROR"]
