"""
Comprehensive tests for timeout configuration system.

Tests cover:
1. Default timeout values (10 minutes)
2. ConfigurationManager timeout methods
3. BaseProvider timeout integration
4. Config file persistence
5. Priority system (kwargs > config > defaults)
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import configuration components
from abstractcore.config.manager import (
    ConfigurationManager,
    TimeoutConfig,
    AbstractCoreConfig
)


class TestTimeoutDefaults:
    """Test default timeout values."""

    def test_timeout_config_defaults(self):
        """Verify TimeoutConfig has correct default values."""
        config = TimeoutConfig()
        assert config.default_timeout == 600.0, "Default HTTP timeout should be 10 minutes (600s)"
        assert config.tool_timeout == 600.0, "Default tool timeout should be 10 minutes (600s)"

    def test_abstract_core_config_includes_timeouts(self):
        """Verify AbstractCoreConfig includes TimeoutConfig."""
        config = AbstractCoreConfig.default()
        assert hasattr(config, 'timeouts'), "AbstractCoreConfig should have timeouts attribute"
        assert isinstance(config.timeouts, TimeoutConfig), "timeouts should be TimeoutConfig instance"
        assert config.timeouts.default_timeout == 600.0
        assert config.timeouts.tool_timeout == 600.0


class TestConfigurationManagerTimeouts:
    """Test ConfigurationManager timeout methods."""

    def setup_method(self):
        """Create a temporary config directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "abstractcore.json"

    def test_set_default_timeout(self):
        """Test setting default HTTP timeout."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            manager = ConfigurationManager()

            # Set timeout to 300 seconds (5 minutes)
            result = manager.set_default_timeout(300.0)
            assert result is True, "set_default_timeout should return True on success"
            assert manager.config.timeouts.default_timeout == 300.0

            # Verify persistence
            assert manager.config_file.exists(), "Config file should exist"
            with open(manager.config_file, 'r') as f:
                data = json.load(f)
                assert data['timeouts']['default_timeout'] == 300.0

    def test_set_tool_timeout(self):
        """Test setting tool execution timeout."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            manager = ConfigurationManager()

            # Set timeout to 120 seconds (2 minutes)
            result = manager.set_tool_timeout(120.0)
            assert result is True, "set_tool_timeout should return True on success"
            assert manager.config.timeouts.tool_timeout == 120.0

            # Verify persistence
            with open(manager.config_file, 'r') as f:
                data = json.load(f)
                assert data['timeouts']['tool_timeout'] == 120.0

    def test_set_timeout_validates_positive_values(self):
        """Test that timeout setters reject non-positive values."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            manager = ConfigurationManager()

            # Test negative value
            result = manager.set_default_timeout(-10.0)
            assert result is False, "Should reject negative timeout"

            # Test zero value
            result = manager.set_tool_timeout(0.0)
            assert result is False, "Should reject zero timeout"

    def test_get_default_timeout(self):
        """Test retrieving default HTTP timeout."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            manager = ConfigurationManager()
            timeout = manager.get_default_timeout()
            assert timeout == 600.0, "Should return default 600s"

            # Change and retrieve
            manager.set_default_timeout(450.0)
            timeout = manager.get_default_timeout()
            assert timeout == 450.0, "Should return updated timeout"

    def test_get_tool_timeout(self):
        """Test retrieving tool execution timeout."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            manager = ConfigurationManager()
            timeout = manager.get_tool_timeout()
            assert timeout == 600.0, "Should return default 600s"

            # Change and retrieve
            manager.set_tool_timeout(180.0)
            timeout = manager.get_tool_timeout()
            assert timeout == 180.0, "Should return updated timeout"


class TestConfigFilePersistence:
    """Test timeout configuration persistence in config file."""

    def setup_method(self):
        """Create a temporary config directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / ".abstractcore" / "config" / "abstractcore.json"

    def test_timeouts_saved_to_config_file(self):
        """Test that timeouts are saved to config file."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            manager = ConfigurationManager()

            # Set custom timeouts
            manager.set_default_timeout(420.0)
            manager.set_tool_timeout(240.0)

            # Verify file content
            assert manager.config_file.exists()
            with open(manager.config_file, 'r') as f:
                data = json.load(f)

            assert 'timeouts' in data, "Config file should have 'timeouts' section"
            assert data['timeouts']['default_timeout'] == 420.0
            assert data['timeouts']['tool_timeout'] == 240.0

    def test_timeouts_loaded_from_config_file(self):
        """Test that timeouts are loaded from existing config file."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            # Create config file with custom timeouts
            config_dir = Path(self.temp_dir) / ".abstractcore" / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "abstractcore.json"

            config_data = {
                "vision": {"strategy": "disabled"},
                "embeddings": {"provider": "huggingface", "model": "all-minilm-l6-v2"},
                "app_defaults": {},
                "default_models": {},
                "api_keys": {},
                "cache": {},
                "logging": {},
                "timeouts": {
                    "default_timeout": 900.0,
                    "tool_timeout": 360.0
                }
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f)

            # Load configuration
            manager = ConfigurationManager()

            # Verify loaded timeouts
            assert manager.config.timeouts.default_timeout == 900.0
            assert manager.config.timeouts.tool_timeout == 360.0

    def test_missing_timeouts_section_uses_defaults(self):
        """Test that missing timeout section in config uses defaults."""
        with patch.object(Path, 'home', return_value=Path(self.temp_dir)):
            # Create config file WITHOUT timeouts section (legacy config)
            config_dir = Path(self.temp_dir) / ".abstractcore" / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "abstractcore.json"

            config_data = {
                "vision": {"strategy": "disabled"},
                "embeddings": {"provider": "huggingface", "model": "all-minilm-l6-v2"},
                "app_defaults": {},
                "default_models": {},
                "api_keys": {},
                "cache": {},
                "logging": {}
                # No "timeouts" section
            }

            with open(config_file, 'w') as f:
                json.dump(config_data, f)

            # Load configuration
            manager = ConfigurationManager()

            # Should use defaults
            assert manager.config.timeouts.default_timeout == 600.0
            assert manager.config.timeouts.tool_timeout == 600.0


class TestBaseProviderTimeoutIntegration:
    """Test BaseProvider integration with timeout configuration."""

    def _create_concrete_provider(self, **kwargs):
        """Create a concrete implementation of BaseProvider for testing."""
        from abstractcore.providers.base import BaseProvider

        class ConcreteProvider(BaseProvider):
            """Concrete provider for testing."""
            def _generate_internal(self, *args, **kwargs):
                pass
            def list_available_models(self, **kwargs):
                return ["test-model"]
            def get_capabilities(self):
                return ["text"]

        return ConcreteProvider(**kwargs)

    def test_provider_reads_timeout_from_config(self):
        """Test that BaseProvider reads timeout from config when not explicitly provided."""
        # Mock config manager to return custom timeout
        mock_config_manager = MagicMock()
        mock_config_manager.get_default_timeout.return_value = 450.0
        mock_config_manager.get_tool_timeout.return_value = 300.0

        with patch('abstractcore.config.get_config_manager', return_value=mock_config_manager):
            # Create provider without explicit timeout
            provider = self._create_concrete_provider(model="test-model")

            # Should use config values
            assert provider._timeout == 450.0, "Should use config default_timeout"
            assert provider._tool_timeout == 300.0, "Should use config tool_timeout"

    def test_provider_explicit_timeout_overrides_config(self):
        """Test that explicit timeout parameter overrides config."""
        # Mock config manager
        mock_config_manager = MagicMock()
        mock_config_manager.get_default_timeout.return_value = 450.0
        mock_config_manager.get_tool_timeout.return_value = 300.0

        with patch('abstractcore.config.get_config_manager', return_value=mock_config_manager):
            # Create provider with explicit timeout
            provider = self._create_concrete_provider(model="test-model", timeout=120.0, tool_timeout=60.0)

            # Should use explicit values, not config
            assert provider._timeout == 120.0, "Explicit timeout should override config"
            assert provider._tool_timeout == 60.0, "Explicit tool_timeout should override config"

    def test_provider_fallback_when_config_unavailable(self):
        """Test that provider uses hardcoded defaults when config unavailable."""
        # Mock config manager to raise exception
        with patch('abstractcore.config.get_config_manager', side_effect=Exception("Config unavailable")):
            # Create provider
            provider = self._create_concrete_provider(model="test-model")

            # Should use hardcoded defaults
            assert provider._timeout == 600.0, "Should use hardcoded default (600s) when config unavailable"
            assert provider._tool_timeout == 600.0, "Should use hardcoded default (600s) when config unavailable"


class TestTimeoutPrioritySystem:
    """Test timeout priority system: kwargs > config > defaults."""

    def _create_concrete_provider(self, **kwargs):
        """Create a concrete implementation of BaseProvider for testing."""
        from abstractcore.providers.base import BaseProvider

        class ConcreteProvider(BaseProvider):
            """Concrete provider for testing."""
            def _generate_internal(self, *args, **kwargs):
                pass
            def list_available_models(self, **kwargs):
                return ["test-model"]
            def get_capabilities(self):
                return ["text"]

        return ConcreteProvider(**kwargs)

    def test_priority_explicit_kwargs_highest(self):
        """Test that explicit kwargs have highest priority."""
        mock_config_manager = MagicMock()
        mock_config_manager.get_default_timeout.return_value = 450.0
        mock_config_manager.get_tool_timeout.return_value = 300.0

        with patch('abstractcore.config.get_config_manager', return_value=mock_config_manager):
            provider = self._create_concrete_provider(model="test", timeout=90.0, tool_timeout=45.0)

            # Explicit kwargs should win
            assert provider._timeout == 90.0
            assert provider._tool_timeout == 45.0

    def test_priority_config_over_defaults(self):
        """Test that config values override defaults."""
        mock_config_manager = MagicMock()
        mock_config_manager.get_default_timeout.return_value = 720.0  # 12 minutes
        mock_config_manager.get_tool_timeout.return_value = 360.0   # 6 minutes

        with patch('abstractcore.config.get_config_manager', return_value=mock_config_manager):
            provider = self._create_concrete_provider(model="test")

            # Config values should override defaults
            assert provider._timeout == 720.0
            assert provider._tool_timeout == 360.0

    def test_priority_defaults_when_no_config_or_kwargs(self):
        """Test that defaults are used when no config or kwargs provided."""
        with patch('abstractcore.config.get_config_manager', side_effect=Exception("No config")):
            provider = self._create_concrete_provider(model="test")

            # Should use hardcoded defaults
            assert provider._timeout == 600.0
            assert provider._tool_timeout == 600.0


class TestTimeoutGettersSetters:
    """Test timeout getter and setter methods on BaseProvider."""

    def _create_concrete_provider(self, **kwargs):
        """Create a concrete implementation of BaseProvider for testing."""
        from abstractcore.providers.base import BaseProvider

        class ConcreteProvider(BaseProvider):
            """Concrete provider for testing."""
            def _generate_internal(self, *args, **kwargs):
                pass
            def list_available_models(self, **kwargs):
                return ["test-model"]
            def get_capabilities(self):
                return ["text"]

        return ConcreteProvider(**kwargs)

    def test_get_timeout(self):
        """Test get_timeout method."""
        provider = self._create_concrete_provider(model="test", timeout=300.0)
        assert provider.get_timeout() == 300.0

    def test_set_timeout(self):
        """Test set_timeout method."""
        provider = self._create_concrete_provider(model="test", timeout=300.0)
        provider.set_timeout(450.0)
        assert provider.get_timeout() == 450.0
        assert provider._timeout == 450.0

    def test_get_tool_timeout(self):
        """Test get_tool_timeout method."""
        provider = self._create_concrete_provider(model="test", tool_timeout=180.0)
        assert provider.get_tool_timeout() == 180.0

    def test_set_tool_timeout(self):
        """Test set_tool_timeout method."""
        provider = self._create_concrete_provider(model="test", tool_timeout=180.0)
        provider.set_tool_timeout(240.0)
        assert provider.get_tool_timeout() == 240.0
        assert provider._tool_timeout == 240.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
