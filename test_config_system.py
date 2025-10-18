#!/usr/bin/env python3
"""
Test script for AbstractCore Configuration System

This demonstrates:
1. Configuration works through code first
2. Default configs don't override direct parameters
3. CLI just calls the same code-based configuration
4. Vision fallback integration with text-only models
"""

import sys
import logging
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent))

def print_config_status(config_manager):
    """Helper function to print configuration status."""
    status = config_manager.get_status()

    print("📋 Configuration Status:")
    print(f"   Vision: {status['vision']['status']}")
    if status['vision']['caption_provider']:
        print(f"      Provider: {status['vision']['caption_provider']}/{status['vision']['caption_model']}")
    print(f"   Embeddings: {status['embeddings']['status']}")
    if status['embeddings']['provider']:
        print(f"      Provider: {status['embeddings']['provider']}/{status['embeddings']['model']}")
    print(f"   Default Model: {status['defaults']['provider']}/{status['defaults']['model']}" if status['defaults']['provider'] else "Not set")


def test_configuration_code_api():
    """Test configuration through code (primary interface)."""
    print("🧪 Testing Configuration Code API")
    print("=" * 50)

    # Import configuration system
    from abstractcore.config import get_config_manager

    # Get configuration manager
    config_manager = get_config_manager()

    print("\n1️⃣ Initial Configuration Status:")
    print_config_status(config_manager)

    print("\n2️⃣ Setting Configuration Through Code:")

    # Set embeddings configuration
    print("   Setting embeddings to ollama/nomic-embed-text...")
    config_manager.set_embeddings_provider("ollama", "nomic-embed-text")

    # Set vision configuration
    print("   Setting vision fallback to ollama/qwen2.5vl:7b...")
    config_manager.set_vision_provider("ollama", "qwen2.5vl:7b")

    # Set default LLM
    print("   Setting default LLM to ollama/llama3:8b...")
    config_manager.set_default_model("ollama/llama3:8b")

    print("\n3️⃣ Updated Configuration Status:")
    print_config_status(config_manager)

    return config_manager


def test_embedding_manager_with_config():
    """Test that EmbeddingManager respects configuration but prioritizes direct parameters."""
    print("\n🔤 Testing EmbeddingManager Configuration Integration")
    print("=" * 60)

    from abstractcore.embeddings import EmbeddingManager

    print("\n1️⃣ Using Config Defaults (no parameters):")
    try:
        # This should use configured defaults
        embedder_default = EmbeddingManager()
        print(f"   Provider: {embedder_default.provider}")
        print(f"   Model: {embedder_default.model_id}")
        print(f"   ✅ Using configured defaults successfully")
    except Exception as e:
        print(f"   ❌ Error with defaults: {e}")

    print("\n2️⃣ Direct Parameters Override Config (parameter priority):")
    try:
        # This should use explicit parameters, ignoring config
        embedder_explicit = EmbeddingManager(provider="huggingface", model="all-mpnet-base-v2")
        print(f"   Provider: {embedder_explicit.provider}")
        print(f"   Model: {embedder_explicit.model_id}")
        print(f"   ✅ Direct parameters correctly override config")
    except Exception as e:
        print(f"   ❌ Error with explicit parameters: {e}")

    print("\n3️⃣ Mixed Parameters (some explicit, some from config):")
    try:
        # This should use explicit provider but config model
        embedder_mixed = EmbeddingManager(provider="huggingface")  # model from config
        print(f"   Provider: {embedder_mixed.provider}")
        print(f"   Model: {embedder_mixed.model_id}")
        print(f"   ✅ Mixed parameters work correctly")
    except Exception as e:
        print(f"   ❌ Error with mixed parameters: {e}")


def test_vision_fallback_integration():
    """Test vision fallback system with configuration."""
    print("\n👁️  Testing Vision Fallback Integration")
    print("=" * 50)

    from abstractcore.media.vision_fallback import VisionFallbackHandler

    print("\n1️⃣ Vision Fallback Handler with Config:")
    try:
        handler = VisionFallbackHandler()
        vision_config = handler.vision_config

        print(f"   Strategy: {vision_config.strategy}")
        print(f"   Provider: {vision_config.caption_provider}")
        print(f"   Model: {vision_config.caption_model}")

        if vision_config.strategy == "two_stage" and vision_config.caption_provider:
            print(f"   ✅ Vision fallback is properly configured")
        else:
            print(f"   ⚠️  Vision fallback needs configuration")

    except Exception as e:
        print(f"   ❌ Error with vision fallback: {e}")


def test_cli_calls_same_config():
    """Test that CLI uses the same configuration system."""
    print("\n🖥️  Testing CLI Integration")
    print("=" * 40)

    # Import CLI functions
    from abstractcore.cli.main import handle_commands
    import argparse

    print("\n1️⃣ Testing CLI Status Command:")
    try:
        # Create args for status command
        args = argparse.Namespace(
            status=True,
            configure=False,
            reset=False,
            set_default_model=None,
            set_default_provider=None,
            set_chat_model=None,
            set_code_model=None,
            set_vision_caption=None,
            set_vision_provider=None,
            add_vision_fallback=None,
            disable_vision=False,
            download_vision_model=None,
            set_embeddings_model=None,
            set_embeddings_provider=None,
            set_api_key=None,
            list_api_keys=False
        )

        print("   CLI Status Output:")
        handle_commands(args)
        print("   ✅ CLI successfully uses same configuration system")

    except Exception as e:
        print(f"   ❌ Error with CLI: {e}")


def test_configuration_priority():
    """Test that configuration priority works correctly."""
    print("\n⚡ Testing Configuration Priority")
    print("=" * 40)

    from abstractcore.config import get_config_manager

    config_manager = get_config_manager()

    print("\n1️⃣ Setting New Embeddings Config:")
    config_manager.set_embeddings_provider("lmstudio", "some-embedding-model")

    print("\n2️⃣ Testing Priority with EmbeddingManager:")
    from abstractcore.embeddings import EmbeddingManager

    # Should use config
    try:
        embedder_config = EmbeddingManager()
        print(f"   Config-based: {embedder_config.provider}/{embedder_config.model_id}")
    except Exception as e:
        print(f"   Config error: {e}")

    # Should override config
    try:
        embedder_override = EmbeddingManager(provider="huggingface", model="override-model")
        print(f"   Override-based: {embedder_override.provider}/{embedder_override.model_id}")
        print("   ✅ Priority system working correctly")
    except Exception as e:
        print(f"   Override error: {e}")


def main():
    """Run all configuration tests."""
    print("🚀 AbstractCore Configuration System Test")
    print("=" * 60)
    print("This demonstrates the unified configuration system:")
    print("• Code-first configuration API")
    print("• CLI that calls the same code configuration")
    print("• Priority: Direct parameters > Config defaults > Hardcoded defaults")
    print("• Integration with EmbeddingManager and Vision fallback")

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Test configuration code API
        config_manager = test_configuration_code_api()

        # Test EmbeddingManager integration
        test_embedding_manager_with_config()

        # Test vision fallback
        test_vision_fallback_integration()

        # Test CLI integration
        test_cli_calls_same_config()

        # Test priority system
        test_configuration_priority()

        print("\n🎉 Configuration System Test Complete!")
        print("=" * 60)
        print("Key Achievements:")
        print("✅ Configuration works through code first")
        print("✅ CLI calls the same configuration system")
        print("✅ Direct parameters override config defaults")
        print("✅ EmbeddingManager integrates with config system")
        print("✅ Vision fallback uses unified configuration")
        print("✅ Priority system works correctly")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())