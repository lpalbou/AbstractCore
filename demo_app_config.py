#!/usr/bin/env python3
"""
Demo: AbstractCore Apps with Centralized Configuration

This demonstrates how AbstractCore apps can integrate with the centralized
configuration system while respecting the priority system:
1. Explicit parameters (highest priority)
2. Centralized configuration defaults
3. App-specific hardcoded fallbacks (lowest priority)
"""

import sys
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent))

def demo_app_configuration():
    """Demo how apps would integrate with centralized configuration."""
    print("🚀 AbstractCore Apps Configuration Integration Demo")
    print("=" * 70)

    # Import our helper functions
    from abstractcore.config.app_helpers import get_app_llm_config, show_config_status_for_app, get_app_default_config
    from abstractcore.config import get_config_manager

    print("\n1️⃣ Current App-Specific Defaults:")
    print("-" * 40)

    apps = ["summarizer", "extractor", "judge"]
    for app_name in apps:
        defaults = get_app_default_config(app_name)
        print(f"   {app_name}: {defaults['provider']}/{defaults['model']}")
        print(f"      Description: {defaults['description']}")
        print()

    print("\n2️⃣ Configuration Priority System Demo:")
    print("-" * 40)

    # Test different scenarios for each app
    for app_name in apps:
        print(f"\n📱 {app_name.upper()} App Configuration:")

        # Scenario 1: No explicit params, no config (uses app defaults)
        provider1, model1 = get_app_llm_config(app_name)
        print(f"   No params, no config: {provider1}/{model1}")

        # Scenario 2: Explicit provider only (mixed)
        provider2, model2 = get_app_llm_config(app_name, explicit_provider="openai")
        print(f"   Explicit provider only: {provider2}/{model2}")

        # Scenario 3: Both explicit params (full override)
        provider3, model3 = get_app_llm_config(app_name, explicit_provider="anthropic", explicit_model="claude-3-5-haiku")
        print(f"   Both explicit params: {provider3}/{model3}")

    print("\n3️⃣ Setting Global Defaults:")
    print("-" * 40)

    # Set a global default
    config_manager = get_config_manager()
    print("   Setting global default: ollama/llama3:8b")
    config_manager.set_default_model("ollama/llama3:8b")

    print("\n4️⃣ After Setting Global Defaults:")
    print("-" * 40)

    for app_name in apps:
        print(f"\n📱 {app_name.upper()} App (with global config):")

        # Now this should use the global config instead of app defaults
        provider1, model1 = get_app_llm_config(app_name)
        print(f"   No params, with config: {provider1}/{model1}")

        # This should still override
        provider2, model2 = get_app_llm_config(app_name, explicit_provider="openai", explicit_model="gpt-4o")
        print(f"   Explicit override: {provider2}/{model2}")

    print("\n5️⃣ Configuration Status:")
    print("-" * 40)
    show_config_status_for_app("summarizer")

    print("\n✅ Key Benefits of This Integration:")
    print("=" * 70)
    print("✅ Consistent defaults across all AbstractCore apps")
    print("✅ User can set global defaults once: abstractcore --set-default-model")
    print("✅ Apps respect explicit parameters (no breaking changes)")
    print("✅ Graceful fallback to app-specific defaults if config unavailable")
    print("✅ Clear error messages with configuration guidance")
    print("✅ No code duplication - all apps use same configuration logic")


def demo_embedding_integration():
    """Demo how embedding configuration would work."""
    print("\n\n🔤 Embeddings Configuration Integration Demo")
    print("=" * 60)

    from abstractcore.config.app_helpers import get_embeddings_config
    from abstractcore.config import get_config_manager

    print("\n1️⃣ Default Embeddings Configuration:")
    provider1, model1 = get_embeddings_config()
    print(f"   Default: {provider1}/{model1}")

    print("\n2️⃣ Setting Custom Embeddings Default:")
    config_manager = get_config_manager()
    config_manager.set_embeddings_provider("ollama", "nomic-embed-text")

    provider2, model2 = get_embeddings_config()
    print(f"   After config: {provider2}/{model2}")

    print("\n3️⃣ Explicit Override:")
    provider3, model3 = get_embeddings_config("huggingface", "all-mpnet-base-v2")
    print(f"   Explicit override: {provider3}/{model3}")


def demo_cli_integration():
    """Demo how this would work with CLI commands."""
    print("\n\n🖥️ CLI Integration Demo")
    print("=" * 40)

    print("\nHow apps would work with centralized configuration:")
    print()

    print("# Set global defaults")
    print("abstractcore --set-default-model openai/gpt-4o-mini")
    print("abstractcore --set-embeddings-provider ollama nomic-embed-text")
    print()

    print("# Now ALL apps use these defaults unless overridden")
    print("summarizer document.txt                    # Uses openai/gpt-4o-mini")
    print("extractor data.pdf                        # Uses openai/gpt-4o-mini")
    print("judge content.md                          # Uses openai/gpt-4o-mini")
    print()

    print("# Individual apps can still override")
    print("summarizer doc.txt --provider ollama --model gemma3:2b")
    print()

    print("# Configuration status works for apps")
    print("abstractcore --status                     # Shows global defaults")


if __name__ == "__main__":
    demo_app_configuration()
    demo_embedding_integration()
    demo_cli_integration()

    print("\n\n🎯 SUMMARY")
    print("=" * 70)
    print("This demo shows how AbstractCore apps can integrate with centralized")
    print("configuration while maintaining backward compatibility and the priority")
    print("system: explicit parameters > config defaults > app fallbacks")
    print()
    print("To implement this in the actual apps, we would:")
    print("1. Import the app_helpers functions")
    print("2. Replace hardcoded provider/model logic with get_app_llm_config()")
    print("3. Update error messages to show configuration guidance")
    print("4. Maintain all existing CLI argument compatibility")