"""
Vision Configuration CLI Commands

Handles CLI commands for vision fallback configuration:
- abstractcore --set-vision-caption
- abstractcore --set-vision-provider
- abstractcore --vision-status
- abstractcore --list-vision
- abstractcore --download-vision-model
"""

import argparse
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

def handle_vision_commands(args) -> bool:
    """
    Handle vision-related CLI commands.

    Returns True if a vision command was processed, False otherwise.
    """
    from ..media.vision_fallback import VisionFallbackHandler

    handler = VisionFallbackHandler()

    if hasattr(args, 'set_vision_caption') and args.set_vision_caption:
        return handle_set_vision_caption(handler, args.set_vision_caption)

    elif hasattr(args, 'set_vision_provider') and args.set_vision_provider:
        provider, model = args.set_vision_provider
        return handle_set_vision_provider(handler, provider, model)

    elif hasattr(args, 'vision_status') and args.vision_status:
        return handle_vision_status(handler)

    elif hasattr(args, 'list_vision') and args.list_vision:
        return handle_list_vision(handler)

    elif hasattr(args, 'download_vision_model') and args.download_vision_model:
        model_name = args.download_vision_model if args.download_vision_model != True else "blip-base-caption"
        return handle_download_vision_model(handler, model_name)

    elif hasattr(args, 'configure') and args.configure == 'vision':
        return handle_configure_vision(handler)

    return False

def handle_set_vision_caption(handler: 'VisionFallbackHandler', model: str) -> bool:
    """Handle --set-vision-caption command."""
    print(f"🔧 Setting vision caption model: {model}")

    # Try to determine provider from model name
    provider = detect_provider_from_model(model)
    if not provider:
        print("❌ Could not determine provider from model name.")
        print("💡 Use --set-vision-provider instead: abstractcore --set-vision-provider ollama qwen2.5vl:7b")
        return True

    success = handler.set_vision_provider(provider, model)
    if success:
        print(f"✅ Vision caption model set to {provider}/{model}")
        print("🎯 Vision fallback is now enabled for text-only models")
        print("\n💡 Test it: Use any text-only model with an image")
    else:
        print(f"❌ Failed to set vision caption model {provider}/{model}")
        print("💡 Check that the provider and model are available")

    return True

def handle_set_vision_provider(handler: 'VisionFallbackHandler', provider: str, model: str) -> bool:
    """Handle --set-vision-provider command."""
    print(f"🔧 Setting vision provider: {provider}/{model}")

    success = handler.set_vision_provider(provider, model)
    if success:
        print(f"✅ Vision provider set to {provider}/{model}")
        print("🎯 Vision fallback is now enabled for text-only models")
        print("\n💡 Test it: Use any text-only model with an image")
    else:
        print(f"❌ Failed to set vision provider {provider}/{model}")
        print("💡 Check that the provider and model are available")
        print("💡 Make sure the model supports vision capabilities")

    return True

def handle_vision_status(handler: 'VisionFallbackHandler') -> bool:
    """Handle --vision-status command."""
    print("🔍 Vision Configuration Status")
    print("=" * 50)

    status = handler.get_status()

    # Strategy
    strategy = status.get('strategy', 'unknown')
    print(f"📋 Strategy: {strategy}")

    # Primary provider
    primary = status.get('primary_provider')
    if primary:
        provider_str = f"{primary['provider']}/{primary['model']}"
        status_icon = "✅" if primary['status'] == 'available' else "❌"
        print(f"🎯 Primary: {status_icon} {provider_str}")
    else:
        print("🎯 Primary: ❌ Not configured")

    # Fallback providers
    fallbacks = status.get('fallback_providers', [])
    if fallbacks:
        print(f"🔄 Fallbacks:")
        for fallback in fallbacks:
            provider_str = f"{fallback['provider']}/{fallback['model']}"
            status_icon = "✅" if fallback['status'] == 'available' else "❌"
            print(f"   {status_icon} {provider_str}")
    else:
        print("🔄 Fallbacks: None configured")

    # Local models
    local_models = status.get('local_models', [])
    if local_models:
        print(f"💾 Local Models:")
        for model in local_models:
            print(f"   ✅ {model['name']}")
    else:
        print("💾 Local Models: None downloaded")

    # Recommendations
    recommendations = status.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")

    print("=" * 50)
    return True

def handle_list_vision(handler: 'VisionFallbackHandler') -> bool:
    """Handle --list-vision command."""
    print("📋 Vision Configuration Examples (non-exhaustive)")
    print("=" * 60)

    print("\n🔧 EXAMPLE PROVIDER/MODEL PAIRS")
    print("-" * 30)

    examples = [
        "lmstudio/qwen/qwen2.5-vl-7b",
        "huggingface/Salesforce/blip-image-captioning-base",
        "mlx/<vision-model>",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "openai-compatible/my-vision-model",
    ]
    for example in examples:
        print(f"  • {example}")

    print("\n💾 DOWNLOADABLE MODELS")
    print("-" * 30)
    download_models = [
        "blip-base-caption (~990MB) - Basic image captioning",
        "git-base (~400MB) - Lightweight Microsoft GIT model",
        "vit-gpt2 (~500MB) - ViT + GPT-2 captioning model"
    ]

    for model in download_models:
        print(f"  • {model}")

    print("\n📖 CONFIGURATION COMMANDS")
    print("-" * 30)
    print("  abstractcore --set-vision-provider PROVIDER MODEL")
    print("  abstractcore --add-vision-fallback PROVIDER MODEL")
    print("  abstractcore --disable-vision")
    print("  abstractcore --download-vision-model")
    print("  abstractcore --download-vision-model blip-base-caption")
    print("  abstractcore --vision-status")
    print("  abstractcore --config")

    print("\n💡 QUICK START")
    print("-" * 30)
    print("  1. Choose any provider/model:")
    print("     abstractcore --set-vision-provider PROVIDER MODEL")
    print("  2. (Optional) Download an offline caption model:")
    print("     abstractcore --download-vision-model blip-base-caption")
    print("  3. Verify:")
    print("     abstractcore --vision-status")

    print("=" * 60)
    return True

def handle_download_vision_model(handler: 'VisionFallbackHandler', model_name: str) -> bool:
    """Handle --download-vision-model command."""
    print(f"📥 Downloading vision model: {model_name}")

    # Available models for download
    AVAILABLE_MODELS = {
        "blip-base-caption": {
            "url": "Salesforce/blip-image-captioning-base",
            "size": "990MB",
            "description": "Basic image captioning model"
        },
        "git-base": {
            "url": "microsoft/git-base",
            "size": "400MB",
            "description": "Lightweight Microsoft GIT model"
        },
        "vit-gpt2": {
            "url": "nlpconnect/vit-gpt2-image-captioning",
            "size": "500MB",
            "description": "ViT + GPT-2 captioning model"
        }
    }

    if model_name not in AVAILABLE_MODELS:
        print(f"❌ Model '{model_name}' not available for download")
        print("\n📋 Available models:")
        for name, info in AVAILABLE_MODELS.items():
            print(f"  • {name} ({info['size']}) - {info['description']}")
        return True

    model_info = AVAILABLE_MODELS[model_name]
    print(f"📊 Model: {model_info['description']}")
    print(f"📦 Size: {model_info['size']}")
    print(f"🔗 Source: {model_info['url']}")

    # Check if transformers is available
    try:
        import transformers
    except ImportError:
        print("❌ transformers library not installed")
        print("💡 Install with: pip install transformers torch")
        return True

    # Create models directory
    models_dir = Path(handler.config.local_models_path).expanduser()
    model_path = models_dir / model_name

    if model_path.exists():
        print(f"✅ Model already downloaded at {model_path}")

        # Enable the downloaded model
        handler.config.strategy = "two_stage"
        handler._save_config(handler.config)
        print("🎯 Vision fallback enabled with local model")
        return True

    try:
        print("🔄 Downloading model...")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Download using transformers
        from transformers import AutoProcessor, AutoModel

        # Download model and processor
        processor = AutoProcessor.from_pretrained(model_info['url'], use_fast=False)
        model = AutoModel.from_pretrained(model_info['url'])

        # Save to local directory
        processor.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))

        print(f"✅ Model downloaded successfully to {model_path}")

        # Enable vision fallback with this model
        handler.config.strategy = "two_stage"
        handler._save_config(handler.config)

        print("🎯 Vision fallback enabled with local model")
        print("\n💡 Test it: Use any text-only model with an image")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Check internet connection and disk space")

        # Clean up partial download
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)

    return True

def _prompt_provider_and_model() -> Optional[Tuple[str, str]]:
    """Prompt for provider/model with provider-agnostic guidance."""
    print("Vision fallback supports any provider and any model (local or cloud).")
    print("Examples (non-exhaustive):")
    print("  lmstudio/qwen/qwen2.5-vl-7b, huggingface/Salesforce/blip-image-captioning-base, mlx/<vision-model>")
    print("  openai/gpt-4o, anthropic/claude-3-5-sonnet, openai-compatible/my-vision-model")
    print("Tip: use `abstractcore --download-vision-model` for offline caption models.")

    provider_raw = input("Enter vision provider id (or provider/model): ").strip()
    model = input("Enter vision model name (or leave blank if provider/model): ").strip()
    if provider_raw and not model and "/" in provider_raw:
        provider, model = provider_raw.split("/", 1)
    else:
        provider = provider_raw
    if not provider or not model:
        print("❌ Provider and model are required")
        return None
    return provider, model


def handle_configure_vision(handler: 'VisionFallbackHandler') -> bool:
    """Handle --configure vision command (interactive setup)."""
    print("🔧 Interactive Vision Configuration")
    print("=" * 50)

    print("\nChoose your vision configuration strategy:")
    print("1. Set provider/model (any provider)")
    print("2. Download lightweight local model")
    print("3. Show current status")
    print("4. Disable vision fallback")

    try:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            return configure_provider(handler)
        elif choice == "2":
            return configure_download_model(handler)
        elif choice == "3":
            return handle_vision_status(handler)
        elif choice == "4":
            handler.disable()
            print("✅ Vision fallback disabled")
            return True
        else:
            print("❌ Invalid choice")
            return True

    except KeyboardInterrupt:
        print("\n👋 Configuration cancelled")
        return True

def configure_provider(handler: 'VisionFallbackHandler') -> bool:
    """Interactive configuration for any provider/model pair."""
    print("\n🔧 Configure Vision Provider")
    print("-" * 30)
    try:
        result = _prompt_provider_and_model()
        if not result:
            return True
        provider, model = result
        success = handler.set_vision_provider(provider, model)
        if success:
            print(f"✅ Vision provider configured: {provider}/{model}")
        else:
            print(f"❌ Failed to configure {provider}/{model}")
    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid input or cancelled")

    return True


def configure_local_provider(handler: 'VisionFallbackHandler') -> bool:
    """Legacy wrapper (kept for backward compatibility)."""
    return configure_provider(handler)


def configure_cloud_provider(handler: 'VisionFallbackHandler') -> bool:
    """Legacy wrapper (kept for backward compatibility)."""
    return configure_provider(handler)

def configure_download_model(handler: 'VisionFallbackHandler') -> bool:
    """Interactive configuration for downloading models."""
    print("\n📥 Download Vision Model")
    print("-" * 30)

    models = ["blip-base-caption", "git-base", "vit-gpt2"]
    print("Available models for download:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    try:
        model_choice = input("Choose model (1-3): ").strip()
        model_idx = int(model_choice) - 1

        if model_idx < 0 or model_idx >= len(models):
            print("❌ Invalid model choice")
            return True

        model = models[model_idx]
        return handle_download_vision_model(handler, model)

    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid input or cancelled")

    return True

def detect_provider_from_model(model: str) -> Optional[str]:
    """Try to detect provider from model name patterns."""
    model_lower = model.lower()

    # Common model name patterns
    if any(pattern in model_lower for pattern in ['qwen2.5vl', 'llama3.2-vision', 'granite']):
        return "ollama"
    elif any(pattern in model_lower for pattern in ['gpt-', 'o1-']):
        return "openai"
    elif any(pattern in model_lower for pattern in ['claude-']):
        return "anthropic"
    elif '/' in model and any(pattern in model_lower for pattern in ['unsloth', 'gguf']):
        return "huggingface"
    elif '/' in model:
        return "lmstudio"

    return None

def add_vision_arguments(parser: argparse.ArgumentParser):
    """Add vision-related arguments to argument parser."""
    vision_group = parser.add_argument_group('vision configuration')

    vision_group.add_argument('--set-vision-caption', metavar='MODEL',
                             help='Set vision caption model (auto-detects provider)')
    vision_group.add_argument('--set-vision-provider', nargs=2, metavar=('PROVIDER', 'MODEL'),
                             help='Set vision provider and model explicitly')
    vision_group.add_argument('--vision-status', action='store_true',
                             help='Show current vision configuration status')
    vision_group.add_argument('--list-vision', action='store_true',
                             help='List available vision configuration options')
    vision_group.add_argument('--download-vision-model', nargs='?', const=True, metavar='MODEL',
                             help='Download vision model for offline use (default: blip-base-caption)')
    vision_group.add_argument('--configure', choices=['vision'],
                             help='Interactive configuration mode')
