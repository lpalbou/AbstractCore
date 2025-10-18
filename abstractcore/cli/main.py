#!/usr/bin/env python3
"""
AbstractCore CLI - Unified Configuration System

Provides configuration commands for all AbstractCore settings:
- Default models and providers
- Vision fallback configuration
- Embeddings settings
- API keys and authentication
- Provider preferences

Usage:
    # General configuration
    abstractcore --set-default-model ollama/llama3:8b
    abstractcore --set-default-provider ollama
    abstractcore --status
    abstractcore --configure

    # Vision configuration
    abstractcore --set-vision-caption qwen2.5vl:7b
    abstractcore --set-vision-provider ollama --model qwen2.5vl:7b

    # Embeddings configuration
    abstractcore --set-embeddings-model sentence-transformers/all-MiniLM-L6-v2
    abstractcore --set-embeddings-provider huggingface

    # API keys
    abstractcore --set-api-key openai sk-...
    abstractcore --set-api-key anthropic ant_...
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abstractcore.config import get_config_manager

def download_vision_model(model_name: str = "blip-base-caption") -> bool:
    """Download a vision model for local use."""
    AVAILABLE_MODELS = {
        "blip-base-caption": {
            "hf_id": "Salesforce/blip-image-captioning-base",
            "size": "990MB",
            "description": "BLIP base image captioning model"
        },
        "blip-large-caption": {
            "hf_id": "Salesforce/blip-image-captioning-large",
            "size": "1.8GB",
            "description": "BLIP large image captioning model (better quality)"
        },
        "vit-gpt2": {
            "hf_id": "nlpconnect/vit-gpt2-image-captioning",
            "size": "500MB",
            "description": "ViT + GPT-2 image captioning model (CPU friendly)"
        },
        "git-base": {
            "hf_id": "microsoft/git-base",
            "size": "400MB",
            "description": "Microsoft GIT base captioning model (smallest)"
        }
    }

    if model_name not in AVAILABLE_MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False

    model_info = AVAILABLE_MODELS[model_name]
    print(f"📋 Model: {model_info['description']} ({model_info['size']})")

    try:
        # Check if transformers is available
        try:
            import transformers
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
            from transformers import GitProcessor, GitForCausalLM
        except ImportError:
            print("❌ Required libraries not found. Installing transformers...")
            import subprocess
            import sys

            # Install transformers and dependencies
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "torchvision", "Pillow"])
            print("✅ Installed transformers and dependencies")

            # Re-import after installation
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
            from transformers import GitProcessor, GitForCausalLM

        # Create models directory
        from pathlib import Path
        models_dir = Path.home() / ".abstractcore" / "models" / model_name
        models_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Download path: {models_dir}")
        print(f"🔄 Downloading {model_info['description']}...")

        hf_id = model_info["hf_id"]

        # Download based on model type
        if "blip" in model_name:
            print("📥 Downloading BLIP model and processor...")
            processor = BlipProcessor.from_pretrained(hf_id, cache_dir=str(models_dir))
            model = BlipForConditionalGeneration.from_pretrained(hf_id, cache_dir=str(models_dir))

            # Save to specific directory structure
            processor.save_pretrained(models_dir / "processor")
            model.save_pretrained(models_dir / "model")

        elif "vit-gpt2" in model_name:
            print("📥 Downloading ViT-GPT2 model...")
            model = VisionEncoderDecoderModel.from_pretrained(hf_id, cache_dir=str(models_dir))
            feature_extractor = ViTImageProcessor.from_pretrained(hf_id, cache_dir=str(models_dir))
            tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(models_dir))

            # Save components
            model.save_pretrained(models_dir / "model")
            feature_extractor.save_pretrained(models_dir / "feature_extractor")
            tokenizer.save_pretrained(models_dir / "tokenizer")

        elif "git" in model_name:
            print("📥 Downloading GIT model...")
            processor = GitProcessor.from_pretrained(hf_id, cache_dir=str(models_dir))
            model = GitForCausalLM.from_pretrained(hf_id, cache_dir=str(models_dir))

            processor.save_pretrained(models_dir / "processor")
            model.save_pretrained(models_dir / "model")

        # Create a marker file to indicate successful download
        marker_file = models_dir / "download_complete.txt"
        with open(marker_file, 'w') as f:
            f.write(f"Model: {model_info['description']}\n")
            f.write(f"HuggingFace ID: {hf_id}\n")
            f.write(f"Downloaded: {Path(__file__).parent}\n")

        print(f"✅ Successfully downloaded {model_info['description']}")
        print(f"📁 Model saved to: {models_dir}")

        # Configure AbstractCore to use this model
        from abstractcore.config import get_config_manager
        config_manager = get_config_manager()
        config_manager.set_vision_caption(f"local/{model_name}")

        print(f"✅ Configured AbstractCore to use local model: {model_name}")
        print(f"🎯 Vision fallback is now enabled!")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_arguments(parser: argparse.ArgumentParser):
    """Add all AbstractCore configuration arguments."""

    # General configuration
    parser.add_argument("--status", action="store_true",
                       help="Show current AbstractCore configuration status")
    parser.add_argument("--configure", action="store_true",
                       help="Interactive configuration setup")
    parser.add_argument("--reset", action="store_true",
                       help="Reset all configuration to defaults")

    # Default model settings
    parser.add_argument("--set-default-model", metavar="MODEL",
                       help="Set default model (format: provider/model or model)")
    parser.add_argument("--set-default-provider", metavar="PROVIDER",
                       help="Set default provider")
    parser.add_argument("--set-chat-model", metavar="MODEL",
                       help="Set default chat model")
    parser.add_argument("--set-code-model", metavar="MODEL",
                       help="Set default code model")

    # Vision configuration
    parser.add_argument("--set-vision-caption", metavar="MODEL",
                       help="Set vision caption model (format: provider/model or model)")
    parser.add_argument("--set-vision-provider", nargs=2, metavar=("PROVIDER", "MODEL"),
                       help="Set vision provider and model explicitly")
    parser.add_argument("--add-vision-fallback", nargs=2, metavar=("PROVIDER", "MODEL"),
                       help="Add fallback provider/model to vision chain")
    parser.add_argument("--disable-vision", action="store_true",
                       help="Disable vision fallback")
    parser.add_argument("--download-vision-model", nargs="?", const="blip-base-caption", metavar="MODEL",
                       help="Download local vision model (default: blip-base-caption, 990MB)")

    # Embeddings configuration
    parser.add_argument("--set-embeddings-model", metavar="MODEL",
                       help="Set embeddings model (format: provider/model or model)")
    parser.add_argument("--set-embeddings-provider", nargs="?", const=True, metavar="PROVIDER",
                       help="Set embeddings provider")

    # API keys
    parser.add_argument("--set-api-key", nargs=2, metavar=("PROVIDER", "KEY"),
                       help="Set API key for provider")
    parser.add_argument("--list-api-keys", action="store_true",
                       help="List API key status for all providers")

def print_status():
    """Print comprehensive configuration status."""
    config_manager = get_config_manager()
    status = config_manager.get_status()

    print("📋 AbstractCore Configuration Status")
    print("=" * 60)

    # Default models
    print("\n🎯 Default Models:")
    defaults = status["defaults"]
    if defaults["provider"] and defaults["model"]:
        print(f"   Primary: {defaults['provider']}/{defaults['model']}")
    else:
        print(f"   Primary: {defaults['model'] or '❌ Not set'}")
    print(f"   Provider: {defaults['provider'] or '❌ Not set'}")
    print(f"   Chat: {defaults['chat_model'] or '❌ Not set'}")
    print(f"   Code: {defaults['code_model'] or '❌ Not set'}")

    # Vision configuration
    print("\n👁️  Vision Fallback:")
    vision = status["vision"]
    print(f"   Strategy: {vision['strategy']}")
    print(f"   Status: {vision['status']}")
    if vision["caption_provider"] and vision["caption_model"]:
        print(f"   Primary: {vision['caption_provider']}/{vision['caption_model']}")
    if vision["fallback_chain_length"] > 0:
        print(f"   Fallback chain: {vision['fallback_chain_length']} entries")

    # Embeddings
    print("\n🔗 Embeddings:")
    embeddings = status["embeddings"]
    print(f"   Status: {embeddings['status']}")
    if embeddings["provider"] and embeddings["model"]:
        print(f"   Model: {embeddings['provider']}/{embeddings['model']}")

    # API Keys
    print("\n🔑 API Keys:")
    for provider, status_text in status["api_keys"].items():
        print(f"   {provider}: {status_text}")

    print(f"\n📁 Config file: {status['config_file']}")

def interactive_configure():
    """Interactive configuration setup."""
    config_manager = get_config_manager()

    print("🚀 AbstractCore Interactive Configuration")
    print("=" * 50)

    # Ask about default model
    print("\n1. Default Model Setup")
    default_choice = input("Set a default model? [y/N]: ").lower().strip()
    if default_choice == 'y':
        model = input("Enter model (provider/model format): ").strip()
        if model:
            config_manager.set_default_model(model)
            print(f"✅ Set default model to: {model}")

    # Ask about vision
    print("\n2. Vision Fallback Setup")
    vision_choice = input("Configure vision fallback for text-only models? [y/N]: ").lower().strip()
    if vision_choice == 'y':
        print("Choose vision setup method:")
        print("  1. Use existing Ollama model (e.g., qwen2.5vl:7b)")
        print("  2. Use cloud API (OpenAI/Anthropic)")
        print("  3. Download local model (coming soon)")

        method = input("Choice [1-3]: ").strip()
        if method == "1":
            model = input("Enter Ollama model name: ").strip()
            if model:
                config_manager.set_vision_caption(model)
                print(f"✅ Set vision model to: {model}")
        elif method == "2":
            provider = input("Enter provider (openai/anthropic): ").strip()
            model = input("Enter model name: ").strip()
            if provider and model:
                config_manager.set_vision_provider(provider, model)
                print(f"✅ Set vision to: {provider}/{model}")

    # Ask about API keys
    print("\n3. API Keys Setup")
    api_choice = input("Configure API keys? [y/N]: ").lower().strip()
    if api_choice == 'y':
        for provider in ["openai", "anthropic", "google"]:
            key = input(f"Enter {provider} API key (or press Enter to skip): ").strip()
            if key:
                config_manager.set_api_key(provider, key)
                print(f"✅ Set {provider} API key")

    print("\n✅ Configuration complete! Run 'abstractcore --status' to see current settings.")

def handle_commands(args) -> bool:
    """Handle AbstractCore configuration commands."""
    config_manager = get_config_manager()
    handled = False

    # Status and configuration
    if args.status:
        print_status()
        handled = True

    if args.configure:
        interactive_configure()
        handled = True

    if args.reset:
        config_manager.reset_configuration()
        print("✅ Configuration reset to defaults")
        handled = True

    # Default model settings
    if args.set_default_model:
        config_manager.set_default_model(args.set_default_model)
        print(f"✅ Set default model to: {args.set_default_model}")
        handled = True

    if args.set_default_provider:
        config_manager.set_default_provider(args.set_default_provider)
        print(f"✅ Set default provider to: {args.set_default_provider}")
        handled = True

    if args.set_chat_model:
        config_manager.set_chat_model(args.set_chat_model)
        print(f"✅ Set chat model to: {args.set_chat_model}")
        handled = True

    if args.set_code_model:
        config_manager.set_code_model(args.set_code_model)
        print(f"✅ Set code model to: {args.set_code_model}")
        handled = True

    # Vision configuration
    if args.set_vision_caption:
        config_manager.set_vision_caption(args.set_vision_caption)
        print(f"✅ Set vision caption model to: {args.set_vision_caption}")
        handled = True

    if args.set_vision_provider:
        provider, model = args.set_vision_provider
        config_manager.set_vision_provider(provider, model)
        print(f"✅ Set vision provider to: {provider}/{model}")
        handled = True

    if args.add_vision_fallback:
        provider, model = args.add_vision_fallback
        config_manager.add_vision_fallback(provider, model)
        print(f"✅ Added vision fallback: {provider}/{model}")
        handled = True

    if args.disable_vision:
        config_manager.disable_vision()
        print("✅ Disabled vision fallback")
        handled = True

    if args.download_vision_model:
        print(f"📥 Starting download of vision model: {args.download_vision_model}")
        success = download_vision_model(args.download_vision_model)
        if success:
            print(f"✅ Successfully downloaded and configured: {args.download_vision_model}")
        else:
            print(f"❌ Failed to download: {args.download_vision_model}")
        handled = True

    # Embeddings configuration
    if args.set_embeddings_model:
        config_manager.set_embeddings_model(args.set_embeddings_model)
        print(f"✅ Set embeddings model to: {args.set_embeddings_model}")
        handled = True

    if args.set_embeddings_provider:
        if isinstance(args.set_embeddings_provider, str):
            config_manager.set_embeddings_provider(args.set_embeddings_provider)
            print(f"✅ Set embeddings provider to: {args.set_embeddings_provider}")
        handled = True

    # API keys
    if args.set_api_key:
        provider, key = args.set_api_key
        config_manager.set_api_key(provider, key)
        print(f"✅ Set API key for: {provider}")
        handled = True

    if args.list_api_keys:
        status = config_manager.get_status()
        print("🔑 API Key Status:")
        for provider, status_text in status["api_keys"].items():
            print(f"   {provider}: {status_text}")
        handled = True

    return handled

def main(argv: List[str] = None):
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="abstractcore",
        description="AbstractCore Unified Configuration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  abstractcore --status                           # Show current configuration
  abstractcore --configure                       # Interactive setup
  abstractcore --set-default-model ollama/llama3:8b
  abstractcore --set-vision-caption qwen2.5vl:7b
  abstractcore --set-api-key openai sk-...

The configuration system enables:
- Default model settings for consistent behavior
- Vision fallback for text-only models processing images
- Embeddings configuration for semantic search
- API key management for cloud providers
        """
    )

    add_arguments(parser)
    args = parser.parse_args(argv)

    try:
        # Handle configuration commands
        if handle_commands(args):
            return 0

        # If no commands were handled, show help
        parser.print_help()
        return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())