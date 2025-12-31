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

# Import config manager with fallback
try:
    from abstractcore.config import get_config_manager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    get_config_manager = None

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
            processor = BlipProcessor.from_pretrained(hf_id, use_fast=False, cache_dir=str(models_dir))
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
        if CONFIG_AVAILABLE:
            config_manager = get_config_manager()
            # Use the proper HuggingFace model identifier
            config_manager.set_vision_provider("huggingface", hf_id)
        else:
            print("⚠️  Config system not available - manual configuration required")

        print(f"✅ Configured AbstractCore to use HuggingFace model: {hf_id}")
        print(f"🎯 Vision fallback is now enabled!")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_arguments(parser: argparse.ArgumentParser):
    """Add all AbstractCore configuration arguments with organized groups."""

    # General configuration group
    general_group = parser.add_argument_group('General Configuration')
    general_group.add_argument("--status", action="store_true",
                              help="Show current configuration status with change commands")
    general_group.add_argument("--configure", action="store_true",
                              help="Interactive guided setup for first-time users")
    general_group.add_argument("--reset", action="store_true",
                              help="Reset all configuration to built-in defaults")

    # Model configuration group
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--set-global-default", metavar="PROVIDER/MODEL",
                            help="Set fallback model for all apps (e.g., ollama/llama3:8b)")
    model_group.add_argument("--set-app-default", nargs=3, metavar=("APP", "PROVIDER", "MODEL"),
                            help="Set app-specific model (apps: cli, summarizer, extractor, judge)")
    model_group.add_argument("--set-chat-model", metavar="PROVIDER/MODEL",
                            help="Set specialized chat model (optional)")
    model_group.add_argument("--set-code-model", metavar="PROVIDER/MODEL",
                            help="Set specialized coding model (optional)")

    # Authentication group
    auth_group = parser.add_argument_group('Authentication')
    auth_group.add_argument("--set-api-key", nargs=2, metavar=("PROVIDER", "KEY"),
                           help="Set API key for cloud providers (openai, anthropic, google, etc.)")
    auth_group.add_argument("--list-api-keys", action="store_true",
                           help="Show which providers have API keys configured")

    # Media processing group
    media_group = parser.add_argument_group('Media & Vision Configuration')
    media_group.add_argument("--set-vision-provider", nargs=2, metavar=("PROVIDER", "MODEL"),
                            help="Set vision model for image analysis with text-only models")
    media_group.add_argument("--add-vision-fallback", nargs=2, metavar=("PROVIDER", "MODEL"),
                            help="Add backup vision provider to fallback chain")
    media_group.add_argument("--download-vision-model", nargs="?", const="blip-base-caption", metavar="MODEL",
                            help="Download local vision model (default: blip-base-caption, ~1GB)")
    media_group.add_argument("--disable-vision", action="store_true",
                            help="Disable vision fallback for text-only models")

    # Embeddings group
    embed_group = parser.add_argument_group('Embeddings Configuration')
    embed_group.add_argument("--set-embeddings-model", metavar="MODEL",
                            help="Set model for semantic search (format: provider/model)")
    embed_group.add_argument("--set-embeddings-provider", nargs="?", const=True, metavar="PROVIDER",
                            help="Set embeddings provider (huggingface, openai, etc.)")

    # Legacy compatibility (hidden in advanced section)
    legacy_group = parser.add_argument_group('Legacy Options')
    legacy_group.add_argument("--set-default-model", metavar="MODEL",
                             help="Set global default model (use --set-global-default instead)")
    legacy_group.add_argument("--set-default-provider", metavar="PROVIDER",
                             help="Set default provider only (use --set-global-default instead)")
    legacy_group.add_argument("--set-vision-caption", metavar="MODEL",
                             help="DEPRECATED: Use --set-vision-provider instead")

    # Storage and logging group
    storage_group = parser.add_argument_group('Storage & Logging')
    storage_group.add_argument("--set-default-cache-dir", metavar="PATH",
                              help="Set default cache directory for models and data")
    storage_group.add_argument("--set-huggingface-cache-dir", metavar="PATH",
                              help="Set HuggingFace models cache directory")
    storage_group.add_argument("--set-local-models-cache-dir", metavar="PATH",
                              help="Set local vision/embedding models cache directory")
    storage_group.add_argument("--set-log-base-dir", metavar="PATH",
                              help="Set directory for log files")

    # Logging control group
    logging_group = parser.add_argument_group('Logging Control')
    logging_group.add_argument("--set-console-log-level", metavar="LEVEL",
                              choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
                              help="Set console logging level (default: WARNING)")
    logging_group.add_argument("--set-file-log-level", metavar="LEVEL",
                              choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"],
                              help="Set file logging level (default: DEBUG)")
    logging_group.add_argument("--enable-debug-logging", action="store_true",
                              help="Enable debug logging for both console and file")
    logging_group.add_argument("--disable-console-logging", action="store_true",
                              help="Disable all console logging output")
    logging_group.add_argument("--enable-file-logging", action="store_true",
                              help="Enable saving logs to files")
    logging_group.add_argument("--disable-file-logging", action="store_true",
                              help="Disable file logging")

    # Streaming configuration group
    streaming_group = parser.add_argument_group('Streaming Configuration')
    streaming_group.add_argument("--stream", choices=["on", "off"],
                                 help="Set default streaming behavior for CLI (on/off)")
    streaming_group.add_argument("--enable-streaming", action="store_true",
                                help="Enable streaming by default for CLI")
    streaming_group.add_argument("--disable-streaming", action="store_true",
                                help="Disable streaming by default for CLI")

    # Timeout configuration group
    timeout_group = parser.add_argument_group('Timeout Configuration')
    timeout_group.add_argument("--set-default-timeout", type=float, metavar="SECONDS",
                              help="Set default HTTP request timeout in seconds (default: 7200 = 2 hours)")
    timeout_group.add_argument("--set-tool-timeout", type=float, metavar="SECONDS",
                              help="Set tool execution timeout in seconds (default: 600 = 10 minutes)")

def print_status():
    """Print comprehensive configuration status with improved readability."""
    if not CONFIG_AVAILABLE or get_config_manager is None:
        print("❌ Configuration system not available")
        print("💡 The AbstractCore configuration module is missing")
        return
    
    config_manager = get_config_manager()
    status = config_manager.get_status()

    # Header with clear context
    print("📋 AbstractCore Default Configuration Status")
    print("   (Explicit parameters in commands override these defaults)")
    print("=" * 75)

    # ESSENTIAL SECTION - What users care about most
    print("\n┌─ ESSENTIAL CONFIGURATION")
    print("│")

    # App defaults with improved formatting
    print("│  🎯 Application Defaults")
    app_defaults = status["app_defaults"]

    apps = [
        ("CLI (utils)", app_defaults["cli"]),
        ("Summarizer", app_defaults["summarizer"]),
        ("Extractor", app_defaults["extractor"]),
        ("Judge", app_defaults["judge"]),
        ("Intent", app_defaults["intent"])
    ]

    for app_name, app_info in apps:
        status_icon = "✅" if app_info["provider"] and app_info["model"] else "⚠️"
        model_text = f"{app_info['provider']}/{app_info['model']}" if app_info["provider"] and app_info["model"] else "Using global fallback"
        print(f"│     {status_icon} {app_name:<12} {model_text}")

    # Global fallback
    print("│")
    print("│  🌐 Global Fallback")
    defaults = status["global_defaults"]
    if defaults["provider"] and defaults["model"]:
        print(f"│     ✅ Default         {defaults['provider']}/{defaults['model']}")
    else:
        print(f"│     ⚠️  Default         Using built-in default (huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF)")

    # Show specialized models if set
    chat_model = defaults['chat_model']
    code_model = defaults['code_model']
    if chat_model or code_model:
        print("│     ┌─ Specialized Models")
        if chat_model:
            print(f"│     │  💬 Chat          {chat_model}")
        if code_model:
            print(f"│     │  💻 Code          {code_model}")

    # API Keys status (simplified)
    print("│")
    print("│  🔑 Provider Access")
    api_keys = status["api_keys"]
    configured_keys = [provider for provider, status_text in api_keys.items() if "✅" in status_text]
    missing_keys = [provider for provider, status_text in api_keys.items() if "❌" in status_text]

    if configured_keys:
        print(f"│     ✅ Configured       {', '.join(configured_keys)}")
    if missing_keys:
        print(f"│     ⚠️  Missing keys     {', '.join(missing_keys)}")

    print("└─")

    # SECONDARY SECTION - Important but less frequently changed
    print("\n┌─ SECONDARY CONFIGURATION")
    print("│")

    # Vision with user-friendly descriptions
    print("│  👁️  Media Processing")
    vision = status["vision"]
    strategy_desc = {
        "two_stage": "Smart captioning for text-only models",
        "disabled": "Media processing disabled",
        "basic_metadata": "Basic metadata extraction only"
    }
    vision_status = "✅ Ready" if "✅" in vision['status'] else "⚠️ Not configured"
    strategy_text = strategy_desc.get(vision['strategy'], vision['strategy'])
    print(f"│     {vision_status:<12} {strategy_text}")
    if vision["caption_provider"] and vision["caption_model"]:
        print(f"│     📷 Vision Model     {vision['caption_provider']}/{vision['caption_model']}")

    # Embeddings
    print("│")
    print("│  🔗 Embeddings")
    embeddings = status["embeddings"]
    emb_status = "✅ Ready" if "✅" in embeddings['status'] else "⚠️ Not configured"
    print(f"│     {emb_status:<12} {embeddings['provider']}/{embeddings['model']}")

    # Streaming configuration
    print("│")
    print("│  🌊 Streaming")
    streaming = status["streaming"]
    stream_status = "✅ Enabled" if streaming['cli_stream_default'] else "⚠️ Disabled"
    stream_desc = "Real-time response display by default" if streaming['cli_stream_default'] else "Complete response display by default"
    print(f"│     {stream_status:<12} {stream_desc}")

    print("└─")

    # ADVANCED SECTION - System-level settings
    print("\n┌─ ADVANCED CONFIGURATION")
    print("│")

    # Logging with dual system display
    print("│  📝 Logging")
    logging_info = status["logging"]

    console_level = logging_info['console_level']
    file_level = logging_info['file_level']
    file_enabled = logging_info['file_logging_enabled']

    # Console logging status
    console_status = "✅" if console_level not in ["NONE", "CRITICAL"] else "❌"
    print(f"│     {console_status} Console        {console_level}")

    # File logging status
    if file_enabled:
        file_status = "✅"
        print(f"│     {file_status} File           {file_level}")
    else:
        file_status = "❌"
        print(f"│     {file_status} File           Disabled")

    # Overall summary
    if console_level == "NONE" and not file_enabled:
        overall_desc = "No logging output"
    elif console_level == "DEBUG" and file_enabled:
        overall_desc = "Full debug logging enabled"
    elif file_enabled:
        overall_desc = "Dual logging active"
    else:
        overall_desc = "Console logging only"

    print(f"│     📊 Summary        {overall_desc}")

    # Cache (simplified)
    print("│")
    print("│  💾 Storage")
    cache = status["cache"]
    print(f"│     ✅ Configured      Cache: {cache['default_cache_dir']}")

    # Timeouts
    print("│")
    print("│  ⏱️  Timeouts")
    timeouts = status["timeouts"]
    default_timeout = timeouts['default_timeout']
    tool_timeout = timeouts['tool_timeout']

    # Format timeout values for display (convert seconds to minutes if >= 60)
    def format_timeout(seconds):
        if seconds >= 60:
            minutes = seconds / 60
            if minutes == int(minutes):
                return f"{int(minutes)}m"
            else:
                return f"{minutes:.1f}m"
        else:
            return f"{int(seconds)}s"

    default_timeout_str = format_timeout(default_timeout)
    tool_timeout_str = format_timeout(tool_timeout)

    print(f"│     ⏱️  HTTP Requests   {default_timeout_str} ({default_timeout}s)")
    print(f"│     🔧 Tool Execution  {tool_timeout_str} ({tool_timeout}s)")

    print("└─")

    # HELP SECTION - Separate actionable commands
    print("\n┌─ QUICK CONFIGURATION COMMANDS")
    print("│")
    print("│  🚀 Common Tasks")
    print("│     abstractcore --set-global-default PROVIDER MODEL")
    print("│     abstractcore --set-app-default APPNAME PROVIDER MODEL")
    print("│     abstractcore --set-api-key PROVIDER YOUR_KEY")
    print("│")
    print("│  🔧 Media & Behavior")
    print("│     abstractcore --set-vision-provider PROVIDER MODEL")
    print("│     abstractcore --download-vision-model  (local models)")
    print("│     abstractcore --stream on/off")
    print("│     abstractcore --enable-streaming / --disable-streaming")
    print("│")
    print("│  📊 Logging & Storage")
    print("│     abstractcore --enable-debug-logging")
    print("│     abstractcore --set-console-log-level LEVEL")
    print("│     abstractcore --set-file-log-level LEVEL")
    print("│     abstractcore --enable-file-logging / --disable-file-logging")
    print("│     abstractcore --set-default-cache-dir PATH")
    print("│")
    print("│  ⏱️  Performance & Timeouts")
    print("│     abstractcore --set-default-timeout SECONDS  (HTTP requests, default: 7200)")
    print("│     abstractcore --set-tool-timeout SECONDS  (Tool execution, default: 600)")
    print("│")
    print("│  🎯 Specialized Models")
    print("│     abstractcore --set-chat-model PROVIDER/MODEL")
    print("│     abstractcore --set-code-model PROVIDER/MODEL")
    print("│     abstractcore --set-embeddings-model PROVIDER/MODEL")
    print("│")
    print("│  🎛️  Advanced")
    print("│     abstractcore --configure  (interactive setup)")
    print("│     abstractcore --reset  (reset to defaults)")
    print("│     abstractcore --list-api-keys  (check API status)")
    print("│")
    print("│  📖 More Help")
    print("│     abstractcore --help")
    print("│     docs/centralized-config.md")
    print("└─")

    print(f"\n📁 Configuration file: {status['config_file']}")

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
    if not CONFIG_AVAILABLE or get_config_manager is None:
        print("❌ Error: Configuration system not available")
        print("💡 The AbstractCore configuration module is missing or not properly installed")
        print("💡 Please reinstall AbstractCore or check your installation")
        return True
    
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

    # Global default model settings
    if args.set_global_default:
        config_manager.set_global_default_model(args.set_global_default)
        print(f"✅ Set global default to: {args.set_global_default}")
        handled = True

    if args.set_default_model:  # Legacy compatibility
        config_manager.set_global_default_model(args.set_default_model)
        print(f"✅ Set global default to: {args.set_default_model}")
        handled = True

    if args.set_default_provider:
        config_manager.set_global_default_provider(args.set_default_provider)
        print(f"✅ Set global default provider to: {args.set_default_provider}")
        handled = True

    # App-specific defaults
    if args.set_app_default:
        app, provider, model = args.set_app_default
        try:
            config_manager.set_app_default(app, provider, model)
            print(f"✅ Set {app} default to: {provider}/{model}")
        except ValueError as e:
            print(f"❌ Error: {e}")
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
        print("⚠️  WARNING: --set-vision-caption is deprecated")
        print("💡 Use instead: abstractcore --set-vision-provider PROVIDER MODEL")
        print("   This provides clearer, more reliable configuration")
        print()
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

    # Cache configuration
    if args.set_default_cache_dir:
        config_manager.set_default_cache_dir(args.set_default_cache_dir)
        print(f"✅ Set default cache directory to: {args.set_default_cache_dir}")
        handled = True

    if args.set_huggingface_cache_dir:
        config_manager.set_huggingface_cache_dir(args.set_huggingface_cache_dir)
        print(f"✅ Set HuggingFace cache directory to: {args.set_huggingface_cache_dir}")
        handled = True

    if args.set_local_models_cache_dir:
        config_manager.set_local_models_cache_dir(args.set_local_models_cache_dir)
        print(f"✅ Set local models cache directory to: {args.set_local_models_cache_dir}")
        handled = True

    # Logging configuration
    if args.set_console_log_level:
        config_manager.set_console_log_level(args.set_console_log_level)
        print(f"✅ Set console log level to: {args.set_console_log_level}")
        handled = True

    if args.set_file_log_level:
        config_manager.set_file_log_level(args.set_file_log_level)
        print(f"✅ Set file log level to: {args.set_file_log_level}")
        handled = True

    if args.set_log_base_dir:
        config_manager.set_log_base_dir(args.set_log_base_dir)
        print(f"✅ Set log base directory to: {args.set_log_base_dir}")
        handled = True

    if args.enable_debug_logging:
        config_manager.enable_debug_logging()
        print("✅ Enabled debug logging for both console and file")
        handled = True

    if args.disable_console_logging:
        config_manager.disable_console_logging()
        print("✅ Disabled console logging")
        handled = True

    if args.enable_file_logging:
        config_manager.enable_file_logging()
        print("✅ Enabled file logging")
        handled = True

    if args.disable_file_logging:
        config_manager.disable_file_logging()
        print("✅ Disabled file logging")
        handled = True

    # Streaming configuration
    if args.stream:
        enabled = args.stream == "on"
        config_manager.set_streaming_default("cli", enabled)
        status = "enabled" if enabled else "disabled"
        print(f"✅ CLI streaming {status} by default")
        handled = True

    if args.enable_streaming:
        config_manager.enable_cli_streaming()
        print("✅ Enabled CLI streaming by default")
        handled = True

    if args.disable_streaming:
        config_manager.disable_cli_streaming()
        print("✅ Disabled CLI streaming by default")
        handled = True

    # Timeout configuration
    if args.set_default_timeout:
        try:
            config_manager.set_default_timeout(args.set_default_timeout)
            # Format display (show in minutes if >= 60 seconds)
            if args.set_default_timeout >= 60:
                minutes = args.set_default_timeout / 60
                display = f"{minutes:.1f} minutes" if minutes != int(minutes) else f"{int(minutes)} minutes"
            else:
                display = f"{args.set_default_timeout} seconds"
            print(f"✅ Set default HTTP timeout to: {display} ({args.set_default_timeout}s)")
        except ValueError as e:
            print(f"❌ Error: {e}")
        handled = True

    if args.set_tool_timeout:
        try:
            config_manager.set_tool_timeout(args.set_tool_timeout)
            # Format display (show in minutes if >= 60 seconds)
            if args.set_tool_timeout >= 60:
                minutes = args.set_tool_timeout / 60
                display = f"{minutes:.1f} minutes" if minutes != int(minutes) else f"{int(minutes)} minutes"
            else:
                display = f"{args.set_tool_timeout} seconds"
            print(f"✅ Set tool execution timeout to: {display} ({args.set_tool_timeout}s)")
        except ValueError as e:
            print(f"❌ Error: {e}")
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
QUICK START:
  abstractcore --status                           # Show current configuration
  abstractcore --configure                       # Interactive guided setup

COMMON TASKS:
  # Set default model for all apps
  abstractcore --set-global-default ollama llama3:8b

  # Set different models for specific apps
  abstractcore --set-app-default cli lmstudio qwen/qwen3-next-80b
  abstractcore --set-app-default summarizer openai gpt-4o-mini
  abstractcore --set-app-default extractor ollama qwen3:4b-instruct

  # Configure API keys
  abstractcore --set-api-key openai sk-your-key-here
  abstractcore --set-api-key anthropic your-anthropic-key

  # Setup vision for images (with text-only models)
  abstractcore --set-vision-provider ollama qwen2.5vl:7b
  abstractcore --download-vision-model

  # Configure logging
  abstractcore --enable-debug-logging            # Enable debug mode
  abstractcore --set-console-log-level WARNING   # Reduce console output
  abstractcore --enable-file-logging             # Save logs to files

SPECIALIZED MODELS:
  abstractcore --set-chat-model openai/gpt-4o-mini      # For chat applications
  abstractcore --set-code-model anthropic/claude-3-5-sonnet  # For coding tasks

PRIORITY SYSTEM:
  1. Explicit parameters (highest):  summarizer doc.pdf --provider openai --model gpt-4o
  2. App-specific config:           --set-app-default summarizer openai gpt-4o-mini
  3. Global config:                 --set-global-default openai/gpt-4o-mini
  4. Built-in defaults (lowest):    huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF

APPS:
  cli        Interactive CLI (python -m abstractcore.utils.cli)
  summarizer Document summarization (summarizer document.pdf)
  extractor  Entity/relationship extraction (extractor data.txt)
  judge      Text evaluation and scoring (judge essay.md)

TROUBLESHOOTING:
  abstractcore --status                          # Check current settings
  abstractcore --reset                          # Reset to defaults
  abstractcore --list-api-keys                  # Check API key status

  If apps show "no provider/model configured":
  abstractcore --set-global-default ollama llama3:8b

DOCUMENTATION: docs/centralized-config.md
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