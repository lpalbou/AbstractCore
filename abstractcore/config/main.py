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
    abstractcore --configure  # alias: --config

    # Vision fallback (for text-only models)
    abstractcore --set-vision-provider ollama qwen2.5vl:7b
    abstractcore --disable-vision

    # Audio/video defaults (attachments)
    abstractcore --set-audio-strategy auto          # requires: pip install abstractvoice
    abstractcore --set-stt-language fr              # optional STT hint
    abstractcore --set-video-strategy auto          # frames fallback requires ffmpeg
    abstractcore --set-video-max-frames 6

    # Embeddings configuration
    abstractcore --set-embeddings-model sentence-transformers/all-MiniLM-L6-v2
    abstractcore --set-embeddings-provider huggingface

    # API keys
    abstractcore --set-api-key openai sk-...
    abstractcore --set-api-key anthropic ant_...
    abstractcore --set-api-key portkey pk_...
"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
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
    general_group.add_argument(
        "--configure",
        "--config",
        action="store_true",
        help="Interactive guided setup for first-time users (alias: --config)",
    )
    general_group.add_argument("--reset", action="store_true",
                              help="Reset all configuration to built-in defaults")
    general_group.add_argument(
        "--install",
        action="store_true",
        help="Check all subsystems and download/install missing models and dependencies",
    )
    general_group.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-accept all downloads during --install (non-interactive)",
    )

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
                           help="Set API key for cloud providers (openai, anthropic, openrouter, portkey, google, etc.)")
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
    media_group.add_argument(
        "--set-audio-strategy",
        choices=["native_only", "speech_to_text", "auto"],
        help=(
            "Set default audio handling strategy for attachments "
            "(default: auto when `abstractvoice` is installed; otherwise native_only)."
        ),
    )
    media_group.add_argument(
        "--set-stt-backend-id",
        metavar="BACKEND_ID",
        help="Set preferred STT backend id for capability plugins (optional).",
    )
    media_group.add_argument(
        "--set-stt-language",
        metavar="LANG",
        help="Set default STT language hint (optional; e.g. en, fr).",
    )
    media_group.add_argument(
        "--set-video-strategy",
        choices=["native_only", "frames_caption", "auto"],
        help="Set default video handling strategy for attachments (default: auto).",
    )
    media_group.add_argument(
        "--set-video-max-frames",
        metavar="N",
        type=int,
        help="Set max frames sampled for video frame fallback (default: 3).",
    )
    media_group.add_argument(
        "--set-video-max-frames-native",
        metavar="N",
        type=int,
        help="Set max frames used for native video-capable models (default: 8).",
    )
    media_group.add_argument(
        "--set-video-frame-format",
        choices=["jpg", "png"],
        help="Set extracted frame format for video frame fallback (default: jpg).",
    )
    media_group.add_argument(
        "--set-video-sampling-strategy",
        choices=["uniform", "keyframes"],
        help="Set frame sampling strategy for video frame fallback (default: uniform).",
    )
    media_group.add_argument(
        "--set-video-max-frame-side",
        metavar="PX",
        type=int,
        help="Set max side length for extracted video frames (default: 1024).",
    )
    media_group.add_argument(
        "--set-video-max-size-bytes",
        metavar="BYTES",
        type=int,
        help="Set max allowed video size for processing (bytes). Use 0 to clear.",
    )

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
                              help="Set default HTTP request timeout in seconds (default: 7200 = 2 hours; 0 = unlimited)")
    timeout_group.add_argument("--set-tool-timeout", type=float, metavar="SECONDS",
                              help="Set tool execution timeout in seconds (default: 600 = 10 minutes; 0 = unlimited)")

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

    # Audio (STT fallback via capability plugins)
    audio = status.get("audio", {})
    audio_strategy = str(audio.get("strategy") or "native_only").strip()
    abstractvoice_installed = False
    try:
        import importlib.util

        abstractvoice_installed = importlib.util.find_spec("abstractvoice") is not None
    except Exception:
        abstractvoice_installed = False
    audio_desc = {
        "native_only": "Native audio only (errors on text-only models)",
        "speech_to_text": "Speech-to-text fallback (requires `abstractvoice`)",
        "auto": "Native when supported, otherwise STT (requires `abstractvoice`)",
        "caption": "Audio caption fallback (reserved)",
    }
    audio_strategy_norm = audio_strategy.strip().lower()
    if audio_strategy_norm in {"speech_to_text", "auto"} and abstractvoice_installed:
        audio_status = "✅ Enabled"
    elif audio_strategy_norm in {"speech_to_text", "auto"} and not abstractvoice_installed:
        audio_status = "⚠️ Disabled"
    else:
        audio_status = "⚠️ Disabled"
    print(f"│     🎧 Audio           {audio_status:<10} {audio_desc.get(audio_strategy_norm, audio_strategy)}")
    stt_backend_id = audio.get("stt_backend_id")
    stt_language = audio.get("stt_language")
    if stt_backend_id:
        print(f"│     🔎 STT backend     {stt_backend_id}")
    if stt_language:
        print(f"│     🌐 STT language    {stt_language}")

    # Video (native where supported; otherwise frames fallback via ffmpeg)
    video = status.get("video", {})
    video_strategy = str(video.get("strategy") or "auto").strip()
    video_desc = {
        "native_only": "Native video only (errors unless the model supports video input)",
        "frames_caption": "Frames fallback (sample frames via ffmpeg; requires vision handling)",
        "auto": "Native when supported, otherwise frames fallback (ffmpeg)",
    }
    video_status = "✅ Enabled" if video_strategy in {"frames_caption", "auto"} else "⚠️ Strict"
    print(f"│     🎞️  Video           {video_status:<10} {video_desc.get(video_strategy, video_strategy)}")
    max_frames = video.get("max_frames")
    max_frames_native = video.get("max_frames_native")
    frame_format = video.get("frame_format")
    sampling_strategy = video.get("sampling_strategy")
    max_frame_side = video.get("max_frame_side")
    max_video_size_bytes = video.get("max_video_size_bytes")
    details = []
    if max_frames is not None:
        details.append(f"max_frames={max_frames}")
    if max_frames_native is not None:
        details.append(f"max_frames_native={max_frames_native}")
    if sampling_strategy:
        details.append(f"sampling={sampling_strategy}")
    if frame_format:
        details.append(f"format={frame_format}")
    if max_frame_side is not None:
        details.append(f"max_side={max_frame_side}")
    if max_video_size_bytes is not None:
        details.append(f"max_bytes={max_video_size_bytes}")
    if details:
        print(f"│     🖼️  Video frames    {', '.join(details)}")

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
    print("│     abstractcore --set-global-default PROVIDER/MODEL")
    print("│     abstractcore --set-app-default APPNAME PROVIDER MODEL")
    print("│     abstractcore --set-api-key PROVIDER YOUR_KEY")
    print("│")
    print("│  🔧 Media & Behavior")
    print("│     abstractcore --set-vision-provider PROVIDER MODEL")
    print("│     abstractcore --download-vision-model  (local models)")
    print("│     abstractcore --set-audio-strategy {native_only|speech_to_text|auto}")
    print("│     abstractcore --set-stt-language LANG  (optional)")
    print("│     abstractcore --set-video-strategy {native_only|frames_caption|auto}")
    print("│     abstractcore --set-video-max-frames N  (frame fallback budget)")
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
    print("│     abstractcore --set-default-timeout SECONDS  (HTTP requests, default: 7200; 0 = unlimited)")
    print("│     abstractcore --set-tool-timeout SECONDS  (Tool execution, default: 600; 0 = unlimited)")
    print("│")
    print("│  🎯 Specialized Models")
    print("│     abstractcore --set-chat-model PROVIDER/MODEL")
    print("│     abstractcore --set-code-model PROVIDER/MODEL")
    print("│     abstractcore --set-embeddings-model PROVIDER/MODEL")
    print("│")
    print("│  🎛️  Advanced")
    print("│     abstractcore --configure / --config  (interactive setup)")
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

            # Determine the provider from the model string.
            if "/" in model:
                selected_provider = model.split("/", 1)[0].lower()
            else:
                selected_provider = "ollama"

            # For local providers, the base URL matters — ask about it.
            # These are the providers where the server address is user-configurable.
            _LOCAL_PROVIDER_ENV_VARS = {
                "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434"),
                "lmstudio": ("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "vllm": ("VLLM_BASE_URL", "http://localhost:8000/v1"),
                "openai-compatible": ("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1"),
            }
            if selected_provider in _LOCAL_PROVIDER_ENV_VARS:
                env_var, default_url = _LOCAL_PROVIDER_ENV_VARS[selected_provider]
                current_url = os.environ.get(env_var, "")
                if current_url:
                    print(f"   ℹ️  {env_var} is already set to: {current_url}")
                else:
                    print(f"   ℹ️  {selected_provider} uses env var {env_var} (default: {default_url})")
                    base_url = input(f"   Enter base URL for {selected_provider} (or press Enter for default): ").strip()
                    if base_url:
                        os.environ[env_var] = base_url
                        print(f"   ✅ Set {env_var}={base_url} (current session)")
                        print(f"   💡 To make this permanent, add to your shell profile:")
                        print(f'      export {env_var}="{base_url}"')
                    else:
                        print(f"   ✅ Using default: {default_url}")

    # Ask about vision
    print("\n2. Vision Fallback Setup")
    vision_choice = input("Configure vision fallback for text-only models? [y/N]: ").lower().strip()
    if vision_choice == 'y':
        print("Vision fallback supports any provider and any model (local or cloud).")
        print("Examples (non-exhaustive):")
        print("  lmstudio/qwen/qwen2.5-vl-7b, huggingface/Salesforce/blip-image-captioning-base, mlx/<vision-model>")
        print("  openai/gpt-4o, anthropic/claude-3-5-sonnet, openai-compatible/my-vision-model")
        print("Tip: use `abstractcore --download-vision-model` for offline caption models.")

        provider_raw = input("Enter vision provider id (or provider/model): ").strip()
        model = input("Enter vision model name (or leave blank if provider/model): ").strip()
        # Allow provider/model in a single input to keep the prompt flexible.
        if provider_raw and not model and "/" in provider_raw:
            provider, model = provider_raw.split("/", 1)
        else:
            provider = provider_raw
        if provider and model:
            # Keep the vision fallback provider-agnostic to match runtime capabilities.
            config_manager.set_vision_provider(provider, model)
            print(f"✅ Set vision to: {provider}/{model}")
        else:
            print("⚠️  Skipped vision fallback (provider and model are required).")

    # Ask about API keys
    print("\n3. API Keys Setup")
    api_choice = input("Configure API keys? [y/N]: ").lower().strip()
    if api_choice == 'y':
        for provider in ["openai", "anthropic", "openrouter", "portkey", "google"]:
            key = input(f"Enter {provider} API key (or press Enter to skip): ").strip()
            if key:
                config_manager.set_api_key(provider, key)
                print(f"✅ Set {provider} API key")

    # Ask about audio strategy (voice/STT fallback for audio attachments)
    print("\n4. Audio Strategy (voice/STT fallback)")
    print("How should AbstractCore handle audio attachments with text-only models?")
    print("  auto           — native when supported, otherwise STT via abstractvoice (recommended)")
    print("  speech_to_text — always transcribe via abstractvoice")
    print("  native_only    — only models with native audio support (no fallback)")
    audio_choice = input("Audio strategy [auto]: ").strip().lower()
    if not audio_choice:
        audio_choice = "auto"
    if audio_choice in ("native_only", "auto", "speech_to_text"):
        config_manager.set_audio_strategy(audio_choice)
        print(f"✅ Set audio strategy to: {audio_choice}")
        if audio_choice in ("auto", "speech_to_text"):
            print("   💡 Requires: pip install abstractvoice")
    else:
        print("⚠️  Invalid choice; keeping existing audio strategy.")

    # Ask about video strategy (frame fallback for video attachments)
    print("\n5. Video Strategy (frame fallback)")
    print("How should AbstractCore handle video attachments?")
    print("  auto           — native when supported, otherwise sample frames via ffmpeg (recommended)")
    print("  frames_caption — always sample frames via ffmpeg")
    print("  native_only    — only models with native video support (no fallback)")
    video_choice = input("Video strategy [auto]: ").strip().lower()
    if not video_choice:
        video_choice = "auto"
    if video_choice in ("native_only", "auto", "frames_caption"):
        config_manager.set_video_strategy(video_choice)
        print(f"✅ Set video strategy to: {video_choice}")
        if video_choice in ("auto", "frames_caption"):
            print("   💡 Requires: ffmpeg/ffprobe on PATH + a vision-capable model or vision fallback")
    else:
        print("⚠️  Invalid choice; keeping existing video strategy.")

    # Ask about embeddings provider/model
    print("\n6. Embeddings Setup")
    print("Embeddings are used for semantic search, RAG pipelines, and knowledge graph retrieval.")
    print("Supported: huggingface (local), ollama, lmstudio, openai, openrouter, portkey, openai-compatible.")
    emb_choice = input("Configure embeddings provider/model? [y/N]: ").lower().strip()
    if emb_choice == 'y':
        print("Examples:")
        print("  huggingface/all-minilm-l6-v2          (local, lightweight, default)")
        print("  huggingface/BAAI/bge-small-en-v1.5    (local, good quality)")
        print("  ollama/nomic-embed-text               (local, via Ollama server)")
        print("  lmstudio/<embedding-model>            (local, via LMStudio server)")
        print("  openai/text-embedding-3-small         (cloud, requires API key)")
        print("  openrouter/<embedding-model>          (cloud, via OpenRouter gateway)")
        print("  portkey/<embedding-model>             (cloud, via Portkey gateway)")
        print("  openai-compatible/<model>             (any OpenAI-compatible endpoint)")
        emb_model = input("Enter embeddings model (provider/model format, or press Enter to keep default): ").strip()
        if emb_model:
            # Validate the provider before saving
            if "/" in emb_model:
                emb_prov = emb_model.split("/", 1)[0].lower()
            else:
                emb_prov = "huggingface"
            _valid_emb_provs = ("huggingface", "ollama", "lmstudio", "openai",
                                "openrouter", "portkey", "openai-compatible")
            if emb_prov not in _valid_emb_provs:
                print(f"⚠️  Provider '{emb_prov}' is not supported for embeddings.")
                print(f"   Supported: {', '.join(_valid_emb_provs)}")
                print(f"   Keeping current config.")
            else:
                config_manager.set_embeddings_model(emb_model)
                print(f"✅ Set embeddings model to: {emb_model}")
        else:
            print("   ✅ Keeping default: huggingface/all-minilm-l6-v2")

    # Ask about console log verbosity
    print("\n7. Console Logging Verbosity")
    print("Choose console verbosity level:")
    print("  none | error | warning | info | debug")
    level = input("Console log level [error]: ").strip().lower()
    if not level:
        level = "error"
    level_map = {
        "none": "NONE",
        "error": "ERROR",
        "warning": "WARNING",
        "info": "INFO",
        "debug": "DEBUG",
    }
    if level in level_map:
        config_manager.set_console_log_level(level_map[level])
        print(f"✅ Set console log level to: {level_map[level]}")
    else:
        print("⚠️  Invalid level; keeping existing console log level.")

    print("\n✅ Configuration complete! Run 'abstractcore --status' to see current settings.")


# ---------------------------------------------------------------------------
# Preflight: comprehensive readiness check
# ---------------------------------------------------------------------------

def _ask_yes(prompt: str, auto_accept: bool) -> bool:
    """Ask a yes/no question. If *auto_accept*, return True without prompting."""
    if auto_accept:
        print(f"{prompt} → auto-accepted (--yes)")
        return True
    answer = input(f"{prompt} [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def install_check(auto_accept: bool = False) -> None:
    """Check all subsystems and download/install missing models and dependencies.

    For each area, report ✅ (ready), ⚠️ (degraded but functional),
    or ❌ (missing / broken).  When a gap is fixable (download, prefetch),
    offer to fix it — or auto-fix if *auto_accept* is True.
    """
    if not CONFIG_AVAILABLE or get_config_manager is None:
        print("❌ Configuration system not available — cannot run preflight.")
        return

    config_manager = get_config_manager()
    status = config_manager.get_status()

    print("📦 AbstractCore Install Check")
    print("=" * 60)

    total = 0
    passed = 0
    warnings = 0
    failed = 0
    actions: list[str] = []  # summary of actions taken

    def _pass(label: str, detail: str = ""):
        nonlocal total, passed
        total += 1
        passed += 1
        suffix = f" — {detail}" if detail else ""
        print(f"  ✅ {label}{suffix}")

    def _warn(label: str, detail: str = ""):
        nonlocal total, warnings
        total += 1
        warnings += 1
        suffix = f" — {detail}" if detail else ""
        print(f"  ⚠️  {label}{suffix}")

    def _fail(label: str, detail: str = ""):
        nonlocal total, failed
        total += 1
        failed += 1
        suffix = f" — {detail}" if detail else ""
        print(f"  ❌ {label}{suffix}")

    # ------------------------------------------------------------------
    # 1. Default model configured
    # ------------------------------------------------------------------
    print("\n┌─ 1. Default Model")
    defaults = status["global_defaults"]
    if defaults["provider"] and defaults["model"]:
        _pass("Default model", f"{defaults['provider']}/{defaults['model']}")
    else:
        _fail("Default model", "not configured")
        print("     💡 Fix: abstractcore --config  (step 1)")

    # ------------------------------------------------------------------
    # 2. Provider reachable (for local providers, quick HTTP probe)
    # ------------------------------------------------------------------
    print("\n┌─ 2. Provider Connectivity")
    provider = (defaults.get("provider") or "").lower()
    _PROVIDER_HEALTH_URLS = {
        "ollama": (os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), "/"),
        "lmstudio": (os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"), "/models"),
        "vllm": (os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), "/models"),
    }
    if provider in _PROVIDER_HEALTH_URLS:
        base, path = _PROVIDER_HEALTH_URLS[provider]
        url = f"{base.rstrip('/')}{path}"
        try:
            import httpx
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code < 500:
                _pass(f"{provider} reachable", url)
            else:
                _warn(f"{provider} responded {resp.status_code}", url)
        except Exception as exc:
            _fail(f"{provider} unreachable", f"{url} ({type(exc).__name__})")
            print(f"     💡 Ensure your {provider} server is running.")
    elif provider in ("openai", "anthropic", "openrouter", "portkey", "google"):
        # Cloud providers — check API key instead of connectivity
        api_keys = status.get("api_keys", {})
        key_status = api_keys.get(provider, "")
        if "✅" in key_status:
            _pass(f"{provider} API key", "configured")
        else:
            _fail(f"{provider} API key", "missing")
            print(f"     💡 Fix: abstractcore --set-api-key {provider} <YOUR_KEY>")
    elif provider:
        _warn(f"{provider}", "connectivity check not implemented for this provider")
    else:
        _warn("Provider", "no default provider configured")

    # ------------------------------------------------------------------
    # 3. Embeddings model
    # ------------------------------------------------------------------
    print("\n┌─ 3. Embeddings")
    emb_cfg = status.get("embeddings", {})
    emb_model = emb_cfg.get("model", "all-minilm-l6-v2")
    emb_provider = (emb_cfg.get("provider") or "huggingface").lower()

    # Supported embeddings providers (source of truth: EmbeddingManager.__init__ in
    # abstractcore/embeddings/manager.py).
    # Server-based providers: the model is served remotely — no local download needed.
    _SERVER_EMB_PROVIDERS = {"lmstudio", "ollama", "openai", "openrouter", "portkey", "openai-compatible"}

    _SUPPORTED_EMB_PROVIDERS = {"huggingface", "ollama", "lmstudio", "openai",
                                "openrouter", "portkey", "openai-compatible"}

    if emb_provider not in _SUPPORTED_EMB_PROVIDERS:
        _fail("Embeddings provider", f"'{emb_provider}' is not supported for embeddings")
        print(f"     💡 Supported providers: {', '.join(sorted(_SUPPORTED_EMB_PROVIDERS))}")
        print(f"     💡 Fix: abstractcore --set-embeddings-model <provider>/<model>")
        print(f"     💡 Examples: huggingface/all-minilm-l6-v2, ollama/nomic-embed-text, openai/text-embedding-3-small")
    elif emb_provider in _SERVER_EMB_PROVIDERS:
        _pass("Embeddings provider", f"{emb_provider}/{emb_model} (served by {emb_provider})")
        # Quick reachability probe for local servers
        _EMB_SERVER_URLS = {
            "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "lmstudio": os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        }
        if emb_provider in _EMB_SERVER_URLS:
            base = _EMB_SERVER_URLS[emb_provider]
            try:
                import httpx
                resp = httpx.get(f"{base.rstrip('/')}/models", timeout=3.0)
                if resp.status_code < 500:
                    _pass(f"Embeddings server ({emb_provider})", f"reachable at {base}")
                else:
                    _warn(f"Embeddings server ({emb_provider})", f"responded {resp.status_code} at {base}")
            except Exception:
                _warn(f"Embeddings server ({emb_provider})", f"unreachable at {base} — ensure it is running")
        elif emb_provider in ("openai", "openrouter", "portkey"):
            # Cloud provider — check API key
            api_keys = status.get("api_keys", {})
            key_status = api_keys.get(emb_provider, "")
            if "✅" in key_status:
                _pass(f"{emb_provider} API key (for embeddings)", "configured")
            else:
                _warn(f"{emb_provider} API key (for embeddings)", f"not set — required for {emb_provider} embeddings")
                print(f"     💡 Fix: abstractcore --set-api-key {emb_provider} <YOUR_KEY>")
    else:
        # HuggingFace / local provider: needs sentence-transformers + a downloaded model.
        try:
            import importlib.util
            st_available = importlib.util.find_spec("sentence_transformers") is not None
        except Exception:
            st_available = False

        if st_available:
            # Check if the model is cached locally
            model_cached = False
            try:
                cache_dir = Path(os.environ.get(
                    "SENTENCE_TRANSFORMERS_HOME",
                    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"),
                ))
                hub_dir = cache_dir / "hub"
                model_slug = emb_model.replace("/", "--")
                if hub_dir.exists():
                    candidates = list(hub_dir.glob(f"models--*{model_slug}*"))
                    model_cached = len(candidates) > 0
            except Exception:
                pass

            if model_cached:
                _pass("Embeddings model", f"{emb_provider}/{emb_model} (cached)")
            else:
                _warn("Embeddings model", f"{emb_provider}/{emb_model} (not cached — will download on first use)")
                if _ask_yes("     Download embeddings model now?", auto_accept):
                    try:
                        print(f"     ⏳ Downloading {emb_model}...")
                        from sentence_transformers import SentenceTransformer
                        SentenceTransformer(emb_model)
                        _pass("Embeddings model", f"{emb_model} downloaded ✅")
                        actions.append(f"Downloaded embeddings model: {emb_model}")
                    except Exception as exc:
                        _warn("Embeddings download", str(exc))
        else:
            _warn("Embeddings", f"sentence-transformers not installed ({emb_provider}/{emb_model} configured)")
            if _ask_yes('     Install embeddings dependencies now? (pip install "abstractcore[embeddings]")', auto_accept):
                try:
                    print('     ⏳ Running: pip install "abstractcore[embeddings]"')
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "abstractcore[embeddings]"],
                        check=True,
                    )
                    _pass("Embeddings deps", "installed ✅")
                    actions.append("Installed abstractcore[embeddings]")
                    # Now try to download the model too
                    if _ask_yes(f"     Download embeddings model ({emb_model}) now?", auto_accept):
                        try:
                            print(f"     ⏳ Downloading {emb_model}...")
                            from sentence_transformers import SentenceTransformer
                            SentenceTransformer(emb_model)
                            _pass("Embeddings model", f"{emb_model} downloaded ✅")
                            actions.append(f"Downloaded embeddings model: {emb_model}")
                        except Exception as exc:
                            _warn("Embeddings download", str(exc))
                except Exception as exc:
                    _warn("Embeddings install", str(exc))

    # ------------------------------------------------------------------
    # 4. Vision fallback
    # ------------------------------------------------------------------
    print("\n┌─ 4. Vision Fallback")
    vision = status.get("vision", {})
    vision_strategy = vision.get("strategy", "disabled")
    if vision_strategy != "disabled" and vision.get("caption_provider") and vision.get("caption_model"):
        _pass("Vision fallback", f"{vision['caption_provider']}/{vision['caption_model']}")
    elif vision_strategy == "disabled":
        _warn("Vision fallback", "disabled (image input on text-only models will fail)")
        print("     💡 Fix: abstractcore --set-vision-provider <PROVIDER> <MODEL>")
        print("     💡  Or: abstractcore --download-vision-model  (local offline model)")
        if _ask_yes("     Download a local vision model (blip-base-caption, ~1GB)?", auto_accept):
            success = download_vision_model("blip-base-caption")
            if success:
                actions.append("Downloaded local vision model: blip-base-caption")
    else:
        _warn("Vision fallback", f"strategy={vision_strategy} but model not fully configured")

    # ------------------------------------------------------------------
    # 5. Audio / Voice (STT + TTS via abstractvoice)
    # ------------------------------------------------------------------
    print("\n┌─ 5. Voice & Audio")
    try:
        import importlib.util
        av_available = importlib.util.find_spec("abstractvoice") is not None
    except Exception:
        av_available = False

    if av_available:
        _pass("abstractvoice", "installed")

        # Check STT model (faster-whisper)
        stt_ready = False
        try:
            # faster-whisper caches under HF cache; check for a small model
            hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            hub = hf_cache / "hub"
            if hub.exists():
                stt_candidates = list(hub.glob("models--Systran--faster-whisper-*"))
                stt_ready = len(stt_candidates) > 0
        except Exception:
            pass

        if stt_ready:
            _pass("STT model (faster-whisper)", "cached")
        else:
            _warn("STT model (faster-whisper)", "not prefetched")
            if _ask_yes("     Prefetch STT model (small, ~500MB)?", auto_accept):
                try:
                    print("     ⏳ Running: abstractvoice-prefetch --stt small")
                    subprocess.run(
                        [sys.executable, "-m", "abstractvoice", "download", "--stt", "small"],
                        check=False,  # command may exit 0 even on failure
                    )
                    # Re-check filesystem — don't trust exit code alone
                    stt_rechecked = False
                    try:
                        if hub.exists():
                            stt_rechecked = len(list(hub.glob("models--Systran--faster-whisper-*"))) > 0
                    except Exception:
                        pass
                    if stt_rechecked:
                        _pass("STT model", "prefetched ✅")
                        actions.append("Prefetched STT model: small")
                    else:
                        _warn("STT prefetch", "download failed — model not found in HF cache (network issue?)")
                except Exception as exc:
                    _warn("STT prefetch", f"download failed ({exc})")

        # Check TTS model (Piper)
        piper_dir = Path.home() / ".piper" / "models"
        piper_has_models = piper_dir.exists() and any(piper_dir.glob("*.onnx"))
        if piper_has_models:
            _pass("TTS model (Piper)", "voice models found")
        else:
            _warn("TTS model (Piper)", "no voice models prefetched")
            if _ask_yes("     Prefetch Piper English voice (~50MB)?", auto_accept):
                try:
                    print("     ⏳ Running: abstractvoice-prefetch --piper en")
                    subprocess.run(
                        [sys.executable, "-m", "abstractvoice", "download", "--piper", "en"],
                        check=False,  # command may exit 0 even on failure
                    )
                    # Re-check filesystem — don't trust exit code alone
                    piper_has_models = piper_dir.exists() and any(piper_dir.glob("*.onnx"))
                    if piper_has_models:
                        _pass("TTS model", "prefetched ✅")
                        actions.append("Prefetched TTS voice: en")
                    else:
                        _warn("TTS prefetch", "download failed — no .onnx model found in ~/.piper/models/ (network issue?)")
                except Exception as exc:
                    _warn("TTS prefetch", f"download failed ({exc})")
    else:
        _warn("abstractvoice", "not installed (TTS/STT unavailable)")
        print('     💡 Fix: pip install abstractvoice')
        print('     💡 Then: abstractvoice-prefetch --stt small --piper en')

    # Audio strategy
    audio = status.get("audio", {})
    audio_strategy = (audio.get("strategy") or "native_only").strip().lower()
    if audio_strategy in ("auto", "speech_to_text") and av_available:
        _pass("Audio strategy", audio_strategy)
    elif audio_strategy in ("auto", "speech_to_text") and not av_available:
        _fail("Audio strategy", f"{audio_strategy} configured but abstractvoice not installed")
    else:
        _warn("Audio strategy", f"{audio_strategy} (audio attachments limited to native-capable models)")

    # ------------------------------------------------------------------
    # 6. Video (ffmpeg)
    # ------------------------------------------------------------------
    print("\n┌─ 6. Video Processing")
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        _pass("ffmpeg + ffprobe", "found on PATH")
    elif ffmpeg_path:
        _warn("ffprobe", "not found on PATH (video frame extraction may fail)")
    else:
        _warn("ffmpeg", "not found on PATH (video frame fallback unavailable)")
        print("     💡 Install ffmpeg: https://ffmpeg.org/download.html")
        print("     💡 macOS: brew install ffmpeg")
        print("     💡 Ubuntu: sudo apt install ffmpeg")

    video = status.get("video", {})
    video_strategy = (video.get("strategy") or "auto").strip().lower()
    if video_strategy in ("auto", "frames_caption"):
        if ffmpeg_path:
            _pass("Video strategy", video_strategy)
        else:
            _warn("Video strategy", f"{video_strategy} but ffmpeg not available")
    else:
        _warn("Video strategy", f"{video_strategy} (video input limited to native-capable models)")

    # ------------------------------------------------------------------
    # 7. Image generation (abstractvision — optional)
    # ------------------------------------------------------------------
    print("\n┌─ 7. Image Generation (optional)")
    try:
        import importlib.util
        avis_available = importlib.util.find_spec("abstractvision") is not None
    except Exception:
        avis_available = False

    if avis_available:
        _pass("abstractvision", "installed")
    else:
        _warn("abstractvision", "not installed (image generation unavailable)")
        print('     💡 Fix: pip install abstractvision')

    # ------------------------------------------------------------------
    # 8. API keys summary
    # ------------------------------------------------------------------
    print("\n┌─ 8. API Keys")
    api_keys = status.get("api_keys", {})
    has_any_key = False
    for prov, key_status in api_keys.items():
        if "✅" in key_status:
            _pass(f"{prov} key", "configured")
            has_any_key = True
        else:
            _warn(f"{prov} key", "not set")
    if not has_any_key:
        print("     ℹ️  No cloud API keys configured (local-only usage is fine)")
        print("     💡 Fix: abstractcore --set-api-key <provider> <key>")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"📦 Install Summary: {passed} passed, {warnings} warnings, {failed} failed  (total: {total})")

    if actions:
        print("\n📋 Actions taken:")
        for action in actions:
            print(f"   • {action}")

    if failed == 0 and warnings == 0:
        print("\n🟢 All systems ready!")
    elif failed == 0:
        print(f"\n🟡 Functional with {warnings} warning(s). Run 'abstractcore --status' for details.")
    else:
        print(f"\n🔴 {failed} critical issue(s). Fix them and re-run 'abstractcore --install'.")

    print()


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

    if args.install:
        install_check(auto_accept=args.yes)
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

    # Audio configuration (speech-to-text fallback via capability plugins, e.g. abstractvoice)
    if getattr(args, "set_audio_strategy", None):
        ok = config_manager.set_audio_strategy(args.set_audio_strategy)
        if ok:
            print(f"✅ Set audio strategy to: {args.set_audio_strategy}")
        else:
            print(f"❌ Error: Invalid audio strategy: {args.set_audio_strategy}")
        handled = True

    if getattr(args, "set_stt_backend_id", None) is not None:
        ok = config_manager.set_stt_backend_id(args.set_stt_backend_id)
        if ok:
            if args.set_stt_backend_id:
                print(f"✅ Set STT backend id to: {args.set_stt_backend_id}")
            else:
                print("✅ Cleared STT backend id")
        else:
            print("❌ Error: Failed to update STT backend id")
        handled = True

    if getattr(args, "set_stt_language", None) is not None:
        ok = config_manager.set_stt_language(args.set_stt_language)
        if ok:
            if args.set_stt_language:
                print(f"✅ Set STT language to: {args.set_stt_language}")
            else:
                print("✅ Cleared STT language")
        else:
            print("❌ Error: Failed to update STT language")
        handled = True

    # Video configuration (native video where supported; otherwise frames fallback via ffmpeg)
    if getattr(args, "set_video_strategy", None):
        ok = config_manager.set_video_strategy(args.set_video_strategy)
        if ok:
            print(f"✅ Set video strategy to: {args.set_video_strategy}")
        else:
            print(f"❌ Error: Invalid video strategy: {args.set_video_strategy}")
        handled = True

    if getattr(args, "set_video_max_frames", None) is not None:
        ok = config_manager.set_video_max_frames(args.set_video_max_frames)
        if ok:
            print(f"✅ Set video max frames (fallback) to: {args.set_video_max_frames}")
        else:
            print(f"❌ Error: Invalid video max frames: {args.set_video_max_frames}")
        handled = True

    if getattr(args, "set_video_max_frames_native", None) is not None:
        ok = config_manager.set_video_max_frames_native(args.set_video_max_frames_native)
        if ok:
            print(f"✅ Set video max frames (native) to: {args.set_video_max_frames_native}")
        else:
            print(f"❌ Error: Invalid video max frames native: {args.set_video_max_frames_native}")
        handled = True

    if getattr(args, "set_video_frame_format", None):
        ok = config_manager.set_video_frame_format(args.set_video_frame_format)
        if ok:
            print(f"✅ Set video frame format to: {args.set_video_frame_format}")
        else:
            print(f"❌ Error: Invalid video frame format: {args.set_video_frame_format}")
        handled = True

    if getattr(args, "set_video_sampling_strategy", None):
        ok = config_manager.set_video_sampling_strategy(args.set_video_sampling_strategy)
        if ok:
            print(f"✅ Set video sampling strategy to: {args.set_video_sampling_strategy}")
        else:
            print(f"❌ Error: Invalid video sampling strategy: {args.set_video_sampling_strategy}")
        handled = True

    if getattr(args, "set_video_max_frame_side", None) is not None:
        ok = config_manager.set_video_max_frame_side(args.set_video_max_frame_side)
        if ok:
            print(f"✅ Set video max frame side to: {args.set_video_max_frame_side}")
        else:
            print(f"❌ Error: Invalid video max frame side: {args.set_video_max_frame_side}")
        handled = True

    if getattr(args, "set_video_max_size_bytes", None) is not None:
        ok = config_manager.set_video_max_video_size_bytes(args.set_video_max_size_bytes)
        if ok:
            if int(args.set_video_max_size_bytes) <= 0:
                print("✅ Cleared video max size")
            else:
                print(f"✅ Set video max size to: {args.set_video_max_size_bytes} bytes")
        else:
            print(f"❌ Error: Invalid video max size bytes: {args.set_video_max_size_bytes}")
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
    # #[WARNING:TIMEOUT]
    if args.set_default_timeout is not None:
        try:
            ok = config_manager.set_default_timeout(args.set_default_timeout)
            if not ok:
                raise RuntimeError("Configuration update failed")
            # Format display (show in minutes if >= 60 seconds)
            if args.set_default_timeout <= 0:
                display = "unlimited"
            elif args.set_default_timeout >= 60:
                minutes = args.set_default_timeout / 60
                display = f"{minutes:.1f} minutes" if minutes != int(minutes) else f"{int(minutes)} minutes"
            else:
                display = f"{args.set_default_timeout} seconds"
            suffix = "" if args.set_default_timeout <= 0 else f" ({args.set_default_timeout}s)"
            print(f"✅ Set default HTTP timeout to: {display}{suffix}")
        except Exception as e:
            print(f"❌ Failed to set default HTTP timeout: {e}")
        handled = True

    # #[WARNING:TIMEOUT]
    if args.set_tool_timeout is not None:
        try:
            ok = config_manager.set_tool_timeout(args.set_tool_timeout)
            if not ok:
                raise RuntimeError("Configuration update failed")
            # Format display (show in minutes if >= 60 seconds)
            if args.set_tool_timeout <= 0:
                display = "unlimited"
            elif args.set_tool_timeout >= 60:
                minutes = args.set_tool_timeout / 60
                display = f"{minutes:.1f} minutes" if minutes != int(minutes) else f"{int(minutes)} minutes"
            else:
                display = f"{args.set_tool_timeout} seconds"
            suffix = "" if args.set_tool_timeout <= 0 else f" ({args.set_tool_timeout}s)"
            print(f"✅ Set tool execution timeout to: {display}{suffix}")
        except Exception as e:
            print(f"❌ Failed to set tool execution timeout: {e}")
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
  abstractcore --configure                       # Interactive guided setup (7 steps)
  abstractcore --install                          # Check & install missing models/deps
  abstractcore --install --yes                   # Auto-download everything that's missing

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
  abstractcore --set-code-model anthropic/claude-haiku-4-5    # For coding tasks (cost-effective default)

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
    logging.basicConfig(level=logging.ERROR)
    sys.exit(main())
