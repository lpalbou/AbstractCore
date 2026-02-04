"""
AbstractCore Configuration Manager

Provides centralized configuration management for AbstractCore.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class VisionConfig:
    """Vision configuration settings."""
    strategy: str = "disabled"
    caption_provider: Optional[str] = None
    caption_model: Optional[str] = None
    fallback_chain: list = None
    local_models_path: Optional[str] = None

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = []


@dataclass
class AudioConfig:
    """Audio configuration settings (input policy + optional fallback)."""
    # Default: do not silently change semantics. Allow native audio when supported,
    # otherwise error unless the caller explicitly requests STT/caption.
    strategy: str = "native_only"  # native_only|speech_to_text|caption|auto
    # Optional preferred STT backend (capabilities plugin backend_id).
    stt_backend_id: Optional[str] = None
    stt_language: Optional[str] = None
    # Reserved for future "audio caption" backends.
    caption_provider: Optional[str] = None
    caption_model: Optional[str] = None
    fallback_chain: list = None

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = []


@dataclass
class VideoConfig:
    """Video configuration settings (input policy + optional fallback)."""
    # Default: best-effort usability. Prefer native video when supported; otherwise fall back
    # to sampled frames routed through existing image/vision handling.
    strategy: str = "auto"  # native_only|frames_caption|auto

    # Frame sampling controls for frames-based fallback.
    max_frames: int = 3
    # Native video models typically require more temporal coverage than the fallback path.
    # This default is used when the selected model supports native video input (v0: HF only).
    max_frames_native: int = 8
    frame_format: str = "jpg"  # jpg|png
    sampling_strategy: str = "uniform"  # uniform|keyframes

    # Downscale extracted frames (preserve aspect ratio; never upscale). Helps memory + token pressure.
    # Applies to both frames_caption fallback and HF native video ingestion (which uses ffmpeg frames).
    max_frame_side: int = 1024

    # Maximum video size allowed for processing (bytes). None => use media handler defaults.
    max_video_size_bytes: Optional[int] = None


@dataclass
class EmbeddingsConfig:
    """Embeddings configuration settings."""
    provider: Optional[str] = "huggingface"
    model: Optional[str] = "all-minilm-l6-v2"


@dataclass
class AppDefaults:
    """Per-application default configurations."""
    cli_provider: Optional[str] = "huggingface"
    cli_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    summarizer_provider: Optional[str] = "huggingface"
    summarizer_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    extractor_provider: Optional[str] = "huggingface"
    extractor_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    judge_provider: Optional[str] = "huggingface"
    judge_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    intent_provider: Optional[str] = "huggingface"
    intent_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


@dataclass
class MaintenanceConfig:
    """Maintenance agent configuration (triage, stewardship)."""

    # LLM assist (optional, local-first).
    triage_llm_enabled: bool = False
    triage_llm_base_url: str = "http://localhost:1234"
    triage_llm_model: str = "qwen/qwen3-next-80b"
    triage_llm_temperature: float = 0.2
    triage_llm_max_tokens: int = 800
    triage_llm_timeout_s: float = 30.0


@dataclass
class EmailConfig:
    """Email defaults (SMTP outbound + IMAP inbound).

    These defaults are used by framework-native comms tools and gateway bridges when
    explicit parameters are omitted (env vars still take precedence).
    """

    # SMTP (outbound)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password_env_var: str = "EMAIL_PASSWORD"
    smtp_use_starttls: bool = True
    from_email: Optional[str] = None
    reply_to: Optional[str] = None

    # IMAP (inbound)
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password_env_var: str = "EMAIL_PASSWORD"
    imap_folder: str = "INBOX"


@dataclass
class DefaultModels:
    """Global default model configurations."""
    global_provider: Optional[str] = None
    global_model: Optional[str] = None
    chat_model: Optional[str] = None
    code_model: Optional[str] = None


@dataclass
class ApiKeysConfig:
    """API keys configuration."""
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    openrouter: Optional[str] = None
    google: Optional[str] = None


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    default_cache_dir: str = "~/.cache/abstractcore"
    huggingface_cache_dir: str = "~/.cache/huggingface"
    local_models_cache_dir: str = "~/.abstractcore/models"
    glyph_cache_dir: str = "~/.abstractcore/glyph_cache"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    console_level: str = "WARNING"
    file_level: str = "DEBUG"
    file_logging_enabled: bool = False
    log_base_dir: Optional[str] = None
    verbatim_enabled: bool = True
    console_json: bool = False
    file_json: bool = True


@dataclass
class StreamingConfig:
    """Streaming configuration settings."""
    cli_stream_default: bool = False


@dataclass
class TimeoutConfig:
    """Timeout configuration settings."""
    # Default HTTP timeout for LLM providers (in seconds).
    # This is used as the *process-wide* default unless overridden per-provider/per-call.
    default_timeout: float = 7200.0  # 2 hours
    tool_timeout: float = 600.0     # 10 minutes for tool execution (in seconds)


@dataclass
class OfflineConfig:
    """Offline-first configuration settings."""
    offline_first: bool = True  # AbstractCore is designed offline-first for open source LLMs
    allow_network: bool = False  # Allow network access when offline_first is True (for API providers)
    force_local_files_only: bool = True  # Force local_files_only for HuggingFace transformers


@dataclass
class AbstractCoreConfig:
    """Main configuration class."""
    vision: VisionConfig
    audio: AudioConfig
    video: VideoConfig
    embeddings: EmbeddingsConfig
    app_defaults: AppDefaults
    default_models: DefaultModels
    api_keys: ApiKeysConfig
    cache: CacheConfig
    logging: LoggingConfig
    streaming: StreamingConfig
    timeouts: TimeoutConfig
    offline: OfflineConfig
    maintenance: MaintenanceConfig
    email: EmailConfig

    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            vision=VisionConfig(),
            audio=AudioConfig(),
            video=VideoConfig(),
            embeddings=EmbeddingsConfig(),
            app_defaults=AppDefaults(),
            default_models=DefaultModels(),
            api_keys=ApiKeysConfig(),
            cache=CacheConfig(),
            logging=LoggingConfig(),
            streaming=StreamingConfig(),
            timeouts=TimeoutConfig(),
            offline=OfflineConfig(),
            maintenance=MaintenanceConfig(),
            email=EmailConfig(),
        )


class ConfigurationManager:
    """Manages AbstractCore configuration."""

    def __init__(self):
        self.config_dir = Path.home() / ".abstractcore" / "config"
        self.config_file = self.config_dir / "abstractcore.json"
        self.config = self._load_config()
        self._provider_config: Dict[str, Dict[str, Any]] = {}  # Runtime config (not persisted)

    def _load_config(self) -> AbstractCoreConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception:
                # If loading fails, return default config
                return AbstractCoreConfig.default()
        else:
            return AbstractCoreConfig.default()

    def _dict_to_config(self, data: Dict[str, Any]) -> AbstractCoreConfig:
        """Convert dictionary to config object."""
        # Create config objects from dictionary data
        vision = VisionConfig(**data.get('vision', {}))
        audio = AudioConfig(**data.get('audio', {}))
        video = VideoConfig(**data.get('video', {}))
        embeddings = EmbeddingsConfig(**data.get('embeddings', {}))
        app_defaults = AppDefaults(**data.get('app_defaults', {}))
        default_models = DefaultModels(**data.get('default_models', {}))
        api_keys = ApiKeysConfig(**data.get('api_keys', {}))
        cache = CacheConfig(**data.get('cache', {}))
        logging = LoggingConfig(**data.get('logging', {}))
        streaming = StreamingConfig(**data.get('streaming', {}))
        timeouts = TimeoutConfig(**data.get('timeouts', {}))
        offline = OfflineConfig(**data.get('offline', {}))
        maintenance = MaintenanceConfig(**data.get('maintenance', {}))
        email_cfg = EmailConfig(**data.get('email', {}))

        return AbstractCoreConfig(
            vision=vision,
            audio=audio,
            video=video,
            embeddings=embeddings,
            app_defaults=app_defaults,
            default_models=default_models,
            api_keys=api_keys,
            cache=cache,
            logging=logging,
            streaming=streaming,
            timeouts=timeouts,
            offline=offline,
            maintenance=maintenance,
            email=email_cfg,
        )

    def _save_config(self):
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary
        config_dict = {
            'vision': asdict(self.config.vision),
            'audio': asdict(self.config.audio),
            'video': asdict(self.config.video),
            'embeddings': asdict(self.config.embeddings),
            'app_defaults': asdict(self.config.app_defaults),
            'default_models': asdict(self.config.default_models),
            'api_keys': asdict(self.config.api_keys),
            'cache': asdict(self.config.cache),
            'logging': asdict(self.config.logging),
            'streaming': asdict(self.config.streaming),
            'timeouts': asdict(self.config.timeouts),
            'offline': asdict(self.config.offline),
            'maintenance': asdict(self.config.maintenance),
        }

        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def set_vision_provider(self, provider: str, model: str) -> bool:
        """Set vision provider and model."""
        try:
            self.config.vision.strategy = "two_stage"
            self.config.vision.caption_provider = provider
            self.config.vision.caption_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_vision_caption(self, model: str) -> bool:
        """Set vision caption model (deprecated)."""
        # Auto-detect provider from model name
        provider = self._detect_provider_from_model(model)
        if provider:
            return self.set_vision_provider(provider, model)
        return False

    def _detect_provider_from_model(self, model: str) -> Optional[str]:
        """Detect provider from model name."""
        model_lower = model.lower()
        
        if any(x in model_lower for x in ['qwen2.5vl', 'llama3.2-vision', 'llava']):
            return "ollama"
        elif any(x in model_lower for x in ['gpt-4', 'gpt-4o']):
            return "openai"
        elif any(x in model_lower for x in ['claude-3']):
            return "anthropic"
        elif '/' in model:
            return "lmstudio"
        
        return None

    def get_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            "config_file": str(self.config_file),
            "vision": {
                "strategy": self.config.vision.strategy,
                "status": "✅ Ready" if self.config.vision.caption_provider else "❌ Not configured",
                "caption_provider": self.config.vision.caption_provider,
                "caption_model": self.config.vision.caption_model
            },
            "video": {
                "strategy": self.config.video.strategy,
                "max_frames": self.config.video.max_frames,
                "max_frames_native": getattr(self.config.video, "max_frames_native", None),
                "frame_format": self.config.video.frame_format,
                "sampling_strategy": getattr(self.config.video, "sampling_strategy", None),
                "max_frame_side": getattr(self.config.video, "max_frame_side", None),
            },
            "app_defaults": {
                "cli": {
                    "provider": self.config.app_defaults.cli_provider,
                    "model": self.config.app_defaults.cli_model
                },
                "summarizer": {
                    "provider": self.config.app_defaults.summarizer_provider,
                    "model": self.config.app_defaults.summarizer_model
                },
                "extractor": {
                    "provider": self.config.app_defaults.extractor_provider,
                    "model": self.config.app_defaults.extractor_model
                },
                "judge": {
                    "provider": self.config.app_defaults.judge_provider,
                    "model": self.config.app_defaults.judge_model
                },
                "intent": {
                    "provider": self.config.app_defaults.intent_provider,
                    "model": self.config.app_defaults.intent_model
                }
            },
            "global_defaults": {
                "provider": self.config.default_models.global_provider,
                "model": self.config.default_models.global_model,
                "chat_model": self.config.default_models.chat_model,
                "code_model": self.config.default_models.code_model
            },
            "embeddings": {
                "status": "✅ Ready",
                "provider": self.config.embeddings.provider,
                "model": self.config.embeddings.model
            },
            "streaming": {
                "cli_stream_default": self.config.streaming.cli_stream_default
            },
            "logging": {
                "console_level": self.config.logging.console_level,
                "file_level": self.config.logging.file_level,
                "file_logging_enabled": self.config.logging.file_logging_enabled
            },
            "timeouts": {
                "default_timeout": self.config.timeouts.default_timeout,
                "tool_timeout": self.config.timeouts.tool_timeout
            },
            "cache": {
                "default_cache_dir": self.config.cache.default_cache_dir,
                "huggingface_cache_dir": self.config.cache.huggingface_cache_dir,
                "local_models_cache_dir": self.config.cache.local_models_cache_dir,
                "glyph_cache_dir": self.config.cache.glyph_cache_dir,
            },
            "api_keys": {
                "openai": "✅ Set" if self.config.api_keys.openai else "❌ Not set",
                "anthropic": "✅ Set" if self.config.api_keys.anthropic else "❌ Not set",
                "openrouter": "✅ Set" if self.config.api_keys.openrouter else "❌ Not set",
                "google": "✅ Set" if self.config.api_keys.google else "❌ Not set"
            },
            "offline": {
                "offline_first": self.config.offline.offline_first,
                "allow_network": self.config.offline.allow_network,
                "status": "🔒 Offline-first" if self.config.offline.offline_first else "🌐 Network-enabled"
            }
        }

    def reset_configuration(self) -> bool:
        """Reset all configuration to built-in defaults."""
        try:
            self.config = AbstractCoreConfig.default()
            self._provider_config.clear()
            self._save_config()
            return True
        except Exception:
            return False

    def set_global_default_model(self, provider_model: str) -> bool:
        """Set global default model in provider/model format."""
        try:
            if '/' in provider_model:
                provider, model = provider_model.split('/', 1)
            else:
                # Assume it's just a model name, use default provider
                provider = "ollama"
                model = provider_model
            
            self.config.default_models.global_provider = provider
            self.config.default_models.global_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_default_model(self, provider_model: str) -> bool:
        """Legacy alias for setting the global default model."""
        return self.set_global_default_model(provider_model)

    def set_global_default_provider(self, provider: str) -> bool:
        """Set global default provider (legacy)."""
        try:
            provider = str(provider or "").strip()
            if not provider:
                raise ValueError("Provider cannot be empty")
            self.config.default_models.global_provider = provider
            self._save_config()
            return True
        except Exception:
            return False

    def set_chat_model(self, provider_model: str) -> bool:
        """Set specialized chat model (provider/model string)."""
        try:
            model = str(provider_model or "").strip()
            if not model:
                raise ValueError("Model cannot be empty")
            self.config.default_models.chat_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_code_model(self, provider_model: str) -> bool:
        """Set specialized code model (provider/model string)."""
        try:
            model = str(provider_model or "").strip()
            if not model:
                raise ValueError("Model cannot be empty")
            self.config.default_models.code_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_embeddings_model(self, provider_model: str) -> bool:
        """Set embeddings provider/model from a provider/model string (preferred)."""
        try:
            value = str(provider_model or "").strip()
            if not value:
                raise ValueError("Embeddings model cannot be empty")

            if "/" in value:
                provider, model = value.split("/", 1)
                self.config.embeddings.provider = provider.strip() or self.config.embeddings.provider
                self.config.embeddings.model = model.strip()
            else:
                self.config.embeddings.model = value

            self._save_config()
            return True
        except Exception:
            return False

    def set_embeddings_provider(self, provider: str) -> bool:
        """Set embeddings provider."""
        try:
            value = str(provider or "").strip()
            if not value:
                raise ValueError("Embeddings provider cannot be empty")
            self.config.embeddings.provider = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_default_cache_dir(self, path: str) -> bool:
        """Set default cache directory for AbstractCore."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("Cache directory cannot be empty")
            self.config.cache.default_cache_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_huggingface_cache_dir(self, path: str) -> bool:
        """Set HuggingFace cache directory."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("HuggingFace cache directory cannot be empty")
            self.config.cache.huggingface_cache_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_local_models_cache_dir(self, path: str) -> bool:
        """Set local models cache directory."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("Local models cache directory cannot be empty")
            self.config.cache.local_models_cache_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_log_base_dir(self, path: str) -> bool:
        """Set log base directory."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("Log base directory cannot be empty")
            self.config.logging.log_base_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_console_log_level(self, level: str) -> bool:
        """Set console logging level."""
        try:
            value = str(level or "").strip().upper()
            if not value:
                raise ValueError("Console log level cannot be empty")
            self.config.logging.console_level = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_file_log_level(self, level: str) -> bool:
        """Set file logging level."""
        try:
            value = str(level or "").strip().upper()
            if not value:
                raise ValueError("File log level cannot be empty")
            self.config.logging.file_level = value
            self._save_config()
            return True
        except Exception:
            return False

    def enable_debug_logging(self) -> bool:
        """Enable debug logging for both console and file."""
        try:
            self.config.logging.console_level = "DEBUG"
            self.config.logging.file_level = "DEBUG"
            self._save_config()
            return True
        except Exception:
            return False

    def disable_console_logging(self) -> bool:
        """Disable console logging output."""
        try:
            self.config.logging.console_level = "NONE"
            self._save_config()
            return True
        except Exception:
            return False

    def enable_file_logging(self) -> bool:
        """Enable file logging."""
        try:
            self.config.logging.file_logging_enabled = True
            self._save_config()
            return True
        except Exception:
            return False

    def disable_file_logging(self) -> bool:
        """Disable file logging."""
        try:
            self.config.logging.file_logging_enabled = False
            self._save_config()
            return True
        except Exception:
            return False

    def set_streaming_default(self, app_name: str, enabled: bool) -> bool:
        """Set default streaming behavior for a given app (currently: cli)."""
        try:
            app = str(app_name or "").strip().lower()
            if app != "cli":
                return False
            self.config.streaming.cli_stream_default = bool(enabled)
            self._save_config()
            return True
        except Exception:
            return False

    def get_streaming_default(self, app_name: str) -> bool:
        """Get default streaming behavior for a given app (currently: cli)."""
        app = str(app_name or "").strip().lower()
        if app == "cli":
            return bool(self.config.streaming.cli_stream_default)
        return False

    def enable_cli_streaming(self) -> bool:
        """Enable streaming by default for the CLI."""
        return self.set_streaming_default("cli", True)

    def disable_cli_streaming(self) -> bool:
        """Disable streaming by default for the CLI."""
        return self.set_streaming_default("cli", False)

    def add_vision_fallback(self, provider: str, model: str) -> bool:
        """Add a vision fallback provider/model to the chain."""
        try:
            provider_val = str(provider or "").strip()
            model_val = str(model or "").strip()
            if not provider_val or not model_val:
                raise ValueError("Provider and model are required")

            self.config.vision.fallback_chain.append({"provider": provider_val, "model": model_val})
            # If vision is configured at all, assume two_stage (caption -> text model).
            if not self.config.vision.strategy or self.config.vision.strategy == "disabled":
                self.config.vision.strategy = "two_stage"
            self._save_config()
            return True
        except Exception:
            return False

    def disable_vision(self) -> bool:
        """Disable vision fallback for text-only models."""
        try:
            self.config.vision.strategy = "disabled"
            self.config.vision.caption_provider = None
            self.config.vision.caption_model = None
            self.config.vision.fallback_chain = []
            self._save_config()
            return True
        except Exception:
            return False


    def set_app_default(self, app_name: str, provider: str, model: str) -> bool:
        """Set app-specific default provider and model."""
        try:
            if app_name == "cli":
                self.config.app_defaults.cli_provider = provider
                self.config.app_defaults.cli_model = model
            elif app_name == "summarizer":
                self.config.app_defaults.summarizer_provider = provider
                self.config.app_defaults.summarizer_model = model
            elif app_name == "extractor":
                self.config.app_defaults.extractor_provider = provider
                self.config.app_defaults.extractor_model = model
            elif app_name == "judge":
                self.config.app_defaults.judge_provider = provider
                self.config.app_defaults.judge_model = model
            elif app_name == "intent":
                self.config.app_defaults.intent_provider = provider
                self.config.app_defaults.intent_model = model
            else:
                raise ValueError(f"Unknown app: {app_name}")
            
            self._save_config()
            return True
        except Exception:
            return False

    def set_api_key(self, provider: str, key: str) -> bool:
        """Set API key for a provider."""
        try:
            if provider == "openai":
                self.config.api_keys.openai = key
            elif provider == "anthropic":
                self.config.api_keys.anthropic = key
            elif provider == "openrouter":
                self.config.api_keys.openrouter = key
            elif provider == "google":
                self.config.api_keys.google = key
            else:
                return False
            
            self._save_config()
            return True
        except Exception:
            return False

    def get_app_default(self, app_name: str) -> Tuple[str, str]:
        """Get default provider and model for an app."""
        app_defaults = self.config.app_defaults

        if app_name == "cli":
            return app_defaults.cli_provider, app_defaults.cli_model
        elif app_name == "summarizer":
            return app_defaults.summarizer_provider, app_defaults.summarizer_model
        elif app_name == "extractor":
            return app_defaults.extractor_provider, app_defaults.extractor_model
        elif app_name == "judge":
            return app_defaults.judge_provider, app_defaults.judge_model
        elif app_name == "intent":
            return app_defaults.intent_provider, app_defaults.intent_model
        else:
            # Return default fallback
            return "huggingface", "unsloth/Qwen3-4B-Instruct-2507-GGUF"

    def set_default_timeout(self, timeout: float) -> bool:
        """Set default HTTP request timeout in seconds."""
        try:
            # #[WARNING:TIMEOUT]
            # Contract: allow `0` to mean "unlimited" (the provider layer normalizes <=0 to None).
            timeout_f = float(timeout)
            if timeout_f < 0:
                raise ValueError("Timeout must be >= 0 (0 = unlimited)")
            self.config.timeouts.default_timeout = timeout_f
            self._save_config()
            return True
        except Exception:
            return False

    def set_tool_timeout(self, timeout: float) -> bool:
        """Set tool execution timeout in seconds."""
        try:
            # #[WARNING:TIMEOUT]
            # Contract: allow `0` to mean "unlimited".
            timeout_f = float(timeout)
            if timeout_f < 0:
                raise ValueError("Timeout must be >= 0 (0 = unlimited)")
            self.config.timeouts.tool_timeout = timeout_f
            self._save_config()
            return True
        except Exception:
            return False

    def get_default_timeout(self) -> float:
        """Get default HTTP request timeout in seconds."""
        return self.config.timeouts.default_timeout

    def get_tool_timeout(self) -> float:
        """Get tool execution timeout in seconds."""
        return self.config.timeouts.tool_timeout

    def set_offline_first(self, enabled: bool) -> bool:
        """Enable or disable offline-first mode."""
        try:
            self.config.offline.offline_first = enabled
            self._save_config()
            return True
        except Exception:
            return False

    def set_allow_network(self, enabled: bool) -> bool:
        """Allow network access when in offline-first mode."""
        try:
            self.config.offline.allow_network = enabled
            self._save_config()
            return True
        except Exception:
            return False

    def is_offline_first(self) -> bool:
        """Check if offline-first mode is enabled."""
        return self.config.offline.offline_first

    def is_network_allowed(self) -> bool:
        """Check if network access is allowed."""
        return self.config.offline.allow_network

    def should_force_local_files_only(self) -> bool:
        """Check if local_files_only should be forced for transformers."""
        return self.config.offline.force_local_files_only

    def configure_provider(self, provider: str, **kwargs) -> None:
        """
        Configure runtime settings for a provider.

        Args:
            provider: Provider name ('ollama', 'lmstudio', 'openai', 'anthropic')
            **kwargs: Configuration options (base_url, timeout, etc.)

        Example:
            configure_provider('ollama', base_url='http://192.168.1.100:11434')
        """
        provider = provider.lower()
        if provider not in self._provider_config:
            self._provider_config[provider] = {}

        for key, value in kwargs.items():
            if value is None:
                # Remove config (revert to env var / default)
                self._provider_config[provider].pop(key, None)
            else:
                self._provider_config[provider][key] = value

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get runtime configuration for a provider.

        Args:
            provider: Provider name

        Returns:
            Dict with configured settings, or empty dict if no config
        """
        return self._provider_config.get(provider.lower(), {}).copy()

    def clear_provider_config(self, provider: Optional[str] = None) -> None:
        """
        Clear runtime provider configuration.

        Args:
            provider: Provider name, or None to clear all
        """
        if provider is None:
            self._provider_config.clear()
        else:
            self._provider_config.pop(provider.lower(), None)


# Global instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager
