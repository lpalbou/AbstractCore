"""
HuggingFace provider implementation with GGUF support.
Supports both transformers models and GGUF models via llama-cpp-python.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator, Type

# Import config manager to respect offline-first settings
from ..config.manager import get_config_manager

# Get config instance and set offline environment variables if needed
_config = get_config_manager()
if _config.is_offline_first():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

# Enable MPS fallback for Apple Silicon to handle unsupported operations
# This prevents "MPS: Unsupported Border padding mode" errors in vision models
try:
    import torch
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
except ImportError:
    pass  # torch not available, skip MPS setup

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ModelNotFoundError, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

# Try to import transformers (standard HuggingFace support)
try:
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try to import llama-cpp-python (GGUF support)
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

# Try to import Outlines (native structured output for transformers models)
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False

# We no longer download models - cache-only approach
# huggingface_hub not required for basic operation


def _get_local_model_path(model_name: str) -> Optional[str]:
    """Get local cache path for a HuggingFace model if it exists."""
    # Use centralized configuration for cache directory
    config = _config
    hf_cache_dir = Path(config.config.cache.huggingface_cache_dir).expanduser()
    
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache_path = hf_cache_dir / "hub" / model_cache_name / "snapshots"
    
    if model_cache_path.exists():
        snapshot_dirs = [d for d in model_cache_path.iterdir() if d.is_dir()]
        if snapshot_dirs:
            return str(snapshot_dirs[0])  # Return first snapshot
    return None


class HuggingFaceProvider(BaseProvider):
    """HuggingFace provider with dual support for transformers and GGUF models"""

    def __init__(self, model: str = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                 device: Optional[str] = None,
                 n_gpu_layers: Optional[int] = None,
                 structured_output_method: str = "auto",
                 **kwargs):

        # Handle legacy context_size parameter with deprecation warning
        context_size = kwargs.pop("context_size", None)
        if context_size is not None:
            import warnings
            warnings.warn(
                "The 'context_size' parameter is deprecated. Use 'max_tokens' instead. "
                "context_size will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = context_size

        super().__init__(model, **kwargs)
        self.provider = "huggingface"

        # Handle timeout parameter for local models
        self._handle_timeout_parameter(kwargs)

        # Structured output method: "auto", "native_outlines", "prompted"
        # auto: Use Outlines if available (for transformers), otherwise prompted (default)
        # native_outlines: Force Outlines (error if unavailable)
        # prompted: Always use prompted fallback (fastest for transformers, still 100% success)
        # Note: GGUF models always use llama-cpp-python native support regardless of this setting
        self.structured_output_method = structured_output_method

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        # Store provider-specific configuration
        self.n_gpu_layers = n_gpu_layers
        self.model_type = None  # Will be "transformers" or "gguf"
        self.device = device
        
        # Store transformers-specific parameters
        self.transformers_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['trust_remote_code', 'torch_dtype', 'device_map', 'load_in_8bit', 'load_in_4bit', 'attn_implementation']
        }
        
        # Store device preference for custom models
        self.preferred_device = kwargs.get('device_map', 'auto')

        # Model instances
        self.tokenizer = None
        self.processor = None  # For vision models
        self.model_instance = None
        self.pipeline = None
        self.llm = None  # For GGUF models

        # Detect model type and load accordingly
        if self._is_gguf_model(model):
            if not LLAMACPP_AVAILABLE:
                raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self.model_type = "gguf"
            self._setup_device_gguf()
            self._load_gguf_model()
        else:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not installed. Install with: pip install transformers torch")
            self.model_type = "transformers"
            self._setup_device_transformers()
            self._load_transformers_model()

    def unload(self) -> None:
        """
        Unload the model from memory.

        For GGUF models, calls llm.close() to free llama.cpp resources.
        For transformers models, clears model and tokenizer references.
        """
        import gc
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # Try to properly close the Llama object (GGUF models)
                if hasattr(self.llm, 'close'):
                    self.llm.close()
                # Clear the reference
                self.llm = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.tokenizer = None
                
            if hasattr(self, 'processor') and self.processor is not None:
                self.processor = None

            if hasattr(self, 'model') and hasattr(self, 'model') and self.model is not None:
                # For transformers models, clear the model
                self.model = None

            # Force garbage collection to free memory immediately
            gc.collect()
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def __del__(self):
        """Properly clean up resources to minimize garbage collection issues"""
        try:
            self.unload()
        except Exception:
            # Silently handle any cleanup errors - this is expected during shutdown
            pass

    def _is_gguf_model(self, model: str) -> bool:
        """Detect if the model is a GGUF model"""
        # Check if it's a .gguf file path
        if model.endswith('.gguf'):
            return True

        # Check if local file exists with .gguf extension
        model_path = Path(model)
        if model_path.exists() and model_path.suffix == '.gguf':
            return True

        # Check if it's a HF repo with GGUF in the name (various formats)
        model_lower = model.lower()
        if 'gguf' in model_lower:
            # Handle formats like:
            # - "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
            # - "unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF" (cache format)
            # - "repo/model-GGUF"
            return True

        return False

    def _is_vision_model(self, model: str) -> bool:
        """Detect if the model is a vision model that requires special handling"""
        model_lower = model.lower()
        
        # Known vision models that require AutoModelForImageTextToText
        vision_models = [
            'glyph',           # zai-org/Glyph
            'glm-4.1v',        # GLM-4.1V variants
            'glm4v',           # GLM4V architecture
            'qwen-vl',         # Qwen-VL models
            'qwen2-vl',        # Qwen2-VL models
            'qwen2.5-vl',      # Qwen2.5-VL models
            'llava',           # LLaVA models
            'instructblip',    # InstructBLIP models
            'blip2',           # BLIP2 models
            'flamingo',        # Flamingo models
        ]
        
        return any(vision_keyword in model_lower for vision_keyword in vision_models)

    def _setup_device_transformers(self):
        """Setup device for transformers models"""
        if not TRANSFORMERS_AVAILABLE:
            return

        if self.device:
            self.device = self.device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def _setup_device_gguf(self):
        """Setup device for GGUF models"""
        # Auto-detect GPU layers if not specified
        if self.n_gpu_layers is None:
            if self.device == "cuda" or (self.device is None and self._has_cuda()):
                # Use all layers on GPU for CUDA
                self.n_gpu_layers = -1
            elif self.device == "mps" or (self.device is None and self._has_metal()):
                # Use GPU layers for Metal
                self.n_gpu_layers = 1  # Metal works differently
            else:
                # CPU only
                self.n_gpu_layers = 0

    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _has_metal(self) -> bool:
        """Check if Metal (Apple Silicon) is available"""
        try:
            import platform
            return platform.system() == "Darwin" and platform.processor() == "arm"
        except:
            return False

    def _load_transformers_model(self):
        """Load standard HuggingFace transformers model"""
        try:
            # Check if this is a vision model that requires special handling
            if self._is_vision_model(self.model):
                return self._load_vision_model()
            
            # Load tokenizer with transformers-specific parameters
            tokenizer_kwargs = {k: v for k, v in self.transformers_kwargs.items() 
                              if k in ['trust_remote_code']}
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                tokenizer_kwargs['local_files_only'] = True
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, **tokenizer_kwargs)
            
            # Load model with all transformers-specific parameters
            # Try AutoModelForCausalLM first, fall back to AutoModel for custom models
            model_kwargs = self.transformers_kwargs.copy()
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                model_kwargs['local_files_only'] = True
            
            try:
                self.model_instance = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
            except ValueError as e:
                if "Unrecognized configuration class" in str(e) or "glm4v" in str(e).lower():
                    # Fall back to AutoModel for custom models like DeepSeek-OCR
                    self.model_instance = AutoModel.from_pretrained(self.model, **model_kwargs)
                else:
                    raise

            # Move to device (only if not using device_map)
            if self.device in ["cuda", "mps"] and 'device_map' not in self.transformers_kwargs:
                self.model_instance = self.model_instance.to(self.device)

            # Create pipeline - handle custom models that don't support text-generation
            device_arg = 0 if self.device == "cuda" else -1
            if self.device == "mps":
                device_arg = -1

            try:
                # Don't pass device argument if using device_map (accelerate)
                if 'device_map' in self.transformers_kwargs:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_instance,
                        tokenizer=self.tokenizer
                    )
                else:
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model_instance,
                        tokenizer=self.tokenizer,
                        device=device_arg
                    )
            except ValueError as e:
                if "not supported for text-generation" in str(e) or "accelerate" in str(e):
                    # For custom models like DeepSeek-OCR, skip pipeline creation
                    # We'll handle generation directly through the model
                    self.pipeline = None
                else:
                    raise

        except Exception as e:
            error_str = str(e).lower()
            if ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                available_models = self.list_available_models()
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model}: {str(e)}")

    def _load_vision_model(self):
        """Load vision model using AutoModelForImageTextToText and AutoProcessor"""
        try:
            # Suppress progress bars during model loading unless in debug mode
            import os
            from transformers.utils import logging as transformers_logging
            
            if not self.debug:
                # Disable transformers progress bars
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                transformers_logging.set_verbosity_error()
                # Disable tqdm progress bars
                os.environ['DISABLE_TQDM'] = '1'
            
            # Load processor for vision models (handles both text and images)
            processor_kwargs = {k: v for k, v in self.transformers_kwargs.items() 
                              if k in ['trust_remote_code']}
            # Enable trust_remote_code for custom architectures like GLM4V
            processor_kwargs['trust_remote_code'] = True
            # Set use_fast=True to avoid the slow processor warning
            processor_kwargs['use_fast'] = True
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                processor_kwargs['local_files_only'] = True
            
            # Use local cache path if offline mode is enabled and model is cached
            model_path = self.model
            if _config.should_force_local_files_only():
                local_path = _get_local_model_path(self.model)
                if local_path:
                    model_path = local_path
                    processor_kwargs.pop('local_files_only', None)  # Remove since we're using local path
                    self.logger.debug(f"Loading processor from local cache: {local_path}")
            
            self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
            
            # Load vision model using AutoModelForImageTextToText with trust_remote_code
            vision_kwargs = self.transformers_kwargs.copy()
            vision_kwargs['trust_remote_code'] = True
            # Respect offline-first configuration
            if _config.should_force_local_files_only():
                vision_kwargs['local_files_only'] = True
            
            # Use local cache path if offline mode is enabled and model is cached
            model_path = self.model
            if _config.should_force_local_files_only():
                local_path = _get_local_model_path(self.model)
                if local_path:
                    model_path = local_path
                    vision_kwargs.pop('local_files_only', None)  # Remove since we're using local path
                    self.logger.debug(f"Loading model from local cache: {local_path}")
            
            self.model_instance = AutoModelForImageTextToText.from_pretrained(model_path, **vision_kwargs)
            
            # Restore logging levels if they were suppressed
            if not self.debug:
                # Restore transformers logging
                transformers_logging.set_verbosity_warning()
                # Remove tqdm suppression
                if 'DISABLE_TQDM' in os.environ:
                    del os.environ['DISABLE_TQDM']
            
            # Move to device (only if not using device_map)
            if self.device in ["cuda", "mps"] and 'device_map' not in self.transformers_kwargs:
                self.model_instance = self.model_instance.to(self.device)
            
            # For vision models, we don't use the standard pipeline
            self.pipeline = None
            
            self.logger.info(f"Successfully loaded vision model {self.model} using AutoModelForImageTextToText")
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for transformers version issues
            if 'glm4v' in error_str and 'does not recognize this architecture' in error_str:
                import transformers
                current_version = transformers.__version__
                raise RuntimeError(
                    f"GLM4V architecture requires transformers>=4.57.1, but you have {current_version}. "
                    f"Please upgrade: pip install transformers>=4.57.1"
                )
            elif ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                available_models = self.list_available_models()
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace vision model {self.model}: {str(e)}")

    def _find_gguf_in_cache(self, model_name: str) -> Optional[str]:
        """Find GGUF model in HuggingFace cache (cache-only, no downloading)"""

        # Normalize model name to cache format
        # Convert "unsloth/model" or "unsloth--model" to "models--unsloth--model"
        cache_name = self._normalize_to_cache_format(model_name)

        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_dir = cache_base / cache_name

        if not model_cache_dir.exists():
            return None

        # Look for GGUF files in snapshots
        snapshots_dir = model_cache_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        # Find the latest snapshot (most recent directory)
        try:
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if not snapshot_dirs:
                return None

            # Use the most recent snapshot
            latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)

            # Look for GGUF files in the snapshot
            gguf_files = list(latest_snapshot.glob("*.gguf"))

            if gguf_files:
                # Prefer certain quantizations
                preferred_quants = ['Q4_K_M', 'Q5_K_M', 'Q4_0', 'Q4_1', 'Q5_0', 'Q8_0']

                for quant in preferred_quants:
                    for gguf_file in gguf_files:
                        if quant in gguf_file.name:
                            return str(gguf_file)

                # Return first available GGUF file
                return str(gguf_files[0])

        except Exception:
            pass

        return None

    def _normalize_to_cache_format(self, model_name: str) -> str:
        """Convert model name to HuggingFace cache directory format"""
        # Remove any ":filename" suffix
        if ':' in model_name:
            model_name = model_name.split(':', 1)[0]

        # Handle different input formats:
        if model_name.startswith('models--'):
            # Already in cache format
            return model_name
        elif '/' in model_name:
            # Standard format: "unsloth/model" -> "models--unsloth--model"
            return f"models--{model_name.replace('/', '--')}"
        elif '--' in model_name and not model_name.startswith('models--'):
            # Cache format without prefix: "unsloth--model" -> "models--unsloth--model"
            return f"models--{model_name}"
        else:
            # Single name, assume it's just the model part
            return f"models--{model_name}"

    def _load_gguf_model(self):
        """Load GGUF model using llama-cpp-python (cache-only, no downloading)"""
        import os
        try:
            model_path = None

            # First, try as a direct file path
            if Path(self.model).exists() and self.model.endswith('.gguf'):
                model_path = self.model
            else:
                # Try to find in HuggingFace cache
                model_path = self._find_gguf_in_cache(self.model)

            if not model_path:
                # Model not found in cache - provide graceful fallback
                self._handle_gguf_not_found()
                return

            # Verify file exists and is accessible
            if not Path(model_path).exists():
                raise FileNotFoundError(f"GGUF model file not found: {model_path}")

            # Determine chat format for function calling
            chat_format = None
            model_lower = self.model.lower()
            if 'qwen' in model_lower or 'coder' in model_lower:
                # Qwen models often support function calling
                chat_format = "chatml-function-calling"
            elif 'functionary' in model_lower:
                chat_format = "functionary-v2"

            # Initialize llama-cpp-python with stderr redirected to our logger
            llama_kwargs = {
                "model_path": model_path,
                "n_ctx": self.max_tokens,  # Use unified max_tokens for context window
                "n_gpu_layers": self.n_gpu_layers,
                "chat_format": chat_format,
                "verbose": self.debug,  # Use debug flag for verbose output
                "n_threads": os.cpu_count() // 2 if os.cpu_count() else 4,
                # Additional performance settings
                "n_batch": 512,
                "use_mmap": True,
                "use_mlock": False
            }

            # Redirect stderr during initialization to avoid GGUF loading noise
            from contextlib import redirect_stderr

            if self.debug:
                # In debug mode, keep stderr visible
                self.llm = Llama(**llama_kwargs)
            else:
                # In non-debug mode, silence stderr during model loading
                with open(os.devnull, 'w') as devnull:
                    with redirect_stderr(devnull):
                        self.llm = Llama(**llama_kwargs)

        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model {self.model}: {str(e)}")

    def _handle_gguf_not_found(self):
        """Handle GGUF model not found with graceful fallback like other providers"""
        # Suggest the correct repo format
        suggested_repo = self._suggest_correct_repo_format(self.model)

        # List any similar models in cache
        similar_models = self._find_similar_gguf_models()

        error_parts = [
            f"❌ GGUF model '{self.model}' not found in HuggingFace cache.",
            "",
            "💡 To download this model, run:",
            f"   huggingface-cli download {suggested_repo}",
            "",
            "🔍 Suggested formats:",
            f"   • Correct: '{suggested_repo}'",
            f"   • Your input: '{self.model}'",
        ]

        if similar_models:
            error_parts.extend([
                "",
                "📂 Similar GGUF models found in cache:",
            ])
            for model in similar_models[:5]:  # Show max 5
                error_parts.append(f"   • {model}")

        error_parts.extend([
            "",
            "📖 For more info: https://huggingface.co/docs/hub/en/gguf",
            "🔧 AbstractCore only uses cached models - we never download automatically."
        ])

        error_message = "\n".join(error_parts)
        raise ModelNotFoundError(error_message)

    def _suggest_correct_repo_format(self, model_name: str) -> str:
        """Suggest the correct repository format"""
        # Handle various input formats and suggest the standard format
        if model_name.startswith('models--'):
            # "models--unsloth--model" -> "unsloth/model"
            parts = model_name.replace('models--', '').split('--', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"

        elif '--' in model_name and not '/' in model_name:
            # "unsloth--model" -> "unsloth/model"
            parts = model_name.split('--', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"

        # Return as-is if already in correct format or unknown format
        return model_name

    def _find_similar_gguf_models(self) -> List[str]:
        """Find similar GGUF models in cache"""
        cache_base = Path.home() / ".cache" / "huggingface" / "hub"

        if not cache_base.exists():
            return []

        similar_models = []
        try:
            for cache_dir in cache_base.iterdir():
                if cache_dir.is_dir() and 'gguf' in cache_dir.name.lower():
                    # Convert cache format back to standard format
                    if cache_dir.name.startswith('models--'):
                        repo_name = cache_dir.name.replace('models--', '').replace('--', '/', 1)
                        similar_models.append(repo_name)

        except Exception:
            pass

        return sorted(similar_models)

    def _handle_timeout_parameter(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle timeout parameter for HuggingFace provider.
        
        Since HuggingFace models run locally (both transformers and GGUF),
        timeout parameters don't apply. If a non-None timeout is provided,
        issue a warning and treat it as None (infinity).
        
        Args:
            kwargs: Initialization kwargs that may contain timeout
        """
        timeout_value = kwargs.get('timeout')
        if timeout_value is not None:
            import warnings
            warnings.warn(
                f"HuggingFace provider runs models locally and does not support timeout parameters. "
                f"Provided timeout={timeout_value} will be ignored and treated as None (unlimited).",
                UserWarning,
                stacklevel=3
            )
            # Force timeout to None for local models
            self._timeout = None
        else:
            # Keep None value (unlimited timeout is appropriate for local models)
            self._timeout = None

    def _update_http_client_timeout(self) -> None:
        """
        HuggingFace provider doesn't use HTTP clients for model inference.
        Local models (transformers and GGUF) don't have timeout constraints.
        """
        # No-op for local models - they don't use HTTP clients
        pass

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate response using appropriate backend"""

        if self.model_type == "gguf":
            return self._generate_gguf(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
        else:
            return self._generate_transformers(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)

    def _generate_transformers(self,
                               prompt: str,
                               messages: Optional[List[Dict[str, str]]] = None,
                               system_prompt: Optional[str] = None,
                               tools: Optional[List[Dict[str, Any]]] = None,
                               media: Optional[List['MediaContent']] = None,
                               stream: bool = False,
                               response_model: Optional[Type[BaseModel]] = None,
                               **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using transformers backend with optional Outlines native structured output"""

        if not self.pipeline:
            # Handle vision models that use processor instead of pipeline
            if self.processor and hasattr(self.model_instance, 'generate'):
                return self._generate_vision_model(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
            # Handle custom models like DeepSeek-OCR that don't support standard pipelines
            elif hasattr(self.model_instance, 'infer'):
                return self._generate_custom_model(prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
            else:
                return GenerateResponse(
                    content="Error: Transformers model not loaded or doesn't support generation",
                    model=self.model,
                    finish_reason="error"
                )

        # Native structured output via Outlines (if configured and available)
        should_use_outlines = (
            response_model and
            PYDANTIC_AVAILABLE and
            not stream and
            self.structured_output_method != "prompted"  # Skip if explicitly prompted
        )

        if should_use_outlines:
            # Check if Outlines is required but unavailable
            if self.structured_output_method == "native_outlines" and not OUTLINES_AVAILABLE:
                return GenerateResponse(
                    content="Error: structured_output_method='native_outlines' requires Outlines library. Install with: pip install abstractcore[huggingface]",
                    model=self.model,
                    finish_reason="error"
                )

            # Try Outlines if available (auto or native_outlines mode)
            if OUTLINES_AVAILABLE:
                try:
                    # Cache Outlines model wrapper to avoid re-initialization
                    if not hasattr(self, '_outlines_model') or self._outlines_model is None:
                        self.logger.debug("Creating Outlines model wrapper for native structured output")
                        self._outlines_model = outlines.from_transformers(
                            self.model_instance,
                            self.tokenizer
                        )

                    # Build input text (same as normal generation)
                    input_text = self._build_input_text_transformers(prompt, messages, system_prompt, tools)

                    # Create constrained generator with JSON schema
                    self.logger.debug(f"Using Outlines native structured output for {response_model.__name__}")
                    generator = self._outlines_model(
                        input_text,
                        outlines.json_schema(response_model),
                        max_tokens=kwargs.get("max_tokens", self.max_tokens or 512)
                    )

                    # Validate and return
                    validated_obj = response_model.model_validate(generator)

                    return GenerateResponse(
                        content=validated_obj.model_dump_json(),
                        model=self.model,
                        finish_reason="stop",
                        validated_object=validated_obj
                    )
                except Exception as e:
                    # If native_outlines was explicitly requested, don't fall back
                    if self.structured_output_method == "native_outlines":
                        return GenerateResponse(
                            content=f"Error: Outlines native structured output failed: {str(e)}",
                            model=self.model,
                            finish_reason="error"
                        )
                    # Otherwise fall back to prompted approach
                    self.logger.debug(f"Outlines generation failed, falling back to prompted: {e}")
                    # Continue with normal generation below

        # Build input text with tool and media support
        # Handle media content first if present
        if media:
            try:
                from ..media.handlers import LocalMediaHandler
                media_handler = LocalMediaHandler("huggingface", self.model_capabilities, model_name=self.model)

                # Create multimodal message combining text and media
                multimodal_message = media_handler.create_multimodal_message(prompt, media)

                # For local providers, we get text-embedded content
                if isinstance(multimodal_message, str):
                    prompt = multimodal_message
                else:
                    # If we get a structured message, extract the content
                    if isinstance(multimodal_message, dict) and "content" in multimodal_message:
                        if isinstance(multimodal_message["content"], list):
                            # Find text content in the structured message
                            text_content = ""
                            for item in multimodal_message["content"]:
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                    break
                            prompt = text_content or prompt
                        else:
                            prompt = str(multimodal_message["content"])
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        input_text = self._build_input_text_transformers(prompt, messages, system_prompt, tools)

        # Generation parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_new_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", 0.9)
        seed_value = kwargs.get("seed", self.seed)

        try:
            if stream:
                return self._stream_generate_transformers_with_tools(input_text, max_new_tokens, temperature, top_p, tools, kwargs.get('tool_call_tags'), seed_value)
            else:
                response = self._single_generate_transformers(input_text, max_new_tokens, temperature, top_p, seed_value)

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    response = self._handle_prompted_tool_execution(response, tools)

                return response

        except Exception as e:
            return GenerateResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _generate_custom_model(self,
                              prompt: str,
                              messages: Optional[List[Dict[str, str]]] = None,
                              system_prompt: Optional[str] = None,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              media: Optional[List['MediaContent']] = None,
                              stream: bool = False,
                              response_model: Optional[Type[BaseModel]] = None,
                              **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using custom model methods (e.g., DeepSeek-OCR's infer method)"""
        
        import time
        import tempfile
        import os
        start_time = time.time()
        
        try:
            # Handle media content for vision models like DeepSeek-OCR
            if media and len(media) > 0:
                # Use the first image for OCR
                media_item = media[0]
                
                # DeepSeek-OCR expects image file path
                if hasattr(media_item, 'file_path') and media_item.file_path:
                    image_file = str(media_item.file_path)
                else:
                    # If no file path, save media content to temp file
                    from PIL import Image
                    
                    if hasattr(media_item, 'content') and media_item.content:
                        # Handle base64 content
                        if media_item.content_format == 'BASE64':
                            import base64
                            image_data = base64.b64decode(media_item.content)
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_file.write(image_data)
                            temp_file.close()
                            image_file = temp_file.name
                        else:
                            return GenerateResponse(
                                content="Error: Unsupported media format for DeepSeek-OCR",
                                model=self.model,
                                finish_reason="error"
                            )
                    else:
                        return GenerateResponse(
                            content="Error: No valid image content found",
                            model=self.model,
                            finish_reason="error"
                        )
                
                # Use DeepSeek-OCR's infer method
                try:
                    # Create temporary output directory for DeepSeek-OCR
                    temp_output_dir = tempfile.mkdtemp()
                    
                    # Patch DeepSeek-OCR for MPS/CPU compatibility if needed
                    if self.device == "mps" or (self.device is None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                        self._patch_deepseek_for_mps()
                    
                    result = self.model_instance.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path=temp_output_dir,  # DeepSeek-OCR requires output path
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=False
                    )
                    
                    # Clean up temp output directory
                    import shutil
                    shutil.rmtree(temp_output_dir, ignore_errors=True)
                    
                    # Clean up temp file if created
                    if 'temp_file' in locals() and os.path.exists(image_file):
                        os.unlink(image_file)
                    
                    # Calculate generation time
                    gen_time = (time.time() - start_time) * 1000
                    
                    return GenerateResponse(
                        content=result if isinstance(result, str) else str(result),
                        model=self.model,
                        finish_reason="stop",
                        input_tokens=len(prompt.split()),  # Rough estimate
                        output_tokens=len(str(result).split()) if result else 0,
                        gen_time=gen_time
                    )
                    
                except Exception as e:
                    return GenerateResponse(
                        content=f"Error during DeepSeek-OCR inference: {str(e)}",
                        model=self.model,
                        finish_reason="error"
                    )
            else:
                return GenerateResponse(
                    content="Error: DeepSeek-OCR requires image input",
                    model=self.model,
                    finish_reason="error"
                )
                
        except Exception as e:
            return GenerateResponse(
                content=f"Error in custom model generation: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _generate_vision_model(self,
                              prompt: str,
                              messages: Optional[List[Dict[str, str]]] = None,
                              system_prompt: Optional[str] = None,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              media: Optional[List['MediaContent']] = None,
                              stream: bool = False,
                              response_model: Optional[Type[BaseModel]] = None,
                              **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using vision model (Glyph, GLM-4.1V, etc.)"""
        
        import time
        start_time = time.time()
        
        # Import torch safely
        try:
            import torch
        except ImportError:
            return GenerateResponse(
                content="Error: PyTorch not available for vision model generation",
                model=self.model,
                finish_reason="error",
                gen_time=0.0
            )
        
        try:
            # Build messages for vision model
            chat_messages = []
            
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            
            if messages:
                chat_messages.extend(messages)
            
            # Build user message with media content
            user_content = []
            
            # Add text content
            if prompt:
                user_content.append({"type": "text", "text": prompt})
            
            # Add media content (images)
            if media:
                for media_item in media:
                    if hasattr(media_item, 'file_path') and media_item.file_path:
                        # Use file path directly
                        user_content.append({
                            "type": "image",
                            "url": str(media_item.file_path)
                        })
                    elif hasattr(media_item, 'content') and media_item.content:
                        # Handle base64 content
                        if media_item.content_format == 'BASE64':
                            # Create data URL for base64 content
                            mime_type = getattr(media_item, 'mime_type', 'image/png')
                            data_url = f"data:{mime_type};base64,{media_item.content}"
                            user_content.append({
                                "type": "image",
                                "url": data_url
                            })
            
            # Add user message
            chat_messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Process messages using the processor
            inputs = self.processor.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model_instance.device)
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", self.max_output_tokens or 512),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": True,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # Add seed if provided
            seed_value = kwargs.get("seed", self.seed)
            if seed_value is not None:
                torch.manual_seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
            
            # Generate response
            # For Apple Silicon, move inputs to CPU if MPS causes issues
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    generated_ids = self.model_instance.generate(**inputs, **generation_kwargs)
                except RuntimeError as e:
                    if "MPS: Unsupported Border padding mode" in str(e):
                        self.logger.warning("MPS Border padding mode error detected, falling back to CPU")
                        # Move model and inputs to CPU
                        cpu_model = self.model_instance.to('cpu')
                        cpu_inputs = {k: v.to('cpu') if hasattr(v, 'to') else v for k, v in inputs.items()}
                        generated_ids = cpu_model.generate(**cpu_inputs, **generation_kwargs)
                        # Move model back to original device
                        self.model_instance.to(self.model_instance.device)
                    else:
                        raise e
            else:
                generated_ids = self.model_instance.generate(**inputs, **generation_kwargs)
            
            # Decode response
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Calculate generation time
            gen_time = (time.time() - start_time) * 1000
            
            # Calculate token usage
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = len(generated_ids[0]) - input_tokens
            
            return GenerateResponse(
                content=output_text.strip(),
                model=self.model,
                finish_reason="stop",
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens
                },
                gen_time=gen_time
            )
            
        except Exception as e:
            gen_time = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0.0
            return GenerateResponse(
                content=f"Error in vision model generation: {str(e)}",
                model=self.model,
                finish_reason="error",
                gen_time=gen_time
            )

    def _patch_deepseek_for_mps(self):
        """Patch DeepSeek-OCR model to work with MPS instead of CUDA"""
        import types
        
        def patched_infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
            """Patched infer method that uses MPS instead of CUDA"""
            import torch
            
            # Determine the best available device
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # Call the original infer method but patch tensor.cuda() calls
            original_cuda = torch.Tensor.cuda
            
            def patched_cuda(tensor, device=None, non_blocking=False, **kwargs):
                """Redirect .cuda() calls to the appropriate device"""
                if device == 'mps' or (device is None and torch.backends.mps.is_available()):
                    return tensor.to('mps', non_blocking=non_blocking)
                elif torch.cuda.is_available():
                    return original_cuda(tensor, device, non_blocking, **kwargs)
                else:
                    return tensor.to('cpu', non_blocking=non_blocking)
            
            # Temporarily patch the cuda method
            torch.Tensor.cuda = patched_cuda
            
            try:
                # Move model to the appropriate device first
                self.to(device)
                
                # Call original infer with device patching
                return self._original_infer(tokenizer, prompt, image_file, output_path, base_size, image_size, crop_mode, test_compress, save_results, eval_mode)
            finally:
                # Restore original cuda method
                torch.Tensor.cuda = original_cuda
        
        # Only patch if not already patched
        if not hasattr(self.model_instance, '_original_infer'):
            self.model_instance._original_infer = self.model_instance.infer
            self.model_instance.infer = types.MethodType(patched_infer, self.model_instance)

    def _generate_gguf(self,
                       prompt: str,
                       messages: Optional[List[Dict[str, str]]] = None,
                       system_prompt: Optional[str] = None,
                       tools: Optional[List[Dict[str, Any]]] = None,
                       media: Optional[List['MediaContent']] = None,
                       stream: bool = False,
                       response_model: Optional[Type[BaseModel]] = None,
                       **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Generate using GGUF backend with llama-cpp-python"""

        if not self.llm:
            return GenerateResponse(
                content="Error: GGUF model not loaded",
                model=self.model,
                finish_reason="error"
            )

        # Build messages for chat completion
        chat_messages = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        if messages:
            chat_messages.extend(messages)

        # Handle media content for the user message - use proper vision format for GGUF models
        if media:
            try:
                from ..architectures.detection import supports_vision

                # Check if this model supports vision natively
                if supports_vision(self.model):
                    # Use HuggingFace multimodal format for vision-capable GGUF models
                    user_message_content = []

                    # Add text content
                    user_message_content.append({"type": "text", "text": prompt})

                    # Add media content
                    for media_item in media:
                        if hasattr(media_item, 'file_path') and media_item.file_path:
                            # Use file:// URL format as specified in HuggingFace docs
                            file_path = str(media_item.file_path)
                            if not file_path.startswith('file://'):
                                file_path = f"file://{file_path}"
                            user_message_content.append({
                                "type": "image",
                                "image": file_path
                            })
                        elif hasattr(media_item, 'content') and media_item.content:
                            # For base64 or other content, we might need to save to temp file
                            import tempfile
                            import base64
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                                if isinstance(media_item.content, str) and media_item.content.startswith('data:'):
                                    # Handle base64 data URLs
                                    header, data = media_item.content.split(',', 1)
                                    decoded_data = base64.b64decode(data)
                                    tmp_file.write(decoded_data)
                                else:
                                    tmp_file.write(media_item.content)
                                tmp_file.flush()
                                user_message_content.append({
                                    "type": "image",
                                    "image": f"file://{tmp_file.name}"
                                })
                else:
                    # Fallback to text-based media handling for non-vision models
                    from ..media.handlers import LocalMediaHandler
                    media_handler = LocalMediaHandler("huggingface", self.model_capabilities, model_name=self.model)
                    multimodal_message = media_handler.create_multimodal_message(prompt, media)
                    user_message_content = multimodal_message if isinstance(multimodal_message, str) else prompt

            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install abstractcore[media]")
                user_message_content = prompt
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")
                user_message_content = prompt
        else:
            user_message_content = prompt

        chat_messages.append({"role": "user", "content": user_message_content})

        # Prepare parameters using unified system
        unified_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_output_tokens = self._get_provider_max_tokens_param(unified_kwargs)

        generation_kwargs = {
            "messages": chat_messages,
            "max_tokens": max_output_tokens,  # This is max_output_tokens for llama-cpp
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": stream
        }

        # Add seed if provided (GGUF/llama-cpp supports seed)
        seed_value = kwargs.get("seed", self.seed)
        if seed_value is not None:
            generation_kwargs["seed"] = seed_value

        # Add native structured output support (llama-cpp-python format)
        # llama-cpp-python supports native structured outputs using the response_format parameter
        # This provides server-side guaranteed schema compliance
        if response_model and PYDANTIC_AVAILABLE:
            json_schema = response_model.model_json_schema()
            generation_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": json_schema
                }
            }

        # Handle tools - both native and prompted support
        has_native_tools = False
        if tools:
            # Check if model supports native tools - but fall back to prompted for now
            # TODO: Re-enable native tools once parameter default handling is fixed
            if False and self.llm.chat_format in ["chatml-function-calling", "functionary-v2"]:
                # Use unified tool handler for consistent formatting
                openai_tools = self.tool_handler.prepare_tools_for_native(tools)
                generation_kwargs["tools"] = openai_tools

                # Debug: Print what we're sending to the model
                print(f"DEBUG: Sending tools to HuggingFace model (unified handler):")
                import json
                print(json.dumps(openai_tools, indent=2))

                # Don't use auto for streaming (limitation of llama-cpp-python)
                if not stream:
                    generation_kwargs["tool_choice"] = "auto"
                has_native_tools = True

            elif self.tool_handler.supports_prompted:
                # Add tools as system prompt for prompted models
                system_text = (
                    chat_messages[0].get("content", "")
                    if chat_messages and chat_messages[0].get("role") == "system"
                    else ""
                )
                include_tool_list = "## Tools (session)" not in str(system_text)
                tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
                if chat_messages and chat_messages[0]["role"] == "system":
                    chat_messages[0]["content"] += f"\n\n{tool_prompt}"
                else:
                    chat_messages.insert(0, {"role": "system", "content": tool_prompt})
                generation_kwargs["messages"] = chat_messages

        try:
            if stream:
                return self._stream_generate_gguf_with_tools(generation_kwargs, tools, has_native_tools, kwargs.get('tool_call_tags'))
            else:
                response = self._single_generate_gguf(generation_kwargs)

                # Handle tool execution for both native and prompted responses
                if tools and (response.has_tool_calls() or
                             (self.tool_handler.supports_prompted and response.content)):
                    response = self._handle_tool_execution_gguf(response, tools, has_native_tools)

                return response

        except Exception as e:
            if stream:
                # Return error as a generator
                def error_generator():
                    yield GenerateResponse(
                        content=f"Error: {str(e)}",
                        model=self.model,
                        finish_reason="error"
                    )
                return error_generator()
            else:
                return GenerateResponse(
                    content=f"Error: {str(e)}",
                    model=self.model,
                    finish_reason="error"
                )

    def _single_generate_gguf(self, kwargs: Dict[str, Any]) -> GenerateResponse:
        """Generate single response using GGUF"""
        response = self.llm.create_chat_completion(**kwargs)

        choice = response['choices'][0]
        message = choice['message']

        # Extract tool calls if present
        tool_calls = None
        if 'tool_calls' in message:
            tool_calls = []
            for tc in message['tool_calls']:
                tool_calls.append({
                    "id": tc.get('id'),
                    "type": tc.get('type', 'function'),
                    "name": tc['function']['name'],
                    "arguments": tc['function']['arguments']
                })

        # Extract usage
        usage = None
        if 'usage' in response:
            usage = {
                "prompt_tokens": response['usage'].get('prompt_tokens', 0),
                "completion_tokens": response['usage'].get('completion_tokens', 0),
                "total_tokens": response['usage'].get('total_tokens', 0)
            }

        # Fix HTML escaping in llama-cpp-python responses
        content = message.get('content', '')
        if content:
            import html
            content = html.unescape(content)

        return GenerateResponse(
            content=content,
            model=self.model,
            finish_reason=choice.get('finish_reason', 'stop'),
            usage=usage,
            tool_calls=tool_calls
        )

    def _stream_generate_gguf(self, kwargs: Dict[str, Any], tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream response using GGUF with tool tag rewriting support"""
        stream = self.llm.create_chat_completion(**kwargs)

        current_tool_call = None
        accumulated_arguments = ""
        
        # Initialize tool tag rewriter if needed
        rewriter = None
        buffer = ""
        if tool_call_tags:
            try:
                from ..tools.tag_rewriter import create_tag_rewriter
                rewriter = create_tag_rewriter(tool_call_tags)
            except ImportError:
                pass

        for chunk in stream:
            if 'choices' not in chunk or not chunk['choices']:
                continue

            choice = chunk['choices'][0]
            delta = choice.get('delta', {})

            # Handle text content
            if 'content' in delta and delta['content']:
                # Fix HTML escaping in streaming content
                content = delta['content']
                if content:
                    import html
                    content = html.unescape(content)
                    
                    # Apply tool tag rewriting if enabled
                    if rewriter:
                        rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                        content = rewritten_content

                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason=choice.get('finish_reason')
                )

            # Handle tool calls
            if 'tool_calls' in delta:
                for tc in delta['tool_calls']:
                    if 'function' in tc:
                        if tc.get('id'):  # New tool call
                            if current_tool_call and accumulated_arguments:
                                # Yield the previous tool call
                                current_tool_call['arguments'] = accumulated_arguments
                                yield GenerateResponse(
                                    content="",
                                    model=self.model,
                                    tool_calls=[current_tool_call]
                                )

                            # Start new tool call
                            current_tool_call = {
                                "id": tc.get('id'),
                                "type": tc.get('type', 'function'),
                                "name": tc['function'].get('name'),
                                "arguments": ""
                            }
                            accumulated_arguments = tc['function'].get('arguments', '')
                        else:
                            # Accumulate arguments
                            if current_tool_call:
                                accumulated_arguments += tc['function'].get('arguments', '')

            # Handle finish reason
            if choice.get('finish_reason'):
                # Yield any pending tool call
                if current_tool_call and accumulated_arguments:
                    current_tool_call['arguments'] = accumulated_arguments
                    yield GenerateResponse(
                        content="",
                        model=self.model,
                        finish_reason=choice['finish_reason'],
                        tool_calls=[current_tool_call]
                    )
                else:
                    yield GenerateResponse(
                        content="",
                        model=self.model,
                        finish_reason=choice['finish_reason']
                    )

    def _single_generate_transformers(self, input_text: str, max_new_tokens: int,
                                     temperature: float, top_p: float, seed: Optional[int] = None) -> GenerateResponse:
        """Generate single response using transformers (original implementation)"""
        try:
            # Set seed for deterministic generation if provided
            if seed is not None:
                try:
                    import torch
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except ImportError:
                    pass  # Skip seeding if torch not available

            # Track generation time
            start_time = time.time()
            
            outputs = self.pipeline(
                input_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                truncation=True,
                return_full_text=False
            )
            
            gen_time = round((time.time() - start_time) * 1000, 1)

            if outputs and len(outputs) > 0:
                response_text = outputs[0]['generated_text'].strip()

                # Calculate token usage using centralized utilities
                usage = self._calculate_usage(input_text, response_text)

                return GenerateResponse(
                    content=response_text,
                    model=self.model,
                    finish_reason="stop",
                    usage=usage,
                    gen_time=gen_time
                )
            else:
                return GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop",
                    gen_time=gen_time
                )

        except Exception as e:
            gen_time = round((time.time() - start_time) * 1000, 1) if 'start_time' in locals() else 0.0
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error",
                gen_time=gen_time
            )

    def _calculate_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate token usage using centralized token utilities."""
        from ..utils.token_utils import TokenUtils

        input_tokens = TokenUtils.estimate_tokens(prompt, self.model)
        output_tokens = TokenUtils.estimate_tokens(response, self.model)
        total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Keep legacy keys for backward compatibility
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens
        }

    def _stream_generate_transformers(self, input_text: str, max_new_tokens: int,
                                     temperature: float, top_p: float, tool_call_tags: Optional[str] = None, seed: Optional[int] = None) -> Iterator[GenerateResponse]:
        """Stream response using transformers (simulated, original implementation) with tool tag rewriting support"""
        try:
            # HuggingFace doesn't have native streaming, so we simulate it
            full_response = self._single_generate_transformers(input_text, max_new_tokens, temperature, top_p, seed)

            if full_response.content:
                # Apply tool tag rewriting if enabled
                content = full_response.content
                if tool_call_tags:
                    try:
                        from ..tools.tag_rewriter import create_tag_rewriter
                        rewriter = create_tag_rewriter(tool_call_tags)
                        content = rewriter.rewrite_text(content)
                    except ImportError:
                        pass
                
                words = content.split()
                for i, word in enumerate(words):
                    chunk_content = word + (" " if i < len(words) - 1 else "")
                    yield GenerateResponse(
                        content=chunk_content,
                        model=self.model,
                        finish_reason="stop" if i == len(words) - 1 else None
                    )
            else:
                yield GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop"
                )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_input_text_transformers(self, prompt: str, messages: Optional[List[Dict[str, str]]],
                                      system_prompt: Optional[str], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build input text for transformers model with tool support"""

        # Add tools to system prompt if provided
        final_system_prompt = system_prompt
        if tools and self.tool_handler.supports_prompted:
            include_tool_list = True
            if final_system_prompt and "## Tools (session)" in final_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if final_system_prompt:
                final_system_prompt += f"\n\n{tool_prompt}"
            else:
                final_system_prompt = tool_prompt

        # Check if model has chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template if available
            chat_messages = []

            if final_system_prompt:
                chat_messages.append({"role": "system", "content": final_system_prompt})

            if messages:
                chat_messages.extend(messages)

            chat_messages.append({"role": "user", "content": prompt})

            try:
                return self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback if chat template fails
                pass

        # Build simple conversational format
        text_parts = []

        if system_prompt:
            text_parts.append(f"System: {system_prompt}\n")

        if messages:
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                text_parts.append(f"{role}: {content}\n")

        text_parts.append(f"User: {prompt}\n")
        text_parts.append("Assistant:")

        return "".join(text_parts)

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities supported by this provider"""
        capabilities = ["chat", "streaming"]

        if self.model_type == "gguf":
            capabilities.append("gguf")
            if self.llm and self.llm.chat_format:
                capabilities.append("tools")
        else:
            # Check for specific model capabilities
            model_lower = self.model.lower()

            if "gpt2" in model_lower or "dialogpt" in model_lower:
                capabilities.append("dialogue")

            if "codegen" in model_lower or "starcoder" in model_lower or "coder" in model_lower:
                capabilities.append("code")

        return capabilities

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        if self.model_type == "gguf":
            return self.llm is not None
        else:
            return self.pipeline is not None


    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter appropriate for the model type"""
        max_output_tokens = kwargs.get("max_output_tokens", self.max_output_tokens)

        if self.model_type == "gguf":
            # For GGUF models, this is the generation limit
            return max_output_tokens
        else:
            # For transformers, this is max_new_tokens
            return max_output_tokens


    def _stream_generate_transformers_with_tools(self, input_text: str, max_new_tokens: int,
                                               temperature: float, top_p: float,
                                               tools: Optional[List[Dict[str, Any]]] = None,
                                               tool_call_tags: Optional[str] = None, seed: Optional[int] = None) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate_transformers(input_text, max_new_tokens, temperature, top_p, tool_call_tags, seed):
            collected_content += chunk.content
            yield chunk

        # Handle tool execution if we have tools and content
        if tools and self.tool_handler.supports_prompted and collected_content:
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop"
            )

            # Handle tool execution using base method
            final_response = self._handle_prompted_tool_execution(complete_response, tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )

    def _handle_tool_execution_gguf(self, response: GenerateResponse, tools: List[Dict[str, Any]], has_native_tools: bool) -> GenerateResponse:
        """Handle tool execution for GGUF responses - both native and prompted"""
        if has_native_tools and response.has_tool_calls():
            # Handle native tool calls using base method
            tool_calls = self._convert_native_tool_calls_to_standard(response.tool_calls)
            return self._execute_tools_with_events(response, tool_calls)
        elif self.tool_handler.supports_prompted and response.content:
            # Handle prompted tool calls using base method
            return self._handle_prompted_tool_execution(response, tools)

        return response

    def _stream_generate_gguf_with_tools(self, generation_kwargs: Dict[str, Any],
                                       tools: Optional[List[Dict[str, Any]]] = None,
                                       has_native_tools: bool = False,
                                       tool_call_tags: Optional[str] = None) -> Iterator[GenerateResponse]:
        """Stream generate GGUF with tool execution at the end"""
        collected_content = ""
        collected_tool_calls = []

        # Stream the response content
        for chunk in self._stream_generate_gguf(generation_kwargs, tool_call_tags):
            collected_content += chunk.content
            if chunk.tool_calls:
                collected_tool_calls.extend(chunk.tool_calls)
            yield chunk

        # Handle tool execution if we have tools and content/calls
        if tools and (collected_tool_calls or
                     (self.tool_handler.supports_prompted and collected_content)):
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop",
                tool_calls=collected_tool_calls
            )

            # Handle tool execution using simplified method
            final_response = self._handle_tool_execution_gguf(complete_response, tools, has_native_tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """
        List available HuggingFace models from local cache (excluding MLX models).

        Args:
            **kwargs: Optional parameters including:
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        try:
            from .model_capabilities import filter_models_by_capabilities

            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if not hf_cache.exists():
                return []

            models = []
            for item in hf_cache.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    # Convert models--microsoft--DialoGPT-medium to microsoft/DialoGPT-medium
                    model_name = item.name.replace("models--", "").replace("--", "/")

                    # CRITICAL: Exclude MLX models from HuggingFace list
                    # Any model with "mlx" in the name should be classified as MLX, not HuggingFace
                    if "mlx" not in model_name.lower():
                        models.append(model_name)

            models = sorted(models)

            # Apply new capability filtering if provided
            input_capabilities = kwargs.get('input_capabilities')
            output_capabilities = kwargs.get('output_capabilities')
            
            if input_capabilities or output_capabilities:
                models = filter_models_by_capabilities(
                    models, 
                    input_capabilities=input_capabilities,
                    output_capabilities=output_capabilities
                )


            return models

        except Exception:
            return []
