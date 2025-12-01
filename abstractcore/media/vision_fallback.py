"""
Vision Fallback System for Text-Only Models

Implements two-stage pipeline: vision model → description → text-only model
Uses unified AbstractCore configuration system.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)


class VisionNotConfiguredError(Exception):
    """Raised when vision fallback is requested but not configured."""
    pass


class VisionFallbackHandler:
    """
    Handles vision fallback for text-only models using two-stage pipeline.

    When a text-only model receives an image:
    1. Uses configured vision model to generate description
    2. Provides description to text-only model for processing

    Uses the unified AbstractCore configuration system.
    """

    def __init__(self, config_manager=None):
        """Initialize with configuration manager."""
        if config_manager is None:
            try:
                from abstractcore.config import get_config_manager
                self.config_manager = get_config_manager()
            except ImportError:
                # Config module not available - use fallback behavior
                logger.warning("Config module not available, vision fallback disabled")
                self.config_manager = None
        else:
            self.config_manager = config_manager

    @property
    def vision_config(self):
        """Get vision configuration from unified config system."""
        if self.config_manager is None:
            # Return a minimal config object when config system is not available
            class FallbackVisionConfig:
                strategy = "disabled"
                caption_provider = None
                caption_model = None
                fallback_chain = []
                local_models_path = None
            return FallbackVisionConfig()
        return self.config_manager.config.vision

    def create_description(self, image_path: str, user_prompt: str = None) -> str:
        """
        Generate description using configured vision model.

        Args:
            image_path: Path to the image file
            user_prompt: Original user prompt for context

        Returns:
            Description string to be used by text-only model

        Raises:
            VisionNotConfiguredError: When vision fallback is not configured
        """
        if self.vision_config.strategy == "disabled":
            raise VisionNotConfiguredError("Vision fallback is disabled")

        if not self._has_vision_capability():
            raise VisionNotConfiguredError("No vision capability configured")

        try:
            return self._generate_with_fallback(image_path)
        except Exception as e:
            logger.debug(f"Vision fallback failed: {e}")
            raise VisionNotConfiguredError(f"Vision fallback generation failed: {e}")

    def _has_vision_capability(self) -> bool:
        """Check if any vision capability is configured."""
        return (
            (self.vision_config.caption_provider is not None and
             self.vision_config.caption_model is not None) or
            len(self.vision_config.fallback_chain) > 0 or
            self._has_local_models()
        )

    def _has_local_models(self) -> bool:
        """Check if any local vision models are available."""
        models_dir = Path(self.vision_config.local_models_path).expanduser()
        return models_dir.exists() and any(models_dir.iterdir())

    def _generate_with_fallback(self, image_path: str) -> str:
        """Try vision models in fallback chain order."""
        # Try primary provider first
        if self.vision_config.caption_provider and self.vision_config.caption_model:
            try:
                description = self._generate_description(
                    self.vision_config.caption_provider,
                    self.vision_config.caption_model,
                    image_path
                )
                return description
            except Exception as e:
                logger.debug(f"Primary vision provider failed: {e}")

        # Try fallback chain
        for provider_config in self.vision_config.fallback_chain:
            try:
                description = self._generate_description(
                    provider_config["provider"],
                    provider_config["model"],
                    image_path
                )
                return description
            except Exception as e:
                logger.debug(f"Vision provider {provider_config} failed: {e}")
                continue

        # Try local models
        if self._has_local_models():
            try:
                description = self._generate_local_description(image_path)
                return description
            except Exception as e:
                logger.debug(f"Local vision model failed: {e}")

        raise Exception("All vision fallback providers failed")

    def _generate_description(self, provider: str, model: str, image_path: str) -> str:
        """Generate description using specified provider and model."""
        try:
            # Import here to avoid circular imports
            from abstractcore import create_llm

            vision_llm = create_llm(provider, model=model)
            response = vision_llm.generate(
                "Provide a detailed description of this image in 3-4 sentences. Be precise about specific landmarks, buildings, objects, and details. If you recognize specific places or things, name them accurately. Describe naturally without phrases like 'this image shows'.",
                media=[image_path]
            )
            return response.content.strip()
        except Exception as e:
            logger.debug(f"Failed to generate description with {provider}/{model}: {e}")
            raise

    def _generate_local_description(self, image_path: str) -> str:
        """Generate description using local vision model."""
        try:
            models_dir = Path(self.vision_config.local_models_path).expanduser()

            # Look for downloaded vision models
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and ("caption" in model_dir.name.lower() or "blip" in model_dir.name.lower() or "vit" in model_dir.name.lower() or "git" in model_dir.name.lower()):
                    try:
                        # Check if download is complete
                        if not (model_dir / "download_complete.txt").exists():
                            logger.debug(f"Model {model_dir.name} download incomplete")
                            continue

                        description = self._use_local_model(model_dir, image_path)
                        if description:
                            return description

                    except Exception as e:
                        logger.debug(f"Local model {model_dir} failed: {e}")
                        continue

            raise Exception("No working local models found")
        except ImportError:
            raise Exception("transformers library not available for local models")

    def _use_local_model(self, model_dir: Path, image_path: str) -> str:
        """Use a specific local model to generate description."""
        from PIL import Image

        model_name = model_dir.name

        if "blip" in model_name:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            # Load BLIP model and processor
            processor = BlipProcessor.from_pretrained(model_dir / "processor", use_fast=False)
            model = BlipForConditionalGeneration.from_pretrained(model_dir / "model")

            # Process image
            image = Image.open(image_path).convert('RGB')
            inputs = processor(image, return_tensors="pt")

            # Generate description
            out = model.generate(**inputs, max_length=50, num_beams=5)
            description = processor.decode(out[0], skip_special_tokens=True)
            return description

        elif "vit-gpt2" in model_name:
            from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

            # Load ViT-GPT2 components
            model = VisionEncoderDecoderModel.from_pretrained(model_dir / "model")
            feature_extractor = ViTImageProcessor.from_pretrained(model_dir / "feature_extractor")
            tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

            # Process image
            image = Image.open(image_path).convert('RGB')
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

            # Generate description
            output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
            description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return description

        elif "git" in model_name:
            from transformers import GitProcessor, GitForCausalLM

            # Load GIT model and processor
            processor = GitProcessor.from_pretrained(model_dir / "processor")
            model = GitForCausalLM.from_pretrained(model_dir / "model")

            # Process image
            image = Image.open(image_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")

            # Generate description
            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
            description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return description

        else:
            # Try generic image-to-text pipeline
            from transformers import pipeline
            captioner = pipeline("image-to-text", model=str(model_dir))
            result = captioner(image_path)
            if result and len(result) > 0:
                return result[0]["generated_text"]

        return None

    def _show_setup_instructions(self) -> str:
        """Return helpful setup instructions for users."""
        return """⚠️  Vision capability not configured for text-only models.

To enable image analysis with text-only models:
1. Download local model: abstractcore --download-vision-model
2. Use existing model: abstractcore --set-vision-caption qwen2.5vl:7b
3. Use cloud API: abstractcore --set-vision-provider openai --model gpt-4o
4. Interactive setup: abstractcore --configure

Current status: abstractcore --status"""

    def get_status(self) -> Dict[str, Any]:
        """Get current vision configuration status using unified config."""
        return self.config_manager.get_status()["vision"]

    def is_enabled(self) -> bool:
        """Check if vision fallback is enabled and configured."""
        return (self.vision_config.strategy == "two_stage" and
                self._has_vision_capability())


# Convenience functions for easy integration
def has_vision_capability() -> bool:
    """Check if vision fallback is configured and enabled."""
    handler = VisionFallbackHandler()
    return handler.is_enabled()


def create_image_description(image_path: str, user_prompt: str = None) -> str:
    """Create image description for text-only models."""
    handler = VisionFallbackHandler()
    return handler.create_description(image_path, user_prompt)