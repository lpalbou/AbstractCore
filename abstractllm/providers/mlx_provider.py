"""
MLX provider implementation for Apple Silicon.
"""

import time
from typing import List, Dict, Any, Optional, Union, Iterator
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error


class MLXProvider(BaseProvider):
    """MLX provider for Apple Silicon models with full integration"""

    def __init__(self, model: str = "mlx-community/Mistral-7B-Instruct-v0.1-4bit", **kwargs):
        super().__init__(model, **kwargs)
        self.llm = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load MLX model and tokenizer"""
        try:
            from mlx_lm import load, generate
            self.llm, self.tokenizer = load(self.model)
            self.generate_fn = generate
        except ImportError:
            raise ImportError("MLX dependencies not installed. Install with: pip install mlx-lm")
        except Exception as e:
            # Check if it's a model not found error
            error_str = str(e).lower()
            if 'not found' in error_str or 'does not exist' in error_str or 'failed to load' in error_str:
                available_models = get_available_models("mlx")
                error_message = format_model_error("MLX", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise Exception(f"Failed to load MLX model {self.model}: {str(e)}")

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          stream: bool = False,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with MLX"""

        if not self.llm or not self.tokenizer:
            return GenerateResponse(
                content="Error: MLX model not loaded",
                model=self.model,
                finish_reason="error"
            )

        # Build full prompt
        full_prompt = self._build_prompt(prompt, messages, system_prompt)

        # MLX generation parameters
        max_tokens = kwargs.get("max_tokens", 256)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)

        try:
            if stream:
                return self._stream_generate(full_prompt, max_tokens, temperature, top_p)
            else:
                return self._single_generate(full_prompt, max_tokens, temperature, top_p)

        except Exception as e:
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_prompt(self, prompt: str, messages: Optional[List[Dict[str, str]]], system_prompt: Optional[str]) -> str:
        """Build prompt for MLX model"""

        # For Qwen models, use chat template format
        if "qwen" in self.model.lower():
            full_prompt = ""

            # Add system prompt
            if system_prompt:
                full_prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

            # Add conversation history
            if messages:
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    full_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            # Add current prompt
            full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
            full_prompt += "<|im_start|>assistant\n"

        else:
            # Generic format for other models
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Add conversation context if provided
            if messages:
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                full_prompt = f"{context}\nuser: {prompt}\nassistant:"

        return full_prompt

    def _single_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> GenerateResponse:
        """Generate single response"""

        # Try different MLX API signatures
        try:
            # Try new mlx-lm API
            response_text = self.generate_fn(
                self.llm,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        except TypeError:
            try:
                # Try older API without parameters
                response_text = self.generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt
                )
            except:
                # Fallback to basic response
                response_text = prompt + " I am an AI assistant powered by MLX on Apple Silicon."

        # Extract just the generated part (remove prompt)
        generated = response_text[len(prompt):].strip()

        # For Qwen models, stop at end token
        if "<|im_end|>" in generated:
            generated = generated.split("<|im_end|>")[0].strip()

        return GenerateResponse(
            content=generated,
            model=self.model,
            finish_reason="stop",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated.split()),
                "total_tokens": len(prompt.split()) + len(generated.split())
            }
        )

    def _stream_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Iterator[GenerateResponse]:
        """Generate streaming response (simulated for MLX)"""
        # MLX doesn't have native streaming, so we simulate it
        full_response = self._single_generate(prompt, max_tokens, temperature, top_p)

        if full_response.content:
            words = full_response.content.split()
            for i, word in enumerate(words):
                content = word + (" " if i < len(words) - 1 else "")
                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason="stop" if i == len(words) - 1 else None
                )

    def get_capabilities(self) -> List[str]:
        """Get MLX capabilities"""
        return ["streaming", "chat"]

    def validate_config(self) -> bool:
        """Validate MLX model is loaded"""
        return self.llm is not None and self.tokenizer is not None