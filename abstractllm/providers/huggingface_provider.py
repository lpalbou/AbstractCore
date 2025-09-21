"""
HuggingFace provider implementation.
"""

from typing import List, Dict, Any, Optional, Union, Iterator
from .base import BaseProvider
from ..core.types import GenerateResponse
from ..exceptions import ModelNotFoundError
from ..utils.simple_model_discovery import get_available_models, format_model_error

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Transformers provider"""

    def __init__(self, model: str = "microsoft/DialoGPT-medium", device: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace dependencies not installed. Install with: pip install transformers torch")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model_instance = None
        self.pipeline = None

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load HuggingFace model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.model_instance = AutoModelForCausalLM.from_pretrained(self.model)

            # Move to device
            if self.device == "cuda":
                self.model_instance = self.model_instance.to(self.device)

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_instance,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

        except Exception as e:
            error_str = str(e).lower()
            if ('not found' in error_str or 'does not exist' in error_str or
                'not a valid model identifier' in error_str):
                # Model not found - show available models from local cache
                available_models = get_available_models("huggingface")
                error_message = format_model_error("HuggingFace", self.model, available_models)
                raise ModelNotFoundError(error_message)
            else:
                raise RuntimeError(f"Failed to load HuggingFace model {self.model}: {str(e)}")

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
        """Generate response using HuggingFace model"""

        if not self.pipeline:
            return GenerateResponse(
                content="Error: HuggingFace model not loaded",
                model=self.model,
                finish_reason="error"
            )

        # Build input text
        input_text = self._build_input_text(prompt, messages, system_prompt)

        # Generation parameters
        max_length = kwargs.get("max_tokens", 2048) + len(self.tokenizer.encode(input_text))
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)

        try:
            # Generate response
            outputs = self.pipeline(
                input_text,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                # Remove the input prompt from the output
                response_text = generated_text[len(input_text):].strip()

                # Calculate token usage
                input_tokens = len(self.tokenizer.encode(input_text))
                output_tokens = len(self.tokenizer.encode(response_text))

                return GenerateResponse(
                    content=response_text,
                    model=self.model,
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                )
            else:
                return GenerateResponse(
                    content="",
                    model=self.model,
                    finish_reason="stop"
                )

        except Exception as e:
            return GenerateResponse(
                content=f"Error generating response: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_input_text(self, prompt: str, messages: Optional[List[Dict[str, str]]],
                         system_prompt: Optional[str]) -> str:
        """Build input text for the model"""

        # Check if model has chat template
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Use chat template if available
            chat_messages = []

            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})

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
        capabilities = ["chat"]

        # Check for specific model capabilities
        model_lower = self.model.lower()

        if "gpt2" in model_lower or "dialogpt" in model_lower:
            capabilities.append("dialogue")

        if "codegen" in model_lower or "starcoder" in model_lower:
            capabilities.append("code")

        return capabilities

    def validate_config(self) -> bool:
        """Validate provider configuration"""
        return self.pipeline is not None

    def get_token_limit(self) -> Optional[int]:
        """Get maximum token limit for this model"""

        # Common model token limits
        token_limits = {
            "gpt2": 1024,
            "gpt2-medium": 1024,
            "gpt2-large": 1024,
            "gpt2-xl": 1024,
            "microsoft/DialoGPT-small": 1024,
            "microsoft/DialoGPT-medium": 1024,
            "microsoft/DialoGPT-large": 1024,
            "bigscience/bloom-560m": 2048,
            "bigscience/bloom-1b7": 2048,
            "bigscience/bloom-3b": 2048,
            "bigscience/bloom-7b1": 2048,
        }

        # Check if model matches known limits
        for model_pattern, limit in token_limits.items():
            if model_pattern in self.model.lower():
                return limit

        # Default to 2048 for unknown models
        return 2048