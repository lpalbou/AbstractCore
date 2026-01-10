"""
vLLM provider implementation with advanced features.

vLLM exposes an OpenAI-compatible API (chat completions, models, embeddings) plus
additional management endpoints and request extensions:
- Guided Decoding: guided_regex, guided_json, guided_grammar
- Beam Search: best_of, use_beam_search
- Multi-LoRA management: load_adapter, unload_adapter, list_adapters

This provider subclasses `OpenAICompatibleProvider` and injects vLLM-specific request
extensions via `payload["extra_body"]`.
"""

from typing import Any, Dict, List, Optional

from .openai_compatible_provider import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM provider for high-throughput GPU inference with advanced features."""

    PROVIDER_ID = "vllm"
    PROVIDER_DISPLAY_NAME = "vLLM"
    BASE_URL_ENV_VAR = "VLLM_BASE_URL"
    API_KEY_ENV_VAR = "VLLM_API_KEY"  # Optional; some deployments sit behind auth
    DEFAULT_BASE_URL = "http://localhost:8000/v1"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, base_url=base_url, api_key=api_key, **kwargs)

    def _mutate_payload(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        extra_body_updates: Dict[str, Any] = {}

        guided_regex = kwargs.get("guided_regex")
        if guided_regex:
            extra_body_updates["guided_regex"] = guided_regex

        guided_json = kwargs.get("guided_json")
        if guided_json:
            extra_body_updates["guided_json"] = guided_json

        guided_grammar = kwargs.get("guided_grammar")
        if guided_grammar:
            extra_body_updates["guided_grammar"] = guided_grammar

        best_of = kwargs.get("best_of")
        use_beam_search = kwargs.get("use_beam_search", False)
        if use_beam_search or best_of:
            extra_body_updates["use_beam_search"] = bool(use_beam_search)
            if best_of is not None:
                extra_body_updates["best_of"] = best_of

        # Allow callers to pass raw extra_body (merge with our computed updates).
        caller_extra_body = kwargs.get("extra_body")
        if isinstance(caller_extra_body, dict) and caller_extra_body:
            extra_body_updates = {**caller_extra_body, **extra_body_updates}

        if extra_body_updates:
            existing = payload.get("extra_body")
            if isinstance(existing, dict) and existing:
                payload["extra_body"] = {**existing, **extra_body_updates}
            else:
                payload["extra_body"] = extra_body_updates

        return payload

    # vLLM-specific methods

    def load_adapter(self, adapter_name: str, adapter_path: str) -> str:
        """
        Load a LoRA adapter dynamically without restarting the server.

        Args:
            adapter_name: Name to identify the adapter (e.g., "sql-expert")
            adapter_path: Path to the LoRA adapter weights

        Returns:
            Success message
        """
        management_url = self.base_url.rstrip("/").replace("/v1", "")

        response = self.client.post(
            f"{management_url}/v1/load_lora_adapter",
            json={"lora_name": adapter_name, "lora_path": adapter_path},
            headers=self._get_headers(),
        )
        self._raise_for_status(response, request_url=f"{management_url}/v1/load_lora_adapter")
        return f"Adapter '{adapter_name}' loaded successfully"

    def unload_adapter(self, adapter_name: str) -> str:
        """Unload a LoRA adapter from memory."""
        management_url = self.base_url.rstrip("/").replace("/v1", "")

        response = self.client.post(
            f"{management_url}/v1/unload_lora_adapter",
            json={"lora_name": adapter_name},
            headers=self._get_headers(),
        )
        self._raise_for_status(response, request_url=f"{management_url}/v1/unload_lora_adapter")
        return f"Adapter '{adapter_name}' unloaded successfully"

    def list_adapters(self) -> List[str]:
        """List currently loaded LoRA adapters."""
        management_url = self.base_url.rstrip("/").replace("/v1", "")

        response = self.client.get(
            f"{management_url}/v1/lora_adapters",
            headers=self._get_headers(),
        )
        self._raise_for_status(response, request_url=f"{management_url}/v1/lora_adapters")
        data = response.json()
        if isinstance(data, dict):
            adapters = data.get("adapters", [])
            return adapters if isinstance(adapters, list) else []
        return []

    # Standard AbstractCore methods

    def get_capabilities(self) -> List[str]:
        """Get vLLM capabilities."""
        capabilities = ["streaming", "chat", "tools", "structured_output"]
        capabilities.extend(["guided_decoding", "multi_lora", "beam_search"])
        return capabilities

