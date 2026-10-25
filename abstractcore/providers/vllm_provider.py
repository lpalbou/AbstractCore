"""
vLLM provider implementation with advanced features.

vLLM exposes an OpenAI-compatible API (chat completions, models, embeddings) plus
additional management endpoints and request extensions:
- Guided Decoding: guided_regex, guided_json, guided_grammar
- Beam Search: best_of, use_beam_search
- Multi-LoRA management: load_adapter, unload_adapter, list_adapters

This provider subclasses `OpenAICompatibleProvider` and injects vLLM-specific request
extensions via `payload["extra_body"]`.

TODO(vllm): Add an opt-in integration test that exercises streaming + LoRA management endpoints
once a reachable vLLM server is available in the test environment.
"""

import warnings
from typing import Any, Dict, List, Optional

from .openai_compatible_provider import OpenAICompatibleProvider
from .base import ThinkingControlHandling


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
        """vLLM request extensions, sent as TOP-LEVEL request fields.

        WHY FLATTEN (2026-10-25). `extra_body` is an OpenAI-Python-SDK convention: the SDK
        merges its keys into the top level of the JSON body. This provider posts the JSON
        itself (httpx), so a nested `"extra_body": {...}` key reached vLLM literally — and
        vLLM's `OpenAIBaseModel` (`model_config = ConfigDict(extra="allow")` plus a
        validator that only *logs* unknown keys at debug level) silently IGNORES it.
        Every thinking control this provider sent (`chat_template_kwargs.enable_thinking`,
        the effort template variable, `thinking_token_budget`) therefore never reached the
        template while metadata reported it applied. vLLM's `ChatCompletionRequest` reads
        `chat_template_kwargs`, `reasoning_effort` and `thinking_token_budget` as
        top-level fields (vllm/entrypoints/openai/chat_completion/protocol.py, main,
        read 2026-10-25). Unverified live (no vLLM server on this machine).

        The superclass hook runs first, so a caller-provided `reasoning_effort=` /
        `chat_template_kwargs=` / `extra_body=` is forwarded too (it was dropped before:
        this override never called super).
        """
        payload = super()._mutate_payload(payload, **kwargs)

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

        extra_body = payload.pop("extra_body", None)
        merged: Dict[str, Any] = dict(extra_body) if isinstance(extra_body, dict) else {}
        merged.update(extra_body_updates)
        for key, value in merged.items():
            if key == "chat_template_kwargs" and isinstance(value, dict):
                # A caller's top-level chat_template_kwargs= wins per key.
                top = payload.get("chat_template_kwargs")
                payload["chat_template_kwargs"] = {**value, **(top if isinstance(top, dict) else {})}
            else:
                payload.setdefault(key, value)

        return payload

    def _apply_provider_thinking_kwargs(
        self,
        *,
        enabled: Optional[bool],
        level: Optional[str],
        kwargs: Dict[str, Any],
    ) -> tuple[Dict[str, Any], ThinkingControlHandling]:
        """Map the unified `thinking=` control to vLLM request fields (asset-driven).

        vLLM renders the model's own chat template server-side
        (https://docs.vllm.ai/en/latest/features/reasoning_outputs/):
        - `chat_template_kwargs` reach the template; request-level kwargs win over server
          defaults. The boolean switch is sent under the name the model DECLARES
          (`thinking_control.template_kwarg`) — never a guessed `enable_thinking` for a
          template that has no such variable (that was a claim with no effect);
        - `reasoning_effort` (top level; none|minimal|low|medium|high|xhigh|max) is passed
          to the template AND mapped by vLLM to `enable_thinking` (none -> false) when the
          request carries no explicit `enable_thinking`; used for models that declare
          `reasoning_levels` but no template switch (DeepSeek V4, Mistral Small 4, GPT-OSS);
        - `thinking_token_budget` (top level) caps reasoning tokens ONLY when the server
          runs with `--reasoning-parser`; it is sent as a best-effort cap for levels and is
          never by itself the basis of a level claim (a budget is not the model's effort).

        The fields are stashed under `extra_body` here and flattened to the top level by
        `_mutate_payload`.
        """
        if enabled is None and level is None:
            return kwargs, ThinkingControlHandling()

        surfaces = self._thinking_control_surfaces()
        declared = self._model_reasoning_levels()
        new_kwargs = dict(kwargs)
        extra_body = new_kwargs.get("extra_body")
        extra_body_dict: Dict[str, Any] = dict(extra_body) if isinstance(extra_body, dict) else {}
        ctk = extra_body_dict.get("chat_template_kwargs")
        ctk_dict: Dict[str, Any] = dict(ctk) if isinstance(ctk, dict) else {}

        def _done(handled_ed: bool, handled_lvl: bool) -> tuple[Dict[str, Any], ThinkingControlHandling]:
            if ctk_dict:
                extra_body_dict["chat_template_kwargs"] = ctk_dict
            if extra_body_dict:
                new_kwargs["extra_body"] = extra_body_dict
            return new_kwargs, ThinkingControlHandling(
                handled_enable_disable=handled_ed, handled_level=handled_lvl
            )

        # Harmony (GPT-OSS): vLLM builds the harmony SYSTEM message itself from the
        # request's `reasoning_effort`; the `Reasoning:` line BaseProvider injects lands
        # in a developer/system turn instead. Send the effort natively too.
        arch = self.architecture_config if isinstance(self.architecture_config, dict) else {}
        caps = self.model_capabilities if isinstance(self.model_capabilities, dict) else {}
        msg_fmt = str(arch.get("message_format") or "").strip().lower()
        resp_fmt = str(caps.get("response_format") or "").strip().lower()
        if msg_fmt == "harmony" or resp_fmt == "harmony":
            effort = level if isinstance(level, str) and level else ("low" if enabled is False else "medium")
            if "reasoning_effort" not in kwargs:
                new_kwargs["reasoning_effort"] = effort
            return _done(True, True)

        # Effort SWITCH (`thinking_control.effort_disable_value`, HY V3 / Hy-Memory
        # `reasoning_effort=no_think|low|high`): the template has no `enable_thinking`,
        # and an unset effort renders no_think. Send only the effort value; decline when
        # there is no honest value (an undeclared level).
        if surfaces.effort_is_switch():
            from ..architectures.thinking_controls import effort_switch_value

            value = effort_switch_value(surfaces, enabled=enabled, level=level, declared_levels=declared)
            if value is None:
                return kwargs, ThinkingControlHandling()
            ctk_dict[surfaces.effort_template_kwarg] = value
            return _done(True, enabled is not False and isinstance(level, str) and bool(level))

        handled_ed = False
        handled_lvl = False
        if surfaces.template_kwarg:
            ctk_dict[surfaces.template_kwarg] = bool(enabled) if enabled is not None else True
            handled_ed = True
            if enabled is not False and isinstance(level, str) and level:
                # Nemotron-style "low effort" boolean (thinking stays on, fewer tokens).
                if surfaces.low_effort_template_kwarg and level in {"minimal", "low"}:
                    ctk_dict[surfaces.low_effort_template_kwarg] = True
                    handled_lvl = True
                # String effort template variable (Qwen3.8 `reasoning_effort`), gated on a
                # declared level so a template that raises on unknown efforts never sees one.
                if surfaces.effort_template_kwarg and (not declared or level in declared):
                    ctk_dict[surfaces.effort_template_kwarg] = level
                    handled_lvl = True
        elif surfaces.budget_template_kwarg:
            # Checked BEFORE declared levels: the budget variable is what the template reads
            # (Seed-OSS declares levels too, but its template ignores reasoning_effort).
            # Integer budget template variable (Seed-OSS `thinking_budget`).
            budget_map = {"minimal": 256, "low": 512, "medium": 1024, "high": 4096, "xhigh": 8192}
            if enabled is False:
                ctk_dict[surfaces.budget_template_kwarg] = 0
            else:
                ctk_dict[surfaces.budget_template_kwarg] = int(budget_map.get(str(level or ""), 1024))
                handled_lvl = isinstance(level, str) and level in budget_map
            handled_ed = True
        elif declared:
            # Levels without a template switch: the OpenAI-standard top-level field.
            effort: Optional[str] = None
            if enabled is False:
                if "none" in declared:
                    effort = "none"
                else:
                    effort = next((x for x in ("minimal", "low", "medium", "high", "xhigh") if x in declared), None)
                    if effort:
                        warnings.warn(
                            f"thinking='off' requested for model '{self.model}', but reasoning_effort cannot "
                            f"be fully disabled (supported: {declared}); using reasoning_effort={effort!r}.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
            elif isinstance(level, str) and level:
                effort = level
            else:
                effort = next((x for x in ("medium", "high", "low") if x in declared), None)
            if effort:
                if "reasoning_effort" not in kwargs:
                    new_kwargs["reasoning_effort"] = effort
                handled_ed = True
                handled_lvl = True
        else:
            # No declared surface (or a prompt-token-only one, which BaseProvider applies):
            # nothing honest to send.
            return kwargs, ThinkingControlHandling()

        # Best-effort per-request reasoning token cap (needs --reasoning-parser server-side).
        if enabled is not False and isinstance(level, str) and level:
            budget = {"minimal": 256, "low": 512, "medium": 1024, "high": 4096, "xhigh": 8192}.get(level)
            if budget is not None and not surfaces.budget_template_kwarg:
                extra_body_dict["thinking_token_budget"] = int(budget)

        return _done(handled_ed, handled_lvl)

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
