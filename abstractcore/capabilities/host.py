from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from .types import CoreTextResult


class DefaultCoreTextGenerationService:
    """Narrow text-only service exposed to capability plugins by an AbstractCore host."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def _enter(self) -> None:
        if bool(getattr(self._owner, "_capability_host_text_generation_active", False)):
            raise RuntimeError("Capability host text generation cannot be re-entered recursively.")
        setattr(self._owner, "_capability_host_text_generation_active", True)

    def _exit(self) -> None:
        try:
            setattr(self._owner, "_capability_host_text_generation_active", False)
        except Exception:
            pass

    def _base_kwargs(
        self,
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Union[bool, str]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "stream": False,
            "media": None,
            "output": None,
        }
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if thinking is not None:
            kwargs["thinking"] = thinking
        return kwargs

    def generate_text(
        self,
        prompt: str = "",
        *,
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking: Optional[Union[bool, str]] = None,
        purpose: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CoreTextResult:
        self._enter()
        try:
            kwargs = self._base_kwargs(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                thinking=thinking,
            )
            response = self._owner.generate(
                prompt=str(prompt or ""),
                messages=[dict(m) for m in messages] if messages is not None else None,
                system_prompt=system_prompt,
                **kwargs,
            )
        finally:
            self._exit()

        content = getattr(response, "content", response)
        meta: Dict[str, Any] = {}
        if isinstance(getattr(response, "metadata", None), dict):
            meta.update(getattr(response, "metadata"))
        if purpose:
            meta["purpose"] = str(purpose)
        if metadata:
            meta["request_metadata"] = dict(metadata)
        meta["text_only"] = True
        meta["service"] = "abstractcore.host_text/v1"
        return CoreTextResult(
            content=str(content or ""),
            model=getattr(response, "model", getattr(self._owner, "model", None)),
            usage=dict(getattr(response, "usage", {}) or {}) if getattr(response, "usage", None) else None,
            finish_reason=getattr(response, "finish_reason", None),
            metadata=meta,
        )

    def generate_structured(
        self,
        prompt: str = "",
        *,
        response_model: Optional[Any] = None,
        json_schema: Optional[Mapping[str, Any]] = None,
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        purpose: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        if response_model is None:
            if json_schema is not None:
                raise ValueError("json_schema structured host text generation is not exposed by this service yet; use response_model.")
            return self.generate_text(
                prompt,
                messages=messages,
                system_prompt=system_prompt,
                purpose=purpose,
                metadata=metadata,
            )

        self._enter()
        try:
            return self._owner.generate(
                prompt=str(prompt or ""),
                messages=[dict(m) for m in messages] if messages is not None else None,
                system_prompt=system_prompt,
                response_model=response_model,
                stream=False,
                media=None,
                output=None,
            )
        finally:
            self._exit()


@dataclass(frozen=True)
class DefaultCapabilityHostContext:
    owner: Any

    @property
    def text(self) -> DefaultCoreTextGenerationService:
        return DefaultCoreTextGenerationService(self.owner)

    def service(self, name: str) -> Any:
        service_name = str(name or "").strip().lower()
        if service_name in {"text", "text_generation", "core_text", "abstractcore.text"}:
            return self.text
        raise KeyError(f"Unknown capability host service: {name}")
