from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..types import DecisionRequest, DecisionResponse


@runtime_checkable
class DecisionEngine(Protocol):
    """What every decision engine exposes to providers, the server and callers."""

    model_id: str
    provider: Optional[str]

    @property
    def backend_name(self) -> str: ...

    @property
    def precision(self) -> Optional[str]: ...

    @property
    def kind(self) -> str:
        """The decision mechanism: ``kev`` (pointer head), ``lev`` / ``openjev`` (label
        logits), ``laya`` (marker scorer), ``systemone`` (remote endpoint)."""
        ...

    def decide(
        self,
        state: Any = None,
        questions: Optional[Dict[str, Any]] = None,
        *,
        request: Optional[DecisionRequest] = None,
        truncate: Optional[bool] = None,
    ) -> DecisionResponse: ...

    def unload(self) -> None: ...
