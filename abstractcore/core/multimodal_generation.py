"""Small result types for opt-in multimodal generation.

These types intentionally live outside ``GenerateResponse`` so existing text
generation keeps its long-standing return shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import GenerateResponse


@dataclass
class GeneratedItem:
    """Generated binary/media output, such as an image or speech audio."""

    modality: str
    task: str
    data: Optional[bytes] = None
    artifact_ref: Optional[Dict[str, Any]] = None
    content_type: Optional[str] = None
    format: Optional[str] = None
    backend_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedResource:
    """Reusable resource produced by generation, such as a cloned voice id."""

    modality: str
    task: str
    resource_type: str
    resource_id: str
    name: Optional[str] = None
    backend_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    artifact_ref: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationIssue:
    """Bounded warning/error record for multimodal generation."""

    modality: str
    task: str
    message: str
    type: str = "error"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalGenerateResponse:
    """Response returned only when ``generate(..., output=...)`` is used."""

    text: Optional[GenerateResponse] = None
    outputs: Dict[str, List[GeneratedItem]] = field(default_factory=dict)
    resources: Dict[str, List[GeneratedResource]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[GenerationIssue] = field(default_factory=list)
    errors: List[GenerationIssue] = field(default_factory=list)

    @property
    def content(self) -> Optional[str]:
        """Expose generated text content for session/history compatibility."""
        if self.text is not None:
            return self.text.content
        content = self.metadata.get("content")
        return str(content) if content is not None else None

    def add_output(self, modality: str, item: GeneratedItem) -> None:
        self.outputs.setdefault(str(modality), []).append(item)

    def add_resource(self, modality: str, resource: GeneratedResource) -> None:
        self.resources.setdefault(str(modality), []).append(resource)


__all__ = [
    "GeneratedItem",
    "GeneratedResource",
    "GenerationIssue",
    "MultimodalGenerateResponse",
]
