"""
Media enrichment metadata contract (v0).

This module defines a provider-agnostic, response-level transparency hook for
"input enrichment fallbacks" where non-text inputs (image/audio/video) are
converted into short grounded text context (caption/transcript/frames) so a
text-only model can proceed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


MEDIA_ENRICHMENT_KEY = "media_enrichment"

EnrichmentStatus = Literal["used", "skipped", "error"]
BackendKind = Literal["llm", "plugin", "local_model", "unknown"]


class EnrichmentBackend(TypedDict, total=False):
    kind: BackendKind
    provider: str
    model: str
    backend_id: str
    source: str


class MediaEnrichmentItem(TypedDict, total=False):
    status: EnrichmentStatus
    input_modality: str  # image|audio|video (string for forward-compat)
    input_index: int
    input_name: str
    policy: str
    summary_kind: str  # caption|transcript|frames (string for forward-compat)
    backend: EnrichmentBackend
    injected_text: str
    injected_chars: int
    artifact: Dict[str, Any]
    error: str


def build_enrichment_item(
    *,
    status: EnrichmentStatus,
    input_modality: str,
    summary_kind: str,
    policy: Optional[str] = None,
    backend: Optional[Dict[str, Any]] = None,
    input_index: Optional[int] = None,
    input_name: Optional[str] = None,
    injected_text: Optional[str] = None,
    artifact: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> MediaEnrichmentItem:
    item: MediaEnrichmentItem = {
        "status": status,
        "input_modality": str(input_modality),
        "summary_kind": str(summary_kind),
    }
    if isinstance(policy, str) and policy.strip():
        item["policy"] = policy.strip()
    if isinstance(backend, dict) and backend:
        # Preserve only JSON-serializable primitives; callers should keep this small.
        cleaned: Dict[str, Any] = {}
        for k in ("kind", "provider", "model", "backend_id", "source"):
            v = backend.get(k)
            if isinstance(v, str) and v.strip():
                cleaned[str(k)] = v.strip()
        if cleaned:
            item["backend"] = cleaned  # type: ignore[assignment]
    if isinstance(input_index, int):
        item["input_index"] = int(input_index)
    if isinstance(input_name, str) and input_name.strip():
        item["input_name"] = input_name.strip()
    if isinstance(injected_text, str) and injected_text.strip():
        txt = injected_text.strip()
        item["injected_text"] = txt
        item["injected_chars"] = len(txt)
    if isinstance(artifact, dict) and artifact:
        item["artifact"] = artifact
    if isinstance(error, str) and error.strip():
        item["error"] = error.strip()
    return item


def merge_enrichment_metadata(
    metadata: Optional[Dict[str, Any]],
    enrichments: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = metadata if isinstance(metadata, dict) else {}
    if not enrichments:
        return out

    existing = out.get(MEDIA_ENRICHMENT_KEY)
    if not isinstance(existing, list):
        existing = []
        out[MEDIA_ENRICHMENT_KEY] = existing

    for item in enrichments:
        if isinstance(item, dict) and item:
            existing.append(item)

    return out

