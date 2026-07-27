"""Capability plugins + facades (voice/audio/vision/music/scene3d).

This module provides a dependency-light integration surface for optional
capability packages (e.g. `abstractvoice`, `abstractvision`) without making
`abstractcore` a hard dependency sink.

Design constraints:
- No plugin imports at `import abstractcore` time.
- Plugins are discovered lazily via Python entry points.
- Plugins must be import-light; heavy ML stacks must not import at module import time.
"""

import threading
from typing import Any, Dict, List, Optional

from .errors import CapabilityUnavailableError
from .host import DefaultCapabilityHostContext, DefaultCoreTextGenerationService
from .registry import CapabilityRegistry


class _DiscoveryCapabilityOwner:
    """Minimal owner for the shared discovery registry (mirrors the server's
    `_ServerCapabilityOwner` precedent): capability TOOLS and policies need a
    registry, not a text-generation host. Backends that genuinely need an LLM
    owner are constructed by real hosts (AbstractCore instances, the server),
    never through this discovery singleton."""

    config: Dict[str, Any] = {}
    model: str = "abstractcore-discovery"

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise RuntimeError("The discovery capability owner does not expose text generation.")


_shared_registry: Optional[CapabilityRegistry] = None
_shared_registry_lock = threading.Lock()


def shared_capability_registry() -> CapabilityRegistry:
    """Process-wide registry for capability DISCOVERY surfaces (plugin tools,
    approval policies, availability facts) — the accessor upper packages
    (runtime/gateway) use so they never import capability plugin packages
    directly (layering rule, laurent dm#16 / commons c4210; seat: camera,
    draft for core's owner review). Lazily constructed under a lock (two
    threads racing first use must share ONE registry, adversary P1-1);
    entry-point plugins load on first read through the registry's own
    ensure-loaded path, which serializes the load itself."""
    global _shared_registry
    if _shared_registry is None:
        with _shared_registry_lock:
            if _shared_registry is None:
                _shared_registry = CapabilityRegistry(_DiscoveryCapabilityOwner())
    return _shared_registry


def capability_tools(capability: Optional[str] = None) -> Any:
    """Module-level convenience over the shared registry: the tools a
    capability plugin contributed (list of ToolDefinition, callables intact),
    `[]`/`{}` when absent. See `CapabilityRegistry.capability_tools`."""
    return shared_capability_registry().capability_tools(capability)


def capability_tool_facts(capability: str) -> Dict[str, Dict[str, bool]]:
    """Module-level convenience over the shared registry: the RISK FACTS a
    plugin declared for its tools ({tool: {fact: bool}}), `{}` when absent —
    factless tools derive as unvetted/top-band (fail-closed)."""
    return shared_capability_registry().capability_tool_facts(capability)


def capability_tool_policy(capability: str) -> Dict[str, List[str]]:
    """Module-level convenience over the shared registry: the approval
    partition a plugin registered ({"auto_approve": [...],
    "require_approval": [...]}), `{}` when absent — hosts fail closed."""
    return shared_capability_registry().capability_tool_policy(capability)
from .types import (
    ArtifactRef,
    ArtifactStoreLike,
    AudioCapability,
    BytesOrArtifactRef,
    CapabilityArtifactRef,
    CapabilityHostContext,
    CapabilityInvokeResult,
    CapabilityModelInfo,
    CapabilityOperationInfo,
    CapabilityProviderInfo,
    CoreTextGenerationService,
    CoreTextResult,
    GenerateWithOutputsResult,
    MusicCapability,
    CameraCapability,
    Scene3dCapability,
    VisionCapability,
    VoiceCapability,
    is_artifact_ref,
)
from .vision_catalog import get_local_vision_cache_catalog

__all__ = [
    "ArtifactRef",
    "ArtifactStoreLike",
    "AudioCapability",
    "capability_tool_facts",
    "capability_tool_policy",
    "capability_tools",
    "shared_capability_registry",
    "BytesOrArtifactRef",
    "CapabilityArtifactRef",
    "CapabilityHostContext",
    "CapabilityInvokeResult",
    "CapabilityModelInfo",
    "CapabilityOperationInfo",
    "CapabilityProviderInfo",
    "CapabilityRegistry",
    "CapabilityUnavailableError",
    "CoreTextGenerationService",
    "CoreTextResult",
    "DefaultCapabilityHostContext",
    "DefaultCoreTextGenerationService",
    "GenerateWithOutputsResult",
    "get_local_vision_cache_catalog",
    "MusicCapability",
    "CameraCapability",
    "Scene3dCapability",
    "VisionCapability",
    "VoiceCapability",
    "is_artifact_ref",
]
