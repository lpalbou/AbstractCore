"""Capability plugins + facades (voice/audio/vision/music).

This module provides a dependency-light integration surface for optional
capability packages (e.g. `abstractvoice`, `abstractvision`) without making
`abstractcore` a hard dependency sink.

Design constraints:
- No plugin imports at `import abstractcore` time.
- Plugins are discovered lazily via Python entry points.
- Plugins must be import-light; heavy ML stacks must not import at module import time.
"""

from .errors import CapabilityUnavailableError
from .host import DefaultCapabilityHostContext, DefaultCoreTextGenerationService
from .registry import CapabilityRegistry
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
    VisionCapability,
    VoiceCapability,
    is_artifact_ref,
)
from .vision_catalog import get_local_vision_cache_catalog

__all__ = [
    "ArtifactRef",
    "ArtifactStoreLike",
    "AudioCapability",
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
    "VisionCapability",
    "VoiceCapability",
    "is_artifact_ref",
]
