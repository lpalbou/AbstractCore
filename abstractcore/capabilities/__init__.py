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
from .registry import CapabilityRegistry
from .types import (
    ArtifactRef,
    ArtifactStoreLike,
    AudioCapability,
    BytesOrArtifactRef,
    GenerateWithOutputsResult,
    MusicCapability,
    VisionCapability,
    VoiceCapability,
    is_artifact_ref,
)

__all__ = [
    "ArtifactRef",
    "ArtifactStoreLike",
    "AudioCapability",
    "BytesOrArtifactRef",
    "CapabilityRegistry",
    "CapabilityUnavailableError",
    "GenerateWithOutputsResult",
    "MusicCapability",
    "VisionCapability",
    "VoiceCapability",
    "is_artifact_ref",
]

