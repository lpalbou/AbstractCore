"""Scene3D backend selector helpers.

These helpers are intentionally dependency-light so they can be reused by:
- local library-mode generation (`llm.generate(..., output={"modality": "scene3d", ...})`)
- direct Python capability calls (`llm.scene3d.generate(...)`)
"""

from __future__ import annotations

from typing import Any, Optional


SCENE3D_BACKEND_IDS = {
    "abstract3d:triposr",
    "abstract3d:step1x-local",
    "abstract3d:hunyuan3d21-local",
    "abstract3d:trellis2-local",
}

SCENE3D_BACKEND_ALIASES = {
    "triposr": "abstract3d:triposr",
    "step1x": "abstract3d:step1x-local",
    "step1x-local": "abstract3d:step1x-local",
    # License-gated backend: selectable by alias, but abstract3d refuses to
    # download or run it without the explicit Tencent license acknowledgment
    # (ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1 / scene3d_hunyuan_license_accepted).
    "hunyuan3d21": "abstract3d:hunyuan3d21-local",
    "hunyuan3d": "abstract3d:hunyuan3d21-local",
    "hunyuan3d21-local": "abstract3d:hunyuan3d21-local",
    "trellis2": "abstract3d:trellis2-local",
    "trellis2-local": "abstract3d:trellis2-local",
}


def _selector_text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace("_", "-")
    return None


def resolve_scene3d_backend_id(*values: Any, allow_unknown: bool = False) -> Optional[str]:
    """Resolve exact scene3d backend selectors into a concrete plugin backend_id."""

    for value in values:
        text = _selector_text(value)
        if not text:
            continue
        if text in SCENE3D_BACKEND_IDS:
            return text
        alias = SCENE3D_BACKEND_ALIASES.get(text)
        if alias:
            return alias
        if allow_unknown:
            return text
    return None


__all__ = [
    "SCENE3D_BACKEND_ALIASES",
    "SCENE3D_BACKEND_IDS",
    "resolve_scene3d_backend_id",
]
