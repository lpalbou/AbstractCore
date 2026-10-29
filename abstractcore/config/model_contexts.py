"""Per-model context settings: the context a local model is LOADED with, set by a person.

Core ADR 0010 (operator ruling 2026-10-29): the load-time fit rule and the allocation ladder
choose how much context to reserve when nothing is set; the chosen context is shown with the
model's advertised one, and a person can set it here per model. A set context is a pin: it is
loaded as is, and when it does not fit the load is refused with the readable sentence (never a
smaller one chosen behind the person's back). A request longer than the loaded context fails
with `ContextWindowExceededError`; nothing is ever cut.

Stored in the AbstractCore config document under `model_contexts`: `{"<provider>/<model>": tokens}`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

_KEY = "model_contexts"


def _key(provider: str, model: str) -> str:
    return f"{str(provider or '').strip().lower()}/{str(model or '').strip()}"


def _read(config_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
    from .manager import resolve_config_file

    path = resolve_config_file(config_file=config_file)
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def configured_model_context(provider: str, model: str, *, config_file: Optional[Union[str, Path]] = None) -> Optional[int]:
    """The context a person set for this model, or None."""
    value = (_read(config_file).get(_KEY) or {}).get(_key(provider, model))
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def set_model_context(provider: str, model: str, tokens: Optional[int], *,
                      config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Set (tokens > 0) or clear (None) the context of one model; returns the stored map.
    Atomic write of the whole document (other keys untouched)."""
    from .manager import resolve_config_file

    if tokens is not None:
        tokens = int(tokens)
        if tokens < 512:
            raise ValueError("a model context is at least 512 tokens")
    path = Path(resolve_config_file(config_file=config_file))
    doc = _read(config_file)
    contexts = dict(doc.get(_KEY) or {})
    if tokens is None:
        contexts.pop(_key(provider, model), None)
    else:
        contexts[_key(provider, model)] = tokens
    doc[_KEY] = contexts
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    return contexts


_CONCURRENCY_KEY = "model_concurrency"


def configured_model_concurrency(provider: str, model: str, *,
                                 config_file: Optional[Union[str, Path]] = None) -> Optional[int]:
    """How many generations may run at once on this local model (its slot pool's bound), or None
    (the engine's default: llama.cpp on a GPU / Metal 4, CPU-only 1)."""
    try:
        value = (_read(config_file).get(_CONCURRENCY_KEY) or {}).get(_key(provider, model))
        return int(value) if value else None
    except Exception:
        return None


def set_model_concurrency(provider: str, model: str, slots: Optional[int], *,
                          config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Set (slots >= 1) or clear (None) the concurrency bound of one model; applies at its next load."""
    from .manager import resolve_config_file

    if slots is not None:
        slots = int(slots)
        if slots < 1:
            raise ValueError("a model's concurrency is at least 1")
    path = Path(resolve_config_file(config_file=config_file))
    doc = _read(config_file)
    current = dict(doc.get(_CONCURRENCY_KEY) or {})
    if slots is None:
        current.pop(_key(provider, model), None)
    else:
        current[_key(provider, model)] = slots
    doc[_CONCURRENCY_KEY] = current
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    return current
