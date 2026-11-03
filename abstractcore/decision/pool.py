"""The resident decision providers of this process.

One place keeps the provider instances that hold a loaded decision model, so the
server's ``/v1/decisions``, a text provider asked for ``output="decision"`` with a
capability-default route, and the runtime all share the same loaded weights
instead of each loading their own. At most ``MAX_RESIDENT`` models stay loaded;
the least recently used one that no request is using is unloaded when another is
needed. Loading happens outside the pool's lock (one load lock per model), so a
request on a resident model never waits for another model's weights.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..utils.structured_logging import get_logger

logger = get_logger(__name__)

MAX_RESIDENT = 2

# The provider tokens a `provider/model` decision handle may start with; anything else
# before the first slash belongs to the model id (`jaredpalmer/kev-4b` is one handle).
PROVIDER_NAMES = frozenset({"huggingface", "hf", "transformers", "mlx", "gguf", "llamacpp", "typesafe", "systemone"})
LOCAL_PROVIDERS = frozenset({"huggingface", "mlx"})
ALIASES = frozenset({"kev-latest", "jev-latest"})
_PROVIDER_ALIASES = {"hf": "huggingface", "transformers": "huggingface", "gguf": "huggingface", "llamacpp": "huggingface"}

Key = Tuple[str, str]

_RESIDENT: Dict[Key, Any] = {}          # insertion order = least recently used first
_IN_USE: Dict[Key, int] = {}            # requests currently inside `use()`
_PENDING_UNLOAD: Dict[Key, Any] = {}    # evicted while in use: unloaded when the last user leaves
_LOAD_LOCKS: Dict[Key, threading.Lock] = {}
_LOCK = threading.Lock()


def split_handle(handle: str) -> Tuple[Optional[str], str]:
    """``(provider, model)`` of a decision handle: the provider when the handle starts with
    one (``mlx/jaredpalmer/kev-4b``), else ``None`` and the handle whole."""
    text = str(handle or "").strip()
    head, _, rest = text.partition("/")
    if rest and head.lower() in PROVIDER_NAMES:
        return _PROVIDER_ALIASES.get(head.lower(), head.lower()), rest
    return None, text


def parse_handle(handle: str) -> Tuple[str, str]:
    """``(provider, model)`` with the local engine chosen for a bare handle: a GGUF runs on
    huggingface (llama.cpp); anything else on mlx when mlx-lm is installed, else huggingface."""
    provider, model = split_handle(handle)
    if provider is not None:
        return provider, model
    from ..utils.model_cache import is_gguf_model_ref

    if is_gguf_model_ref(model):
        return "huggingface", model
    try:
        import mlx_lm  # noqa: F401  # type: ignore

        return "mlx", model
    except Exception:
        return "huggingface", model


def _key(provider: str, model: str) -> Key:
    key = (str(provider or "").strip().lower(), str(model or "").strip())
    if not key[0] or not key[1]:
        raise ValueError("a decision model needs both a provider and a model")
    return key


def _check_is_decision_model(key: Key) -> None:
    """Refuse a local handle that is not a decision model BEFORE any weights load."""
    if key[0] not in LOCAL_PROVIDERS:
        return
    from .loader import is_decision_model

    if not is_decision_model(key[1], key[0]):
        raise ValueError(
            f"{key[1]!r} is not a decision model on {key[0]}. Decision output needs a Kev-family checkpoint "
            "(a LoRA adapter with head.pt, an AbstractCore decision conversion, or a decision GGUF), Lev or Laya; "
            "for example mlx/jaredpalmer/kev-4b or huggingface/ggml-org/Kev-4B-GGUF:Q4_K_M."
        )


def _unload(key: Key, llm: Any) -> None:
    try:
        llm.unload_model(key[1])
    except Exception as exc:  # noqa: BLE001 - an unload failure must not fail a request
        logger.warning("decision provider unload failed", provider=key[0], model=key[1], error=str(exc))


def _evict_locked() -> List[Tuple[Key, Any]]:
    """Under `_LOCK`: drop least recently used models no request is using until the pool fits."""
    evicted: List[Tuple[Key, Any]] = []
    for key in list(_RESIDENT):
        if len(_RESIDENT) <= MAX_RESIDENT:
            break
        if _IN_USE.get(key, 0) > 0:
            continue
        evicted.append((key, _RESIDENT.pop(key)))
    return evicted


def _acquire(
    key: Key,
    *,
    hold: bool,
    factory: Optional[Callable[[str, str], Any]],
    options: Dict[str, Any],
) -> Any:
    """The resident provider for `key`, loaded if needed. With `hold`, the in-use count is
    raised in the same locked section that hands the instance out, so no eviction or
    `release()` can unload it before the caller's `use()` block ends."""
    with _LOCK:
        llm = _RESIDENT.pop(key, None)
        if llm is not None:
            _RESIDENT[key] = llm  # most recently used
            if hold:
                _IN_USE[key] = _IN_USE.get(key, 0) + 1
            return llm
        load_lock = _LOAD_LOCKS.setdefault(key, threading.Lock())
    with load_lock:  # one load per model; other models stay available meanwhile
        with _LOCK:
            llm = _RESIDENT.pop(key, None)
            if llm is not None:
                _RESIDENT[key] = llm
                if hold:
                    _IN_USE[key] = _IN_USE.get(key, 0) + 1
                return llm
        _check_is_decision_model(key)
        if factory is not None:
            llm = factory(key[0], key[1])
        else:
            from ..core.factory import create_llm

            llm = create_llm(key[0], model=key[1], **options)
        with _LOCK:
            _RESIDENT[key] = llm
            if hold:
                _IN_USE[key] = _IN_USE.get(key, 0) + 1
            evicted = _evict_locked()
    _unload_all(evicted)
    return llm


def _unload_all(items: List[Tuple[Key, Any]]) -> None:
    """Unload evicted models under their load lock, so a reload of the same key waits for
    the unload instead of putting a second copy in memory."""
    for key, llm in items:
        with _LOCK:
            load_lock = _LOAD_LOCKS.setdefault(key, threading.Lock())
        with load_lock:
            _unload(key, llm)


def decision_provider(
    provider: str,
    model: str,
    *,
    factory: Optional[Callable[[str, str], Any]] = None,
    **options: Any,
) -> Any:
    """The provider instance holding ``provider/model`` as a decision model.

    Created on first use (``create_llm(provider, model=model, **options)`` unless
    ``factory`` builds it), then reused; ``options`` only matter on creation. A
    handle that is not a decision model is refused before anything loads. Around a
    request, prefer ``use()``: it keeps the model resident until the request ends.
    """
    return _acquire(_key(provider, model), hold=False, factory=factory, options=options)


@contextmanager
def use(provider: str, model: str, *, factory: Optional[Callable[[str, str], Any]] = None, **options: Any) -> Iterator[Any]:
    """``with use(provider, model) as llm:`` — the resident provider, protected from
    unloading until the block ends (an eviction or `release()` meanwhile is deferred)."""
    key = _key(provider, model)
    llm = _acquire(key, hold=True, factory=factory, options=options)
    try:
        yield llm
    finally:
        pending: List[Tuple[Key, Any]] = []
        with _LOCK:
            left = _IN_USE.get(key, 0) - 1
            if left > 0:
                _IN_USE[key] = left
            else:
                _IN_USE.pop(key, None)
                deferred = _PENDING_UNLOAD.pop(key, None)
                if deferred is not None:
                    pending.append((key, deferred))
                pending.extend(_evict_locked())
        _unload_all(pending)


def latest() -> Optional[Key]:
    """``(provider, model)`` of the decision model used most recently, if any is resident."""
    with _LOCK:
        return list(_RESIDENT.keys())[-1] if _RESIDENT else None


def resident() -> List[Key]:
    """Every resident ``(provider, model)``, least recently used first."""
    with _LOCK:
        return list(_RESIDENT.keys())


def release(provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[int, int]:
    """Unload resident decision models (all of them, or the one named).

    Returns ``(unloaded, deferred)``: a model a request is using is unloaded when that
    request ends and counts as deferred."""
    with _LOCK:
        if provider is None:
            keys = list(_RESIDENT)
        else:
            keys = [k for k in _RESIDENT if k == (str(provider).strip().lower(), str(model or "").strip())]
        dropped: List[Tuple[Key, Any]] = []
        deferred = 0
        for key in keys:
            llm = _RESIDENT.pop(key)
            if _IN_USE.get(key, 0) > 0:
                _PENDING_UNLOAD[key] = llm
                deferred += 1
            else:
                dropped.append((key, llm))
    _unload_all(dropped)
    with _LOCK:
        for key, _ in dropped:
            if key not in _RESIDENT and key not in _IN_USE:
                _LOAD_LOCKS.pop(key, None)
    return len(dropped), deferred


def _reset_for_tests() -> None:
    with _LOCK:
        _RESIDENT.clear()
        _IN_USE.clear()
        _PENDING_UNLOAD.clear()
        _LOAD_LOCKS.clear()


__all__ = [
    "ALIASES",
    "LOCAL_PROVIDERS",
    "MAX_RESIDENT",
    "PROVIDER_NAMES",
    "decision_provider",
    "latest",
    "parse_handle",
    "release",
    "resident",
    "split_handle",
    "use",
]
