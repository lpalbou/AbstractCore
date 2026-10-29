"""ONE llama.cpp weights instance per GGUF model in a process, any number of contexts on it.

A llama.cpp `Llama` is two things: the model (the weights, `_internals.LlamaModel`) and a
context (its KV cache and compute buffers, `_internals.LlamaContext`). llama.cpp lets any
number of contexts run on one model, each with its own KV, concurrently (the decode releases
the GIL). llama-cpp-python's `Llama()` always loads the weights again, so before this module
every provider instance of a resident GGUF -- a second account's runtime pool, a gateway-side
client, a server request -- was a FULL second copy (measured 2026-10-29: 11.8 -> 23.3 -> 34.9 GB
RSS for three holders of one 4B model; the residency failsafe then refused the second copy).

Here every `Llama()` built through `shared_llama(...)` gets a BORROWED handle on the one
`LlamaModel` of its weights key (resolved path + the model parameters that shape the weights:
GPU layers, mmap, mlock; never the context size). The contexts built on it are the slots of the
model's slot pool (`gguf_slots`, ADR 0011): provider instances own no context, a generation
borrows a slot, slots run in parallel on the same weights -- never a second copy to get
concurrency. The weights are freed when the last context built on them closes (or is collected).

Mechanism: llama-cpp-python builds the model as `internals.LlamaModel(...)` inside
`Llama.__init__`; while a thread is inside `shared_llama`, that constructor returns a handle on
the shared model (loading it on first use). Outside `shared_llama` nothing changes (the
embedder, tests' fake engines, other libraries' `Llama()`).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional

_LOCK = threading.RLock()
_TLS = threading.local()
_INSTALLED = False


class SharedWeights:
    """The one `LlamaModel` of a weights key and the handles borrowed on it."""

    def __init__(self, key: str, model: Any) -> None:
        self.key = key
        self.model = model
        self.handles = 0  # open borrowed handles (one per alive context built on it)
        self.first_n_ctx: Optional[int] = None  # the context the loading holder settled on


_WEIGHTS: Dict[str, SharedWeights] = {}


class _BorrowedModel:
    """What a `Llama` holds as its `_model`: every attribute is the shared model's; `close()`
    (run by `Llama.close()` / `__del__`) returns the handle, and the last one frees the weights."""

    def __init__(self, shared: SharedWeights) -> None:
        self.__dict__["_shared"] = shared
        self.__dict__["_closed"] = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_shared"].model, name)

    def close(self) -> None:
        if self.__dict__["_closed"]:
            return
        self.__dict__["_closed"] = True
        _return_handle(self.__dict__["_shared"])


def _return_handle(shared: SharedWeights) -> None:
    with _LOCK:
        shared.handles -= 1
        if shared.handles > 0:
            return
        if _WEIGHTS.get(shared.key) is shared:
            _WEIGHTS.pop(shared.key, None)
        model, shared.model = shared.model, None
    if model is not None:
        try:
            model.close()
        except Exception:  # noqa: BLE001 - freeing is best effort, like Llama.close()
            pass


def _install() -> None:
    """Make `llama_cpp._internals.LlamaModel` honour `shared_llama` (once per process)."""
    global _INSTALLED
    if _INSTALLED:
        return
    with _LOCK:
        if _INSTALLED:
            return
        import llama_cpp._internals as internals  # type: ignore

        original = internals.LlamaModel

        class _SharedLlamaModel(original):  # type: ignore[misc, valid-type]
            """`internals.LlamaModel` that hands out the shared weights inside `shared_llama`."""

            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                key = getattr(_TLS, "key", None)
                if key is None:
                    return original(*args, **kwargs)
                with _LOCK:
                    shared = _WEIGHTS.get(key)
                    if shared is None or shared.model is None:
                        shared = SharedWeights(key, original(*args, **kwargs))
                        _WEIGHTS[key] = shared
                        _TLS.loaded = True
                    shared.handles += 1
                    return _BorrowedModel(shared)

        internals.LlamaModel = _SharedLlamaModel
        _INSTALLED = True


def weights_key(model_path: Any, *, n_gpu_layers: Any = None, use_mmap: Any = True, use_mlock: Any = False,
                main_gpu: Any = 0, tensor_split: Any = None) -> str:
    try:
        path = os.path.realpath(str(model_path))
    except Exception:  # noqa: BLE001
        path = str(model_path)
    return f"{path}|gpu_layers={n_gpu_layers}|mmap={bool(use_mmap)}|mlock={bool(use_mlock)}|main_gpu={main_gpu}|split={tensor_split}"


def key_for_kwargs(kwargs: Dict[str, Any]) -> str:
    return weights_key(kwargs.get("model_path"), n_gpu_layers=kwargs.get("n_gpu_layers"),
                       use_mmap=kwargs.get("use_mmap", True), use_mlock=kwargs.get("use_mlock", False),
                       main_gpu=kwargs.get("main_gpu", 0), tensor_split=kwargs.get("tensor_split"))


def resident(key: str) -> Optional[SharedWeights]:
    """The shared weights of `key` if they are loaded now."""
    with _LOCK:
        shared = _WEIGHTS.get(key)
        return shared if shared is not None and shared.model is not None else None


def weights_resident_for(kwargs: Dict[str, Any]) -> bool:
    return resident(key_for_kwargs(kwargs)) is not None


def shared_llama(llama_cls: Any) -> Callable[..., Any]:
    """`llama_cls(**kwargs)` that borrows the resident weights of its key (or loads them once).
    The returned engine says which: `engine._abstractcore_weights_borrowed` (True when the
    weights were already resident) and `engine._abstractcore_weights_key`."""

    def build(**kwargs: Any) -> Any:
        try:
            _install()
        except Exception:  # noqa: BLE001 - no llama_cpp internals (a fake engine): plain build
            return llama_cls(**kwargs)
        key = key_for_kwargs(kwargs)
        prev_key, prev_loaded = getattr(_TLS, "key", None), getattr(_TLS, "loaded", None)
        _TLS.key, _TLS.loaded = key, False
        try:
            engine = llama_cls(**kwargs)
            loaded = bool(_TLS.loaded)
        finally:
            _TLS.key, _TLS.loaded = prev_key, prev_loaded
        shared_handle = isinstance(getattr(engine, "_model", None), _BorrowedModel)
        try:
            engine._abstractcore_weights_key = key if shared_handle else None
            engine._abstractcore_weights_borrowed = bool(shared_handle and not loaded)
        except Exception:  # noqa: BLE001
            pass
        return engine

    return build


def weights_identity(llm: Any) -> Optional[int]:
    """One id per weights instance: the shared model behind a borrowed handle, else the engine's own
    model. Residency rows count the weights once per identity (not once per holder)."""
    model = getattr(llm, "_model", None)
    if model is None:
        return None
    shared = model.__dict__.get("_shared") if isinstance(model, _BorrowedModel) else None
    if shared is not None:
        return id(shared)
    return id(model)


def report() -> List[Dict[str, Any]]:
    """`[{key, handles}]` for every resident shared weights instance (diagnostics, tests)."""
    with _LOCK:
        return [{"key": s.key, "handles": int(s.handles)} for s in _WEIGHTS.values() if s.model is not None]
