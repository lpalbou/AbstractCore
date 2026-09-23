"""Observe mlx-vlm's chunked prefill, chunk by chunk (mid-prefill progress).

mlx-lm's `generate_step` takes a `prompt_progress_callback(processed, total)`;
mlx-vlm's (0.7.x) does not. Its chunked prefill loop in
`mlx_vlm.generate.ar.generate_step` reports each chunk ONLY to a progress bar:

    with tqdm(total=total_tokens, desc="Prefill", unit="tok", disable=not verbose) as pbar:
        while inputs_embeds.shape[1] > 1:
            ...                                # one prefill_step_size chunk
            mx.eval([c.state for c in prompt_cache])
            pbar.update(n_to_process)

`pbar.update` is called after the chunk's KV state has been EVALUATED, so it is
an exact per-chunk measurement of the tokens fed so far (`total` is the number
of tokens this call feeds: the prompt minus whatever mlx-vlm restored from its
prompt cache / APC; the final token is processed together with the first
sample, so the bar tops out at `total - 1`).

This module swaps the module-global `tqdm` name that `generate_step` resolves
at call time for a thin wrapper which forwards every call to the real tqdm and,
ONLY when the current thread bound an observer with `observe_prefill(...)` and
the bar is the "Prefill" bar, also reports `(fed_processed, fed_total)`. With no
observer bound it is behaviourally the real tqdm (same arguments, same output,
same `disable`), so other threads and other callers are unaffected.

Thread binding is per thread because `generate_step` runs on the thread that
advances the generator: the native runtime's worker thread for the exclusive
lane, the consumer's thread for the in-process mlx-vlm stream. The observer is
captured when the bar is CREATED, so an observer bound around one `next()` is
enough and never leaks into a later bar.

If the installed mlx-vlm no longer exposes that seam (the name is missing, or
someone else already replaced it), nothing is patched and `install()` returns
False with a warning: that lane then reports no mid-prefill progress, and its
events carry the total only — never an invented count.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

PrefillObserver = Callable[[int, int], None]

_BOUND = threading.local()
_INSTALL_LOCK = threading.Lock()
_INSTALLED: Optional[bool] = None
_PREFILL_BAR_DESC = "Prefill"


def _current_observer() -> Optional[PrefillObserver]:
    observer = getattr(_BOUND, "observer", None)
    return observer if callable(observer) else None


def _make_wrapper(real_tqdm: Any) -> type:
    class _ObservedPrefillBar:
        """Forwards to the real tqdm; also reports the Prefill bar's progress."""

        _abstractcore_prefill_observer = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = real_tqdm(*args, **kwargs)
            observer = _current_observer() if kwargs.get("desc") == _PREFILL_BAR_DESC else None
            total = kwargs.get("total")
            try:
                self._total = int(total) if total is not None else None
            except (TypeError, ValueError):
                self._total = None
            self._observer = observer if self._total is not None else None
            self._done = 0
            self._report()

        def _report(self) -> None:
            observer = self._observer
            if observer is None:
                return
            try:
                observer(self._done, self._total)
            except Exception:  # noqa: BLE001 - observability must never break a prefill
                logger.debug("mlx-vlm prefill observer raised; detaching it", exc_info=True)
                self._observer = None

        def update(self, n: Any = 1) -> Any:
            try:
                self._done += int(n)
            except (TypeError, ValueError):
                pass
            result = self._inner.update(n)
            self._report()
            return result

        def __enter__(self) -> "_ObservedPrefillBar":
            self._inner.__enter__()
            return self

        def __exit__(self, *exc: Any) -> Any:
            return self._inner.__exit__(*exc)

        def __iter__(self) -> Iterator[Any]:
            return iter(self._inner)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    _ObservedPrefillBar.__name__ = "tqdm"
    return _ObservedPrefillBar


def install() -> bool:
    """Install the observing bar into `mlx_vlm.generate.ar` once. True when active."""

    global _INSTALLED
    if _INSTALLED is not None:
        return _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED is not None:
            return _INSTALLED
        try:
            from mlx_vlm.generate import ar as _ar
        except Exception as exc:  # noqa: BLE001
            logger.warning("mlx-vlm prefill progress unavailable: cannot import mlx_vlm.generate.ar (%s)", exc)
            _INSTALLED = False
            return False
        current = getattr(_ar, "tqdm", None)
        if getattr(current, "_abstractcore_prefill_observer", False):
            _INSTALLED = True
            return True
        try:
            from tqdm import tqdm as _real_tqdm
        except Exception as exc:  # noqa: BLE001
            logger.warning("mlx-vlm prefill progress unavailable: tqdm import failed (%s)", exc)
            _INSTALLED = False
            return False
        if current is not _real_tqdm:
            logger.warning(
                "mlx-vlm prefill progress unavailable: mlx_vlm.generate.ar.tqdm is %r, not the tqdm "
                "this module knows how to observe; mid-prefill progress stays off for mlx-vlm lanes",
                current,
            )
            _INSTALLED = False
            return False
        _ar.tqdm = _make_wrapper(_real_tqdm)
        _INSTALLED = True
        return True


@contextmanager
def observe_prefill(observer: Optional[PrefillObserver]) -> Iterator[bool]:
    """Bind `observer(fed_processed, fed_total)` for mlx-vlm prefill bars created
    on THIS thread inside the block. Yields whether the seam is active."""

    if observer is None:
        yield False
        return
    active = install()
    previous = getattr(_BOUND, "observer", None)
    _BOUND.observer = observer if active else None
    try:
        yield active
    finally:
        _BOUND.observer = previous


__all__ = ["install", "observe_prefill"]
