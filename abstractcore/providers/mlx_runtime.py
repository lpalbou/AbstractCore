"""Bounded, process-local execution owner for native MLX VLM sessions.

The runtime owns execution, not artifact loading. MLX/VLM imports are lazy and
every tensor, processor, drafter and cache operation runs on its one worker.
Target decoding continuously admits compatible requests. Upstream MTP cannot
extend an active speculative batch, so that mode is explicitly called a cohort.
"""

from __future__ import annotations

import copy
import hashlib
import json
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Iterator, Optional


class NativeRuntimeError(RuntimeError):
    """Typed native failure; only explicit request-local codes bypass retries.

    Backend/Metal faults retain the default ``backend_error`` classification.
    Overload, cancellation and rejected controls say nothing about model health.
    """
    _REQUEST_STATUS = {
        "queue_full": 503, "queue_timeout": 504, "cancelled": 409,
        "invalid_request": 400, "invalid_controls": 400, "no_drafter": 400,
        "missing_lease": 409, "owner_busy": 409, "runtime_closed": 409,
        "stream_backpressure": 429,
    }

    def __init__(self, message, *, code="backend_error"):
        super().__init__(message)
        self.code = code
        self.request_local = code in self._REQUEST_STATUS
        self.http_status = self._REQUEST_STATUS.get(code, 500)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise TypeError(f"Native sampling values must be immutable data, not {type(value).__name__}")


def _thaw(value):
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class NativeRequest:
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    draft_tokens: int = 0
    cache_key: Optional[str] = None
    cache_scope: str = "local"
    media: tuple = ()
    sampling: Mapping = field(default_factory=dict)
    stop: tuple = ()
    owner_id: str = ""

    def __post_init__(self):
        if not isinstance(self.prompt, str):
            raise TypeError("Native prompt must be rendered text")
        for name, minimum in (("max_tokens", 1), ("draft_tokens", 0), ("top_k", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.temperature, bool) or not 0 <= self.temperature < float("inf"):
            raise ValueError("temperature must be finite and nonnegative")
        if isinstance(self.top_p, bool) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or None")
        if not isinstance(self.cache_scope, str) or not isinstance(self.owner_id, str):
            raise TypeError("cache_scope and owner_id must be strings")
        if self.cache_key is not None and not isinstance(self.cache_key, str):
            raise TypeError("cache_key must be a string or None")
        if not isinstance(self.sampling, Mapping):
            raise TypeError("sampling must be a mapping")
        object.__setattr__(self, "sampling", _freeze(self.sampling))
        stops = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)
        if any(not isinstance(s, str) or not s for s in stops):
            raise ValueError("stop sequences must be nonempty strings")
        object.__setattr__(self, "stop", stops)
        media = []
        for index, data in self.media:
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("media indexes must be nonnegative integers")
            if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
                raise ValueError("media must contain nonempty image bytes")
            media.append((index, bytes(data)))
        object.__setattr__(self, "media", tuple(media))


@dataclass(frozen=True)
class NativeResult:
    text: str = ""
    token: Optional[int] = None
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    cached_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    metadata: dict = field(default_factory=dict)
    media_records: list = field(default_factory=list)


@dataclass(frozen=True)
class NativePrefillProgress:
    """A mid-prefill observation, delivered to `stream(on_prefill_progress=...)`.

    Never yielded by the result iterator: consumers that did not ask for it see
    exactly the stream they saw before. Exactly one axis is set:

    * batched lane: `processed_tokens` on the whole prompt (restored prefix
      included) with `prompt_tokens` and `cached_tokens` from the upstream
      `PromptProcessingBatch` row;
    * exclusive lane: `fed_processed` / `fed_total` from mlx-vlm's chunked
      prefill (tokens this call feeds; restored tokens are not in `fed_total`).
    """

    processed_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    fed_processed: Optional[int] = None
    fed_total: Optional[int] = None

    def as_kwargs(self) -> dict:
        return {k: v for k, v in (
            ("processed_tokens", self.processed_tokens), ("prompt_tokens", self.prompt_tokens),
            ("cached_tokens", self.cached_tokens), ("fed_processed", self.fed_processed),
            ("fed_total", self.fed_total)) if v is not None}


@dataclass(eq=False)
class _Job:
    request: NativeRequest
    output: queue.Queue
    submitted: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None
    started: float = 0.0
    first_token_at: float = 0.0
    mode: str = "exclusive"
    peak_batch_size: int = 1
    cohort_size: int = 1
    prompt_tokens: int = 0
    generation_tokens: int = 0
    cached_tokens: int = 0
    prompt_tps: float = 0.0
    cache: Any = None
    media_records: list = field(default_factory=list)
    detokenizer: Any = None
    pending_text: str = ""
    draft_before: tuple = (0, 0, 0)
    reports_prefill: bool = False


@dataclass(eq=False)
class _Control:
    callback: Callable
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


def _load_backend():
    import mlx.core as mx
    from mlx_vlm import stream_generate
    from mlx_vlm.generate.ar import BatchGenerator
    from mlx_vlm.apc import semantic_extra_hash

    return SimpleNamespace(mx=mx, BatchGenerator=BatchGenerator, stream_generate=stream_generate,
                           semantic_extra_hash=semantic_extra_hash)


class _ResultIterator:
    """A closeable iterator that may be cancelled while another thread waits."""

    def __init__(self, runtime, job, on_prefill_progress=None):
        self._runtime, self._job = runtime, job
        self._closed = False
        self._on_prefill_progress = on_prefill_progress if callable(on_prefill_progress) else None

    def __iter__(self):
        return self

    def __next__(self):
        while not self._closed:
            if self._job.cancelled.is_set() and self._job.error is None:
                self.close()
                break
            try:
                item = self._job.output.get(timeout=0.05)
                if self._closed:
                    break
                if isinstance(item, NativePrefillProgress):
                    # Delivered on the CONSUMER's thread, never yielded: the
                    # result stream stays byte-identical for every consumer.
                    listener = self._on_prefill_progress
                    if listener is not None:
                        try:
                            listener(item)
                        except Exception:  # noqa: BLE001 - observability never breaks a stream
                            self._on_prefill_progress = None
                    continue
                return item
            except queue.Empty:
                if self._job.done.is_set():
                    self._closed = True
                    if self._job.error is not None:
                        raise self._job.error
                    break
        raise StopIteration

    def close(self):
        self._closed = True
        if not self._job.done.is_set():
            self._job.cancelled.set()
            with self._runtime._condition:
                self._runtime._condition.notify_all()

    def __del__(self):
        self.close()


class NativeRuntime:
    """One execution thread with bounded admission and per-consumer delivery."""

    def __init__(self, model, processor, drafter=None, draft_kind=None, *,
                 max_batch_size=4, max_queue_size=32, batch_wait_ms=10,
                 queue_timeout_s=120, output_queue_size=256, cache_factory=None,
                 on_close=None):
        for name, value in (("max_batch_size", max_batch_size), ("max_queue_size", max_queue_size),
                            ("output_queue_size", output_queue_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= batch_wait_ms <= 1000 or not 0 < queue_timeout_s < float("inf"):
            raise ValueError("Invalid native runtime queue deadline or batch wait")
        self.model, self.processor = model, processor
        self.drafter, self.draft_kind = drafter, draft_kind
        self.max_batch_size, self.max_queue_size = max_batch_size, max_queue_size
        self.batch_wait_s, self.queue_timeout_s = batch_wait_ms / 1000.0, queue_timeout_s
        self.output_queue_size, self.cache_factory = output_queue_size, cache_factory
        self._on_close, self._close_error = on_close, None
        self._condition = threading.Condition(threading.RLock())
        self._pending = deque()
        self._jobs = set()
        self._owners = set()
        self._retiring_owners = set()
        self._retirement_keepalive = {}
        self._deferred_retirement = False
        # A final release may time out while a control/backend operation still
        # owns this worker. Retain its identity so that same owner can finish
        # closing later, without reopening admission or accepting unknown IDs.
        self._pending_release_owner = None
        self._closing = False
        self._closed = threading.Event()
        self._backend = None
        self._counters = dict(submitted=0, completed=0, cancelled=0, failed=0,
                              rejected=0, peak_batch_size=0)
        self._thread = threading.Thread(target=self._worker, name="abstractcore-native-mlx", daemon=True)
        self._thread.start()

    def acquire(self) -> str:
        with self._condition:
            if self._closing:
                raise NativeRuntimeError("Native MLX runtime is closed", code="runtime_closed")
            owner = uuid.uuid4().hex
            self._owners.add(owner)
            return owner

    def release(self, owner_id):
        with self._condition:
            if owner_id is not None and owner_id == self._pending_release_owner:
                last = True
            else:
                if owner_id not in self._owners:
                    raise NativeRuntimeError("Unknown native MLX runtime owner", code="missing_lease")
                if any(j.request.owner_id == owner_id for j in self._jobs):
                    raise NativeRuntimeError("Cannot release native MLX owner with active or queued requests; close its streams first", code="owner_busy")
                self._owners.remove(owner_id)
                self._retiring_owners.discard(owner_id)
                self._retirement_keepalive.pop(owner_id, None)
                last = not self._owners
                if last:
                    self._pending_release_owner = owner_id
                    self._closing = True  # No acquire may race the final release.
        if last:
            self.close()
            with self._condition:
                self._pending_release_owner = None

    def retire(self, owner_id, *, keepalive=None):
        """Nonblocking finalizer retirement; explicit unload uses ``release``.

        Cancel only this owner's work and surrender its lease at the next safe
        worker boundary. ``keepalive`` preserves a weakly registered session
        until its final worker cleanup, preventing a duplicate model load while
        cancelled prefill still owns its tensors. Repeated retirement is safe.
        """
        with self._condition:
            if owner_id not in self._owners and owner_id != self._pending_release_owner:
                return False
            if owner_id is None:
                return False
            self._deferred_retirement = True
            self._retiring_owners.add(owner_id)
            if keepalive is not None and not self._closed.is_set():
                self._retirement_keepalive[owner_id] = keepalive
            for job in self._jobs:
                if job.request.owner_id == owner_id:
                    job.cancelled.set()
            self._retire_idle_owners_locked()
            self._condition.notify_all()
        return True

    def _retire_idle_owners_locked(self):
        """Called with the condition held, never joins or closes GPU resources."""
        active_owners = {job.request.owner_id for job in self._jobs}
        for owner in tuple(self._retiring_owners):
            if owner in active_owners:
                continue
            self._retiring_owners.discard(owner)
            self._owners.discard(owner)
            if owner == self._pending_release_owner:
                self._pending_release_owner = None
            if self._owners or self._closed.is_set():
                self._retirement_keepalive.pop(owner, None)
            if not self._owners:
                self._closing = True

    def stream(self, request: NativeRequest, *, cancel_event=None,
               on_prefill_progress=None) -> Iterator[NativeResult]:
        """Admit `request`; iterate its results.

        `on_prefill_progress(NativePrefillProgress)` (optional) is called on the
        iterating thread for each mid-prefill observation the lane can make —
        per `prefill_step_size` chunk on both the batched and the exclusive
        lane, never after the first token. A lane that cannot observe its
        prefill (an unchunked one-shot prefill, e.g. under an MTP drafter)
        simply never calls it.
        """
        if not isinstance(request, NativeRequest):
            raise TypeError("NativeRuntime.stream requires NativeRequest")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event or None")
        if on_prefill_progress is not None and not callable(on_prefill_progress):
            raise TypeError("on_prefill_progress must be callable or None")
        with self._condition:
            if self._closing:
                raise NativeRuntimeError("Native MLX runtime is closed", code="runtime_closed")
            if request.owner_id not in self._owners:
                raise NativeRuntimeError("Native MLX request must hold a live runtime lease", code="missing_lease")
            if request.owner_id in self._retiring_owners:
                raise NativeRuntimeError("Native MLX request owner is retiring", code="missing_lease")
            if request.draft_tokens and self.drafter is None:
                raise NativeRuntimeError("Native MTP requested but this runtime has no drafter", code="no_drafter")
            if cancel_event is not None and cancel_event.is_set():
                raise NativeRuntimeError("Native MLX request was cancelled before admission", code="cancelled")
            if len(self._pending) >= self.max_queue_size:
                self._counters["rejected"] += 1
                raise NativeRuntimeError("Native MLX waiting queue is full; retry after active requests finish", code="queue_full")
            job = _Job(request=request, output=queue.Queue(self.output_queue_size))
            if cancel_event is not None:
                job.cancelled = cancel_event
            job.reports_prefill = on_prefill_progress is not None
            self._jobs.add(job)
            self._pending.append(job)
            self._counters["submitted"] += 1
            self._condition.notify_all()
        return _ResultIterator(self, job, on_prefill_progress)

    def generate(self, request: NativeRequest, *, cancel_event=None) -> NativeResult:
        iterator, pieces, last = self.stream(request, cancel_event=cancel_event), [], None
        try:
            for last in iterator:
                pieces.append(last.text)
            cancelled = iterator._job.cancelled.is_set()
        finally:
            iterator.close()
        if last is None or last.finish_reason is None:
            raise NativeRuntimeError("Native MLX generation ended without a terminal result",
                                     code="cancelled" if cancelled else "backend_error")
        return replace(last, text="".join(pieces))

    def control(self, callback):
        if threading.current_thread() is self._thread:
            return callback()
        control = _Control(callback)
        with self._condition:
            if self._closing:
                raise NativeRuntimeError("Native MLX runtime is closed", code="runtime_closed")
            if len(self._pending) >= self.max_queue_size:
                raise NativeRuntimeError("Native MLX control queue is full", code="queue_full")
            self._pending.append(control)
            self._condition.notify_all()
        control.done.wait()
        if control.error is not None:
            raise control.error
        return control.result

    def stats(self):
        with self._condition:
            queued = sum(isinstance(j, _Job) for j in self._pending)
            return {**self._counters, "queued": queued, "active": len(self._jobs) - queued,
                    "owners": len(self._owners), "closed": self._closed.is_set(),
                    "retiring_owners": len(self._retiring_owners),
                    "close_error": str(self._close_error) if self._close_error is not None else None,
                    "closing": self._closing, "max_batch_size": self.max_batch_size,
                    "max_queue_size": self.max_queue_size}

    def close(self):
        with self._condition:
            self._closing = True
            for job in self._jobs:
                job.cancelled.set()
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                raise NativeRuntimeError("Native MLX worker has not stopped; resources remain owned until its GPU operation returns")
            if self._close_error is not None:
                raise NativeRuntimeError(f"Native MLX close hook failed: {self._close_error}") from self._close_error

    def _finish(self, job, error=None):
        with self._condition:
            if job.done.is_set():
                return
            if error is not None:
                job.error = error
            state = "failed" if job.error else "cancelled" if job.cancelled.is_set() else "completed"
            self._counters[state] += 1
            self._jobs.discard(job)
            job.done.set()
            self._retire_idle_owners_locked()
            self._condition.notify_all()

    def _discard_expired(self):
        now = time.monotonic()
        with self._condition:
            kept = deque()
            for item in self._pending:
                if isinstance(item, _Job) and item.cancelled.is_set():
                    self._finish(item)
                elif isinstance(item, _Job) and now - item.submitted >= self.queue_timeout_s:
                    self._finish(item, NativeRuntimeError("Native MLX request timed out waiting for admission", code="queue_timeout"))
                else:
                    kept.append(item)
            self._pending = kept

    def _worker(self):
        try:
            while True:
                self._discard_expired()
                with self._condition:
                    while not self._pending and not self._closing:
                        self._condition.wait()
                    if self._closing:
                        break
                    item = self._pending.popleft()
                if isinstance(item, _Control):
                    try:
                        item.result = item.callback()
                    except BaseException as exc:
                        item.error = exc
                    finally:
                        item.done.set()
                    continue
                if item.cancelled.is_set():
                    self._finish(item)
                    continue
                try:
                    if self._backend is None:
                        self._backend = _load_backend()
                    if self._batchable(item.request):
                        self._run_batch(item)
                    else:
                        self._run_exclusive(item)
                except BaseException as exc:
                    self._finish(item, exc)
        finally:
            with self._condition:
                for job in tuple(self._jobs):
                    self._finish(job, NativeRuntimeError("Native MLX runtime closed before request completed", code="cancelled"))
                for item in self._pending:
                    if isinstance(item, _Control):
                        item.error = NativeRuntimeError("Native MLX runtime closed before control operation", code="cancelled")
                        item.done.set()
                self._pending.clear()
            if self._on_close is not None:
                try:
                    self._on_close()
                except BaseException as exc:
                    self._close_error = exc
                    if self._deferred_retirement:
                        # No synchronous caller remains to observe a finalizer's
                        # close failure. Keep it inspectable and log it visibly.
                        import logging
                        logging.getLogger(__name__).error("Native MLX deferred close hook failed: %s", exc)
            self._closed.set()
            with self._condition:
                self._retirement_keepalive.clear()

    @staticmethod
    def _batchable(request):
        # BatchGenerator has a single sampler. Rich per-request processors and
        # random seeds therefore run exclusively rather than being ignored.
        neutral = {"min_p": (None, 0, 0.0), "repetition_penalty": (None, 1, 1.0),
                   "presence_penalty": (None, 0, 0.0), "frequency_penalty": (None, 0, 0.0),
                   "logit_bias": (None, {})}
        return request.temperature == 0 and all(
            k in neutral and _thaw(v) in neutral[k] for k, v in request.sampling.items()
        )

    def _cache_for(self, job):
        if job.request.cache_key and self.cache_factory:
            return self.cache_factory(job.request)
        return None

    def _group_key(self, job):
        return job.request.draft_tokens, id(job.cache)

    def _take_compatible(self, first, limit, *, initial=False):
        selected = []
        deadline = time.monotonic() + (self.batch_wait_s if initial else 0)
        while len(selected) < limit:
            self._discard_expired()
            with self._condition:
                if self._closing:
                    break
                if not self._pending:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._condition.wait(remaining)
                        continue
                    break
                job = self._pending[0]
                if not isinstance(job, _Job) or not self._batchable(job.request):
                    break  # A control or incompatible request is a fairness barrier.
                if job.request.draft_tokens != first.request.draft_tokens:
                    break
                self._pending.popleft()
            try:
                job.cache = self._cache_for(job)
            except BaseException as exc:
                self._finish(job, exc)
                continue
            if self._group_key(job) != self._group_key(first):
                with self._condition:
                    self._pending.appendleft(job)
                break
            selected.append(job)
        return selected

    @staticmethod
    def _tenant(request, layout):
        # This is cache partitioning, not authenticated tenant authorization.
        return json.dumps(["abstractcore-native-vlm", request.cache_scope, request.cache_key,
                           request.draft_tokens, layout], separators=(",", ":"))

    def _prepare(self, job):
        request, model, processor = job.request, self.model, self.processor
        tokenizer = getattr(processor, "tokenizer", processor)
        mx = self._backend.mx
        if request.media:
            from .mlx_qwen4 import prepare_images

            parts = [(i, SimpleNamespace(content=data, file_path=None)) for i, data in request.media]
            inputs, job.media_records = prepare_images(model, processor, request.prompt, parts)
            ids = inputs["input_ids"]
            pixel_values, mask = inputs.get("pixel_values"), inputs.get("mask")
            extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "mask")}
        else:
            ids = mx.array([tokenizer.encode(request.prompt, add_special_tokens=False)])
            pixel_values, mask, extra = None, None, {}
        # Even pure text needs this call: Qwen's position/rope preparation is
        # model-specific and skipping it corrupts left-padded batched prompts.
        embeddings = model.get_input_embeddings(ids, pixel_values, mask=mask, **extra)
        kwargs = {**extra, **{k: v for k, v in embeddings.to_dict().items() if v is not None}}
        tokens = ids.tolist()[0]
        if not tokens:
            raise NativeRuntimeError("Native MLX prompt rendered to zero tokens", code="invalid_request")
        job.prompt_tokens = len(tokens)
        if job.cache is not None:
            kwargs["_apc_tenant"] = self._tenant(request, "batch")
            image_hash = 0
            if request.media:
                digest = hashlib.sha256()
                for index, data in request.media:
                    digest.update(index.to_bytes(8, "big"))
                    digest.update(len(data).to_bytes(8, "big"))
                    digest.update(data)
                image_hash = int.from_bytes(digest.digest()[:8], "big")
                kwargs["_apc_image_hash"] = image_hash
            # Text embeddings are entirely determined by the token chain and
            # immutable model identity. Hashing their entire payload would
            # synchronize/copy GPU tensors and invalidate a reusable prefix
            # whenever only the text suffix changed. Image bytes and upstream
            # model/processor semantic dependencies remain part of the salt.
            # The cache store independently fingerprints weights/config/engine.
            kwargs["_apc_semantic_hash"] = self._backend.semantic_extra_hash(
                tenant=kwargs["_apc_tenant"], image_hash=image_hash, media=None,
                model=self.model, processor=self.processor,
            )
        job.detokenizer = copy.copy(processor.detokenizer)
        job.detokenizer.reset()
        return tokens, kwargs

    def _draft_counts(self):
        # Upstream's batched counter accumulates MEAN accepted tokens per
        # round, not the aggregate across rows. Preserve fractional means.
        return (int(getattr(self.drafter, "speculative_total_rounds", 0)),
                float(getattr(self.drafter, "speculative_total_accepted", 0.0)),
                int(getattr(self.drafter, "speculative_total_drafted", 0)))

    def _snapshot(self, job, text="", token=None, finish_reason=None, *, source=None):
        now = time.monotonic()
        elapsed = now - (job.first_token_at or job.started or now)
        generation_tps = max(0, job.generation_tokens - 1) / elapsed if elapsed > 0 else 0.0
        metadata = {"execution": {"mode": job.mode, "peak_batch_size": job.peak_batch_size,
                                  "queue_s": max(0.0, job.started - job.submitted),
                                  "ttft_s": job.first_token_at - job.submitted if job.first_token_at else None}}
        if job.request.draft_tokens:
            rounds, accepted, drafted = [max(0, b - a) for a, b in zip(job.draft_before, self._draft_counts())]
            metadata["speculation"] = {
                "used": drafted > 0, "num_draft_tokens": job.request.draft_tokens,
                "stats": {"rounds": rounds, "accepted_tokens": accepted, "drafted_tokens": drafted,
                          "accounting": "cohort" if job.cohort_size > 1 else "request",
                          "counter_semantics": ("sum_of_round_means_not_batch_token_totals"
                                                if job.cohort_size > 1 else "request_tokens")},
            }
        peak = float(getattr(source, "peak_memory", 0.0) if source is not None else 0.0)
        if not peak:
            peak = self._backend.mx.get_peak_memory() / 1e9
        return NativeResult(text=text, token=token, finish_reason=finish_reason,
                            prompt_tokens=job.prompt_tokens, generation_tokens=job.generation_tokens,
                            cached_tokens=job.cached_tokens, prompt_tps=job.prompt_tps,
                            generation_tps=float(getattr(source, "generation_tps", generation_tps)),
                            peak_memory=peak, metadata=metadata, media_records=copy.deepcopy(job.media_records))

    def _emit(self, job, result):
        if job.cancelled.is_set():
            return False
        try:
            job.output.put_nowait(result)
            return True
        except queue.Full:
            job.error = NativeRuntimeError("Native MLX stream consumer is too slow; bounded output queue overflowed", code="stream_backpressure")
            job.cancelled.set()
            return False

    @staticmethod
    def _emit_prefill(job, **fields):
        """Queue one mid-prefill observation for a job whose consumer asked.

        Each observation is CUMULATIVE, so one that finds the output queue
        half full is skipped rather than allowed to crowd out result items
        (whose overflow fails the request): the next chunk's observation
        supersedes it. Nothing is queued after the first token.
        """
        if not job.reports_prefill or job.first_token_at or job.cancelled.is_set():
            return False
        maxsize = job.output.maxsize or 0
        if maxsize and job.output.qsize() >= maxsize // 2:
            return False
        try:
            job.output.put_nowait(NativePrefillProgress(**fields))
            return True
        except queue.Full:
            return False

    _PROMPT_BATCH_ROW_FIELDS = ("_prompt_uids", "_prompt_tokens_per_row", "_cached_tokens_per_row",
                                "_suffix_lens", "_left_padding_per_row", "_processed_prompt_columns",
                                "_right_pad_per_row")
    _prompt_batch_shape_warned = False

    @classmethod
    def _prompt_batch_rows(cls, prompt_batch):
        """Per-row prefill position of an upstream `PromptProcessingBatch`.

        mlx-vlm (0.7.x) prefills one `prefill_step_size` chunk per
        `BatchGenerator.next()` and exposes the per-row counters only as
        private attributes (its public `prompt_progress()` answers after the
        LAST chunk). Rows are aligned with `_prompt_uids`; a row's real tokens
        processed are its restored prefix plus the suffix columns consumed so
        far (right-padded rows start at column 0, left-padded rows after their
        padding) — the same arithmetic as upstream's
        `_row_real_tokens_processed`. A shape change is reported once, loudly,
        and then nothing is reported: never a guessed position.
        """
        try:
            uids = list(prompt_batch._prompt_uids)
            totals = [int(n) for n in prompt_batch._prompt_tokens_per_row]
            cached = [int(n) for n in prompt_batch._cached_tokens_per_row]
            suffix = [int(n) for n in prompt_batch._suffix_lens]
            left = [int(n) for n in prompt_batch._left_padding_per_row]
            columns = int(prompt_batch._processed_prompt_columns)
            right_padded = prompt_batch._right_pad_per_row is not None
        except (AttributeError, TypeError, ValueError) as exc:
            if not cls._prompt_batch_shape_warned:
                cls._prompt_batch_shape_warned = True
                import logging
                logging.getLogger(__name__).warning(
                    "Native MLX batched prefill progress unavailable: upstream PromptProcessingBatch "
                    "no longer exposes %s (%s)", ", ".join(cls._PROMPT_BATCH_ROW_FIELDS), exc)
            return []
        rows = []
        for i, uid in enumerate(uids):
            if i >= len(totals) or i >= len(cached) or i >= len(suffix) or i >= len(left):
                break
            done = columns if right_padded else columns - left[i]
            done = min(suffix[i], max(0, done))
            rows.append((uid, min(totals[i], cached[i] + done), totals[i], cached[i]))
        return rows

    @staticmethod
    def _text_delta(job, text, final=False):
        job.pending_text += text
        stops = job.request.stop
        matches = [job.pending_text.find(s) for s in stops if s in job.pending_text]
        if matches:
            text = job.pending_text[:min(matches)]
            job.pending_text = ""
            return text, True
        keep = 0
        if not final:
            for stop in stops:
                for n in range(1, min(len(stop), len(job.pending_text) + 1)):
                    if job.pending_text.endswith(stop[:n]):
                        keep = max(keep, n)
        text = job.pending_text[:-keep] if keep else job.pending_text
        job.pending_text = job.pending_text[-keep:] if keep else ""
        return text, False

    def _run_batch(self, first):
        first.cache = self._cache_for(first)
        batch = [first] + self._take_compatible(first, self.max_batch_size - 1, initial=True)
        engine, active = None, {}
        mode = "cohort" if first.request.draft_tokens else "continuous"
        baseline = self._draft_counts()
        try:
            engine = self._backend.BatchGenerator(
                getattr(self.model, "language_model", self.model), self.processor,
                completion_batch_size=self.max_batch_size, prefill_batch_size=self.max_batch_size,
                prefill_step_size=256, compute_logprobs=False, greedy_sampling=True,
                draft_model=self.drafter if first.request.draft_tokens else None,
                draft_kind=self.draft_kind if first.request.draft_tokens else None,
                draft_block_size=first.request.draft_tokens + 1 if first.request.draft_tokens else None,
                apc_manager=first.cache,
            )

            def insert(jobs):
                for job in jobs:
                    if job.cancelled.is_set():
                        self._finish(job)
                        continue
                    job.started = time.monotonic()
                    job.mode, job.draft_before = mode, baseline
                    job.cohort_size = len(batch) if mode == "cohort" else 1
                    try:
                        tokens, kwargs = self._prepare(job)
                        if job.cancelled.is_set():
                            self._finish(job)
                            continue
                        uid = engine.insert([tokens], max_tokens=[job.request.max_tokens],
                                            prompt_kwargs=[kwargs])[0]
                        active[uid] = job
                    except BaseException as exc:
                        self._finish(job, exc)

            def observe_batch_sizes(prompt_responses, responses):
                # Admitted/queued jobs need not be rows of the same tensor
                # batch (e.g. incompatible vision shapes or APC warm states).
                groups = [{r.uid for r in prompt_responses}, {r.uid for r in responses}]
                for name in ("_prompt_batch", "_generation_batch"):
                    current = getattr(engine, name, None)
                    groups.append(set(getattr(current, "uids", ()) or ()))
                peak = max((len(group) for group in groups), default=0)
                with self._condition:
                    self._counters["peak_batch_size"] = max(self._counters["peak_batch_size"], peak)
                for group in groups:
                    for uid in group:
                        if uid in active:
                            active[uid].peak_batch_size = max(active[uid].peak_batch_size, len(group))

            insert(batch)
            while active:
                self._discard_expired()
                for uid, job in tuple(active.items()):
                    if job.cancelled.is_set():
                        # False during multirow prefill means upstream cannot
                        # remove yet. Tombstone locally; never deliver its tokens.
                        if engine.remove(uid):
                            active.pop(uid)
                            self._finish(job)
                if not active or all(job.cancelled.is_set() for job in active.values()):
                    break
                if mode == "continuous":
                    insert(self._take_compatible(first, self.max_batch_size - len(active)))
                    # Upstream admits a new prefill only when free decode
                    # slots >= prefill_batch_size. A fixed full-size prefill
                    # batch would silently turn continuous mode into cohorts.
                    current = getattr(engine, "_generation_batch", None)
                    decoding = len(current) if current is not None else 0
                    engine.prefill_batch_size = max(1, self.max_batch_size - decoding)
                prompt_responses, responses = engine.next()
                observe_batch_sizes(prompt_responses, responses)
                # Mid-prefill progress: `next()` advanced the prompt batch by
                # at most one chunk; report where each listening row stands.
                prompt_batch = getattr(engine, "_prompt_batch", None)
                if prompt_batch is not None and any(j.reports_prefill for j in active.values()):
                    for uid, processed, total, cached in self._prompt_batch_rows(prompt_batch):
                        job = active.get(uid)
                        if job is not None:
                            self._emit_prefill(job, processed_tokens=processed,
                                               prompt_tokens=total, cached_tokens=cached)
                for response in prompt_responses:
                    job = active.get(response.uid)
                    if job is not None:
                        job.prompt_tokens = int(response.prompt_tokens)
                        job.prompt_tps = float(getattr(response, "prompt_tps", 0.0))
                        job.cached_tokens = int(getattr(response, "cached_tokens", 0))
                for response in responses:
                    job = active.get(response.uid)
                    if job is None:
                        continue
                    if job.cancelled.is_set():
                        if response.finish_reason is not None:
                            active.pop(response.uid)
                            self._finish(job)
                        continue
                    token, finish = response.token, response.finish_reason
                    if token is not None:
                        if not job.first_token_at:
                            job.first_token_at = time.monotonic()
                        job.generation_tokens += 1
                        if finish != "stop":
                            job.detokenizer.add_token(int(token))
                    if finish is not None:
                        job.detokenizer.finalize()
                    delta, stopped = self._text_delta(job, job.detokenizer.last_segment, finish is not None)
                    if stopped:
                        finish = "stop"
                        engine.remove(response.uid)
                    self._emit(job, self._snapshot(job, delta, token, finish))
                    if finish is not None:
                        active.pop(response.uid)
                        self._finish(job)
                if not engine.has_work and active:
                    raise NativeRuntimeError("Native batch engine stopped without terminal responses")
        except BaseException as exc:
            for job in batch + list(active.values()):
                self._finish(job, exc)
        finally:
            for job in active.values():
                if not job.done.is_set():
                    self._finish(job, None if job.cancelled.is_set() else NativeRuntimeError("Native batch interrupted"))
            if engine is not None:
                rounds = getattr(getattr(engine, "_generation_batch", None), "_rounds_iter", None)
                if rounds is not None and hasattr(rounds, "close"):
                    rounds.close()
                engine.close()

    def _run_exclusive(self, job):
        job.started, job.mode = time.monotonic(), "exclusive"
        job.draft_before = self._draft_counts()
        request = job.request
        kwargs = _thaw(request.sampling)
        # Never let a dictionary override normalized admission controls.
        reserved = {"sampler", "max_tokens", "temperature", "top_p", "top_k", "seed",
                    "draft_model", "draft_kind", "draft_block_size", "apc_manager", "apc_tenant"}
        if reserved.intersection(kwargs):
            raise NativeRuntimeError("Native sampling dictionary contains reserved execution controls", code="invalid_controls")
        supported = {"min_p", "repetition_penalty", "repetition_context_size", "presence_penalty",
                     "presence_context_size", "frequency_penalty", "frequency_context_size", "logit_bias"}
        if set(kwargs) - supported:
            raise NativeRuntimeError("Unsupported native sampling controls: " + ", ".join(sorted(set(kwargs) - supported)), code="invalid_controls")
        neutral = {"repetition_penalty": (None, 1, 1.0), "presence_penalty": (None, 0, 0.0),
                   "frequency_penalty": (None, 0, 0.0), "logit_bias": (None, {})}
        if request.draft_tokens and any(kwargs.get(k) not in values for k, values in neutral.items()):
            raise NativeRuntimeError("Native MTP cannot honor logits processors; disable speculation for this request", code="invalid_controls")
        kwargs.update(max_tokens=request.max_tokens, temperature=request.temperature,
                      top_p=request.top_p, top_k=request.top_k, seed=request.seed,
                      prefill_step_size=256)
        if request.draft_tokens:
            kwargs.update(draft_model=self.drafter, draft_kind=self.draft_kind,
                          draft_block_size=request.draft_tokens + 1)
        job.cache = self._cache_for(job)
        if job.cache is not None:
            kwargs.update(apc_manager=job.cache, apc_tenant=self._tenant(request, "exclusive"))
        if request.media:
            from .mlx_qwen4 import prepare_images

            parts = [(i, SimpleNamespace(content=data, file_path=None)) for i, data in request.media]
            inputs, job.media_records = prepare_images(self.model, self.processor, request.prompt, parts)
            kwargs.update(inputs)
        if job.cancelled.is_set():
            self._finish(job)
            return
        generator = self._backend.stream_generate(self.model, self.processor, request.prompt, **kwargs)
        error, terminal_seen = None, False
        # Mid-prefill progress: mlx-vlm's chunked prefill runs on THIS thread
        # inside the first `next()`, and reports each evaluated chunk to its
        # "Prefill" bar (mlx_prefill_observer). Bound for the job only.
        from .mlx_prefill_observer import observe_prefill

        def _observe_prefill(done, total):
            self._emit_prefill(job, fed_processed=int(done), fed_total=int(total))

        observing = observe_prefill(_observe_prefill if job.reports_prefill else None)
        observing.__enter__()
        try:
            for result in generator:
                self._discard_expired()
                if job.cancelled.is_set():
                    break
                token = getattr(result, "token", None)
                if token is not None and not job.first_token_at:
                    job.first_token_at = time.monotonic()
                job.prompt_tokens = int(getattr(result, "prompt_tokens", job.prompt_tokens))
                job.generation_tokens = int(getattr(result, "generation_tokens", job.generation_tokens))
                job.cached_tokens = int(getattr(result, "cached_tokens", 0))
                job.prompt_tps = float(getattr(result, "prompt_tps", 0.0))
                backend_finish = getattr(result, "finish_reason", None)
                delta, stopped = self._text_delta(job, getattr(result, "text", ""), final=backend_finish is not None)
                finish = "stop" if stopped else backend_finish
                self._emit(job, self._snapshot(job, delta, token, finish, source=result))
                if finish is not None:
                    terminal_seen = True
                    break
            if not job.cancelled.is_set() and not terminal_seen:
                delta, _ = self._text_delta(job, "", final=True)
                finish = "length" if job.generation_tokens >= request.max_tokens else "stop"
                self._emit(job, self._snapshot(job, delta, finish_reason=finish))
        except BaseException as exc:
            error = exc
        finally:
            observing.__exit__(None, None, None)
            try:
                generator.close()
            except BaseException as exc:
                error = error or exc
            self._finish(job, error)
