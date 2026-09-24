"""Host jobs: downloads, deletes and engine installs as pollable background work.

Contract E (`host_job_v1`). Every long action a console can start on the HOST
-- fetching weights, deleting weights, running an engine installer -- is a
job with ONE dict shape, so every poller (core web console, gateway web
console, both terminal consoles, the CLI) renders all three kinds with the
same code.

THE THREE PROPERTIES, and why each exists:

- NOT ON THE CALLER'S THREAD. A download is minutes of I/O; the request that
  starts it gets a job id back at once and polls.
- SINGLE-FLIGHT PER KEY. A second request for the same artifact JOINS the
  running job (`joined` counts them) instead of starting a second `lms get`
  writing the same files. Engine installs are stricter: a second install
  while one runs is refused (`JobBusy`), because two package managers racing
  on one host is never what anyone meant.
- CANCELLABLE. A subprocess is terminated (then killed); an HTTP stream stops
  at its next line; a Hugging Face download stops at its next progress tick.
  Cancellation is cooperative through `JobControl`, which the materializer
  reads from a context variable, so provider code needs no job plumbing.

CROSS-PROCESS VISIBILITY (optional). A registry built with `persist_dir`
writes each job's snapshot as `<job_id>.json` there, and honours a
`<job_id>.cancel` marker file. That is what lets `abstractcore models jobs`
(a fresh process) list work started by `abstractcore serve` or by a detached
`models download --detach`, and `abstractcore models cancel <id>` stop it --
without any IPC beyond files. Snapshots whose owning process has died are
reported `failed` ("owner process exited"), never left `running` forever.

Field names stay compatible with the gateway's `model_downloads.py` job dict
(`job`, `events`, `elapsed_s`, `result` are emitted as aliases), so the
gateway can delegate to this registry without breaking its pollers.

PROGRESS YOU CAN SEE (download jobs). `status` is the coarse lifecycle
(queued|running|completed|failed|cancelled) every existing poller reads. On
top of it a download job carries the fine-grained progress contract:

- `state`: queued | resolving | downloading | verifying | installing | done |
  failed | cancelled | stalled.
- `bytes_done` / `bytes_total` (aliases of `downloaded_bytes`/`total_bytes`),
  `percent`, `size_unknown` + `size_note` when the source cannot say.
- `bytes_per_second` over a recent window (it DECAYS to 0 when bytes stop --
  a frozen speed is a lie), `eta_s`, `updated_at`.
- `files`: `[{name, bytes_done, bytes_total, state}]`, `current_file`.
- `message`: one plain sentence ("Downloading model.safetensors (2 of 5) ·
  1.2 GB of 4.8 GB · 38 MB/s · 1 min left"); the provider's own last line is
  kept verbatim in `detail`.
- STALL: a job that receives no bytes for `stall_after_s` (default 15 s,
  `ABSTRACTCORE_DOWNLOAD_STALL_S`) turns `stalled`, says so in `message`,
  logs it (logger + `log_tail` + `transitions`), and turns back to
  `downloading` by itself the moment bytes move again.

ONE registry TICKER thread (every 0.5 s, alive only while a job is active)
recomputes speed/ETA/stall and persists the
snapshot, so a poller sees fresh numbers even when the provider is silent.
Every tick that changed something is also appended to `progress_events`
(in memory, uncapped, read with `HostJobRegistry.events()`) and, with a
`persist_dir`, to `<job_id>.events.jsonl` -- never rate-limited by count
(ADR-0026).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "HOST_JOB_SCHEMA",
    "JOB_KINDS",
    "JOB_STATUSES",
    "DOWNLOAD_STATES",
    "default_stall_after_s",
    "format_bytes",
    "format_duration",
    "JobBusy",
    "JobCancelled",
    "JobControl",
    "HostJobRegistry",
    "current_job_control",
    "default_registry",
    "default_jobs_dir",
    "read_persisted_jobs",
    "read_persisted_job",
    "request_cancel",
    "start_download_job",
    "start_delete_job",
    "cli_equivalent_download",
    "cli_equivalent_delete",
    "cli_equivalent_engine_install",
]

HOST_JOB_SCHEMA = "host_job_v1"
JOB_KINDS = ("download", "delete", "engine_install")
JOB_STATUSES = ("queued", "running", "completed", "failed", "cancelled")
DOWNLOAD_STATES = (
    "queued",
    "resolving",
    "downloading",
    "verifying",
    "installing",
    "done",
    "failed",
    "cancelled",
    "stalled",
)
_ACTIVE = ("queued", "running")
_MAX_TAIL = 60
_MAX_JOBS = 40
_PERSIST_MIN_INTERVAL_S = 0.5
_KIND_PREFIX = {"download": "dl", "delete": "rm", "engine_install": "eng"}
_TICK_S = 0.5
_SPEED_WINDOW_S = 5.0
_DEFAULT_STALL_S = 15.0
# States in which "no bytes arrived" means something is wrong. Verifying and
# installing legitimately move no bytes (hashing a 5 GB blob takes a while).
_STALLABLE = ("resolving", "downloading", "stalled")
_PHASES = ("resolving", "downloading", "verifying", "installing")

logger = logging.getLogger("abstractcore.host_jobs")


def default_stall_after_s() -> float:
    """`$ABSTRACTCORE_DOWNLOAD_STALL_S` (seconds without bytes), default 15."""

    raw = str(os.getenv("ABSTRACTCORE_DOWNLOAD_STALL_S") or "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_STALL_S
    except ValueError:
        value = _DEFAULT_STALL_S
    return value if value > 0 else _DEFAULT_STALL_S


def format_bytes(value: Any) -> str:
    """Decimal units, the way disks and hubs count: `1.2 GB`, `38 MB`, `512 KB`."""

    if not isinstance(value, (int, float)) or value < 0:
        return "?"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            n = value / scale
            return f"{n:.1f} {unit}" if n < 10 or unit == "GB" else f"{n:.0f} {unit}"
    return f"{int(value)} B"


def format_duration(seconds: Any) -> str:
    """`45 s`, `3 min`, `1 h 20 min` -- rounded up, never `0 s` for work left."""

    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "?"
    s = int(seconds + 0.999)
    if s < 60:
        return f"{max(1, s)} s"
    if s < 3600:
        return f"{(s + 59) // 60} min"
    h, rest = divmod(s, 3600)
    return f"{h} h {rest // 60} min" if rest >= 60 else f"{h} h"


class JobBusy(RuntimeError):
    """A job for this key is already running and joining is not allowed."""

    def __init__(self, message: str, job: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.job = job


class JobCancelled(RuntimeError):
    """Raised inside a job when its control has been cancelled."""


def _now_iso(ts: Optional[float] = None) -> Optional[str]:
    if ts is None:
        return None
    import datetime as _dt

    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_iso_ms(ts: Optional[float]) -> Optional[str]:
    """ISO-8601 UTC with milliseconds: `updated_at` moves every 0.5 s."""

    if ts is None:
        return None
    import datetime as _dt

    stamp = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


def default_jobs_dir() -> Path:
    """`$ABSTRACTCORE_JOBS_DIR`, else `<abstractcore config dir>/jobs`."""

    explicit = str(os.getenv("ABSTRACTCORE_JOBS_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    base = str(os.getenv("ABSTRACTCORE_CONFIG_DIR") or "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".abstractcore" / "config"
    return root / "jobs"


# ---------------------------------------------------------------------------
# JobControl: what provider code sees
# ---------------------------------------------------------------------------


class JobControl:
    """The cooperative-cancel handle a running job exposes to provider code."""

    def __init__(self, job_id: str, on_cancel_probe: Optional[Callable[[], bool]] = None):
        self.job_id = job_id
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._procs: List[subprocess.Popen] = []
        self._callbacks: List[Callable[[], None]] = []
        self._probe = on_cancel_probe

    @property
    def event(self) -> threading.Event:
        return self._event

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Run `callback` when the job is cancelled (at once if it already is).

        For transfers that BLOCK in a read -- an HTTP stream that went quiet
        cannot notice a flag -- closing the socket is the only way to stop
        within a second.
        """

        with self._lock:
            self._callbacks.append(callback)
        if self._event.is_set():
            self._fire(callback)

    @staticmethod
    def _fire(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:
            pass

    def is_cancelled(self) -> bool:
        if not self._event.is_set() and self._probe is not None:
            try:
                if self._probe():
                    self.cancel()
            except Exception:
                pass
        return self._event.is_set()

    def check(self) -> None:
        if self.is_cancelled():
            raise JobCancelled(f"job {self.job_id} cancelled")

    def register_process(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._procs.append(proc)
        if self._event.is_set():
            self._terminate(proc)

    def cancel(self) -> None:
        if self._event.is_set():
            return
        self._event.set()
        with self._lock:
            procs = list(self._procs)
            callbacks = list(self._callbacks)
        for proc in procs:
            self._terminate(proc)
        for callback in callbacks:
            self._fire(callback)

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """SIGTERM the process GROUP (installers spawn children), then SIGKILL."""

        def _signal(sig: int) -> None:
            try:
                if os.name != "nt" and getattr(proc, "_abstractcore_own_group", False):
                    os.killpg(proc.pid, sig)
                elif sig == 15:
                    proc.terminate()
                else:
                    proc.kill()
            except Exception:
                pass

        def _kill() -> None:
            try:
                if proc.poll() is None:
                    _signal(15)
                    try:
                        proc.wait(timeout=5.0)
                    except Exception:
                        _signal(9)
            except Exception:
                pass

        threading.Thread(target=_kill, daemon=True).start()


_current_control: "contextvars.ContextVar[Optional[JobControl]]" = contextvars.ContextVar(
    "abstractcore_host_job_control", default=None
)


def current_job_control() -> Optional[JobControl]:
    """The `JobControl` of the job running on this thread, if any."""

    return _current_control.get()


# ---------------------------------------------------------------------------
# The job record
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    job_id: str
    kind: str
    key: str
    provider: Optional[str] = None
    artifact: Optional[str] = None
    engine: Optional[str] = None
    status: str = "queued"
    percent: Optional[float] = None
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    message: str = ""
    log_tail: List[str] = field(default_factory=list)
    command: List[str] = field(default_factory=list)
    dry_run: bool = False
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    joined: int = 0
    cli_equivalent: str = ""
    result: Optional[Dict[str, Any]] = None
    pid: int = field(default_factory=os.getpid)
    control: Optional[JobControl] = None
    # --- the progress contract (see the module docstring) -------------------
    state: str = "queued"
    updated_at: float = field(default_factory=time.time)
    detail: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)
    current_file: Optional[str] = None
    size_unknown: Optional[bool] = None
    size_note: Optional[str] = None
    bytes_per_second: Optional[float] = None
    eta_s: Optional[int] = None
    stall_after_s: float = _DEFAULT_STALL_S
    last_bytes_at: Optional[float] = None
    stalled_since: Optional[float] = None
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    samples: List[Any] = field(default_factory=list)
    progress_events: List[Dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    def _size_unknown(self) -> bool:
        if self.size_unknown is not None:
            return bool(self.size_unknown)
        return bool(self.state in ("downloading", "stalled") and not self.total_bytes and self.downloaded_bytes)

    def to_dict(self) -> Dict[str, Any]:
        end = self.finished_at or time.time()
        stalled_for = None
        if self.state == "stalled" and self.stalled_since is not None:
            stalled_for = round(time.time() - self.stalled_since, 1)
        return {
            "schema": HOST_JOB_SCHEMA,
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "provider": self.provider,
            "artifact": self.artifact,
            "engine": self.engine,
            "percent": self.percent,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "message": self.message,
            "log_tail": list(self.log_tail),
            "command": list(self.command),
            "dry_run": bool(self.dry_run),
            "started_at": _now_iso(self.started_at),
            "finished_at": _now_iso(self.finished_at),
            "error": self.error,
            "joined": int(self.joined),
            "cli_equivalent": self.cli_equivalent,
            "result": dict(self.result) if isinstance(self.result, dict) else None,
            "pid": self.pid,
            # Aliases for the gateway's model_downloads.py pollers.
            "job": self.job_id,
            "events": list(self.log_tail),
            "elapsed_s": round(end - self.started_at, 1),
            # The progress contract.
            "state": self.state,
            "bytes_done": self.downloaded_bytes,
            "bytes_total": self.total_bytes,
            "size_unknown": self._size_unknown(),
            "size_note": self.size_note,
            "bytes_per_second": self.bytes_per_second,
            "eta_s": self.eta_s,
            "updated_at": _now_iso_ms(self.updated_at),
            "detail": self.detail,
            "files": [dict(f) for f in self.files],
            "current_file": self.current_file,
            "stall_after_s": self.stall_after_s,
            "stalled_for_s": stalled_for,
            "transitions": [dict(t) for t in self.transitions],
            "progress_events_count": len(self.progress_events),
            "cancel_requested": bool(self.cancel_requested),
        }


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


Runner = Callable[["JobContext"], Dict[str, Any]]

# `DownloadStatus` value -> contract phase. COMPLETE/ERROR/CANCELLED are not
# phases: the job's END (the runner's result) decides done/failed/cancelled.
_STATUS_PHASE = {"starting": "resolving", "downloading": "downloading", "verifying": "verifying"}


def _phase_of(phase: Any, status: Any) -> Optional[str]:
    explicit = str(phase or "").strip().lower()
    if explicit in _PHASES:
        return explicit
    raw = getattr(status, "value", status)
    return _STATUS_PHASE.get(str(raw or "").strip().lower())


class JobContext:
    """What a runner receives: progress/log sinks plus the cancel handle."""

    def __init__(self, registry: "HostJobRegistry", job_id: str, control: JobControl):
        self._registry = registry
        self.job_id = job_id
        self.control = control

    def progress(self, progress: Any) -> None:
        """Accepts a `DownloadProgress` (or any object/dict with the same fields)."""

        get = (lambda k: progress.get(k)) if isinstance(progress, dict) else (lambda k: getattr(progress, k, None))
        self._registry._update(
            self.job_id,
            message=str(get("message") or "").strip(),
            percent=get("percent"),
            downloaded_bytes=get("downloaded_bytes"),
            total_bytes=get("total_bytes"),
            phase=_phase_of(get("phase"), get("status")),
            files=get("files"),
            current_file=get("current_file"),
            size_unknown=get("size_unknown"),
            size_note=get("size_note"),
        )

    def log(self, line: str) -> None:
        self._registry._update(self.job_id, message=str(line or "").strip())

    def set_command(self, argv: List[str]) -> None:
        self._registry._update(self.job_id, command=list(argv))


class HostJobRegistry:
    """In-process registry of host jobs. Thread-safe; one per process is normal."""

    def __init__(
        self,
        *,
        max_jobs: int = _MAX_JOBS,
        persist_dir: Optional[Path] = None,
        tick_s: float = _TICK_S,
        stall_after_s: Optional[float] = None,
        speed_window_s: float = _SPEED_WINDOW_S,
    ):
        self._lock = threading.RLock()
        self._jobs: Dict[str, _Job] = {}
        self._by_key: Dict[str, str] = {}
        self._max_jobs = int(max_jobs)
        self._persist_dir = Path(persist_dir).expanduser() if persist_dir else None
        self._last_persist: Dict[str, float] = {}
        self._tick_s = float(tick_s) if tick_s and tick_s > 0 else _TICK_S
        self._stall_after_s = float(stall_after_s) if stall_after_s and stall_after_s > 0 else None
        self._speed_window_s = float(speed_window_s) if speed_window_s and speed_window_s > 0 else _SPEED_WINDOW_S
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._ticker: Optional[threading.Thread] = None

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
        """Call `callback(snapshot)` on every progress change (for push streams).

        Returns a function that removes the listener. Callbacks run on the
        job's thread and must not block.
        """

        with self._lock:
            self._listeners.append(callback)

        def _remove() -> None:
            with self._lock:
                try:
                    self._listeners.remove(callback)
                except ValueError:
                    pass

        return _remove

    def events(self, job_id: str) -> Optional[List[Dict[str, Any]]]:
        """Every progress event of a job in this process (uncapped), or None."""

        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            return [dict(e) for e in job.progress_events] if job is not None else None

    # --- public API -----------------------------------------------------------

    @property
    def persist_dir(self) -> Optional[Path]:
        return self._persist_dir

    def start(
        self,
        *,
        kind: str,
        key: str,
        runner: Runner,
        provider: Optional[str] = None,
        artifact: Optional[str] = None,
        engine: Optional[str] = None,
        command: Optional[List[str]] = None,
        dry_run: bool = False,
        cli_equivalent: str = "",
        join: bool = True,
        job_id: Optional[str] = None,
        run_inline: bool = False,
    ) -> Dict[str, Any]:
        """Start (or join) a job. Returns its snapshot.

        `join=False` raises `JobBusy` when a job for `key` is active.
        `run_inline=True` runs the job on the calling thread (the CLI's
        foreground mode) and returns the FINISHED snapshot.
        """

        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}; expected one of {', '.join(JOB_KINDS)}")
        with self._lock:
            existing = self._jobs.get(self._by_key.get(key, ""))
            if existing is not None and existing.status in _ACTIVE:
                if not join:
                    raise JobBusy(f"a {kind} job is already running for {key} ({existing.job_id})", existing.to_dict())
                existing.joined += 1
                self._persist(existing, force=True)
                return existing.to_dict()
            jid = job_id or f"{_KIND_PREFIX.get(kind, 'job')}_{uuid.uuid4().hex[:12]}"
            job = _Job(
                job_id=jid,
                kind=kind,
                key=key,
                provider=provider,
                artifact=artifact,
                engine=engine,
                command=list(command or []),
                dry_run=bool(dry_run),
                cli_equivalent=cli_equivalent,
                message="queued",
                stall_after_s=self._stall_after_s or default_stall_after_s(),
            )
            self._transition(job, "queued", "queued")
            job.control = JobControl(jid, on_cancel_probe=self._cancel_marker_probe(jid))
            self._jobs[jid] = job
            self._by_key[key] = jid
            self._prune_locked()
            self._persist(job, force=True)

        self._ensure_ticker()
        if run_inline:
            self._run(jid, runner)
            return self.get(jid) or job.to_dict()
        try:
            threading.Thread(target=self._run, args=(jid, runner), name=f"host-job-{jid}", daemon=True).start()
        except Exception as exc:
            self._finish(jid, status="failed", error=f"could not start the job worker: {exc}")
        return self.get(jid) or job.to_dict()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            return job.to_dict() if job is not None else None

    def list(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)
            return [j.to_dict() for j in jobs if kind is None or j.kind == kind]

    def active_for(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(self._by_key.get(key, ""))
            if job is None or job.status not in _ACTIVE:
                return None
            return job.to_dict()

    def active(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        return [j for j in self.list(kind) if j.get("status") in _ACTIVE]

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Request cancellation. Returns the snapshot, or None when unknown here."""

        with self._lock:
            job = self._jobs.get(str(job_id or "").strip())
            if job is None:
                return None
            if job.status not in _ACTIVE:
                return job.to_dict()
            control = job.control
            job.message = "cancelling"
            job.cancel_requested = True
            job.updated_at = time.time()
            self._append_tail(job, "cancel requested")
            self._persist(job, force=True)
        if control is not None:
            control.cancel()
        return self.get(job_id)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            snap = self.get(job_id)
            if snap is None or snap.get("status") not in _ACTIVE:
                return snap
            if deadline is not None and time.monotonic() >= deadline:
                return snap
            time.sleep(0.05)

    def reset_for_tests(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._by_key.clear()

    # --- internals ---------------------------------------------------------------

    def _cancel_marker_probe(self, job_id: str) -> Optional[Callable[[], bool]]:
        if self._persist_dir is None:
            return None
        marker = self._persist_dir / f"{job_id}.cancel"
        return marker.exists

    def _run(self, job_id: str, runner: Runner) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            control = job.control or JobControl(job_id)
            job.status = "running"
            job.message = job.message if job.message and job.message != "queued" else "running"
            now = time.time()
            job.last_bytes_at = now
            if job.kind == "download":
                self._transition(job, "resolving", "preparing the download")
                job.message = "Preparing the download"
            else:
                self._transition(job, "running", "running")
            self._persist(job, force=True)
        ctx = JobContext(self, job_id, control)

        # The registry's TICKER (one thread for all active jobs) turns an
        # on-disk cancel marker into a real cancel even while the runner is
        # blocked in a subprocess read, and keeps speed/ETA/stall honest while
        # the provider is silent. It is started by `start()`.
        stop_watch = threading.Event()

        token = _current_control.set(control)
        try:
            result = runner(ctx) or {}
        except JobCancelled:
            result = {"ok": False, "status": "cancelled", "message": "cancelled"}
        except Exception as exc:
            result = {"ok": False, "status": "failed", "message": str(exc)}
        finally:
            _current_control.reset(token)
            stop_watch.set()

        if control.is_cancelled() and not result.get("ok"):
            self._finish(job_id, status="cancelled", result=result, error=None)
            return
        ok = bool(result.get("ok"))
        self._finish(
            job_id,
            status="completed" if ok else "failed",
            result=result,
            error=None if ok else str(result.get("message") or "failed"),
        )

    def _append_tail(self, job: _Job, line: str) -> None:
        if not line:
            return
        if job.log_tail and job.log_tail[-1] == line:
            return
        job.log_tail.append(line)
        if len(job.log_tail) > _MAX_TAIL:
            del job.log_tail[: len(job.log_tail) - _MAX_TAIL]

    def _update(
        self,
        job_id: str,
        *,
        message: str = "",
        percent: Any = None,
        downloaded_bytes: Any = None,
        total_bytes: Any = None,
        command: Optional[List[str]] = None,
        phase: Optional[str] = None,
        files: Any = None,
        current_file: Any = None,
        size_unknown: Any = None,
        size_note: Any = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = time.time()
            job.updated_at = now
            is_download = job.kind == "download"
            if message:
                if is_download:
                    job.detail = message
                job.message = message
            if isinstance(percent, (int, float)):
                job.percent = round(float(percent), 2)
            moved = False
            if isinstance(downloaded_bytes, int) and not isinstance(downloaded_bytes, bool):
                previous = job.downloaded_bytes
                job.downloaded_bytes = int(downloaded_bytes)
                if previous is None or downloaded_bytes > previous:
                    moved = bool(downloaded_bytes > (previous or 0))
                if previous is not None and downloaded_bytes < previous:
                    job.samples.clear()  # a source that re-based its count
                job.samples.append((now, int(downloaded_bytes)))
            if isinstance(total_bytes, int) and total_bytes > 0:
                job.total_bytes = int(total_bytes)
            if not isinstance(percent, (int, float)) and isinstance(downloaded_bytes, int) and job.total_bytes:
                job.percent = round(min(100.0, downloaded_bytes / job.total_bytes * 100.0), 2)
            if command is not None:
                job.command = list(command)
            if isinstance(files, list):
                job.files = [dict(f) for f in files if isinstance(f, dict)]
            if current_file is not None:
                job.current_file = str(current_file) or None
            if size_unknown is not None:
                job.size_unknown = bool(size_unknown)
            if size_note:
                job.size_note = str(size_note)
            if moved:
                job.last_bytes_at = now
            if is_download:
                self._apply_phase(job, phase, moved, now)
                self._recompute(job, now)
            # Only STATE changes earn a tail line: a byte counter that ticks
            # thousands of times must not become thousands of rows.
            if message and percent is None and downloaded_bytes is None:
                self._append_tail(job, message)
            elif message and not job.log_tail:
                self._append_tail(job, message)
            self._persist(job)
            self._record(job, now)

    # --- the progress contract ---------------------------------------------------

    def _transition(self, job: _Job, state: str, why: str) -> None:
        if job.state == state and job.transitions:
            return
        job.state = state
        job.transitions.append({"at": _now_iso_ms(time.time()), "state": state, "why": why})

    def _apply_phase(self, job: _Job, phase: Optional[str], moved: bool, now: float) -> None:
        if job.status not in _ACTIVE:
            return
        if job.state == "stalled":
            if moved:
                stalled_for = now - (job.stalled_since or now)
                job.stalled_since = None
                why = f"resumed: bytes are moving again after {format_duration(stalled_for)} without data"
                logger.info("download %s %s", job.job_id, why)
                self._append_tail(job, why)
                self._transition(job, "downloading", why)
            elif phase in ("verifying", "installing"):
                job.stalled_since = None
                self._transition(job, phase, job.detail or phase)
            return
        if phase and phase != job.state:
            if phase == "downloading" or phase in ("verifying", "installing") or job.state in ("queued", "resolving"):
                if phase in ("resolving", "downloading") and job.state in ("resolving", "queued"):
                    job.last_bytes_at = now  # the no-bytes clock starts with the phase
                self._transition(job, phase, job.detail or phase)
        elif moved and job.state in ("queued", "resolving"):
            self._transition(job, "downloading", "first bytes arrived")

    def _speed(self, job: _Job, now: float) -> Optional[float]:
        samples = job.samples
        if not samples:
            return None
        window = self._speed_window_s
        horizon = now - window * 1.6
        while len(samples) > 2 and samples[1][0] < horizon:
            samples.pop(0)
        anchor = samples[0]
        for sample in samples:
            if sample[0] <= now - window:
                anchor = sample
            else:
                break
        latest = samples[-1]
        span = now - anchor[0]
        if span < 0.4:
            return None
        return max(0.0, (latest[1] - anchor[1]) / span)

    def _recompute(self, job: _Job, now: float) -> None:
        """Speed, ETA, stall and the plain-language message of an active download."""

        if job.status not in _ACTIVE:
            return
        speed = self._speed(job, now)
        job.bytes_per_second = round(speed, 1) if speed is not None else None
        remaining = None
        if job.total_bytes and job.downloaded_bytes is not None:
            remaining = max(0, job.total_bytes - job.downloaded_bytes)
        if remaining is not None and speed and speed > 0 and job.state in ("downloading",):
            job.eta_s = int(remaining / speed + 0.999)
        elif remaining == 0:
            job.eta_s = 0
        else:
            job.eta_s = None

        quiet = now - (job.last_bytes_at or now)
        if job.state in _STALLABLE and job.state != "stalled" and quiet >= job.stall_after_s and not job.cancel_requested:
            job.stalled_since = job.last_bytes_at or now
            where = (
                f" at {format_bytes(job.downloaded_bytes)}" + (f" of {format_bytes(job.total_bytes)}" if job.total_bytes else "")
                if job.downloaded_bytes
                else ""
            )
            why = f"stalled: no bytes received for {format_duration(quiet)}{where}"
            logger.warning("download %s (%s %s) %s", job.job_id, job.provider, job.artifact, why)
            self._append_tail(job, why)
            self._transition(job, "stalled", why)
        job.message = self._compose(job, now)

    def _compose(self, job: _Job, now: float) -> str:
        if job.cancel_requested:
            return "Cancelling…"
        done, total = job.downloaded_bytes, job.total_bytes
        amount = ""
        if isinstance(done, int) and total:
            amount = f"{format_bytes(done)} of {format_bytes(total)}"
        elif isinstance(done, int) and done > 0:
            amount = f"{format_bytes(done)} so far"
        elif total:
            amount = f"{format_bytes(total)} to fetch"
        if job.state == "stalled":
            quiet = now - (job.stalled_since or now)
            parts = [f"Stalled: no data for {format_duration(quiet)}"]
            if amount:
                parts.append(amount)
            parts.append("still trying, it resumes by itself when data flows again")
            return " · ".join(parts)
        if job.state == "downloading":
            head = "Downloading"
            if job.current_file:
                names = [str(f.get("name")) for f in job.files]
                if job.current_file in names and len(names) > 1:
                    head = f"Downloading {job.current_file} ({names.index(job.current_file) + 1} of {len(names)})"
                else:
                    head = f"Downloading {job.current_file}"
            parts = [head]
            if amount:
                parts.append(amount)
            if job.bytes_per_second is not None:
                parts.append(f"{format_bytes(job.bytes_per_second)}/s")
            if job.eta_s is not None and job.eta_s > 0:
                parts.append(f"{format_duration(job.eta_s)} left")
            elif total is None and job._size_unknown():
                parts.append("total size unknown" + (f" ({job.size_note})" if job.size_note else ""))
            return " · ".join(parts)
        label = {"resolving": "Preparing", "verifying": "Verifying", "installing": "Installing"}.get(job.state)
        if label:
            parts = [label]
            if job.detail and job.detail.lower() not in (label.lower(), "preparing the download"):
                parts.append(job.detail)
            if amount and job.state != "resolving":
                parts.append(amount)
            elif total and job.state == "resolving":
                parts.append(amount)
            return " · ".join(parts)
        return job.message

    def _record(self, job: _Job, now: float) -> None:
        """Append a progress event when something a human would see changed."""

        if job.kind != "download":
            return
        event = {
            "t": _now_iso_ms(now),
            "state": job.state,
            "bytes_done": job.downloaded_bytes,
            "bytes_total": job.total_bytes,
            "percent": job.percent,
            "bytes_per_second": job.bytes_per_second,
            "eta_s": job.eta_s,
            "current_file": job.current_file,
            "message": job.message,
        }
        last = job.progress_events[-1] if job.progress_events else None
        if last is not None and all(last.get(k) == event[k] for k in event if k != "t"):
            return
        job.progress_events.append(event)
        if self._persist_dir is not None:
            try:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                with open(self._persist_dir / f"{job.job_id}.events.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, sort_keys=True) + "\n")
            except Exception:
                pass
        if self._listeners:
            snap = job.to_dict()
            for listener in list(self._listeners):
                try:
                    listener(snap)
                except Exception:
                    pass

    def _ensure_ticker(self) -> None:
        with self._lock:
            if getattr(self, "_ticker", None) is not None:
                return
            self._ticker = threading.Thread(target=self._tick_loop, name="host-jobs-ticker", daemon=True)
            ticker = self._ticker
        try:
            ticker.start()
        except Exception as exc:
            # No ticker is degraded (no stall detection, speed only on
            # provider updates), never fatal: the job itself still runs.
            logger.warning("host jobs: could not start the progress ticker: %s", exc)
            with self._lock:
                if self._ticker is ticker:
                    self._ticker = None

    def _tick_loop(self) -> None:
        while True:
            time.sleep(self._tick_s)
            with self._lock:
                active = [(j.job_id, j.control) for j in self._jobs.values() if j.status in _ACTIVE]
                if not active:
                    self._ticker = None
                    return
            for job_id, control in active:
                if self._persist_dir is not None and control is not None and not control.event.is_set():
                    control.is_cancelled()
                self._tick(job_id)

    def _tick(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in _ACTIVE:
                return
            now = time.time()
            if job.kind == "download":
                self._recompute(job, now)
                job.updated_at = now
            self._persist(job)
            self._record(job, now)

    def _finish(
        self,
        job_id: str,
        *,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.result = dict(result) if isinstance(result, dict) else None
            job.finished_at = time.time()
            job.error = error
            if isinstance(result, dict):
                message = str(result.get("message") or "").strip()
                if message:
                    job.message = message
                    self._append_tail(job, message)
                cmd = result.get("command")
                if isinstance(cmd, list) and cmd and not job.command:
                    job.command = [str(c) for c in cmd]
            if status == "cancelled":
                job.message = "cancelled"
                self._append_tail(job, "cancelled")
            if status == "completed" and not job.dry_run:
                job.percent = 100.0
            self._finish_progress(job, status, result)
            # Free the single-flight slot: a failed pull is worth retrying.
            if self._by_key.get(job.key) == job.job_id:
                self._by_key.pop(job.key, None)
            self._persist(job, force=True)
            if self._persist_dir is not None:
                try:
                    (self._persist_dir / f"{job_id}.cancel").unlink()
                except Exception:
                    pass

    def _finish_progress(self, job: _Job, status: str, result: Optional[Dict[str, Any]]) -> None:
        """The contract's terminal state, with numbers that agree with it."""

        now = job.finished_at or time.time()
        job.updated_at = now
        job.bytes_per_second = None
        job.stalled_since = None
        final = {"completed": "done", "failed": "failed", "cancelled": "cancelled"}.get(status, status)
        why = str((result or {}).get("message") or status)
        if job.kind != "download":
            self._transition(job, final, why)
            return
        if status == "completed":
            job.eta_s = 0
            outcome = str((result or {}).get("status") or "")
            if job.total_bytes and not job.dry_run and outcome not in ("already_installed", "planned"):
                job.downloaded_bytes = job.total_bytes
            for entry in job.files:
                if entry.get("state") not in ("done", "skipped"):
                    entry["state"] = "done"
                    if entry.get("bytes_total"):
                        entry["bytes_done"] = entry["bytes_total"]
            job.current_file = None
            if outcome == "completed" and job.downloaded_bytes:
                took = max(0.0, now - job.started_at)
                job.message = f"Downloaded {format_bytes(job.downloaded_bytes)} in {format_duration(took)}"
        else:
            job.eta_s = None
            for entry in job.files:
                if entry.get("state") in ("downloading", "pending", "stalled"):
                    entry["state"] = final
        job.size_unknown = False if status == "completed" else job.size_unknown
        self._transition(job, final, why)
        self._append_tail(job, f"{final}: {why}" if final != why else final)
        # The terminal event is always recorded, even if the tick just wrote one.
        job.progress_events.append(
            {
                "t": _now_iso_ms(now),
                "state": final,
                "bytes_done": job.downloaded_bytes,
                "bytes_total": job.total_bytes,
                "percent": job.percent,
                "bytes_per_second": None,
                "eta_s": job.eta_s,
                "current_file": None,
                "message": job.message,
                "error": job.error,
            }
        )
        if self._persist_dir is not None:
            try:
                with open(self._persist_dir / f"{job.job_id}.events.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(job.progress_events[-1], sort_keys=True) + "\n")
            except Exception:
                pass
        if self._listeners:
            snap = job.to_dict()
            for listener in list(self._listeners):
                try:
                    listener(snap)
                except Exception:
                    pass

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status not in _ACTIVE),
            key=lambda j: j.finished_at or j.started_at,
        )
        for job in finished[: max(0, len(self._jobs) - self._max_jobs)]:
            self._jobs.pop(job.job_id, None)
            if self._by_key.get(job.key) == job.job_id:
                self._by_key.pop(job.key, None)
        if self._persist_dir is not None:
            _prune_persisted(self._persist_dir, self._max_jobs)

    def _persist(self, job: _Job, *, force: bool = False) -> None:
        if self._persist_dir is None:
            return
        now = time.monotonic()
        if not force and now - self._last_persist.get(job.job_id, 0.0) < _PERSIST_MIN_INTERVAL_S:
            return
        self._last_persist[job.job_id] = now
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            target = self._persist_dir / f"{job.job_id}.json"
            tmp = target.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(job.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cross-process view (files)
# ---------------------------------------------------------------------------


def _pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except Exception:
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _normalize_persisted(snap: Dict[str, Any]) -> Dict[str, Any]:
    if snap.get("status") in _ACTIVE and not _pid_alive(snap.get("pid")):
        snap = dict(snap)
        snap["status"] = "failed"
        snap["error"] = "owner process exited before the job finished"
        snap["message"] = snap["error"]
    return snap


def read_persisted_jobs(directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every job snapshot on disk, newest first (dead owners reported failed)."""

    root = Path(directory) if directory else default_jobs_dir()
    out: List[Dict[str, Any]] = []
    try:
        files = list(root.glob("*.json"))
    except Exception:
        return out
    for path in files:
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(snap, dict) and snap.get("job_id"):
            out.append(_normalize_persisted(snap))
    out.sort(key=lambda s: str(s.get("started_at") or ""), reverse=True)
    return out


def read_persisted_job(job_id: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    root = Path(directory) if directory else default_jobs_dir()
    safe = str(job_id or "").strip()
    if not safe or "/" in safe or "\\" in safe or safe.startswith("."):
        return None
    try:
        snap = json.loads((root / f"{safe}.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return _normalize_persisted(snap) if isinstance(snap, dict) else None


def request_cancel(job_id: str, directory: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Ask the owning process to cancel a persisted job (marker file)."""

    root = Path(directory) if directory else default_jobs_dir()
    snap = read_persisted_job(job_id, root)
    if snap is None:
        return None
    if snap.get("status") in _ACTIVE:
        try:
            (root / f"{snap['job_id']}.cancel").write_text(str(time.time()), encoding="utf-8")
        except Exception:
            pass
        snap = dict(snap)
        snap["message"] = "cancel requested"
    return snap


def _prune_persisted(directory: Path, keep: int) -> None:
    try:
        snaps = read_persisted_jobs(directory)
    except Exception:
        return
    finished = [s for s in snaps if s.get("status") not in _ACTIVE]
    for snap in finished[keep:]:
        for suffix in (".json", ".events.jsonl"):
            try:
                (directory / f"{snap['job_id']}{suffix}").unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# The process-wide registry and the two model verbs
# ---------------------------------------------------------------------------


_default_lock = threading.Lock()
_default: Optional[HostJobRegistry] = None


def default_registry() -> HostJobRegistry:
    """The process-wide registry; persists snapshots under `default_jobs_dir()`
    unless `ABSTRACTCORE_JOBS_PERSIST=0`."""

    global _default
    with _default_lock:
        if _default is None:
            persist = str(os.getenv("ABSTRACTCORE_JOBS_PERSIST") or "1").strip().lower() not in {"0", "false", "no", "off"}
            _default = HostJobRegistry(persist_dir=default_jobs_dir() if persist else None)
        return _default


def set_default_registry(registry: Optional[HostJobRegistry]) -> None:
    """Swap the process-wide registry (tests, embedding hosts)."""

    global _default
    with _default_lock:
        _default = registry


def cli_equivalent_download(provider: str, artifact: str, dry_run: bool = False) -> str:
    return f"abstractcore models download {provider} {artifact}" + (" --dry-run" if dry_run else "")


def cli_equivalent_delete(provider: str, artifact: str, dry_run: bool = False, force: bool = False) -> str:
    return (
        f"abstractcore models delete {provider} {artifact} --yes"
        + (" --force" if force else "")
        + (" --dry-run" if dry_run else "")
    )


def cli_equivalent_engine_install(engine: str, dry_run: bool = False) -> str:
    return f"abstractcore engines install {engine} --yes" + (" --dry-run" if dry_run else "")


def start_download_job(
    provider: str,
    artifact: str,
    *,
    dry_run: bool = False,
    base_url: Optional[str] = None,
    expected_bytes: Optional[int] = None,
    registry: Optional[HostJobRegistry] = None,
    run_inline: bool = False,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `model_materializer.download` as a job (single-flight per provider/artifact)."""

    from . import model_materializer as mm

    pid = mm._provider_id(provider)
    ref = str(artifact or "").strip()
    if not pid or not ref:
        raise ValueError("a provider and an artifact are required")
    reg = registry or default_registry()

    def runner(ctx: JobContext) -> Dict[str, Any]:
        outcome = mm.download(
            pid,
            ref,
            progress_cb=ctx.progress,
            base_url=base_url,
            dry_run=dry_run,
            expected_bytes=expected_bytes,
        )
        data = outcome.to_dict()
        if outcome.command:
            ctx.set_command(list(outcome.command))
        return data

    return reg.start(
        kind="download",
        key=f"download:{pid}/{ref}",
        runner=runner,
        provider=pid,
        artifact=ref,
        command=mm._planned_command(pid, ref),
        dry_run=dry_run,
        cli_equivalent=cli_equivalent_download(pid, ref, dry_run),
        run_inline=run_inline,
        job_id=job_id,
    )


def start_delete_job(
    provider: str,
    artifact: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    base_url: Optional[str] = None,
    registry: Optional[HostJobRegistry] = None,
    run_inline: bool = False,
    with_companions: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run `model_materializer.delete_artifact` as a job.

    `with_companions=True` also removes the model's MTP companion(s) no other
    installed model uses; otherwise the result carries `companion_offer`.

    Callers that must REFUSE on blockers (loaded model, shared cache) check
    `model_materializer.delete_blockers()` first; the job itself also refuses
    and finishes `failed` with `result.status == "refused"`.
    """

    from . import model_materializer as mm

    pid = mm._provider_id(provider)
    ref = str(artifact or "").strip()
    if not pid or not ref:
        raise ValueError("a provider and an artifact are required")
    reg = registry or default_registry()

    def runner(ctx: JobContext) -> Dict[str, Any]:
        ctx.log(f"deleting {pid} {ref}" + (" (dry run)" if dry_run else ""))
        result = mm.delete_artifact(pid, ref, dry_run=dry_run, force=force, base_url=base_url, with_companions=with_companions)
        if result.get("command"):
            ctx.set_command([str(c) for c in result["command"]])
        return result

    return reg.start(
        kind="delete",
        key=f"delete:{pid}/{ref}",
        runner=runner,
        provider=pid,
        artifact=ref,
        dry_run=dry_run,
        cli_equivalent=cli_equivalent_delete(pid, ref, dry_run, force),
        join=True,
        run_inline=run_inline,
    )


# ---------------------------------------------------------------------------
# Detached execution: `python -m abstractcore.config.host_jobs run <spec.json>`
# ---------------------------------------------------------------------------


def spawn_detached(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Start a job in a DETACHED child process and return its queued snapshot.

    The terminal console's CLI transport needs a job id it can poll after the
    `abstractcore` process that started it has exited; the child owns the job
    and persists its snapshot under `default_jobs_dir()`.
    """

    import sys
    import tempfile

    kind = str(spec.get("kind") or "")
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    job_id = f"{_KIND_PREFIX.get(kind, 'job')}_{uuid.uuid4().hex[:12]}"
    spec = dict(spec, job_id=job_id)
    jobs_dir = default_jobs_dir()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    fd, spec_path = tempfile.mkstemp(prefix=f"{job_id}.", suffix=".spec.json", dir=str(jobs_dir))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh)
    log_path = jobs_dir / f"{job_id}.log"
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":  # pragma: no cover - windows only
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    # The child snapshots the HF offline flags at ITS import as "what the
    # operator set" (`config.manager`); hand it the operator's values, not an
    # offline flag written in THIS process after start, or its explicit
    # download would refuse in the operator's name.
    from .manager import explicit_download_hf_env

    kwargs["env"] = explicit_download_hf_env()[0]
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(  # noqa: S603 - argv, fixed module entry point
            [sys.executable, "-m", "abstractcore.config.host_jobs", "run", spec_path],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
    queued = {
        "schema": HOST_JOB_SCHEMA,
        "job_id": job_id,
        "job": job_id,
        "kind": kind,
        "status": "queued",
        "provider": spec.get("provider"),
        "artifact": spec.get("artifact"),
        "engine": spec.get("engine"),
        "percent": None,
        "downloaded_bytes": None,
        "total_bytes": None,
        "message": "queued (detached)",
        "log_tail": [],
        "events": [],
        "command": [],
        "dry_run": bool(spec.get("dry_run")),
        "started_at": _now_iso(time.time()),
        "finished_at": None,
        "error": None,
        "joined": 0,
        "cli_equivalent": "",
        "result": None,
        "pid": proc.pid,
        "elapsed_s": 0.0,
        # The progress contract, so a poller that races the child's start
        # reads the same shape it will read a moment later.
        "state": "queued",
        "bytes_done": None,
        "bytes_total": None,
        "size_unknown": False,
        "size_note": None,
        "bytes_per_second": None,
        "eta_s": None,
        "updated_at": _now_iso_ms(time.time()),
        "detail": "",
        "files": [],
        "current_file": None,
        "stall_after_s": default_stall_after_s(),
        "stalled_for_s": None,
        "transitions": [{"at": _now_iso_ms(time.time()), "state": "queued", "why": "queued (detached)"}],
        "progress_events_count": 0,
        "cancel_requested": False,
    }
    # Write the queued snapshot now so a poll that races the child's start
    # finds the job instead of a 404.
    target = jobs_dir / f"{job_id}.json"
    if not target.exists():
        try:
            target.write_text(json.dumps(queued, indent=1, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
    return queued


def _run_spec(spec_path: str) -> int:
    try:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    finally:
        try:
            Path(spec_path).unlink()
        except Exception:
            pass
    kind = spec.get("kind")
    registry = default_registry()
    if kind == "download":
        snap = start_download_job(
            spec["provider"],
            spec["artifact"],
            dry_run=bool(spec.get("dry_run")),
            registry=registry,
            run_inline=True,
            job_id=spec.get("job_id"),
        )
    elif kind == "engine_install":
        from .engines import engine_install

        snap = engine_install(
            spec["engine"],
            dry_run=bool(spec.get("dry_run")),
            force=bool(spec.get("force")),
            registry=registry,
            run_inline=True,
            job_id=spec.get("job_id"),
        )
    else:
        return 2
    return 0 if snap.get("status") == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - exercised through spawn_detached
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "run":
        sys.exit(_run_spec(sys.argv[2]))
    print("usage: python -m abstractcore.config.host_jobs run <spec.json>", file=sys.stderr)
    sys.exit(2)
