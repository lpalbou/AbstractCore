"""Send rate limits per account: rolling hour and rolling day, durable across restarts.

The counter is a small JSON file of send timestamps (the last 24 hours only) next to the
account store, updated under an exclusive file lock so concurrent processes share one count.

    limiter = SendRateLimiter(path, SendLimits(per_hour=100, per_day=1000))
    token = limiter.reserve()          # raises EmailRateLimited when a window is full
    try: send(...)
    except NothingSent: limiter.refund(token)

A reservation counts one message whatever its number of recipients.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List

from .errors import EmailRateLimited
from .models import SendLimits

HOUR_S = 3600.0
DAY_S = 86400.0


try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None


@contextmanager
def _locked(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
            try:
                yield
            finally:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


class SendRateLimiter:
    def __init__(self, path: Path, limits: SendLimits, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.limits = limits
        self._clock = clock

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _read(self) -> List[float]:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        sends = doc.get("sends") if isinstance(doc, dict) else None
        out: List[float] = []
        for v in sends or []:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out

    def _write(self, sends: List[float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}-{secrets.token_hex(4)}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"v": 1, "sends": sends}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def _counts(self, sends: List[float], now: float) -> Dict[str, int]:
        return {
            "last_hour": sum(1 for t in sends if now - t < HOUR_S),
            "last_day": sum(1 for t in sends if now - t < DAY_S),
        }

    def usage(self) -> Dict[str, int]:
        now = self._clock()
        with _locked(self._lock_path):
            sends = [t for t in self._read() if now - t < DAY_S]
        c = self._counts(sends, now)
        return {
            "used_last_hour": c["last_hour"],
            "used_last_day": c["last_day"],
            "per_hour": self.limits.per_hour,
            "per_day": self.limits.per_day,
        }

    def reserve(self) -> float:
        now = self._clock()
        with _locked(self._lock_path):
            sends = [t for t in self._read() if now - t < DAY_S]
            c = self._counts(sends, now)
            for window, used, limit, span in (
                ("hour", c["last_hour"], self.limits.per_hour, HOUR_S),
                ("day", c["last_day"], self.limits.per_day, DAY_S),
            ):
                if used >= limit:
                    in_window = sorted(t for t in sends if now - t < span)
                    retry_after = 0.0
                    if limit > 0 and in_window:
                        # The slot frees when the (used - limit + 1)-th oldest send leaves the window.
                        retry_after = max(0.0, in_window[used - limit] + span - now)
                    raise EmailRateLimited(
                        f"The send limit of {limit} message(s) per {window} is reached ({used} sent in the last {window}); nothing was sent.",
                        "Wait for the window to pass, or raise the limit in the email settings "
                        f"(`abstractcore email limits set --per-{window} <n>`).",
                        details={
                            "window": window,
                            "limit": limit,
                            "used": used,
                            "retry_after_s": round(retry_after, 1) if limit > 0 else None,
                        },
                    )
            sends.append(now)
            self._write(sends)
        return now

    def refund(self, token: float) -> None:
        with _locked(self._lock_path):
            sends = self._read()
            try:
                sends.remove(token)
            except ValueError:
                return
            self._write(sends)
