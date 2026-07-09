"""Persistent shell sessions (backlog 0215).

The default `execute_command` tool is a one-shot `subprocess.run(...)`: `cd`, `export`, virtualenv
activation, and interactive REPLs do not persist across calls. This module adds a persistent,
long-lived shell whose working directory and environment survive between commands, plus a
`write_stdin` primitive for feeding input to an interactive process — the core of a Codex-style
`unified_exec` surface.

Transport: a PTY (pseudo-terminal), NOT a plain pipe. An earlier pipe-based design delivered the
exit-code sentinel over the same stdin as the command, so a stdin-consuming command (`cat`, a bare
REPL, `read`), `set -x`, or an unterminated quote could swallow or desync the sentinel; binary
output killed the reader thread. The PTY design fixes all of these:

- The command and its sentinel are written on ONE line (`<cmd>; printf … "$?"`). The shell reads and
  parses the whole line before executing, so a stdin-consuming command inside `<cmd>` cannot eat the
  sentinel (verified: `cat`, `set -x`, binary blobs all keep the boundary intact).
- Terminal ECHO is disabled, so the command bytes we write are not echoed back into the read stream
  (the sentinel appears only via `printf`).
- Output post-processing (`\r\n`) is normalized to `\n`; undecodable bytes are `errors="replace"`d
  so binary output never kills the reader.
- A STARTUP DRAIN consumes the shell's banner/first prompt so it is never attributed to the first
  command.

Scope / non-goals: POSIX shells only (bash preferred; Windows out of scope); this is NOT an OS
sandbox (that is a separate tier, backlog 0062) — a persistent session is at least as powerful as
the one-shot tool and must be gated by the same approval/workspace policy by its callers. Output is
bounded for the model with an explicit `#[WARNING:TRUNCATION]` marker (ADR-0026); the session keeps
running regardless. A command with no output and no return (a bare `cat` with no input, an infinite
loop) still hits the per-call timeout, which marks the session `_degraded`; the next `run()` then
resyncs deterministically (Ctrl-C + a discard-until-sync sentinel) so stale output does not bleed
into it, though a SIGINT-ignoring foreground process still warrants recycling the session.

The engine is hardened and adversarially verified (bounded O(n) sentinel scan + capped memory on
huge output, deterministic post-timeout resync, fd-leak-free reopen, process-group teardown). It is
still a host-owned PRIMITIVE — exposing it as an agent-callable tool is a separate, approval-gated,
run-scoped step (see backlog 0215) because a persistent shell, once approved, escapes per-call
workspace-cwd confinement exactly like `execute_command`.
"""

from __future__ import annotations

import atexit
import os
import queue
import shutil
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

_DEFAULT_TIMEOUT_S = 120.0
_MAX_OUTPUT_CHARS = 20000
_STARTUP_DRAIN_TIMEOUT_S = 5.0
# PTY read size: 64 KiB keeps big-output commands to few syscalls/queue hops (small reads made a
# multi-MB command hundreds of round-trips). The reader is event-driven — run() wakes the instant a
# chunk lands — so this only affects large-output throughput, not short-command latency.
_READ_CHUNK_BYTES = 65536

# Sentinel object marking end-of-stream on the reader queue.
_EOF = object()


def _pick_shell() -> Optional[str]:
    for candidate in ("bash", "sh"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _shell_argv(shell_path: str) -> list[str]:
    # Non-login, non-rc shell to avoid profile banners / user rc noise. `sh` ignores these flags
    # it doesn't know only if they exist; dash accepts neither, so only add them for bash.
    if os.path.basename(shell_path).startswith("bash"):
        return [shell_path, "--norc", "--noprofile"]
    return [shell_path]


@dataclass
class ShellCommandResult:
    stdout: str
    exit_code: Optional[int]
    timed_out: bool = False
    truncated: bool = False


@dataclass
class ShellSession:
    """A persistent PTY-backed shell with durable cwd/env across commands."""

    session_id: str
    shell_path: str
    cwd: Optional[str] = None
    _pid: Optional[int] = field(default=None, repr=False)
    _master_fd: Optional[int] = field(default=None, repr=False)
    _reader: Optional[threading.Thread] = field(default=None, repr=False)
    _chunks: "queue.Queue" = field(default_factory=queue.Queue, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closed: bool = False
    _degraded: bool = False  # set after a timeout; next run() drains stale buffered output first

    def start(self, *, env: Optional[Dict[str, str]] = None) -> None:
        import pty
        import termios

        run_env = dict(os.environ)
        # No interactive prompt strings in the output stream; venv activate scripts rewrite PS1
        # unless VIRTUAL_ENV_DISABLE_PROMPT is set, which would re-introduce prompt noise.
        run_env.update({"PS1": "", "PS2": "", "VIRTUAL_ENV_DISABLE_PROMPT": "1"})
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})

        pid, master_fd = pty.fork()
        if pid == 0:  # child
            try:
                if self.cwd:
                    os.chdir(self.cwd)
            except Exception:
                pass
            argv = _shell_argv(self.shell_path)
            try:
                os.execvpe(argv[0], argv, run_env)
            except Exception:
                os._exit(127)
            return  # unreachable

        # Parent.
        self._pid = pid
        self._master_fd = master_fd

        # Disable ECHO (so our written command bytes aren't echoed back) and ONLCR (so newlines
        # are not translated to \r\n in the output stream).
        try:
            attrs = termios.tcgetattr(master_fd)
            attrs[3] = attrs[3] & ~termios.ECHO  # lflag
            attrs[1] = attrs[1] & ~termios.ONLCR  # oflag
            termios.tcsetattr(master_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        def _drain() -> None:
            while True:
                try:
                    data = os.read(master_fd, _READ_CHUNK_BYTES)
                except OSError:
                    break  # EIO on child exit
                if not data:
                    break
                self._chunks.put(data.decode("utf-8", errors="replace"))
            self._chunks.put(_EOF)

        self._reader = threading.Thread(target=_drain, daemon=True)
        self._reader.start()

        # Startup drain: run a sync sentinel and discard everything up to it (the shell banner /
        # first prompt), so it is never attributed to the first real command.
        self._run_raw("", timeout_s=_STARTUP_DRAIN_TIMEOUT_S)
        # Disable job control (`set +m`). A PTY makes bash interactive, which turns job control ON
        # and puts every `&` background job in its OWN process group — so a single killpg(shell pg)
        # on close would ORPHAN those children. With job control off, background jobs stay in the
        # shell's process group and close()'s killpg reaps the whole tree. Job listing (`jobs`) and
        # `&` still work.
        self._run_raw("set +m", timeout_s=_STARTUP_DRAIN_TIMEOUT_S)

    def is_alive(self) -> bool:
        if self._pid is None or self._closed:
            return False
        try:
            done, _ = os.waitpid(self._pid, os.WNOHANG)
            # waitpid returns (0, 0) while alive; (pid, status) once it exited.
            return done == 0
        except ChildProcessError:
            return False
        except Exception:
            return True

    def _read_until_sentinel(self, sentinel: str, *, deadline: float) -> tuple[str, Optional[int], bool]:
        """Accumulate output until `sentinel:<code>\\n` appears. Returns (output, exit_code, timed_out).

        Bounded in BOTH cpu and memory (adversarial-review hardening):
        - Incremental marker search (a search offset), so a large output is scanned once — O(n),
          not O(n²) which turned a finished big-output command into a FALSE timeout.
        - The accumulated buffer is capped: once output exceeds a head cap we stop growing it and
          keep only a small rolling tail for marker detection, so multi-hundred-MB output cannot
          blow up the reader's heap. `_run_raw` applies the final `_MAX_OUTPUT_CHARS` preview cap.
        """
        marker = sentinel + ":"
        head_cap = _MAX_OUTPUT_CHARS + 4096
        window = max(4096, 4 * len(marker))
        head = ""            # bounded output capture (<= head_cap)
        head_search_from = 0  # incremental find offset into `head`
        tail = ""            # rolling last `window` chars once we stop growing head
        truncated = False

        def _finish(out: str, after: str) -> tuple[str, Optional[int], bool]:
            code_str = after.split("\n", 1)[0].strip()
            try:
                code: Optional[int] = int(code_str)
            except ValueError:
                code = None
            # The sentinel printf prepends a newline; drop the single trailing newline it added.
            if out.endswith("\n"):
                out = out[:-1]
            return self._normalize(out), code, False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._degraded = True
                return self._normalize(head), None, True
            try:
                chunk = self._chunks.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if chunk is _EOF:
                return self._normalize(head), None, False

            if not truncated:
                head += chunk
                idx = head.find(marker, max(0, head_search_from - (len(marker) - 1)))
                if idx != -1:
                    after = head[idx + len(marker):]
                    if "\n" not in after:
                        # Exit-code newline not fully arrived; wait, but don't rescan from 0.
                        head_search_from = idx
                        continue
                    return _finish(head[:idx], after)
                head_search_from = len(head)
                if len(head) > head_cap:
                    # Switch to bounded mode: freeze the captured head, keep only a rolling tail.
                    truncated = True
                    tail = head[-window:]
            else:
                tail = (tail + chunk)[-window:]
                idx = tail.find(marker)
                if idx != -1:
                    after = tail[idx + len(marker):]
                    if "\n" not in after:
                        continue
                    # Middle was dropped; return the captured head (‹_run_raw› adds the truncation
                    # note since head_cap > _MAX_OUTPUT_CHARS) and the exit code from the tail.
                    code_str = after.split("\n", 1)[0].strip()
                    try:
                        code = int(code_str)
                    except ValueError:
                        code = None
                    return self._normalize(head), code, False

    @staticmethod
    def _normalize(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _write(self, data: str) -> None:
        assert self._master_fd is not None
        os.write(self._master_fd, data.encode("utf-8", errors="replace"))

    def _run_raw(self, command: str, *, timeout_s: float) -> ShellCommandResult:
        sentinel = f"__ABSTRACT_SENTINEL_{uuid.uuid4().hex}__"
        printf = f"printf '\\n%s:%s\\n' '{sentinel}' \"$?\"\n"
        # NEWLINE sentinel: the printf runs as its own statement on the line AFTER the command, so
        # multi-line commands, heredocs (`cat <<EOF … EOF`), pipelines, and quoted blocks stay
        # syntactically intact (a same-line `; printf` would collide with a heredoc terminator or an
        # unterminated quote). A genuinely INTERACTIVE command that reads the tty with no supplied
        # input (a bare `cat`, a REPL) will consume the printf line and the call will hit its
        # timeout — the honest behavior for a still-running interactive process; the caller feeds it
        # via write_stdin or recycles the session after the timeout.
        line = printf if not command else f"{command}\n{printf}"
        self._write(line)
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        out, code, timed_out = self._read_until_sentinel(sentinel, deadline=deadline)
        truncated = False
        if len(out) > _MAX_OUTPUT_CHARS:
            #[WARNING:TRUNCATION] bounded session-command preview for the model
            out = out[:_MAX_OUTPUT_CHARS].rstrip() + "\n… (truncated; full output exceeded the preview bound)"
            truncated = True
        return ShellCommandResult(stdout=out, exit_code=code, timed_out=timed_out, truncated=truncated)

    def run(self, command: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> ShellCommandResult:
        """Run one command in the persistent session; returns its output + exit code.

        Serialized per session (one command at a time). Working directory and environment set by a
        command persist into the next.
        """
        if not self.is_alive():
            raise RuntimeError(f"shell session '{self.session_id}' is not running")
        cmd = str(command or "")
        with self._lock:
            # If a PRIOR command timed out, a foreground process may still be running and consuming
            # the tty (e.g. a bare `cat`/REPL). Resync DETERMINISTICALLY: send Ctrl-C, then a fresh
            # SYNC sentinel, and discard everything up to and including it. Late stale output from
            # the interrupted command arrives BEFORE the sync sentinel and is dropped — unlike a
            # fixed-sleep drain, this closes the race where output emitted just after the sleep
            # window bled into the next command. (If the foreground process IGNORES SIGINT, the sync
            # sentinel can't run and we fall back to draining what's buffered; recycle the session.)
            if self._degraded:
                self._resync()
                self._degraded = False
            return self._run_raw(cmd, timeout_s=timeout_s)

    def _resync(self, *, timeout_s: float = 5.0) -> None:
        try:
            self._write("\x03")  # ETX -> SIGINT to the foreground process group
        except Exception:
            pass
        sync = f"__ABSTRACT_SYNC_{uuid.uuid4().hex}__"
        try:
            self._write(f"printf '\\n%s\\n' '{sync}'\n")
        except Exception:
            return
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        seen = ""
        while time.monotonic() < deadline:
            try:
                chunk = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is _EOF:
                return
            seen += chunk
            if sync in seen:
                return
        # Sync sentinel never arrived (SIGINT-ignoring foreground process). Best-effort drain.
        while True:
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                break

    def write_stdin(self, data: str) -> None:
        """Feed raw input to the session (for interactive prompts / REPLs)."""
        if not self.is_alive():
            raise RuntimeError(f"shell session '{self.session_id}' is not running")
        with self._lock:
            self._write(data if data.endswith("\n") else data + "\n")

    def read_output(self, *, timeout_s: float = 2.0, idle_s: float = 0.5, max_chars: int = _MAX_OUTPUT_CHARS) -> ShellCommandResult:
        """Drain whatever output the session produces within a bounded window.

        The companion to `write_stdin` for interactive processes (REPLs, prompts): there is no
        sentinel to wait for, so we read until the stream goes quiet for `idle_s` (or `timeout_s`
        elapses). Never resyncs and never touches `_degraded` — an interactive foreground process
        is a legitimate state here, not a failure. Exit code is unknowable in this mode (None).
        """
        if not self.is_alive():
            raise RuntimeError(f"shell session '{self.session_id}' is not running")
        with self._lock:
            deadline = time.monotonic() + max(0.1, float(timeout_s))
            idle = max(0.05, float(idle_s))
            out = ""
            got_any = False
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                # After the first data arrives, a quiet gap of `idle_s` ends the read early.
                wait_s = min(deadline - now, idle if got_any else (deadline - now))
                try:
                    chunk = self._chunks.get(timeout=max(0.05, wait_s))
                except queue.Empty:
                    if got_any:
                        break
                    continue
                if chunk is _EOF:
                    break
                out += chunk
                got_any = True
                if len(out) > max_chars + 4096:
                    break
            out = self._normalize(out)
            truncated = False
            if len(out) > max_chars:
                #[WARNING:TRUNCATION] bounded interactive-read preview for the model
                out = out[:max_chars].rstrip() + "\n… (truncated; full output exceeded the preview bound)"
                truncated = True
            return ShellCommandResult(stdout=out, exit_code=None, timed_out=False, truncated=truncated)

    def close(self) -> None:
        self._closed = True
        pid = self._pid
        fd = self._master_fd
        # Kill the whole process group (the PTY session leader), reaping foreground + background
        # children so nothing is orphaned; then reap the shell to avoid a zombie.
        if pid is not None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
            try:
                os.waitpid(pid, 0)
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        self._pid = None
        self._master_fd = None


def namespaced_session_id(namespace: str, session_id: str) -> str:
    """Build a registry key scoped to a namespace (e.g. a run id).

    Hosts inject the namespace at their trust boundary so callers cannot reach another
    scope's sessions by guessing ids. An empty namespace means process-global.
    """
    ns = str(namespace or "").strip()
    sid = str(session_id or "").strip() or "main"
    return f"{ns}::{sid}" if ns else sid


class ShellSessionRegistry:
    """Process-local registry of persistent shell sessions keyed by id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ShellSession] = {}
        self._lock = threading.Lock()

    def open(self, *, session_id: Optional[str] = None, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> ShellSession:
        shell_path = _pick_shell()
        if not shell_path:
            raise RuntimeError("No POSIX shell (bash/sh) found for a persistent session")
        sid = str(session_id or f"sh_{uuid.uuid4().hex[:12]}")
        with self._lock:
            existing = self._sessions.get(sid)
            if existing is not None:
                if existing.is_alive():
                    return existing
                # Dead session being replaced: close it first so its PTY master fd is released
                # (reopen-after-abnormal-death would otherwise leak the fd).
                try:
                    existing.close()
                except Exception:
                    pass
            session = ShellSession(session_id=sid, shell_path=shell_path, cwd=cwd)
            session.start(env=env)
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Optional[ShellSession]:
        with self._lock:
            return self._sessions.get(str(session_id))

    def close(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(str(session_id), None)
        if session is None:
            return False
        session.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            try:
                s.close()
            except Exception:
                pass

    def close_namespace(self, namespace: str) -> int:
        """Close every session whose key belongs to `namespace` (see namespaced_session_id).

        Hosts call this at their scope's end-of-life boundary (e.g. a run reaching a
        terminal state) so approved sessions never outlive the work they were approved for.
        Returns the number of sessions closed.
        """
        ns = str(namespace or "").strip()
        if not ns:
            return 0
        prefix = f"{ns}::"
        with self._lock:
            keys = [k for k in self._sessions if k.startswith(prefix)]
            sessions = [self._sessions.pop(k) for k in keys]
        for s in sessions:
            try:
                s.close()
            except Exception:
                pass
        return len(sessions)


# Process-wide default registry (host-owned; never persisted in RunState).
_DEFAULT_REGISTRY = ShellSessionRegistry()

# Hygiene: reap any still-open sessions (and their process groups) at interpreter exit so a
# crashed/interrupted host never strands shell children.
atexit.register(_DEFAULT_REGISTRY.close_all)


def get_shell_session_registry() -> ShellSessionRegistry:
    return _DEFAULT_REGISTRY


__all__ = [
    "ShellSession",
    "ShellSessionRegistry",
    "ShellCommandResult",
    "get_shell_session_registry",
    "namespaced_session_id",
]
