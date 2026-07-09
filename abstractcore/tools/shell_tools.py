"""Agent-callable persistent shell tools (backlog 0220).

These expose the persistent PTY shell engine (`shell_session.py`) as tools a model can call:
`shell_exec` runs commands in a session whose working directory and environment persist across
calls; `shell_write_stdin` feeds input to an interactive foreground process (REPL, prompt);
`shell_close` ends a session.

Safety posture (deliberate, per backlog 0220):
- OPT-IN: not part of the default toolsets. Hosts enable them explicitly
  (`ABSTRACT_ENABLE_SHELL_TOOLS=1` for the default-toolset path, or by wiring `SHELL_TOOLS`
  into their executor).
- APPROVAL-GATED by default at the runtime layer (listed in `_DEFAULT_REQUIRE_APPROVAL`
  alongside `execute_command`).
- HONESTLY LABELED: the schema states that a session is NOT an OS sandbox and is NOT durable —
  see the tool descriptions. Sessions escape per-call working-directory confinement once
  approved, exactly like `execute_command`.
- SCOPED: hosts inject `_registry_namespace` (hidden from the model schema) at their trust
  boundary so one run's sessions are unreachable from another run. The AbstractRuntime
  TOOL_CALLS handler overwrites this argument with the run id and closes the namespace when
  the run reaches a terminal state.

Non-durability contract: sessions live in THIS process only. After a host restart/replay, a
session id resolves to a fresh shell; every fresh open is explicitly announced in the tool
output ("new session started") so the model never assumes state survived.
"""

from __future__ import annotations

from typing import Optional

from .core import tool
from .shell_session import (
    ShellCommandResult,
    get_shell_session_registry,
    namespaced_session_id,
)

_DEFAULT_SESSION_ID = "main"

# Per-call ceiling. Models routinely confuse units (600000 "ms" arrives as seconds) and a
# foreground command that needs longer than this should be backgrounded (`&`) instead of
# holding the act/observe round open. The session itself keeps running after a timeout.
_MAX_TIMEOUT_S = 600.0


def _coerce_float(value, default: float, *, cap: Optional[float] = None) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    if f <= 0:
        return default
    return min(f, cap) if cap else f


def _new_session_notice(session_id: str) -> str:
    return (
        f"[new shell session '{session_id}' started — no state carries over from before "
        "(a session never survives a host restart; cwd/env/venv must be re-established)]"
    )


def _render_result(result: ShellCommandResult, *, prefix_lines: Optional[list] = None) -> str:
    parts: list[str] = list(prefix_lines or [])
    if result.timed_out:
        parts.append(
            "[timed out — the foreground process may still be running. Feed it input with "
            "shell_write_stdin, retry later, or shell_close the session to kill it.]"
        )
    if result.exit_code is not None:
        parts.append(f"exit_code: {result.exit_code}")
    out = result.stdout if result.stdout else "(no output)"
    parts.append(out)
    return "\n".join(parts)


@tool(
    description=(
        "Run a command in a PERSISTENT shell session (cwd/env/venv persist across calls). "
        "NOT a sandbox (execute_command-level trust); NOT durable (never survives a host restart)."
    ),
    when_to_use=(
        "Multi-step shell workflows where state must persist: activate a venv then run tools in it, "
        "cd and work there, start a background process (`&`) and inspect it later. A 'new shell "
        "session' notice in output means prior state is gone."
    ),
    hide_args=["_registry_namespace"],
    examples=[
        {"description": "Create and use a venv across calls", "arguments": {"command": "python3 -m venv .venv && source .venv/bin/activate && pip install requests"}},
        {"description": "Next call, same session: venv still active", "arguments": {"command": "python -c 'import requests; print(requests.__version__)'"}},
        {"description": "Parallel session with its own state", "arguments": {"command": "cd /tmp && ls", "session_id": "scratch"}},
    ],
)
def shell_exec(
    command: str,
    session_id: str = _DEFAULT_SESSION_ID,
    working_directory: Optional[str] = None,
    timeout: float = 120,
    _registry_namespace: str = "",
) -> str:
    """Run `command` in the persistent session `session_id`, creating it if needed.

    `working_directory` sets the INITIAL cwd when the session is first created (ignored on an
    existing session — use `cd` there). `timeout` bounds this call in seconds; on timeout the
    session survives and resyncs on the next call.
    """
    cmd = str(command or "").strip()
    if not cmd:
        return "Error: command must be a non-empty string"
    sid = str(session_id or "").strip() or _DEFAULT_SESSION_ID
    key = namespaced_session_id(_registry_namespace, sid)
    registry = get_shell_session_registry()

    existing = registry.get(key)
    created = existing is None or not existing.is_alive()
    try:
        session = registry.open(
            session_id=key,
            cwd=str(working_directory).strip() if created and working_directory else None,
        )
        result = session.run(cmd, timeout_s=_coerce_float(timeout, 120.0, cap=_MAX_TIMEOUT_S))
    except Exception as e:
        return f"Error: shell session failed: {e}"

    prefix = [_new_session_notice(sid)] if created else None
    return _render_result(result, prefix_lines=prefix)


@tool(
    description=(
        "Send a line of input to the interactive foreground process of a persistent shell session "
        "(REPL, prompt), then return the output produced within a short window."
    ),
    when_to_use=(
        "Only when a prior shell_exec started an interactive process now waiting for input (a REPL, "
        "a confirmation prompt). Not for running normal commands — use shell_exec for those."
    ),
    hide_args=["_registry_namespace"],
    examples=[
        {"description": "Answer a confirmation prompt", "arguments": {"input": "y"}},
        {"description": "Evaluate in a REPL started earlier", "arguments": {"input": "print(6*7)", "read_timeout": 3}},
    ],
)
def shell_write_stdin(
    input: str,
    session_id: str = _DEFAULT_SESSION_ID,
    read_timeout: float = 5,
    _registry_namespace: str = "",
) -> str:
    """Write `input` (newline-terminated) to the session's foreground process and read output."""
    sid = str(session_id or "").strip() or _DEFAULT_SESSION_ID
    key = namespaced_session_id(_registry_namespace, sid)
    session = get_shell_session_registry().get(key)
    if session is None or not session.is_alive():
        return (
            f"Error: no active shell session '{sid}'. Start one with shell_exec first "
            "(sessions do not survive host restarts)."
        )
    try:
        session.write_stdin(str(input if input is not None else ""))
        result = session.read_output(timeout_s=_coerce_float(read_timeout, 5.0, cap=_MAX_TIMEOUT_S))
    except Exception as e:
        return f"Error: shell session failed: {e}"
    return _render_result(result)


@tool(
    description=(
        "Close a persistent shell session, killing its process group (foreground and background "
        "children). Sessions are also closed automatically when the run ends."
    ),
    when_to_use=(
        "When done with a session whose processes should stop now (e.g. a dev server you "
        "started), or to recycle a session stuck on an interactive process."
    ),
    hide_args=["_registry_namespace"],
    examples=[{"description": "Close the default session", "arguments": {}}],
)
def shell_close(
    session_id: str = _DEFAULT_SESSION_ID,
    _registry_namespace: str = "",
) -> str:
    """Close session `session_id` and reap its process group."""
    sid = str(session_id or "").strip() or _DEFAULT_SESSION_ID
    key = namespaced_session_id(_registry_namespace, sid)
    closed = get_shell_session_registry().close(key)
    if closed:
        return f"Shell session '{sid}' closed (process group terminated)."
    return f"No active shell session '{sid}' (nothing to close)."


SHELL_TOOLS = [shell_exec, shell_write_stdin, shell_close]

__all__ = ["shell_exec", "shell_write_stdin", "shell_close", "SHELL_TOOLS"]
