"""Process-tree kill for subprocess timeouts (shared: browser_probe + execute_command).

The recurring hazard (browser_tools + execute_command, runtime c5004): a child
launched through a shell or a driver spawns grandchildren that ESCAPE the
child's process group — `chrome-headless-shell` calls `setsid()`; a shell's
backgrounded or re-sessioned grandchild does the same — so
`os.killpg(child_group)` misses them. Worse, with CAPTURED pipes the
post-timeout read WAITS FOR PIPE EOF, and an orphaned grandchild still holding
stdout pins the calling thread FOREVER even after the direct child "died" (the
classic subprocess gotcha behind a 40-minute stuck tick).

Killing the whole descendant TREE — enumerated by ANCESTRY before the child is
reaped (afterwards the grandchildren reparent to init and the links are gone) —
reaches the escaped groups AND lets the captured pipes close so the reader
unblocks. Verified by browser_tools' deterministic setsid-escaped-grandchild
test; this module is the one implementation both call sites share.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Dict, List


def descendant_pids(root_pid: int) -> List[int]:
    """All descendant PIDs of root_pid (POSIX, via one `ps` sweep).

    Walks ancestry (parent→child), so it reaches a grandchild even after it
    left the process group via setsid(). MUST be called BEFORE the child is
    killed (afterwards descendants reparent to init and the child links vanish).
    One retry — a transient `ps` failure would otherwise silently reduce the
    kill to the worker group alone (leaking the escaped tree).
    """
    out = ""
    for _attempt in range(2):
        try:
            out = subprocess.run(
                ["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, timeout=5
            ).stdout
            if out:
                break
        except Exception:
            continue
    if not out:
        return []
    children: Dict[int, List[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                pid, ppid = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            children.setdefault(ppid, []).append(pid)
    seen: List[int] = []
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            if c not in seen and c != root_pid:
                seen.append(c)
                stack.append(c)
    return seen


def hard_kill_tree(proc: "subprocess.Popen") -> None:
    """SIGKILL the worker AND every descendant process/group (leak-proof).

    POSIX: enumerate descendants first (while the tree is intact), kill the
    worker's own group, then SIGKILL each descendant's process group and the
    descendant itself — so a `setsid()`'d grandchild's group and its children
    die too. Windows: `proc.kill()` (no group-escape there; best-effort).

    Guarded throughout: a descendant reaped between enumeration and kill (or a
    `getpgid` on a dead pid) must never raise — the point is to leave nothing
    alive, not to account for every pid.
    """
    if os.name != "posix":
        try:
            proc.kill()
        except Exception:
            pass
        return
    descendants = descendant_pids(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # worker + same-group children (shell, driver)
    except Exception:
        pass
    killed_groups = set()
    for dpid in descendants:
        try:
            gid = os.getpgid(dpid)
            if gid not in killed_groups:
                killed_groups.add(gid)
                os.killpg(gid, signal.SIGKILL)  # a setsid()'d grandchild's own group + its children
        except Exception:
            pass
        try:
            os.kill(dpid, signal.SIGKILL)  # belt: the pid directly
        except Exception:
            pass


__all__ = ["descendant_pids", "hard_kill_tree"]
