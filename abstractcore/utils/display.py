"""Can a browser open on the person's screen from this process?

The one Python rule, the same as the console crate's `display_from`
(console-tui/src/screens/mod.rs):

- an SSH session (`SSH_CONNECTION`, `SSH_CLIENT` or `SSH_TTY` set) has no
  display of the person's own: a browser would open on the REMOTE machine's
  screen, or fail -- on every OS;
- on Linux/BSD, a graphical session sets `DISPLAY` (X11) or
  `WAYLAND_DISPLAY`; with neither there is no screen at all (a headless
  server, a container, a text console).

Without a display nothing is launched and the caller prints the URL.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping, Optional

__all__ = ["no_display_reason", "open_url"]

_SSH_VARS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")


def no_display_reason(*, platform: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Why no browser can open on the person's screen here, or None when one can.

    `platform` is a `sys.platform` value (default: this one).
    """

    env = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform

    def is_set(key: str) -> bool:
        return bool(env.get(key))

    if any(is_set(key) for key in _SSH_VARS):
        return "this is an SSH session: a browser would open on the remote machine, not yours"
    if plat not in ("darwin", "win32") and not is_set("DISPLAY") and not is_set("WAYLAND_DISPLAY"):
        return "no graphical session: DISPLAY and WAYLAND_DISPLAY are unset"
    return None


def open_url(url: str) -> bool:
    """Open `url` in a browser when there is a display; True when one was launched."""

    if no_display_reason() is not None:
        return False
    import webbrowser

    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False
