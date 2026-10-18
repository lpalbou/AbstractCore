"""Console output that never crashes on a character the terminal's code page lacks.

Windows sends a redirected or piped stream (``> file``, ``| Select-String``, CI logs) through
the ANSI code page, often cp1252, which has no emoji, arrows or box-drawing characters; a
print of one raised ``UnicodeEncodeError`` and ended the command. The command-line entry
points call `safe_stdio()` first: an encoding error then writes a replacement character.
Interactive consoles (UTF-16 through the console API) and UTF-8 streams are left as they are.
"""

from __future__ import annotations

import sys


def safe_stdio() -> None:
    """Make stdout/stderr replace characters their encoding cannot represent."""

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = str(getattr(stream, "encoding", "") or "").lower().replace("_", "-")
        if encoding in {"utf-8", "utf8", "utf-8-sig"}:
            continue
        try:
            reconfigure(errors="replace")
        except Exception:  # a stream that cannot be reconfigured keeps its behavior
            pass
