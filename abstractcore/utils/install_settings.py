"""The three AbstractCore install settings, as the commands install hints show.

- light: ``pip install abstractcore`` -- every remote provider, tools, media inputs, the server
  and the capability plugins. A missing light dependency means an old or broken install, so the
  hint is an upgrade.
- apple: ``pip install "abstractcore[apple]"`` -- light + every local engine for Apple silicon.
- gpu: ``pip install "abstractcore[gpu]"`` -- light + every local engine for NVIDIA / AMD Linux.

See docs/installation.md. No other setting exists; never print another extra name.
"""

from __future__ import annotations

import platform
import sys
from typing import Optional

LIGHT_INSTALL = "pip install -U abstractcore"
APPLE_INSTALL = 'pip install "abstractcore[apple]"'
GPU_INSTALL = 'pip install "abstractcore[gpu]"'


def local_engines_setting() -> Optional[str]:
    """The setting that carries local engines on this machine: ``apple``, ``gpu`` or None.

    Apple silicon macOS -> ``apple``; Linux -> ``gpu``; other hosts (Intel Macs, Windows) have
    no local-engine setting and run local models through Ollama or LM Studio on light.
    """
    if sys.platform == "darwin":
        return "apple" if platform.machine().lower() in {"arm64", "aarch64"} else None
    if sys.platform.startswith("linux"):
        return "gpu"
    return None


def local_engines_install_command() -> str:
    """The install command for this machine's local engines, or both when the host has none."""
    setting = local_engines_setting()
    if setting == "apple":
        return APPLE_INSTALL
    if setting == "gpu":
        return GPU_INSTALL
    return f"{APPLE_INSTALL} (Apple silicon) or {GPU_INSTALL} (Linux with an NVIDIA or AMD GPU)"
