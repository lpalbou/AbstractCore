"""The three AbstractCore install settings, as the commands install hints show.

- light: ``pip install abstractcore`` -- every remote provider, tools, media inputs, the server
  and the capability plugins. A missing light dependency means an old or broken install, so the
  hint is an upgrade.
- apple: ``pip install "abstractcore[apple]"`` -- light + every local engine for Apple silicon.
- gpu: ``pip install "abstractcore[gpu]"`` -- light + every local engine for NVIDIA / AMD Linux.

See docs/installation.md. No other setting exists; never print another extra name, and never
tell a user to install a bare package (transformers, httpx, playwright, ...)
to get an AbstractCore capability (operator ruling 2026-09-29).

A host with no local-engine setting (an Intel Mac, Windows): ``abstractcore[apple]`` cannot
resolve there (no MLX / PyTorch wheels for Intel macOS) and ``abstractcore[gpu]`` does not install
(vLLM ships Linux wheels only; on Windows it only resolves to an sdist that does not build). A
capability that needs a local engine is then plainly *not available on this machine with the
three settings* -- `not_available_here` -- never a bare-package workaround.
"""

from __future__ import annotations

import platform
import sys
from typing import Optional

LIGHT_INSTALL = "pip install -U abstractcore"
APPLE_INSTALL = 'pip install "abstractcore[apple]"'
GPU_INSTALL = 'pip install "abstractcore[gpu]"'

_SETTING_INSTALL = {"apple": APPLE_INSTALL, "gpu": GPU_INSTALL}


def host_setting(os_id: Optional[str], arch: Optional[str]) -> Optional[str]:
    """The local-engine setting for a host given as `host_profile.normalize_os` /
    `normalize_arch` values: ``apple`` (macOS arm64), ``gpu`` (Linux) or None."""

    os_key = str(os_id or "").strip().lower()
    arch_key = str(arch or "").strip().lower()
    if os_key == "darwin":
        return "apple" if arch_key in {"arm64", "aarch64"} else None
    if os_key == "linux":
        return "gpu"
    return None


def local_engines_setting() -> Optional[str]:
    """The setting that carries local engines on this machine: ``apple``, ``gpu`` or None.

    Apple silicon macOS -> ``apple``; Linux -> ``gpu``; other hosts (Intel Macs, Windows) have
    no local-engine setting and run local models through Ollama or LM Studio on light.
    """
    if sys.platform == "darwin":
        return host_setting("darwin", platform.machine().lower())
    if sys.platform.startswith("linux"):
        return "gpu"
    return None


def setting_install_command(setting: Optional[str], *, upgrade: bool = False) -> Optional[str]:
    """`APPLE_INSTALL` / `GPU_INSTALL` for ``apple`` / ``gpu`` (with ``-U`` when `upgrade`),
    else None."""

    command = _SETTING_INSTALL.get(str(setting or ""))
    if command and upgrade:
        command = command.replace("pip install ", "pip install -U ", 1)
    return command


def local_engines_install_command() -> Optional[str]:
    """The install command for this machine's local engines, or None when the host has none."""

    return setting_install_command(local_engines_setting())


def not_available_here(what: str) -> str:
    """The plain sentence for a capability no install setting provides on this host."""

    return (
        f"{what} is not available on this machine with AbstractCore's install settings "
        "(light: every remote provider; apple: Apple silicon; gpu: Linux with an NVIDIA or AMD GPU); "
        "use a remote provider or a local server such as Ollama or LM Studio"
    )


def setting_hint(what: str, setting: Optional[str], *, upgrade: bool = False) -> str:
    """``Install with: <setting command>`` (``Upgrade with: ...`` with `upgrade`), or
    `not_available_here` when `setting` is None."""

    command = setting_install_command(setting, upgrade=upgrade)
    if command:
        return f"{'Upgrade' if upgrade else 'Install'} with: {command}"
    return not_available_here(what)


def local_engines_hint(what: str, *, upgrade: bool = False) -> str:
    """How to get a local-engine capability (`what`) on THIS machine, as one sentence."""

    return setting_hint(what, local_engines_setting(), upgrade=upgrade)


def light_repair_hint() -> str:
    """A missing light dependency: the install is old or broken."""

    return f"the AbstractCore install is incomplete; repair it with: {LIGHT_INSTALL}"
