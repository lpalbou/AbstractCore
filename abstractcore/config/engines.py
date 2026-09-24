"""Local inference engines: are they here, are they running, how to install them.

Contract B (`engines_status_v1`). An ENGINE is the third-party software that
runs local weights: Ollama, LM Studio, MLX (mlx-lm), llama.cpp, vLLM, and the
Hugging Face stack. This module answers three questions about each, and is
the ONE implementation both entry points (the AbstractCore CLI/server and
the AbstractGateway, which inherits it) use:

    engine_inventory(probe)              -> installed? version? running? reachable?
    engine_install_plan(engine, os, arch) -> the EXACT command that would install it
    engine_install(engine, dry_run)      -> run that command as a host job

HOST SAFETY, the rules this module will not bend:

1. FIXED ARGV FROM AN ALLOWLIST. Every command an install can run is a
   constant in `_PLANS` below, chosen by (engine, os, available tool). No
   part of it comes from a request, a config file or an environment variable;
   there is nothing to inject into. The three vendor bootstrap one-liners
   (Ollama and LM Studio's own install scripts) are constant strings passed to
   `sh -c` / `powershell -Command` -- the exact commands the vendors document.
2. ONE INSTALL AT A TIME. A second install while one runs is refused
   (`JobBusy` -> HTTP 409), never queued behind the first.
3. OPT-IN OFF LOOPBACK. `engine_install_allowed()` is True for local CLI use;
   the core server turns it on by default only when bound to loopback
   (`ABSTRACTCORE_ALLOW_ENGINE_INSTALL` overrides). Installs run on the host
   that runs THIS process, which for a server is not the browser's machine.
4. SHOW BEFORE DOING. `dry_run` returns the plan (argv, notes, whether it
   needs sudo/UAC) without running anything; every job records its CLI
   equivalent so a human can run the same command by hand.
5. NO SILENT PRIVILEGE. Installers run with stdin closed: a sudo password
   prompt fails fast with the tool's own words instead of hanging a job.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "ENGINES_SCHEMA",
    "ENGINE_IDS",
    "EngineInstallRefused",
    "engine_ids",
    "engine_inventory",
    "engine_status",
    "engine_install_plan",
    "engine_install",
    "engine_install_allowed",
    "set_engine_install_allowed",
    "engine_download_url",
]

ENGINES_SCHEMA = "engines_status_v1"
ENGINE_IDS: Tuple[str, ...] = ("ollama", "lmstudio", "mlx", "llamacpp", "vllm", "huggingface")

# Module-level policy knob. True for local CLI use (the operator IS the host);
# servers override per request with their own policy (see server routes).
ALLOW_ENGINE_INSTALL: bool = True

_VERSION_TIMEOUT_S = 5.0
_INSTALL_TIMEOUT_S = 60 * 60
_version_cache: Dict[str, Tuple[float, Any]] = {}
_version_lock = threading.Lock()
_VERSION_TTL_S = 60.0


class EngineInstallRefused(RuntimeError):
    """The install was refused by policy (not allowed, unsupported, no plan)."""

    def __init__(self, message: str, *, reason: str = "refused", plan: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.reason = reason
        self.plan = plan


def engine_ids() -> List[str]:
    return list(ENGINE_IDS)


def set_engine_install_allowed(value: bool) -> None:
    global ALLOW_ENGINE_INSTALL
    ALLOW_ENGINE_INSTALL = bool(value)


def engine_install_allowed() -> bool:
    """`ABSTRACTCORE_ALLOW_ENGINE_INSTALL` when set, else the module knob."""

    raw = str(os.getenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(ALLOW_ENGINE_INSTALL)


# ---------------------------------------------------------------------------
# Static engine facts
# ---------------------------------------------------------------------------

_ENGINE_META: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "name": "Ollama",
        "kind": "local_server",
        "docs_url": "https://docs.ollama.com",
        "download_url": "https://ollama.com/download",
        "provider": "ollama",
    },
    "lmstudio": {
        "name": "LM Studio",
        "kind": "local_server",
        "docs_url": "https://lmstudio.ai/docs",
        "download_url": "https://lmstudio.ai/download",
        "provider": "lmstudio",
    },
    "mlx": {
        "name": "MLX (mlx-lm)",
        "kind": "local_engine",
        "docs_url": "https://github.com/ml-explore/mlx-lm",
        "download_url": "https://pypi.org/project/mlx-lm/",
        "provider": "mlx",
    },
    "llamacpp": {
        "name": "llama.cpp",
        "kind": "local_engine",
        "docs_url": "https://github.com/ggml-org/llama.cpp",
        "download_url": "https://github.com/ggml-org/llama.cpp/releases",
        "provider": "huggingface",
    },
    "vllm": {
        "name": "vLLM",
        "kind": "local_server",
        "docs_url": "https://docs.vllm.ai",
        "download_url": "https://docs.vllm.ai/en/latest/getting_started/installation/",
        "provider": "vllm",
    },
    "huggingface": {
        "name": "Hugging Face (transformers)",
        "kind": "local_engine",
        "docs_url": "https://huggingface.co/docs/transformers",
        "download_url": "https://pypi.org/project/transformers/",
        "provider": "huggingface",
    },
}

# Vendor bootstrap one-liners: CONSTANTS, exactly as the vendors document them.
_OLLAMA_SH = "curl -fsSL https://ollama.com/install.sh | sh"
_OLLAMA_PS1 = "irm https://ollama.com/install.ps1 | iex"
_LMS_SH = "curl -fsSL https://lmstudio.ai/install.sh | bash"
_LMS_PS1 = "irm https://lmstudio.ai/install.ps1 | iex"
_WINGET_FLAGS = ["-e", "--accept-source-agreements", "--accept-package-agreements"]


def _powershell(command: str) -> List[str]:
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]


def _pip_argv(packages: List[str], extra: Optional[List[str]] = None, *, prefer_uv: Optional[bool] = None) -> Tuple[str, List[str]]:
    """`(method, argv)` installing into THIS interpreter's environment.

    `uv` venvs ship without pip; when pip is missing and `uv` is on PATH the
    plan uses `uv pip install --python <this interpreter>`, which targets the
    same environment.
    """

    use_uv = prefer_uv
    if use_uv is None:
        use_uv = importlib.util.find_spec("pip") is None and shutil.which("uv") is not None
    if use_uv:
        return "pip", ["uv", "pip", "install", "--python", sys.executable, *packages, *(extra or [])]
    return "pip", [sys.executable, "-m", "pip", "install", *packages, *(extra or [])]


# ---------------------------------------------------------------------------
# Support matrix
# ---------------------------------------------------------------------------


def _support(engine: str, os_id: str, arch: str, accelerator: Optional[str]) -> Tuple[bool, Optional[str]]:
    if engine == "mlx":
        if os_id == "darwin" and arch == "arm64":
            return True, None
        return False, "MLX runs only on Apple Silicon Macs (macOS, arm64)"
    if engine == "lmstudio":
        if os_id == "darwin" and arch != "arm64":
            return False, "LM Studio on macOS requires Apple Silicon (arm64) and macOS 14+"
        if os_id in {"darwin", "linux", "windows"}:
            return True, None
        return False, f"LM Studio has no build for {os_id}"
    if engine == "vllm":
        if os_id == "linux" and accelerator == "cuda":
            return True, None
        return False, (
            "vLLM runs on Linux with an NVIDIA GPU (CUDA); on this host use a remote vLLM server "
            "through VLLM_BASE_URL instead"
        )
    if engine in {"ollama", "llamacpp", "huggingface"}:
        if os_id in {"darwin", "linux", "windows"}:
            return True, None
        return False, f"no supported build for {os_id}"
    return False, f"unknown engine {engine!r}"


# ---------------------------------------------------------------------------
# Install plans (the allowlist)
# ---------------------------------------------------------------------------


def _plan(
    method: Optional[str],
    argv: Optional[List[str]],
    *,
    url: Optional[str],
    notes: str,
    requires_admin: bool = False,
    alternatives: Optional[List[Dict[str, Any]]] = None,
    available: bool = True,
    estimated_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "available": bool(available and argv),
        "method": method,
        "argv": list(argv or []),
        "url": url,
        "requires_confirmation": True,
        "requires_admin": bool(requires_admin),
        "estimated_bytes": estimated_bytes,
        "notes": notes,
        "alternatives": list(alternatives or []),
    }


def engine_install_plan(
    engine_id: str,
    os_name: Optional[str] = None,
    arch: Optional[str] = None,
    *,
    accelerator: Optional[str] = None,
    tools: Optional[Dict[str, bool]] = None,
    prefer_uv: Optional[bool] = None,
) -> Dict[str, Any]:
    """The contract-B `install` block for one engine on one (os, arch).

    `tools` ({"brew": bool, "winget": bool}) defaults to what is on PATH here;
    pass it explicitly to plan for another host (tests, remote planning).
    """

    from ..utils.host_profile import normalize_arch, normalize_os

    eid = str(engine_id or "").strip().lower()
    if eid not in _ENGINE_META:
        raise KeyError(f"unknown engine {engine_id!r}; known: {', '.join(ENGINE_IDS)}")
    os_id = normalize_os(os_name)
    arch_id = normalize_arch(arch)
    have = dict(tools or {})
    if "brew" not in have:
        have["brew"] = shutil.which("brew") is not None
    if "winget" not in have:
        have["winget"] = shutil.which("winget") is not None
    meta = _ENGINE_META[eid]
    url = meta["download_url"]
    supported, reason = _support(eid, os_id, arch_id, accelerator)
    if not supported:
        return _plan(None, None, url=url, notes=reason or "not supported on this host", available=False)

    if eid == "ollama":
        if os_id == "darwin":
            script = _plan(
                "script",
                ["sh", "-c", _OLLAMA_SH],
                url=url,
                notes=(
                    "Runs Ollama's official installer: installs Ollama.app in /Applications and links "
                    "/usr/local/bin/ollama (may ask for your password -- run it in a terminal if the job "
                    "stops at a sudo prompt), then starts the app."
                ),
                requires_admin=True,
            )
            if have["brew"]:
                return _plan(
                    "brew",
                    ["brew", "install", "ollama"],
                    url=url,
                    notes=(
                        "Installs the Ollama CLI and server with Homebrew (no sudo). Start it with "
                        "`ollama serve`, or `brew services start ollama` to run it at login."
                    ),
                    alternatives=[{k: script[k] for k in ("method", "argv", "notes", "requires_admin")}],
                )
            return script
        if os_id == "linux":
            return _plan(
                "script",
                ["sh", "-c", _OLLAMA_SH],
                url=url,
                notes=(
                    "Runs Ollama's official installer: NEEDS SUDO. Installs to /usr/local, creates the "
                    "`ollama` system user and an enabled systemd `ollama.service`. Without passwordless "
                    "sudo, run this command in a terminal instead."
                ),
                requires_admin=True,
            )
        # windows
        script = _plan(
            "script",
            _powershell(_OLLAMA_PS1),
            url=url,
            notes="Runs Ollama's official PowerShell installer (OllamaSetup.exe, per-user, no admin).",
        )
        if have["winget"]:
            return _plan(
                "winget",
                ["winget", "install", "--id", "Ollama.Ollama", *_WINGET_FLAGS],
                url=url,
                notes="Installs Ollama with winget (per-user scope under %LOCALAPPDATA%\\Programs\\Ollama; no admin).",
                alternatives=[{k: script[k] for k in ("method", "argv", "notes", "requires_admin")}],
            )
        return script

    if eid == "lmstudio":
        alternatives: List[Dict[str, Any]] = []
        if os_id == "darwin" and have["brew"]:
            alternatives.append(
                {
                    "method": "brew",
                    "argv": ["brew", "install", "--cask", "lm-studio"],
                    "notes": "Installs the LM Studio desktop app with Homebrew.",
                    "requires_admin": False,
                }
            )
        if os_id == "windows" and have["winget"]:
            alternatives.append(
                {
                    "method": "winget",
                    "argv": ["winget", "install", "--id", "ElementLabs.LMStudio", *_WINGET_FLAGS],
                    "notes": "Installs the LM Studio desktop app with winget.",
                    "requires_admin": False,
                }
            )
        bootstrap = ["bash", "-c", _LMS_SH] if os_id != "windows" else _powershell(_LMS_PS1)
        linux_note = " On Linux it may need `libatomic1` (sudo apt install libatomic1)." if os_id == "linux" else ""
        return _plan(
            "download_page",
            bootstrap,
            url=url,
            notes=(
                "The LM Studio desktop app is a download from the page above. The command installs "
                "LM Studio's HEADLESS daemon (llmster) and the `lms` CLI under ~/.lmstudio (no admin, "
                "no GUI); start it with `lms daemon up` and `lms server start`." + linux_note
            ),
            alternatives=alternatives,
        )

    if eid == "mlx":
        method, argv = _pip_argv(["mlx-lm"], prefer_uv=prefer_uv)
        return _plan(
            method,
            argv,
            url=url,
            notes=f"Installs mlx and mlx-lm into this Python environment ({sys.executable}); no admin.",
        )

    if eid == "llamacpp":
        method, argv = _pip_argv(["llama-cpp-python"], prefer_uv=prefer_uv)
        alternatives = []
        if os_id == "darwin" and have["brew"]:
            alternatives.append(
                {
                    "method": "brew",
                    "argv": ["brew", "install", "llama.cpp"],
                    "notes": "Installs the llama.cpp binaries (llama-server, llama-cli) with Homebrew. "
                    "Note: llama-server defaults to port 8080, the gateway's port.",
                    "requires_admin": False,
                }
            )
        if os_id == "windows" and have["winget"]:
            alternatives.append(
                {
                    "method": "winget",
                    "argv": ["winget", "install", "--id", "ggml.llamacpp", *_WINGET_FLAGS],
                    "notes": "Installs the llama.cpp binaries with winget.",
                    "requires_admin": False,
                }
            )
        accel_note = {
            "darwin": " Builds with Metal on Apple Silicon.",
            "linux": " Builds for CPU unless CMAKE_ARGS=-DGGML_CUDA=on is set (then a CUDA toolkit is needed).",
            "windows": " Needs a C/C++ compiler (Visual Studio Build Tools) when no prebuilt wheel matches.",
        }.get(os_id, "")
        return _plan(
            method,
            argv,
            url=url,
            notes=(
                "Installs llama-cpp-python (the in-process GGUF engine AbstractCore's huggingface "
                f"provider uses) into this Python environment; may compile from source (minutes).{accel_note}"
            ),
            alternatives=alternatives,
        )

    if eid == "vllm":
        use_uv = prefer_uv if prefer_uv is not None else (importlib.util.find_spec("pip") is None and shutil.which("uv") is not None)
        if use_uv:
            argv = ["uv", "pip", "install", "--python", sys.executable, "vllm", "--torch-backend=auto"]
        else:
            argv = [sys.executable, "-m", "pip", "install", "vllm"]
        return _plan(
            "pip",
            argv,
            url=url,
            notes="Installs vLLM (several GB: PyTorch + CUDA wheels) into this Python environment; NVIDIA GPU with compute capability >= 7.5 required.",
        )

    # huggingface
    method, argv = _pip_argv(["abstractcore[huggingface]"], prefer_uv=prefer_uv)
    return _plan(
        method,
        argv,
        url=url,
        notes="Installs transformers, torch and huggingface_hub (AbstractCore's huggingface extra) into this Python environment; several GB.",
    )


def engine_download_url(engine_id: str) -> str:
    eid = str(engine_id or "").strip().lower()
    if eid not in _ENGINE_META:
        raise KeyError(f"unknown engine {engine_id!r}; known: {', '.join(ENGINE_IDS)}")
    return str(_ENGINE_META[eid]["download_url"])


# ---------------------------------------------------------------------------
# Presence / version detection
# ---------------------------------------------------------------------------


def _cached_version(key: str, produce: Callable[[], Any]) -> Any:
    now = time.monotonic()
    with _version_lock:
        hit = _version_cache.get(key)
        if hit is not None and now - hit[0] < _VERSION_TTL_S:
            return hit[1]
    value = produce()
    with _version_lock:
        _version_cache[key] = (now, value)
    return value


def _reset_caches_for_tests() -> None:
    with _version_lock:
        _version_cache.clear()


def _run_text(argv: List[str]) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=_VERSION_TIMEOUT_S)  # noqa: S603
    except Exception:
        return ""
    return re.sub(r"\x1b\[[0-9;]*m", "", (proc.stdout or "") + "\n" + (proc.stderr or ""))


def _plist_version(app: Path) -> Optional[str]:
    try:
        with open(app / "Contents" / "Info.plist", "rb") as fh:
            value = plistlib.load(fh).get("CFBundleShortVersionString")
        return str(value) if value else None
    except Exception:
        return None


def _module_version(dist: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _mac_app(name: str) -> Optional[Path]:
    """`/Applications/<name>`, else `~/Applications/<name>` (a user-level install, no admin rights)."""

    for folder in (Path("/Applications"), Path.home() / "Applications"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def _detect_ollama(os_id: str) -> Dict[str, Any]:
    cli = shutil.which("ollama")
    location = cli
    app: Optional[Path] = None
    if os_id == "darwin":
        app = _mac_app("Ollama.app")
        if app is not None:
            location = location or str(app)
            # The app ships its CLI inside the bundle; /usr/local/bin/ollama is only a link
            # the app offers to create (admin), so an install without it is still complete.
            bundled = app / "Contents" / "Resources" / "ollama"
            if cli is None and bundled.exists():
                cli = str(bundled)
    elif os_id == "windows":
        local = os.getenv("LOCALAPPDATA") or ""
        candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe" if local else None
        if candidate is not None and candidate.exists():
            location = location or str(candidate)
            cli = cli or str(candidate)
    version: Optional[str] = None
    if cli:
        text = _cached_version(f"ollama:{cli}", lambda: _run_text([cli, "--version"]))
        m = re.search(r"(?:client version is|ollama version is|version)\s+v?(\d+\.\d+[\w.\-]*)", text)
        version = m.group(1) if m else None
    if version is None and app is not None:
        version = _plist_version(app)
    return {"installed": bool(location), "install_location": location, "version": version}


def _detect_lmstudio(os_id: str) -> Dict[str, Any]:
    from .model_materializer import _lms_cli

    cli = _lms_cli()
    location = None
    version: Optional[str] = None
    app: Optional[Path] = None
    if os_id == "darwin":
        app = _mac_app("LM Studio.app")
    elif os_id == "windows":
        local = os.getenv("LOCALAPPDATA") or ""
        candidate = Path(local) / "Programs" / "LM Studio" if local else None
        if candidate is not None and candidate.exists():
            app = candidate
    if app is not None:
        location = str(app)
        version = _plist_version(app) if os_id == "darwin" else None
    cli_version: Optional[str] = None
    if cli:
        location = location or cli
        text = _cached_version(f"lms:{cli}", lambda: _run_text([cli, "version"]))
        m = re.search(r"CLI commit:\s*([0-9a-f]+)", text)
        cli_version = f"cli-{m.group(1)}" if m else None
    return {
        "installed": bool(location),
        "install_location": location,
        "version": version or cli_version,
        "cli": cli,
        "cli_version": cli_version,
    }


def _detect_module(engine: str) -> Dict[str, Any]:
    module, dist = {
        "mlx": ("mlx", "mlx"),
        "llamacpp": ("llama_cpp", "llama-cpp-python"),
        "vllm": ("vllm", "vllm"),
        "huggingface": ("transformers", "transformers"),
    }[engine]
    present = _find_spec(module)
    info: Dict[str, Any] = {
        "installed": present,
        "install_location": sys.executable if present else None,
        "version": _module_version(dist) if present else None,
    }
    if engine == "mlx":
        info["mlx_lm_version"] = _module_version("mlx-lm") if _find_spec("mlx_lm") else None
    if engine == "llamacpp":
        server = shutil.which("llama-server")
        info["llama_server"] = server
        if not present and server:
            info.update(installed=True, install_location=server)
    if engine == "vllm" and not present and shutil.which("vllm"):
        info.update(installed=True, install_location=shutil.which("vllm"))
    if engine == "huggingface":
        info["huggingface_hub_version"] = _module_version("huggingface_hub") if _find_spec("huggingface_hub") else None
    return info


def _detect(engine: str, os_id: str) -> Dict[str, Any]:
    if engine == "ollama":
        return _detect_ollama(os_id)
    if engine == "lmstudio":
        return _detect_lmstudio(os_id)
    return _detect_module(engine)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _server_rows(probe: bool) -> Dict[str, Dict[str, Any]]:
    try:
        from .model_materializer import provider_inventory

        rows = provider_inventory(probe=probe)
    except Exception:
        return {}
    return {str(r.get("provider")): r for r in rows if r.get("provider") in {"ollama", "lmstudio", "vllm"}}


def _models_count(reachability: str) -> Optional[int]:
    m = re.search(r"\((\d+) models?\)", str(reachability or ""))
    return int(m.group(1)) if m else None


def engine_status(engine_id: str, *, probe: bool = False, host: Optional[Dict[str, Any]] = None, server_rows: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One contract-B engine row."""

    from ..utils.host_profile import host_profile

    eid = str(engine_id or "").strip().lower()
    if eid not in _ENGINE_META:
        raise KeyError(f"unknown engine {engine_id!r}; known: {', '.join(ENGINE_IDS)}")
    profile = host or host_profile()
    os_id, arch, accel = profile.get("os"), profile.get("arch"), profile.get("accelerator")
    meta = _ENGINE_META[eid]
    supported, reason = _support(eid, str(os_id), str(arch), accel)
    kind = meta["kind"]
    if eid == "vllm" and not supported:
        kind = "remote_only"

    detected = _detect(eid, str(os_id))
    rows = server_rows if server_rows is not None else _server_rows(probe)
    server = rows.get(eid) or {}
    running: Optional[bool] = None
    reachable: Optional[bool] = None
    base_url: Optional[str] = server.get("base_url") or None
    models_count: Optional[int] = None
    if probe and kind == "local_server":
        reachable = server.get("reachable")
        running = reachable
        models_count = _models_count(server.get("reachability", "")) if reachable else None
    if probe and eid == "vllm" and base_url and server.get("reachable") is not None:
        reachable = server.get("reachable")
        running = reachable if supported else None

    row: Dict[str, Any] = {
        "id": eid,
        "name": meta["name"],
        "kind": kind,
        "provider": meta["provider"],
        "supported_on_host": supported,
        "unsupported_reason": reason,
        "installed": bool(detected.get("installed")),
        "version": detected.get("version"),
        "install_location": detected.get("install_location"),
        "running": running,
        "base_url": base_url,
        "reachable": reachable,
        "reachability": server.get("reachability") or None,
        "models_count": models_count,
        "install": engine_install_plan(eid, str(os_id), str(arch), accelerator=accel),
        "docs_url": meta["docs_url"],
    }
    for key in ("cli", "cli_version", "mlx_lm_version", "llama_server", "huggingface_hub_version"):
        if key in detected:
            row[key] = detected[key]
    return row


def engine_inventory(probe: bool = False, *, host: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Contract B (`engines_status_v1`): every engine, one row each.

    `probe=True` adds ONE cheap GET per local server (running / reachable /
    models_count). Presence and versions never touch the network.
    """

    from ..utils.host_profile import host_profile, utc_now_iso

    profile = host or host_profile()
    rows = _server_rows(probe)
    engines = [engine_status(eid, probe=probe, host=profile, server_rows=rows) for eid in ENGINE_IDS]
    return {
        "schema": ENGINES_SCHEMA,
        "engines": engines,
        "probed": bool(probe),
        "install_allowed": engine_install_allowed(),
        "host": {k: profile.get(k) for k in ("os", "arch", "accelerator", "gpu_name")},
        "generated_at": utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# Install (a host job)
# ---------------------------------------------------------------------------


def engine_install(
    engine_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    allow: Optional[bool] = None,
    registry: Any = None,
    run_inline: bool = False,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the engine's install plan as a `host_job_v1` job (kind `engine_install`).

    Raises `EngineInstallRefused` (policy: not allowed, unsupported, no plan)
    and `host_jobs.JobBusy` (an engine install is already running). An engine
    that is already installed finishes at once as `already_installed` unless
    `force`. `dry_run` finishes at once with the command it WOULD run.
    """

    from ..utils.host_profile import host_profile
    from . import host_jobs
    from .model_materializer import _run_streaming

    eid = str(engine_id or "").strip().lower()
    if eid not in _ENGINE_META:
        raise EngineInstallRefused(f"unknown engine {engine_id!r}; known: {', '.join(ENGINE_IDS)}", reason="unknown_engine")
    permitted = engine_install_allowed() if allow is None else bool(allow)
    if not permitted and not dry_run:
        raise EngineInstallRefused(
            "engine installs are disabled on this host (allow_engine_install is off; "
            "set ABSTRACTCORE_ALLOW_ENGINE_INSTALL=1 to enable)",
            reason="not_allowed",
        )
    profile = host_profile()
    status = engine_status(eid, probe=False, host=profile, server_rows={})
    plan = status["install"]
    if not status["supported_on_host"]:
        raise EngineInstallRefused(status["unsupported_reason"] or "not supported on this host", reason="unsupported", plan=plan)
    if not plan.get("available") or not plan.get("argv"):
        raise EngineInstallRefused(plan.get("notes") or "no install command for this host", reason="no_plan", plan=plan)

    argv = [str(a) for a in plan["argv"]]
    reg = registry or host_jobs.default_registry()
    already = bool(status["installed"]) and not force

    def runner(ctx: "host_jobs.JobContext") -> Dict[str, Any]:
        base = {"engine": eid, "command": argv, "plan": plan}
        if dry_run:
            return dict(base, ok=True, status="planned", message="would run: " + " ".join(argv))
        if already:
            return dict(
                base,
                ok=True,
                status="already_installed",
                message=f"{status['name']} is already installed" + (f" ({status['version']})" if status.get("version") else ""),
            )
        env = dict(os.environ)
        env.setdefault("NONINTERACTIVE", "1")
        env.setdefault("HOMEBREW_NO_AUTO_UPDATE", "1")
        outcome = _run_streaming(argv, "engine", eid, ctx.progress, env=env)
        data = dict(base, ok=bool(outcome.ok), status=outcome.status, message=outcome.message, output=outcome.output[-4000:])
        if outcome.ok:
            _reset_caches_for_tests()
            after = engine_status(eid, probe=False, host=profile, server_rows={})
            data["installed_after"] = after["installed"]
            data["version_after"] = after["version"]
            if not after["installed"]:
                data["message"] += " (the command succeeded but the engine is not detected yet; a new shell or PATH update may be needed)"
        return data

    return reg.start(
        kind="engine_install",
        key="engine_install",
        runner=runner,
        engine=eid,
        command=argv,
        dry_run=dry_run,
        cli_equivalent=host_jobs.cli_equivalent_engine_install(eid, dry_run),
        join=False,
        run_inline=run_inline,
        job_id=job_id,
    )
