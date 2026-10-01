"""Is the engine behind an in-process route INSTALLED on this machine?

A capability route can fail at first use for three different reasons, and
every grid says which one (they are never folded together):

    route_unavailable  this HOST cannot run the engine at all (MLX on Linux):
                       `capability_defaults.configured_routes_unavailable`
    engine_missing     the host can run it, but the engine's software is not
                       installed in this Python environment: THIS module
    weights            the engine is here, the model files are not
                       ("not downloaded"): `model_materializer.probe`

Only IN-PROCESS engines are judged: software this process imports to run the
route. A server provider (LM Studio, Ollama, vLLM, OpenAI-compatible) may be
running on another machine, and a cloud provider runs anywhere, so nothing
installed here decides whether they work.

    provider      engine        installed when                      owner of the answer
    mlx           mlx           `mlx_lm` is importable              AbstractCore's MLX provider
    huggingface   llamacpp      `llama_cpp` is importable           AbstractCore's GGUF lane
                  huggingface   `transformers` and `torch`          AbstractCore's Transformers lane
    mlx-gen       mlx-gen       `abstractvision` and `mlx-gen`      AbstractVision's MLX-Gen backend

Every install command is one of the three settings (light repair, `abstractcore[apple]`,
`abstractcore[gpu]`); a host with no local-engine setting gets no command and a plain
not-available sentence.
    voice routes  <engine>      abstractvoice's own answer          `abstractvoice.engine_runtime`

VOICE: AbstractVoice owns which packages each of its engines needs
(Supertonic: onnxruntime, faster-whisper: faster_whisper, ...), so the answer
is its public `engine_runtime_status` (abstractvoice >= the `voice` extra's
floor in pyproject.toml). An AbstractVoice too old to have that API is itself
the missing engine of every local voice route (upgrade command included):
guessing its engines' packages here is exactly the drift the API exists to
stop, and one old package must not take the whole grid down. With
AbstractVoice absent, every local voice route is missing it.

Every install command core writes itself targets THIS interpreter
(`engines.pip_install_command`, the Engines screen rows' argv).

Every check is `importlib.util.find_spec` / package metadata: nothing is
imported, so a grid of every route costs milliseconds.
"""

from __future__ import annotations

import importlib.util
import shlex
from importlib import metadata
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "ABSTRACTVOICE_ENGINE_RUNTIME_FLOOR",
    "provider_engine_installed",
    "provider_engines_installed",
    "route_engine_missing",
    "routes_engine_missing",
    "voice_engine_id",
]

# The first AbstractVoice with `abstractvoice.engine_runtime` (the public
# runtime probe). Mirrors the floor of the `voice` extra in pyproject.toml;
# the release stager raises both together.
ABSTRACTVOICE_ENGINE_RUNTIME_FLOOR = "0.13.2"

# Voice providers that run remotely: AbstractVoice needs nothing beyond its
# core install for them, and whether they are configured is not a runtime
# question.
_REMOTE_VOICE_PROVIDERS = frozenset({"openai", "openai-compatible"})

# The speech-input provider ids AbstractVoice's AbstractCore plugin accepts on
# an STT route besides its engine ids (`integrations/abstractcore_plugin.py`:
# `_norm_compat_provider_id`, `_engine_aliases`, `_stt_model_ids_for_provider`).
# Its public `engine_runtime` alias table (abstractvoice 0.13.2) does not carry
# them, so they are mirrored here, explicitly; everything else goes through
# `engine_runtime.normalize_engine_id` (underscores, the TTS aliases). A new
# alias upstream belongs in `engine_runtime._ALIASES` and then here.
_STT_PROVIDER_ALIASES = {
    "whisper": "faster-whisper",
    "local": "faster-whisper",
    "faster_whisper": "faster-whisper",
    "transformers": "transformers-asr",
    "transformers_asr": "transformers-asr",
    "hf": "transformers-asr",
    "hf-asr": "transformers-asr",
}

# What to do about a voice route whose provider is no AbstractVoice engine:
# (what the id is not, the engine kind to list, the fix).
_VOICE_PICK = {
    "input.voice": ("a transcription engine", "stt", "Pick a transcription engine on the Multimodal page."),
    "output.voice": ("a voice engine", "tts", "Pick a voice engine on the Multimodal page."),
}


def voice_engine_id(provider: Any, key: Any = None) -> str:
    """The AbstractVoice engine id a voice route's provider names: `whisper` /
    `local` -> `faster-whisper`, `hf` -> `transformers-asr` on `input.voice`
    (the plugin's aliases), else the id as given (lowercased). No lookup of
    whether the engine exists: `route_engine_missing` answers that."""

    pid = str(provider or "").strip().lower()
    if str(key or "").strip().lower() == "input.voice":
        return _STT_PROVIDER_ALIASES.get(pid, pid)
    return pid


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _distributed(dist: str) -> bool:
    try:
        metadata.distribution(dist)
        return True
    except metadata.PackageNotFoundError:
        return False


def _pip_command(*packages: str) -> str:
    from .engines import pip_install_command

    return pip_install_command(*packages)


def _engine_plan_command(engine_id: str) -> str:
    """The Engines screen's own install command for a Python engine row
    (`engines.engine_install_plan`: the one allowlist)."""

    from .engines import engine_install_plan

    plan = engine_install_plan(engine_id)
    argv = plan.get("argv") or []
    return shlex.join(str(a) for a in argv)


def _missing(engine: str, name: str, reason: str, install: Optional[str], engine_row: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"engine": engine, "name": name, "reason": reason, "install": install}
    if engine_row:
        # The Engines screen row that installs it (`i` there runs `install`).
        out["engine_row"] = engine_row
    return out


def _python_engine(engine: str, name: str, modules: Tuple[str, ...], what: str) -> Optional[Dict[str, Any]]:
    absent = [m for m in modules if not _importable(m)]
    if not absent:
        return None
    from .engines import engine_install_plan

    plan = engine_install_plan(engine)
    if not plan.get("available"):
        # No local-engine setting on this host (or llama.cpp on Windows, which the installer
        # adds): the plan's notes say so plainly.
        return _missing(
            engine,
            name,
            f"{name} is not installed in this Python environment ({', '.join(absent)} missing); {what}. "
            f"{plan.get('notes')}",
            None,
            engine_row=engine,
        )
    install = _engine_plan_command(engine)
    return _missing(
        engine,
        name,
        f"{name} is not installed in this Python environment ({', '.join(absent)} missing); {what}. "
        f"Install it with: {install}",
        install,
        engine_row=engine,
    )


def _setting_install(what: str) -> Tuple[Optional[str], str]:
    """`(install, sentence)` for a local engine on THIS machine: its local-engine setting
    (`abstractcore[apple]` / `abstractcore[gpu]`, into this interpreter), or no command and
    the plain not-available sentence. Never a bare package or a plugin's own extra
    (operator ruling 2026-09-29: users only ever install one of the three settings)."""

    from ..utils.install_settings import local_engines_setting, not_available_here

    setting = local_engines_setting()
    if not setting:
        return None, not_available_here(what)
    install = _pip_command(f"abstractcore[{setting}]")
    return install, f"Install it with: {install}"


def _mlx_gen() -> Optional[Dict[str, Any]]:
    absent = [d for d in ("abstractvision", "mlx-gen") if not _distributed(d)]
    if not absent:
        return None
    if absent == ["abstractvision"]:
        # AbstractVision is part of the light install: missing means a broken install.
        install = _pip_command("-U", "abstractcore")
        sentence = f"Install it with: {install}"
    else:
        install, sentence = _setting_install("MLX-Gen image and video generation")
    return _missing(
        "mlx-gen",
        "MLX-Gen (AbstractVision)",
        f"MLX-Gen image and video generation is not installed in this Python environment "
        f"({', '.join(absent)} missing). {sentence}",
        install,
    )


def _voice(provider: str, key: str = "") -> Optional[Dict[str, Any]]:
    provider = voice_engine_id(provider, key)
    if provider in _REMOTE_VOICE_PROVIDERS:
        return None
    if not _distributed("abstractvoice"):
        # AbstractVoice is part of the light install: a missing one is an old or broken install.
        install = _pip_command("-U", "abstractcore")
        return _missing(
            provider,
            "AbstractVoice",
            f"voice runs in AbstractVoice, which is not installed in this Python environment. "
            f"Install it with: {install}",
            install,
        )
    try:
        from abstractvoice.engine_runtime import engine_runtime_status, known_engines
    except ImportError:
        # The light install's floor IS this floor: upgrading AbstractCore upgrades AbstractVoice.
        install = _pip_command("-U", "abstractcore")
        return _missing(
            provider,
            "AbstractVoice",
            f"abstractvoice {_dist_version('abstractvoice')} has no public engine runtime probe "
            f"(abstractvoice.engine_runtime); AbstractCore needs abstractvoice>={ABSTRACTVOICE_ENGINE_RUNTIME_FLOOR}. "
            f"Install it with: {install}",
            install,
        )
    try:
        status = engine_runtime_status(provider)
    except ValueError:
        # Not an engine AbstractVoice has: the route cannot run. The reason
        # says what the id is not and what to do (a download source such as
        # `huggingface` on a transcription route is the usual case), with the
        # engines AbstractVoice lists for that kind (`known_engines`).
        what, kind, fix = _VOICE_PICK.get(key, ("an AbstractVoice engine", None, "Pick a voice engine on the Multimodal page."))
        engines = ", ".join(known_engines(kind))
        return _missing(
            provider,
            provider,
            f"{provider!r} is not {what} AbstractVoice has (it has: {engines}). {fix}",
            None,
        )
    if status.installed:
        return None
    # AbstractVoice's own `install_command` / `reason` name its standalone extra
    # (`abstractvoice[supertonic]`); inside AbstractCore every voice engine's runtime
    # arrives with the local-engine setting (abstractvoice[all-apple] / [all-gpu]).
    missing = ", ".join(status.missing_modules) or "its runtime"
    install, sentence = _setting_install(f"The {status.label} voice engine")
    return _missing(
        status.engine,
        status.label,
        f"{status.label} is not installed in this Python environment ({missing} missing). {sentence}",
        install,
    )


def _dist_version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "(unknown version)"


# In-process providers a RECOMMENDATION may name, and what makes each usable in this Python
# environment (the host profile's `engines_installed`, read by the recommendation so the light
# install profile is never handed an engine it does not have). Voice engines are AbstractVoice's
# own answer (`route_engine_missing(..., key=)`), not judged here: they ship with the light
# profile and asking needs AbstractVoice imported.
def provider_engine_installed(provider: Any) -> Optional[bool]:
    """True / False for an in-process provider a recommendation can name (mlx, mlx-gen,
    diffusers, acestep), None for any other provider (not judged). Lookups only."""

    pid = str(provider or "").strip().lower()
    if pid == "mlx":
        return _importable("mlx_lm")
    if pid == "mlx-gen":
        return _distributed("abstractvision") and _distributed("mlx-gen")
    if pid in ("diffusers", "acestep"):
        return _importable("torch")
    return None


def provider_engines_installed() -> Dict[str, bool]:
    """`{provider: installed}` for every provider `provider_engine_installed` judges."""

    out: Dict[str, bool] = {}
    for pid in ("mlx", "mlx-gen", "diffusers", "acestep"):
        value = provider_engine_installed(pid)
        if value is not None:
            out[pid] = bool(value)
    return out


def route_engine_missing(provider: Any, model: Any = None, key: Any = None) -> Optional[Dict[str, Any]]:
    """`{engine, name, reason, install[, engine_row]}` when the in-process
    engine behind this route is not installed here, else None.

    `key` is the route key (`output.voice`, `input.voice`, ...): voice routes
    are AbstractVoice engines whatever their provider id. `model` decides the
    Hugging Face lane (a GGUF reference runs on llama.cpp, anything else on
    Transformers). Providers that are not in-process return None.
    """

    pid = str(provider or "").strip().lower()
    if not pid:
        return None
    route_key = str(key or "").strip().lower()
    if route_key in ("input.voice", "output.voice"):
        return _voice(pid, route_key)
    if pid == "mlx":
        return _python_engine("mlx", "MLX (mlx-lm)", ("mlx_lm",), "the mlx provider runs models with it")
    if pid == "mlx-gen":
        return _mlx_gen()
    if pid == "huggingface":
        from ..utils.model_cache import is_gguf_model_ref

        if is_gguf_model_ref(str(model or "")):
            return _python_engine(
                "llamacpp", "llama.cpp (llama-cpp-python)", ("llama_cpp",), "GGUF models run in it"
            )
        return _python_engine(
            "huggingface", "Hugging Face Transformers", ("transformers", "torch"), "Transformers models run in it"
        )
    return None


def routes_engine_missing(routes: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """`{key: engine_missing}` for every route in `{key: route-like}` whose
    in-process engine is not installed (the grid, the apply plan)."""

    from .capability_defaults import clean_capability_route_default

    out: Dict[str, Dict[str, Any]] = {}
    for key, value in routes.items():
        route = clean_capability_route_default(value)
        flag = route_engine_missing(route.provider, route.model, key)
        if flag is not None:
            out[str(key)] = flag
    return out
