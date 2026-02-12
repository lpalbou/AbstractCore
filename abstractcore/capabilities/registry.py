from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .errors import CapabilityUnavailableError
from .types import AudioCapability, MusicCapability, VisionCapability, VoiceCapability


_PLUGIN_ENTRYPOINT_GROUP = "abstractcore.capabilities_plugins"

T = TypeVar("T")


@dataclass(frozen=True)
class CapabilityBackendInfo:
    capability: str
    backend_id: str
    priority: int
    description: Optional[str] = None
    install_hint: Optional[str] = None
    config_hint: Optional[str] = None


@dataclass(frozen=True)
class _BackendRegistration:
    info: CapabilityBackendInfo
    factory: Callable[[Any], Any]


class CapabilityRegistry:
    """Host/session-scoped capability registry.

    Plugins register capability backends via `register_*_backend(...)`.
    The registry resolves a default backend per capability deterministically:
    - explicit preference (config) beats order,
    - otherwise highest priority wins (tie-breaker: backend_id).
    """

    def __init__(self, owner: Any, *, preferred_backends: Optional[Dict[str, str]] = None) -> None:
        self._owner = owner
        self._preferred_backends: Dict[str, str] = {}
        if isinstance(preferred_backends, dict):
            self._preferred_backends = {str(k): str(v) for k, v in preferred_backends.items() if str(k) and str(v)}

        self._plugins_loaded = False
        self._plugin_errors: List[Dict[str, Any]] = []
        self._plugins_seen: List[Dict[str, Any]] = []

        # capability -> backend_id -> registration
        self._registrations: Dict[str, Dict[str, _BackendRegistration]] = {}
        # (capability, backend_id) -> backend instance
        self._instances: Dict[Tuple[str, str], Any] = {}

        self.voice = _VoiceFacade(self)
        self.audio = _AudioFacade(self)
        self.vision = _VisionFacade(self)
        self.music = _MusicFacade(self)

    def set_preferred_backend(self, capability: str, backend_id: str) -> None:
        cap = str(capability or "").strip()
        bid = str(backend_id or "").strip()
        if not cap or not bid:
            return
        self._preferred_backends[cap] = bid

    def _ensure_plugins_loaded(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True

        try:
            from importlib.metadata import entry_points
        except Exception:  # pragma: no cover
            entry_points = None  # type: ignore[assignment]

        if entry_points is None:
            self._plugin_errors.append(
                {
                    "group": _PLUGIN_ENTRYPOINT_GROUP,
                    "error": "importlib.metadata.entry_points is not available",
                }
            )
            return

        eps: List[Any] = []
        try:
            raw = entry_points()
            if hasattr(raw, "select"):
                eps = list(raw.select(group=_PLUGIN_ENTRYPOINT_GROUP))  # type: ignore[attr-defined]
            else:  # Python 3.8/3.9 compat
                eps = list(getattr(raw, "get", lambda _k, _d=None: _d)(_PLUGIN_ENTRYPOINT_GROUP, []) or [])
        except Exception as e:
            self._plugin_errors.append({"group": _PLUGIN_ENTRYPOINT_GROUP, "error": str(e)})
            return

        for ep in eps:
            name = getattr(ep, "name", None)
            value = getattr(ep, "value", None)
            self._plugins_seen.append({"name": name, "value": value})
            try:
                obj = ep.load()
            except Exception as e:
                self._plugin_errors.append({"name": name, "value": value, "error": str(e)})
                continue

            try:
                if callable(obj):
                    obj(self)
                else:
                    register = getattr(obj, "register", None)
                    if callable(register):
                        register(self)
                    else:
                        raise TypeError("Plugin entry point must be callable or expose .register(registry)")
            except Exception as e:
                self._plugin_errors.append({"name": name, "value": value, "error": str(e)})

    def register_backend(
        self,
        *,
        capability: str,
        backend_id: str,
        factory: Callable[[Any], Any],
        priority: int = 0,
        description: Optional[str] = None,
        install_hint: Optional[str] = None,
        config_hint: Optional[str] = None,
    ) -> None:
        cap = str(capability or "").strip()
        bid = str(backend_id or "").strip()
        if not cap:
            raise ValueError("capability must be a non-empty string")
        if not bid:
            raise ValueError("backend_id must be a non-empty string")
        if not callable(factory):
            raise ValueError("factory must be callable")

        info = CapabilityBackendInfo(
            capability=cap,
            backend_id=bid,
            priority=int(priority or 0),
            description=str(description).strip() if isinstance(description, str) and description.strip() else None,
            install_hint=str(install_hint).strip() if isinstance(install_hint, str) and install_hint.strip() else None,
            config_hint=str(config_hint).strip() if isinstance(config_hint, str) and config_hint.strip() else None,
        )
        self._registrations.setdefault(cap, {})[bid] = _BackendRegistration(info=info, factory=factory)

    def register_voice_backend(
        self,
        *,
        backend_id: str,
        factory: Callable[[Any], VoiceCapability],
        priority: int = 0,
        description: Optional[str] = None,
        install_hint: Optional[str] = None,
        config_hint: Optional[str] = None,
    ) -> None:
        self.register_backend(
            capability="voice",
            backend_id=backend_id,
            factory=factory,
            priority=priority,
            description=description,
            install_hint=install_hint,
            config_hint=config_hint,
        )

    def register_audio_backend(
        self,
        *,
        backend_id: str,
        factory: Callable[[Any], AudioCapability],
        priority: int = 0,
        description: Optional[str] = None,
        install_hint: Optional[str] = None,
        config_hint: Optional[str] = None,
    ) -> None:
        self.register_backend(
            capability="audio",
            backend_id=backend_id,
            factory=factory,
            priority=priority,
            description=description,
            install_hint=install_hint,
            config_hint=config_hint,
        )

    def register_vision_backend(
        self,
        *,
        backend_id: str,
        factory: Callable[[Any], VisionCapability],
        priority: int = 0,
        description: Optional[str] = None,
        install_hint: Optional[str] = None,
        config_hint: Optional[str] = None,
    ) -> None:
        self.register_backend(
            capability="vision",
            backend_id=backend_id,
            factory=factory,
            priority=priority,
            description=description,
            install_hint=install_hint,
            config_hint=config_hint,
        )

    def register_music_backend(
        self,
        *,
        backend_id: str,
        factory: Callable[[Any], MusicCapability],
        priority: int = 0,
        description: Optional[str] = None,
        install_hint: Optional[str] = None,
        config_hint: Optional[str] = None,
    ) -> None:
        self.register_backend(
            capability="music",
            backend_id=backend_id,
            factory=factory,
            priority=priority,
            description=description,
            install_hint=install_hint,
            config_hint=config_hint,
        )

    def _default_install_hint(self, capability: str) -> Optional[str]:
        cap = str(capability or "").strip().lower()
        if cap == "voice" or cap == "audio":
            return "pip install abstractvoice"
        if cap == "vision":
            return "pip install abstractvision"
        if cap == "music":
            return "pip install abstractmusic"
        return None

    def _select_backend_id(self, capability: str) -> str:
        regs = self._registrations.get(capability) or {}
        if not regs:
            raise CapabilityUnavailableError(
                capability=capability,
                reason="No backends registered",
                install_hint=self._default_install_hint(capability),
            )

        preferred = self._preferred_backends.get(capability)
        if isinstance(preferred, str) and preferred in regs:
            return preferred

        # Deterministic default: highest priority, tie-break on backend_id.
        sorted_ids = sorted(regs.keys())
        best_id = sorted_ids[0]
        best_prio = int(regs[best_id].info.priority)
        for bid in sorted_ids[1:]:
            prio = int(regs[bid].info.priority)
            if prio > best_prio:
                best_id = bid
                best_prio = prio
        return best_id

    def _get_instance(self, capability: str) -> Any:
        self._ensure_plugins_loaded()
        bid = self._select_backend_id(capability)
        key = (capability, bid)
        if key in self._instances:
            return self._instances[key]

        reg = (self._registrations.get(capability) or {}).get(bid)
        if reg is None:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=f"Selected backend '{bid}' is not registered",
                install_hint=self._default_install_hint(capability),
                details={"backend_id": bid},
            )

        try:
            inst = reg.factory(self._owner)
        except CapabilityUnavailableError:
            raise
        except Exception as e:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=f"Backend '{bid}' failed to initialize: {e}",
                install_hint=reg.info.install_hint or self._default_install_hint(capability),
                config_hint=reg.info.config_hint,
                details={"backend_id": bid},
            ) from e

        self._instances[key] = inst
        return inst

    def get_voice(self) -> VoiceCapability:
        out = self._get_instance("voice")
        return out  # type: ignore[return-value]

    def get_audio(self) -> AudioCapability:
        out = self._get_instance("audio")
        return out  # type: ignore[return-value]

    def get_vision(self) -> VisionCapability:
        out = self._get_instance("vision")
        return out  # type: ignore[return-value]

    def get_music(self) -> MusicCapability:
        out = self._get_instance("music")
        return out  # type: ignore[return-value]

    def status(self) -> Dict[str, Any]:
        """Return a JSON-safe capability availability snapshot."""
        self._ensure_plugins_loaded()

        out: Dict[str, Any] = {
            "group": _PLUGIN_ENTRYPOINT_GROUP,
            "plugins_loaded": bool(self._plugins_loaded),
            "plugins_seen": list(self._plugins_seen),
            "plugin_errors": list(self._plugin_errors),
            "capabilities": {},
        }

        for cap in ["voice", "audio", "vision", "music"]:
            regs = self._registrations.get(cap) or {}
            backends = sorted(
                [
                    {
                        "backend_id": r.info.backend_id,
                        "priority": int(r.info.priority),
                        "description": r.info.description,
                    }
                    for r in regs.values()
                ],
                key=lambda b: (-(int(b.get("priority") or 0)), str(b.get("backend_id") or "")),
            )
            available = bool(regs)
            selected = None
            if available:
                try:
                    selected = self._select_backend_id(cap)
                except Exception:
                    selected = None
            out["capabilities"][cap] = {
                "available": available,
                "selected_backend": selected,
                "backends": backends,
            }
            if not available:
                out["capabilities"][cap]["install_hint"] = self._default_install_hint(cap)

        return out


class _VoiceFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def tts(self, text: str, **kwargs: Any) -> Any:
        return self._registry.get_voice().tts(text, **kwargs)

    def stt(self, audio: Any, **kwargs: Any) -> Any:
        return self._registry.get_voice().stt(audio, **kwargs)


class _AudioFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        return self._registry.get_audio().transcribe(audio, **kwargs)


class _VisionFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def t2i(self, prompt: str, **kwargs: Any) -> Any:
        return self._registry.get_vision().t2i(prompt, **kwargs)

    def i2i(self, prompt: str, image: Any, **kwargs: Any) -> Any:
        return self._registry.get_vision().i2i(prompt, image, **kwargs)

    def t2v(self, prompt: str, **kwargs: Any) -> Any:
        return self._registry.get_vision().t2v(prompt, **kwargs)

    def i2v(self, image: Any, **kwargs: Any) -> Any:
        return self._registry.get_vision().i2v(image, **kwargs)


class _MusicFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def t2m(self, prompt: str, **kwargs: Any) -> Any:
        return self._registry.get_music().t2m(prompt, **kwargs)

