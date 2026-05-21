from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

from .errors import CapabilityUnavailableError
from .host import DefaultCapabilityHostContext
from .types import (
    AudioCapability,
    CapabilityHostContext,
    CapabilityModelInfo,
    CapabilityOperationInfo,
    CapabilityProviderInfo,
    MusicCapability,
    VisionCapability,
    VoiceCapability,
)


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

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "capability": self.capability,
            "backend_id": self.backend_id,
            "priority": int(self.priority),
        }
        if self.description is not None:
            out["description"] = self.description
        if self.install_hint is not None:
            out["install_hint"] = self.install_hint
        if self.config_hint is not None:
            out["config_hint"] = self.config_hint
        return out


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
        self._host_context: Optional[CapabilityHostContext] = None

    @property
    def host_context(self) -> CapabilityHostContext:
        if self._host_context is None:
            self._host_context = DefaultCapabilityHostContext(self._owner)
        return self._host_context

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
            return 'pip install "abstractcore[voice]"'
        if cap == "vision":
            return 'pip install "abstractcore[vision]"'
        if cap == "music":
            return 'pip install "abstractcore[music]"'
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
        return self._get_instance_for_backend_id(capability, bid)

    def _get_instance_for_backend_id(self, capability: str, backend_id: str) -> Any:
        """Return (and cache) a backend instance for a specific backend_id.

        This is used for request-scoped selection (especially music) without
        mutating the registry's process-wide preferred backend.
        """
        self._ensure_plugins_loaded()
        cap = str(capability or "").strip()
        bid = str(backend_id or "").strip()
        if not cap or not bid:
            raise CapabilityUnavailableError(
                capability=cap or "unknown",
                reason="Capability backend_id is required.",
                install_hint=self._default_install_hint(cap or "unknown"),
                details={"backend_id": bid},
            )

        regs = self._registrations.get(cap) or {}
        if bid not in regs:
            raise CapabilityUnavailableError(
                capability=cap,
                reason=f"Requested backend '{bid}' is not registered.",
                install_hint=self._default_install_hint(cap),
                details={"backend_id": bid},
            )

        key = (capability, bid)
        if key in self._instances:
            return self._instances[key]

        reg = regs.get(bid)
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

    def _is_backend_registered(self, capability: str, backend_id: str) -> bool:
        self._ensure_plugins_loaded()
        cap = str(capability or "").strip()
        bid = str(backend_id or "").strip()
        if not cap or not bid:
            return False
        return bid in (self._registrations.get(cap) or {})

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

    def list_backend_infos(self, capability: Optional[str] = None) -> List[CapabilityBackendInfo]:
        """Return registered backend metadata without instantiating backend factories."""
        self._ensure_plugins_loaded()
        cap_filter = str(capability or "").strip() or None
        out: List[CapabilityBackendInfo] = []
        for cap, regs in sorted(self._registrations.items()):
            if cap_filter is not None and cap != cap_filter:
                continue
            out.extend(reg.info for reg in regs.values())
        return sorted(out, key=lambda info: (info.capability, -int(info.priority), info.backend_id))

    def list_capabilities(self) -> Dict[str, Any]:
        return dict(self.status().get("capabilities") or {})

    def available_providers(self, capability: str, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        cap = str(capability or "").strip()
        backend = self._get_instance(cap)
        method = getattr(backend, "available_providers", None)
        if not callable(method):
            method = getattr(backend, "list_available_providers", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability=cap,
                reason=f"The selected {cap} capability backend does not expose available_providers(task=...).",
                install_hint=self._default_install_hint(cap),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        raw = _call_capability_discovery_method(method, task=task, kind=task)
        return [item.to_dict() for item in _normalize_provider_records(raw, capability=cap, task=task, backend_id=getattr(backend, "backend_id", None))]

    def list_models(
        self,
        capability: str,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cap = str(capability or "").strip()
        backend = self._get_instance(cap)
        method = getattr(backend, "list_models", None)
        if not callable(method):
            method = getattr(backend, "list_provider_models", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability=cap,
                reason=f"The selected {cap} capability backend does not expose list_models(task=..., provider=...).",
                install_hint=self._default_install_hint(cap),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        provider_s = provider_id or provider
        raw = _call_capability_discovery_method(method, task=task, kind=task, provider=provider_s, provider_id=provider_s)
        return [
            item.to_dict()
            for item in _normalize_model_records(
                raw,
                capability=cap,
                task=task,
                provider_id=provider_s,
                backend_id=getattr(backend, "backend_id", None),
            )
        ]

    def list_operations(self, capability: str, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        cap = str(capability or "").strip()
        backend = self._get_instance(cap)
        method = getattr(backend, "list_operations", None)
        if not callable(method):
            method = getattr(backend, "get_capabilities", None)
        if not callable(method):
            return []
        raw = _call_capability_discovery_method(method, task=task, kind=task)
        return [
            item.to_dict()
            for item in _normalize_operation_records(
                raw,
                capability=cap,
                task=task,
            )
        ]

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


def _call_residency_mapping_method(
    backend: Any,
    capability: str,
    method_names: Tuple[str, ...],
    request: Mapping[str, Any],
    *,
    install_hint: Optional[str],
) -> Dict[str, Any]:
    for method_name in method_names:
        method = getattr(backend, method_name, None)
        if callable(method):
            out = method(dict(request or {}))
            if not isinstance(out, dict):
                raise CapabilityUnavailableError(
                    capability=capability,
                    reason=f"The selected {capability} capability backend returned a non-object residency payload.",
                    install_hint=install_hint,
                    details={"backend_id": getattr(backend, "backend_id", None), "method": method_name},
                )
            return dict(out)
    raise CapabilityUnavailableError(
        capability=capability,
        reason=(
            f"The selected {capability} capability backend does not expose "
            "load_resident_model/list_resident_models/unload_resident_model."
        ),
        install_hint=install_hint,
        details={"backend_id": getattr(backend, "backend_id", None), "methods": list(method_names)},
    )


def _call_residency_list_method(
    backend: Any,
    capability: str,
    method_names: Tuple[str, ...],
    filters: Optional[Mapping[str, Any]],
    *,
    install_hint: Optional[str],
) -> List[Dict[str, Any]]:
    for method_name in method_names:
        method = getattr(backend, method_name, None)
        if callable(method):
            out = method(dict(filters or {}))
            if not isinstance(out, list):
                raise CapabilityUnavailableError(
                    capability=capability,
                    reason=f"The selected {capability} capability backend returned a non-list residency payload.",
                    install_hint=install_hint,
                    details={"backend_id": getattr(backend, "backend_id", None), "method": method_name},
                )
            return [dict(item) for item in out if isinstance(item, dict)]
    raise CapabilityUnavailableError(
        capability=capability,
        reason=(
            f"The selected {capability} capability backend does not expose "
            "list_loaded_models/list_resident_models."
        ),
        install_hint=install_hint,
        details={"backend_id": getattr(backend, "backend_id", None), "methods": list(method_names)},
    )


def _call_capability_discovery_method(
    method: Callable[..., Any],
    *,
    kind: Optional[str] = None,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    provider_id: Optional[str] = None,
) -> Any:
    params: Dict[str, inspect.Parameter] = {}
    accepts_kwargs = False
    try:
        params = dict(inspect.signature(method).parameters)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    except (TypeError, ValueError):
        accepts_kwargs = True

    kwargs: Dict[str, Any] = {}
    if kind is not None and (accepts_kwargs or "kind" in params):
        kwargs["kind"] = str(kind or "")
    if task is not None and (accepts_kwargs or "task" in params):
        kwargs["task"] = str(task or "")
    if provider is not None and (accepts_kwargs or "provider" in params):
        kwargs["provider"] = provider
    if provider_id is not None and (accepts_kwargs or "provider_id" in params):
        kwargs["provider_id"] = provider_id
    if kwargs:
        return method(**kwargs)
    return method()


def _list_from_payload(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, dict):
        for key in ("providers", "models", "operations", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return list(value)
            if isinstance(value, dict):
                return [dict(v, provider_id=k) if isinstance(v, dict) else {"provider_id": k, "value": v} for k, v in value.items()]
        return [dict(v, provider_id=k) if isinstance(v, dict) else {"provider_id": k, "value": v} for k, v in raw.items()]
    return [raw]


def _str_list(value: Any, fallback: Optional[str] = None) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v or "").strip()]
    if fallback:
        return [fallback]
    return []


def _normalize_provider_records(
    raw: Any,
    *,
    capability: str,
    task: Optional[str],
    backend_id: Any,
) -> List[CapabilityProviderInfo]:
    out: List[CapabilityProviderInfo] = []
    for item in _list_from_payload(raw):
        if isinstance(item, CapabilityProviderInfo):
            out.append(item)
            continue
        if isinstance(item, str):
            provider_id = item.strip()
            data: Dict[str, Any] = {}
        elif isinstance(item, Mapping):
            data = dict(item)
            provider_id = str(
                data.get("provider_id")
                or data.get("provider")
                or data.get("id")
                or data.get("name")
                or ""
            ).strip()
        else:
            continue
        if not provider_id:
            continue
        tasks = _str_list(data.get("tasks"), task)
        display = str(data.get("display_name") or data.get("name") or provider_id)
        status = str(data.get("status") or "available")
        metadata = dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {}
        local = bool(data.get("local"))
        remote = bool(data.get("remote"))
        backend_id_s = str(data.get("backend_id") or backend_id or "").strip() or None
        out.append(
            CapabilityProviderInfo(
                provider_id=provider_id,
                display_name=display,
                capability=capability,
                tasks=tasks,
                local=local,
                remote=remote,
                status=status,
                backend_id=backend_id_s,
                installed=data.get("installed") if isinstance(data.get("installed"), bool) else None,
                configured=data.get("configured") if isinstance(data.get("configured"), bool) else None,
                reachable=data.get("reachable") if isinstance(data.get("reachable"), bool) else None,
                selected=data.get("selected") if isinstance(data.get("selected"), bool) else None,
                install_hint=data.get("install_hint") if isinstance(data.get("install_hint"), str) else None,
                config_hint=data.get("config_hint") if isinstance(data.get("config_hint"), str) else None,
                metadata=metadata,
            )
        )
    return out


def _normalize_model_records(
    raw: Any,
    *,
    capability: str,
    task: Optional[str],
    provider_id: Optional[str],
    backend_id: Any,
) -> List[CapabilityModelInfo]:
    out: List[CapabilityModelInfo] = []
    for item in _list_from_payload(raw):
        if isinstance(item, CapabilityModelInfo):
            out.append(item)
            continue
        if isinstance(item, str):
            model_id = item.strip()
            data: Dict[str, Any] = {}
        elif isinstance(item, Mapping):
            data = dict(item)
            model_id = str(data.get("model_id") or data.get("model") or data.get("id") or data.get("name") or "").strip()
        else:
            continue
        if not model_id:
            continue
        provider_s = str(data.get("provider_id") or data.get("provider") or provider_id or "").strip()
        out.append(
            CapabilityModelInfo(
                model_id=model_id,
                provider_id=provider_s,
                capability=capability,
                tasks=_str_list(data.get("tasks"), task),
                modalities=_str_list(data.get("modalities")),
                local=bool(data.get("local")),
                remote=bool(data.get("remote")),
                status=str(data.get("status") or "available"),
                backend_id=str(data.get("backend_id") or backend_id or "") or None,
                routed_model=data.get("routed_model") if isinstance(data.get("routed_model"), str) else None,
                formats=_str_list(data.get("formats")),
                source=data.get("source") if isinstance(data.get("source"), str) else None,
                recommended=data.get("recommended") if isinstance(data.get("recommended"), bool) else None,
                license=data.get("license") if isinstance(data.get("license"), str) else None,
                commercial_allowed=data.get("commercial_allowed") if isinstance(data.get("commercial_allowed"), bool) else None,
                raw_metadata=dict(data.get("raw_metadata") or data.get("metadata") or {})
                if isinstance(data.get("raw_metadata") or data.get("metadata"), dict)
                else {},
            )
        )
    return out


def _normalize_operation_records(raw: Any, *, capability: str, task: Optional[str]) -> List[CapabilityOperationInfo]:
    out: List[CapabilityOperationInfo] = []
    for item in _list_from_payload(raw):
        if isinstance(item, CapabilityOperationInfo):
            out.append(item)
            continue
        if isinstance(item, str):
            operation_id = item.strip()
            data: Dict[str, Any] = {}
        elif isinstance(item, Mapping):
            data = dict(item)
            operation_id = str(data.get("operation_id") or data.get("operation") or data.get("task") or data.get("id") or "").strip()
        else:
            continue
        if not operation_id:
            continue
        out.append(
            CapabilityOperationInfo(
                operation_id=operation_id,
                capability=str(data.get("capability") or capability),
                task=str(data.get("task") or task or operation_id),
                input_modalities=_str_list(data.get("input_modalities")),
                output_modalities=_str_list(data.get("output_modalities")),
                parameter_schema=dict(data.get("parameter_schema")) if isinstance(data.get("parameter_schema"), dict) else None,
                required_parameters=_str_list(data.get("required_parameters")),
                artifact_output=data.get("artifact_output") if isinstance(data.get("artifact_output"), bool) else None,
                metadata=dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {},
            )
        )
    return out


def _normalized_provider_id(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _provider_ids_match(left: Any, right: Any) -> bool:
    return bool(_normalized_provider_id(left)) and _normalized_provider_id(left) == _normalized_provider_id(right)


def _clone_models_from_voice_catalog(catalog: Any, *, provider: Optional[str] = None) -> List[str]:
    if not isinstance(catalog, dict):
        return []

    mapping = catalog.get("cloning_models_by_provider")
    if isinstance(mapping, dict):
        out: List[str] = []
        for provider_id, values in mapping.items():
            if provider is not None and not _provider_ids_match(provider_id, provider):
                continue
            if isinstance(values, list):
                out.extend(str(item).strip() for item in values if isinstance(item, str) and str(item).strip())
        return list(dict.fromkeys(out))

    compat = catalog.get("compatibility_catalog")
    if not isinstance(compat, dict):
        return []
    providers = compat.get("providers")
    if not isinstance(providers, dict):
        return []
    cloning = providers.get("cloning")
    if not isinstance(cloning, dict):
        return []

    out: List[str] = []
    for provider_id, entry in cloning.items():
        if provider is not None and not _provider_ids_match(provider_id, provider):
            continue
        models = entry.get("models") if isinstance(entry, dict) else None
        if not isinstance(models, dict):
            continue
        for model_name in models:
            text = str(model_name or "").strip()
            if text and text != "*":
                out.append(text)
    return list(dict.fromkeys(out))


class _VoiceFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @property
    def backend_id(self) -> Optional[str]:
        try:
            return str(getattr(self._registry.get_voice(), "backend_id"))
        except Exception:
            return None

    def tts(self, text: str, **kwargs: Any) -> Any:
        return self._registry.get_voice().tts(text, **kwargs)

    def stt(self, audio: Any, **kwargs: Any) -> Any:
        return self._registry.get_voice().stt(audio, **kwargs)

    def load_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return _call_residency_mapping_method(
            self._registry.get_voice(),
            "voice",
            ("load_resident_model", "load_model"),
            request,
            install_hint=self._registry._default_install_hint("voice"),
        )

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        return _call_residency_list_method(
            self._registry.get_voice(),
            "voice",
            ("list_loaded_models", "list_resident_models"),
            filters,
            install_hint=self._registry._default_install_hint("voice"),
        )

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        return _call_residency_list_method(
            self._registry.get_voice(),
            "voice",
            ("list_resident_models",),
            filters,
            install_hint=self._registry._default_install_hint("voice"),
        )

    def unload_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return _call_residency_mapping_method(
            self._registry.get_voice(),
            "voice",
            ("unload_resident_model", "unload_model"),
            request,
            install_hint=self._registry._default_install_hint("voice"),
        )

    def list_profiles(self, *, kind: str = "tts") -> List[Dict[str, Any]]:
        backend = self._registry.get_voice()
        method = getattr(backend, "list_profiles", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend does not expose list_profiles(kind=...).",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        out = method(kind=str(kind or "tts"))
        return list(out or [])

    def list_models(self, *, kind: str = "tts", provider: Optional[str] = None) -> List[str]:
        backend = self._registry.get_voice()
        normalized_kind = str(kind or "tts").strip().lower() or "tts"
        method = getattr(backend, "list_models", None)
        if callable(method):
            out = _call_capability_discovery_method(method, kind=normalized_kind, provider=provider)
            return [str(item) for item in list(out or []) if str(item or "").strip()]
        if normalized_kind == "tts":
            return self.list_tts_models(provider=provider)
        if normalized_kind == "stt":
            return self.list_stt_models(provider=provider)
        if normalized_kind == "cloning":
            return self.list_cloning_models(provider=provider)
        raise CapabilityUnavailableError(
            capability="voice",
            reason=f"The selected voice capability backend does not expose list_models(kind={normalized_kind!r}, provider=...).",
            install_hint=self._registry._default_install_hint("voice"),
            details={"backend_id": getattr(backend, "backend_id", None), "kind": normalized_kind, "provider": provider},
        )

    def list_tts_models(self, provider: Optional[str] = None) -> List[str]:
        backend = self._registry.get_voice()
        method = getattr(backend, "list_tts_models", None)
        if not callable(method):
            generic = getattr(backend, "list_models", None)
            if callable(generic):
                out = _call_capability_discovery_method(generic, kind="tts", provider=provider)
                return [str(item) for item in list(out or []) if str(item or "").strip()]
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend does not expose list_tts_models(provider=...).",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None), "provider": provider},
            )
        out = _call_capability_discovery_method(method, provider=provider)
        return [str(item) for item in list(out or []) if str(item or "").strip()]

    def list_stt_models(self, provider: Optional[str] = None) -> List[str]:
        backend = self._registry.get_voice()
        method = getattr(backend, "list_stt_models", None)
        if not callable(method):
            generic = getattr(backend, "list_models", None)
            if callable(generic):
                out = _call_capability_discovery_method(generic, kind="stt", provider=provider)
                return [str(item) for item in list(out or []) if str(item or "").strip()]
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend does not expose list_stt_models(provider=...).",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None), "provider": provider},
            )
        out = _call_capability_discovery_method(method, provider=provider)
        return [str(item) for item in list(out or []) if str(item or "").strip()]

    def list_cloning_models(self, provider: Optional[str] = None) -> List[str]:
        backend = self._registry.get_voice()
        method = getattr(backend, "list_cloning_models", None)
        if callable(method):
            out = _call_capability_discovery_method(method, provider=provider)
            return [str(item) for item in list(out or []) if str(item or "").strip()]

        generic = getattr(backend, "list_models", None)
        if callable(generic):
            out = _call_capability_discovery_method(generic, kind="cloning", provider=provider)
            return [str(item) for item in list(out or []) if str(item or "").strip()]

        catalog_method = getattr(backend, "voice_catalog", None)
        if callable(catalog_method):
            models = _clone_models_from_voice_catalog(catalog_method(), provider=provider)
            if models:
                return models

        raise CapabilityUnavailableError(
            capability="voice",
            reason="The selected voice capability backend does not expose clone model discovery.",
            install_hint=self._registry._default_install_hint("voice"),
            details={"backend_id": getattr(backend, "backend_id", None), "provider": provider},
        )

    def voice_catalog(self) -> Dict[str, Any]:
        backend = self._registry.get_voice()
        method = getattr(backend, "voice_catalog", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend does not expose voice_catalog().",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        out = method()
        if not isinstance(out, dict):
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend returned a non-object voice catalog.",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        return dict(out)

    def available_providers(self) -> Dict[str, Any]:
        """Return lightweight provider availability for voice backends.

        This is intended for UI/catalog routes; implementations must avoid
        constructing heavy runtimes.
        """
        backend = self._registry.get_voice()
        method = getattr(backend, "available_providers", None)
        if not callable(method):
            method = getattr(backend, "list_available_providers", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend does not expose available_providers().",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        out = method()
        if not isinstance(out, dict):
            raise CapabilityUnavailableError(
                capability="voice",
                reason="The selected voice capability backend returned a non-object provider availability payload.",
                install_hint=self._registry._default_install_hint("voice"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        return dict(out)

    def clone(self, audio: Any, **kwargs: Any) -> Any:
        backend = self._registry.get_voice()
        for method_name in ("clone", "clone_voice"):
            method = getattr(backend, method_name, None)
            if callable(method):
                return method(audio, **kwargs)
        raise CapabilityUnavailableError(
            capability="voice",
            reason=(
                "voice.clone is not exposed by the selected voice capability backend. "
                "Use AbstractCore Server's /v1/voice/clone route or an AbstractVoice version/backend "
                "that implements voice cloning."
            ),
            install_hint=self._registry._default_install_hint("voice"),
        )


class _AudioFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @property
    def backend_id(self) -> Optional[str]:
        try:
            return str(getattr(self._registry.get_audio(), "backend_id"))
        except Exception:
            return None

    def transcribe(self, audio: Any, **kwargs: Any) -> Any:
        return self._registry.get_audio().transcribe(audio, **kwargs)

    def load_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return _call_residency_mapping_method(
            self._registry.get_audio(),
            "audio",
            ("load_resident_model", "load_model"),
            request,
            install_hint=self._registry._default_install_hint("audio"),
        )

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        return _call_residency_list_method(
            self._registry.get_audio(),
            "audio",
            ("list_loaded_models", "list_resident_models"),
            filters,
            install_hint=self._registry._default_install_hint("audio"),
        )

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        return _call_residency_list_method(
            self._registry.get_audio(),
            "audio",
            ("list_resident_models",),
            filters,
            install_hint=self._registry._default_install_hint("audio"),
        )

    def unload_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return _call_residency_mapping_method(
            self._registry.get_audio(),
            "audio",
            ("unload_resident_model", "unload_model"),
            request,
            install_hint=self._registry._default_install_hint("audio"),
        )


class _VisionFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @property
    def backend_id(self) -> Optional[str]:
        try:
            return str(getattr(self._registry.get_vision(), "backend_id"))
        except Exception:
            return None

    def t2i(self, prompt: str, **kwargs: Any) -> Any:
        return self._registry.get_vision().t2i(prompt, **kwargs)

    def load_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return _call_residency_mapping_method(
            self._registry.get_vision(),
            "vision",
            ("load_resident_model", "load_model"),
            request,
            install_hint=self._registry._default_install_hint("vision"),
        )

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        return _call_residency_list_method(
            self._registry.get_vision(),
            "vision",
            ("list_loaded_models", "list_resident_models"),
            filters,
            install_hint=self._registry._default_install_hint("vision"),
        )

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        return _call_residency_list_method(
            self._registry.get_vision(),
            "vision",
            ("list_resident_models",),
            filters,
            install_hint=self._registry._default_install_hint("vision"),
        )

    def unload_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return _call_residency_mapping_method(
            self._registry.get_vision(),
            "vision",
            ("unload_resident_model", "unload_model"),
            request,
            install_hint=self._registry._default_install_hint("vision"),
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        backend = self._registry.get_vision()
        method = getattr(backend, "list_provider_models", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability="vision",
                reason="The selected vision capability backend does not expose list_provider_models(task=...).",
                install_hint=self._registry._default_install_hint("vision"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        out = method(task=task)
        return list(out or [])

    def available_providers(self, *, task: Optional[str] = None) -> Dict[str, Any]:
        """Return lightweight provider availability for vision backends."""
        backend = self._registry.get_vision()
        method = getattr(backend, "available_providers", None)
        if not callable(method):
            method = getattr(backend, "list_available_providers", None)
        if not callable(method):
            raise CapabilityUnavailableError(
                capability="vision",
                reason="The selected vision capability backend does not expose available_providers(task=...).",
                install_hint=self._registry._default_install_hint("vision"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        out = method(task=task)
        if not isinstance(out, dict):
            raise CapabilityUnavailableError(
                capability="vision",
                reason="The selected vision capability backend returned a non-object provider availability payload.",
                install_hint=self._registry._default_install_hint("vision"),
                details={"backend_id": getattr(backend, "backend_id", None)},
            )
        return dict(out)

    def i2i(self, prompt: str, image: Any, **kwargs: Any) -> Any:
        return self._registry.get_vision().i2i(prompt, image, **kwargs)

    def t2v(self, prompt: str, **kwargs: Any) -> Any:
        return self._registry.get_vision().t2v(prompt, **kwargs)

    def i2v(self, image: Any, **kwargs: Any) -> Any:
        return self._registry.get_vision().i2v(image, **kwargs)


class _MusicFacade:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @property
    def backend_id(self) -> Optional[str]:
        try:
            return str(getattr(self._registry.get_music(), "backend_id"))
        except Exception:
            return None

    def available_providers(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._registry.available_providers("music", task=task)

    def list_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._registry.list_models("music", task=task, provider=provider, provider_id=provider_id)

    def list_provider_models(
        self,
        *,
        task: Optional[str] = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.list_models(task=task, provider=provider, provider_id=provider_id)

    def list_operations(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._registry.list_operations("music", task=task)

    def capability_catalog(self, *, task: Optional[str] = None) -> Dict[str, Any]:
        return {
            "capability": "music",
            "backend_id": self.backend_id,
            "task": task,
            "providers": self.available_providers(task=task),
            "models": self.list_models(task=task),
            "operations": self.list_operations(task=task),
        }

    def t2m(self, prompt: str, **kwargs: Any) -> Any:
        return self._registry.get_music().t2m(prompt, **kwargs)

    def generate(self, prompt: str, *, task: Optional[str] = None, **kwargs: Any) -> Any:
        """Generate music/audio through the selected music capability backend.

        `t2m(...)` remains the minimal plugin contract. This alias gives Python
        callers the same `generate(...)` shape used by other Core surfaces while
        keeping backend implementations modality-specific.
        """
        normalized_task = str(task or kwargs.pop("operation", "") or "text_to_music").strip().lower()
        normalized_task = normalized_task.replace("-", "_")
        if normalized_task in {"", "music", "song", "t2m"}:
            normalized_task = "text_to_music"
        if normalized_task not in {"text_to_music", "music_generation", "lyrics_to_music", "text_to_audio"}:
            raise CapabilityUnavailableError(
                capability="music",
                reason=f"Unsupported music generation task: {normalized_task!r}.",
                install_hint=self._registry._default_install_hint("music"),
                details={"backend_id": self.backend_id, "task": normalized_task},
            )

        # Request-scoped backend selection (do not silently fall back).
        #
        # Prefer explicit `backend` / `music_backend`. If absent, also accept
        # `provider` when it matches a known backend alias (Gateway historically
        # used provider-like names such as "ace-step").
        from .music_selectors import resolve_music_backend_id, resolve_music_provider_hint

        backend_selector = kwargs.get("backend")
        if backend_selector is None:
            backend_selector = kwargs.get("music_backend")

        provider_selector = kwargs.get("provider")
        provider_hint = resolve_music_provider_hint(provider_selector)

        requested_backend_id: Optional[str] = None
        consumed_provider_alias = False

        if backend_selector is not None:
            raw = str(backend_selector).strip()
            if raw:
                resolved = resolve_music_backend_id(raw)
                candidate = resolved or raw
                alt = raw.lower().replace("_", "-")
                if (
                    not self._registry._is_backend_registered("music", candidate)
                    and self._registry._is_backend_registered("music", alt)
                ):
                    candidate = alt
                if not self._registry._is_backend_registered("music", candidate):
                    raise CapabilityUnavailableError(
                        capability="music",
                        reason=f"Unknown music backend selector: {raw!r}.",
                        install_hint=self._registry._default_install_hint("music"),
                        details={"backend": raw},
                    )
                requested_backend_id = candidate

        if requested_backend_id is None:
            raw_provider = str(provider_selector).strip() if isinstance(provider_selector, str) else ""
            resolved = resolve_music_backend_id(raw_provider)
            if resolved and self._registry._is_backend_registered("music", resolved):
                requested_backend_id = resolved
                consumed_provider_alias = True

        if requested_backend_id is not None:
            backend = self._registry._get_instance_for_backend_id("music", requested_backend_id)
        else:
            backend = self._registry.get_music()

        # Do not forward routing-only selectors into the backend call.
        call_kwargs = dict(kwargs)
        call_kwargs.pop("backend", None)
        call_kwargs.pop("music_backend", None)
        if consumed_provider_alias:
            call_kwargs.pop("provider", None)

        # Fail early on known backend/model mismatches to avoid confusing
        # backend-specific runtime errors.
        model_id = call_kwargs.get("model") or call_kwargs.get("model_id") or call_kwargs.get("music_model_id")
        target_backend_id = requested_backend_id
        if target_backend_id is None:
            try:
                bid = getattr(backend, "backend_id", None)
            except Exception:
                bid = None
            if isinstance(bid, str) and bid.strip():
                target_backend_id = bid.strip()

        if isinstance(model_id, str) and model_id.strip() and isinstance(target_backend_id, str) and target_backend_id:
            model_s = model_id.strip().lower()
            if target_backend_id == "abstractmusic:stable-audio" and "stable-audio-3" in model_s:
                raise CapabilityUnavailableError(
                    capability="music",
                    reason=(
                        "Requested backend 'stable-audio' cannot run Stable Audio 3 models. "
                        "Use backend='stable-audio-3' for stabilityai/stable-audio-3-*."
                    ),
                    install_hint=self._registry._default_install_hint("music"),
                    details={"backend_id": target_backend_id, "model": model_id},
                )
            if target_backend_id == "abstractmusic:stable-audio-3" and "stable-audio-open" in model_s:
                raise CapabilityUnavailableError(
                    capability="music",
                    reason=(
                        "Requested backend 'stable-audio-3' cannot run Stable Audio Open models. "
                        "Use backend='stable-audio' for stabilityai/stable-audio-open-*."
                    ),
                    install_hint=self._registry._default_install_hint("music"),
                    details={"backend_id": target_backend_id, "model": model_id},
                )

        method = getattr(backend, "generate", None)
        if callable(method):
            out = method(prompt, task=normalized_task, **call_kwargs)
        else:
            out = backend.t2m(prompt, **call_kwargs)

        # Ensure truthful backend reporting without requiring every plugin to
        # remember to include it.
        if requested_backend_id is not None:
            if isinstance(out, dict):
                out = dict(out)
                out.setdefault("backend_id", requested_backend_id)
            else:
                out = {"data": out, "backend_id": requested_backend_id}
        return out
