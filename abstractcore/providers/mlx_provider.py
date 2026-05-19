"""
MLX provider implementation for Apple Silicon.
"""

import json
import time
import uuid
import inspect
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Iterator, Type, TYPE_CHECKING

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None

# Try to import Outlines (native structured output for MLX models)
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False

from .base import BaseProvider, ThinkingControlHandling
from ..core.types import GenerateResponse
from ..exceptions import ProviderAPIError, ModelNotFoundError, format_model_error
from ..tools import UniversalToolHandler, execute_tools
from ..events import EventType

if TYPE_CHECKING:
    from ..media.types import MediaContent


class MLXProvider(BaseProvider):
    """MLX provider for Apple Silicon models with full integration"""

    def __init__(self, model: str = "mlx-community/Mistral-7B-Instruct-v0.1-4bit",
                 structured_output_method: str = "auto", **kwargs):
        super().__init__(model, **kwargs)
        self.provider = "mlx"

        # Handle timeout parameter for local models
        self._handle_timeout_parameter(kwargs)

        # Structured output method: "auto", "native_outlines", "prompted"
        # auto: Use Outlines if available, otherwise prompted (default)
        # native_outlines: Force Outlines (error if unavailable)
        # prompted: Always use prompted fallback (fastest, still 100% success)
        self.structured_output_method = structured_output_method

        # Initialize tool handler
        self.tool_handler = UniversalToolHandler(model)

        self.llm = None
        self.tokenizer = None
        self._resolved_model_id: Optional[str] = None
        self._load_model()

    def supports_prompt_cache(self) -> bool:
        """MLX supports KV prompt caches via `mlx_lm.models.cache`."""
        return True

    def prompt_cache_supports_kv_source_of_truth(self) -> bool:
        """MLX KV caches are mutable and can serve as the context source-of-truth."""
        return True

    def _apply_provider_thinking_kwargs(self, *, enabled, level=None, kwargs: Dict[str, Any]):
        """Map unified thinking control into MLX prompt serialization state.

        mlx-lm local generation takes an already-serialized prompt. For Qwen reasoning
        templates, the robust local disable control is to serialize the assistant
        generation prompt in no-thinking mode (`<think>\n\n</think>\n\n`) before
        generation. The actual serialization happens in `_build_prompt_fragment`.
        """
        new_kwargs = dict(kwargs or {})
        if self.architecture in {"qwen3", "qwen3_5", "qwen3_6"}:
            if enabled is False:
                new_kwargs["_acore_mlx_enable_thinking"] = False
                return new_kwargs, ThinkingControlHandling(
                    handled_enable_disable=True,
                    handled_level=False,
                )
            if enabled is True or level is not None:
                new_kwargs["_acore_mlx_enable_thinking"] = True
                return new_kwargs, ThinkingControlHandling(
                    handled_enable_disable=True,
                    handled_level=False,
                )
        return new_kwargs, ThinkingControlHandling()

    def _prompt_cache_backend_create(self) -> Optional[Any]:
        try:
            from mlx_lm.models.cache import make_prompt_cache
        except Exception:
            return None
        try:
            return make_prompt_cache(self.llm)
        except Exception:
            return None

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Optional[Any]:
        """Best-effort deep clone of an MLX prompt cache."""
        if cache_value is None:
            return None

        def _clone_layer(layer: Any) -> Any:
            from_state = getattr(layer.__class__, "from_state", None)
            state_attr: Any = None
            if callable(from_state):
                try:
                    state_attr = getattr(layer, "state", None)
                except Exception:
                    state_attr = None
            if callable(from_state):
                try:
                    state_val = state_attr() if callable(state_attr) else state_attr
                    meta_attr = getattr(layer, "meta_state", None)
                    meta_val = meta_attr() if callable(meta_attr) else meta_attr
                    if state_val is not None:
                        try:
                            sig = inspect.signature(from_state)
                            if len(sig.parameters) == 2:
                                return from_state(state_val, meta_val)
                            if len(sig.parameters) == 1:
                                return from_state(state_val)
                        except Exception:
                            pass

                        # Fallback: try the common 2-arg then 1-arg patterns.
                        try:
                            return from_state(state_val, meta_val)
                        except TypeError:
                            return from_state(state_val)

                    # Some MLX-LM cache layers (notably KVCache) cannot serialize an "empty" state.
                    # Fall back to constructing a new empty instance when state is unavailable.
                    try:
                        empty = layer.__class__()  # type: ignore[call-arg]
                        try:
                            if meta_val is not None and hasattr(empty, "meta_state"):
                                empty.meta_state = meta_val  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        return empty
                    except Exception:
                        return None
                except Exception:
                    return None
            if hasattr(layer, "copy"):
                try:
                    return layer.copy()
                except Exception:
                    return None
            return None

        # MLX-LM prompt caches are typically a list of per-layer KVCache objects.
        if isinstance(cache_value, list):
            cloned: List[Any] = []
            for layer in cache_value:
                c = _clone_layer(layer)
                if c is None:
                    return None
                cloned.append(c)
            return cloned

        if isinstance(cache_value, tuple):
            cloned_layers: List[Any] = []
            for layer in cache_value:
                c = _clone_layer(layer)
                if c is None:
                    return None
                cloned_layers.append(c)
            return tuple(cloned_layers)

        # Fallback: single cache object.
        return _clone_layer(cache_value)

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        if cache_value is None:
            return 0
        try:
            if isinstance(cache_value, (list, tuple)):
                for layer in cache_value:
                    if hasattr(layer, "size"):
                        try:
                            s = int(layer.size())
                        except Exception:
                            s = None
                        if isinstance(s, int) and s > 0:
                            return s
                    if hasattr(layer, "offset"):
                        try:
                            off = int(getattr(layer, "offset", 0))
                        except Exception:
                            off = 0
                        if off > 0:
                            return off
                return 0
        except Exception:
            pass
        return None

    def _build_prompt_fragment(
        self,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        prefilled_modules: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Build a prompt fragment intended to be appended to an existing prompt_cache."""

        prefilled = set()
        if prefilled_modules:
            for item in prefilled_modules:
                try:
                    norm = str(item or "").strip().lower()
                except Exception:
                    norm = ""
                if norm:
                    prefilled.add(norm)

        base_system_prompt = system_prompt
        tool_system_prompt = None
        if tools and self.tool_handler.supports_prompted and "tools" not in prefilled:
            include_tool_list = True
            if base_system_prompt and "## Tools (session)" in base_system_prompt:
                include_tool_list = False
            tool_prompt = self.tool_handler.format_tools_prompt(tools, include_tool_list=include_tool_list)
            if tool_prompt:
                tool_system_prompt = tool_prompt

        def _as_text(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        is_qwen = "qwen" in self.model.lower()
        parts: List[str] = []

        if base_system_prompt and "system" not in prefilled:
            if is_qwen:
                parts.append(f"<|im_start|>system\n{base_system_prompt}<|im_end|>\n")
            else:
                parts.append(f"{base_system_prompt.strip()}\n\n")

        if tool_system_prompt:
            if is_qwen:
                parts.append(f"<|im_start|>system\n{tool_system_prompt}<|im_end|>\n")
            else:
                parts.append(f"{tool_system_prompt.strip()}\n\n")

        if messages:
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "user")
                content = _as_text(msg.get("content"))
                if is_qwen:
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
                else:
                    parts.append(f"{role}: {content}\n")

        if isinstance(prompt, str) and prompt:
            if is_qwen:
                parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
            else:
                parts.append(f"user: {prompt}\n")

        if add_generation_prompt:
            if is_qwen:
                parts.append("<|im_start|>assistant\n")
                if enable_thinking is False:
                    parts.append("<think>\n\n</think>\n\n")
            else:
                parts.append("assistant:")

        return "".join(parts)

    def _prompt_cache_backend_append(
        self,
        cache_value: Any,
        *,
        prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> bool:
        _ = kwargs
        if cache_value is None:
            return False

        fragment = self._build_prompt_fragment(
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=bool(add_generation_prompt),
            enable_thinking=kwargs.get("_acore_mlx_enable_thinking"),
        )
        if not fragment:
            return True

        try:
            from mlx_lm.models.cache import trim_prompt_cache
        except Exception:
            trim_prompt_cache = None

        # Best-effort prefill: MLX-LM generates at least one token; trim it to end exactly at the fragment boundary.
        generated = 0
        try:
            gen = self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt=fragment,
                prompt_cache=cache_value,
                max_tokens=1,
            )
            for _chunk in gen:
                generated += 1
        except TypeError:
            try:
                gen = self.stream_generate_fn(
                    self.llm,
                    self.tokenizer,
                    fragment,
                    prompt_cache=cache_value,
                    max_tokens=1,
                )
                for _chunk in gen:
                    generated += 1
            except Exception:
                return False
        except Exception:
            return False

        if trim_prompt_cache is not None and generated > 0:
            try:
                trim_prompt_cache(cache_value, generated)
            except Exception:
                pass

        return True

    def prompt_cache_set(
        self,
        key: str,
        *,
        make_default: bool = True,
        warm_prompt: Optional[str] = None,
        ttl_s: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Create/reset a prompt cache for the given key (best-effort)."""
        _ = kwargs
        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            return False
        if not super().prompt_cache_set(normalized, make_default=make_default):
            return False

        try:
            from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        except Exception:
            return False

        cache_obj = make_prompt_cache(self.llm)

        # Best-effort warm: MLX-LM always generates at least 1 token, so we trim it back.
        if isinstance(warm_prompt, str) and warm_prompt.strip():
            try:
                gen = self.stream_generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt=warm_prompt,
                    prompt_cache=cache_obj,
                    max_tokens=1,
                )
                for _ in gen:
                    break
                try:
                    trim_prompt_cache(cache_obj, 1)
                except Exception:
                    pass
            except Exception:
                pass

        try:
            self._prompt_cache_store.set(normalized, cache_obj, ttl_s=ttl_s, meta={"backend": "mlx"})
        except Exception:
            return False
        return True

    def prompt_cache_save(
        self,
        key: str,
        filename: str,
        *,
        q8: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Save an MLX KV prompt cache to a `.safetensors` file (model-locked; best-effort)."""
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        normalized = self._normalize_prompt_cache_key(key)
        if normalized is None:
            raise ValueError("prompt cache key must be a non-empty string")

        cache_obj = self._prompt_cache_store.get(normalized)
        if cache_obj is None:
            raise ValueError(f"prompt cache key '{normalized}' does not exist")

        try:
            from mlx_lm.models.cache import save_prompt_cache
        except Exception as e:
            raise ImportError(
                "MLX prompt cache saving requires mlx-lm (install: `pip install \"abstractcore[mlx]\"`)."
            ) from e

        out_meta: Dict[str, Any] = dict(meta or {})
        out_meta.setdefault("format", "abstractcore-prompt-cache/v1")
        out_meta.setdefault("provider", str(getattr(self, "provider", "mlx")))
        out_meta.setdefault("model", str(getattr(self, "model", "")))
        resolved_model_id = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if resolved_model_id:
            out_meta.setdefault("model_resolved_id", resolved_model_id)
        out_meta.setdefault("saved_at", datetime.now().isoformat())

        try:
            tok = self._prompt_cache_backend_token_count(cache_obj)
            if isinstance(tok, int) and tok >= 0:
                out_meta.setdefault("token_count", tok)
        except Exception:
            pass

        cache_to_save = cache_obj
        if q8:
            try:
                cache_to_save = [layer.to_quantized(group_size=64, bits=8) for layer in cache_obj]
                out_meta["quantized"] = "q8"
            except Exception:
                # Best-effort: fall back to full precision.
                cache_to_save = cache_obj

        # mlx_lm saves KV caches via safetensors metadata, which requires string keys + values.
        def _meta_value(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                if isinstance(value, (dict, list, tuple)):
                    return json.dumps(value, ensure_ascii=False)
            except Exception:
                pass
            return str(value)

        out_meta_str: Dict[str, str] = {str(k): _meta_value(v) for k, v in out_meta.items() if isinstance(k, str) and k}

        save_prompt_cache(str(filename), cache_to_save, metadata=out_meta_str)

        return {
            "supported": True,
            "operation": "save",
            "provider": str(getattr(self, "provider", "mlx")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": out_meta_str,
        }

    def prompt_cache_load(
        self,
        filename: str,
        *,
        key: Optional[str] = None,
        make_default: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load an MLX KV prompt cache from a `.safetensors` file (model-locked; best-effort)."""
        _ = kwargs
        if not self.supports_prompt_cache():
            raise ValueError("Prompt caching is not supported for this provider/model.")

        try:
            from mlx_lm.models.cache import load_prompt_cache
        except Exception as e:
            raise ImportError(
                "MLX prompt cache loading requires mlx-lm (install: `pip install \"abstractcore[mlx]\"`)."
            ) from e

        loaded_cache, meta = load_prompt_cache(str(filename), return_metadata=True)
        meta_dict: Dict[str, Any] = dict(meta or {}) if isinstance(meta, dict) else {}

        required_ids = {
            str(meta_dict.get("model") or "").strip(),
            str(meta_dict.get("model_id") or "").strip(),
            str(meta_dict.get("model_resolved_id") or "").strip(),
        }
        required_ids.discard("")
        current_ids = {str(getattr(self, "model", "") or "").strip()}
        resolved_model_id = str(getattr(self, "_resolved_model_id", "") or "").strip()
        if resolved_model_id:
            current_ids.add(resolved_model_id)
        current_ids.discard("")
        if required_ids and not (required_ids & current_ids):
            raise ValueError(
                "Prompt cache model mismatch: "
                f"cache expects one of {sorted(required_ids)!r}, current provider is {sorted(current_ids)!r}."
            )

        if not required_ids:
            # Best-effort structural check: layer count mismatch is a strong signal of wrong model.
            try:
                expected = self._prompt_cache_backend_create()
                if isinstance(expected, (list, tuple)) and isinstance(loaded_cache, (list, tuple)):
                    if len(expected) != len(loaded_cache):
                        raise ValueError(
                            "Prompt cache appears incompatible with the current model (layer count mismatch)."
                        )
            except Exception:
                pass

        new_key = key
        normalized = self._normalize_prompt_cache_key(new_key) if new_key is not None else None
        if normalized is None:
            normalized = f"cache:{uuid.uuid4().hex[:12]}"

        store_meta: Dict[str, Any] = {
            "backend": "mlx",
            "loaded_from": str(filename),
        }
        store_meta.update(meta_dict)
        try:
            tok = self._prompt_cache_backend_token_count(loaded_cache)
            if isinstance(tok, int) and tok >= 0:
                store_meta.setdefault("token_count", tok)
        except Exception:
            pass

        self._prompt_cache_store.set(normalized, loaded_cache, meta=store_meta)
        if make_default:
            self._default_prompt_cache_key = normalized

        return {
            "supported": True,
            "operation": "load",
            "provider": str(getattr(self, "provider", "mlx")),
            "model": str(getattr(self, "model", "")),
            "key": normalized,
            "filename": str(filename),
            "meta": store_meta,
        }

    def _load_model(self):
        """Load MLX model and tokenizer"""
        try:
            from mlx_lm import load, generate, stream_generate
            import mlx.core as mx
            import os
            from contextlib import redirect_stdout, redirect_stderr
            from pathlib import Path

            # Respect AbstractCore's offline-first defaults: never download model files on-demand.
            try:
                from ..config.manager import get_config_manager

                _cfg = get_config_manager()
                if _cfg.is_offline_first():
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
            except Exception:
                pass

            from ..utils.model_cache import (
                default_hf_hub_cache_dirs,
                default_lmstudio_model_dirs,
                resolve_hf_snapshot_dir,
                resolve_lmstudio_hub_manifest,
                resolve_lmstudio_model_dir,
            )

            # Upstream compatibility: mlx-lm may call `mx.metal.device_info()` which is deprecated in recent MLX.
            # Patch the deprecated entrypoint to the supported API so the warning is fixed (not silenced).
            try:
                if hasattr(mx, "device_info") and hasattr(mx, "metal") and hasattr(mx.metal, "device_info"):
                    mx.metal.device_info = mx.device_info  # type: ignore[attr-defined]
            except Exception:
                pass

            # Clean model name - remove trailing slashes that cause HuggingFace validation errors
            clean_model_name = self.model.rstrip("/")

            def _has_weights(d: Path) -> bool:
                """Best-effort check to avoid triggering downloads on missing weights."""
                try:
                    if not d.is_dir():
                        return False
                except Exception:
                    return False
                patterns = ("*.safetensors", "*.npz", "*.bin", "*.pt", "*.pth")
                for pat in patterns:
                    try:
                        if any(d.glob(pat)):
                            return True
                    except Exception:
                        continue
                return False

            def _looks_like_gguf_dir(d: Path) -> bool:
                try:
                    if not d.is_dir():
                        return False
                except Exception:
                    return False
                try:
                    return any(p.suffix.lower() == ".gguf" for p in d.iterdir())
                except Exception:
                    return False

            # Resolve to a local directory (cache-only). Do not pass a repo id into mlx-lm,
            # as it can trigger Hub network requests even when cached.
            load_dir: Optional[Path] = None
            explicit_path = Path(clean_model_name).expanduser()
            if explicit_path.is_dir():
                load_dir = explicit_path
            else:
                load_dir = resolve_lmstudio_model_dir(clean_model_name, base_dirs=default_lmstudio_model_dirs())
                if load_dir is None:
                    snap = resolve_hf_snapshot_dir(clean_model_name, cache_dirs=default_hf_hub_cache_dirs())
                    if snap is not None and _has_weights(snap):
                        load_dir = snap

            if load_dir is None or _looks_like_gguf_dir(load_dir):
                hint_lines: list[str] = []
                if load_dir is not None and _looks_like_gguf_dir(load_dir):
                    hint_lines.append(
                        f"Found GGUF files under '{load_dir}', but the MLX provider cannot load GGUF."
                    )
                    hint_lines.append(
                        "Use `--provider huggingface` (GGUF) or `--provider lmstudio` for GGUF-backed models."
                    )
                else:
                    manifest_path = resolve_lmstudio_hub_manifest(clean_model_name)
                    if manifest_path is not None:
                        try:
                            raw = manifest_path.read_text(encoding="utf-8")
                            manifest = json.loads(raw) if raw.strip() else {}
                            deps = manifest.get("dependencies") if isinstance(manifest, dict) else None
                            if isinstance(deps, list) and deps:
                                for dep in deps:
                                    if not isinstance(dep, dict):
                                        continue
                                    for src in dep.get("sources") or []:
                                        if not isinstance(src, dict):
                                            continue
                                        if str(src.get("type") or "").strip().lower() != "huggingface":
                                            continue
                                        user = str(src.get("user") or "").strip()
                                        repo = str(src.get("repo") or "").strip()
                                        if not user or not repo:
                                            continue
                                        repo_id = f"{user}/{repo}"
                                        lm_dir = resolve_lmstudio_model_dir(
                                            repo_id, base_dirs=default_lmstudio_model_dirs()
                                        )
                                        if lm_dir is None:
                                            continue
                                        try:
                                            ggufs = sorted([p for p in lm_dir.glob("*.gguf") if p.is_file()])
                                        except Exception:
                                            ggufs = []
                                        if ggufs:
                                            hint_lines.append(
                                                f"LM Studio hub entry found for '{clean_model_name}', but it resolves to GGUF files (e.g. '{ggufs[0].name}')."
                                            )
                                            hint_lines.append(
                                                "MLX provider cannot load GGUF; use `--provider huggingface` (GGUF) or `--provider lmstudio`."
                                            )
                                            break
                                    if hint_lines:
                                        break
                        except Exception:
                            pass

                searched_lms = [str(p) for p in default_lmstudio_model_dirs()]
                searched_hf = [str(p) for p in default_hf_hub_cache_dirs()]
                headline = (
                    f"❌ MLX model '{clean_model_name}' not found locally (downloads are disabled)."
                    if load_dir is None
                    else f"❌ MLX provider cannot load '{clean_model_name}' (GGUF detected; downloads are disabled)."
                )
                msg = (
                    f"{headline}\n\n"
                    f"Searched LM Studio caches:\n  - "
                    + "\n  - ".join(searched_lms or ["(none found)"])
                    + "\n\n"
                    f"Searched HuggingFace hub caches:\n  - "
                    + "\n  - ".join(searched_hf or ["(none found)"])
                    + "\n"
                )
                if hint_lines:
                    msg += "\n" + "\n".join(hint_lines) + "\n"
                msg += "\nTip: download explicitly (e.g. with `huggingface-cli download ...`) or pass a local model directory path."
                raise ModelNotFoundError(msg)

            load_target = str(load_dir)
            self._resolved_model_id = load_target

            # Silence the "Fetching" progress bar by redirecting stdout/stderr
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    try:
                        self.llm, self.tokenizer = load(load_target)
                    except ValueError as e:
                        msg = str(e)
                        low = msg.lower()
                        if "model type" in low and "not supported" in low:
                            model_type = None
                            try:
                                cfg_path = Path(load_target) / "config.json"
                                if cfg_path.is_file():
                                    raw = cfg_path.read_text(encoding="utf-8", errors="ignore")
                                    cfg = json.loads(raw) if raw.strip() else {}
                                    model_type = cfg.get("model_type") if isinstance(cfg, dict) else None
                            except Exception:
                                model_type = None

                            mlx_lm_version = None
                            try:  # pragma: no cover
                                import mlx_lm  # type: ignore

                                mlx_lm_version = getattr(mlx_lm, "__version__", None)
                            except Exception:
                                mlx_lm_version = None

                            ver_s = f" (mlx-lm {mlx_lm_version})" if mlx_lm_version else ""
                            extra_hint = ""
                            if str(model_type or "").strip().lower() == "gemma4":
                                extra_hint = (
                                    "\n"
                                    "Note:\n"
                                    "  - Gemma 4 MLX models require a newer mlx-lm build (>=0.31.2).\n"
                                    "    If that version is not available on PyPI yet, install mlx-lm from source until it is released.\n"
                                )

                            raise ModelNotFoundError(
                                f"❌ MLX provider cannot load '{clean_model_name}' from '{load_target}'.\n\n"
                                f"Detected model_type={model_type!r}, but the installed mlx-lm does not support it{ver_s}.\n\n"
                                "Try one of:\n"
                                "  - Use provider='huggingface' (transformers) for this local model directory\n"
                                "  - Use provider='lmstudio' if you are running LM Studio's local server\n"
                                "  - Upgrade mlx-lm once a release with this model_type is published on PyPI\n"
                                f"{extra_hint}"
                            ) from e
                        raise

            self.generate_fn = generate
            self.stream_generate_fn = stream_generate
        except ImportError:
            raise ImportError("MLX dependencies not installed. Install with: pip install mlx-lm")
        except Exception as e:
            # Check if it's a model not found error
            error_str = str(e).lower()
            if "not found" in error_str or "does not exist" in error_str or "failed to load" in error_str:
                available_models = self.list_available_models()
                error_message = format_model_error("MLX", self.model, available_models)
                raise ModelNotFoundError(error_message)
            raise Exception(f"Failed to load MLX model {self.model}: {str(e)}")

    def unload_model(self, model_name: str) -> None:
        """
        Unload the MLX model from memory.

        Clears model and tokenizer references and forces garbage collection
        to free GPU/CPU memory immediately.
        """
        import gc
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                # Clear MLX model
                del self.llm
                self.llm = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # Clear tokenizer
                del self.tokenizer
                self.tokenizer = None

            if hasattr(self, 'generate_fn'):
                self.generate_fn = None

            if hasattr(self, 'stream_generate_fn'):
                self.stream_generate_fn = None

            # Force garbage collection to free memory immediately
            gc.collect()
        except Exception as e:
            # Log but don't raise - unload should be best-effort
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error during unload: {e}")

    def _handle_timeout_parameter(self, kwargs: Dict[str, Any]) -> None:
        """
        Handle timeout parameter for MLX provider.

        Since MLX models run locally on Apple Silicon,
        timeout parameters don't apply. If a non-None timeout is provided,
        issue a warning and treat it as None (infinity).

        Args:
            kwargs: Initialization kwargs that may contain timeout
        """
        timeout_value = kwargs.get('timeout')
        if timeout_value is not None:
            import warnings
            warnings.warn(
                f"MLX provider runs models locally on Apple Silicon and does not support timeout parameters. "
                f"Provided timeout={timeout_value} will be ignored and treated as None (unlimited).",
                UserWarning,
                stacklevel=3
            )
            # Force timeout to None for local models
            self._timeout = None
        else:
            # Keep None value (unlimited timeout is appropriate for local models)
            self._timeout = None

    def _update_http_client_timeout(self) -> None:
        """
        MLX provider doesn't use HTTP clients for model inference.
        Local models on Apple Silicon don't have timeout constraints.
        """
        # No-op for local models - they don't use HTTP clients
        pass

    def generate(self, *args, **kwargs):
        """Public generate method that includes telemetry"""
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(self,
                          prompt: str,
                          messages: Optional[List[Dict[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          tools: Optional[List[Dict[str, Any]]] = None,
                          media: Optional[List['MediaContent']] = None,
                          stream: bool = False,
                          response_model: Optional[Type[BaseModel]] = None,
                          **kwargs) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """Internal generation with MLX and optional Outlines native structured output"""

        if not self.llm or not self.tokenizer:
            return GenerateResponse(
                content="Error: MLX model not loaded",
                model=self.model,
                finish_reason="error"
            )

        prompt_cache_prefilled_modules = kwargs.pop("prompt_cache_prefilled_modules", None)
        if isinstance(prompt_cache_prefilled_modules, tuple):
            prompt_cache_prefilled_modules = list(prompt_cache_prefilled_modules)
        if isinstance(prompt_cache_prefilled_modules, str):
            prompt_cache_prefilled_modules = [prompt_cache_prefilled_modules]
        if not isinstance(prompt_cache_prefilled_modules, list):
            prompt_cache_prefilled_modules = None
        mlx_enable_thinking = kwargs.get("_acore_mlx_enable_thinking")

        # Native structured output via Outlines (if configured and available)
        should_use_outlines = (
            response_model and
            PYDANTIC_AVAILABLE and
            not stream and
            self.structured_output_method != "prompted"  # Skip if explicitly prompted
        )

        if should_use_outlines:
            # Check if Outlines is required but unavailable
            if self.structured_output_method == "native_outlines" and not OUTLINES_AVAILABLE:
                return GenerateResponse(
                    content="Error: structured_output_method='native_outlines' requires Outlines library. Install with: pip install \"abstractcore[mlx]\"",
                    model=self.model,
                    finish_reason="error"
                )

            # Try Outlines if available (auto or native_outlines mode)
            if OUTLINES_AVAILABLE:
                try:
                    # Cache Outlines MLX model wrapper to avoid re-initialization
                    if not hasattr(self, '_outlines_model') or self._outlines_model is None:
                        self.logger.debug("Creating Outlines MLX model wrapper for native structured output")
                        self._outlines_model = outlines.from_mlxlm(self.llm, self.tokenizer)

                    # Build full prompt (same as normal generation)
                    processed_prompt = prompt
                    full_prompt = self._build_prompt(processed_prompt, messages, system_prompt, tools)

                    # Create constrained generator with JSON schema
                    self.logger.debug(f"Using Outlines native structured output for {response_model.__name__}")
                    generator = self._outlines_model(
                        full_prompt,
                        outlines.json_schema(response_model),
                        max_tokens=kwargs.get("max_tokens", self.max_tokens or 512)
                    )

                    # Validate and return
                    validated_obj = response_model.model_validate(generator)

                    return GenerateResponse(
                        content=validated_obj.model_dump_json(),
                        model=self.model,
                        finish_reason="stop",
                        validated_object=validated_obj
                    )
                except Exception as e:
                    # If native_outlines was explicitly requested, don't fall back
                    if self.structured_output_method == "native_outlines":
                        return GenerateResponse(
                            content=f"Error: Outlines native structured output failed: {str(e)}",
                            model=self.model,
                            finish_reason="error"
                        )
                    # Otherwise fall back to prompted approach
                    self.logger.debug(f"Outlines generation failed, falling back to prompted: {e}")
                    # Continue with normal generation below

        # Handle media content first if present
        processed_prompt = prompt
        media_enrichment = None
        if media:
            try:
                from ..media.handlers import LocalMediaHandler
                media_handler = LocalMediaHandler("mlx", self.model_capabilities, model_name=self.model)

                # Create multimodal message combining text and media
                multimodal_message = media_handler.create_multimodal_message(prompt, media)
                media_enrichment = getattr(media_handler, "media_enrichment", None)

                # For MLX (local provider), we get text-embedded content
                if isinstance(multimodal_message, str):
                    processed_prompt = multimodal_message
                else:
                    # If we get a structured message, extract the content
                    if isinstance(multimodal_message, dict) and "content" in multimodal_message:
                        if isinstance(multimodal_message["content"], list):
                            # Find text content in the structured message
                            text_content = ""
                            for item in multimodal_message["content"]:
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                    break
                            processed_prompt = text_content or prompt
                        else:
                            processed_prompt = str(multimodal_message["content"])
            except ImportError:
                self.logger.warning("Media processing not available. Install with: pip install \"abstractcore[media]\"")
            except Exception as e:
                self.logger.warning(f"Failed to process media content: {e}")

        # Build full prompt with tool support
        full_prompt = self._build_prompt(
            processed_prompt,
            messages,
            system_prompt,
            tools,
            prefilled_modules=prompt_cache_prefilled_modules,
            enable_thinking=mlx_enable_thinking if isinstance(mlx_enable_thinking, bool) else None,
        )

        # MLX generation parameters using unified system
        generation_kwargs = self._prepare_generation_kwargs(**kwargs)
        max_tokens = self._get_provider_max_tokens_param(generation_kwargs)
        temperature = generation_kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", 0.9)
        seed_value = generation_kwargs.get("seed")
        prompt_cache = None
        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
            prompt_cache = self._prompt_cache_store.get(prompt_cache_key.strip())
            if prompt_cache is None:
                self.prompt_cache_set(prompt_cache_key.strip(), make_default=False)
                prompt_cache = self._prompt_cache_store.get(prompt_cache_key.strip())

        try:
            if stream:
                return self._stream_generate_with_tools(
                    full_prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    tools,
                    kwargs.get('tool_call_tags'),
                    seed_value,
                    prompt_cache,
                )
            else:
                response = self._single_generate(
                    full_prompt, max_tokens, temperature, top_p, seed_value, prompt_cache
                )
                if media_enrichment:
                    from ..media.enrichment import merge_enrichment_metadata

                    response.metadata = merge_enrichment_metadata(response.metadata, media_enrichment)

                # Handle tool execution for prompted models
                if tools and self.tool_handler.supports_prompted and response.content:
                    response = self._handle_prompted_tool_execution(response, tools)

                return response

        except Exception as e:
            return GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def _build_prompt(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        prefilled_modules: Optional[List[str]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Build prompt for MLX model with tool support."""
        return self._build_prompt_fragment(
            prompt=str(prompt or ""),
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            add_generation_prompt=True,
            prefilled_modules=prefilled_modules,
            enable_thinking=enable_thinking,
        )

    def _single_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
    ) -> GenerateResponse:
        """Generate single response"""

        # Handle seed parameter (MLX supports seed via mx.random.seed)
        if seed is not None:
            import mlx.core as mx
            mx.random.seed(seed)
            self.logger.debug(f"Set MLX random seed to {seed} for deterministic generation")

        # Track generation time
        start_time = time.time()

        # Try different MLX API signatures
        try:
            # Try new mlx-lm API
            response_text = self.generate_fn(
                self.llm,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
                prompt_cache=prompt_cache,
            )
        except TypeError:
            try:
                # Try older API without parameters
                response_text = self.generate_fn(
                    self.llm,
                    self.tokenizer,
                    prompt
                )
            except:
                # Fallback to basic response
                response_text = prompt + " I am an AI assistant powered by MLX on Apple Silicon."

        gen_time = round((time.time() - start_time) * 1000, 1)

        # Use the full response as-is - preserve all content including thinking
        generated = response_text.strip()

        return GenerateResponse(
            content=generated,
            model=self.model,
            finish_reason="stop",
            usage=self._calculate_usage(prompt, generated),
            gen_time=gen_time
        )

    def _calculate_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate token usage using centralized token utilities."""
        from ..utils.token_utils import TokenUtils

        input_tokens = TokenUtils.estimate_tokens(prompt, self.model)
        output_tokens = TokenUtils.estimate_tokens(response, self.model)
        total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # Keep legacy keys for backward compatibility
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens
        }

    def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tool_call_tags: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
    ) -> Iterator[GenerateResponse]:
        """Generate real streaming response using MLX stream_generate with tool tag rewriting support"""
        try:
            # Handle seed parameter (MLX supports seed via mx.random.seed)
            if seed is not None:
                import mlx.core as mx
                mx.random.seed(seed)
                self.logger.debug(f"Set MLX random seed to {seed} for deterministic streaming generation")

            # Initialize tool tag rewriter if needed
            rewriter = None
            buffer = ""
            if tool_call_tags:
                try:
                    from ..tools.tag_rewriter import create_tag_rewriter
                    rewriter = create_tag_rewriter(tool_call_tags)
                except ImportError:
                    pass

            # Use MLX's native streaming with minimal parameters
            for response in self.stream_generate_fn(
                self.llm,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
            ):
                # Each response has a .text attribute with the new token(s)
                content = response.text

                # Apply tool tag rewriting if enabled
                if rewriter and content:
                    rewritten_content, buffer = rewriter.rewrite_streaming_chunk(content, buffer)
                    content = rewritten_content

                yield GenerateResponse(
                    content=content,
                    model=self.model,
                    finish_reason=None,  # MLX doesn't provide finish reason in stream
                    raw_response=response
                )

        except Exception as e:
            yield GenerateResponse(
                content=f"Error: {str(e)}",
                model=self.model,
                finish_reason="error"
            )

    def get_capabilities(self) -> List[str]:
        """Get MLX capabilities"""
        return ["streaming", "chat"]

    def validate_config(self) -> bool:
        """Validate MLX model is loaded"""
        return self.llm is not None and self.tokenizer is not None

    # Removed override - using BaseProvider method with JSON capabilities

    def _get_provider_max_tokens_param(self, kwargs: Dict[str, Any]) -> int:
        """Get max tokens parameter for MLX generation"""
        # For MLX, max_tokens is the max output tokens
        return kwargs.get("max_output_tokens", self.max_output_tokens)


    def _stream_generate_with_tools(
        self,
        full_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_call_tags: Optional[str] = None,
        seed: Optional[int] = None,
        prompt_cache: Optional[Any] = None,
    ) -> Iterator[GenerateResponse]:
        """Stream generate with tool execution at the end"""
        collected_content = ""

        # Stream the response content
        for chunk in self._stream_generate(
            full_prompt, max_tokens, temperature, top_p, tool_call_tags, seed, prompt_cache
        ):
            collected_content += chunk.content or ""
            yield chunk

        # Handle tool execution if we have tools and content
        if tools and self.tool_handler.supports_prompted and collected_content:
            # Create complete response for tool processing
            complete_response = GenerateResponse(
                content=collected_content,
                model=self.model,
                finish_reason="stop"
            )

            # Handle tool execution using base method
            final_response = self._handle_prompted_tool_execution(complete_response, tools)

            # If tools were executed, yield the tool results as final chunk
            if final_response.content != collected_content:
                tool_results_content = final_response.content[len(collected_content):]
                yield GenerateResponse(
                    content=tool_results_content,
                    model=self.model,
                    finish_reason="stop"
                )

    @classmethod
    def list_available_models(cls, **kwargs) -> List[str]:
        """
        List available MLX models from local caches.

        This includes:
        - HuggingFace hub cache (~/.cache/huggingface/hub) for any repo containing "mlx"
        - LM Studio cache (~/.lmstudio/models) for any org/model containing "mlx"

        Args:
            **kwargs: Optional parameters including:
                - input_capabilities: List of ModelInputCapability enums to filter by input capability
                - output_capabilities: List of ModelOutputCapability enums to filter by output capability

        Returns:
            List of model names, optionally filtered by capabilities
        """
        from pathlib import Path
        from .model_capabilities import filter_models_by_capabilities

        try:
            model_set = set()

            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache.exists():
                for item in hf_cache.iterdir():
                    if item.is_dir() and item.name.startswith("models--"):
                        # Convert models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit to mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
                        model_name = item.name.replace("models--", "").replace("--", "/")

                        # Include ANY model with "mlx" in the name (case-insensitive)
                        # This captures: mlx-community/*, */mlx-*, *-mlx-*, etc.
                        if "mlx" in model_name.lower():
                            model_set.add(model_name)

            lmstudio_models = Path.home() / ".lmstudio" / "models"
            if lmstudio_models.exists():
                # LM Studio stores models under: ~/.lmstudio/models/<org>/<model>/*
                for org_dir in lmstudio_models.iterdir():
                    if not org_dir.is_dir():
                        continue
                    # These org folders are MLX by design (model names may not include "mlx")
                    include_all_in_org = org_dir.name.lower() in {"mlx-community", "lmstudio-community"}
                    for model_dir in org_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        model_name = f"{org_dir.name}/{model_dir.name}"
                        if include_all_in_org or "mlx" in model_name.lower():
                            model_set.add(model_name)

            models = sorted(model_set)

            # Apply new capability filtering if provided
            input_capabilities = kwargs.get('input_capabilities')
            output_capabilities = kwargs.get('output_capabilities')

            if input_capabilities or output_capabilities:
                models = filter_models_by_capabilities(
                    models, 
                    input_capabilities=input_capabilities,
                    output_capabilities=output_capabilities
                )


            return models

        except Exception:
            return []
