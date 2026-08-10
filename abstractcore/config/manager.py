"""
AbstractCore Configuration Manager

Provides centralized configuration management for AbstractCore.
"""

import copy
import json
import os
import uuid
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timezone

from .capability_defaults import (
    TEXT_ROUTE_KEY,
    TEXT_ROUTE_STORAGE_KEY,
    CapabilityDefaultsConfig,
    CapabilityRouteDefault,
    capability_defaults_from_dict,
    capability_route_broad_key,
    capability_route_key,
    capability_route_task_keys,
    capability_route_tasks_cover_broad,
    iter_capability_default_specs,
    split_capability_default_route,
)
from .provider_profiles import (
    ProviderProfile,
    ProviderProfilesConfig,
    api_key_fingerprint,
    normalize_base_url,
    normalize_profile_id,
    normalize_provider_family,
    normalize_string_list,
    profile_id_from_virtual_provider,
    provider_profiles_from_dict,
    split_api_key_value,
)

_MISSING = object()


def merge_store_documents(
    baseline: Optional[Dict[str, Any]],
    mine: Dict[str, Any],
    disk: Dict[str, Any],
) -> Dict[str, Any]:
    """Three-way merge of the config store, key by key, recursing into objects.

    THE RULE: a writer publishes what it CHANGED and preserves everything else.
    `baseline` is the document the writer last saw on disk, `mine` is what it
    wants to publish, `disk` is what is there now. Per key:

      - in `mine` only, or changed by me      -> mine wins
      - unchanged by me, changed on disk      -> DISK wins (another writer)
      - in `baseline`, absent from `mine`,
        and unchanged on disk                 -> a DELETE I made; stays deleted
      - absent from `baseline` and `mine`,
        present on disk                       -> another writer ADDED it; kept

    Without this, an atomic whole-file publish is a silent revert of every
    change made since the writer loaded (incident 2026-08-01: a route row that
    simply was not there any more). Lists are compared and replaced whole --
    they are values here (allowlists, fallback chains), never mergeable sets.
    """

    base = baseline if isinstance(baseline, dict) else {}
    out: Dict[str, Any] = {}
    # `mine` first so the published key order stays the writer's order; disk-only
    # keys (another writer's additions) follow.
    for key in list(mine.keys()) + [k for k in disk.keys() if k not in mine]:
        b = base.get(key, _MISSING)
        m = mine.get(key, _MISSING)
        d = disk.get(key, _MISSING)

        if m is _MISSING:
            # I do not carry this key. Deliberate delete, or someone else's row?
            if b is not _MISSING and b == d:
                continue  # I deleted it and nobody touched it since -> stays gone
            out[key] = copy.deepcopy(d)
            continue

        if d is _MISSING:
            # Gone from disk: either another writer deleted it, or it is new here.
            if b is not _MISSING and b == m:
                continue  # I did not change it; honour their delete
            out[key] = copy.deepcopy(m)
            continue

        if isinstance(m, dict) and isinstance(d, dict):
            out[key] = merge_store_documents(b if isinstance(b, dict) else {}, m, d)
            continue

        if b is not _MISSING and m == b and d != b:
            out[key] = copy.deepcopy(d)  # only the other writer changed it
        else:
            out[key] = copy.deepcopy(m)
    return out


_SERVER_AUTH_TOKEN_ENV_VAR = "ABSTRACTCORE_AUTH_TOKEN"
_PROVIDER_MODEL_PREFIXES = {
    "anthropic",
    "google",
    "huggingface",
    "lmstudio",
    "ollama",
    "openai",
    "openai-compatible",
    "openai_compatible",
    "openrouter",
    "portkey",
    "vllm",
}


def _split_provider_model(value: str, *, default_provider: str) -> Tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Model cannot be empty")
    if ":" in raw:
        provider, model = raw.split(":", 1)
        provider_clean = provider.strip().lower()
        if provider_clean in _PROVIDER_MODEL_PREFIXES:
            model_clean = model.strip()
            if not model_clean:
                raise ValueError("Model cannot be empty")
            return provider.strip(), model_clean
    if "/" in raw:
        provider, model = raw.split("/", 1)
        model_clean = model.strip()
        if not model_clean:
            raise ValueError("Model cannot be empty")
        return provider.strip() or default_provider, model_clean
    return default_provider, raw


@dataclass
class VisionConfig:
    """Vision configuration settings."""
    strategy: str = "disabled"
    caption_provider: Optional[str] = None
    caption_model: Optional[str] = None
    fallback_chain: list = None
    local_models_path: Optional[str] = None

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = []


@dataclass
class AudioConfig:
    """Audio configuration settings (input policy + optional fallback)."""
    # Default: "auto" — use native audio when supported. Non-native STT fallback
    # is authorized by an explicit input.voice capability default or per-call policy.
    strategy: str = "auto"  # native_only|speech_to_text|caption|auto
    # Optional preferred STT backend (capabilities plugin backend_id).
    stt_backend_id: Optional[str] = None
    stt_language: Optional[str] = None
    # Reserved for future "audio caption" backends.
    caption_provider: Optional[str] = None
    caption_model: Optional[str] = None
    fallback_chain: list = None

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = []


@dataclass
class VideoConfig:
    """Video configuration settings (input policy + optional fallback)."""
    # Default: prefer native video when supported. Sampled-frame fallback is
    # authorized by model capabilities or an explicit input.video capability default.
    strategy: str = "auto"  # native_only|frames_caption|auto

    # Frame sampling controls for frames-based fallback.
    max_frames: int = 3
    # Native video models typically require more temporal coverage than the fallback path.
    # This default is used when the selected model supports native video input (v0: HF only).
    max_frames_native: int = 8
    frame_format: str = "jpg"  # jpg|png
    sampling_strategy: str = "uniform"  # uniform|keyframes

    # Downscale extracted frames (preserve aspect ratio; never upscale). Helps memory + token pressure.
    # Applies to both frames_caption fallback and HF native video ingestion (which uses ffmpeg frames).
    max_frame_side: int = 1024

    # Maximum video size allowed for processing (bytes). None => use media handler defaults.
    max_video_size_bytes: Optional[int] = None


@dataclass
class EmbeddingsConfig:
    """Embeddings configuration settings."""
    provider: Optional[str] = "huggingface"
    model: Optional[str] = "all-minilm-l6-v2"
    base_url: Optional[str] = None


@dataclass
class AppDefaults:
    """Per-application default configurations."""
    cli_provider: Optional[str] = "huggingface"
    cli_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    summarizer_provider: Optional[str] = "huggingface"
    summarizer_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    extractor_provider: Optional[str] = "huggingface"
    extractor_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    judge_provider: Optional[str] = "huggingface"
    judge_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    intent_provider: Optional[str] = "huggingface"
    intent_model: Optional[str] = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


@dataclass
class MaintenanceConfig:
    """Maintenance agent configuration (triage, stewardship)."""

    # LLM assist (optional, local-first).
    triage_llm_enabled: bool = False
    triage_llm_base_url: str = "http://localhost:1234"
    triage_llm_model: str = "qwen/qwen3-next-80b"
    triage_llm_temperature: float = 0.2
    triage_llm_max_tokens: int = 800
    # #[WARNING:TIMEOUT] Triage LLM client timeout, 7200s (ADR-0027 §2/§4).
    # This dataclass is the AUTHORITATIVE source: `MaintenanceConfig` is
    # always serialized into `~/.abstractcore/config/abstractcore.json`, so
    # `abstractgateway.maintenance.llm_assist` reads THIS number via
    # `stored.get("triage_llm_timeout_s", DEFAULT_TRIAGE_LLM_TIMEOUT_S)` and
    # the module-level 7200s fallback there is NEVER reached. It was 30.0:
    # raising only the gateway fallback left every triage call capped at 30s
    # (local prompt processing alone exceeds that), surfaced as an opaque
    # "LLM assist failed". Configure via `maintenance.triage_llm_timeout_s`
    # or ABSTRACT_TRIAGE_LLM_TIMEOUT_S; `0` = no client timeout.
    triage_llm_timeout_s: float = 7200.0


@dataclass
class EmailConfig:
    """Email defaults (SMTP outbound + IMAP inbound).

    These defaults are used by framework-native comms tools and gateway bridges when
    explicit parameters are omitted (env vars still take precedence).
    """

    # SMTP (outbound)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password_env_var: str = "EMAIL_PASSWORD"
    smtp_use_starttls: bool = True
    from_email: Optional[str] = None
    reply_to: Optional[str] = None

    # IMAP (inbound)
    imap_host: str = ""
    imap_port: int = 993
    imap_username: str = ""
    imap_password_env_var: str = "EMAIL_PASSWORD"
    imap_folder: str = "INBOX"


@dataclass
class DefaultModels:
    """Global default model configurations."""
    global_provider: Optional[str] = None
    global_model: Optional[str] = None
    chat_model: Optional[str] = None
    code_model: Optional[str] = None


@dataclass
class ApiKeysConfig:
    """API keys configuration."""
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    openrouter: Optional[str] = None
    portkey: Optional[str] = None
    openai_compatible: Optional[str] = None
    vllm: Optional[str] = None
    google: Optional[str] = None


@dataclass
class ServerConfig:
    """OpenAI-compatible HTTP gateway configuration."""

    # Inbound server auth token. When set, clients authenticate with
    # `Authorization: Bearer <token>` and can use provider keys configured on
    # the server.
    auth_token: Optional[str] = None

    # Dangerous local/dev escape hatch. Production should keep this false.
    allow_unauthenticated: bool = False

    # Comma-separated allowlists matching the corresponding server env vars.
    base_url_allowlist: Optional[str] = None
    url_fetch_allowlist: Optional[str] = None

    # Local HTTP media path controls.
    media_root: Optional[str] = None
    allow_local_files: bool = False

    # Optional default bind for `python -m abstractcore.server.app`.
    host: Optional[str] = None
    port: Optional[int] = None


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    default_cache_dir: str = "~/.cache/abstractcore"
    huggingface_cache_dir: str = "~/.cache/huggingface"
    local_models_cache_dir: str = "~/.abstractcore/models"
    glyph_cache_dir: str = "~/.abstractcore/glyph_cache"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    console_level: str = "ERROR"
    file_level: str = "DEBUG"
    file_logging_enabled: bool = False
    log_base_dir: Optional[str] = None
    verbatim_enabled: bool = True
    console_json: bool = False
    file_json: bool = True


@dataclass
class StreamingConfig:
    """Streaming configuration settings."""
    cli_stream_default: bool = False


@dataclass
class TimeoutConfig:
    """Timeout configuration settings."""
    # #[WARNING:TIMEOUT] Process-wide LLM HTTP budget, 7200s (ADR-0027 §2:
    # high safeguard for correctness-critical paths; `0` = unlimited).
    # Overridden per-provider/per-call, and by `abstractruntime`'s
    # authoritative per-effect budget under orchestration (ADR-0014 §1).
    # Config key: `timeouts.default_timeout`.
    default_timeout: float = 7200.0  # 2 hours
    # #[WARNING:TIMEOUT] One tool call, 7200s. ADR-0014 §2 fixes the tool
    # default at 7200s alongside the LLM default; the former 600s here was a
    # second, LOWER floor that could truncate a healthy long-running tool
    # (build, test suite, indexing run) below the orchestrator's budget
    # whenever abstractcore was used directly. Config key:
    # `timeouts.tool_timeout` (`0` = unlimited).
    tool_timeout: float = 7200.0    # 2 hours (ADR-0014 §2), matches default_timeout


@dataclass
class OfflineConfig:
    """Offline-first configuration settings."""
    offline_first: bool = True  # AbstractCore is designed offline-first for open source LLMs
    allow_network: bool = False  # Allow network access when offline_first is True (for API providers)
    force_local_files_only: bool = True  # Force local_files_only for HuggingFace transformers


@dataclass
class AbstractCoreConfig:
    """Main configuration class."""
    vision: VisionConfig
    audio: AudioConfig
    video: VideoConfig
    embeddings: EmbeddingsConfig
    app_defaults: AppDefaults
    default_models: DefaultModels
    capability_defaults: CapabilityDefaultsConfig
    provider_profiles: ProviderProfilesConfig
    api_keys: ApiKeysConfig
    server: ServerConfig
    cache: CacheConfig
    logging: LoggingConfig
    streaming: StreamingConfig
    timeouts: TimeoutConfig
    offline: OfflineConfig
    maintenance: MaintenanceConfig
    email: EmailConfig

    @classmethod
    def default(cls):
        """Create default configuration."""
        return cls(
            vision=VisionConfig(),
            audio=AudioConfig(),
            video=VideoConfig(),
            embeddings=EmbeddingsConfig(),
            app_defaults=AppDefaults(),
            default_models=DefaultModels(),
            capability_defaults=CapabilityDefaultsConfig(),
            provider_profiles=ProviderProfilesConfig(),
            api_keys=ApiKeysConfig(),
            server=ServerConfig(),
            cache=CacheConfig(),
            logging=LoggingConfig(),
            streaming=StreamingConfig(),
            timeouts=TimeoutConfig(),
            offline=OfflineConfig(),
            maintenance=MaintenanceConfig(),
            email=EmailConfig(),
        )


class CapabilityDefaultWriteError(ValueError):
    """A capability-default write failed, and the message says why.

    A `ValueError` on purpose: the AbstractCore server and the AbstractGateway
    seam already map `ValueError` to a 400, so the reason reaches the operator
    through every entry point without any of them learning a new type.
    """


def _capability_write_error(
    action: str,
    kind: Any,
    modality: Any,
    task: Any,
    cause: Exception,
) -> "CapabilityDefaultWriteError":
    route = ".".join(str(part) for part in (kind, modality, task) if part)
    return CapabilityDefaultWriteError(
        f"Failed to {action} capability default {route or kind!r}: "
        f"{type(cause).__name__}: {cause}"
    )


def resolve_config_file(
    config_dir: Optional[Union[str, Path]] = None,
    config_file: Optional[Union[str, Path]] = None,
) -> Path:
    """WHERE the config lives, without opening it.

    Pure and cheap (no I/O beyond `expanduser`): this is the resolution
    `ConfigurationManager.__init__` performs, extracted so a caller that only
    needs the PATH does not pay a full load to learn it. `stat`-ing this path
    is how a host tells whether the store moved under it, and that check runs
    per run -- building a manager for it re-read and re-parsed the very file
    the check exists to avoid re-reading (measured 133us vs 1.7us for the stat).
    """

    if config_file is not None:
        return Path(config_file).expanduser()
    env_config_file = os.getenv("ABSTRACTCORE_CONFIG_FILE")
    if env_config_file:
        return Path(env_config_file).expanduser()
    env_config_dir = os.getenv("ABSTRACTCORE_CONFIG_DIR")
    base = Path(config_dir or env_config_dir or (Path.home() / ".abstractcore" / "config")).expanduser()
    return base / "abstractcore.json"


class ConfigurationManager:
    """Manages AbstractCore configuration."""

    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        *,
        apply_env: bool = True,
    ):
        # Backward-compatible meta flags (stored at top-level in the JSON file).
        self._audio_strategy_explicit = False

        self.config_file = resolve_config_file(config_dir, config_file)
        self.config_dir = self.config_file.parent
        self._apply_env = bool(apply_env)
        # Warn-once memo for shadowed-key warnings (dm#201): initialized BEFORE
        # _load_config/_apply_api_keys_to_env run (init-order rule: config-
        # restored state initializes before the load call that consumes it).
        self._shadowed_key_warned: set = set()
        # THE STORE AS THIS MANAGER LAST SAW IT (see `_save_config`): the raw
        # document at load time, then the document each save publishes. It is
        # the BASELINE of the three-way merge that keeps a save from reverting
        # another writer's rows. `{}` means "no trustworthy baseline" (fresh
        # install or unreadable file) and the merge degrades to a whole publish.
        self._store_baseline: Dict[str, Any] = self._read_store_document() or {}
        self.config = self._load_config()
        self._apply_smart_defaults()
        if self._apply_env:
            self._apply_api_keys_to_env()
            self._apply_server_config_to_env()
        self._provider_config: Dict[str, Dict[str, Any]] = {}  # Runtime config (not persisted)
        # Process-lifetime provider profiles injected by hosts (e.g. a client
        # materializing gateway-registered endpoint profiles). Deliberately a
        # SEPARATE dict from config.provider_profiles: if injected rows lived
        # in the persisted mapping, any unrelated _save_config() would silently
        # write host-derived profiles into the local config file.
        self._runtime_provider_profiles: Dict[str, ProviderProfile] = {}

    def _filter_dataclass_kwargs(self, cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in data.items() if k in allowed}

    def _has_abstractvoice(self) -> bool:
        try:
            return importlib.util.find_spec("abstractvoice") is not None
        except Exception:
            return False

    def _apply_smart_defaults(self) -> None:
        """Apply non-persisted, environment-aware defaults.

        Goal: keep legacy `audio.strategy="auto"` usable for native-capable audio
        models while Core capability defaults decide whether STT fallback is enabled.
        """
        try:
            audio = getattr(self.config, "audio", None)
            if audio is None:
                return

            # Respect explicit user choice.
            if bool(getattr(self, "_audio_strategy_explicit", False)):
                return

            raw = str(getattr(audio, "strategy", "") or "").strip().lower()
            has_av = bool(self._has_abstractvoice())
            if has_av and raw in {"", "native_only", "native", "disabled"}:
                audio.strategy = "auto"
            elif (not has_av) and raw in {"auto", "speech_to_text", "stt"}:
                audio.strategy = "native_only"
        except Exception:
            # Never fail config initialization due to smart defaults.
            return

    def _apply_api_keys_to_env(self) -> None:
        """Reconcile config-persisted API keys with os.environ.

        Providers read API keys from environment variables (e.g. OPENAI_API_KEY).
        This bridges the gap: keys saved via ``abstractcore --set-api-key`` are
        written to the config JSON but must appear in os.environ for providers
        to find them.

        Precedence (operator ruling dm#201, 2026-07-22): cloud API keys are the
        RULED exception to behavior-env elimination — they stay env-INHERITABLE
        by default (they exist for other apps too; no migration forced). BUT a
        key redefined in the config ALWAYS supersedes the env var: a rotated
        console/wizard key must apply to all of this process's traffic, not
        just the lanes that happened to lack an export (the key-precedence
        inversion conflict, env-conflict report angle A #3). A shadowed env
        key is warned once with fingerprints (never key material).
        """
        _KEY_MAP = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "portkey": "PORTKEY_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
            "vllm": "VLLM_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        try:
            api_keys = self.config.api_keys
            applied_this_pass: set = set()
            for attr, env_var in _KEY_MAP.items():
                key = getattr(api_keys, attr, None)
                if not key:
                    continue  # env-inheritable by default: nothing configured, env stands
                if env_var in applied_this_pass:
                    # Two config fields share one env var (openai /
                    # openai_compatible → OPENAI_API_KEY): first field wins,
                    # matching the historical injection order.
                    continue
                env_value = os.environ.get(env_var)
                if env_value and env_value != key:
                    shadow_state = (env_var, api_key_fingerprint(key), api_key_fingerprint(env_value))
                    if shadow_state not in self._shadowed_key_warned:
                        self._shadowed_key_warned.add(shadow_state)
                        import logging

                        logging.getLogger(__name__).warning(
                            f"#FALLBACK {env_var} from the environment (fingerprint "
                            f"{api_key_fingerprint(env_value)}) is SHADOWED by the key configured via "
                            f"`abstractcore --config` (fingerprint {api_key_fingerprint(key)}) — the "
                            f"configured key applies (operator ruling dm#201). Unset the env var or "
                            f"clear the configured key to silence this."
                        )
                os.environ[env_var] = key
                applied_this_pass.add(env_var)
        except Exception:
            # Never fail config initialization.
            pass

    def _apply_server_config_to_env(self) -> None:
        """Inject persisted server settings into os.environ when env vars are absent.

        Environment variables always win. This lets deployments override local
        config through Docker/Kubernetes/secrets managers while still making
        `abstractcore --config` useful for local server runs.
        """
        try:
            server = self.config.server

            value_map = {
                _SERVER_AUTH_TOKEN_ENV_VAR: server.auth_token,
                "ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST": server.base_url_allowlist,
                "ABSTRACTCORE_SERVER_URL_FETCH_ALLOWLIST": server.url_fetch_allowlist,
                "ABSTRACTCORE_SERVER_MEDIA_ROOT": server.media_root,
                "HOST": server.host,
                "PORT": str(server.port) if server.port is not None else None,
            }
            for env_var, value in value_map.items():
                if env_var == _SERVER_AUTH_TOKEN_ENV_VAR and os.environ.get(_SERVER_AUTH_TOKEN_ENV_VAR):
                    continue
                if value is not None and str(value).strip() and not os.environ.get(env_var):
                    os.environ[env_var] = str(value).strip()

            if server.allow_unauthenticated and not os.environ.get("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED"):
                os.environ["ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED"] = "1"
            if server.allow_local_files and not os.environ.get("ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES"):
                os.environ["ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES"] = "1"
        except Exception:
            # Never fail config initialization.
            pass

    def _read_store_document(self) -> Optional[Dict[str, Any]]:
        """The store EXACTLY as it is on disk right now, or `None`.

        `None` means "there is nothing trustworthy to merge against" -- the file
        is absent, unreadable, or not a JSON object -- and the caller then
        publishes its own document wholesale, which is the pre-merge behaviour.
        """
        try:
            if not self.config_file.exists():
                return None
            with open(self.config_file, "r") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _adopt_baseline_for_delete(self, *path: str) -> None:
        """Make an EXPLICIT delete of `path` survive the save-time merge.

        The merge reads "absent from mine, present on disk, absent from my
        baseline" as ANOTHER writer's row and keeps it -- which is right for a
        key this manager never knew about, and wrong for a key an operator just
        asked it to delete. `clear-default`/`delete-provider` say the key must
        go even if it appeared after this manager loaded, so the on-disk value
        is copied into the baseline first: the merge then sees a row this
        manager was carrying and dropped, which is exactly a delete.
        """
        if not path:
            return
        disk = self._read_store_document()
        if disk is None:
            return
        node: Any = disk
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return
            node = node[part]
        target = self._store_baseline
        for part in path[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[path[-1]] = copy.deepcopy(node)

    def _load_config(self) -> AbstractCoreConfig:
        """Load configuration from file or create default.

        CRITICAL DATA-SAFETY INVARIANT (incident 2026-07-11): an existing config
        file that fails to parse must NEVER be silently replaced by all-defaults.
        The old behavior returned `AbstractCoreConfig.default()` on any read
        error, and the next `_save_config()` then OVERWROTE the (recoverable)
        file with defaults — silently discarding the operator's settings,
        including capability routes like the entity embedding model. That was a
        reassertion vector for the stale embedding default: a partial/corrupt
        read → defaults → the framework's old `all-minilm-l6-v2` reappears with
        no warning. We now BACK UP the unreadable file (timestamped, once) and
        log LOUDLY before falling back, so nothing is lost and the operator sees
        it. Happy-path behavior is unchanged.
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                # Backward compatible: accept both top-level and nested flags.
                if isinstance(data, dict):
                    if "audio_strategy_explicit" in data:
                        self._audio_strategy_explicit = bool(data.get("audio_strategy_explicit"))
                    else:
                        nested = data.get("audio", {})
                        if isinstance(nested, dict) and "strategy_explicit" in nested:
                            self._audio_strategy_explicit = bool(nested.get("strategy_explicit"))
                return self._dict_to_config(data)
            except OSError as e:
                # THE FILE IS FINE; THE READ FAILED. A full disk, an EIO, a
                # momentarily unreadable mount — none of these are evidence that
                # the operator's config is bad, so falling back to defaults here
                # would be a guess, and the next _save_config() would publish
                # that guess OVER a perfectly good store. That is the exact
                # incident-2026-07-11 loss path, reached through a different
                # door. Observed 2026-08-02: a full volume produced a
                # `.corrupt-*.bak` quarantine of a file that parsed cleanly both
                # before and after.
                #
                # So: refuse to proceed. A loud failure is recoverable; a silent
                # defaults-regeneration is not. No quarantine copy either —
                # there is nothing wrong with the file to preserve.
                raise OSError(
                    f"Could not READ the AbstractCore config at {self.config_file}: {e}. "
                    "The file itself may be intact — this is an I/O failure (a full disk is "
                    "the usual cause). Refusing to continue with default settings, which "
                    "would overwrite your configuration on the next save."
                ) from e
            except Exception as e:
                # Never silently destroy a config we could not PARSE: preserve
                # it so the operator can recover the lost settings, and make the
                # degradation loud instead of a silent defaults regeneration.
                # Deliberately NO recommended-defaults seed on this branch: the
                # operator's settings are recoverable from the backup and must
                # not be shadowed by recommendations in the meantime.
                self._backup_unreadable_config(e)
                return AbstractCoreConfig.default()
        else:
            # FRESH INSTALL (no config file has ever existed here): seed the
            # framework's recommended defaults so a new install works out of
            # the box (operator ruling 2026-08-01: text qwen3.5-9b, voice
            # supertonic, image flux.2-klein-4b). Ordinary rows once written —
            # visible in every grid, overridable from either entry point,
            # always beaten by request pins. The `seeded` marker records the
            # provenance; file existence (not the marker) gates re-seeding.
            config = AbstractCoreConfig.default()
            from .capability_defaults import seed_recommended_capability_defaults

            seed_recommended_capability_defaults(config.capability_defaults)
            return config

    def _backup_unreadable_config(self, error: Exception) -> None:
        """Copy an unparseable config aside (timestamped) and warn loudly.

        Best-effort and never raises — a backup failure must not block startup,
        but the WARNING always fires so a silent regeneration can never happen
        unnoticed again."""
        backup_path: Optional[Path] = None
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = self.config_file.with_suffix(
                self.config_file.suffix + f".corrupt-{stamp}.bak"
            )
            # Copy raw bytes (not the parsed form) so nothing is lost/normalized.
            backup_path.write_bytes(self.config_file.read_bytes())
            try:
                os.chmod(backup_path, 0o600)
            except Exception:
                pass
        except Exception:
            backup_path = None
        try:
            import logging

            logging.getLogger(__name__).warning(
                "#FALLBACK abstractcore config at %s could not be parsed (%s); "
                "falling back to DEFAULTS for this session. The unreadable file "
                "was backed up to %s — recover your settings (provider/model, "
                "embedding model, capability routes) from it; a save will "
                "otherwise overwrite it with defaults.",
                str(self.config_file),
                error,
                str(backup_path) if backup_path else "(backup failed)",
            )
        except Exception:
            pass

    def _dict_to_config(self, data: Dict[str, Any]) -> AbstractCoreConfig:
        """Convert dictionary to config object."""
        # Create config objects from dictionary data
        vision = VisionConfig(**self._filter_dataclass_kwargs(VisionConfig, data.get('vision', {})))
        audio = AudioConfig(**self._filter_dataclass_kwargs(AudioConfig, data.get('audio', {})))
        video = VideoConfig(**self._filter_dataclass_kwargs(VideoConfig, data.get('video', {})))
        embeddings = EmbeddingsConfig(**self._filter_dataclass_kwargs(EmbeddingsConfig, data.get('embeddings', {})))
        app_defaults = AppDefaults(**self._filter_dataclass_kwargs(AppDefaults, data.get('app_defaults', {})))
        default_models = DefaultModels(**self._filter_dataclass_kwargs(DefaultModels, data.get('default_models', {})))
        capability_defaults = capability_defaults_from_dict(data.get('capability_defaults', {}))
        provider_profiles = provider_profiles_from_dict(data.get('provider_profiles', {}))
        api_keys = ApiKeysConfig(**self._filter_dataclass_kwargs(ApiKeysConfig, data.get('api_keys', {})))
        server = ServerConfig(**self._filter_dataclass_kwargs(ServerConfig, data.get('server', {})))
        cache = CacheConfig(**self._filter_dataclass_kwargs(CacheConfig, data.get('cache', {})))
        logging = LoggingConfig(**self._filter_dataclass_kwargs(LoggingConfig, data.get('logging', {})))
        streaming = StreamingConfig(**self._filter_dataclass_kwargs(StreamingConfig, data.get('streaming', {})))
        timeouts = TimeoutConfig(**self._filter_dataclass_kwargs(TimeoutConfig, data.get('timeouts', {})))
        offline = OfflineConfig(**self._filter_dataclass_kwargs(OfflineConfig, data.get('offline', {})))
        maintenance = MaintenanceConfig(**self._filter_dataclass_kwargs(MaintenanceConfig, data.get('maintenance', {})))
        email_cfg = EmailConfig(**self._filter_dataclass_kwargs(EmailConfig, data.get('email', {})))

        return AbstractCoreConfig(
            vision=vision,
            audio=audio,
            video=video,
            embeddings=embeddings,
            app_defaults=app_defaults,
            default_models=default_models,
            capability_defaults=capability_defaults,
            provider_profiles=provider_profiles,
            api_keys=api_keys,
            server=server,
            cache=cache,
            logging=logging,
            streaming=streaming,
            timeouts=timeouts,
            offline=offline,
            maintenance=maintenance,
            email=email_cfg,
        )

    def _save_config(self):
        """Save configuration to file.

        ONE STORE, MANY WRITERS. `abstractcore config`, the AbstractCore
        console-TUI, the AbstractCore server's PUT and the Gateway seam all
        write THIS file, and nothing serializes them. Publishing is atomic
        (write a temp file, `os.replace` it over the target), but the temp file
        MUST be unique per writer.

        A shared `abstractcore.json.tmp` corrupted the live config (reproduced
        2026-08-01, 8 concurrent writers: 41 of 1200 concurrent reads saw a
        truncated or interleaved file). The sequence: writer B truncates the
        shared temp while writer A is mid-`json.dump` into the same inode; A
        renames that inode over the config; B's still-open fd then keeps
        writing INTO THE LIVE CONFIG FILE. The next process to start reads a
        broken store, backs it up and falls back to defaults -- the operator's
        settings appear to vanish. A unique temp name makes every writer's
        bytes private until its own atomic publish, which is exactly
        last-writer-wins with no torn state.

        LAST-WRITER-WINS PER FIELD, NOT PER FILE (incident 2026-08-01, the
        operator's `output.image: mflux/flux2-klein-9b` row). Atomicity alone
        made the file always parseable and always WRONG in the same way: this
        method serialises the WHOLE in-memory config, so a manager that loaded
        the store at T0 and saved one unrelated field at T2 republished its T0
        snapshot over everything written in between. A long-lived process (a
        server, a console session, an entity loop) holds such a snapshot for
        hours; a single `set_default_timeout()` from it silently deleted a
        capability route another writer had added. Nothing warned, nothing
        failed, and the row simply was not there any more.

        So a save now publishes a THREE-WAY MERGE of (baseline, mine, disk):
        the baseline is the document this manager last saw on disk, `mine` is
        what it is about to write, `disk` is what is there right now. A field
        this manager changed wins; a field it did NOT change keeps whatever is
        on disk; a key it deliberately dropped (`clear-default`,
        `delete-provider`) stays dropped, because "absent from mine but present
        in the baseline unchanged" is a DELETE, not an omission. That is the
        per-field preservation `update_capability_default` gives one route,
        applied to the whole document -- every writer preserves what it does
        not manage. With no concurrent writer, `disk == baseline` and the merge
        is exactly `mine`, so the happy path is byte-identical to before.
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary
        config_dict = {
            # Meta flags (top-level for backward compatibility).
            "audio_strategy_explicit": bool(getattr(self, "_audio_strategy_explicit", False)),
            'vision': asdict(self.config.vision),
            'audio': asdict(self.config.audio),
            'video': asdict(self.config.video),
            'embeddings': asdict(self.config.embeddings),
            'app_defaults': asdict(self.config.app_defaults),
            'default_models': asdict(self.config.default_models),
            'capability_defaults': self.config.capability_defaults.to_dict(),
            'provider_profiles': self.config.provider_profiles.to_dict(),
            'api_keys': asdict(self.config.api_keys),
            'server': asdict(self.config.server),
            'cache': asdict(self.config.cache),
            'logging': asdict(self.config.logging),
            'streaming': asdict(self.config.streaming),
            'timeouts': asdict(self.config.timeouts),
            'offline': asdict(self.config.offline),
            'maintenance': asdict(self.config.maintenance),
            'email': asdict(self.config.email),
        }

        # The merge is a no-op when nothing else wrote since this manager
        # loaded; `_read_store_document()` returning None (absent/unreadable/
        # not an object) means there is nothing to merge against.
        disk = self._read_store_document()
        published = (
            config_dict
            if disk is None
            else merge_store_documents(self._store_baseline, config_dict, disk)
        )

        # Unique per writer (pid + a random token): same directory, so the
        # publish stays a same-filesystem `rename`, which is atomic.
        tmp = self.config_file.with_suffix(
            self.config_file.suffix + f".{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            with open(tmp, 'w') as f:
                json.dump(published, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(tmp, 0o600)
            except Exception:
                pass
            tmp.replace(self.config_file)
        except BaseException:
            # A failed save must not leave debris beside the store; the
            # config itself is untouched because nothing was published.
            try:
                tmp.unlink()
            except Exception:
                pass
            raise
        # The NEW baseline is what this manager INTENDED to write, not the
        # merged document: rows another writer contributed must stay "not
        # mine", or the next save would read their absence from `mine` as a
        # delete and revert them after all.
        self._store_baseline = copy.deepcopy(config_dict)
        try:
            os.chmod(self.config_file, 0o600)
        except Exception:
            pass

    def set_vision_provider(self, provider: str, model: str) -> bool:
        """Set vision provider and model."""
        try:
            self.config.vision.strategy = "two_stage"
            self.config.vision.caption_provider = provider
            self.config.vision.caption_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_audio_strategy(self, strategy: str) -> bool:
        """Set default audio handling strategy (native_only|speech_to_text|auto)."""
        raw = str(strategy or "").strip().lower()
        if raw in {"native"}:
            raw = "native_only"
        if raw in {"stt"}:
            raw = "speech_to_text"
        if raw not in {"native_only", "speech_to_text", "auto", "caption"}:
            return False
        try:
            self.config.audio.strategy = raw
            self._audio_strategy_explicit = True
            self._save_config()
            return True
        except Exception:
            return False

    def set_stt_backend_id(self, backend_id: Optional[str]) -> bool:
        """Set preferred STT backend id for capability plugins (optional)."""
        bid = str(backend_id or "").strip()
        try:
            self.config.audio.stt_backend_id = bid or None
            self._save_config()
            return True
        except Exception:
            return False

    def set_stt_language(self, language: Optional[str]) -> bool:
        """Set default STT language hint (optional)."""
        lang = str(language or "").strip()
        try:
            self.config.audio.stt_language = lang or None
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_strategy(self, strategy: str) -> bool:
        """Set default video handling strategy (native_only|frames_caption|auto)."""
        raw = str(strategy or "").strip().lower()
        if raw in {"native"}:
            raw = "native_only"
        if raw in {"frames", "frame_caption"}:
            raw = "frames_caption"
        if raw not in {"native_only", "frames_caption", "auto"}:
            return False
        try:
            self.config.video.strategy = raw
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_max_frames(self, max_frames: int) -> bool:
        """Set max sampled frames for video frames fallback (>= 1)."""
        try:
            n = int(max_frames)
        except Exception:
            return False
        if n < 1:
            return False
        try:
            self.config.video.max_frames = n
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_max_frames_native(self, max_frames_native: int) -> bool:
        """Set max frames for native video-capable models (>= 1)."""
        try:
            n = int(max_frames_native)
        except Exception:
            return False
        if n < 1:
            return False
        try:
            self.config.video.max_frames_native = n
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_frame_format(self, frame_format: str) -> bool:
        """Set extracted frame image format (jpg|png)."""
        raw = str(frame_format or "").strip().lower()
        if raw in {"jpeg"}:
            raw = "jpg"
        if raw not in {"jpg", "png"}:
            return False
        try:
            self.config.video.frame_format = raw
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_sampling_strategy(self, sampling_strategy: str) -> bool:
        """Set frame sampling strategy (uniform|keyframes)."""
        raw = str(sampling_strategy or "").strip().lower()
        if raw in {"keyframe"}:
            raw = "keyframes"
        if raw not in {"uniform", "keyframes"}:
            return False
        try:
            self.config.video.sampling_strategy = raw
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_max_frame_side(self, max_frame_side: int) -> bool:
        """Set max side length for extracted frames (preserves aspect ratio; never upscales)."""
        try:
            n = int(max_frame_side)
        except Exception:
            return False
        if n < 1:
            return False
        try:
            self.config.video.max_frame_side = n
            self._save_config()
            return True
        except Exception:
            return False

    def set_video_max_video_size_bytes(self, max_video_size_bytes: Optional[int]) -> bool:
        """Set maximum allowed video size for processing (bytes). Use 0/None to clear."""
        if max_video_size_bytes is None:
            value = None
        else:
            try:
                value_i = int(max_video_size_bytes)
            except Exception:
                return False
            value = None if value_i <= 0 else value_i

        try:
            self.config.video.max_video_size_bytes = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_vision_caption(self, model: str) -> bool:
        """Set vision caption model (deprecated)."""
        # Auto-detect provider from model name
        provider = self._detect_provider_from_model(model)
        if provider:
            return self.set_vision_provider(provider, model)
        return False

    def _detect_provider_from_model(self, model: str) -> Optional[str]:
        """Detect provider from model name."""
        model_lower = model.lower()

        if any(x in model_lower for x in ['qwen2.5vl', 'llama3.2-vision', 'llava']):
            return "ollama"
        elif any(x in model_lower for x in ['gpt-4', 'gpt-4o']):
            return "openai"
        elif any(x in model_lower for x in ['claude-3']):
            return "anthropic"
        elif '/' in model:
            return "lmstudio"

        return None

    def get_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        text_route = self.stored_capability_default("input", "text")
        embedding_route = self.get_capability_default("embedding", "text")
        embedding_route_configured = bool(
            embedding_route.get("provider")
            or embedding_route.get("model")
            or embedding_route.get("base_url")
            or embedding_route.get("options")
        )
        if embedding_route_configured:
            embedding_provider = embedding_route.get("provider")
            embedding_model = embedding_route.get("model")
            embedding_base_url = embedding_route.get("base_url")
            embedding_status = "✅ Ready" if embedding_provider and embedding_model else "⚠️ Partially configured"
            embedding_source = embedding_route.get("source") or "abstractcore.capability_defaults"
        else:
            embedding_provider = self.config.embeddings.provider
            embedding_model = self.config.embeddings.model
            embedding_base_url = self.config.embeddings.base_url
            embedding_status = "✅ Ready" if embedding_provider and embedding_model else "❌ Not configured"
            embedding_source = "abstractcore.embeddings_legacy"
        return {
            "config_file": str(self.config_file),
            "vision": {
                "strategy": self.config.vision.strategy,
                "status": "✅ Ready" if self.config.vision.caption_provider else "❌ Not configured",
                "caption_provider": self.config.vision.caption_provider,
                "caption_model": self.config.vision.caption_model
            },
            "audio": {
                "strategy": self.config.audio.strategy,
                "stt_backend_id": self.config.audio.stt_backend_id,
                "stt_language": self.config.audio.stt_language,
            },
            "video": {
                "strategy": self.config.video.strategy,
                "max_frames": self.config.video.max_frames,
                "max_frames_native": getattr(self.config.video, "max_frames_native", None),
                "frame_format": self.config.video.frame_format,
                "sampling_strategy": getattr(self.config.video, "sampling_strategy", None),
                "max_frame_side": getattr(self.config.video, "max_frame_side", None),
                "max_video_size_bytes": getattr(self.config.video, "max_video_size_bytes", None),
            },
            "app_defaults": {
                "cli": {
                    "provider": self.config.app_defaults.cli_provider,
                    "model": self.config.app_defaults.cli_model
                },
                "summarizer": {
                    "provider": self.config.app_defaults.summarizer_provider,
                    "model": self.config.app_defaults.summarizer_model
                },
                "extractor": {
                    "provider": self.config.app_defaults.extractor_provider,
                    "model": self.config.app_defaults.extractor_model
                },
                "judge": {
                    "provider": self.config.app_defaults.judge_provider,
                    "model": self.config.app_defaults.judge_model
                },
                "intent": {
                    "provider": self.config.app_defaults.intent_provider,
                    "model": self.config.app_defaults.intent_model
                }
            },
            # The text capability route is the store for the global default, so
            # status reports the route. `default_models.global_*` is the legacy
            # spelling the `--set-default-model` flag also writes; reporting it
            # directly would name a stale model any time the route was set by
            # `config set-default` or from the Gateway.
            "global_defaults": {
                "provider": text_route.get("provider") or self.config.default_models.global_provider,
                "model": text_route.get("model") or self.config.default_models.global_model,
                "reasoning": text_route.get("reasoning"),
                "source": "abstractcore.capability_defaults" if text_route else "abstractcore.default_models",
                "chat_model": self.config.default_models.chat_model,
                "code_model": self.config.default_models.code_model,
                "legacy": {
                    "provider": self.config.default_models.global_provider,
                    "model": self.config.default_models.global_model,
                },
            },
            "capability_defaults": self.list_capability_defaults(),
            "provider_profiles": self.list_provider_profiles(),
            "embeddings": {
                "status": embedding_status,
                "provider": embedding_provider,
                "model": embedding_model,
                "base_url": embedding_base_url,
                "route": "embedding.text",
                "source": embedding_source,
                "legacy": {
                    "provider": self.config.embeddings.provider,
                    "model": self.config.embeddings.model,
                    "base_url": self.config.embeddings.base_url,
                },
            },
            "streaming": {
                "cli_stream_default": self.config.streaming.cli_stream_default
            },
            "logging": {
                "console_level": self.config.logging.console_level,
                "file_level": self.config.logging.file_level,
                "file_logging_enabled": self.config.logging.file_logging_enabled
            },
            "timeouts": {
                "default_timeout": self.config.timeouts.default_timeout,
                "tool_timeout": self.config.timeouts.tool_timeout
            },
            "cache": {
                "default_cache_dir": self.config.cache.default_cache_dir,
                "huggingface_cache_dir": self.config.cache.huggingface_cache_dir,
                "local_models_cache_dir": self.config.cache.local_models_cache_dir,
                "glyph_cache_dir": self.config.cache.glyph_cache_dir,
            },
            "api_keys": {
                "openai": "✅ Set" if self.config.api_keys.openai else "❌ Not set",
                "anthropic": "✅ Set" if self.config.api_keys.anthropic else "❌ Not set",
                "openrouter": "✅ Set" if self.config.api_keys.openrouter else "❌ Not set",
                "portkey": "✅ Set" if self.config.api_keys.portkey else "❌ Not set",
                "openai-compatible": "✅ Set" if self.config.api_keys.openai_compatible else "❌ Not set",
                "vllm": "✅ Set" if self.config.api_keys.vllm else "❌ Not set",
                "google": "✅ Set" if self.config.api_keys.google else "❌ Not set"
            },
            "server": {
                "auth_token": "✅ Set" if self.config.server.auth_token else "❌ Not set",
                "allow_unauthenticated": bool(self.config.server.allow_unauthenticated),
                "auth_mode": (
                    "server_token"
                    if self.config.server.auth_token
                    else ("unauthenticated_dev" if self.config.server.allow_unauthenticated else "provider_key_only")
                ),
                "base_url_allowlist": self.config.server.base_url_allowlist,
                "url_fetch_allowlist": self.config.server.url_fetch_allowlist,
                "media_root": self.config.server.media_root,
                "allow_local_files": bool(self.config.server.allow_local_files),
                "host": self.config.server.host,
                "port": self.config.server.port,
            },
            "offline": {
                "offline_first": self.config.offline.offline_first,
                "allow_network": self.config.offline.allow_network,
                "status": "🔒 Offline-first" if self.config.offline.offline_first else "🌐 Network-enabled"
            }
        }

    def reset_configuration(self) -> bool:
        """Reset all configuration to built-in defaults."""
        try:
            self.config = AbstractCoreConfig.default()
            self._audio_strategy_explicit = False
            self._provider_config.clear()
            self._save_config()
            # Re-apply environment-aware defaults for the current process (non-persisted).
            self._apply_smart_defaults()
            return True
        except Exception:
            return False

    def list_provider_profiles(self, *, include_disabled: bool = True) -> list[Dict[str, Any]]:
        """Return redacted local provider endpoint profiles (runtime-injected included)."""
        rows: list[Dict[str, Any]] = []
        for profile in sorted(self.config.provider_profiles.profiles.values(), key=lambda p: p.id.lower()):
            if not include_disabled and not profile.enabled:
                continue
            rows.append(profile.public_dict())
        seen = {str(r.get("id") or "").lower() for r in rows}
        for pid, profile in sorted(self._runtime_provider_profiles.items()):
            if pid in seen:
                continue  # persisted profiles win on id collision
            if not include_disabled and not profile.enabled:
                continue
            row = profile.public_dict()
            row["source"] = "runtime"
            rows.append(row)
        return rows

    def get_provider_profile(self, profile_id: str) -> Optional[ProviderProfile]:
        """Return one provider profile by plain id or virtual provider id."""
        try:
            pid = normalize_profile_id(profile_id)
        except Exception:
            return None
        found = self.config.provider_profiles.profiles.get(pid.lower())
        if found is not None:
            return found
        return self._runtime_provider_profiles.get(pid.lower())

    def resolve_provider_profile(self, provider: str, *, require_enabled: bool = True) -> Optional[ProviderProfile]:
        """Resolve ``endpoint:<id>`` or a plain profile id to a provider profile."""
        try:
            raw = str(provider or "").strip()
            pid = profile_id_from_virtual_provider(raw) or normalize_profile_id(raw)
            profile = self.config.provider_profiles.profiles.get(pid.lower())
            if profile is None:
                profile = self._runtime_provider_profiles.get(pid.lower())
            if profile is None:
                return None
            if require_enabled and not profile.enabled:
                return None
            return profile
        except Exception:
            return None

    def register_runtime_provider_profile(self, profile: Union[Dict[str, Any], ProviderProfile]) -> ProviderProfile:
        """Register a PROCESS-LIFETIME provider endpoint profile (never persisted).

        Hosts use this to materialize externally-defined endpoint providers
        (e.g. profiles registered on an AbstractGateway) so `create_llm`
        resolves them like local `endpoint:<id>` profiles. The profile lives
        only in this process: it is stored outside `config.provider_profiles`
        so `_save_config()` can never leak it to disk, and persisted profiles
        always win on id collision. Validation errors raise (unsupported
        provider_family must fail loudly, never at first generate)."""
        if isinstance(profile, ProviderProfile):
            candidate = profile
        else:
            data = dict(profile or {})
            allowed = {f.name for f in fields(ProviderProfile)}
            candidate = ProviderProfile(**{k: v for k, v in data.items() if k in allowed})
        pid = normalize_profile_id(candidate.id)
        normalize_provider_family(candidate.provider_family)
        normalize_base_url(candidate.base_url)
        normalize_string_list(candidate.allowed_models)
        self._runtime_provider_profiles[pid.lower()] = candidate
        return candidate

    def clear_runtime_provider_profiles(self) -> None:
        """Drop all process-lifetime injected profiles."""
        self._runtime_provider_profiles.clear()

    def set_provider_profile(
        self,
        profile_id: str,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        provider_family: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        clear_api_key: bool = False,
        allowed_models: Optional[list[str]] = None,
        enabled: Optional[bool] = None,
        scope: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        created_at: Optional[str] = None,
    ) -> ProviderProfile:
        """Create or update a provider endpoint profile.

        `scope` and `capabilities` are the hosting columns a Gateway console
        sets on the same row (operator ruling 2026-08-01: a profile created
        from EITHER console is one profile, in this store). `created_at` exists
        for a migration importing a row that already has a history; ordinary
        callers leave it alone.
        """
        pid = normalize_profile_id(profile_id)
        existing = self.config.provider_profiles.profiles.get(pid.lower())
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        parsed_api_key = None
        parsed_api_key_env_var = None
        if api_key is not None:
            parsed_api_key, parsed_api_key_env_var = split_api_key_value(api_key)

        profile = ProviderProfile(
            id=pid,
            display_name=display_name if display_name is not None else (existing.display_name if existing else pid),
            description=description if description is not None else (existing.description if existing else ""),
            provider_family=provider_family if provider_family is not None else (existing.provider_family if existing else "openai-compatible"),
            base_url=base_url if base_url is not None else (existing.base_url if existing else ""),
            api_key=(
                ""
                if clear_api_key
                else (parsed_api_key if parsed_api_key is not None else (existing.api_key if existing else ""))
            ),
            api_key_env_var=(
                ""
                if clear_api_key
                else (parsed_api_key_env_var if parsed_api_key_env_var is not None else (existing.api_key_env_var if existing else ""))
            ),
            allowed_models=allowed_models if allowed_models is not None else (existing.allowed_models if existing else []),
            enabled=bool(enabled) if enabled is not None else (existing.enabled if existing else True),
            scope=scope if scope is not None else (existing.scope if existing else "gateway"),
            capabilities=capabilities if capabilities is not None else (list(existing.capabilities) if existing else None),
            created_at=(existing.created_at if existing else (created_at or now)),
            updated_at=now,
        )

        # Run explicit normalization here to surface errors before saving.
        normalize_provider_family(profile.provider_family)
        normalize_base_url(profile.base_url)
        normalize_string_list(profile.allowed_models)

        self.config.provider_profiles.profiles[profile.id.lower()] = profile
        self._save_config()
        return profile

    def delete_provider_profile(self, profile_id: str) -> bool:
        """Delete a local provider endpoint profile."""
        pid = normalize_profile_id(profile_id)
        removed = self.config.provider_profiles.profiles.pop(pid.lower(), None)
        if removed is None:
            return False
        self._adopt_baseline_for_delete("provider_profiles", "profiles", pid.lower())
        self._save_config()
        return True

    def get_capability_default(
        self,
        kind: str,
        modality: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the effective default route for a capability."""
        try:
            key = capability_route_key(*self._route_parts(kind, modality, task))
        except Exception:
            return {}

        if key == TEXT_ROUTE_KEY:
            route = self.config.capability_defaults.routes.get(TEXT_ROUTE_STORAGE_KEY)
            if route and route.configured():
                out = route.to_dict()
                out.update(
                    {
                        "key": key,
                        "source": "abstractcore.capability_defaults",
                        "derived_from": "input.text",
                        "read_only": True,
                    }
                )
                return out
            return {"key": key, "source": "not_configured", "derived_from": "input.text", "read_only": True}

        route = self.config.capability_defaults.routes.get(key)
        if route and route.configured():
            out = route.to_dict()
            out.update({"key": key, "source": "abstractcore.capability_defaults"})
            if key == capability_route_key("input", "image"):
                out = self._decorate_image_input_default(out)
            elif key == capability_route_key("input", "video"):
                out.setdefault("overrideable", True)
            elif key == capability_route_key("input", "sound"):
                out.setdefault("overrideable", True)
            elif key == capability_route_key("input", "music"):
                out.setdefault("overrideable", True)
            return out

        out = {"key": key, "source": "not_configured"}
        if key == capability_route_key("input", "image"):
            out = self._decorate_image_input_default(out)
        elif key == capability_route_key("input", "video"):
            out = self._decorate_video_input_default(out)
        elif key == capability_route_key("input", "sound"):
            out = self._decorate_sound_input_default(out)
        elif key == capability_route_key("input", "music"):
            out = self._decorate_music_input_default(out)
        return out

    def apply_recommended_capability_defaults(
        self,
        *,
        only: Optional[list] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Bring THIS store's routes to the framework recommendation.

        The fresh-install seed never touches an existing store, which is right
        for safety and left "make my machine match the recommendation" with no
        product answer at all (operator report 2026-08-01: "I asked for
        qwen3.5-9b everywhere, I see qwen3-0.6b"). This is that answer, and it
        is the same action at every surface -- CLI, Gateway endpoint, both
        console-TUIs -- because it is implemented once, here.

        A route the operator configured DIFFERENTLY is kept and reported, never
        silently replaced; `force` is the explicit "yes, overrule me". Extra
        fields on the row (a pinned `base_url`, a reasoning effort, plugin
        options) are preserved in every case: they describe this machine, not
        the recommendation. `dry_run` plans and writes nothing.
        """

        from .capability_defaults import plan_recommended_capability_defaults

        plan = plan_recommended_capability_defaults(
            self.config.capability_defaults.routes, only=only, force=force
        )
        applied: list = []
        for row in plan:
            if row["changed"] and not dry_run:
                self.update_capability_default(
                    row["key"],
                    provider=row["recommended"].get("provider"),
                    model=row["recommended"].get("model"),
                )
            applied.append(dict(row))
        return {
            "ok": True,
            "dry_run": bool(dry_run),
            "force": bool(force),
            "config_file": str(self.config_file),
            "changed": sum(1 for row in applied if row["changed"]),
            "kept": sum(1 for row in applied if row["action"] == "kept"),
            "already": sum(1 for row in applied if row["action"] == "already"),
            "routes": applied,
        }

    def list_capability_defaults(self) -> list[Dict[str, Any]]:
        """Return all known capability routes with explicit persisted defaults."""
        rows: list[Dict[str, Any]] = []
        for spec in iter_capability_default_specs():
            route = self.get_capability_default(spec.key)
            row = {**spec.to_dict(), **route}
            # `configured` mirrors `CapabilityRouteDefault.configured()` field for
            # field, reasoning included: a route that carries only a reasoning
            # default is a configured route, and a row that claimed otherwise
            # would hide the operator's setting from every reader of this grid.
            row["configured"] = bool(
                route.get("provider")
                or route.get("model")
                or route.get("base_url")
                or route.get("reasoning")
                or route.get("options")
            )
            if spec.key == capability_route_key("input", "image"):
                row = self._decorate_image_input_default(row)
            elif spec.key == capability_route_key("input", "video") and not row["configured"]:
                row = self._decorate_video_input_default(row)
            elif spec.key == capability_route_key("input", "sound") and not row["configured"]:
                row = self._decorate_sound_input_default(row)
            elif spec.key == capability_route_key("input", "music") and not row["configured"]:
                row = self._decorate_music_input_default(row)
            rows.append(row)
        return self._decorate_route_hierarchy(rows)

    @staticmethod
    def _decorate_route_hierarchy(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Stamp the parent/child facts every surface needs onto the grid.

        ONE DERIVATION, FOUR GRIDS. `output.image` is the PARENT of
        `output.image.*` -- the value that answers every image task without a row
        of its own -- and all four surfaces (web console, both console-TUIs, the
        CLI) used to draw it as a flat sibling ABOVE its own children with a red
        "not configured", which is what made an operator ask whether the row was
        dead code. Deriving it here means no surface re-derives it and they
        cannot drift.

        Fields added:
          `broad_key`      on a task row: the modality cell it falls back to
          `task_keys`      on a modality cell: the task rows that override it
          `covered_by_tasks`  on an UNSET modality cell whose task rows are all
                           configured -- the state is then benign ("not needed"),
                           because nothing can reach the parent. Deliberately NOT
                           `covered_by`, which drives read-only/editability: this
                           row stays editable, since setting it is the simple
                           path for an operator who wants one image model.
          `inherits_broad` on an UNSET task row whose parent IS configured -- the
                           MIRROR of the same confusion, and the shape a fresh
                           install has: the seed writes `output.image` alone, so
                           three red "not configured" task rows sat under a
                           working parent and read as "image editing is not set
                           up" when the parent answers every one of them.
        """

        by_key = {str(row.get("key") or ""): row for row in rows}
        for row in rows:
            key = str(row.get("key") or "")
            broad_key = capability_route_broad_key(key)
            if broad_key:
                row["broad_key"] = broad_key
                if not row.get("configured") and by_key.get(broad_key, {}).get("configured"):
                    row["inherits_broad"] = True
                continue
            task_keys = capability_route_task_keys(key)
            if not task_keys:
                continue
            row["task_keys"] = list(task_keys)
            if not row.get("configured") and capability_route_tasks_cover_broad(key, by_key):
                row["covered_by_tasks"] = True
        return rows

    def _decorate_image_input_default(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return self._decorate_text_covered_input_default(row, "image", read_only=True)

    def _decorate_video_input_default(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return self._decorate_text_covered_input_default(
            row,
            "video",
            coverage_mode="video_frames",
            overrideable=True,
            read_only=False,
        )

    def _decorate_sound_input_default(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return self._decorate_text_covered_input_default(row, "sound", overrideable=True, read_only=False)

    def _decorate_music_input_default(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return self._decorate_text_covered_input_default(row, "music", overrideable=True, read_only=False)

    def _decorate_text_covered_input_default(
        self,
        row: Dict[str, Any],
        modality: str,
        *,
        coverage_mode: Optional[str] = None,
        overrideable: bool = False,
        read_only: bool = True,
    ) -> Dict[str, Any]:
        text_route = self.config.capability_defaults.routes.get(TEXT_ROUTE_STORAGE_KEY)
        if not text_route or not text_route.provider or not text_route.model:
            return row
        if not self._model_supports_input(text_route.model, modality):
            return row
        out = dict(row)
        out.update(
            {
                "provider": text_route.provider,
                "model": text_route.model,
                "base_url": text_route.base_url,
                "source": "abstractcore.capability_defaults",
                "configured": True,
                "covered_by": "input.text",
                "read_only": read_only,
            }
        )
        if coverage_mode:
            out["coverage_mode"] = coverage_mode
        if overrideable:
            out["overrideable"] = True
        if text_route.options:
            out["options"] = dict(text_route.options)
        return out

    @staticmethod
    def _model_supports_input(model: str, modality: str) -> bool:
        try:
            from ..providers.model_capabilities import (
                ModelInputCapability,
                model_matches_input_capabilities,
                model_supports_capability_route,
            )

            normalized_modality = str(modality or "").strip().lower()
            try:
                if model_supports_capability_route(str(model or ""), capability_route_key("input", normalized_modality)):
                    return True
            except Exception:
                pass

            capability = {
                "image": ModelInputCapability.IMAGE,
                "audio": ModelInputCapability.AUDIO,
                "sound": ModelInputCapability.SOUND,
                "voice": ModelInputCapability.VOICE,
                "music": ModelInputCapability.MUSIC,
                "video": ModelInputCapability.VIDEO,
            }.get(normalized_modality)
            if capability is None:
                return False
            return bool(model_matches_input_capabilities(str(model or ""), [capability]))
        except Exception:
            return False

    @staticmethod
    def _route_parts(kind: str, modality: Optional[str], task: Optional[str]) -> Tuple[str, str, Optional[str]]:
        """Normalize a route named either as `"output.text"` or as `(kind, modality)`."""
        if modality is None:
            return split_capability_default_route(kind)
        return split_capability_default_route(kind, modality, task)

    def storage_capability_route_key(self, kind: str, modality: Optional[str] = None, task: Optional[str] = None) -> str:
        """The key a capability route is STORED under.

        `output.text` is the name callers use; AbstractCore canonicalizes it to
        the storage key `input.text`, and every writer must agree on that or two
        rows describe one route. THE ONE implementation of that rule: the
        set/clear paths resolve their key through here rather than repeating it.
        """
        kind, modality, task = self._route_parts(kind, modality, task)
        key = capability_route_key(kind, modality, task)
        return TEXT_ROUTE_STORAGE_KEY if key == TEXT_ROUTE_KEY else key

    def stored_capability_default(
        self,
        kind: str,
        modality: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The route row exactly as it is PERSISTED, or `{}`.

        Unlike `get_capability_default`, this never returns a row derived from
        another route (`input.image` covered by `input.text`, say). A partial
        update must merge over what is stored, or it would persist a derivation
        as if an operator had asked for it.
        """
        try:
            key = self.storage_capability_route_key(kind, modality, task)
        except Exception:
            return {}
        route = self.config.capability_defaults.routes.get(key)
        if route is None or not route.configured():
            return {}
        return route.to_dict()

    def update_capability_default(
        self,
        kind: str,
        modality: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> bool:
        """Persist a capability default route, PRESERVING the fields not named.

        A route is stored as a whole row, so a writer that named only `provider`
        and `model` would clear every other field of that row. AbstractCore has
        more than one writer -- the `abstractcore config` CLI, the AbstractCore
        server's config routes, and AbstractGateway through them -- and if they
        do not share this rule they silently overwrite each other's settings.

        Pass a field to change it; pass `""` to clear it; leave it `None` and it
        keeps whatever is stored. `clear_capability_default` drops the whole row.
        """
        stored = self.stored_capability_default(kind, modality, task)
        if not stored:
            return self.set_capability_default(
                kind,
                modality,
                provider=provider,
                model=model,
                base_url=base_url,
                reasoning=reasoning,
                options=options,
                task=task,
            )
        merged_options = options if isinstance(options, dict) else stored.get("options")
        return self.set_capability_default(
            kind,
            modality,
            provider=stored.get("provider") if provider is None else provider,
            model=stored.get("model") if model is None else model,
            base_url=stored.get("base_url") if base_url is None else base_url,
            reasoning=stored.get("reasoning") if reasoning is None else reasoning,
            options=dict(merged_options) if isinstance(merged_options, dict) else {},
            task=task,
        )

    def set_capability_default(
        self,
        kind: str,
        modality: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        reasoning: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
    ) -> bool:
        """Persist a capability default route as a WHOLE ROW.

        Every field the caller leaves `None` is cleared. Use
        `update_capability_default` for a partial write that keeps the fields it
        does not name.

        A FAILURE CARRIES ITS REASON. This used to swallow every exception into
        a bare `False`, and every surface above it then printed a reason-free
        "Failed to set capability default <route>": the CLI, the Gateway's HTTP
        400, both TUIs. A write that fails once and cannot be reproduced is
        undiagnosable that way (reported 2026-08-01). Now it raises
        `CapabilityDefaultWriteError` naming the route, with the original
        exception as `__cause__`; the boolean return is kept for callers that
        test it and is only ever `True`.
        """
        try:
            kind, modality, task = self._route_parts(kind, modality, task)
            key = self.storage_capability_route_key(kind, modality, task)
            route = CapabilityRouteDefault(
                provider=str(provider).strip() if isinstance(provider, str) and provider.strip() else None,
                model=str(model).strip() if isinstance(model, str) and model.strip() else None,
                base_url=str(base_url).strip() if isinstance(base_url, str) and base_url.strip() else None,
                reasoning=str(reasoning).strip().lower() if isinstance(reasoning, str) and reasoning.strip() else None,
                options=dict(options or {}) if isinstance(options, dict) else {},
            )
            if route.configured():
                self.config.capability_defaults.routes[key] = route
                if key == TEXT_ROUTE_STORAGE_KEY:
                    self.config.capability_defaults.routes.pop(TEXT_ROUTE_KEY, None)
            else:
                self.config.capability_defaults.routes.pop(key, None)
                self._adopt_baseline_for_delete("capability_defaults", "routes", key)
            self._sync_embeddings_from_capability_default(key)
            self._save_config()
            return True
        except CapabilityDefaultWriteError:
            raise
        except Exception as exc:
            raise _capability_write_error("set", kind, modality, task, exc) from exc

    def clear_capability_default(self, kind: str, modality: Optional[str] = None, task: Optional[str] = None) -> bool:
        """Clear one persisted capability default route.

        Raises `CapabilityDefaultWriteError` on failure -- same reason as
        `set_capability_default`: a reason-free `False` is undiagnosable at
        every surface above it.
        """
        try:
            kind, modality, task = self._route_parts(kind, modality, task)
            key = self.storage_capability_route_key(kind, modality, task)
            self.config.capability_defaults.routes.pop(key, None)
            self._adopt_baseline_for_delete("capability_defaults", "routes", key)
            if key == TEXT_ROUTE_STORAGE_KEY:
                self.config.capability_defaults.routes.pop(TEXT_ROUTE_KEY, None)
                self._adopt_baseline_for_delete("capability_defaults", "routes", TEXT_ROUTE_KEY)
            self._sync_embeddings_from_capability_default(key)
            self._save_config()
            return True
        except CapabilityDefaultWriteError:
            raise
        except Exception as exc:
            raise _capability_write_error("clear", kind, modality, task, exc) from exc

    def set_global_default_model(self, provider_model: str) -> bool:
        """Set the global default model, in provider/model format.

        This is the legacy spelling of "set the text-generation route". The
        capability route is the store, so the write goes there and keeps the
        rest of that row: naming a model must not discard the reasoning effort,
        base URL or options the route already carries.
        """
        try:
            provider, model = _split_provider_model(provider_model, default_provider="ollama")

            self.config.default_models.global_provider = provider
            self.config.default_models.global_model = model
            key = TEXT_ROUTE_STORAGE_KEY
            stored = self.config.capability_defaults.routes.get(key)
            self.config.capability_defaults.routes[key] = CapabilityRouteDefault(
                provider=provider,
                model=model,
                base_url=stored.base_url if stored else None,
                reasoning=stored.reasoning if stored else None,
                options=dict(stored.options) if stored and stored.options else {},
            )
            self.config.capability_defaults.routes.pop(TEXT_ROUTE_KEY, None)
            self._save_config()
            return True
        except Exception:
            return False

    def set_default_model(self, provider_model: str) -> bool:
        """Legacy alias for setting the global default model."""
        return self.set_global_default_model(provider_model)

    def set_global_default_provider(self, provider: str) -> bool:
        """Set global default provider (legacy)."""
        try:
            provider = str(provider or "").strip()
            if not provider:
                raise ValueError("Provider cannot be empty")
            self.config.default_models.global_provider = provider
            self._save_config()
            return True
        except Exception:
            return False

    def set_chat_model(self, provider_model: str) -> bool:
        """Set specialized chat model (provider/model string)."""
        try:
            model = str(provider_model or "").strip()
            if not model:
                raise ValueError("Model cannot be empty")
            self.config.default_models.chat_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_code_model(self, provider_model: str) -> bool:
        """Set specialized code model (provider/model string)."""
        try:
            model = str(provider_model or "").strip()
            if not model:
                raise ValueError("Model cannot be empty")
            self.config.default_models.code_model = model
            self._save_config()
            return True
        except Exception:
            return False

    def set_embeddings_model(self, provider_model: str) -> bool:
        """Set embeddings provider/model from a provider/model string (preferred)."""
        try:
            value = str(provider_model or "").strip()
            if not value:
                raise ValueError("Embeddings model cannot be empty")

            provider, model = _split_provider_model(value, default_provider=str(self.config.embeddings.provider or "huggingface"))
            self.config.embeddings.provider = provider.strip() or self.config.embeddings.provider
            self.config.embeddings.model = model.strip()

            self._sync_embedding_capability_default()
            self._save_config()
            return True
        except Exception:
            return False

    def set_embeddings_provider(self, provider: str) -> bool:
        """Set embeddings provider."""
        try:
            value = str(provider or "").strip()
            if not value:
                raise ValueError("Embeddings provider cannot be empty")
            self.config.embeddings.provider = value
            self._sync_embedding_capability_default()
            self._save_config()
            return True
        except Exception:
            return False

    def set_embeddings_base_url(self, base_url: Optional[str]) -> bool:
        """Set optional embeddings provider base URL."""
        try:
            value = str(base_url or "").strip().rstrip("/")
            self.config.embeddings.base_url = value or None
            self._sync_embedding_capability_default()
            self._save_config()
            return True
        except Exception:
            return False

    def _sync_embedding_capability_default(self) -> None:
        """Mirror `embeddings` onto the `embedding.text` capability route.

        The legacy `embeddings` section names a provider, a model and a base
        URL. The route it mirrors onto can carry more than that, so the mirror
        keeps the fields it has nothing to say about: setting an embeddings
        model must not drop plugin options set on the same route.
        """
        key = capability_route_key("embedding", "text")
        stored = self.config.capability_defaults.routes.get(key)
        route = CapabilityRouteDefault(
            provider=self.config.embeddings.provider,
            model=self.config.embeddings.model,
            base_url=self.config.embeddings.base_url,
            reasoning=stored.reasoning if stored else None,
            options=dict(stored.options) if stored and stored.options else {},
        )
        if route.configured():
            self.config.capability_defaults.routes[key] = route
        else:
            self.config.capability_defaults.routes.pop(key, None)

    def _sync_embeddings_from_capability_default(self, key: str) -> None:
        """Mirror the `embedding.text` capability route back onto `embeddings`.

        The capability route is the store the whole framework routes on, and the
        `embeddings` section is the shape the embeddings commands and
        `--show-config` read. The mirror runs in both directions so the two
        never report different answers for the same question, whichever entry
        point wrote last.
        """
        if key != capability_route_key("embedding", "text"):
            return
        route = self.config.capability_defaults.routes.get(key)
        if route is None:
            self.config.embeddings.provider = None
            self.config.embeddings.model = None
            self.config.embeddings.base_url = None
            return
        self.config.embeddings.provider = route.provider
        self.config.embeddings.model = route.model
        self.config.embeddings.base_url = route.base_url

    def set_default_cache_dir(self, path: str) -> bool:
        """Set default cache directory for AbstractCore."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("Cache directory cannot be empty")
            self.config.cache.default_cache_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_huggingface_cache_dir(self, path: str) -> bool:
        """Set HuggingFace cache directory."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("HuggingFace cache directory cannot be empty")
            self.config.cache.huggingface_cache_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_local_models_cache_dir(self, path: str) -> bool:
        """Set local models cache directory."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("Local models cache directory cannot be empty")
            self.config.cache.local_models_cache_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_log_base_dir(self, path: str) -> bool:
        """Set log base directory."""
        try:
            value = str(path or "").strip()
            if not value:
                raise ValueError("Log base directory cannot be empty")
            self.config.logging.log_base_dir = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_console_log_level(self, level: str) -> bool:
        """Set console logging level."""
        try:
            value = str(level or "").strip().upper()
            if not value:
                raise ValueError("Console log level cannot be empty")
            self.config.logging.console_level = value
            self._save_config()
            return True
        except Exception:
            return False

    def set_file_log_level(self, level: str) -> bool:
        """Set file logging level."""
        try:
            value = str(level or "").strip().upper()
            if not value:
                raise ValueError("File log level cannot be empty")
            self.config.logging.file_level = value
            self._save_config()
            return True
        except Exception:
            return False

    def enable_debug_logging(self) -> bool:
        """Enable debug logging for both console and file."""
        try:
            self.config.logging.console_level = "DEBUG"
            self.config.logging.file_level = "DEBUG"
            self._save_config()
            return True
        except Exception:
            return False

    def disable_console_logging(self) -> bool:
        """Disable console logging output."""
        try:
            self.config.logging.console_level = "NONE"
            self._save_config()
            return True
        except Exception:
            return False

    def enable_file_logging(self) -> bool:
        """Enable file logging."""
        try:
            self.config.logging.file_logging_enabled = True
            self._save_config()
            return True
        except Exception:
            return False

    def disable_file_logging(self) -> bool:
        """Disable file logging."""
        try:
            self.config.logging.file_logging_enabled = False
            self._save_config()
            return True
        except Exception:
            return False

    def set_streaming_default(self, app_name: str, enabled: bool) -> bool:
        """Set default streaming behavior for a given app (currently: cli)."""
        try:
            app = str(app_name or "").strip().lower()
            if app != "cli":
                return False
            self.config.streaming.cli_stream_default = bool(enabled)
            self._save_config()
            return True
        except Exception:
            return False

    def get_streaming_default(self, app_name: str) -> bool:
        """Get default streaming behavior for a given app (currently: cli)."""
        app = str(app_name or "").strip().lower()
        if app == "cli":
            return bool(self.config.streaming.cli_stream_default)
        return False

    def enable_cli_streaming(self) -> bool:
        """Enable streaming by default for the CLI."""
        return self.set_streaming_default("cli", True)

    def disable_cli_streaming(self) -> bool:
        """Disable streaming by default for the CLI."""
        return self.set_streaming_default("cli", False)

    def add_vision_fallback(self, provider: str, model: str) -> bool:
        """Add a vision fallback provider/model to the chain."""
        try:
            provider_val = str(provider or "").strip()
            model_val = str(model or "").strip()
            if not provider_val or not model_val:
                raise ValueError("Provider and model are required")

            self.config.vision.fallback_chain.append({"provider": provider_val, "model": model_val})
            # If vision is configured at all, assume two_stage (caption -> text model).
            if not self.config.vision.strategy or self.config.vision.strategy == "disabled":
                self.config.vision.strategy = "two_stage"
            self._save_config()
            return True
        except Exception:
            return False

    def disable_vision(self) -> bool:
        """Disable vision fallback for text-only models."""
        try:
            self.config.vision.strategy = "disabled"
            self.config.vision.caption_provider = None
            self.config.vision.caption_model = None
            self.config.vision.fallback_chain = []
            self._save_config()
            return True
        except Exception:
            return False


    def set_app_default(self, app_name: str, provider: str, model: str) -> bool:
        """Set app-specific default provider and model."""
        try:
            if app_name == "cli":
                self.config.app_defaults.cli_provider = provider
                self.config.app_defaults.cli_model = model
            elif app_name == "summarizer":
                self.config.app_defaults.summarizer_provider = provider
                self.config.app_defaults.summarizer_model = model
            elif app_name == "extractor":
                self.config.app_defaults.extractor_provider = provider
                self.config.app_defaults.extractor_model = model
            elif app_name == "judge":
                self.config.app_defaults.judge_provider = provider
                self.config.app_defaults.judge_model = model
            elif app_name == "intent":
                self.config.app_defaults.intent_provider = provider
                self.config.app_defaults.intent_model = model
            else:
                raise ValueError(f"Unknown app: {app_name}")

            self._save_config()
            return True
        except Exception:
            return False

    def set_api_key(self, provider: str, key: str) -> bool:
        """Set API key for a provider."""
        try:
            provider_key = str(provider or "").strip().lower().replace("-", "_")
            if provider_key not in {f.name for f in fields(ApiKeysConfig)}:
                return False
            setattr(self.config.api_keys, provider_key, key)

            self._save_config()
            self._apply_api_keys_to_env()
            return True
        except Exception:
            return False

    def set_server_auth_token(self, token: Optional[str]) -> bool:
        """Set or clear the HTTP server auth token."""
        try:
            value = str(token or "").strip()
            self.config.server.auth_token = value or None
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def set_server_allow_unauthenticated(self, enabled: bool) -> bool:
        """Allow unauthenticated HTTP server requests for explicit local/dev use."""
        try:
            self.config.server.allow_unauthenticated = bool(enabled)
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def set_server_base_url_allowlist(self, allowlist: Optional[str]) -> bool:
        """Set request-level provider base_url override allowlist."""
        try:
            value = str(allowlist or "").strip()
            self.config.server.base_url_allowlist = value or None
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def set_server_url_fetch_allowlist(self, allowlist: Optional[str]) -> bool:
        """Set URL media fetch allowlist for otherwise blocked non-public targets."""
        try:
            value = str(allowlist or "").strip()
            self.config.server.url_fetch_allowlist = value or None
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def set_server_media_root(self, path: Optional[str]) -> bool:
        """Set the safe local media root for HTTP requests."""
        try:
            value = str(path or "").strip()
            self.config.server.media_root = value or None
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def set_server_allow_local_files(self, enabled: bool) -> bool:
        """Enable or disable unsafe unrestricted local file access for HTTP requests."""
        try:
            self.config.server.allow_local_files = bool(enabled)
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def set_server_bind(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        """Set default bind host/port for `python -m abstractcore.server.app`."""
        try:
            if host is not None:
                host_value = str(host or "").strip()
                self.config.server.host = host_value or None
            if port is not None:
                port_i = int(port)
                if port_i < 1 or port_i > 65535:
                    return False
                self.config.server.port = port_i
            self._save_config()
            self._apply_server_config_to_env()
            return True
        except Exception:
            return False

    def get_app_default(self, app_name: str) -> Tuple[str, str]:
        """Get default provider and model for an app."""
        app_defaults = self.config.app_defaults

        if app_name == "cli":
            return app_defaults.cli_provider, app_defaults.cli_model
        elif app_name == "summarizer":
            return app_defaults.summarizer_provider, app_defaults.summarizer_model
        elif app_name == "extractor":
            return app_defaults.extractor_provider, app_defaults.extractor_model
        elif app_name == "judge":
            return app_defaults.judge_provider, app_defaults.judge_model
        elif app_name == "intent":
            return app_defaults.intent_provider, app_defaults.intent_model
        else:
            # Return default fallback
            return "huggingface", "unsloth/Qwen3-4B-Instruct-2507-GGUF"

    def set_default_timeout(self, timeout: float) -> bool:
        """Set default HTTP request timeout in seconds."""
        try:
            # #[WARNING:TIMEOUT]
            # Contract: allow `0` to mean "unlimited" (the provider layer normalizes <=0 to None).
            timeout_f = float(timeout)
            if timeout_f < 0:
                raise ValueError("Timeout must be >= 0 (0 = unlimited)")
            self.config.timeouts.default_timeout = timeout_f
            self._save_config()
            return True
        except Exception:
            return False

    def set_tool_timeout(self, timeout: float) -> bool:
        """Set tool execution timeout in seconds."""
        try:
            # #[WARNING:TIMEOUT]
            # Contract: allow `0` to mean "unlimited".
            timeout_f = float(timeout)
            if timeout_f < 0:
                raise ValueError("Timeout must be >= 0 (0 = unlimited)")
            self.config.timeouts.tool_timeout = timeout_f
            self._save_config()
            return True
        except Exception:
            return False

    def get_default_timeout(self) -> float:
        """Get default HTTP request timeout in seconds."""
        return self.config.timeouts.default_timeout

    def get_tool_timeout(self) -> float:
        """Get tool execution timeout in seconds."""
        return self.config.timeouts.tool_timeout

    def set_offline_first(self, enabled: bool) -> bool:
        """Enable or disable offline-first mode."""
        try:
            self.config.offline.offline_first = enabled
            self._save_config()
            return True
        except Exception:
            return False

    def set_allow_network(self, enabled: bool) -> bool:
        """Allow network access when in offline-first mode."""
        try:
            self.config.offline.allow_network = enabled
            self._save_config()
            return True
        except Exception:
            return False

    def is_offline_first(self) -> bool:
        """Check if offline-first mode is enabled."""
        return self.config.offline.offline_first

    def is_network_allowed(self) -> bool:
        """Check if network access is allowed."""
        return self.config.offline.allow_network

    def should_force_local_files_only(self) -> bool:
        """Check if local_files_only should be forced for transformers."""
        return self.config.offline.force_local_files_only

    def configure_provider(self, provider: str, **kwargs) -> None:
        """
        Configure runtime settings for a provider.

        Args:
            provider: Provider name ('ollama', 'lmstudio', 'openai', 'anthropic')
            **kwargs: Configuration options (base_url, timeout, etc.)

        Example:
            configure_provider('ollama', base_url='http://192.168.1.100:11434')
        """
        provider = provider.lower()
        if provider not in self._provider_config:
            self._provider_config[provider] = {}

        for key, value in kwargs.items():
            if value is None:
                # Remove config (revert to env var / default)
                self._provider_config[provider].pop(key, None)
            else:
                self._provider_config[provider][key] = value

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get runtime configuration for a provider.

        Args:
            provider: Provider name

        Returns:
            Dict with configured settings, or empty dict if no config
        """
        provider_key = str(provider or "").strip().lower()
        runtime_config = self._provider_config.get(provider_key, {}).copy()

        profile = self.resolve_provider_profile(provider_key)
        if profile is None:
            return runtime_config

        profile_resolution = profile.private_resolution()
        profile_config = {
            "provider_family": profile_resolution.get("provider_family"),
            "base_url": profile_resolution.get("base_url"),
            "api_key": profile_resolution.get("api_key"),
            "allowed_models": profile_resolution.get("allowed_models") or [],
            "virtual_provider": profile_resolution.get("virtual_provider"),
            "provider_profile_id": profile_resolution.get("id"),
        }
        profile_config = {k: v for k, v in profile_config.items() if v not in (None, "", [])}
        return {**profile_config, **runtime_config}

    def clear_provider_config(self, provider: Optional[str] = None) -> None:
        """
        Clear runtime provider configuration.

        Args:
            provider: Provider name, or None to clear all
        """
        if provider is None:
            self._provider_config.clear()
        else:
            self._provider_config.pop(provider.lower(), None)


# Global instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager
