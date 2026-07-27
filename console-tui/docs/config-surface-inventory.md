# AbstractCore Config-Surface Inventory

**Purpose**: authoritative reference for building the AbstractCore configuration console TUI
(`abstractcore/console-tui`, Rust on abstracttui). Read-only probe of
`/Users/albou/tmp/abstractframework/abstractcore` (abstractcore **2.13.38**,
`abstractcore/utils/version.py:13`; `requires-python >=3.9`, `pyproject.toml:49`).
Every claim cites `file:line` in the abstractcore repo. Line numbers are from the working tree
on 2026-07-25.

**TL;DR integration verdict**: the TUI is a **local config editor**. Read state from the config
JSON file directly; perform **all writes through `abstractcore` CLI subprocesses** where a setter
exists (they enforce coupled-field invariants the file format does not express); direct
read-modify-write (unknown-keys-preserving, tmp+rename, 0600) **only** for the handful of
config-only fields that have no CLI setter. Use the HTTP server only as an *optional* live-test
target when detected — it is not a config API. Details in §10.

---

## 1. Where config lives

| Fact | Value | Citation |
|---|---|---|
| Default path | `~/.abstractcore/config/abstractcore.json` | `abstractcore/config/manager.py:348-349` |
| Path overrides | `ABSTRACTCORE_CONFIG_FILE` (full path) wins over `ABSTRACTCORE_CONFIG_DIR` (dir containing `abstractcore.json`); constructor args win over both | `manager.py:338-349` |
| Format | JSON, `indent=2`, trailing newline. **No comments possible, none preserved** | `manager.py:637-639` |
| Owner module | `abstractcore.config.manager.ConfigurationManager` (singleton via `get_config_manager()`) | `manager.py:325, 1868-1877` |
| Schema | Nested `@dataclass`es composed into `AbstractCoreConfig` (17 sections, §3) | `manager.py:73-322` |
| Write path | Whole-file rewrite from dataclasses on **every** setter: `tmp = <file>.tmp` → `json.dump` → `chmod 0600` → `tmp.replace(file)` → `chmod 0600`. Atomic rename; fixed tmp name (two concurrent writers can collide on the tmp file, but rename keeps the target file always-valid) | `manager.py:609-648` |
| File perms | `0600` on tmp and final | `manager.py:640-648` |
| Locking | **None.** No flock/fcntl anywhere in the config package (only `utils/data_registry.py` uses locking, unrelated). Concurrent writers are last-writer-wins | grep of `abstractcore/` for `fcntl|flock|lockf|FileLock` |
| Missing file | `AbstractCoreConfig.default()` in memory; nothing written until the first setter runs | `manager.py:529-530` |
| Partial file | Missing sections → section defaults; **unknown keys inside a known section are silently dropped at load** (`_filter_dataclass_kwargs` keeps only dataclass field names); unknown top-level sections are ignored at load and **not written back** (save rebuilds the dict from the 17 known sections + 1 meta flag) | `manager.py:368-372, 568-607, 609-634` |
| Corrupt file | NEVER silently replaced: raw bytes copied to `<file>.corrupt-<YYYYmmdd-HHMMSS>.bak` (0600) + loud `#FALLBACK` warning, then in-memory defaults. **The next save still overwrites the corrupt original with defaults** — the backup is the only recovery artifact | `manager.py:495-566` |
| Top-level meta flag | `audio_strategy_explicit` (bool) sits at the JSON top level beside the sections (legacy nested `audio.strategy_explicit` also accepted at load) | `manager.py:514-521, 616` |
| Live ground truth | The real file on this machine has exactly: `api_keys, app_defaults, audio, audio_strategy_explicit, cache, capability_defaults, default_models, email, embeddings, logging, maintenance, offline, provider_profiles, server, streaming, timeouts, video, vision` | probed 2026-07-25 (keys only) |

### Config → environment injection at load (two OPPOSITE precedence rules)

`ConfigurationManager.__init__` (unless `apply_env=False`) injects config into `os.environ`:

1. **API keys: config WINS over env** (operator ruling dm#201, 2026-07-22). Each configured key
   is written into its env var *unconditionally*; a differing pre-existing env value is warned
   once with sha256[:8] fingerprints (never key material) and then **shadowed**
   (`manager.py:405-461`). Key→env map (`manager.py:422-430`):
   `openai→OPENAI_API_KEY`, `anthropic→ANTHROPIC_API_KEY`, `openrouter→OPENROUTER_API_KEY`,
   `portkey→PORTKEY_API_KEY`, `openai_compatible→OPENAI_API_KEY` (**shares** the var with
   `openai`; first-in-map wins, i.e. `openai`), `vllm→VLLM_API_KEY`, `google→GOOGLE_API_KEY`.
2. **Server settings: env WINS over config.** Injected only when the env var is absent
   (`manager.py:463-493`): `server.auth_token→ABSTRACTCORE_AUTH_TOKEN`,
   `base_url_allowlist→ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST`,
   `url_fetch_allowlist→ABSTRACTCORE_SERVER_URL_FETCH_ALLOWLIST`,
   `media_root→ABSTRACTCORE_SERVER_MEDIA_ROOT`, `host→HOST`, `port→PORT`,
   `allow_unauthenticated→ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1`,
   `allow_local_files→ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1`.

> **Doc contradiction (TUI must follow the code)**: `docs/centralized-config.md:25` still claims
> "Environment variables always take precedence over config-persisted keys" — stale for API keys
> since dm#201. Any "Applies now" pane must model direction 1 for keys and direction 2 for server
> settings.

### Non-persisted smart default (display honesty)

`audio.strategy` is rewritten **in memory** at every load unless `audio_strategy_explicit` is
true: with `abstractvoice` installed, `""|native_only|native|disabled` → `auto`; without it,
`auto|speech_to_text|stt` → `native_only` (`manager.py:380-403`). The value the TUI reads from
disk is therefore NOT always the value a Python process sees. Render both ("on disk" vs
"effective") or replicate the rule.

### Other config-related env vars (not part of the JSON round trip)

`ABSTRACTCORE_CONSOLE_LOG_LEVEL` / `ABSTRACTCORE_FILE_LOG_LEVEL` / `ABSTRACTCORE_LOG_BASE_DIR`
(logging overrides), `ABSTRACTCORE_DEBUG`, `ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG`
(server skips config→env injection; `server/app.py:169-186`), plus the large
`ABSTRACTCORE_VISION_*` family (server vision backend selection, not the config file) and
provider vars in §8. Full inventory obtainable via `rg -o 'ABSTRACTCORE_[A-Z0-9_]+'`.

---

## 2. Full key inventory (the TUI's data model)

**Counts**: 17 sections + 1 top-level meta flag; **85 scalar fields**; `capability_defaults.routes`
(24 enumerable route keys, §4); `provider_profiles.profiles` (open-ended map, §5).

All defaults below are the dataclass defaults (`manager.py:73-322`). "CLI" names the setter
surface (§6/§7 mark wizard coverage).

### vision (`manager.py:73-84`)
| Field | Default | Valid values | CLI |
|---|---|---|---|
| strategy | `"disabled"` | `two_stage`, `disabled`, `basic_metadata` (doc: `docs/centralized-config.md:687`) | `--set-vision-provider` sets `two_stage`; `--disable-vision` |
| caption_provider | null | any provider id | `--set-vision-provider P M` |
| caption_model | null | any model id | same |
| fallback_chain | `[]` | list of `{provider, model}` | `--add-vision-fallback P M` (append-only; **no CLI remove**) |
| local_models_path | null | path | **no CLI** |

### audio (`manager.py:87-103`)
| Field | Default | Valid values | CLI |
|---|---|---|---|
| strategy | `"auto"` | `native_only`, `speech_to_text`, `caption`, `auto` (aliases `native`,`stt` normalized; `caption` reserved, accepted by manager but not by the CLI flag choices) | `--set-audio-strategy` (choices exclude `caption`, `main.py:279-286`); sets `audio_strategy_explicit=true` (`manager.py:661-676`) |
| stt_backend_id | null | capabilities-plugin backend id | `--set-stt-backend-id` (blank clears) |
| stt_language | null | e.g. `en`, `fr` | `--set-stt-language` (blank clears) |
| caption_provider / caption_model / fallback_chain | null / null / `[]` | reserved | **no CLI** |

### video (`manager.py:106-126`)
| Field | Default | Valid values | CLI |
|---|---|---|---|
| strategy | `"auto"` | `native_only`, `frames_caption`, `auto` | `--set-video-strategy` |
| max_frames | 3 | int ≥ 1 | `--set-video-max-frames` |
| max_frames_native | 8 | int ≥ 1 | `--set-video-max-frames-native` |
| frame_format | `"jpg"` | `jpg`, `png` (`jpeg`→`jpg`) | `--set-video-frame-format` |
| sampling_strategy | `"uniform"` | `uniform`, `keyframes` | `--set-video-sampling-strategy` |
| max_frame_side | 1024 | int ≥ 1 | `--set-video-max-frame-side` |
| max_video_size_bytes | null | int > 0 or null (≤0 clears) | `--set-video-max-size-bytes` |

### embeddings (`manager.py:129-134`) — legacy pair of the `embedding.text` route
| Field | Default | CLI |
|---|---|---|
| provider | `"huggingface"` | `--set-embeddings-provider`; `--set-embeddings-model P/M` |
| model | `"all-minilm-l6-v2"` | `--set-embeddings-model` |
| base_url | null | `--set-embeddings-base-url` (blank clears) |

Every embeddings setter also **mirrors into `capability_defaults.routes["embedding.text"]`**
(`_sync_embedding_capability_default`, `manager.py:1421-1431`). Reads prefer the route and fall
back to this legacy section (`get_status`, `manager.py:830-848`). Supported embedding providers
(source of truth `abstractcore/embeddings/models.py:71-145`): `huggingface` (in-process,
needs local model files), `lmstudio`, `ollama`, `vllm`, `openai`, `openai-compatible`,
`openrouter`, `portkey`. **The wizard's own validation list omits `vllm`**
(`main.py:943-944`) — use `embeddings/models.py` as truth, not the wizard.

### app_defaults (`manager.py:137-149`) — 5 apps × (provider, model)
`cli`, `summarizer`, `extractor`, `judge`, `intent`; all default
`huggingface` / `unsloth/Qwen3-4B-Instruct-2507-GGUF`. CLI: `--set-app-default APP PROVIDER MODEL`
(valid apps enforced at `manager.py:1599-1623`).

### default_models (`manager.py:190-196`)
| Field | Default | CLI |
|---|---|---|
| global_provider / global_model | null / null | `--set-global-default P/M` (also `P:M` for the known prefixes at `manager.py:37-49`; bare model → provider `ollama`, `manager.py:1326`). **Coupled write**: also sets route `input.text` and clears `output.text` (`manager.py:1323-1338`) |
| chat_model | null | `--set-chat-model` (stored as one `provider/model` string) |
| code_model | null | `--set-code-model` |

Priority chain for apps: explicit args > app default > global default > built-in
`huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF` (`main.py:2254-2258`).

### capability_defaults — see §4. `{version: 1, routes: {key → {provider?, model?, base_url?, reasoning?, options{}}}}`

### provider_profiles — see §5. `{profiles: {id(lower) → 11-field profile}}`

### api_keys (`manager.py:199-208`) — **PLAINTEXT SECRETS**
`openai, anthropic, openrouter, portkey, openai_compatible, vllm, google` (all null by default).
CLI: `--set-api-key PROVIDER KEY` (name normalized `-`→`_`, must be a dataclass field,
`manager.py:1625-1637`); `--list-api-keys` (status only). Note JSON key is
`openai_compatible` (underscore) while the CLI accepts `openai-compatible`. `google` is stored
and injected to `GOOGLE_API_KEY` but there is **no google provider in the registry** (§8) —
reserved (`docs/centralized-config.md:730`).

### server (`manager.py:211-233`)
| Field | Default | CLI |
|---|---|---|
| auth_token | null | `--set-server-auth-token` / `--clear-server-auth-token` |
| allow_unauthenticated | false | `--allow-unauthenticated-server` / `--disallow-…` |
| base_url_allowlist | null | `--set-server-base-url-allowlist CSV` (blank clears) |
| url_fetch_allowlist | null | `--set-server-url-fetch-allowlist CSV` |
| media_root | null | `--set-server-media-root PATH` |
| allow_local_files | false | `--allow-server-local-files` / `--disallow-…` |
| host / port | null / null (port 1-65535) | `--set-server-host` / `--set-server-port` |

Auth modes derived in status: `server_token` (token set) / `unauthenticated_dev` /
`provider_key_only` (`manager.py:944-948`).

### cache (`manager.py:236-242`)
`default_cache_dir` `~/.cache/abstractcore`, `huggingface_cache_dir` `~/.cache/huggingface`,
`local_models_cache_dir` `~/.abstractcore/models`, `glyph_cache_dir`
`~/.abstractcore/glyph_cache`. CLI: `--set-default-cache-dir`, `--set-huggingface-cache-dir`,
`--set-local-models-cache-dir`; **glyph_cache_dir has no CLI**.

### logging (`manager.py:245-254`)
| Field | Default | Valid values | CLI |
|---|---|---|---|
| console_level | `"ERROR"` | `DEBUG,INFO,WARNING,ERROR,CRITICAL,NONE` | `--set-console-log-level` |
| file_level | `"DEBUG"` | same | `--set-file-log-level` |
| file_logging_enabled | false | bool | `--enable/--disable-file-logging` |
| log_base_dir | null (runtime default `~/.abstractcore/logs`) | path | `--set-log-base-dir` |
| verbatim_enabled | true | bool | **no CLI** |
| console_json | false | bool | **no CLI** |
| file_json | true | bool | **no CLI** |

Convenience: `--enable-debug-logging` (both→DEBUG), `--disable-console-logging` (→NONE).

### streaming (`manager.py:257-260`)
`cli_stream_default` false. CLI: `--stream on|off`, `--enable/--disable-streaming`.

### timeouts (`manager.py:263-269`)
`default_timeout` 7200.0 s (provider HTTP), `tool_timeout` 600.0 s. CLI: `--set-default-timeout`,
`--set-tool-timeout`. **Contract: `0` = unlimited; negatives rejected** (`manager.py:1738-1764`).

### offline (`manager.py:272-277`) — **config-file-only, no CLI flags at all**
`offline_first` true, `allow_network` false, `force_local_files_only` true. (Manager methods
`set_offline_first`/`set_allow_network` exist at `manager.py:1774-1790` but nothing in
`config/main.py` wires them.)

### maintenance (`manager.py:152-162`) — **config-file-only**
`triage_llm_enabled` false, `triage_llm_base_url` `http://localhost:1234`,
`triage_llm_model` `qwen/qwen3-next-80b`, `triage_llm_temperature` 0.2,
`triage_llm_max_tokens` 800, `triage_llm_timeout_s` 30.0.

### email (`manager.py:165-187`) — **config-file-only**
SMTP: `smtp_host` "", `smtp_port` 587, `smtp_username` "", `smtp_password_env_var`
`"EMAIL_PASSWORD"` (env var NAME, not a secret), `smtp_use_starttls` true, `from_email` null,
`reply_to` null. IMAP: `imap_host` "", `imap_port` 993, `imap_username` "",
`imap_password_env_var` `"EMAIL_PASSWORD"`, `imap_folder` `"INBOX"`. Env vars still take
precedence per the docstring.

---

## 3. Capability routing defaults (the modern routing surface)

Contract module: `abstractcore/config/capability_defaults.py` (shared with AbstractGateway).

- **Key grammar**: `kind.modality` or `kind.modality.task`. Kinds: `input, output, embedding,
  rerank`; modalities: `text, image, video, voice, sound, music, scene3d`; tasks:
  `text_to_image, image_to_image, image_upscale, text_to_video, image_to_video,
  text_to_scene3d, image_to_scene3d` (`capability_defaults.py:17-27`). Rich alias maps
  normalize input (`stt`→voice, `audio`→sound, `t2i`→text_to_image, …;
  `capability_defaults.py:29-87`).
- **Route value**: `provider?, model?, base_url?, reasoning?, options{}` — all optional; a route
  is "configured" if ANY field is set (`capability_defaults.py:117-142`). Unknown scalar keys in
  a loaded route dict are folded into `options` rather than dropped
  (`capability_defaults.py:246-287`) — the ONE part of the config with lossless unknown-key
  handling. Unparseable route keys are silently skipped at load (`capability_defaults.py:296-302`).
- **The 24 enumerable route specs** (`iter_capability_default_specs`,
  `capability_defaults.py:313-352`): `input.{text,image,video,voice,sound,music,scene3d}`,
  `output.text`, `output.image` (+`.text_to_image`, `.image_to_image`, `.image_upscale`),
  `output.video` (+`.text_to_video`, `.image_to_video`), `output.voice`, `output.sound`,
  `output.music`, `output.scene3d` (+`.text_to_scene3d`, `.image_to_scene3d`),
  `embedding.text`, `embedding.image`, `rerank.text` — each with label, task, `package_hint`
  and `option_examples` (e.g. `output.voice → {"voice": "default"}`,
  `output.image.image_upscale → {"resolution": "2x", "softness": 0.25}`) the TUI should surface.
- **`output.text` is a read-only alias of `input.text`**: reads derive it, writes/clears to
  `output.text` are redirected to `input.text` (`manager.py:1126-1139, 1285-1286, 1313-1317`).
  Never render `output.text` as independently editable.
- **Coverage decoration**: when `input.text`'s model is registry-known to accept a modality,
  `input.image/video/sound/music` report `covered_by: "input.text"`; `input.image` coverage is
  `read_only: true`, the others are `overrideable: true` (`manager.py:1184-1234`). Capability
  knowledge comes from `providers/model_capabilities.py` over the **read-only asset**
  `abstractcore/assets/model_capabilities.json` (~316 KB; `model_capabilities.py:1-24`).
- **CLI**: `abstractcore config defaults [--json]`, `config set-default ROUTE [--provider
  --model --base-url --reasoning --option K=V…]`, `config clear-default ROUTE`
  (`main.py:1688-1707`). `--option` values are JSON-parsed when possible (`main.py:1395-1414`).
  All `config` subcommands accept `--config-file` / `--config-dir` (`main.py:1684-1685`).
- **HTTP twin** (when a server runs): `GET /v1/config/capability-defaults`,
  `PUT/DELETE /v1/config/capability-defaults/{kind}/{modality}[/{task}]`
  (`server/app.py:6588-6692`).

Live-verified JSON shape of `abstractcore config defaults --json` (run 2026-07-25 via the
framework venv): `{ok, version, authority: "abstractcore.local", writable, source, config_file,
routes: [{key, kind, modality, label, task, provider?, model?, base_url?, options?, source,
configured, covered_by?, read_only?, overrideable?, package_hint?, option_examples?}], errors}`
(`main.py:1442-1452` + `manager.py:1166-1182`).

---

## 4. Provider endpoint profiles (`endpoint:<id>` virtual providers)

Module: `abstractcore/config/provider_profiles.py`. A profile makes `endpoint:<id>` resolvable
as a provider everywhere (create_llm, discovery, routes).

- **Fields** (`provider_profiles.py:127-155`): `id` (regex `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$`),
  `display_name`, `description`, `provider_family` (one of **8**: `anthropic, lmstudio, ollama,
  openai, openai-compatible, openrouter, portkey, vllm` — `provider_profiles.py:22-31`; NOT
  huggingface/mlx), `base_url` (must start `http(s)://`, trailing `/` stripped), `api_key`
  (**plaintext**, or literal `EMPTY`), `api_key_env_var` (env NAME reference — CLI syntax
  `--api-key '$VAR'` or `'${VAR}'` stores the reference instead of the value,
  `provider_profiles.py:96-105`), `allowed_models` (fixed allowlist; empty = live discovery),
  `enabled`, `created_at`, `updated_at`.
- Resolution: env-var reference wins over stored raw key when the env var is non-empty
  (`provider_profiles.py:161-166`).
- **Redaction built in**: `public_dict()` exposes `api_key_set` + `api_key_fingerprint`
  (sha256[:8]) and never the key (`provider_profiles.py:168-185`); `abstractcore config
  providers --json` uses it (`main.py:1455-1483`). The TUI should display from this surface.
- CLI: `config providers|provider <id>|set-provider <id> [--family --base-url --api-key
  --clear-api-key --name --description --allow-model… --clear-models --enabled|--disabled]
  |delete-provider <id>` (`main.py:1709-1739`).
- Runtime-injected profiles (`register_runtime_provider_profile`) are process-lifetime only and
  deliberately never persisted; persisted profiles win on id collision (`manager.py:360-366,
  1021-1042`). The TUI will never see them in the file — don't try to model them.

---

## 5. The `abstractcore --config` wizard (verified against code)

Entry: `abstractcore --configure` / `--config` → `interactive_configure()`
(`main.py:208-213, 1786-1788, 730-975`). **8 phases, ~26 prompts**, sequential, each phase
optional. Every answer immediately persists via a manager setter (one full-file rewrite per
answer). The AGENTS.md-remembered phase list matches, with one correction: **STT
backend/language and file logging are NOT wizard questions** (flags only).

| # | Phase | Prompts and valid values | Writes | Citation |
|---|---|---|---|---|
| 1 | Default Model Setup | `Set a default model? [y/N]`; model in `provider/model`; if provider ∈ {ollama, lmstudio, vllm, openai-compatible} and its env var is unset: base-URL prompt. Defaults offered: `OLLAMA_BASE_URL` `http://localhost:11434`, `LMSTUDIO_BASE_URL` `http://localhost:1234/v1`, `VLLM_BASE_URL` `http://localhost:8000/v1`, `OPENAI_BASE_URL` `http://localhost:1234/v1` | `default_models.global_*` **+ route `input.text`**; base URL goes to **session env only** (never persisted; wizard prints the `export` line) | `main.py:737-774` |
| 2 | Vision Fallback | `[y/N]`; provider id (or `provider/model` in one input) + model | `vision.strategy=two_stage`, `caption_provider/model` | `main.py:776-798` |
| 3 | API Keys | `[y/N]`; one prompt per provider: `openai, anthropic, openrouter, portkey, openai-compatible, vllm, google` (blank skips) | `api_keys.*` + env injection | `main.py:800-808` |
| 4 | HTTP Server / Gateway Auth | `[y/N]`; token (blank skip, `generate` → `secrets.token_urlsafe(32)`, `clear`); unauth `[y/N]`; base_url allowlist CSV (blank=loopback-only, `clear`); URL fetch allowlist CSV; media root (blank/`clear`); unrestricted local files `[y/N]`; host (blank=runtime default); port | all `server.*` fields | `main.py:810-885` |
| 5 | Audio Strategy | `native_only\|speech_to_text\|auto`, default `auto` | `audio.strategy` + `audio_strategy_explicit=true` | `main.py:887-902` |
| 6 | Video Strategy | `native_only\|frames_caption\|auto`, default `auto` | `video.strategy` | `main.py:904-919` |
| 7 | Embeddings | `[y/N]`; `provider/model` string; provider validated against `(huggingface, ollama, lmstudio, openai, openrouter, portkey, openai-compatible)` (vllm missing here — wizard bug/gap) | `embeddings.*` + route `embedding.text` | `main.py:921-953` |
| 8 | Console Logging Verbosity | `none\|error\|warning\|info\|debug`, default `error` | `logging.console_level` | `main.py:955-973` |

### Gap list — settable in config but NOT covered by the wizard (the TUI can exceed it)

Via other CLI flags/subcommands (§2 tables name them): all **capability routes** except
`input.text`/`embedding.text`, **provider profiles**, **app_defaults** (5 apps),
`chat_model`/`code_model`, STT `stt_backend_id`/`stt_language`, all six **video detail knobs**,
vision `fallback_chain` (add-only), **cache dirs** (3 of 4), file logging
(`file_level`, `file_logging_enabled`, `log_base_dir`), **streaming**, **timeouts**.

Config-file-only (no CLI anywhere — TUI would be the FIRST UI for these):
`offline.*` (3), `maintenance.*` (6), `email.*` (12), `logging.verbatim_enabled/console_json/
file_json`, `vision.local_models_path`, `cache.glyph_cache_dir`, `audio.caption_*`/
`fallback_chain` (reserved), removing a vision fallback entry.

### Related non-wizard verbs
`--status` (human-formatted tree — **do not scrape**; `main.py:399-728`), `--reset` (full reset
to defaults + save; `manager.py:963-974`), `--install [--yes]` (readiness preflight WITH
side effects — downloads models, pip-installs extras; `main.py:991-1392`),
`--download-vision-model [NAME]` (downloads + reconfigures vision; `main.py:65-199`).
The TUI should treat `--install`/`--download-vision-model` as explicitly user-confirmed actions,
never background probes.

---

## 6. Providers: registry, config shape, discovery

### Registry (`abstractcore/providers/registry.py:117-261`)
**10 static providers** + dynamic `endpoint:<id>` profiles (§4 — `list_provider_names()` appends
enabled profiles' virtual ids, `registry.py:328-342`):

| Provider | local | auth req. | default_model | base URL env (default) | key env |
|---|---|---|---|---|---|
| openai | no | yes | `gpt-5-nano-2025-08-07` | `OPENAI_BASE_URL` (api.openai.com) | `OPENAI_API_KEY` (`openai_provider.py:67-72`) |
| anthropic | no | yes | `claude-haiku-4-5` | `ANTHROPIC_BASE_URL` | `ANTHROPIC_API_KEY` (`anthropic_provider.py:45-50`) |
| ollama | yes | no | `qwen3:4b-instruct-2507-q4_K_M` | `OLLAMA_BASE_URL` then `OLLAMA_HOST` (`http://localhost:11434`) (`ollama_provider.py:41-42`) | — |
| lmstudio | yes | no | `qwen/qwen3-4b-2507` | `LMSTUDIO_BASE_URL` (`http://localhost:1234/v1`) (`lmstudio_provider.py:153-155`) | none |
| mlx | yes | no | `mlx-community/Qwen3-4B` | — (in-process) | — |
| huggingface | yes | optional | `unsloth/Qwen3-4B-Instruct-2507-GGUF` | — (in-process) | — |
| vllm | yes | optional | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `VLLM_BASE_URL` (`http://localhost:8000/v1`) (`vllm_provider.py:28-30`) | `VLLM_API_KEY` |
| openai-compatible | yes | optional | `default` | `OPENAI_BASE_URL` (`http://localhost:1234/v1`) (`openai_compatible_provider.py:156-158`) | `OPENAI_API_KEY` |
| openrouter | no | yes | `openai/gpt-4o-mini` | `OPENROUTER_BASE_URL` (openrouter.ai/api/v1) (`openrouter_provider.py:23-25`) | `OPENROUTER_API_KEY`, config fallback (`openrouter_provider.py:52-59`) |
| portkey | no | optional | `default` | `PORTKEY_BASE_URL` (api.portkey.ai/v1) (`portkey_provider.py:50-52`) | `PORTKEY_API_KEY`, config fallback (`portkey_provider.py:144-152`) |

Resolution order in the OpenAI-compatible family: explicit param > `*_ENV_VAR` > class default /
config fallback (`openai_compatible_provider.py:474-505`). There is **no `google` provider**
despite the `api_keys.google` field. Runtime (non-persisted) per-provider overrides also exist
via `configure_provider()` (`manager.py:1804-1853`) but die with the process.

### Discovery / test-connection semantics
- `get_available_models(provider, …)` (`registry.py:407-503`): for `endpoint:<id>` with an
  `allowed_models` list, returns the fixed list WITHOUT network unless `force_live_discovery`
  (`registry.py:428-431`). For live listing it instantiates the provider; **OpenAI-compatible
  subclasses are constructed with `model="default"` to skip model validation during discovery**
  (the AGENTS.md "discovery bypass" note, verified `registry.py:474-487`).
- **Opt-in probing**: `openai-compatible` and `vllm` refuse discovery (return `[]`, or raise
  with `raise_on_error`) unless a base_url is explicitly configured (kwarg, env, or runtime
  config) — their localhost defaults are "often wrong" and cause noisy timeouts
  (`registry.py:93-115, 451-469`). The TUI must NOT auto-probe these two without a base_url.
- `get_provider_status(name)` → `{status: available|no_models|error, model_count, models, …}`
  (`registry.py:505-550`) — powers the server `/providers` endpoint.
- CLI: `abstractcore config models PROVIDER [--live] [--raise-on-error] [--json]` and
  `abstractcore config test-provider PROVIDER [--json]` (= models with `live=true,
  raise_on_error=true`; `main.py:1636-1676, 1741-1751`). These accept plain provider ids,
  profile ids, or `endpoint:<id>` (`main.py:1608-1621`).

### model_capabilities.json — read-only registry
`abstractcore/assets/model_capabilities.json` (~316 KB, schema beside it) is the canonical
model-capability registry, consumed via `providers/model_capabilities.py` and
`architectures/detection.py`. Per AGENTS.md it is edited by maintainers when models ship. The
TUI **displays** capability-derived facts (coverage badges, capability filtering of model lists)
and must never write these assets.

---

## 7. Verification paths (what "Review & Test" should invoke)

Cheap → expensive; all as `abstractcore` subprocesses unless a server is detected:

1. **Config parse check**: `abstractcore config defaults --json` (also proves the Python side
   reads the file the TUI wrote). ~1.4 s observed via the framework venv.
2. **Profile/provider reachability**: `abstractcore config test-provider <id> --json` — forces
   live `/v1/models` discovery; exit 1 + `❌ Error:` on failure (`main.py:1669-1676`). Model
   enumeration for pickers: `config models <provider> [--live] --json`.
3. **Local server health probes** (what `--install` does, `main.py:1049-1067`): GET
   `OLLAMA_BASE_URL/` (or `/api/tags`), `LMSTUDIO_BASE_URL/models`, `VLLM_BASE_URL/models`,
   3 s timeout. The TUI can do these natively in Rust (ureq) — they are provider servers, not
   abstractcore.
4. **Cloud key presence** is checked as configured/not-configured, not by billing calls
   (`main.py:1069-1077`). A REAL key test = a live models-list via `test-provider` on that
   provider.
5. **Full preflight**: `abstractcore --install` (interactive downloads; gate behind explicit
   user action) — checks default model, provider connectivity, embeddings (incl. cache
   presence), vision, voice (abstractvoice + STT/TTS model caches), ffmpeg, abstractvision,
   API keys (`main.py:991-1392`).
6. **Generation smoke test** (optional, spends tokens): there is no dedicated CLI "test
   generate" one-shot; the honest options are a `/v1/chat/completions` call against a running
   `abstractcore serve` or a tiny `python -c "from abstractcore import create_llm; …"`
   subprocess. Keep it explicitly user-triggered and labeled as spending tokens.
7. **If a server is already running** (config `server.host/port`, default bind `0.0.0.0:8000`,
   `server/app.py:8941-8950`): `GET /health` (unauthenticated), `GET /providers`,
   `GET /v1/models`, `GET /v1/config/capability-defaults` mirror the CLI reads over HTTP
   (`server/app.py:4837, 7025, 6804, 6588`). Auth: `Authorization: Bearer <server.auth_token>`
   when configured.

---

## 8. Integration shape — the honest options and the recommendation

**(a) Read/write the config file directly + shell out to the `abstractcore` CLI.**
Viable: file format is plain JSON with a dataclass schema; CLI setters exist for ~80% of fields;
`config … --json` subcommands are machine-readable and live-verified.

**(b) Drive a long-lived `abstractcore` process.**
Not viable: there is no config REPL/JSONL protocol. `abstractcore` is one-shot argparse
(`main.py:2196-2292`); `abstractcore-chat` is a chat REPL, not a config surface. Building the
TUI on a nonexistent protocol would mean writing and maintaining a Python-side daemon first.

**(c) The HTTP server's config endpoints.**
Partial only: `/v1/config/capability-defaults` CRUD + read-only discovery. No HTTP surface for
api_keys, logging, server hardening, app defaults, profiles, etc. The server is optional,
usually not running on a workstation being configured, and requires the very auth token the TUI
is configuring. Disqualified as the primary channel.

**RECOMMENDATION: (a), in a specific split.**

1. **Reads**: parse `~/.abstractcore/config/abstractcore.json` directly (honor
   `ABSTRACTCORE_CONFIG_FILE`/`_DIR`). Instant, works with no venv resolution. Refresh/verify
   with `abstractcore config defaults --json` and `config providers --json` where the DERIVED
   view matters (coverage decorations, output.text aliasing, redacted profiles) — deriving those
   in Rust would duplicate registry logic that drifts.
2. **Writes**: prefer `abstractcore` CLI subprocesses for every field that has a setter. This is
   not squeamishness — the setters enforce **coupled-field invariants that a direct file write
   silently breaks**: `--set-global-default` also writes route `input.text`
   (`manager.py:1323-1338`); embeddings setters mirror into `embedding.text`
   (`manager.py:1421-1431`); `--set-audio-strategy` sets the `audio_strategy_explicit` meta flag
   (`manager.py:661-676`); `output.text` writes must redirect to `input.text`
   (`manager.py:1285-1286`); profile writes normalize/validate family, URL, id
   (`manager.py:1092-1095`). A field-by-field Rust re-implementation of these rules is a
   permanent drift liability.
3. **Direct writes only for the CLI-less fields** (§6 gap list: offline, maintenance, email,
   logging json/verbatim flags, `vision.local_models_path`, `glyph_cache_dir`, fallback-chain
   removal), using: fresh re-read → minimal key mutation → **preserve every unknown key** →
   `tmp` + rename + `0600`. One combined save per user action, not per keystroke.
4. **Binary resolution**: `$ABSTRACTCORE_BIN` override → `abstractcore` on PATH → known venv
   fallbacks (e.g. `~/tmp/abstractframework/.venv/bin/abstractcore`, which exists and works).
   Surface the resolved binary + its `config_file` (echoed in every `--json` payload) in the UI
   so the "which config am I editing" question is always answered.
5. **Custom config-file targeting asymmetry**: `abstractcore config …` accepts
   `--config-file/--config-dir` (`main.py:1684-1685`), but the flags CLI (`--set-*`) does NOT —
   for those, set `ABSTRACTCORE_CONFIG_FILE` in the subprocess env (`manager.py:342-349`).
6. Server HTTP: optional enrichment only (live health/providers panel when detected).

Latency budget: ~1-1.5 s per CLI call (measured). Batch reads at screen entry; run writes on
explicit save; never call the CLI on render.

---

## 9. Risk map — what the TUI must NEVER do

1. **Never rewrite the file from a Rust-side schema snapshot.** The Python manager itself drops
   unknown keys on its round trip (`manager.py:368-372, 609-634`) — a NEWER abstractcore than
   the TUI's model will have fields the TUI doesn't know; dropping them on a direct write
   destroys user config that Python would have kept. Preserve-unknown-keys on every direct
   write; and conversely, never stash TUI-private metadata in the file (the next Python save
   deletes it).
2. **Never auto-save on open, and never save after a failed parse.** Python's own corrupt-file
   path backs up then falls back to defaults; the *next save* overwrites the original with
   defaults (`manager.py:495-566`). A TUI that opens a broken file and immediately persists
   "what it loaded" completes exactly the data-loss incident this code was hardened against
   (2026-07-11). On parse failure: show the error + the `.corrupt-*.bak`/`.bak-repair-*` files
   (both exist in the wild) and stop.
3. **Never print/log/echo secret values.** Plaintext at rest: `api_keys.*`,
   `provider_profiles.*.api_key`, `server.auth_token` (file is 0600). Display `Set/Not set` +
   sha256[:8] fingerprints (`provider_profiles.py:120-124`), prefer the pre-redacted
   `config providers --json`. Careful with subprocess argv: `--set-api-key PROVIDER KEY` puts
   the key in the process list — acceptable for a local single-user tool but never in TUI logs
   or error toasts; offer the `$ENV_VAR` reference form for profiles.
4. **Never break the coupled invariants** (global default ↔ `input.text`; embeddings ↔
   `embedding.text`; `output.text` → `input.text`; `audio_strategy_explicit`). Symptom of
   getting this wrong: `--status` shows one model while runtime resolves another. Writing
   through the CLI (§8) makes this class unreachable.
5. **Never model one env-precedence direction.** API keys: config **overrides** env (with
   shadow warning); server settings: env **overrides** config (§1). The docs are stale on the
   former. Any "Applies now:" line must name value + source per rule — and never present a
   picker's first item as configuration (the gateway console's combo-fabrication defect,
   AGENTS.md 2026-07-17).
6. **No lock exists — behave accordingly.** A running `abstractcore serve`, another CLI, or the
   wizard can write between TUI read and TUI write; whole-file replace means last-writer-wins.
   Mitigate: re-read immediately before write, detect mtime/content drift since load and prompt,
   write atomically. Don't invent an advisory lock the Python side won't honor.
7. **Never auto-probe `openai-compatible`/`vllm` without a configured base_url**
   (`registry.py:451-469`) and never run `--install`/`--download-vision-model` implicitly (they
   download gigabytes / pip-install into the env).
8. **Never edit the asset registries** (`assets/model_capabilities.json`,
   `assets/architecture_formats.json`) — read-only maintainer-owned inputs.
9. **Never scrape human output** (`--status`, wizard text, emoji lines). Machine surfaces:
   the JSON file, `config … --json`, exit codes (`0` ok / `1` error with `❌ Error:` on stdout,
   `main.py:1753-1760`).
10. **Field-level foot-guns worth inline validation** (from `docs/troubleshooting.md:23-48,
    894-907` + code): whitespace/copy errors in API keys; wrong model names (registry default
    exists per provider — offer `config models` pickers); LM Studio server toggle off / context
    length too small (400s); Ollama not running; `timeouts` `0` = unlimited (not "disabled");
    `server.allow_unauthenticated=true` and `allow_local_files=true` are flagged UNSAFE in
    status — render with warning tones; embeddings provider must be in
    `embeddings/models.py:71-145` (the wizard's own list is missing `vllm`); `api_keys.google`
    configures a provider that doesn't exist (label it "reserved").

---

## 10. Quick reference — machine surfaces the TUI will call

```text
READ   ~/.abstractcore/config/abstractcore.json          # direct parse (ground truth at rest)
READ   abstractcore config defaults --json                # derived routes view + config_file echo
READ   abstractcore config providers --json               # redacted profiles
READ   abstractcore config models <provider> [--live] --json
TEST   abstractcore config test-provider <id> --json      # live reachability, exit code honest
WRITE  abstractcore --set-... / config set-default / config set-provider ...
WRITE  (direct, unknown-keys-preserving, tmp+rename+0600) # only CLI-less fields (§6 gap list)
OPT    GET http://<host>:<port>/health | /providers | /v1/models | /v1/config/capability-defaults
```
