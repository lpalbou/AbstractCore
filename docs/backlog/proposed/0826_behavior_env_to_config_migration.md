# Proposed: migrate remaining BEHAVIOR env vars to centralized config (operator dm#177)

## Metadata
- Created: 2026-07-21
- Status: Proposed (partial migration shipped 2026-07-21; remainder tracked here)
- Origin: operator ruling dm#177 ("kill behavior env vars — configurable on gateway/console"),
  relayed at agora commons c4157. Rule: BEHAVIOR (engine/model/voice, feature toggles) never in
  env; DEPLOYMENT (ports, bind, dirs, secrets, endpoints) may stay.

## Shipped 2026-07-21 (this pass, one fable5 adversary)
- The voice/music incident twin: `audio_endpoints._capability_config()` now merges the deprecated
  env source UNDER the centralized config's `capability_defaults` routes (output.voice / input.voice
  / output.music), config-wins, warn-once, env-only compat preserved; music RESIDENCY warms the same
  config; route-option translation + STT model fan-out to `voice_whisper_model`; call-site pins.

## Shipped 2026-07-22 (vision half, one fable5 adversary — SHIP-WITH-FIXES folded)
- The vision incident twin (the F13 hole below, now closed for the shipped scope):
  `vision_endpoints.py` is config-first for backend kind, per-backend model defaults, upstream
  base URL, sdcpp full-model + component options, catalog/advertising seeding, and the proxy
  model advertising site — env demoted to labeled `#FALLBACK` (warn-once naming the ACTUALLY-SET
  env var). Modality-scoped: image lanes read `output.image`, video lanes read `output.video`
  (adversary P1: an Image Output route must never steer `/v1/videos/*`); residency follows the
  requested task's modality. Route row = ONE backend identity (provider-less models attributed
  by shape, withheld when unattributable). Env-only deployments byte-identical; the
  `abstractvision` package-hint alias maps to auto for CONFIG values only. Audio clone-engine
  direct-env bypass also closed (`voice_cloning_engine` via `_capability_config`).

## Shipped 2026-07-25 (the "REMAINING in the vision lane" tier)
- **Task-specific routes consulted**: `_vision_route_defaults(modality, task)` reads the task row
  first (`output.image.text_to_image` / `.image_to_image` / `.image_upscale`,
  `output.video.text_to_video` / `.image_to_video`); a CONFIGURED task row wins WHOLESALE (one
  backend identity, never field-merged), broad row serves when the task row is absent — the
  generate_contract semantics. Task is threaded through backend resolution
  (`_effective_backend_kind`/`_resolve_backend`), per-backend model defaults, proxy base_url/model,
  sdcpp settings, request-parts shaping, the endpoints (t2i/i2i/upscale/t2v/i2v incl. jobs
  variants), and residency loads (config task set only). `_image_upscale_route_defaults` now seeds
  from the `image_upscale` TASK row ONLY (deliberate: the broad `output.image` row is a generation
  identity — falling back to it would aim upscales at a t2i model; the built-in SeedVR2 default
  stays the no-config fallback), so the endpoint seed no longer defeats the configured task route.
- **Route-option fan-out to diffusers/mflux/proxy** (audio's pattern ported):
  `_route_options_for(backend_kind, modality, task)` + per-lane known-key registry
  (`_ROUTE_OPTION_KEYS`) + `_diffusers_backend_settings` (device, torch_dtype, allow_download,
  auto_retry_fp32) / `_mflux_backend_settings` (base_model, model_dir, allow_download) /
  `_proxy_upstream_settings` (the four upstream path overrides + image_to_video_mode) — config
  wins, env demoted to labeled `#FALLBACK` (warn-once naming the ACTUALLY-SET env var), options
  never leak across backends/modalities. Unknown option keys warn ONCE as "left to the request
  layer unverified" — NOT falsely as dropped, because route options are also request-level
  generation parameters folded by `ResolvedGenerateRouteEntry.apply_to_output_spec` (the shipped
  upscale example's `resolution`/`softness`). Vision diverges from audio's raw pass-through
  deliberately: the vision backend configs are typed dataclasses, so unknown keys cannot be
  forwarded without a TypeError. Provider-less rows carrying options warn (no lane can claim them).
- **mflux base model config-first**: `ABSTRACTCORE_VISION_MFLUX_BASE_MODEL` was env-only in BOTH
  consumers (`_resolve_backend` mlx branch AND the catalog builder env_map); now the mflux route's
  `options.base_model` wins in both (catalog builder also seeds model_dir/allow_download from
  options, and sdcpp `options.diffusion_model`, keeping advertising == execution).
- **i2v modality fix folded in**: the two image-to-video lanes (`_videos_edits_impl` + jobs i2v)
  called `_is_remote_vision_request`/`_create_vision_generation_core` WITHOUT `modality="video"` —
  backend resolution read the IMAGE route (the adversary-P1 bleed the t2v lanes had already fixed).
  Now aligned; same fix inside `_resolve_backend`'s sdcpp component reads and residency's
  `_resolve_backend` call (both previously image-scoped regardless of the resolved modality).
- Validation: `tests/server/test_vision_config_precedence.py` (+16 tests: task precedence incl.
  wholesale-wins + upscale task-only seed, per-lane config-wins + env-only byte-parity, unknown /
  request-level / provider-less option warnings, cross-backend isolation) and an end-to-end
  endpoint pin (`test_images_generations_uses_task_route_over_broad_route`). 134 passed across the
  four touched server test files.

## REMAINING in the vision lane (after the 2026-07-25 pass)
- Timeout classification: audio treats remote timeout as behavior; vision deliberately leaves
  `ABSTRACTCORE_VISION_TIMEOUT_S` env/catalog-only (unchanged this pass).
- Advertising lanes stay broad-route: `_configured_vision_provider_model_entries(task)` reads the
  broad image route regardless of the task filter (a video task filter reads the IMAGE route —
  pre-existing modality gap), and `_vision_catalog_config_from_env` seeds the catalog core from the
  broad row only (task-agnostic by design). Left untouched: fixing the advertising modality gap is
  a behavior change for configured-video deployments and deserves its own focused pass.
- `OPENAI_IMAGE_MODEL_ID` / `OPENAI_IMAGE_MODEL` advertising path remains env-only.

## DEFERRED — genuinely need a config-home design (align with gateway's config architecture)
- Provider feature/hardware toggles with no config home: `ABSTRACTCORE_GGUF_CONTROL_PLANE`,
  `ABSTRACTCORE_GGUF_METAL_UNSAFE` (a deliberate dev/experimental escape hatch — may stay env),
  `ABSTRACTCORE_HF_DEVICE`, `ABSTRACTCORE_EMBEDDINGS_STRICT`.
- `ABSTRACTCORE_FETCH_URL_PDF_NATIVE_MODEL` (pdf_routing): hardcoded `gpt-4.1-mini`, env-only, no
  config home. Adjacent class (a model choice living only in env) — deferral defensible only because
  choosing its config home touches the same in-flight design; recorded here so it is not forgotten.
- CLI-only behavior vars (`ABSTRACTCORE_CLI_KV_REFRESH_TOOLS_AT`, `ABSTRACTCORE_CLI_COUNTRY`): the CLI
  has no console; open question whether these move to the config file.

## Validation
- Config-wins precedence pins per lane; call-site pins (re-pointing to env-only fails); env-only
  byte-parity; warn-once cadence.
