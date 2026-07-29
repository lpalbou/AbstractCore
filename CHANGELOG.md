# Changelog

All notable changes to AbstractCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **analyze_media resolves delegated sight through the SESSION MODEL first (operator ruling 2026-07-26, c3977 amendment; backlog 0837-B; fable5-designed, core half of a cross-lane fix)**: the operator ruled "I should NOT need to set a fallback vision model for a model that already has vision — fallbacks are SOLELY for models that do NOT have vision." `analyze_media` now resolves in that order: (1) the RUN'S OWN route when its model declares vision (capability-gated local read via `get_media_capabilities` — the same signal the media stack uses to decide native attachment, so the gate and the nested `generate(media=...)` cannot disagree; never constructs a client, so no network validation) → runs the image through the session route natively (new `VisionFallbackHandler.create_description_via_route(provider, model, image, prompt)`), fallback never consulted; (2) the configured vision fallback ONLY when the session model lacks vision (its sole purpose per the ruling), for unstamped calls (byte-identical to prior behavior), or as a labeled `#FALLBACK` backstop after a session-route runtime failure; (3) honest three-way refusal naming WHICH model lacked vision + where to configure (never suggesting a model point at itself). The session route arrives via a hidden host-injected `_session_route` param (`hide_args`, participants-stamp precedent) of shape `{"provider", "model"}` ONLY — core REFUSES raw transport (`base_url`/keys): `analyze_media` is read-only/auto-approvable and `hide_args` hides-but-doesn't-enforce, so accepting a model-authored URL would turn an auto-approved tool into a local-file-bytes egress channel; bounding to provider+model caps abuse at operator-credentialed routes (the configured-fallback destination class). Degrades byte-identical when unstamped. 25 analyze_media tests (14 pre-existing byte-identity + 11 new). Cross-lane (flagged to owners): runtime stamps `_session_route` in the TOOL_CALLS handler from `_runtime.provider`/`model` (stamp-or-strip); gateway's 0837-A becomes a loud "vision: not configured" status belt for genuinely vision-less setups (auto-seeding is dead by the ruling — it would manufacture the self-pointing fallback); code-tui runs end-to-end acceptance.
- **Vision config-migration remainder (0826): task-specific routes + route-option fan-out + mflux base-model config-first (operator dm#177, 2026-07-25, one fable5 pass)**: closes the "REMAINING in the vision lane" tier of backlog 0826. (1) **Task-specific routes** — `vision_endpoints.py` now reads `output.image.{text_to_image,image_to_image,image_upscale}` / `output.video.{text_to_video,image_to_video}` FIRST, falling back to the broad `output.image`/`output.video` row; a configured task row wins WHOLESALE (one backend identity, never field-merged — the same semantics `generate_contract.resolve_capability_default_route` uses). `task` is threaded (keyword, default `None` = task-less callers byte-unchanged) through the backend/model/base-url/options resolvers and every endpoint passes its task (t2i/edits/upscale/t2v/i2v, sync + jobs). `_image_upscale_route_defaults` now seeds from the `image_upscale` task row only (the hardcoded SeedVR2 seed previously defeated a configured upscale route), with the built-in as the no-config fallback. (2) **Route-option fan-out** — options previously reached only the sdcpp lane; diffusers (device/torch_dtype/allow_download/auto_retry_fp32), mflux (base_model/model_dir/allow_download), and proxy (upstream paths + image_to_video_mode) lanes now consume their route options with config-first precedence (env labeled `#FALLBACK`, warn naming the actually-set env var), and unknown/untranslated option keys WARN once instead of dropping silently. Honest divergence from audio recorded: vision backend configs are typed dataclasses (unknown keys can't be forwarded), and route options double as request-level params folded downstream, so the warning says "left to the request layer unverified", never falsely "dropped". (3) **mflux base_model config-first** — `options.base_model` on an mflux route now wins over `ABSTRACTCORE_VISION_MFLUX_BASE_MODEL` in both the resolver and the catalog builder (advertising kept equal to execution). Folded-in fix: the two image-to-video lanes resolved backends without `modality="video"`, so an image route steered `/v1/videos/edits` (the same cross-modality bleed the t2v lanes had already fixed) — corrected in the edits/jobs/residency call sites. Env-only deployments byte-identical. 134 server tests green (17 new precedence pins). Residuals left with rationale in 0826: advertising lanes stay broad-route (a deeper pre-existing video-task-reads-image-route gap deserves its own pass), timeout classification stays env/catalog-only (open question), and the deferred cross-package tier (GGUF/HF/embeddings toggles, PDF model, CLI vars) untouched.
- **Vision behavior config is now config-first on the standalone server (operator ruling dm#177/dm#194; env-conflict report angle A #4, 2026-07-22)**: the image/video endpoints consulted ZERO capability defaults — the console PUT `output.image` routes into the very config this server owns while env on the host decided execution, and the mere PRESENCE of an exported `OPENAI_BASE_URL` flipped a configured local setup to the OpenAI-compatible proxy. `vision_endpoints.py` now reads the centralized config's `output.image`/`output.video` routes FIRST (backend kind, per-backend model defaults, upstream base URL, sdcpp full-model + component options, catalog/advertising seeding), with env retained as a labeled `#FALLBACK` below config — warn-once naming the env var that is actually set. The route row is ONE backend identity (an mflux route's model never leaks into the diffusers/sdcpp/proxy lanes; provider-less models attributed by shape, withheld when unattributable). A fable5 adversary review (SHIP-WITH-FIXES) was folded the same night: modality scoping so an Image Output route never steers `/v1/videos/*` (video lanes read `output.video`; residency follows the requested task), config-first proxy ADVERTISING (a config-only proxy route used to execute but advertise `[]`), plugin wire spelling for the proxy backend, sdcpp shadow warning, actually-set env var named in override warnings, and the `abstractvision` package-hint alias restricted to config-sourced values (env-only deployments stay byte-identical). The audio clone-engine direct-env bypass (`ABSTRACTVOICE_CLONING_ENGINE`) was closed in the same pass. 17 precedence pins; full server suite green.
- **API-key precedence: a config-set key now SUPERSEDES the env var (operator ruling dm#201, 2026-07-22)**: cloud API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …) remain env-INHERITABLE by default — the ruled exception to behavior-env elimination (they exist for other apps; no migration forced). BUT `_apply_api_keys_to_env` used to inject config-persisted keys only when the env var was ABSENT ("environment always wins"), so a key rotated via `abstractcore --config` applied only to lanes that happened to lack an export. Inverted per the ruling: a configured key now always overwrites the env var at injection, and a shadowed env key is warned once per state with key FINGERPRINTS (never key material). The `openai`/`openai_compatible` config fields share `OPENAI_API_KEY`; first-field-wins is preserved explicitly. Endpoint profiles' `api_key_env_var` indirection is untouched (config explicitly pointing AT an env var is config winning). 6 pins; config suite 55 green.
- **Voice/music behavior config is now config-first on the standalone server (operator ruling dm#177, 2026-07-21)**: exported `ABSTRACTVOICE_*`/`ABSTRACTMUSIC_*` env vars (e.g. `ABSTRACTVOICE_TTS_ENGINE`) could silently override the engine/model configured via `abstractcore --config` — the same silent-override incident the gateway hit. The server's capability core now reads the centralized config's `capability_defaults` routes (`output.voice`/`input.voice`/`output.music`) FIRST, with env retained only as a labeled `#FALLBACK` below config: config wins, env-only deployments still work byte-identically, and a warn-once message names the env vars to unset. Music residency warms the same config (no catalog-vs-synthesis divergence), the configured STT model reaches local whisper (`voice_whisper_model`), and route options are translated to plugin keys instead of silently no-op'ing. A fable5 adversary hardened the change (F1 music-residency, F3 option translation, F4 STT fan-out, F5 option-never-clobbers-identity, F9 warn-once, F11 call-site pins). Remaining behavior env vars (generative-vision model selectors, provider feature toggles, PDF native model) are tracked in backlog 0826. 11 precedence pins; full server suite green.

### Added
- **`fetch_url` SSRF guard — per-hop destination validation + allowlist + header hygiene (operator/code convergence, 2026-07-28)**: new `abstractcore/tools/fetch_url_ssrf.py` validates every outbound hop (including redirects via `SSRFGuardAdapter` on the requests session): cloud metadata IPs/hostnames deny-always (169.254.169.254, 100.100.100.200, fd00:ec2::254, metadata.google.internal/goog); non-public destinations deny-by-default with `ip.is_global is False` as the CGNAT catch-all (100.64/10); operator allowlist via `ABSTRACTCORE_FETCH_URL_ALLOW` as host:port entries (localhost and 127.0.0.1 are distinct spellings); model-supplied Authorization/Cookie/Proxy-* headers stripped before send. Meta-refresh follows reuse the guarded session. 11 new pins; fetch_url + execute_command suites 64 green. the OpenAI-compat server DROPPED model reasoning at the boundary in both modes — `convert_to_openai_response` never emitted it and the SSE lane forwarded content only. Non-streamed chat completions now carry `message.reasoning_content` (the DeepSeek/vLLM/LM Studio de-facto key, the same one AbstractCore's own OpenAI-compatible provider reads back — the server round-trips through its own client); streamed responses forward `delta.reasoning_content` chunks (incremental deltas as they arrive; the trailing aggregate is sent exactly once ONLY when no deltas were seen, so accumulating clients never double). `ChatCompletionRequest` accepts OpenAI-native `reasoning_effort` (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`) as an alias folded into the unified `thinking` control when `thinking` is absent (standard OpenAI clients were silently ignored before). `/v1/models` entries gain a `reasoning` block (`thinking_support` + `reasoning_levels`) sourced via the new `lookup_registry_model_capabilities()` seam — registry-resolved knowledge only, NO architecture-default fallback and no unknown-model warning spam: an unknown model gets NO block rather than a fabricated `thinking_support: false`, so a coupled provider→model→effort selector can distinguish "not a reasoning model" from "unknown". Also fixed en route: a `usage=None` response crashed `convert_to_openai_response` with AttributeError.
- **Tool risk tiers: declared FACTS + one versioned derivation, and the comms tools join the inventory (operator order dm#221, 11-seat design + adversaries, 2026-07-22/23)**: the framework needed the operator to KNOW what "powers" a granted agent/entity holds. The converged design is TWO AXES — `capability_class` (boundary: who may even see a tool; structural, runtime-owned) and a RISK band (what a user grants) — and the risk band is DERIVED from per-tool FACTS, never hand-assigned. New `abstractcore/tools/risk_facts.py` hosts the ONE versioned max-wins mapping (core hosts it as code, import-never-copy; gateway+runtime own its CONTENT, `RISK_MAPPING_VERSION` served on every row): `destructive_capable` or facts-undeclared → destroy(4); `comms_send`/`captures_environment`/`standing_effect` → outreach(3); `mutating`/`remote_write_capable` → act(2); else observe(1). Facts-undeclared gates at the top band but PRESENTS as `unvetted` (never "destructive" — overclaim kills the signal); `model_cost` (budget) and `model_controlled_destination` (approval-rule) are band-neutral (facts feed two rules). The fact vocabulary is closed and danger-when-true, validated AT THE DESK (`register_capability_tool_facts` / `capability_tool_facts` on the capability registry refuse unknown spellings; a typo can never silently under-derive a tier). Inventory schema **v2→v3**: the 5 risk facts + derived `risk_tier`/`risk_rank`/`risk_presentation`/`risk_mapping_version` on every row, AND `comms_tools` (email/WhatsApp) + `telegram_tools` JOIN the scanned modules — the dm#221 tool-surfacing audit found the v1 exclusion made 9 comms tools invisible to every host's discovery; they now surface WITH facts (`send_*` → outreach) instead of unvetted-at-top. `execute_command`/`shell_*` carry `destructive_capable` (the argv clamp: rm/git are programs inside the shell; per-argument refinement stays approval-time, never a grant-time discount). The comms `send_*` tools gained the `write` side-effect tag to match their `remote_write_capable` fact (the decide-the-fact-once cross-surface guard). 26 builtin tools total; risk-facts + inventory + capability-surface + tag suites green. Custom-grant path needs no core delta beyond serving complete rows (name lists are the finest grain). A fable5 adversary review (SHIP-WITH-FIXES) was folded before ship: (P0) `derive_risk({})` returned observe — an EMPTY facts dict is now treated as UNDECLARED (fail-closed → unvetted/top), because the idiomatic consumer join `derive_risk(facts_map.get(name, {}))` must never render an unknown tool safe; `register_capability_tool_facts` refuses an empty per-tool declaration at the desk as the belt; (P2) `BuiltinToolDescriptor` risk defaults flipped to destroy/unvetted (a hand-built row is born highest-risk, never observe); (P2) a version-pinned canonical derivation table test makes a band change without a `RISK_MAPPING_VERSION` bump land red; (P2) the act-band containment assumption (the workspace wall separates bounded writes from shell reach) is documented in `derive_risk`; (honesty) `comms_send` narrowed to "a purpose-built human-messaging lane" so the fact stays a true checkable claim vs a webhook POST. The one deferred item is filed to the runtime seat: `fetch_url`'s band-neutral `model_controlled_destination` fact only holds if the approval layer consumes it, but runtime's default policy currently auto-approves `fetch_url` ("Network read-only") — contradicting core's `remote_write_capable` fact and the 2026-07-12 "fetch_url is never read-only-safe" ruling.
- **`send_email` per-call risk refiner declaration (operator ruling dm#244, 2026-07-23)**: the operator ruled `send_email` should modulate by recipient — sending to the registered operator email is auto, others require approval, and custom mode can auto-accept wholesale. This is per-call classification (like `execute_command`'s read-only-git proof), not a flat tier flip. Core's half: `send_email` now declares `risk_refiner="send_email_recipient@v1"` as BAND-NEUTRAL metadata — the grant-time band stays `outreach` (the ceiling and deny-safe default), and the refiner rides on the row so the enforcement layer dispatches on DATA instead of a hardcoded tool-name (the clean version of the currently-hardcoded read-only-git dispatch). New `risk_facts` surface: `KNOWN_REFINER_IDS` + `validate_refiner_id` (versioned `name@vN`; unknown/unversioned refuses at the desk), `derive_risk(facts, refiner=...)` (the refiner NEVER changes the derived band — a factless row stays `unvetted` even with a refiner declared: you cannot refine what you cannot classify), and inventory `_REFINER_BY_NAME` fail-closed both ways (a refiner naming no scanned tool refuses). `risk_refiner` rides every inventory row + `to_dict`. Ownership split per the shipped contract: core hosts the fact + refiner-id (done); runtime runs `send_email_recipient@v1` at the approval point; gateway serves the registered operator email (config-first settings row) + records grant acts. The refiner is LOWER-ONLY (can drop a proven-self call below `outreach`, never raise), so the ceiling holds on every call it cannot prove self — the deny-safe default is structural. A fable5 adversary (SHIP-WITH-FIXES, band-neutrality proven over all 258 fact inputs) was folded: (P1) an UNDECLARED row now STRIPS the refiner entirely (`derive_risk(None, refiner=...)` → no refiner) — "you cannot refine what you cannot classify", so an unvetted row never serves a lowering hook; (P2) `RiskAssessment.__post_init__` validates the refiner-id at construction (a direct bogus id refuses, not a silent-unlabeled row), every `KNOWN_REFINER_IDS` entry is pinned `@vN`-versioned, and both build-time refiner guards (unscanned-tool and unknown-id) are now test-pinned like their classification siblings. Pins across risk-facts + inventory suites; full tools+capabilities suite 482 green.
- **`analyze_code` multi-language engine + never-refuse fallback (operator order 2026-07-22, three fable5 reviewers folded)**: the code-navigation tool supported only python/js/html/r and REFUSED everything else (live incident: `main.rs` → "Unsupported code language", forcing agents to re-read whole files raw). New declarative engine (`tools/code_analysis.py`): one outliner driven by per-language `LanguageSpec` rows — adding a language is adding DATA — now covering rust, go, java, c, c++, c#, swift, kotlin, ruby, php, shell, sql, css, markdown, yaml, toml, json, dockerfile, makefile, terraform/hcl, protobuf (21 engine + 4 legacy deep lanes, unchanged). Unknown-but-readable text degrades to a labeled GENERIC outline (metrics, top-level structure, TODO markers) — only binary is an error; shebangs route extensionless scripts (bash → shell, python → the deep AST lane). Reviewer folds, all pinned: (A, fidelity) the brace scanner is string- and block-comment-aware and SHARED with the balance lint (a `"}"` literal or `/* } */` comment no longer ends extents early while the lint vouches ok); braceless declarations abort their forward scan at the next declaration (kotlin `data class` no longer steals the next block's extent); ruby handles one-line `def x; end`, assigned `= case` blocks, heredocs and `=begin` comments; Allman-style C/C++ definitions, init-list/plain constructors and swift `init`/`deinit` outline; `#if 0` code, shell heredocs and markdown fences are skipped as data. (B, usefulness/contract) nothing in the monorepo keyed on the old refusal (verified); JSON/JSONL dispatches BEFORE the minified guard (compact package-lock/ledger files keep their parse lane); truncated sections name the `search_files` recovery path; `language="text"` means the generic lane without a spurious #FALLBACK; a misspelled hint on an unambiguous extension honors the file with a notice. (C, performance/encoding) two P0s fixed: an exponential-backtracking C function regex hung the tool on ordinary `* * *` comment banners (>15s → 64ms), and per-declaration extent scans were quadratic when braces never close (4000 unclosed decls: 91s → 0.2s via a one-pass extent index); java/c#/c++ modifier-chain regexes rewritten with atomic-group emulation; the 4MB truncation boundary no longer mojibakes the whole file (incremental UTF-8 decode); UTF-16/32 BOM files decode with a label instead of a false "binary" claim; UTF-8 BOMs are stripped; line numbers split on `\n` only so they always match `read_file(start_line)` (str.splitlines' form-feed splitting drifted them); rust drops `'` from quote tracking (lifetimes). 47 pins; tools suite green (2 pre-existing pty-flaky shell-session tests fail identically on the clean tree).: `CapabilityRegistry.register_capability_tools(capability, tools)` + `register_capability_tool_policy(capability, {"auto_approve": [...], "require_approval": [...]})` let a capability plugin (camera today, any future plugin) surface its `@tool` functions and their approval partition at plugin-registration time; upper packages consume them via `abstractcore.capabilities.capability_tools(name)` / `capability_tool_policy(name)` (process-shared registry, thread-safe first-use) instead of importing the plugin package — runtime can drop its direct `abstractcamera` import. Contract facts, adversary-hardened (fable5, BLOCK→folds): reads ensure entry-point plugins are loaded (a fresh process no longer answers `[]` for an installed plugin — the silent-vanish window); returned tools are ISOLATED copies (fresh `parameters`/`tags`/`examples`, shared callable) so one consumer's in-place schema rewrite cannot poison the registry or siblings; the approval partition is derive-never-copy (the plugin computes it from its own classification facts; core stores and serves); absence fails closed (no policy = everything approval-required); same-object re-export dedupes but two DIFFERENT definitions claiming one name refuse loudly (inventory discipline); nameless tools and bare-string contributions refuse. 32 pins in the capabilities suite.
- **`analyze_media` tool — delegated sight for text-only agents (backlog 0825, operator-ruled GO 2026-07-21)**: a default agent tool that answers a question about an image by running the operator's CONFIGURED vision model and returning bounded text, so an agent that captured an image mid-loop can see it without the main model being a VLM. Rides the existing vision-fallback config (no second model knob), refuses loudly with the setup hint when unconfigured, and a fable5 adversarial review hardened it before ship: (1) an **honesty gate** PIL-verifies the file actually decodes as an image before dispatch — a corrupt/truncated capture or a renamed non-image now REFUSES instead of getting a confident, provenance-stamped description of a placeholder the model never saw; (2) a configured-but-failing route surfaces the real runtime cause (new `VisionGenerationError`) instead of a misleading "not configured"; (3) the nested vision call runs one attempt with a bounded 120s per-attempt timeout (no retry stacking, no 2h config default); (4) the image-suffix set is aligned to the media layer's real support; (5) classified read-only with `model_cost=True` (inventory schema v2) so hosts budget/approve the nested-LLM cost distinctly. 13 tool pins + updated inventory pins; tools+media suites 490 green.
- **Retry wall-clock budget (`RetryConfig.max_total_wall_clock_s` / `create_llm(..., retry_wall_clock_budget_s=...)`, 2026-07-21)**: per-attempt timeouts stack across retries — 3 attempts at the 600s config default timeout wedged an interactive entity visit for exactly 30 minutes against a hanging endpoint. When the budget is set, no retry starts once the sequence's elapsed wall clock exceeds it; the first attempt is never gated (legitimate long generations are not retries); exhaustion raises the last error loudly (`#FALLBACK` + `RETRY_EXHAUSTED` with `reason=wall_clock_budget_exhausted`). Default `None` preserves existing behavior. Interactive lanes should set it to their user-facing patience window. 5 pins; retry-lane suites 87 green.
- **KV-artifact cache-dtype axis + q8 storage unlocked (backlog 0817, axis 5, 2026-07-21)**: the durable bloc lane hardcoded `quantization="fp"` and rejected everything else, leaving MLX's 8-bit KV quantization unreachable. `ensure_bloc_kv_artifact`/`load_bloc_kv_artifact` (and the `/acore/blocs/kv/ensure|load` endpoints) now take `quantization: "fp"|"q8"` — a stored/requested mismatch recompiles at the REQUESTED dtype with a labeled `#FALLBACK`, an unknown stored dtype refuses (never guess another build's tensor layout), and q8 against a provider that doesn't declare a `q8` save parameter raises up front (a `**kwargs` writer would silently store fp under a q8 label). Pre-axis manifests read as "fp": the existing artifact corpus stays valid under default requests. 8 new pins; full suite 2256 green. Follow-up same night: the MLX save path's SILENT fp fallback on q8 conversion failure was removed — hybrid/recurrent architectures (state-space layers carry `ArraysCache` with no `to_quantized`: Qwen3.5+ hybrids, LFM2, Mamba-class) now raise loudly naming the unquantizable cache type instead of storing fp bytes under a q8 label; standard and sliding-window attention caches (`KVCache`/`RotatingKVCache`) quantize as before.
- **KV-artifact weights-identity gate (backlog 0817, axis 4, 2026-07-20)**: a checkpoint swap under the SAME model id — force-pushed HF revision, re-quantized GGUF, edited shards — left text/tokenizer/config traces identical while the weights that computed the cached tensors were gone (runtime c1734). New `providers/weights_fingerprint.py` records a CHEAP tiered identity (hub revision sha → local fileset + safetensors-header digests → GGUF size + header-slice digest; full-content hashing of multi-GB weights deliberately out of scope, stated honest limit for same-size same-header value edits). GGUF does NOT abstain on this axis — it owns the in-file tokenizer/config signals axes 2-3 deferred to it. Ensure-gate refuses+recompiles on mismatch, MLX load-gate raises, pre-axis artifacts reuse with labeled `#FALLBACK`, unavailable state abstains; not part of `binding_id`. 17 new pins; full suite 2248 green; live-verified (real snapshot dir resolves its commit sha; 4.9 GB GGUF fingerprints in 2.5 ms).

### Fixed
- **`execute_command` timeout results now include the output captured before the kill, and the timeout parameter is clamped to 600 seconds and documented as SECONDS (8h20m production hang, code-tui root-cause, 2026-07-27)**: a model passed `timeout=30000` meaning milliseconds; nothing capped the value, so one tool call waited 8 hours 20 minutes — and when the timeout finally fired, the result carried only "Tool timeout after 30000s" with every captured byte discarded (the model then diagnosed the real failure nine seconds after being shown any output; the loop was starved of input, not intelligence). Two fixes. (1) OUTPUT ON TIMEOUT: after the process-tree kill, the tool drains the pipes with a second `communicate()` (hard-bounded at 5s — the reliable collection point, since the `TimeoutExpired` exception does not carry partial output on every platform) and returns the pre-kill stdout/stderr both in the rendered text (tail-kept at the normal 20000/5000-char caps with a `#TRUNCATION` label — the end of the output is usually where the failure shows) and in the structured fields (full, for durable evidence). When nothing was captured, the message says so and why ("a backgrounded child writing to files or inheriting the terminal is invisible here"); if the bounded drain itself gives up because something the kill could not reach still holds the pipes, the bytes read so far are salvaged from the drain exception (POSIX) and the render says the capture may be incomplete. (2) CLAMP + SCHEMA TEACHING: new module constant `EXECUTE_COMMAND_MAX_TIMEOUT_S = 600.0` caps the model-supplied timeout; when the clamp fires, a note naming the requested and applied values rides BOTH the success and the timeout renders, pointing longer work at "run it in the background and poll". Zero, negative, NaN and infinite values fall back to the 300s default (NaN previously slipped through the `<= 0` check and would have made `communicate()` wait forever — worse than the incident). The `timeout` parameter now carries a JSON-Schema `description` ("Seconds (max 600; values above are clamped). Time limit before the command tree is killed.") via the same post-definition injection channel as `edit_file`'s line-range descriptions — the docstring never reaches the model, so the schema is the only surface that can teach the unit at call time. The default timeout is unchanged, and unclamped success renders are byte-identical (pinned). 12 new tests including both grandchild-holding-pipe drain shapes (real tree-kill: fast drain with both processes' output; simulated kill miss: bounded give-up + salvage); execute_command suite 17 green.
- **Nested `create_llm("endpoint:<id>")` during tool execution can now resolve HOST-registered endpoint profiles (live incident `Unknown provider: endpoint:airelay`, converged cross-package fix, 2026-07-26)**: `endpoint:*` profiles can live on a HOST (AbstractGateway per-principal store) rather than in `~/.abstractcore` — the run's own LLM calls resolve them through a resolver the host attaches to provider INSTANCES (`resolve_provider_endpoint_profile`), but a nested tool-side construction (`analyze_media`'s session route → `create_llm(provider, model)`) has no instance to inherit from, so the registry raised `Unknown provider` for a profile the host could resolve. New stdlib-only module `providers/endpoint_context.py`: the host enters `use_provider_endpoint_profile_resolver(resolver)` around tool execution (ContextVar; token-reset nesting, inner wins, `None` masks; model-unreachable by construction — the resolver never rides tool args or any model-writable surface). The registry consults the ambient resolver ONLY after local config misses an `endpoint:*`-shaped spec (local config FIRST — operator-profile routes byte-unchanged, pinned by a tripwire test; non-endpoint specs never consult). On a hit the payload (gateway `private_resolution()` shape: `provider_family`/`provider`, `base_url`, `api_key`, `allowed_models`, `enabled`) normalizes through the local `ProviderProfile` dataclass — unknown fields dropped, no/unsupported family or disabled → labeled `#FALLBACK` miss (missing fields are a MISS, never a guess; never a crash; secrets scrubbed from all messages) — and construction flows through the SAME `create_provider_instance` path as local profiles (one shared `_host_profile_transport_config` underlay mirrors the config manager's key set for transport kwargs; live model discovery probes the profile's transport too). The constructed instance gets the resolver attached (the BaseProvider setattr-propagation pattern) so nested/fallback constructions inherit the host's reach, and the video-route fallback in `BaseProvider` falls back to the ambient resolver when no instance-level one exists. Error honesty: with a resolver present and BOTH sources missing, `Unknown provider: endpoint:<id>` now names both (local config + host-injected resolver); bare core (no resolver installed) keeps today's error strings byte-identical on both raise sites (pinned). 17 new tests incl. the incident end-to-end (analyze_media stamped with an endpoint session route resolving through the injected resolver into a real offline provider construction); analyze_media 25 + factory/registry/profiles/video-policy suites stay green (100 total). channel-separated reasoning (Ollama `thinking`, LM Studio native `reasoning.delta`, OpenAI-compatible `reasoning_content`, Anthropic `thinking_delta`) arrived only as per-chunk fragments under `metadata["reasoning"]` with NO aggregate anywhere — a stream fold that reads the final response's reasoning (the runtime rehydration contract) captured one fragment or nothing. Contract now: intermediate chunks carry verbatim `metadata["reasoning_delta"]` fragments (live-tail channel; whitespace preserved — Ollama previously `.strip()`ed each delta, corrupting word boundaries on join, and `extract_reasoning_from_message` gained `strip=False` for the delta lane); `BaseProvider` joins deltas (tolerating legacy per-chunk `reasoning` spellings) and guarantees the stream's TRAILING chunk carries the COMPLETE `metadata["reasoning"]`, merged with inline `<think>` stripper output and re-carrying last-seen usage. Anthropic streaming previously DROPPED `thinking_delta` events entirely (only `.text`/`.partial_json` were handled) — extended-thinking text now streams as `reasoning_delta` chunks and aggregates like every other provider. `GenerateResponse.reasoning` aliases `reasoning_delta` last (chunk-level display compat); `BasicSession` persists reasoning into assistant-message metadata in all four generate paths (sync/async × stream/non-stream; history-only — provider replay stays role/content); the CLI prefers the trailing aggregate over joined deltas. Live-verified against LM Studio (qwen3-0.6b): inline-think stream ends with a clean 1k-char aggregate, non-stream captures, and the native-REST rejection fallback stays loud. New suites: streaming delta aggregation, Anthropic stream capture, session persistence.
- **`thinking=` escaped-regex defects: "extra high" alias raised ValueError and Harmony `Reasoning:` lines duplicated (reasoning audit, 2026-07-26)**: two raw strings carried doubled backslashes (`[\\s\\-]+`, `^\\s*Reasoning\\s*:...`), so the character class matched literal backslashes instead of whitespace — the documented `thinking="extra high"` / `"x high"` aliases raised ValueError instead of normalizing to `xhigh` — and the Harmony system-prompt `Reasoning:` line replacement could never match an existing line, PREPENDING a second conflicting `Reasoning:` directive instead of replacing it (GPT-OSS reads that line as the effort control; two lines is undefined behavior). Both regexes fixed; normalization + replacement/prepend paths pinned.
- **Anthropic thinking mapping: no silent `xhigh`→`max` escalation, `disabled` omitted on adaptive-only models, `budget_tokens` respects API floors (reasoning audit, 2026-07-26)**: (1) `thinking="xhigh"` on adaptive models was silently escalated to `output_config.effort="max"` when `max_effort_supported` — spending MORE than the caller asked; `xhigh` is a valid adaptive effort value and now passes through verbatim ("max" stays unreachable until the unified vocabulary grows an explicit level; "minimal" maps to the documented floor "low"). (2) `thinking="on"` without a level no longer forces `effort="medium"` — adaptive rides alone and the API default applies. (3) `thinking="off"` sent `thinking: {"type": "disabled"}` unconditionally, which adaptive-only models (Opus 4.7, Fable 5, Mythos 5 per Anthropic/AWS docs) reject with HTTP 400 — those models (new registry field `thinking_disable_supported: false`) now express off by OMITTING the thinking parameter, with a warning. (4) The extended-thinking budget clamp could emit requests the API is documented to reject: `budget_tokens` has a hard 1024 minimum and `max_tokens` must be STRICTLY greater — the clamp now floors at 1024, caps at `max_output_tokens - 1`, and refuses to enable thinking (loudly) when `max_output_tokens <= 1024` instead of sending a known-invalid request.
- **LM Studio `thinking="off"` was reported effective while the model kept reasoning (live find, 2026-07-26)**: for models declaring both an `enable_thinking` template kwarg and the Qwen assistant-prefill hard switch, the template-kwarg claim ("handled") SKIPPED the prefill fallback — but LM Studio's OpenAI-compatible endpoint ignores `chat_template_kwargs` for some formats (live: qwen3-0.6b GGUF produced 1.1k chars of reasoning under `thinking_effective: "off"`). The prefill hard switch now applies for LM Studio even when the template kwarg was sent (both artifacts express the same off state); HF-GGUF keeps the handled gate (its renderers place the marker themselves). Live re-verified: off now yields zero reasoning in both stream and non-stream.
- **Registry: o-series effort levels were missing, gpt-5-mini/nano lacked "minimal" (reasoning audit, 2026-07-26)**: `o1`, `o3`, `o3-mini`, `o4-mini` had `thinking_support` but NO `reasoning_levels`, so the OpenAI provider never sent `reasoning_effort` for them — effort requests warned and did nothing. Added `["low","medium","high"]` per the OpenAI reasoning guide (o1-mini deliberately stays level-less: it rejects the parameter; "minimal" arrived with gpt-5, "xhigh" only after gpt-5.1-codex-max). `gpt-5-mini`/`gpt-5-nano` gained `"minimal"` (gpt-5 family launch docs). New assets fields documented in `assets/README.md` (`reasoning_levels`, `thinking_budget`, `thinking_control_mode`, `thinking_disable_supported`, `max_effort_supported`) and pinned by a registry-fields test with the evidence trail.
- **`skim_websearch` required-term filter no longer zeroes out on backend-fused snippets (live incident 2026-07-26, operator screenshots)**: a query about "Game Boy" returned `{"fetched":15,"matched":0,"returned":0}` because the `ddgs` library's result normalization strips HTML highlight tags with an empty replacement (`ddgs/utils.py::_normalize_text`, applied to every title/body via `ddgs/results.py`), fusing words at tag boundaries ("TheGameBoyhas foursoundchannels") — the multi-word term "game boy" is not an exact substring of fused text (mechanism reproduced byte-for-byte against the library; upstream fact, not forked). Two independent layers shipped: (1) our own `duckduckgo.html` fallback extractor had the identical defect (`tag_re.sub("", ...)`) and now replaces tags with a space then collapses whitespace runs (also normalizes `&nbsp;`); (2) `skim_websearch` term matching gained a whitespace-elided fallback scoped to whitespace-bearing terms only ("game boy" → "gameboy" against per-field whitespace-stripped haystacks; primary exact-substring check unchanged and first) — this also catches legitimate source spellings ("GameBoy" on gg8) and nbsp variants; results matched only via the fallback add one labeled `note` so the model treats snippets as degraded leads. Zero-match hints are now cause-aware: mostly-empty backend snippets → suggests `require_in='title_snippet'`/`'all'`; terms present in title/url but not snippet → names the wider scope that matches; otherwise the original hint. Also: `required_terms` accepts JSON-encoded string lists (`'["Game Boy"]'` — tool transports deliver list args as source text; the literal brackets previously became part of the term), `counts.fetched` counts only dict rows (truthful), and the decorator example teaching `match="all"` snippet-only filtering was replaced with the safer `match="any"` + `require_in="title_snippet"` combination. The elided comparison stays a skim relevance filter only — never generalize it to exact-match tools (`search_files`). 12 new synthetic-backend tests; web tool suites green.
- **`edit_file` now teaches batching (operator ruling 2026-07-26, "batching: yes")**: agents were making one edit per turn even for independent edits, because nothing told them otherwise. The tool's guidance now says it plainly: edits to different files can be sent as several `edit_file` calls in the same turn; several changes to the same file belong in one unified-diff call (many hunks, applied atomically) instead of sequential pattern calls, whose earlier edits can shift what the later ones depend on. This is core's half of the cross-package batching fix; the agent-prompt and TUI-display halves ship in their own packages.
- **`edit_file` line-range UX rebuilt from a live failure trace (operator screenshots 2026-07-26, fable5-adversaried)**: a live abstractcode-tui run called `edit_file(start_line=0, end_line=0, pattern=...)` (0-based habit), got refused one parameter per turn ("Invalid start_line 0…" and only THEN "Invalid end_line 0…"), and later `end_line=210` on a 196-line file — burning a model turn per refusal for intents that were unambiguous. Root cause runs deeper than the messages: the exported tool schema carried NO parameter descriptions at all (`ToolDefinition.from_function` emits `{type, default}` only), so no model was ever TOLD the params are 1-based. Fixes (SOTA posture: Claude Code's clamp-tolerant 1-based `view_range` with `-1` EOF sentinel): (1) BATCHED validation — one refusal names every invalid range param with the file's real line count, per-param teaching, and "omit start_line/end_line to search the whole file"; (2) tolerance for unambiguous intent — `start_line=0` clamps to 1 and `end_line>total` clamps to EOF, each with a visible `Note (line range)` disposition; `-1` is the documented EOF sentinel; `end_line=0` stays refused (genuinely ambiguous) with full teaching; (3) stale-scope misses are now DEFINITIVE — a scoped no-match probes the whole file and either reports "N match(es) exist outside this range (first at line Z); line numbers may be stale — re-read or omit the range" or "not found anywhere: fix the pattern, not the range" (previously a speculative "may exist outside"); (4) scoped successes disclose `(searched lines X-Y)` and name exact matches outside the scope; (5) `start_line`/`end_line` now carry JSON-Schema `description`s in the exported tool definition (1-based teaching at call time — closing the actual gap that produced the trace); (6) docstring/description now name all THREE modes (find/replace, range-replace with `pattern=""`, unified diff) and the partial-replacement hint teaches re-read-before-reusing-line-numbers (line numbers shift after every edit). 12 new range-UX regression tests incl. a byte-identity pin of the unscoped render; tests/tools 576 passed / 21 skipped. Framework-wide remainder filed: parameter descriptions are still invisible for every OTHER tool (native schema emission + prompted-mode compact format) — see backlog 0838.
- **`analyze_code` next-step hint no longer steers models into needless `edit_file(start_line/end_line)` calls (auto-hint audit, live trace 2026-07-26)**: every outline's second line taught "then edit_file(start_line/end_line) for a bounded edit" — presenting the optional 1-based search-scope limiters as THE calling convention. A live abstractcode-tui trace showed a model dutifully passing them (0-based, unneeded), getting refused, and burning a turn; the hint also taught no staleness discipline (outline line numbers shift after every successful edit). The hint — previously two byte-duplicated copies (deep lanes in `common_tools.py`, engine lanes in `code_analysis.py`) — is now ONE constant (`code_analysis.ANALYZE_CODE_NEXT_STEP_HINT`) teaching the real contract: edit_file wants a short UNIQUE pattern with no line params; start_line/end_line (1-based) only disambiguate repeated matches or bound a range replace (`pattern=""`); line numbers go stale after every edit — re-run analyze_code or re-read before reusing them. The analyze_code docstring's `edit_file` mention aligned; a dead `search_files(output_mode="context")` code comment in read_file corrected (that mode never existed; context rides `context_lines=N`). Hint-contract tests pin the semantics, cross-lane consistency, the 1-based numbering truth, and the anti-regression (the old `edit_file(start_line/end_line)` teaching can never reappear).
- **analyze_media escalation: browser_probe now DECLARES its screenshot as media + hint honesty (operator escalation via code-tui c5568, fable5-investigated, 2026-07-25)**: an operator hit `analyze_media` refusing ("no vision model configured for delegated sight") mid-run while the SESSION model (gpt-5.6-sol) had visual encoders. The adversary settled the real causes and corrected two of my earlier assumptions: (a) the refusal is a first-run CONFIG gap (`VisionConfig.strategy` defaults to disabled), NOT a capability-detection miss — the fuzzy matcher already resolved `gpt-5.6-*` onto the `gpt-5` row (vision=True), so the in-tree registry additions fix context-window/sampling-param PRECISION but do NOT change the refusal; (b) a mid-run capture CAN reach a vision-capable main model IF the tool declares a `media` output (the shipped `c3969` shape-A sight lane, as the camera tools do) — it was the only-sight-path fallback only for tools that report a path as TEXT (like `browser_probe`). Core-lane fixes: (1) `browser_probe` now returns `{"rendered": <report>, "media": [screenshot_path]}` when a screenshot is captured (opt-in — plain string otherwise, so the pre-existing contract holds for the 99% no-screenshot case), so a sight-lane host attaches the shot to the next model call and a vision-capable main model sees the render NATIVELY without any vision-fallback config — the direct unblock for the incident; (2) the `analyze_media` refusal hint corrected to the THREE-WAY geometry (agent lane-owner correction c5662, superseding an intermediate two-way scoping of mine that wrongly cited a browser_probe screenshot as text-path-only after I had just made browser_probe DECLARE media): (i) a capture tool that declares media + a vision-capable main model → seen natively on the next call (analyze_media not needed); (ii) declared media + text-only main model → vision fallback/analyze_media; (iii) undeclared capture (path printed as text / hand-fed) → analyze_media is the only sight. The hint now says so and points at configuring the fallback at the session endpoint for cases (ii)/(iii). Immediate zero-code operator remedy: `abstractcore --config` → point the vision fallback at the session endpoint (works today, even on the stale registry). Remaining cross-package options (gateway first-run vision seeding; a `_session_route` hidden-param + runtime stamp for tool-side session-model delegation, which needs a c3977 ruling amendment) filed as proposed 0837. 74 browser/inventory/analyze_media tests green. Also folded a raster-set consistency fix (code-tui c5573): `media/auto_handler.py`'s `image_formats` set was missing `tif` (had only `tiff`) while `analyze_media` and `types.py`'s `EXTENSION_TO_MEDIA_TYPE` both accept `.tif` — a bare `.tif` was classified an image everywhere except that one support check; now aligned. Canonical accepted raster set: `{.jpg,.jpeg,.png,.gif,.webp,.bmp,.tif,.tiff}` (MIME via `mimetypes.guess_type` → `image/tiff` for both tiff spellings); honest caveat recorded that most vision providers reject TIFF/BMP so the provider-reliable subset is png/jpeg/gif/webp. media_handling suite green (1 unrelated OpenAI-429 quota failure).
- **`edit_file` exact-match-first escape handling (0829) — you can now write a literal escape sequence into source (operator-approved, tools audit 2026-07-25, adversary-confirmed defect + fable5 review)**: `edit_file` unconditionally rewrote literal `\n`/`\t`/`\r` in BOTH `pattern` and `replacement` into real control characters BEFORE matching (`_normalize_escape_sequences` at both call sites), so a caller inserting a literal escape into source code (e.g. `sep = "\n"` in JS, `re.split("\\n", x)` in Python) got a real line break written — silent corruption on unguarded types (.js/.c/.md/.txt) and an UNFIXABLE retry loop on guarded ones (.py/.json/.yaml: the parse guard blamed the caller's edit without revealing the tool's rewrite). No encoding could get a literal backslash-n through, and regex patterns/templates using `\n`/`\t` were corrupted before `re.compile`/`re.sub`. Fix: `pattern`/`replacement` are kept VERBATIM; escape-normalization survives only as a LABELED fallback on the PATTERN — if the raw literal pattern matches nothing but the escape-normalized pattern does (a weak model over-escaping the pattern), the tool swaps to it and discloses the unescape in the output; the replacement is NEVER rewritten (a weak model that over-escapes only the replacement now writes a visible, correctable literal instead of silent corruption), and regex mode is never normalized. The escape note rides EVERY post-swap surface so a refusal or no-op can never hide a rewrite — the success render, both guard-refusal paths (python-syntax + generic), the ambiguity refusal, and the "no changes applied" path. Matches how Claude Code/Cursor/aider all do character-exact matching. A fable5 depth/quality review returned SOLID (no blocking defect) and its two low-severity polish items were folded (the missing disclosure on the ambiguity + no-op surfaces, and two non-discriminating regex tests replaced with discriminating ones that fail on the old code). Live-verified across the headline case, the fallback, regex integrity, range-replace, and match-literal-escape-source; 68 existing edit_file tests + 10 new literal-escape regression tests + full tools suite green.
- **Tool audit fixes 0827/0828/0831/0834/0835 — five builtin-tool defects/gaps closed (operator-directed, tools quality audit 2026-07-25, each verified + regression-tested)**:
  - **`skim_files` structure detection was entirely dead (0827)**: the five structure/heading regexes were raw strings with DOUBLED backslashes (`r"^#{1,6}\\s+\\S"` etc.), matching a literal backslash instead of the whitespace class — so heading/setext/list/`def`-`class`/sentence detection never fired and the tool silently fell back to bookend + paragraph sampling. Un-doubled and HOISTED to module-level compiled constants (`_SKIM_HEADING_RE`/`_SKIM_HR_RE`/`_SKIM_LIST_RE`/`_SKIM_CODE_DECL_RE`/`_SKIM_SENTENCE_END_RE`) so they are unit-testable in isolation and cannot be silently re-broken inside the closure. Regression test asserts each pattern is live + a mid-file heading is surfaced. 9 skim tests green.
  - **`read_file` char-offset continuation corrupted non-ASCII files (0828)**: `_partial_chunk` opened the file in TEXT mode and `seek()`ed with a value computed in CHARACTERS — but `TextIOWrapper.seek` treats a non-cookie argument as BYTES, so on any non-ASCII file the `start_char=` continuation named in the `#TRUNCATION` footer landed at the wrong position, silently re-reading overlapping content and splitting a multibyte codepoint into a U+FFFD at the chunk head. Now byte-true: binary read, byte seek, decode on a codepoint boundary (trim up to 3 trailing bytes when not at EOF so the seam never splits a char), byte-true continuation offset; the range-overflow path counts bytes too. `start_char` is documented as a byte offset (the model copies it verbatim from the footer). Verified: a 200k-char multibyte file reassembles EXACTLY across 3 chunks with zero U+FFFD; ASCII parity preserved. 2 regression tests.
  - **`search_files` silently discarded four parameters (0831)**: `context_lines`/`case_sensitive`/`output_mode`/`ignore_dirs` were accepted and dropped (`_ = (...)`) — the exact silent-default the suite's arg-coercion philosophy forbids. Now all honored: `context_lines` (0-10) emits surrounding lines with a `-` separator (vs `:` for matches) and `--` between non-adjacent groups — the single highest-leverage token saver, avoiding a follow-up `read_file` per hit; `case_sensitive` toggles `re.IGNORECASE`; `output_mode` supports content|files_with_matches|count with an explicit error for unknown values (never silent); `ignore_dirs` extends the skip set; the multiline lane uses `content.split("\n")` for line-model consistency with the numbering and bounds the whole-file read (labeled `#TRUNCATION` when capped). `context_lines=0` output is byte-identical to the previous format (backward compatible). A fable5 review hardened it: the `files_with_matches`/`count` modes now LABEL the multiline byte-cap (they previously undercounted or returned a false "No matches" for a match past the cap — the unlabeled-truncation class the ADR forbids), `ignore_dirs` accepts a list as well as a comma-string (models send arrays for plural params), the multiline cap is a monkeypatchable module constant, and the line loop breaks early once the match cap is hit (was scanning to EOF). 9 regression tests.
  - **`analyze_media` had no `.abstractignore` gate (0834)**: it was the ONE file-reading tool that shipped file bytes to a possibly-remote vision route without the ignore-policy check every sibling enforces. Added the check BEFORE the vision handler is even imported, so an ignored path is refused and no bytes leave the host. Regression test uses a tripwire handler to assert zero dispatch on an ignored path.
  - **Consistency & robustness polish (0835)**: (a) IMAP `IMAP4_SSL(host, port, timeout=timeout)` at both `list_emails`/`read_email` sites — the connect was previously unbounded (the socket timeout was set only AFTER the constructor had connected), so a black-holing host pinned the tool for the OS TCP default; the SMTP lane already passed `timeout=`, IMAP now matches. (b) `list_files` `head_limit` default unified to 10 across signature/docstring/string-coercion fallback (the docstring + fallback said 25, drifting from the signature's 10); and path-shaped globs (`src/*.py`, `**/*.py`) now return an explicit hint (patterns match NAMES only — scope with `directory_path`, use `search_files` for path/content) instead of silently matching nothing. (c) both telegram tools gained `when_to_use`/`examples` (they were the only 2 of 27 builtins without them).
- **Registry refresh: Anthropic gen-5 models + architecture patterns (operator directive, 2026-07-25)**: `model_capabilities.json` was missing the entire 5th-generation Anthropic lineup — `claude-opus-5` (flagship, released 2026-07-24), `claude-opus-4-7`, `claude-sonnet-5`, `claude-fable-5`, and `claude-mythos-5` (Project Glasswing). All added from the official Anthropic model catalog: 1M context, 128K max output, native tools + structured output, high-resolution vision (Sonnet 5 at 2576px), and — LOAD-BEARING — `thinking_control_mode: "adaptive"` on every gen-5 entry so the Anthropic provider routes reasoning through `output_config.effort` (with `max_effort_supported: true` mapping `xhigh`→`max`) instead of the deprecated `thinking: {budget_tokens}` path that gen-5 models reject with a 400. Verified: capability lookups resolve vision/context/effort for all five, and the thinking-control mapper emits the adaptive+effort shape with zero `budget_tokens`. `architecture_formats.json` `claude`/`gpt` pattern lists refreshed with current-generation dash-form tokens (documentary — the broad `claude`/`gpt` substring patterns already matched, verified via `detect_architecture`). 278 model rows total; both assets parse clean.
- **gpt-5.6 family added to the capability registry + `analyze_media` hint honesty (code-tui incident, commons c5520, 2026-07-25)**: a live coder run on `endpoint:airelay / gpt-5.6-sol` had `analyze_media` refuse with advice the agent lane cannot follow, and the root cause was two-fold. (1) REGISTRY GAP: `model_capabilities.json` had 17 gpt-5* rows but none for the 5.6 family — `gpt-5.6-sol`/`-terra`/`-luna` (GA 2026-07-09) were invisible to every capability lookup, so nothing could know these models have visual encoders (vision routing, context-floor checks, token-param selection all blind). Added all three from the official OpenAI model cards: 1,050,000-token context, 128,000 max output, text+image input / text output, native tools + structured output, thinking support, `max_completion_tokens`, `gpt-5.6` alias → Sol; verified the effort-suffixed slugs the operator actually runs (`gpt-5.6-sol-medium`) resolve through fuzzy matching. (2) HINT HONESTY: the `VisionNotConfiguredError` refusal said "…or use a vision-capable main model" — unreachable advice in the agent lane, where a tool-captured image (e.g. a `browser_probe` screenshot) never enters the main model's image lane regardless of its encoders; `context.attachments` is the only media path into agent LLM calls and mid-run captures are not session attachments. The hint now leads with the remedy that works in EVERY lane (configure the vision fallback — any vision-capable route, including the caller's own chat endpoint/model) and states explicitly that for tool-captured images the fallback is the only sight path. Tools suite 534 green.
- **LLM read-idle (no-progress) timeout — a stalled stream now aborts at the socket instead of pinning a worker for hours (runtime routing c5004 "face 2", confirmed c5041, 2026-07-23)**: a no-progress LLM stream held a worker up to the full `DEFAULT_LLM_TIMEOUT_S` (7200s / 2h) — the 40-minute-stuck-tick class. Reframing finding: the no-progress bound ALREADY EXISTS in the HTTP layer (httpx's `read` timeout — the max wait for the NEXT response chunk — and requests' `(connect, read)` tuple), it was just CONFLATED with the total (a single timeout value set `read == 7200s`, present-but-useless). So the fix is separation, not a watchdog (a thread watchdog would be a lie — unkillable threads). New provider param `read_idle_timeout_s` (base `_read_idle_timeout`, None = unchanged so every consumer that doesn't opt in is byte-identical) sets the httpx client's `read` timeout distinctly from the total `connect`/`write`/`pool`; the runtime factory threads it per-lane (entity 120s, non-entity default). Shared helper `providers/_http.py::build_read_idle_timeout` wires every httpx lane with one implementation: `openai_compatible_provider` (+ its lmstudio/openrouter/portkey/vllm subclasses — the primary lane, incl. the OVH/vLLM endpoints where the wedge lived), `ollama_provider`, and `openai_provider`'s audio blocks. Live-verified: a stalling server (total 30s, read_idle 2s) aborts at 2.0s, not 30s. A read timeout is already a RetryManager-classified retryable, so no new error plumbing. 6 pins + live smoke. FOLLOW-ON: the SDK lanes (native-OpenAI chat via the openai SDK, Anthropic via its SDK) accept an `httpx.Timeout` and get the same read-idle in a separate pass; the httpx lanes (the wedge lane included) are covered now.
- **`execute_command` timeout now kills the whole process TREE — no orphaned grandchildren, no pinned tick thread (runtime routing c5004 "face 1", 2026-07-23)**: `subprocess.run(shell=True, timeout=, capture_output=True)` on timeout killed only the direct child (the shell), leaving its grandchildren orphaned; and — the subtler, load-bearing half — the captured-pipe read WAITS FOR PIPE EOF, so an orphaned grandchild still holding stdout kept that read (and the calling tick thread) alive FOREVER even after the shell "died" (the classic subprocess gotcha behind a live 40-minute stuck tick, e.g. an orphaned browser). Now `execute_command` runs via `Popen(start_new_session=True)` + `communicate(timeout)`, and on timeout does a process-TREE kill (reaching a `setsid()`-escaped grandchild like a launched browser) then a HARD-BOUNDED drain — so the captured pipes close and the thread can never be pinned. The tree-kill is the exact pattern proven in `browser_tools` this week, now extracted to one shared stdlib-only module `abstractcore/tools/process_tree.py` (`descendant_pids` + `hard_kill_tree`) that both `browser_probe` and `execute_command` use (no duplication; `browser_tools` aliases the historical names). Normal/non-zero/`capture_output=False` semantics preserved; new pins: the incident shape (backgrounded grandchild holding stdout + shell sleeping past timeout → returns at ~timeout, honest timeout verdict, zero orphans) + stderr/non-zero capture. execute_command + browser tree-kill pins green. (Face 2 — a socket-level no-progress LLM timeout in the provider client — is claimed as a separate follow-on.)
- **`fetch_url` 401 guidance no longer invites a credential paste at loopback/private hosts (code-tui 401-incident root-cause, commons c4978, 2026-07-23)**: the `auth_required` (401) suggestion said "supply credentials via headers if you have them" unconditionally — a prompt-injection footgun when a model holds a token in context (e.g. the gateway bearer) and the target is the loopback control plane (the incident: an agent fetch_url'd its own gateway and got a correct 401). Now the classifier takes the final (post-redirect) host and, for loopback / private / link-local / `localhost` / `.local` targets (`_is_loopback_or_private_host`, via stdlib `ipaddress`), gives safe guidance instead — "do NOT paste tokens; a local control plane authenticates through its own session, not a model-supplied token." Public hosts keep the original hint; a host-less call keeps the backward-compatible default. This is the softer sub-fix that needs no ruling; the broader same-origin/control-plane GUARD on fetch_url is the operator's call against the standing base64-exfil "ONE protection" ruling (routed to the operator, deliberately not changed here). 61 guidance + egress-screen pins green.
- **`model_controlled_destination` now set on `fetch_url` and `browser_probe` (runtime ruling, commons c4879, 2026-07-23)**: both adversaries flagged that `risk_facts.py` cited `fetch_url`'s URL as THE archetype of `model_controlled_destination`, yet the inventory set it on neither `fetch_url` nor `browser_probe` — an inconsistency I raised to the enforcement-lane owner. Runtime ruled: set it on both (their approval lane consumes it via a risk-rank ceiling-SKIP so a model-controlled-egress tool is never silenced by an operator ceiling — the guard derives from the fact, never a name list that can rot). Both tools' egress destination IS model-chosen (`fetch_url`'s URL/method; `browser_probe` navigates + executes a model-chosen URL's JS). The fact is band-NEUTRAL — both stay `act` (mcd feeds the approval rule, not the tier fold). `RISK_MAPPING_VERSION` deliberately NOT bumped: the version tracks the fact vocabulary + derivation mapping (both unchanged), a band-neutral per-tool classification change propagates because consumers derive from the served fact fresh (runtime's guard "holds by derivation"), and a bump with an unchanged mapping would contradict what the version means and force spurious re-derivation. Inventory + risk_facts suites 36 green.
- **Image `/v1/images/generations` param-shaping consulted the VIDEO route (regression from the vision config-wins work, found by an adversary 2026-07-23)**: `_image_generation_request_parts` passed `modality="video"` to `_effective_backend_kind`, which is modality-sensitive (it reads the config `output.<modality>` route). So an operator running DIFFERENT image and video backends (e.g. image = local diffusers, video = an OpenAI-compatible proxy) got image requests shaped by the video route — `width`/`height` dropped and a `size` string sent even though the local image backend needs `width`/`height` — the exact cross-modality bleed the config-wins work set out to kill, mirrored. One-line fix (`"video"` → `"image"`); the sibling `_video_generation_request_parts` was already correct. Pinned both directions (image=local+video=proxy keeps width/height; image=proxy+video=local size-shapes) — the differing-routes case the prior tests never exercised. Vision config-precedence suite 19 green.
- **`list_files` no longer walks + stats the whole tree before applying `head_limit` (operator directive dm 2026-07-23, extends the search-perf wave)**: the same eager-enumeration class hit `list_files` — `list_files(head_limit=100)` over a 130k-file tree walked, `is_ignored`'d, AND mtime-stat-sorted all 130k entries before slicing to 100. Three fixes per the operator's design: (1) entries STREAM and collection STOPS at `head_limit` + a bounded look-ahead — the old global most-recent-first sort is PRESERVED for normal trees (when the stream exhausts within the budget, the full matched set is sorted with exact counts), and only a tree LARGER than the budget switches to fast stream-order; (2) a more-files hint on the large-tree path advises narrowing (subfolder / specific pattern / tighter regex) instead of blind re-listing; (3) a bounded per-extension/per-subfolder composition summary (`"404 .py; 100 .md | subfolders: pkg/ 404"`) built ONLY from the streamed look-ahead, labeled "of what was scanned" (partial) on large trees — never a second walk (the operator's "if not computationally costly" qualifier). `target/` default-ignore + the `is_ignored` double-`resolve` cut from the `search_files` fix help `list_files` for free. Regression pins: `is_ignored` calls stay bounded over a 2000-file tree; normal trees keep exact counts + most-recent-first; large-tree summary labeled partial. Full tools suite green.
- **`list_files` adversary fold (fable5, 2026-07-23 — verdict SHIP-WITH-FIXES, all three findings fixed)**: the equivalence harness proved the streaming rewrite byte-identical to the old output in 13/16 cases and the stat attribution proved the perf claim (the 500-entry look-ahead is never mtime-stat'd; ~cap-bounded `is_ignored` calls; 62–151ms flat across 5k–50k trees vs the old full-tree walk). Three regressions it caught are folded: (F1/P1) the empty-result message lost the old "matching hidden entries exist" disambiguator — an agent probing for `.env` was told nothing matched; the streaming generator now flags pattern-matching hidden entries as it skips them (including hidden dirs pruned from the recursive walk, which never reach the per-entry loop) and the empty branch names the recourse. (F2/P2) directories with dots polluted the extension composition (`pkg.v0.data/` counted as a `.data` file) — the generator now yields `(path, is_dir)` (free from the walk in recursive mode; one bounded `is_dir()` per collected entry otherwise) and composition counts files only. (F3/P2) the non-recursive path materialized the whole directory via `list(iterdir())` (~130k transient Path objects on the incident tree) — it now iterates the generator directly with the `PermissionError` try around the loop. New pins: the hidden-hint message (both modes), files-only composition, and the previously unpinned collect-cap boundary (510 exact vs 511 "many"). list_files+UX suites 27 green; tools suite 455 green (1 unrelated flaky fd-count test passes in isolation).
- **`search_files` no longer walks the whole tree before matching — an 8m39s single call fixed (forensic report from code-tui, 2026-07-23)**: a `search_files` call over a ~196k-file tree ran 8 minutes 39 seconds (ledger-proven), because enumeration was EAGER — `os.walk` collected the full candidate list, running `AbstractIgnore.is_ignored()` (which itself called `path.resolve()` TWICE per file) plus a 1KB utf-8 binary sniff on EVERY file, and `max_hits` capped only the match loop, never the walk. Four fixes: (A) enumeration is now a LAZY generator streamed into the match loop, so hitting `max_hits` stops the `os.walk` early and the binary sniff runs only on files actually reached; the exact remainder count is kept for small trees via a bounded budget (≤500 extra candidates) and degrades to a "more may exist" note on large trees instead of paying the full walk. (B) `target/` (the Rust/JVM build tree) is now a default ignore in both the tool's inline set and `AbstractIgnore` defaults — the twin of `node_modules`/`dist`/`build` (a 61k-file `target/` was half of the incident tree). (C) `AbstractIgnore._rel` no longer re-`resolve()`s a path the caller already resolved — the double `resolve()` was ~52% of the warm-cache scan cost. (D) results carry a scan-cost line (`scanned N files in Ss — narrow path/file_pattern or add .abstractignore`) on large/slow scans, teaching the model to narrow instead of walking blind (the `include_hidden=true`-over-a-giant-dot-dir foot-gun). Regression pins: `is_ignored` file-level calls stay bounded near `max_hits` + the remainder budget over a 2005-file tree; `target/` pruned by default. Full tools suite green.
- **`coerce_arguments` no longer kills tool calls carrying explicit `null` for typed optional parameters** (found + fixed by the code seat during the memgraph benchmark, adopted by core 2026-07-20): callers routinely emit `{"head_limit": null}` for optionals; JSON Schema treats null as a distinct type, so coercing it would be invention and raising turned well-formed optionals into hard failures (a flow-composed `list_files` gate died on every dispatch). An explicit `None` for a typed non-string field now DROPS the key with a `#FALLBACK` warning — the tool's own default applies, same outcome as omission. String parameters keep their existing `None` passthrough (existing sentinel behavior consumers may rely on; non-string fields previously raised, so nothing ever received `None` there). Coercion suite 23 green; tools suite 371 green.

### Added
- **`git_read_only@v1` per-call refiner declared on `execute_command` (runtime built it c5042, code-tui client-proof retires c5050, 2026-07-23)**: `execute_command` now carries `risk_refiner="git_read_only@v1"` (added to `risk_facts.KNOWN_REFINER_IDS` + `inventory._REFINER_BY_NAME`), the same architecture as `send_email_recipient@v1` — core DECLARES the refiner id on the row, the enforcement lane (runtime) implements the two-stage conservative proof (positional write verbs like `remote set-url`/`reflog expire`, write/exec flags, wrappers, globals-before-verb, and shell operators all ASK; only a proven read-only `git` invocation auto-approves). The grant-time band stays `destroy(4)` (the ceiling + deny-safe default; the refiner is band-NEUTRAL and only LOWERS a proven-safe call at the approval point). Declaring it retires the thin clients' hand-rolled git allowlists (code-tui's 330-line shell proof, abstractcode's Python twin). Pinned: execute_command carries the refiner, band unchanged, exactly two declared refiner carriers (send_email + execute_command) with the fail-closed stale-entry check; inventory + risk_facts suites 40 green.
- **Session permission-mode vocabulary served by core (tier/gating clean-pathways ruling, commons c4909-c5028, 2026-07-23)**: the operator ruled that thin clients must READ+FORWARD and the server owns the gating logic — but the session-autonomy posture ladder (`read-only`/`write`/`full-auto`) was CLIENT-INVENTED (a name table in each client that duplicates + drifts from core's served facts). Core now SERVES the ladder in `risk_facts.py` (its natural home, beside the risk mapping the ladder reads): `PERMISSION_MODES` (the ruled words), `PERMISSION_MODE_SEMANTICS` (one-line per mode so a client renders without inventing copy), and `permission_mode_max_auto_rank(mode)` → the risk-rank CEILING each mode auto-approves up to ({read-only:1 observe, write:2 observe+act, full-auto:4 all}; unknown→read-only, fail-CLOSED never fail-open). A client forwards the WORD; the server derives per-tool auto-vs-ask by comparing each tool's served `risk_rank` to the ceiling — no client name table. `permission_mode_auto_approves(mode, *, risk_rank, model_controlled_destination)` is the ONE served decision that combines the ceiling with the mcd belt (a `model_controlled_destination` tool — fetch_url/browser_probe — is `act(2)` ≤ the write ceiling, so the ceiling ALONE would wrongly auto it under write; the helper makes it ASK below full-auto), so a consumer cannot apply the ceiling and forget the belt (gateway c5053: "a consumer that cannot forget the belt beats a documented warning"). require-always-wins > per-tool overrides > mode default remains the enforcement-lane precedence. Content is the enforcement lane's to change under `RISK_MAPPING_VERSION`, same rule as the fact→band mapping. Pinned: the vocabulary + ceilings, fail-closed unknown, and cross-checks against the risk band (observe autos read-only; act autos write not read-only; destroy asks under write). risk_facts suite 17 green.
- **Capability default-off declaration hook (`register_capability_tool_defaults` / `capability_tool_defaults`, runtime ruling c4886 + gateway c4892, 2026-07-23)**: camera capture tools (photo/video/open/detection) were silently available to any agent in "untouched = workflow decides" mode — a privacy gap (operator ask via abstractcode-tui). The room converged, and core corrected its own first instinct: a static `default_enabled` flag in the registration was the wrong home (enablement is config-toggleable state, not a registration constant — dm#177). The DERIVE-OVER-HARDCODE design instead: a capability DECLARES its privacy/default-off class through core (same shape as the approval partition `register_capability_tool_policy`: `{"default_disabled": [tool names]}`), and the gateway SEEDS those tools disabled in its settings registry — so default-off follows the plugin's declaration, never a hardcoded `"camera"` string a maintainer must remember to extend for the next environment-capturing plugin (mic/screen); the forgotten-name failure is silent availability, the exact class this closes. SEED-ONLY (dm#194, runtime's precision): the declaration seeds the default only when the operator's setting is unset — an explicit console enable supersedes it forever after; the host owns that precedence, core only carries the seed. Seat split: core ships the hook (this change); abstractcamera declares its class; gateway derives the seed + serves the per-tool `enabled:false` discovery row (both thin clients already read that path, zero client change); runtime's `disabled_toolsets` composition filter (already shipped, 5 pins) enforces absence-from-run-registration so a workflow bypassing discovery still can't get a seeded-off tool. 8 new pins (register/retrieve, copy-isolation, re-register-replaces, dedupe, empty/None clears — distinct from facts' empty-dict refusal since "no tool off" is legitimate, bad-shape refusal, capability-general); capability-tools surface suite 33 green.
- **`browser_probe` — headless-browser render verification tool (operator-approved dm:core--laurent#21, 2 fable5 adversaries, 2026-07-23)**: answers the one question `read_file` cannot — *does this page actually render?* An agent writes an HTML/JS app, the source looks right, and the page renders blank (OpenAI's codex hit the same class, issue #14755, which used a `browser_probe(require_nonblank=…, target=…)` shape). New `abstractcore/tools/browser_tools.py`: a Playwright-backed probe that renders a URL or local HTML file in a headless Chromium shell and returns a `PASS`/`FAIL` verdict + navigation outcome, HTTP status, `readyState`, title, visible-text stats, per-check results (`require_nonblank` = visible text OR canvas/svg/img/video so WebGL/canvas games aren't false-blanked; `expect_selector`; `expect_text`), captured console errors + uncaught exceptions, blocked-network report, redirect disclosure, and an optional screenshot path for `analyze_media`. Design decisions: (1) it's a CORE TOOL behind an optional **`browser`** extra (not a capability plugin — capability plugins are server modalities; the ~200-400MB Chromium is a `playwright install` post-step either way), lazy-imported like `requests`/`bs4` with a two-step install hint when absent; (2) the whole probe runs in a WORKER SUBPROCESS killed at `timeout_s` + launch grace via a process-TREE kill — the real "never hangs on an infinite-JS-loop page / never leaks a browser" guarantee (an in-process CDP call has no timeout and can wedge; `chrome-headless-shell` `setsid()`s out of the worker's group so a group-only kill would orphan it — `_hard_kill_tree` walks ancestry to reach it); (3) readiness is a CONTENT signal, never `networkidle` (refused with teaching: background polling/websockets make it flaky by design) or `sleep`; (4) inventory classification `{"mutating": False, "remote_write_capable": True}` → risk band `act`, byte-identical to `fetch_url` (navigating a model-controlled URL executes that page's JS = the same escape-hatch class; screenshots land only in a fresh temp dir = no local-write surface); (5) local `file://` targets BLOCK outbound network by default (anti-exfil, blocked requests reported) and — a browser CORS limitation the report DETECTS and flags — cannot load ES modules or `fetch()`, so a modern module/fetch app renders empty as a file and must be served over http. Two adversarial reviews (design + security/robustness) plus a third VERIFICATION review of the fold ran; all endorsed the packaging, string return, and risk band. Folded and live-verified: the design review's P0 (ES-module `file://` false-blank steering agents wrong) and four P1s (iframe-scope disclosure, check-less-PASS qualification, the macOS browser-leak, redirect disclosure); then the verification review's three P1s — redirect false-positive on URL normalization (a slashless host like the tool's own `http://127.0.0.1:8000` example got a phantom "Redirected to"; now compares normalized components + suppresses `about:blank`), a false "network blocked" claim when `allow_network=True` (now reports "network ALLOWED"), and leak tests contaminable by concurrent playwright agents on the same machine (now fingerprinted via a unique `TMPDIR` so the check is agent-isolated) — plus P2s (never-responding-server honest "no response committed" instead of a snapshot hang misdiagnosed as an infinite loop; capture-time CORS flag on untruncated text; `wait_until`-aware timeout wording; `_descendant_pids` retry; screenshot-dir lock). The macOS leak was structurally confirmed (`chrome-headless-shell` `setsid()`s into its own group, escaping `os.killpg`) and the tree-kill proven by a deterministic `setsid`-escaped-grandchild test needing no browser. Registered in the builtin inventory (schema v3); 34 tests (live render fixtures + the deterministic tree-kill pin + normalization/allow-network/no-commit pins); tools + inventory suites 56 green. Post-ship (agent seam, commons c4932): the `executor` tag added to the decorator — abstractagent's ReAct verifier preference is declaration-driven (verification probes PREFER artifact-executing tools over read-only review; execution catches what LLM-read review blesses), and browser_probe is exactly that class; the tag is now test-pinned as a cross-package contract.
- **KV-artifact model-config geometry gate (backlog 0817, axis 3, 2026-07-20)**: a `config.json` edit under the SAME model id — a `rope_theta` retune, a longrope/yarn `rope_scaling` block, a sliding-window layout change — re-defines what saved KV tensors MEAN while leaving no textual or tokenizer trace, so a reused artifact was silently positionally wrong. New `providers/model_config_fingerprint.py` hashes a CURATED set of geometry keys (RoPE family, window/attention layout, positional envelope, `model_type`; top-level and `text_config`) so irrelevant config churn (`transformers_version`, dtype, name metadata) never false-invalidates the corpus. Recorded at compile into the bloc-KV manifest (`model_config_fingerprint`) and every MLX artifact's metadata; ensure-gate refuses+recompiles on mismatch, the MLX load-gate raises with the model live, pre-axis artifacts reuse with a labeled `#FALLBACK`, unavailable current state abstains. GGUF deliberately abstains (config travels inside the weights file — the weights-identity axis owns it). 18 new pins; full suite 2232 green; live-verified on a real cached config.

### Documentation
- **Camera and 3D capability plugins documented across the doc set**: `docs/capabilities.md`, `docs/server.md`, `docs/README.md`, `README.md`, `llms.txt`, and `llms-full.txt` now cover the optional `abstract3d` (`llm.scene3d`, `/v1/scene3d/generations`) and `abstractcamera` (`llm.camera`, `/v1/camera/*`) plugins alongside voice/vision/music — install extras, the `generate`/facade surface, the extension HTTP endpoints, and the honest `501`-when-absent contract.

### Fixed
- **Blocking generation no longer runs ON the server event loop (music + TTS speech lanes; twin of abstract3d's scene3d adversary finding, 2026-07-19)**: `/v1/audio/music`, `/v1/audio/speech`, `/v1/audio/speech/stream` and their provider-scoped aliases were `async def` handlers that await NOTHING and call BLOCKING synthesis/generation — they executed on the event loop, serializing the ENTIRE server behind one generation for its full duration (a multi-minute music render or a wedged TTS call froze every other request — the same head-of-line CLASS as the 2026-07-17 voice outage, though gateway confirmed that incident ran abstractvoice in-process, not through core's HTTP server). Converted to sync `def` (FastAPI runs sync handlers in its threadpool); the genuinely-awaiting lanes (upload `file.read()` in transcriptions/voice-clone) stay async. Docstrings pin the rule: do not "modernize" back to `async def` without a real off-loop dispatch. Server suite 233 green; full suite 2213 green.

### Added
- **Ornith 35B GGUF (and every embedded-ChatML GGUF) reaches the fast prompt-cache lane (backlog 0821 closed, operator-directed 2026-07-19; one adversarial subagent as asked)**: GGUF control-plane detection (`_gguf_prompt_cache_control_plane_chat_format`) now admits models whose EMBEDDED Jinja chat template is ChatML — by template CONTENT (both turn markers + a cached probe render proving the template is ChatML-SHAPED: the last user turn's content inside an `<|im_start|>…<|im_end|>` pair with a generation prompt after it), never by model name or llama.cpp's guessed format id. The admitted lane is the proven `llama-cpp-chat-template` renderer (the model's OWN template — preserves Ornith's `<think>` generation opening that plain ChatML would drop); probe failure falls back to `keyed` (pre-0821 behavior, fail-safe). LIVE on the real `ornith-1.0-35b-Q4_K_M.gguf`, twice: `mode=local_control_plane`, warm/cold 0.352 → 0.338, fact recall correct, zero warnings. The requested fable5 adversary returned HOLDS-WITH-GAPS and all gaps were folded same-day: (F1, P1, reproduced — pre-existing and WIDER than 0821) the snapshot prefix reader trusted the cache map KEY length while fallback-lane writers (llama.cpp's own save after `create_chat_completion`) save states holding `len(key)−1` tokens — the reader skipped eval'ing the missing token and served KV with every later position shifted (wrong output, zero errors); fixed by `_gguf_state_held_tokens` (reuse what the state HOLDS per its own `n_tokens`/`input_ids`, eval the remainder, refuse states disagreeing with their key). (F2) marker-MENTIONING non-ChatML templates no longer pass the probe. (F3) a template refusing a conversation shape mid-session (Ornith raises on mid-history system messages) degraded on the non-stream lane but escaped as a raw ValueError at the consumer's first `next()` on the STREAM lane — render now degrades to a `finish_reason="error"` chunk on both. F4 (no lock over the control-plane prefill→sample window under concurrent keyed generates; pre-existing) recorded as a follow-up in the completed item. 24 pins in `tests/huggingface/test_gguf_prompt_cache_control_plane.py`; full suite 2190 green.
- **Sliding-window models reclassified as NOT functional in-process caches (operator ruling 2026-07-19)**: `docs/prompt-caching.md` now states the verdict directly on the MLX matrix row — GPT-OSS (128-token window) and Gemma-3/4-class (512–1024) give delta reuse only while the WHOLE transcript fits under the window, because a `RotatingKVCache` physically discards positions past it, making the rewind the delta lane needs impossible; a window smaller than any real agent transcript is not a functional cache, whatever the mechanism's correctness. The recommendation on the row: serve those models via LM Studio/vLLM, whose forward-only slot/prefix cache works regardless of the window.

- **KV-artifact tokenizer-identity gate (backlog 0817, axis 2, 2026-07-18)**: a durable KV artifact encodes the token-id stream ONE tokenizer + chat template produced; a `tokenizer.json`/chat-template refresh under the SAME model id changes text→ids and made a reused artifact silently wrong (`rendered_recipe_sha256` hashes TEXT and cannot see it — the audit's #1-danger missing axis). New provider hook `prompt_cache_tokenizer_fingerprint()` (base default `""`) with MLX + HF-transformers overrides via the new `providers/tokenizer_fingerprint.py`: sha256 over the fast tokenizer's complete serialized state (`backend_tokenizer.to_str()`) + chat template text + special-token ids → `tokenizer-full:sha256:<24hex>` (weaker `tokenizer-vocab:` tier for slow tokenizers; the tier prefix makes cross-tier compares fail-safe mismatches; GGUF deliberately `""` — its tokenizer travels inside the weights file, the weights axis owns that signal). Recorded at compile into the bloc-KV manifest (`tokenizer_fingerprint`) and into EVERY MLX artifact's safetensors meta at `prompt_cache_save`. Gates share one three-way verdict (`check_tokenizer_fingerprint`): ensure-time (`_validate_existing_manifest`) refuses+recompiles on mismatch with `#FALLBACK`, reuses pre-axis artifacts UNVERIFIED with `#FALLBACK`, and ABSTAINS silently when the current tokenizer is unavailable (ensure may run before the model loads — the load gate re-checks); MLX `prompt_cache_load` (tokenizer live) raises a loud ValueError on mismatch naming both fingerprints and the recompile fix. NOT part of `binding_id` (backfill safety, same rule as the engine axis). Live-verified on real tokenizers (stability, template-mutation flip, mlx TokenizerWrapper == inner HF tokenizer). 18 pins (`tests/providers/test_tokenizer_fingerprint_unit.py`, `tests/test_bloc_kv_tokenizer_fingerprint.py`); full suite 2182 green.

### Fixed
- **Capability route rows no longer leak across backends (live incident 2026-07-17, assistant DM + laurent's offline voice outage; adversary-hardened same day)**: a configured capability default route (e.g. `output.voice` = supertonic + `options: {voice: M1}`) had its `options` merged into EVERY matching output spec — including one that explicitly named a DIFFERENT provider/model. A piper TTS request with no voice inherited the supertonic route's `voice: M1` and failed all attempts with `Unknown voice_id: M1`. Root cause in `core/generate_contract.py`: `_route_entry` adopted the default row's `options` unconditionally while provider/model stayed per-field explicit-wins, so the resolved entry mixed one route's backend with another route's options. Fix, generalized after a fable5 red-team found the options-only first cut left same-class gaps: a route row is ONE coherent backend identity, and `_route_row_contribution()` now grades how much of it may ride — `none` when the explicit spec redirects the backend itself (provider conflict, or an explicit provider against a row anchored only by base_url): the row contributes NOTHING, closing the adversary's identity-fill leak (explicit `provider=piper` used to inherit the route's `model=supertonic-3`); `no_options` when the spec moves WITHIN the backend (different model on the configured provider, or same provider on a different base_url — the moved-server/proxy pattern): identity fields still fill, options drop; `full` otherwise. All comparisons case-insensitive; drops recorded as `field_sources["options"] = "dropped:explicit_route_override"` so route summaries stay honest; the classic base_url-only proxy override keeps its configured provider/model (pinned). Same rule applied to the STT-fallback lane (`_pop_stt_route_params` in `providers/base.py`): an explicit `stt_provider` contradicting the `input.voice` route no longer inherits the route's model (openai was silently asked for faster-whisper models). 8 regression pins (`tests/test_generate_contract.py`, `tests/media_handling/test_audio_stt_fallback.py`); full suite 2165 green.

### Added
- **`workflow-memory` data-home kind (semantics-ruled, 2026-07-15)**: the machine data registry's closed kind set (`abstractcore/utils/data_registry.py` `DATA_HOME_KINDS`) gains `workflow-memory` for abstractflow's per-workflow memory graphs (ruling dm:memory--semantics#4; plans/flow-memory-nodes.md §4/§9). Kind posture: OPERATOR data — registers `safe_to_purge=True` by default (purge/delete legal). Anti-laundering boundary carried in the governance note: an entity home or any file under one must NEVER register as `workflow-memory` — and the registry's nesting guard refuses the ancestor/descendant overlap structurally (live-verified: a workflow-memory row under a registered entity-home is refused), so the purgeable kind cannot become a route around the entity-home never-purge right. Kind-set pin updated (`tests/utils/test_data_registry.py`).

### Fixed
- **Structured output survives strict-schema backends (schema-rejection → prompted fallback, 2026-07-15)**: OpenAI-strict validators (subscription relays, responses-API backends) refuse whole requests with deterministic 4xxs when the attached JSON schema violates strict rules ("every object schema must set `additionalProperties`...", `invalid_json_schema`, "'required' is required to be supplied..."). A free-form dict (`{"type":"object"}` without `properties`) is inexpressible under those rules by construction — the operator-reported airelay/gpt-5.4 422s on every ReAct review cycle (`response_model=ReActVerifier`). Fix (new `structured/schema_compat.py` + handler wiring): a conservative, evidence-based classifier (`is_schema_rejection_error` — requires a non-auth/rate-limit 4xx AND a known strict-validator signature; auth, context-length, rate-limit, 5xx keep their existing semantics) routes schema refusals to the existing PROMPTED structured-output lane with a `#FALLBACK` warning, and a process-lifetime registry keyed by (provider class, base_url, model, schema fingerprint) skips the doomed native attempt on later calls instead of re-hitting the 4xx per cycle. The prompted lane's JSON extraction also strips reasoning noise (`<think>`/Harmony markup) via the shared `normalize_assistant_text` before regex extraction. Works for every schema on every backend — native when accepted, prompted when refused, no provider special-cases. Live-verified: free-form-dict schema on airelay gpt-5.4 falls back and validates; LM Studio native lane unaffected (no rejection recorded). 18 pins in `tests/structured/test_schema_rejection_fallback.py`.
- **Tool-call HISTORY with non-string argument values 400'd LM Studio (minja lacks `safe`) and crashed HF-GGUF template render — both lanes fixed (operator-reported "what are the news today fails 100%", root-caused live on Ornith-1.0-35B, 2026-07-15)**: Qwen3-Coder-convention GGUF chat templates render replayed assistant tool-call arguments per entry as `args_value | string if args_value is string else args_value | tojson | safe`. Two DIFFERENT failures behind one symptom: (1) **LM Studio** renders with a minja-class engine that lacks the `safe` filter, so the first request whose history carries any non-string argument value (e.g. `{"query": "news", "num_results": 10}` — every abstractcode `web_search` cycle 2) failed template rendering with HTTP 400 `Unknown StringValue filter: safe` (live-verified: all-string args 200, one int arg → the exact operator error). Fix (REACTIVE, evolved 2026-07-15 from an earlier unconditional hook — the maintainer's constraint: no new uncontrolled side effects): each non-string argument VALUE is JSON-stringified by `stringify_tool_call_history_argument_values` (`10 → "10"`, `{"a": 1} → '{"a": 1}'`; keys stay, strings untouched) so the template takes its `| string` branch, which renders byte-identical output to the `tojson` branch (property-pinned against the real Ornith template under the HF-transformers tojson convention) — but the transform now applies ONLY once the server has PROVEN it cannot render the standard payload. Unconditional application silently quoted prior args for the ~10/18 installed tool GGUFs whose templates render the WHOLE dict via `arguments | tojson` (Llama-3.2/Qwen3/Granite/Ministral conventions). Mechanism (mirrors the `stream_options`/`prompt_cache_key` rejection-latch pattern): the first call sends the standard wire; a template-render failure (`_TEMPLATE_RENDER_ERROR_RE`, extended with `cannot apply filter`) latches the per-instance flag `_lmstudio_minja_arg_stringify_needed`, emits ONE `#FALLBACK` warning naming the cause, and retries the SAME request ONCE with stringified history args; later calls on the latched instance stringify proactively in `_mutate_payload` (no wasted first attempt). Wired at all four request sites (sync/async × stream/non-stream) via the `_render_400_repaired_payload` hook — base `OpenAICompatibleProvider` never repairs (vLLM/OpenRouter/Portkey render with real Jinja2; shared wire untouched, test-pinned). Bounded by construction: the hook returns None once latched (stream-recursion guard), when the transform would be a no-op (a render failure with all-string args has a different cause — no burned retry), and on non-render 400s; a failing retry raises `InvalidRequestError`, which the outer RetryManager classifies non-retryable, so one logical call is never more than two HTTP requests (pinned through the full `generate()` path). Live find while validating: on `stream: true` LM Studio reports the render failure as the FIRST SSE event of an HTTP 200, not a connection-time 400 — the stream lanes therefore also consult a `_stream_error_event_repaired_payload` hook before anything was yielded (repair re-establishes the stream invisibly; errors after the first yielded chunk stay loud, never re-requested — replaying delivered content would duplicate it). `_raise_for_status`'s template-render warning remains for TERMINAL failures (retry also failed, no-op transform, or non-LM-Studio servers). Copy-on-write (caller session history never mutated); idempotent (a latched re-apply never double-encodes); live RESPONSE tool_calls parsing untouched. Live-verified on the operator's LM Studio + Ornith-1.0-35B: non-stream turn-2 transcript → attempts `[400, 200]`, `finish_reason=stop`, correct summary; latched second call → one stringified attempt; fresh streaming instance → error-event repair → re-established stream, `finish_reason=stop`, correct content. Reactive pins in `tests/providers/test_lmstudio_reactive_stringify_unit.py` (23 tests: bounded single retry, latch persistence, no-op skip, LM-Studio-only scope, sync+async+stream+non-stream, in-stream error-event lane, idempotent re-apply, payload preservation). (2) **HF-GGUF** (llama-cpp-python) renders the same template with real Jinja2 (`safe` exists) but the OpenAI wire convention carries `arguments` as a JSON STRING, and `tool_call.arguments|items` on a string raises `TypeError: Can only get item pairs from a mapping` — fixed by the fallback-lane bridge `_gguf_normalize_tool_call_arguments_for_template` (JSON-string arguments parsed to dicts before `create_chat_completion`), now pinned against the real template in llama-cpp's exact jinja2 environment. 12 pins in `tests/providers/test_lmstudio_tool_history_stringify_unit.py`, 5 in `tests/providers/test_huggingface_gguf_tool_history_template_render_unit.py` (real template fixture: `tests/fixtures/ornith_chat_template.jinja`).
- **Output-token cap no longer imposes a SILENT budget (ADR 0001 no-silent-degradation + use-full-capability; flow investigation, maintainer-flagged 2026-07-15)**: every OpenAI / OpenAI-compatible call previously shipped `max_tokens = <registry max_output_tokens>` even when the caller imposed NO cap — a silent per-call output ceiling (e.g. 8192 for `gpt-oss-120b`, whose context is 128k), so any response needing more was truncated with only `finish_reason=length` (unsurfaced) as the signal. Fix: (1) a sentinel in `core/interface.py` distinguishes "caller specified a cap" from "defaulted" via `self._max_output_tokens_explicit`; (2) `_prepare_generation_kwargs` no longer fabricates the registry default when the caller gave none; (3) new `_resolve_output_token_cap()` returns the explicit cap, else `None` ("omit — use full capability") unless the provider's API requires a bound (`_requires_output_cap()`); (4) the openai / openai-compatible call sites (sync + async) omit the token param when `None`; (5) providers that require a bound keep sending the model's true registry max (Anthropic Messages API, Ollama `num_predict`, LM Studio native REST via `_requires_output_cap=True`); (6) **no silent truncation**: `_annotate_output_truncation()` warns (`RuntimeWarning`) + sets `metadata["output_truncated"]=True` whenever `finish_reason=length`. Explicit caller caps (per-call or constructor) are honored verbatim and still clamped to the model's hard max. Live-verified on OVH vLLM `gpt-oss-120b`: no-cap request omits `max_tokens` and completes `finish_reason=stop`; an explicit `max_output_tokens=64` is sent and its truncation is annotated + warned; a constructor cap of 100 is honored. Regression pins in `tests/providers/test_output_token_no_silent_budget.py`. **Owner review + fable5 red-team follow-through (core, 2026-07-15):** the fix's promise was completed on the lanes the same-day patch missed — (a) the ASYNC lane now annotates truncation (`agenerate` non-stream + a per-chunk `_annotate_async_stream` wrapper); it previously returned a truncated result with zero signal (live-proven on OVH/LMStudio). (b) STREAMING truncation is annotated per processed chunk. (c) An instance-level `requires_output_cap=True` knob (constructor kwarg / endpoint-profile field, consulted by `_requires_output_cap()`) forces the bound for arbitrary `base_url` servers whose omit-default TRUNCATES — the confirmed case is Text-Generation-Inference (~100-token default on absent `max_tokens`); no new subclass needed. (d) Ollama non-streaming now maps `done_reason`→`finish_reason` (was hardcoded "stop", making the truncation guarantee a no-op on the default processing provider). (e) the `gpt-oss-120b`/`gpt-oss-20b` registry `max_output_tokens` under-cap (8192) is corrected to the true 128000 ceiling (OVH accepted 40000; the clamp had reduced explicit large requests to 8192). Omit-safety was live-verified across OVH vLLM, LM Studio server, and OpenRouter (incl. Anthropic-via-OpenRouter). Remaining finish_reason-truthfulness lanes (Ollama streaming, Anthropic sync-stream, MLX/HF) + the constructor-clamp semantics + local-runaway cost note are tracked in backlog 0824. Full suite green (2112 passed).

### Changed
- **Prompt-cache compatibility LIVE-VERIFIED for Qwen3.5 / Qwen3.6 / Gemma 4 / Ornith 1.0 on MLX + GGUF (4 fable5 subagents, 2026-07-14)**: the documented behavior was confirmed by running the real AbstractCore MLX and GGUF cache lanes against on-machine models (new tool `scripts/verify_prompt_cache_families.py` — a 3-turn growing-prefix ReAct shape over one cache key with a fact-recall correctness gate; JSON reports + `mlx_lm.make_prompt_cache` census probes retained). **MLX**: `Qwen3-4B-Instruct-2507-4bit` (pure attention) → `outcome=hit_extend`, fed 37 of 485 tk, ×6 end-to-end at a 12.7k-tk prefix; `Qwen3.5-4B-MLX-4bit` / `Qwen3.6-27B-4bit` / `Qwen3.6-35B-A3B-4bit` / `Ornith-1.0-9B-4bit` (hybrids) confirmed as the untrimmable 3:1 Gated DeltaNet arch, **census on the 4B: exactly 24×`ArraysCache` + 8×`KVCache`, `can_trim_prompt_cache=False`** — the doc's architecture claim proven at the cache-object level (these now use the snapshot/restore lane below for warm reuse); `gemma-4-31b-mxfp4` (sliding-window) → `hit_extend` under the window. Every model kept fact-recall correct — the delta lane never returns wrong context. **GGUF (llama.cpp 0.3.23)**: `qwen35`/`qwen35moe` (Qwen3.5/3.6/Ornith hybrids) and `gemma4` all LOAD and generate correctly (arch support confirmed). Doc updated with per-model ✅ verified markers.
- **Prompt-cache compatibility documented for Qwen3.5 / Qwen3.6 / Gemma 4 / Ornith 1.0 (HF research, 2026-07-14)**: recorded in `docs/prompt-caching.md` after verifying architectures on Hugging Face. Qwen3.5, **Qwen3.6 (the SAME `qwen3_5`/`qwen3_5_moe` architecture — HF loads both with the same classes)**, and **Ornith 1.0 (9B/35B/397B, Qwen3.5 post-trains)** all use a 3:1 Gated DeltaNet (linear-attention) + full-attention hybrid stack whose recurrent layers are untrimmable `ArraysCache` → AbstractCore's in-process MLX/HF delta-trim lane rebuilds per warm call with a `#FALLBACK` (correct, no in-process delta savings); growing-prefix reuse still helps on the server lane (LM Studio/vLLM) when the deployed engine has the hybrid-checkpoint prefix cache (upstream mlx-lm #1006, checkpoint-based not trim-based). Gemma 4 is a sliding-window (local) + global-attention hybrid with trimmable `RotatingKVCache` → full delta while the transcript is under the sliding window (512/1024), then rebuild-per-call — same lane as Gemma 3. Registry coverage confirmed COMPLETE for all four families: every HF size present (Qwen3.5 0.8B–397B, Qwen3.6 27B/35B-A3B, Gemma 4 E2B/E4B/12B/26B-A4B/31B, Ornith 9B/35B/397B), and FP8/GPTQ/GGUF/mlx-community-quant variant names all resolve to full capabilities via the architecture patterns + fuzzy match (verified live against real HF repo listings; no registry gaps).

### Added
- **Process-lifetime provider endpoint profiles (`register_runtime_provider_profile`, 2026-07-15)**: hosts can now materialize externally-defined endpoint providers (e.g. profiles registered on an AbstractGateway) into the running process so `create_llm("endpoint:<id>", ...)` resolves them without local `~/.abstractcore` config. `ConfigurationManager.register_runtime_provider_profile(profile)` validates through `ProviderProfile` (unsupported `provider_family` fails loudly) and stores in a SEPARATE in-memory dict consulted by `resolve_provider_profile`/`get_provider_profile`/`list_provider_profiles` — deliberately outside `config.provider_profiles`, so an unrelated `_save_config()` can never silently persist host-derived profiles to disk (leak-tested); persisted profiles win on id collision; `clear_runtime_provider_profiles()` drops them. First consumer: AbstractCode's launch-time `--provider` resolution against the gateway provider catalog. Pins in `abstractcode/tests/test_provider_resolution.py` (injection + never-persists) ride the consumer repo.
- **MLX warm-cache reuse for Gated-DeltaNet hybrids via a snapshot/restore lane (2026-07-15)**: Qwen3.5 / Qwen3.6 / Ornith 1.0 (and pure-SSM models) hold a recurrent `ArraysCache` state that cannot be trimmed, so the token-trim delta path did not apply and warm turns re-prefilled the whole prompt. A recurrent state cannot be rewound but it CAN be copied, so `MLXProvider` now keeps one `copy.deepcopy` cache snapshot per `prompt_cache_key` at the last prefill boundary (keyed by the exact tokens it holds) and, when the next full-context prompt extends it, restores the copy and prefills only the suffix — forward-only reuse, the same discipline llama.cpp's GGUF lane and mlx_lm's own server (`LRUPromptCache`) use for these architectures. New telemetry outcome `hit_restore` (in `GenerateResponse.metadata["prompt_cache"]`, alongside `hit_full`/`hit_extend`/`cold`/`rebuilt`/`bypassed`/`append`/`off`); turn 1 pays a full prefill (`rebuilt`), a divergent prefix rebuilds and re-snapshots, one snapshot per key (the growing boundary evicts its predecessor), snapshots cleared with the key. Live-verified on `Qwen3.5-4B-MLX-4bit` (6.6k-token transcript, one key): turns 2–3 `outcome=hit_restore` feeding ~24 of 6.6k tokens, byte-exact restore-vs-cold at temperature 0 (probe), fact recall correct; the pure-attention trim path (`hit_extend`) is unchanged. Prefix-only/`CachedSession` append lane and durable artifact caches are untouched (artifact keys still bypass to protect the shared cache). Full design + adversary evidence in `untracked/mlx-gguf-cache-parity.md`. 5 pins in `tests/providers/test_mlx_hybrid_snapshot_lane.py`.

- **`fetch_url`/`skim_url` base64 URL screen — decode-and-inspect, params kept (operator ruling, 2026-07-14, final)**: after an SSRF-guard exploration the operator ruled the fetch security must be dead simple and fully functional — keep URL query parameters (they select the right page; stripping breaks fetches) and add exactly one protection: refuse a URL that carries a base64-ENCODED PAYLOAD anywhere (path/query/fragment) — the model-authored data-exfil signature — returning `error_class="blocked_encoded_url"`. The discriminator is DECODE-AND-INSPECT, not a character-class guess (this resolves the operator's suspicion that a base64 secret and a legitimate base64 identifier can't be told apart — they CAN, because base64 is reversible): candidates (base64url-alphabet runs ≥24 chars, so `-`/`_` payloads are NOT split — closing the tokenizer evasion) are base64-decoded (both alphabets, padding repaired) and flagged ONLY when they decode to MEANINGFUL data (≥85% printable ASCII, or valid UTF-8 ≥75%) over ≥12 bytes. An exfiltrated secret is real information (keys, JSON, text, PII) → decodes printable → BLOCK; a random IDENTIFIER (Google Drive file id, git SHA, UUID, nonce) decodes to high-entropy noise → ALLOW. So Google Drive/Docs URLs, REST paths, UUIDs, hex digests, hyphen/underscore slugs, and normal queries all fetch cleanly, while a base64 secret in the URL (including url-safe with `-`/`_`) is refused. The candidate token is scanned across the **netloc, path, query, and fragment** (fable5 FP/perf adversary found the authority was unscanned — a readable secret in userinfo `<b64>@host` or a subdomain label needs no obfuscation; now covered). Runs on the initial URL + meta-refresh follower for `fetch_url` and on `skim_url` (sibling fetch, same surface); URL fetched intact, headers pass through, redirects follow normally. Two fable5 adversaries validated it: **0 false positives** on ~360k random ids/hashes/slugs and **0/73** mainstream content URLs (Google Drive/Docs, S3 presigned, GitHub SHA paths, YouTube, npm, UUID/ULID/nanoid all allowed); every block in every corpus was a genuine decodable base64 payload; performance is µs-scale and linear (no ReDoS). ENCODED-COMPRESSED exfil (gzip/bz2/xz/zip/zstd/lz4-then-base64) is caught by decoded magic bytes. HONEST RESIDUALS (documented): headerless raw-deflate / encrypted-then-base64 decode to random bytes (allowed); secrets < ~16 decoded bytes; split-across-params or percent-encoded payloads — all deliberate obfuscation above the "obvious case" bar. KNOWN TENSION escalated to the operator (ruling-vs-goal, not a bug): legitimate base64 URL *parameters* (GraphQL/Relay pagination cursors, SSO `RelayState`/`next` redirects, view-state, continuation tokens) decode to readable content and ARE blocked — they are byte-identical in shape to an exfil secret, so no decode gate can separate them; blocking them is the ruling faithfully enforced. Deliberately the only screen — no SSRF/metadata/allowlist/STRICT machinery. 36 pins in `tests/tools/test_fetch_url_simple_egress_screen.py`; full tools suite 365 green.

- **KV-artifact engine-identity gate (backlog 0817, first axis, 2026-07-14)**: a durable KV cache artifact is only valid under the inference engine + version that produced it (mlx_lm / transformers / llama.cpp own the serialized cache layout; a version change can silently alter what a saved cache means). Before this, an engine upgrade left old artifacts loadable with NO error — the silently-wrong-cache class. New provider hook `prompt_cache_engine_fingerprint()` (base default `""`; MLX → `mlx_lm==<version>`, HF → `transformers==<version>`/`llama_cpp==<version>`) is recorded into the bloc-KV manifest + artifact metadata at compile, and `_validate_existing_manifest` gates on it: a recorded fingerprint that DIFFERS from the current engine REFUSES the stale-layout artifact and recompiles (`#FALLBACK` logged); an ABSENT fingerprint (pre-0817 artifact) is reused UNVERIFIED with a labeled `#FALLBACK` (no corpus-wide invalidation, never silent). Deliberately NOT part of `binding_id` — adding it would change every pre-0817 manifest's recomputed binding and reject the existing corpus (backfill safety). This is the first of the audited validity axes (the full gap matrix — tokenizer/template fingerprint, model-config hash, weights identity, cache dtype, position offset — is in `docs/backlog/planned/0817_*`); each remaining axis lands as its own refuse-loudly gate. 7 pins in `tests/test_bloc_kv_engine_fingerprint.py`; live-verified (`mlx_lm==0.31.3`); 23 existing bloc_kv pins green.

### Fixed
- **MLX rendered non-"qwen" ChatML models (Ornith) as plain text (2026-07-15)**: `_build_prompt_fragment` decided ChatML framing by `"qwen" in model_name`, so any `message_format: "im_start_end"` model whose NAME lacks "qwen" — notably Ornith 1.0 (arch `qwen3_5_agentic`) — fell through to the plain `role: content` fallback with ZERO `<|im_start|>`/`<|im_end|>` markers on the live generate path (`_build_prompt` → `_build_prompt_fragment`), not just the cache path. The decision now reads the registry's `message_format` (`im_start_end` → ChatML), with the name substring kept only as a fallback for a model missing its arch config; genuinely non-ChatML archs keep the existing plain fallback. 2 pins in `tests/providers/test_mlx_single_system_block_unit.py`.
- **Tool-placement policy was copy-pasted across 9 sites (one silently dropped tools) — now one shared helper (S4 refactor, 2026-07-15)**: the "merge the prompted tool block into ONE system turn" policy (the `supports_prompted` gate, the `## Tools (session)` dedup sentinel, the one-system-turn `\n\n` merge) was duplicated across MLX/GGUF/transformers/Ollama/Anthropic/OpenAI-compatible, had drifted four ways, and the transformers no-template fallback (`_build_input_text_transformers`) used the raw `system_prompt` and **silently dropped the tool declaration** on any template-less model. Collapsed to one free function `abstractcore.tools.merge_tools_into_system(handler, system, tools)` that duck-types on any handler exposing `supports_prompted` + `format_tools_prompt` (so provider test doubles compose without change), byte-identical to the prior per-site output (`f"{system}\n\n{tools}"`), so prompt-cache byte-parity holds. The transformers fallback now renders the merged system (tools survive). OpenAI-compatible/Anthropic keep native-first routing (prose only when `not supports_native`). 6 pins in `tests/tools/test_merge_tools_into_system.py`.
- **MLX cache clone aliased hybrid recurrent state (fable5 parity adversaries, 2026-07-15)**: `_prompt_cache_backend_clone` cloned via `layer.from_state(layer.state)`, but `ArraysCache.state` returns the live per-layer array LIST and `from_state` re-assigns it — so a "clone" of a Gated-DeltaNet hybrid cache ALIASED the parent's mutable recurrent slots, and a later generation step-write silently corrupted the other copy (a latent defect for hybrid module-cache forks via `prompt_cache_prepare_modules`, independent of the snapshot lane; reproduced by probe). The clone now uses `copy.deepcopy` first (measured 0.2–3 ms on a 4B hybrid; the source's lazy state is materialized before the copy so it is truly independent), keeping the `from_state` path only as a fallback. `KVCache` clones were accidentally safe (growth rebinds), which is why pure-attention models never hit it.
- **GGUF model given as an absolute/relative FILESYSTEM PATH failed to load (operator-reported, 2026-07-15)**: launching with `--model /Users/…/.lmstudio/models/org/Model-GGUF` (or any on-disk path to a `.gguf` file or a directory containing one) raised `ModelNotFoundError`, while the HF repo-id form (`org/Model-GGUF`) loaded the exact same file. Root cause: `_find_gguf_in_cache()` and `_is_gguf_model()` only understood repo-ids / HF-cache names — an absolute path fell through `_to_repo_id` (which returned the whole path minus slashes) and matched nothing, so a model sitting right on disk was reported "not found in local caches". Fix: both methods now resolve a direct filesystem path FIRST — a `.gguf` file is used directly; a directory is globbed for `*.gguf` (preferred-quant pick, honoring a trailing `:quant` selector) — before any cache lookup, so a path always wins and never gets mis-parsed as a repo id. Non-path strings and missing paths fall through unchanged (a bad path raises nothing; cache resolution still runs). Live-verified: the reported `~/.lmstudio/models/deepreinforce-ai/Ornith-1.0-35B-GGUF` path now resolves and loads (llama.cpp Metal init reached, model loaded), the repo-id form is unchanged, and a plain transformers hub id is not misclassified as GGUF. 7 pins in `tests/huggingface/test_hf_gguf_absolute_path_resolution.py`.
- **GGUF control-plane lane gave ZERO reuse on plain `generate()` — now persists its prefill snapshot (fable5 GGUF adversary, 2026-07-14)**: the local-control-plane generate path prefilled with `save_state=False`, so a plain `generate(messages=…, prompt_cache_key="k")` growing-prefix loop — the runtime's actual calling convention, without explicit `prompt_cache_update` — wrote NOTHING back to the keyed cache: every warm turn re-prefilled the whole prompt, and the lane's `llm.reset()` additionally forfeited llama.cpp's own in-place `n_past` prefix reuse, making the control-plane lane strictly SLOWER than the `ABSTRACTCORE_GGUF_CONTROL_PLANE=0` fallback at long prompts (measured 15.6 s vs 1.06 s warm on a 9k-tk prompt). Fix: the plain-generate path now `save_state=True` — it persists the prefilled snapshot keyed by the (with-generation-prompt) prompt tokens, and because turn N+1 of a growing-prefix loop continues exactly after turn N's generation prompt, the saved key is a TRUE token prefix of the next prompt, so `_gguf_prefill_prompt_cache` loads it and evaluates only the growing suffix. Live result on this lane (10k-tk system prompt, one key, growing turns): **~9 s cold → ~0.6 s warm** on `Qwen3-4B-Instruct-2507` and `gemma-4-E4B` GGUFs; `Qwen3.5-4B` GGUF (Gated DeltaNet hybrid) warm ~1 s with fact-recall correct — the snapshot round-trips the recurrent state, which is why load-then-eval works where llama.cpp's in-place partial-KV trim refuses. Correctness-safe (true-prefix KV reuse only), coexists with `prompt_cache_update`'s transcript-aligned states (longest true prefix wins), RAM-bounded by `LlamaRAMCache` eviction (the growing snapshot evicts its smaller predecessor). 3 pins in `tests/huggingface/test_gguf_control_plane_plain_generate_reuse.py`; 21 existing control-plane pins green. (Follow-up: Ornith GGUF ships an embedded ChatML Jinja template that format detection classifies as `keyed` rather than `local_control_plane` — still correct, still reused via llama.cpp-native, but off the snapshot lane; tracked in backlog 0821.)
- **KV artifacts now persist the fed-token-id record — bloc caches join the MLX delta lattice (backlog 0819, adversary P0-1, 2026-07-14)**: the fed-token-id record (the exact ids a cache encodes — the bookkeeping the whole delta lane rides on) was store-meta only and NEVER written into saved artifacts, so every artifact-backed cache loaded as "warm-unknown" and the full-context lane BYPASSED it: load cost + RAM, then a FULL re-prefill — negative value under the runtime's calling convention. Now: `prompt_cache_save` persists the record into artifact metadata (JSON string under the safetensors string-value constraint; covers the bloc compile lane with zero bloc_kv changes — one save boundary, one fix), and `prompt_cache_load` parses it back to a real int list with VERIFIED admission (a record longer than the loaded cache cannot be a true token-prefix — dropped loudly with `#FALLBACK`, protective bypass preserved; shorter records are legitimate per the freeze invariant and the trim arithmetic handles the generated tail). Recorded artifact-backed keys now join the full-context delta lattice (LCP → trim → suffix-feed) with one artifact-only protection: a DIVERGENT prompt bypasses instead of trimming, so a single divergent call can never degrade a shared stable bloc cache down to a stub. Record-less legacy artifacts keep the honest bypass (recompile mints records; reconstruct-at-load deliberately deferred behind 0817's render fingerprint — a reconstructed record without that check is the silently-wrong-cache class). RIDER (runtime seam condition): `GenerateResponse.metadata["prompt_cache"]` telemetry struct on the sync key lane — `mode`/`key`, `outcome` (`hit_full`|`hit_extend`|`cold`|`bypassed`|`rebuilt`|`append`|`off`), MEASURED `cached_tokens`/`fed_tokens`, `bloc_sha256`/`artifact_sha256`/`binding_id` when bound, `degraded_reason` with `#FALLBACK` — the ledger can now explain 90s-vs-2s turns instead of a log line. Validation: 15 new pins (`tests/providers/test_mlx_bloc_artifact_record_persistence.py`) + the 26 existing delta pins green; LIVE two-process proof (`scripts/bloc_artifact_delta_live_check.py`, Qwen3-4B-Instruct-2507-4bit): compile → fresh-process load → full-context ask answered correctly with cached=578/fed=32 (94.8% of prefill skipped, `hit_extend` with binding shas); a hybrid-architecture run (Qwen3.5 class) verified the honest `bypassed`/"not trimmable" telemetry.

### Added
- **Side-effect tags at tool definition sites (abstractagent ask, 2026-07-14)**: consumers now classify side-effect tools from `ToolDefinition.tags` instead of curated name lists (abstractagent's repeat-guard reads them via its `tool_tags_map`; curated lists rot, definition-site tags don't). The six mutating builtins (`write_file`, `edit_file`, `execute_command`, `shell_exec`, `shell_write_stdin`, `shell_close`) carry `tags=["mutating"]`; `fetch_url` carries `tags=["write", "remote_write"]` — NOT mutating (local host state untouched) but its model-controlled `method`/`data` can POST/PUT/DELETE remotely (the 2026-07-12 never-read-only-safe finding, now machine-readable at the definition site). MCP tools were already born tagged (`["mcp", "mcp_server:<id>"]` in `mcp.tool_source`) and the handler's dict→ToolDefinition conversion preserves tags end-to-end. Read lanes stay untagged by contract — over-tagging would make deny-safe consumers skip legitimate repeat reads. Consistency is test-pinned against `tools/inventory.py`'s classification map (one fact, two surfaces, cannot drift) and tags stay OUT of native wire payloads (strict provider schemas reject unknown keys). 6 pins in `tests/tools/test_tool_tags_side_effect.py`.

- **Data-home self-registration: core's caches now register at first write (2026-07-13)**: the machine-level data registry shipped with zero writers — core, the primitive's owner, is now its first. New best-effort lane in `abstractcore.utils.data_registry`: `ensure_data_home_registered` (never raises into a data-writing path; per-process dedup; ONE `#FALLBACK` warning per name on refusal with silent retry on the next call so transient failures heal; `ABSTRACTFRAMEWORK_DATA_REGISTRY_DISABLE=1` kill switch) and `ensure_core_data_homes` (once-per-process `register_core_data_homes`). Wired at the actual write moments: HuggingFace/MLX/LMStudio provider construction registers the HF hub cache + LM Studio's report-only model dir; `EmbeddingManager` registers the default `~/.abstractcore/embeddings` vector cache and the hub cache for local models; file logging registers `abstractcore-logs` when it creates the directory; the glyph PDF renderer registers `abstractcore-glyph-cache`; `--download-vision-model` registers `abstractcore-local-models`. THE TWO LARGEST core-owned dirs are covered (agency's measured-table find — together 20x everything else): `FileBlocStore.upsert` registers `abstractcore-blocs` (~/.abstractcore/blocs, 500-GB-class KV prompt-cache artifacts; also probed by `register_core_data_homes` for read-only processes), and `~/.abstractcore/prompt_cache_repl_sessions` (28 GB, written by the RETIRED save feature of the prompt-cache REPL demo — no current writer, confirmed by git archaeology) registers as a report-only `sessions` row: its JSON transcripts are user-elected saves, never a bulk purge. Container rule learned from a live incident (372 pytest-tmp rows self-registered during a gateway suite run): constructor-custom cache/bloc dirs do NOT self-register — they live inside their caller's data home and ride that container's row; only machine-level default dirs register. Test isolation: an autouse conftest fixture points `ABSTRACTFRAMEWORK_DATA_REGISTRY` at a per-test path so unit tests never write the developer's real registry. 7 new tests; 32 green in the registry suite.

- **Registry refresh: 14 recent Hugging Face models + 6 architectures (2026-07-13)**: extends `model_capabilities.json` (249 → 263 entries) and `architecture_formats.json` (54 → 60) with the June/July 2026 open-weights wave, every fact verified against the repo's `config.json` / `chat_template.jinja` / model card on access date. New capability entries: Ornith 1.0 9B/35B/397B (DeepReinforce agentic-coding post-trains of Qwen3.5 — the 35B is the substrate serving entity sessions via LM Studio), GLM-5.1 (~200K ctx) and GLM-5.2 (1M ctx, IndexShare, effort levels none/high/max), Tencent Hy3 (295B-A21B, ':opensource'-suffixed special tokens), MiniMax-M3 (~428B-A23B vision MoE, 1M ctx, `<mm:think>`), OpenBMB MiniCPM5-1B (131K ctx compact reasoner), Meituan LongCat-2.0 (1.6T-A48B, `<longcat_*>` token family), Qwen-AgentWorld-35B-A3B + InternScience Agents-A1 (Qwen3.5-MoE agentic post-trains), NVIDIA Nemotron-Labs-3-Puzzle-75B-A9B and Nemotron-Labs-Audex-30B-A3B (audio-in/audio-out on Cascade-2), and Baidu Unlimited-OCR. New architecture entries: `qwen3_5_agentic` (Qwen3.5 base + Qwen3-Coder `<function=...>` XML tool convention — Ornith/AgentWorld/Agents-A1 all verified on it), `glm5_dsa`, `minimax_m3`, `hunyuan3`, `longcat`, `minicpm5`; `nemotron_hybrid_moe` gains the `nemotron-labs` pattern. Alias-only mappings where the checkpoint is NOT a new model: DeepSeek-V4-Pro/Flash-DSpark (same checkpoints + bolt-on speculative-decoding module, per the model card) resolve to the V4 entries; ThinkingCap-Qwen3.6-27B → qwen3.6-27b; the high-download yuxinlu1 gemma-4-12B fine-tune GGUFs → gemma-4-12b-it. Correction: the `glm-5` entry carried GLM-4.5's parameter figures (355B/32B); the official card states 744B/40B — fixed with the card quote in notes. Models whose output cap is not separately documented take a conservative 8,192 default with an explicit raise-per-deployment note (never a guessed number).

### Fixed
- **Bloc store text/meta writes are now atomic (runtime adversary find, 2026-07-13)**: `FileBlocStore` wrote `content.txt`/`meta.json`/`meta.jsonld` with plain `write_text` — a concurrent bloc compiler reading a TORN `content.txt` would hash what it read and stamp a self-consistent manifest of the WRONG text, undetectable after the fact (the shared per-host store makes concurrent writers the normal case under the framework-level bloc direction). All bloc text writes now go through unique-tmp + `os.replace` (readers see old bytes or new bytes, never a mixture); pinned incl. a no-tmp-litter assertion in `tests/test_file_bloc_store_unit.py`.

- **MCP tool names now survive strict native endpoints (abstractagent find, 2026-07-13)**: namespaced MCP tool names (`mcp::server::tool`) went onto the wire verbatim in native tool declarations — OpenAI/Anthropic-strict endpoints enforce `^[a-zA-Z0-9_-]{1,64}$` and 400 the WHOLE request, so one MCP tool in the list killed the call. New `abstractcore.tools.wire_naming`: `wire_safe_tool_name` (pure/deterministic; already-safe names pass through byte-identical — zero change for every builtin tool; unsafe names sanitize + carry an 8-hex hash of the ORIGINAL so distinct originals can never collide on the wire) wired at both declaration boundaries (`UniversalToolHandler.prepare_tools_for_native` + the Anthropic provider's own formatter), and `resolve_wire_tool_name` (stateless recomputation, no alias map to go stale) wired at the single response-normalization choke point so the model's alias answers map back to the original tool name before execution. Prompted lane deliberately untouched (prompt text validates nothing). 12 pins in `tests/tools/test_wire_safe_tool_names.py`.

- **Ollama `num_ctx` is now forwardable (abstractagent find, 2026-07-13)**: `llm_kwargs={"num_ctx": ...}` landed in interface config and was never read — the options builder sent only temperature/num_predict/top_p/top_k/seed, so the stack could not request a context window per call and Ollama silently truncated long prompts to the model default. Both payload builders (sync + async) now forward `num_ctx` with per-call > constructor precedence; absence sends nothing (the model default is never second-guessed); invalid values raise loudly instead of being silently dropped (the silent-truncation class this fixes). 5 pins in `tests/providers/test_ollama_num_ctx_unit.py`.

- **bge/e5/gte embedders were invisible to the capability filter (gateway adversary find, 2026-07-13)**: the embedding name heuristic only matched "embed"-ish substrings, so `bge-m3`, `multilingual-e5-large`, `gte-large-en-v1.5` and family were EXCLUDED from `output_type=embeddings` model lists and OFFERED as chat models. Fix at both layers: (1) `_is_embedding_model` now also matches the bge/e5/gte family tokens as bounded name segments (regex with separator boundaries — plain substring matching would false-positive on chat names; `phi-3.5`, `gemma3n:e4b`, `claude-3-5-*` verified unaffected); (2) seven registry entries with `model_type: "embedding"` added: bge-m3, bge-large-en-v1.5, multilingual-e5-large, gte-large-en-v1.5, and the Qwen3-Embedding 0.6B/4B/8B family (the previously-noted registry gap), each with card-verified context/dims/MRL facts. Classification matrix pinned green: 9 embedder spellings (incl. quant tags and org prefixes) list as embedders, 11 chat names stay chat.

- **MLX warm-cache double-prefill (adversarial find B2, 2026-07-12)**: `MLXProvider.generate` built the FULL rendered prompt and passed it together with the warm KV cache resolved by `prompt_cache_key` — and `mlx_lm` has no common-prefix dedup, so caching ON prefilled the whole transcript ON TOP of its own KV every call: ~2x the prefill cost of caching OFF, with the transcript duplicated in-context. This is the HuggingFace provider's delta pattern ported to the token level: each cache's exact fed token ids are recorded in its store-entry meta (`fed_token_ids`); on a warm call the new prompt is tokenized, LCP'd against the record, the cache is trimmed to the shared prefix (`trim_prompt_cache`, covering the previous call's generated tokens and any divergence in one arithmetic), and ONLY the suffix ids are fed. Delta discipline applies only to FULL-CONTEXT callers (`messages=` present — they re-send the whole logical context every call); prompt-only callers (`CachedSession` KV mode, direct accumulators) send fragments over a cache that IS the context and keep append semantics untouched (LCP arithmetic there would trim away the session). Fail-safe lattice: a warm cache of UNKNOWN composition under a full-context caller is rebuilt fresh with a one-time `#FALLBACK` warning (one honest cold prefill — correct for a caller that re-sends everything, and it kills the double prefill for pre-fix caches too); trim refusal or PARTIAL trim (`trim_prompt_cache` returns the count actually trimmed; hybrid-cache architectures whose `ArraysCache` layers are untrimmable are refused by `can_trim_prompt_cache`) → fresh cache; tokenizer failure → bypass the cache. The control-plane lane stays record-true (`prompt_cache_update` extends the id record from the exact fragment the backend append fed — and only when the record still exactly describes the cache head, never across a generated-tokens gap; `prompt_cache_set(warm_prompt=...)` records from birth; an unknown head is never recorded over, and a record FREEZES once generated tokens sit between fragments), so the runtime's per-call cache prepare composes with the delta path end-to-end. Usage accounting keeps reporting the full logical prompt (`usage_prompt`). Second adversarial wave folded (1 fable5 rerun; 1 P0 + 6 P1): the record encoder mirrors mlx_lm's BOS inference (`add_special_tokens` from whether the text starts with the BOS literal — plain encode ran ONE TOKEN LONG for BOS-templated architectures like gemma-turn, silently skewing every trim); warm-but-UNCOUNTABLE caches (pure-SSM/`CacheList` architectures with no `size()`/`offset`) are distinguished from cold via `empty()` and take the fresh-rebuild lane instead of reviving the double prefill with a false record; fresh rebuilds preserve the entry's meta (minus the stale id record) and TTL, and loaded/bound ARTIFACT caches are bypassed rather than destroyed (durable-bloc binding survives); the stats surface (`get_prompt_cache_stats`, served verbatim over HTTP) ships `fed_token_count` — never the raw ids, which decode back to the full prompt text; every benefit-less lane (untrimmable architectures, uncountable state) warns `#FALLBACK` once per key; `messages=[]` counts as full-context (only `messages=None` selects the append lane); the append stash is lock-guarded against cross-key races. 24 pins in `tests/providers/test_mlx_prompt_cache_delta.py` including generate()-wiring tests (suffix-feed, record-on-error, empty-messages discriminator) and a real partial-trim test.

### Added
- **`docs/prompt-caching.md` — measured performance section (2026-07-12)**: cross-lane benchmark matrix (LM Studio server cache, MLX in-process, GGUF/llama.cpp in-process, HF transformers `past_key_values` delta) measured on Apple Silicon with a content-correctness gate: warm-prefill ratios ×8–×24 at 8–16k-token prefixes; the LM Studio server cache matches on the byte-stable prefix alone (survives client restarts, `prompt_cache_key` not required for that lane); hybrid-architecture note (Qwen3.5-class untrimmable `ArraysCache` layers → fresh-rebuild fallback, correct output at ~cache-off cost); prefill-vs-decode guidance. Numbers audited by the code seat's benchmark wave (drafted cross-seat, reviewed and landed by core).

- **Authoritative builtin-tool inventory (framework tool-inventory expansion, core's half, 2026-07-12)**: new `abstractcore.tools.inventory` — `list_builtin_tool_inventory()` / `list_builtin_tool_names()` / `builtin_tool_inventory_as_dicts()` + `BuiltinToolDescriptor` (re-exported from `abstractcore.tools`). The enumeration downstream packages DERIVE from instead of hand-copying (runtime's grant tool universe, gateway door declarations, the served capability matrix): programmatically scanned from the `@tool` definitions in `common_tools` + `shell_tools` (13 + 3; a hand-list desync is structurally impossible), deterministic byte-stable ordering, lazy imports (a minimal install can call it — verified down the media-processor chain). Each descriptor carries ONLY core-owned facts: `name`, `owner`, `module` (provenance only — `(owner, name)` is the join key), `mutating`, `remote_write_capable`, `act_only`, `description`, `parameters` (deep-copied so served rows never alias the live provider-facing schema); tier/containment vocabularies stay with their owning seats. Classification is one exhaustive map, fail-closed in EVERY direction: an unclassified new tool refuses the whole inventory, a stale entry naming no scanned tool refuses, and duplicate-name claims refuse (within and across modules). Adversarial-review wave folded (1 fable5; 3 P1s + 8 P2s, no P0): the honesty split of `mutating` (local host state) from `remote_write_capable` (fetch_url's model-controlled `method`/`data` can POST — a name-based approval layer auto-approving on `mutating=False` alone would let unattended loops write remotely), the deep-copy schema isolation, the stale-entry refusal, duplicate/ordering fixture pins, strict-JSON serialization pin (`allow_nan=False`), `INVENTORY_SCHEMA_VERSION`, and declared comms exclusions (email/WhatsApp AND telegram — out of the v1 directive scope, pinned). 15 tests in `tests/tools/test_builtin_tool_inventory.py`.

### Added
- **Machine-level data-home registry (`abstractcore.utils.data_registry`, 2026-07-13)**: the primitive half of the cache-management split ruled by the cross-package vote (v-gtytw8: split 7-1) and operator-signed. Every package/app registers the data directories it writes (`register_data_home` — idempotent register-at-first-write) into `~/.abstractframework/data_registry.json` (env `ABSTRACTFRAMEWORK_DATA_REGISTRY`); the gateway console enumerates them (`list_data_homes(include_sizes=True)`) and purges through `purge_data_home` (dry-run supported). Safety lattice (adversary-hardened): `safe_to_purge` is OWNER-declared and enforced (refusals name the owner — entity homes register unsafe by construction); nested/overlapping homes are REFUSED at registration (an ancestor purge would bypass a child's declaration — the entity-home amputation class); the registry's own container directory cannot be registered; purge refuses a path whose resolution changed since registration (symlink-swap TOCTOU), deletes contents never the home dir, never follows symlinks out, and skips protected subtrees with accounting even against hand-edited registries; corrupt registry files are refused loudly, never silently regenerated. Cross-process locking via `fcntl.flock`/`msvcrt.locking` (kernel-released on death — no stale-sweep double-acquire race). Core self-registers the HF hub cache (safe to purge, re-downloadable) and LM Studio's model dir (report-only, never touched) via `register_core_data_homes()`. 25 tests.

### Fixed
- **Production-readiness adversary fold (2026-07-13, whole-package fable5 review — 7 P1 / 10 P2, top findings fixed same-day)**:
  - *Tool registry lied about successful dict outputs*: `{"success": True, "message": ...}` was reported to the model as a FAILED call (any non-empty `message` counted as an error signal) — after the tool's side effect had already run. Explicit success markers now short-circuit; `message` alone is a payload field, only `error` signals failure unmarked. 3 new pins.
  - *Async boundary drift*: `agenerate()` silently ignored `max_tokens` (never renamed to `max_output_tokens`) and never applied unified `thinking` controls — identical arguments produced different requests than `generate()`. The async boundary now mirrors the sync rename + `_apply_thinking_request`, and carries thinking metadata onto non-streamed responses.
  - *Async streamed media drop*: `_async_stream_generate` never forwarded `media` — async streamed calls on MLX/HF silently answered without ever seeing the caller's images/documents.
  - *Bare-array structured output truncated to one item*: `_extract_json` only recognized `{...}`, so `[{...}, {...}]` lost every element but the first, then validated cleanly through the single-list-wrapper coercer. Arrays (bare + fenced) now survive extraction whole; pinned.
  - *HF transformers cached lane double-appended the user prompt* into `state.messages` (full-context branch + shared post-generation block) — a later `prompt_cache_update` re-rendered the cache with the question twice.
  - *Circuit breaker counted non-retryable failures*: caller bugs (auth/invalid request) opened the breaker; under endpoint damping one misconfigured caller could block every healthy instance on a shared endpoint. Breaker now records transient classes only; pinned.
  - *Module-chain clone fallback built mislabeled caches*: when the found prefix failed to clone, `prompt_cache_prepare_modules` silently started from EMPTY while skipping the prefix modules — a cache keyed as the full chain with missing content. Clone failure now rebuilds the whole chain from module 0 with a `#FALLBACK` warning.
  - *MLX append-stash race*: the fed-token-id stash was written/read without the lock in two of three paths — cross-thread record pollination across keys. All stash accesses now ride `_append_stash_lock` (RLock).
  - *`generate(stream=True, response_model=...)` without tools* crashed with a bare AttributeError on a generator; it now raises the same clear ValueError as the hybrid path.
  - *`prompt_cache_update` erased per-key TTL overrides* (re-`set` with `ttl_s=None` reverted to store default); the base lane now preserves the existing TTL like MLX's `_fresh_full_feed` already did.
  - *MLX Outlines lane used the context window as the output cap* (`self.max_tokens`); it now honors `max_output_tokens` so structured truncation-retry bumps reach it.
  - Filed (not yet fixed, tracked): async non-stream lane still bypasses RetryManager + `normalize_assistant_text`; streamed HTTP failures bypass RetryManager (generator fires outside `execute_with_retry`); usage lost when it rides a fully-buffered final content chunk consumed by the tool detector; MLX catch-all converts generation failures into success-shaped `Error: ...` responses; registry TypeError-retry can double-execute side-effecting tools (fix: signature pre-bind); `detect_tool_calls` dead branch; `fetch_url` docstring stale.

- **Mid-stream server error events now raise loudly (live operator drive, 2026-07-13)**: LM Studio evicted a model MID-STREAM (memory pressure from a concurrent 35B load) and sent `data: {"error": {"message": "Model unloaded."}}` as an SSE event — the streaming parser skipped it (no `choices`, no `usage`), so the stream ended looking like a normal stop and the consumer kept a TRUNCATED answer with zero signal. `_stream_generate`/`_async_stream_generate` now raise `ProviderAPIError` (the retryable class) on any `{"error": ...}` stream event, object or string shape. Pinned in `tests/providers/test_streamed_usage_accounting.py`.

- **The FINAL streamed chunk carries usage again (live operator drive, 2026-07-13)**: when the incremental thinking-tag stripper buffers a tail (reasoning models), the unified stream emits one trailing finalize chunk AFTER the provider's usage chunk — consumers reading accounting from the last chunk (the OpenAI convention) saw `usage=None` even though the server reported real numbers. The unified stream now tracks the last-seen usage and re-carries it on the trailing finalize chunk. Live-verified against LM Studio (`../scripts/core_operator_drive.py` L2); the 9-leg drive script is the regression harness for the full operator surface (generate/stream/tools/structured/session/MLX delta/embeddings/fetch_url).

- **Structured-output examples are now schema-faithful (operator incident, 2026-07-12/13)**: `_create_example_from_schema` exampled every array as `["example_item"]` and never recursed — small models copy the example nearly verbatim, so an array-of-objects field taught the model to answer with a literal placeholder string that then failed pydantic validation (live incident: `ReActVerifier_next_tool_callsItem … input_value='example_item'` surfaced in an operator session at cycle 1). The generator now builds one element from the array's `items` schema, recurses through nested `properties`, resolves local `$ref`s against `$defs` (how pydantic nests models), examples `anyOf`/`oneOf` unions by their first non-null variant (Optional fields), and prefers schema-provided `examples`/`default`/`enum`/`const` values over synthetic placeholders. Test bar: the generated example must VALIDATE against the model it teaches. This is the core half of the review-mode re-default condition recorded by the agent/code seats.

- **Transformers cache crop: "didn't raise" is not "was exact" (adversarial find, 2026-07-12)**: transformers deliberately no-ops `Cache.crop` on linear-attention/mamba layers, and some hybrid layer classes (Zamba) inherit the no-op over their ATTENTION half — the cached lane would have run warm calls on visibly wrong context with zero signal. `_transformers_crop_cache` now verifies post-crop (`get_seq_length() <= keep_tokens`; a no-op crop reads past the crop point → refuse → fresh rebuild, never wrong context), and hybrids whose attention layers crop exactly while linear layers keep O(1) recurrent state are an accepted-but-LABELED approximation (one-time `#FALLBACK` naming the class; measured ×12 on Qwen3.5-4B with the content gate passing). The cached lane's rebuild branch also warns once per key (`#FALLBACK … rebuilding fresh per warm call`) — parity with the MLX lane's labeled-degradation discipline; sliding-window models past their fill (Gemma-4 class) previously paid a silent full prefill every warm call.

- **Plain-`chatml` GGUFs now reach the local control plane (2026-07-12)**: llama-cpp-python's byte-exact template guess `"chatml"` was not admitted by the control-plane alias map even though the exact ChatML renderer (and its dedicated tokenization branch) already served it — plain-ChatML GGUFs stayed `keyed` for no technical reason.

- **`prompt_cache_prepare_modules` sorted tools, breaking byte-prefix identity with `generate` (runtime-lane cache miss, 2026-07-12)**: `PromptCacheModule.normalized()` alphabetized the tools list, so the module lane rendered tools in a different order than `generate` renders the same list — the byte-prefix equality that in-process delta feeds rely on broke at the SECOND tool, and every warm full-context call through the module-prepared lane (the runtime client's shape) re-prefilled the whole tools section and everything after it. The sort also created a false cache identity (two callers with different orders shared one fingerprint while producing different bytes). Caller order is now the identity: normalization preserves it; unstable callers get distinct keys, which distinct bytes deserve.

- **Module-chain caches now carry fed-token-id records (MLX delta through `prepare_modules`/`fork`, 2026-07-12)**: caches built by `prompt_cache_prepare_modules` had no fed-token-id record, so a session cache forked from a module chain was warm-with-unknown-composition and every full-context `generate` fell to the fresh-rebuild lane (correct, zero savings). New provider hook `_prompt_cache_append_record_meta` threads provider bookkeeping through the module build; the MLX implementation derives each module cache's record from the prior module's record plus the exact fragment the backend append fed (any uncertainty → no record, never a guess). `prompt_cache_fork` already copies meta, so forked session caches inherit a true record and the generate-side delta engages end-to-end: live-verified through the real runtime ReAct lane — warm generate feeds dropped from ~8,200 tokens to ~240 (Qwen3-4B-Instruct-2507-4bit, 8.5k-token transcript).

- **HF transformers cached lane ignored `messages` on warm calls (stale-context answers, 2026-07-12)**: `_single_generate_transformers_cached` consumed `messages`/`system_prompt` only on the FIRST call of a key; every warm call rendered its delta from `prompt` alone. The runtime/ReAct shape passes the whole transcript via `messages` with `prompt=""` every call — so warm calls fed a near-empty fragment over the stale cache and the model answered the PREVIOUS question (verified live: warm call returned the prior answer, `messages` silently dropped). Now the same caller-shape discriminator as the MLX delta lane: `messages is not None` = full context → the newly rendered transcript is LCP'd against the recorded `state.prompt_tokens`, the KV cache is CROPPED back to the shared prefix (`DynamicCache.crop`), and only the suffix is prefilled; divergence on a crop-refusing cache (hybrid architectures) rebuilds fresh — one cold prefill, never a stale-context answer. Prompt-only callers keep the append lane byte-identically. Live-verified on `SmolLM2-135M-Instruct` (~8k-token transcript): warm 1.3s vs same-bytes cold 4.1s with identical outputs at temperature 0. 4 pins in `tests/huggingface/test_transformers_cached_full_context_delta.py`.

- **GGUF + transformers cached-lane usage dicts now carry normalized keys (2026-07-12)**: the GGUF chat path and two transformers cached-lane returns emitted only legacy `prompt_tokens`/`completion_tokens` spellings; consumers reading the normalized `input_tokens`/`output_tokens` keys (the cross-provider contract) saw `None`. All usage dicts from the HuggingFace provider now carry both spellings.

- **Streamed OpenAI-compatible usage accounting no longer goes dark (cross-seat find by the code seat, 2026-07-12)**: every streamed call through the OpenAI-compatible provider reported `usage=None` — the whole accounting lane (`/usage`, cache stats, run footers) was blind under streaming. Two halves, both fixed sync + async: (1) streamed payloads now request the usage chunk via the standard `stream_options: {"include_usage": true}` (never sent on non-streamed requests); strict servers that 400-reject the field get the same drop-retry-latch treatment as `prompt_cache_key` (`#FALLBACK` warning, per-instance latch, streaming itself never breaks over an accounting extra). (2) The stream parsers surface usage from BOTH server shapes: usage riding the last content-bearing chunk (LM Studio style) and the OpenAI-style final chunk with EMPTY `choices` — which the old parser silently skipped (its `choices`-guarded branch was the only yield, so a volunteered usage chunk died there even without `stream_options`). Usage-only chunks yield a content-less `GenerateResponse` that passes the streaming post-processor untouched (its no-content path forwards chunks verbatim); servers that report nothing still yield `usage=None` — absent stays distinguishable from zero, no fabricated counts. 8 pins in `tests/providers/test_streamed_usage_accounting.py`.

- **OpenAI-compatible provider now honors registry parameter constraints (latent 400 class, 2026-07-11)**: the OpenAI-compatible payload builder sent `temperature`/`top_p` (and kwargs penalties/seed) unconditionally and hardcoded `max_tokens`, ignoring `model_capabilities.json`'s `unsupported_parameters` and `token_param_name` — the constraints the OpenAI and Portkey providers have honored since the capability-filtering wave (v2.13.0). Any restricted model served through an OpenAI-compatible endpoint (LiteLLM-style proxy in front of o-series/GPT-5-class APIs, strict vLLM deployments) received parameters its API rejects and failed with a 400 the registry existed to prevent. New `_apply_model_parameter_constraints()` runs at BOTH payload sites (sync + async, before `_mutate_payload` so subclass hooks see the filtered payload): declared-unsupported sampling params (`temperature`, `top_p`, `top_k`, penalties, `repetition_penalty`, `seed`) are dropped (debug-logged, no per-call warning noise per the standing convention — registry is the authoritative enforcement), and the output cap is RENAMED to the registry's `token_param_name` — never dropped. Absent registry fields = byte-identical payloads (backward-compatible; all local/self-hosted models unaffected). Adversarial-review wave (1 fable5 adversary, findings folded):
  1. **Fuzzy-match blast-radius guard (the adversary's real P1)**: capability lookup falls back to substring partial matching, so a LOCAL model whose name merely CONTAINS a restricted registry key (e.g. `Skywork-o1-Open-Llama-3.1-8B` catching key `o1`) would have silently lost `temperature` and had its cap renamed to a key llama.cpp-class servers ignore → UNCAPPED generation. Wire-contract fields (`unsupported_parameters`, `token_param_name`) are now STRIPPED from partial-match results unless the match is PREFIX-ALIGNED at a token boundary (`gpt-5-2025-08-07` → `gpt-5` keeps them; midfix family inference does not) — `_partial_match_is_prefix_aligned()` in `architectures/detection.py`. Soft capabilities (vision, context, tool_support) keep inheriting unchanged.
  2. **Portkey composition**: the unsolicited-default cap strip now recognizes the base-renamed `max_completion_tokens` spelling too (an unsolicited default cap could otherwise leak to backends — e.g. Azure deployments that 400 above the deployment limit); explicit caps still rename correctly.
  3. **Malformed-registry fail-safety**: `_is_parameter_supported` treats a non-list `unsupported_parameters` as absent (a comma-joined STRING would have become substring matching); `_get_token_param_name` normalizes null/empty to `max_tokens` (a verbatim `None` would have renamed the cap to a JSON `null` key). New registry lint in `tests/assets/test_model_capabilities_schema.py` pins field shapes + the invariant "`max_tokens` in unsupported_parameters requires `token_param_name: max_completion_tokens`".
  4. **Sync/async drift closed (adversary drive-by, pre-existing)**: the async payload builder never sent `prompt_cache_key` — async callers silently lost session cache identity; now sync-parity with the same rejection-latch semantics.
  Pinned in `tests/providers/test_openai_compatible_parameter_filtering.py` (16 tests: drop/rename/backward-compat, async parity, STREAMING payload pin, real-registry end-to-end, fuzzy-collision + prefix-aligned pins, portkey composition, malformed-shape fail-safety, prompt_cache_key async parity).
- **Config load never silently regenerates to defaults (embedding-incident root-cause class, 2026-07-11)**: `ConfigurationManager._load_config` returned `AbstractCoreConfig.default()` on ANY read/parse error, and the next `_save_config()` then OVERWROTE the (recoverable) file with all-defaults — silently discarding operator settings including capability routes, and reasserting the stale framework embedding default (`all-minilm-l6-v2`) with no warning. This was a reassertion vector in the Mnemosyne embedding incident (a partial/corrupt config read → defaults → old embedding model reappears). Now an unparseable existing config is BACKED UP (timestamped `.corrupt-<ts>.bak`, raw bytes preserved) with a loud `#FALLBACK` warning naming the backup, before falling back to defaults for the session — nothing is lost and the degradation is never silent. Happy path (valid or missing config) is unchanged; a valid config never spawns a backup. Pinned in `tests/config/test_config_load_never_silently_regenerates.py`.

### Added
- **Embedding served-model cross-check (rogue-label defense, incident 2026-07-11)**: the OpenAI-compatible `/v1/embeddings` response carries a `model` field naming what the server ACTUALLY served — the only server-side label truth in the stack, previously discarded. `EmbeddingManager` now records it (`served_model`, surfaced in `get_cache_stats()`) and cross-checks it against the requested `model_id`, emitting a loud one-time `#FALLBACK` warning on a genuine mismatch (case/prefix/tag-tolerant, so label formatting variance does not warn). Warn-only by design and served-route-scoped (the HuggingFace-local path has no server label): it is the SIGNAL layer BENEATH the memory store's `embedding_pin`, which stays the enforcement authority — a mismatch here surfaces a lying config/server at the wire (turning a silent rogue label into a logged disagreement) but never refuses on its own. Requested + endorsed by the memory seat under the incident's resolution order. Pinned in `tests/embeddings/test_served_model_cross_check.py`.

### Fixed
- **`fetch_url` hardening — robust fetch + information-rich extraction (maintainer directive, 2026-07-11)**: a summoned entity got "found no readable text" on four perfectly-readable pages (techxplore, budgyapp, a beehiiv newsletter, nextbigfuture). A deep-focus investigation with 4 adversarial subagents (each curating a raw-byte gold reference + attacking the failing code path) found and fixed a layered failure, verified against a mechanical fact-recall harness (committed fixtures: `tests/tools/fetch_url_fixtures/`, all 4 at 100% fact recall / 0 boilerplate junk):
  1. **Primary content contract (the P0):** `fetch_url` returned `raw_text`/`normalized_text`/`rendered` but no obvious `content`/`title` key, so every clean HTML fetch looked empty to a consumer reading `result["content"]`. The result now exposes first-class `content` (structure-preserving markdown via the existing `_html_to_markdown` renderer — headings/lists/links kept, no sentence-splitting), `title`, and `description`. `_extract_main_content()` is the new shared extractor.
  2. **403 bot-challenge resilience:** a bounded same-profile retry ladder on transient statuses (403/429/5xx) that honors `Retry-After` (capped). The default UA is now honest+identified (`AbstractCore-FetchTool/1.0 (+url)`) — live testing showed the honest UA is whitelisted where browser-impersonation profiles drew challenges (incoherent browser-UA-on-bot-TLS fingerprint), so we never escalate to spoofing.
  3. **Actionable error contract:** persistent HTTP failures return `error_class` (bot_challenge/rate_limited/auth_required/not_found/gone/server_error/client_error), a `retryable` flag, and concrete `suggestions` — instead of dumping raw headers and discarding the response body.
  4. **Extraction quality:** text-signature consent-banner removal (`_strip_consent_banners` — catches utility-class CMP overlays the class-name scan misses; GDPR-refusal by construction, no cookie ever accepted); author-box/in-content-widget pruning; and a readability densest-container fallback (`_select_densest_container`) so pages without semantic selectors (beehiiv/Substack) no longer select the whole `<body>` and drag in nav/footer.
  5. **Never-empty contract:** an HTML 200 that yields no real content and matches a JS/anti-bot challenge signature returns an actionable `bot_challenge`/`js_required`/`empty_content` error (with server-reachable title/description), not a silent empty success. Live-verified across a 10-URL roster spanning encyclopedia (EN + non-English), framework/MDN docs, JSON API, arXiv, aggregator, independent blog, and news (`tests/tools/test_fetch_url_roster_live.py`, env-gated).
- **Bare-string `prompt_cache_binding` trap defused (entity visit lane incident, 2026-07-11)**: a bare string passed as `prompt_cache_binding` with no `prompt_cache_key` coerced to `{"binding_id": s}` and then failed binding validation 100% of the time with a message that never named the fix ("Prompt cache binding validation requires a non-empty prompt cache key"). This was a vocabulary collision: hosts meaning *per-session cache identity* (which is `prompt_cache_key`, best-effort) reached the *strict durable-bloc artifact verification* param. The provider boundary now refuses EARLY with `code="prompt_cache_binding_bare_string"`, naming BOTH params and their semantics. Deliberately NOT sugar: silently downgrading the strict param to the best-effort one would break the verification guarantee without a sound. The legitimate shorthand (bare-string `binding_id` WITH a `prompt_cache_key`, verified against loaded meta) is unchanged. Pinned in `tests/test_bloc_kv.py`.

### Added
- **Retry collapse (entity-topology plan item 12 / core C3)** — four pieces, all opt-in or
  default-preserving (design: `docs/backlog/proposed/0816...md`, both reviewer asks
  discharged; built on laurent's c398 approval):
  1. `RetryConfig.single_attempt()` construction preset — the readable core half of
     collapsing the double retry stack (provider makes exactly ONE attempt; the host's
     outer RetryPolicy owns attempts/backoff; circuit breaker stays active). The Harmony
     400 carve-out survives the collapse (test-pinned: single-attempt inner + outer
     resample still absorbs the race).
  2. Cancellable backoff: `generate(..., cancel_event=threading.Event())` threads into
     `RetryManager.execute_with_retry`; backoff waits become ≤1s cancellable slices and
     raise `RetryCancelledError` (`[retry cancelled by host]`, carries `last_error`;
     classifies non-retryable) within ~1s of the signal. Without a cancel event the wait
     is one plain sleep — byte-identical legacy behavior.
  3. Retry-After honoring: `ProviderError` gains `retry_after_s`; the OpenAI-compatible
     `_raise_for_status` extracts the `Retry-After` header (seconds or HTTP-date) on
     429/5xx; the retry layer uses `min(max(server_wait, jitter), max_delay)` — the
     server's own signal beats the jitter guess, capped by our max so a hostile header
     can never park a worker.
  4. Per-endpoint damping (`core/endpoint_damping.py`, opt-in
     `create_llm(..., endpoint_damping=True)`): provider instances targeting one endpoint
     share a circuit breaker (first trip stops the other N−1 — herd demo test: 8 instances,
     3 real probes instead of 8) and a bounded retry-waiter budget; budget exhaustion
     FAILS FAST with the last typed error labeled `[retry budget exhausted for endpoint …]`
     (status_code preserved) — never a queue. Keyed `(base_url, model)`; per-process by
     design (cross-process fairness is item 11's admission lane — boundary pin).
  Piece 2b rider: `_handle_api_error` wrap sites now preserve `status_code` (and
  Retry-After) from wrapped SDK/httpx exceptions (`_status_code_from_exception`) — with
  inner retries collapsed, the outer status-code-first classifier is the only 4xx gate
  and must never be starved by stringified errors. 16 tests
  (`tests/core/test_retry_collapse_c3.py`).
- **`ToolDefinition.act_only` host-side policy attribute** (entity-topology plan item 7 /
  G1 pre-condition "diary words never rest outside the book"; spelling ruled on a2a thread
  0013): tools like an entity's `diary_read` are declared ACT-ONLY on the tool contract —
  hosts (runtime effect handlers, agent observe nodes, ledger writers) key their durable
  channels on the flag, persisting only the act-frame (id + reason + gist), never the
  returned words. First-class typed bool, NOT a `tags` entry, so the dataclass default IS
  the fail-closed policy (undeclared = normal durable tool; a typo'd tag would fail open —
  the wrong direction for a privacy attribute). Additive serialization (`to_dict()` emits
  the key only when true), dict round-trip preserved in the handler, `@tool(act_only=True)`
  decorator support, and deliberately NEVER copied into native provider payloads (strict
  servers reject unknown fields; enforcement is host-side — model-facing guidance belongs
  in the tool description). Core carries the declaration only; enforcement is the
  runtime/agent lane. 8 tests (`tests/tools/test_tool_definition_act_only.py`).
- **gpt-oss-120b/OVH regression harness** (entity-topology consensus plan, core C2): the
  native-tool-call probe is demoted to a permanent regression harness pinning both behaviors
  on the path summoned entities ride. Offline
  (`tests/providers/test_harmony_transient_artifact_regression.py`, 10 tests): the Harmony
  400 signature (`unexpected tokens remaining in message header` + sibling shapes) maps to
  retryable `ProviderAPIError` while plain 400s keep `InvalidRequestError` (no blanket 400
  retry); the RetryManager chain resamples the transient class exactly once; a full
  `generate()` absorbs one artifact 400 into a successful resample at the wire path; the
  operator-declared fallback pair (LMStudio `qwen/qwen3.6-35b-a3b`, laurent c157) stays
  registry-READY (native tools, 262K context). Live
  (`tests/providers/test_gpt_oss_120b_ovh_live_regression.py`, env-gated behind
  `ABSTRACTCORE_RUN_LIVE_API_TESTS=1`): the two-leg native tool-call round-trip
  (structured `tool_calls` out, `role:"tool"` result back in) on `endpoint:ovh-provider` /
  `gpt-oss-120b` — verified passing live 2026-07-10. The live test deliberately never
  auto-falls-back to another endpoint (a harness that switches endpoints hides the
  regression it exists to catch).

### Fixed
- **Harmony generation artifacts are classified transient, not invalid-request**
  (maintainer directive 2026-07-09: "maybe it's something our parser in
  abstractcore could self-correct"): gpt-oss models on vLLM sometimes emit
  output that violates their own Harmony template (e.g. an unclosed
  `to=...` recipient header, primed by tool-ish prompt content); the
  server's strict `openai-harmony` parser rejects the MODEL'S OWN OUTPUT
  and surfaces it as HTTP 400 `unexpected tokens remaining in message
  header` on a perfectly valid request (vllm#23567, openai/harmony#38/#80;
  upstream lenient-parser fix vllm#28303 not yet on all deployments —
  OVH included). The OpenAI-compatible provider now maps this signature to
  `ProviderAPIError` (retryable — the RetryManager resamples) instead of
  `InvalidRequestError` (never retried), turning a sampling race that
  killed ~21 unattended entity-loop ticks in one night into a transparent
  retry. Request payloads were verified clean; identical payloads pass on
  resample.

### Added
- **Persistent shell sessions** (`tools/shell_session.py`, agency-parity 0215): a PTY-backed
  `ShellSession`/`ShellSessionRegistry` whose working directory and environment persist across
  commands (unlike one-shot `execute_command`), with `write_stdin` for interactive input and
  exit-code capture. Hardened + adversarially verified: bounded O(n) sentinel scan with capped
  memory on huge output (a finished multi-MB command returns its real exit code, not a false
  timeout), deterministic post-timeout resync (Ctrl-C + discard-until-sync-sentinel, no stale
  bleed), fd-leak-free reopen, and process-group teardown (background children reaped). Handles the
  cases a pipe transport could not: stdin-consuming commands (`cat`), `set -x`, binary output, and
  command-echo suppression. Host-owned primitive; NOT yet exposed as an agent tool (see 0215).
- **Agent-callable persistent shell tools** (`tools/shell_tools.py`, agency-parity 0220):
  `shell_exec` / `shell_write_stdin` / `shell_close` expose the shell-session engine to models —
  cwd/env/venv persist across calls per `session_id`. Deliberately opt-in and honestly labeled:
  schemas state "NOT a sandbox" (execute_command-level trust) and "NOT durable" (a fresh session
  always announces "new shell session" so state loss after a host restart is never silent). The
  session registry namespace is a hidden trust-boundary argument stamped by the host; per-call
  timeout capped at 600s; `read_output` added to the engine (bounded quiet-gap drain, pairs with
  `write_stdin`); `close_namespace`/`namespaced_session_id` + atexit reaping added to the registry.
  Live-verified on OVH `gpt-oss-120b`: venv create → pip install → run inside the venv across
  separate calls (the one-shot `execute_command` arm of the same task silently installed into the
  WRONG environment while claiming success — the exact failure class this closes).
- **Centralized schema-aware tool-argument coercion** (`tools/arg_coercion.py`, backlog 039):
  arguments are coerced to their declared JSON-schema type at dispatch in both the registry and the
  runtime executor, so string flags from prompted/XML tool formats (`allow_dangerous="false"`,
  `use_regex="false"`, `preview_only="false"`) become real booleans instead of truthy strings.
  Un-coercible typed values fail loudly as a tool error; containers are never stringified.

### Added
- **`create_llm(..., prompt_cache_key=...)` construction convenience** (agency-parity 0221,
  maintainer-directed; audited + design-attacked by two adversarial reviews): sets the provider
  instance's default prompt-cache key at construction so instance-per-session callers (entity
  drivers, one-shot ingest) are cached without threading a per-call kwarg. The registry consumes
  the param (never reaches provider `__init__`); explicit per-call keys always win (including
  explicit `None` to disable); unsupported providers/models degrade with a `#FALLBACK` warning,
  never an error. Pooled clients are guarded runtime-side: `MultiLocalAbstractCoreLLMClient`
  strips the param from pooled kwargs (one instance serves all sessions; the LLM_CALL handler
  injects session-scoped keys per call instead). 5 tests.

### Fixed
- **PRODUCTION INCIDENT (2026-07-09): strict OpenAI-compatible servers reject non-leading system
  messages** — a gateway assistant's FIRST message failed on OVH (vLLM-based) with HTTP 400
  "System message must be at the beginning." The trigger was the runtime's tail-appended
  attachment-index message keeping `role: "system"` (the 0212 relocation was validated against
  native OpenAI and Anthropic — which accept/wrap it — but never against the OpenAI-compatible
  family production actually runs on). Fix at the transport boundary, mirroring the shipped
  Anthropic behavior: `OpenAICompatibleProvider` now merges a leading system run into one first
  message and converts every non-leading system message into a `<system_instruction>`-wrapped
  user message, deferred past tool-result runs so tool-call adjacency holds (sync + async paths).
  This also covers in-stream `BasicSession` compaction summaries, which would have hit the same
  400. Reproduced pre-fix and verified post-fix live on the exact production model
  (OVH `Qwen3.5-397B-A17B`); 9 unit tests pin the contract (incl. one driving real generate() to the HTTP boundary — the wire path, not just the helper)
  (`test_openai_compatible_strict_system_messages.py`). Serving processes must be restarted to
  pick up the fix. Adversarial-audit hardening (2026-07-09, critic wave): structured
  content-part lists are text-extracted instead of stringified to Python reprs; an all-empty
  leading system run no longer emits an empty system message; and the async generation path
  gained the sync path's "no user message" fallback (some templates fail system-only requests
  with "No user query found").
- **Anthropic prompt caching was an active cost increase** (agency-parity 0221, live-falsified
  2026-07-08): the provider sent a TOP-LEVEL `cache_control` request param, which marks only the
  last cacheable block — the volatile transcript tail in agent-loop requests. Live result: the
  full-prompt 1.25x cache-WRITE premium on every call with zero reads (7,042/7,043-token writes,
  0 reads on consecutive calls); below the model's minimum cacheable size, a silent no-op. Now an
  explicit `cache_control` breakpoint is placed on the last `system` text block (after tools/system
  folding, sync + async), caching the byte-stable tools+system head: live-verified write 6,302 on
  call 1 → read 6,302 on call 2, and through the full ReAct loop (write 5,022 once → read 5,022 on
  every subsequent call; net +7,784 token-equivalents saved on a 3-call run). Messages are
  deliberately unmarked (a volatile tail cannot be re-read). Caller-placed `cache_control` blocks
  are respected (the provider defers instead of risking the 4-breakpoint API 400). Doc-verified:
  the official docs describe this exact trap ("place the breakpoint at the end of the static
  prefix, not on the varying block") and the per-model minimums (4,096 tokens for Haiku 4.5 —
  below it, marking is a silent no-op with zero extra cost). 7 unit tests pin the contract.
- **Anthropic cache usage was dropped and input undercounted**: `cache_read_input_tokens` /
  `cache_creation_input_tokens` were discarded, and since Anthropic's raw `input_tokens` EXCLUDES
  cache traffic, dashboards undercounted prompt size whenever caching engaged. `_build_usage_dict`
  now reports inclusive `input_tokens` and normalized cross-provider keys: `cached_input_tokens`
  (read) + `cache_write_tokens` (write premium) — also surfaced by the OpenAI and OpenAI-compatible
  providers when the server reports cache details. Contract: absent = "mechanism cannot report",
  0 = "measured zero" (regression tooling must not conflate them).
- **`edit_file` CRLF preservation** (agency-parity 0216 follow-up): files were read with universal
  newlines, so every edit silently rewrote CRLF files to LF (whole-file corruption for CRLF
  codebases; the existing preservation branches were dead code). Reads now keep the real line
  endings, all matching runs on LF-normalized text, and the file's dominant style is restored at
  the write boundary (`newline=""` on write, so Windows never double-translates). Mixed-endings
  files normalize to the dominant style with an explicit note in the tool output. CRLF patterns
  from models match CRLF files; preview mode never writes.
- **`edit_file` unified-diff deletion of `-- `-prefixed lines** (agency-parity 0216 follow-up):
  deleting a line whose content starts with `-- ` (SQL/Lua comments) produces a diff line
  `--- <text>`, which the parser misread as a new file header and refused. The hunk-body collector
  now counts old-side lines against the header-declared length and only treats `--- ` as a header
  once the old side is consumed — the fix arbitrates exactly this prefix collision; header counts
  remain hints elsewhere, and genuine multi-file diffs are still refused.
- **`prompt_cache_key` hard-rejection fallback**: some OpenAI-compatible servers (e.g. OVH AI
  Endpoints) 400-reject the best-effort `prompt_cache_key` field instead of ignoring it, which
  failed every generation once runtime prompt caching defaulted on. All four request paths
  (sync/async × single/stream) now detect that specific rejection, drop the key, retry once, and
  stop sending it for the provider instance's lifetime (`#FALLBACK` logged). Unrelated 400s are
  never blindly retried.
- **Tool schema type inference under `from __future__ import annotations`** (backlog 039):
  `ToolDefinition.from_function` now resolves stringized (PEP 563) annotations via
  `typing.get_type_hints` (with a bare-string fallback map), so `bool`/`int`/`Optional[int]`/
  `List[str]` parameters in modules like `common_tools.py` declare the correct JSON-schema type
  instead of silently defaulting to `"string"` — which had made schema-aware coercion a no-op for
  every flag on `edit_file`/`execute_command`.
- **Typed thinking-control surfaces**: `thinking_control` in the asset registries is now a typed object (`prompt_disable_token`, `template_kwarg`, `assistant_prefill_disable`, `budget_template_kwarg`, `low_effort_template_kwarg`, `request_param`) resolved by `abstractcore.architectures.thinking_controls`. Provider hooks (OpenAI-compatible, HuggingFace, MLX) read the declared surfaces instead of hardcoded architecture sets, so new model families get thinking controls by registry update alone.
- **LM Studio native reasoning control**: `thinking=` now maps to LM Studio's documented native REST `reasoning` field (`/api/v1/chat`) for reasoning-capable models whose only declared surface is a chat-template kwarg (e.g. Gemma 4) — reasoning traces are captured into `metadata["reasoning"]` and undeclared effort levels are clamped to `on`/`off` to avoid HTTP 400s.
- **LM Studio native streaming + image input**: the native route now serves `stream=True` via typed SSE events (`reasoning.delta` → per-chunk `metadata["reasoning"]`, `message.delta` → content, `chat.end` → usage incl. `completion_tokens_details.reasoning_tokens`) and vision requests via `{"type": "image", "data_url": ...}` input parts. Only what `/api/v1/chat` genuinely rejects falls back to the OpenAI-compatible endpoint — custom tools, assistant-history messages, `response_format`, and non-image media (verified live: HTTP 400 `unrecognized_keys`/`invalid_union`; matches the official endpoint feature table) — with an ADR-0001 warning naming the blocker and the effective behavior.
- **Model registry entries**: added `nemotron-3-nano-4b` (dense hybrid edge SLM; previously fell back to generic defaults with a 16K context) and `grok-4` (always-on invisible reasoning; new `reasoning_output: false` capability field records that reasoning is billed but never returned).
- **Invisible-reasoning billing evidence**: the OpenAI-compatible provider now preserves `usage.completion_tokens_details` (incl. `reasoning_tokens`) and `prompt_tokens_details`; the server's `/v1/responses` and chat shims forward real token detail breakdowns instead of hardcoding zeros.

### Fixed
- **Silent system-message drop in native OpenAI/Anthropic providers**: `role:"system"` messages inside `messages` were deleted (sync and async paths). This silently discarded system prompts arriving via `messages` — most severely for AbstractCore Server clients, whose leading system message (personas/guardrails) never reached `openai/...` or `anthropic/...` backends while `ollama/`/`lmstudio/` backends worked. Now: OpenAI passes system messages through verbatim at their original position (the Chat Completions API accepts them anywhere); Anthropic merges a leading system run into its top-level `system` parameter and converts non-leading system messages in place into `<system_instruction>`-wrapped user turns (position preserved; never inserted between a tool_use and its tool_result; conversions counted in `metadata["system_role_user_wrapped"]`). The Anthropic history builder is also defensive against messages missing `role`/`content` keys (previously `KeyError`).
- **BasicSession compaction summary never reached the model**: the session boundary filtered ALL system messages before every provider call, including the `[CONVERSATION HISTORY]` summary that `compact()` stores as a system message. The session now excludes only its own `system_prompt` duplicate and delivers other system messages (e.g. compaction summaries) in-stream.
- **HF native-video history collapse dropped system turns**: the text-only history block for HuggingFace native-video models now includes `SYSTEM:` lines instead of silently omitting mid-stream system messages.
- **Prompt pollution on thinking-off**: template-variable control names (e.g. Gemma 4 / Qwen 3.6 `enable_thinking`) were appended to the user prompt as literal text by the generic disable fallback while reporting the control as handled. The fallback now appends only declared `prompt_disable_token` surfaces (GLM `/nothink`, Qwen `/no_think`).
- **Honest thinking metadata (ADR-0001)**: `thinking_effective` is only reported when a real control artifact was applied; unhandled requests now emit a RuntimeWarning stating that the model/server default thinking behavior remains in effect (reasoning may still be generated and billed) instead of silently claiming `off`.
- **Truncated reasoning leak**: unterminated thinking blocks (e.g. `finish_reason=length` before the closing tag) are auto-closed and captured into `metadata["reasoning"]` with a `(...)` truncation marker (#TRUNCATION logged) instead of leaking raw reasoning and tag markup into visible content — in both non-streaming and streaming paths.
- **Streaming latency with thinking off**: architectures with `thinking_tags` no longer buffer the whole stream waiting for a possible reasoning-first block when thinking is effectively disabled; visible content now streams incrementally.

## [2.13.38] - 2026-06-14

### Added
- **Workspace-path utilities**: added shared `abstractcore.utils.workspace_paths`
  helpers for canonical workspace path normalization, mount alias generation,
  and safe root-bound path resolution.
- **File-family utilities**: added shared `abstractcore.utils.file_filters`
  helpers for extension normalization and media/code/document family matching.

### Changed
- **Voice plugin floor**: raised AbstractVoice integration requirements to
  `abstractvoice>=0.10.18` so Core installs pick up the corrected local TTS
  model discovery surface used by Gateway and AbstractFlow.

## [2.13.37] - 2026-06-13

### Fixed
- **Release packaging guard**: `tests/test_packaging_extras.py` now verifies AbstractVision extra wiring by dependency prefix instead of hard-coding one historical plugin floor, so AbstractCore patch releases no longer fail CI when the released AbstractVision minimum is bumped intentionally.

## [2.13.36] - 2026-06-13

### Added
- **Vision adapter discovery**: added `llm.vision.list_provider_adapters(...)` plus `GET /v1/vision/adapters` so Core can surface installed compatible LoRA adapters for a selected provider/model/task route without duplicating AbstractVision compatibility truth.
- **Batch generated media through Core**: added first-class batch delegation for `t2i`, `i2i`, `t2v`, and `i2v` through the Python capability facade, unified `generate(..., output=...)`, sync media routes, and async vision jobs.

### Changed
- **Vision plugin floor**: raised AbstractVision integration requirements to `abstractvision>=0.3.26` so Core installs pick up the released plugin exposure for adapter discovery and batch request delegation.
- **Generated media contract**: image/video output specs now preserve `count`/`n`, explicit `seeds`, stacked `lora_adapters`, `flow_shift`, and typed `guidance_2` across the full Core facade/server boundary instead of flattening them into generic extras or repeating the same singular call.
- **Server async vision boundary**: `/v1/vision/jobs/images/*` and `/v1/vision/jobs/videos/*` now delegate batch seed planning, backend request construction, and progress-method selection through `ServerVisionFacade`, so the server keeps HTTP/job orchestration while AbstractVision integration remains the source of request semantics.

### Fixed
- **Server batch seed planning**: `/v1/images/*`, `/v1/videos/*`, and async `/v1/vision/jobs/*` routes now return all requested outputs and honor explicit seed lists instead of reusing one singular generation path.
- **Remote/local bridge parity**: the server-local capability bridge now preserves typed LoRA stacks and task-specific video controls for both direct local backends and OpenAI-compatible proxy paths.
- **Async progress totals without explicit steps**: async image, image-edit, text-to-video, and image-to-video jobs now preserve backend-reported denoise totals when the caller omits `steps`, instead of pre-committing misleading totals from request defaults or frame counts.

## [2.13.35] - 2026-06-07

### Added
- **Image upscaling routes**: added `/v1/images/upscale`, `/{provider}/v1/images/upscale`, and async `/v1/vision/jobs/images/upscale` with polling/progress support.
- **Generated image upscaling**: `generate(..., output={"task": "image_upscale"})` routes source images through the AbstractVision upscaler capability.
- **HTTP server CLI**: added `abstractcore serve` as the first-class command for starting the OpenAI-compatible AbstractCore server. Existing module and uvicorn entrypoints remain available for compatibility.
- **MLX-Gen reference-image edits**: `/v1/images/edits` and async `/v1/vision/jobs/images/edits` now accept repeated multipart `reference_images` files and forward them to AbstractVision backends for composition/style-reference image edits.
- **Image progress events**: async image generation/edit jobs now capture AbstractVision `on_progress(event)` payloads in `progress.last_event`, matching the existing video job surface.
- **Wan A14B second guidance**: video generation routes, async video jobs, and generated-video output specs now accept typed `guidance_2` for dual-transformer video models.

### Removed
- **Capability default CLI compatibility flags**: removed the top-level `abstractcore --set-capability-default` / `--clear-capability-default` form. Use `abstractcore config set-default`, `abstractcore config defaults`, and `abstractcore config clear-default` instead.

### Changed
- **Permissive PDF media path**: moved the default `PDFProcessor` and `media`/aggregate install profiles from PyMuPDF-family packages to the BSD-licensed `pypdf` baseline. PyMuPDF4LLM and `pymupdf-layout` remain available only through the explicit `pdf-pymupdf-commercial` opt-in extra.
- **Vision plugin floor**: raised AbstractVision integration requirements to `abstractvision>=0.3.22` so Core installs pick up MLX-Gen `0.18.13`, SeedVR2 image upscaling, canonical q8/q4 upscaler packages, and the current upscaler progress event surface.
- **Vision job progress semantics**: normalized server job payloads now preserve AbstractVision `step_progress` and `frame_progress`; `progress` follows the backend event's canonical progress value, which is denoise-step progress for MLX-Gen.
- **Generated media examples**: updated Core docs and OpenAPI examples to use task-specific MLX-Gen A14B text-to-video and image-to-video model ids.

### Fixed
- **PDF capability truth**: the default `pypdf` processor no longer advertises image extraction support, and page-level text extraction errors are reported as warnings instead of aborting the whole document.
- **Vision upscaler discovery**: local vision catalogs now surface MLX-Gen models that only support `image_upscale`, including canonical `AbstractFramework/seedvr2-{3b,7b}-{8bit,4bit}` packages.
- **Generated image callback forwarding**: server-local generated image/edit dispatch now forwards top-level progress callbacks and backend-specific parameters through the same AbstractVision `extra` path used for video generation.
- **Reference media routing**: unified Python image-edit generation forwards `media` items with `reference`, `style`, or `context` roles as AbstractVision `reference_images`.
- **Upscale Swagger examples**: multipart OpenAPI examples now cover direct, provider-scoped, and async SeedVR2 image upscaling routes.
- **Python 3.9 configuration import**: `ConfigurationManager` constructor annotations remain importable on Python 3.9.

## [2.13.32] - 2026-06-03

### Added
- **Provider endpoint profiles**: added first-class Core config support for reusable OpenAI-compatible/provider endpoint profiles, including model discovery with profile-specific base URLs and API keys.
- **Capability default CLI**: expanded `abstractcore config` so provider/model defaults can be set, listed, cleared, and discovered from the same Core config surface used by Gateway.
- **Audio understanding registry**: added reviewed audio-understanding model metadata for Qwen Omni/Audio candidates while keeping Qwen3.6 text/vision models marked as non-audio.

### Changed
- **Capability routing metadata**: refined multimodal route metadata for text, image, video, speech, sound effects, music, and embeddings so hosts can distinguish generation routes and fallback routes.
- **Generated media outputs**: tightened output-spec normalization and media typing so image/video/voice/music/sound artifacts are routed consistently through Core.
- **Plugin floors**: raised AbstractMusic to `>=0.1.13` and AbstractVision to `>=0.3.19`.

### Fixed
- **Provider discovery**: model listing now honors per-provider/profile base URLs instead of falling back to the global provider endpoint.
- **Embedding discovery**: embedding managers use provider endpoint profile configuration when resolving remote embedding models.
- **Media content round-trips**: audio/video/media payload normalization preserves content dictionaries used by Gateway sandbox and Runtime calls.

## [2.13.31] - 2026-05-31

### Changed
- **Vision plugin floor**: raised AbstractVision integration requirements to `abstractvision>=0.3.18` so Core installs pick up the MLX-Gen `0.18.8` runtime floor and Wan 2.2 A14B text-to-video/image-to-video catalog support.

### Fixed
- **Embedding endpoint validation**: LM Studio, vLLM, and generic OpenAI-compatible embedding clients now skip eager chat-model catalogue validation for embedding-only setup. Embedding requests still surface provider errors at call time, but incomplete `/models` catalogues no longer disable remote embeddings before the first request.
- **Remote-light voice/audio extras**: `abstractcore[voice]` and `abstractcore[audio]` now install the AbstractVoice capability plugin without `omnivoice`, `torch`, or `torchaudio`. Local OmniVoice engines remain in the explicit local aggregate profiles such as `abstractcore[all-apple]` and `abstractcore[all-gpu]`.

## [2.13.30] - 2026-05-29

### Changed
- **Vision plugin floor**: raised AbstractVision integration requirements to `abstractvision>=0.3.17` so Core installs pick up the MLX-Gen `0.18.7` runtime floor and latest Wan video/model fixes.
- **Voice plugin floor**: raised AbstractVoice integration requirements to `abstractvoice>=0.10.17` so Core optional installs pick up the one-shot TTS CLI release and current voice package metadata.

## [2.13.29] - 2026-05-26

### Added
- **Video generation through Core**: added Python `generate(..., output={"task":"text_to_video"|"image_to_video"})` callback forwarding for AbstractVision progress events, plus OpenAI-compatible `/v1/videos/generations`, `/v1/videos/edits`, and async `/v1/vision/jobs/videos/*` routes.
- **Video job progress**: async video jobs now capture normalized backend progress events in `progress.last_event` while preserving step/frame counters for polling clients.

### Changed
- **Vision plugin floor**: raised AbstractVision integration requirements to `abstractvision>=0.3.16` so Core installs pick up MLX-Gen 0.18.6, exact model id routing, and text/image-to-video support.

### Fixed
- **Generated media callback boundary**: top-level progress callbacks supplied to multimodal `generate(...)` calls are attached to generated image/video output specs instead of leaking into the text-provider kwargs path.

### Verified
- `pytest tests/test_packaging_extras.py tests/test_output_specs.py tests/test_multimodal_generate_output.py tests/server/test_server_vision_image_endpoints.py tests/capabilities/test_vision_catalog_helper.py tests/server/test_server_model_residency_control_plane.py -q`

## [2.13.28] - 2026-05-26

### Changed
- **Capability plugin floors**: updated optional capability plugin install floors to `abstractvoice>=0.10.16` and `abstractvision>=0.3.14` so Core installs consume the latest OmniVoice catalog and MLX-Gen vision surfaces.
- **Capability defaults**: added shared capability default configuration support for server and downstream hosts.
- **Embedding configuration**: expanded remote embedding provider configuration and server routing coverage.

### Fixed
- **Vision catalog propagation**: preserved AbstractVision's canonical `mlx-gen` q4/q8 model catalog through Core discovery without hardcoded provider fallbacks.

## [2.13.27] - 2026-05-23

### Changed
- **Capability plugin floors**: updated optional capability plugin install floors to `abstractvoice>=0.10.15`, `abstractvision>=0.3.13`, and `abstractmusic>=0.1.12` (plus matching turnkey profiles and docs references).

### Verified
- **Hermetic test suite**: `pytest` passes with local/provider/live tests disabled (CI-style defaults).

## [2.13.26] - 2026-05-23

### Changed
- **Server music routing contract**: `/v1/audio/music` now documents `provider` as the music backend selector (aligned with the server’s 422 rejection of legacy `backend` / `music_backend` fields).

### Fixed
- **Dev server plugin resolution**: the server now prefers a sibling `../abstractmusic/src` checkout (alongside `abstractvision` and `abstractvoice`) when `ABSTRACTCORE_DEV_PREFER_SIBLINGS=1`, avoiding stale site-packages imports during local plugin development.
- **README install matrix**: added the missing `abstractcore[music]` extra so users can install the AbstractMusic capability plugin directly from the main install section.

### Verified
- **Hermetic test suite**: `pytest` passes with local/provider/live tests disabled (CI-style defaults).

## [2.13.25] - 2026-05-22

### Changed
- **Capability plugin floors**: updated optional capability plugin install floors to `abstractvoice>=0.10.14`, `abstractvision>=0.3.9`, and `abstractmusic>=0.1.8` (plus matching turnkey profiles).
- **Docs**: refreshed plugin-floor references across README, Server, Capabilities, and `llms*.txt`.

### Verified
- **Capability residency integration**: `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload` were validated against the released plugin builds for STT/TTS, voice-clone engine preloads, and local MFLUX image residency.

## [2.13.24] - 2026-05-21

### Changed
- **Lightweight music integration**: raised the optional `abstractcore[music]` floor to `abstractmusic>=0.1.4`, which installs AbstractMusic's lightweight remote-capable base package without local model runtime extras.
- **Remote ACE Music routing**: `music_backend` / server `backend` selectors now recognize `acemusic`, `ace-music`, `remote`, and related ACE aliases as `abstractmusic:acemusic`, while still allowing explicit plugin backend ids.
- **Music server formats**: `/v1/audio/music` and `/{provider}/v1/audio/music` now accept and document `wav`, `mp3`, and `flac`, matching the remote ACE Music backend's advertised formats.

### Fixed
- **Music API diagnostics**: plugin-side upstream 5xx/timeouts, including ACE Music HTTP 504 responses, now preserve gateway-style HTTP statuses instead of being collapsed into generic 500 errors.

## [2.13.23] - 2026-05-21

### Added
- **Generic capability plugin contract**: added the planned Core/plugin contract record for optional modality plugins, including shared provider/model discovery, typed task methods, host text-generation access, and cycle-free plugin integration expectations.
- **Music capability integration**: added first-class music output routing through `llm.generate(..., output="music")`, the `llm.music.generate(...)` facade, and typed server routes for `POST /v1/audio/music` and `POST /{provider}/v1/audio/music` when `abstractmusic` is installed.
- **Memory bloc maintenance control plane**: added public local helpers and matching server operations to list, delete, and prune blocs or provider/model KV artifacts while preserving live-binding safety checks for loaded cache keys.

### Changed
- **Capability registry consistency**: normalized voice, audio, vision, and music plugin discovery around a shared Core-owned registry surface so optional plugins can expose capabilities without hard dependencies or import cycles.
- **Server media routing**: music requests now support request-level backend/model/provider routing and typed music parameters instead of relying only on environment-selected plugin defaults.
- **Documentation set**: refreshed README, API, server, capabilities, memory-bloc, backlog, and LLM index docs for the new music and bloc-maintenance surfaces.

### Verified
- **Music smoke proofs**: generated valid 3-second WAV outputs through both Python `generate(..., output="music")` and the server music route using the local AbstractMusic ACE-Step backend.

## [2.13.22] - 2026-05-20

### Added
- **Provider-wide durable memory bloc caches**: unified exact durable bloc KV artifacts across MLX, HuggingFace Transformers, and HuggingFace GGUF, including shared Python/server APIs, provider-native artifact formats, manifest validation, and request-time `prompt_cache_binding` proof.
- **Durable cache validation tooling and reports**: added the durable bloc cache benchmark script plus real-provider validation reports covering processing-phase speedups, correct cached answers, artifact sizes, and provider compatibility limits.
- **HuggingFace cache-state coverage**: expanded Transformers prompt-cache save/load coverage for standard dynamic caches, sliding-window caches, Qwen3.5 hybrid cache state, and Mamba-style tensor state; expanded GGUF persistence around llama.cpp RAM-cache state.
- **Prompt-cache planning records**: completed the unified bloc-cache, HF Transformers, and HF GGUF backlog items; accepted ADR 0007 for durable memory bloc cache binding; kept speculative superbloc/exact-prefix recipe and live snapshot persistence work proposed.

### Changed
- **Generation defaults**: providers now consume `inference_parameters` from model/architecture metadata for omitted sampling knobs such as `temperature`, `top_p`, and `top_k`; Hugging Face Transformers also applies loaded `generation_config.json` defaults when present.
- **MLX sampling controls**: MLX generation now builds an `mlx-lm` sampler from unified `temperature`, `top_p`, and `top_k` values instead of ignoring those controls at decode time.
- **Prompt-cache compatibility metadata**: architecture and model capability assets now capture cache, reasoning/thinking, quantization, and generation-parameter defaults used by provider capability discovery.
- **Voice/audio compatibility floors**: optional voice/audio install profiles now target `abstractvoice>=0.10.11` and `omnivoice>=0.1.5`.

### Fixed
- **HuggingFace greedy decoding**: Transformers pipeline generation now treats `temperature=0` as greedy decoding (`do_sample=false`) instead of forwarding an invalid sampling temperature.
- **HuggingFace model compatibility failures**: unsupported FP8-on-MPS and broken quantized Transformers load paths now fail explicitly instead of being mistaken for prompt-cache failures.
- **Prompt-cache abstraction boundaries**: live prompt-cache snapshot persistence is now documented as a proposed local-admin decision, not as a durable bloc or thin-client binding surface.

## [2.13.21] - 2026-05-20

### Added
- **ADR baseline**: added an accepted ADR set covering engineering guardrails, validation/evidence, provider and capability ownership boundaries, server trust boundaries, source-first fixes, and the planned text-generation adapter lifecycle contract.
- **Prompt-cache research backlog**: recorded narrower proposed backlog items for exact-prefix memory-cluster recipes, external cache-binding semantics, and transformers/GGUF parity boundaries so future cache work starts from current code reality instead of stale assumptions.

### Changed
- **Loaded-runtime execution model**: gateway-loaded local runtimes now route prompt-cache control-plane calls, bloc-KV ensure/load operations, and chat generation through one stable provider worker thread, while streaming responses bridge out through an unbounded queue instead of tying provider progress to client drain speed.
- **Planning and docs ownership**: the package backlog now points to `docs/backlog/overview.md` as the canonical planning entry point, the old ad hoc `docs/KnowledgeBase.md` was retired in favor of ADR-backed durable policy, and the adapter example docs now describe the current vLLM lifecycle reality without claiming portable `model=` hot-switching.

### Fixed
- **Loaded-runtime thread affinity**: loaded local runtimes no longer risk wedging when streaming cleanup happens on a different ASGI worker thread, and thread-affine local prompt-cache/bloc reuse paths now stay on the same provider thread across save/load/update/generate operations.

## [2.13.20] - 2026-05-20

### Added
- **Public local vision cache catalog helper**: added `abstractcore.capabilities.get_local_vision_cache_catalog()` and `abstractcore.capabilities.vision_catalog.get_local_vision_cache_catalog()` as dependency-light local cached-vision snapshot helpers for Runtime, Gateway, and other in-process consumers.

### Changed
- **Server local vision catalog delegation**: `/v1/vision/models` now delegates its local cache snapshot to the public helper, keeps server-only active-backend state in the route layer, and avoids duplicate local cache scans within the same request path.
- **AbstractVision release target**: optional vision-enabled install profiles now target `abstractvision>=0.3.8`, the current validated plugin release for this Core boundary update.

### Fixed
- **Runtime/Core discovery boundary**: local cached-vision discovery no longer needs to import `abstractcore.server.vision_endpoints`, which removes accidental FastAPI/server-extra coupling from non-server consumers.

## [2.13.19] - 2026-05-19

### Fixed
- **Python 3.9 server imports**: replaced Python 3.10 union annotations in the gateway and single-model endpoint request paths so the release test matrix passes on the project-supported Python 3.9 runtime.

## [2.13.18] - 2026-05-19

### Added
- **Task-aware model residency**: generalized `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload` into a single residency control plane for `text_generation`, `image_generation`, `tts`, and `stt`, while keeping omitted `task` backward-compatible with existing text-generation runtime loading.
- **Capability residency facade methods**: exposed optional Python residency hooks on `llm.vision`, `llm.voice`, and `llm.audio`: `load_resident_model(...)`, `list_loaded_models(...)`, `list_resident_models(...)`, and `unload_resident_model(...)`.
- **Developer message compatibility**: server chat requests now accept OpenAI-style `developer` messages, preserving them for OpenAI and normalizing them for providers that only support system/user/assistant/tool roles.

### Changed
- **Server image residency reuse**: image loading now uses the same server backend cache as `/v1/images/*`, calls backend preload/unload hooks when available, clears load records on cache eviction, and reports remote OpenAI-compatible image providers as `configured` rather than locally loaded.
- **Voice/audio residency routing**: `task=tts` and `task=stt` now route through the shared AbstractVoice-backed capability core used by speech and transcription endpoints, so Core owns the stable control-plane contract while AbstractVoice owns model-specific warmup semantics.
- **Model residency contract**: `loaded_new` is now treated as a load-call event signal, not a `loaded` alias. Capability-backed loads return `loaded_new=true` only when the backend explicitly reports or clearly implies that the call transitioned the model from not loaded to loaded.
- **Runtime documentation and backlog status**: documented task-aware residency in the server docs, server module README, memory-bloc docs, and moved the residency proposal to completed with an implementation report.

### Fixed
- **MLX Qwen no-thinking control**: `thinking="off"` for Qwen-family MLX models now serializes the Qwen no-thinking assistant prefill so models such as Qwen3.6 stop emitting visible `<think>` content when reasoning is disabled.

### Removed
- **Vision-specific model control endpoints**: removed the public `/v1/vision/model/load` and `/v1/vision/model/unload` endpoints from the server surface and OpenAPI docs now that `/acore/models/*` is the stable model residency API.

## [2.13.17] - 2026-05-19

### Added
- **Shared Responses/chat request surface**: `/v1/responses` now accepts the same shared text-inference controls as `/v1/chat/completions` for OpenAI-style `input` payloads, including routing (`base_url`), agent format conversion, reasoning control, prompt-cache fields, and standard generation knobs such as `stop`/`seed`/penalties.

### Changed
- **Prompt-cache control-plane consistency**: `/acore/prompt_cache/update` now accepts optional `thinking` on both the gateway and `AbstractEndpoint`, and `BaseProvider.prompt_cache_update()` applies reasoning control before appending cached prompt state so cache-prefilled requests stay aligned with later generation calls.
## [2.13.16] - 2026-05-19

### Added
- **Gateway warm-runtime control plane**: added `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload` so the multi-provider server can keep local runtimes warm and expose a stable runtime selector for follow-up prompt-cache and memory-bloc operations.
- **Direct gateway bloc/prompt-cache orchestration**: the server now supports local prompt-cache and MLX bloc-KV control-plane calls against loaded gateway runtimes instead of requiring every workflow to proxy through a separate `AbstractEndpoint`.
- **MLX bloc artifact integrity coverage**: added focused unit/integration coverage for suffix-preserving artifact writes, resolved-model-id cache loading, gateway runtime reuse, and local/proxied bloc control-plane behavior.

### Changed
- **Gateway reuse path**: `/v1/chat/completions` now reuses a matching warm runtime when one has already been loaded into the gateway, including MLX prompt-cache/bloc workflows that need model state to stay hot across requests.
- **Control-plane contract**: prompt-cache and memory-bloc server routes now accept `runtime_id` as the cleanest selector for a loaded runtime when multiple warm runtimes share the same `provider` + `model`, with `provider`/`model` and optional `base_url` remaining as stable fallback selectors.
- **Documentation set**: server, endpoint, memory-bloc, and server README docs now describe both direct gateway mode and upstream `AbstractEndpoint` proxy mode, and they document that `/acore/blocs/kv/load` returns `artifact.key` for reuse as `prompt_cache_key`.

### Fixed
- **Bloc control-plane HTTP semantics**: `AbstractEndpoint` memory-bloc routes now return real HTTP error statuses for missing blocs/manifests and execution failures instead of always returning `200` with `ok: false`.
- **Real MLX artifact persistence**: bloc KV temp artifact handling now preserves the original artifact suffix, which fixes practical save/load behavior against real MLX prompt-cache files.
- **MLX metadata generation reuse path**: bloc metadata generation now consumes the shared MLX bloc-KV loader path correctly and avoids mutating the provider default cache key as a side effect.

## [2.13.15] - 2026-05-18

### Added
- **Voice and vision provider-availability catalogs**: the capability registry now exposes lightweight `available_providers()` queries for voice and vision backends so server routes can report what is configured without constructing heavy local runtimes.
- **Qwen3.6 MTP GGUF catalog entries**: added capability metadata for `unsloth/Qwen3.6-27B-MTP-GGUF` and `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, plus explicit GGUF quant selector resolution for model ids such as `:Q4_K_M` and `:UD-Q4_K_M`.

### Changed
- **Canonical server auth naming**: the HTTP gateway now uses `ABSTRACTCORE_AUTH_TOKEN` consistently across runtime config, CLI commands, Python config helpers, docs, and Swagger guidance.
- **Swagger auth behavior**: `/docs` keeps the native Swagger `Authorize` flow, but only advertises the AbstractCore bearer scheme when server auth is actually enabled and validates the token through `/acore/auth/validate` before storing it client-side.
- **Media gateway consistency**: image/audio catalog and generation routes now share a cleaner provider/model/base_url override contract and use the current AbstractVoice/AbstractVision capability package floors (`abstractvoice>=0.10.3`, `abstractvision>=0.3.6`).
- **Apple aggregate profile alignment**: `abstractcore[all-apple]` now matches the current Apple-local dependency stack required by `abstractvoice[all-apple]`, `abstractvision[all-apple]`, `mflux`, and the newer llama.cpp bindings.
- **OpenAI-compatible env surface**: the generic OpenAI-compatible provider, registry, config manager, and related docs/tests now consistently use `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

### Fixed
- **`all-apple` install resolution**: repaired the resolver conflict between `numpy`, `Pillow`, `torch`, `mflux`, and plugin extra floors so `pip install -e ".[all-apple]"` no longer backtracks across an impossible dependency graph.
- **Swagger false authorization state**: the browser docs no longer show a misleading authorized state for arbitrary bearer values when server auth is configured.
- **Generated media provider metadata**: multimodal output routing now preserves explicit voice/vision provider selectors in generated artifacts and resource metadata instead of collapsing them back to the active LLM provider class name.

## [2.13.14] - 2026-05-13

### Fixed
- Generated image, voice, and transcription output specs now pass per-call media model selectors through capability plugins while keeping runtime LLM provider routing out of plugin kwargs.
- Media classification now honors dict `content_type` metadata so artifact-backed, extensionless audio remains valid for transcription.

### Changed
- Raised AbstractVision and AbstractVoice capability floors to `abstractvision>=0.3.5` and `abstractvoice>=0.9.4`.


## [2.13.13] - 2026-05-12

### Added
- Added a Voice capability `list_stt_models()` contract and `/v1/audio/transcriptions/models` server catalog route so Gateway and thin clients can discover speech-to-text models instead of hard-coding defaults.

### Changed
- Raised AbstractVoice capability floors to `abstractvoice>=0.9.3` for voice-enabled install profiles.

## [2.13.12] - 2026-05-08

### Changed
- **Capability plugin floors**: optional vision/voice/music install profiles now
  require `abstractvision>=0.3.3`, `abstractvoice>=0.9.2`, and
  `abstractmusic>=0.1.1`.
- **Native aggregate profiles**: `abstractcore[all-apple]` now cascades to
  `abstractvision[all-apple]`, `abstractvoice[all-apple]`, and
  `abstractmusic[all-apple]`; `abstractcore[all-gpu]` now cascades to the
  matching `all-gpu` capability packages.
- **Profile boundary**: `abstractcore[apple]` remains the MLX local LLM alias
  and `abstractcore[gpu]` remains the vLLM local LLM alias. Full media/capability
  installs use `all-apple` or `all-gpu`.

## [2.13.11] - 2026-05-08

### Added
- **Capability catalog discovery**: added `llm.vision.list_provider_models(...)`,
  `llm.voice.list_profiles(...)`, `llm.voice.list_tts_models()`, and
  `llm.voice.voice_catalog()` facade methods over optional capability plugins.
- **Server media catalog routes**: added `GET /v1/vision/provider_models`,
  `GET /v1/audio/voices`, and `GET /v1/audio/speech/models` so thin clients can
  discover image models, TTS models, and voice profiles without importing
  AbstractVision or AbstractVoice directly.

### Changed
- **Plugin compatibility floors**: optional voice/audio extras now require
  `abstractvoice>=0.9.1`; optional vision extras now require
  `abstractvision>=0.3.2` so Core can rely on the released plugin catalog
  boundary while keeping local engines behind explicit plugin extras.
- **Install profile alignment**: added `abstractcore[apple]` as the hardware
  alias for the MLX local LLM stack and `abstractcore[gpu]` as the hardware
  alias for the vLLM local LLM stack, while keeping `all-apple` and `all-gpu`
  as broader aggregate profiles.

### Fixed
- **Audio catalog route HTTP status preservation**: `/v1/audio/voices` and
  `/v1/audio/speech/models` now preserve route-level `HTTPException` statuses
  for server-held credential auth failures and invalid/disallowed `base_url`
  overrides instead of wrapping them as `502` catalog failures.

## [2.13.10] - 2026-05-07

### Fixed
- **Task-only text generation selectors**: `generate(...)` and `agenerate(...)` calls with
  `output={"task": "text_generation"}` now normalize through the public output selector contract as
  `modality="text"` and follow the normal chat/text generation path instead of being treated as
  generated media or non-chat dispatch.

### Changed
- **Backlog completion**: moved the task-only text generation output-normalization proposal to
  completed with an implementation report, acceptance-criteria results, and validation notes.

## [2.13.9] - 2026-05-07

### Added
- **Public output selector contract for runtimes**: added `abstractcore.core.output_specs` so AbstractRuntime and other durable callers can identify and normalize `generate(..., output=...)` selectors without importing private provider helpers or maintaining a drift-prone mirror of AbstractCore dispatch semantics.
- **Output routing guardrail helpers**: exposed helpers for selector detection, output-spec normalization, generated-media detection, non-chat dispatch detection, runtime metadata stripping, and backend plugin kwargs extraction.
- **Selector contract tests**: added parity tests for string, dict, list, unsupported, alias-normalization, transcription, generated-media, non-chat-dispatch, and runtime-metadata cases.

### Changed
- **Provider selector delegation**: `BaseProvider._is_acore_output_request(...)`, `_normalize_output_spec(...)`, `_normalize_output_specs(...)`, and `_output_plugin_kwargs(...)` now delegate to the public Core helper module while preserving existing behavior and compatibility quirks, including the current string-vs-dict `"audio"` selector behavior.
- **Backlog completion**: moved the public output selector proposal to completed with an implementation report and validation notes.

## [2.13.8] - 2026-05-07

### Added
- **Unified generated media output**: `generate(..., output=...)` now supports a narrow opt-in multimodal path over optional capability plugins. `output="image"` routes to AbstractVision image generation/edit, while `output="voice"` routes to AbstractVoice TTS or voice clone/register depending on whether audio media is supplied. Text-only `generate(...)` remains unchanged.
- **Multimodal result types**: added `MultimodalGenerateResponse`, `GeneratedItem`, and `GeneratedResource` so generated binary artifacts and reusable resources such as cloned voices have separate, inspectable result shapes.
- **Unified output tests**: added fake-plugin coverage for image generation, image edit, TTS, voice clone/register, transcription, multi-output text chaining, streaming rejection, and provider-kwarg backward compatibility.

### Changed
- **Plugin compatibility floors**: optional voice/audio extras now require `abstractvoice>=0.9.0`; optional vision extras now require `abstractvision>=0.3.1`.
- **Media input normalization**: media dicts now accept the public `{"type": "...", "path": "...", "role": "..."}` shape and preserve roles for output routing.
- **Async generated media parity**: `agenerate(..., output=...)` now uses the same central multimodal dispatcher as sync generation instead of bypassing optional voice/vision plugins in native async providers.
- **Generated media routing hardening**: task-only output specs now infer their modality, masked image edits infer image-to-image correctly, empty `output=[]` remains a provider kwarg, and ambiguous audio+voice clone requests are rejected.
- **AbstractVoice clone compatibility**: the library clone path can reuse AbstractVoice's `VoiceManager` clone methods when the capability shim exposes TTS/STT but not a direct `clone(...)` method.
- **Generated artifact metadata**: output items now record backend/provider identity when available, forward TTS backend kwargs, decode base64 `MediaContent` payloads for plugin calls, and store returned raw bytes through `artifact_store` when provided.
- **Server media wiring**: synchronous image generation/edit routes and local/plugin audio speech, transcription, and voice-clone routes now reuse the same `generate(..., output=...)` dispatcher while preserving their OpenAI-compatible HTTP contracts.
- **Server documentation**: documented tested curl examples for image generation, image edit, TTS, STT, and image analysis, and moved the unified multimodal generation backlog item to completed with a completion report.

## [2.13.7] - 2026-05-07

### Fixed
- **GHCR image packaging**: corrects the 2.13.6 image release path by installing the exact PyPI release wheel by direct URL from PyPI metadata, avoiding PyPI simple-index propagation lag during release builds.
- **Docker image scope**: the published server image remains a lightweight remote/server gateway with `abstractcore[server,remote,media,tokens,compression]`. AbstractVoice and AbstractVision local plugin runtimes remain optional custom-image installs because their current packages pull large native inference stacks; remote OpenAI-compatible audio and image routes still work in the default image.
- **Docker configuration docs**: server-image examples now show explicit secret values in `.env` files instead of shell interpolation that `docker run --env-file` would treat literally.

## [2.13.6] - 2026-05-07

### Added
- **Provider/model image routing for local and remote vision**: `/v1/images/generations` and `/v1/images/edits` use provider/model ids for explicit routing. Local models use `diffusers/default`, `diffusers/<huggingface-repo>`, or `sdcpp/default`; remote OpenAI-compatible image endpoints use `openai-compatible/<model>` with a configured base URL. AbstractCore no longer hardcodes a local image model as its default.
- **AbstractVision environment compatibility**: server image endpoints now understand `ABSTRACTVISION_*` configuration aliases in addition to `ABSTRACTCORE_VISION_*`, making AbstractCore Server and direct AbstractVision plugin setups easier to share.
- **Vision route regression coverage**: added tests for OpenAI-compatible image proxy success, image edit proxy success, provider/model local Diffusers routing, rejection of removed local/default aliases, and the packaging split between lightweight server installs and optional local vision runtimes.
- **Voice/audio plugin extras**: added `abstractcore[voice]` and `abstractcore[audio]` as lightweight aliases for installing the compatible `abstractvoice` plugin path without making the default or server extras heavier.
- **OpenAI-compatible voice cloning route**: `/v1/voice/clone` can forward to AbstractVoice-compatible/OpenAI-compatible voice-clone endpoints with provider/model routing and loopback-safe `base_url` overrides, while preserving local AbstractVoice fallback when configured.
- **Server image plugin coverage**: the GHCR server image now installs `abstractcore[server,remote,media,tokens,compression,voice,vision]` so AbstractVoice and AbstractVision plugin entry points are available by default for remote/OpenAI-compatible voice and vision capability paths.

### Changed
- **Cleaner image generation contract**: omitted `model` now selects the configured AbstractVision/OpenAI-compatible image default only when the server environment provides one. Explicit image models must use provider/model routing such as `diffusers/default`, `sdcpp/default`, or `openai-compatible/<model>`. The JSON generation schema documents `width`/`height` instead of OpenAI's legacy `size`; legacy `size` is still accepted and translated for compatibility.
- **Server endpoint documentation pass**: expanded the server docs with a complete endpoint map, parameter tables for image/audio/embedding/model-discovery routes, provider-specific chat guidance, local vision model/job helper docs, and prompt-cache control-plane docs; Swagger now groups core routes under explicit tags instead of the default bucket.
- **Safer local vision downloads**: local Diffusers server generation is cache-only by default, matching AbstractVision 0.2.6. Set `ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=1` or `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` only when runtime model downloads are intentional.
- **Plugin compatibility floors**: optional vision extras now require `abstractvision>=0.2.6`, and optional voice/audio extras require `abstractvoice>=0.8.5`. Server-local generation, direct `llm.vision` plugin calls, local/remote audio fallback, and voice-clone routing were rechecked against those latest plugin releases.

### Fixed
- **Swagger TTS preview reliability**: `/docs` now patches Swagger's audio preview for authenticated binary `POST` responses by converting generated audio to a browser `blob:` URL; `/v1/audio/speech` examples now prefer WAV and audio responses include inline filename headers, while preserving MP3 and other formats for API clients.
- **Strict image upstream compatibility**: OpenAI-compatible image proxy requests no longer forward local-only top-level fields such as `seed`, `steps`, `guidance_scale`, or `negative_prompt` by default; custom upstreams can still receive those fields through `extra` / `extra_json`.
- **Swagger audio response docs**: `/v1/audio/speech` now advertises binary `audio/*` responses in OpenAPI instead of documenting the successful response as JSON.
- **Swagger media examples and error docs**: OpenAPI now provides complete executable examples for image, audio, voice-clone, vision-job, prompt-cache, and model-load request bodies, and documents standard AbstractCore error responses so Swagger no longer labels common 4xx/5xx responses as undocumented.

## [2.13.5] - 2026-05-06

### Added
- **Local audio model alias**: `/v1/audio/speech` and `/v1/audio/transcriptions` now accept `model="abstractvoice/default"` for local `abstractvoice` plugin fallback, which makes OpenAI SDK-style clients usable without relying on an empty model string. The earlier `local/abstractvoice` spelling remains accepted as a backward-compatible alias.

### Documentation
- Clarified `abstractvoice` 0.8.4 compatibility: the base AbstractCore plugin path can install on Python 3.9, while Python 3.10+ remains recommended because optional/heavier engines such as OpenF5/F5-TTS, Chroma, and OmniVoice are Python 3.10+ paths.

## [2.13.4] - 2026-05-04

### Added
- **Remote embeddings in the OpenAI-compatible server**: `/v1/embeddings` now routes `openai/...`, `openrouter/...`, `portkey/...`, `openai-compatible/...`, and `lmstudio/...` models in addition to the existing local/native embedding providers. OpenAI-compatible fields such as `dimensions`, `encoding_format`, and `user` are forwarded where supported, and `base_url` can target loopback/local OpenAI-compatible embedding endpoints under the existing server allowlist policy.
- **Remote STT/TTS server routing**: `/v1/audio/transcriptions` and `/v1/audio/speech` now route to remote provider endpoints when `model` is supplied (`openai/...`, `openrouter/...`, `portkey/...`, `openai-compatible/...`) while preserving the existing `abstractvoice` capability-plugin fallback when `model` is omitted.
- **Dependency-light image proxy routes**: `/v1/images/generations` and `/v1/images/edits` can proxy to an OpenAI-compatible upstream without installing local Diffusers/stable-diffusion.cpp vision runtimes. Local image generation remains opt-in via `abstractcore[server,vision]`.
- **Swagger UI authentication support**: `/docs` now exposes an OpenAPI Bearer auth scheme so users can click `Authorize` and run authenticated requests directly from the browser. Docs/schema stay public by default so Swagger can load before auth; `ABSTRACTCORE_SERVER_PROTECT_DOCS=1` protects them for locked-down deployments.
- **GHCR server image release path**: the release workflow now publishes `ghcr.io/lpalbou/abstractcore-server:<version>` after PyPI publishing succeeds. The image is built from the PyPI package with `abstractcore[server,remote,media,tokens,compression]==<version>`.

### Changed
- **Server embedding errors are strict**: HTTP embedding requests now surface upstream/provider failures as errors instead of silently returning zero-vector fallbacks.
- **Server extra remains remote-friendly**: `abstractcore[server]` now installs the FastAPI server stack without pulling local image-generation runtimes; install `abstractcore[server,vision]` for local Diffusers/sdcpp image generation.
- **Server docs for deployment and remote modalities**: updated server documentation for remote embeddings, remote audio, provider-key handling, OpenAI-compatible local endpoints, and the PyPI-backed Docker image.

### Fixed
- **Remote embedding parameter forwarding**: server-backed embedding providers now receive requested dimensions so OpenAI-compatible providers can perform provider-native dimension reduction instead of only local truncation.

## [2.13.3] - 2026-05-04

### Added
- **Centralized server auth config**: `abstractcore --config` and direct config commands now cover the hardened HTTP server auth model. Users can persist the AbstractCore server master key, unauthenticated local/dev mode, `base_url` and URL-fetch allowlists, safe media root, local-file toggle, and default server bind host/port.

### Changed
- **Provider key config coverage**: centralized API-key storage now includes `openai-compatible` and `vllm` in addition to OpenAI, Anthropic, OpenRouter, Portkey, and Google. Persisted provider and server settings are injected into environment variables only when deployment env vars are absent.
- **Configuration wizard coverage**: the interactive wizard's HTTP server step now covers the full persisted server security surface, including URL-fetch allowlists, unsafe local-file toggles, and default bind host/port.

## [2.13.2] - 2026-05-03

### Added
- **Model registry refresh**: added capability entries and architecture detection for Gemma 4, Qwen3.6, Mistral Medium 3.5, Kimi K2.6, DeepSeek V4 Pro/Flash, NVIDIA Nemotron 3 Nano Omni, and IBM Granite 4.1 models.

### Changed
- **Package maturity metadata**: updated PyPI classifiers to `Development Status :: 5 - Production/Stable` and added Science/Research, Information Technology, and typed-package classifiers.
- **Qwen3.6 thinking controls**: Qwen3.6 now uses the same `enable_thinking` request handling path as Qwen3 and Qwen3.5 in local/OpenAI-compatible providers.

## [2.13.1] - 2026-05-03

### Added
- **Install extras for common deployment paths**: added `remote` as a lightweight hosted-SDK bundle (`openai` + `anthropic`) and explicit no-dependency extras for `openrouter`, `portkey`, and `openai-compatible` so installation commands are clearer and compose cleanly.
- **Automated release workflow**: pushing a `vX.Y.Z` tag now validates the version/changelog, runs tests, builds docs, builds and checks distributions, publishes to PyPI via Trusted Publishing, and creates a GitHub Release with notes from `CHANGELOG.md`.
- **GGUF streaming regression coverage**: added a focused unit test ensuring HuggingFace/GGUF streaming setup errors are returned as error responses with the original message.

### Changed
- **Version bumped to 2.13.1** for the install-quality and release-automation cleanup.
- **Structured native test runtime**: simplified `tests/structured/test_comprehensive_native.py` so normal runs use a fast fake-native handler regression test, local provider inference is gated behind `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1`, the three-level live matrix is opt-in with `ABSTRACTCORE_RUN_COMPREHENSIVE_NATIVE_STRUCTURED_TESTS=1`, and native structured skip output no longer prints huge local model inventories.
- **Install guidance**: README and docs now emphasize the lightweight core install, `abstractcore[remote]` for hosted SDKs, composable extras, `all-apple` for Apple Silicon local stacks, and `all-gpu` for NVIDIA/vLLM stacks. The legacy `all-non-mlx` extra remains available but is no longer promoted as a primary install path.
- **Product positioning**: README and comparison docs now present AbstractCore as an offline-capable, open-source-first provider layer that can run local, self-hosted, hosted, or hybrid deployments from the same `create_llm(...)` application code.
- **Comparison guide**: refreshed `docs/comparison.md` with a clearer AbstractCore vs LiteLLM/LangChain/LangGraph/LlamaIndex distinction, including offline/self-hosted/remote deployment posture and AbstractFramework ecosystem positioning.
- **Lint configuration**: updated Ruff settings to the current `[tool.ruff.lint]` / `[tool.ruff.lint.per-file-ignores]` layout.
- **Formatting baseline**: removed the full-repo W293 blank-line-with-whitespace noise so focused lint checks can be more meaningful.
- **System prompt alias compatibility**: provider `generate()`/`agenerate()` calls now accept `system=` as a warned alias for `system_prompt=`, prefer explicit `system_prompt=` when both are supplied, and remove the alias before provider-specific kwargs are dispatched.
- **Structured output system alias**: direct `StructuredOutputHandler.generate_structured()` calls now apply the same warned `system=` alias handling.
- **CI Python matrix**: GitHub CI now tests Python 3.9, 3.10, 3.11, 3.12, and 3.13; NumPy dependency markers allow NumPy 2.x on Python 3.13 while keeping the existing NumPy 1.x constraint on older supported Python versions.

### Fixed
- **Optional import hygiene**: provider/interface type-only media references now use `TYPE_CHECKING` so core/provider imports stay lightweight and do not pull optional media modules at runtime.
- **HuggingFace/GGUF streaming errors**: streaming setup failures now preserve the original exception text in the returned error chunk instead of closing over an exception variable that Python clears after the `except` block.
- **Glyph text renderer fallback**: PIL text-width fallback now uses the active font size when Pillow lacks `textbbox`/`textsize`.
- **Server auth/provider credential routing**: `ABSTRACTCORE_SERVER_API_KEY` now acts as the server master key for all configured providers, `X-AbstractCore-Provider-API-Key` overrides only the requested upstream provider, and `Authorization` is forwarded as a provider key only when server auth is not configured. Body/query `api_key` fields remain disabled and secret-bearing headers/URLs are redacted.
- **Server request hardening**: request-level `base_url` overrides now default to loopback or explicit allowlists, remote overrides cannot silently inherit server environment API keys, URL media fetches block non-public targets across redirects, and HTTP-request local media paths require an explicit safe root or unsafe opt-in.
- **Server URL allowlists**: URL-based allowlist entries now parse and compare scheme, exact host, effective port, and path-segment prefixes to prevent host-confusion and path-prefix bypasses.
- **CachedSession system alias safety**: prompt-cache key/KV modes now warn and strip per-call `system=` overrides in sync and async generation so cached session context cannot be silently desynced.
- **LM Studio unload for reasoning REPL**: `LMStudioProvider.unload_model()` now uses LM Studio's native REST unload endpoint and resolves model keys/variants to loaded instance IDs so `examples/reasoning/qwen_thinking_repl.py` can free LM Studio models via `:unload` (automatic unload-on-switch remains HuggingFace-only).
- **CI vs local provider test gating**: clarified and fixed the split between real implementation tests and GitHub CI. Most provider tests intentionally exercise real providers with real SDKs, API keys, local model servers, and model caches. GitHub CI does not have access to LLM provider credentials or local inference services, so credential/local-provider-dependent tests now skip in that environment instead of failing during provider construction, while still running normally in a configured local test environment.

### Documentation
- README badges now include GitHub Actions CI status and tested Python versions read from the CI matrix.
- Clarified tool calling defaults (pass-through) and removed misleading “tools executed” wording from the quick start.
- Documented `CachedSession` more consistently across core docs and `llms*.txt` (getting started, API map, sessions, structured output hybrid note).
- Updated install examples across README, getting started, prerequisites, FAQ, troubleshooting, media docs, app docs, and contributing guidance.


## [2.13.0] - 2026-05-02

### Added
- **Prompt caching sessions**: `CachedSession` selects the best prompt-cache strategy automatically (KV mode for MLX + HuggingFace transformers; otherwise stable `prompt_cache_key`).
- **File “boxes” for large contexts**: `CachedSession.attach_files()` extracts text from attached files and appends one immutable transcript “box” per file (reused via KV/prefix caches).
- **Prompt cache persistence + REPL demo**: providers expose `prompt_cache_save()` / `prompt_cache_load()` when supported (capability-gated); `examples/prompt_caching/prompt_cache_repl_demo.py` reports TTFT/TIFT + cache token counts.
- **HuggingFace transformers KV reuse**: cross-call KV caching (`past_key_values` / `DynamicCache`) keyed by `prompt_cache_key`, including the local control plane (`prepare_modules`/`fork`/`update`) and `.safetensors` save/load.
- **Memory blocs (persistent file ↔ bloc ↔ KV artifacts)**: `FileBlocStore` stores extracted text snapshots and optional per-(provider,model) KV artifacts; `generate_bloc_metadata_jsonld()` produces JSON-LD metadata using `abstractcore/assets/bloc-schema.jsonld`.
- **Reasoning/thinking controls**: `GenerateResponse.reasoning` property, thinking/tag stripping in streaming, and expanded `thinking_support` / `reasoning_levels` coverage in model capability assets.
- **Model registry expansion**: improved model/variant detection, new capability entries (incl. Gemma 4), and tooling to normalize vendor/model id variants.
- **Telegram tooling**: expanded Telegram Bot API tools + tests, and improved tool transcript handling in the OpenAI provider.

### Changed
- **Capability-driven parameter filtering**: providers enforce `unsupported_parameters` and `token_param_name` from `model_capabilities.json` (reduces model-name heuristics).
- **GGUF prompt caching**: stable-prefix control plane for supported chat formats (delta-only append/update to reduce prompt re-rendering for long sessions).
- **Prompt-cache REPL observability**: clearer TTFT/TIFT + throughput reporting and attach timing breakdown (extract vs cache work).

### Fixed
- **Reasoning model token parameter mapping**: consistent `max_completion_tokens` vs `max_tokens` handling via `token_param_name`.
- **MLX prompt cache persistence**: safetensors metadata is stringified to satisfy `mlx-lm`, and cache cloning handles newer cache layer variants.
- **GGUF prompt cache persistence**: NumPy 2.x compatibility fixes for metadata encoding.

### Documentation
- Updated docs for prompt caching and memory blocs.

## [2.12.0] - 2026-02-12

### Added
- **`--install` readiness check**: comprehensive check of all subsystems (default model, provider connectivity, embeddings model, vision fallback, STT/TTS models, ffmpeg, abstractvision, API keys). Reports ✅/⚠️/❌ for each area and offers to download/install missing models interactively. Use `--yes` (`-y`) to auto-accept all downloads for non-interactive environments (e.g. `abstractcore --install --yes`).
- **Embeddings: 7 providers supported** (was 3). `EmbeddingManager` now accepts `openai`, `openrouter`, `portkey`, and `openai-compatible` in addition to the existing `huggingface`, `ollama`, and `lmstudio`. Added `OpenAIProvider.embed()` method; gateway providers (`OpenRouterProvider`, `PortkeyProvider`) already inherit `embed()` from `OpenAICompatibleProvider`. All server/cloud providers return embeddings in OpenAI-compatible format.
- **Interactive config wizard (`--config`) — expanded to 7 steps**:
  - Step 1: now asks for **base URL** when the selected provider is a local server (ollama, lmstudio, vllm, openai-compatible). Shows the env var name, current value if set, default URL, and prints the `export` command for shell persistence.
  - Step 4 (NEW): **Audio strategy** — defaults to `auto` on Enter. Asks about `native_only` / `auto` / `speech_to_text` for audio attachment handling. Mentions `abstractvoice` dependency when needed.
  - Step 5 (NEW): **Video strategy** — defaults to `auto` on Enter. Asks about `native_only` / `auto` / `frames_caption` for video attachment handling. Mentions `ffmpeg` dependency when needed.
  - Step 6 (NEW): **Embeddings provider/model** — asks for embeddings configuration with examples across all 7 supported providers. Validates provider before saving.
  - Step 7: Console logging verbosity (renumbered from step 4).

### Changed
- Interactive config wizard now covers all major configuration areas (model, base URL, vision, API keys, audio, video, embeddings, logging). Previously only covered model, vision, API keys, and logging.
- **`--install` embeddings check**: now provider-aware — server-based providers (ollama, lmstudio, openai, openrouter, portkey, openai-compatible) check reachability or API key instead of trying to download via `sentence-transformers`. When `sentence-transformers` is missing, `--install` offers to `pip install "abstractcore[embeddings]"` and then download the model.

### Fixed
- **Audio strategy default changed from `native_only` to `auto`**: the `AudioConfig.strategy` default was `native_only`, which caused audio attachments to fail on text-only models unless the user explicitly configured it. Changed to `auto` (matching `VideoConfig.strategy` which was already `auto`). With `auto`, audio works seamlessly when `abstractvoice` is installed (STT fallback) and raises a clear error with install hints when it is not.
- **Config-persisted API keys now injected into environment**: API keys saved via `abstractcore --set-api-key` (or `--config`) were stored in `~/.abstractcore/config/abstractcore.json` but providers only read from `os.environ` (e.g. `OPENAI_API_KEY`). Added `_apply_api_keys_to_env()` to bridge config-persisted keys into the environment at config load time. Environment variables always take precedence (config keys are injected only when the env var is absent).
- **`--install` TTS/STT severity**: failed model downloads are now reported as `⚠️` (warning) instead of `❌` (critical) since TTS/STT are optional subsystems.
- **`--install` TTS/STT verification**: download results are now verified by re-checking the filesystem instead of trusting the subprocess exit code (some prefetch commands exit 0 even on failure).

## [2.11.9] - 2026-02-09

### Changed
- Documentation and internal improvements.

## [2.11.8] - 2026-02-08

### Added
- **Portkey provider**: OpenAI-compatible gateway with config-based routing (env: `PORTKEY_API_KEY`, `PORTKEY_CONFIG`; optional `PORTKEY_BASE_URL`).
- **Tests**: Portkey provider payload adaptation, reasoning model restrictions, explicit-None handling, and base URL validation.

### Changed
- **Portkey payload hygiene**: forward optional generation parameters only when explicitly set.
- **Token parameter mapping**: use `max_completion_tokens` for OpenAI reasoning families (gpt-5/o1); keep legacy `max_tokens` for other backends.
- **Reasoning model compatibility**: drop unsupported parameters (temperature/top_p/penalties) with structured logging.
- **Error diagnostics**: base URL validation and improved DNS/connectivity hints.
- **Server logging**: route Python warnings through structured logging; avoid raw stderr warnings at default ERROR verbosity.
- **Server UX**: print internal/external access URLs outside logging on startup.
- **OpenAPI schema**: normalize request examples to prevent `/openapi.json` validation failures.

### Fixed
- Config CLI: interactive vision fallback now accepts any provider/model and uses provider-agnostic guidance.
- Config CLI: interactive console logging default now uses ERROR to match package defaults.

### Documentation
- Portkey usage guidance added across core docs.
- Media docs: clarified vision fallback examples as provider-agnostic.
- Server docs: moved interactive API docs links to the top of the page.

## [2.11.6] - 2026-02-06

### Added
- Config CLI: video defaults (`--set-video-*`) and `--config` alias for interactive setup.

### Changed
- Faster CLI startup by lazily importing optional web parsing deps in `abstractcore.tools.common_tools`.
- Docs: clarified requirements and configuration for image/video/audio fallbacks (including `abstractcore --config`).


## [2.11.5] - 2026-02-06

### Changed
- STT fallback when abstractvoice is installed
- faster utils.cli with lazy loading of the providers

## [2.11.3] - 2026-02-04

### Changed
- Updated the timeout settings (abstractcore config 3600s)

## [2.11.2] - 2026-02-04

### Added
- **Skim tool benchmarks**: added `examples/tools/skim_tools_benchmark.py` to measure output footprint and latency for `skim_websearch`/`web_search` and `skim_url`/`fetch_url`.
- **Import-safety test**: added a test to ensure `import abstractcore` does not eagerly import optional deps (`requests`, `bs4`, `sentence_transformers`, `pymupdf*`, ...).

### Changed
- **Skim outputs stay compact**: `skim_websearch` now truncates long titles/snippets to keep tool outputs prompt-friendly by default.
- **Tool guidance for prompted models**: tool prompts now render short `when_to_use` hints for small tool sets and a few high-impact tools (edit/write/execute + web triage tools).
- **Tool examples**: globally-capped examples now include `skim_websearch`/`skim_url` earlier so models learn the token-efficient web triage workflow.
- **Native tool payload compatibility**: native tool schemas no longer include non-standard metadata keys (`tags`, `when_to_use`, `examples`) to avoid strict provider schema validation failures.
- **Docs accuracy**: clarified `fetch_url` behavior for PDFs/binaries and documented the recommended `skim_*` → `fetch_*` workflow in the docs entry points.

## [2.11.1] - 2026-02-04

### Added
- **Security policy**: added `SECURITY.md` with responsible disclosure guidance.
- **API overview doc**: added `docs/api.md` as a user-facing map of the public Python API.
- **FAQ**: added `docs/faq.md` and linked it from the docs entry points.
- **Events + logging docs**: added `docs/events.md` and `docs/structured-logging.md`.
- **Skim tools**: added `skim_url` (fast URL triage) and `skim_websearch` (compact/filtered search) to keep agent prompts smaller when you only need “what is this about?”.

### Changed
- **Install composition (default stays small)**: docs and packaging emphasize a lightweight core install, with heavy features enabled via explicit extras (`tools`, `media`, `embeddings`, `server`, provider SDKs).
- **Dependency compatibility**: relaxed `abstractcore[huggingface]` `transformers` upper bound to `<6` so it can co-install with `abstractcore[mlx]` (as `mlx-lm` currently pins `transformers==5.0.0rc*`).
- **Documentation polish**: refreshed wording and navigation for external users; ensured internal links/anchors resolve across docs.
- **Skim output footprint**: tuned `skim_url` defaults (smaller preview/headings) and made `skim_websearch` JSON compact so tool outputs are more token-efficient by default.
- **Web search URLs**: `web_search` now unwraps DuckDuckGo redirect URLs (more readable links; smaller tool outputs).

### Fixed
- **Docs accuracy**: aligned event fields and examples with the current codebase (events, telemetry, and usage data).
- **Optional imports**: made Telegram Bot API tools import-safe when `requests` is not installed (returns a clear `abstractcore[tools]` install hint when used).
- **HTML extraction edge cases**: improved main-content selection/pruning so `fetch_url`/`skim_url` previews don’t get wiped by over-aggressive boilerplate removal on some pages.

## [2.11.0] - 2026-01-28

### Added
- **MLX throughput benchmarking**: `examples/performance/mlx_concurrency_benchmark.py` to sweep concurrency with continuous batching (`mlx-lm`) and generate summary CSVs + PNG plots.

### Changed
- **MLX install extras**: refreshed/clarified `mlx` + `mlx-bench` optional dependencies for Apple Silicon throughput benchmarking.

### Fixed
- **Embedding model detection**: treat `model_type: "embedding"` as the canonical signal; add `nomic-embed-text-v1.5` (incl. LMStudio alias `text-embedding-nomic-embed-text-v1.5@q6_k`) to `assets/model_capabilities.json`.
- **MLX model discovery**: `MLXProvider.list_available_models()` now also scans LM Studio's local cache (`~/.lmstudio/models`) (including `lmstudio-community/*` and `mlx-community/*`) and loads from those local directories when present.
- **GPT-OSS (Harmony) on MLX**: improved prompt formatting (prefers tokenizer chat templates), extracts Harmony transcripts into clean `content` (stores reasoning in `metadata.reasoning`), and propagates correct `finish_reason` (`stop`/`length`) for truncation handling.

### Documentation
- **Concurrency guide**: added MLX concurrency benchmarking notes and tracked benchmark plots/CSVs under `docs/assets/` so docs don't depend on the ignored `test_results/` folder.

## [2.10.1] - 2026-01-11

### Fixed
- **Config CLI parity**: implemented missing `ConfigurationManager` methods used by `abstractcore` config commands (streaming defaults, embeddings config, cache dirs, logging controls, vision fallback chain).
- **OpenAI-compatible auth**: `openai-compatible` provider now reads `OPENAI_COMPATIBLE_API_KEY` when set.
- **CLI provider selection**: `abstractcore.utils.cli` now exposes `openrouter`, `openai-compatible`, and `vllm` in `--provider` choices (and updates usage examples).
- **CLI token controls**: `abstractcore.utils.cli` now supports `--max-output-tokens` and interactive `/max-tokens` + `/max-output-tokens`.

### Documentation
- Updated provider/config/CLI/server docs to reflect OpenAI-compatible consolidation, OpenRouter usage, current Claude model naming, and `base_url` usage for OpenAI-compatible endpoints.

## [2.10.0] - 2026-01-10

### Added
- **OpenRouter provider**: `create_llm("openrouter", ...)` via the OpenAI-compatible API (`https://openrouter.ai/api/v1`), with config support for `OPENROUTER_API_KEY`.

### Changed
- **OpenAI-compatible consolidation**: refactored `OpenAICompatibleProvider` into the shared implementation and made `LMStudioProvider` / `VLLMProvider` thin subclasses.
- **Config**: added `api_keys.openrouter` support and wiring for `abstractcore --set-api-key openrouter ...`.
- **Defaults**: updated Anthropic default model to `claude-haiku-4-5`.

### Fixed
- **Test stability**: live-network and local-server provider tests are consistently opt-in via env flags; tracing tests no longer require a running Ollama server.
- **Media validation**: `AnthropicMediaHandler.validate_media_for_model()` now relies on centralized vision capability detection for newer Claude naming (e.g. `claude-haiku-4-5`).

## [2.9.1] - 2026-01-07

### Fixed
- **Packaging / installability**: `pip install abstractcore` now includes `beautifulsoup4` so `import abstractcore` does not fail due to `ModuleNotFoundError: bs4`.

## [2.9.0] - 2025-01-06

### Added

- **MCP (Model Context Protocol) Integration**: First-class support for MCP servers
  - New `abstractcore.mcp` package with HTTP and stdio client implementations
  - `McpClient` for HTTP-based MCP servers with session management
  - `McpStdioClient` for local stdio-based MCP server processes
  - `McpToolSource` for automatic tool discovery and schema normalization
  - Tool namespacing (`mcp:server_name:tool_name`) to prevent collisions
  - Comprehensive test coverage for MCP integration

- **Model Support**: Added 5 new models to capabilities database
  - `claude-haiku-4-5`: Claude Haiku 4.5 with 64K max output, 200K context
  - `claude-opus-4-5`: Claude Opus 4.5 with 64K max output, 200K context
  - `glm-4.7`: GLM-4.7 358B MoE with enhanced coding and reasoning (32K output, 128K context)
  - `minimax-m2.1`: MiniMax M2.1 229B MoE optimized for coding (128K output, 200K context)
  - `nemotron-3-nano-30b-a3b`: NVIDIA Nemotron 30B hybrid MoE (23 Mamba-2 + 6 Attention layers, 256K context)

- **Architecture Support**: Added `nemotron_hybrid_moe` architecture in `architecture_formats.json` for hybrid Mamba-2/Attention models

- **Model Name Resolution**: Enhanced architecture detection to strip provider prefixes (`nvidia`, `azure`, `bedrock`, `fireworks`, `gemini`, `google`, `groq`, `together`, etc.) from model names for capability lookups (e.g., `lmstudio/qwen/qwen3-next-80b` → `qwen3-next-80b`)

- **Tools Infrastructure**:
  - Filesystem ignore policy (`abstractcore.tools.abstractignore`) with `.abstractignore` support and default patterns for `*.d/` runtime directories
  - Argument canonicalization (`arg_canonicalizer.py`) for flexible parameter naming (e.g., `file_path`/`filepath`/`path`)
  - JSON-ish parser (`abstractcore.utils.jsonish`) for robust LLM-generated JSON parsing
  - Tool schema now includes `required_args` field in `ToolDefinition.to_dict()`

- **Documentation**:
  - GLM-4.6V tool format troubleshooting guide (`docs/misc/glm-4.6v-tool-format-inconsistency.md`)
  - Enhanced `docs/tool-calling.md` with best practices
  - Backlog organization with `docs/backlog/README.md` and completed items moved to subdirectory

### Changed

- **Tool Output Format** (Breaking): Core tools now return structured JSON
  - `execute_command`: Returns `{success, return_code, stdout, stderr, rendered}` dict
  - `fetch_url`: Returns `{rendered, raw_text, normalized_text, ...}` dict
  - Maintains `rendered` field for human-readable output
  - Tool Registry supports structured failure reporting

- **Provider Enhancements**:
  - `max_tokens` parameter (if provided without `max_output_tokens`) is automatically mapped to `max_output_tokens` for backward compatibility with callers using legacy terminology. Within AbstractCore, `max_output_tokens` remains the first-class citizen alongside `max_input_tokens` and `max_tokens` (context window)
  - Centralized timeout configuration from `abstractcore/config`
  - Server endpoint `/v1/chat/completions` accepts `timeout_s` request field
  - Refactored tool prompt handling for better model-specific format support
  - Enhanced performance tracking with detailed timing metrics

- **File Operations**:
  - `read_file` max lines increased from 600 to 1000
  - `list_files` now includes directories and uses relative paths
  - `edit_file` enhanced with idempotent insertion behavior, better error messages, diff observability

### Fixed

- **Provider Fixes**:
  - **Anthropic**: Unknown `claude*` models default to native tool calling; `claude-haiku-4-5` and `claude-opus-4-5` properly recognized; `role="tool"` messages converted to `tool_result` content blocks
  - **OpenAI-Compatible**: Fixed tool call normalization for wrapped tool names (e.g., `"{function-name: write_file}"`)
  - **Ollama**: Added `metadata._provider_request` for provider-wire observability
  - **VLLM**: Enhanced tool call handling
  - **LMStudio**: Improved timeout handling
  - **All**: Normalized timeout errors, enhanced metadata handling, better architecture detection

- **Tool Fixes**:
  - **Web Search**: Prefer `ddgs` with fallback to `duckduckgo_search`; bounded retries with query cleaning; region fallback; relevance scoring
  - **File Operations**: `write_file` now requires `content` parameter; `edit_file` improved diagnostics; enhanced `search_files` and `read_file` context handling
  - **Code Analysis**: Enhanced `analyze_code` documentation

- **Tool Calling Infrastructure**:
  - Parser handles doubled tags, broken closing tags, unescaped control characters
  - Bracket prefix support for alternative formats
  - Better Nemotron XMLish format handling
  - Wrapped tool name mapping in `BaseProvider`
  - Enhanced tag rewriting and normalization

- **Model Capabilities**:
  - Caching for default capabilities warnings (reduces log noise)
  - Updated multiple models to "native" tool support (including `qwen3-next-80b-a3b`)
  - Proper max output token clamping with better error messages

- **Testing**: Added 30+ new test files for MCP, tool calling, providers, filesystem policy, streaming, and packaging

### Migration Notes

- **Tool Outputs**: Update code parsing `execute_command` or `fetch_url` outputs to handle dicts with `rendered` field
- **File Operations**: Explicitly provide `content` parameter to `write_file` (use `content=""` for empty files)
- **Claude Models**: Review tool support settings for Claude 4.5 models (now default to native)

### Statistics

- **43 commits** improving tools, providers, MCP integration, and infrastructure
- **120 files changed**: 8,738 insertions, 12,472 deletions
- **5 new models** added to capabilities database (135 total models)
- **30+ new test files** for comprehensive coverage
- **21,385 total lines changed** across the codebase

## [2.8.1 - 2025-12-21

### Added
Add workflow event types: Introduce new event types for workflow progress tracking

- Added EVENT_TYPE constants for workflow steps: WORKFLOW_STEP_STARTED, WORKFLOW_STEP_COMPLETED, WORKFLOW_STEP_WAITING, and WORKFLOW_STEP_FAILED.
- Enhances event tracking capabilities for durable execution processes.



## [2.8.0] - 2025-12-18

### Added
- **Model Support**: Added 15+ new models including GLM-4.6V, Qwen3-VL series, Devstral, GPT-OSS, MiniMax-M2, and Granite-4.0-H
  - Vision models with enhanced OCR (32 languages) and visual agent capabilities
  - MoE models with detailed expert configurations and quantization specs
  - Coding models optimized for agentic workflows
- **Architecture Support**: Added 8 new architectures (glm4v_moe, mistral3, ministral3, granitemoehybrid, gpt_oss, qwen3_vl, qwen3_vl_moe, minimax_m2, harmony)
- **Compression Modes**: Added `CompressionMode` enum for chat history summarization (LIGHT/STANDARD/HEAVY)
- **Trace Metadata**: Added HTTP header extraction for distributed tracing support
- **Token Budget Control**: `BasicSummarizer` now supports AUTO mode for token management
  - `max_tokens=-1` (AUTO): Uses model's full context window capability
  - `max_tokens=N`: Hard limit for deployment constraints (GPU/RAM)
  - Same logic applies to `max_output_tokens`
  - CLI supports `--max-tokens auto` or specific values

### Enhanced
- **Tool Call Parsing**: Improved robustness with sanitization for malformed LLM output
  - Handles doubled tags, broken closing tags, and unescaped control characters
  - String-aware JSON escaping preserves structural whitespace
- **Summarization**: Smart token budget management prevents OOM while optimizing performance
  - AUTO mode uses model's full capability
  - Hard limits respect deployment constraints (GPU memory)
  - Reduces API calls on large-context models (up to 12x improvement)
  - Fallback parsing when structured output fails
- **File Editing**: Added flexible whitespace matching and unified diff support to `edit_file`
  - Matches patterns ignoring indentation differences
  - Preserves file's original indentation style
- **Error Handling**: Added fallback strategies throughout for improved reliability

### Fixed
- **Async Trace Capture**: Improved reliability of trace capture in `agenerate()` for async LLM calls

### Technical Details
- All changes maintain backward compatibility
- Default changed to `max_tokens=-1` (AUTO) for optimal performance
- Token limits prevent OOM in memory-constrained environments
- Added deprecation warnings for `execute_tools` parameter

## [2.6.7] - 2025-12-13

### Fixed
- Made PIL/Pillow a required core dependency
  - Providers need media handling, so PIL cannot be optional
  - Fixes import errors when using abstractcore without explicit media installation
  - Modified files: `pyproject.toml`, `abstractcore/media/utils/image_scaler.py`, `abstractcore/utils/vlm_token_calculator.py`

## [2.6.6] - 2025-12-13

### Fixed
- Fixed `NameError: name 'Image' is not defined` when importing tools module without PIL/Pillow installed
  - `image_scaler.py` used PIL types in annotations but imported conditionally, causing NameError instead of ImportError
  - Changed to direct imports with clear error messages
  - Core functionality (`tools`, `create_llm`) now works without PIL installed
  - Modified files: `abstractcore/media/utils/image_scaler.py`, `abstractcore/utils/vlm_token_calculator.py`

- Fixed `compression` installation group to depend on `media` (includes Pillow)

- Added missing installation groups: `all-non-mlx`, `all-providers-non-mlx`, `local-providers-non-mlx`

## [2.6.5] - 2025-12-10

### Added
- **Dynamic Base URL Support for Server Endpoint**: POST parameter for runtime base_url configuration
  - **New Parameter**: `base_url` field in `/v1/chat/completions` request body
  - **Use Case**: Connect to custom OpenAI-compatible endpoints without environment variables
  - **Example**: `{"model": "openai-compatible/model-name", "base_url": "http://localhost:1234/v1", ...}`
  - **Integration**: Works with openai-compatible provider and any provider supporting base_url
  - **Logging**: Custom base URLs logged with 🔗 emoji for easy debugging
  - **Priority**: POST parameter > environment variable > provider default
  - **Zero Breaking Changes**: Optional parameter, existing code unchanged

### Fixed
- **OpenAI-Compatible Provider Model Listing**: Fixed `/v1/models?provider=openai-compatible` endpoint
  - **Root Cause**: Provider validation rejected "default" placeholder model used by registry for model discovery
  - **Solution**: Skip model validation when model == "default" (registry placeholder)
  - **Impact**: `/v1/models` endpoint now correctly lists all 27 models from LMStudio/llama.cpp servers
  - **Verified**: Works with environment variable (`OPENAI_COMPATIBLE_BASE_URL`) configuration
  - **Model Prefix**: All models returned with correct `openai-compatible/` prefix

### Enhanced
- **Provider Registry**: Added openai-compatible to instance-based model listing
  - **Previous**: Attempted static method call, failed with openai-compatible
  - **Fixed**: Added "openai-compatible" to instance-based providers list alongside ollama, lmstudio, anthropic
  - **Benefit**: Proper model discovery with base_url injection from environment variables

### Technical Details
- **Files Modified**:
  - `abstractcore/server/app.py` (added base_url field to ChatCompletionRequest, ~18 lines)
  - `abstractcore/providers/openai_compatible_provider.py` (skip validation for "default" model, ~3 lines)
  - `abstractcore/providers/registry.py` (added openai-compatible to instance providers, 1 line)
  - `abstractcore/utils/version.py` (version bump to 2.6.5)
- **Architecture**: Clean parameter injection pattern, minimal code changes
- **Testing**: Validated with LMStudio server on localhost:1234 (qwen/qwen3-next-80b model)

### Usage Examples
```bash
# POST with dynamic base_url parameter (NEW in v2.6.5)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-compatible/qwen/qwen3-next-80b",
    "messages": [{"role": "user", "content": "Hello"}],
    "base_url": "http://localhost:1234/v1"
  }'

# List models with environment variable (FIXED in v2.6.5)
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:1234/v1"
curl http://localhost:8080/v1/models?provider=openai-compatible
# Returns all 27 models with openai-compatible/ prefix
```

## [2.6.4] - 2025-12-10

### Added
- **vLLM Provider**: Dedicated provider for high-throughput GPU inference on NVIDIA CUDA hardware
  - **Native vLLM Features**: Exposes guided decoding, Multi-LoRA, and beam search capabilities
  - **Guided Decoding**: `guided_regex`, `guided_json`, `guided_grammar` parameters for 100% syntax-safe code generation
  - **Multi-LoRA Support**: `load_adapter()`, `unload_adapter()`, `list_adapters()` for dynamic adapter management
  - **Beam Search**: `best_of`, `use_beam_search` parameters for higher accuracy on complex tasks
  - **Full Async Support**: Native async implementation with lazy-loaded httpx.AsyncClient
  - **OpenAI-Compatible**: Uses `/v1/chat/completions` endpoint while exposing vLLM extensions via `extra_body`
  - **Shared Cache**: Automatically shares HuggingFace cache with HF/MLX providers via `HF_HOME`
  - **Environment Variables**: `VLLM_BASE_URL` (default: `http://localhost:8000/v1`), `VLLM_API_KEY` (optional)
  - **Default Model**: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (or use Qwen2.5-Coder-7B-Instruct for testing)
  - **Registry Integration**: Listed in `get_all_providers_status()` alongside other 6 providers
  - **Implementation**: 823 lines of provider code, 371 lines of tests, comprehensive GPU testing guide
  - **Use Cases**: Production GPU deployments, multi-GPU tensor parallelism, specialized AI agents with LoRA adapters

- **OpenAI-Compatible Generic Provider**: Universal provider for any OpenAI-compatible API endpoint
  - **Maximum Compatibility**: Works with llama.cpp, text-generation-webui, LocalAI, FastChat, Aphrodite, SGLang, proxies
  - **Optional Authentication**: API key support (optional, many local servers don't require it)
  - **Feature Parity**: Chat completions, streaming, async, embeddings, structured output, prompted tools
  - **Environment Variables**: `OPENAI_COMPATIBLE_BASE_URL` (default: `http://localhost:8080/v1`), `OPENAI_COMPATIBLE_API_KEY` (optional)
  - **Default Model**: `"default"` (server-dependent)
  - **8 Providers Total**: Completes provider ecosystem alongside OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace, vLLM
  - **Implementation**: 764 lines of provider code, 328 lines of tests
  - **Architecture**: Inherits from BaseProvider, uses httpx for HTTP communication
  - **Use Cases**: llama.cpp local servers, text-generation-webui deployments, OpenAI-compatible proxies, custom endpoints
  - **Future Enhancement**: Planned refactoring to create base class for vLLM/LMStudio to reduce code duplication (see `docs/backlog/`)

### Documentation
- **Hardware Requirements**: Updated README.md and docs/prerequisites.md with hardware compatibility warnings
  - Added "Hardware" column to provider table (MLX: Apple Silicon only, vLLM: NVIDIA CUDA only)
  - Clear installation guidance per hardware platform
- **Multi-GPU Setup**: Complete guide for tensor parallelism on 4x NVIDIA L4 GPUs
  - Startup commands for single GPU, multi-GPU, production with LoRA
  - Key parameters documentation (`--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-num-seqs`)
  - OOM troubleshooting based on real deployment experience
- **Testing Infrastructure**: GPU test scripts for quick verification and comprehensive integration testing
  - `test-repl-gpu.py`: Interactive REPL for direct vLLM provider testing
  - `test-gpu.py`: Full stack test with AbstractCore server + curl examples
  - FastDoc UI available at `http://localhost:8080/docs` when server running

### Deployment Experience
- Validated on **4x NVIDIA L4 GPUs** (23GB VRAM each, Scaleway Paris)
- Successfully resolved multi-GPU tensor parallelism requirements
- Fixed sampler warm-up OOM by reducing `--max-num-seqs` from 256 to 128
- Documented Triton kernel compilation issues with MoE models (recommend 7B models for reliability)

### Technical Details
- **Files Created**:
  - `abstractcore/providers/vllm_provider.py` (823 lines)
  - `abstractcore/providers/openai_compatible_provider.py` (764 lines)
  - `tests/providers/test_vllm_provider.py` (371 lines)
  - `tests/providers/test_openai_compatible_provider.py` (328 lines)
- **Files Modified**:
  - `abstractcore/providers/registry.py` (added 2 provider registrations)
  - `abstractcore/providers/__init__.py` (exported 2 new providers)
  - `README.md` (hardware requirements)
  - `docs/prerequisites.md` (multi-GPU setup guide)
- **Architecture**: Both providers inherit from BaseProvider (not OpenAIProvider) for clean httpx implementation
- **Pattern**: vLLM uses `extra_body` for vLLM-specific params; OpenAI-compatible is pure OpenAI-compatible
- **Branch**: `vllm-provider` (pending merge to main)

## [2.6.3] - 2025-12-10

### Changed
- **More Stringent Assessment Scoring**: BasicJudge now applies rigorous, context-aware scoring to prevent grade inflation (2025-12-10)
  - **Anti-Grade-Inflation**: Explicit guidance to avoid defaulting to high scores (3-4) for adequate work
  - **Context-Aware Criteria**: Scores criteria based on task type (e.g., innovation=1-2 for routine calculations, not 3)
  - **Task-Appropriate Expectations**: Different rubrics for routine tasks vs creative work vs complex problem-solving
  - **New Evaluation Step**: "Assess if each criterion meaningfully applies to this task (if not, score 1-2)"
  - **Impact**: More accurate and fair assessments that distinguish between routine competence and genuine excellence
  - **Example**: Basic arithmetic now correctly scores innovation=1-2 (routine formula), not 3 (adequate innovation)
  - **Zero Breaking Changes**: Assessment API unchanged, only internal scoring logic improved

### Added
- **Complete Score Visibility**: `session.generate_assessment()` now returns all predefined criterion scores in structured format
  - **New Field**: `scores` dict containing clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence
  - **Before**: Only overall_score, custom_scores, and text feedback visible
  - **After**: Full transparency with individual scores for both predefined and custom criteria
  - **Impact**: Users can now see exactly how each criterion was scored, not just overall and custom scores
  - **Backward Compatible**: New `scores` field added to assessment result without breaking existing code

### Technical Details
- **Files Modified**: `abstractcore/processing/basic_judge.py` (scoring principles), `abstractcore/core/session.py` (score extraction)
- **Prompt Enhancement**: Added "SCORING PRINCIPLES - CRITICAL" section with 6 explicit guidelines
- **Implementation**: ~15 lines added to scoring rubric, ~10 lines to session assessment storage

## [2.6.2] - 2025-12-01

### Added
- **Programmatic Provider Configuration**: Runtime configuration API for provider settings without environment variables (2025-12-01)
  - **Simple API**: `configure_provider()`, `get_provider_config()`, `clear_provider_config()` functions
  - **Runtime Configuration**: Set provider base URLs and other settings programmatically
  - **Automatic Application**: All future `create_llm()` calls automatically use configured settings
  - **Provider Discovery**: `get_all_providers_with_models()` automatically uses runtime configuration
  - **Use Cases**:
    - Web UI settings pages: Configure providers through user interfaces
    - Docker startup scripts: Read from custom env vars and configure programmatically
    - Integration testing: Set mock server URLs without environment variables
    - Multi-tenant deployments: Configure different base URLs per tenant
  - **Priority System**: Constructor parameter > Runtime configuration > Environment variable > Default value
  - **Implementation**: ~65 lines across 3 files (config/manager.py, config/__init__.py, providers/registry.py)
  - **Testing**: 9/9 tests passing with real implementations (no mocking)
  - **Zero Breaking Changes**: Optional runtime configuration, all existing code works unchanged
  - **Feature Request**: Extension of Digital Article team's base URL configuration request

### Documentation
- **README.md**: Added Programmatic Configuration section with use cases and priority system
- **llms.txt**: Added feature line for v2.6.2
- **llms-full.txt**: Added comprehensive section with Web UI, Docker, testing, and multi-tenant examples
- **FEATURE_REQUEST_RESPONSE_ENV_VARS.md**: Updated with programmatic API examples

### Technical Details
- **Architecture**: Runtime-only (in-memory), not persisted to config JSON file
- **Injection Point**: `ProviderRegistry.create_provider_instance()` merges runtime config into kwargs
- **Pattern**: `merged_kwargs = {**runtime_config, **kwargs}` ensures user kwargs take precedence
- **Backward Compatibility**: All 6 providers work automatically via registry injection
- **Test Coverage**: Unit tests for config methods, provider creation, precedence, and registry integration

## [2.6.1] - 2025-12-01

### Added
- **Environment Variable Support for Provider Base URLs**: Ollama and LMStudio providers now respect environment variables for custom base URLs (2025-12-01)
  - **Ollama Provider**: Supports `OLLAMA_BASE_URL` and `OLLAMA_HOST` environment variables
  - **LMStudio Provider**: Supports `LMSTUDIO_BASE_URL` environment variable
  - **Provider Discovery**: `get_all_providers_with_models()` automatically respects environment variables when checking provider availability
  - **Use Cases**:
    - Remote Ollama servers (e.g., GPU server on `http://192.168.1.100:11434`)
    - Docker/Kubernetes deployments with custom networking
    - Non-standard ports for multi-instance deployments (e.g., `:11435`, `:1235`)
    - Accurate provider availability detection in distributed environments
  - **Priority System**: Programmatic `base_url` parameter > Environment variable > Default value
  - **Implementation**: ~30 lines across 2 providers, follows existing OpenAI/Anthropic pattern
  - **Testing**: 12/12 tests passing with real implementations (no mocking)
  - **Zero Breaking Changes**: Optional environment variables, defaults unchanged, fully backward compatible
  - **Feature Request**: Submitted by Digital Article team for computational notebook deployment

### Documentation
- **README.md**: Added Environment Variables section with examples for all providers
- **llms.txt**: Added feature line for v2.6.1
- **llms-full.txt**: Added comprehensive Environment Variables section with use cases and code examples

### Technical Details
- **Architecture**: Consistent with OpenAI/Anthropic providers (implemented in v2.6.0)
- **Pattern**: `base_url or os.getenv("PROVIDER_BASE_URL") or default_value`
- **Providers Updated**: `ollama_provider.py`, `lmstudio_provider.py`
- **Test Coverage**: Unit tests for env var reading, precedence, defaults, and integration with provider registry

## [2.6.0] - 2025-12-01

### Added
- **Model Download API**: Provider-agnostic async model download with progress reporting (2025-12-01)
  - **Top-Level Function**: `from abstractcore import download_model` - simple, discoverable API
  - **Async Progress Reporting**: Real-time status updates via async generator pattern
  - **Provider Support**:
    - ✅ **Ollama**: Full progress with percent and bytes via `/api/pull` streaming NDJSON
    - ✅ **HuggingFace**: Start/complete messages via `huggingface_hub.snapshot_download`
    - ✅ **MLX**: Same as HuggingFace (uses HF Hub internally)
  - **Progress Information**: `DownloadProgress` dataclass with status, message, percent, downloaded_bytes, total_bytes
  - **Error Handling**: Clear error messages for connection failures, missing models, and gated repositories
  - **Use Cases**: Docker deployments, automated setup, web UIs with SSE streaming, batch downloads
  - **Implementation**: ~240 lines in `abstractcore/download.py`, 11/11 tests passing with real implementations
  - **Zero Breaking Changes**: New functionality only, fully backward compatible

- **Custom Base URL Support**: Configure custom API endpoints for OpenAI and Anthropic providers (2025-12-01)
  - **OpenAI Provider**: `base_url` parameter + `OPENAI_BASE_URL` environment variable
  - **Anthropic Provider**: `base_url` parameter + `ANTHROPIC_BASE_URL` environment variable
  - **Use Cases**:
    - OpenAI-compatible proxies (Portkey, etc.) for observability, caching, cost management
    - Local OpenAI-compatible servers
    - Enterprise gateways for security and compliance
    - Custom endpoints for testing and development
  - **Configuration Methods**: Programmatic parameter (recommended) or environment variables
  - **Implementation**: ~30 lines across 2 providers, follows Ollama/LMStudio pattern
  - **Testing**: 8/10 tests passing, 2 appropriately skipped (OpenAI model validation with test keys)
  - **Zero Breaking Changes**: Optional parameter with None default, fully backward compatible
  - **Note**: Azure OpenAI NOT supported (requires AzureOpenAI SDK class)

- **Production-Ready Native Async Support**: Complete async/await implementation with validated 6-7.5x performance improvement (2025-11-30)
  - **Native Async Providers**: Ollama, LMStudio, OpenAI, Anthropic now use native async clients (httpx.AsyncClient, AsyncOpenAI, AsyncAnthropic)
  - **Performance Validated**:
    - Ollama: 7.5x faster for concurrent requests
    - LMStudio: 6.5x faster for concurrent requests
    - OpenAI: 6.0x faster for concurrent requests
    - Anthropic: 7.4x faster for concurrent requests
  - **Fallback Providers**: MLX and HuggingFace use `asyncio.to_thread()` (industry standard for non-async libraries)
  - **Implementation Time**: 15-16 hours (vs 80-120 hours originally planned) - simplified approach
  - **Code Changes**: ~529 lines across 4 provider files (Ollama, LMStudio native implementations)
  - **Zero Breaking Changes**: All sync APIs unchanged, async purely additive
  - **Testing**: Comprehensive validation with real models (no mocking), 100% success rate

- **Structured Logging Standardization**: Completed migration of 14 core modules to structured logging (2025-12-01)
  - **100% Migration Rate**: 14/14 target files successfully migrated to `get_logger()` from `abstractcore.utils.structured_logging`
  - **Modules Migrated**: tools/ (6 files), architectures/, core/, embeddings/, media/, providers/, utils/
  - **Simplified Approach**: 2 hours implementation (vs 6-12 hours originally planned) - 5-6x more efficient
  - **SOTA Compliance**: Follows PEP 282, Django, FastAPI, and cloud-native patterns
  - **Zero Breaking Changes**: Fully backward compatible, all tests passing
  - **Benefits**: Consistent structured logs, JSON output support, cloud-native ready, improved observability

### Enhanced
- **Async Documentation**:
  - Updated README.md with performance data and provider-specific details
  - Educational [async CLI demo](examples/cli/async_cli_demo.py) with 8 core async/await patterns
  - Created comprehensive async guide in docs/async-guide.md
  - Backlog documents: `async-mlx-hf.md` (investigation), `batching.md` (future enhancement)

- **Observability**: Consistent structured logging across all critical infrastructure
  - Module-level loggers using `get_logger(__name__)` pattern
  - Structured fields support for machine-readable logs (ELK/Datadog/Splunk)
  - Cloud-native JSON output ready
  - No file dependencies (stdout/stderr only)

### Technical Details
- **Architecture**:
  - `BaseProvider._agenerate_internal()` as extension point for native async
  - Lazy-loaded async clients (zero overhead for sync-only users)
  - Proper async cleanup in `unload()` methods
  - Pattern follows SOTA from LangChain, LiteLLM, Pydantic-AI
- **Why MLX/HF use fallback**: Libraries don't expose async APIs, direct function calls (no HTTP layer)
- **SOTA Validation**: Research confirmed approach matches industry best practices

### Performance
- **Average Speedup**: ~7x faster for concurrent requests across all providers
- **Real Concurrency**: True async I/O overlap for network providers (HTTP client/server architecture)
- **Fallback Efficiency**: MLX/HF keep event loop responsive for mixing with async I/O operations

### Documentation
- [Async/Await Support](README.md#async) - Updated usage examples
- [Async Guide](docs/async-guide.md) - Comprehensive examples and patterns
- [Async CLI Demo](examples/cli/async_cli_demo.py) - Educational reference for learning

## [2.5.4] - 2025-11-27

### Added
- **Async/Await Support**: Native async API for concurrent LLM requests with 3-10x performance improvement
  - **`agenerate()` Method**: Async version of `generate()` works with all 6 providers (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
  - **Concurrent Execution**: Use `asyncio.gather()` for parallel requests with proven 3.52x speedup on real workloads
  - **Async Streaming**: Full streaming support with `AsyncIterator` for real-time token generation
  - **Session Async**: `BasicSession.agenerate()` maintains conversation history in async workflows
  - **Zero Breaking Changes**: All sync APIs continue to work unchanged - async is purely additive
  - **FastAPI Compatible**: Works seamlessly with async web frameworks and non-blocking applications
  - **Real Concurrency Verified**: Benchmark tests confirm true async concurrency, not fake async wrappers
  - **Implementation**: ~90 lines in 2 files using `asyncio.to_thread()` for thread-pool async execution
  - **Files Modified**: `abstractcore/providers/base.py`, `abstractcore/core/session.py`
  - **Tests**: Comprehensive test suite with real provider implementations (no mocking) in `tests/async/`

- **Cross-Platform Installation Options**: New installation extras for Linux/Windows users
  - `abstractcore[all-non-mlx]` - Complete installation without MLX (for Linux/Windows)
  - `abstractcore[all-providers-non-mlx]` - All providers except MLX
  - `abstractcore[local-providers-non-mlx]` - Ollama and LMStudio without MLX
  - Fixes installation failures when trying to install MLX on non-macOS systems
  - Comprehensive installation guide: `docs/installation-guide.md`
  - Updated README with platform-specific installation instructions

### Enhanced
- **Async Documentation**: Comprehensive documentation updates across all guides
  - **README.md**: Added async to Key Features and dedicated Async/Await section with examples
  - **docs/getting-started.md**: New Section 6 covering async patterns and use cases
  - **docs/api-reference.md**: Complete API documentation for `agenerate()` methods
  - **docs/README.md**: Added async to Essential Guides navigation
  - **llms.txt**: Added async code examples and capabilities for AI consumption
  - **llms-full.txt**: Comprehensive async section with 4 subsections (basic, streaming, session, multi-provider)

### Fixed
- **Platform Compatibility**: `pip install abstractcore[all]` no longer fails on Linux/Windows
  - Previously, `abstractcore[all]` would fail on non-macOS systems due to MLX dependencies
  - Users should now use `abstractcore[all-non-mlx]` on Linux/Windows for complete installation

### Technical
- **Async Implementation Details**:
  - Uses `asyncio.to_thread()` to run sync methods in thread pool without blocking event loop
  - Proper `AsyncIterator` protocol for streaming responses
  - Works with all existing provider implementations automatically via `BaseProvider`
  - Full parameter passthrough for all generation options
  - Tested with real LLM calls across all providers

### Performance
- **Verified Speedup**: Benchmark testing shows 3.52x improvement for concurrent requests
  - Sequential: 0.93s for 3 requests
  - Concurrent: 0.26s for 3 requests with `asyncio.gather()`
  - Real async concurrency confirmed (not fake async wrappers)

### Use Cases
- Batch document processing
- Multi-provider consensus/comparison
- Non-blocking web applications (FastAPI, async frameworks)
- Parallel data extraction tasks
- High-throughput API endpoints

## [2.5.3] - 2025-11-10

### Added
- Added programmatic interaction tracing to capture complete LLM interaction history, enabling debugging, compliance, and performance analysis.
- Introduced provider-level and session-level tracing with customizable metadata and automatic trace collection.
- Implemented trace retrieval and export utilities for JSONL, JSON, and Markdown formats.
- Enhanced documentation and examples for interaction tracing usage and benefits.
- Comprehensive test coverage added for tracing functionality, ensuring reliability and correctness.

- **MiniMax M2 Model Support**: Added comprehensive detection for MiniMax M2 Mixture-of-Experts model
  - **Model Specs**: 230B total parameters with 10B active (MoE architecture)
  - **Capabilities**: Native tool calling, structured outputs, interleaved thinking with `<think>` tags
  - **Context Window**: 204K tokens (industry-leading), optimized for coding and agentic workflows
  - **Variant Detection**: Supports all distribution formats:
    - `minimax-m2` (canonical name)
    - `MiniMaxAI/MiniMax-M2` (HuggingFace official)
    - `mlx-community/minimax-m2` (MLX quantized)
    - `unsloth/MiniMax-M2-GGUF` (GGUF format)
  - **Case-Insensitive**: All variants detected regardless of case (e.g., `MiniMax-M2`, `MINIMAX-m2`)
  - **Source**: Official MiniMax documentation (minimax-m2.org, HuggingFace, GitHub)
  - **License**: Apache-2.0 with no commercial restrictions
  - **Note**: Added single entry in `model_capabilities.json` with comprehensive aliases for automatic detection across all distribution formats

- **[EXPERIMENTAL] Glyph Visual-Text Compression**: Renders long text as optimized images for VLM processing
  - ⚠️ **Vision Model Requirement**: ONLY works with vision-capable models (gpt-4o, claude-3-5-sonnet, llama3.2-vision, etc.)
  - ⚠️ **Error Handling**: `glyph_compression="always"` raises `UnsupportedFeatureError` if model lacks vision support
  - ⚠️ **Auto Mode**: `glyph_compression="auto"` (default) logs warning and falls back to text processing for non-vision models
  - PIL-based text rendering with custom font support and proper DPI scaling
  - Markdown-like formatting with hierarchical headers, bold/italic text, and smart newline handling
  - Multi-column layout support with configurable spacing and margins
  - Special OCRB font family support with separate regular/italic variants and stroke-based bold effect
  - Font customization via `--font` (by name) and `--font-path` (by file) parameters
  - Research-based VLM token calculator with provider-specific formulas
  - Thread-safe caching system in `~/.abstractcore/glyph_cache/`
  - Optional dependencies: `pip install abstractcore[compression]` (removed ReportLab dependency)
  - Vision capability validation in `AutoMediaHandler._should_apply_compression()`

### Enhanced
- **Model Capability Filtering**: Clean, type-safe system for filtering models by input/output capabilities
  - **Input Capabilities**: Filter by what models can analyze (TEXT, IMAGE, AUDIO, VIDEO)
  - **Output Capabilities**: Filter by what models generate (TEXT, EMBEDDINGS)
  - **Python API**: `list_available_models(input_capabilities=[...], output_capabilities=[...])`
  - **HTTP API**: `/v1/models?input_type=image&output_type=text`
  - **All Providers**: Works consistently across OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace

- **Text File Support**: Media module now supports 90+ text-based file extensions with intelligent content detection
  - **Expanded Mappings**: Added support for programming languages (.py, .js, .r, .R, .rs, .go, .jl, etc.), notebooks (.ipynb, .rmd), config files (.yaml, .toml, .ini), web files (.css, .vue, .svelte), build scripts (.sh, .dockerfile), and more
  - **Smart Detection**: Unknown extensions are analyzed via content sampling (UTF-8, Latin-1, etc.) to automatically detect text files
  - **Programmatic Access**: New `get_all_supported_extensions()` and `get_supported_extensions_by_type()` functions for querying supported formats
  - **CLI Enhancement**: `@filepath` syntax now works with ANY text-based file (R scripts, Jupyter notebooks, SQL files, etc.)
  - **Fallback Processing**: TextProcessor handles all text files via plain text fallback, ensuring universal support
- **Model Capabilities**: Added 50+ VLM models (Mistral Small 3.1/3.2, LLaMA 4, Qwen3-VL, Granite Vision)
- **Detection System**: All model queries go through `detection.py` with structured logging
- **Token Calculation**: Accurate image tokenization using model-specific parameters
- **Offline-First Architecture**: AbstractCore now enforces offline-first operation by default
  - Added centralized offline configuration in `config/manager.py` 
  - HuggingFace provider loads models directly from local cache when offline
  - Environment variables (`TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`) set automatically
  - Uses centralized cache directory configuration
  - Designed primarily for open source LLMs with full offline capability
- **HuggingFace Provider**: Added vision model support for GLM4V architecture (Glyph, GLM-4.1V)
  - Upgraded transformers requirement to >=4.57.1 for GLM4V architecture support
  - Added `_is_vision_model()` detection for AutoModelForImageTextToText models
  - Added `_load_vision_model()` and `_generate_vision_model()` methods
  - Proper multimodal message handling with AutoProcessor
  - Suppressed progress bars and processor warnings during model loading
- **Vision Compression**: Enhanced test script with exact token counting from API responses
  - Added `--detail` parameter for Qwen3-VL token optimization (`low`, `high`, `auto`, `custom`)
  - Added `--target-tokens` parameter for precise token control per image
  - Improved compression ratio calculation using actual vs estimated tokens
  - Added model-specific context window validation and warnings
- **Media Handler Architecture**: Clarified OpenAI vs Local handler usage patterns
  - LMStudio uses OpenAIMediaHandler for vision models (API compatibility)
  - Ollama uses LocalMediaHandler with custom image array format
  - Added comprehensive architecture documentation and diagrams

### Fixed
- **Cache Creation**: Automatic directory creation with proper error handling
- **Dependency Validation**: Structured logging for missing libraries  
- **Compression Pipeline**: Fixed parameter passing and quality threshold bypass
- **GLM4V Architecture**: Fixed `KeyError: 'glm4v'` when loading Glyph and GLM-4.1V models
- **Text Formatting Performance**: Fixed infinite loop in inline formatting parser for large files
- **Text Pagination**: Implemented proper multi-image splitting for long texts
- **Literal Newline Handling**: Fixed `\\n` sequences not being converted to actual newlines
- **Token Estimation**: Added model-specific visual token calculations and context overflow protection
- **Media Path Logging**: Fixed media output paths not showing in INFO logs
- **Qwen3-VL Context Management**: Auto-adjusts detail level to prevent memory allocation errors
- **LMStudio GLM-4.1V Compatibility**: Documented LMStudio's internal vision config limitations
- **HuggingFace GLM4V Support**: Added proper error handling for transformers version requirements
- Requires vision-capable models (llama3.2-vision, qwen2.5vl, gpt-4o, claude-3-5-sonnet, zai-org/Glyph)
- System dependency on poppler-utils may require manual installation on some systems
- Quality assessment heuristics may be overly conservative for some document types

## [2.5.2] - 2025-10-26

### Added
- **Native Structured Output Support for HuggingFace GGUF Models**: HuggingFace provider now supports server-side schema enforcement for GGUF models via llama-cpp-python's `response_format` parameter
  - GGUF models loaded through HuggingFace provider automatically get native structured output support
  - Uses the same OpenAI-compatible `response_format` parameter as LMStudio
  - Server-side schema enforcement validates output against the provided schema
  - Transformers models continue to use prompted approach as fallback
  - Provider registry updated to advertise structured output capability
- **Native Structured Output via Outlines for HuggingFace Transformers**: HuggingFace Transformers models now support native structured output via optional Outlines integration
  - Constrained decoding ensures 100% schema compliance without validation retries
  - Optional dependency - only installed with `pip install abstractcore[huggingface]`
  - Automatic detection and activation when Outlines is available
  - Graceful fallback to prompted approach if Outlines not installed
  - Works with any transformers-compatible model
  - Server-side logit filtering guarantees valid token selection
- **Native Structured Output via Outlines for MLX**: MLX models now support native structured output via optional Outlines integration
  - Constrained decoding on Apple Silicon with 100% schema compliance
  - Optional dependency - only installed with `pip install abstractcore[mlx]`
  - Automatic detection and activation when Outlines is available
  - Graceful fallback to prompted approach if Outlines not installed
  - Optimized for Apple M-series processors
  - Zero validation retries required

### Changed
- **StructuredOutputHandler**: Enhanced provider detection to identify HuggingFace GGUF models, Transformers with Outlines, and MLX with Outlines as having native support
  - Checks for `model_type == "gguf"` to determine GGUF native support
  - Checks for `model_type == "transformers"` with Outlines availability for Transformers native support
  - Checks for Outlines availability for MLX native support
  - GGUF models benefit from llama-cpp-python's constrained sampling
  - Transformers and MLX models benefit from Outlines constrained decoding when available
  - Automatic fallback to prompted strategy if Outlines not installed
- **Structured Output Control**: Added `structured_output_method` parameter to HuggingFace and MLX providers for explicit control
  - `"auto"` (default): Use Outlines if available, fallback to prompted
  - `"native_outlines"`: Force Outlines usage (error if unavailable)
  - `"prompted"`: Always use prompted fallback (recommended - fastest, 100% success)
  - Allows users to optimize for performance vs theoretical guarantees
- **Model Capabilities**: Verified and documented native structured output support for Ollama and LMStudio providers
  - Ollama: Confirmed correct implementation using `format` parameter with full JSON schema
  - LMStudio: Documented existing OpenAI-compatible `response_format` implementation
  - Both providers leverage server-side schema enforcement for schema compliance
- **Dependencies**: Added Outlines as optional dependency for HuggingFace and MLX providers
  - `pip install abstractcore[huggingface]` now includes Outlines for native structured output
  - `pip install abstractcore[mlx]` now includes Outlines for native structured output
  - Base installation remains lightweight - Outlines only installed when needed

### Fixed
- **HuggingFace Provider**: Added missing `response_model` parameter propagation through internal generation methods
  - Fixed `_generate_internal()` to pass `response_model` to both GGUF and transformers backends
  - Both `_generate_gguf()` and `_generate_transformers()` now accept and handle `response_model` parameter
- **Provider Registry**: Added `"structured_output"` to supported features for Ollama, LMStudio, HuggingFace, and MLX providers
  - Ensures accurate capability reporting for structured output functionality

### Performance Notes

**Surprising Findings from Comprehensive Testing** (October 26, 2025):

Extensive testing on Apple Silicon M4 Max revealed unexpected performance characteristics:

**MLX Provider** (mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit):
- **Prompted fallback**: 745-4,193ms, 100% success rate
- **Outlines native**: 2,031-9,840ms, 100% success rate
- **Overhead**: 173-409% slower with Outlines constrained generation
- **Conclusion**: Both approaches achieve 100% schema compliance, but prompted is 2-5x faster

**Key Insight**: The prompted approach (client-side validation) achieves identical 100% success rate at significantly better performance than Outlines' server-side constrained generation. This is contrary to typical expectations where server-side constraints should be more reliable.

**Recommendation**:
- Default to `structured_output_method="prompted"` for best performance with proven reliability
- Use `structured_output_method="native_outlines"` only when theoretical guarantees are required despite performance cost
- The `"auto"` setting uses Outlines if installed, which may impact performance without improving reliability

This finding suggests that for these specific models and use cases, the overhead of constrained decoding outweighs its benefits when client-side validation already achieves 100% success.

## [2.5.1] - 2025-10-24

### Added
- New `intent` CLI application for analyzing conversation intents and detecting deception patterns
- `/intent` command in interactive CLI to analyze participant motivations in real-time conversations
- Support for multi-participant conversation analysis with focus on specific participants
- **Native Structured Output Support**: LMStudio provider now supports server-side schema enforcement via OpenAI-compatible `response_format` parameter
  - Structured outputs are now guaranteed to match the provided schema without retry logic
  - Works seamlessly with Pydantic models through the existing `response_model` parameter
  - Provider registry updated to advertise structured output capability

### Changed
- Renamed "Internal CLI" to "AbstractCore CLI" throughout documentation
- File renamed: `docs/internal-cli.md` → `docs/acore-cli.md`
- **Model Capabilities**: Updated 50+ Ollama-compatible models to report native structured output support (Llama, Qwen, Gemma, Mistral, Phi families)
  - This reflects the actual server-side schema enforcement capabilities these models have when used with Ollama
- **Provider Registry**: Added `"structured_output"` to supported features for both Ollama and LMStudio providers

### Fixed
- Updated all documentation cross-references to use new CLI naming
- **Ollama Provider**: Improved documentation of native structured output implementation (was already correct, now better documented)
- **StructuredOutputHandler**: Enhanced provider detection logic to correctly identify Ollama and LMStudio as having native support regardless of configuration

## [2.4.9] - 2025-10-21

### Fixed
- **Configuration System**: Fixed missing configuration module that caused `'NoneType' object is not callable` error
  - Renamed `abstractcore/cli` to `abstractcore/config` to match expected import path
  - Added complete configuration manager implementation with vision, embeddings, and app defaults
  - Fixed `abstractcore --set-vision-provider` and all other configuration commands

## [2.4.7] - 2025-10-21

### Fixed
- **Tools Dependencies**: Added missing `requests` dependency to core requirements and created `tools` optional extra for enhanced functionality

### Added

#### Consistent Token Terminology
- **Unified Token Naming**: Standardized token terminology across AbstractCore to match input parameter naming
  - `GeneratedResponse` now provides `input_tokens`, `output_tokens`, `total_tokens` properties
  - Maintains backward compatibility with legacy `prompt_tokens` and `completion_tokens` keys
  - All providers now use consistent terminology in usage dictionaries
  - Token counts sourced from: Provider APIs (OpenAI, Anthropic, LMStudio) or AbstractCore's `token_utils.py` (MLX, HuggingFace)

#### Token Count Source Transparency
- **Provider-Specific Token Handling**: Clear documentation of token count sources
  - **From Provider APIs**: OpenAI, Anthropic, LMStudio (native API token counts)
  - **From AbstractCore**: MLX, HuggingFace providers (calculated using `token_utils.py`)
  - **Mixed Sources**: Ollama (combination of provider and calculated tokens)
- **Consistent Interface**: All providers normalized through unified `GeneratedResponse.usage` structure

#### Generation Time Tracking
- **Universal Timing**: Added `gen_time` property to `GeneratedResponse` across all providers (in milliseconds)
  - **Precise Measurement**: Tracks actual API call duration for network-based providers (OpenAI, Anthropic, LMStudio, Ollama)
  - **Local Processing Time**: Measures inference time for local providers (MLX, HuggingFace)
  - **Simulated Timing**: Local providers include realistic timing simulation
  - **Precision**: Rounded to 1 decimal place for clean, readable output
- **Performance Insights**: Enables performance monitoring, optimization, and comparative analysis across providers
- **Summary Integration**: Generation time automatically included in `response.get_summary()` output

## [2.4.6] - 2025-10-21

### Added

#### Enhanced fetch_url Tool Performance
- **Optimized HTML Parsing**: Added lxml parser support for 2-3x faster HTML processing (with html.parser fallback)
- **Session-Based Connection Reuse**: Improved network performance through connection pooling
- **Enhanced Encoding Detection**: Multiple encoding fallback strategies for better text decoding reliability
- **Improved Content Extraction**: Better main content detection, removes navigation/footer/sidebar elements
- **Smart Download Chunking**: Optimized chunk sizes based on content type (32KB for binary, 16KB for text)
- **Better JSON Formatting**: Smart truncation at logical boundaries for improved readability

#### Universal SEED and Temperature Control
- **Unified Parameter Support**: Added comprehensive `seed` and `temperature` parameter support across all 6 providers
  - **Provider-Level**: All providers now accept `seed` and `temperature` parameters in constructor and generate() calls
  - **Session-Level**: BasicSession now supports persistent `temperature` and `seed` parameters across conversation
  - **Parameter Inheritance**: Session parameters are used as defaults, can be overridden per generate() call
  - **Consistent Interface**: Same API works across OpenAI, Anthropic, HuggingFace, Ollama, LMStudio, and MLX providers

#### Provider-Specific SEED Implementation
- **OpenAI**: Native `seed` parameter support for deterministic outputs (except reasoning models like o1)
- **Anthropic**: Graceful fallback with debug logging (Claude API doesn't support seed natively)
- **HuggingFace**: Full seed support for both transformers (`torch.manual_seed()`) and GGUF models (`llama-cpp-python`)
- **Ollama**: Native `seed` parameter support via options
- **LMStudio**: OpenAI-compatible `seed` parameter support
- **MLX**: Graceful fallback with debug logging (MLX-LM has limited seed support)

#### Enhanced Temperature Control
- **Consistent Handling**: Improved temperature parameter consistency across all providers
- **Session Persistence**: Temperature can be set at session level and persists across generate() calls
- **Provider Defaults**: Each provider maintains its own default temperature (0.7) when not specified

### Enhanced

#### Architectural Improvements (Post-Implementation Review)
- **Interface-Level Parameter Declaration**: Moved `temperature` and `seed` to `AbstractCoreInterface` for consistent contract
- **Eliminated Code Duplication**: Removed redundant parameter initialization from all 6 providers (DRY principle)
- **Centralized Parameter Logic**: Added `_extract_generation_params()` helper method for consistent parameter extraction
- **Cleaner Provider Code**: Providers now focus only on their specific configuration, inheriting common parameters
- **Robust Fallback Hierarchy**: kwargs → instance variables → interface defaults with elegant one-liner implementation

#### Session Management
- **Parameter Persistence**: Session-level temperature and seed are maintained across conversation
- **Flexible Override**: Per-call parameters override session defaults without changing session state
- **Enhanced Documentation**: Updated session docstrings with parameter descriptions

### Technical Details

#### Implementation Strategy & Architecture Review
- **Non-Breaking**: All changes are backward compatible - existing code continues to work
- **Provider-Agnostic**: Same seed/temperature API works regardless of underlying provider capabilities
- **Graceful Degradation**: Providers that don't support seed log debug messages instead of failing
- **Clean Architecture**: Leveraged existing parameter inheritance system in BaseProvider

#### Code Quality Improvements (Independent Review)
- **Eliminated Duplication**: Removed 12 lines of identical parameter initialization across 6 providers
- **Interface Contract**: Parameters now declared at interface level, ensuring consistent API contract
- **Centralized Logic**: Single `_extract_generation_params()` method replaces scattered parameter handling
- **Simplified Providers**: Each provider reduced by 2-4 lines, focusing only on provider-specific concerns
- **Maintainability**: Future parameter additions only require interface-level changes, not per-provider updates

#### Usage Examples
```python
# Provider-level parameters
llm = create_llm("openai", model="gpt-4", temperature=0.3, seed=42)
response = llm.generate("Hello", temperature=0.8)  # Override temperature for this call

# Session-level parameters
session = BasicSession(provider=llm, temperature=0.5, seed=123)
response1 = session.generate("First message")  # Uses session temperature=0.5, seed=123
response2 = session.generate("Second message", temperature=0.9)  # Override temperature, keep seed
```

### Architecture Review Summary

After independent analysis, the implementation was **refactored for maximum elegance and maintainability**:

#### Original Issues Identified
- Code duplication across 6 providers (12 identical lines)
- Inconsistent parameter handling patterns
- Missing interface-level parameter contract
- Scattered parameter extraction logic

#### Architectural Improvements Applied
- **Interface-Level Declaration**: Parameters moved to `AbstractCoreInterface` for consistent contract
- **DRY Principle**: Eliminated all parameter duplication across providers
- **Centralized Logic**: Single `_extract_generation_params()` method for consistent behavior
- **Cleaner Providers**: Each provider reduced by 2-4 lines, focusing only on provider-specific concerns
- **Future-Proof**: New parameters require only interface-level changes, not per-provider updates

#### Quality Metrics
- **Lines Reduced**: 12 lines of duplication eliminated
- **Maintainability**: 83% reduction in parameter-related code across providers
- **Consistency**: 100% uniform parameter handling across all 6 providers
- **Extensibility**: New parameters can be added with 2 lines instead of 12

See [Generation Parameters Architecture](docs/generation-parameters.md) for detailed technical analysis.

### Testing & Verification

#### Comprehensive Test Suite
- **Basic Parameter Tests**: `tests/test_seed_temperature_basic.py` - CI/CD compatible parameter handling tests
- **Determinism Tests**: `tests/test_seed_determinism.py` - Real-world determinism verification across providers
- **Manual Verification**: `tests/manual_seed_verification.py` - Interactive script for testing actual determinism
- **Test Documentation**: `tests/README_SEED_TESTING.md` - Complete testing guide and troubleshooting

#### Provider Support Verification
- **OpenAI**: ✅ Native seed support (verified deterministic)
- **Anthropic**: ❌ No seed support (issues UserWarning when seed provided)
- **HuggingFace**: ✅ Full support for transformers and GGUF models
- **Ollama**: ✅ Native seed support via options
- **LMStudio**: ✅ OpenAI-compatible seed support
- **MLX**: ✅ Native seed support via mx.random.seed() (corrected implementation)

#### Real-World Testing & Verification ✅
**Empirically Verified**: All providers except Anthropic achieve true determinism with `seed + temperature=0`:

```bash
# Verified deterministic behavior (100% success rate):
✅ OpenAI (gpt-3.5-turbo): Same seed → Identical outputs
✅ Ollama (gemma3:1b): Same seed → Identical outputs  
✅ MLX (Qwen3-4B): Same seed → Identical outputs
⚠️ Anthropic (claude-3-haiku): temperature=0 → Consistent outputs (no seed support)
```

**Test Commands**:
```bash
# Test all available providers
python tests/manual_seed_verification.py

# Test specific provider determinism
python tests/manual_seed_verification.py --provider openai --prompt "Count to 5"
```

## [2.4.5] - 2025-10-21

### Fixed

#### Critical Package Distribution Bug
- **Missing Media Subpackages**: Fixed critical package installation bug where media subpackages were not included in distribution
  - **Issue**: `pyproject.toml` only listed `abstractcore.media` parent package but not its subpackages
  - **Impact**: Import `from abstractcore import create_llm` failed with `ModuleNotFoundError: No module named 'abstractcore.media.processors'`
  - **Missing Packages**:
    - `abstractcore.media.processors` (ImageProcessor, PDFProcessor, OfficeProcessor, TextProcessor)
    - `abstractcore.media.handlers` (OpenAIMediaHandler, AnthropicMediaHandler, LocalMediaHandler)
    - `abstractcore.media.utils` (image_scaler utilities)
  - **Solution**: Explicitly added all media subpackages to packages list in `pyproject.toml`
  - **Root Cause**: When explicitly listing packages in pyproject.toml, setuptools does NOT auto-discover subpackages
  - **Workaround for 2.4.4**: Use `from abstractcore.core.factory import create_llm` instead of `from abstractcore import create_llm`
  - **Credit**: Bug discovered and reported during production deployment testing

#### Missing CLI Package
- **Missing abstractcore.cli Module**: Fixed missing `abstractcore.cli` package from distribution
  - **Issue**: CLI entry point `abstractcore` command referenced `abstractcore.cli.main:main` but module was not included in package
  - **Impact**: Configuration CLI commands would fail after installation from PyPI
  - **Solution**: Added `abstractcore.cli` to packages list in `pyproject.toml`

### Added

#### CLI Entry Point Improvements
- **New Entry Points**: Added convenient aliases to clarify CLI purpose and improve user experience
  - `abstractcore-config`: Alias for `abstractcore` command (configuration CLI for settings, API keys, models)
  - `abstractcore-chat`: New entry point for interactive REPL (`abstractcore.utils.cli` → LLM interaction)
  - **Purpose**: Distinguish between configuration CLI (manage settings) and interactive chat CLI (talk to LLMs)
  - **Backwards Compatible**: All existing commands continue to work (`abstractcore`, `python -m abstractcore.utils.cli`)

### Technical

#### Package Configuration
- **Updated packages list** in `pyproject.toml` to include all required modules:
  ```toml
  packages = [
      # ... existing packages ...
      "abstractcore.media",
      "abstractcore.media.processors",  # ✅ Added
      "abstractcore.media.handlers",    # ✅ Added
      "abstractcore.media.utils",       # ✅ Added
      "abstractcore.cli"                # ✅ Added
  ]
  ```
- **Verification**: All 19 packages now properly included in distribution
- **Testing**: Recommended to always test `pip install` from built wheel before PyPI release

### Benefits
- **Installation Works**: Users can now successfully `pip install abstractcore[all]` or `pip install abstractcore[media]`
- **Complete Media System**: All media processing capabilities (images, PDFs, Office docs) now accessible after installation
- **Clear CLI Commands**: Users have obvious entry points for different CLI purposes
- **Production Ready**: Package installation thoroughly tested and verified

### Migration Guide

No migration needed - this is a pure bug fix release. If you experienced installation issues with 2.4.4:

1. **Upgrade**: `pip install --upgrade abstractcore`
2. **Verify**: `python -c "from abstractcore import create_llm; print('✅ Works!')"`
3. **Use new CLI aliases** (optional):
   - `abstractcore-config --status` instead of `abstractcore --status`
   - `abstractcore-chat` instead of `python -m abstractcore.utils.cli`

## [2.4.4] - 2025-10-21

### Added

#### Provider Health Check System
- **NEW `.health()` Method**: Unified health check interface for all providers
  - **Structured Response**: Consistent health status format across all providers
  - **Connectivity Testing**: Uses `list_available_models()` as implicit connectivity test
  - **Smart Timeout Management**: Configurable timeout (default: 5.0s) with automatic restoration
  - **Never Throws**: Errors captured in response structure, never raises exceptions
  - **Rich Information**: Returns status, provider name, model list, model count, error message, and latency
  - **Universal Compatibility**: Works with all provider types (API, local, server-based)
  - **Override-able**: Providers can customize health check logic if needed

#### Health Check Response Structure
```python
{
    "status": bool,              # True if provider is healthy/online
    "provider": str,             # Provider class name (e.g., "OllamaProvider")
    "models": List[str] | None,  # Available models if online, None if offline
    "model_count": int,          # Number of models available (0 if offline)
    "error": str | None,         # Error message if offline, None if healthy
    "latency_ms": float          # Health check duration in milliseconds
}
```

### Fixed

#### HuggingFace Token Counting Consistency
- **Centralized Token Counter**: Fixed HuggingFace provider to use centralized `TokenUtils` for consistency
  - **Problem**: HuggingFace was the only provider using provider-specific `tokenizer.encode()` for token counting
  - **Solution**: Added `_calculate_usage()` method matching MLX provider pattern using `TokenUtils.estimate_tokens()`
  - **Impact**: All local providers now consistently use centralized token counting infrastructure
  - **Benefits**:
    - ✅ Consistency across all providers (MLX, HuggingFace)
    - ✅ Robustness when tokenizer unavailable (GGUF models)
    - ✅ Content-type detection for better accuracy (code vs text vs JSON)
    - ✅ Model-family adjustments (qwen, llama, mistral tokenization patterns)

### Enhanced

#### Token Usage Tracking
- **Comprehensive Token Capture**: All providers consistently capture THREE token metrics
  - **prompt_tokens**: Input/context tokens (system prompt + history + current prompt)
  - **completion_tokens**: Generated/output tokens (model's response)
  - **total_tokens**: Sum of prompt + completion (used for billing/quotas)
  - **API Providers**: OpenAI, Anthropic, Ollama, LMStudio use exact API-provided counts
  - **Local Providers**: MLX, HuggingFace use centralized `TokenUtils` estimation

### Technical

#### Token Counting Implementation
- **Centralized Infrastructure**: Located at `abstractcore/utils/token_utils.py`
  - `TokenUtils.estimate_tokens(text, model)`: Fast estimation with content-type detection
  - `TokenUtils.count_tokens(text, model, method)`: Flexible counting (auto/precise/fast)
  - `TokenUtils.count_tokens_precise(text, model)`: Accurate counting with tiktoken when available
  - Multi-tiered strategy: tiktoken (precise) → provider tokenizer → model-aware heuristics → fast fallback

#### Files Modified
- `abstractcore/providers/base.py`: Added `health()` method (lines 870-965)
- `abstractcore/providers/huggingface_provider.py`:
  - Added `_calculate_usage()` method using centralized TokenUtils (lines 890-902)
  - Updated `_single_generate_transformers()` to use centralized token counting (lines 867-868)

### Benefits
- **Health Monitoring**: Simple interface to check provider connectivity and availability
- **Consistency**: Unified token counting across all providers with same methodology
- **Production Ready**: Built-in timeout management prevents hanging health checks
- **Developer Experience**: Rich health information enables better error handling and monitoring
- **Maintainability**: Single centralized token counter to update/improve

### Migration Guide

#### For Health Check Users
New `.health()` method available on all providers:

```python
from abstractcore.core.factory import create_llm

# Check single provider
provider = create_llm("ollama", model="llama2")
health = provider.health(timeout=3.0)

if health["status"]:
    print(f"✅ {health['provider']} is healthy!")
    print(f"   📦 {health['model_count']} models available")
    print(f"   ⏱️  {health['latency_ms']}ms response time")
else:
    print(f"❌ {health['provider']} is offline")
    print(f"   Error: {health['error']}")
```

#### For Token Counting
No changes required - all existing code continues to work. HuggingFace provider now uses the same centralized token counting infrastructure as other local providers, improving consistency and accuracy.

## [2.4.3] - 2025-10-20

### Major Features

#### OpenAI Responses API Compatibility
- **NEW `/v1/responses` Endpoint**: 100% compatible with OpenAI's Responses API format
  - **input_file Support**: Native support for `{"type": "input_file", "file_url": "..."}` in content arrays
  - **Backward Compatible**: Existing `messages` format continues to work alongside new `input` format
  - **Automatic Format Detection**: Server automatically detects and converts between OpenAI and legacy formats
  - **Streaming Support**: Optional streaming with `"stream": true` for real-time responses (defaults to `false`)
  - **Universal File Processing**: Works with all file types (PDF, DOCX, XLSX, CSV, images) across all providers

#### Enhanced File Attachment System
- **type="file" Support**: New content type alongside `"text"` and `"image_url"` for explicit file attachments
  - **Unified Format**: `{"type": "file", "file_url": {"url": "..."}}` works consistently across all endpoints
  - **Multiple Sources**: Supports HTTP(S) URLs, local file paths, and base64 data URLs
  - **Content-Type Detection**: Intelligent file type detection from headers and URL extensions
  - **Generic Downloader**: Replaces image-only downloader with universal file download supporting 15+ file types

#### Production-Grade PDF Processing
- **Complete Text Extraction**: Full PDF content extraction using PyMuPDF4LLM with formatting preservation
  - **40,000+ Character Support**: Successfully tested with large documents (Berkshire Hathaway annual letter)
  - **LLM-Optimized Output**: Markdown formatting with preserved tables, headers, and structure
  - **Automatic Installation**: Added PyMuPDF4LLM, PyMuPDF, and Pillow to dependencies
  - **Graceful Fallbacks**: Multi-level fallback ensures content extraction even if advanced processing fails

#### Centralized Configuration System
- **Global Configuration Management**: Unified configuration at `~/.abstractcore/config/abstractcore.json`
  - **App-Specific Defaults**: Set different models for CLI, summarizer, extractor, and judge apps
  - **Global Fallbacks**: Configure fallback models when app-specific settings aren't available
  - **API Key Management**: Centralized API key storage for all providers
  - **Cache Configuration**: Configurable cache directories for HuggingFace, local models, and general cache
  - **Logging Control**: Console and file logging levels with enable/disable commands
  - **Streaming Defaults**: Configure default streaming behavior for CLI applications

#### Comprehensive Media Handling System
- **Universal Media API**: Same `media=[]` parameter works across all providers with automatic format conversion
  - **Image Processing**: Automatic resolution optimization for each model's maximum capability (GPT-4o: 4096px, Claude 3.5: 1568px, qwen2.5vl: 3584px)
  - **Document Processing**: Full support for PDF, DOCX, XLSX, PPTX with complete content extraction
  - **Data Files**: CSV, TSV, JSON, XML with intelligent parsing and analysis
  - **Provider-Specific Formatting**: Automatic conversion to OpenAI JSON, Anthropic Messages API, or local text embedding
  - **Error Handling**: Multi-level fallback strategy ensures users always get meaningful results

#### Vision Capabilities and Fallback System
- **Vision Fallback for Text-Only Models**: Transparent two-stage pipeline enables image processing for any model
  - **Automatic Detection**: Identifies when text-only models receive images and activates fallback
  - **One-Command Setup**: `abstractcore --download-vision-model` downloads and configures BLIP vision model
  - **Flexible Configuration**: Supports local models (BLIP, ViT-GPT2, GIT), Ollama, LMStudio, and cloud APIs
  - **Transparent Operation**: Users don't need to change code - system handles vision fallback automatically

### Server Enhancements

#### Enhanced Debug and Logging
- **Command-Line Arguments**: Added `--debug`, `--host`, and `--port` flags for flexible server startup
  - **Debug Mode**: `--debug` enables comprehensive request/response logging with timing metrics
  - **Custom Binding**: `--host` and `--port` allow custom server addresses (default: 127.0.0.1:8000)
  - **Environment Integration**: Follows centralized config patterns with `ABSTRACTCORE_DEBUG` variable

- **Comprehensive Error Reporting**: Enhanced 422 validation error handling with actionable diagnostics
  - **Field-Level Details**: Shows exact field path, validation message, and problematic input
  - **Request Body Capture**: In debug mode, logs full request body for troubleshooting
  - **Structured Logging**: JSON-formatted logs with client IP, timing, and error context
  - **Before vs After**: "422 Unprocessable Entity" now shows detailed field validation errors

#### Media Processing Integration
- **OpenAI Vision API Format**: Full support for `image_url` objects with base64 data URLs and HTTP(S) URLs
- **File Processing Pipeline**: Automatic media extraction, validation, and cleanup with request-specific prefixes
- **Size Limits**: 10MB per file, 32MB total per request with comprehensive validation
- **Cleanup Logic**: Automatic temporary file cleanup for `abstractcore_img_*`, `abstractcore_file_*`, and `abstractcore_b64_*` prefixes
- **Prompt Adaptation**: Intelligent prompt adaptation based on file types to avoid confusion

### Fixed

#### Critical Runtime Issues
- **Time Module Scoping**: Removed redundant local `import time` statements causing "cannot access local variable" errors
  - Fixed in lines 1995-1996 and 2123-2124 of `abstractcore/server/app.py`
  - Now uses global time import consistently throughout server

- **Boolean Syntax**: Corrected JavaScript boolean syntax (`false`/`true`) to Python syntax (`False`/`True`)
  - Fixed in lines 625, 813, 824, 1170, 1181, 1214 across request examples and defaults

- **Streaming Default**: Changed `/v1/responses` endpoint default from `stream=True` to `stream=False`
  - Aligns with OpenAI API standard behavior (streaming opt-in, not opt-out)
  - Line 361 in `OpenAIResponsesRequest` model

#### Swagger UI Integration
- **Payload Input Issue**: Fixed `/v1/responses` endpoint not showing request body in Swagger "Try it out"
  - Replaced raw `Request` parameter with proper FastAPI `Body(...)` annotation
  - Added comprehensive examples for OpenAI format, legacy format, file analysis, and streaming
  - Lines 1148-1220 now properly expose request schema to OpenAPI documentation

#### Media Processing Reliability
- **PDF Download Failures**: Created generic file downloader replacing image-only version
  - Added proper `Accept: */*` headers instead of image-specific headers
  - Comprehensive content-type mapping for PDF, DOCX, XLSX, CSV, and 10+ other types
  - URL extension fallback when content-type header missing
  - Lines 1502-1627 in `abstractcore/server/app.py`

### Enhanced

#### CLI Applications
- **Centralized Configuration Integration**: All CLI apps (summarizer, extractor, judge) now use centralized config
  - Apps respect `abstractcore --set-app-default` configuration
  - Fallback to global defaults when app-specific config not set
  - Enhanced `--debug` mode for all applications

- **Vision Configuration CLI**: New `abstractcore/cli/vision_config.py` for vision fallback setup
  - Interactive configuration wizard
  - Model download commands
  - Status checking and validation

#### Documentation
- **Centralized Configuration**: Created `docs/centralized-config.md` with complete configuration system documentation
  - All available commands with examples
  - Configuration file format and priority system
  - Troubleshooting guide and common tasks

- **Media Handling System**: Comprehensive `docs/media-handling-system.md` with production-tested examples
  - "How It Works Behind the Scenes" section explaining multi-layer architecture
  - Provider-specific formatting documentation (OpenAI JSON, Anthropic Messages API)
  - Real-world CLI usage examples with verified working commands
  - Model compatibility matrix and resolution limits

- **Server Documentation**: Updated `docs/server.md` with `/v1/responses` endpoint details
  - OpenAI Responses API format examples
  - File attachment workflows
  - Streaming configuration
  - Media processing capabilities

### Technical

#### Architecture Improvements
- **Provider Registry Enhancement**: Leverages centralized provider registry for model discovery
  - `/providers` endpoint returns complete provider metadata
  - No hardcoded provider lists - all dynamic discovery
  - Registry version 2.0 indicators in API responses

- **Message Preprocessing**: New `MessagePreprocessor` for `@filename` syntax in CLI
  - Extracts file attachments from text
  - Validates file existence
  - Cleans text for LLM processing

- **Media Type Detection**: Intelligent file type detection and processor selection
  - AutoMediaHandler coordinates specialized processors
  - ImageProcessor, PDFProcessor, OfficeProcessor, TextProcessor
  - Graceful fallback ensures processing never fails completely

#### Test Coverage
- **Media Examples**: Added comprehensive test assets in `tests/media_examples/`
  - PDF reports, Office documents, spreadsheets, presentations
  - CSV/TSV data files with various encodings
  - Image examples with metadata

- **Server Testing**: Enhanced test suite for media processing and OpenAI compatibility
  - Real file processing tests (not mocked)
  - Cross-provider media handling verification
  - Streaming with media attachments

### Breaking Changes
None. All changes maintain full backward compatibility with version 2.4.x.

### Migration Guide

#### For Server Users
The `/v1/responses` endpoint now accepts both OpenAI's `input` format and our legacy `messages` format:

**OpenAI Responses API Format (Recommended):**
```json
{
  "model": "gpt-4o",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Analyze this document"},
        {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
      ]
    }
  ],
  "stream": false
}
```

**Legacy Format (Still Supported):**
```json
{
  "model": "openai/gpt-4",
  "messages": [
    {"role": "user", "content": "Tell me a story"}
  ],
  "stream": false
}
```

**Note**: Streaming is now opt-in (set `"stream": true`) instead of automatic, matching OpenAI's behavior.

#### For Configuration Users
New centralized configuration system available:

```bash
# Set global default model
abstractcore --set-global-default ollama/llama3:8b

# Set app-specific defaults
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default extractor ollama qwen3:4b-instruct

# Configure logging
abstractcore --set-console-log-level WARNING
abstractcore --enable-file-logging

# Check current configuration
abstractcore --status
```

Configuration is stored in `~/.abstractcore/config/abstractcore.json` and respects priority:
1. Explicit parameters (highest priority)
2. App-specific configuration
3. Global configuration
4. Hardcoded defaults (lowest priority)

#### For Media Processing Users
Media processing now supports explicit file types:

**CLI (Using @filename syntax):**
```bash
python -m abstractcore.utils.cli --prompt "Analyze @report.pdf and @chart.png"
```

**Python API:**
```python
response = llm.generate(
    "Analyze these documents",
    media=["report.pdf", "chart.png", "data.xlsx"]
)
```

**Server API (New type="file"):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this file"},
        {"type": "file", "file_url": {"url": "https://example.com/doc.pdf"}}
      ]
    }
  ]
}
```

All formats work identically across all providers with automatic format conversion.

### Dependencies Added
- `pymupdf4llm` (0.0.27): LLM-optimized PDF text extraction
- `pymupdf` (1.26.5): Core PDF processing library
- `pydantic` (2.12.3): Request validation and serialization
- `fastapi`: Enhanced with latest features
- `pillow` (12.0.0): Image processing support

### Benefits
- **Users**: Seamless file attachment across all providers with `@filename` CLI syntax and `media=[]` API
- **Developers**: OpenAI-compatible server endpoints with comprehensive media processing
- **Production**: Robust error handling, detailed logging, and graceful degradation
- **Configuration**: Single source of truth for all package-wide preferences and defaults

## [2.4.3] - 2025-10-19

### Fixed
- **Media System Critical Fixes**: Resolved implementation issues preventing full media processing functionality
  - **PDF Processing**: Fixed `output_format` parameter conflict in `PDFProcessor._create_media_content()` call (line 128) causing "got multiple values for keyword argument" error
  - **Office Document Processing**: Fixed element iteration errors in `OfficeProcessor` by replacing `convert_to_dict()` approach with direct element processing for DOCX, XLSX, and PPTX files
  - **Unstructured Library Integration**: Updated office processor to work correctly with current unstructured library API, eliminating "'NarrativeText' object is not iterable" and "'Table' object is not iterable" errors

### Enhanced
- **Production-Ready Media System**: All file types now working perfectly with comprehensive content extraction
  - **PDF Files**: Full text extraction with formatting preservation using PyMuPDF4LLM
  - **Word Documents**: Complete document analysis with structure preservation (DOCX)
  - **Excel Spreadsheets**: Sheet-by-sheet content extraction with intelligent data analysis (XLSX)
  - **PowerPoint Presentations**: Slide content extraction with comprehensive presentation analysis (PPTX)
  - **CSV/TSV Files**: Intelligent data parsing with quality assessment and recommendations
  - **Images**: Seamless vision model integration with existing test infrastructure

- **Server Debug Support**: Comprehensive debug mode for troubleshooting API issues
  - **Command Line Interface**: Added `--debug`, `--host`, and `--port` arguments to server startup with comprehensive help
  - **Enhanced Error Logging**: Detailed 422 validation error reporting with field-level diagnostics and request body capture
  - **Request/Response Tracking**: Full HTTP request logging with client information, timing metrics, and structured JSON output
  - **Centralized Configuration Integration**: Follows centralized config system patterns with environment variable support
  - **Before vs After**: Uninformative "422 Unprocessable Entity" messages now provide actionable field validation details

### Verified
- **CLI Integration**: Confirmed `@filename` syntax works flawlessly across all file types
  - Tested with real files: PDF reports, Office documents, spreadsheets, presentations, data files, and images
  - Cross-provider compatibility verified with OpenAI, Anthropic, and LMStudio providers
  - All examples documented in `docs/media-handling-system.md` are production-tested and working

### Documentation
- **Comprehensive Media System Documentation**: Completely rewrote `docs/media-handling-system.md` to reflect actual implementation
  - Added detailed "How It Works Behind the Scenes" section explaining the multi-layer architecture
  - Documented provider-specific formatting (OpenAI JSON, Anthropic Messages API, local text embedding)
  - Added real-world CLI usage examples with verified working commands
  - Included cross-provider workflow diagrams and error handling strategies
- **Architecture Documentation**: Updated `docs/architecture.md` with comprehensive media system architecture section
  - Added media processing workflow diagrams and component descriptions
  - Documented graceful fallback strategy and provider-specific formatting
  - Included unified media API documentation and CLI integration details

### Technical
- **Robust Error Handling**: Multi-level fallback strategy ensures users always get meaningful results
  - Advanced processing with specialized libraries (PyMuPDF4LLM, Unstructured)
  - Basic processing fallbacks for text extraction
  - Metadata-only fallbacks when all else fails
  - System never crashes or fails completely
- **Test Infrastructure**: Leveraged existing `tests/vision_examples/` with production-quality test assets
  - 5 high-quality images with comprehensive JSON metadata for validation
  - Real-world testing with actual provider APIs and file processing

### Benefits
- **Users**: Can immediately attach any file type using `@filename` syntax with excellent analysis results
- **Developers**: Universal `media=[]` parameter works identically across all providers
- **Production**: Reliable media processing with comprehensive error handling and graceful degradation
- **CLI**: Simple file attachment workflow that works with all supported file formats

## [2.4.2] - 2025-10-16

### Added
- **Centralized Provider Registry System**: Unified provider discovery and metadata management
  - **Single Source of Truth**: Created `abstractcore/providers/registry.py` with `ProviderRegistry` class for centralized provider management
  - **Package-wide Discovery Function**: `get_all_providers_with_models()` provides unified access to ALL providers with complete metadata
  - **Complete Model Lists**: Fixed truncation issue - now returns all models without "... and X more" truncation
  - **Rich Metadata**: Installation instructions, features, authentication requirements, supported capabilities automatically available
  - **HTTP API Integration**: Server `/providers` endpoint now uses centralized registry (registry_version: "2.0")
  - **Dynamic Discovery**: Automatically discovers providers without hardcoding, eliminating manual synchronization

### Enhanced
- **Factory System**: Simplified `create_llm()` from 70+ line if/elif chain to single registry call while maintaining full backward compatibility
- **Server Endpoints**: Enhanced `/providers` endpoint with comprehensive metadata including model counts, features, and installation instructions
- **Documentation**: Added "Provider Discovery" section to both `llms.txt` and `llms-full.txt` with Python API and HTTP API examples
- **Error Messages**: Improved error messages with dynamic provider lists from registry

### Fixed
- **Manual Provider Synchronization**: Eliminated need to manually update provider lists across factory.py, server/app.py, and documentation
- **Model List Truncation**: Fixed "... and X more" truncation - now returns complete model lists for all providers
- **Provider Metadata Inconsistency**: Centralized all provider information including features, authentication requirements, and installation extras

### Technical
- **Comprehensive Test Suite**: Added 50 tests in `tests/provider_registry/` covering core functionality, server integration, and factory integration
- **Lazy Loading**: Provider classes loaded on-demand for better performance and memory usage
- **Backward Compatibility**: All existing code continues to work unchanged - no breaking changes
- **Extensible Architecture**: Easy to add new providers by registering them in the centralized registry

### Benefits
- **Developers**: Single function to discover all providers programmatically
- **Server Users**: Enhanced `/providers` endpoint with rich metadata
- **Maintainers**: No more manual provider list synchronization across multiple files
- **Documentation**: Always up-to-date provider information in docs

## [2.4.1] - 2025-10-16

### Fixed
- **Critical Package Distribution Fix**: Fixed `ModuleNotFoundError: No module named 'abstractcore.exceptions'` that occurred when installing from PyPI
  - Added missing `abstractcore.exceptions` and `abstractcore.media` packages to the setuptools configuration in `pyproject.toml`
  - This issue was introduced during the refactoring process when these modules were not included in the package distribution list
  - Users can now successfully import `from abstractcore import create_llm` after installing from PyPI
  - Verified fix by building and testing the wheel package with the corrected configuration

## [2.4.0] - 2025-10-15

### Breaking Changes
- **Complete Rebranding**: Comprehensive rename from "AbstractLLM" to "AbstractCore" throughout the entire project
  - **Package Name**: Internal package `abstractllm/` → `abstractcore/` to align with published package name
  - **Product Name**: "AbstractLLM Core" → "AbstractCore" in all documentation and branding
  - **Import statements**: All `from abstractcore import ...` must become `from abstractcore import ...`
  - **Console scripts**: Entry points changed from `abstractllm.apps.*` to `abstractcore.apps.*`
  - **Interface names**: `AbstractLLMInterface` → `AbstractCoreInterface`, `AbstractLLMError` → `AbstractCoreError`
  - **Environment variables**: `ABSTRACTLLM_*` → `ABSTRACTCORE_*` (e.g., `ABSTRACTCORE_ONNX_VERBOSE`)
  - **Cache directories**: `~/.abstractllm/` → `~/.abstractcore/`
  - **Log files**: `abstractllm_*.log` → `abstractcore_*.log`
  - **Module paths**: All absolute imports updated throughout codebase
  - **Impact**: This affects all users - complete migration required from AbstractLLM to AbstractCore branding
  
### Migration Guide
To migrate from 2.3.x to 2.4.0, update all references to AbstractLLM:

**1. Import Statements:**
```python
# Before (2.3.x)
from abstractcore import create_llm
from abstractllm.processing import BasicSummarizer
from abstractllm.embeddings import EmbeddingManager

# After (2.4.0+)
from abstractcore import create_llm
from abstractcore.processing import BasicSummarizer  
from abstractcore.embeddings import EmbeddingManager
```

**2. Interface Names:**
```python
# Before (2.3.x) 
from abstractllm.core.interface import AbstractLLMInterface

# After (2.4.0+)
from abstractcore.core.interface import AbstractCoreInterface
```

**3. Environment Variables:**
```bash
# Before (2.3.x)
export ABSTRACTLLM_ONNX_VERBOSE=1

# After (2.4.0+)
export ABSTRACTCORE_ONNX_VERBOSE=1
```

**4. Console Scripts:**
Console scripts remain the same (both `summarizer` and `abstractcore-summarizer` work), but internal module paths have changed to `abstractcore.apps.*`.

### Technical
- **Directory Structure**: Renamed main package directory from `abstractllm/` to `abstractcore/`
- **Configuration Updates**: Updated `pyproject.toml` with new package names, console scripts, and version paths
- **Build System**: Cleaned and regenerated all build artifacts with correct package structure
- **Documentation**: Updated all code examples, CLI usage, and module references across documentation
- **Examples**: Updated all example files with new import statements
- **Tests**: Updated all test imports and references throughout test suite

## [2.3.9] - 2025-10-25
### Fixed
- **Timeout Handling**: Comprehensive timeout parameter handling across all providers
  - All providers now properly handle `timeout=None` (infinity) as the default
  - **HuggingFace Provider**: Issues warning when non-None timeout is provided (local models don't support timeouts)
  - **MLX Provider**: Issues warning when non-None timeout is provided (local models don't support timeouts)  
  - **Local Providers**: Accept timeout parameters appropriately
  - **API Providers** (OpenAI, Anthropic, Ollama, LMStudio): Properly pass timeout to HTTP clients
  - Added `_update_http_client_timeout()` method for providers that need to update client timeouts
- Setting timeout default to None (infinity)

## [2.3.8] - 2025-10-25
### Fixed
- Issue with the version

## [2.3.7] - 2025-10-25

### Fixed
- **Syntax Warning**: Fixed invalid escape sequence `\(` in `common_tools.py` docstring example
- **CLI Enhancement**: Added optional focus parameter to `/compact` command for targeted conversation summarization
  - Usage: `/compact [focus]` where focus can be "technical details", "key decisions", etc.
  - Leverages existing `BasicSummarizer` focus functionality for more precise compaction
  - Maintains backward compatibility (no focus = default behavior)

## [2.3.6] - 2025-10-14

### Added
- **Vector Embeddings**: SOTA open-source models with EmbeddingGemma as default, ONNX optimization, multi-provider support (HuggingFace, Ollama, LMStudio)
- **Processing Applications**: BasicSummarizer, BasicExtractor, BasicJudge with CLI tools and structured output
- **GitHub Pages Website**: Professional documentation site with responsive design and provider showcase
- **Unified Streaming Architecture**: Real-time tool call detection and execution across all providers
- **Memory Management**: Provider unload() methods for resource management in constrained environments
- **Session Management**: Complete serialization with analytics (summary, assessment, facts)
- **CLI Enhancements**: Interactive REPL with tool integration, session persistence, and comprehensive help system

### Fixed
- **Critical Tool Compatibility**: Tools + structured output now work together with sequential execution pattern
- **Ollama Endpoint Selection**: Fixed verbose responses by using correct `/api/chat` endpoint
- **Streaming Tool Execution**: Consistent formatting between streaming and non-streaming modes
- **Architecture Detection**: Corrected Qwen3-Next models and universal tool call parsing
- **Session Serialization**: Fixed parameter consistency and tool result integration
- **Timeout Configuration**: Unified timeout management across all components (default: 5 minutes)
- **Package Dependencies**: Made processing module core dependency, fixed installation extras

### Enhanced
- **Multi-Provider Embedding**: Unified API across HuggingFace, Ollama, LMStudio with caching and optimization
- **Tool Call Syntax Rewriting**: Server-side format conversion for agentic CLI compatibility
- **Documentation**: Consolidated and professional tone, comprehensive tool calling guide
- **Token Management**: Helper methods and validation with provider-specific recommendations
- **Test Coverage**: 346+ tests with real models, comprehensive provider testing

### Technical
- **Event System**: Real-time monitoring and observability with OpenTelemetry compatibility
- **Circuit Breakers**: Netflix Hystrix pattern with exponential backoff retry strategy
- **FastAPI Server**: OpenAI-compatible endpoints with comprehensive parameter support
- **Model Discovery**: Heuristic-based filtering and provider-specific routing

## [2.3.5] - 2025-10-14

### Fixed

#### CRITICAL: Tools + Structured Output Compatibility
- **Problem**: AbstractCore's `tools` and `response_model` parameters were mutually exclusive, preventing users from combining function calling with structured output validation
- **Root Cause**: `StructuredOutputHandler` bypassed normal tool execution flow and tried to validate tool call JSON against Pydantic model
- **Solution**: Implemented sequential execution pattern - tools execute first, then structured output uses results as context
- **Impact**: Enables sophisticated LLM applications requiring both function calling and structured output validation
- **Usage**: `llm.generate(tools=[func], response_model=Model, execute_tools=True)` now works seamlessly
- **Limitation**: Streaming not supported in hybrid mode (clear error message provided)

#### Enhanced BaseProvider Interface
- **Added**: `generate()` method to BaseProvider implementing AbstractCoreInterface
- **Fixed**: Proper delegation from `generate()` to `generate_with_telemetry()` with full parameter passthrough
- **Impact**: Ensures consistent API behavior across all provider implementations

### Technical

#### Implementation Details
- Added `_handle_tools_with_structured_output()` method with sequential execution strategy
- Modified `generate_with_telemetry()` to detect and route hybrid requests appropriately
- Enhanced prompt engineering to inject tool execution results into structured output context
- Maintained full backward compatibility for single-mode usage (tools-only or structured-only)

#### Files Modified
- `abstractcore/providers/base.py`: Added hybrid handling logic and generate() method implementation
- Sequential execution: Tool execution → Context enhancement → Structured output generation
- Clean error handling with descriptive messages for unsupported combinations

#### Test Results
✅ Tools-only mode: Works correctly  
✅ Structured output-only mode: Works correctly  
✅ **NEW**: Hybrid mode (tools + structured output): Now works correctly  
✅ Backward compatibility: All existing functionality preserved  
✅ Error handling: Clear messages for unsupported streaming + hybrid combination

## [2.3.4] - 2025-10-14

### Added

#### State-of-the-Art GitHub Pages Website
- **Professional Website**: Created comprehensive GitHub Pages website at `https://lpalbou.github.io/AbstractCore/`
- **Modern UI/UX**: Responsive design with dark/light theme toggle, smooth animations, and mobile-first approach
- **Interactive Features**: Code block copy functionality, smooth scrolling navigation, and dynamic theme switching
- **Provider Showcase**: Visual display of all supported LLM providers (OpenAI, Anthropic, Ollama, MLX, LMStudio, HuggingFace)
- **SEO Optimization**: Complete sitemap.xml, robots.txt, and meta tags for search engine visibility
- **LLM Integration**: Added `llms.txt` and `llms-full.txt` files for enhanced LLM compatibility and content discovery

#### Comprehensive Tool Calling Documentation
- **New Documentation**: Created `docs/tool-calling.md` with complete coverage of the tool calling system
- **Rich Decorator Examples**: Documented the full capabilities of the `@tool` decorator including metadata injection
- **Architecture-Aware Formatting**: Explained how tool definitions adapt to different model architectures (Qwen, LLaMA, Gemma)
- **Tool Syntax Rewriting**: Integrated comprehensive documentation of Tag Rewriter and Syntax Rewriter systems
- **Real-World Examples**: Showcased actual tools from `common_tools.py` with full metadata and system prompt integration

### Enhanced

#### Documentation Consolidation and Cleanup
- **Professional Tone**: Removed pretentious language, excessive emojis, and marketing hype from all documentation
- **Consolidated Content**: Merged `tool-syntax-rewriting.md` into comprehensive `tool-calling.md` documentation
- **Fixed Cross-References**: Updated all internal links in README.md, docs/README.md, and getting-started.md
- **Consistent Styling**: Standardized documentation format and removed redundant content
- **HTML Documentation**: Created HTML versions of all documentation for the GitHub Pages website

#### Website Architecture
- **Static Site Generation**: Pure HTML/CSS/JavaScript implementation for maximum performance and compatibility
- **Asset Organization**: Structured asset directory with optimized SVG logos and provider icons
- **GitHub Pages Optimization**: Added `.nojekyll` file and proper CNAME configuration for custom domains
- **Documentation Integration**: Seamless integration between website and documentation with consistent navigation

### Technical

#### Files Added
- `index.html`: Main landing page with hero section, features showcase, and provider display
- `assets/css/main.css`: Comprehensive styling with CSS variables for theming and responsive design
- `assets/js/main.js`: Interactive functionality including theme switching and mobile navigation
- `llms.txt`: Concise LLM-friendly project overview with key documentation links
- `llms-full.txt`: Complete documentation content aggregated for LLM consumption
- `docs/tool-calling.html`: HTML version of comprehensive tool calling documentation
- `robots.txt` and `sitemap.xml`: SEO optimization files for search engine discovery

#### Documentation Updates
- Enhanced `docs/tool-calling.md` with complete `@tool` decorator capabilities and real-world examples
- Updated README.md, docs/README.md, and docs/getting-started.md with professional tone and correct links
- Removed redundant `docs/tool-syntax-rewriting.md` after content integration
- Fixed all cross-references and internal navigation links

#### GitHub Pages Deployment
- Created clean `gh-pages` branch with optimized website content
- Implemented proper GitHub Pages configuration with SEO optimization
- Added comprehensive LLM compatibility files for enhanced discoverability
- Structured deployment ready for custom domain configuration

### Impact
- **Enhanced Developer Experience**: Professional website provides clear project overview and easy navigation
- **Improved Documentation Quality**: Consolidated, professional documentation without redundancy or pretentious language
- **Better LLM Integration**: Structured `llms.txt` files enable better LLM understanding and interaction with the project
- **Increased Discoverability**: SEO-optimized website improves project visibility and accessibility
- **Comprehensive Tool Documentation**: Complete coverage of tool calling system with practical examples and architecture details

## [2.3.3] - 2025-10-14

### Fixed

#### ONNX Runtime Warning Suppression
- **Problem**: ONNX Runtime displayed verbose CoreML execution provider warnings on macOS during embedding model initialization
- **Root Cause**: ONNX Runtime logs informational messages about CoreML partitioning and node assignment directly to stderr, bypassing Python's warning system
- **Solution**: Added ONNX Runtime log level configuration in `_suppress_onnx_warnings()` to suppress harmless informational messages
- **Impact**: Cleaner console output during embedding operations while preserving debugging capability via `ABSTRACTLLM_ONNX_VERBOSE=1` environment variable
- **Technical**: Set `onnxruntime.set_default_logger_severity(3)` to suppress warnings that don't affect performance or quality

## [2.3.2] - 2025-10-14

### Fixed

#### Critical Ollama Endpoint Selection Bug
- **Problem**: Ollama provider was generating excessively verbose responses (1000+ characters for simple questions like "What is 2+2?")
- **Root Cause**: Provider incorrectly used `/api/generate` endpoint for all requests, including tool-enabled conversations
- **Solution**: Updated endpoint selection logic to use `/api/chat` by default, following Ollama's API design recommendations
- **Impact**: Reduced response length from 977+ characters to 15 characters for simple queries, eliminated "infinite text" generation issue
- **Technical**: Modified `_generate_internal()` method to use `use_chat_format = tools is not None or messages is not None or True` for proper endpoint routing

#### Session Serialization Parameter Consistency
- **Problem**: Inconsistent parameter naming between `session.add_message()` using `name` and `session.generate()` using `username`
- **Root Cause**: Parameter standardization was incomplete during metadata redesign
- **Solution**: Standardized both methods to use `name` parameter, aligning with `session_schema.json` specification
- **Impact**: Consistent API across session methods, improved developer experience

#### Tool Execution Results in Live Sessions
- **Problem**: Tool execution results were missing from chat history during live CLI sessions but appeared after session reload
- **Root Cause**: Tool results were not being added to session message history during execution
- **Solution**: Modified `_execute_tool_calls()` in CLI to explicitly add `role="tool"` messages with execution metadata
- **Impact**: Tool results now immediately available to assistant during conversation, consistent behavior between live and serialized sessions

#### Common Tools Defensive Programming
- **Problem**: `list_files` and `search_files` tools failed with type errors when `head_limit` parameter was passed as string
- **Root Cause**: LLM-generated tool calls sometimes provided numeric parameters as strings
- **Solution**: Added defensive type conversion with fallback to default values on `ValueError`
- **Impact**: Improved tool reliability and error handling

### Enhanced

#### Comprehensive Session Management System
- **Session Serialization**: Complete session state preservation including provider, model, parameters, system prompt, tool registry, and conversation history
- **Optional Analytics**: Added `generate_summary()`, `generate_assessment()`, and `extract_facts()` methods for session-level insights
- **Versioned Schema**: Implemented `session-archive/v1` format with JSON schema validation in `abstractcore/assets/session_schema.json`
- **CLI Integration**: Added `/save <file> [--summary] [--assessment] [--facts]` and `/load <file>` commands with optional analytics generation
- **Backward Compatibility**: Graceful handling of legacy session formats during load operations

#### Enhanced CLI User Experience
- **Improved Help System**: Comprehensive, aesthetically pleasing help text with detailed command documentation and usage examples
- **Tool Integration**: Added `search_files` tool to CLI with full documentation and status reporting
- **Better Banner**: Informative startup banner with quick commands and available tools overview
- **Parameter Documentation**: Clear documentation of `/save` command options and usage patterns

#### Metadata System Redesign
- **Extensible Metadata**: Moved `name` field into `metadata` dictionary for better extensibility
- **Location Support**: Added `location` property backed by `metadata['location']` for geographical context
- **Property-Based Access**: Clean API with `message.name` and `message.location` properties while maintaining metadata flexibility
- **Backward Compatibility**: Automatic migration of legacy `name` field to `metadata['name']` during deserialization

### Technical

#### Files Modified
- `abstractcore/providers/ollama_provider.py`: Fixed endpoint selection logic to use `/api/chat` by default
- `abstractcore/core/session.py`: Enhanced serialization, standardized parameter naming, added analytics methods
- `abstractcore/core/types.py`: Redesigned metadata system with property-based access
- `abstractcore/utils/cli.py`: Improved help system, added tool integration, enhanced save/load commands
- `abstractcore/tools/common_tools.py`: Added defensive programming for parameter type handling
- `abstractcore/assets/session_schema.json`: Created comprehensive JSON schema for session validation
- `docs/session.md`: New documentation explaining session management and serialization benefits

#### Test Results
✅ Ollama responses now concise (15 chars vs 977+ chars previously)  
✅ Session serialization preserves complete state including analytics  
✅ Tool execution results properly integrated into live chat history  
✅ Parameter consistency across all session methods  
✅ Defensive tool parameter handling prevents type errors  
✅ Backward compatibility maintained for existing session files

## [2.3.0] - 2025-10-12

### Major Changes

#### Server Simplification and Enhancement
- Simplified server implementation in `abstractcore/server/app.py` (reduced from ~4000 to ~1500 lines)
- Removed complex model discovery in favor of direct provider queries
- Added comprehensive endpoint documentation with OpenAI-style descriptions
- Enhanced request/response models with detailed parameter descriptions and examples

#### Multi-Provider Embedding Support
- `EmbeddingManager` now supports three providers: HuggingFace, Ollama, and LMStudio
- Unified embedding API across all providers with automatic format conversion
- Provider-specific caching for isolation and performance
- Backward compatible with existing HuggingFace-only code (default provider)

#### Tool Call Syntax Rewriting
- Added `syntax_rewriter.py` for server-side tool call format conversion
- Supports multiple formats: OpenAI, Codex, Qwen3, LLaMA3, Gemma, XML
- Automatic format detection based on headers, user-agent, and model name
- Enables seamless integration with agentic CLIs (Codex, Crush, Gemini CLI)

#### Model Discovery and Filtering
- Added `/v1/models?type=text-embedding` endpoint for filtering embedding models
- Heuristic-based model type detection (embedding vs text-generation)
- Embedding patterns: "embed", "all-minilm", "bert-", "-bert", "bge-", "gte-", etc.
- Provider-specific model filtering via query parameters

### Server Enhancements

#### API Endpoints
- Enhanced `/v1/embeddings` endpoint with multi-provider support
- Added `type` parameter to `/v1/models` for model type filtering (text-generation/text-embedding)
- Improved `/v1/chat/completions` with comprehensive parameter documentation
- Added `/{provider}/v1/chat/completions` for provider-specific requests
- Enhanced `/v1/responses` endpoint for agentic CLI compatibility
- Updated `/providers` endpoint with detailed provider information

#### Request/Response Models
- Added detailed field descriptions and examples to all Pydantic models
- `EmbeddingRequest`: Comprehensive parameter explanations using OpenAI reference style
- `ChatCompletionRequest`: Enhanced with field-level documentation and examples
- `ChatMessage`: Detailed role and content descriptions with use cases
- Default examples updated to use working models

#### Format Conversion
- Automatic tool call format conversion for different agentic CLIs
- Support for custom tool call tags via `agent_format` parameter
- Configurable tool execution (server-side vs client-side)
- Environment variable configuration for default formats

### Core Library Improvements

#### Embeddings
- Provider parameter added to `EmbeddingManager.__init__()` (default: "huggingface")
- `embed()` and `embed_batch()` methods now delegate to provider-specific implementations
- Ollama provider: Added `embed()` method using `/api/embeddings` endpoint
- LMStudio provider: Added `embed()` method using `/v1/embeddings` endpoint
- Cache naming includes provider for proper isolation

#### Providers
- Enhanced provider base classes with improved error handling
- Better streaming support across all providers
- Consistent timeout handling and retry logic
- Improved tool call detection and parsing

#### Exception Handling
- Added `UnsupportedProviderError` for better error messages
- Enhanced exception types for embedding-specific errors
- Improved error context and debugging information

### Documentation Overhaul

#### Consolidated Documentation
- Merged `common-mistakes.md` into `troubleshooting.md` with cross-references
- Merged `server-api-reference.md` into simplified `server.md` (1006 → 479 lines)
- Created comprehensive `docs/README.md` as navigation hub
- Removed redundant documentation files (8 files consolidated)

#### New Documentation
- Created `tool-syntax-rewriting.md` covering both tag and syntax rewriters
- Enhanced `embeddings.md` with multi-provider support and examples
- Updated `architecture.md` with server architecture and present-tense language
- Improved `getting-started.md` with comprehensive tool documentation

#### Documentation Organization
- Moved `basic-*.md` files to `docs/apps/` subdirectory
- Created `docs/archive/` for superseded documentation
- Added `docs/archive/README.md` explaining archived content
- Updated all cross-references across documentation

#### Documentation Style
- Removed historical/refactoring language ("replaced", "improved", "before/after")
- Converted all documentation to present tense
- Focused on current capabilities and actionable content
- Simplified language for clarity and accessibility

#### Root README Updates
- Added clearer distinction between core library and optional server
- Enhanced documentation section with better organization
- Added "Architecture & Advanced" section
- Improved Quick Links with comprehensive navigation

### Technical Improvements

#### Code Quality
- Removed unused `simple_model_discovery.py` module
- Cleaned up temporary debug files and scripts
- Removed integration.py tool module (functionality moved to providers)
- Better separation of concerns between core and server

#### Testing
- Added comprehensive tests for embedding providers
- Enhanced server endpoint testing
- Improved tool call syntax rewriting tests
- Better test coverage for multi-provider scenarios

### Breaking Changes
None. All changes are backward compatible with version 2.2.x.

### Migration Guide

#### For Embedding Users
If you were using embeddings, no changes needed. The default behavior remains HuggingFace.

To use other providers:
```python
from abstractcore.embeddings import EmbeddingManager

# HuggingFace (default, unchanged)
embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

# Ollama (new)
embedder = EmbeddingManager(model="granite-embedding:278m", provider="ollama")

# LMStudio (new)
embedder = EmbeddingManager(model="text-embedding-all-minilm-l6-v2-embedding", provider="lmstudio")
```

#### For Server Users
Server API endpoints remain compatible. New features:
- Use `?type=text-embedding` to filter embedding models
- Use `agent_format` parameter for custom tool call formats
- Environment variables for default configuration

#### For Documentation Users
- Use `docs/server.md` instead of `server-api-reference.md`
- Use `docs/troubleshooting.md` for all troubleshooting (includes common mistakes)
- Use `docs/README.md` as navigation hub
- Reference `prerequisites.md` instead of deleted `providers.md`

## [2.2.4] - 2025-10-10

### Fixed
- **ONNX Optimization and Warning Management**: Improved embedding performance and user experience
  - **Smart ONNX Model Selection**: EmbeddingManager now automatically selects optimized `model_O3.onnx` for better performance
  - **Warning Suppression**: Eliminated harmless warnings from PyTorch 2.8+ and sentence-transformers during model loading
  - **Graceful Fallbacks**: Multiple fallback layers ensure reliability (optimized ONNX → basic ONNX → PyTorch)
  - **Performance Improvement**: ONNX optimization provides significant speedup for batch embedding operations
  - **Clean Implementation**: Conservative approach with minimal code changes (40 lines) for maintainability

### Technical
- Added `_suppress_onnx_warnings()` context manager to handle known harmless warnings
- Added `_get_optimal_onnx_model()` function for intelligent ONNX variant selection
- Enhanced `_load_model()` with multi-layer fallback strategy and clear logging
- Zero breaking changes - all improvements are additive with sensible defaults

## [2.2.3] - 2025-10-10

### Fixed
- **Installation Package [all] Extra**: Fixed `pip install abstractcore[all]` to truly install ALL modules
  - **Issue**: The `[all]` extra was missing development dependencies (dev, test, docs)
  - **Solution**: Updated `[all]` extra to include complete dependency set (12 total extras)
  - **Coverage**: Now includes all providers, features, and development tools
    - **All Providers** (6): openai, anthropic, ollama, lmstudio, huggingface, mlx
    - **All Features** (3): embeddings, processing, server
    - **All Development** (3): dev, test, docs
  - **Impact**: Users can now confidently use `abstractcore[all]` for complete installation without missing dependencies

### Technical
- **Comprehensive Installation**: `pip install abstractcore[all]` now installs 12 dependency groups
- **Development Ready**: Includes all testing frameworks (pytest-cov, responses), code tools (black, mypy, ruff), and documentation tools (mkdocs)
- **Verified Configuration**: All referenced extras exist and are properly defined with no circular dependencies

## [2.2.2] - 2025-10-10

### Added
- **LLM-as-a-Judge**: Production-ready objective evaluation with structured assessments
  - **BasicJudge** class for critical assessment with constructive skepticism
  - **Multiple file support** with sequential processing to avoid context overflow
  - **Global assessment synthesis** for multi-file evaluations (appears first, followed by individual file results)
  - **Enhanced assessment structure** with judge summary, source reference, and optional criteria details
  - **9 evaluation criteria**: clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence
  - **CLI with simple command**: `judge file1.py file2.py --context="code review"` (console script entry point)
  - **Flexible output formats**: JSON, plain text, YAML with structured scoring (1-5 scale)
  - **Optional global assessment control**: `--exclude-global` flag for original list behavior

### Enhanced
- **Built-in Applications**: BasicJudge added to production-ready application suite
  - **Structured output integration** with Pydantic validation and FeedbackRetry for validation error recovery
  - **Chain-of-thought reasoning** for transparent evaluation with low temperature (0.1) for consistency
  - **Custom criteria support** and reference-based evaluation for specialized assessment needs
  - **Comprehensive error handling** with graceful fallbacks and detailed diagnostics

### Documentation
- **Complete BasicJudge documentation**: Enhanced `docs/basic-judge.md` with API reference, examples, and best practices
  - **Real-world examples**: Code review, documentation assessment, academic writing evaluation, multiple file scenarios
  - **CLI parameter documentation** with practical usage patterns and advanced options
  - **Global assessment examples** showing synthesis of multiple file evaluations
- **Updated README.md**: Added BasicJudge to built-in applications with 30-second examples
- **Internal CLI integration**: Added `/judge` command for conversation quality evaluation with detailed feedback

### Technical
- **Context overflow prevention**: Optimized global assessment prompts to work within model context limits
- **Production-grade architecture**: Proper Pydantic integration, sequential file processing, backward compatibility
- **Console script integration**: Simple `judge` command available after package installation (matches `extractor`, `summarizer`)
- **Full backward compatibility**: All existing functionality preserved, optional features clearly marked

## [2.2.1] - 2025-10-10

### Enhanced
- **Timeout Configuration**: Unified timeout management across all components
  - Updated default HTTP timeout from 180s to 300s (5 minutes) for better reliability with large models
  - All providers now consistently inherit timeout from base configuration
  - Server endpoints updated to use unified 5-minute default
  - Improved handling of large language models (36B+ parameters) that require longer processing time

- **Extractor CLI Improvements**: Enhanced command-line interface for knowledge graph extraction
  - Added `--timeout` parameter with proper validation (30s minimum, 2 hours maximum)
  - Users can now configure timeout for large documents and models: `--timeout 3600` for 60 minutes
  - Improved error messages for timeout validation
  - Better support for processing large documents with resource-intensive models

### Fixed
- **BasicExtractor JSON-LD Consistency**: Resolved structural inconsistencies in knowledge graph output
  - Fixed JSON-LD reference normalization where some providers generated string references instead of proper object format
  - Corrected refinement prompt to match initial extraction format exactly (`@type: "s:Relationship"` vs `@type: "r:provides"`)
  - Added missing `s:name` and `strength` fields in relationship refinement
  - All providers now generate consistent, properly structured JSON-LD output

- **Cross-Provider Compatibility**: Improved extraction reliability across different LLM providers
  - LMStudio models now generate proper JSON-LD object references through automatic normalization
  - Reduced warning noise by converting normalization messages to debug level
  - Enhanced iterative refinement to follow exact same structure rules as initial extraction

### Technical
- **Centralized Timeout Management**: All timeout configuration now emanates from `base.py`
  - Providers inherit timeout via `self._timeout` from BaseProvider class
  - Factory system properly propagates timeout parameters through `**kwargs`
  - No hardcoded timeout values remain in provider implementations
  - Consistent 300-second default across HTTP clients, tool execution, and embeddings

### Documentation
- **Updated Model References**: Modernized documentation to use current recommended models
  - Updated `docs/getting-started.md` to use `qwen3:4b-instruct-2507-q4_K_M` (default) and `qwen3-coder:30b` (premium)
  - Replaced outdated `qwen2.5-coder:7b` references throughout getting started guide
  - Added proper cross-references to reorganized documentation (`server.md`, `acore-cli.md`)
  - Enhanced "What's Next?" section with links to universal API server and CLI documentation

- **Cross-Reference Validation**: Verified all documentation links and anchors
  - Confirmed `docs/prerequisites.md` section anchors match README.md references
  - Validated provider setup links point to correct sections (#openai-setup, #anthropic-setup, etc.)
  - Ensured consistent documentation structure across all guides

## Previous Versions

Previous version history is available in the git commit log.
