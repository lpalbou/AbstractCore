# Local models: browse, check fit, download, delete

AbstractCore can show which local models suit your machine, where they are installed, and
fetch or remove them with each engine's own tool. The same data is available from the CLI
(`abstractcore models ...`, with `--json`), from the HTTP server (`/acore/models/*`), and from
Python. AbstractGateway re-exposes the same payloads, so every console shows the same answers.

Related pages: [Engines](engines.md) (installing Ollama, LM Studio, MLX, llama.cpp),
[Centralized Config](centralized-config.md) (which model each capability uses and
`abstractcore models status`), [Memory and model residency](memory-management.md)
(what is loaded right now), [Server](server.md), and
[Troubleshooting](troubleshooting.md#local-model-downloads) for load and download failures.

## What this covers

| Verb | CLI | HTTP | Python |
|---|---|---|---|
| Describe this machine | `abstractcore host profile --json` | `GET /acore/host/profile` | `abstractcore.utils.host_profile.host_profile()` |
| Browse downloadable models | `abstractcore models catalog [--engine X] [--fits] [--tag T] [--hub] --json` | `GET /acore/models/catalog?q=&engine=&fits=&hub=&tag=` | `abstractcore.config.model_catalog.catalog(...)` |
| Search | `abstractcore models search "<query>" [--engine X] [--fits] [--hub] --json` | same route with `q=` | `model_catalog.search(q, ...)` |
| List installed models | `abstractcore models list [--provider X] --json` | `GET /acore/models/installed?provider=` | `model_materializer.list_installed(provider)` |
| Download | `abstractcore models download <provider> <artifact> [--dry-run] [--detach] [--json]` | `POST /acore/models/download` | `host_jobs.start_download_job(...)` / `model_materializer.download(...)` |
| Delete | `abstractcore models delete <provider> <artifact> [--yes] [--dry-run] [--force] [--json]` | `POST /acore/models/delete` | `model_materializer.delete_artifact(...)` |
| Follow background work | `abstractcore models jobs [<job_id>] [--kind K] [--status S] --json`, `abstractcore models cancel <job_id>` | `GET /acore/jobs`, `GET /acore/jobs/{id}`, `POST /acore/jobs/{id}/cancel` | `host_jobs.default_registry()` |
| Check that an installed model answers | `abstractcore models verify <artifact> [--provider P] [--json]` | none | `abstractcore.config.model_verify.verify_inference(...)` |
| Repair missing `refs/main` | `abstractcore models repair-refs [--dry-run] [--cache-dir DIR] [--json]` | none | `model_materializer.repair_hf_refs(apply=...)` |

Exit codes for every verb: `0` success, `1` error, `2` refused (a policy, a delete blocker, or
a destructive action without `--yes`). A refusal still prints its JSON (`status`, `reason`,
`message`, and `delete_blockers` for deletes).

## Artifact references

An **artifact** names the exact weights to fetch, quantization included:

| Provider | Example artifact | Notes |
|---|---|---|
| `ollama` | `qwen3:8b` | An Ollama tag. |
| `lmstudio` | `qwen/qwen3.5-9b@4bit` | LM Studio model id with `@quant`. A bare id lets LM Studio pick its default build. |
| `mlx` | `mlx-community/Qwen3-8B-4bit` | A Hugging Face repo; MLX repos hold one quantization. |
| `huggingface` | `unsloth/Qwen3-8B-GGUF:Q4_K_M` | A GGUF repo plus `:QUANT`. Only the matching `*Q4_K_M*.gguf` files are fetched. Without `:QUANT` the whole repo is downloaded. |
| `mlx-gen`, `supertonic` | `AbstractFramework/flux.2-klein-4b-8bit`, `supertonic-3` | Image and voice starters. |

## The host profile

`abstractcore host profile --json` (`host_profile_v1`) reports:

- `os`, `arch`, `accelerator` (`metal`, `cuda`, `rocm`, `none`), `gpu_name`, `unified_memory`;
- `ram_bytes`, `vram_bytes` (CUDA, from `nvidia-smi`);
- `ceiling_bytes` and `ceiling_source`: the most memory a model can use. On Apple silicon this is
  the `iogpu.wired_limit_mb` sysctl when set (`metal_wired_limit`), else Metal's recommended
  working set (`metal_recommended`, needs `mlx`), else 75% of RAM (`ram_75pct`). On NVIDIA it is
  total VRAM (`cuda_total`). This is the same ceiling the context estimator uses;
- `free_now_bytes`: memory available right now (available RAM capped at the ceiling; free VRAM on CUDA);
- `disk`: free space for the Hugging Face cache, the LM Studio models folder and the Ollama store.

Values that cannot be measured are `null`.

## The catalog

`abstractcore models catalog --json` returns `model_catalog_v1`: the host profile plus one row
per model. The curated seed ships with AbstractCore
(`abstractcore/assets/model_downloads_catalog.json`, schema alongside) and covers chat, coding,
vision and embedding models across Qwen, Llama, Gemma, Mistral, DeepSeek, GPT-OSS, Phi,
SmolLM, Granite, GLM and common embedding families, plus the recommended fresh-install set
(the recommended text model below, FLUX.2 klein 4B, Supertonic 3). Where an 8-bit build exists
upstream, a row lists it next to the 4-bit one: MLX `-8bit` repos, GGUF `Q8_0` files in the same
repos as the `Q4_K_M` ones, and Ollama `-q8_0` tags.

Each row carries `capabilities` (text, vision, audio, tools, thinking, max_tokens, embedding)
joined from AbstractCore's model capability registry, and one entry per artifact with:

- `presence.status`: `installed`, `absent`, `unknown` (the engine could not be asked), or
  `not_applicable` (remote providers);
- `quant` (the artifact's own quantization label, lowercased, or `null`), `bits` (effective bits
  per weight used by the fit estimate), `quant_class` and `quant_class_source` (see below);
- `options`: route options a recommendation copies with this artifact (the MTP builds carry
  `speculation`); `{}` for most artifacts;
- `companions`: repos downloaded together with this artifact (`[]` for most): an MLX build's MTP
  drafter, as the MLX drafter registry names it (`speculation.mlx_companion_repos`, the same
  answer the download job and the provider use; the seed holds no list of its own). For example
  both `mlx-community/Qwen3.8-27B-4bit` and `Jundot/Qwen3.8-27B-oQ4e-mtp` list
  `mlx-community/Qwen3.8-27B-MTP-4bit`; Qwen3.8 Flash-Next has its MTP head built in and lists
  none. `companion_bytes` is their verified size (the seed's `companion_sizes`, read from the
  Hugging Face file listing), and a catalog-sized `download_bytes` INCLUDES it, because the
  download fetches both and the drafter stays in memory while the model decodes. `null` means
  no companion, or a companion whose size is not recorded (then it is not included and the fit
  notes say so). An engine-reported size is shown as the engine reports it;
- `note`: one sentence about this build, when there is something to say (the MTP builds);
- `download_bytes` and `size_source`: `catalog` (observed on a real engine store, or read from
  the upstream file listing when the seed records an `upstream` check), `engine`
  (reported by the installed engine), `hf_api` (Hugging Face file listing), `estimate`
  (parameters x bits), or `unknown`;
- `fit` (see below), `downloadable`, `supported_on_host`, and the `cli_download` command;
- `recommended`: exactly one artifact per row is pre-selected for this machine. On Apple silicon
  the three text-tier rows pre-select their tier build (below), and otherwise the order is MLX,
  LM Studio, Ollama, Hugging Face. Elsewhere the curated starter wins when the host can run it,
  then the order is Ollama, LM Studio, Hugging Face. Both prefer artifacts that fit and engines
  that are installed. LM Studio and Ollama builds stay listed and downloadable on a Mac;
- `starter` (per row): part of this machine's recommended fresh-install set. Exactly one text
  row is the starter: the row of the recommended text model.

### The recommended text model

One function decides it for every surface (the catalog flags, the fresh-install defaults,
`abstractcore config apply-recommended`, `abstractcore models download --recommended`, and the
Gateway's first-run guide): `abstractcore.config.model_catalog.recommended_text_model()`.

On Apple silicon it is an MLX build chosen by the computer's unified memory, as the host profile
reports it (`ram_bytes`, in GiB):

| Unified memory | Row | Recommended build |
|---|---|---|
| below 24 GiB | `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-MLX-4bit` |
| 24 GiB to below 128 GiB | `qwen3.8-27b` | `mlx-community/Qwen3.8-27B-4bit` |
| 128 GiB and above | `qwen3.8-flash-next` | `mlx-community/Qwen3.8-Flash-Next-4bit` |

A Mac whose memory cannot be read gets the smallest tier, and the pick says so in `tier`. Every
other computer keeps the portable default, LM Studio `qwen/qwen3.5-9b@4bit`.

Each tier also lists an MTP build (native multi-token prediction: `mlx-works/Qwen3.5-9B-oQ4e-mtp`,
`Jundot/Qwen3.8-27B-oQ4e-mtp`, `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`). The module switch
`MTP_RECOMMENDED` (off) decides whether the tier recommends the MTP build instead of the plain
one. Either way the text route keeps the recommendation's MTP policy
(`speculation = {mode: native_mtp, num_draft_tokens: 2, require_acceleration: false}`).

The pick carries the catalog's fit verdict for this machine (`fit`, `fits`). A tier the estimate
says will not fit stays the recommendation: `fits` is `false` and `warning` says so in one
sentence, which the Gateway's guide shows on the **Chat and text** card. A `tight` verdict fits
(`fits` is `true`) and `warning` says it fits tightly. The pick never moves to another tier on
its own. The pick also carries `companions` and `companion_bytes`, and its fit counts the
drafter's bytes.

The three tier rows carry `kv_geometry`, read from each repo's upstream `config.json`: these
models are hybrids where only one layer in four keeps a per-token KV cache (the others are linear
attention with a fixed-size state), so the estimate counts only the full-attention layers. With
the upstream weight size, a Qwen3.8 Flash-Next 4-bit build needs about 109 GiB (103.9 GiB of
weights, 0.2 GiB of KV cache for 8,192 tokens, 5.2 GiB runtime overhead).

### `quant_class`

A normalized quantization family for filtering, derived from the artifact's own `quant` label:

| `quant` label | `quant_class` |
|---|---|
| `2bit`, `q2_k`, `iq2_*` | `2bit` |
| `3bit`, `q3_k_*`, `iq3_*` | `3bit` |
| `4bit`, `q4_k_m`, `q4_0`, `iq4_xs`, `ud-q4_k_xl`, `mxfp4`, `nvfp4`, `int4`, `oq4e` | `4bit` |
| `5bit`, `q5_*` | `5bit` |
| `6bit`, `q6_k` | `6bit` |
| `8bit`, `q8_0`, `fp8`, `int8` | `8bit` |
| `bf16`, `f16`, `fp16` | `16bit` |
| `f32`, `fp32` | `full` |
| no label, or one the table does not know (`iq1_s`) | `unknown` |

When only effective bits are known, `[N, N+1)` bits is `Nbit` (4.5 bits is `4bit`).

`quant_class_source` says where the class came from:

| `quant_class_source` | Meaning |
|---|---|
| `stated` | The reference names its quant (`q4_k_m`, `@4bit`, a `-8bit` repo). |
| `assumed` | A bare Ollama tag (`qwen3.5:9b`) or LM Studio id (`qwen/qwen3.5-9b`): the class of the build the engine fetches by default (Ollama Q4_K_M, LM Studio 4-bit), the same assumption the fit estimate makes. A console labels it "assumed". |
| `null` | No quant information at all: `quant_class` is `unknown`. |

An assumption can be wrong for a given tag: the Ollama registry lists `qwen3.5:0.8b`,
`qwen3.5:2b` and `qwen3-embedding:0.6b` as Q8_0 builds. `quant` and `bits` keep their meaning.

Filters: `q` (every word must start a word of the id, name, vendor, tags or artifacts),
`--engine` (`ollama`, `lmstudio`, `mlx`, `huggingface`, or `llamacpp` for GGUF artifacts),
`--fits` (keep `fits` and `tight` only), `--tag` (`chat`, `coding`, `vision`, `embedding`, ...).

### Hugging Face enrichment (`--hub`)

The catalog works offline. With `--hub` (`hub=true`), AbstractCore asks the Hugging Face API
for exact file sizes and parameter counts of the Hugging Face-hosted artifacts and, when a
query is given, appends up to 20 `hf_search` rows per lane (MLX repos on Apple silicon, GGUF
repos everywhere; a GGUF search result gets the best available quant, Q4_K_M first).
Answers are cached for 24 hours in `~/.abstractcore/config/cache/hf_hub_catalog.json`. A rate
limit (HTTP 429) or an offline machine is reported in `hub.errors` and the offline answer is
returned. Set `HF_TOKEN` for gated repos and higher rate limits.

## Fit verdicts

`fit` answers "will this run here" before any byte is downloaded:

```
W    weights       exact artifact size, else parameters x bits / 8 x 1.03
KV   KV cache      n x 2 x layers x kv_heads x head_dim x 2 bytes (f16), when the geometry is known;
                   otherwise n x 0.5 MiB x (parameters / 8e9)
O    overhead      max(0.5 GiB, 5% of W)
Ceff usable       ceiling - max(2 GiB, 5% of ceiling)
need = W + KV + O          (n = min(8192, context window))
```

| Verdict | Meaning |
|---|---|
| `fits` | `need <= 0.8 x Ceff` |
| `tight` | `need <= Ceff`: runs, with little headroom |
| `partial_offload` | CUDA only: too big for VRAM, but fits VRAM + 75% of RAM (slower) |
| `too_large` | does not fit |
| `unknown` | no size and no parameter count, or no measurable ceiling |

The block also reports `need_bytes` (W + KV + O), `usable_bytes` (Ceff, the amount the verdict
compares `need_bytes` with), `reserve_bytes` (ceiling minus Ceff), `overhead_bytes` (O),
`ceiling_bytes`, `free_now_bytes`, `fits_now` (whether it
fits in memory that is free right now), `disk_ok` (download size plus 5 GiB headroom against
free disk), `max_context` (largest context beside the weights, capped at the model window),
`confidence` (`exact`, `estimated`, `rough`, `unknown`) and human `notes`. Mixture-of-experts
models are sized by total parameters: every expert is resident. Verdicts are advisory; nothing
blocks a download or a load on them.

Python:

```python
from abstractcore.utils.host_profile import host_profile
from abstractcore.utils.model_fit import estimate_fit

fit = estimate_fit(host=host_profile(), params_total=8_200_000_000, quant="q4_k_m", max_tokens=131072)
print(fit["verdict"], fit["need_bytes"], fit["max_context"])
```

## Installed models

`abstractcore models list --json` returns `models_installed_v1`: one row per installed
artifact with `provider`, `artifact`, `quant`, `size_bytes`, `params_total`, `location`,
`loaded`, `catalog_id`, `deletable` and `delete_blockers`, plus `engines_probed`, per-engine
`errors` and `totals`.

Each row also says what the artifact is and what it can do, read from local evidence only
(no model-card fetch):

- `kind`: `model`, `adapter` (a LoRA or PEFT adapter, which cannot be loaded on its own),
  `encoder` or `embedding`; `null` when the local files cannot tell.
- `tasks`: AbstractCore task names such as `text_generation`, `image_to_text`,
  `text_embedding`, `speech_to_text`, `text_to_image`, `image_to_image`; `[]` when unknown.
- `tasks_source`: where the answer came from: `model_card` (the cached `README.md`
  `pipeline_tag`, or a pipeline name among its tags), `config` (`config.json` architectures),
  `adapter_config`, `files` (sentence-transformers files), `engine` (`lms ls --json` `type` and
  `vision`), or `null`.

The Hugging Face cache holds the model card only when the full repository was downloaded
(mlx-lm's own fetch skips `README.md`), so a repository with no card and no recognisable
`config.json` stays `null`/`[]`. Ollama rows are always `null`/`[]`: telling an embedding
tag from a chat tag needs one `/api/show` call per model, which the listing does not make.

| Engine | Source | Size fields |
|---|---|---|
| Ollama | `GET /api/tags` (`/api/ps` for `loaded`) | `size`, `details.parameter_size`, `details.quantization_level` |
| LM Studio | `lms ls --json` (`lms ps --json` for `loaded`) | `sizeBytes`, `paramsString`, `quantization.name`, `maxContextLength` |
| MLX / Hugging Face | the Hugging Face cache (`scan_cache_dir()`) | repo size on disk; rows split into `mlx` and `huggingface` by AbstractCore's MLX detection |

**Which Hugging Face caches are read.** Every lookup (a load, the provider model lists, the
GGUF search, the ONNX probe of an embedding model, the capability probes) reads the same list
of hub caches: the one `huggingface_hub` itself uses (so a cache you moved for
`huggingface_hub` is honoured), then the `cache.huggingface_cache_dir` setting (`hub/` inside
it), so a relocated cache is found wherever you moved it.

**Where `abstractcore --download-vision-model` writes.** Into the `cache.local_models_cache_dir`
setting (default `~/.abstractcore/models`), one folder per model. Move it with
`abstractcore --set-local-models-cache-dir PATH`.

When the local Ollama server is not running, its tags are listed from the on-disk manifests
(marked `engine_not_running`). An engine that cannot be read appears in `errors`, never as an
empty list.

An MLX row whose model has an MTP companion (see [MTP companions](#mtp-companions)) carries
`companions: [{artifact, role: "mtp_companion", installed, size_bytes, location, companion_of}]`.
A cached companion is listed under its model, not as a row of its own; `totals.size_bytes`
counts it once. A companion whose models are all deleted stays a row, with
`role: "mtp_companion"` and `companion_of: []`, so it can still be found and deleted.

## Downloading

```bash
abstractcore models download ollama qwen3:8b
abstractcore models download lmstudio qwen/qwen3.5-9b@4bit
abstractcore models download huggingface unsloth/Qwen3-8B-GGUF:Q4_K_M
abstractcore models download lmstudio qwen/qwen3.5-9b@4bit --dry-run     # show the command only
```

- Every source reports real progress (bytes, total, percent, speed, time left, per file) from
  the first second; see [Download progress](#download-progress).
- Ollama downloads use `POST /api/pull`; LM Studio uses `lms get <artifact> --yes` and verifies
  afterwards that the requested model landed; Hugging Face (and MLX, mlx-gen) uses
  `snapshot_download` (narrowed by `:QUANT`) with the file list and sizes read from the hub
  first; Supertonic voice files are fetched file by file from the revision AbstractVoice pins.
- A download that cannot fit on disk with 5 GiB to spare is refused before it starts (Hugging
  Face computes the size itself; `expected_bytes` arms the check for other engines).
- `--json` streams one `host_job_v1` object per line while the download runs and ends with the
  final job. SIGTERM or Ctrl-C cancels the download, stops the engine tool, and the last line
  reports `cancelled`. `--detach` starts the download in the background and prints the job;
  follow it with `abstractcore models jobs <job_id> --json`.
- A Hugging Face download is pinned to the commit the hub listed
  (`snapshot_download(revision=<sha>)`), so the files watched are the files fetched.
  huggingface_hub writes no `refs/main` for a commit-hash revision, so once every planned
  file is verified whole, AbstractCore writes `<repo>/refs/main` = that commit itself. Any
  loader that resolves the repo by id offline (transformers with `local_files_only`,
  `mlx_lm.load("<id>")`, vLLM, your own scripts) then finds it. An existing `refs/main`
  naming another commit is never overwritten; the completion message says which commit
  loading by name resolves.

### MTP companions

Several MLX models accelerate with a **separate** MTP head repo, the companion. The registry
(`model_capabilities.json`, `speculation.runtimes.mlx.drafter`) names it, and it is the same
entry the MLX provider loads the head from. Loading never downloads, so AbstractCore treats
the companion as part of the model:

| Model | Companion |
|---|---|
| `mlx-works/Qwen3.5-9B-oQ4e-mtp`, `mlx-community/Qwen3.5-9B-MLX-4bit` | `mlx-community/Qwen3.5-9B-MTP-4bit` (164 MB) |
| `Jundot/Qwen3.8-27B-oQ4e-mtp`, `mlx-community/Qwen3.8-27B-4bit` | `mlx-community/Qwen3.8-27B-MTP-4bit` (270 MB) |
| `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` | none: the head is built into the model's own weights |

- `abstractcore models download mlx <model>`, the Gateway's model downloads and
  `--recommended` fetch the companion in the **same job**, after the model. The job's
  `bytes_total` includes it from the start. Its files are rows of the job's `files`, named
  `<companion repo>/<file>` with `role: "mtp_companion"`, and the result lists it under
  `companions`. A cancel stops both.
- The job ends `done` only when both are whole. When the companion fails, the job fails with
  a message naming it, and `models status` / the probe report the model `absent` with the
  reason until the same download is retried. A retry fetches only what is missing.
- `--dry-run` names the companion and its size.
- `models delete mlx <model>` asks whether to delete the companion too, unless another installed
  model uses it. `--with-companion` deletes it without asking and `--keep-companion` keeps it.
  JSON and Gateway results carry `companion_offer` with the exact delete command. A companion
  can also be deleted by its own name.

### Checking a fresh install: `models verify`

```bash
abstractcore models verify mlx-works/Qwen3.5-9B-oQ4e-mtp
abstractcore models verify mlx-community/Qwen3.8-27B-4bit --json
```

This loads ONE installed model through its provider with the default configuration. It never
downloads. It asks "What is the boiling point of water at sea level in degrees Celsius?" at
temperature 0 and checks that the answer contains `100`. For a model with an MTP companion it
also checks that `speculation.used` is `true`. It unloads the model afterwards, and the exit
code is `0` only when every check passed. Use it after a download to confirm that the model
produces sensible output on this machine, not only that its files are present. `--provider`
selects the provider (default `mlx`).

### Repairing `refs/main`

A cached repo can hold a complete snapshot and no `refs/main`, for example one downloaded by
an AbstractCore release before 2.15.0 (see the [CHANGELOG](https://github.com/lpalbou/AbstractCore/blob/main/CHANGELOG.md)). It loads through AbstractCore
(which resolves the snapshot directory itself) but not by id through other tools, which report
"couldn't find them in the cached files". Write the missing refs once:

```bash
abstractcore models repair-refs --dry-run   # list what would be written
abstractcore models repair-refs             # write it
abstractcore models repair-refs --cache-dir ~/.cache/huggingface/hub --json
```

It scans every Hugging Face cache AbstractCore reads (or only `--cache-dir`, repeatable) and
writes `refs/main` only for a repo with **no** `refs/main` and **exactly one complete**
snapshot. Complete means: no `.incomplete` blob and no unfinished download marker in the repo,
no dangling file in the snapshot, every shard a `*.index.json` names present, and at least a
config or a weight file. Each repo that is not already healthy is listed with a status:

| Status | Meaning | Written? |
|---|---|---|
| `repairable` / `repaired` | one complete snapshot; `refs/main` names it | only without `--dry-run` |
| `ambiguous` | several complete snapshots; which one was `main` cannot be known offline | never |
| `no_complete_snapshot` | every snapshot is partial, README-only, or missing a shard (the reason is listed) | never |
| `dangling_ref` | `refs/main` names a snapshot that is not on disk | never (left as found) |
| `error` | the cache or the ref could not be read or written | exit code `1` |

Healthy repos are counted as `ok`. Quarantined caches (`runtime/model-quarantine/*`) are not
scanned. The command is idempotent.

## Deleting

```bash
abstractcore models delete ollama qwen3:8b --dry-run
abstractcore models delete ollama qwen3:8b --yes
abstractcore models delete lmstudio qwen/qwen3.8-27b@q4_k_m --yes --force
```

| Engine | Mechanism |
|---|---|
| Ollama | `DELETE /api/delete` (with `--force`, a loaded model is unloaded first) |
| LM Studio | `lms unload` when loaded (`--force`), then removal of the model's files under the LM Studio models folder and its hub entry |
| MLX / Hugging Face | `scan_cache_dir().delete_revisions(...).execute()` for the whole repo |

A delete is refused (exit `2`, HTTP `409`) when a blocker applies, unless `--force`:

- `loaded`: the model is in memory;
- `remote_engine`: the Ollama server is on another machine;
- `shared_cache:mlx,huggingface`: the Hugging Face cache entry is classified for the other
  engine (the files are shared).

`unknown_location` (a model bundled inside LM Studio) and `engine_not_running` cannot be
forced. Paths outside the known model stores are never removed.

## Jobs

Downloads, deletes and engine installs are jobs (`host_job_v1`): `job_id`, `kind`
(`download`, `delete`, `engine_install`), `status` (`queued`, `running`, `completed`, `failed`,
`cancelled`), `percent`, `downloaded_bytes`, `total_bytes`, `message`, `log_tail` (last 60
lines), `command`, `dry_run`, `started_at`, `finished_at`, `error`, `joined`,
`cli_equivalent`, and `result` (the verb's own outcome).

- A second request for the same download joins the running job (`joined` counts them).
- `cancelled` follows only a cancel request. `cancelled_by` says who made it (`console`, `api`,
  `cli`, `other_process`) and `cancelled_by_user` the account when known. A job that stopped on
  its own (dropped connection, timeout, Hub error, full disk, failed MTP companion, owning
  process gone) is `failed`. Either way `ended_reason` says why in one plain sentence.
- A stopped Hugging Face download keeps the files that finished; the file that was in progress
  starts over on the next download.
- At most 40 finished jobs are kept per process.

### Download progress

A download job also carries these fields (the web consoles and `--json` read them):

| Field | Meaning |
|---|---|
| `state` | `queued`, `resolving`, `downloading`, `verifying`, `installing`, `done`, `failed`, `cancelled`, `stalled` |
| `bytes_done`, `bytes_total` | bytes on disk so far and the total (same values as `downloaded_bytes`, `total_bytes`) |
| `size_unknown`, `size_note` | `true` only when the source cannot say how big the download is, with the reason |
| `percent` | `bytes_done / bytes_total`, 100 when done |
| `bytes_per_second`, `eta_s` | speed over the last 5 seconds (it falls to 0 when bytes stop) and seconds left |
| `files` | `[{name, bytes_done, bytes_total, state}]`; `current_file` is the file arriving now |
| `message` | one sentence, e.g. `Downloading model.safetensors (2 of 5) · 1.2 GB of 4.8 GB · 38 MB/s · 1 min left` |
| `detail` | the engine tool's own last line, unchanged |
| `updated_at` | refreshed at least every 0.5 s while the job runs |
| `transitions` | every state change with its time and reason |
| `stall_after_s`, `stalled_for_s` | the stall threshold and how long the current stall has lasted |

A download that receives no bytes for 15 seconds turns `stalled` and says so
(`Stalled: no data for 23 s · 1.2 GB of 4.8 GB · still trying…`), is logged, and turns back
to `downloading` by itself when bytes arrive again. Set the threshold with
`ABSTRACTCORE_DOWNLOAD_STALL_S`. `verifying` and `installing` never count as stalls.

What each source reports:

| Source | Progress | Cancel |
|---|---|---|
| Hugging Face, MLX, mlx-gen | file list and sizes from the hub before the first byte; per-file bytes from the files being written in the cache | the transfer runs in a child process that is stopped at once; files already complete are kept and reused by the next download; a download that did not finish never reads as installed |
| Ollama | the layers of `/api/pull`, added up into one total | the connection is closed at once; Ollama keeps the layers it has |
| LM Studio | the bytes, total and speed `lms get` prints | answers `lms get`'s own "continue in the background?" question with No, so LM Studio stops too |
| Supertonic (voice) | per-file bytes, sizes read first | stops at once; the partial file is removed |

When `lms get` prints no progress, the job reports the bytes landing in the LM Studio models
folder instead ("LM Studio reports no progress; 850 MB on disk so far").

An explicit download always reaches the Hub. `offline_first` stops on-demand downloads while
a model is *loading*; it never blocks `abstractcore models download` or a download job. The
download runs in a child process whose `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and
`HF_DATASETS_OFFLINE` are the values set before the process started. A value written later
inside the process (by a provider or a library during a load) is not passed on, and the job
log says so ("explicit download: Hub offline flags set in this process after start are not
passed to the download ..."). If you set `HF_HUB_OFFLINE=1` (or `TRANSFORMERS_OFFLINE=1`)
yourself before starting the process, the download is refused at once with a message naming
the variable. Unset it and restart to download.

Hugging Face downloads started from a job use plain HTTP rather than Xet, because Xet writes
a file only when it is complete, so no progress would show. Set `ABSTRACTCORE_HF_XET=1` to use
Xet anyway; progress then moves one whole file at a time.

Every progress update is also appended to `<jobs dir>/<job_id>.events.jsonl`.

Job snapshots are written to `~/.abstractcore/config/jobs/` (override with
`ABSTRACTCORE_JOBS_DIR`; disable with `ABSTRACTCORE_JOBS_PERSIST=0`), so
`abstractcore models jobs` lists work started by `abstractcore serve` or by
`--detach`, and `abstractcore models cancel <job_id>` can stop it.
