# Planned: give the MLX provider real sight via an mlx-vlm vision add-on encoder

## Metadata
- Created: 2026-08-21
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: `0001` (no silent degradation), `0003` (provider/capability boundaries),
  `0008` (provider-owned truth — the pattern this follows)
- ADR impact: No new ADR required. The design is an application of `0003`'s "optional vision
  behavior stays behind capability plugins" and `0001`'s best-effort-must-be-observable rule.
  If the MRoPE precision gap (below) becomes a standing product statement rather than a
  transitional one, that belongs in an ADR.

## Context

`MLXProvider` loads through `mlx_lm` (`abstractcore/providers/mlx_provider.py`, the
`from mlx_lm import load, generate, stream_generate` in `_load_model`). Upstream
`mlx_lm/models/qwen3_5.py::Model.sanitize` explicitly `continue`s past every weight whose key
starts with `vision_tower` or `model.visual`. The vision tower is therefore never instantiated,
and images have nowhere to go. This is not a gap in AbstractCore's media handling — the media
block already behaves correctly given a blind transport.

Six of the nine `mlx-community` checkpoints cached on the development machine ship a vision
tower (297–333 tensors, ~0.9 GB each): Qwen3.5-4B, Qwen3.6-27B, Qwen3.6-35B-A3B,
Qwen3.8-27B (4bit and 8bit), Ornith-1.0-9B. All are `qwen3_5` / `qwen3_5_moe`. AbstractCore
downloads those weights, stores them, and discards them at load.

Measured on `mlx-community/Qwen3.8-27B-4bit`, same checkpoint and same image:

| lane | `model.parameters()` | answer |
|---|---|---|
| mlx-lm (today) | `['language_model']` | *"there is no image in this conversation"* |
| mlx-vlm | `['language_model', 'vision_tower']` | "A red triangle, a blue circle, and the word SEVEN" |

Full evidence, 22 numbered measurements: `untracked/investigation-vision-mlx/EVIDENCE.md`.
Strategy and rejected alternatives: `untracked/investigation/mlx-native-vision-strategy.md`.

## Current code reality

- `_load_model` binds `self.generate_fn` / `self.stream_generate_fn` to `mlx_lm.generate` /
  `mlx_lm.stream_generate`. Both forward `**kwargs` verbatim into `mlx_lm.generate_step`, which
  accepts `input_embeddings`. **No structural change to the generate path is required.**
- The media block reduces a multimodal message to its text part and records `dropped_media`,
  with an accurate warning. That honesty lane is correct and stays.
- `_build_prompt_fragment` hand-renders ChatML and never calls the tokenizer's chat template, so
  AbstractCore keeps full ownership of rendering, thinking control, and tool transcripts.
- `mlx_vlm` 0.6.3 exposes `merge_input_ids_with_image_features` as a public `@staticmethod` on the
  model class (made public upstream in Blaizzy/mlx-vlm PRs #335 and #355, explicitly for external
  integrators), so no mlx-vlm model instance is needed to merge.

## Problem

The capability registry advertises vision for these checkpoints, the weights are on disk, and the
transport cannot carry a single pixel. `analyze_media` can stamp "there is no image in this
conversation" as an observation with provenance — the exact failure the 2026-08-21 delegated-sight
honesty work was written to expose.

## What we want to do

Keep `mlx_lm` as the decoder, always. Use `mlx_vlm` **only** as a vision encoder plus its public
merge staticmethod, and hand the merged embeddings to the existing generate call via
`input_embeddings`.

```
image ─► mlx_vlm vision_tower ─────► image features
                                          │
prompt ─► AbstractCore _build_prompt ─► input_ids ─► mlx_lm embed_tokens ─► text embeds
                                          │                                      │
                                          └──► Model.merge_input_ids_with_image_features
                                                              │
                                                    merged embeddings
                                                              │
                          self.stream_generate_fn(..., input_embeddings=merged)
```

## Why

Because `self.llm` stays an mlx-lm model, every hazard found while evaluating the alternative
(loading the whole model through mlx-vlm) disappears rather than being managed:

| hazard | under the add-on |
|---|---|
| `make_prompt_cache(wrapper)` yields `KVCache × 64` instead of `ArraysCache × 48 + KVCache × 16` (75 % of layers wrong, silently) | does not arise |
| two `generation_stream` objects + a process-global wired limit → the documented Metal SIGABRT | does not arise; only mlx-lm generates |
| `mlx_lm.generate_step` rejecting mlx-vlm's `LanguageModelOutput` | does not arise |
| text-path drift onto a 2645-line reimplementation (vs mlx-lm's 531) | does not arise; text path untouched |
| `mlx_lm==<ver>` engine fingerprint invalidated across the persisted KV corpus | preserved; mlx-lm still owns the KV |

Measured cost, steady state on the 27B with the decoder resident and never unloaded:

| step | MLX active |
|---|---|
| mlx-lm decoder resident | 14.094 GB |
| `+ mlx_vlm.load(lazy=True)` | 14.094 GB (**+0.000**) |
| `+ mx.eval(vision_tower.parameters())` | 14.953 GB (**+0.859**) |

`lazy=True` never materialises the wrapper's language-model half. And mlx-lm's and mlx-vlm's
`embed_tokens` are bit-identical (`max |Δ| == 0.0` on the 27B), so using mlx-lm's costs nothing
in fidelity.

## Requirements

- An image supplied to a sighted MLX checkpoint reaches the model's forward pass. A question about
  image content is answered from the image.
- The text-only path is **byte-identical** to today: same rendering, same tokenizer call, same
  sampler, same prompt cache, same reasoning and tool behaviour.
- `mlx_vlm` is imported only when an image is actually present on a sighted checkpoint. A
  text-only session must never import it.
- The vision tower is loaded lazily, evaluated alone, and the mlx-vlm language model is **never**
  materialised (evaluating its `embed_tokens` costs +0.672 GB for nothing).
- **No prompt-cache reuse on a turn carrying media** (see Non-goals and `0843`).
- A per-family merge adapter with a load-time `inspect.signature` assertion that refuses the lane
  on an unrecognised shape.
- Failure to encode never produces a silent text answer that claims sight. See `0842`.

## Suggested implementation

1. **New module** `abstractcore/providers/mlx_vision_addon.py` (~120 lines). One public method,
   modelled on `lmstudio-ai/mlx-engine`'s `BaseVisionAddOn.compute_embeddings`:
   `compute_embeddings(text_model, rendered_prompt, images_b64) -> (input_ids, merged_embeddings)`.
   Internally: `prepare_inputs` → `vision_tower(pixel_values, grid_thw)` → **mlx-lm's**
   `embed_tokens(ids)` → the public merge staticmethod.
2. **`_load_model`**: a cheap `config.json`-only check (non-empty `vision_config`,
   `language_model_only` is not `True`) recording `self._declares_vision`. Do **not** load
   mlx-vlm here.
3. **Media block**: when the checkpoint declares vision and images are present, lazily construct
   the add-on (first image only, cached on the instance), prefix the derived vision placeholder
   onto `processed_prompt`, and record positive delivery. On any failure, fall through to the
   existing `dropped_media` path with a named reason.
4. **Placeholder derivation**: read `vision_start_token_id` / `image_token_id` /
   `vision_end_token_id` from `config.json` and `tokenizer.decode` them. Do **not** add a
   registry field — all four sighted local checkpoints derive
   `<|vision_start|><|image_pad|><|vision_end|>` correctly this way.
5. **Generate path**: add a keyword-only `input_embeddings=None` to `_single_generate`,
   `_stream_generate`, and `_stream_generate_with_tools`, and pass it into the existing
   `self.generate_fn(...)` / `self.stream_generate_fn(...)` calls. Nothing else moves.
6. **`unload_model`**: also clear the add-on, the processor, `_outlines_model` (which holds a live
   model reference and currently defeats the unload), the prompt-cache store and hybrid snapshots,
   and call `mx.clear_cache()`.
7. **`pyproject.toml`**: a new opt-in extra
   `mlx-vision = ["mlx-vlm>=0.6.3,<1.0.0"]`, with a comment recording that it forces
   `transformers>=5.5.0` and pulls fastapi / starlette / uvicorn / opencv-python / datasets /
   miniaudio / mlx-audio / llguidance. It must **not** join the `mlx` extra.

Anchor edits on symbols, not line numbers — the numbers in earlier drafts of this investigation
were already stale by the time it finished.

## Scope

- Image input on `qwen3_5` and `qwen3_5_moe` checkpoints, single image per request.
- The seven families where both halves of the contract exist today: `qwen3_5`, `qwen3_5_moe`,
  `qwen3_vl`, `qwen3_vl_moe`, `qwen2_vl`, `pixtral`, `mistral3`. Start with the first two.

## Non-goals

- **`gemma3` / `gemma4`.** Verified: neither exposes a public merge staticmethod
  (`prepare_inputs_for_multimodal` and `get_input_embeddings` respectively). An earlier draft of
  this plan named them; that was wrong.
- **Multi-image and video.** `rope_deltas` accumulates across images and the temporal axis
  collapses under 1-D positions. Upstream has not solved cross-turn image caching either
  (Blaizzy/mlx-vlm#832, lmstudio-ai/mlx-engine#287). Refuse with a named reason for now.
- **Prompt-cache reuse across image-bearing turns** — see `0843`.
- **A second full-mlx-vlm runtime lane / a `vision_precision` knob.** Rejected: a per-request knob
  puts both libraries' `generation_stream` objects in flight against a process-global wired limit,
  re-creating the documented SIGABRT. The escape hatch already exists —
  `provider="huggingface"` runs real MRoPE, and `media/vision_fallback.py` routes to a captioner.
- Audio / omni modalities.
- Porting a monkeypatch of mlx-lm internals. `lmstudio-ai/mlx-engine` carries a 25 KB
  commit-pinned patch of mlx-lm's `DecoderLayer` for this; AbstractCore must not own that.

## Dependencies and related tasks

- `0841` — transport-actual media capability. Independent; either can ship first.
- `0842` — media-delivery honesty. **Should ship before or with this item**: this item's failure
  paths depend on a positive delivery assertion existing.
- `0843` — media-aware prompt-cache reuse (proposed, deferred).
- Upstream `ml-explore/mlx-lm#1768` adds `position_ids` for M-RoPE. **Nothing here depends on it.**
  If it merges, the known precision gap closes by passing one more kwarg.

## Expected outcomes

- `generate("what is in this image?", media=[img])` on a sighted MLX checkpoint answers from the
  image instead of denying one exists.
- Text-only behaviour, prompt-cache behaviour, and reasoning/tool parity all unchanged.
- `pip install abstractcore[mlx]` unchanged; vision is opt-in.

## Validation

- **Sight**: on `mlx-community/Qwen3.8-27B-4bit` with a deterministic synthetic image, the answer
  names the shapes and the embedded word. A blind control (no add-on) must fail the same assertion.
- **Text no-regression**: greedy decode, several prompts, token-identical to the pre-change
  provider on the same checkpoint. Not "similar" — identical.
- **Cache composition**: `make_prompt_cache` on the loaded model yields `ArraysCache × 48 +
  KVCache × 16` for `qwen3_5`, and the text-lane save/load round-trip still passes.
- **Memory**: add-on delta over the resident decoder stays under ~1 GB on the 27B; assert the
  mlx-vlm language model is never evaluated.
- **Import hygiene**: a text-only generation on a sighted checkpoint does not import `mlx_vlm`
  (assert on `sys.modules` in a subprocess).
- **Signature guard**: a fake model class with a wrong merge signature causes the lane to refuse
  at load, not to mis-bind positionally at request time.
- **Known-gap documentation test**: the spatial benchmark is recorded, not asserted green — see
  Guidance.
- Existing suites: `tests/providers/`, `tests/media/`-adjacent capability tests, and the MLX
  prompt-cache tests must be unchanged.

## Progress checklist

- [ ] Land `0842` (positive delivery assertion) or land it alongside
- [ ] `mlx_vision_addon.py` with the single `compute_embeddings` entry point
- [ ] Per-family merge adapter + load-time signature assertion
- [ ] Media-block branch + lazy add-on construction + placeholder derivation
- [ ] `input_embeddings` threaded through the three generate methods
- [ ] Prompt-cache bypass on media-bearing turns
- [ ] `unload_model` clears the add-on, processor, Outlines wrapper, cache store, snapshots
- [ ] `mlx-vision` extra in `pyproject.toml`
- [ ] Tests above, including the text no-regression and import-hygiene checks
- [ ] Document the M-RoPE precision gap in the MLX provider docs

## Guidance for the implementing agent

**Do not add a second generate method.** The single clearest failure signal for this work is a
diff that introduces one. `huggingface_provider.py::_generate_vision_model` is 515 lines in one
method — a parallel `_generate_internal` that re-implements prompt lifting, templating, streaming,
tool handling, usage accounting and error responses, and gives up prompt caching entirely. If this
diff grows a `def ..._generate_vision...`, stop and re-read this item.

**Known precision gap, measured, do not treat as a bug.** mlx-lm substitutes plain 1-D RoPE where
Qwen specifies 3-D M-RoPE (`mlx_lm/models/rope_utils.py` recognises `mrope`, asserts the section
length, then returns `nn.RoPE`). On a controlled comparison — identical weights, identical merged
embeddings, only positions differing — 14 spatial questions scored **13/14 with M-RoPE vs 11/14
with 1-D**, and 1-D never won a case. Both losses were position-sensitive (a spatial relation and
a count). This is a bounded quality degradation, not a failure mode, and under ADR 0001 it must be
**annotated**, not absorbed silently. n=14 on one model shows a direction, not a rate; if you need
to defend a default, run a real spatial/OCR benchmark first.

**The two embeddings are interchangeable but the memory cost is not.** Use mlx-lm's `embed_tokens`.
Calling mlx-vlm's produces identical values and materialises a second 248320×5120 table.
