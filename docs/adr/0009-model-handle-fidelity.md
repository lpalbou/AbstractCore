# ADR 0009: Model handle fidelity

Status: Accepted.

## Context

ADR 0001 forbids silent degradation, but it is written about *behaviour* — fallback prompts,
truncation, timeouts, degraded capabilities — and it closes with "This ADR does not ban
fallbacks." It says nothing about the identity of the artifact a provider loads, and that gap
was read as permission.

The concrete failure: `create_llm(provider="huggingface", model="Qwen/Qwen3.6-27B")` returned a
provider with `model_type == "gguf"`. `HuggingFaceProvider.__init__` promoted any handle to GGUF
whenever a local LM Studio Hub manifest existed and *some* GGUF could be resolved from the
caches. A Hub manifest's `baseModel` dependency routinely names a different repository, so the
call followed `Qwen/Qwen3.6-27B` to `lmstudio-community/Qwen3.6-27B-GGUF` and loaded a Q4_K_M
file on llama.cpp — while the requested bf16 transformers weights sat unused in the HF cache a
directory away. There was no warning and no error.

The substitution changed three things at once: the repository, the numeric precision (bf16 to
4-bit), and the execution engine (transformers to llama.cpp, with Metal offload disabled). Two
separate measurement agents came close to publishing GGUF-on-CPU numbers as "27B bf16
transformers"; a third tripped over `model_instance is None`, because the GGUF lane populates
`self.llm` and leaves `self.model_instance` unset.

This is worse than ordinary degradation. A caller who is told a request was truncated can
compensate. A caller silently handed a different model cannot: every downstream number,
comparison and quality judgement is attributed to the wrong artifact, and nothing in the
returned object contradicts them.

The framework's central promise is that requesting a model runs that model. Model identity is
therefore not a quality-of-service concern subject to best-effort handling; it is a correctness
invariant.

## Decision

**A model handle names an artifact. AbstractCore resolves where that artifact lives; it never
loads a different one. When the requested artifact cannot be loaded as named, the call fails
with an actionable error rather than substituting.**

The governing distinction is between *locating* and *substituting*:

- **Locating (allowed, unlimited).** Mapping a handle to the bytes it denotes: a bare repo id to
  its local snapshot, a `.gguf` path to that file, an LM Studio directory to the model inside it,
  a repository id to a specific quantization file within *that same* repository. The artifact is
  the one named; only its location was unknown.
- **Substituting (forbidden).** Returning bytes the handle does not denote: a different
  repository, a different artifact class (GGUF conversion vs transformers weights), or a
  different quantization than an explicit selector requested. The artifact is not the one named.

The accepted rules are:

- A handle that does not name a GGUF is never promoted to one. Alias manifests, local-cache
  contents, and convenience heuristics are not authority to change the artifact class.
- Cross-repository redirection through an alias manifest is permitted only when the caller has
  already asked for the artifact class that redirection produces, and it must be logged.
- An explicit `:quant` selector is honoured exactly or the call fails. It must survive alias
  resolution; falling back to a default quantization silently is forbidden.
- Where a handle genuinely underdetermines the artifact — a GGUF repository id holding several
  quantizations — a default pick is legitimate, but it must be logged with the alternatives
  available (ADR 0001).
- Disambiguation is explicit and minimal. The HuggingFace provider accepts exactly one selector,
  `model_type="gguf" | "transformers"`, and naming the artifact directly (a `.gguf` path, or the
  GGUF repository id) always works. No further flags may be added to re-enable substitution.
- Errors raised under this ADR state what was requested, what was found, why they are not
  interchangeable, and how to ask for either one explicitly. `ModelArtifactMismatchError` is the
  contract type.
- Resolution decisions that remain choices must be observable. The resolution path may not use
  bare `except: pass` around a choice that changes which bytes are loaded.

This ADR amends ADR 0001. Where 0001 says it does not ban fallbacks, that permission does not
extend to model identity: there is no acceptable fallback from one model artifact to another,
warned or otherwise. Loud failure is the only correct behaviour.

## Consequences

### Positive

- A measurement, benchmark, or quality judgement can be attributed to the model that was named.
- Callers relying on the LM Studio Hub alias convenience get an error that tells them exactly
  which one-word change restores it, instead of wrong numbers.
- `model_type` gives GGUF-by-alias an explicit, greppable spelling at every call site that wants
  it.

### Negative

- Handles that previously "worked" by loading a GGUF now raise until the caller says which
  artifact they mean. This is intended; those calls were returning the wrong model.
- Provider authors must distinguish locating from substituting when adding resolution logic,
  which is a judgement call rather than a mechanical rule.

### Neutral

- This ADR does not require every artifact to be available locally, nor does it change
  cache-only, no-download behaviour.
- This ADR does not govern device, dtype, or context-window fallbacks. Those change performance,
  not identity, and remain under ADR 0001's warn-and-document rule.

## Enforcement

- Reviews must reject any resolution path that can return an artifact from a different
  repository, artifact class, or explicitly-requested quantization than the handle names.
- Reviews must reject new flags, env vars, or heuristics that re-enable substitution.
- Any `except` clause in a resolution path must re-raise `ModelArtifactMismatchError`; swallowing
  it converts a loud failure back into a silent one.
- Provider constructors must complete artifact selection before any loader runs, so a rejected
  request never partially loads.

## Validation

- Provider tests must cover: a handle that would have been substituted now raises; the explicit
  selector opts back in; a genuine request for the other artifact still resolves; and an
  ordinary handle with only one artifact present is unaffected.
- Quantization tests must cover an unsatisfiable explicit selector raising rather than
  downgrading, and selector survival across alias resolution.
- Tests must assert the error names the requested handle, the found artifact, and the explicit
  way to ask for each.

## Related

- `abstractcore/providers/huggingface_provider.py` (`_reject_silent_gguf_substitution`,
  `_find_gguf_in_cache`)
- `abstractcore/providers/mlx_provider.py` — the correct prior art: it reads the same Hub
  manifest, but only to build an error hint, and refuses to load a GGUF directory outright.
- `abstractcore/exceptions/__init__.py` (`ModelArtifactMismatchError`)
- `tests/huggingface/test_model_handle_fidelity_unit.py`
- `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md` (amended by this ADR)
- `docs/adr/0008-provider-owned-model-residency-truth.md`
