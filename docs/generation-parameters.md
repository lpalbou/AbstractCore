# Generation Parameters Architecture

This document explains the design and implementation of unified generation parameters (`temperature`, `top_p`, `top_k`, `seed`, and thinking/reasoning controls) across AbstractCore providers.

## Design Principles

### 1. **Interface-First Design**
Parameters are declared at the `AbstractCoreInterface` level, ensuring:
- Consistent API contract across all providers
- Type safety and documentation at the interface level
- Automatic inheritance by all provider implementations

### 2. **DRY (Don't Repeat Yourself)**
Common parameters are handled centrally to avoid:
- Code duplication across providers
- Inconsistent parameter handling
- Maintenance overhead for parameter changes

### 3. **Graceful Degradation**
Providers that don't support certain parameters:
- Accept the parameters without error
- Issue appropriate warnings (e.g., Anthropic seed warning)
- Maintain consistent API behavior
- Provide fallback mechanisms where possible

## Architecture Overview

```python
AbstractCoreInterface (interface.py)
├── temperature: float = 0.7        # Interface-level default
├── seed: Optional[int] = -1         # Random; no seed forwarded
├── thinking: Optional[bool|str] = None  # Unified thinking/reasoning control (best-effort)
└── _validate_parameters()           # Validation logic

BaseProvider (base.py)
├── _prepare_generation_kwargs()     # Unified parameter processing
├── _extract_generation_params()     # Parameter extraction helper
├── _metadata_inference_parameters() # Asset-driven model/architecture defaults
├── _apply_thinking_request()        # Provider-agnostic + provider-specific thinking mapping
└── Parameter fallback hierarchy     # kwargs → instance/assets → defaults

Individual Providers
├── Provider-specific parameters only (top_p, frequency_penalty, etc.)
├── Provider-specific parameter mapping
└── Native API integration
```

## Parameter Hierarchy

Token budgets use `max_tokens` for input plus output, and `max_output_tokens` for generation.
Set the output cap explicitly: an omitted cap can use a provider's full output capability.
A per-call cap above the model's registered output ceiling is reduced to that ceiling, with a
`RuntimeWarning`. For a model that `model_capabilities.json` does not know, the ceiling is only a
guessed default (4096), so a per-call cap is sent as given and the server enforces its real
limit. A cap set on the constructor still bounds every per-call value.
Ollama's thinking path raises a smaller cap to a minimum output budget: 1024 for `minimal`,
2048 for `low` or boolean on, 4096 for `medium`, 8192 for `high`. Use `thinking=False` when
enforcing a small output cap. See [bounded integration recipes](agent-usage.md).

Parameters follow a clear precedence order:

1. **Method-level kwargs** (highest priority)
   ```python
   llm.generate("Hello", temperature=0.9, seed=123)
   ```

2. **Instance-level parameters**
   ```python
   llm = create_llm("openai", temperature=0.5, seed=42)
   ```

3. **Loaded model generation metadata**
   Hugging Face Transformers models may publish `generation_config.json`; when present, AbstractCore reads safe defaults such as `temperature`, `top_p`, and `top_k` from the loaded model. Explicit kwargs and constructor values still win.

4. **Static model / architecture metadata**
   ```json
   {
     "inference_parameters": {
       "temperature": 1.0,
       "top_p": 0.95,
       "top_k": 64
     }
   }
   ```

5. **Interface defaults** (lowest priority)
   ```python
   # temperature=0.7, seed=-1 (random; no seed forwarded)
   ```

### Override Guarantee

`inference_parameters` in `model_capabilities.json`, `architecture_formats.json`, and loaded HF `generation_config.json` are defaults only. They are never hard requirements. For providers/backends that accept a parameter, caller-supplied values always win:

```python
llm = create_llm("huggingface", model="google/gemma-4-E4B-it")  # metadata default: top_k=64
llm.generate("Hello", temperature=0.2, top_p=0.6, top_k=12)      # uses caller values
```

If a backend does not support a parameter, provider-specific compatibility rules still apply. For example, strict gateway providers may drop unsupported parameters rather than forwarding invalid requests.

## Provider-Specific Implementation

### Native Support (OpenAI, Ollama, LM Studio, Hugging Face, MLX)
```python
# Direct parameter mapping to provider API
call_params["temperature"] = params["temperature"]
call_params["top_p"] = params["top_p"]
call_params["top_k"] = params["top_k"]  # when the backend exposes this control
if "seed" in params:
    call_params["seed"] = params["seed"]
```

Notes:
- Hugging Face Transformers applies loaded `generation_config` defaults and forwards `temperature`, `top_p`, and `top_k` into `generate()` / pipeline calls where supported.
- Hugging Face GGUF and Ollama forward sampling controls through their local runtime APIs.
- MLX uses an `mlx-lm` sampler so `temperature`, `top_p`, and `top_k` affect actual decoding rather than only AbstractCore metadata.
- LM Studio receives `top_k` when using the LM Studio provider; the generic OpenAI-compatible provider keeps stricter OpenAI-compatible behavior unless a backend-specific hook handles the parameter.

### Graceful Fallback (Anthropic)
```python
# Accept parameters but log limitation
if "seed" in params:
    self.logger.debug(f"Seed {params['seed']} requested but not supported - logged for debugging")
```

### Portkey gateway (pass-through)

Portkey is a routing gateway that forwards payloads to **many** backends (OpenAI, Anthropic, Gemini, Grok, etc.). To avoid sending defaults that strict models reject, the Portkey provider:

- Forwards optional generation parameters **only when explicitly set** by the user (constructor or `generate()` kwargs).
- Drops unsupported parameters for OpenAI reasoning families (gpt-5/o1), and uses `max_completion_tokens` instead of `max_tokens` for those models.
- Keeps legacy `max_tokens` for non-reasoning families to preserve compatibility with older backends.

## Thinking / Reasoning Control (Unified)

> For the full picture — per-provider transports, model requirements, how to verify a request
> took effect, and the interaction with prompt caching — see
> **[Reasoning Control](reasoning-control.md)**.

Modern models may expose “thinking”/“reasoning effort” as either:
- a **request-side control** (enable/disable or low/medium/high), and/or
- a **separate output channel** (provider fields or inline tags).

AbstractCore exposes a single best-effort parameter:

```python
response = llm.generate("Solve this", thinking=None)      # auto (provider/model default)
response = llm.generate("Solve this", thinking="off")     # try to reduce/disable thinking
response = llm.generate("Solve this", thinking="none")    # alias for "off"
response = llm.generate("Solve this", thinking="on")      # enable thinking
response = llm.generate("Solve this", thinking="low")     # lower effort / smaller budgets (when supported)
response = llm.generate("Solve this", thinking="medium")  # balanced (when supported)
response = llm.generate("Solve this", thinking="high")    # higher effort / larger budgets (when supported)
print(response.reasoning)
```

**Accepted values**: `None|"auto"|"on"|"off"|"none"|True|False|"low"|"medium"|"high"` (legacy aliases: `"minimal"`, `"xhigh"`, `"extra high"`).

**Best-effort mappings (as of May 2026):**
- **OpenAI** (`OpenAIProvider`): Chat Completions `reasoning_effort` (values come from `reasoning_levels` in `model_capabilities.json`). `thinking="off"` maps to `reasoning_effort="none"` when supported; otherwise it falls back to the minimum supported effort with a warning (e.g., `gpt-5.2-pro` → `"medium"`).
- **Anthropic** (`AnthropicProvider`): Messages API `thinking` + (for Claude 4.6 adaptive thinking) `output_config.effort`.
  - Unified levels map to effort: `low|medium|high|xhigh` → `low|medium|high|max` (when supported); `xhigh` falls back to `high` with a warning when `max` is unavailable.
  - For older models, AbstractCore falls back to manual thinking budgets via `thinking: {type:\"enabled\", budget_tokens: ...}` (best-effort; newer models deprecate this).
- **LM Studio / OpenAI-compatible local servers** (`LMStudioProvider`, `OpenAICompatibleProvider`):
  - **Qwen3 / Qwen3.5 / Nemotron**: `chat_template_kwargs.enable_thinking` on OpenAI-compatible servers that accept template kwargs.
    - **LM Studio**: its OpenAI-compatible endpoint ignores `chat_template_kwargs`; the top-level `reasoning_effort` field is what reaches the template (`"none"` = off, any other value = on for on/off models, the declared level for models with levels). See [Reasoning Control](reasoning-control.md#lm-studio-facts). The empty think block (`<think>\n\n</think>\n\n`) is used as the off switch only when the server rejected `reasoning_effort`.
    - Qwen also supports a “soft” `/no_think` / `/think` instruction (stateful across turns), but AbstractCore prefers the stateless hard-switch where needed. See `docs/fallbacks.md`.
    - **Effort levels**: models whose registry entry declares `thinking_control.effort_template_kwarg` (for example Qwen3.8) receive the OpenAI-standard `reasoning_effort` request field, which LM Studio maps into the model's chat template. Templates that expose only a boolean switch treat a level request as “thinking enabled” and warn.
    - **Nemotron**: `thinking="low"` additionally maps to `chat_template_kwargs.low_effort=True` on servers that accept template kwargs (not LM Studio, which ignores them).
  - **Seed‑OSS**: `chat_template_kwargs.thinking_budget` on servers that accept template kwargs (levels map to budgets: low=512, medium=1024, high=4096, xhigh=8192; `off` → 0). On LM Studio, Seed-OSS's declared levels travel as `reasoning_effort` instead.
- **MLX** (`MLXProvider`): the provider serializes prompts locally, so both controls are prompt artifacts. `thinking="off"/"none"` places the empty think block at the generation boundary; a level declared in `thinking_control.effort_system_lines` renders the template's own effort instruction into the system block.
- **HuggingFace (transformers)** (`HuggingFaceProvider`): `enable_thinking` and the declared effort kwarg are passed to `tokenizer.apply_chat_template(...)`, so the model's own template produces the prompt. The cached (KV) renderer writes the same effort instruction into its system block.
- **HuggingFace (GGUF / llama-cpp-python)** (`HuggingFaceProvider` with GGUF models):
  - `llama-cpp-python`’s `Llama.create_chat_completion()` does not forward per-request template kwargs, so thinking-controlled requests are served by AbstractCore's local renderer. Builds that ship an embedded chat template are rendered through it with `enable_thinking` and the declared effort kwarg; plain ChatML builds get the empty think block at the generation boundary and the effort instruction in the system block.
  - Structured output, media, tool-role or content-part histories are served by `create_chat_completion` instead. Those requests carry no control artifact, so AbstractCore reports the control as unhandled and warns rather than claiming it.
  - **Local context note**: model cards may advertise extremely large context windows (e.g. 262k). For GGUF loads, AbstractCore will first try the advertised `max_tokens` (context window); if allocation fails locally it retries with smaller llama.cpp `n_ctx` values (best-effort). Pass `max_tokens=...` to `HuggingFaceProvider()` to explicitly control the runtime `n_ctx`.
- **vLLM**: top-level `chat_template_kwargs.<template switch>` (the name the model declares, e.g. `enable_thinking`), or top-level `reasoning_effort` for models that declare levels and no switch. vLLM ignores a literal nested `extra_body` field, so AbstractCore sends every vLLM extension at the top level of the request.
  - When `thinking` is a level, AbstractCore also sets `thinking_token_budget` (vLLM reasoning budget; effective only when the server runs with `--reasoning-parser`). The budget alone is never reported as an applied effort level.
- **OpenRouter**: the normalized `reasoning` object — `{"effort": <level>}`, `{"effort": "none"}` for off (the lowest declared effort, with a warning, for models whose reasoning is mandatory), `{"enabled": true}` for a bare on. An explicit `reasoning={...}` argument wins.
- **Portkey**: not mapped (the backend behind a config is unknown); a warning says no control was sent.
- **Ollama**: request field `think` (bool for most models; `"low"|"medium"|"high"` for GPT‑OSS)
- **GPT‑OSS (Harmony)**: inject system line `Reasoning: low|medium|high` (traces can’t be fully disabled; `"off"` maps to `"low"` with a warning)

**Output semantics**: when a provider/model exposes reasoning, AbstractCore normalizes it into `GenerateResponse.metadata["reasoning"]` and keeps `GenerateResponse.content` clean using `abstractcore/architectures/response_postprocessing.py` (asset-driven via `assets/model_capabilities.json` + `assets/architecture_formats.json`).

When a requested thinking mode is not supported by a model/provider, AbstractCore emits a `RuntimeWarning` and applies a best-effort approximation:

- If the model advertises `reasoning_levels`, AbstractCore maps the requested level to the nearest supported level (generic ordering: `minimal < low < medium < high < xhigh`) and reports the effective level in the warning.
- If a provider/model can only toggle reasoning on/off (no effort scaling), AbstractCore still enables reasoning for level requests and warns that the requested effort level may be ignored.
- `thinking_effective` reports a level only when a control artifact actually reached the model, so an unenforced request never reads as an enforced one.

### Observability: requested vs effective thinking

When `thinking=` is provided, AbstractCore records the requested and effective thinking mode in `GenerateResponse.metadata`:

- `thinking_requested`: normalized unified request (`"off"`, `"on"`, or a level like `"high"`)
- `thinking_effective`: effective unified control after mappings (for example `"xhigh" → "high"` for a model that only supports up to `"high"`)
- `thinking_level_requested` / `thinking_level_effective`: effort-level details when applicable
- `thinking_handled_enable_disable` / `thinking_handled_level`: whether the provider/model actually implemented the on/off toggle and/or the effort scaling knob
- `thinking_supported_levels`: model-advertised effort enum when available (from assets)
- `thinking_supports_output` / `thinking_supports_control`: asset-driven capability split (model emits reasoning vs model exposes a request-side knob)

These fields make it easier to debug best-effort fallbacks without relying only on warnings.

## Session Integration

Sessions maintain persistent parameters across conversations:

```python
session = BasicSession(
    provider=llm,
    temperature=0.5,    # Default for all messages
    seed=42            # Consistent across conversation
)

# Uses session defaults
response1 = session.generate("Hello")

# Override for specific message
response2 = session.generate("Be creative!", temperature=0.9)
```

For prompt-cache-aware long chats, use `CachedSession` (see `docs/prompt-caching.md`):

```python
from abstractcore import CachedSession

session = CachedSession(provider=llm, system_prompt="You are helpful.", prompt_cache_strategy="auto")
session.generate("Hello")
```

## Future Extensibility

Adding new common parameters requires:
1. Declaration in `AbstractCoreInterface`
2. Logic in `BaseProvider._extract_generation_params()`
3. Optional static metadata in `assets/model_capabilities.json` or `assets/architecture_formats.json`
4. Provider-specific mapping where supported

Provider constructors should only keep provider-specific state. Common defaults should flow through the base fallback hierarchy.

## Performance Considerations

- **Minimal Overhead**: Parameter extraction happens once per generation call
- **Memory Efficient**: No parameter duplication across providers
- **CPU Efficient**: Simple dictionary operations for parameter resolution

## Backward Compatibility

All changes are fully backward compatible:
- Existing code continues to work unchanged
- New parameters are optional with sensible defaults
- Provider behavior remains consistent for existing use cases

## Empirical Verification (Best-Effort)

Determinism across LLM providers is **not guaranteed**. When supported, AbstractCore passes seed-like
controls to providers/backends and recommends `temperature=0` to reduce randomness, but results can
still vary with backend settings, hardware, and model/server updates.

To compare repeatability, keep the prompt, exact weights, backend, seed and generation budgets
fixed, then compare complete outputs from repeated calls. A matching seed is not a guarantee
of identical output across machines or backends.

**Provider-Specific Implementations**:
- **OpenAI**: Native `seed` parameter in API
- **MLX**: `mx.random.seed()` before generation
- **Ollama**: `seed` in options payload
- **HuggingFace**: `torch.manual_seed()` + GGUF native seed
- **LMStudio**: OpenAI-compatible `seed` parameter
- **Anthropic**: Issues `UserWarning` when seed provided

**Testing Commands**:
```bash
# Verify determinism across providers
python tests/manual_seed_verification.py

# Test specific provider
python tests/manual_seed_verification.py --provider openai
```
