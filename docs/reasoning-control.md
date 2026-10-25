# Reasoning Control

AbstractCore exposes one parameter — `thinking=` — for every provider that can reason, and
maps it to whatever control surface each backend actually honors. This page explains what
you can request, what each provider does with the request, and how to verify from the
response that your request took effect.

For the wider parameter vocabulary (`temperature`, `top_p`, `max_output_tokens`, …) see
[Generation Parameters](generation-parameters.md). For the registry fields referenced here,
see `abstractcore/assets/README.md`.

## Request a reasoning mode

```python
from abstractcore import create_llm

llm = create_llm("mlx", model="mlx-community/Qwen3.8-27B-4bit")

llm.generate("Plan the migration", thinking=None)      # model/server default
llm.generate("Plan the migration", thinking="off")     # no reasoning ("none" is an alias)
llm.generate("Plan the migration", thinking="on")      # reasoning enabled, model default depth
llm.generate("Plan the migration", thinking="low")     # brief reasoning
llm.generate("Plan the migration", thinking="medium")  # balanced
llm.generate("Plan the migration", thinking="xhigh")   # deepest, when the model offers it
```

Accepted values: `None` / `"auto"`, `True` / `"on"`, `False` / `"off"` / `"none"`, and the
levels `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` (`"extra high"` is an alias for
`"xhigh"`).

When a model advertises a specific effort enum, a level outside that enum is mapped to the
nearest supported one and the substitution is reported as a `RuntimeWarning`. Requesting
`"high"` from a model offering `low | medium | xhigh` therefore runs at `medium`.

Reasoning text, when the model emits it, is available as `response.reasoning`, and
`response.content` stays clean.

## Precedence

Several things can set the reasoning of one call. The first that applies wins:

1. **`thinking=` on the call.** `thinking="auto"` (or `None` passed through an API that
   distinguishes it) means "no control": nothing is sent and no configured default is added.
2. **A provider-native reasoning field on the call**: `reasoning_effort=`, `reasoning=`
   (LM Studio's native endpoint, OpenRouter's object), or `think=` (Ollama). It is sent as
   given, without level mapping.
3. **The reasoning effort configured on the text route** (`abstractcore config set-default
   input.text --reasoning <level>`, or the gateway console). It applies only when the route
   describes the instance serving the call: the same provider, or the same endpoint profile
   (`endpoint:<id>`), and the same model. A route that sets only a reasoning effort, with no
   provider or model, applies to every instance. An effort configured for another backend is
   not a fact about this one, so it is not applied.
4. **Nothing.** The model or server default stays in effect.

Before 2026-10-25, step 3 applied to every call that did not pass `thinking=`. A machine
configured with `reasoning: minimal` for one route therefore sent `reasoning_effort: "low"`
for `generate(..., reasoning_effort="high")` and on unrelated models.

`response.metadata["thinking_source"]` records which step applied: `"thinking"`,
`"reasoning_effort"` (a raw field), or `"configured_default"`. For a raw `reasoning_effort=`,
`thinking_requested` is the value you passed (`"none"` reads as `"off"`). `thinking_effective`
repeats it only on providers that forward the field verbatim (OpenAI and the OpenAI-compatible
family, LM Studio and vLLM included). OpenRouter is excluded, because it documents a
`reasoning` object rather than a top-level `reasoning_effort`. The route summary in
`response.metadata["_resolved_generate_route"]` reports a configured effort that was not applied
as `reasoning: None`, with `reasoning_source` set to `not_applied:<reason>`.

When a call passes both `thinking=` and `reasoning_effort=`, providers differ. OpenAI and
generic OpenAI-compatible servers send the `thinking=` mapping in place of your field. LM Studio
sends your field. vLLM and OpenRouter send your top-level field next to their own mapping
(`chat_template_kwargs`, or the `reasoning` object), so the server decides between them. Pass
one or the other.

## What each provider sends

Two things are controlled separately: whether the model reasons at all, and how deeply.

| Provider | Enable / disable | Effort level |
|---|---|---|
| OpenAI | `reasoning_effort` (`"none"` when the model accepts it) | `reasoning_effort` |
| Anthropic | `thinking` block | `output_config.effort` (adaptive models) or a thinking budget |
| LM Studio | top-level `reasoning_effort`: `"none"` for off, a non-`"none"` value for on | top-level `reasoning_effort` with the model's declared level |
| OpenAI-compatible servers | `chat_template_kwargs.<template switch>` when the server accepts template kwargs; otherwise `reasoning_effort` for models that declare levels | `chat_template_kwargs.<effort kwarg>`, or `reasoning_effort` for models with no template surface |
| vLLM | top-level `chat_template_kwargs.<template switch>`, or `reasoning_effort` for models that declare levels and no switch | `chat_template_kwargs.<effort kwarg>` or `reasoning_effort`; `thinking_token_budget` as a best-effort cap |
| OpenRouter | `reasoning: {"effort": "none"}` / `reasoning: {"enabled": true}` | `reasoning: {"effort": <level>}` |
| Portkey | not mapped: the backend behind a config is unknown, so no control is sent and a warning says so | not mapped |
| MLX | no-think prefill placed at the generation boundary of the rendered prompt | the template's own effort instruction rendered into the system block |
| HuggingFace (transformers) | `enable_thinking` passed to `apply_chat_template` | the effort kwarg passed to `apply_chat_template`; the cached renderer writes the effort instruction into the system block |
| HuggingFace (GGUF) | no-think prefill placed by the local renderer at the generation boundary | the model's embedded template rendered with the effort kwarg, or the effort instruction written into the system block for plain ChatML builds |
| Ollama | `think` | `think` level for GPT-OSS; other models run with reasoning enabled |
| GPT-OSS (Harmony) | `Reasoning:` system line (traces cannot be fully disabled) | `Reasoning: low \| medium \| high` |

### Per model family

The registry decides which surface a model has; the provider decides how it travels. The
table shows what reaches the server for `thinking="on"` / `"off"` / `"high"`.

| Model family (registry) | LM Studio | vLLM | MLX / HF transformers |
|---|---|---|---|
| Qwen3.5 / 3.6, Laguna, Qwen3 (`template_kwarg`, assistant prefill; no levels) | `reasoning_effort` `medium` / `none` / `medium` (reported as `on`) | `chat_template_kwargs.enable_thinking` true / false / true, plus `thinking_token_budget` for a level (reported as `on`) | template switch; MLX also the no-think prefill |
| Qwen3.8 (`template_kwarg`, `effort_template_kwarg`, levels `low\|medium\|xhigh`) | `reasoning_effort` `medium` / `none` / `medium` (nearest level, warned) | `enable_thinking` plus `reasoning_effort` template kwarg | switch plus the template's effort sentence |
| HY V3 / Hy-Memory (effort switch, levels `low\|high`) | `reasoning_effort` `high` / `none` / `high` | `chat_template_kwargs.reasoning_effort` `high` / `no_think` / `high` | the effort variable `no_think\|low\|high` |
| Gemma 4, Nemotron, GLM-5, LongCat (`template_kwarg` only) | `reasoning_effort` `medium` / `none` / `medium` (reported as `on`) | `chat_template_kwargs.enable_thinking`; Nemotron `low_effort` for `low`/`minimal` | HF: template switch; MLX: not mapped (warns) |
| DeepSeek V4, Mistral Small 4 (levels incl. `none`, no template surface) | `reasoning_effort` `high` / `none` / `high` | `reasoning_effort` | not mapped (warns) |
| GPT-OSS (Harmony, levels `low\|medium\|high`) | `Reasoning:` system line plus `reasoning_effort` | `reasoning_effort` (vLLM builds the Harmony system message from it) plus the system line | `Reasoning:` system line |
| GLM-4.6V (`prompt_disable_token` only) | `/nothink` appended for off; on not mapped (warns) | `/nothink` for off; on not mapped | `/nothink` for off |
| Kimi K2.6 (`request_param: thinking`), surface-less reasoning models | not mapped (warns) | not mapped (warns) | not mapped (warns) |

"Not mapped" means no control artifact is sent, `thinking_effective` is absent, and a
`RuntimeWarning` states that the model or server default stays in effect.

### LM Studio facts

LM Studio's OpenAI-compatible endpoint behaves differently from vLLM, so its lane is pinned
by live evidence (2026-10-25, `qwen/qwen3.6-35b-a3b` MLX and `qwen/qwen3.8-27b`, rendered
prompts read back with `lms log stream -s model --filter input`):

- `chat_template_kwargs`, at the top level or under `extra_body`, and top-level template
  variables such as `enable_thinking` are ignored. AbstractCore still sends them, but never
  counts them as an applied control.
- The top-level `reasoning_effort` field reaches the chat template on every request shape:
  plain prompts, message histories, and tool calls with tool-result history, on both the
  sync and async paths. `"none"` renders the template's own empty think block. For an on/off
  model such as Qwen3.6, every other value turns thinking on and produces the same
  reasoning. For Qwen3.8, LM Studio applies the level, mapping `minimal` to `low` and
  `high` to `xhigh` on its side.
- LM Studio's default for both models was thinking off. Before this fix, `thinking="on"`
  silently stayed off while the metadata reported `on`.
- The assistant no-think prefill is not needed once `reasoning_effort: "none"` is sent.
  Sending both renders the empty think block twice. AbstractCore uses the prefill only as
  the off switch of last resort, when the server has rejected `reasoning_effort` earlier
  in the session.
- The native REST endpoint (`POST /api/v1/chat`, field `reasoning`) cannot carry message
  history, custom tools, or structured output. `thinking=` no longer routes there. Pass
  `reasoning=` explicitly to use that endpoint for a plain prompt.

Local providers serialize prompts themselves, so their controls are prompt artifacts rather
than API fields. AbstractCore reproduces the artifact the model's own chat template would
produce, which keeps a controlled request inside the model's training distribution.

## Model requirements

A level request is enforced only when the model's registry entry declares a control surface
for it — `thinking_control.effort_template_kwarg` for backends that render the template, and
`thinking_control.effort_system_lines` for lanes that serialize prompts locally, with
`reasoning_levels` listing the accepted enum. Qwen3.8 declares all three and supports
`low`, `medium`, and `xhigh`.

Models whose templates expose only an on/off switch — Qwen3.6, for example — accept
`thinking="off"` and `thinking="on"`, and treat a level request as "reasoning enabled" with a
`RuntimeWarning` explaining that effort scaling is unavailable. To add support for a new
model, declare its surfaces in `abstractcore/assets/architecture_formats.json` or
`model_capabilities.json`; no provider code changes are required.

## Verify what actually happened

Every response carries the request and its outcome in `response.metadata`:

```python
response = llm.generate("Plan the migration", thinking="low")
print(response.metadata["thinking_requested"])      # "low"
print(response.metadata["thinking_effective"])      # "low" when enforced
print(response.metadata["thinking_handled_level"])  # True when a real artifact was applied
```

- `thinking_requested` / `thinking_effective` — what you asked for, and what the request
  became after mapping.
- `thinking_handled_enable_disable` / `thinking_handled_level` — whether the on/off switch and
  the effort level were actually applied.
- `thinking_supported_levels` — the model's advertised effort enum.
- `thinking_supports_output` / `thinking_supports_control` — whether the model emits reasoning,
  and whether it exposes a request-side knob.

`thinking_effective` reports a level only when a control artifact reached the model. When a
request cannot be honored, the field reflects the weaker control that was applied and a
`RuntimeWarning` states what remains in effect, so a silently ignored request never reads as a
successful one.

## Requests that fall back

A few request shapes are served by a code path that carries no control artifact. AbstractCore
declines the claim for these rather than reporting an unenforced level:

- **Prefilled system blocs.** In KV cache modes where the system block is already committed to
  the cache, the effort instruction cannot be inserted. Disabling reasoning still works,
  because that control lives at the generation boundary.
- **HuggingFace GGUF with structured output, media, tool-role or content-part histories**, and
  when `ABSTRACTCORE_GGUF_CONTROL_PLANE=0` disables the local renderer. These requests are
  served by `llama-cpp-python`'s chat-completion path, which renders no control artifact.
- **Servers that reject `reasoning_effort`.** AbstractCore drops the field, retries the
  request once so it still succeeds, and warns that the level was not applied. On LM Studio
  the rejection is remembered for the session: later requests decline the claim, and
  `thinking="off"` falls back to the assistant no-think prefill when the model declares one.

## Reasoning control and prompt caching

An effort instruction is part of the system block, so changing levels changes the first tokens
of the prompt. A durable prompt-cache bloc built at one effort level will not prefix-match a
request at another, and the prompt is recomputed. Pin one level per cached session to keep
prefixes warm. See [Prompt Caching](prompt-caching.md).

## Related documentation

- [Generation Parameters](generation-parameters.md) — the full parameter vocabulary and
  precedence rules.
- [Fallbacks](fallbacks.md) — the layered strategy behind the Qwen no-think switch.
- [Capabilities](capabilities.md) — what AbstractCore supports across providers.
- `abstractcore/assets/README.md` — the registry fields that drive all of the above.
