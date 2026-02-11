# Providers: Ollama — Native API vs OpenAI-Compatible (2026-01-09)

## Questions Answered

4) What about `abstractcore/providers/ollama_provider.py`? Is it “just another OpenAI-compatible duplication”, or are there genuine reasons to keep it separate?

## Executive Summary

`OllamaProvider` is **not** “just OpenAI-compatible”.

It uses Ollama’s **native API** (`/api/chat`, `/api/generate`, `/api/embeddings`) and implements Ollama-specific behaviors:
- server-side unload via `keep_alive=0`
- Ollama-specific token option mapping (`num_predict`)
- structured output via Ollama’s `format` field (full JSON schema)
- message-role restrictions (only `system|user|assistant`) and explicit conversion for `tool` messages

Ollama does expose an OpenAI-compatible `/v1` surface in many modern setups, but using it would cost you native features (and can change semantics). Keeping a separate `OllamaProvider` is justified.

## What the code does today (confirmed)

Touch points:
- `abstractcore/abstractcore/providers/ollama_provider.py`
  - `OllamaProvider.unload` (uses `keep_alive=0` on `/api/generate`)
  - `OllamaProvider._convert_messages_for_ollama` (role normalization)
  - `OllamaProvider._generate_internal` (payload and endpoint selection)
  - structured outputs: `payload["format"] = response_model.model_json_schema()`

Media handling:
- For vision-capable Ollama models, Ollama expects an `images: [base64...]` array on chat messages.
- This is handled via `LocalMediaHandler("ollama", ...)` in the provider code.

## Is there duplication with OpenAI-compatible providers?

There is some *conceptual* overlap (chat, streaming, tools) but not a pure duplication:
- Endpoint paths differ (`/api/chat` vs `/v1/chat/completions`).
- Payload shape differs (`options.num_predict`, `format`, `keep_alive`).
- Message constraints differ (no `tool` role in Ollama’s chat schema).

So: it’s not “a 3rd copy” of OpenAI-compatible.

## Should OllamaProvider be reimplemented as OpenAICompatibleProvider + base_url?

Recommendation: **No** (keep native by default), but you can optionally offer both modes.

### Pros of switching to OpenAI-compatible `/v1` for Ollama
- Potentially more uniform payload across providers.
- Reuse of the consolidated OpenAI-compatible base class.

### Cons (why native is worth keeping)
- You lose Ollama-specific memory management semantics (`keep_alive` / unload behavior).
- You may lose Ollama-native structured output (`format` field with full schema).
- You rely on Ollama’s “compatibility layer”, which historically changes behavior over time.

## Actionable Recommendations

### R1) Keep `OllamaProvider` as the native provider

Touch points:
- `abstractcore/abstractcore/providers/ollama_provider.py`

### R2) Offer an explicit “Ollama via OpenAI-compatible /v1” option (optional)

If you want to support users who specifically want OpenAI-style APIs:
- Document: “use `OpenAICompatibleProvider` with `base_url=http://localhost:11434/v1`”
- Or add a separate provider key (example): `ollama-openai` that subclasses the OpenAI-compatible base and sets base_url default to Ollama’s `/v1`.

Touch points (if you do it):
- `abstractcore/abstractcore/providers/registry.py` (register provider)
- `abstractcore/abstractcore/providers/openai_compatible_provider.py` (base)

### R3) Clean up unreachable logic in endpoint selection

In `OllamaProvider._generate_internal`, `use_chat_format` is currently set with `... or True`, which makes the non-chat branch unreachable.

Touch point:
- `abstractcore/abstractcore/providers/ollama_provider.py` (search for `use_chat_format`)

Recommendation:
- Either always use chat and delete the dead branch (simpler), or implement a real decision rule (clearer intent).

