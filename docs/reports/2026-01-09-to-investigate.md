# To Investigate (2026-01-09)

This file collects potentially concerning issues or doc/code discrepancies discovered during the review. Some are definite cleanups; others require product intent clarification.

## Provider architecture and duplication

1) **Major OpenAI-compatible duplication**
   - `abstractcore/abstractcore/providers/openai_compatible_provider.py`
   - `abstractcore/abstractcore/providers/lmstudio_provider.py`
   - `abstractcore/abstractcore/providers/vllm_provider.py`
   Impact: behavioral drift risk as you add OpenRouter and other providers.

2) **Streaming error semantics**
   - Several OpenAI-compatible streaming implementations yield a final chunk like `content="Error: ..."` instead of raising.
   - Touch points:
     - `abstractcore/abstractcore/providers/openai_compatible_provider.py:_stream_generate`
     - `abstractcore/abstractcore/providers/lmstudio_provider.py:_stream_generate`
     - `abstractcore/abstractcore/providers/vllm_provider.py:_stream_generate`
   Risk: callers interpret failed streams as “successful” responses.

3) **Environment-variable “bleed” due to truthy `or` resolution**
   - Providers often use `api_key = api_key or os.getenv(...)` which makes it hard to explicitly disable auth and can pull in unrelated env vars.
   - Touch point:
     - `abstractcore/abstractcore/providers/openai_compatible_provider.py:__init__` (and duplicates)

## Media system

4) **Two different `MediaCapabilities` classes with the same name**
   - `abstractcore/abstractcore/media/types.py:MediaCapabilities` (simple handler capability type)
   - `abstractcore/abstractcore/media/capabilities.py:MediaCapabilities` (richer model capability type)
   Risk: confusion + accidental misuse as features grow.

5) **OpenAIProvider non-vision image handling may skip images**
   - OpenAIProvider uses `OpenAIMediaHandler` unconditionally.
   - For non-vision models, `OpenAIMediaHandler` can skip image content instead of applying vision fallback + placeholder semantics.
   - Touch points:
     - `abstractcore/abstractcore/providers/openai_provider.py:_generate_internal`
     - `abstractcore/abstractcore/media/handlers/openai_handler.py:create_multimodal_message`
     - `abstractcore/abstractcore/media/handlers/local_handler.py:_create_text_embedded_message`
   Needs: clarify desired product behavior (“always fallback if image + non-vision”).

6) **Potential None-path bug in vision fallback local model detection**
   - `VisionFallbackHandler._has_local_models` assumes `local_models_path` is set.
   - Touch point:
     - `abstractcore/abstractcore/media/vision_fallback.py:_has_local_models`
   Likely safe with current config defaults, but worth hardening.

## Ollama provider

7) **Unreachable branch due to `... or True`**
   - `use_chat_format = tools is not None or messages is not None or True`
   - Touch point:
     - `abstractcore/abstractcore/providers/ollama_provider.py` (search `use_chat_format`)
   Action: remove dead code or implement real decision logic.

## Documentation drift

8) **Provider docs imply OpenAI provider covers “OpenAI & compatible APIs”**
   - Touch point:
     - `abstractcore/abstractcore/providers/README.md`
   Action: update to mention `openai_compatible_provider.py` and the OpenAI-compatible family.

9) **Media docs contain pseudo-code that no longer matches the BaseProvider pipeline**
   - Touch points:
     - `abstractcore/abstractcore/media/README.md`
     - `abstractcore/docs/media-handling-system.md`
   Action: update examples to reflect `BaseProvider._process_media_content(...)` as the central ingestion path.

