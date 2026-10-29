# 0866 — A GGUF-only repo in the Hugging Face cache says what it is

**Status**: completed (2026-10-29, unreleased) · **Priority**: P1 · **Created**: 2026-10-29
**Cross-package**: operator QA batch item (c): models "Not in the catalog" must be usable (the gateway console
offers **Use as default** for an installed text model).

## Why
On the OVH server, `models list` reported every GGUF repo outside the catalog with `kind: model, tasks: []`. That
included mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF, Gemma-4-14B-A4B-Heretic-GGUF and XHToken/Spark-X2.6-4B-GGUF.
A catalog GGUF download fetches only the quant file (no `config.json`, no card), so `_hf_local_kind_tasks` had no
evidence. The console could not tell a text model from a TTS or image GGUF.

## What shipped
`abstractcore/config/model_materializer.py` `_gguf_kind_tasks`: for a snapshot with neither a config nor a card
answer, it reads `general.architecture` from the first GGUF header (`read_gguf_architecture`):
- an architecture naming `tts` gives `text_to_speech`;
- `flux` gives `text_to_image`;
- `whisper` gives `speech_to_text`;
- `clip` and tokenizers give no task;
- an embedding repo (its name contains "embed": Qwen3-Embedding's architecture is `qwen3`) gives `embedding` /
  `text_embedding`;
- anything else gives `text_generation`.
`tasks_source: "gguf"`. A repo that ships its own config keeps the config's answer.

## Evidence
- `tests/config/test_installed_gguf_tasks.py`: the server's five real cases (architectures read live on the server:
  qwen35, spark2_5, qwen3 embedding, qwen3-tts, flux), plus a repo with a config. Red without the GGUF step
  (mutation-checked).
- `tests/config`: 1029 passed.

## Follow-ups
- Whether this install's llama.cpp can load an architecture (spark2_5 cannot) is batch item (f), separate.
