# 0872 — Reservation order: the embedder must leave room for the configured text default (ADR 0011 follow-up)

> Type: defect (capacity)
> Created: 2026-10-29 (OVH 8080 pass four; filed by the merge desk on the lead's ruling)
> Priority: high (before the release wave's server pass five)
> Labels: residency, embeddings, llama.cpp, gpu-memory, fit
> Links: [ADR 0011](../../adr/0011-shared-weights-and-concurrent-requests.md), [ADR 0010](../../adr/0010-model-residency-fit-and-eviction.md), [0870](0870_per_model_slot_scheduler_and_shared_gguf_weights.md), core PR #34 (R5: the GGUF embedder reserves the largest input that fits)

## Why
Core PR #34 (amended, 2026-10-29) first made the GGUF embedder's reservation cheap — flash attention and 512-token
micro-batches took its compute scratch on OVH 8080 from 6,710 to 1,808 MB at the full 8,192-token reservation — and
keeps a fit check as the safety net: the input is sized to what fits beside the models already resident minus the
5 % buffer. That safety net is order-dependent: it only yields when the LLM loads FIRST. An embedder that loads on an
EMPTY card (the automation that embeds before anyone chats, a restart that warms embeddings first) takes its whole
reservation, and on a card where the text default's slots are tight (the 16 GB card: a second Kimi 9B slot needs
8.3 GiB) the ~1.8 GB still decides whether the second slot fits. Smaller than before, same shape of defect.

## What
- The embedder's fit considers the CONFIGURED text default (`capability_defaults` output.text, when it is a local
  model) as if it were resident: its weights plus one slot at its reserved context (ADR 0010 point 8 / 0870) are
  subtracted from the free memory before the embedder picks its input size. The row says so
  (`context_reason`: "… leaving room for <text default> and one slot").
- Order independence is the acceptance: load the embedder first, then the text default → the text default gets the
  same number of slots as when it loads first (test on a fake device-memory ledger; a real-weights run on the Linux
  card in the server pass).
- An admin value (`model_contexts` for the embedder) stays exact-or-refused; the refusal names the text default it
  would starve.
- No eviction, no shrinking of a loaded context (ADR 0011): the embedder's reservation is decided at ITS load only.

## Done when
`tests/embeddings/test_embedding_gguf_repo_and_fit_unit.py` has the two-order test (embedder first / LLM first) and
both orders give the text default the same slot count; the row's `context_reason` names the text default; the
server pass shows a second LLM slot with the embedder loaded first (0 → 1 queued at 2 concurrent).
