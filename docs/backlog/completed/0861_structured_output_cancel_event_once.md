# 0861: Structured output hands the host's cancel event over once (GGUF "got multiple values")

**Status**: completed 2026-10-29 (unreleased; ships with the next AbstractCore release).
**Found by**: the OVH release-test server (AbstractCore 2.31.0, gateway 0.17.0): every structured call on
`unsloth/gemma-4-E4B-it-GGUF` (HuggingFaceProvider, GGUF lane) failed with
`HuggingFaceProvider._generate_gguf() got multiple values for keyword argument 'cancel_event'`.

## Cause

`BaseProvider.generate` consumes the host's `cancel_event=` in the plain path's retry loop and forwards it as
`CANCEL_KWARG` (`_cancel_event`); structured output (`response_model=`) branches off earlier, so the raw
`cancel_event` kwarg travelled through the structured handler into `_generate_internal`. The HuggingFace provider
pops `_cancel_event` and passes `cancel_event=` explicitly to its lane, with the raw one still in `**kwargs`.

## Fix

`BaseProvider._structured_handler_kwargs` hands the event to the structured handler under `CANCEL_KWARG` for
providers that support generation cancel (drops it for the others, which would forward an unknown kwarg into their
payload). It is applied where the handler is called: the plain structured branch and the second pass of tools +
structured output. The first pass of tools + structured output keeps the raw kwarg: it is the plain path, which
consumes it (review 2026-10-29: converting it before the hybrid branch made that pass mint a private eject event,
so the host's Stop no longer stopped the model there).

## Evidence

`tests/providers/test_generation_cancel_unit.py::test_structured_output_hands_the_cancel_event_over_once` (both a
cancel-capable and a plain provider; red without the fix); `::test_tools_plus_structured_output_hands_the_host_event_to_both_passes` (red when the conversion moves above the
hybrid branch); `tests/providers` + `tests/structured`: 1774 passed.
