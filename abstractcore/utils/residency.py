"""Host-level model-server residency sweep.

Answers "which models are resident on this host's LOCAL model servers"
(Ollama, LM Studio) without constructing model-bound provider instances.
In-process providers (MLX, HuggingFace) have no server to ask — their
residency lives on the owning provider instance (`list_loaded_models`).

ADR 0008: residency stays provider-owned truth — this module only fans out to
the provider classmethods that query each server's own running-model API. An
unreachable server contributes nothing (unknown stays unknown, never guessed).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# The providers the host sweep covers (the only ones that can contribute
# sweep records). Public: consumers (server routes, abstractruntime) key
# their "can the sweep ever answer this?" checks off this tuple.
SWEEP_PROVIDERS: Tuple[str, ...] = ("ollama", "lmstudio")


def normalize_sweep_model(name: Any) -> str:
    """Case-normalized model name for sweep dedup/filter comparisons.

    Strips Ollama's `:latest` alias — `name` and `name:latest` are the same
    model on an Ollama server."""
    model_s = str(name or "").strip().lower()
    if model_s.endswith(":latest"):
        model_s = model_s[: -len(":latest")]
    return model_s


def sweep_models_match(
    provider: Any,
    model: Any,
    sweep_record: Optional[Mapping[str, Any]] = None,
    instance_ids: Optional[Sequence[str]] = None,
) -> bool:
    """Does a registry/runtime record (`provider`, `model`) name the same
    resident model as a sweep row? The single source of truth for sweep
    dedup — uses the same alias rules the providers themselves use.

    - All providers: `normalize_sweep_model` equality (case + `:latest`).
    - LM Studio: additionally the provider's own substring resolution
      (`_native_rest_loaded_instance_ids_for_model`): a record naming a
      variant of the server key / a loaded instance id is the SAME resident
      model — two rows would double-count its memory.

    `instance_ids` overrides the sweep row's `provider_instance_ids` when the
    caller holds them separately."""
    provider_s = str(provider or "").strip().lower()
    record: Mapping[str, Any] = sweep_record if isinstance(sweep_record, Mapping) else {}
    reg = normalize_sweep_model(model)
    sweep_model = normalize_sweep_model(record.get("model"))
    if not reg or not sweep_model:
        return False
    if reg == sweep_model:
        return True
    if provider_s == "lmstudio":
        ids = instance_ids if instance_ids is not None else record.get("provider_instance_ids")
        candidates = [sweep_model] + [
            str(i).strip().lower() for i in ids or [] if isinstance(i, str)
        ]
        return any(reg == c or reg in c for c in candidates)
    return False


def sweep_loaded_models(timeout_s: float = 2.0) -> List[Dict[str, Any]]:
    """Best-effort sweep of local model servers for resident models.

    Each record is the provider's normalized server enumeration tagged with
    `source: "provider_server"`. A provider whose server is unreachable or
    errors is silently skipped; this function never raises.
    """
    records: List[Dict[str, Any]] = []
    try:
        from ..providers.lmstudio_provider import LMStudioProvider
        from ..providers.ollama_provider import OllamaProvider
    except Exception:
        return records

    for provider_cls in (OllamaProvider, LMStudioProvider):
        try:
            provider_records = provider_cls.list_server_loaded_models(timeout_s=timeout_s)
        except Exception:
            continue
        for record in provider_records or []:
            if not isinstance(record, dict):
                continue
            tagged = dict(record)
            tagged["source"] = "provider_server"
            records.append(tagged)
    return records
