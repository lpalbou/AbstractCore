"""
Local Qwen3.5 thinking-control probe (LM Studio + Ollama + HuggingFace GGUF).

Run:
  python examples/reasoning/local_qwen3_5_thinking_probe.py

This script is intentionally small and self-contained so it can be run on a dev machine
to validate:
- `thinking="off"` suppresses reasoning traces
- `thinking="low|medium|high"` maps to the cleanest available backend knob (best-effort)

It uses AbstractCore providers so results reflect the framework's unified `thinking=`
parameter mapping.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _preview(text: Optional[str], *, max_chars: int = 140) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    s = re.sub(r"\s+", " ", text.strip())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1] + "…"


@dataclass(frozen=True)
class ProbeCase:
    provider: str
    model: str
    thinking: str


def _lmstudio_list_models(*, base_url: str) -> List[str]:
    import httpx

    url = base_url.rstrip("/") + "/models"
    resp = httpx.get(url, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    out: List[str] = []
    for item in (data.get("data") or []):
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            out.append(item["id"].strip())
    return out


def _lmstudio_list_models_native_v1(*, rest_base_url: str) -> List[str]:
    """List models via LM Studio native REST API (/api/v1/models)."""
    import httpx

    url = rest_base_url.rstrip("/") + "/api/v1/models"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    out: List[str] = []
    # Native v1 may return {"data":[...]} similar to OpenAI, or {"models":[...]}.
    items = None
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or data.get("items")
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("model") or item.get("instance_id")
            if isinstance(mid, str) and mid.strip():
                out.append(mid.strip())
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _lmstudio_load_model_native_v1(
    *, rest_base_url: str, model: str, context_length: Optional[int] = None, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load a model via LM Studio native REST API (/api/v1/models/load)."""
    import httpx

    url = rest_base_url.rstrip("/") + "/api/v1/models/load"
    payload: Dict[str, Any] = {"model": model}
    if context_length is not None:
        payload["context_length"] = int(context_length)
    if isinstance(extra, dict) and extra:
        payload.update(extra)
    resp = httpx.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {"raw": data}


def _lmstudio_unload_model_native_v1(*, rest_base_url: str, model: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Unload a model via LM Studio native REST API (/api/v1/models/unload)."""
    import httpx

    unload_url = rest_base_url.rstrip("/") + "/api/v1/models/unload"

    # LM Studio uses instance ids for lifecycle management. We discover the instance id
    # by calling the native model listing endpoint, then unload by instance_id.
    #
    # Some builds appear to accept the model id directly as an instance id; try that first.
    direct_payload: Dict[str, Any] = {"instance_id": model}
    if isinstance(extra, dict) and extra:
        direct_payload.update(extra)
    direct = httpx.post(unload_url, json=direct_payload, timeout=30.0)
    if direct.status_code < 400:
        try:
            out = direct.json()
        except Exception:  # noqa: BLE001
            out = {"raw": direct.text}
        return out if isinstance(out, dict) else {"raw": out}

    list_url = rest_base_url.rstrip("/") + "/api/v1/models"
    resp = httpx.get(list_url, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    items = None
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or data.get("items")
    if not isinstance(items, list):
        items = []

    def _coerce_str(x: Any) -> str:
        return x.strip() if isinstance(x, str) else ""

    needle = model.strip().lower()
    candidates: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        iid = _coerce_str(item.get("instance_id") or item.get("instanceId") or item.get("instance"))
        if not iid:
            # If it's not an instance, we can't unload it.
            continue

        # Try several likely fields for "which model is this instance running?"
        fields: List[str] = []
        for k in ("id", "model", "name", "model_id", "modelId"):
            v = _coerce_str(item.get(k))
            if v:
                fields.append(v)
        nested = item.get("model") if isinstance(item.get("model"), dict) else {}
        if isinstance(nested, dict):
            for k in ("id", "name", "identifier"):
                v = _coerce_str(nested.get(k))
                if v:
                    fields.append(v)

        if any(needle == f.lower() for f in fields) or any(needle in f.lower() for f in fields):
            candidates.append(item)

    if not candidates:
        # Best-effort: if we can't find a loaded instance, treat as already unloaded.
        return {"note": f"no loaded instance found for {model!r}; assumed unloaded"}

    # Prefer the most recently loaded instance if timestamps exist, otherwise take first.
    def _sort_key(item: Dict[str, Any]) -> float:
        for k in ("loaded_at", "loadedAt", "created_at", "createdAt", "ts"):
            v = item.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    candidates.sort(key=_sort_key, reverse=True)
    chosen = candidates[0]
    instance_id = _coerce_str(chosen.get("instance_id") or chosen.get("instanceId") or chosen.get("instance"))
    if not instance_id:
        return {"note": f"no instance_id field found for {model!r}; assumed unloaded"}

    payload: Dict[str, Any] = {"instance_id": instance_id}
    if isinstance(extra, dict) and extra:
        payload.update(extra)

    unload_resp = httpx.post(unload_url, json=payload, timeout=120.0)
    if unload_resp.status_code >= 400:
        raise RuntimeError(
            f"LM Studio unload failed ({unload_resp.status_code}): {unload_resp.text.strip() or 'no response body'}"
        )
    try:
        out = unload_resp.json()
    except Exception:  # noqa: BLE001
        out = {"raw": unload_resp.text}
    return out if isinstance(out, dict) else {"raw": out}


def _choose_lmstudio_model_id(*, ids: Sequence[str], needle: str) -> Optional[str]:
    """Choose a model id for LM Studio by substring match with simple scoring."""
    needle_l = needle.strip().lower()
    if not needle_l:
        return None

    candidates: List[str] = [x for x in ids if isinstance(x, str) and needle_l in x.lower()]
    if not candidates:
        return None

    def score(mid: str) -> Tuple[int, int, int]:
        m = mid.lower()
        # Prefer official-ish ids.
        prefix = 0
        if m.startswith("qwen/") or m.startswith("qwen\\"):
            prefix = 3
        elif m.startswith("mlx-community/"):
            prefix = 2
        elif m.startswith("lmstudio-community/"):
            prefix = 1

        # Prefer non-specialized finetunes for baseline behavior.
        penalty = 0
        for bad in ("uncensored", "distilled", "abliterated", "aggressive", "omnicoder"):
            if bad in m:
                penalty += 2

        # Prefer simpler ids (often the canonical).
        length = len(mid)
        return (prefix - penalty, -length, -m.count("@"))

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _run_one(
    *,
    provider_name: str,
    model: str,
    thinking: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    base_url: Optional[str],
) -> Dict[str, Any]:
    started_ms = _now_ms()
    err: Optional[str] = None
    resp_content: Optional[str] = None
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    raw_message_keys: Optional[List[str]] = None
    raw_reasoning_len: Optional[int] = None
    raw_content_preview: Optional[str] = None
    raw_contains_think_end: Optional[bool] = None
    raw_contains_think_start: Optional[bool] = None
    raw_contains_thinking_process: Optional[bool] = None
    usage: Optional[Dict[str, Any]] = None

    llm = None
    try:
        if provider_name == "lmstudio":
            from abstractcore.providers.lmstudio_provider import LMStudioProvider

            llm = LMStudioProvider(model=model, base_url=base_url)
        elif provider_name == "ollama":
            from abstractcore.providers.ollama_provider import OllamaProvider

            llm = OllamaProvider(model=model, base_url=base_url)
        elif provider_name == "huggingface":
            # GGUF via llama-cpp-python; use CPU-only by default to avoid contending with other runs.
            from abstractcore.providers.huggingface_provider import HuggingFaceProvider

            llm = HuggingFaceProvider(model=model, n_gpu_layers=0)
        else:
            raise ValueError(f"unsupported provider: {provider_name!r}")

        resp = llm.generate(
            prompt,
            thinking=thinking,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        resp_content = getattr(resp, "content", None)
        finish_reason = getattr(resp, "finish_reason", None)
        usage_val = getattr(resp, "usage", None)
        usage = dict(usage_val) if isinstance(usage_val, dict) else None
        meta = getattr(resp, "metadata", None)
        if isinstance(meta, dict):
            reasoning_val = meta.get("reasoning")
            if isinstance(reasoning_val, str) and reasoning_val.strip():
                reasoning = reasoning_val.strip()
            req = meta.get("_provider_request")
            if isinstance(req, dict):
                maybe_payload = req.get("payload")
                if isinstance(maybe_payload, dict):
                    payload = maybe_payload

        raw = getattr(resp, "raw_response", None)
        if isinstance(raw, dict):
            # OpenAI-compatible servers: raw["choices"][0]["message"]
            try:
                choices = raw.get("choices")
                if isinstance(choices, list) and choices:
                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
                    if isinstance(msg, dict) and msg:
                        raw_message_keys = sorted([k for k in msg.keys() if isinstance(k, str)])
                        mc = msg.get("content")
                        if isinstance(mc, str) and mc.strip():
                            raw_content_preview = _preview(mc, max_chars=220)
                            raw_contains_think_end = "</think>" in mc
                            raw_contains_think_start = "<think>" in mc
                            raw_contains_thinking_process = "thinking process" in mc.lower()
                        for rk in ("reasoning", "thinking", "reasoning_content"):
                            rv = msg.get(rk)
                            if isinstance(rv, str) and rv.strip():
                                raw_reasoning_len = len(rv.strip())
                                break
                # Ollama: raw["message"]
                if raw_message_keys is None:
                    msg = raw.get("message") if isinstance(raw.get("message"), dict) else {}
                    if isinstance(msg, dict) and msg:
                        raw_message_keys = sorted([k for k in msg.keys() if isinstance(k, str)])
                        mc = msg.get("content")
                        if isinstance(mc, str) and mc.strip():
                            raw_content_preview = _preview(mc, max_chars=220)
                            raw_contains_think_end = "</think>" in mc
                            raw_contains_think_start = "<think>" in mc
                            raw_contains_thinking_process = "thinking process" in mc.lower()
                        for rk in ("reasoning", "thinking"):
                            rv = msg.get(rk)
                            if isinstance(rv, str) and rv.strip():
                                raw_reasoning_len = len(rv.strip())
                                break
            except Exception:
                raw_message_keys = None
                raw_reasoning_len = None
                raw_content_preview = None
                raw_contains_think_end = None
                raw_contains_think_start = None
                raw_contains_thinking_process = None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        # Local providers should aggressively release RAM/VRAM between runs.
        try:
            if provider_name == "huggingface" and llm is not None and hasattr(llm, "unload_model"):
                llm.unload_model(model)
        except Exception:
            pass

    took_ms = _now_ms() - started_ms
    return {
        "provider": provider_name,
        "model": model,
        "thinking": thinking,
        "ok": err is None,
        "error": err,
        "took_ms": took_ms,
        "finish_reason": finish_reason,
        "content_preview": _preview(resp_content),
        "reasoning_preview": _preview(reasoning, max_chars=140) if reasoning else "",
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "has_reasoning": bool(reasoning),
        "raw_message_keys": raw_message_keys,
        "raw_reasoning_chars": raw_reasoning_len or 0,
        "raw_content_preview": raw_content_preview or "",
        "raw_contains_think_start": bool(raw_contains_think_start),
        "raw_contains_think_end": bool(raw_contains_think_end),
        "raw_contains_thinking_process": bool(raw_contains_thinking_process),
        "usage": usage or {},
        "provider_payload": payload,
    }


def _run_one_preloaded(
    *,
    llm: Any,
    provider_name: str,
    model: str,
    thinking: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    started_ms = _now_ms()
    err: Optional[str] = None
    resp_content: Optional[str] = None
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    raw_message_keys: Optional[List[str]] = None
    raw_reasoning_len: Optional[int] = None
    raw_content_preview: Optional[str] = None
    raw_contains_think_end: Optional[bool] = None
    raw_contains_think_start: Optional[bool] = None
    raw_contains_thinking_process: Optional[bool] = None
    usage: Optional[Dict[str, Any]] = None

    try:
        resp = llm.generate(
            prompt,
            thinking=thinking,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        resp_content = getattr(resp, "content", None)
        finish_reason = getattr(resp, "finish_reason", None)
        usage_val = getattr(resp, "usage", None)
        usage = dict(usage_val) if isinstance(usage_val, dict) else None
        meta = getattr(resp, "metadata", None)
        if isinstance(meta, dict):
            reasoning_val = meta.get("reasoning")
            if isinstance(reasoning_val, str) and reasoning_val.strip():
                reasoning = reasoning_val.strip()
            req = meta.get("_provider_request")
            if isinstance(req, dict):
                maybe_payload = req.get("payload")
                if isinstance(maybe_payload, dict):
                    payload = maybe_payload

        raw = getattr(resp, "raw_response", None)
        if isinstance(raw, dict):
            try:
                choices = raw.get("choices")
                if isinstance(choices, list) and choices:
                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    msg = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}
                    if isinstance(msg, dict) and msg:
                        raw_message_keys = sorted([k for k in msg.keys() if isinstance(k, str)])
                        mc = msg.get("content")
                        if isinstance(mc, str) and mc.strip():
                            raw_content_preview = _preview(mc, max_chars=220)
                            raw_contains_think_end = "</think>" in mc
                            raw_contains_think_start = "<think>" in mc
                            raw_contains_thinking_process = "thinking process" in mc.lower()
                        for rk in ("reasoning", "thinking", "reasoning_content"):
                            rv = msg.get(rk)
                            if isinstance(rv, str) and rv.strip():
                                raw_reasoning_len = len(rv.strip())
                                break
            except Exception:
                raw_message_keys = None
                raw_reasoning_len = None
                raw_content_preview = None
                raw_contains_think_end = None
                raw_contains_think_start = None
                raw_contains_thinking_process = None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    took_ms = _now_ms() - started_ms
    return {
        "provider": provider_name,
        "model": model,
        "thinking": thinking,
        "ok": err is None,
        "error": err,
        "took_ms": took_ms,
        "finish_reason": finish_reason,
        "content_preview": _preview(resp_content),
        "reasoning_preview": _preview(reasoning, max_chars=140) if reasoning else "",
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "has_reasoning": bool(reasoning),
        "raw_message_keys": raw_message_keys,
        "raw_reasoning_chars": raw_reasoning_len or 0,
        "raw_content_preview": raw_content_preview or "",
        "raw_contains_think_start": bool(raw_contains_think_start),
        "raw_contains_think_end": bool(raw_contains_think_end),
        "raw_contains_thinking_process": bool(raw_contains_thinking_process),
        "usage": usage or {},
        "provider_payload": payload,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default="both",
        choices=["both", "all", "lmstudio", "ollama", "huggingface"],
        help="Which backends to probe ('both' = lmstudio+ollama; 'all' = lmstudio+ollama+huggingface)",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default="http://localhost:1234/v1",
        help="LM Studio OpenAI-compatible base URL (should include /v1)",
    )
    parser.add_argument(
        "--lmstudio-rest-base-url",
        default="http://localhost:1234",
        help="LM Studio native REST base URL (no /v1 suffix). Used for optional native load/list probes.",
    )
    parser.add_argument(
        "--lmstudio-load",
        default="",
        help="Optional: model id to load via native REST before probing (e.g. qwen/qwen3.5-4b).",
    )
    parser.add_argument(
        "--lmstudio-model",
        default="",
        help="Optional: probe a single LM Studio model id (skips /v1/models auto-selection)",
    )
    parser.add_argument(
        "--lmstudio-unload-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unload each LM Studio model via native REST after probing it (frees VRAM/RAM).",
    )
    parser.add_argument(
        "--lmstudio-unload-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only unload the specified LM Studio model via native REST, then exit.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--ollama-model",
        default="",
        help="Optional: probe a single Ollama model id (e.g. qwen3.5:9b)",
    )
    parser.add_argument(
        "--hf-model",
        default="",
        help="Optional: probe a single HuggingFace GGUF repo id (e.g. unsloth/Qwen3.5-0.8B-GGUF)",
    )
    parser.add_argument(
        "--include-122b",
        action="store_true",
        default=False,
        help="Include 122B probes (disabled by default to avoid disrupting other inferences).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Small-ish cap for fast probes (increase if your backend emits long thinking traces)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for determinism",
    )
    parser.add_argument(
        "--prompt",
        default="Compute (17*23) - (19*11). Reply with the integer result only.",
        help="Prompt to use for all probes",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    prompt = str(args.prompt or "").strip() or "Compute (17*23) - (19*11). Reply with the integer result only."
    # Primary abstraction contract we want to verify for Qwen3.5:
    # - "none" disables thinking (alias for "off")
    # - "low|medium|high" enable thinking with increasing budgets (best-effort per backend)
    thinking_settings = ["none", "low", "medium", "high"]

    results: List[Dict[str, Any]] = []

    if args.providers in {"both", "all", "lmstudio"} and args.lmstudio_unload_only:
        target = (args.lmstudio_model or args.lmstudio_load or "").strip()
        if not target:
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "results": [
                            {
                                "provider": "lmstudio",
                                "model": "",
                                "thinking": "",
                                "ok": False,
                                "error": "lmstudio_unload_only requires --lmstudio-model (or --lmstudio-load)",
                                "took_ms": 0,
                                "finish_reason": None,
                                "content_preview": "",
                                "reasoning_chars": 0,
                                "has_reasoning": False,
                                "provider_payload": None,
                            }
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 1
        try:
            unload_resp = _lmstudio_unload_model_native_v1(
                rest_base_url=args.lmstudio_rest_base_url,
                model=target,
            )
            print(json.dumps({"prompt": prompt, "results": [{"provider": "lmstudio", "model": target, "ok": True, "provider_payload": {"native_unload": unload_resp}}]}, indent=2, ensure_ascii=False))
            return 0
        except Exception as e:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "results": [
                            {
                                "provider": "lmstudio",
                                "model": target,
                                "ok": False,
                                "error": f"native_unload_failed: {type(e).__name__}: {e}",
                            }
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 1

    # Resolve LM Studio model ids dynamically so this script works with local naming.
    lmstudio_models: Dict[str, str] = {}
    if args.providers in {"both", "all", "lmstudio"}:
        # Optional native-load step: make a model.yaml virtual id visible to the OpenAI-compatible endpoint.
        if isinstance(args.lmstudio_load, str) and args.lmstudio_load.strip():
            try:
                load_resp = _lmstudio_load_model_native_v1(
                    rest_base_url=args.lmstudio_rest_base_url,
                    model=args.lmstudio_load.strip(),
                )
                results.append(
                    {
                        "provider": "lmstudio",
                        "model": args.lmstudio_load.strip(),
                        "thinking": "",
                        "ok": True,
                        "error": None,
                        "took_ms": 0,
                        "finish_reason": None,
                        "content_preview": "",
                        "reasoning_chars": 0,
                        "has_reasoning": False,
                        "provider_payload": {"native_load": load_resp},
                    }
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    {
                        "provider": "lmstudio",
                        "model": args.lmstudio_load.strip(),
                        "thinking": "",
                        "ok": False,
                        "error": f"native_load_failed: {type(e).__name__}: {e}",
                        "took_ms": 0,
                        "finish_reason": None,
                        "content_preview": "",
                        "reasoning_chars": 0,
                        "has_reasoning": False,
                        "provider_payload": None,
                    }
                )

        if isinstance(args.lmstudio_model, str) and args.lmstudio_model.strip():
            lmstudio_models["(single)"] = args.lmstudio_model.strip()
        else:
            ids = _lmstudio_list_models(base_url=args.lmstudio_base_url)
            for size in ("0.8b", "2b", "4b", "9b", "27b", "35b-a3b"):
                needle = f"qwen3.5-{size}"
                chosen = _choose_lmstudio_model_id(ids=ids, needle=needle)
                if chosen:
                    lmstudio_models[size] = chosen
            if args.include_122b:
                chosen = _choose_lmstudio_model_id(ids=ids, needle="qwen3.5-122b-a10b")
                if chosen:
                    lmstudio_models["122b-a10b"] = chosen

    ollama_models: Dict[str, str] = {
        "0.8b": "qwen3.5:0.8b",
        "2b": "qwen3.5:2b",
        "4b": "qwen3.5:4b",
        "9b": "qwen3.5:9b",
        "27b": "qwen3.5:27b",
        "35b": "qwen3.5:35b",
    }
    if args.include_122b:
        ollama_models["122b"] = "qwen3.5:122b"

    hf_models: Dict[str, str] = {
        "0.8b": "unsloth/Qwen3.5-0.8B-GGUF",
        "2b": "unsloth/Qwen3.5-2B-GGUF",
        "4b": "unsloth/Qwen3.5-4B-GGUF",
    }

    def _probe(provider_name: str, model: str, thinking: str, *, base_url: str) -> None:
        results.append(
            _run_one(
                provider_name=provider_name,
                model=model,
                thinking=thinking,
                prompt=prompt,
                max_output_tokens=int(args.max_output_tokens),
                temperature=float(args.temperature),
                base_url=base_url,
            )
        )

    if args.providers in {"both", "all", "lmstudio"}:
        if "(single)" in lmstudio_models:
            model_id = lmstudio_models["(single)"]
            for thinking in thinking_settings:
                _probe("lmstudio", model_id, thinking, base_url=args.lmstudio_base_url)
            if args.lmstudio_unload_after:
                try:
                    unload_resp = _lmstudio_unload_model_native_v1(
                        rest_base_url=args.lmstudio_rest_base_url,
                        model=model_id,
                    )
                    results.append(
                        {
                            "provider": "lmstudio",
                            "model": model_id,
                            "thinking": "",
                            "ok": True,
                            "error": None,
                            "took_ms": 0,
                            "finish_reason": None,
                            "content_preview": "",
                            "reasoning_chars": 0,
                            "has_reasoning": False,
                            "provider_payload": {"native_unload": unload_resp},
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    results.append(
                        {
                            "provider": "lmstudio",
                            "model": model_id,
                            "thinking": "",
                            "ok": False,
                            "error": f"native_unload_failed: {type(e).__name__}: {e}",
                            "took_ms": 0,
                            "finish_reason": None,
                            "content_preview": "",
                            "reasoning_chars": 0,
                            "has_reasoning": False,
                            "provider_payload": None,
                        }
                    )
        else:
            sizes = ["0.8b", "2b", "4b", "9b", "27b", "35b-a3b"]
            if args.include_122b:
                sizes.append("122b-a10b")
            for size in sizes:
                model_id = lmstudio_models.get(size)
                if not model_id:
                    results.append(
                        {
                            "provider": "lmstudio",
                            "model": f"(missing: qwen3.5-{size})",
                            "thinking": "",
                            "ok": False,
                            "error": "Model id not found in /v1/models",
                            "took_ms": 0,
                            "finish_reason": None,
                            "content_preview": "",
                            "reasoning_chars": 0,
                            "has_reasoning": False,
                            "provider_payload": None,
                        }
                    )
                    continue

                for thinking in thinking_settings:
                    _probe("lmstudio", model_id, thinking, base_url=args.lmstudio_base_url)
                if args.lmstudio_unload_after:
                    try:
                        unload_resp = _lmstudio_unload_model_native_v1(
                            rest_base_url=args.lmstudio_rest_base_url,
                            model=model_id,
                        )
                        results.append(
                            {
                                "provider": "lmstudio",
                                "model": model_id,
                                "thinking": "",
                                "ok": True,
                                "error": None,
                                "took_ms": 0,
                                "finish_reason": None,
                                "content_preview": "",
                                "reasoning_chars": 0,
                                "has_reasoning": False,
                                "provider_payload": {"native_unload": unload_resp},
                            }
                        )
                    except Exception as e:  # noqa: BLE001
                        results.append(
                            {
                                "provider": "lmstudio",
                                "model": model_id,
                                "thinking": "",
                                "ok": False,
                                "error": f"native_unload_failed: {type(e).__name__}: {e}",
                                "took_ms": 0,
                                "finish_reason": None,
                                "content_preview": "",
                                "reasoning_chars": 0,
                                "has_reasoning": False,
                                "provider_payload": None,
                            }
                        )

    if args.providers in {"both", "all", "ollama"}:
        if isinstance(args.ollama_model, str) and args.ollama_model.strip():
            model_id = args.ollama_model.strip()
            for thinking in thinking_settings:
                _probe("ollama", model_id, thinking, base_url=args.ollama_base_url)
        else:
            for _size_key, model_id in ollama_models.items():
                for thinking in thinking_settings:
                    _probe("ollama", model_id, thinking, base_url=args.ollama_base_url)

    if args.providers in {"all", "huggingface"}:
        from abstractcore.providers.huggingface_provider import HuggingFaceProvider

        def _probe_hf_model(model_id: str) -> None:
            try:
                hf = HuggingFaceProvider(model=model_id, n_gpu_layers=0)
            except Exception as e:  # noqa: BLE001
                results.append(
                    {
                        "provider": "huggingface",
                        "model": model_id,
                        "thinking": "",
                        "ok": False,
                        "error": f"init_failed: {type(e).__name__}: {e}",
                        "took_ms": 0,
                        "finish_reason": None,
                        "content_preview": "",
                        "reasoning_chars": 0,
                        "has_reasoning": False,
                        "provider_payload": None,
                    }
                )
                return

            try:
                for thinking in thinking_settings:
                    results.append(
                        _run_one_preloaded(
                            llm=hf,
                            provider_name="huggingface",
                            model=model_id,
                            thinking=thinking,
                            prompt=prompt,
                            max_output_tokens=int(args.max_output_tokens),
                            temperature=float(args.temperature),
                        )
                    )
            finally:
                try:
                    hf.unload_model(model_id)
                except Exception:
                    pass

        if isinstance(args.hf_model, str) and args.hf_model.strip():
            _probe_hf_model(args.hf_model.strip())
        else:
            for size in ("0.8b", "2b", "4b"):
                model_id = hf_models.get(size)
                if model_id:
                    _probe_hf_model(model_id)

    # Output JSON for easy copy/paste into backlog reports.
    print(json.dumps({"prompt": prompt, "results": results}, indent=2, ensure_ascii=False))

    # Exit code: non-zero if any probe failed.
    failed = [r for r in results if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
