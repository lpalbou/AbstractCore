#!/usr/bin/env python3
"""Benchmark durable memory bloc cache reuse for one local provider at a time.

The benchmark intentionally runs a single provider/model per process so large local
models are released when the process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from abstractcore import create_llm, ensure_bloc_kv_artifact, load_bloc_kv_artifact
from abstractcore.core.file_blocs import FileBlocStore
from abstractcore.core.file_boxes import FileBox, render_file_box_message


EXPECTED = {
    "launch_window": "Tuesday at 09:30 UTC",
    "inspector": "Mira Chen",
    "checksum": "ACORE-7421",
}

QUESTION = (
    "Using only the attached memory bloc, output exactly one line in this format and no other text: "
    "launch_window=<value>; inspector=<value>; checksum=<value>"
)


def _default_gguf_model() -> str:
    candidates = [
        "~/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/"
        "snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "~/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/"
        "snapshots/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-Q4_K_M.gguf",
        "~/.lmstudio/models/bartowski/mlabonne_Qwen3-4B-abliterated-GGUF/"
        "mlabonne_Qwen3-4B-abliterated-Q4_K_M.gguf",
        "~/.lmstudio/models/bartowski/mlabonne_Qwen3-0.6B-abliterated-GGUF/"
        "mlabonne_Qwen3-0.6B-abliterated-Q4_K_M.gguf",
    ]
    for candidate in candidates:
        local = Path(candidate).expanduser()
        if local.exists():
            return str(local)
    return "unsloth/Qwen3.5-0.8B-GGUF"


CASES: Dict[str, Dict[str, Any]] = {
    "mlx": {
        "provider": "mlx",
        "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "kwargs": {"max_tokens": 4096, "max_output_tokens": 96},
        "generate_kwargs": {"max_output_tokens": 96, "temperature": 0.1, "seed": 123, "thinking": "off"},
    },
    "hf-transformers": {
        "provider": "huggingface",
        "model": "Qwen/Qwen3.5-4B",
        "kwargs": {"max_tokens": 4096, "max_output_tokens": 256, "device": "mps"},
        "generate_kwargs": {"max_output_tokens": 256, "temperature": 0.1, "seed": 123, "thinking": "off"},
    },
    "hf-gguf": {
        "provider": "huggingface",
        "model": _default_gguf_model(),
        "kwargs": {"max_tokens": 4096, "max_output_tokens": 96, "n_gpu_layers": -1},
        "generate_kwargs": {"max_output_tokens": 96, "temperature": 0.1, "seed": 123, "thinking": "off"},
    },
}


def _benchmark_content(repetitions: int) -> str:
    header = f"""Memory Bloc Validation Dossier

Authoritative facts:
- Launch window: {EXPECTED["launch_window"]}
- Inspector: {EXPECTED["inspector"]}
- Checksum: {EXPECTED["checksum"]}

Answering rule: if asked for the launch window, inspector, and checksum, use the
authoritative facts above exactly.
Output rule: do not explain reasoning. Do not write a thinking process. Return only the requested
one-line key/value answer.

"""
    filler = []
    for idx in range(1, repetitions + 1):
        filler.append(
            "Control paragraph "
            f"{idx:03d}: telemetry record, routing note, and archived context. "
            "This paragraph is stable filler used only to make the bloc prefix large enough "
            "for prompt-cache timing. It does not change the authoritative facts."
        )
    return header + "\n".join(filler) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    return str(content or "")


def _score_answer(text: str) -> Dict[str, Any]:
    normalized = str(text or "").lower()
    checks = {
        "has_tuesday": "tuesday" in normalized,
        "has_0930": "09:30" in normalized or "0930" in normalized,
        "has_mira": "mira" in normalized,
        "has_acore_7421": "acore-7421" in normalized or "acore 7421" in normalized,
    }
    return {"correct": all(checks.values()), "checks": checks}


def _time_call(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def run_case(case: str, *, repetitions: int, output: Path | None, model_override: str | None = None) -> Dict[str, Any]:
    cfg = CASES[case]
    model = str(model_override or cfg["model"])
    root = Path(tempfile.mkdtemp(prefix=f"abstractcore-bloc-cache-{case}-"))
    store = FileBlocStore(root_dir=root)
    content = _benchmark_content(repetitions)
    sha = _sha256_text(content)
    record = store.upsert(
        file_meta={
            "path": f"benchmarks/{case}/durable-memory-bloc.txt",
            "sha256": sha,
            "content_sha256": sha,
            "media_type": "text/plain",
            "size_bytes": len(content.encode("utf-8")),
            "content_length": len(content),
            "format": "text",
        },
        content=content,
    )
    file_box_prompt = render_file_box_message(
        FileBox(
            path=record.path,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            mtime_ns=record.mtime_ns,
            sha256=record.sha256,
            content=content,
            content_sha256=record.content_sha256,
            format=record.format,
            content_length=record.content_length,
            estimated_tokens=record.estimated_tokens,
        )
    )

    os.environ.setdefault("ABSTRACTCORE_BLOC_KV_DEBUG", "1")

    llm, load_s = _time_call(
        lambda: create_llm(cfg["provider"], model=model, **dict(cfg.get("kwargs") or {}))
    )
    _, warmup_s = _time_call(
        lambda: llm.generate("Reply with exactly: ready", max_output_tokens=8, temperature=0.1, seed=11)
    )

    gen_kwargs = dict(cfg.get("generate_kwargs") or {})
    thinking = gen_kwargs.get("thinking")
    full_process_key = f"bench:{case}:full-process"
    cached_process_key = f"bench:{case}:cached-process"
    full_processing_probe: Dict[str, Any] = {}
    cached_processing_probe: Dict[str, Any] = {}

    try:
        llm.prompt_cache_set(full_process_key, make_default=False)
        _, full_processing_s = _time_call(
            lambda: llm.prompt_cache_update(
                full_process_key,
                messages=[{"role": "user", "content": file_box_prompt}],
                prompt=QUESTION,
                add_generation_prompt=True,
                thinking=thinking,
            )
        )
        full_processing_probe = {
            "ok": True,
            "cache_key": full_process_key,
            "processing_s": round(full_processing_s, 4),
            "token_count": llm.prompt_cache_token_count(full_process_key),
        }
    except Exception as e:
        full_processing_probe = {
            "ok": False,
            "cache_key": full_process_key,
            "error": str(e),
        }

    uncached_response, uncached_s = _time_call(
        lambda: llm.generate(
            QUESTION,
            messages=[{"role": "user", "content": file_box_prompt}],
            **gen_kwargs,
        )
    )

    ensured, ensure_s = _time_call(
        lambda: ensure_bloc_kv_artifact(provider=llm, store=store, record=record, force_rebuild=True, debug=True)
    )
    loaded, artifact_load_s = _time_call(
        lambda: load_bloc_kv_artifact(provider=llm, store=store, record=record, key=f"bench:{case}", debug=True)
    )
    try:
        _, fork_s = _time_call(
            lambda: llm.prompt_cache_fork(loaded.key, cached_process_key, make_default=False)
        )
        _, cached_processing_s = _time_call(
            lambda: llm.prompt_cache_update(
                cached_process_key,
                prompt=QUESTION,
                add_generation_prompt=True,
                thinking=thinking,
            )
        )
        cached_processing_probe = {
            "ok": True,
            "cache_key": cached_process_key,
            "fork_s_excluded": round(fork_s, 4),
            "processing_s": round(cached_processing_s, 4),
            "token_count": llm.prompt_cache_token_count(cached_process_key),
        }
    except Exception as e:
        cached_processing_probe = {
            "ok": False,
            "cache_key": cached_process_key,
            "error": str(e),
        }

    cached_response, cached_s = _time_call(
        lambda: llm.generate(
            QUESTION,
            prompt_cache_binding=loaded.prompt_cache_binding,
            **gen_kwargs,
        )
    )

    uncached_text = _response_text(uncached_response)
    cached_text = _response_text(cached_response)
    speedup = uncached_s / cached_s if cached_s > 0 else None
    artifact_size = ensured.artifact_path.stat().st_size if ensured.artifact_path.exists() else None
    full_processing_s = full_processing_probe.get("processing_s")
    cached_processing_s = cached_processing_probe.get("processing_s")
    processing_speedup = (
        float(full_processing_s) / float(cached_processing_s)
        if isinstance(full_processing_s, (int, float))
        and isinstance(cached_processing_s, (int, float))
        and float(cached_processing_s) > 0
        else None
    )

    result = {
        "case": case,
        "provider": cfg["provider"],
        "model": model,
        "model_load_s": round(load_s, 4),
        "warmup_s": round(warmup_s, 4),
        "uncached_generation_s": round(uncached_s, 4),
        "ensure_compile_s": round(ensure_s, 4),
        "artifact_load_s": round(artifact_load_s, 4),
        "cached_generation_s": round(cached_s, 4),
        "full_prompt_processing_s": full_processing_probe.get("processing_s"),
        "cached_suffix_processing_s": cached_processing_probe.get("processing_s"),
        "processing_speedup_excluding_load_and_decode": round(processing_speedup, 4)
        if processing_speedup is not None
        else None,
        "generation_speedup": round(speedup, 4) if speedup is not None else None,
        "reuse_including_artifact_load_s": round(artifact_load_s + cached_s, 4),
        "reuse_including_artifact_load_speedup": round(uncached_s / (artifact_load_s + cached_s), 4)
        if artifact_load_s + cached_s > 0
        else None,
        "artifact_path": str(ensured.artifact_path),
        "artifact_size_bytes": artifact_size,
        "manifest_path": str(ensured.manifest_path),
        "cache_backend": ensured.manifest.cache_backend,
        "artifact_format": ensured.manifest.artifact_format,
        "binding_id": ensured.binding_id,
        "binding_validated": bool(llm.prompt_cache_validate_binding(loaded.prompt_cache_binding["key"], loaded.prompt_cache_binding)),
        "token_count": ensured.manifest.token_count,
        "debug": {
            "ensure": ensured.debug,
            "load": loaded.debug,
            "full_processing_probe": full_processing_probe,
            "cached_processing_probe": cached_processing_probe,
        },
        "uncached_answer": uncached_text,
        "cached_answer": cached_text,
        "uncached_metadata": getattr(uncached_response, "metadata", None),
        "cached_metadata": getattr(cached_response, "metadata", None),
        "uncached_score": _score_answer(uncached_text),
        "cached_score": _score_answer(cached_text),
        "store_root": str(root),
    }
    result["semantic_correct"] = bool(
        result["uncached_score"]["correct"] and result["cached_score"]["correct"]
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Override the default model/path for the selected case.")
    parser.add_argument(
        "--allow-incorrect",
        action="store_true",
        help="Do not fail the process if uncached or cached generation misses the expected facts.",
    )
    args = parser.parse_args()

    result = run_case(
        args.case,
        repetitions=max(1, args.repetitions),
        output=args.output,
        model_override=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.allow_incorrect and not result.get("semantic_correct"):
        raise SystemExit(
            "semantic correctness failed: "
            f"uncached={result['uncached_score']['correct']} cached={result['cached_score']['correct']}"
        )


if __name__ == "__main__":
    main()
