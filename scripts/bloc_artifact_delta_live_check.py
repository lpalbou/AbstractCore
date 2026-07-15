"""Live validation for 0819: bloc KV artifacts join the MLX delta lattice.

Two-phase, TWO PROCESSES (the artifact round-trip is the point — the record
must survive a process death, not an in-memory store):

  Phase A (--phase compile): real MLX model, temp FileBlocStore, upsert a
  content text, compile the KV artifact, then read the safetensors header
  back and assert `fed_token_ids` was persisted. Writes a handoff JSON.

  Phase B (--phase ask): FRESH process. Loads the artifact via
  load_bloc_kv_artifact, then issues a RUNTIME-SHAPED full-context generate
  (messages=[bloc user message], prompt=question) with the loaded key and
  asserts from response.metadata["prompt_cache"]:
    - outcome == hit_extend (artifact ADMITTED, not bypassed)
    - cached_tokens ~= the artifact's token count (bloc prefix served from KV)
    - fed_tokens is small (the question suffix, never the bloc again)
    - the answer passes a content-correctness gate.

Run:
  python scripts/bloc_artifact_delta_live_check.py --phase compile
  python scripts/bloc_artifact_delta_live_check.py --phase ask
  (or --phase both: spawns phase B as a subprocess for true process isolation)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# The WORKING TREE is what this check validates — `python scripts/foo.py`
# puts scripts/ (not the repo root) first on sys.path, so a plain import
# would silently exercise the INSTALLED site-packages copy (live incident:
# the first run of this very script compiled an artifact through the
# pre-fix installed package and "disproved" the fix).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Pure-attention model REQUIRED for the hit_extend assertion: hybrid
# architectures (Qwen3.5-class linear-attention layers) refuse trim, so the
# artifact lane honestly bypasses there (run with such a model to see the
# telemetry explain exactly that).
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
WORKDIR = Path("/tmp/bloc_delta_live_check")
HANDOFF = WORKDIR / "handoff.json"

CONTENT = (
    "Project Meridian status report, July 2026.\n"
    "The Meridian pipeline ingests telemetry from 412 ground stations.\n"
    "Its checkpoint cadence is every 90 seconds, and the recovery target is 12 seconds.\n"
    "The lead engineer is Dr. Yuki Tanaka, and the fallback coordinator is Marco Silva.\n"
    "The Q3 budget allocation is 2.4 million euros, of which 30 percent is reserved for storage.\n"
) * 6  # repeat to make the bloc prefix meaningfully large


def phase_compile() -> int:
    import hashlib

    from abstractcore import create_llm
    from abstractcore.core.file_blocs import FileBlocStore
    from abstractcore.core.bloc_kv import ensure_bloc_kv_artifact

    WORKDIR.mkdir(parents=True, exist_ok=True)
    store = FileBlocStore(root_dir=WORKDIR / "blocs")
    source_path = WORKDIR / "meridian_report.txt"
    source_path.write_text(CONTENT, encoding="utf-8")
    sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    record = store.upsert(
        file_meta={
            "path": str(source_path),
            "media_type": "text",
            "size_bytes": len(CONTENT.encode("utf-8")),
            "mtime_ns": source_path.stat().st_mtime_ns,
            "sha256": sha,
            "content_sha256": hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
            "format": "text/plain",
            "content_length": len(CONTENT),
            "estimated_tokens": len(CONTENT) // 4,
        },
        content=CONTENT,
        relpath_base=WORKDIR,
    )
    print(f"[A] bloc upserted sha={record.sha256[:12]} chars={len(CONTENT)}")

    llm = create_llm("mlx", model=MODEL)
    t0 = time.time()
    result = ensure_bloc_kv_artifact(provider=llm, store=store, record=record)
    compile_s = time.time() - t0
    print(f"[A] artifact compiled={result.compiled} in {compile_s:.1f}s -> {result.artifact_path}")

    # Assert the record was persisted into the artifact metadata — read it
    # back through the SAME reader the provider load path uses (mlx_lm
    # prefixes metadata keys in the raw safetensors header and un-prefixes
    # them here; raw-header reads see '1.fed_token_ids').
    from mlx_lm.models.cache import load_prompt_cache

    _, header_meta = load_prompt_cache(str(result.artifact_path), return_metadata=True)
    raw_record = (header_meta or {}).get("fed_token_ids")
    assert raw_record, "FAIL: fed_token_ids missing from artifact metadata (P0-1 not fixed)"
    record_ids = json.loads(raw_record)
    assert isinstance(record_ids, list) and len(record_ids) > 50, "record implausibly small"
    print(f"[A] PASS: artifact metadata carries fed_token_ids ({len(record_ids)} ids)")

    HANDOFF.write_text(json.dumps({
        "bloc_sha256": record.sha256,
        "store_root": str(WORKDIR / "blocs"),
        "artifact_path": str(result.artifact_path),
        "record_len": len(record_ids),
        "token_count": result.manifest.token_count,
    }), encoding="utf-8")
    print(f"[A] handoff written -> {HANDOFF}")
    return 0


def phase_ask() -> int:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    from abstractcore import create_llm
    from abstractcore.core.file_blocs import FileBlocStore
    from abstractcore.core.bloc_kv import (
        _read_bloc_content,
        _render_attached_file_box_recipe,
        _resolve_record,
        load_bloc_kv_artifact,
    )

    store = FileBlocStore(root_dir=Path(handoff["store_root"]))
    record = _resolve_record(store=store, record=None, sha256=handoff["bloc_sha256"], bloc_id=None)
    llm = create_llm("mlx", model=MODEL)

    t0 = time.time()
    loaded = load_bloc_kv_artifact(
        provider=llm, store=store, record=record, key="bloc:live-check", make_default=False,
    )
    load_s = time.time() - t0
    print(f"[B] artifact loaded={loaded.loaded} key={loaded.key} in {load_s:.1f}s")

    key_record = llm._fed_token_ids_for_key(loaded.key)
    assert key_record, "FAIL: loaded key has no fed-token record (load reconstruction broken)"
    print(f"[B] PASS: loaded key carries the record ({len(key_record)} ids)")

    # Runtime-shaped full-context call: the bloc rides as a COMPLETE prior
    # user message (byte-identical to what the compile lane fed); the
    # question is the next message. The delta lane should serve the bloc
    # prefix from KV and feed only the question suffix.
    content = _read_bloc_content(store, record)
    rendered = _render_attached_file_box_recipe(provider=llm, record=record, content=content)
    question = "According to the attached report, who is the lead engineer and what is the checkpoint cadence? Answer in one sentence."

    t1 = time.time()
    response = llm.generate(
        question,
        messages=[{"role": "user", "content": rendered.file_box_prompt}],
        prompt_cache_key=loaded.key,
        max_output_tokens=2048,
        temperature=0.0,
    )
    warm_s = time.time() - t1

    telemetry = (response.metadata or {}).get("prompt_cache") or {}
    print(f"[B] telemetry: {json.dumps(telemetry, indent=2)}")
    print(f"[B] warm call: {warm_s:.2f}s")
    print(f"[B] answer: {response.content[:300]}")

    assert telemetry.get("outcome") == "hit_extend", (
        f"FAIL: expected hit_extend, got {telemetry.get('outcome')!r} "
        f"(reason={telemetry.get('degraded_reason')!r})"
    )
    cached = int(telemetry.get("cached_tokens") or 0)
    fed = int(telemetry.get("fed_tokens") or 0)
    assert cached >= int(handoff["record_len"]) - 2, (
        f"FAIL: cached_tokens {cached} does not cover the bloc prefix ({handoff['record_len']})"
    )
    assert fed < cached / 4, f"FAIL: fed_tokens {fed} is not a small suffix of cached {cached}"
    assert telemetry.get("bloc_sha256") == handoff["bloc_sha256"], "binding identity missing"

    answer = (response.content or "").lower()
    assert "tanaka" in answer, f"content-correctness gate FAILED: {response.content[:200]!r}"
    assert "90" in answer, f"content-correctness gate FAILED (cadence): {response.content[:200]!r}"

    print(
        f"[B] PASS: bloc served from artifact KV — cached={cached} fed={fed} "
        f"({100.0 * cached / max(1, cached + fed):.1f}% of prefill skipped), answer correct."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["compile", "ask", "both"], default="both")
    args = parser.parse_args()
    if args.phase == "compile":
        return phase_compile()
    if args.phase == "ask":
        return phase_ask()
    rc = subprocess.call([sys.executable, __file__, "--phase", "compile"])
    if rc != 0:
        return rc
    # Fresh process for phase B: the round-trip through disk is the point.
    return subprocess.call([sys.executable, __file__, "--phase", "ask"])


if __name__ == "__main__":
    sys.exit(main())
