"""Live twin of `test_mlx_stream_prompt_cache_parity.py`: a real (tiny) MLX model.

Same prompt, same prompt-cache key, two turns (cold, then warm): the streamed
call's LAST chunk carries exactly the sync call's `metadata["prompt_cache"]`,
`usage`, `finish_reason` and text. Loads the smallest MLX model of the local HF
cache READ-ONLY from its snapshot directory (no download, no write to the real
home); skipped when that model or the MLX stack is absent.
"""

from __future__ import annotations

import glob
import importlib.util
import os
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.real_home("loads a tiny MLX model read-only from the operator's HF cache snapshot dir")

LIVE_REPO = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


def _snapshot_dir() -> Optional[str]:
    if not all(importlib.util.find_spec(m) for m in ("mlx", "mlx_lm")):
        return None
    home = os.environ["ABSTRACT_TEST_REAL_HOME"]
    hits = glob.glob(os.path.join(
        home, ".cache", "huggingface", "hub", f"models--{LIVE_REPO.replace('/', '--')}",
        "snapshots", "*", "model.safetensors",
    ))
    return os.path.dirname(sorted(hits)[0]) if hits else None


_SNAPSHOT = _snapshot_dir()


@pytest.mark.skipif(_SNAPSHOT is None, reason=f"{LIVE_REPO} (or the MLX stack) not available locally")
def test_live_sync_and_stream_record_identical_prompt_cache_on_both_turns():
    from abstractcore import create_llm

    def run(stream: bool) -> List[Dict[str, Any]]:
        llm = create_llm("mlx", model=_SNAPSHOT)
        records = []
        try:
            for turn in ("Say hi.", "Say bye."):
                r = llm.generate(turn, messages=[], prompt_cache_key="parity", stream=stream,
                                 max_output_tokens=6, temperature=0)
                if stream:
                    chunks = list(r)
                    last = chunks[-1]
                    records.append({
                        "prompt_cache": (last.metadata or {}).get("prompt_cache"),
                        "usage": last.usage,
                        "finish_reason": last.finish_reason,
                        "content": "".join(c.content or "" for c in chunks),
                    })
                else:
                    records.append({
                        "prompt_cache": (r.metadata or {}).get("prompt_cache"),
                        "usage": r.usage,
                        "finish_reason": r.finish_reason,
                        "content": r.content,
                    })
        finally:
            llm.unload_model(llm.model)
        return records

    sync, streamed = run(False), run(True)
    assert sync[0]["prompt_cache"]["outcome"] == "cold"
    assert sync[1]["prompt_cache"]["outcome"].startswith("hit")
    for s, t in zip(sync, streamed):
        assert t["prompt_cache"] == s["prompt_cache"]
        assert t["usage"] == s["usage"]
        assert t["finish_reason"] == s["finish_reason"]
        assert t["content"] == s["content"]
