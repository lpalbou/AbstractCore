"""Fresh-install inference check on REAL weights (slow, opt-in).

Unit tests could not see mission W2's bug: an MTP-preserving checkpoint loaded
through mlx-lm answered with fluent garbage and no error. This test loads a real
MLX repo the way a fresh install does -- default configuration, no explicit
`speculation=`, loading never downloads -- asks a fixed question and reads the
answer; for a repo with a registry MTP companion it also requires
`speculation.used == True`.

Opt-in because it needs the MLX stack, several GB of weights and minutes:

    ABSTRACTCORE_TEST_FRESH_INSTALL_REPO=mlx-works/Qwen3.5-9B-oQ4e-mtp \\
    HF_HOME=<a cache holding ONLY what `abstractcore models download mlx <repo>` fetched> \\
    python -m pytest -m slow tests/providers/test_mlx_fresh_install_inference_slow.py

(The same check as a command: `abstractcore models verify <repo>`.)
"""

from __future__ import annotations

import importlib.util
import os

import pytest

REPO = os.environ.get("ABSTRACTCORE_TEST_FRESH_INSTALL_REPO", "").strip()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not REPO, reason="set ABSTRACTCORE_TEST_FRESH_INSTALL_REPO to an installed MLX repo"),
    pytest.mark.skipif(
        not all(importlib.util.find_spec(m) for m in ("mlx", "mlx_lm", "mlx_vlm")),
        reason="requires the MLX stack",
    ),
]


def test_fresh_install_answers_sensibly_and_uses_its_companion():
    from abstractcore.config.model_verify import verify_inference

    report = verify_inference("mlx", REPO)
    assert report["checks"][0]["name"] == "installed" and report["checks"][0]["ok"], report
    assert "100" in (report["content"] or ""), f"garbage or wrong answer: {report['content']!r}"
    if report["companions"]:
        assert (report["speculation"] or {}).get("used") is True, report["speculation"]
    assert report["ok"], report
