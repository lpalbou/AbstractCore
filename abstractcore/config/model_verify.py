"""Fresh-install inference check: does an installed model actually answer?

`abstractcore models verify <artifact>` (and the slow test
`tests/providers/test_mlx_fresh_install_inference_slow.py`) load ONE installed
model through its provider exactly as a fresh install would -- the default
configuration, no explicit `speculation=`, loading never downloads -- ask a
fixed factual question at temperature 0, and check the answer.

Why this exists: an MTP-preserving MLX checkpoint loaded through mlx-lm
<= 0.31.3 produced fluent-looking GARBAGE with no error anywhere, and a model
downloaded without its MTP companion quietly ran without acceleration. Both
passed every unit test. The only check that catches that class is one that
loads the real artifacts and reads the real answer.

For a model with a registry companion (`model_materializer.companion_artifacts`)
it also requires `speculation.used == True`: the companion was downloaded with
the model, so MTP must actually run.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional

VERIFY_PROMPT = "What is the boiling point of water at sea level in degrees Celsius? Answer in one sentence."
VERIFY_EXPECT = "100"
VERIFY_MAX_TOKENS = 64


def verify_inference(
    provider: str,
    artifact: str,
    *,
    prompt: str = VERIFY_PROMPT,
    expect: str = VERIFY_EXPECT,
    max_output_tokens: int = VERIFY_MAX_TOKENS,
    require_mtp: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load, ask, check, unload. Never downloads. Returns a report dict (`ok` is the verdict).

    `require_mtp=None` means: require `speculation.used` exactly when the
    artifact has a registry companion.
    """

    from .model_materializer import PRESENCE_INSTALLED, companion_artifacts, probe

    checks: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "provider": provider,
        "artifact": artifact,
        "prompt": prompt,
        "expect": expect,
        "ok": False,
        "checks": checks,
        "content": None,
        "speculation": None,
        "companions": companion_artifacts(provider, artifact),
    }
    if require_mtp is None:
        require_mtp = bool(report["companions"])

    presence = probe(provider, artifact)
    checks.append(
        {
            "name": "installed",
            "ok": presence.status == PRESENCE_INSTALLED,
            "detail": presence.detail or presence.location or presence.status,
        }
    )
    if presence.status != PRESENCE_INSTALLED:
        report["summary"] = f"not installed ({presence.status}): {presence.detail or ''}".strip()
        if presence.instruction:
            report["instruction"] = presence.instruction
        return report

    from .. import create_llm

    llm = None
    try:
        started = time.time()
        llm = create_llm(provider, model=artifact)
        report["load_s"] = round(time.time() - started, 2)
        report["lane"] = {
            "mtp_preserving_checkpoint": bool(getattr(llm, "_mtp_preserving_checkpoint", False)),
            "mlx_vlm": getattr(llm, "_mtp_processor", None) is not None,
            "mtp_active": bool(getattr(llm, "_mtp_active", False)),
        }
        response = llm.generate(
            None,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            seed=7,
            thinking=False,
            max_output_tokens=max_output_tokens,
        )
        content = str(getattr(response, "content", "") or "")
        speculation = (getattr(response, "metadata", None) or {}).get("speculation")
        report["content"] = content
        report["speculation"] = speculation
        checks.append(
            {
                "name": "answer",
                "ok": expect in content,
                "detail": f"expected {expect!r} in the answer",
            }
        )
        if require_mtp:
            used = isinstance(speculation, dict) and speculation.get("used") is True
            why = (speculation or {}).get("message") or (speculation or {}).get("reason") if isinstance(speculation, dict) else None
            checks.append(
                {
                    "name": "mtp_used",
                    "ok": used,
                    "detail": "speculation.used is true" if used else f"MTP did not run: {why or 'no speculation outcome reported'}",
                }
            )
    except Exception as exc:
        checks.append({"name": "load_and_generate", "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    finally:
        if llm is not None:
            try:
                llm.unload_model(artifact)
            except Exception as exc:  # the verdict stands; the unload failure is reported
                checks.append({"name": "unload", "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
            del llm
        gc.collect()
        try:
            import mlx.core as mx  # type: ignore

            mx.clear_cache()
        except Exception:
            pass

    report["ok"] = all(c["ok"] for c in checks)
    failed = [c["name"] for c in checks if not c["ok"]]
    report["summary"] = "answers sensibly" + (" with MTP" if require_mtp else "") if not failed else "FAILED: " + ", ".join(failed)
    return report
