"""END-TO-END bloc-cache correctness gate (gate v2, operator-specified).

Token equality proves the bytes line up. It does NOT prove the KV is reusable.
This file is the other half: a real model, a real 10k+ token prefix built as a
BLOC CHAIN, and answers that can only come from that prefix.

Gate design, in the order the checks must run — a later check is meaningless if
an earlier one fails:

  1. DETERMINISM CONTROL. Same bytes, fresh key, twice -> must match. A lane
     that is not self-deterministic makes every later comparison noise.
  2. CONTEXT-DEPENDENCE CONTROL, WITH ITS POSITIVE HALF. The same questions
     with the context REMOVED must FAIL, *while a general-knowledge question
     asked under the identical bare setup still ANSWERS*. Both halves are
     required: the negative half alone is satisfied by any model that returns
     garbage, so on its own it proves nothing about the question.

     The bare arm must NOT carry the answer-shaping instructions. `Answer ONLY
     from the log ... otherwise reply exactly: NOT IN LOG` scripts the desired
     outcome, so the negative half would measure instruction-following and
     could only fail if the model spontaneously invented "Vashti Renn" — an
     outcome of essentially zero probability, i.e. a check incapable of
     failing. `BARE_SYSTEM` below is stripped to a role line so the model is
     free to confabulate, which is what makes the failure reachable.

     (The previous gate asked "what is 10-3?" over a 300-token context —
     answerable with the context deleted, and too short to exercise a prefix
     at all.)
  3. SEMANTIC GATE. Planted facts recalled from the WARM bloc chain, at three
     depths: ~5%, ~50%, ~95%. Depth matters — a cache that silently drops the
     middle still answers early and late questions.
  4. TELEMETRY GATE. `response.metadata["prompt_cache"]` must report a hit
     whose `cached_tokens` covers essentially the whole prefix. This is the
     assertion the bloc work exists for: before the render-layer fix the chain
     reached ~29% of the prefix.
  5. NUMERIC GATE. Warm-vs-cold log-probabilities at the FIRST GENERATED
     POSITION: top-1 must agree (hard), and `max|delta log p|` must sit under a
     threshold (soft, scale-dependent, `MAX_DELTA_LOGP`). Telemetry says the
     cache was USED; only this says the KV it reused is numerically the same KV
     a cold prefill would have built. Checks 1-4 are all satisfiable by a cache
     that is reused and subtly wrong.

Byte equality between warm and cold is deliberately NOT a gate — mlx_lm chunks
prefill at `prefill_step_size=2048`, so warm and cold land on different chunk
boundaries and argmax flips at near-ties. It gets worse at exactly the 10k-100k
sizes the cache is for. It is recorded as a diagnostic only.

RUNNING: heavy and GPU-exclusive (MLX aborts the process if two generations run
concurrently).

    ABSTRACTCORE_RUN_BLOC_GATE=1 \\
    ABSTRACTCORE_BLOC_GATE_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit \\
    python -m pytest tests/test_prompt_cache_bloc_e2e_gate.py -q -s

Model choice matters: `Qwen3.5-4B-MLX-4bit` is known to FAIL check 1 uncached
(not self-deterministic), so it cannot be used as the gate model.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pytest


RUN = os.getenv("ABSTRACTCORE_RUN_BLOC_GATE") == "1"
MODEL = os.getenv("ABSTRACTCORE_BLOC_GATE_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TARGET_TOKENS = int(os.getenv("ABSTRACTCORE_BLOC_GATE_TOKENS", "10000"))
# PROVISIONAL AND SCALE-DEPENDENT. The only published measurement was at ~4.1k,
# where top-1 always agreed while max|delta log p| reached 1.0; 2.0 is headroom
# over that single point, not a value measured at the 10k-100k operating range.
# `untracked/prompt-cache-bench/harness/check3_numeric_curve.py` measures the
# curve at 10k / 30k / 100k so this number can rest on data at the size it is
# applied to. Move it WITH that data; do not relax it to make a run pass.
MAX_DELTA_LOGP = float(os.getenv("ABSTRACTCORE_BLOC_GATE_MAX_DELTA_LOGP", "2.0"))

requires_model = pytest.mark.skipif(
    not RUN,
    reason="Bloc E2E cache gate is heavy and GPU-exclusive; set ABSTRACTCORE_RUN_BLOC_GATE=1 to run",
)


# --------------------------------------------------------------------------
# Context: a coherent incident log, not repeated filler.
#
# Filler that repeats compresses unnaturally and can mask a prefix bug — a
# broken cache still "answers" when every chunk is interchangeable. Each entry
# here is distinct, and the three planted facts are unique strings that appear
# exactly once, at known depths.
# --------------------------------------------------------------------------

_SUBSYSTEMS = [
    "relay-alpha", "relay-beta", "pump-3", "pump-7", "coolant-loop-A", "coolant-loop-B",
    "vent-stack-2", "vent-stack-5", "feed-valve-11", "feed-valve-12", "sensor-array-N",
    "sensor-array-S", "buffer-tank-1", "buffer-tank-4", "compressor-D", "compressor-G",
]
_ACTIONS = [
    "logged a nominal reading", "requested a manual recheck", "cleared a stale advisory",
    "rotated the duty cycle", "annotated the shift handover", "acknowledged a soft warning",
    "scheduled a filter swap", "recorded a pressure trend", "reset a debounce counter",
    "confirmed the interlock state",
]

PLANTED = [
    # (depth fraction, unique fact line, question, the substrings the answer must contain)
    (
        0.05,
        "Operator Vashti Renn logged the anomaly under case tag GRENDEL-77 and set the "
        "escalation threshold to 42 psi.",
        "According to the log, which operator opened the case tagged GRENDEL-77, and what "
        "escalation threshold did they set? Answer in one sentence.",
        ["Vashti", "Renn", "42"],
    ),
    (
        0.50,
        "After the relay-alpha fault, operator Idris Kwan bypassed coolant-loop-B and this "
        "caused buffer-tank-4 to overfill by 19 litres.",
        "According to the log, what did the operator do after the relay-alpha fault, and what "
        "did it cause? Answer in one sentence.",
        ["Kwan", "coolant-loop-B", "buffer-tank-4"],
    ),
    (
        0.95,
        "The shift closed when technician Perrin Oyelaran replaced feed-valve-12 with unit "
        "serial QX-3391 and signed off at 04:17.",
        "According to the log, who replaced feed-valve-12, with which unit serial, and at what "
        "time did they sign off? Answer in one sentence.",
        ["Oyelaran", "QX-3391", "04:17"],
    ),
]


def _filler_entry(i: int) -> str:
    sub = _SUBSYSTEMS[i % len(_SUBSYSTEMS)]
    act = _ACTIONS[(i * 7) % len(_ACTIONS)]
    hh, mm = (i * 13) % 24, (i * 29) % 60
    return f"[{hh:02d}:{mm:02d}] entry {i:04d} — station {sub}: technician on duty {act}; drift {(i % 37) - 18:+d} mbar."


def build_context(provider: Any, target_tokens: int) -> Tuple[str, List[int]]:
    """A log of at least `target_tokens` tokens with the planted facts at depth.

    Returns `(text, planted_line_indices)`.
    """
    entries: List[str] = []
    i = 0
    while True:
        entries.append(_filler_entry(i))
        i += 1
        if i % 64 == 0:
            probe = "\n".join(entries)
            n = len(provider.prompt_cache_encode_bloc_text(probe) or [])
            if n >= target_tokens:
                break
        if i > 200_000:  # pathological tokenizer; stop rather than hang
            break

    planted_at: List[int] = []
    for depth, fact, _q, _need in PLANTED:
        idx = max(0, min(len(entries) - 1, int(len(entries) * depth)))
        entries.insert(idx, f"[!!] {fact}")
        planted_at.append(idx)
    return "\n".join(entries), planted_at


SYSTEM_INSTRUCTIONS = (
    "You are an incident-log analyst. Answer ONLY from the log below. "
    "If the log does not contain the answer, reply exactly: NOT IN LOG.\n\n"
    "=== INCIDENT LOG ===\n"
)

# The context-REMOVED arm. Deliberately NOT `SYSTEM_INSTRUCTIONS` minus the log:
# the refusal script in those instructions ("reply exactly: NOT IN LOG") is what
# would produce the expected negative result, which makes the control measure
# instruction-following instead of question difficulty. Stripped to the role
# line, a model that "knows" the answer is free to say it — so the control can
# actually fail.
BARE_SYSTEM = "You are an incident-log analyst."

# The positive half of the control. Same bare setup, questions answerable from
# the model's own weights. If these fail, the negative half above is evidence of
# a broken lane, not of context-dependence, and the gate must not proceed.
POSITIVE_CONTROL: List[Tuple[str, List[str]]] = [
    ("What is the capital of France? Answer in one sentence.", ["paris"]),
    ("What is the largest planet in the solar system? Answer in one sentence.", ["jupiter"]),
    ("In what year did Apollo 11 land humans on the Moon? Answer in one sentence.", ["1969"]),
]

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "search_log",
        "description": "Search the incident log for a substring",
        "parameters": {
            "type": "object",
            "properties": {"needle": {"type": "string"}},
            "required": ["needle"],
        },
    }
]


def _answered(text: str, needles: List[str]) -> bool:
    low = str(text or "").lower()
    return all(n.lower() in low for n in needles)


def _tool_calls(r: Any) -> List[Dict[str, Any]]:
    """The response's tool calls, normalized to comparable dicts."""
    calls = getattr(r, "tool_calls", None) or []
    out = []
    for c in calls:
        name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
        args = getattr(c, "arguments", None) or (
            c.get("arguments") if isinstance(c, dict) else None
        )
        out.append({"name": str(name or ""), "arguments": args})
    return out


def _live(r: Any) -> bool:
    """Did the lane produce a well-formed response AT ALL?

    Visible text OR a tool call counts. MEASURED on hardware (dense
    Qwen3-4B-Instruct, 2026-08-03): asked a planted question with the log
    removed but `search_log` still offered, the model returns EMPTY content and
    a `search_log` call — the strongest possible form of "I cannot answer from
    what I have". That is a live, coherent response and must not be conflated
    with the degenerate empty (no content, no reasoning, no calls) this
    liveness check exists to catch.
    """
    if len(_norm(getattr(r, "content", "") or "")) >= 3:
        return True
    return len(_tool_calls(r)) > 0


def _transcript(r: Any) -> str:
    """Canonical comparable form: visible text plus any tool calls."""
    parts = [_norm(getattr(r, "content", "") or "")]
    for c in _tool_calls(r):
        parts.append(f"CALL {c['name']}({json.dumps(c['arguments'], sort_keys=True, default=str)})")
    return " | ".join(p for p in parts if p)


def _refused(text: str) -> bool:
    low = str(text or "").lower()
    return "not in log" in low or not low.strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _ask(
    provider: Any,
    question: str,
    *,
    system_prompt: str,
    cache_key: Optional[str],
    tools: Optional[List[Dict[str, Any]]] = TOOLS,
) -> Any:
    # `messages=[]` is LOAD-BEARING, not decoration. The delta feed picks its
    # lane by `full_context = messages is not None` (P2-8, pinned by
    # tests/providers/test_mlx_prompt_cache_delta.py): with messages omitted the
    # call is PROMPT-ONLY and the warm cache takes APPEND semantics — the whole
    # 11k prompt is fed ON TOP of the cached prefix, telemetry reads
    # `append` with `cached` growing by ~11k per turn, and the telemetry gate
    # below fails while the cache "works". Measured on hardware 2026-08-03
    # (dense, 10k): cached 11315 -> 22696 -> 34080. The renderer ignores an
    # empty list (`if messages:`), so the fed bytes are IDENTICAL either way —
    # only the lane changes.
    kwargs: Dict[str, Any] = {
        "prompt": question,
        "messages": [],
        "system_prompt": system_prompt,
        "tools": tools,
        "temperature": 0.0,
        "seed": 0,
        "max_output_tokens": 96,
    }
    if cache_key is not None:
        kwargs["prompt_cache_key"] = cache_key
    return provider.generate(**kwargs)


def _feed(model, cache, ids, mx, *, step: int = 2048) -> None:
    """Feed `ids` exactly the way `generate_step(max_tokens=0)` does.

    NOT plain step-sized chunking: mlx_lm chunks `len-1` and then runs a
    separate single-token `_step`. The MLX provider prefills every bloc
    fragment through that call (`_prefill_tokens_into_cache`), so modelling the
    warm prefix any other way compares against a plan the provider never
    produces.
    """
    n = len(ids)
    if n == 0:
        return
    i = 0
    while i < n - 1:
        take = min(step, (n - 1) - i)
        model(mx.array(ids[i : i + take])[None], cache=cache)
        mx.eval([c.state for c in cache])
        i += take
        mx.clear_cache()
    model(mx.array(ids[n - 1 :])[None], cache=cache)
    mx.eval([c.state for c in cache])
    mx.clear_cache()


def _logprobs_next(model, cache, last_id, mx):
    logits = model(mx.array([last_id])[None], cache=cache)[:, -1, :]
    lp = (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0)
    mx.eval(lp)
    return lp


def _numeric_gate(provider, prep, system_prompt, question) -> Dict[str, Any]:
    """Warm-vs-cold logprobs at the FIRST GENERATED POSITION.

    WARM reproduces the real bloc-chain construction: each bloc fragment fed as
    its own `generate_step`, boundaries read from the caches the chain actually
    built (`bloc_boundary_tokens` meta), then the suffix on top. COLD is a
    single fresh prefill of the identical ids.
    """
    import gc

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model = provider.llm
    render = provider.prompt_cache_render_bloc_text(
        prompt=question, system_prompt=system_prompt, tools=TOOLS, add_generation_prompt=True
    )
    ids = provider.prompt_cache_encode_bloc_text(render)
    n = len(ids)

    bounds: List[int] = []
    for m in prep["modules"]:
        b = (provider.prompt_cache_key_meta(m["cache_key"]) or {}).get("bloc_boundary_tokens")
        if isinstance(b, int) and b > 0:
            bounds.append(b)
    assert bounds and bounds == sorted(bounds), (
        f"bloc boundaries unusable for the numeric gate: {bounds!r}"
    )
    prefix_len = bounds[-1]
    assert prefix_len < n, "the bloc prefix must be a strict prefix of the live prompt"

    def arm(by_fragment: bool):
        cache = make_prompt_cache(model)
        if by_fragment:
            prev = 0
            for b in bounds:
                _feed(model, cache, ids[prev:b], mx)
                prev = b
            _feed(model, cache, ids[prefix_len : n - 1], mx)
        else:
            _feed(model, cache, ids[: n - 1], mx)
        lp = _logprobs_next(model, cache, ids[n - 1], mx)
        lp = mx.array(lp)
        mx.eval(lp)
        del cache
        gc.collect()
        mx.clear_cache()
        return lp

    lp_cold = arm(False)
    lp_warm = arm(True)
    top1_cold = int(mx.argmax(lp_cold).item())
    top1_warm = int(mx.argmax(lp_warm).item())
    idx = mx.argsort(-lp_cold)[:32]
    out = {
        "prompt_tokens": n,
        "prefix_len": prefix_len,
        "bloc_boundaries": bounds,
        "top1_cold": top1_cold,
        "top1_warm": top1_warm,
        "top1_agree": top1_cold == top1_warm,
        "max_abs_delta_logp": float(mx.max(mx.abs(lp_warm - lp_cold)).item()),
        "max_abs_delta_logp_top32": float(mx.max(mx.abs(lp_warm[idx] - lp_cold[idx])).item()),
    }
    del lp_cold, lp_warm
    gc.collect()
    mx.clear_cache()
    return out


@pytest.fixture(scope="module")
def gate_provider():
    from abstractcore import create_llm

    llm = create_llm("mlx", model=MODEL)
    yield llm
    try:
        llm.unload_model(MODEL)
    except Exception:
        pass


# --------------------------------------------------------------------------
# CPU-only: the gate's FIXTURE is checkable without a model, and it must be —
# a gate whose context is too short, whose facts are duplicated, or whose
# planted depths are wrong measures nothing, and that failure would otherwise
# only show up on a GPU-exclusive run.
# --------------------------------------------------------------------------


def test_gate_context_fixture_is_well_formed():
    from tests.test_prompt_cache_bloc_composition import ToyBPETokenizer, make_renderer

    renderer = make_renderer("test/model", "im_start_end", ToyBPETokenizer())
    # The toy tokenizer is roughly per-character, so ask for a byte budget that
    # a real BPE would comfortably exceed at the same entry count.
    log, planted_at = build_context(renderer, 40_000)
    lines = log.split("\n")

    assert len(lines) > 100
    for (depth, fact, _q, _need), idx in zip(PLANTED, planted_at):
        assert log.count(fact) == 1, "a planted fact must appear exactly once"
        found = next(i for i, line in enumerate(lines) if fact in line)
        assert abs(found / len(lines) - depth) < 0.10, (
            f"planted fact drifted from depth {depth:.0%} to {found / len(lines):.0%}"
        )
    # Distinct entries, not repeated filler: repeated filler compresses
    # unnaturally and can mask a dropped-middle prefix bug.
    body = [ln for ln in lines if not ln.startswith("[!!]")]
    assert len(set(body)) > 0.95 * len(body), "gate context is too repetitive"

    # DISCRIMINATION. `_answered` requires every needle in the ANSWER, so a
    # question is only discriminating if at least one of its needles occurs
    # exactly ONCE in the whole log. Otherwise a model echoing filler could
    # satisfy it without having read the planted line. (Measured on the real
    # Qwen3-4B tokenizer at 10k: 'coolant-loop-B' and 'buffer-tank-4' occur 21x
    # and '42' 10x — only 'Kwan', 'Vashti', 'Renn', 'Oyelaran', 'QX-3391' and
    # '04:17' are unique, and every question owns at least one of them.)
    low = log.lower()
    for depth, _fact, question, needles in PLANTED:
        unique = [n for n in needles if low.count(n.lower()) == 1]
        assert unique, (
            f"question at depth {depth:.0%} has no globally unique needle "
            f"({ {n: low.count(n.lower()) for n in needles} }); it can be satisfied "
            f"by echoing filler: {question}"
        )


def test_gate_questions_require_all_three_needles():
    """An under-specified question measures a tie-break, not correctness."""
    for _depth, fact, question, needles in PLANTED:
        assert len(needles) >= 2, f"question is too easy to satisfy: {question!r}"
        for needle in needles:
            assert needle.lower() in fact.lower(), f"needle {needle!r} is not in its planted fact"
        assert not _answered("I don't know", needles)
        assert _answered(fact, needles)


@requires_model
def test_bloc_chain_e2e_cache_gate(gate_provider):
    provider = gate_provider
    log, planted_at = build_context(provider, TARGET_TOKENS)
    system_prompt = SYSTEM_INSTRUCTIONS + log
    prefix_tokens = len(provider.prompt_cache_encode_bloc_text(system_prompt) or [])
    print(f"\n[gate] model={MODEL} context={prefix_tokens} tk planted_at={planted_at}")
    assert prefix_tokens >= 1000, "gate context must be at least 1000 tokens (target 10k+)"

    # ---- CHECK 1: determinism control -----------------------------------
    # Fresh key each time so nothing is reused; if the lane is not
    # self-deterministic, every later comparison is noise.
    q0 = PLANTED[0][2]
    a = _ask(provider, q0, system_prompt=system_prompt, cache_key="gate:det:a")
    b = _ask(provider, q0, system_prompt=system_prompt, cache_key="gate:det:b")
    # LIVENESS FIRST. `"" == ""` is a determinism pass that measures nothing, and
    # it is the expected shape for a reasoning model that spends its whole budget
    # inside <think> (base.py only raises EmptyCompletionError when `reasoning` is
    # ALSO empty). A TOOL CALL with empty content is live — see `_live` — so the
    # comparison below runs over the full transcript (text + calls), not text only.
    assert _live(a) and _live(b), (
        f"lane produced neither text nor a tool call — determinism cannot be assessed "
        f"and every later check would compare empty strings.\n"
        f"  run A: {a.content!r}\n  run B: {b.content!r}"
    )
    assert _transcript(a) == _transcript(b), (
        f"lane is not self-deterministic at temperature 0 — gate cannot proceed.\n"
        f"  run A: {_transcript(a)!r}\n  run B: {_transcript(b)!r}"
    )
    print(f"[gate] determinism transcript: {_transcript(a)[:120]!r}")
    cold_answers = {}

    # ---- CHECK 2: context-dependence control (both halves) --------------
    # NEGATIVE: without the log the questions MUST fail. Asked under BARE_SYSTEM,
    # not under the refusal-scripted instructions — see the note at BARE_SYSTEM.
    for _depth, _fact, question, needles in PLANTED:
        r = _ask(provider, question, system_prompt=BARE_SYSTEM, cache_key=None)
        assert _live(r), (
            f"context-removed arm produced neither text nor a tool call; 'it did not "
            f"answer' is then vacuous.\n  q: {question}\n  a: {r.content!r}"
        )
        # A `search_log` call with the log gone is not an answer — it is the
        # model saying it NEEDS the context, which is the point of this check.
        # `_answered` reads visible content only, so a call cannot satisfy it.
        assert not _answered(r.content, needles), (
            f"question is answerable WITHOUT the log — invalid gate question.\n"
            f"  q: {question}\n  a: {r.content!r}"
        )
        print(f"[gate] no-log response: {_transcript(r)[:120]!r}")

    # POSITIVE: same bare system prompt, a question the weights can answer. This
    # is what distinguishes "the question needs the log" from "the lane is
    # broken". Asked WITHOUT tools: its job is to prove text GENERATION survived
    # the removal, and a persona-driven `search_log("capital of France")` call
    # would fail this check for a reason that has nothing to do with what it
    # measures. (Tool-calling liveness is already covered by the negative half.)
    for question, needles in POSITIVE_CONTROL:
        r = _ask(provider, question, system_prompt=BARE_SYSTEM, cache_key=None, tools=None)
        assert _answered(r.content, needles), (
            f"general-knowledge positive control FAILED with the log removed, so the "
            f"negative half above is not evidence of context-dependence — it is evidence "
            f"of a broken lane.\n  q: {question}\n  a: {r.content!r}\n  expected: {needles}"
        )
        print(f"[gate] positive control OK: {_norm(r.content)[:80]!r}")

    # ---- Build the BLOC CHAIN: separate `system` and `tools` blocs -------
    prep = provider.prompt_cache_prepare_modules(
        namespace="gate",
        modules=[
            {"module_id": "system", "system_prompt": system_prompt, "add_generation_prompt": False},
            {"module_id": "tools", "tools": TOOLS, "add_generation_prompt": False},
        ],
        make_default=False,
    )
    assert prep.get("supported") is True
    plan = prep.get("bloc_plan") or {}
    print(f"[gate] bloc_plan={plan}")
    assert plan.get("composable") is True, f"blocs did not compose: {plan.get('reason')}"
    assert len(prep["modules"]) == 2, "the system and tools blocs must stay separate"
    assert prep["modules"][0]["cache_key"] != prep["modules"][1]["cache_key"]

    # ---- CHECK 3 + 4: semantic gate and telemetry, from the WARM chain ---
    #
    # ONE FORK PER QUESTION — that is the hierarchical-cache contract ("fork
    # the final prefix into a per-session cache"), and it is also the only
    # shape whose telemetry can be asserted. A single session asked three
    # DIFFERENT questions diverges at the question boundary from turn 2 on
    # (the record holds prefix+Q1+answer; prefix+Q2 is not an extension), and
    # the full-context lattice answers divergence with a fresh rebuild — a
    # correct behavior that would fail this gate for a reason that is not a
    # cache defect. Each fork starts at the chain boundary, so every question
    # is a pure extension: `hit_extend`, cached ~= the whole prefix.
    reuse_ratios: List[float] = []
    for i, (depth, _fact, question, needles) in enumerate(PLANTED):
        session_key = f"gate:session:{i}"
        assert provider.prompt_cache_fork(prep["final_cache_key"], session_key, make_default=False)
        r = _ask(provider, question, system_prompt=system_prompt, cache_key=session_key)
        assert _answered(r.content, needles), (
            f"planted fact at depth {depth:.0%} not recalled from the warm bloc chain.\n"
            f"  q: {question}\n  full response: {_transcript(r)!r}\n  expected all of: {needles}\n"
            f"  (a `CALL search_log(...)` here means the model preferred the offered tool "
            f"over the inline log — a gate-design finding about one-shot QA with tools, "
            f"distinct from a cache failure; the no-cache diagnostic arm below "
            f"distinguishes them)"
        )
        cold_answers[question] = _norm(r.content)

        pc = (r.metadata or {}).get("prompt_cache") or {}
        print(f"[gate] depth={depth:.0%} outcome={pc.get('outcome')} "
              f"cached={pc.get('cached_tokens')} fed={pc.get('fed_tokens')}")
        cached = int(pc.get("cached_tokens") or 0)
        fed = int(pc.get("fed_tokens") or 0)
        assert str(pc.get("outcome") or "").startswith("hit"), (
            f"expected a cache hit at depth {depth:.0%}, got {pc!r}"
        )
        total = max(1, cached + fed)
        reuse_ratios.append(cached / total)

    # The bloc work exists for this number. Before the render-layer fix the
    # chain reached ~29% of the prefix (618 of 2148 tokens) because the tools
    # bloc opened a second `<|im_start|>system` block.
    worst = min(reuse_ratios)
    print(f"[gate] worst prefix reuse across depths: {worst:.1%}")
    assert worst > 0.80, f"bloc chain prefix reuse collapsed to {worst:.1%}; expected >80%"

    # ---- CHECK 5: NUMERIC GATE ------------------------------------------
    # Gate v2 item 4, which this file previously did not implement at all: the
    # docstring's "CHECK 4" is telemetry (outcome + reuse ratio), a different
    # measurement entirely. Numeric soundness of the reused KV needs a logprob
    # comparison, and nothing here was making one.
    #
    # TOP-1 AGREEMENT IS THE GATE. `max|delta log p|` is REPORTED against a
    # threshold that is explicitly provisional and scale-dependent: the only
    # published measurement was at ~4.1k, where top-1 always agreed while
    # max|delta| reached 1.0. Two COLD runs whose ids are identical already
    # diverge when only the prefill chunk plan differs, so a delta is not by
    # itself evidence about the cache — which is exactly why the magnitude is a
    # soft bound and the argmax is the hard one.
    ndeltas = _numeric_gate(provider, prep, system_prompt, PLANTED[0][2])
    print(f"[gate] numeric: top1_agree={ndeltas['top1_agree']} "
          f"max|dlogp|={ndeltas['max_abs_delta_logp']:.4f} "
          f"top32={ndeltas['max_abs_delta_logp_top32']:.4f} "
          f"prefix={ndeltas['prefix_len']} prompt={ndeltas['prompt_tokens']}")
    assert ndeltas["top1_agree"], (
        f"warm and cold disagree on the FIRST GENERATED TOKEN "
        f"(cold={ndeltas['top1_cold']}, warm={ndeltas['top1_warm']}); the reused KV is "
        f"not numerically equivalent to a cold prefill at this scale"
    )
    assert ndeltas["max_abs_delta_logp"] <= MAX_DELTA_LOGP, (
        f"max|delta log p| = {ndeltas['max_abs_delta_logp']:.4f} exceeds the configured "
        f"threshold {MAX_DELTA_LOGP} at {ndeltas['prompt_tokens']} tokens. This bound is "
        f"scale-dependent; if a curve measured at this size says otherwise, move the "
        f"threshold WITH THE DATA (ABSTRACTCORE_BLOC_GATE_MAX_DELTA_LOGP), do not delete "
        f"the check"
    )

    # ---- Diagnostic only, never a gate ----------------------------------
    # Warm vs cold byte equality: mlx_lm chunks prefill at 2048, so warm and
    # cold land on different reduction boundaries and argmax can flip at
    # near-ties. Reported, not asserted.
    for _depth, _fact, question, _needles in PLANTED:
        fresh = _ask(provider, question, system_prompt=system_prompt, cache_key=None)
        same = _norm(fresh.content) == cold_answers[question]
        print(f"[diag] warm==cold bytes: {same}  q={question[:48]!r}")
