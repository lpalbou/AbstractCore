"""Pins for MID-PREFILL progress: `Prefill · 2,100 / 5,642 tokens (37%)`.

A 20k-token prompt can take tens of seconds to prefill, and "Prefill · 5,642
tokens" frozen for that long reads as a hang. Lanes that can observe their
prompt pass chunk by chunk report a cumulative `prefill_processed_tokens`;
lanes that cannot report nothing (never an estimate). These tests pin:

1. the emitter — cadence during prefill (time-limited, never count-capped),
   monotone positions, restored tokens counted first, nothing after the first
   token, nothing on an unknown axis;
2. the lanes that cannot observe — HTTP providers never receive a callback;
3. the native runtime seam — prefill observations reach the listener on the
   consumer thread and are never yielded as results; the batched lane's row
   arithmetic matches upstream's private counters (and the fields exist in the
   installed mlx-vlm);
4. the mlx-vlm prefill-bar observer and the mlx-lm `prompt_progress_callback`;
5. the HuggingFace lanes — llama.cpp `n_batch` slices, and the transformers
   chunk loop on the cached SmolLM2-135M model (processed climbs per chunk).
"""

from __future__ import annotations

import inspect
import os
import queue
import threading
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from abstractcore.providers.generation_progress import TextProgressEmitter


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _emitter(events: List[Dict[str, Any]], clock: _Clock, **kwargs: Any) -> TextProgressEmitter:
    return TextProgressEmitter(events.append, provider="mlx", model="m", clock=clock, **kwargs)


def _prefill_positions(events):
    return [e["prefill_processed_tokens"] for e in events if "prefill_processed_tokens" in e]


# --------------------------------------------------------------------------
# 1. emitter
# --------------------------------------------------------------------------


def test_prefill_progress_flows_at_the_cadence_with_monotone_positions_and_a_rate():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.5)
    em.prefill(prompt_tokens=5642)
    assert "prefill_processed_tokens" not in events[0], "a cold start has no measured position yet"
    for done in range(0, 5642, 256):  # one 256-token chunk every 0.1 s
        em.prefill_progress(processed_tokens=done)
        clock.t += 0.1
    positions = _prefill_positions(events)
    assert all(e["phase"] == "prefill" for e in events)
    assert len(positions) >= 3, "mid-prefill progress never reached the host"
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    gaps = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(events, events[1:])]
    assert all(gap >= 0.5 - 1e-9 for gap in gaps), gaps
    rates = [e["prefill_tokens_per_second"] for e in events if "prefill_tokens_per_second" in e]
    assert rates and all(r == pytest.approx(2560.0, rel=0.01) for r in rates), rates
    assert all(e["generated_tokens"] == 0 and e["prompt_tokens"] == 5642 for e in events)


def test_prefill_progress_is_never_count_capped():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.5)
    em.prefill(prompt_tokens=1_000_000)
    for done in range(1, 501):
        clock.t += 0.5
        em.prefill_progress(processed_tokens=done * 1000)
    assert len(_prefill_positions(events)) == 500


def test_restored_tokens_count_as_processed_at_once():
    # Key-mode lanes know the restored prefix at start: the start event says so.
    events: List[Dict[str, Any]] = []
    em = _emitter(events, _Clock())
    em.prefill(prompt_tokens=5363, cached_tokens=5320)
    assert events[0]["prefill_processed_tokens"] == 5320

    # APC lanes learn it from the first observation: emitted at once, even
    # inside the cadence window, then fed progress is placed after it.
    events2: List[Dict[str, Any]] = []
    clock = _Clock()
    em2 = _emitter(events2, clock, min_interval_s=0.5)
    em2.prefill(prompt_tokens=5363)
    clock.t += 0.01
    assert em2.prefill_progress(fed_processed=0, fed_total=43) is True
    assert events2[-1]["prefill_processed_tokens"] == 5320
    assert events2[-1]["cached_tokens"] == 5320 and events2[-1]["fed_tokens"] == 43


def test_fed_axis_progress_is_placed_after_the_restored_prefix():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.0)
    em.prefill(prompt_tokens=1000, cached_tokens=600)
    for done in (0, 128, 256, 384):
        clock.t += 0.1
        em.prefill_progress(fed_processed=done, fed_total=400)
    assert _prefill_positions(events) == [600, 728, 856, 984]


def test_no_prefill_event_after_the_first_token_or_after_complete():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.0)
    em.prefill(prompt_tokens=100)
    em.prefill_progress(processed_tokens=40)
    em.generation(generated_tokens=1)
    before = len(events)
    assert em.prefill_progress(processed_tokens=99, force=True) is False
    em.complete(generated_tokens=1)
    assert em.prefill_progress(processed_tokens=100, force=True) is False
    assert [e["phase"] for e in events[before:]] == ["complete"]
    assert all(e["phase"] != "prefill" for e in events[2:])


def test_positions_never_go_backwards_and_never_exceed_the_prompt():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.0)
    em.prefill(prompt_tokens=100)
    for done in (50, 30, 50, 120):
        clock.t += 0.1
        em.prefill_progress(processed_tokens=done)
    assert _prefill_positions(events) == [50, 100]


def test_progress_on_an_unknown_axis_is_not_reported():
    events: List[Dict[str, Any]] = []
    em = _emitter(events, _Clock(), min_interval_s=0.0)
    # No prefill start: neither the prompt size nor the restored count is known.
    assert em.prefill_progress(fed_processed=256, fed_total=1000) is False
    assert em.prefill_progress(processed_tokens=256) is False
    # Counts that contradict the known prompt size: say nothing.
    em2 = _emitter(events, _Clock(), min_interval_s=0.0)
    em2.prefill(prompt_tokens=100)
    assert em2.prefill_progress(fed_processed=10, fed_total=5000) is False
    assert _prefill_positions(events) == []


def test_an_inactive_emitter_ignores_prefill_progress():
    em = TextProgressEmitter(None)
    assert em.prefill_progress(processed_tokens=10, prompt_tokens=100) is False


# --------------------------------------------------------------------------
# 2. lanes that cannot observe prefill emit nothing extra
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module, cls",
    [
        ("abstractcore.providers.lmstudio_provider", "LMStudioProvider"),
        ("abstractcore.providers.ollama_provider", "OllamaProvider"),
        ("abstractcore.providers.openai_compatible_provider", "OpenAICompatibleProvider"),
    ],
)
def test_http_providers_never_receive_a_progress_callback(module, cls):
    import importlib

    provider_cls = getattr(importlib.import_module(module), cls)
    provider = provider_cls.__new__(provider_cls)
    assert provider.supports_text_progress_events() is False


def test_in_process_providers_declare_phase_support():
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider
    from abstractcore.providers.mlx_provider import MLXProvider

    assert MLXProvider.__new__(MLXProvider).supports_text_progress_events() is True
    assert HuggingFaceProvider.__new__(HuggingFaceProvider).supports_text_progress_events() is True


# --------------------------------------------------------------------------
# 3. native runtime seam
# --------------------------------------------------------------------------


def _job(reports=True, maxsize=8):
    from abstractcore.providers.mlx_runtime import NativeRequest, _Job

    job = _Job(request=NativeRequest(prompt="p", max_tokens=4), output=queue.Queue(maxsize))
    job.reports_prefill = reports
    return job


def test_result_iterator_delivers_prefill_observations_to_the_listener_and_never_yields_them():
    from abstractcore.providers.mlx_runtime import NativePrefillProgress, NativeResult, NativeRuntime, _ResultIterator

    job = _job()
    seen: List[Any] = []
    job.output.put(NativePrefillProgress(processed_tokens=256, prompt_tokens=1000, cached_tokens=0))
    job.output.put(NativePrefillProgress(fed_processed=512, fed_total=1000))
    job.output.put(NativeResult(text="hi", token=1, finish_reason="stop"))
    job.done.set()
    runtime = NativeRuntime.__new__(NativeRuntime)
    runtime._condition = threading.Condition(threading.RLock())
    results = list(_ResultIterator(runtime, job, seen.append))
    assert [r.text for r in results] == ["hi"]
    assert [s.as_kwargs() for s in seen] == [
        {"processed_tokens": 256, "prompt_tokens": 1000, "cached_tokens": 0},
        {"fed_processed": 512, "fed_total": 1000},
    ]

    # Without a listener the stream is exactly what it was before.
    job2 = _job()
    job2.output.put(NativePrefillProgress(processed_tokens=256, prompt_tokens=1000))
    job2.output.put(NativeResult(text="x", token=1, finish_reason="stop"))
    job2.done.set()
    assert [r.text for r in _ResultIterator(runtime, job2)] == ["x"]


def test_emit_prefill_only_for_listeners_before_the_first_token_and_never_crowds_results():
    from abstractcore.providers.mlx_runtime import NativeRuntime

    silent = _job(reports=False)
    assert NativeRuntime._emit_prefill(silent, processed_tokens=1) is False and silent.output.empty()

    job = _job(maxsize=8)
    for i in range(10):
        NativeRuntime._emit_prefill(job, processed_tokens=i)
    assert job.output.qsize() == 4, "prefill observations may fill at most half the result queue"

    started = _job()
    started.first_token_at = 1.0
    assert NativeRuntime._emit_prefill(started, processed_tokens=1) is False


def test_prompt_batch_rows_match_upstream_arithmetic():
    from abstractcore.providers.mlx_runtime import NativeRuntime

    # Left-padded cold rows: 3 prompts of 1000 / 600 / 300 tokens, 512 columns done.
    left = SimpleNamespace(
        _prompt_uids=[7, 8, 9], _prompt_tokens_per_row=[1000, 600, 300], _cached_tokens_per_row=[0, 0, 0],
        _suffix_lens=[1000, 600, 300], _left_padding_per_row=[0, 400, 700], _processed_prompt_columns=512,
        _right_pad_per_row=None,
    )
    assert NativeRuntime._prompt_batch_rows(left) == [(7, 512, 1000, 0), (8, 112, 600, 0), (9, 0, 300, 0)]

    # Right-padded warm/cold mix: a restored 5,320-token prefix + 43 suffix, and a cold row.
    right = SimpleNamespace(
        _prompt_uids=[1, 2], _prompt_tokens_per_row=[5363, 900], _cached_tokens_per_row=[5320, 0],
        _suffix_lens=[43, 900], _left_padding_per_row=[0, 0], _processed_prompt_columns=256,
        _right_pad_per_row=[857, 0],
    )
    assert NativeRuntime._prompt_batch_rows(right) == [(1, 5363, 5363, 5320), (2, 256, 900, 0)]


def test_prompt_batch_rows_report_nothing_when_upstream_changes_shape():
    from abstractcore.providers.mlx_runtime import NativeRuntime

    assert NativeRuntime._prompt_batch_rows(SimpleNamespace(_prompt_uids=[1])) == []


def test_installed_mlx_vlm_still_exposes_the_counters_the_batched_lane_reads():
    ar = pytest.importorskip("mlx_vlm.generate.ar")
    from abstractcore.providers.mlx_runtime import NativeRuntime

    source = inspect.getsource(ar.PromptProcessingBatch)
    for name in NativeRuntime._PROMPT_BATCH_ROW_FIELDS:
        assert f"self.{name}" in source, f"mlx-vlm PromptProcessingBatch no longer sets {name}"
    assert "self._processed_prompt_columns += n" in inspect.getsource(ar.PromptProcessingBatch.prompt_step)


# --------------------------------------------------------------------------
# 4. mlx-vlm prefill bar + mlx-lm prompt_progress_callback
# --------------------------------------------------------------------------


def test_mlx_vlm_prefill_bar_is_observed_only_while_bound_and_only_for_prefill():
    ar = pytest.importorskip("mlx_vlm.generate.ar")
    from abstractcore.providers.mlx_prefill_observer import install, observe_prefill

    assert install() is True
    assert "Prefill" in inspect.getsource(ar.generate_step) and "pbar.update(n_to_process)" in inspect.getsource(
        ar.generate_step
    ), "mlx-vlm's chunked prefill no longer reports to its Prefill bar"

    seen: List[tuple] = []
    with observe_prefill(lambda done, total: seen.append((done, total))):
        with ar.tqdm(total=1000, desc="Prefill", unit="tok", disable=True) as bar:
            bar.update(256)
            bar.update(256)
        with ar.tqdm(total=10, desc="Something else", disable=True) as other:
            other.update(5)
    with ar.tqdm(total=1000, desc="Prefill", disable=True) as unbound:
        unbound.update(999)
    assert seen == [(0, 1000), (256, 1000), (512, 1000)]


def test_mlx_lm_lane_reports_prefill_through_prompt_progress_callback():
    from abstractcore.providers.mlx_provider import MLXProvider

    clock = _Clock()
    events: List[Dict[str, Any]] = []
    provider = MLXProvider.__new__(MLXProvider)
    provider.llm = object()
    provider.tokenizer = object()

    def stream_generate_fn(model, tokenizer, prompt, **kwargs):
        report = kwargs["prompt_progress_callback"]
        for done in (0, 2048, 4096):
            report(done, 5000)
            clock.t += 0.6
        yield SimpleNamespace(text="ok", token=1, generation_tokens=1, prompt_tokens=5000,
                              prompt_tps=1.0, generation_tps=0.0, finish_reason="stop")

    provider.stream_generate_fn = stream_generate_fn
    em = _emitter(events, clock, min_interval_s=0.5)
    em.prefill(prompt_tokens=5000)
    text = provider._observed_generate(em, prompt="p", max_tokens=4, prompt_cache=None,
                                       sampler_kwargs={}, embed_kwargs={})
    assert text == "ok"
    assert _prefill_positions(events) == [2048, 4096]
    assert [e["phase"] for e in events] == ["prefill", "prefill", "prefill", "generate", "complete"]


def test_native_lanes_receive_the_emitter_hook_not_an_mlx_lm_callback():
    from abstractcore.providers.generation_progress import PREFILL_PROGRESS_KWARG
    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider._native_runtime = object()
    em = TextProgressEmitter(lambda e: None)
    kwargs = provider._prefill_observation_kwargs(em)
    assert set(kwargs) == {PREFILL_PROGRESS_KWARG}
    assert provider._prefill_observation_kwargs(TextProgressEmitter(None)) == {}


# --------------------------------------------------------------------------
# 5. HuggingFace lanes
# --------------------------------------------------------------------------


def test_llama_cpp_prefill_reports_each_n_batch_slice_as_an_absolute_position():
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _hf_progress_bound

    fed: List[int] = []
    llm = SimpleNamespace(n_batch=512, eval=lambda toks: fed.append(len(toks)))
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.model = "gguf"
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = TextProgressEmitter(events.append, provider="huggingface", model="gguf", clock=clock, min_interval_s=0.0)
    em.prefill(prompt_tokens=3000, cached_tokens=1000, fed_tokens=2000)

    orig_eval = llm.eval

    def slow_eval(toks):
        clock.t += 0.2
        orig_eval(toks)

    llm.eval = slow_eval
    with _hf_progress_bound(em):
        provider._gguf_eval_cancellable(llm, list(range(2000)), None, position=1000)
    assert fed == [512, 512, 512, 464]
    assert _prefill_positions(events) == [1000, 1512, 2024, 2536, 3000]

    # Unobserved and uncancellable: ONE eval, byte-identical to before.
    fed.clear()
    provider._gguf_eval_cancellable(llm, list(range(2000)), None, position=1000)
    assert fed == [2000]


def _smollm2_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache("HuggingFaceTB/SmolLM2-135M-Instruct", "config.json")
        return isinstance(hit, str) and os.path.exists(hit)
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _smollm2_cached(), reason="SmolLM2-135M-Instruct is not in the local HF cache")
def test_transformers_prefill_progress_climbs_per_chunk_on_a_real_model(monkeypatch):
    """Real weights, real chunk loop: processed climbs by one chunk per event,
    the restored prefix is counted first on the second turn, and the first
    token ends the prefill phase."""

    monkeypatch.setenv("ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP", "256")
    monkeypatch.setenv("ABSTRACTCORE_PROGRESS_MIN_INTERVAL_S", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from abstractcore import create_llm

    llm = create_llm("huggingface", model="HuggingFaceTB/SmolLM2-135M-Instruct")
    text = "".join(f"Paragraph {i}: ledger records stream to the web client. " for i in range(160))
    history = [{"role": "user", "content": "Read this.\n\n" + text}, {"role": "assistant", "content": "Done."}]

    turn1: List[Dict[str, Any]] = []
    llm.generate("", messages=history + [{"role": "user", "content": "Summarize."}], max_tokens=4,
                 temperature=0.0, prompt_cache_key="missionJ:test", on_progress=turn1.append)
    positions = _prefill_positions(turn1)
    total = turn1[0]["prompt_tokens"]
    assert total > 1024
    assert len(positions) >= 4, turn1
    assert positions == sorted(positions)
    steps = [b - a for a, b in zip(positions, positions[1:])]
    assert all(step == 256 for step in steps[:-1]), steps
    first_generate = next(i for i, e in enumerate(turn1) if e["phase"] == "generate")
    assert all(e["phase"] == "prefill" for e in turn1[:first_generate])
    assert turn1[first_generate]["first_token"] is True
    assert turn1[-1]["phase"] == "complete"

    turn2: List[Dict[str, Any]] = []
    llm.generate("", messages=history + [{"role": "user", "content": "Summarize."},
                                          {"role": "assistant", "content": "Ledger."},
                                          {"role": "user", "content": "Shorter."}],
                 max_tokens=4, temperature=0.0, prompt_cache_key="missionJ:test", on_progress=turn2.append)
    assert turn2[0]["phase"] == "prefill"
    assert turn2[0]["cached_tokens"] > 1024
    assert turn2[0]["prefill_processed_tokens"] == turn2[0]["cached_tokens"]
    assert turn2[-1]["phase"] == "complete"
