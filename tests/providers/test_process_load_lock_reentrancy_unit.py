"""Pinned context creation nests the process load lock without allowing another load."""
import logging
import threading

from abstractcore.providers import gguf_slots, process_residency
from abstractcore.providers.huggingface_provider import HuggingFaceProvider


def test_pinned_gguf_contexts_complete_inside_initial_load_lock(monkeypatch, tmp_path):
    # Isolate the production lock type: a failure leaves only a daemon holding
    # this test's lock, never the process lock used by later tests.
    monkeypatch.setattr(process_residency, "_LOAD_LOCK", type(process_residency.load_lock())())
    allocations = []

    class Engine:
        def __init__(self, **kwargs):
            allocations.append(self)

    class Scheduler:
        def __init__(self, **kwargs):
            self.max_concurrent = kwargs["max_concurrent"]
            self.slots = [kwargs["first_llm"]]

        def add_slot(self, engine):
            self.slots.append(engine)

    monkeypatch.setattr(gguf_slots, "SlotScheduler", Scheduler)
    monkeypatch.setattr(gguf_slots, "register", lambda *args: None)
    monkeypatch.setattr(gguf_slots, "attach", lambda *args: None)
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    provider.__dict__["_llm_own"] = Engine()
    provider.model = "fake-pinned-gguf"
    provider.max_tokens = provider.max_output_tokens = 4096
    provider.n_gpu_layers = 0
    provider._gguf_prompt_cache_lock = threading.Lock()
    provider._generation_worker = object()
    provider.logger = logging.getLogger(__name__)
    provider._gguf_configured_concurrency = lambda: 2
    provider._gguf_slot_fits = lambda *args: None
    provider._gguf_probe_decode = lambda engine: None
    errors = []

    def load():
        try:
            with process_residency.load_lock():
                provider._gguf_create_slot_pool(Engine, {"n_ctx": 4096}, tmp_path / "fake.gguf", "fake")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), "pinned slots must not deadlock the initial model load"
    assert not errors
    assert len(allocations) == 2
    assert provider._gguf_scheduler.pinned_slots == 2


def test_process_load_lock_still_excludes_other_threads(monkeypatch):
    monkeypatch.setattr(process_residency, "_LOAD_LOCK", type(process_residency.load_lock())())
    entered = threading.Event()

    def load():
        with process_residency.load_lock():
            entered.set()

    with process_residency.load_lock():
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        assert not entered.wait(timeout=0.05)
    assert entered.wait(timeout=1)
    thread.join(timeout=1)
