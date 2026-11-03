"""``generate(state, output="decision", questions=...)`` through BaseProvider, with the
decision runtime replaced by a fake: the dispatch, question binding, response shape
and error paths, without weights."""

import json

import numpy as np
import pytest

from abstractcore.core.multimodal_generation import MultimodalGenerateResponse
from abstractcore.core.types import GenerateResponse
from abstractcore.decision.runtime import DecisionModel
from abstractcore.providers.base import BaseProvider

from .test_decision_core import FakeBackend, FakeTokenizer


class _LocalProvider(BaseProvider):
    def __init__(self, provider="mlx", model="fake/kev-4b"):
        super().__init__(model=model)
        self.provider = provider

    def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
        return GenerateResponse(content=f"generated:{prompt}", model=self.model, metadata={})

    def get_capabilities(self):
        return []

    def unload_model(self, model_name):
        return None

    def list_available_models(self, **kwargs):
        return [self.model]


@pytest.fixture
def fake_runtime(monkeypatch):
    loads = []

    def fake_load(provider, model, **kwargs):
        loads.append((provider, model, kwargs))
        return DecisionModel(FakeBackend(), FakeTokenizer(), model_id=model, provider=provider)

    import abstractcore.decision as decision

    monkeypatch.setattr(decision, "load_decision_model", fake_load)
    monkeypatch.setattr(decision, "is_decision_model", lambda model, provider=None: "kev" in model)
    return loads


def test_generate_decision_output_returns_typed_answers(fake_runtime):
    llm = _LocalProvider()
    result = llm.generate(
        "Shoes arrived late and in the wrong size.",
        output="decision",
        questions={
            "department": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": None, "shipping and delivery": None}},
            "escalate": "Does this need urgent human attention?",
        },
    )
    assert isinstance(result, MultimodalGenerateResponse)
    assert result.decision is not None
    assert result.decisions["department"].choice == "shipping and delivery"
    assert result.decisions["escalate"].noul == pytest.approx(0.5)
    assert result.metadata["decision"]["answers"]["escalate"] == {"type": "noul", "noul": 0.5}
    assert json.loads(result.content)["department"]["choice"] == "shipping and delivery"
    assert result.metadata["decision_backend"] == "fake"
    assert fake_runtime == [("mlx", "fake/kev-4b", {})]
    # the runtime is built once and reused
    llm.generate("another document", output="decision", questions={"q": ["a", "bb"]})
    assert len(fake_runtime) == 1


def test_generate_decision_accepts_spec_dict_and_state_override(fake_runtime):
    llm = _LocalProvider()
    result = llm.generate(
        "ignored prompt",
        output={"modality": "decision", "state": {"message": "late order", "charges": 2},
                "questions": {"refund": "Should we refund?"}},
    )
    assert "refund" in result.decisions
    assert result.decision.usage.state_tokens > 1


def test_generate_decision_requires_questions_and_state(fake_runtime):
    llm = _LocalProvider()
    with pytest.raises(ValueError, match="questions="):
        llm.generate("some state", output="decision")
    with pytest.raises(ValueError, match="needs a state"):
        llm.generate("", output="decision", questions={"q": "yes?"})


def test_generate_decision_refuses_non_decision_model_and_remote_provider(fake_runtime):
    with pytest.raises(ValueError, match="not a decision model"):
        _LocalProvider(model="qwen/qwen3-4b").generate("state", output="decision", questions={"q": "yes?"})
    with pytest.raises(ValueError, match="set-default output.decision"):
        _LocalProvider(provider="openai", model="fake/kev-4b").generate("state", output="decision", questions={"q": "yes?"})


def test_text_generation_is_untouched(fake_runtime):
    llm = _LocalProvider()
    assert llm.generate("hello").content == "generated:hello"


def test_decision_options_reach_the_loader(fake_runtime):
    llm = _LocalProvider()
    llm._decision_options = {"prefix_cache_size": 0, "n_ctx": 4096}
    llm.generate("a document", output="decision", questions={"q": ["a", "bb"]})
    assert fake_runtime[-1][2] == {"prefix_cache_size": 0, "n_ctx": 4096}


def test_decision_spec_route_uses_the_resident_pool(monkeypatch):
    """A text provider asked for output='decision' with a routed provider/model (the
    output.decision capability default) runs the resident decision provider, not itself."""
    from abstractcore.decision import pool
    from abstractcore.providers.base import BaseProvider

    from abstractcore.decision.types import DecisionResponse, DecisionUsage, NoulAnswer

    class _Model:
        def decide(self, state, questions, truncate=None):
            return DecisionResponse(model="fake", backend="fake", precision="bf16", answers={"q": NoulAnswer(noul=0.5)},
                                    usage=DecisionUsage(), latency_ms=1)

    class _DecisionLLM:
        def __init__(self, provider, model):
            self.provider, self.model = provider, model

        def decision_model(self):
            return _Model()

        def unload_model(self, model):
            pass

    pool._reset_for_tests()
    created = []
    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: created.append((key[0], key[1], options)) or _DecisionLLM(*key))
    owner = _LocalProvider(provider="ollama", model="qwen3.5:9b")
    with BaseProvider._decision_model_in_use(
        owner, {"modality": "decision", "provider": "mlx", "model": "jaredpalmer/kev-4b", "decision_prefix_cache_size": 0}
    ) as model:
        assert isinstance(model, _Model)
    assert created == [("mlx", "jaredpalmer/kev-4b", {"decision_prefix_cache_size": 0})]


def test_decision_pool_keeps_two_models_resident(monkeypatch):
    from abstractcore.decision import loader, pool

    monkeypatch.setattr(loader, "is_decision_model", lambda model, provider=None: True)
    unloaded = []

    class _LLM:
        def __init__(self, p, m):
            self.key = (p, m)

        def unload_model(self, model):
            unloaded.append(model)

    pool._reset_for_tests()
    a = pool.decision_provider("mlx", "a", factory=_LLM)
    pool.decision_provider("mlx", "b", factory=_LLM)
    assert pool.decision_provider("mlx", "a", factory=lambda p, m: None) is a  # reused, now most recent
    pool.decision_provider("mlx", "c", factory=_LLM)
    assert unloaded == ["b"] and pool.resident() == [("mlx", "a"), ("mlx", "c")] and pool.latest() == ("mlx", "c")
    assert pool.release() == (2, 0) and pool.resident() == []


def test_decision_spec_model_handle_names_its_provider(monkeypatch):
    from abstractcore.decision import pool

    seen = []

    class _LLM:
        def decision_model(self):
            return "model"

    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: seen.append(key) or _LLM())
    owner = _LocalProvider(provider="ollama", model="qwen3.5:9b")
    assert _in_use(owner, {"modality": "decision", "model": "hf/ggml-org/Kev-4B-GGUF:Q4_K_M"}) == "model"
    assert seen == [("huggingface", "ggml-org/Kev-4B-GGUF:Q4_K_M")]


def test_decision_handle_wins_over_the_default_route_provider(monkeypatch):
    """The route resolver copies the output.decision default's provider onto a spec that names
    only a model; a `provider/model` handle still names its own provider."""
    from abstractcore.decision import pool

    seen = []
    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: seen.append(key) or _PoolLLM())
    owner = _LocalProvider(provider="ollama", model="qwen3.5:9b")
    _in_use(owner, {"modality": "decision", "provider": "huggingface", "model": "mlx/jaredpalmer/kev-4b"})
    assert seen == [("mlx", "jaredpalmer/kev-4b")]


def test_bare_decision_handle_picks_the_local_engine(monkeypatch):
    from abstractcore.decision import pool

    seen = []
    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: seen.append(key) or _PoolLLM())
    monkeypatch.setattr(pool, "parse_handle", lambda h: ("huggingface", h))
    owner = _LocalProvider(provider="lmstudio", model="qwen/qwen3.5-9b")
    _in_use(owner, {"modality": "decision", "model": "ggml-org/Kev-4B-GGUF:Q4_K_M"})
    assert seen == [("huggingface", "ggml-org/Kev-4B-GGUF:Q4_K_M")]


def test_kev_latest_reuses_the_resident_model_or_explains(monkeypatch):
    from abstractcore.decision import pool

    pool._reset_for_tests()
    owner = _LocalProvider(provider="openai", model="gpt-5")
    with pytest.raises(ValueError, match="no resident decision model"):
        _in_use(owner, {"modality": "decision", "model": "kev-latest"})
    monkeypatch.setattr(pool, "latest", lambda: ("mlx", "jaredpalmer/kev-0.8b"))
    seen = []
    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: seen.append(key) or _PoolLLM())
    _in_use(owner, {"modality": "decision", "model": "kev-latest"})
    assert seen == [("mlx", "jaredpalmer/kev-0.8b")]


def test_a_hosted_route_keeps_its_own_model_name_never_the_resident_alias(monkeypatch):
    """The output.decision default `typesafe` + `jev-latest` is TypeSafe's hosted Jev: the name is the
    endpoint's own, not the alias of whatever local model was used last."""
    from abstractcore.decision import pool

    pool._reset_for_tests()
    monkeypatch.setattr(pool, "latest", lambda: ("mlx", "jaredpalmer/kev-0.8b"))
    seen = []
    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: seen.append(key) or _PoolLLM())
    owner = _LocalProvider(provider="ollama", model="qwen3.5:9b")
    _in_use(owner, {"modality": "decision", "provider": "typesafe", "model": "jev-latest"})
    _in_use(owner, {"modality": "decision", "provider": "systemone", "model": "kev-latest"})
    _in_use(owner, {"modality": "decision", "provider": "mlx", "model": "kev-latest"})  # a local route: the alias
    assert seen == [("typesafe", "jev-latest"), ("systemone", "kev-latest"), ("mlx", "jaredpalmer/kev-0.8b")]
    assert pool.serves_decisions_only("typesafe") and pool.serves_decisions_only("systemone")
    assert not any(pool.serves_decisions_only(p) for p in ("mlx", "huggingface", "ollama", "", None))


def test_pool_refuses_a_non_decision_model_before_loading(monkeypatch):
    from abstractcore.decision import loader, pool

    pool._reset_for_tests()
    monkeypatch.setattr(loader, "is_decision_model", lambda model, provider=None: False)
    built = []
    with pytest.raises(ValueError, match="is not a decision model"):
        pool.decision_provider("mlx", "mlx-community/Qwen3.5-9B-4bit", factory=lambda p, m: built.append(m))
    assert built == [] and pool.resident() == [] and pool.latest() is None


def test_pool_defers_unloading_a_model_in_use(monkeypatch):
    from abstractcore.decision import loader, pool

    pool._reset_for_tests()
    monkeypatch.setattr(loader, "is_decision_model", lambda model, provider=None: True)
    unloaded = []

    class _LLM:
        def __init__(self, p, m):
            self.key = (p, m)

        def unload_model(self, model):
            unloaded.append(model)

    with pool.use("mlx", "a", factory=_LLM):
        pool.decision_provider("mlx", "b", factory=_LLM)
        pool.decision_provider("mlx", "c", factory=_LLM)  # over capacity: "a" is in use, so "b" goes
        assert unloaded == ["b"] and ("mlx", "a") in pool.resident()
        assert pool.release("mlx", "a") == (0, 1)  # asked to unload while in use: deferred
        assert unloaded == ["b"]
    assert unloaded == ["b", "a"] and pool.resident() == [("mlx", "c")]


class _PoolLLM:
    def decision_model(self):
        return "model"

    def unload_model(self, model):
        pass


def _in_use(owner, spec):
    """Run the production lane (`_decision_model_in_use`) and return what it yields."""
    with BaseProvider._decision_model_in_use(owner, spec) as model:
        return model


def test_pool_never_unloads_a_model_between_hand_out_and_use(monkeypatch):
    """`use()` raises the in-use count in the same locked section that hands the model out,
    so a concurrent loader that evicts cannot unload it under the request."""
    import threading

    from abstractcore.decision import loader, pool

    pool._reset_for_tests()
    monkeypatch.setattr(loader, "is_decision_model", lambda model, provider=None: True)
    dead = []

    class _LLM:
        def __init__(self, p, m):
            self.key = (p, m)
            self.alive = True

        def unload_model(self, model):
            self.alive = False
            dead.append(model)

    stop = threading.Event()
    errors = []

    def requester():
        while not stop.is_set():
            with pool.use("mlx", "a", factory=_LLM) as llm:
                for _ in range(50):
                    if not llm.alive:
                        errors.append("a dead while in use")
                        return

    def loader_thread():
        for i in range(200):
            pool.decision_provider("mlx", f"m{i}", factory=_LLM)
        stop.set()

    t1 = threading.Thread(target=requester)
    t2 = threading.Thread(target=loader_thread)
    t1.start(); t2.start(); t2.join(); stop.set(); t1.join()
    assert errors == []
    assert len(pool.resident()) <= pool.MAX_RESIDENT


def test_a_decision_model_answers_itself_never_the_hosted_default(fake_runtime, monkeypatch):
    """Review of #41, Q1: with an output.decision default of `typesafe` + `jev-latest`, a local Kev
    instance asked for output='decision' with no route answers itself; the document never goes to
    the hosted service because of a global setting. A route the request names still wins, and a
    text model (which cannot decide) still uses the default."""
    from abstractcore.decision import pool

    hosted = []
    monkeypatch.setattr(pool, "_acquire", lambda key, hold, factory, options: hosted.append(key[:2]) or (_ for _ in ()).throw(
        AssertionError(f"routed to {key[:2]}")))
    default = {"output.decision": {"provider": "typesafe", "model": "jev-latest"}}
    questions = {"escalate": "Does this need urgent human attention?"}

    kev = _LocalProvider(provider="mlx", model="fake/kev-4b")
    kev._abstractcore_capability_defaults = dict(default)
    result = kev.generate("Shoes arrived late.", output="decision", questions=questions)
    assert result.metadata["decision_provider"] == "mlx" and hosted == []
    assert fake_runtime == [("mlx", "fake/kev-4b", {})]

    with pytest.raises(AssertionError, match="routed to"):
        kev.generate("Shoes arrived late.", output={"modality": "decision", "provider": "typesafe", "model": "jev-latest"},
                     questions=questions)
    assert hosted == [("typesafe", "jev-latest")]

    text = _LocalProvider(provider="ollama", model="qwen3.5:9b")
    text._abstractcore_capability_defaults = dict(default)
    with pytest.raises(AssertionError, match="routed to"):
        text.generate("Shoes arrived late.", output="decision", questions=questions)
    assert hosted == [("typesafe", "jev-latest"), ("typesafe", "jev-latest")]
