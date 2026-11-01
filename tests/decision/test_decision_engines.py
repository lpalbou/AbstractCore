"""The non-Kev engines with fakes: label-logit readout (Lev / OpenJev conventions),
the remote System One engine (mocked HTTP), Laya's prompt helpers, the typesafe provider."""

import json

import httpx
import numpy as np
import pytest

from abstractcore.decision.engines.labels import LabelLogitDecisionModel
from abstractcore.decision.engines.laya import render_options, temperature_bucket
from abstractcore.decision.engines.remote import RemoteSystemOneModel
from abstractcore.decision.kinds import LEV_N_RATINGS, lev_spec, openjev_spec
from abstractcore.decision.types import DecisionRequest


class WordTokenizer:
    """Every whitespace-separated word is one token; capitalised codes are single tokens."""

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = 1000 + len(self.vocab)
        return self.vocab[w]

    def encode_text(self, text):
        return [self._id(w) for w in text.split()]

    encode_prompt = encode_text


class PreferLabelBackend:
    """Scores label i with logit = preference[i] (so the test controls the answer)."""

    name = "fake"
    precision = "none"

    def __init__(self, preference):
        self.preference = list(preference)
        self.prompts = []

    def last_token_logits(self, ids, candidate_ids):
        self.prompts.append(list(ids))
        return np.asarray(self.preference[: len(candidate_ids)], dtype=np.float32)

    def unload(self):
        pass


def test_lev_choice_averages_two_option_orders_and_reads_noul_as_rating():
    spec = lev_spec({"choice:A:small": 1.0, "noul:A": 1.0, "score:A": 1.0})
    backend = PreferLabelBackend([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # always prefers the FIRST label
    model = LabelLogitDecisionModel(backend, WordTokenizer(), spec, model_id="fake/lev")
    req = DecisionRequest(state="doc", questions={"team": ["returns", "billing"], "urgent": "Urgent?"})
    probs = model.probabilities(req)
    # first-label bias cancels across the two orders: both options end up equal
    assert probs[0][0] == pytest.approx(probs[0][1], abs=1e-6)
    assert len(backend.prompts) == 3   # 2 orders for the choice + 1 for the noul
    # noul: a strong preference for rating 0 -> P(true) near 0
    # P(rating 0) = e^3/(e^3+8) = 0.715, the rest spread evenly: expected rating/8 = 0.16
    assert probs[1][1] == pytest.approx(0.160, abs=0.01) and probs[1][0] + probs[1][1] == pytest.approx(1.0, abs=1e-6)
    assert model.n_outputs(req.questions["urgent"]) == LEV_N_RATINGS


def test_openjev_noul_is_true_first_and_temperature_applies():
    spec = openjev_spec(temperature=0.5, noul_factor=1.0)
    backend = PreferLabelBackend([1.0, 0.0])
    model = LabelLogitDecisionModel(backend, WordTokenizer(), spec, model_id="fake/openjev")
    probs = model.probabilities(DecisionRequest(state="doc", questions={"ok": "Is it fine?"}))
    # label A = "true" for OpenJev; the readout is reported back as [false, true]
    assert probs[0][1] > probs[0][0]
    expected_true = np.exp(2.0) / (np.exp(2.0) + 1.0)   # logit 1.0 / T 0.5
    assert probs[0][1] == pytest.approx(expected_true, abs=1e-5)
    rendered = model.render("the state", "q", "choice", "Which?", [{"key": "a", "description": None, "label": "A"}, {"key": "b", "description": "bee", "label": "B"}])
    assert "[A] a:" in rendered and "[B] b: bee" in rendered and rendered.endswith("<think>\n\n</think>\n\n")


def test_lev_template_renders_scale_for_noul_and_sorted_json():
    spec = lev_spec()
    model = LabelLogitDecisionModel(PreferLabelBackend([0.0] * 9), WordTokenizer(), spec, model_id="fake/lev")
    text = model.render({"b": 1, "a": 2}, "q", "noul", "Is it?", [{"key": "false", "description": None, "label": "A"}, {"key": "true", "description": "yes indeed", "label": "B"}])
    assert '{"a": 2, "b": 1}' in text and "# Scale" in text and "yes: yes indeed" in text
    text = model.render("s", "q", "score", "How much?", [{"key": "0", "description": "low", "label": "A"}, {"key": "1", "description": "high", "label": "B"}])
    assert "A. (level 0 of 1) low" in text


def test_remote_systemone_engine_round_trip():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"dept": {"type": "choice", "choice": "returns", "confidence": 0.4, "probabilities": {"returns": 0.7, "billing": 0.3}},
                        "urgent": {"type": "noul", "noul": 0.93}},
            "usage": {"input_tokens": 50, "output_tokens": 20}, "latency_ms": 210.0,
        })

    engine = RemoteSystemOneModel("https://api.example.test", "jev-latest", api_key="k", provider="typesafe")
    engine._client = httpx.Client(transport=httpx.MockTransport(handler))
    response = engine.decide("late order", {"dept": ["returns", "billing"], "urgent": "Urgent?"})
    assert seen["url"] == "https://api.example.test/v1/systemone" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "jev-latest" and seen["body"]["questions"]["dept"]["criteria"] == {"returns": None, "billing": None}
    assert "criteria" not in seen["body"]["questions"]["urgent"]
    assert response["dept"].choice == "returns" and response["urgent"].noul == 0.93
    assert response.model == "jev-1.13.0" and response.usage.input_tokens == 50 and response.backend == "remote"


def test_remote_systemone_engine_maps_errors():
    from abstractcore.decision import ContextOverflow
    from abstractcore.exceptions import AuthenticationError

    def handler(request: httpx.Request) -> httpx.Response:
        code = int(request.headers.get("x-code", "401"))
        return httpx.Response(code, json={"error": {"message": "nope"}})

    engine = RemoteSystemOneModel("https://api.example.test", "jev-latest", headers={"x-code": "401"})
    engine._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(AuthenticationError):
        engine.decide("s", {"q": "yes?"})
    engine.extra_headers["x-code"] = "422"
    with pytest.raises(ContextOverflow):
        engine.decide("s", {"q": "yes?"})


def test_laya_prompt_helpers():
    assert render_options("noul", None) == ["false: no, the statement does not hold", "true: yes, the statement holds"]
    assert render_options("choice", {"a": None, "b": {"x": 1}}) == ["a", "b: {\"x\": 1}"]
    assert render_options("score", ["Calm", "Angry"]) == ["level 0: Calm", "level 1: Angry"]
    assert temperature_bucket("choice", 2) == "choice:2" and temperature_bucket("choice", 7) == "choice:6-10" and temperature_bucket("score", 4) == "score:3-5"


def test_typesafe_provider_uses_remote_engine(monkeypatch):
    from abstractcore.providers.typesafe_provider import SystemOneProvider, TypeSafeProvider

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TypeSafe API key"):
        TypeSafeProvider(model="jev-latest")
    llm = TypeSafeProvider(model="jev-latest", api_key="secret")
    engine = llm.decision_model()
    assert engine.endpoint == "https://api.typesafe.ai/v1/systemone" and llm.model_type == "decision"
    from abstractcore.exceptions import DecisionModelError

    with pytest.raises(DecisionModelError, match="decision model"):
        llm.generate("hello")
    local = SystemOneProvider(model="kev-latest", base_url="http://localhost:8009/")
    assert local.decision_model().endpoint == "http://localhost:8009/v1/systemone"
    with pytest.raises(ValueError, match="base_url"):
        SystemOneProvider(model="x")


class PrefixAwareBackend(PreferLabelBackend):
    """Scores like PreferLabelBackend but through prefix() / suffix_logits(), recording the split."""

    def __init__(self, preference):
        super().__init__(preference)
        self.prefixes = []
        self.suffixes = []

    def prefix(self, ids):
        self.prefixes.append(list(ids))
        return {"ids": list(ids)}

    def suffix_logits(self, handle, suffixes, candidate_ids):
        self.suffixes.append([list(s) for s in suffixes])
        return [self.last_token_logits(handle["ids"] + list(s), candidate_ids) for s in suffixes]

    def release(self, handle):
        pass


def test_label_engine_reuses_the_shared_prompt_prefix():
    spec = lev_spec({"choice:A:small": 1.0, "noul:A": 1.0})
    tok = WordTokenizer()
    plain = LabelLogitDecisionModel(PreferLabelBackend([2.0, 1.0, 0.0] + [0.0] * 6), tok, spec, model_id="fake/lev")
    aware_backend = PrefixAwareBackend([2.0, 1.0, 0.0] + [0.0] * 6)
    aware = LabelLogitDecisionModel(aware_backend, tok, spec, model_id="fake/lev")
    req = DecisionRequest(state="a long shared evidence text about an order", questions={"team": ["returns", "billing"], "urgent": "Urgent?"})
    expected = plain.probabilities(req)
    got = aware.probabilities(req)
    for e, g in zip(expected, got):
        assert np.allclose(e, g, atol=1e-6)
    # one prefix for the whole request, three suffixes (two option orders + the noul), prefix covers the evidence
    assert len(aware_backend.prefixes) == 1 and len(aware_backend.suffixes) == 1 and len(aware_backend.suffixes[0]) == 3
    assert aware.last_prefix_tokens == len(aware_backend.prefixes[0])
    assert tok._id("evidence") in aware_backend.prefixes[0]
    # the prefix is reused by a second request on the same document
    aware.probabilities(DecisionRequest(state="a long shared evidence text about an order", questions={"other": ["x", "y"]}))
    assert len(aware_backend.prefixes) == 1


def test_laya_window_plan_and_aggregation():
    from abstractcore.decision.engines.laya import aggregate_windows, window_plan

    assert window_plan(100, 300) == [(0, 100)]
    plan = window_plan(1000, 300)
    assert plan[0] == (0, 300) and plan[-1][1] == 1000 and all(b - a <= 300 for a, b in plan)
    assert all(plan[i + 1][0] - plan[i][0] == 150 for i in range(len(plan) - 2))
    with pytest.raises(ValueError):
        window_plan(1000, 300, stride=400)
    noul = aggregate_windows("noul", [np.array([0.9, 0.1]), np.array([0.3, 0.7]), np.array([0.8, 0.2])])
    assert noul[1] == pytest.approx(0.7)
    choice = aggregate_windows("choice", [np.array([0.4, 0.6]), np.array([0.95, 0.05])])
    assert choice[0] == pytest.approx(0.95)
