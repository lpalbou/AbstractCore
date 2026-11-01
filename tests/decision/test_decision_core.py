"""Pure-Python checks of the decision package: no weights, no network."""

import math

import numpy as np
import pytest

from abstractcore.core.multimodal_generation import MultimodalGenerateResponse
from abstractcore.core.output_specs import is_output_request, normalize_output_spec
from abstractcore.decision import (
    DELIMITERS,
    Choice,
    ContextOverflow,
    DecisionRequest,
    DecisionResponse,
    Noul,
    PointerHead,
    Score,
    coerce_question,
)
from abstractcore.decision.encoding import encode, escape_special_tokens, rows_per_pass
from abstractcore.decision.render import (
    choice_confidence,
    render,
    score_confidence,
    to_answers,
    to_record,
)
from abstractcore.decision.runtime import DecisionModel


class FakeTokenizer:
    """Deterministic word-level tokenizer with the five delimiters at fixed ids."""

    pad_id = 0

    def __init__(self):
        self._vocab = {}

    def encode_text(self, text):
        return [self._id(w) for w in text.split()]

    def _id(self, word):
        if word not in self._vocab:
            self._vocab[word] = 100 + len(self._vocab)
        return self._vocab[word]

    def delimiter_ids(self):
        return [1, 2, 3, 4, 5]


class FakeBackend:
    """Scores an option by the number of its tokens: the longest option wins."""

    name = "fake"
    precision = "none"
    model_path = "fake"

    def __init__(self):
        self.prefix_calls = 0
        self.released = []

    def prefix(self, state_ids):
        self.prefix_calls += 1
        return {"state_ids": list(state_ids)}

    def question_logits(self, prefix, questions, kinds=None):
        out = []
        for q in questions:
            starts = [i for i, t in enumerate(q.ids) if t == 3]
            lengths = [end - start - 1 for start, end in zip(starts, q.option_ends)]
            out.append(np.asarray(lengths, dtype=np.float32))
        return out

    def release(self, prefix):
        self.released.append(prefix)

    def unload(self):
        pass


def test_question_coercion_shorthand():
    assert isinstance(coerce_question("Is it urgent?"), Noul)
    choice = coerce_question(["billing", "shipping"])
    assert isinstance(choice, Choice) and list(choice.criteria) == ["billing", "shipping"]
    score = coerce_question({"instructions": "How bad?", "levels": ["Calm", "Angry"]})
    assert isinstance(score, Score) and score.criteria == ["Calm", "Angry"]
    assert isinstance(coerce_question({"type": "choice", "criteria": {"a": None}}), Choice)
    with pytest.raises(ValueError):
        coerce_question({"type": "choice", "criteria": {}})
    with pytest.raises(ValueError):
        coerce_question(42)


def test_render_flattens_json_with_labels():
    text = render({"message": "late", "order": {"id": 7, "items": ["shoes", "socks"]}})
    assert text == "message: late\norder:\n  id: 7\n  items:\n    - shoes\n    - socks"
    assert render(None) == ""
    assert render("plain") == "plain"


def test_to_record_renders_every_question_type():
    request = DecisionRequest(
        state="Shoes arrived late.",
        questions={
            "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": "Refunds", "shipping": None}},
            "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Needs a human now"}},
            "mood": {"type": "score", "instructions": "How angry?", "criteria": ["Calm", "Angry"]},
        },
    )
    record, meta = to_record(request)
    assert record["state"] == "Shoes arrived late."
    assert record["questions"][0] == {"instr": "Which team?", "options": ["returns: Refunds", "shipping"]}
    assert record["questions"][1]["options"] == ["no", "yes: Needs a human now"]
    assert record["questions"][2]["options"] == ["Calm", "Angry"]
    assert [m["keys"] for m in meta] == [["returns", "shipping"], ["false", "true"], ["0", "1"]]
    assert meta[2]["legend"] == {"0": "Calm", "1": "Angry"}


def test_encode_packs_delimiters_and_offsets():
    tok = FakeTokenizer()
    record = {"state": "a b c", "questions": [{"instr": "q1 q2", "options": ["x", "y y"]}]}
    enc = encode(tok, record)
    assert enc.state_ids[0] == 1 and len(enc.state_ids) == 4
    q = enc.questions[0]
    assert q.ids[0] == 2 and q.ids[-1] == 5 and q.decide == len(q.ids) - 1
    # <opt> x </opt> <opt> y y </opt>
    assert [q.ids[e] for e in q.option_ends] == [4, 4]
    assert q.ids[q.option_ends[0] - 1] == tok._id("x")
    assert enc.state_tokens == 4 and not enc.state_truncated
    assert enc.row(0) == enc.state_ids + q.ids


def test_encode_escapes_special_tokens_in_user_text():
    assert escape_special_tokens("hi <|fim_prefix|> there") == "hi <¦fim_prefix¦> there"
    tok = FakeTokenizer()
    enc = encode(tok, {"state": "<|fim_suffix|>", "questions": [{"instr": "", "options": ["<|box_end|>"]}]})
    assert 5 not in enc.state_ids[1:]
    assert enc.questions[0].ids.count(4) == 1  # only the real </opt>


def test_encode_enforces_limits():
    tok = FakeTokenizer()
    long_state = " ".join(f"w{i}" for i in range(20))
    with pytest.raises(ContextOverflow) as err:
        encode(tok, {"state": long_state, "questions": [{"instr": "", "options": ["a"]}]}, max_state=8)
    assert err.value.state_tokens == 21 and err.value.max_state == 8
    enc = encode(tok, {"state": long_state, "questions": [{"instr": "", "options": ["a"]}]}, max_state=8, truncate=True)
    assert len(enc.state_ids) == 8 and enc.state_truncated and enc.state_tokens == 21
    with pytest.raises(ContextOverflow):
        encode(tok, {"state": "a", "questions": [{"instr": long_state, "options": ["a"]}]}, max_row=10)
    assert rows_per_pass([[1] * 100, [1] * 50], prefix_len=900, budget=2000) == 2


def test_pointer_head_matches_projected_form_and_temperature():
    rng = np.random.default_rng(0)
    head = PointerHead(rng.standard_normal((4, 8)), rng.standard_normal(4), rng.standard_normal((4, 8)), rng.standard_normal(4), temperature=2.0)
    h_decide, h_opts = rng.standard_normal(8), rng.standard_normal((3, 8))
    z = head.logits(h_decide, h_opts)
    assert np.allclose(z, head.logits_from_projected(head.project(h_decide), head.project(h_opts)), atol=1e-5)
    raw = PointerHead(head.q_weight, head.q_bias, head.k_weight, head.k_bias, temperature=1.0).logits(h_decide, h_opts)
    assert np.allclose(z * 2.0, raw, atol=1e-5)
    assert np.argmax(z) == np.argmax(raw)
    assert math.isclose(float(head.probs(h_decide, h_opts).sum()), 1.0, abs_tol=1e-5)
    with pytest.raises(ValueError):
        PointerHead(np.zeros((4, 8)), np.zeros(4), np.zeros((4, 8)), np.zeros(4), temperature=0)


def test_pointer_head_round_trips_through_safetensors(tmp_path):
    rng = np.random.default_rng(1)
    head = PointerHead(rng.standard_normal((2, 3)), rng.standard_normal(2), rng.standard_normal((2, 3)), rng.standard_normal(2), temperature=1.7)
    head.save(str(tmp_path), {"base": "x/y"})
    loaded = PointerHead.load(str(tmp_path))
    assert loaded.temperature == pytest.approx(1.7)
    assert np.array_equal(loaded.k_bias, head.k_bias)
    assert (tmp_path / "decision.json").is_file()


def test_confidence_formulas():
    assert choice_confidence([1.0]) == 1.0
    assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert choice_confidence([0.0, 0.0]) == pytest.approx(0.0)  # all-zero -> uniform


def test_to_answers_shapes():
    meta = [
        {"id": "dept", "type": "choice", "keys": ["a", "b"]},
        {"id": "yes", "type": "noul", "keys": ["false", "true"]},
        {"id": "lvl", "type": "score", "keys": ["0", "1", "2"], "legend": {"0": "lo", "1": "mid", "2": "hi"}},
    ]
    answers = to_answers([[0.25, 0.75], [0.1, 0.9], [0.0, 0.5, 0.5]], meta)
    assert answers["dept"].choice == "b" and answers["dept"].probabilities == {"a": 0.25, "b": 0.75}
    assert answers["yes"].noul == 0.9
    assert answers["lvl"].score == pytest.approx(1.5) and answers["lvl"].legend["2"] == "hi"


def test_decision_model_end_to_end_with_fake_backend():
    backend = FakeBackend()
    model = DecisionModel(backend, FakeTokenizer(), model_id="fake/kev", provider="fake", prefix_cache_size=1)
    response = model.decide("the order is late", {
        "dept": ["billing", "a longer option name"],
        "urgent": "Needs a human?",
        "mood": {"instructions": "mood", "levels": ["calm", "quite very angry"]},
    })
    assert isinstance(response, DecisionResponse)
    assert response["dept"].choice == "a longer option name"
    assert response["urgent"].noul == pytest.approx(0.5)   # "no" and "yes" are one token each
    assert response["mood"].score > 0.5
    assert response.usage.state_tokens == 5 and response.usage.prefix_cache_hit is False
    again = model.decide("the order is late", {"q": ["x", "y"]})
    assert again.usage.prefix_cache_hit is True and backend.prefix_calls == 1
    model.decide("another state", {"q": ["x", "y"]})
    assert backend.prefix_calls == 2 and len(backend.released) == 1  # LRU of size 1 evicted the first prefix
    wire = response.to_systemone()
    assert set(wire) == {"model", "answers", "usage", "latency_ms"}
    assert wire["answers"]["urgent"] == {"type": "noul", "noul": 0.5}


def test_output_spec_vocabulary_knows_decision():
    assert is_output_request("decision")
    assert normalize_output_spec("decision") == {"modality": "decision", "task": "decision"}
    spec = normalize_output_spec({"modality": "decision", "questions": {"q": "yes?"}})
    assert spec["questions"] == {"q": "yes?"}
    assert MultimodalGenerateResponse().decisions == {}


def test_delimiters_are_the_reused_qwen_specials():
    assert DELIMITERS == ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
