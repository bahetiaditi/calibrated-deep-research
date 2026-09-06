"""Tests for contradiction detection.

The NLI model is faked; what is tested is everything around it — label
resolution, cross-source scoping, symmetry, thresholding and the f5 rate.
Those are where the correctness risks live, and all of them are silent
failures if wrong.
"""
import pytest

from src.config import Config
from src.rag.contradiction import (
    DEFAULT_LABELS,
    MAX_PREMISE_CHARS,
    ContradictionDetector,
    _softmax,
)
from src.state import Passage

CFG = Config({"retrieval": {"contradiction": {
    "model": "cross-encoder/nli-deberta-v3-small",
    "max_pairs_per_subquestion": 15,
    "contradiction_threshold": 0.60,
    "cross_source_only": True,
    "symmetric": True,
}}})


def passage(pid, text, source="paperA", section="Results"):
    return Passage(
        id=pid, source_type="arxiv", source_id=source, source_domain="arxiv.org",
        title="", text=text, section=section, published="2023-01-01",
        sub_question_id="sq1", retrieval_score=0.1, rerank_score=5.0,
    )


class FakeNLI:
    """Emits three logits per pair, ordered (contradiction, entail, neutral).

    `contradicting` is a set of frozensets of text fragments that conflict.
    """

    def __init__(self, contradicting=None, id2label=None, error=None,
                 scalar=False):
        self.contradicting = contradicting or set()
        self.error = error
        self.scalar = scalar
        self.calls = []

        class Inner:
            pass

        class Config_:
            pass

        inner, conf = Inner(), Config_()
        conf.id2label = id2label if id2label is not None else {
            0: "contradiction", 1: "entailment", 2: "neutral"
        }
        inner.config = conf
        self.model = inner

    def predict(self, pairs):
        self.calls.append(list(pairs))
        if self.error:
            raise self.error
        out = []
        for a, b in pairs:
            conflict = any(
                x in a and y in b for pair in self.contradicting
                for x, y in (tuple(pair), tuple(pair)[::-1])
            )
            if self.scalar:
                out.append(0.95 if conflict else 0.05)
            elif conflict:
                out.append([4.0, -2.0, -1.0])     # contradiction dominates
            else:
                out.append([-3.0, 3.0, 0.0])      # entailment dominates
        return out


def detector(model=None, **kw):
    return ContradictionDetector(CFG, model=model or FakeNLI(), **kw)


# --- the acceptance check --------------------------------------------------


def test_contradicting_pair_is_detected():
    """C12 criterion, half one."""
    model = FakeNLI(contradicting={frozenset({"rank 8 is optimal",
                                              "rank 64 is optimal"})})
    records = detector(model).detect([
        passage("P1", "For 7B models rank 8 is optimal.", source="paperA"),
        passage("P2", "For 7B models rank 64 is optimal.", source="paperB"),
    ])
    assert len(records) == 1
    assert {records[0].passage_a, records[0].passage_b} == {"P1", "P2"}
    assert records[0].score >= 0.60
    assert records[0].cross_source is True


def test_agreeing_pair_is_not_detected():
    """C12 criterion, half two."""
    records = detector().detect([
        passage("P1", "LoRA reduces trainable parameters.", source="paperA"),
        passage("P2", "LoRA lowers the number of trained weights.", source="paperB"),
    ])
    assert records == []


# --- label resolution ------------------------------------------------------


def test_contradiction_index_read_from_model_config():
    """Hardcoding the index is a silent correctness bug: a wrong index gives
    plausible numbers pointing the wrong way, and nothing errors."""
    model = FakeNLI(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    assert detector(model).contradiction_index() == 2


def test_index_respects_a_reordered_head():
    model = FakeNLI(id2label={0: "neutral", 1: "contradiction", 2: "entailment"})
    d = detector(model)
    assert d.contradiction_index() == 1


def test_falls_back_to_documented_order_without_labels():
    model = FakeNLI(id2label={})
    assert detector(model).contradiction_index() == DEFAULT_LABELS.index(
        "contradiction"
    )


def test_index_is_cached():
    model = FakeNLI()
    d = detector(model)
    assert d.contradiction_index() == d.contradiction_index()


# --- cross-source scoping --------------------------------------------------


def test_same_source_pairs_are_excluded_by_default():
    """A hypothesis stated before it is refuted is not sources disagreeing,
    and counting it would inflate f5 on a corpus that actually agrees."""
    model = FakeNLI(contradicting={frozenset({"we expect X", "X does not hold"})})
    records = detector(model).detect([
        passage("P1", "we expect X to improve results.", source="paperA"),
        passage("P2", "X does not hold in our experiments.", source="paperA"),
    ])
    assert records == []


def test_same_source_pairs_included_when_configured():
    cfg = Config({"retrieval": {"contradiction": {
        "model": "m", "max_pairs_per_subquestion": 15,
        "contradiction_threshold": 0.60, "cross_source_only": False,
        "symmetric": True,
    }}})
    model = FakeNLI(contradicting={frozenset({"we expect X", "X does not hold"})})
    records = ContradictionDetector(cfg, model=model).detect([
        passage("P1", "we expect X to improve results.", source="paperA"),
        passage("P2", "X does not hold in our experiments.", source="paperA"),
    ])
    assert len(records) == 1
    assert records[0].cross_source is False


# --- symmetry --------------------------------------------------------------


def test_both_directions_are_scored():
    model = FakeNLI()
    detector(model).detect([
        passage("P1", "alpha", source="A"),
        passage("P2", "beta", source="B"),
    ])
    assert len(model.calls[0]) == 2      # one pair, both directions


def test_contradiction_found_in_one_direction_only_is_kept():
    """NLI heads are asymmetric in practice; missing a real contradiction
    costs more than the extra forward pass."""
    class OneWay(FakeNLI):
        def predict(self, pairs):
            self.calls.append(list(pairs))
            # Only the reversed direction fires.
            return [
                [4.0, -2.0, -1.0] if a.startswith("beta") else [-3.0, 3.0, 0.0]
                for a, b in pairs
            ]

    records = detector(OneWay()).detect([
        passage("P1", "alpha claim", source="A"),
        passage("P2", "beta claim", source="B"),
    ])
    assert len(records) == 1


# --- thresholding and budget -----------------------------------------------


def test_below_threshold_is_ignored():
    class Weak(FakeNLI):
        def predict(self, pairs):
            return [[0.2, 0.1, 0.1] for _ in pairs]   # ~0.37 after softmax

    assert detector(Weak()).detect([
        passage("P1", "a", source="A"), passage("P2", "b", source="B"),
    ]) == []


def test_pair_budget_is_respected():
    model = FakeNLI()
    passages = [passage(f"P{i}", f"text {i}", source=f"paper{i}") for i in range(10)]
    detector(model).detect(passages, max_pairs=4)
    assert len(model.calls[0]) == 8      # 4 pairs x 2 directions


def test_records_sorted_by_score():
    class Graded(FakeNLI):
        def predict(self, pairs):
            return [[float(4 - i % 3), -2.0, -1.0] for i, _ in enumerate(pairs)]

    records = detector(Graded()).detect(
        [passage(f"P{i}", f"t{i}", source=f"p{i}") for i in range(4)]
    )
    assert [r.score for r in records] == sorted(
        [r.score for r in records], reverse=True
    )


# --- degradation -----------------------------------------------------------


def test_fewer_than_two_passages_yields_nothing():
    assert detector().detect([passage("P1", "only one", source="A")]) == []


def test_empty_text_passages_are_skipped():
    assert detector().detect([
        passage("P1", "   ", source="A"), passage("P2", "real", source="B"),
    ]) == []


def test_model_failure_returns_no_records():
    """Missing contradiction data is a weaker signal, not a broken run."""
    assert detector(FakeNLI(error=RuntimeError("OOM"))).detect([
        passage("P1", "a", source="A"), passage("P2", "b", source="B"),
    ]) == []


def test_scalar_head_is_handled():
    model = FakeNLI(contradicting={frozenset({"up", "down"})}, scalar=True)
    records = detector(model).detect([
        passage("P1", "the value goes up", source="A"),
        passage("P2", "the value goes down", source="B"),
    ])
    assert len(records) == 1


def test_long_passages_are_truncated_before_scoring():
    """NLI models are trained on sentence pairs, not 1000-char chunks."""
    model = FakeNLI()
    detector(model).detect([
        passage("P1", "x" * 5000, source="A"),
        passage("P2", "y" * 5000, source="B"),
    ])
    premise, hypothesis = model.calls[0][0]
    assert len(premise) <= MAX_PREMISE_CHARS
    assert len(hypothesis) <= MAX_PREMISE_CHARS


# --- feature f5 ------------------------------------------------------------


def test_contradiction_rate_is_the_fraction_of_scored_pairs():
    model = FakeNLI(contradicting={frozenset({"rank 8", "rank 64"})})
    rate, records = detector(model).contradiction_rate([
        passage("P1", "rank 8 is best", source="A"),
        passage("P2", "rank 64 is best", source="B"),
        passage("P3", "unrelated content", source="C"),
    ])
    assert len(records) == 1
    assert rate == pytest.approx(1 / 3)


def test_rate_is_zero_with_too_few_passages():
    """No disagreement was OBSERVED, which differs from evidence agreeing.
    Thinness is already captured by f1, f2 and f4."""
    rate, records = detector().contradiction_rate([passage("P1", "one", source="A")])
    assert rate == 0.0 and records == []


def test_record_carries_sections_for_downstream_weighting():
    """A Related Work chunk describes someone else's claim, so a conflict
    with it may be a competing method rather than a genuine dispute."""
    model = FakeNLI(contradicting={frozenset({"faster", "slower"})})
    records = detector(model).detect([
        passage("P1", "our method is faster", source="A", section="Results"),
        passage("P2", "the method is slower", source="B", section="Related Work"),
    ])
    assert records[0].section_a == "Results"
    assert records[0].section_b == "Related Work"


def test_record_serialises_for_state():
    model = FakeNLI(contradicting={frozenset({"up", "down"})})
    record = detector(model).detect([
        passage("P1", "goes up", source="A"), passage("P2", "goes down", source="B"),
    ])[0].to_dict()
    assert set(record) >= {"passage_a", "passage_b", "score", "cross_source"}


def test_scan_is_traced():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    detector(tracer=FakeTracer()).detect([
        passage("P1", "a", source="A"), passage("P2", "b", source="B"),
    ])
    name, payload = events[0]
    assert name == "contradiction_scan" and payload["pairs_scored"] == 1


def test_softmax_normalises():
    assert sum(_softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)