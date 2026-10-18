"""Tests for cross-encoder reranking.

The model is faked (loading bge-reranker-base costs ~500MB and seconds per
test); what is tested is the contract around it — ordering, truncation,
failure handling, and the logit-preservation decision that feature f3
depends on.
"""
import math

import pytest

from src.config import Config
from src.rag.fusion import FusedResult
from src.rag.rerank import Reranker, rerank_features, sigmoid
from src.state import Passage

CFG = Config({"retrieval": {"rerank": {
    "model": "BAAI/bge-reranker-base",
    "input_candidates": 30, "output_top_k": 5, "batch_size": 16,
}}})


def result(pid, text, fusion_rank=1):
    passage = Passage(
        id=pid, source_type="arxiv", source_id="src", source_domain="arxiv.org",
        title="", text=text, section="Method", published="2023-01-01",
        sub_question_id="sq1", retrieval_score=0.01, rerank_score=None,
    )
    return FusedResult(passage=passage, rrf_score=0.01, dense_rank=fusion_rank)


class FakeCrossEncoder:
    """Scores by keyword presence, mimicking real logit magnitudes."""

    def __init__(self, scores=None, error=None):
        self.scores = scores
        self.error = error
        self.calls = []

    def predict(self, pairs, batch_size=None):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        if self.error:
            raise self.error
        if self.scores is not None:
            return self.scores[: len(pairs)]
        return [8.0 if "answer" in p[1] else -3.0 for p in pairs]


def reranker(model=None, **kw):
    return Reranker(CFG, model=model or FakeCrossEncoder(), **kw)


# --- sigmoid and the saturation problem ------------------------------------


def test_sigmoid_maps_logits_to_probability():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(8.0) > 0.99
    assert sigmoid(-8.0) < 0.01


def test_sigmoid_handles_extreme_negatives_without_overflow():
    assert sigmoid(-800.0) == pytest.approx(0.0, abs=1e-12)


def test_saturation_would_destroy_the_f3_signal():
    """Why raw logits are persisted. Two clearly different evidence
    strengths collapse to an indistinguishable gap in probability space —
    exactly where f3 is supposed to say 'sharp peak, not flat mush'."""
    logit_gap = 9.0 - 7.5
    prob_gap = sigmoid(9.0) - sigmoid(7.5)
    assert logit_gap == 1.5
    assert prob_gap < 0.001


# --- reranking behaviour ---------------------------------------------------


def test_reranking_reorders_by_score():
    """The C11 acceptance criterion: reranking measurably reorders."""
    results = [
        result("P1", "irrelevant filler text", fusion_rank=1),
        result("P2", "this passage contains the answer", fusion_rank=2),
        result("P3", "more filler", fusion_rank=3),
    ]
    out = reranker().rerank("what is the answer", results)
    assert out[0].passage["id"] == "P2"
    assert out[0].dense_rank == 2      # pre-rerank position is preserved


def test_raw_logit_is_stored_not_a_probability():
    out = reranker(FakeCrossEncoder(scores=[7.25])).rerank("q", [result("P1", "x")])
    assert out[0].passage["rerank_score"] == pytest.approx(7.25)
    assert out[0].passage["rerank_score"] > 1.0      # not squashed into [0,1]


def test_output_truncated_to_top_k():
    results = [result(f"P{i}", f"text {i}") for i in range(10)]
    assert len(reranker().rerank("q", results)) == 5     # config output_top_k


def test_explicit_top_k_overrides_config():
    results = [result(f"P{i}", f"text {i}") for i in range(10)]
    assert len(reranker().rerank("q", results, top_k=2)) == 2


def test_only_input_candidates_are_scored():
    """Cross-encoders are expensive; scoring the whole corpus is the failure
    mode reranking exists to avoid."""
    model = FakeCrossEncoder()
    results = [result(f"P{i}", f"text {i}") for i in range(50)]
    reranker(model).rerank("q", results, input_candidates=12)
    assert len(model.calls[0]["pairs"]) == 12


def test_batch_size_comes_from_config():
    model = FakeCrossEncoder()
    reranker(model).rerank("q", [result("P1", "x")])
    assert model.calls[0]["batch_size"] == 16


def test_query_is_paired_with_passage_text():
    model = FakeCrossEncoder()
    reranker(model).rerank("my query", [result("P1", "passage body")])
    assert model.calls[0]["pairs"][0] == ("my query", "passage body")


def test_scores_are_descending():
    results = [result(f"P{i}", "answer" if i % 2 else "filler") for i in range(6)]
    out = reranker().rerank("q", results)
    scores = [r.passage["rerank_score"] for r in out]
    assert scores == sorted(scores, reverse=True)


# --- degradation -----------------------------------------------------------


def test_model_failure_falls_back_to_fusion_order():
    """A worse ordering is recoverable; a crashed run is not."""
    results = [result("P1", "a"), result("P2", "b"), result("P3", "c")]
    out = reranker(FakeCrossEncoder(error=RuntimeError("OOM"))).rerank("q", results)
    assert [r.passage["id"] for r in out] == ["P1", "P2", "P3"]


def test_failure_is_traced():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    reranker(FakeCrossEncoder(error=RuntimeError("x")),
             tracer=FakeTracer()).rerank("q", [result("P1", "a")])
    assert events[0][0] == "rerank_failed"


def test_empty_candidates_returns_empty():
    assert reranker().rerank("q", []) == []


def test_empty_query_returns_empty():
    assert reranker().rerank("  ", [result("P1", "x")]) == []


def test_rerank_is_traced():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    reranker(tracer=FakeTracer()).rerank(
        "q", [result("P1", "filler"), result("P2", "answer")]
    )
    name, payload = events[0]
    assert name == "rerank"
    assert payload["candidates"] == 2 and payload["kept"] == 2


# --- features f1-f3 --------------------------------------------------------


def test_features_computed_from_logits():
    results = [result(f"P{i}", "x") for i in range(6)]
    out = reranker(FakeCrossEncoder(scores=[9.0, 8.0, 7.0, 6.0, 7.5, 1.0])).rerank(
        "q", results, top_k=6
    )
    features = rerank_features(out)
    assert features["f1_max_rerank"] == pytest.approx(9.0)
    assert features["f2_mean_top3"] == pytest.approx((9.0 + 8.0 + 7.5) / 3)
    assert features["f3_score_gap"] == pytest.approx(9.0 - 6.0)


def test_f3_is_none_with_fewer_than_five_results():
    """None is honest. Zero would read as 'no gap' — a strong claim about
    ambiguity the data does not license."""
    out = reranker(FakeCrossEncoder(scores=[5.0, 4.0])).rerank(
        "q", [result("P1", "a"), result("P2", "b")]
    )
    assert rerank_features(out)["f3_score_gap"] is None


def test_features_none_when_nothing_reranked():
    assert rerank_features([]) == {
        "f1_max_rerank": None, "f2_mean_top3": None, "f3_score_gap": None,
    }


def test_flat_scores_yield_a_small_gap():
    """The 'flat mush' case f3 exists to detect."""
    out = reranker(FakeCrossEncoder(scores=[2.0, 1.99, 1.98, 1.97, 1.96])).rerank(
        "q", [result(f"P{i}", "x") for i in range(5)], top_k=5
    )
    assert rerank_features(out)["f3_score_gap"] < 0.1


# --- the activation-function trap (found live at C17) ----------------------


def test_probability_signature_is_detected():
    """sentence-transformers defaults activation_fn to nn.Sigmoid when
    num_labels == 1, which bge-reranker-base is. Left at the default,
    predict() returns probabilities and C11's decision is silently reversed:
    ranking is unchanged (sigmoid is monotonic), nothing errors, and only f3
    stops carrying information."""
    from src.rag.rerank import looks_like_probabilities

    assert looks_like_probabilities([0.969, 0.961, 0.950, 0.889, 0.880])


def test_logits_are_not_mistaken_for_probabilities():
    from src.rag.rerank import looks_like_probabilities

    assert not looks_like_probabilities([9.0, 7.5, 6.0, 2.0, -3.0])


def test_small_in_range_sample_does_not_trigger():
    """A logit can legitimately fall in [0,1]; the check only fires on the
    saturation signature, and it warns rather than transforming."""
    from src.rag.rerank import looks_like_probabilities

    assert not looks_like_probabilities([0.4, 0.2])


def test_sigmoid_output_is_warned_about_not_silently_accepted():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    model = FakeCrossEncoder(scores=[0.97, 0.96, 0.95, 0.89, 0.88])
    r = reranker(model, tracer=FakeTracer())
    out = r.rerank("q", [result(f"P{i}", "x") for i in range(5)], top_k=5)

    assert any(name == "rerank_activation_warning" for name, _ in events)
    # The scores are NOT transformed — guessing an inverse sigmoid would be
    # worse than a loud warning.
    assert out[0].passage["rerank_score"] == pytest.approx(0.97)


def test_warning_fires_only_once():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append(name)

    r = reranker(FakeCrossEncoder(scores=[0.97, 0.96, 0.95, 0.9, 0.88]),
                 tracer=FakeTracer())
    for _ in range(3):
        r.rerank("q", [result(f"P{i}", "x") for i in range(5)], top_k=5)
    assert events.count("rerank_activation_warning") == 1


def test_the_measured_cost_of_the_bug():
    """What this actually cost, from a real C17 run: f3 came out at 0.08
    because scores above ~0.95 compress into a few thousandths. The
    sufficiency model would have been fit on a feature that cannot
    distinguish strong evidence from marginally stronger evidence."""
    squashed = reranker(FakeCrossEncoder(
        scores=[0.969, 0.965, 0.961, 0.955, 0.889]
    )).rerank("q", [result(f"P{i}", "x") for i in range(5)], top_k=5)
    raw = reranker(FakeCrossEncoder(
        scores=[9.0, 8.5, 8.0, 7.0, 2.5]
    )).rerank("q", [result(f"P{i}", "x") for i in range(5)], top_k=5)

    assert rerank_features(squashed)["f3_score_gap"] < 0.1
    assert rerank_features(raw)["f3_score_gap"] > 6.0