"""Tests for retrieval metrics and the R1-R4 harness.

Metrics are checked against hand-computed values. If nDCG is wrong, every
conclusion drawn from the R1-R4 table is wrong in a way that looks entirely
plausible, so these are worth being pedantic about.
"""
import json

import pytest

from analysis.retrieval_eval import (
    CONFIGS,
    IRRELEVANT,
    PARTIAL,
    RELEVANT,
    ConfigScore,
    Judgments,
    RetrievalEvaluator,
    check_expected_ordering,
    dcg_at_k,
    evaluate_ranking,
    gain,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    render_table,
)


# --- gains and DCG ---------------------------------------------------------


def test_exponential_gain():
    """One fully relevant passage is worth three partial ones — which matches
    how evidence works here: a passage that answers the sub-question beats
    three that circle it."""
    assert gain(IRRELEVANT) == 0.0
    assert gain(PARTIAL) == 1.0
    assert gain(RELEVANT) == 3.0


def test_dcg_discounts_by_log_rank():
    # rank 1 -> /log2(2)=1, rank 2 -> /log2(3)
    expected = 3.0 / 1.0 + 3.0 / 1.5849625007
    assert dcg_at_k([2, 2], 2) == pytest.approx(expected, rel=1e-6)


def test_dcg_respects_k():
    assert dcg_at_k([2, 2, 2], 1) == pytest.approx(3.0)


def test_order_matters():
    assert dcg_at_k([2, 0], 2) > dcg_at_k([0, 2], 2)


# --- nDCG ------------------------------------------------------------------


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([2, 1], [2, 1], 10) == pytest.approx(1.0)


def test_reversed_ranking_scores_less():
    assert ndcg_at_k([1, 2], [2, 1], 10) < 1.0


def test_ideal_uses_all_judgments_not_just_retrieved():
    """Normalising against only what was retrieved would score a system that
    found one relevant passage as perfectly as one that found ten."""
    partial = ndcg_at_k([2], [2, 2, 2], 10)
    complete = ndcg_at_k([2, 2, 2], [2, 2, 2], 10)
    assert partial < complete == pytest.approx(1.0)


def test_no_relevant_judgments_scores_zero():
    assert ndcg_at_k([0, 0], [0, 0], 10) == 0.0


def test_irrelevant_results_contribute_nothing():
    assert ndcg_at_k([0, 0, 2], [2], 10) < ndcg_at_k([2], [2], 10)


# --- recall ----------------------------------------------------------------


def test_recall_is_pool_relative():
    assert recall_at_k([2, 0], [2, 2, 1], 20) == pytest.approx(1 / 3)


def test_recall_counts_partial_by_default():
    assert recall_at_k([1, 1], [1, 1], 20) == pytest.approx(1.0)


def test_recall_threshold_can_require_full_relevance():
    assert recall_at_k([1, 1], [1, 1], 20, threshold=RELEVANT) == 0.0


def test_recall_respects_cutoff():
    assert recall_at_k([0] * 20 + [2], [2], 20) == 0.0


def test_recall_with_no_relevant_is_zero():
    assert recall_at_k([0], [0], 20) == 0.0


# --- MRR -------------------------------------------------------------------


def test_mrr_finds_first_fully_relevant():
    assert reciprocal_rank([0, 1, 2]) == pytest.approx(1 / 3)


def test_mrr_zero_when_nothing_relevant():
    assert reciprocal_rank([0, 1, 1]) == 0.0


# --- judgments -------------------------------------------------------------


@pytest.fixture
def labels_file(tmp_path):
    path = tmp_path / "retrieval_labels.json"
    path.write_text(json.dumps({
        "queries": [
            {"id": "q1", "query": "what is RRF",
             "judgments": {"Pa": 2, "Pb": 1, "Pc": 0}},
            {"id": "q2", "query": "what is BM25",
             "judgments": {"Pd": 2, "Pe": 0}},
            {"id": "q3", "query": "unanswerable",
             "judgments": {"Pf": 0, "Pg": 0}},
            {"id": "q4", "query": "partly labelled",
             "judgments": {"Ph": 2, "Pi": None}},
        ]
    }))
    return path


def test_load_judgments(labels_file):
    j = Judgments.load(labels_file)
    assert j.grade("q1", "Pa") == 2
    assert j.queries["q1"] == "what is RRF"


def test_unjudged_passage_is_irrelevant(labels_file):
    """Valid only because the pool is the union of all systems compared."""
    assert Judgments.load(labels_file).grade("q1", "Punseen") == 0


def test_null_grades_are_dropped_as_unlabelled(labels_file):
    assert "Pi" not in Judgments.load(labels_file).by_query["q4"]


def test_has_relevant(labels_file):
    j = Judgments.load(labels_file)
    assert j.has_relevant("q1") and not j.has_relevant("q3")


def test_coverage_summary(labels_file):
    coverage = Judgments.load(labels_file).coverage()
    assert coverage["queries"] == 4
    assert coverage["queries_with_relevant"] == 3


# --- the R1-R4 harness -----------------------------------------------------


class FakeResult:
    def __init__(self, pid):
        self.passage = {"id": pid}


class FakeRetriever:
    """Returns a scripted ranking per (config, query)."""

    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def retrieve(self, query, *, top_k, dense_only=False, sparse_only=False):
        config = "R1" if dense_only else "R2" if sparse_only else "R3"
        self.calls.append((config, query))
        return [FakeResult(p) for p in self.rankings[config][query][:top_k]]


class FakeReranker:
    def __init__(self, rankings):
        self.rankings = rankings

    def rerank(self, query, results, top_k=None):
        return [FakeResult(p) for p in self.rankings["R4"][query][:top_k or 20]]


@pytest.fixture
def evaluator(labels_file):
    rankings = {
        "R1": {"what is RRF": ["Pc", "Pa"], "what is BM25": ["Pe", "Pd"],
               "partly labelled": ["Ph"]},
        "R2": {"what is RRF": ["Pb", "Pc"], "what is BM25": ["Pd", "Pe"],
               "partly labelled": ["Ph"]},
        "R3": {"what is RRF": ["Pa", "Pb"], "what is BM25": ["Pd", "Pe"],
               "partly labelled": ["Ph"]},
        "R4": {"what is RRF": ["Pa", "Pb"], "what is BM25": ["Pd", "Pe"],
               "partly labelled": ["Ph"]},
    }
    return RetrievalEvaluator(
        FakeRetriever(rankings), Judgments.load(labels_file),
        reranker=FakeReranker(rankings),
    )


def test_all_four_configs_score(evaluator):
    scores = evaluator.evaluate()
    assert [s.config for s in scores] == ["R1", "R2", "R3", "R4"]
    assert all(0.0 <= s.ndcg_at_10 <= 1.0 for s in scores)


def test_queries_without_relevant_judgments_are_excluded(evaluator):
    """They cannot discriminate between systems; including them drags every
    score toward zero equally and dilutes the comparison."""
    assert evaluator.evaluate()[0].queries == 3     # q3 excluded


def test_better_ranking_scores_higher(evaluator):
    by_name = {s.config: s for s in evaluator.evaluate()}
    assert by_name["R3"].ndcg_at_10 > by_name["R1"].ndcg_at_10


def test_r4_requires_a_reranker(labels_file):
    rankings = {c: {"what is RRF": ["Pa"]} for c in ("R1", "R2", "R3")}
    ev = RetrievalEvaluator(FakeRetriever(rankings), Judgments.load(labels_file))
    with pytest.raises(ValueError, match="R4 requires a reranker"):
        ev.rank("R4", "what is RRF")


def test_unknown_config_raises(evaluator):
    with pytest.raises(ValueError, match="unknown configuration"):
        evaluator.rank("R9", "q")


def test_no_relevant_judgments_at_all_raises(tmp_path):
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"queries": [
        {"id": "q1", "query": "x", "judgments": {"Pa": 0}}
    ]}))
    ev = RetrievalEvaluator(FakeRetriever({"R1": {}}), Judgments.load(path))
    with pytest.raises(ValueError, match="no queries have any relevant"):
        ev.evaluate()


# --- the exit criterion ----------------------------------------------------


def test_expected_ordering_passes_when_bets_hold():
    scores = [
        ConfigScore("R1", "d", 0.50, 0.6, 0.5, 3),
        ConfigScore("R2", "s", 0.45, 0.5, 0.4, 3),
        ConfigScore("R3", "f", 0.60, 0.7, 0.6, 3),
        ConfigScore("R4", "r", 0.68, 0.7, 0.7, 3),
    ]
    assert check_expected_ordering(scores) == []


def test_fusion_failing_to_beat_single_arm_is_reported():
    """§8's exit criterion allows this outcome — it is a finding to explain,
    not a failure to hide."""
    scores = [
        ConfigScore("R1", "d", 0.70, 0.6, 0.5, 3),
        ConfigScore("R2", "s", 0.45, 0.5, 0.4, 3),
        ConfigScore("R3", "f", 0.60, 0.7, 0.6, 3),
        ConfigScore("R4", "r", 0.68, 0.7, 0.7, 3),
    ]
    problems = check_expected_ordering(scores)
    assert any("RRF is not helping" in p for p in problems)


def test_rerank_failing_to_beat_fusion_is_reported():
    scores = [
        ConfigScore("R1", "d", 0.50, 0.6, 0.5, 3),
        ConfigScore("R2", "s", 0.45, 0.5, 0.4, 3),
        ConfigScore("R3", "f", 0.60, 0.7, 0.6, 3),
        ConfigScore("R4", "r", 0.55, 0.7, 0.7, 3),
    ]
    assert any("not earning its latency" in p
               for p in check_expected_ordering(scores))


def test_table_renders_every_config(evaluator):
    table = render_table(evaluator.evaluate())
    for name in CONFIGS:
        assert name in table
    assert "nDCG@10" in table and "recall@20" in table


def test_evaluate_ranking_shape(labels_file):
    metrics = evaluate_ranking("q1", ["Pa", "Pb"], Judgments.load(labels_file))
    assert set(metrics) == {"ndcg", "recall", "mrr"}