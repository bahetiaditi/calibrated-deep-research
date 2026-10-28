"""C27.9: a rerank score is relevance to a query, so it belongs to the
sub-question that asked, not to the passage.

Live symptom: five different sub-questions all reported f1 = 0.35 at
decision time — the score of one passage they all retrieved, measured
against the FIRST sub-question's query, because merge_evidence keeps the
first copy of a passage and its score with it.
"""
import pytest

from src.config import Config
from src.state import merge_evidence, new_state, new_sub_question
from src.sufficiency.features import features_for_sub_question
from tests.decision_fixtures import passage
from tests.test_adaptive_retriever import (
    FakeProvider,
    FakeReranker,
    FakeRetriever,
    build,
)
from tests.test_adaptive_retriever import passage as rpassage


def shared_state(score_a, score_b, with_maps=True):
    state = new_state("q")
    a = new_sub_question("about A", sub_question_id="sq1")
    b = new_sub_question("about B", sub_question_id="sq2")
    pa = passage("Pshared00000", "sq1", score_a)
    pb = passage("Pshared00000", "sq2", score_b)
    state["evidence"] = merge_evidence(merge_evidence([], [pa]), [pb])
    a["evidence_ids"] = b["evidence_ids"] = ["Pshared00000"]
    if with_maps:
        a["evidence_scores"] = {"Pshared00000": score_a}
        b["evidence_scores"] = {"Pshared00000": score_b}
    state["plan"] = [a, b]
    return state


def test_a_shared_passage_is_scored_by_each_sub_questions_own_query():
    """The reproduction. Before C27.9, sq2 read 0.35 — sq1's score."""
    state = shared_state(0.35, 6.0)
    assert state["evidence"][0]["rerank_score"] == 0.35     # stored: first copy
    assert features_for_sub_question(state, "sq1")["f1_max_rerank"] == 0.35
    assert features_for_sub_question(state, "sq2")["f1_max_rerank"] == 6.0


def test_plans_from_before_c27_9_fall_back_to_the_stored_score():
    state = shared_state(0.35, 6.0, with_maps=False)
    assert features_for_sub_question(state, "sq2")["f1_max_rerank"] == 0.35


def test_the_real_retriever_records_each_sub_questions_scores():
    """End to end through retrieval: one passage, two sub-questions, two
    different relevances — and two different decision-time features."""
    impl = FakeRetriever(by_query={
        "about A": [rpassage("P1", 9.0, "X"), rpassage("P2", 8.0, "Y")],
        "about B": [rpassage("P1", 1.0, "X"), rpassage("P3", 0.5, "Z")],
    })
    retriever = build(retriever=impl, provider=FakeProvider([
        {"action": "give_up", "reasoning": "nothing better"}]))
    state = new_state("q")
    state["plan"] = [new_sub_question("about A", sub_question_id="sq1"),
                     new_sub_question("about B", sub_question_id="sq2")]
    for sq in state["plan"]:
        sq["suggested_route"] = "arxiv_meta"

    for sqid in ("sq1", "sq2"):
        update = retriever.retrieve_one(state, sqid)
        state["plan"] = update["plan"]
        state["evidence"] = merge_evidence(state["evidence"], update.get("evidence"))

    sq1, sq2 = state["plan"]
    assert sq1["evidence_scores"]["P1"] == 9.0
    assert sq2["evidence_scores"]["P1"] == 1.0
    assert features_for_sub_question(state, "sq1")["f1_max_rerank"] == 9.0
    assert features_for_sub_question(state, "sq2")["f1_max_rerank"] == 1.0


def test_a_passage_retrieved_twice_by_one_sub_question_keeps_its_best_score():
    impl = FakeRetriever(by_query={
        "hard": [rpassage("P1", 1.0, "X")],
        "rewritten": [rpassage("P1", 7.0, "X"), rpassage("P2", 6.0, "Y")],
    })
    retriever = build(retriever=impl, provider=FakeProvider([{
        "action": "wider", "route": "web", "query": "rewritten", "reasoning": "r"}]))
    state = new_state("q")
    state["plan"] = [new_sub_question("hard", sub_question_id="sq1")]
    state["plan"][0]["suggested_route"] = "arxiv_meta"
    final = retriever.retrieve_all(state)
    assert final["plan"][0]["evidence_scores"]["P1"] == 7.0


def test_new_sub_questions_start_with_an_empty_map():
    assert new_sub_question("x")["evidence_scores"] == {}
