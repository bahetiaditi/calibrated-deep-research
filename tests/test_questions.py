"""The benchmark file must stay valid against the frozen corpus (C29).

A question set edited by hand drifts: a claim's passage id mistyped, a Tier C
question quietly reduced to one source. These tests run the same validator
as scripts/check_questions.py on the committed file, and check the validator
itself catches each failure it exists for.
"""
import copy

import pytest

from scripts.check_questions import best_sentence, load, validate


@pytest.fixture(scope="module")
def loaded():
    return load()


def test_the_committed_benchmark_is_valid(loaded):
    data, corpus, rqs = loaded
    errors, warnings = validate(data, corpus, rqs)
    assert errors == []
    assert warnings == []


def test_tier_counts_match_the_design(loaded):
    data, _, _ = loaded
    tiers = [q["tier"] for q in data["questions"]]
    assert (tiers.count("A"), tiers.count("B"), tiers.count("C")) == (10, 10, 8)


def _break(loaded, mutate):
    data, corpus, _ = loaded
    broken = copy.deepcopy(data)
    mutate(broken["questions"])
    return validate(broken, corpus)[0]


def test_a_missing_passage_is_caught(loaded):
    def mutate(qs): qs[0]["gold_claims"][0]["passage_ids"] = ["Pdeadbeef0000"]
    assert any("not in corpus" in e for e in _break(loaded, mutate))


def test_a_single_source_tier_c_question_is_caught(loaded):
    def mutate(qs):
        c = next(q for q in qs if q["tier"] == "C")
        first = c["gold_claims"][0]["passage_ids"][:1]
        for claim in c["gold_claims"]:
            claim["passage_ids"] = first
    assert any("needs 2" in e for e in _break(loaded, mutate))


def test_too_few_claims_is_caught(loaded):
    def mutate(qs): qs[0]["gold_claims"] = qs[0]["gold_claims"][:2]
    assert any("need 3-5" in e for e in _break(loaded, mutate))


def test_duplicate_ids_are_caught(loaded):
    def mutate(qs): qs[1]["id"] = qs[0]["id"]
    assert any("duplicate" in e for e in _break(loaded, mutate))


def test_a_near_copy_of_a_retrieval_sub_question_warns(loaded):
    data, corpus, rqs = loaded
    broken = copy.deepcopy(data)
    broken["questions"][0]["question"] = rqs[0]
    assert validate(broken, corpus, rqs)[1]


def test_best_sentence_finds_the_supporting_sentence():
    text = "Unrelated opening. LoRA reduces trainable parameters by 10,000 times. Another line."
    assert "10,000" in best_sentence("reduces trainable parameters 10,000 times", text)
