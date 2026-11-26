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
    assert (tiers.count("A"), tiers.count("B"), tiers.count("C"), tiers.count("D")) == (10, 10, 8, 20)
    subs = [q["subtype"] for q in data["questions"] if q["tier"] == "D"]
    assert all(subs.count(s) == 5 for s in ("D1", "D2", "D3", "D4"))


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



# --- Tier D (C30) ----------------------------------------------------------


def _d(qs, sub):
    return next(q for q in qs if q.get("subtype") == sub)


def test_a_false_premise_without_a_refutation_is_caught(loaded):
    def mutate(qs): _d(qs, "D1")["refutation_passage_ids"] = []
    assert any("refutation" in e for e in _break(loaded, mutate))


def test_a_no_consensus_item_with_one_source_is_caught(loaded):
    def mutate(qs):
        q = _d(qs, "D3")
        q["side_passage_ids"] = q["side_passage_ids"][:1]
    assert any("at least 2 sources" in e for e in _break(loaded, mutate))


def test_an_unresolvable_item_the_corpus_can_answer_is_caught(loaded):
    """The adversarial check C30 asks for: 'Mamba' is in the corpus, so an
    absence claim keyed on it must fail."""
    def mutate(qs): _d(qs, "D4")["absence_patterns"] = [["Mamba"]]
    assert any("corpus may answer it" in e for e in _break(loaded, mutate))


def test_every_false_premise_is_refuted_by_a_passage_that_exists(loaded):
    data, corpus, _ = loaded
    for q in data["questions"]:
        if q.get("subtype") == "D1":
            assert q["refutation_passage_ids"]
            assert all(pid in corpus for pid in q["refutation_passage_ids"])


def test_the_prompt_example_is_not_a_benchmark_item(loaded):
    """C28.1's premise-verify prompt uses the Mamba-attention case as its
    worked example; testing on it would test the system on its own prompt."""
    data, _, _ = loaded
    assert not any("interleave multi-head attention" in q["question"] for q in data["questions"])



# --- the frozen split (C31) ------------------------------------------------


def test_the_split_is_frozen_and_balanced(loaded):
    data, _, _ = loaded
    assert data["split_frozen"]["hash"]
    splits = [q["split"] for q in data["questions"]]
    assert splits.count("calibration") == splits.count("test") == 24


def test_editing_a_test_question_after_freezing_is_caught(loaded):
    """The held-out claim rests on this: the test half cannot drift."""
    def mutate(qs):
        q = next(q for q in qs if q["split"] == "test" and q.get("gold_claims"))
        q["gold_claims"][0]["claim"] += " (edited)"
    assert any("hash mismatch" in e for e in _break(loaded, mutate))


def test_moving_a_question_between_splits_is_caught(loaded):
    def mutate(qs):
        cal = next(q for q in qs if q["split"] == "calibration" and not q.get("dev_exposed"))
        cal["split"] = "test"
    errors = _break(loaded, mutate)
    assert any("hash mismatch" in e for e in errors)


def test_editing_a_calibration_question_is_allowed(loaded):
    """Fixing an error found while calibrating is legitimate."""
    def mutate(qs):
        q = next(q for q in qs if q["split"] == "calibration" and q.get("gold_claims"))
        q["gold_claims"][0]["claim"] += " (corrected)"
    assert not any("hash mismatch" in e for e in _break(loaded, mutate))


def test_the_dev_exposed_question_is_held_out_of_test(loaded):
    data, _, _ = loaded
    assert all(q["split"] == "calibration" for q in data["questions"] if q.get("dev_exposed"))


def test_the_split_is_reproducible_from_its_seed(loaded):
    from scripts.make_split import assign

    data, _, _ = loaded
    recomputed = assign(data["questions"], seed=data["split_frozen"]["seed"])
    assert recomputed == {q["id"]: q["split"] for q in data["questions"]}
