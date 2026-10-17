"""Tests for the premise checker.

The property under test is the three-way distinction. Collapsing
UNVERIFIABLE into CONTRADICTED would make the system refuse every question
whose presuppositions the corpus happens not to restate — which is most of
them — and the project would report a beautiful false-answer rate achieved
by answering almost nothing.
"""
import json

import pytest

from src.config import Config
from src.nodes.premise_checker import (
    CONTRADICTED,
    SUPPORTED,
    UNVERIFIABLE,
    PremiseCheckResult,
    PremiseChecker,
    Presupposition,
    premise_checker_node,
)
from src.state import Passage

CFG = Config({
    "sufficiency": {"hard_rules": {"premise_grounding_min": 0.35}},
})


def passage(pid, text, source="2312.00752v2", section="Method"):
    return Passage(
        id=pid, source_type="arxiv", source_id=source,
        source_domain="arxiv.org", title="t", text=text, section=section,
        published="2023-12-01", sub_question_id="", retrieval_score=0.5,
        rerank_score=7.0,
    )


class FakeResult:
    def __init__(self, p):
        self.passage = p


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else [
            passage("P1", "Mamba-2 uses a selective state space mechanism.")
        ]
        self.error = error
        self.queries = []

    def retrieve(self, query, *, top_k, **kw):
        self.queries.append(query)
        if self.error:
            raise self.error
        return [FakeResult(p) for p in self.results]


class FakeReranker:
    def rerank(self, query, results, *, top_k=None):
        return list(results)[: top_k or 10]


class FakeProvider:
    """Scripted JSON responses, in call order."""

    def __init__(self, responses, error=None):
        self.responses = list(responses)
        self.error = error
        self.calls = []

    def complete_json(self, system, user, *, role=None, required_keys=None, **kw):
        self.calls.append({"system": system, "user": user, "role": role})
        if self.error:
            raise self.error
        return self.responses.pop(0), None


def checker(provider, retriever=None, **kw):
    return PremiseChecker(
        CFG, provider=provider,
        retriever=retriever or FakeRetriever(),
        reranker=FakeReranker(), **kw,
    )


EXTRACT_ONE = {"presuppositions": [
    {"text": "Mamba-2 uses rotary position embeddings", "centrality": "core"}
]}


# --- the three-way distinction ---------------------------------------------


def test_contradicted_core_presupposition_fails_the_check():
    """The Tier D1 case. One retrieval round instead of three paragraphs
    about a mechanism that was never there."""
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{
            "claim_index": 0, "verdict": "CONTRADICTED",
            "reasoning": "the paper describes a different mechanism",
            "correction": "Mamba-2 uses a selective state space mechanism, "
                          "not rotary position embeddings.",
            "cited": ["P01"],
        }]},
    ])
    result = checker(provider).check("Why does Mamba-2 use RoPE in its SSM blocks?")
    assert result.failed is True
    assert result.grounding_score == 0.0
    assert "selective state space" in result.correction
    assert "false premise" in result.rationale


def test_unverifiable_does_not_fail_the_check():
    """The decision this node turns on. Treating silence as refutation would
    abstain on every question the corpus does not happen to restate."""
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{
            "claim_index": 0, "verdict": "UNVERIFIABLE",
            "reasoning": "the evidence does not discuss this",
            "correction": "", "cited": [],
        }]},
    ])
    result = checker(provider).check("Why does Mamba-2 use RoPE?")
    assert result.failed is False
    assert result.grounding_score == 0.5      # midpoint, not zero
    # The rationale must say what actually happened: the corpus was silent,
    # which is not the same as the premise being confirmed.
    assert "silent" in result.rationale
    assert "unverified" in result.rationale


def test_supported_presupposition_passes():
    provider = FakeProvider([
        {"presuppositions": [
            {"text": "Mamba-2 has SSM blocks", "centrality": "core"}
        ]},
        {"verdicts": [{
            "claim_index": 0, "verdict": "SUPPORTED",
            "reasoning": "stated directly", "correction": "", "cited": ["P01"],
        }]},
    ])
    result = checker(provider).check("How do Mamba-2's SSM blocks work?")
    assert result.failed is False
    assert result.grounding_score == 1.0
    assert "confirmed" in result.rationale


def test_unknown_verdict_degrades_to_unverifiable():
    """A malformed verdict must never become a contradiction — that would
    abstain on a parse error."""
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{"claim_index": 0, "verdict": "PROBABLY_NOT",
                       "reasoning": "", "correction": "", "cited": []}]},
    ])
    result = checker(provider).check("q")
    assert result.presuppositions[0].verdict == UNVERIFIABLE
    assert result.failed is False


# --- core vs incidental ----------------------------------------------------


def test_incidental_contradiction_does_not_fail_the_check():
    """The question survives a wrong incidental detail, slightly reworded."""
    provider = FakeProvider([
        {"presuppositions": [
            {"text": "Mamba-2 exists", "centrality": "core"},
            {"text": "Mamba-2 was published in 2024", "centrality": "incidental"},
        ]},
        {"verdicts": [
            {"claim_index": 0, "verdict": "SUPPORTED", "reasoning": "",
             "correction": "", "cited": ["P01"]},
            {"claim_index": 1, "verdict": "CONTRADICTED",
             "reasoning": "published 2023", "correction": "2023", "cited": ["P01"]},
        ]},
    ])
    result = checker(provider).check("q")
    assert result.failed is False
    assert result.grounding_score == 1.0   # f7 uses core presuppositions only


def test_grounding_averages_over_core_only():
    provider = FakeProvider([
        {"presuppositions": [
            {"text": "a", "centrality": "core"},
            {"text": "b", "centrality": "core"},
            {"text": "c", "centrality": "incidental"},
        ]},
        {"verdicts": [
            {"claim_index": 0, "verdict": "SUPPORTED", "reasoning": "",
             "correction": "", "cited": []},
            {"claim_index": 1, "verdict": "UNVERIFIABLE", "reasoning": "",
             "correction": "", "cited": []},
            {"claim_index": 2, "verdict": "CONTRADICTED", "reasoning": "",
             "correction": "", "cited": []},
        ]},
    ])
    result = checker(provider).check("q")
    assert result.grounding_score == pytest.approx(0.75)   # (1.0 + 0.5) / 2


# --- extraction ------------------------------------------------------------


def test_open_question_with_no_presuppositions_passes():
    """Scoring an open question as ungrounded would penalise exactly the
    Tier C questions the system should handle well."""
    provider = FakeProvider([{"presuppositions": []}])
    result = checker(provider).check("What approaches exist for RAG evaluation?")
    assert result.ran and not result.failed
    assert result.grounding_score == 1.0
    assert len(provider.calls) == 1     # no verification call needed


def test_presuppositions_are_capped():
    provider = FakeProvider([
        {"presuppositions": [
            {"text": f"claim {i}", "centrality": "core"} for i in range(9)
        ]},
        {"verdicts": []},
    ])
    assert len(checker(provider).check("q").presuppositions) == 4


def test_blank_presuppositions_are_dropped():
    provider = FakeProvider([
        {"presuppositions": [{"text": "  ", "centrality": "core"},
                             {"text": "real", "centrality": "core"}]},
        {"verdicts": []},
    ])
    assert len(checker(provider).check("q").presuppositions) == 1


def test_empty_question_does_not_run():
    result = checker(FakeProvider([])).check("   ")
    assert result.ran is False and result.failed is False


# --- batching and cost -----------------------------------------------------


def test_verification_is_one_call_for_all_presuppositions():
    """Three to five calls would make the premise check cost more than the
    research it guards, against a budget of roughly nine (§7.1)."""
    provider = FakeProvider([
        {"presuppositions": [
            {"text": "a", "centrality": "core"},
            {"text": "b", "centrality": "core"},
            {"text": "c", "centrality": "core"},
        ]},
        {"verdicts": [
            {"claim_index": i, "verdict": "SUPPORTED", "reasoning": "",
             "correction": "", "cited": []} for i in range(3)
        ]},
    ])
    checker(provider).check("q")
    assert len(provider.calls) == 2      # extract + verify, regardless of count


def test_budget_is_charged():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{"claim_index": 0, "verdict": "SUPPORTED",
                       "reasoning": "", "correction": "", "cited": []}]},
    ])
    checker(provider, budget=budget).check("q")
    assert budget.llm_calls_used == 2
    assert budget.retrieval_calls_used == 1


# --- citation labels -------------------------------------------------------


def test_labels_map_back_to_real_passage_ids():
    """A model asked to echo a content-addressed id transcribes it wrong,
    and a citation that does not resolve is indistinguishable from a
    fabricated one (the C18 problem, solved the same way)."""
    retriever = FakeRetriever(results=[
        passage("Pa3f9c2b1d004", "Mamba-2 uses selective state spaces."),
    ])
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{"claim_index": 0, "verdict": "CONTRADICTED",
                       "reasoning": "", "correction": "x", "cited": ["P01"]}]},
    ])
    result = checker(provider, retriever=retriever).check("q")
    assert result.presuppositions[0].cited_passage_ids == ["Pa3f9c2b1d004"]


def test_unknown_label_is_ignored():
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{"claim_index": 0, "verdict": "SUPPORTED",
                       "reasoning": "", "correction": "", "cited": ["P99"]}]},
    ])
    assert checker(provider).check("q").presuppositions[0].cited_passage_ids == []


def test_each_presupposition_is_retrieved_separately():
    retriever = FakeRetriever()
    provider = FakeProvider([
        {"presuppositions": [{"text": "claim one", "centrality": "core"},
                             {"text": "claim two", "centrality": "core"}]},
        {"verdicts": []},
    ])
    checker(provider, retriever=retriever).check("q")
    assert retriever.queries == ["claim one", "claim two"]


# --- degradation -----------------------------------------------------------


def test_extraction_failure_does_not_fail_the_run():
    """Degrading to 'no premise information' costs the hard rule, not the
    question — the calibrated threshold still governs the decision."""
    result = checker(FakeProvider([], error=RuntimeError("provider down"))).check("q")
    assert result.ran is False
    assert result.failed is False
    assert result.error


def test_verification_failure_does_not_fail_the_run():
    class HalfBroken(FakeProvider):
        def complete_json(self, *a, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                return EXTRACT_ONE, None
            raise RuntimeError("timeout")

    result = checker(HalfBroken([])).check("q")
    assert result.ran is False and result.failed is False


def test_retrieval_failure_yields_unverifiable_not_failure():
    provider = FakeProvider([EXTRACT_ONE])
    result = checker(provider, retriever=FakeRetriever(error=RuntimeError("x"))).check("q")
    assert result.failed is False
    assert result.grounding_score == 0.5


# --- state integration -----------------------------------------------------


def test_node_writes_premise_check_into_state():
    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{"claim_index": 0, "verdict": "CONTRADICTED",
                       "reasoning": "r", "correction": "c", "cited": []}]},
    ])
    update = premise_checker_node(checker(provider))({"question": "q"})
    assert update["premise_check"]["failed"] is True
    assert update["premise_check"]["presuppositions"][0]["verdict"] == CONTRADICTED


def test_result_is_json_serialisable():
    """It is checkpointed at every node, so a field LangGraph cannot
    serialise surfaces as a mid-run failure."""
    result = PremiseCheckResult(
        ran=True,
        presuppositions=[Presupposition(text="x", verdict=SUPPORTED)],
        grounding_score=1.0,
    )
    assert json.loads(json.dumps(result.to_dict()))["ran"] is True


def test_check_is_traced():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    provider = FakeProvider([
        EXTRACT_ONE,
        {"verdicts": [{"claim_index": 0, "verdict": "CONTRADICTED",
                       "reasoning": "", "correction": "", "cited": []}]},
    ])
    checker(provider, tracer=FakeTracer()).check("q")
    name, payload = events[0]
    assert name == "premise_check"
    assert payload["failed"] is True and payload["verdicts"] == [CONTRADICTED]