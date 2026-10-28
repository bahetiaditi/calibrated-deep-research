"""Tests for the critic.

The grounding metrics are computed from this node, so its errors become the
reported numbers. The properties that matter: discourse sentences stay out
of the denominator, an unparseable verdict never becomes a pass, and the
whole draft costs one call.
"""
import pytest

from src.config import Config
from src.nodes.critic import (
    CONTRADICTED,
    NOT_A_CLAIM,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    Critic,
    CriticReport,
    contradicted_passages,
    critic_node,
    segment,
)
from src.state import Passage, new_state

CFG = Config({})

PID = {i: f"P{i:012x}" for i in range(1, 9)}


def passage(pid, text="the evidence text", section="Results",
            source="paper-A", sub_question_id="sq1"):
    return Passage(
        id=pid, source_type="arxiv", source_id=source,
        source_domain="arxiv.org", title="t", text=text, section=section,
        published="2024-01-01", sub_question_id=sub_question_id,
        retrieval_score=0.5, rerank_score=7.0,
    )


class FakeProvider:
    def __init__(self, verdicts=None, error=None):
        self.verdicts = verdicts
        self.error = error
        self.calls = []

    def complete_json(self, system, user, **kw):
        self.calls.append({"system": system, "user": user})
        if self.error:
            raise self.error
        return {"verdicts": self.verdicts or []}, None


def critic(verdicts=None, **kw):
    return Critic(CFG, provider=FakeProvider(verdicts), **kw)


def state_with(draft, passages=None, contradictions=None):
    state = new_state("q")
    state["draft"] = draft
    state["evidence"] = passages or [passage(PID[1]), passage(PID[2])]
    state["contradictions"] = contradictions or []
    return state


# --- segmentation ----------------------------------------------------------


def test_sentences_are_split_and_citations_extracted():
    sentences = segment(f"First claim. [{PID[1]}] Second claim. [{PID[2]}]")
    assert len(sentences) == 2
    assert sentences[0].citations == [PID[1]]
    assert sentences[1].citations == [PID[2]]


def test_multiple_citations_on_one_sentence():
    s = segment(f"A claim. [{PID[1]}][{PID[2]}]")[0]
    assert s.citations == [PID[1], PID[2]]


def test_repeated_citation_is_deduplicated():
    s = segment(f"A claim [{PID[1]}] and again [{PID[1]}].")[0]
    assert s.citations == [PID[1]]


def test_uncited_sentence_has_no_citations():
    assert segment("Just a sentence with no citation.")[0].citations == []


def test_empty_draft_segments_to_nothing():
    assert segment("") == []


# --- three categories (the C18 constraint) ---------------------------------


def test_discourse_sentences_are_excluded_from_the_denominator():
    """The C18 run produced four uncited sentences that were correct
    SUMMARIES of claims cited above. Counting those as UNSUPPORTED would put
    them in the hallucination rate and make the metric measure the
    synthesizer's citation habits rather than the system's grounding."""
    draft = (
        f"FlashAttention-2 reaches 73% of peak. [{PID[1]}] "
        f"The second approach differs. "
        f"PagedAttention pages the KV cache. [{PID[2]}]"
    )
    report = critic([
        {"index": 0, "verdict": SUPPORTED, "reasoning": "stated"},
        {"index": 1, "verdict": NOT_A_CLAIM, "reasoning": "organises the text"},
        {"index": 2, "verdict": SUPPORTED, "reasoning": "stated"},
    ]).review(state_with(draft))

    assert report.claims == 2
    assert report.discourse_sentences == 1
    assert report.unsupported_rate == 0.0


def test_the_model_can_reclassify_a_cited_sentence_as_discourse():
    draft = f"In what follows we compare them. [{PID[1]}]"
    report = critic([
        {"index": 0, "verdict": NOT_A_CLAIM, "reasoning": "organises the text"},
    ]).review(state_with(draft))
    assert report.claims == 0
    assert report.discourse_sentences == 1


def test_short_fragments_are_treated_as_discourse():
    report = critic([]).review(state_with("Yes. No."))
    assert report.claims == 0
    assert report.discourse_sentences == 2


def test_a_long_uncited_sentence_is_a_claim_not_discourse():
    draft = "FlashAttention-2 reaches seventy three percent of theoretical peak."
    report = critic([
        {"index": 0, "verdict": UNSUPPORTED, "reasoning": "no citation"},
    ]).review(state_with(draft))
    assert report.claims == 1
    assert report.findings[0]["verdict"] == UNSUPPORTED


# --- verdicts --------------------------------------------------------------


def test_the_six_case_acceptance_check():
    """C23's criterion: three good and three bad citations, all labelled."""
    draft = " ".join(
        f"Claim number {i} asserting something checkable. [{PID[i]}]"
        for i in range(1, 7)
    )
    passages = [passage(PID[i]) for i in range(1, 7)]
    report = critic([
        {"index": 0, "verdict": SUPPORTED, "reasoning": "stated directly"},
        {"index": 1, "verdict": SUPPORTED, "reasoning": "stated directly"},
        {"index": 2, "verdict": SUPPORTED, "reasoning": "implied clearly"},
        {"index": 3, "verdict": UNSUPPORTED, "reasoning": "wrong entity"},
        {"index": 4, "verdict": UNSUPPORTED, "reasoning": "passage is silent"},
        {"index": 5, "verdict": CONTRADICTED, "reasoning": "says the opposite"},
    ]).review(state_with(draft, passages))

    assert report.claims == 6
    assert report.count(SUPPORTED) == 3
    assert report.count(UNSUPPORTED) == 2
    assert report.count(CONTRADICTED) == 1
    assert report.grounding_precision == 0.5
    assert report.unsupported_rate == 0.5


def test_an_uncited_claim_is_unsupported_whatever_the_model_says():
    """The model classifies claim-vs-discourse; the ABSENCE of a citation
    decides the verdict, since the model cannot have checked support that
    was never cited."""
    draft = "A long uncited claim about memory access patterns in kernels."
    report = critic([
        {"index": 0, "verdict": SUPPORTED, "reasoning": "looks right to me"},
    ]).review(state_with(draft))
    assert report.findings[0]["verdict"] == UNSUPPORTED
    assert "no citation" in report.findings[0]["reasoning"]


def test_a_missing_verdict_defaults_to_unsupported():
    draft = f"A claim about something checkable here. [{PID[1]}]"
    report = critic([]).review(state_with(draft))
    assert report.findings[0]["verdict"] == UNSUPPORTED


def test_an_unparseable_verdict_never_becomes_a_pass():
    """Claiming support the critic never gave is the one error the grounding
    metric cannot survive."""
    draft = f"A claim about something checkable here. [{PID[1]}]"
    report = critic([
        {"index": 0, "verdict": "PROBABLY_FINE", "reasoning": ""},
    ]).review(state_with(draft))
    assert report.findings[0]["verdict"] == UNSUPPORTED


def test_partial_is_neither_supported_nor_counted_as_hallucination():
    draft = f"A claim about something checkable here. [{PID[1]}]"
    report = critic([
        {"index": 0, "verdict": PARTIAL, "reasoning": "drops a qualifier"},
    ]).review(state_with(draft))
    assert report.grounding_precision == 0.0
    assert report.unsupported_rate == 0.0


# --- what the model is shown -----------------------------------------------


def test_only_cited_passages_are_sent():
    provider = FakeProvider([{"index": 0, "verdict": SUPPORTED, "reasoning": ""}])
    state = state_with(
        f"A claim about something checkable. [{PID[1]}]",
        [passage(PID[1], "cited text"), passage(PID[2], "uncited text")],
    )
    Critic(CFG, provider=provider).review(state)
    system = provider.calls[0]["system"]
    assert "cited text" in system
    assert "uncited text" not in system


def test_section_is_shown_for_attribution():
    """A Related Work passage describes someone else's work — one of the
    three traps the prompt names."""
    provider = FakeProvider([{"index": 0, "verdict": SUPPORTED, "reasoning": ""}])
    state = state_with(
        f"A claim about something checkable. [{PID[1]}]",
        [passage(PID[1], section="Related Work")],
    )
    Critic(CFG, provider=provider).review(state)
    assert "[Related Work]" in provider.calls[0]["system"]


def test_contradicted_passages_are_flagged():
    provider = FakeProvider([{"index": 0, "verdict": PARTIAL, "reasoning": ""}])
    state = state_with(
        f"A claim about something checkable. [{PID[1]}]",
        [passage(PID[1])],
        contradictions=[{"passage_a": PID[1], "passage_b": PID[2],
                         "score": 0.8, "cross_source": True}],
    )
    Critic(CFG, provider=provider).review(state)
    assert "CONTRADICTED BY OTHER SOURCES" in provider.calls[0]["system"]


def test_same_source_contradictions_are_not_flagged():
    """Only cross-source disagreement counts (C12)."""
    assert contradicted_passages([
        {"passage_a": PID[1], "passage_b": PID[2], "cross_source": False},
    ]) == set()


def test_an_unresolvable_citation_is_marked_in_the_prompt():
    provider = FakeProvider([{"index": 0, "verdict": UNSUPPORTED, "reasoning": ""}])
    state = state_with(f"A claim about something checkable. [{PID[7]}]", [])
    Critic(CFG, provider=provider).review(state)
    assert "PASSAGE NOT FOUND" in provider.calls[0]["system"]


# --- cost ------------------------------------------------------------------


def test_the_whole_draft_costs_one_call():
    """Per-claim calls would be 8-10 per question against a budget of about
    9, so the critic alone would exceed the entire run (§7.1)."""
    provider = FakeProvider([
        {"index": i, "verdict": SUPPORTED, "reasoning": ""} for i in range(6)
    ])
    draft = " ".join(
        f"Claim number {i} asserting something checkable. [{PID[1]}]"
        for i in range(6)
    )
    Critic(CFG, provider=provider).review(state_with(draft))
    assert len(provider.calls) == 1


def test_budget_charged_once():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    critic([{"index": 0, "verdict": SUPPORTED, "reasoning": ""}],
           budget=budget).review(
        state_with(f"A claim about something checkable. [{PID[1]}]")
    )
    assert budget.llm_calls_used == 1


# --- degradation -----------------------------------------------------------


def test_no_draft_returns_an_error_not_a_crash():
    assert critic([]).review(new_state("q")).error


def test_provider_failure_marks_claims_unverified_rather_than_supported():
    """A critic that crashes loses the whole verification pass. Degrading to
    unsupported keeps the run alive and is visible in the findings."""
    c = Critic(CFG, provider=FakeProvider(error=RuntimeError("down")))
    report = c.review(state_with(
        f"A claim about something checkable. [{PID[1]}]"
    ))
    assert report.error
    assert report.findings[0]["verdict"] == UNSUPPORTED
    assert "critic unavailable" in report.findings[0]["reasoning"]


def test_metrics_are_none_with_no_claims():
    """A draft that asserts nothing has no grounding to measure; 0.0 and 1.0
    would both be inventions."""
    report = CriticReport()
    assert report.grounding_precision is None
    assert report.unsupported_rate is None


# --- state integration -----------------------------------------------------


def _with_plan(state, owned):
    from src.state import new_sub_question

    plan = []
    for sqid, pids in owned.items():
        sq = new_sub_question(f"text of {sqid}", sub_question_id=sqid)
        sq["evidence_ids"] = list(pids)
        plan.append(sq)
    state["plan"] = plan
    return state


def test_findings_carry_the_owning_sub_question_from_the_plan():
    """Ownership comes from SubQuestion.evidence_ids, the mapping C17 made
    authoritative (C27.2)."""
    state = _with_plan(state_with(f"A claim about something checkable. [{PID[1]}]",
                                  [passage(PID[1])]), {"sq3": [PID[1]]})
    report = critic([{"index": 0, "verdict": SUPPORTED, "reasoning": ""}]).review(state)
    assert report.findings[0]["sub_question_id"] == "sq3"


def test_a_stale_passage_payload_does_not_decide_ownership():
    """What happened live: a passage served from the persistent store
    carried a sub_question_id from another run. The claim was attributed to
    it, and the router's RE_RETRIEVE on that claim reopened nothing."""
    stale = passage(PID[1], sub_question_id="rq07")
    state = _with_plan(state_with(f"A claim about something checkable. [{PID[1]}]",
                                  [stale]), {"sq1": [PID[1]]})
    report = critic([{"index": 0, "verdict": UNSUPPORTED, "reasoning": ""}]).review(state)
    assert report.findings[0]["sub_question_id"] == "sq1"


def test_a_claim_citing_only_unowned_evidence_is_unowned():
    state = _with_plan(state_with(f"A claim about something checkable. [{PID[1]}]",
                                  [passage(PID[1], sub_question_id="sq1")]), {"sq1": []})
    report = critic([{"index": 0, "verdict": UNSUPPORTED, "reasoning": ""}]).review(state)
    assert report.findings[0]["sub_question_id"] == ""


def test_live_sub_questions_own_shared_evidence_before_abandoned_ones():
    from src.nodes.critic import passage_owners

    state = _with_plan(new_state("q"), {"sq1": ["P1"], "sq2": ["P1"]})
    state["plan"][0]["status"] = "abandoned"
    assert passage_owners(state) == {"P1": "sq2"}


def test_action_taken_is_left_for_c24():
    report = critic([{"index": 0, "verdict": UNSUPPORTED, "reasoning": "x"}]).review(
        state_with(f"A claim about something checkable. [{PID[1]}]")
    )
    assert report.findings[0]["action_taken"] == ""


def test_node_writes_findings_into_state():
    update = critic_node(
        critic([{"index": 0, "verdict": SUPPORTED, "reasoning": "ok"}])
    )(state_with(f"A claim about something checkable. [{PID[1]}]"))
    assert update["critic_findings"][0]["verdict"] == SUPPORTED


def test_review_is_traced():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    critic([{"index": 0, "verdict": SUPPORTED, "reasoning": ""}],
           tracer=FakeTracer()).review(
        state_with(f"A claim about something checkable. [{PID[1]}]")
    )
    name, payload = events[0]
    assert name == "critic"
    assert payload["claims"] == 1 and payload["supported"] == 1