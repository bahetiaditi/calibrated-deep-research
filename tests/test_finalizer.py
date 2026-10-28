"""Tests for the finalizer (C28).

What a reader sees is the product. These assert on the report text: that a
dropped claim is gone, a weak one is marked where it stands, a withheld
draft never appears, footnotes and sources agree, and the uncalibrated
heuristic is never presented as the calibrated layer.
"""
import re

from src.config import Config
from src.nodes.critic import segment
from src.nodes.finalizer import Finalizer
from src.state import new_state, new_sub_question

P = [f"P{c * 12}" for c in "abc"]          # content-addressed-looking ids
CFG = Config({})


def passage(pid, title):
    return {"id": pid, "source_type": "arxiv", "source_id": f"2307.{pid[1:5]}",
            "source_domain": "arxiv.org", "title": title, "text": "t",
            "section": "Results", "published": "2023-07-17",
            "sub_question_id": "sq1", "retrieval_score": 0.5, "rerank_score": 6.0}


DRAFT = (f"FlashAttention-2 partitions work across thread blocks. [{P[0]}] "
         f"This reduces shared-memory traffic by half. [{P[1]}]\n\n"
         f"PagedAttention reads KV blocks from contiguous memory. [{P[2]}] "
         f"The evidence does not establish the exact cache layout.")


def findings_for(draft, plan):
    """verdict/action per sentence index, from the real segmenter."""
    out = []
    for sentence in segment(draft):
        if sentence.index in plan:
            verdict, action = plan[sentence.index]
            out.append({"claim": sentence.text, "sub_question_id": "sq1",
                        "cited_passage_ids": sentence.citations, "verdict": verdict,
                        "reasoning": "", "action_taken": action})
    return out


def state(decision="ANSWER", rule="threshold", calibrated=False, draft=DRAFT,
          findings=None, gaps=(), **record):
    s = new_state("How do FlashAttention-2 and PagedAttention differ?")
    s["plan"] = [new_sub_question("sub", sub_question_id="sq1")]
    s["evidence"] = [passage(P[0], "FlashAttention-2"), passage(P[1], "FA2 again"),
                     passage(P[2], "vLLM")]
    s["draft"] = draft
    s["critic_findings"] = findings if findings is not None else findings_for(draft, {
        0: ("SUPPORTED", "ACCEPT"),
        1: ("PARTIAL", "HEDGE"),
        2: ("CONTRADICTED", "DROP_CLAIM"),
    })
    s["terminal_decision"] = decision
    s["decision"] = {"rule": rule, "score": 0.5, "scorer": "heuristic",
                     "calibrated": calibrated, "gaps": list(gaps), **record}
    s["decision_rationale"] = "Abstained: score 0.00 < tau_partial 0.33."
    return s


def report(s):
    return Finalizer(CFG).finalize(s)


# --- ANSWER: verification applied in the text ------------------------------


def test_a_dropped_claim_is_gone_from_the_report():
    text = report(state())
    assert "contiguous memory" not in text
    assert "1 claim(s) removed" in text


def test_a_weak_claim_is_marked_where_it_stands():
    text = report(state())
    assert re.search(r"by half\. \[\d\] \*\[partially supported\]\*", text)


def test_supported_claims_and_gap_statements_carry_no_mark():
    text = report(state())
    assert "thread blocks. [1]\n" in text or "thread blocks. [1] This" in text
    assert "exact cache layout." in text
    assert "layout. *[" not in text


def test_footnotes_are_numbered_in_order_and_sources_match():
    text = report(state())
    body, sources = text.split("## Sources")
    assert "[1]" in body and "[2]" in body and "[3]" not in body
    assert "[1] FlashAttention-2" in sources and "[2] FA2 again" in sources
    assert "vLLM" not in sources          # cited only by the dropped claim


def test_no_internal_ids_reach_the_reader():
    assert not re.search(r"P[0-9a-f]{12}", report(state()))


def test_verification_counts_what_the_text_shows():
    text = report(state())
    assert "1 of 2 claims are supported" in text
    assert "1 marked *partially supported*" in text


# --- PARTIAL ---------------------------------------------------------------


def test_partial_lists_question_level_gaps():
    text = report(state("PARTIAL", gaps=[
        "could not establish: What memory access pattern does FlashAttention-2 use?"]))
    assert "PARTIAL ANSWER" in text
    assert "## What could not be established" in text
    assert "- What memory access pattern does FlashAttention-2 use?\n" in text


def test_an_unaffordable_rewrite_is_flagged_to_the_reader():
    text = report(state("PARTIAL", partial_rewrite="unaffordable"))
    assert "written to answer the whole question" in text


# --- ABSTAIN ---------------------------------------------------------------


def test_a_refuted_premise_shows_the_premise_and_the_correction():
    s = state("ABSTAIN", rule="premise_refuted", draft="",
              withheld_draft="Mamba interleaves attention layers because…")
    s["premise_check"] = {"ran": True, "failed": True, "presuppositions": [
        {"text": "Mamba contains multi-head attention layers", "verdict": "CONTRADICTED"},
        {"text": "Mamba has selective SSM blocks", "verdict": "SUPPORTED"}],
        "correction": "Mamba is an attention-free architecture."}
    text = report(s)
    assert "rests on a premise the evidence contradicts" in text
    assert "*Mamba contains multi-head attention layers*" in text
    assert "Mamba has selective SSM blocks" not in text
    assert "attention-free architecture" in text


def test_a_withheld_draft_never_reaches_the_report():
    s = state("ABSTAIN", draft="", withheld_draft="A CONFIDENT WRONG REPORT")
    text = report(s)
    assert "CONFIDENT WRONG" not in text
    assert "## Sources" not in text and "## Answer" not in text


def test_a_system_failure_says_it_is_not_about_the_evidence():
    s = state("ABSTAIN", rule="system_failure", draft="", error="boom")
    assert "not a judgment about the evidence" in report(s)


# --- markers and honesty ---------------------------------------------------


def test_stray_markers_are_stripped_and_the_sentence_still_gets_its_mark():
    """The critic saw '… half [unresolved].'; the draft is stripped of the
    marker before rendering. Keyed unstripped, the verdict would silently
    not apply."""
    draft = (f"FlashAttention-2 partitions work. [{P[0]}] "
             f"This reduces traffic by half [unresolved]. [{P[1]}]")
    findings = findings_for(draft, {0: ("SUPPORTED", "ACCEPT"),
                                    1: ("UNSUPPORTED", "UNRESOLVED")})
    text = report(state(draft=draft, findings=findings))
    assert "[unresolved]" not in text
    assert "*[unverified]*" in text


def test_the_heuristic_is_never_presented_as_calibrated():
    assert "Uncalibrated" in report(state(calibrated=False))
    assert "Uncalibrated" not in report(state(calibrated=True))


def test_a_hard_rule_decision_shows_no_score():
    """The premise rule bypasses tau; a score beside it would imply the
    scorer decided."""
    s = state("ABSTAIN", rule="premise_refuted", draft="")
    s["premise_check"] = {"ran": True, "failed": True, "presuppositions": []}
    text = report(s)
    assert "score" not in text.lower().split("## why")[0]
    assert "Uncalibrated" not in text


def test_the_node_writes_final_report():
    from src.nodes.finalizer import finalizer_node

    update = finalizer_node(Finalizer(CFG))(state())
    assert update["final_report"].startswith("# How do FlashAttention-2")


# --- C28 housekeeping ------------------------------------------------------


def test_merge_is_read_from_the_structured_field():
    from src.sufficiency.scorer import sub_question_scores
    from tests.decision_fixtures import STRONG, WEAK, decision_state

    s = decision_state(STRONG, WEAK, statuses={"sq2": "abandoned"})
    s["plan"][1]["covered_by"] = "sq1"
    s["plan"][1]["rationale"] = "no parseable text here"
    assert not sub_question_scores(s)[1].included


def test_the_dead_scoring_stage_is_gone():
    import src.controller as controller

    assert not hasattr(controller, "SUFFICIENCY_SCORER")
    assert "sufficiency_scorer" not in controller.ALL_NODES
