"""Tests for the graph state.

Two things carry real weight here: the reducers (a wrong one silently
corrupts the sufficiency features) and the round-trip through LangGraph's
checkpointer, which is the C14 acceptance criterion.
"""
import pytest

from src.state import (
    TERMINAL_STATUSES,
    ClaimVerification,
    Passage,
    ResearchState,
    SubQuestion,
    all_terminal,
    evidence_for,
    find_sub_question,
    independent_sources,
    make_passage_id,
    merge_contradictions,
    merge_evidence,
    new_state,
    new_sub_question,
    pending_sub_questions,
)


def passage(pid, **over):
    base = dict(
        id=pid, source_type="arxiv", source_id="1706.03762v5",
        source_domain="arxiv.org", title="t", text="body",
        section="Results", published="2017-06-12", sub_question_id="sq1",
        retrieval_score=0.5, rerank_score=7.0,
    )
    base.update(over)
    return Passage(**base)


# --- passage ids (C6, retained) --------------------------------------------


def test_passage_id_is_deterministic():
    assert make_passage_id("a", "t", "Results") == make_passage_id("a", "t", "Results")


def test_section_is_part_of_the_id():
    """A claim from Related Work describes someone else's contribution, so
    the same sentence in two sections is not the same evidence."""
    assert (make_passage_id("a", "t", "Results")
            != make_passage_id("a", "t", "Related Work"))


def test_field_boundaries_cannot_collide():
    assert make_passage_id("ab", "c") != make_passage_id("a", "bc")


# --- evidence reducer ------------------------------------------------------


def test_evidence_accumulates():
    merged = merge_evidence([passage("P1")], [passage("P2")])
    assert [p["id"] for p in merged] == ["P1", "P2"]


def test_duplicate_evidence_is_dropped():
    """Re-retrieval legitimately returns passages already in state. Counting
    one piece of evidence twice inflates f2 and f4 — the exact failure the
    abstention layer exists to prevent."""
    merged = merge_evidence([passage("P1")], [passage("P1"), passage("P2")])
    assert [p["id"] for p in merged] == ["P1", "P2"]


def test_first_copy_wins_on_duplicate():
    """A passage keeps the scores from the round that first found it.
    Overwriting would let one sub-question's rerank score silently replace
    another's, and feature extraction reads those scores."""
    merged = merge_evidence(
        [passage("P1", rerank_score=9.0, sub_question_id="sq1")],
        [passage("P1", rerank_score=1.0, sub_question_id="sq2")],
    )
    assert merged[0]["rerank_score"] == 9.0
    assert merged[0]["sub_question_id"] == "sq1"


def test_evidence_reducer_handles_none():
    assert merge_evidence(None, None) == []
    assert len(merge_evidence(None, [passage("P1")])) == 1
    assert len(merge_evidence([passage("P1")], None)) == 1


def test_evidence_without_id_is_skipped():
    assert merge_evidence([], [{"text": "no id"}]) == []


def test_evidence_reducer_does_not_mutate_input():
    original = [passage("P1")]
    merge_evidence(original, [passage("P2")])
    assert len(original) == 1


# --- contradiction reducer -------------------------------------------------


def test_contradictions_accumulate():
    merged = merge_contradictions(
        [{"passage_a": "P1", "passage_b": "P2", "score": 0.9}],
        [{"passage_a": "P3", "passage_b": "P4", "score": 0.8}],
    )
    assert len(merged) == 2


def test_reversed_pair_is_the_same_finding():
    """Contradiction is symmetric. Storing both directions would double f5's
    numerator while the denominator counts each pair once."""
    merged = merge_contradictions(
        [{"passage_a": "P1", "passage_b": "P2", "score": 0.9}],
        [{"passage_a": "P2", "passage_b": "P1", "score": 0.9}],
    )
    assert len(merged) == 1


def test_contradiction_reducer_handles_none():
    assert merge_contradictions(None, None) == []


# --- constructors ----------------------------------------------------------


def test_new_state_populates_every_field():
    """Nodes return partial updates, so ResearchState is total=False. A fully
    populated initial state means a node reading it never meets a missing
    key."""
    state = new_state("why does Mamba-2 use RoPE?")
    for key in ResearchState.__annotations__:
        assert key in state, f"new_state omits {key}"


def test_new_state_carries_the_question():
    assert new_state("q")["question"] == "q"


def test_new_state_generates_a_trace_id():
    assert new_state("q")["trace_id"] != new_state("q")["trace_id"]


def test_new_sub_question_defaults():
    sq = new_sub_question("what is RRF", rationale="needs definition")
    assert sq["status"] == "pending"
    assert sq["rounds"] == 0
    assert sq["sufficiency"] is None
    assert sq["attempted_routes"] == [] and sq["attempted_queries"] == []
    assert sq["rationale"] == "needs definition"


def test_sub_question_ids_are_unique():
    assert new_sub_question("a")["id"] != new_sub_question("b")["id"]


def test_sub_question_carries_parent_for_splits():
    """D1 can split a sub-question; the lineage has to survive or plan
    revision is unauditable."""
    child = new_sub_question("narrower", parent_id="sq1")
    assert child["parent_id"] == "sq1"


# --- helpers ---------------------------------------------------------------


@pytest.fixture
def state():
    s = new_state("root question")
    s["plan"] = [
        new_sub_question("a", sub_question_id="sq1"),
        new_sub_question("b", sub_question_id="sq2"),
        new_sub_question("c", sub_question_id="sq3"),
    ]
    s["plan"][1]["status"] = "sufficient"
    s["plan"][2]["status"] = "abandoned"
    s["evidence"] = [
        passage("P1", sub_question_id="sq1", source_id="paperA"),
        passage("P2", sub_question_id="sq1", source_id="paperA"),
        passage("P3", sub_question_id="sq2", source_id="paperB"),
    ]
    return s


def test_find_sub_question(state):
    assert find_sub_question(state, "sq2")["text"] == "b"
    assert find_sub_question(state, "nope") is None


def test_pending_excludes_terminal_statuses(state):
    assert [sq["id"] for sq in pending_sub_questions(state)] == ["sq1"]


def test_terminal_statuses_are_the_three_that_stop_work():
    assert TERMINAL_STATUSES == {"sufficient", "insufficient", "abandoned"}


def test_all_terminal(state):
    assert not all_terminal(state)
    state["plan"][0]["status"] = "insufficient"
    assert all_terminal(state)


def test_empty_plan_is_not_terminal():
    """An empty plan means the planner has not run — routing to the decider
    here would abstain before doing any work."""
    assert not all_terminal(new_state("q"))


def test_evidence_for_sub_question(state):
    assert {p["id"] for p in evidence_for(state, "sq1")} == {"P1", "P2"}


def test_independent_sources_counts_distinct_papers(state):
    """f4 keys on source_id, so two chunks of one paper are one source but
    two different papers are two."""
    assert independent_sources(state["evidence"]) == 2
    assert independent_sources(evidence_for(state, "sq1")) == 1


def test_independent_sources_empty():
    assert independent_sources([]) == 0


# --- the C14 acceptance check ----------------------------------------------


def test_state_round_trips_through_the_checkpointer(tmp_path):
    """C14's criterion. LangGraph serialises state at every node, so a field
    it cannot round-trip would surface as a mid-run failure at C19 rather
    than here."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    def planner(state: ResearchState) -> dict:
        return {
            "plan": [new_sub_question("sub one", sub_question_id="sq1")],
            "plan_revisions": 1,
        }

    def retriever(state: ResearchState) -> dict:
        return {
            "evidence": [passage("P1"), passage("P2")],
            "contradictions": [{"passage_a": "P1", "passage_b": "P2",
                                "score": 0.81, "cross_source": True}],
        }

    def decider(state: ResearchState) -> dict:
        return {
            "terminal_decision": "PARTIAL",
            "decision_rationale": "one sub-question unresolved",
            "unresolved_gaps": ["no source gives the number"],
            "critic_findings": [ClaimVerification(
                claim="x reaches 73%", sub_question_id="sq1",
                cited_passage_ids=["P1"], verdict="SUPPORTED",
                reasoning="stated directly", action_taken="accept",
            )],
        }

    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever)
    builder.add_node("decider", decider)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "decider")
    builder.add_edge("decider", END)

    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite")) as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        final = graph.invoke(new_state("root question"), config)

        assert final["question"] == "root question"
        assert final["plan"][0]["id"] == "sq1"
        assert {p["id"] for p in final["evidence"]} == {"P1", "P2"}
        assert final["terminal_decision"] == "PARTIAL"
        assert final["critic_findings"][0]["verdict"] == "SUPPORTED"
        assert final["unresolved_gaps"] == ["no source gives the number"]

        # Reloading from the checkpointer must give the same state back.
        restored = graph.get_state(config).values
        assert restored["plan"][0]["text"] == "sub one"
        assert len(restored["evidence"]) == 2
        assert restored["contradictions"][0]["score"] == 0.81


def test_reducer_applies_across_nodes(tmp_path):
    """Two nodes both returning evidence must accumulate, not overwrite —
    and the duplicate must not double-count."""
    from langgraph.graph import END, START, StateGraph

    def first(state: ResearchState) -> dict:
        return {"evidence": [passage("P1")]}

    def second(state: ResearchState) -> dict:
        return {"evidence": [passage("P1"), passage("P2")]}

    builder = StateGraph(ResearchState)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    final = builder.compile().invoke(new_state("q"))
    assert [p["id"] for p in final["evidence"]] == ["P1", "P2"]


def test_partial_updates_leave_other_fields_intact():
    """total=False lets a node return only what it owns. A node that had to
    echo the whole state back would make every node a merge conflict."""
    from langgraph.graph import END, START, StateGraph

    def only_draft(state: ResearchState) -> dict:
        return {"draft": "some text"}

    builder = StateGraph(ResearchState)
    builder.add_node("n", only_draft)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)

    state = new_state("q")
    state["plan"] = [new_sub_question("keep me", sub_question_id="sq1")]
    final = builder.compile().invoke(state)
    assert final["draft"] == "some text"
    assert final["plan"][0]["text"] == "keep me"