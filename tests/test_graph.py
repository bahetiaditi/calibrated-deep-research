"""Tests for graph assembly.

Fake nodes throughout — no LLM, no retrieval. What is under test is that the
hub-and-spoke wiring puts the controller in charge of every transition, that
the whole run checkpoints, and that a resumed run continues rather than
restarts.
"""
import pytest

from src.config import Config
from src.controller import PLANNER, PREMISE_CHECKER, RETRIEVER, SYNTHESIZER
from src.graph import Components, build_graph, run_research
from src.state import new_state, new_sub_question

CFG = Config({
    "agency": {"d1_plan_revision": False,
               "skip_research_on_failed_premise": True},
    "budget": {"max_rounds_per_subquestion": 3},
})


def passage(pid):
    return {
        "id": pid, "source_type": "arxiv", "source_id": "p",
        "source_domain": "arxiv.org", "title": "t", "text": f"text {pid}",
        "section": "Results", "published": "2023-01-01",
        "sub_question_id": "sq1", "retrieval_score": 0.5, "rerank_score": 7.0,
    }


class Spy:
    """Records which nodes ran, in order."""

    def __init__(self):
        self.visits: list[str] = []

    def premise(self, failed=False):
        def node(state):
            self.visits.append("premise_checker")
            return {"premise_check": {"ran": True, "failed": failed,
                                      "correction": "actually Z"}}
        return node

    def planner(self, n=2):
        def node(state):
            self.visits.append("planner")
            return {"plan": [
                new_sub_question(f"sub {i}", sub_question_id=f"sq{i + 1}")
                for i in range(n)
            ]}
        return node

    def retriever(self, found=True):
        def node(state):
            self.visits.append("retriever")
            pending = [sq for sq in state["plan"] if sq["status"] == "pending"]
            if not pending:
                return {}
            target = pending[0]["id"]
            plan = []
            for sq in state["plan"]:
                if sq["id"] == target:
                    sq = {**sq, "status": "sufficient" if found else "insufficient",
                          "rounds": sq["rounds"] + 1,
                          "evidence_ids": [f"P{target}"]}
                plan.append(sq)
            return {"plan": plan,
                    "evidence": [passage(f"P{target}")] if found else []}
        return node

    def synthesizer(self):
        def node(state):
            self.visits.append("synthesizer")
            return {"draft": f"a report citing {len(state['evidence'])} passages"}
        return node


def components(spy, **over):
    base = Components(
        premise_checker=spy.premise(),
        planner=spy.planner(),
        retriever=spy.retriever(),
        synthesizer=spy.synthesizer(),
    )
    for key, value in over.items():
        setattr(base, key, value)
    return base


def run(spy, question="How do X and Y differ?", **over):
    graph = build_graph(components(spy, **over), config=CFG)
    return graph.invoke(new_state(question))


# --- assembly --------------------------------------------------------------


def test_registered_nodes_only():
    comps = Components(planner=lambda s: {}, retriever=lambda s: {})
    assert set(comps.registered()) == {PLANNER, RETRIEVER}


def test_empty_components_rejected():
    with pytest.raises(ValueError, match="no nodes registered"):
        build_graph(Components(), config=CFG)


# --- the happy path --------------------------------------------------------


def test_full_pipeline_runs_in_order():
    spy = Spy()
    final = run(spy)
    assert spy.visits == [
        "premise_checker", "planner", "retriever", "retriever", "synthesizer"
    ]
    assert final["draft"]


def test_retriever_loops_once_per_sub_question():
    spy = Spy()
    run(spy, planner=Spy().planner(4))
    assert spy.visits.count("retriever") == 4


def test_evidence_accumulates_across_rounds():
    spy = Spy()
    final = run(spy, planner=Spy().planner(3))
    assert len(final["evidence"]) == 3


def test_state_survives_the_whole_run():
    spy = Spy()
    final = run(spy, question="my question")
    assert final["question"] == "my question"
    assert final["premise_check"]["ran"] is True
    assert len(final["plan"]) == 2


# --- the controller is in charge ------------------------------------------


def test_missing_node_is_skipped_without_changing_routing_code():
    """Registering a node must change routing without editing routing."""
    spy = Spy()
    graph = build_graph(
        Components(planner=spy.planner(1), retriever=spy.retriever(),
                   synthesizer=spy.synthesizer()),
        config=CFG,
    )
    graph.invoke(new_state("q"))
    assert "premise_checker" not in spy.visits
    assert spy.visits[0] == "planner"


def test_refuted_premise_skips_research_and_synthesis():
    """The cheap path: the correction is already in state, so spending the
    budget elaborating a refuted premise buys nothing."""
    spy = Spy()
    final = run(spy, premise_checker=spy.premise(failed=True))
    assert spy.visits == ["premise_checker"]
    assert final["draft"] == ""
    assert final["premise_check"]["correction"] == "actually Z"


def test_failed_retrieval_still_terminates():
    spy = Spy()
    final = run(spy, retriever=spy.retriever(found=False))
    assert spy.visits.count("retriever") == 2
    assert "synthesizer" not in spy.visits      # no evidence, no report
    assert final["draft"] == ""


def test_no_infinite_loop_when_the_synthesizer_produces_nothing():
    """The loop the controller test caught: an empty draft keeps
    `not state['draft']` true forever."""
    spy = Spy()

    from src.nodes.synthesizer import Synthesizer, synthesizer_node

    class EmptyProvider:
        def complete(self, system, user, **kw):
            spy.visits.append("synthesizer")

            class R:
                text = "   "
            return R()

    empty_synth = synthesizer_node(
        Synthesizer(Config({}), provider=EmptyProvider())
    )
    final = run(spy, synthesizer=empty_synth)
    assert spy.visits.count("synthesizer") == 1
    assert final is not None


# --- checkpointing ---------------------------------------------------------


def test_run_checkpoints_and_resumes(tmp_path):
    """§7.1 needs this: the eval spans days and both providers rate-limit, so
    a run that cannot resume has to restart and re-spend its quota."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    spy = Spy()
    db = str(tmp_path / "ckpt.sqlite")
    config = {"configurable": {"thread_id": "t1"}}

    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(components(spy), config=CFG, checkpointer=saver)
        final = graph.invoke(new_state("q"), config)
        assert final["draft"]

    # A fresh process re-opens the same database.
    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(components(spy), config=CFG, checkpointer=saver)
        restored = graph.get_state(config).values
        assert restored["draft"] == final["draft"]
        assert len(restored["plan"]) == 2


def test_resuming_a_finished_run_does_not_redo_the_work(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    spy = Spy()
    db = str(tmp_path / "ckpt.sqlite")
    config = {"configurable": {"thread_id": "t1"}}

    with SqliteSaver.from_conn_string(db) as saver:
        graph = build_graph(components(spy), config=CFG, checkpointer=saver)
        graph.invoke(new_state("q"), config)
        visits_after_first = len(spy.visits)
        graph.invoke(None, config)          # resume
        assert len(spy.visits) == visits_after_first


# --- recursion guard -------------------------------------------------------


def test_recursion_limit_catches_a_routing_loop():
    """A guard against a routing bug, not a feature. If it fires in normal
    operation the controller is wrong."""
    from langgraph.errors import GraphRecursionError

    spy = Spy()

    def never_finishes(state):
        spy.visits.append("retriever")
        return {}                            # never marks anything terminal

    graph = build_graph(components(spy, retriever=never_finishes), config=CFG)
    with pytest.raises(GraphRecursionError):
        graph.invoke(new_state("q"), {"recursion_limit": 8})