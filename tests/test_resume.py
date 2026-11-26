"""C32: run_research really resumes.

Until C32, run_research invoked the graph with a fresh initial state every
time. LangGraph treats input on an existing thread as a new run from START:
every node re-ran and reducer fields accumulated twice. Its docstring said
it resumed; C19 had only tested resumption directly on the graph.
"""
from src.config import Config
from src.graph import Components, run_research
from src.state import new_sub_question


class Spender:
    """A node's helper object: _attach binds the run's budget onto it."""
    budget = None
    tracer = None


def components(log, crash_once):
    spender = Spender()

    def premise(state):
        log.append("premise")
        return {"premise_check": {"ran": True, "failed": False}}

    def planner(state):
        log.append("planner")
        sq = new_sub_question("what?", sub_question_id="sq1")
        # A different gap on every execution: a re-run from START would add
        # a second one through the merge_gaps reducer; a resume cannot.
        return {"plan": [sq], "unresolved_gaps": [f"gap from planner run {len(log)}"]}

    def retriever(state):
        log.append("retriever")
        spender.budget.spend("retrieval", 2)
        spender.budget.spend("llm", 1)
        sq = dict(state["plan"][0]); sq["status"] = "sufficient"; sq["evidence_ids"] = ["P1"]
        passage = {"id": "P1", "source_type": "arxiv", "source_id": "x", "source_domain": "x",
                   "title": "t", "text": "evidence text", "section": None, "published": None,
                   "sub_question_id": "sq1", "retrieval_score": 0.5, "rerank_score": 5.0}
        return {"plan": [sq], "evidence": [passage]}

    def synthesizer(state):
        log.append("synthesizer")
        if crash_once and log.count("synthesizer") == 1:
            raise RuntimeError("simulated crash mid-run")
        return {"draft": "a draft [P1]", "draft_mode": "answer"}

    return Components(premise_checker=premise, planner=planner,
                      retriever=retriever, synthesizer=synthesizer)


def cfg(tmp_path):
    return Config({
        "run": {"checkpoint_db": str(tmp_path / "ckpt.sqlite"),
                "trace_dir": str(tmp_path / "traces")},
        "budget": {"llm_calls_max": 25, "retrieval_calls_max": 20,
                   "wall_clock_max_s": 900, "max_rounds_per_subquestion": 3},
        "agency": {"d1_plan_revision": False},
    })


def test_an_interrupted_run_resumes_where_it_stopped(tmp_path):
    log = []
    comps = components(log, crash_once=True)
    first = run_research("q", components=comps, config=cfg(tmp_path),
                         thread_id="t1", trace_id="t1-a", resume=True)
    assert first.error and "simulated crash" in first.error
    assert log == ["premise", "planner", "retriever", "synthesizer"]

    second = run_research("q", components=comps, config=cfg(tmp_path),
                          thread_id="t1", trace_id="t1-b", resume=True)
    assert second.error is None
    assert second.resumed == "resumed"
    # Only the failed node ran again.
    assert log == ["premise", "planner", "retriever", "synthesizer", "synthesizer"]
    assert second.state["draft"] == "a draft [P1]"


def test_resume_does_not_double_accumulate_reducer_fields(tmp_path):
    """The defect a re-invoke with input causes: evidence added twice."""
    log = []
    comps = components(log, crash_once=True)
    run_research("q", components=comps, config=cfg(tmp_path), thread_id="t2",
                 trace_id="t2-a", resume=True)
    second = run_research("q", components=comps, config=cfg(tmp_path), thread_id="t2",
                          trace_id="t2-b", resume=True)
    assert len(second.state["unresolved_gaps"]) == 1


def test_the_budget_spend_survives_the_resume(tmp_path):
    """Open item 2: a resumed run used to get its whole budget back."""
    log = []
    comps = components(log, crash_once=True)
    run_research("q", components=comps, config=cfg(tmp_path), thread_id="t3",
                 trace_id="t3-a", resume=True)
    second = run_research("q", components=comps, config=cfg(tmp_path), thread_id="t3",
                          trace_id="t3-b", resume=True)
    # Restored, not re-spent: the retriever did not run again.
    assert log.count("retriever") == 1
    assert second.budget.retrieval_calls_used == 2
    assert second.budget.llm_calls_used == 1


def test_the_budget_is_in_state_after_every_node(tmp_path):
    log = []
    result = run_research("q", components=components(log, crash_once=False),
                          config=cfg(tmp_path), thread_id="t4", trace_id="t4-a")
    assert result.state["budget"]["retrieval_calls_used"] == 2
    assert "elapsed_s" in result.state["budget"]


def test_a_finished_run_is_returned_without_running_anything(tmp_path):
    log = []
    comps = components(log, crash_once=False)
    run_research("q", components=comps, config=cfg(tmp_path), thread_id="t5",
                 trace_id="t5-a", resume=True)
    n = len(log)
    again = run_research("q", components=comps, config=cfg(tmp_path), thread_id="t5",
                         trace_id="t5-b", resume=True)
    assert again.resumed == "complete"
    assert len(log) == n


def test_without_resume_a_thread_starts_fresh(tmp_path):
    log = []
    comps = components(log, crash_once=False)
    run_research("q", components=comps, config=cfg(tmp_path), thread_id="t6a", trace_id="x1")
    run_research("q", components=comps, config=cfg(tmp_path), thread_id="t6b", trace_id="x2")
    assert log.count("premise") == 2
