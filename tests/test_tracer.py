"""Tests for structured tracing.

The trace is evidence, not logging. Two consumers depend on its correctness:
C35 verifies an ablation actually froze its decision surface by counting
decision events, and C38 annotates failures against MAST by replaying them.
"""
import json

import pytest

from src.tracing.tracer import (
    MAX_LIST_ITEMS,
    MAX_STRING_CHARS,
    DecisionSurface,
    EventType,
    TraceReader,
    Tracer,
)


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces", trace_id="t1", question="why?")


def read(tracer):
    return TraceReader.load(tracer.path)


# --- format ----------------------------------------------------------------


def test_every_line_is_valid_json(tracer):
    with tracer.node("planner"):
        tracer.note("hello", x=1)
    tracer.close()
    for line in tracer.path.read_text().splitlines():
        json.loads(line)


def test_run_start_recorded_with_question(tracer):
    tracer.close()
    start = read(tracer).of_type(EventType.RUN_START)[0]
    assert start["payload"]["question"] == "why?"
    assert start["payload"]["trace_id"] == "t1"


def test_sequence_is_monotonic(tracer):
    for i in range(5):
        tracer.note(f"n{i}")
    tracer.close()
    seqs = [e["seq"] for e in read(tracer).events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))


def test_file_named_by_trace_id(tracer):
    assert tracer.path.name == "t1.jsonl"


def test_disabled_tracer_writes_nothing(tmp_path):
    t = Tracer(tmp_path / "tr", trace_id="x", enabled=False)
    t.note("ignored")
    t.close()
    assert not (tmp_path / "tr" / "x.jsonl").exists()


# --- node spans ------------------------------------------------------------


def test_node_span_emits_enter_and_exit_with_duration(tracer):
    with tracer.node("retriever"):
        pass
    tracer.close()
    r = read(tracer)
    assert r.nodes_visited() == ["retriever"]
    assert r.of_type(EventType.NODE_EXIT)[0]["duration_ms"] >= 0


def test_events_inside_a_span_are_attributed_to_that_node(tracer):
    with tracer.node("critic"):
        tracer.note("checking")
    tracer.close()
    note = read(tracer).of_type(EventType.NOTE)[0]
    assert note["payload"]["node"] == "critic"


def test_exception_in_node_is_recorded_and_reraised(tracer):
    with pytest.raises(ValueError):
        with tracer.node("planner"):
            raise ValueError("bad plan")
    tracer.close()
    errors = read(tracer).errors()
    assert errors[0]["payload"]["error_type"] == "ValueError"
    assert "bad plan" in errors[0]["payload"]["error"]


def test_node_stack_unwinds_after_error(tracer):
    with pytest.raises(ValueError):
        with tracer.node("a"):
            raise ValueError("x")
    tracer.note("after")
    tracer.close()
    # 'after' must not still be attributed to the failed node. With the stack
    # unwound there is no node key at all, so the payload is omitted entirely.
    note = read(tracer).of_type(EventType.NOTE)[0]
    assert note.get("payload", {}).get("node") != "a"


# --- routing ---------------------------------------------------------------


def test_route_records_reason_and_state_summary(tracer):
    tracer.route(
        "planner",
        reason="sub-question insufficient after 3 rounds",
        state_summary={"plan_size": 4, "budget_used": 0.6},
        from_node="controller",
    )
    tracer.close()
    frm, to, reason = read(tracer).routes()[0]
    assert (frm, to) == ("controller", "planner")
    assert "insufficient" in reason


# --- decision surfaces (the C35 ablation check) ----------------------------


def test_decision_records_alternatives_and_rationale(tracer):
    tracer.decision(
        DecisionSurface.D2_ROUTE_AND_QUERY,
        chosen="arxiv_fulltext",
        alternatives=["arxiv_meta", "web"],
        rationale="abstract lacked the number",
        inputs={"attempted_routes": ["arxiv_meta"]},
    )
    tracer.close()
    d = read(tracer).decisions(DecisionSurface.D2_ROUTE_AND_QUERY)[0]["payload"]
    assert d["chosen"] == "arxiv_fulltext"
    assert "web" in d["alternatives"]
    assert d["was_adaptive"] is True


def test_ablated_surface_reports_zero_adaptive_decisions(tracer):
    """This is exactly C35's check: A1 freezes D1, so the trace of an A1 run
    must show zero adaptive plan revisions. A non-zero count means the
    ablation did not take effect and its result is invalid."""
    tracer.decision(
        DecisionSurface.D1_PLAN_REVISION,
        chosen="keep", rationale="static policy", was_adaptive=False,
    )
    tracer.decision(
        DecisionSurface.D3_DEPTH_AND_STOPPING, chosen="deeper", was_adaptive=True,
    )
    tracer.close()
    counts = read(tracer).adaptive_decision_counts()
    assert counts["D1_plan_revision"] == 0
    assert counts["D3_depth_and_stopping"] == 1


def test_all_five_surfaces_present_in_counts(tracer):
    tracer.close()
    counts = read(tracer).adaptive_decision_counts()
    for surface in DecisionSurface:
        assert surface.value in counts


# --- llm and retrieval -----------------------------------------------------


def test_llm_totals_separate_live_from_cached(tracer):
    tracer.llm_call(role="judgment", model="m", provider="groq",
                    prompt_tokens=100, completion_tokens=50, total_tokens=150,
                    latency_s=1.0, from_cache=False)
    tracer.llm_call(role="judgment", model="m", provider="groq",
                    prompt_tokens=100, completion_tokens=50, total_tokens=150,
                    latency_s=0.0, from_cache=True)
    tracer.close()
    totals = read(tracer).llm_totals()
    assert totals == {
        "calls": 2, "live_calls": 1, "cached_calls": 1,
        "total_tokens": 300, "live_tokens": 150, "by_model": {"m": 2},
    }


def test_retrieval_recorded(tracer):
    tracer.retrieval(route="web", query="flashattention memory",
                     n_results=6, latency_s=0.4, sub_question_id="sq1")
    tracer.close()
    ev = read(tracer).of_type(EventType.RETRIEVAL)[0]
    assert ev["name"] == "web" and ev["payload"]["n_results"] == 6


# --- payload guards --------------------------------------------------------


def test_long_strings_are_truncated(tracer):
    tracer.note("big", blob="x" * (MAX_STRING_CHARS + 500))
    tracer.close()
    blob = read(tracer).of_type(EventType.NOTE)[0]["payload"]["blob"]
    assert len(blob) < MAX_STRING_CHARS + 200
    assert "+500 chars" in blob


def test_long_lists_are_truncated(tracer):
    tracer.note("big", items=list(range(MAX_LIST_ITEMS + 10)))
    tracer.close()
    items = read(tracer).of_type(EventType.NOTE)[0]["payload"]["items"]
    assert len(items) == MAX_LIST_ITEMS + 1
    assert "more" in str(items[-1])


def test_nested_payloads_are_truncated(tracer):
    tracer.note("nested", d={"inner": {"blob": "y" * (MAX_STRING_CHARS + 100)}})
    tracer.close()
    blob = read(tracer).of_type(EventType.NOTE)[0]["payload"]["d"]["inner"]["blob"]
    assert "chars" in blob


# --- robustness ------------------------------------------------------------


def test_truncated_final_line_does_not_lose_the_trace(tracer):
    """A process killed mid-write leaves a partial line. Losing that one
    event must not cost the whole file."""
    tracer.note("good")
    tracer.close()
    with tracer.path.open("a") as fh:
        fh.write('{"seq": 99, "type": "note", "na')
    r = read(tracer)
    assert len(r.events) >= 2
    assert any(e["name"] == "good" for e in r.of_type(EventType.NOTE))


def test_context_manager_records_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with Tracer(tmp_path / "tr", trace_id="cm") as t:
            raise RuntimeError("exploded")
    r = TraceReader.load(tmp_path / "tr" / "cm.jsonl")
    assert r.errors()
    assert r.of_type(EventType.RUN_END)[0]["payload"]["ok"] is False


def test_summary_shape(tracer):
    with tracer.node("planner"):
        tracer.decision(DecisionSurface.D1_PLAN_REVISION, chosen="split")
    tracer.close()
    s = read(tracer).summary()
    assert s["trace_id"] == "t1"
    assert s["nodes"] == ["planner"]
    assert s["decisions"]["D1_plan_revision"] == 1