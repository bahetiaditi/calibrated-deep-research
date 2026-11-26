"""C37.1: gpt-oss's dagger citations, the synthesis log, explicit re-runs.

The live failure (A03, calibration): the delivered draft cited every
sentence as `【P09†L1-L3】`. C27.4's normaliser knew `【P09】` but not the
locator suffix, so all 12 citations stayed plain text, the critic saw 8
uncited claims, and the run was ANSWERED at score 1.00 with 0 of 8 verified.
The tests that matter assert the EFFECT: the critic sees passages, the
record shows the loss, and a superseded result stays visible.
"""
import json
import operator
from types import SimpleNamespace
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from src.config import Config
from src.eval.runner import EvalRefused, EvalRunner
from src.nodes.critic import segment
from src.nodes.synthesizer import Synthesizer, normalize_citations, synthesizer_node
from src.state import new_state
from tests.test_synthesizer import FakeProvider, state_with

BASE = Config({"llm": {"cache": {"dir": "data/cache/llm"}},
               "run": {"checkpoint_db": "data/checkpoints.sqlite", "trace_dir": "traces"}})

# Two sentences of the A03 draft, verbatim (labels as the model wrote them).
A03 = ("Chinchilla is a dense transformer language model containing 70 billion "
       "parameters, roughly one‑quarter the size of Gopher’s 280 billion‑parameter "
       "architecture【P09†L1-L3】【P16†L1-L3】. It was trained on 1.4 trillion "
       "tokens while consuming the same overall FLOP budget【P09†L4-L7】【P16†L4-L6】.")


def ids(n):
    return [f"Pa{i:02d}{'f' * 9}" for i in range(n)]


def synthesize(text, n=20):
    state = state_with(ids=ids(n))
    return Synthesizer(Config({}), provider=FakeProvider(text)).synthesize(state), state


# --- the normaliser ----------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("A 【P09†L1-L3】.", "A [P09]."),
    ("A 【P09†L1-L3】【P16†L4-L6】.", "A [P09][P16]."),
    ("A 【P09†source】.", "A [P09]."),
    ("A [P02†L4, P05†L1-L2].", "A [P02][P05]."),
    ("A 【P3†L1】.", "A [P03]."),
])
def test_dagger_locators_become_the_instructed_form(raw, expected):
    text, count = normalize_citations(raw)
    assert text == expected and count >= 1


def test_a_line_range_in_a_locator_is_not_read_as_a_passage_range():
    """`L1-L3` is lines 1-3 of P09: it must not cite P01..P03. A plain
    passage range in the same brackets still expands."""
    assert normalize_citations("A 【P09†L1-L3】.")[0] == "A [P09]."
    assert normalize_citations("A 【P02-P04】.")[0] == "A [P02][P03][P04]."


@pytest.mark.parametrize("raw", [
    "A [Pa1b2c3d4e5f6†L1] stays.",       # a content-addressed id is never a label
    "Footnote† P1 is prose.",
])
def test_what_is_not_a_dagger_citation_is_left_alone(raw):
    assert normalize_citations(raw) == (raw, 0)


def test_the_a03_draft_reaches_the_critic_cited():
    """The effect. Before C37.1 both sentences reached the critic with no
    citation and nothing was counted as normalised."""
    draft, _ = synthesize(A03)
    assert draft.unparsed_citations == 0
    sentences = segment(draft.text)
    assert [s.citations for s in sentences] == [[ids(20)[8], ids(20)[15]]] * 2
    assert "†" not in draft.text and "【" not in draft.text


# --- the synthesis log ---------------------------------------------------------


def test_the_node_records_lost_citations_in_state():
    node = synthesizer_node(Synthesizer(Config({}), provider=FakeProvider(
        "Claim one {P01}. Claim two [source: P02].")))
    update = node(state_with(ids=ids(3)))
    (entry,) = update["synthesis_log"]
    assert entry["unparsed_citations"] == 2 and entry["passages_cited"] == 0
    assert entry["mode"] == "answer"


def test_the_log_accumulates_across_syntheses_through_the_graph():
    """A03 synthesized twice; the SECOND draft lost its citations. A
    replace-semantics field would have kept only one of the two."""
    from src.state import ResearchState

    def first(_):
        return {"synthesis_log": [{"unparsed_citations": 0}]}

    def second(_):
        return {"synthesis_log": [{"unparsed_citations": 12}]}

    g = StateGraph(ResearchState)
    g.add_node("first", first)
    g.add_node("second", second)
    g.add_edge(START, "first")
    g.add_edge("first", "second")
    g.add_edge("second", END)
    out = g.compile().invoke(new_state("q"))
    assert [e["unparsed_citations"] for e in out["synthesis_log"]] == [0, 12]


# --- the runner: record, warning, explicit re-run ------------------------------


class FakeRun:
    def __init__(self, unparsed=0):
        self.calls, self.unparsed = [], unparsed

    def __call__(self, question, *, config, thread_id, trace_id, resume):
        self.calls.append(thread_id)
        return SimpleNamespace(
            state={"terminal_decision": "ANSWER", "final_report": "r",
                   "decision": {"rule": "threshold", "score": 0.9, "calibrated": False},
                   "synthesis_log": [{"unparsed_citations": self.unparsed}]},
            budget=None, resumed="", node_visits=[], trace_path="", error=None)


def make(tmp_path, run, split="calibration", **kw):
    return EvalRunner("agent", split, tmp_path, base_config=BASE, run_fn=run, **kw)


def test_the_record_carries_the_synthesis_log_and_the_run_warns(tmp_path):
    run = FakeRun(unparsed=12)
    s = make(tmp_path, run).run(limit=1)
    qid = s.done[0]
    record = json.loads((tmp_path / "agent" / "calibration" / f"{qid}.json").read_text())
    assert record["synthesis_log"] == [{"unparsed_citations": 12}]
    assert s.unparsed == {qid: 12}


def test_a_clean_run_raises_no_warning(tmp_path):
    assert make(tmp_path, FakeRun()).run(limit=1).unparsed == {}


def test_an_explicit_rerun_replaces_a_completed_result_and_keeps_what_it_replaced(tmp_path):
    run = FakeRun()
    first = make(tmp_path, run).run(limit=2).done
    qid = first[0]
    s = make(tmp_path, run).run(rerun={qid: "C37.1: citations lost"})
    # Only the named question ran, on a NEW thread: resuming the finished
    # a0 thread would return the old state unchanged.
    assert run.calls[2:] == [f"agent:calibration:{qid}:a1"]
    assert s.rerun == [qid]
    record = json.loads((tmp_path / "agent" / "calibration" / f"{qid}.json").read_text())
    assert record["attempt"] == 1
    assert record["supersedes"]["attempt"] == 0
    assert record["supersedes"]["thread_id"] == f"agent:calibration:{qid}:a0"
    assert record["supersedes"]["reason"] == "C37.1: citations lost"


def test_without_rerun_a_completed_question_is_still_skipped(tmp_path):
    run = FakeRun()
    qid = make(tmp_path, run).run(limit=1).done[0]
    s = make(tmp_path, run).run(only=[qid])
    assert s.skipped == [qid] and len(run.calls) == 1


def test_rerun_is_refused_on_the_test_split(tmp_path):
    with pytest.raises(EvalRefused, match="calibration-only"):
        make(tmp_path, FakeRun(), split="test", held_out=True).run(rerun={"A01": "x"})


def test_rerun_needs_a_reason_and_an_existing_result(tmp_path):
    run = FakeRun()
    qid = make(tmp_path, run).run(limit=1).done[0]
    with pytest.raises(EvalRefused, match="needs a reason"):
        make(tmp_path, run).run(rerun={qid: " "})
    other = [q["id"] for q in make(tmp_path, run).questions if q["id"] != qid][0]
    with pytest.raises(EvalRefused, match="no result yet"):
        make(tmp_path, run).run(rerun={other: "x"})
    with pytest.raises(EvalRefused, match="not in the calibration split"):
        make(tmp_path, run).run(rerun={"ZZ99": "x"})


def test_reissuing_the_same_rerun_finishes_the_batch_instead_of_repeating_it(tmp_path):
    """C37.3: quota can stop a batch of re-runs half way. The same command
    the next day must skip what it already re-ran for that reason."""
    run = FakeRun()
    a, b = make(tmp_path, run).run(limit=2).done
    reason = "C37.3: homogeneous calibration"
    make(tmp_path, run).run(rerun={a: reason})            # a done, then quota "stopped"
    n = len(run.calls)
    s = make(tmp_path, run).run(rerun={a: reason, b: reason})
    assert run.calls[n:] == [f"agent:calibration:{b}:a1"]  # only b
    assert s.skipped == [a]
    # A NEW reason is a new decision and re-runs again.
    make(tmp_path, run).run(rerun={a: "a later fix"})
    assert run.calls[-1] == f"agent:calibration:{a}:a2"
