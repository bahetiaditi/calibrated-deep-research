"""C37: judging and scoring end to end, with a fake judge."""
import json

from analysis.score import apply_tau, score_config
from src.eval import judge as J
from src.eval import metrics as M
from src.eval.runner import load_questions

QS = {q["id"]: q for q in load_questions("all")[0]}
AC = next(q for q in QS.values() if q["tier"] == "A" and q["split"] == "calibration")
D1 = next(q for q in QS.values() if q.get("subtype") == "D1" and q["split"] == "calibration")
PID = next(pid for c in AC["gold_claims"] for pid in c["passage_ids"])


class FakeJudge:
    def __init__(self, d_label="CORRECT", coverage_label="PRESENT"):
        self.d_label, self.coverage_label, self.calls = d_label, coverage_label, []

    def complete_json(self, system, user, **kw):
        self.calls.append(user[:40])
        if "gold claims" in user.lower():
            n = user.count("\n[")
            return {"claims": [{"i": i, "label": self.coverage_label} for i in range(n + 1)]}, None
        if "should not" in user.lower():
            return {"label": self.d_label, "reason": "r"}, None
        return {"sentences": [{"i": 0, "label": "SUPPORTED"}, {"i": 1, "label": "UNSUPPORTED"}]}, None


def write(tmp_path, config, records):
    d = tmp_path / config / "calibration"; d.mkdir(parents=True, exist_ok=True)
    for r in records:
        (d / f"{r['id']}.json").write_text(json.dumps(r))


def agent_records():
    draft = f"Claim one is stated here. [{PID}] Claim two is uncited."
    return [
        {"id": AC["id"], "tier": "A", "terminal_decision": "ANSWER", "rule": "threshold",
         "score": 0.8, "final_report": "report text", "draft": draft, "draft_mode": "answer",
         "decision": {}, "budget": {"llm_calls_used": 12}},
        {"id": D1["id"], "tier": "D", "terminal_decision": "ABSTAIN", "rule": "premise_refuted",
         "score": 0.0, "final_report": "the premise is false", "draft": "", "draft_mode": "",
         "decision": {}, "budget": {"llm_calls_used": 2}},
    ]


def corpus():
    return {PID: {"id": PID, "text": "passage text"}}


def test_sentences_carry_their_own_citations():
    after = J.sentences_with_citations(f"First claim. [{PID}] Second claim has none.")
    before = J.sentences_with_citations(f"First claim [{PID}]. Second claim has none.")
    assert after == [("First claim.", [PID]), ("Second claim has none.", [])]
    assert before[0][1] == [PID] and before[1][1] == []


def test_scoring_judges_once_then_reads_the_cache(tmp_path):
    write(tmp_path, "agent", agent_records())
    judge = FakeJudge()
    outs, _ = score_config("agent", "calibration", tmp_path, QS, corpus(), judge, judge=True)
    first = len(judge.calls)
    assert first == 3                     # coverage (A), grounding (A), tier_d (D)
    outs2, _ = score_config("agent", "calibration", tmp_path, QS, corpus(), judge, judge=True)
    assert len(judge.calls) == first      # all from judged/ cache
    s = M.summarise(outs2)
    assert s["ac_accuracy_answered"] == 1.0 and s["d_correct"] == 1.0
    assert s["hallucination"] == 0.5


def test_an_unreadable_tier_d_judgment_is_not_counted_against_the_system(tmp_path):
    write(tmp_path, "agent", agent_records())
    outs, _ = score_config("agent", "calibration", tmp_path, QS, corpus(),
                           FakeJudge(d_label="???"), judge=True)
    s = M.summarise(outs)
    assert s["unjudged"] == 1 and s["d_false_answer"] is None


def test_an_abstained_ac_question_is_graded_on_its_answer_draft(tmp_path):
    r = agent_records()[0] | {"terminal_decision": "ABSTAIN", "final_report": "no answer",
                              "draft": "", "decision": {"withheld_draft": "the answer draft",
                                                         "withheld_draft_mode": "answer"}}
    write(tmp_path, "agent", [r])
    outs, _ = score_config("agent", "calibration", tmp_path, QS, corpus(), FakeJudge(), judge=True)
    judged = json.loads((tmp_path / "agent" / "calibration" / "judged" / f"{AC['id']}.json").read_text())
    assert "coverage_answer" in judged and "coverage_delivered" not in judged
    assert outs[0].kind == "abstained" and outs[0].correct_if_answered


def test_b3_decisions_wait_for_tau_then_apply_it(tmp_path):
    pending = [{"id": "x", "rule": "threshold_pending", "score": 0.7, "terminal_decision": None},
               {"id": "y", "rule": "threshold_pending", "score": 0.3, "terminal_decision": None}]
    assert apply_tau(pending, tmp_path / "missing.json") is False
    (tmp_path / "tau.json").write_text(json.dumps({"tau": 0.5}))
    assert apply_tau(pending, tmp_path / "tau.json") is True
    assert [r["terminal_decision"] for r in pending] == ["ANSWER", "ABSTAIN"]
