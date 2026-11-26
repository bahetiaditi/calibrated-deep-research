"""C37.2: grading corrections, made before any test-split result exists.

Each test asserts what the NUMBER does, not that a code path ran: a missing
label must not move the hallucination rate; a passage's supporting sentence
past character 900 must reach the judge; a stale judgment must not be mixed
with current ones.
"""
import json
import re

from analysis.score import score_config, spot_check
from src.eval import judge as J
from src.eval import metrics as M
from src.eval.runner import load_questions

QS = {q["id"]: q for q in load_questions("all")[0]}
AC = next(q for q in QS.values() if q["tier"] == "A" and q["split"] == "calibration")
PA, PB = "P" + "a" * 12, "P" + "b" * 12
ITEM_RE = re.compile(r"^\[(\d+)\] ", re.M)


class Judge:
    """Answers grounding prompts with `label_for(i)`; None = omit that item."""

    def __init__(self, label_for, budget=None, coverage_for=None):
        self.label_for, self.budget, self.coverage_for = label_for, budget, coverage_for
        self.prompts = []

    def prompt_token_budget(self, role):
        return self.budget

    def complete_json(self, system, user, **kw):
        self.prompts.append(user)
        shown = [int(i) for i in ITEM_RE.findall(user)]
        if "gold claims" in user.lower():
            f = self.coverage_for or (lambda i: "PRESENT")
            return {"claims": [{"i": i, "label": f(i)} for i in shown if f(i)]}, None
        return {"sentences": [{"i": i, "label": self.label_for(i)}
                              for i in shown if self.label_for(i)]}, None


def corpus(**texts):
    return {pid: {"id": pid, "text": t} for pid, t in texts.items()}


# --- unjudged is not a verdict ------------------------------------------------


def test_a_label_the_judge_never_returns_is_unjudged_not_unsupported():
    draft = f"Claim zero holds [{PA}]. Claim one holds [{PA}]. Claim two holds [{PA}]."
    judge = Judge(lambda i: None if i == 1 else "SUPPORTED")
    g = J.grounding(judge, draft, corpus(**{PA: "Claims hold."}))
    assert [it["label"] for it in g["items"]] == ["SUPPORTED", "UNJUDGED", "SUPPORTED"]
    # The effect on the number: C37 would have reported 1/3 here.
    assert M.hallucination_rate([it["label"] for it in g["items"]]) == 0.0


def test_the_missing_item_is_re_asked_once_alone_and_recovered():
    draft = f"Claim zero holds [{PA}]. Claim one holds [{PA}]."
    calls = []

    def label_for(i):
        calls.append(i)
        return None if (i == 1 and calls.count(1) == 1) else "SUPPORTED"

    judge = Judge(label_for)
    g = J.grounding(judge, draft, corpus(**{PA: "Claims hold."}))
    assert [it["label"] for it in g["items"]] == ["SUPPORTED", "SUPPORTED"]
    assert len(judge.prompts) == 2
    assert ITEM_RE.findall(judge.prompts[1]) == ["1"]      # only the missing one


def test_an_unjudged_gold_claim_removes_the_question_from_accuracy_not_from_the_count():
    labels = J.coverage(Judge(None, coverage_for=lambda i: None if i == 0 else "PRESENT"),
                        "q", ["g0", "g1", "g2"], "report")
    assert labels == ["UNJUDGED", "PRESENT", "PRESENT"]
    record = {"id": AC["id"], "tier": "A", "terminal_decision": "ANSWER", "rule": "threshold",
              "score": 0.9, "budget": {}}
    o = M.grade(record, {"coverage_delivered": labels, "coverage_answer": labels})
    assert o.unjudged and o.selective_unjudged and o.coverage is None
    s = M.summarise([o])
    assert s["ac_accuracy_answered"] is None and s["unjudged"] == 1
    assert s["aurc"] is None                       # not scored as a wrong answer either


# --- the judge reads the passage --------------------------------------------------


def test_a_supporting_sentence_past_character_900_reaches_the_judge():
    filler = "Background sentence about something else entirely. " * 24   # ~1250 chars
    text = filler + "The model was trained for 3.5 days on eight GPUs."
    judge = Judge(lambda i: "SUPPORTED")
    J.grounding(judge, f"It trained for 3.5 days [{PA}].", corpus(**{PA: text}))
    assert len(text) > 1200
    assert "3.5 days on eight GPUs" in judge.prompts[0]


def test_over_budget_passages_are_trimmed_at_sentence_ends_and_counted():
    text = "First sentence is here. " * 120
    judge = Judge(lambda i: "SUPPORTED", budget=900)
    g = J.grounding(judge, f"A claim [{PA}].", corpus(**{PA: text}))
    shown = judge.prompts[0].split(f"<{PA}> ", 1)[1].split("\n")[0]
    assert g["trimmed"] == 1
    assert shown.endswith("is here. […]")          # a whole sentence, then the marker
    assert len(shown) < len(text.strip())


def test_long_drafts_are_judged_in_batches():
    draft = " ".join(f"Claim number {i} holds [{PA}]." for i in range(20))
    judge = Judge(lambda i: "SUPPORTED")
    g = J.grounding(judge, draft, corpus(**{PA: "Claims hold."}), batch_size=15)
    assert len(judge.prompts) == 2
    assert [it["label"] for it in g["items"]] == ["SUPPORTED"] * 20
    assert ITEM_RE.findall(judge.prompts[1])[0] == "15"   # indices keep their place


def test_a_citation_to_no_corpus_passage_is_judged_as_citing_nothing():
    g = J.grounding(Judge(lambda i: "UNSUPPORTED"), f"A claim [{PB}].", corpus(**{PA: "x"}))
    assert g["items"][0]["cites"] == []


# --- the number, decomposed ------------------------------------------------------


def test_hallucination_splits_into_uncited_and_miscitation():
    labels = ["UNSUPPORTED", "UNSUPPORTED", "SUPPORTED", "SUPPORTED", "UNSUPPORTED", "GAP"]
    cited = [False, False, True, True, True, False]
    total = M.hallucination_rate(labels)
    uncited, mis = M.hallucination_parts(labels, cited)
    assert total == 3 / 5 and uncited == 2 / 5 and mis == 1 / 3
    # uncited + cited-UNSUPPORTED/factual adds back to the total
    assert abs(uncited + 1 / 5 - total) < 1e-9


def test_the_summary_reports_the_parts():
    record = {"id": AC["id"], "tier": "A", "terminal_decision": "ANSWER", "rule": "threshold",
              "score": 0.9, "budget": {}}
    judged = {"coverage_delivered": ["PRESENT"], "coverage_answer": ["PRESENT"],
              "grounding_items": [{"i": 0, "label": "UNSUPPORTED", "cites": []},
                                  {"i": 1, "label": "SUPPORTED", "cites": [PA]}]}
    s = M.summarise([M.grade(record, judged)])
    assert s["hallucination"] == 0.5 and s["halluc_uncited"] == 0.5 and s["miscitation"] == 0.0


# --- stale judgments, and a sheet a human can check ----------------------------


def write(tmp_path, record, judged=None):
    d = tmp_path / "agent" / "calibration"
    (d / "judged").mkdir(parents=True, exist_ok=True)
    (d / f"{record['id']}.json").write_text(json.dumps(record))
    if judged is not None:
        (d / "judged" / f"{record['id']}.json").write_text(json.dumps(judged))


def ac_record():
    pid = next(p for c in AC["gold_claims"] for p in c["passage_ids"])
    return {"id": AC["id"], "tier": "A", "terminal_decision": "ANSWER", "rule": "threshold",
            "score": 0.8, "final_report": "report", "draft": f"One claim holds [{pid}].",
            "draft_mode": "answer", "decision": {}, "budget": {}}, pid


def test_a_judgment_from_an_older_grader_is_rejudged_not_mixed(tmp_path):
    record, pid = ac_record()
    old = {"coverage_delivered": ["PRESENT"], "grounding": ["UNSUPPORTED"]}   # C37: no version
    write(tmp_path, record, old)
    judge = Judge(lambda i: "SUPPORTED")
    outs, _ = score_config("agent", "calibration", tmp_path, QS, corpus(**{pid: "x"}),
                           judge, judge=True)
    assert judge.prompts                                     # it was judged again
    saved = json.loads((tmp_path / "agent" / "calibration" / "judged" / f"{AC['id']}.json").read_text())
    assert saved["judge_version"] == J.JUDGE_VERSION
    assert outs[0].grounding == ("SUPPORTED",)


def test_a_rerun_record_is_judged_afresh(tmp_path):
    """C37.1 re-runs keep the question id. A judgment of the replaced record
    must not be reused for the new one."""
    record, pid = ac_record()
    record |= {"thread_id": "agent:calibration:X:a0", "finished_at": "t0"}
    write(tmp_path, record)
    judge = Judge(lambda i: "UNSUPPORTED")
    score_config("agent", "calibration", tmp_path, QS, corpus(**{pid: "x"}), judge, judge=True)
    n = len(judge.prompts)
    score_config("agent", "calibration", tmp_path, QS, corpus(**{pid: "x"}), judge, judge=True)
    assert len(judge.prompts) == n                       # same record: cached
    write(tmp_path, record | {"thread_id": "agent:calibration:X:a1", "finished_at": "t1"})
    judge.label_for = lambda i: "SUPPORTED"
    outs, _ = score_config("agent", "calibration", tmp_path, QS, corpus(**{pid: "x"}),
                           judge, judge=True)
    assert len(judge.prompts) > n and outs[0].grounding == ("SUPPORTED",)


def test_with_no_judge_a_stale_judgment_is_skipped_not_used(tmp_path):
    record, pid = ac_record()
    write(tmp_path, record, {"grounding": ["UNSUPPORTED"]})
    outs, _ = score_config("agent", "calibration", tmp_path, QS, {}, None, judge=False)
    assert outs == []


def test_the_spot_check_sheet_puts_each_label_beside_its_sentence(tmp_path, monkeypatch):
    import analysis.score as S
    record, pid = ac_record()
    write(tmp_path, record, {
        "judge_version": J.JUDGE_VERSION,
        "coverage_delivered": ["PRESENT"] * len(AC["gold_claims"]),
        "grounding_items": [{"i": 0, "sentence": "One claim holds.", "cites": [pid],
                             "label": "SUPPORTED"}]})
    monkeypatch.setattr(S, "RESULTS", tmp_path)
    sheet = spot_check("calibration", tmp_path, ["agent"], QS).read_text()
    assert "| 0 | SUPPORTED | " + pid + " | One claim holds. |" in sheet
    assert AC["gold_claims"][0]["claim"][:40] in sheet
