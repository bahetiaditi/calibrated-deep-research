#!/usr/bin/env python3
"""Judge every result and compute the metrics table (C37).

    python -m analysis.score --split calibration
    python -m analysis.score --split calibration --configs agent B2 B3
    python -m analysis.score --split calibration --no-judge     # metrics from cached judgments
    python -m analysis.score --split calibration --spot-check   # 20% sheet for a human

Judgments are cached per question under <config>/<split>/judged/, so a
re-run costs nothing and an interrupted one resumes. The judge is Gemini
flash-lite (its own quota, separate from the runs). Writes
results/metrics_<split>.md and .json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from src.config import get_config
from src.eval import judge as J
from src.eval import metrics as M
from src.eval.baselines import answer_draft, load_checkpoint_state
from src.eval.runner import _write_atomic, eval_config, load_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eval"
RESULTS = ROOT / "results"
CORPUS = ROOT / "benchmark" / "retrieval_corpus.json"
ORDER = ["agent", "B0", "B1", "B2", "B3", "A1", "A2", "A3", "A4", "A5"]
SPOT_FRACTION, SPOT_SEED = 0.2, 37


def load_records(out: Path, config: str, split: str) -> list[dict]:
    d = out / config / split
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))
            if p.name != "manifest.json"] if d.exists() else []


def apply_tau(records: list[dict], tau_path: Path) -> bool:
    """B3's decisions wait for its threshold, chosen at C33."""
    pending = [r for r in records if r.get("rule") == "threshold_pending"]
    if not pending or not tau_path.exists():
        return not pending
    tau = json.loads(tau_path.read_text())["tau"]
    for r in pending:
        r["terminal_decision"] = "ANSWER" if (r.get("score") or 0) >= tau else "ABSTAIN"
        r["rule"] = "threshold"
    return True


def judge_record(record: dict, question: dict, provider, corpus: dict, db: Path) -> dict:
    out: dict = {}
    decision = record.get("terminal_decision")
    delivered = record.get("final_report") or record.get("draft") or ""
    state = None
    if record.get("draft") is None and record.get("thread_id"):
        state = load_checkpoint_state(db, record["thread_id"])
    answer = answer_draft(record, state)

    if question["tier"] == "D":
        if record.get("rule") != "system_failure" and delivered:
            out["tier_d"] = J.tier_d(provider, question, delivered)
    else:
        gold = [c["claim"] for c in question["gold_claims"]]
        if decision == "ANSWER" and delivered:
            labels = J.coverage(provider, question["question"], gold, delivered)
            out["coverage_delivered"] = out["coverage_answer"] = labels
        else:
            if decision == "PARTIAL" and delivered:
                out["coverage_delivered"] = J.coverage(provider, question["question"], gold, delivered)
            if answer:
                out["coverage_answer"] = J.coverage(provider, question["question"], gold, answer)

    raw = record.get("draft") if record.get("draft") is not None else (state or {}).get("draft", "")
    if decision in M.ANSWERED and raw:
        out["grounding"] = J.grounding(provider, raw, corpus)
    return out


def score_config(config: str, split: str, out: Path, questions: dict, corpus: dict,
                 provider, *, judge: bool) -> tuple[list[M.Outcome], bool]:
    records = load_records(out, config, split)
    decided = apply_tau(records, out / "B3" / "tau.json") if config == "B3" else True
    jdir = out / config / split / "judged"
    jdir.mkdir(parents=True, exist_ok=True)
    db = out / "checkpoints.sqlite"
    outcomes = []
    for r in records:
        path = jdir / f"{r['id']}.json"
        if path.exists():
            judged = json.loads(path.read_text())
        elif judge:
            judged = judge_record(r, questions[r["id"]], provider, corpus, db)
            _write_atomic(path, judged)
        else:
            continue
        outcomes.append(M.grade(r, judged))
    return outcomes, decided


COLUMNS = [("n", "n"), ("failures", "fail"), ("ac_accuracy_answered", "A–C acc (answered)"),
           ("ac_false_abstention", "A–C false abstain"), ("ac_mean_coverage", "A–C coverage"),
           ("d_correct", "D correct"), ("d_false_answer", "D false answer"),
           ("d_wrong_reason", "D wrong reason"), ("hallucination", "halluc."),
           ("aurc", "AURC"), ("acc_at_80_coverage", "acc@80%"), ("ece", "ECE"),
           ("mean_llm_calls", "LLM calls")]


def table(rows: dict[str, dict], notes: dict[str, str]) -> str:
    head = "| config | " + " | ".join(c for _, c in COLUMNS) + " |"
    sep = "|---" * (len(COLUMNS) + 1) + "|"
    lines = [head, sep]
    for name, s in rows.items():
        lines.append(f"| {name}{notes.get(name, '')} | " +
                     " | ".join(M.fmt(s.get(k)) for k, _ in COLUMNS) + " |")
    return "\n".join(lines)


def spot_check(split: str, out: Path, configs: list[str], questions: dict) -> Path:
    items = []
    for c in configs:
        jdir = out / c / split / "judged"
        for p in sorted(jdir.glob("*.json")) if jdir.exists() else []:
            items.append((c, p.stem, json.loads(p.read_text())))
    rng = random.Random(SPOT_SEED)
    sample = rng.sample(items, max(1, round(len(items) * SPOT_FRACTION))) if items else []
    lines = [f"# Judge spot-check — {split} ({len(sample)} of {len(items)} judgments, seed {SPOT_SEED})",
             "", "For each: read the output, then mark whether you agree with every label.",
             "Agreement rate = agreed / total; report it with the metrics (§6.2).", ""]
    for c, qid, j in sample:
        rec = json.loads((out / c / split / f"{qid}.json").read_text())
        q = questions[qid]
        lines += [f"## {c} · {qid} — {q['question']}", "",
                  f"- decision: `{rec.get('terminal_decision')}` ({rec.get('rule')})",
                  f"- judge: `{json.dumps(j)[:600]}`", "",
                  "<details><summary>output</summary>", "",
                  (rec.get("final_report") or rec.get("draft") or "")[:3000], "", "</details>", "",
                  "- [ ] agree with all labels", ""]
    path = RESULTS / f"spot_check_{split}.md"
    path.write_text("\n".join(lines))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calibration", choices=["calibration", "test"])
    ap.add_argument("--configs", nargs="*", default=ORDER)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--spot-check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    all_q, _ = load_questions("all")
    questions = {q["id"]: q for q in all_q}
    corpus = {p["id"]: p for p in json.loads(CORPUS.read_text())["passages"]}
    provider = None
    if not args.no_judge:
        from src.llm.provider import LLMProvider
        provider = LLMProvider(eval_config(get_config(), "agent", args.split, args.out))

    rows, notes, present = {}, {}, []
    for c in args.configs:
        outcomes, decided = score_config(c, args.split, args.out, questions, corpus,
                                         provider, judge=not args.no_judge)
        if not outcomes:
            continue
        present.append(c)
        rows[c] = M.summarise(outcomes)
        if not decided:
            notes[c] = " ¹"
            for k in ("ac_accuracy_answered", "ac_false_abstention", "ac_mean_coverage",
                      "d_correct", "d_false_answer", "d_wrong_reason"):
                rows[c][k] = None

    md = [f"# Metrics — {args.split} split", "", table(rows, notes), ""]
    if notes:
        md.append("¹ decisions pending: B3's threshold is chosen at C33; score metrics only.")
    md.append("\nRules: REFERENCE.md, C37 entry. Judge: gemini-3.1-flash-lite (JUDGE role).")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"metrics_{args.split}.md").write_text("\n".join(md) + "\n")
    (RESULTS / f"metrics_{args.split}.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(md))
    if args.spot_check:
        print(f"\nspot-check sheet: {spot_check(args.split, args.out, present, questions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
