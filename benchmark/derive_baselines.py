#!/usr/bin/env python3
"""Derive B2 and B3 from the agent's finished runs (C34).

    python -m benchmark.derive_baselines --split calibration
    python -m benchmark.derive_baselines --split test --held-out      # C36

B2 (always answer): the agent's critiqued answer-mode draft, no LLM call.
B3 (verbalized confidence): one judgment call per question, asking the same
pinned model how confident it is in that draft. Questions where the agent
skipped research on a refuted premise have no draft; B2 is run LIVE for
those (the B2 configuration turns skipping off). Idempotent: existing
derived records are kept. See src/eval/baselines.py for why deriving makes
the comparison paired rather than merely cheaper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_config
from src.eval import baselines as bl
from src.eval.runner import EvalRunner, _write_atomic, eval_config, load_questions

OUT = Path(__file__).resolve().parents[1] / "results" / "eval"


def derive(split: str, out: Path, *, provider=None, live_runner=None,
           base_config=None) -> dict:
    agent_dir = out / "agent" / split
    records = [json.loads(p.read_text()) for p in sorted(agent_dir.glob("*.json"))
               if p.name != "manifest.json"]
    records = [r for r in records if r.get("rule") != "system_failure"]
    cfg = eval_config(base_config or get_config(), "agent", split, out)
    if provider is None:
        from src.llm.provider import LLMProvider
        provider = LLMProvider(cfg)
    db = Path(cfg.get("run.checkpoint_db"))

    summary = {"b2": 0, "b3": 0, "b3_calls": 0, "live_b2": [], "skipped": 0}
    for config in ("B2", "B3"):
        d = out / config / split
        d.mkdir(parents=True, exist_ok=True)
        _write_atomic(d / "manifest.json", {"config": config, "split": split,
                                            "derived_from": str(agent_dir)})
    for record in records:
        qid = record["id"]
        state = None
        if record.get("draft") is None:            # written before C34
            state = bl.load_checkpoint_state(db, record.get("thread_id", ""))
        draft = bl.answer_draft(record, state)

        b2_path = out / "B2" / split / f"{qid}.json"
        if bl.needs_live_b2(record):
            if not b2_path.exists():
                summary["live_b2"].append(qid)
        elif not b2_path.exists():
            _write_atomic(b2_path, bl.b2_record(record, draft))
            summary["b2"] += 1

        b3_path = out / "B3" / split / f"{qid}.json"
        if b3_path.exists():
            summary["skipped"] += 1
            continue
        confidence, reason = None, ""
        if record.get("rule") not in ("premise_refuted", "system_failure", "no_evidence") and draft:
            question = record.get("question") or _question_text(qid)
            confidence, reason = bl.verbalized_confidence(question, draft, provider)
            summary["b3_calls"] += 1
        _write_atomic(b3_path, bl.b3_record(record, draft, confidence, reason))
        summary["b3"] += 1

    if summary["live_b2"] and live_runner is not None:
        live_runner(summary["live_b2"])
    return summary


def _question_text(qid: str) -> str:
    qs, _ = load_questions("all")
    return next(q["question"] for q in qs if q["id"] == qid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calibration", choices=["calibration", "test"])
    ap.add_argument("--held-out", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-live", action="store_true", help="skip the live B2 runs")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.held_out:
        print("refused: the test split is the held-out half; pass --held-out (C36)")
        return 2

    def live(ids):
        print(f"running B2 live for {len(ids)} question(s) with no agent draft: {', '.join(ids)}")
        EvalRunner("B2", args.split, args.out, held_out=args.held_out).run(only=ids)

    s = derive(args.split, args.out, live_runner=None if args.no_live else live)
    print(f"B2 derived: {s['b2']}  B3 derived: {s['b3']} ({s['b3_calls']} confidence calls)  "
          f"already present: {s['skipped']}")
    if s["live_b2"]:
        print(f"B2 live: {', '.join(s['live_b2'])}" + (" (skipped: --no-live)" if args.no_live else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
