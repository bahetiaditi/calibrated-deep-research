#!/usr/bin/env python3
"""Run one configuration over one split (C32).

    python -m benchmark.run_eval --config agent --split calibration
    python -m benchmark.run_eval --config agent --split calibration --limit 2
    python -m benchmark.run_eval --config agent --split test --held-out    # C36, once
    python -m benchmark.run_eval --list

Resumable: re-running the same command skips completed questions, resumes an
interrupted one from its checkpoint, and re-runs system failures. Stops when
a pinned model's day quota runs out; run it again after the reset.
Outputs under results/eval/<config>/<split>/, one JSON per question.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.eval.configs import CONFIGS
from src.eval.runner import EvalRefused, EvalRunner

OUT = Path(__file__).resolve().parents[1] / "results" / "eval"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="agent")
    ap.add_argument("--split", default="calibration", choices=["calibration", "test", "all"])
    ap.add_argument("--limit", type=int, help="attempt at most N questions this invocation")
    ap.add_argument("--only", nargs="*", help="question ids")
    ap.add_argument("--held-out", action="store_true", help="confirm a test-split run (C36)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.list:
        for c in CONFIGS.values():
            flag = "" if c.available else f"   [{c.note}]"
            print(f"  {c.name:<6} {c.isolates}{flag}")
        return 0
    try:
        runner = EvalRunner(args.config, args.split, args.out, held_out=args.held_out)
        s = runner.run(limit=args.limit, only=args.only)
    except EvalRefused as exc:
        print(f"refused: {exc}")
        return 2

    print(f"\n{s.config} / {s.split}: {len(s.done)} done, {len(s.skipped)} skipped "
          f"(already complete), {len(s.resumed)} resumed, {len(s.rerun)} re-run, "
          f"{len(s.failed)} failed")
    if s.failed:
        print(f"  failed: {', '.join(s.failed)} — re-run the same command to retry")
    if s.stopped_on_quota:
        print(f"  stopped at {s.stopped_on_quota}: a pinned model is out of day quota. "
              f"Run the same command after the reset (00:00 UTC).")
    remaining = runner.remaining()
    print(f"  {len(remaining)} question(s) remaining in this split"
          + (f": {', '.join(remaining)}" if 0 < len(remaining) <= 12 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
