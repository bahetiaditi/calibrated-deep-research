#!/usr/bin/env python3
"""C37.3 live check: the critic on the two claims it got wrong.

    python -m scripts.smoke_c37_3

One critic call on gpt-oss-120b (the eval pin), development cache. The draft
is the two sentences from the calibration runs, citing the same frozen-corpus
passages:

  A06  "…eight parallel attention heads…" [Pb6eeca1bdcd7]
       before C37.3: UNSUPPORTED, "describes the authors' model, not the
       original base Transformer" — the critic was not shown the title
  A05  "…trained for 3.5 days on eight GPUs" [P25b038f40d30]
       before C37.3: UNSUPPORTED — the support sits at character 872 and the
       critic was shown 800

Prints what the critic was shown for each passage and its verdicts. Both
should now be SUPPORTED. A model judgment can still differ run to run; a
WARN here is a finding to read, not a test failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import get_config
from src.eval.configs import EVAL_PINS
from src.nodes.critic import Critic
from src.state import new_state, new_sub_question

ROOT = Path(__file__).resolve().parents[1]
CORPUS = {p["id"]: p for p in json.loads(
    (ROOT / "benchmark" / "retrieval_corpus.json").read_text())["passages"]}
CASES = [
    ("A06", "Pb6eeca1bdcd7",
     "The original \u201cbase\u201d Transformer architecture employs eight parallel "
     "attention heads in each multi\u2011head attention layer"),
    ("A05", "P25b038f40d30",
     "For the English\u2011to\u2011French experiment the paper states that the model "
     "was trained for 3.5 days on eight GPUs"),
]
GREEN, YELLOW, BOLD, DIM, RESET = "\033[32m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"


def main() -> int:
    cfg = get_config().with_overrides(dict(EVAL_PINS))
    critic = Critic(cfg)
    state = new_state("smoke C37.3")
    sq = new_sub_question("sq", sub_question_id="sq1")
    sq["evidence_ids"] = [pid for _, pid, _ in CASES]
    state["plan"] = [sq]
    state["evidence"] = [dict(CORPUS[pid], sub_question_id="sq1") for _, pid, _ in CASES]
    state["draft"] = " ".join(f"{claim} [{pid}]." for _, pid, claim in CASES)

    for qid, pid, _ in CASES:
        shown = critic._render_passage(pid, CORPUS[pid], False)
        print(f"{BOLD}{qid} — what the critic is shown{RESET}\n{DIM}{shown}{RESET}\n")

    report = critic.review(state)
    if report.error:
        print(f"{YELLOW}critic error: {report.error}{RESET}")
        return 1
    passed = 0
    for (qid, pid, _), f in zip(CASES, report.findings):
        good = f["verdict"] == "SUPPORTED"
        passed += good
        tag = f"{GREEN}  PASS{RESET}" if good else f"{YELLOW}  WARN{RESET}"
        print(f"{tag} {qid}: {f['verdict']} — {f['reasoning']}")
    print(f"\n{passed} of {len(CASES)} now SUPPORTED (both were UNSUPPORTED before C37.3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
