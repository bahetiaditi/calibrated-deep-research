#!/usr/bin/env python3
"""C12 acceptance check — real NLI contradiction detection.

    python -m scripts.smoke_c12

Zero API calls. Downloads ~280 MB of model weights on first run, then CPU only.

The C12 criterion: "on a hand-built contradicting pair, contradiction is
detected; on an agreeing pair, it is not."

Also verifies the thing a unit test cannot: that this checkpoint's label
ordering is what we think it is. A wrong contradiction index produces
plausible numbers pointing the wrong way, with no error anywhere.
"""
from __future__ import annotations

import sys
import time

from src.config import get_config
from src.rag.contradiction import ContradictionDetector
from src.state import Passage, make_passage_id

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)


def ok(m): print(f"{GREEN}  PASS{RESET} {m}")
def warn(m): print(f"{YELLOW}  WARN{RESET} {m}")
def bad(m): print(f"{RED}  FAIL{RESET} {m}")


def passage(source, text, section="Results"):
    return Passage(
        id=make_passage_id(source, text, section),
        source_type="arxiv", source_id=source, source_domain="arxiv.org",
        title=source, text=text, section=section, published="2024-01-01",
        sub_question_id="sq1", retrieval_score=0.1, rerank_score=6.0,
    )


# Genuine disagreement across sources — the Tier D "no consensus" shape.
CONFLICTING = [
    passage("paperA",
            "We find that a LoRA rank of 8 is sufficient for 7B models; "
            "increasing the rank beyond 8 yields no further gains and wastes "
            "parameters."),
    passage("paperB",
            "Our experiments show that a LoRA rank of 8 is far too small for "
            "7B models; ranks of at least 64 are required to match full "
            "fine-tuning quality."),
]

# Same claim, different wording.
AGREEING = [
    passage("paperC",
            "LoRA substantially reduces the number of trainable parameters by "
            "learning low-rank update matrices while the base model stays frozen."),
    passage("paperD",
            "By freezing the pretrained weights and training only low-rank "
            "adapters, LoRA cuts the count of parameters that must be updated."),
]

# Two chunks of ONE paper that superficially conflict — the artefact that
# cross-source scoping exists to exclude.
SAME_SOURCE = [
    passage("paperE",
            "One might expect that increasing adapter rank always improves "
            "downstream accuracy on small datasets.", section="Introduction"),
    passage("paperE",
            "Contrary to that expectation, higher ranks overfit and accuracy "
            "decreases on our smallest dataset.", section="Results"),
]


def main() -> int:
    print("\n=== C12 smoke test: contradiction detection ===")
    print(f"{DIM}First run downloads ~280 MB of model weights. CPU only.{RESET}\n")

    cfg = get_config()
    detector = ContradictionDetector(cfg)
    failures = 0

    print(f"{DIM}--- label ordering for this checkpoint{RESET}")
    t0 = time.time()
    index = detector.contradiction_index()
    load_s = time.time() - t0
    config = getattr(getattr(detector.model, "model", None), "config", None)
    labels = getattr(config, "id2label", None)
    print(f"{DIM}       model loaded in {load_s:.1f}s{RESET}")
    print(f"{DIM}       id2label: {labels}{RESET}")
    print(f"{DIM}       contradiction index resolved to: {index}{RESET}")
    if labels and any("contradict" in str(v).lower() for v in labels.values()):
        ok("contradiction label resolved from the model's own config")
    else:
        warn("model exposes no usable id2label; using the documented default — "
             "verify the direction of the results below carefully")

    print(f"\n{DIM}--- disagreeing sources (should be detected){RESET}")
    rate, records = detector.contradiction_rate(CONFLICTING)
    if records:
        ok(f"contradiction detected, score {records[0].score:.3f}, f5 rate {rate:.2f}")
        print(f"{DIM}       {records[0].source_a} vs {records[0].source_b} "
              f"(cross_source={records[0].cross_source}){RESET}")
    else:
        bad("genuine cross-source disagreement was NOT detected — Tier D "
            "no-consensus questions will not trigger abstention")
        failures += 1

    print(f"\n{DIM}--- agreeing sources (should NOT be detected){RESET}")
    agree_rate, agree_records = detector.contradiction_rate(AGREEING)
    if not agree_records:
        ok(f"paraphrase correctly not flagged, f5 rate {agree_rate:.2f}")
    else:
        bad(f"false positive on agreeing sources, score "
            f"{agree_records[0].score:.3f} — f5 would inflate on a corpus "
            f"that actually agrees")
        failures += 1

    print(f"\n{DIM}--- same-source artefact (should be excluded from f5){RESET}")
    same_rate, same_records = detector.contradiction_rate(SAME_SOURCE)
    if not same_records:
        ok("same-source pair excluded — a hypothesis refuted later in the "
           "same paper is not sources disagreeing")
    else:
        bad("same-source pair counted toward f5")
        failures += 1

    print(f"\n{DIM}--- discrimination margin{RESET}")
    if records and agree_records:
        margin = records[0].score - agree_records[0].score
        print(f"{DIM}       conflict {records[0].score:.3f} vs "
              f"agreement {agree_records[0].score:.3f}{RESET}")
        if margin > 0.2:
            ok(f"margin {margin:.3f} — threshold "
               f"{detector.threshold} sits comfortably between them")
        else:
            warn(f"margin only {margin:.3f}; the threshold may need tuning "
                 f"before the calibration run at C33")
    elif records:
        ok(f"conflict scored {records[0].score:.3f}, agreement scored below "
           f"threshold entirely")

    print(f"\n{DIM}--- mixed set: does f5 track the amount of disagreement{RESET}")
    mixed = CONFLICTING + AGREEING
    mixed_rate, mixed_records = detector.contradiction_rate(mixed)
    print(f"{DIM}       {len(mixed_records)} contradictions over "
          f"{len(mixed)} passages -> f5 = {mixed_rate:.3f}{RESET}")
    if 0.0 < mixed_rate < 1.0:
        ok("f5 is a graded signal, not a binary flag — usable by the "
           "logistic sufficiency model")
    else:
        warn(f"f5 = {mixed_rate:.2f} on a deliberately mixed set; check the "
             f"threshold")

    print(f"\n{DIM}--- cost{RESET}")
    t0 = time.time()
    detector.detect(mixed)
    elapsed = time.time() - t0
    pairs = detector.max_pairs
    print(f"{DIM}       {elapsed:.2f}s for this scan; budget is {pairs} pairs "
          f"x 2 directions per sub-question{RESET}")
    if elapsed < 20.0:
        ok("acceptable on CPU")
    else:
        warn("slow; consider lowering retrieval.contradiction."
             "max_pairs_per_subquestion")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C12 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())