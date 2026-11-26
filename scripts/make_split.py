#!/usr/bin/env python3
"""Freeze the calibration/test split (C31).

    python -m scripts.make_split              # freeze, if not already frozen
    python -m scripts.make_split --refreeze   # deliberate re-freeze (loud)

Zero API calls. §6.3: stratified 50/50, 24 calibration (fits the sufficiency
model and tau, C33) and 24 test (reported numbers, C36), "touched once, at
the end". Stratified across tiers AND Tier D sub-types.

Allocation (deterministic; the seed only chooses WHICH items within a
stratum):
  A 5/5, B 5/5, C 4/4 — even.
  D1..D4 have 5 items each, so they alternate 3/2 and 2/3, giving 10/10.

Two rules beyond §6.3:
  - A question tagged `dev_exposed` (B01 — a near-copy of the development
    smoke question) is forced into CALIBRATION. The held-out half must not
    contain anything the system was debugged on.
  - The test half is hashed: SHA-256 over each test question's full content
    plus both splits' membership. scripts/check_questions.py fails if the
    hash no longer matches — editing a test question, or moving a question
    between splits, after the freeze is caught, not merely discouraged.
    Calibration questions stay editable: fixing an error found while
    calibrating is legitimate; changing what the reported numbers rest on
    is not.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "benchmark" / "questions.json"
SEED = 31                     # the commit number; any fixed value would do
CALIBRATION, TEST = "calibration", "test"

# How many of each stratum go to calibration.
CAL_COUNT = {"A": 5, "B": 5, "C": 4, "D1": 3, "D2": 2, "D3": 3, "D4": 2}


def stratum(q) -> str:
    return q["subtype"] if q["tier"] == "D" else q["tier"]


def assign(questions, seed: int = SEED) -> dict[str, str]:
    rng = random.Random(seed)
    by_stratum: dict[str, list] = {}
    for q in sorted(questions, key=lambda q: q["id"]):
        by_stratum.setdefault(stratum(q), []).append(q)

    split: dict[str, str] = {}
    for name, items in sorted(by_stratum.items()):
        forced = [q for q in items if q.get("dev_exposed")]
        free = [q for q in items if not q.get("dev_exposed")]
        rng.shuffle(free)
        need = CAL_COUNT[name] - len(forced)
        if need < 0:
            raise SystemExit(f"stratum {name}: more dev_exposed items than calibration slots")
        for q in forced + free[:need]:
            split[q["id"]] = CALIBRATION
        for q in free[need:]:
            split[q["id"]] = TEST
    return split


def split_hash(questions) -> str:
    """Test questions' full content plus both splits' membership."""
    test = sorted((q for q in questions if q.get("split") == TEST), key=lambda q: q["id"])
    payload = {
        "test": [json.dumps(q, sort_keys=True, ensure_ascii=False) for q in test],
        "calibration_ids": sorted(q["id"] for q in questions if q.get("split") == CALIBRATION),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> int:
    data = json.loads(QUESTIONS.read_text())
    if data.get("split_frozen") and "--refreeze" not in sys.argv:
        print("Split already frozen "
              f"({data['split_frozen']['date']}, hash {data['split_frozen']['hash'][:12]}…).")
        print("Re-freezing after results exist invalidates the held-out claim. "
              "If you mean it: --refreeze, and record why in REFERENCE.md.")
        return 1

    split = assign(data["questions"])
    for q in data["questions"]:
        q["split"] = split[q["id"]]
    data["split_frozen"] = {
        "date": date.today().isoformat(),
        "seed": SEED,
        "method": "stratified by tier and Tier D sub-type; dev_exposed forced to calibration",
        "hash": split_hash(data["questions"]),
    }
    QUESTIONS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    cal = [q for q in data["questions"] if q["split"] == CALIBRATION]
    test = [q for q in data["questions"] if q["split"] == TEST]
    print(f"frozen: {len(cal)} calibration / {len(test)} test, hash "
          f"{data['split_frozen']['hash'][:12]}…")
    for name in sorted(CAL_COUNT):
        c = sum(stratum(q) == name for q in cal)
        t = sum(stratum(q) == name for q in test)
        print(f"  {name:<3} {c} / {t}")
    print("test:", ", ".join(sorted(q["id"] for q in test)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
