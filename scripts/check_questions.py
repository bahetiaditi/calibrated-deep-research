#!/usr/bin/env python3
"""Validate benchmark/questions.json against the frozen corpus; write a review sheet.

    python -m scripts.check_questions            # validate + write the review sheet
    python -m scripts.check_questions --strict   # also fail on warnings

Zero API calls. Written at C29, whose check is "every question is answerable
from accessible sources — verify by hand before committing". The script
cannot verify a claim; it makes verifying one fast. For every gold claim it
prints the sentence from each cited passage that shares the most words with
the claim, into benchmark/review_questions.md, where a reviewer ticks or
rejects each claim.

Structural rules it enforces (REFERENCE §6.1–6.2):
  - tier counts: A=10, B=10, C=8 (Tier D arrives at C30)
  - 3-5 gold claims per question, each citing at least one corpus passage
  - every cited passage id exists in the frozen corpus
  - Tiers B and C cite at least two distinct sources: B is "multi-source
    synthesis", C "judgment across many sources"
And one warning: a question whose wording heavily overlaps one of the 30
retrieval sub-questions (benchmark/retrieval_questions.json). Topic overlap
is unavoidable — the corpus was built around those sub-questions — but a
near-copy would measure answer quality on the items that tuned retrieval.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "benchmark" / "questions.json"
CORPUS = ROOT / "benchmark" / "retrieval_corpus.json"
RETRIEVAL_QS = ROOT / "benchmark" / "retrieval_questions.json"
SHEET = ROOT / "benchmark" / "review_questions.md"

EXPECTED = {"A": 10, "B": 10, "C": 8}
MIN_SOURCES = {"B": 2, "C": 2}
OVERLAP_WARN = 0.5
STOP = set("the a an of to in and or for on with what how does do is are by as at from "
           "that this it its be their they which when why vs".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
            if w not in STOP and len(w) > 1}


def load():
    data = json.loads(QUESTIONS.read_text())
    corpus = {p["id"]: p for p in json.loads(CORPUS.read_text())["passages"]}
    rq = json.loads(RETRIEVAL_QS.read_text())
    items = rq.get("questions", rq) if isinstance(rq, dict) else rq
    texts = []
    for it in (items.values() if isinstance(items, dict) else items):
        if isinstance(it, dict):
            texts.append(it.get("text") or it.get("question") or "")
        elif isinstance(it, str):
            texts.append(it)
    return data, corpus, [t for t in texts if t]


def validate(data, corpus, retrieval_questions=()) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    questions = data.get("questions") or []
    ids = [q.get("id") for q in questions]
    for qid, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate id {qid}")
    tiers = Counter(q.get("tier") for q in questions)
    for tier, n in EXPECTED.items():
        if tiers.get(tier, 0) != n:
            errors.append(f"tier {tier}: {tiers.get(tier, 0)} questions, expected {n}")

    for q in questions:
        qid, tier = q.get("id"), q.get("tier")
        claims = q.get("gold_claims") or []
        if not 3 <= len(claims) <= 5:
            errors.append(f"{qid}: {len(claims)} gold claims (need 3-5)")
        sources = set()
        for i, claim in enumerate(claims):
            pids = claim.get("passage_ids") or []
            if not (claim.get("claim") or "").strip():
                errors.append(f"{qid} claim {i}: empty")
            if not pids:
                errors.append(f"{qid} claim {i}: cites no passage")
            for pid in pids:
                if pid not in corpus:
                    errors.append(f"{qid} claim {i}: passage {pid} not in corpus")
                else:
                    sources.add(_source(corpus[pid]))
        need = MIN_SOURCES.get(tier)
        if need and len(sources) < need:
            errors.append(f"{qid}: tier {tier} cites {len(sources)} source(s), needs {need}")
        qw = words(q.get("question", ""))
        for rq in retrieval_questions:
            rw = words(rq)
            if qw and rw and len(qw & rw) / len(qw | rw) >= OVERLAP_WARN:
                warnings.append(f"{qid} overlaps retrieval sub-question: {rq!r}")
    return errors, warnings


def _source(passage) -> str:
    sid = passage.get("source_id") or ""
    m = re.search(r"(\d{4}\.\d{4,5})", sid)        # arXiv id, any version or URL form
    return m.group(1) if m else re.sub(r"^https?://(www\.)?", "", sid).split("/")[0]


def best_sentence(claim: str, text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    cw = words(claim)
    return max(sentences, key=lambda s: len(cw & words(s)), default="")[:400]


def review_sheet(data, corpus) -> str:
    review = data.get("review") or {}
    status = (f"**Status: {review['status']} ({review.get('date', '')}).** "
              f"{review.get('method', '')}. Corrected: {'; '.join(review.get('corrected', []))}."
              if review else "**Status: not yet reviewed.**")
    out = ["# Gold-claim review (C29)", "", status, "",
           "For each claim, check the quoted sentence actually supports it. "
           "Mark `[x]` to accept; edit or delete the claim in "
           "`benchmark/questions.json` otherwise, then re-run the checker.", ""]
    for q in data["questions"]:
        out += [f"## {q['id']} (Tier {q['tier']}) — {q['question']}", ""]
        if q.get("notes"):
            out += [f"_{q['notes']}_", ""]
        for i, claim in enumerate(q["gold_claims"], 1):
            out.append(f"- [ ] **{i}.** {claim['claim']}")
            for pid in claim["passage_ids"]:
                p = corpus.get(pid)
                if p:
                    out.append(f"    - `{pid}` {_source(p)} ({p.get('section') or '-'}): "
                               f"\"{best_sentence(claim['claim'], p['text'])}\"")
            out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    data, corpus, rqs = load()
    errors, warnings = validate(data, corpus, rqs)
    for e in errors:
        print(f"ERROR   {e}")
    for w in warnings:
        print(f"WARN    {w}")
    SHEET.write_text(review_sheet(data, corpus))
    n = len(data["questions"])
    claims = sum(len(q["gold_claims"]) for q in data["questions"])
    print(f"\n{n} questions, {claims} gold claims; {len(errors)} error(s), "
          f"{len(warnings)} warning(s). Review sheet: {SHEET.relative_to(ROOT)}")
    if errors or ("--strict" in sys.argv and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
