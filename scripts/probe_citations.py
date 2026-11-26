#!/usr/bin/env python3
"""Re-read every eval record's text with the current citation normaliser (C37.1).

    python -m scripts.probe_citations
    python -m scripts.probe_citations --config agent --split calibration

Zero API calls. For each record it counts label-shaped tokens (`P09`,
`【P09†L1-L3】`, ...) left in the delivered report and the raw draft, before
and after `normalize_citations`. A record whose text still holds label-shaped
tokens AFTER normalisation contains a citation form the synthesizer does not
understand — the class of defect C27.4 and C37.1 each fixed one instance of.
Records with `synthesis_log` (C37.1+) also show what the run itself counted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.nodes.synthesizer import _LEFTOVER_RE, normalize_citations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eval"
# Footnote numbers like [1] are the finalizer's, not the model's.
_FORMS_RE = re.compile(r"[\[【(][^\]】)]{0,40}P\d{1,3}[^\]】)]{0,40}[\]】)]")


def label_tokens(text: str) -> int:
    return len(_LEFTOVER_RE.findall(text or ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agent")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    d = args.out / args.config / args.split
    paths = sorted(p for p in d.glob("*.json") if p.name != "manifest.json")
    if not paths:
        print(f"no records under {d}")
        return 1
    print(f"{'id':6} {'field':12} {'raw':>4} {'after':>5}  forms seen")
    unresolved = 0
    for p in paths:
        r = json.loads(p.read_text())
        for field in ("final_report", "draft"):
            text = r.get(field) or ""
            if not text:
                continue
            raw = label_tokens(text)
            after = label_tokens(normalize_citations(text)[0])
            # Resolved citations become [P01]-style labels, which the leftover
            # pattern also matches; count only forms outside that shape.
            after -= len(re.findall(r"\[P\d{2,3}\]", normalize_citations(text)[0]))
            forms = sorted(set(_FORMS_RE.findall(text)))[:3]
            if raw:
                print(f"{r['id']:6} {field:12} {raw:>4} {max(after, 0):>5}  {forms}")
            unresolved += max(after, 0)
        for i, e in enumerate(r.get("synthesis_log") or []):
            if e.get("unparsed_citations"):
                print(f"{r['id']:6} synthesis#{i}  run counted {e['unparsed_citations']} unparsed")
    print(f"\nlabel-shaped tokens the current normaliser still cannot read: {unresolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
