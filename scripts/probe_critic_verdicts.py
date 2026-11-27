#!/usr/bin/env python3
"""Which eval records had critic sentences left unjudged (C37.4)?

    python -m scripts.probe_critic_verdicts
    python -m scripts.probe_critic_verdicts --config agent --split calibration

Zero API calls. Two sources, because the committed record cannot show
everything:

  record   final critic findings whose reasoning says "no verdict returned".
           Visible for CITED sentences only in records written before C37.4 —
           an uncited sentence's reasoning was overwritten with "no citation".
  trace    every critic LLM call of the record's attempt (local traces, which
           are gitignored), with its completion tokens. A call at or near the
           judgment role's output cap was cut off, and every sentence after
           the cut went unjudged, whether cited or not.

A record needs re-running under C37.4 only if either source shows a gap: the
fix is a no-op when every verdict came back.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eval"
NO_VERDICT = "no verdict returned"
NEAR_CAP = 0.97


def output_cap() -> int:
    cfg = get_config()
    chain = cfg.get("llm.roles.judgment") or []
    return int(chain[0].get("max_output_tokens", 2048)) if chain else 2048


def trace_for(record: dict, traces: Path) -> Path | None:
    p = Path(record.get("trace_path") or "")
    if p.is_file():
        return p
    # Absolute paths differ between machines; fall back to the id pattern.
    pattern = f"{record['config']}-{record['split']}-{record['id']}-a{record.get('attempt', 0)}-*.jsonl"
    matches = sorted(traces.glob(pattern))
    return matches[-1] if matches else None


def critic_calls(path: Path) -> list[dict]:
    calls = []
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = e.get("payload") or {}
        if e.get("type") == "llm_call" and payload.get("node") == "critic":
            calls.append(payload)
    return calls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agent")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    d = args.out / args.config / args.split
    traces = args.out / "traces"
    cap = output_cap()
    rerun = []
    print(f"judgment output cap: {cap} tokens; flagged at >= {int(cap * NEAR_CAP)}\n")
    print(f"{'id':6} {'att':>3}  {'no-verdict':>10}  critic calls (completion tokens)")
    for p in sorted(x for x in d.glob("*.json") if x.name != "manifest.json"):
        r = json.loads(p.read_text())
        missing = sum(NO_VERDICT in (f.get("reasoning") or "")
                      for f in r.get("critic_findings") or [])
        tpath = trace_for(r, traces)
        if tpath is None:
            calls_txt, capped = "(no local trace)", False
        else:
            calls = critic_calls(tpath)
            toks = [int(c.get("completion_tokens") or 0) for c in calls]
            capped = any(t >= cap * NEAR_CAP for t in toks)
            calls_txt = ", ".join(f"{t}{' CAPPED' if t >= cap * NEAR_CAP else ''}"
                                  f"{' (cache)' if c.get('from_cache') else ''}"
                                  for t, c in zip(toks, calls)) or "(none)"
        flag = missing > 0 or capped
        if flag:
            rerun.append(r["id"])
        print(f"{r['id']:6} {r.get('attempt', 0):>3}  {missing:>10}  {calls_txt}"
              f"{'   <- affected' if flag else ''}")
    print(f"\naffected by C37.4 (re-run these): {' '.join(rerun) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
