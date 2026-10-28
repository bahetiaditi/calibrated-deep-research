#!/usr/bin/env python3
"""Read one run's trace as a timeline.

    python -m scripts.inspect_trace                      # latest trace
    python -m scripts.inspect_trace smoke-c27-premise    # latest matching prefix
    python -m scripts.inspect_trace traces/<id>.jsonl    # a specific file

Zero API calls. Written after the C27 live run, where two questions needed
answering from the trace rather than by inference: what the premise checker
actually concluded, and why a re-synthesized draft skipped the critic. C38's
failure annotation will need the same view for every failed run, so it is a
script rather than a one-off.

Prints, in order: node entries, routing decisions with their reason, LLM
calls (role, model, cache hit, fallback), retrievals, the premise check's
per-presupposition verdicts, synthesis notes with draft digests, and every
D1-D5 decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.config import get_config
from src.tracing.tracer import TraceReader

DIM, BOLD, YELLOW, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


def find(arg: str | None) -> Path:
    if arg and Path(arg).is_file():
        return Path(arg)
    directory = get_config().path("run.trace_dir")
    pattern = f"{arg}*.jsonl" if arg else "*.jsonl"
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        sys.exit(f"no trace matching {pattern!r} in {directory}")
    return matches[-1]


def main() -> int:
    path = find(sys.argv[1] if len(sys.argv) > 1 else None)
    reader = TraceReader.load(path)
    print(f"\n{BOLD}{path.name}{RESET}\n")

    llm = {"calls": 0, "cached": 0, "fell_back": 0, "models": {}}
    for event in reader.events:
        kind, name = event.get("type"), event.get("name")
        p = event.get("payload", {})
        t = f"{DIM}{event.get('elapsed_ms', 0) / 1000:7.1f}s{RESET}"

        if kind == "node_enter":
            print(f"{t} {BOLD}[{name}]{RESET}")
        elif kind == "route":
            state = p.get("state") or {}
            extra = ""
            if "draft_reviewed" in state:
                extra = f"  draft_reviewed={state['draft_reviewed']}"
            print(f"{t}   -> {name:<16} {DIM}{p.get('reason', '')}{extra}{RESET}")
        elif kind == "llm_call":
            llm["calls"] += 1
            llm["cached"] += bool(p.get("from_cache"))
            llm["fell_back"] += bool(p.get("fell_back"))
            llm["models"][name] = llm["models"].get(name, 0) + 1
            flags = " ".join(f for f, on in (("CACHE", p.get("from_cache")),
                                             ("FALLBACK", p.get("fell_back"))) if on)
            print(f"{t}      llm  {p.get('role', ''):<10} {name:<24} "
                  f"{p.get('total_tokens', 0):>6} tok  {flags}")
        elif kind == "retrieval":
            print(f"{t}      ret  {name:<14} n={p.get('n_results')}  "
                  f"{DIM}{str(p.get('query', ''))[:70]!r}{RESET}")
        elif kind == "note" and name == "premise_check":
            print(f"{t}      {YELLOW}premise{RESET} failed={p.get('failed')} "
                  f"f7={p.get('grounding_score')} verdicts={p.get('verdicts')}")
            print(f"{t}      {DIM}{p.get('rationale', '')}{RESET}")
        elif kind == "note" and name == "synthesis":
            print(f"{t}      {YELLOW}synthesis{RESET} mode={p.get('mode')} "
                  f"chars={p.get('chars')} cited={p.get('passages_cited')}/"
                  f"{p.get('passages_offered')} digest={p.get('digest', '(not recorded)')}")
        elif kind == "decision":
            adaptive = "" if p.get("was_adaptive") else " (static)"
            print(f"{t}      {YELLOW}{name}{RESET}{adaptive} chosen={p.get('chosen')}  "
                  f"{DIM}{str(p.get('rationale', ''))[:110]}{RESET}")
        elif kind == "error":
            print(f"{t}      {YELLOW}ERROR{RESET} {p.get('error_type')}: {p.get('error')}")

    print(f"\n{BOLD}LLM calls{RESET}  {llm['calls']} total, {llm['cached']} cached, "
          f"{llm['fell_back']} on a fallback model")
    for model, n in sorted(llm["models"].items(), key=lambda kv: -kv[1]):
        print(f"   {model:<28} {n}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
