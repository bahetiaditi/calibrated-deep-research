#!/usr/bin/env python3
"""Print a finished run's final STATE from the checkpoint database.

    python -m scripts.show_run                       # latest run
    python -m scripts.show_run smoke-c27-answerable  # latest matching prefix
    python -m scripts.show_run <thread_id> --full    # drafts untruncated

Zero API calls. The companion to inspect_trace.py: the trace says what
happened, this shows what it produced — the decision record, the withheld
draft, the final draft, every critic finding with its owner and action, and
the gaps. C27.4 needed it because the trace records a synthesis as "cited
0/22" but not the text, and the text was the only way to see which citation
form the model had used. C38's failure annotation needs both views.

The thread id of a run is its trace id (the trace file's stem).
"""
from __future__ import annotations

import sys
import textwrap

from src.config import get_config

BOLD, DIM, YELLOW, RESET = "\033[1m", "\033[2m", "\033[33m", "\033[0m"


def latest_thread(prefix: str | None) -> str:
    trace_dir = get_config().path("run.trace_dir")
    pattern = f"{prefix}*.jsonl" if prefix else "*.jsonl"
    matches = sorted(trace_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        sys.exit(f"no trace matching {pattern!r} in {trace_dir}")
    return matches[-1].stem


def load_state(thread_id: str) -> dict:
    from langgraph.checkpoint.sqlite import SqliteSaver

    db = get_config().path("run.checkpoint_db")
    if not db.is_file():
        sys.exit(f"no checkpoint database at {db}")
    with SqliteSaver.from_conn_string(str(db)) as saver:
        snapshot = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if snapshot is None:
        sys.exit(f"no checkpoint for thread {thread_id!r} — was it run with "
                 f"checkpointing on?")
    return dict(snapshot.checkpoint.get("channel_values") or {})


def block(title: str, text: str, full: bool) -> None:
    print(f"\n{BOLD}{title}{RESET}  {DIM}({len(text or '')} chars){RESET}")
    if not text:
        print(f"{DIM}   (empty){RESET}")
        return
    body = text if full else (text[:1500] + (" …" if len(text) > 1500 else ""))
    print(textwrap.indent(textwrap.fill(body, 96, replace_whitespace=False), "   "))


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    trace_dir = get_config().path("run.trace_dir")
    if args and (trace_dir / f"{args[0]}.jsonl").is_file():
        thread = args[0]                      # an exact thread id
    else:
        thread = latest_thread(args[0] if args else None)
    state = load_state(thread)
    record = state.get("decision") or {}

    print(f"\n{BOLD}{thread}{RESET}\n{state.get('question', '')}")
    print(f"\n{BOLD}decision{RESET}  {state.get('terminal_decision')}  "
          f"rule={record.get('rule')}  score={record.get('score')}  "
          f"scorer={record.get('scorer')}  calibrated={record.get('calibrated')}")
    print(f"{DIM}   {state.get('decision_rationale', '')}{RESET}")
    if record.get("partial_rewrite"):
        print(f"{DIM}   partial rewrite: {record.get('partial_rewrite')}{RESET}")

    block(f"withheld draft ({record.get('withheld_draft_mode') or '—'})",
          record.get("withheld_draft") or "", full)
    block(f"final draft ({state.get('draft_mode') or '—'})",
          state.get("draft") or "", full)

    findings = state.get("critic_findings") or []
    print(f"\n{BOLD}critic findings on the final draft{RESET}  ({len(findings)})")
    for f in findings:
        owner = f.get("sub_question_id") or "—"
        cites = len(f.get("cited_passage_ids") or [])
        print(f"   {YELLOW}{f.get('verdict', ''):<12}{RESET} {owner:<6} "
              f"cites={cites} {f.get('action_taken', ''):<12} "
              f"{DIM}{str(f.get('claim', ''))[:90]}{RESET}")

    for sq in record.get("sub_questions") or []:
        if not sq.get("verdicts"):
            continue
        print(f"{DIM}   at decision time {sq['id']}: {sq['verdicts']}{RESET}")

    gaps = state.get("unresolved_gaps") or []
    print(f"\n{BOLD}gaps{RESET}  ({len(gaps)})")
    for gap in gaps:
        print(f"   {DIM}{gap[:130]}{RESET}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
