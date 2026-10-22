#!/usr/bin/env python3
"""Inspect and repair the local quota ledger.

    python -m scripts.quota status
    python -m scripts.quota reset --model openai/gpt-oss-120b
    python -m scripts.quota reset --all

Why a repair tool exists at all
-------------------------------
The ledger is a *local prediction* of what the providers will allow, not a
record they keep. That makes it useful — it refuses a call before the quota
is spent discovering a 429 — and it makes it capable of being wrong in a way
the provider cannot correct.

A C20 bug wrote each model's entire daily cap into the ledger on a JSON
formatting failure, so three models read as fully spent while almost nothing
had actually been consumed. Nothing errored; the symptom was simply that work
stopped. A ledger that can be wrong in the conservative direction needs a way
to be inspected and reset, or the only remedy is waiting out a day that never
needed to pass.

Reset is safe in the sense that matters: if the real quota *is* spent, the
provider returns 429 and the provider layer handles it exactly as designed —
record the exhaustion and move to the next model. The cost of resetting
wrongly is one wasted request; the cost of not resetting is a lost day.
"""
from __future__ import annotations

import argparse
import sys

from src.config import get_config
from src.llm.provider import LLMProvider

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def _bar(used: int, cap: int | None, width: int = 24) -> str:
    if not cap:
        return f"{DIM}{'·' * width}{RESET}"
    fraction = min(1.0, used / cap)
    filled = int(fraction * width)
    colour = RED if fraction > 0.9 else YELLOW if fraction > 0.6 else GREEN
    return f"{colour}{'█' * filled}{DIM}{'·' * (width - filled)}{RESET}"


def cmd_status(_args) -> int:
    cfg = get_config()
    provider = LLMProvider(cfg)
    snapshot = provider.ledger.snapshot()

    print(f"\n{BOLD}Quota ledger{RESET}")
    print(f"{DIM}{provider.ledger.path}{RESET}\n")

    suspicious: list[str] = []
    for model, usage in snapshot.items():
        print(f"{BOLD}{model}{RESET}")
        for dimension in ("rpm", "tpm", "rpd", "tpd"):
            used, cap = usage[dimension]
            cap_text = str(cap) if cap is not None else "—"
            print(f"   {dimension:<5}{_bar(used, cap)} "
                  f"{used:>7,}/{cap_text:<9}")
        used, cap = usage["tpd"]
        if cap and used >= cap:
            suspicious.append(model)
        print()

    if suspicious:
        print(f"{YELLOW}{len(suspicious)} model(s) read as fully spent on "
              f"tokens today.{RESET}")
        print(f"{DIM}If that does not match the provider console, the ledger "
              f"is over-counting.\nCheck console.groq.com → Settings → "
              f"Limits, then:\n    python -m scripts.quota reset --all{RESET}")
        print(f"{DIM}Resetting wrongly costs one rejected request; the "
              f"provider records the\n429 and moves on. Not resetting costs "
              f"a day.{RESET}\n")
    else:
        print(f"{GREEN}No model is at its daily ceiling.{RESET}\n")
    return 0


def cmd_reset(args) -> int:
    cfg = get_config()
    provider = LLMProvider(cfg)
    ledger = provider.ledger

    models = list(ledger.limits) if args.all else [args.model]
    if not args.all and not args.model:
        print(f"{RED}pass --model NAME or --all{RESET}")
        return 1
    unknown = [m for m in models if m not in ledger.limits]
    if unknown:
        print(f"{RED}unknown model(s): {unknown}{RESET}")
        print(f"{DIM}known: {sorted(ledger.limits)}{RESET}")
        return 1

    with ledger._lock:  # noqa: SLF001 — deliberate: this is the repair path
        for model in models:
            ledger._states.pop(model, None)  # noqa: SLF001
        ledger._save()  # noqa: SLF001

    print(f"\n{GREEN}reset {len(models)} model(s){RESET}")
    for model in models:
        print(f"{DIM}   {model}{RESET}")
    print(f"\n{DIM}The ledger now believes nothing has been spent today. If "
          f"the real quota is\ngone, the next call returns 429 and the "
          f"provider records it properly.{RESET}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show the ledger").set_defaults(func=cmd_status)

    reset = sub.add_parser("reset", help="clear recorded usage")
    reset.add_argument("--model", help="one model name")
    reset.add_argument("--all", action="store_true", help="every model")
    reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())