#!/usr/bin/env python3
"""C7 acceptance check — live web search.

    python -m scripts.smoke_c7

Costs zero LLM tokens and ~2 Tavily credits (of 1000/month). ddgs is free but
rate-limits hard, so this deliberately makes few requests.

The C7 check is "both return valid passages; DDG rate-limit exception is
caught and falls back".
"""
from __future__ import annotations

import sys

from src.config import get_config
from src.state import Passage
from src.tools.web_search import (
    DuckDuckGoSearchProvider,
    TavilySearchProvider,
    WebSearchTool,
    get_search_provider,
    make_search_ledger,
)

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
)

QUERY = "reciprocal rank fusion retrieval evaluation"
REQUIRED = set(Passage.__annotations__)


def ok(m: str) -> None:
    print(f"{GREEN}  PASS{RESET} {m}")


def warn(m: str) -> None:
    print(f"{YELLOW}  WARN{RESET} {m}")


def bad(m: str) -> None:
    print(f"{RED}  FAIL{RESET} {m}")


def check(passages: list[dict]) -> list[str]:
    problems = []
    for p in passages:
        missing = REQUIRED - set(p)
        if missing:
            problems.append(f"missing {sorted(missing)}")
        if not p.get("text") or not p.get("source_id"):
            problems.append("empty text or url")
        if p.get("source_type") != "web":
            problems.append(f"source_type={p.get('source_type')!r}")
    return problems


def show(passages: list[dict]) -> None:
    for p in passages[:2]:
        print(f"{DIM}       {p['source_domain']:24s} {len(p['text']):>5d} chars  "
              f"{p['title'][:44]}{RESET}")


def main() -> int:
    print("\n=== C7 smoke test: web search ===\n")
    cfg = get_config()
    failures = 0

    print(f"{DIM}--- provider: tavily (pre-extracted content, costs credits){RESET}")
    tavily = TavilySearchProvider(cfg, ledger=make_search_ledger(cfg))
    if not tavily.is_available():
        warn("TAVILY_API_KEY not set — skipping")
    else:
        try:
            passages = tavily.search(QUERY, max_results=3, sub_question_id="sq1")
            problems = check(passages)
            if not passages:
                bad("tavily returned nothing")
                failures += 1
            elif problems:
                bad(f"schema problems: {problems[:3]}")
                failures += 1
            else:
                ok(f"{len(passages)} passages")
                show(passages)
                avg = sum(len(p["text"]) for p in passages) / len(passages)
                if avg > 300:
                    ok(f"content is substantive (avg {avg:.0f} chars) — citable")
                else:
                    warn(f"avg only {avg:.0f} chars; snippets are hard to verify")
        except Exception as exc:  # noqa: BLE001
            bad(f"tavily: {type(exc).__name__}: {exc}")
            failures += 1

    print(f"\n{DIM}--- provider: ddgs (free, snippets, rate-limits hard){RESET}")
    ddg = DuckDuckGoSearchProvider(cfg)
    try:
        passages = ddg.search(QUERY, max_results=3, sub_question_id="sq1")
        if passages and not check(passages):
            ok(f"{len(passages)} passages")
            show(passages)
        elif not passages:
            warn("ddgs returned nothing (rate limited or blocked) — "
                 "the chain covers this, see fallback below")
        else:
            bad(f"schema problems: {check(passages)[:3]}")
            failures += 1
    except Exception as exc:  # noqa: BLE001
        warn(f"ddgs raised {type(exc).__name__} — the chain must absorb this")

    print(f"\n{DIM}--- fallback: broken primary must not fail the search{RESET}")

    class BrokenProvider(DuckDuckGoSearchProvider):
        name = "broken"

        def search(self, query, *, max_results, sub_question_id=""):
            from ddgs.exceptions import RatelimitException
            raise RatelimitException("simulated 429")

    chain = WebSearchTool(
        [BrokenProvider(cfg), tavily if tavily.is_available() else ddg],
        config=cfg,
    )
    passages = chain.search(QUERY, max_results=2)
    if passages:
        ok(f"rate-limited primary absorbed; fell through and got "
           f"{len(passages)} passages")
    else:
        bad("fallback produced nothing")
        failures += 1

    print(f"\n{DIM}--- whole chain down returns [] rather than raising{RESET}")
    dead = WebSearchTool([BrokenProvider(cfg)], config=cfg)
    try:
        if dead.search(QUERY) == []:
            ok("returns [] — 'nothing found' is information D2/D3 act on")
        else:
            bad("expected empty result")
            failures += 1
    except Exception as exc:  # noqa: BLE001
        bad(f"raised into the caller: {type(exc).__name__}: {exc}")
        failures += 1

    print(f"\n{DIM}--- factory + credit ledger{RESET}")
    tool = get_search_provider(cfg)
    ok(f"chain order: {[p.name for p in tool.providers]}")
    led = make_search_ledger(cfg)
    snap = led.snapshot()["tavily"]
    used, cap = snap["rpd"]
    ok(f"tavily credits today: {used}/{cap} "
       f"(monthly allowance {cfg.get('sources.web.tavily_credits_per_month')})")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C7 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())