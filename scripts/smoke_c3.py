#!/usr/bin/env python3
"""C3 acceptance check — run once, manually, with real API keys.

    python -m scripts.smoke_c3

Costs ONE live LLM call. The whole point is that the second and third
identical calls cost nothing at all.
"""
from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")

from src.config import get_config  # noqa: E402
from src.llm.provider import LLMProvider, Role  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def ok(m: str) -> None:
    print(f"{GREEN}  PASS{RESET} {m}")


def bad(m: str) -> None:
    print(f"{RED}  FAIL{RESET} {m}")


def main() -> int:
    print("\n=== C3 smoke test: prompt-hash cache ===\n")
    cfg = get_config()
    provider = LLMProvider(cfg)
    failures = 0

    if not provider.cache.enabled:
        bad("cache is disabled in config.yaml (llm.cache.enabled)")
        return 1

    # A unique prompt so the check is meaningful on a warm cache.
    marker = f"run-{int(time.time())}"
    system = "You are terse. Answer in at most eight words."
    user = f"[{marker}] Name one advantage of hybrid retrieval over dense-only."

    print(f"{DIM}--- call 1: expected live{RESET}")
    before = provider.ledger.snapshot()
    t0 = time.time()
    first = provider.complete(system, user, role=Role.JUDGMENT)
    live_s = time.time() - t0
    if first.from_cache:
        bad("first call was served from cache — marker collision?")
        return 1
    ok(f"live via {first.model}: {first.text.strip()[:52]!r}")
    print(f"{DIM}       {first.total_tokens} tokens, {live_s:.2f}s{RESET}")

    spent = provider.ledger.snapshot()[first.model]["tpd"][0]

    print(f"\n{DIM}--- call 2: identical prompt, expected cached{RESET}")
    t0 = time.time()
    second = provider.complete(system, user, role=Role.JUDGMENT)
    cached_s = time.time() - t0

    if not second.from_cache:
        bad("second identical call was NOT cached")
        failures += 1
    elif second.text != first.text:
        bad("cached text differs from the original response")
        failures += 1
    else:
        speedup = live_s / cached_s if cached_s > 0 else float("inf")
        ok(f"served from cache, byte-identical ({cached_s * 1000:.1f}ms, ~{speedup:.0f}x faster)")

    now_spent = provider.ledger.snapshot()[first.model]["tpd"][0]
    if now_spent != spent:
        bad(f"cache hit charged quota: {spent} -> {now_spent}")
        failures += 1
    else:
        ok(f"no quota charged for the hit (still {now_spent:,} tokens today)")

    print(f"\n{DIM}--- call 3: use_cache=False, expected live (C37 blind judge){RESET}")
    third = provider.complete(system, user, role=Role.JUDGMENT, use_cache=False)
    if third.from_cache:
        bad("use_cache=False still returned a cached response")
        failures += 1
    else:
        ok(f"bypassed cache as requested ({third.total_tokens} tokens)")

    print(f"\n{DIM}--- call 4: changed prompt, expected miss{RESET}")
    fourth = provider.complete(system, user + " Answer in French.", role=Role.JUDGMENT)
    if fourth.from_cache:
        bad("a changed prompt was served from cache — key is too weak")
        failures += 1
    else:
        ok("changed prompt correctly missed")

    print(f"\n{DIM}--- cache{RESET}")
    print(f"  {provider.cache.report()}")
    print(f"\n{DIM}--- quota{RESET}")
    print(provider.quota_report())

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C3 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())