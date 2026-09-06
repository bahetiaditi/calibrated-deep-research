#!/usr/bin/env python3
"""C2 acceptance check — run once, manually, with real API keys.

    python -m scripts.smoke_c2

Costs roughly 4 LLM calls: two Groq (judgment + mechanical), one Gemini
judge, one structured-output call. That is a deliberate, small spend against
a tight daily budget — it verifies the whole layer end to end and is the only
place in the project that calls the real APIs before Phase 1.

The unit tests already prove retry, fallback, and quota logic with fakes.
This proves the SDK wiring, model names, and credentials are actually right.
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

from src.config import get_config  # noqa: E402
from src.llm.provider import LLMProvider, Role, available_providers  # noqa: E402
from src.llm.rate_limit import QuotaExhausted  # noqa: E402

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}  PASS{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}  FAIL{RESET} {msg}")


def main() -> int:
    print("\n=== C2 smoke test: LLM provider layer ===\n")

    keys = available_providers()
    for name, present in keys.items():
        (ok if present else fail)(f"{name} API key present")
    if not keys["groq"]:
        print("\nGROQ_API_KEY is required. Copy .env.example to .env.")
        return 1

    cfg = get_config()
    provider = LLMProvider(cfg)
    failures = 0

    for role in (Role.JUDGMENT, Role.MECHANICAL):
        chain = [s.model for s in provider.chains[role]]
        print(f"\n{DIM}--- {role.value}: {chain}{RESET}")
        try:
            r = provider.complete(
                system="You are terse. Answer in at most five words.",
                user="What is the capital of France?",
                role=role,
            )
            ok(f"{r.model} -> {r.text.strip()[:60]!r}")
            print(
                f"{DIM}       {r.prompt_tokens} in + {r.completion_tokens} out "
                f"= {r.total_tokens} tokens, {r.latency_s:.2f}s, "
                f"attempts={r.attempts}, fell_back={r.fell_back}{RESET}"
            )
            if r.total_tokens <= 0:
                fail("token usage not reported — quota accounting would be blind")
                failures += 1
        except (QuotaExhausted, Exception) as exc:  # noqa: BLE001
            fail(f"{role.value}: {type(exc).__name__}: {exc}")
            failures += 1

    print(f"\n{DIM}--- structured output{RESET}")
    try:
        parsed, r = provider.complete_json(
            system=(
                "You output only JSON. Schema: "
                '{"sub_questions": [{"id": "sq1", "text": "..."}]}'
            ),
            user="Decompose: 'How does QLoRA reduce memory?' into 2 sub-questions.",
            role=Role.JUDGMENT,
            required_keys=["sub_questions"],
        )
        n = len(parsed["sub_questions"])
        ok(f"parsed {n} sub-questions from {r.model}")
        if n == 0:
            fail("empty decomposition")
            failures += 1
    except Exception as exc:  # noqa: BLE001
        fail(f"structured output: {type(exc).__name__}: {exc}")
        failures += 1

    if keys["gemini"]:
        print(f"\n{DIM}--- judge (independent provider){RESET}")
        try:
            r = provider.complete(
                system="Reply with exactly one word: SUPPORTED or UNSUPPORTED.",
                user=(
                    "Claim: 'Paris is the capital of France.'\n"
                    "Passage: 'Paris is the capital and largest city of France.'\n"
                    "Does the passage support the claim?"
                ),
                role=Role.JUDGE,
            )
            ok(f"{r.model} -> {r.text.strip()[:40]!r} ({r.total_tokens} tokens)")
        except Exception as exc:  # noqa: BLE001
            fail(f"judge: {type(exc).__name__}: {exc}")
            failures += 1
    else:
        print(f"\n{DIM}--- judge skipped (no GOOGLE_API_KEY){RESET}")

    print(f"\n{DIM}--- quota ledger after this run{RESET}")
    print(provider.quota_report())
    print(
        f"\n{DIM}Ledger persisted to {cfg.path('llm.quota_ledger_path')}\n"
        f"It survives restarts on purpose — a resumed eval must not re-spend "
        f"a quota that is already gone.{RESET}"
    )

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
        return 1
    print(f"{GREEN}C2 acceptance: all checks passed.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())