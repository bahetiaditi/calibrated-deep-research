#!/usr/bin/env python3
"""Find out which Gemini models this key can actually CALL.

    python -m scripts.probe_gemini

Listing a model is not the same as being able to call it: gemini-2.5-flash
appears in models.list() with a 1M context window and still returns
404 "no longer available to new users" on generateContent. Only a real call
settles it.

Cost: one tiny call per candidate (~15 tokens each). With a 20 RPD ceiling
that is a real but small spend, and it is the only way to know.
"""
from __future__ import annotations

import sys

from src.config import require_env

# Ordered cheapest/most-likely-free first. Deliberately excludes the
# floating aliases (gemini-flash-latest, gemini-flash-lite-latest): a judge
# whose version changes underneath you makes scores from different days
# incomparable, which defeats the purpose of a held-out test split.
CANDIDATES = [
    "gemini-3.1-flash-lite",   # 15 RPM / 250K TPM /   500 RPD — current judge
    "gemini-3.5-flash",        #  5 RPM / 250K TPM /    20 RPD
    "gemma-4-31b-it",          # 30 RPM /  16K TPM / 14.4K RPD — big upside
    "gemma-4-26b-a4b-it",      # 30 RPM /  16K TPM / 14.4K RPD
    "gemini-3.6-flash",        # retry: earlier 400 may have been our fault
    "gemini-3.5-flash-lite",   # 15 RPM / 250K TPM /   500 RPD if it works
]

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def main() -> int:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))

    print(f"\nProbing {len(CANDIDATES)} candidates with a minimal call each.\n")
    working: list[tuple[str, int]] = []

    for model in CANDIDATES:
        # Some models reject thinking_config outright ("Thinking budget is not
        # supported"): Gemma has no thinking mode at all. That is our invalid
        # argument, not a model limitation, so retry without it before
        # concluding the model is unusable.
        for attempt, use_thinking in enumerate((True, False)):
            try:
                cfg: dict = {
                    "temperature": 0.0,
                    "max_output_tokens": 64,
                    "automatic_function_calling": (
                        types.AutomaticFunctionCallingConfig(disable=True)
                    ),
                }
                if use_thinking:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

                resp = client.models.generate_content(
                    model=model,
                    contents="Reply with the single word: OK",
                    config=types.GenerateContentConfig(**cfg),
                )
                usage = resp.usage_metadata
                total = int(getattr(usage, "total_token_count", 0) or 0)
                thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
                text = (resp.text or "").strip()[:24]
                note = "" if use_thinking else "  [no thinking_config]"

                if not text:
                    # A 200 with empty text is a FAILURE, not a pass. Seen on
                    # gemini-3.7/3.8-flash: they ignore thinking_budget: 0 and
                    # burn the whole output budget on hidden thinking.
                    print(
                        f"{RED}  FAIL{RESET} {model:24s} 200 but EMPTY text "
                        f"({total} tok, {thoughts} thinking){note}"
                    )
                    break

                print(
                    f"{GREEN}  OK  {RESET} {model:24s} {total:>4d} tok "
                    f"({thoughts} thinking)  {text!r}{note}"
                )
                working.append((model, use_thinking))
                break

            except Exception as exc:  # noqa: BLE001
                msg = str(exc).split("\n")[0]
                retryable = "hinking" in msg and attempt == 0
                if retryable:
                    continue
                code = getattr(exc, "code", None) or getattr(exc, "status_code", "?")
                print(f"{RED}  FAIL{RESET} {model:24s} [{code}] {msg[:78]}")
                break

    print()
    if not working:
        print(f"{RED}No candidate is callable. Gemini cannot serve as judge —{RESET}")
        print("tell Claude and we will move the judge role to a Groq model that")
        print("is not used by the judgment chain (qwen/qwen3.8-27b).")
        return 1

    print(f"\n{GREEN}Callable models:{RESET}")
    for name, use_thinking in working:
        flag = "thinking_budget: 0" if use_thinking else "OMIT thinking_budget"
        print(f"    {name:24s} -> {flag}")
    print(
        f"\n{DIM}A model needing OMIT must have no thinking_budget key in its\n"
        f"config.yaml block — passing it returns 400.\n\n"
        f"Next: read the RPM / TPM / RPD row for the model you pick from the\n"
        f"AI Studio rate-limit page and put those exact numbers in its limits:\n"
        f"block. Do not guess them — the ledger is only as accurate as these.{RESET}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())