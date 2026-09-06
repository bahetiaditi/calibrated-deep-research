#!/usr/bin/env python3
"""List the Gemini models this API key can actually call.

    python -m scripts.list_gemini_models

Why this exists: AI Studio's rate-limit page lists models that are visible
but NOT callable. gemini-2.5-flash showed limits of 5 RPM / 20 RPD on this
project, yet the API returns 404 "no longer available to new users". The
model catalog is the authoritative source; the rate-limit page is not.

Costs zero generation tokens — listing models is a metadata call.
"""
from __future__ import annotations

import sys

from src.config import require_env


def main() -> int:
    from google import genai

    client = genai.Client(api_key=require_env("GOOGLE_API_KEY"))

    rows = []
    for model in client.models.list():
        actions = list(getattr(model, "supported_actions", None) or [])
        if "generateContent" not in actions:
            continue  # embedding / TTS / other non-chat endpoints
        rows.append(
            (
                model.name.replace("models/", ""),
                getattr(model, "input_token_limit", None),
                getattr(model, "output_token_limit", None),
                (getattr(model, "display_name", "") or "")[:38],
            )
        )

    rows.sort()
    print(f"\n{len(rows)} models support generateContent with this key:\n")
    print(f"{'model string':42s} {'in':>9s} {'out':>7s}  display name")
    print("-" * 100)
    for name, tin, tout, disp in rows:
        print(f"{name:42s} {str(tin):>9s} {str(tout):>7s}  {disp}")

    print(
        "\nPick the cheapest/fastest FLASH model from this list for the judge "
        "role and put that exact string in config.yaml under llm.roles.judge.\n"
        "Then confirm its limits on the AI Studio rate-limit page and update\n"
        "the `limits:` block to match.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())