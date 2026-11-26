"""C30.1: the premise verifier must see the end of a passage.

Live (C30 probe, D1-03): a Chinchilla Related Work passage describes Kaplan
et al.'s conclusion and then refutes it in its final sentence. Cut at 700
characters, mid-sentence, the verifier saw only the description, called the
false premise SUPPORTED, and cited the passage that refutes it.
"""
import json
from pathlib import Path

from src.config import Config
from src.nodes.premise_checker import (
    MIN_EVIDENCE_CHARS,
    TRIM_MARKER,
    PremiseChecker,
    _trim_at_sentence,
)

CORPUS = {p["id"]: p for p in json.loads(
    (Path(__file__).resolve().parents[1] / "benchmark" / "retrieval_corpus.json").read_text())["passages"]}
LIVE = CORPUS["Pf3481f6f16bd"]
REFUTATION = "should scale at roughly the same rate"


class Capture:
    def __init__(self, budget=None):
        self.budget = budget
        self.user = None

    def prompt_token_budget(self, role):
        return self.budget

    def complete_json(self, system, user, **kw):
        self.user = user
        return {"verdicts": []}, None


def verify_prompt(provider, n_copies=1):
    from src.nodes.premise_checker import Presupposition

    checker = PremiseChecker(Config({}), provider=provider, retriever=object(), reranker=object())
    evidence = [(f"P{i + 1:02d}", LIVE) for i in range(n_copies)]
    pre = [Presupposition(text="The Chinchilla authors concluded model size should be "
                               "scaled much faster than training data", centrality="core")]
    try:
        checker._verify("Why did Chinchilla conclude…?", pre, evidence)
    except Exception:  # noqa: BLE001 — only the rendered prompt matters here
        pass
    return provider.user


def test_the_live_passage_now_reaches_the_verifier_whole():
    prompt = verify_prompt(Capture(budget=5900))
    assert REFUTATION in prompt


def test_before_c30_1_the_refutation_was_cut():
    """The defect, reproduced: a 700-character hard cut."""
    assert REFUTATION not in LIVE["text"][:700]
    assert REFUTATION in LIVE["text"]


def test_trimming_never_cuts_mid_sentence():
    text = " ".join(LIVE["text"].split())
    out = _trim_at_sentence(text, 700)
    assert out.endswith("." + TRIM_MARKER)
    assert "In contrast, our analysis predicts that both quantities" not in out


def test_text_within_the_cap_is_untouched():
    assert _trim_at_sentence("Short. Passage.", 100) == "Short. Passage."


def test_a_tight_budget_trims_at_sentence_ends_and_marks_it():
    prompt = verify_prompt(Capture(budget=1500), n_copies=8)
    assert TRIM_MARKER.strip() in prompt


def test_no_passage_is_trimmed_below_the_floor():
    out = _trim_at_sentence("word " * 400, MIN_EVIDENCE_CHARS)
    assert len(out) >= MIN_EVIDENCE_CHARS // 2


def test_the_prompt_carries_the_attribution_rule():
    from src.prompts import load

    prompt = load("premise_verify")
    assert "ATTRIBUTION" in prompt and "Related Work" in prompt
    assert "[…]" in prompt
