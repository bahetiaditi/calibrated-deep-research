"""C37.3: every node sees a passage the same way — id, title, same window.

Two calibration defects, reproduced on the real corpus passages:
  A06  the critic saw "1706.03762v7" with no title and rejected "the base
       Transformer uses eight heads" as describing some other model
  A05  the critic saw 800 characters, the writer 900; the support for
       "3.5 days on eight GPUs" (character 872) was outside the critic's view
"""
import json
from pathlib import Path

from src.config import Config
from src.nodes.critic import Critic
from src.nodes.premise_checker import PremiseChecker, Presupposition
from src.nodes.synthesizer import Synthesizer
from src.state import (
    EVIDENCE_TRIM_MARKER,
    EVIDENCE_WINDOW_CHARS,
    evidence_window,
    new_state,
    new_sub_question,
    source_label,
)

CORPUS = {p["id"]: p for p in json.loads(
    (Path(__file__).resolve().parents[1] / "benchmark" / "retrieval_corpus.json").read_text())["passages"]}
A06_PASSAGE, A05_PASSAGE = CORPUS["Pb6eeca1bdcd7"], CORPUS["P25b038f40d30"]


class Capture:
    def __init__(self, budget=None):
        self.budget, self.system, self.user = budget, None, None

    def prompt_token_budget(self, role):
        return self.budget

    def complete_json(self, system, user, **kw):
        self.system, self.user = system, user
        return {"verdicts": []}, None


def critic_prompt(passage, claim):
    state = new_state("q")
    sq = new_sub_question("sq", sub_question_id="sq1")
    sq["evidence_ids"] = [passage["id"]]
    state["plan"] = [sq]
    state["evidence"] = [dict(passage, sub_question_id="sq1")]
    state["draft"] = f"{claim} [{passage['id']}]."
    provider = Capture()
    Critic(Config({}), provider=provider).review(state)
    return provider.system


def test_the_critic_is_told_which_paper_the_a06_passage_comes_from():
    prompt = critic_prompt(A06_PASSAGE, "The base Transformer uses eight attention heads")
    assert "Attention Is All You Need" in prompt
    assert "h = 8 parallel attention layers" in prompt


def test_the_critic_sees_the_a05_support_the_writer_saw():
    """Character 872: inside the writer's 900, outside the critic's old 800."""
    text = " ".join(A05_PASSAGE["text"].split())
    assert 800 < text.find("3.5 days") < EVIDENCE_WINDOW_CHARS
    prompt = critic_prompt(A05_PASSAGE, "It was trained for 3.5 days on eight GPUs")
    assert "after training for 3.5 days on eight GP" in prompt


def test_writer_and_critic_see_identical_text_for_every_corpus_passage():
    """The invariant, over all 982 passages: whatever the synthesizer was
    shown, the critic is shown — no claim can be rejected for text the
    writer read and the verifier did not."""
    synth = Synthesizer(Config({}), provider=Capture())
    critic = Critic(Config({}), provider=Capture())
    for p in CORPUS.values():
        writer_body = synth._render_evidence([p]).split("\n", 1)[1]
        critic_body = critic._render_passage(p["id"], p, False).split("\n", 1)[1]
        assert writer_body == critic_body, p["id"]


def test_the_synthesizer_is_shown_titles_too():
    body = Synthesizer(Config({}), provider=Capture())._render_evidence([A06_PASSAGE])
    assert body.startswith('P01  1706.03762v7 "Attention Is All You Need"')


def test_the_premise_verifier_is_shown_titles():
    provider = Capture(budget=5900)
    checker = PremiseChecker(Config({}), provider=provider, retriever=object(), reranker=object())
    try:
        checker._verify("q?", [Presupposition(text="The Transformer uses 8 heads",
                                               centrality="core")], [("P01", A06_PASSAGE)])
    except Exception:  # noqa: BLE001 — only the prompt matters
        pass
    assert '"Attention Is All You Need"' in provider.user


def test_premise_fitting_counts_the_longer_headers():
    """A title can be 120 characters; with a fixed 60-per-header estimate,
    many passages under a tight budget overran it. The rendered prompt must
    fit the budget it was fitted to."""
    from src.llm.provider import estimate_tokens

    # Many short sentences, so sentence-boundary trimming is fine-grained
    # enough for the header arithmetic to decide where the cut lands.
    long_title = dict(A05_PASSAGE, title="X" * 200,
                      text=" ".join(f"Short sentence {k}." for k in range(200)))
    budget = 3000
    provider = Capture(budget=budget)
    checker = PremiseChecker(Config({}), provider=provider, retriever=object(), reranker=object())
    evidence = [(f"P{i + 1:02d}", long_title) for i in range(8)]
    try:
        checker._verify("q?", [Presupposition(text="claim", centrality="core")], evidence)
    except Exception:  # noqa: BLE001
        pass
    assert estimate_tokens("You are a precise evidence checker. Output only JSON.",
                           provider.user, 0) <= budget


def test_a_cut_passage_says_it_continues_and_a_whole_one_does_not():
    long = "word " * 400
    assert evidence_window(long).endswith(EVIDENCE_TRIM_MARKER)
    assert evidence_window("short passage.") == "short passage."


def test_source_label_shapes():
    assert source_label({"source_id": "1706.03762v7", "title": "Attention Is All You Need"}) \
        == '1706.03762v7 "Attention Is All You Need"'
    assert source_label({"source_id": "x"}) == "x"
    assert len(source_label({"source_id": "x", "title": "T" * 500})) < 130
