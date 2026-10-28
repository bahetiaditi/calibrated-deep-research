"""Tests for the synthesizer.

The citation machinery is what matters. A citation that does not resolve is
indistinguishable from a fabricated one, so the project's headline metric
depends on this node never emitting one.
"""
import pytest

from src.config import Config
from src.nodes.synthesizer import (
    MAX_PASSAGES,
    Synthesizer,
    synthesizer_node,
)
from src.state import Passage, merge_evidence, new_state, new_sub_question

CFG = Config({})


def passage(pid, text="some evidence text", section="Results", source=None):
    return Passage(
        id=pid, source_type="arxiv", source_id=source or "2307.08691v1",
        source_domain="arxiv.org", title="t", text=text, section=section,
        published="2023-01-01", sub_question_id="sq1",
        retrieval_score=0.5, rerank_score=7.0,
    )


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeProvider:
    def __init__(self, text="", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def complete(self, system, user, *, role=None, **kw):
        self.calls.append({"system": system, "user": user, "role": role})
        if self.error:
            raise self.error
        return FakeResponse(self.text)


def state_with(n=3, ids=None):
    ids = ids or [f"Pa{i}{'f' * 10}" for i in range(n)]
    state = new_state("How does FlashAttention-2 reduce memory traffic?")
    state["plan"] = [new_sub_question("sub one", sub_question_id="sq1")]
    state["plan"][0]["evidence_ids"] = ids
    state["evidence"] = [passage(pid) for pid in ids]
    return state


def synth(text="", **kw):
    return Synthesizer(CFG, provider=FakeProvider(text), **kw)


# --- the label map ---------------------------------------------------------


def test_prompt_uses_short_sequential_labels():
    """Content-addressed ids are a bad thing to ask a model to transcribe;
    a slip produces a citation that cannot be distinguished from a
    fabrication."""
    provider = FakeProvider("Claim. [P01]")
    Synthesizer(CFG, provider=provider).synthesize(state_with(3))
    evidence_block = provider.calls[0]["system"].split("EVIDENCE")[-1]
    assert "P01" in evidence_block and "P03" in evidence_block
    # The real content-addressed id must never reach the model.
    assert "Pa0ffffffffff" not in provider.calls[0]["system"]


def test_labels_are_rewritten_to_real_passage_ids():
    draft = synth("FlashAttention-2 tiles the computation. [P02]").synthesize(
        state_with(3)
    )
    assert "[Pa1ffffffffff]" in draft.text
    assert "[P02]" not in draft.text
    assert draft.cited_passage_ids == ["Pa1ffffffffff"]


def test_multiple_citations_on_one_claim():
    draft = synth("A claim supported twice. [P01][P03]").synthesize(state_with(3))
    assert len(draft.cited_passage_ids) == 2


def test_repeated_citation_counted_once():
    draft = synth("One. [P01] Two. [P01]").synthesize(state_with(3))
    assert draft.cited_passage_ids == ["Pa0ffffffffff"]


def test_label_map_is_returned_for_inspection():
    draft = synth("x [P01]").synthesize(state_with(3))
    assert draft.label_map["P01"] == "Pa0ffffffffff"
    assert len(draft.label_map) == 3


# --- invalid citations -----------------------------------------------------


def test_invented_label_is_stripped_not_kept():
    """An uncited claim is marked UNSUPPORTED by the critic — correct and
    visible. A citation pointing at nothing would either crash the critic's
    lookup or be silently skipped, and a claim that LOOKS cited but was never
    checked is the exact failure this project measures."""
    draft = synth("A real claim. [P01] An invented one. [P47]").synthesize(
        state_with(3)
    )
    assert "[P47]" not in draft.text
    assert draft.invalid_labels == ["P47"]
    assert "An invented one." in draft.text        # the claim survives, uncited


def test_invalid_label_count_is_recorded():
    """Burying this would hide a real problem: a model inventing labels is a
    signal about prompt quality."""
    draft = synth("a [P90] b [P91] c [P01]").synthesize(state_with(3))
    assert draft.invalid_labels == ["P90", "P91"]
    assert draft.citation_count == 1


def test_stripping_tidies_whitespace():
    draft = synth("A claim [P99] .").synthesize(state_with(3))
    assert "  " not in draft.text
    assert " ." not in draft.text


def test_every_surviving_citation_resolves():
    """The C18 acceptance criterion: 100% of citation ids exist."""
    import re

    draft = synth(
        "One. [P01] Two. [P02] Three. [P99] Four. [P03]"
    ).synthesize(state_with(3))
    cited = re.findall(r"\[(P[0-9a-f]{12})\]", draft.text)
    known = set(draft.label_map.values())
    assert cited and all(c in known for c in cited)


# --- evidence selection ----------------------------------------------------


def test_evidence_grouped_by_sub_question_not_globally_sorted():
    """A global sort lets one well-covered sub-question take every slot and
    starve the others, producing a report that answers part of the question
    confidently and ignores the rest."""
    state = new_state("q")
    state["plan"] = [
        new_sub_question("a", sub_question_id="sq1"),
        new_sub_question("b", sub_question_id="sq2"),
    ]
    state["plan"][0]["evidence_ids"] = ["Pa", "Pb"]
    state["plan"][1]["evidence_ids"] = ["Pc"]
    state["evidence"] = [passage("Pa"), passage("Pb"), passage("Pc")]

    provider = FakeProvider("x [P03]")
    Synthesizer(CFG, provider=provider).synthesize(state)
    # Search the evidence block: the instructions mention P01/P03 as examples.
    block = provider.calls[0]["system"].split("EVIDENCE")[-1]
    assert block.index("P01") < block.index("P02") < block.index("P03")


def test_shared_evidence_is_offered_once():
    state = new_state("q")
    state["plan"] = [
        new_sub_question("a", sub_question_id="sq1"),
        new_sub_question("b", sub_question_id="sq2"),
    ]
    state["plan"][0]["evidence_ids"] = ["Pa", "Pb"]
    state["plan"][1]["evidence_ids"] = ["Pb"]       # shared
    state["evidence"] = [passage("Pa"), passage("Pb")]
    draft = synth("x [P01]").synthesize(state)
    assert len(draft.label_map) == 2


def test_abandoned_sub_question_evidence_is_demoted_not_discarded():
    """An abandoned sub-question was given up on because it was malformed,
    so its evidence was retrieved for a bad query. But a passage can be a
    poor answer to one question and a good one for another, and discarding
    evidence the system paid to retrieve is a stronger claim than the status
    supports. Ordering it last means MAX_PASSAGES drops it first."""
    state = new_state("q")
    state["plan"] = [
        new_sub_question("live", sub_question_id="sq1"),
        new_sub_question("dead", sub_question_id="sq2"),
    ]
    state["plan"][0]["evidence_ids"] = ["Plive"]
    state["plan"][1]["evidence_ids"] = ["Pdead"]
    state["plan"][1]["status"] = "abandoned"
    state["evidence"] = [passage("Pdead"), passage("Plive")]

    draft = synth("x [P01]").synthesize(state)
    assert draft.label_map["P01"] == "Plive"      # live evidence first
    assert "Pdead" in draft.label_map.values()    # but still available


def test_section_is_shown_to_the_model():
    """The prompt asks the model to distinguish a measured result from a
    cited one; it cannot without knowing the section."""
    state = state_with(1)
    state["evidence"] = [passage("Pa0ffffffffff", section="Related Work")]
    provider = FakeProvider("x [P01]")
    Synthesizer(CFG, provider=provider).synthesize(state)
    assert "[Related Work]" in provider.calls[0]["system"]


def test_passage_cap_is_enforced():
    state = state_with(40)
    draft = synth("x [P01]").synthesize(state)
    assert len(draft.label_map) == MAX_PASSAGES


def test_long_passages_are_truncated():
    state = state_with(1)
    state["evidence"] = [passage("Pa0ffffffffff", text="word " * 1000)]
    provider = FakeProvider("x [P01]")
    Synthesizer(CFG, provider=provider).synthesize(state)
    assert len(provider.calls[0]["system"]) < 6000


# --- modes -----------------------------------------------------------------


def test_partial_mode_uses_the_partial_prompt():
    provider = FakeProvider("x [P01]")
    Synthesizer(CFG, provider=provider).synthesize(
        state_with(3), mode="partial", gaps=["the backward-pass figure"]
    )
    system = provider.calls[0]["system"]
    assert "UNRESOLVED" in system
    assert "the backward-pass figure" in system


def test_partial_mode_forbids_speculation_explicitly():
    provider = FakeProvider("x [P01]")
    Synthesizer(CFG, provider=provider).synthesize(state_with(3), mode="partial")
    assert "do not soften" in provider.calls[0]["system"].lower()


def test_answer_mode_still_warns_about_gaps():
    provider = FakeProvider("x [P01]")
    Synthesizer(CFG, provider=provider).synthesize(
        state_with(3), gaps=["something unresolved"]
    )
    assert "NOT ESTABLISHED" in provider.calls[0]["system"]


def test_mode_recorded_on_the_draft():
    assert synth("x [P01]").synthesize(state_with(3), mode="partial").mode == "partial"


# --- degradation -----------------------------------------------------------


def test_no_evidence_returns_an_empty_draft():
    assert synth("x").synthesize(new_state("q")).error


def test_empty_question_returns_an_empty_draft():
    state = state_with(3)
    state["question"] = "  "
    assert synth("x [P01]").synthesize(state).error


def test_provider_failure_does_not_raise():
    s = Synthesizer(CFG, provider=FakeProvider(error=RuntimeError("down")))
    draft = s.synthesize(state_with(3))
    assert draft.error and draft.text == ""


def test_draft_with_no_citations_at_all():
    draft = synth("A long claim with no citation whatsoever in it.").synthesize(
        state_with(3)
    )
    assert draft.citation_count == 0
    assert draft.uncited_sentences == 1


def test_short_sentences_are_not_counted_as_uncited_claims():
    """Transitions are not claims."""
    draft = synth("Yes. No. [P01] A properly long claim goes here now. [P02]"
                  ).synthesize(state_with(3))
    assert draft.uncited_sentences == 0


# --- budget, tracing, node -------------------------------------------------


def test_budget_charged_once():
    from src.budget import Budget

    budget = Budget(llm_calls_max=25, retrieval_calls_max=20,
                    wall_clock_max_s=900)
    Synthesizer(CFG, provider=FakeProvider("x [P01]"),
                budget=budget).synthesize(state_with(3))
    assert budget.llm_calls_used == 1


def test_synthesis_is_traced():
    events = []

    class FakeTracer:
        def note(self, name, **kw):
            events.append((name, kw))

    Synthesizer(CFG, provider=FakeProvider("x [P01] y [P88]"),
                tracer=FakeTracer()).synthesize(state_with(3))
    name, payload = events[0]
    assert name == "synthesis"
    assert payload["passages_cited"] == 1
    assert payload["invalid_labels"] == ["P88"]


def test_node_writes_draft():
    update = synthesizer_node(synth("A claim. [P01]"))(state_with(3))
    assert "[Pa0ffffffffff]" in update["draft"]


def test_node_does_not_run_on_the_abstain_path():
    """§3.3: there is no report to write, and generating one would invite
    the finalizer to show it."""
    provider = FakeProvider("should not be generated")
    state = state_with(3)
    state["terminal_decision"] = "ABSTAIN"
    update = synthesizer_node(Synthesizer(CFG, provider=provider))(state)
    assert update["draft"] == ""
    assert provider.calls == []


def test_node_uses_partial_mode_on_the_partial_path():
    provider = FakeProvider("x [P01]")
    state = state_with(3)
    state["terminal_decision"] = "PARTIAL"
    state["unresolved_gaps"] = ["a gap"]
    synthesizer_node(Synthesizer(CFG, provider=provider))(state)
    assert "UNRESOLVED" in provider.calls[0]["system"]

# --- C27.3: the prompt must fit the judgment chain -------------------------

from src.llm.provider import estimate_tokens  # noqa: E402

LONG = "Measured result with enough detail to be worth citing. " * 16   # ~880 chars


class BudgetedProvider(FakeProvider):
    def __init__(self, budget, text="A claim. [P01]"):
        super().__init__(text)
        self.budget = budget

    def prompt_token_budget(self, role):
        return self.budget


def two_sub_question_state(per_sq=12):
    state = new_state("How do FlashAttention-2 and PagedAttention differ?")
    state["plan"] = []
    state["evidence"] = []
    for sqid in ("sq1", "sq2"):
        sq = new_sub_question(f"text {sqid}", sub_question_id=sqid)
        ids = [f"P{sqid}{i:02d}{'a' * 8}" for i in range(per_sq)]
        sq["evidence_ids"] = ids
        state["plan"].append(sq)
        state["evidence"].extend(passage(pid, text=LONG) for pid in ids)
    return state


def test_the_prompt_is_trimmed_to_the_chains_budget():
    """The C27.2 smoke: 24 passages came to ~6.1K tokens and no judgment
    model could accept the call at all."""
    provider = BudgetedProvider(budget=3000)
    draft = Synthesizer(CFG, provider=provider).synthesize(two_sub_question_state())
    assert draft.error is None
    call = provider.calls[0]
    assert estimate_tokens(call["system"], call["user"], 0) <= 3000


def test_trimming_is_fair_across_sub_questions():
    """Cutting the grouped list's tail would delete the last sub-question's
    evidence entirely — the starvation C18's grouping prevents. With
    round-robin trimming both keep evidence, within one passage of each
    other, and each keeps its BEST passages."""
    provider = BudgetedProvider(budget=2500)
    state = two_sub_question_state()
    draft = Synthesizer(CFG, provider=provider).synthesize(state)
    shown = set(draft.label_map.values())
    assert 2 <= len(shown) < 24
    ranked = [sq["evidence_ids"] for sq in state["plan"]]
    kept = [[pid for pid in ids if pid in shown] for ids in ranked]
    assert kept[0] and kept[1]
    assert abs(len(kept[0]) - len(kept[1])) <= 1
    for ids, mine in zip(ranked, kept):
        assert mine == ids[: len(mine)]           # a prefix: best first


def test_trimming_records_how_much_was_dropped():
    events = []

    class Tracer:
        def note(self, name, **kw):
            events.append(kw)

    provider = BudgetedProvider(budget=2500)
    Synthesizer(CFG, provider=provider, tracer=Tracer()).synthesize(
        two_sub_question_state())
    assert events[0]["passages_dropped_for_budget"] > 0
    assert events[0]["passages_offered"] + events[0]["passages_dropped_for_budget"] == 24


def test_nothing_is_trimmed_when_it_already_fits():
    provider = BudgetedProvider(budget=100_000)
    draft = Synthesizer(CFG, provider=provider).synthesize(
        two_sub_question_state(per_sq=3))
    assert len(draft.label_map) == 6


def test_labels_match_the_passages_actually_shown():
    """The label map is built after trimming. Built before, P01..P24 would
    include labels for dropped passages, and a citation to one would resolve
    to evidence the model never saw."""
    provider = BudgetedProvider(budget=2500, text="A claim. [P01] Another. [P02]")
    draft = Synthesizer(CFG, provider=provider).synthesize(two_sub_question_state())
    system = provider.calls[0]["system"]
    labels_in_prompt = {f"P{i:02d}" for i in range(1, 25) if f"\nP{i:02d}  " in "\n" + system}
    assert set(draft.label_map) == labels_in_prompt
    assert set(draft.cited_passage_ids) <= set(draft.label_map.values())


def test_a_budget_too_small_for_one_passage_is_an_error_not_an_empty_call():
    provider = BudgetedProvider(budget=50)
    draft = Synthesizer(CFG, provider=provider).synthesize(two_sub_question_state())
    assert draft.error and "prompt budget" in draft.error
    assert provider.calls == []
