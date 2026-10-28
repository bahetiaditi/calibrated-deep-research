"""C27.4: citation forms the synthesizer must understand.

The live failure: an answer-mode draft cited 0 of 22 passages. Only `[P01]`
was parsed; any other form stayed plain text, the critic saw uncited claims,
and the run's decision, gaps and partial rewrite were all built on that.
The test that matters is the last group: that a variant citation reaches the
CRITIC as a citation.
"""
import pytest

from src.config import Config
from src.nodes.critic import segment
from src.nodes.synthesizer import Synthesizer, normalize_citations
from tests.test_synthesizer import FakeProvider, state_with


@pytest.mark.parametrize("raw,expected", [
    ("A [P01, P03].", "A [P01][P03]."),
    ("A [P01,P03].", "A [P01][P03]."),
    ("A [P1; P4].", "A [P01][P04]."),
    ("A [P01 and P02].", "A [P01][P02]."),
    ("A [P01 & P02].", "A [P01][P02]."),
    ("A 【P02】.", "A [P02]."),
    ("A (P05, P06).", "A [P05][P06]."),
    ("A [P02-P04].", "A [P02][P03][P04]."),
    ("A [P02–P04].", "A [P02][P03][P04]."),
    ("A [P7].", "A [P07]."),
])
def test_variants_become_the_instructed_form(raw, expected):
    text, count = normalize_citations(raw)
    assert text == expected and count == 1


@pytest.mark.parametrize("raw", [
    "A [P01].", "A [P01][P03].",
    "A [Pa1b2c3d4e5f6] stays.",          # a content-addressed id
    "Table P1 shows nothing cited.",      # prose, not a citation
    "A [P01-P40].",                       # an implausible range is left alone
])
def test_what_is_not_a_variant_is_left_alone(raw):
    assert normalize_citations(raw) == (raw, 0)


def ids(n):
    return [f"Pa{i}{'f' * 10}" for i in range(n)]


def synthesize(text, n=4):
    state = state_with(ids=ids(n))
    return Synthesizer(Config({}), provider=FakeProvider(text)).synthesize(state), state


def test_a_variant_citation_resolves_to_passages():
    draft, _ = synthesize("Claim one [P01, P02]. Claim two 【P03】.")
    assert draft.cited_passage_ids == ids(4)[:3]
    assert draft.normalised_citations == 2
    assert draft.unparsed_citations == 0


def test_an_invented_label_inside_a_group_is_dropped_and_the_rest_kept():
    draft, _ = synthesize("Claim [P01, P99].")
    assert draft.cited_passage_ids == [ids(4)[0]]
    assert draft.invalid_labels == ["P99"]


def test_an_unparseable_citation_is_counted_not_silently_lost():
    draft, _ = synthesize("Claim one {P01}. Claim two [source: P02].")
    assert draft.cited_passage_ids == []
    assert draft.unparsed_citations == 2


def test_the_critic_sees_a_variant_citation_as_a_citation():
    """The effect: before C27.4 this sentence reached the critic uncited,
    and the absence of a citation decided its verdict."""
    draft, _ = synthesize("FlashAttention-2 reaches 73% of peak [P01, P02].")
    sentences = segment(draft.text)
    assert sentences[0].citations == ids(4)[:2]


def test_the_trace_reports_normalised_and_unparsed_counts():
    events = []

    class Tracer:
        def note(self, name, **kw):
            events.append(kw)

    state = state_with(ids=ids(3))
    Synthesizer(Config({}), provider=FakeProvider("A [P01, P02]. B {P03}."),
                tracer=Tracer()).synthesize(state)
    assert events[0]["normalised_citations"] == 1
    assert events[0]["unparsed_citations"] == 1
