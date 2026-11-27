"""C37.4: a critic call that parses but returns too few verdicts.

Live (A09, calibration a1): the critic's output was cut at the token cap,
`extract_json` closed the truncated array, the call "succeeded", and the four
sentences after the cut defaulted to UNSUPPORTED "no verdict returned" —
including correctly cited results. Split-on-failure (C28.3) never fired,
because nothing raised.
"""
import re

from src.config import Config
from src.nodes.critic import MAX_SPLIT_DEPTH, NO_VERDICT, SUPPORTED, UNSUPPORTED, Critic
from tests.test_critic import PID, passage, state_with

CFG = Config({})


class TruncatingProvider:
    """Returns verdicts for at most `keep` sentences per call, as a cut-off
    output repaired into valid JSON does. `keep=None` judges everything."""

    def __init__(self, keep, verdict="SUPPORTED"):
        self.keep, self.verdict, self.batches = keep, verdict, []

    def complete_json(self, system, user, **kw):
        indices = [int(i) for i in re.findall(r"^\[(\d+)\] ", system, flags=re.M)]
        self.batches.append(indices)
        shown = indices if self.keep is None else indices[: self.keep]
        return {"verdicts": [{"index": i, "verdict": self.verdict, "reasoning": "ok"}
                             for i in shown]}, None


def draft(n, cited=True):
    cite = f" [{PID[1]}]" if cited else ""
    return " ".join(f"Claim number {i} about attention kernels is checkable.{cite}"
                    for i in range(n))


def review(provider, n=10, cited=True, **kw):
    return Critic(CFG, provider=provider, **kw).review(
        state_with(draft(n, cited), [passage(PID[1])]))


def test_the_sentences_after_a_cut_are_judged_not_defaulted():
    """The effect: before C37.4, 4 of these 10 were UNSUPPORTED."""
    provider = TruncatingProvider(keep=6)
    report = review(provider, n=10)
    assert [f["verdict"] for f in report.findings] == [SUPPORTED] * 10
    assert report.missing_verdicts == 0
    # The re-ask carried only the four missing sentences.
    assert provider.batches[1] == provider.batches[0][6:]


def test_repeated_truncation_is_bounded_and_the_remainder_is_counted():
    provider = TruncatingProvider(keep=2)
    report = review(provider, n=10)
    assert len(provider.batches) == 1 + MAX_SPLIT_DEPTH          # 10, then 8, then 6
    judged = [f for f in report.findings if NO_VERDICT not in f["reasoning"]]
    unjudged = [f for f in report.findings if NO_VERDICT in f["reasoning"]]
    assert len(judged) == 2 * (1 + MAX_SPLIT_DEPTH)
    assert all(f["verdict"] == UNSUPPORTED for f in unjudged)     # never a pass
    assert report.missing_verdicts == len(unjudged) == 10 - len(judged)


def test_a_complete_answer_costs_exactly_one_call():
    """The fix is a no-op when nothing is missing — which is why records
    with every verdict present need no re-run."""
    provider = TruncatingProvider(keep=None)
    review(provider, n=10)
    assert len(provider.batches) == 1


def test_an_unjudged_uncited_sentence_says_it_was_unjudged():
    """It might have been a gap statement or discourse; 'no citation' alone
    would hide that the critic never looked at it."""
    report = review(TruncatingProvider(keep=0), n=3, cited=False)
    assert all(NO_VERDICT in f["reasoning"] for f in report.findings)
    assert report.missing_verdicts == 3


def test_a_failing_re_ask_keeps_the_verdicts_already_returned():
    class FailSecond(TruncatingProvider):
        def complete_json(self, system, user, **kw):
            if self.batches:
                self.batches.append([])
                raise RuntimeError("429")
            return super().complete_json(system, user, **kw)

    report = review(FailSecond(keep=6), n=10)
    verdicts = [f["verdict"] for f in report.findings]
    assert verdicts[:6] == [SUPPORTED] * 6 and verdicts[6:] == [UNSUPPORTED] * 4
    assert report.error is None and report.missing_verdicts == 4


def test_the_trace_reports_the_re_ask():
    events = []

    class Tracer:
        def note(self, name, **kw):
            events.append((name, kw))

    review(TruncatingProvider(keep=6), n=10, tracer=Tracer())
    re_asks = [kw for name, kw in events if name == "critic_missing_verdicts"]
    assert re_asks == [{"sentences": 10, "missing": 4, "depth": 1}]
    (critic_note,) = [kw for name, kw in events if name == "critic"]
    assert critic_note["missing_verdicts"] == 0
