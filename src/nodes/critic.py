"""Critic — per-claim entailment judgment (C23).

This is the node the project's grounding metrics are computed from, so its
errors become the reported numbers. Three decisions follow from that.

**Three categories, not two.** A draft contains cited claims, uncited claims,
and sentences that assert nothing. The C18 run produced four uncited
sentences that were correct *summaries* of claims cited immediately above;
counting those as UNSUPPORTED would have put them in the hallucination rate
(§5.5) and made the metric measure the synthesizer's citation habits rather
than the system's grounding. `NOT_A_CLAIM` sentences are excluded from the
denominator entirely rather than scored.

The classification is deliberately biased: when the model is unsure, the
prompt tells it to call something a CLAIM. Judging a transition is harmless;
excusing a claim is not. The error that matters is the one that hides an
unsupported assertion.

**One call for the whole draft.** §7.1's arithmetic makes this a hard
requirement rather than an optimisation: per-claim calls would be 8-10 per
question against a budget of roughly 9, so the critic alone would exceed the
entire run. Batching also reduces variance — one judge seeing all claims
together applies a more consistent standard than ten independent calls.

**Every substantial sentence is classified, cited or not.** A cited sentence
needs two judgments — is it a claim, and does its citation support it. An
uncited one needs only the first: if it is a claim it is UNSUPPORTED by
definition. Skipping uncited sentences was cheaper and wrong, because it
conflated "no citation" with "not a claim", and no length heuristic
distinguishes "The second approach differs" from "FlashAttention-2 reaches
73% of peak". Both land in the hallucination rate if guessed wrong, in
opposite directions.

Five-way routing on what to *do* about a verdict is C24. This node decides
only what is true.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.config import Config, get_config
from src.llm.provider import LLMProvider, Role
from src.prompts import render
from src.state import ClaimVerification, Passage, ResearchState

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(P[0-9a-f]{12})\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")
# One or more citations at the very start of a fragment, which means the
# writer put them after the full stop of the previous sentence.
LEADING_CITATION = re.compile(r"^(?:\[P[0-9a-f]{12}\]\s*)+")

SUPPORTED = "SUPPORTED"
PARTIAL = "PARTIAL"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"
NOT_A_CLAIM = "NOT_A_CLAIM"
VERDICTS = (SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED)

# Below this a sentence is almost certainly connective tissue. Used only to
# skip the model, never to assign a verdict — anything longer is classified
# by the model, which is what the three-category prompt is for.
MIN_SENTENCE_CHARS = 25

MAX_PASSAGE_CHARS = 800


@dataclass
class Sentence:
    index: int
    text: str
    citations: list[str] = field(default_factory=list)

    @property
    def is_cited(self) -> bool:
        return bool(self.citations)


@dataclass
class CriticReport:
    findings: list[ClaimVerification] = field(default_factory=list)
    discourse_sentences: int = 0
    error: str | None = None

    @property
    def claims(self) -> int:
        return len(self.findings)

    def count(self, verdict: str) -> int:
        return sum(1 for f in self.findings if f["verdict"] == verdict)

    @property
    def grounding_precision(self) -> float | None:
        """Fraction of claims whose citation actually supports them.

        None with no claims — a draft that asserts nothing has no grounding
        to measure, and reporting 0.0 or 1.0 would both be inventions.
        """
        if not self.findings:
            return None
        return self.count(SUPPORTED) / len(self.findings)

    @property
    def unsupported_rate(self) -> float | None:
        """§5.5's hallucination rate: claims with no support or bad support.

        The denominator is CLAIMS, never sentences. Counting discourse
        sentences would depress this uniformly across the agent and all four
        baselines — not changing which wins, but making the absolute numbers
        meaningless and incomparable to published figures.
        """
        if not self.findings:
            return None
        bad = self.count(UNSUPPORTED) + self.count(CONTRADICTED)
        return bad / len(self.findings)


class Critic:
    def __init__(
        self,
        config: Config | None = None,
        *,
        provider: LLMProvider | None = None,
        tracer: Any = None,
        budget: Any = None,
    ) -> None:
        self.cfg = config or get_config()
        self.provider = provider or LLMProvider(self.cfg, tracer=tracer)
        self.tracer = tracer
        self.budget = budget

    # -- main entry point ---------------------------------------------------

    def review(self, state: ResearchState) -> CriticReport:
        draft = (state.get("draft") or "").strip()
        if not draft:
            return CriticReport(error="no draft to review")

        sentences = segment(draft)
        evidence = {p["id"]: p for p in (state.get("evidence") or [])}
        contradicted = contradicted_passages(state.get("contradictions") or [])

        # Every sentence long enough to assert something goes to the model,
        # cited or not — but for different reasons.
        #
        # A cited sentence needs two judgments: is it a claim, and does its
        # citation support it. An uncited one needs only the first: if it is
        # a claim it is UNSUPPORTED by definition, since there is nothing to
        # verify it against. Skipping uncited sentences entirely was cheaper
        # and wrong — it conflated "no citation" with "not a claim", and a
        # length heuristic cannot tell "The second approach differs" (28
        # characters, pure discourse) from "FlashAttention-2 reaches 73% of
        # peak" (also short, a claim). Both end up in the hallucination rate
        # if guessed wrong, in opposite directions.
        judged = [s for s in sentences if len(s.text) >= MIN_SENTENCE_CHARS]
        trivial = len(sentences) - len(judged)

        findings: list[ClaimVerification] = []
        discourse = trivial

        if judged:
            try:
                verdicts = self._judge(state, judged, evidence, contradicted)
            except Exception as exc:  # noqa: BLE001
                # A critic that crashes loses the whole verification pass.
                # Degrading to "unverified" keeps the run alive and is
                # visible in the findings rather than silently absent.
                log.warning("critic judgment failed: %s", exc)
                return CriticReport(
                    findings=[
                        self._finding(s, UNSUPPORTED,
                                      f"critic unavailable ({type(exc).__name__})",
                                      evidence, _owner(s, evidence))
                        for s in judged
                    ],
                    discourse_sentences=discourse,
                    error=str(exc),
                )

            for sentence in judged:
                verdict, reasoning = verdicts.get(
                    sentence.index, (UNSUPPORTED, "no verdict returned")
                )
                if verdict == NOT_A_CLAIM:
                    discourse += 1
                    continue
                if not sentence.is_cited:
                    # The model classified it as a claim; the absence of a
                    # citation decides the verdict, whatever the model said
                    # about support it could not have checked.
                    verdict = UNSUPPORTED
                    reasoning = (
                        "no citation — nothing to verify the claim against"
                    )
                findings.append(
                    self._finding(sentence, verdict, reasoning, evidence,
                                  _owner(sentence, evidence))
                )

        report = CriticReport(findings=findings, discourse_sentences=discourse)
        self._trace(report, len(sentences))
        return report

    # -- the single batched call --------------------------------------------

    def _judge(
        self,
        state: ResearchState,
        sentences: Sequence[Sentence],
        evidence: dict[str, Passage],
        contradicted: set[str],
    ) -> dict[int, tuple[str, str]]:
        cited_ids: list[str] = []
        for sentence in sentences:
            for pid in sentence.citations:
                if pid not in cited_ids:
                    cited_ids.append(pid)

        rendered_sentences = "\n".join(
            f"[{s.index}] {s.text}\n      cites: "
            f"{', '.join(s.citations) or '(none)'}"
            for s in sentences
        )
        rendered_passages = "\n\n".join(
            self._render_passage(pid, evidence.get(pid), pid in contradicted)
            for pid in cited_ids
        ) or "(no cited passage could be resolved)"

        parsed, _ = self.provider.complete_json(
            system=render(
                "critic",
                sentences=rendered_sentences,
                passages=rendered_passages,
            ),
            user="Judge every sentence.",
            role=Role.JUDGMENT,
            required_keys=["verdicts"],
        )
        self._charge()

        out: dict[int, tuple[str, str]] = {}
        for raw in parsed.get("verdicts") or []:
            if not isinstance(raw, dict):
                continue
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError):
                continue
            verdict = str(raw.get("verdict") or "").upper().strip()
            if verdict not in (*VERDICTS, NOT_A_CLAIM):
                # An unparseable verdict must not become a pass. Claiming
                # support the critic never gave is the one error the
                # grounding metric cannot survive.
                verdict = UNSUPPORTED
            out[index] = (verdict, str(raw.get("reasoning") or "")[:300])
        return out

    def _render_passage(
        self, pid: str, passage: Passage | None, is_contradicted: bool
    ) -> str:
        if passage is None:
            # A citation that does not resolve is indistinguishable from a
            # fabricated one — C18 strips those, so reaching here means the
            # evidence store and the draft have diverged.
            return f"{pid}\n(PASSAGE NOT FOUND — treat any claim citing it "
            f"as unsupported)"

        header = f"{pid}  {passage.get('source_id', '?')}"
        section = passage.get("section")
        if section:
            # Shown because attribution is one of the three traps in the
            # prompt: a Related Work passage describes someone else's work.
            header += f"  [{section}]"
        if is_contradicted:
            header += "  [CONTRADICTED BY OTHER SOURCES]"
        text = " ".join((passage.get("text") or "").split())
        return f"{header}\n{text[:MAX_PASSAGE_CHARS]}"

    # -- plumbing -----------------------------------------------------------

    def _finding(
        self,
        sentence: Sentence,
        verdict: str,
        reasoning: str,
        evidence: dict[str, Passage],
        sub_question_id: str,
    ) -> ClaimVerification:
        return ClaimVerification(
            claim=sentence.text,
            sub_question_id=sub_question_id,
            cited_passage_ids=list(sentence.citations),
            verdict=verdict,  # type: ignore[typeddict-item]
            reasoning=reasoning,
            action_taken="",   # filled by the five-way routing at C24
        )

    def _charge(self) -> None:
        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

    def _trace(self, report: CriticReport, sentences: int) -> None:
        if self.tracer is None:
            return
        self.tracer.note(
            "critic",
            sentences=sentences,
            claims=report.claims,
            discourse=report.discourse_sentences,
            supported=report.count(SUPPORTED),
            partial=report.count(PARTIAL),
            unsupported=report.count(UNSUPPORTED),
            contradicted=report.count(CONTRADICTED),
            grounding_precision=(
                round(report.grounding_precision, 4)
                if report.grounding_precision is not None else None
            ),
        )


def critic_node(critic: Critic):
    """LangGraph node: draft -> state['critic_findings']."""

    def node(state: dict) -> dict:
        report = critic.review(state)  # type: ignore[arg-type]
        update: dict[str, Any] = {"critic_findings": report.findings}
        if report.error:
            update["error"] = report.error
        return update

    return node


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment(draft: str) -> list[Sentence]:
    """Split a draft into sentences, extracting citations.

    Regex rather than a parser: the drafts are plain prose with bracketed
    ids, and spaCy would add a 50MB dependency to split on full stops. The
    known weakness is abbreviations ("et al. showed"), which produce a short
    fragment — harmless, because short fragments are classified as discourse
    rather than scored as claims.
    """
    pieces: list[str] = []
    for raw in SENTENCE_SPLIT.split(draft or ""):
        text = " ".join(raw.split())
        if not text:
            continue
        # A citation written AFTER the full stop — "…reaches 73%. [P1a2b…]"
        # — would otherwise be split onto the following sentence, crediting
        # the wrong claim and leaving the real one uncited. Models produce
        # both orders, so a leading citation is re-attached to the sentence
        # it actually supports.
        leading = LEADING_CITATION.match(text)
        if leading and pieces:
            pieces[-1] = f"{pieces[-1]} {leading.group(0).strip()}"
            text = text[leading.end():].strip()
            if not text:
                continue
        pieces.append(text)

    return [
        Sentence(
            index=i,
            text=text,
            citations=list(dict.fromkeys(CITATION_RE.findall(text))),
        )
        for i, text in enumerate(pieces)
    ]


def contradicted_passages(contradictions: Sequence[dict[str, Any]]) -> set[str]:
    """Passage ids involved in a cross-source disagreement (C12).

    A claim citing one of these picked a side; the prompt asks for PARTIAL
    unless the disagreement is acknowledged.
    """
    out: set[str] = set()
    for record in contradictions:
        if not record.get("cross_source", True):
            continue
        for key in ("passage_a", "passage_b"):
            pid = record.get(key)
            if pid:
                out.add(str(pid))
    return out


def _owner(sentence: Sentence, evidence: dict[str, Passage]) -> str:
    """Which sub-question this claim belongs to, via its first citation."""
    for pid in sentence.citations:
        passage = evidence.get(pid)
        if passage and passage.get("sub_question_id"):
            return str(passage["sub_question_id"])
    return ""