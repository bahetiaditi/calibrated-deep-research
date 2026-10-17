"""Premise checker — the cleanest abstention signal available (§3.3).

Runs *before* planning. It extracts what a question takes for granted,
retrieves against each presupposition, and judges whether the corpus supports
it. "Why does Mamba-2 use rotary position embeddings in its SSM blocks?"
presupposes that Mamba-2 uses rotary position embeddings. It does not. A
system that checks this first abstains in one retrieval round; a system that
does not discovers the problem after writing three paragraphs about a
mechanism that was never there — if it discovers it at all.

The decision this node turns on
-------------------------------
What counts as a *failed* premise. There are three outcomes, not two, and
collapsing them is the mistake that would wreck the project's headline
metric:

  SUPPORTED     the corpus confirms it
  CONTRADICTED  the corpus positively says otherwise
  UNVERIFIABLE  the corpus is silent

Only **CONTRADICTED** triggers the hard abstention rule in §5.4. Treating
UNVERIFIABLE as failure would make the system refuse every question whose
presuppositions the corpus happens not to mention — which is most of them,
since a paper does not restate its own premises. The false-abstention rate on
Tiers A-C (§5.5) would be catastrophic, and the project would report a
beautiful false-answer rate achieved by answering almost nothing.

Absence of evidence is not evidence of absence, and here that cliché has a
measurable cost attached. The verification prompt says so explicitly and the
tie-break goes to UNVERIFIABLE.

Cost
----
Two LLM calls per question regardless of how many presuppositions are found:
one to extract, one to verify them all together. Verifying one at a time
would be three to five calls against a budget of roughly nine (§7.1), which
would make the premise check cost more than the research it guards. The same
batching lesson C23 has to apply to the critic.

Feature f7
----------
`grounding_score` is §5.2's premise-grounding feature. It is computed over
*core* presuppositions only — an incidental one being unverifiable says
little about whether the question is well-posed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.config import Config, get_config
from src.llm.provider import LLMProvider, Role
from src.llm.structured import StructuredOutputError
from src.prompts import render
from src.state import Passage

log = logging.getLogger(__name__)

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
UNVERIFIABLE = "UNVERIFIABLE"
VALID_VERDICTS = (SUPPORTED, CONTRADICTED, UNVERIFIABLE)

# Grounding contributed by each verdict, for feature f7.
# UNVERIFIABLE scores at the midpoint rather than zero: silence is genuinely
# uninformative, and scoring it as a failure would make f7 a proxy for
# "corpus coverage" instead of "is this question well-posed".
_GROUNDING = {SUPPORTED: 1.0, UNVERIFIABLE: 0.5, CONTRADICTED: 0.0}

MAX_PRESUPPOSITIONS = 4
EVIDENCE_PER_PRESUPPOSITION = 4
MAX_EVIDENCE_CHARS = 700


@dataclass
class Presupposition:
    text: str
    centrality: str = "core"           # core | incidental
    verdict: str = UNVERIFIABLE
    reasoning: str = ""
    correction: str = ""
    cited_passage_ids: list[str] = field(default_factory=list)

    @property
    def is_core(self) -> bool:
        return self.centrality != "incidental"


@dataclass
class PremiseCheckResult:
    """What the node writes into `state["premise_check"]`."""

    ran: bool = False
    presuppositions: list[Presupposition] = field(default_factory=list)
    grounding_score: float = 1.0       # feature f7
    failed: bool = False               # True only on a CONTRADICTED core claim
    correction: str = ""               # what is actually the case
    rationale: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["presuppositions"] = [asdict(p) for p in self.presuppositions]
        return data

    @property
    def contradicted(self) -> list[Presupposition]:
        return [p for p in self.presuppositions if p.verdict == CONTRADICTED]


class PremiseChecker:
    def __init__(
        self,
        config: Config | None = None,
        *,
        provider: LLMProvider | None = None,
        retriever: Any = None,
        reranker: Any = None,
        tracer: Any = None,
        budget: Any = None,
    ) -> None:
        self.cfg = config or get_config()
        self.provider = provider or LLMProvider(self.cfg, tracer=tracer)
        self._retriever = retriever
        self._reranker = reranker
        self.tracer = tracer
        self.budget = budget
        self.grounding_min = float(
            self.cfg.get("sufficiency.hard_rules.premise_grounding_min", 0.35)
        )

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from src.rag.fusion import HybridRetriever

            self._retriever = HybridRetriever(self.cfg, tracer=self.tracer)
        return self._retriever

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            from src.rag.rerank import Reranker

            self._reranker = Reranker(self.cfg, tracer=self.tracer)
        return self._reranker

    # -- main entry point ---------------------------------------------------

    def check(self, question: str) -> PremiseCheckResult:
        question = (question or "").strip()
        if not question:
            return PremiseCheckResult(ran=False, rationale="empty question")

        try:
            presuppositions = self._extract(question)
        except (StructuredOutputError, Exception) as exc:  # noqa: BLE001
            # A failed premise check must not fail the run. Degrading to
            # "no premise information" costs the hard rule, not the question:
            # the calibrated threshold still governs the decision.
            log.warning("presupposition extraction failed: %s", exc)
            return PremiseCheckResult(
                ran=False, rationale="extraction failed", error=str(exc)
            )

        if not presuppositions:
            # Open questions genuinely presuppose nothing checkable. Scoring
            # them as ungrounded would penalise exactly the Tier C questions
            # the system should answer well.
            result = PremiseCheckResult(
                ran=True, presuppositions=[], grounding_score=1.0,
                rationale="no checkable presuppositions",
            )
            self._trace(result)
            return result

        evidence = self._gather(presuppositions)
        if not evidence:
            result = PremiseCheckResult(
                ran=True,
                presuppositions=presuppositions,
                grounding_score=_mean(
                    _GROUNDING[UNVERIFIABLE] for _ in presuppositions
                ),
                rationale="no evidence retrieved for any presupposition",
            )
            self._trace(result)
            return result

        try:
            self._verify(question, presuppositions, evidence)
        except Exception as exc:  # noqa: BLE001
            log.warning("presupposition verification failed: %s", exc)
            result = PremiseCheckResult(
                ran=False, presuppositions=presuppositions,
                rationale="verification failed", error=str(exc),
            )
            self._trace(result)
            return result

        return self._summarise(presuppositions)

    # -- steps --------------------------------------------------------------

    def _extract(self, question: str) -> list[Presupposition]:
        parsed, _ = self.provider.complete_json(
            system=render("premise_extract"),
            user=f"QUESTION\n{question}",
            role=Role.JUDGMENT,
            required_keys=["presuppositions"],
        )
        self._charge_llm()

        out: list[Presupposition] = []
        for raw in parsed.get("presuppositions") or []:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            centrality = str(raw.get("centrality") or "core").lower()
            out.append(Presupposition(
                text=text,
                centrality="incidental" if centrality == "incidental" else "core",
            ))
        return out[:MAX_PRESUPPOSITIONS]

    def _gather(
        self, presuppositions: Sequence[Presupposition]
    ) -> list[tuple[str, Passage]]:
        """Retrieve evidence for each presupposition, labelled for citation.

        Labels are P01, P02… rather than the content-addressed passage ids:
        a model asked to echo `Pa3f9c2b1d004` transcribes it wrong, and a
        citation that does not resolve is indistinguishable from a
        fabricated one. The same mapping is what C18 needs for the
        synthesizer.
        """
        labelled: list[tuple[str, Passage]] = []
        seen: set[str] = set()

        for presupposition in presuppositions:
            try:
                results = self.retriever.retrieve(
                    presupposition.text, top_k=EVIDENCE_PER_PRESUPPOSITION * 3
                )
                results = self.reranker.rerank(
                    presupposition.text, results,
                    top_k=EVIDENCE_PER_PRESUPPOSITION,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("premise retrieval failed for %r: %s",
                            presupposition.text[:60], exc)
                continue
            self._charge_retrieval()

            for result in results:
                passage = result.passage
                if passage["id"] in seen:
                    continue
                seen.add(passage["id"])
                labelled.append((f"P{len(labelled) + 1:02d}", passage))
        return labelled

    def _verify(
        self,
        question: str,
        presuppositions: list[Presupposition],
        evidence: list[tuple[str, Passage]],
    ) -> None:
        claims = "\n".join(
            f"[{i}] {p.text}" for i, p in enumerate(presuppositions)
        )
        rendered = "\n\n".join(
            f"{label} ({p.get('source_id', '?')}"
            f"{', ' + p['section'] if p.get('section') else ''})\n"
            f"{(p.get('text') or '')[:MAX_EVIDENCE_CHARS]}"
            for label, p in evidence
        )
        label_to_id = {label: p["id"] for label, p in evidence}

        parsed, _ = self.provider.complete_json(
            system="You are a precise evidence checker. Output only JSON.",
            user=render(
                "premise_verify",
                question=question, claims=claims, evidence=rendered,
            ),
            role=Role.JUDGMENT,
            required_keys=["verdicts"],
        )
        self._charge_llm()

        for raw in parsed.get("verdicts") or []:
            if not isinstance(raw, dict):
                continue
            try:
                index = int(raw.get("claim_index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(presuppositions):
                continue

            verdict = str(raw.get("verdict") or "").upper().strip()
            if verdict not in VALID_VERDICTS:
                # An unparseable verdict must not become a contradiction.
                verdict = UNVERIFIABLE

            target = presuppositions[index]
            target.verdict = verdict
            target.reasoning = str(raw.get("reasoning") or "")[:400]
            target.correction = str(raw.get("correction") or "")[:400]
            target.cited_passage_ids = [
                label_to_id[str(c)]
                for c in (raw.get("cited") or [])
                if str(c) in label_to_id
            ]

    def _summarise(self, presuppositions: list[Presupposition]) -> PremiseCheckResult:
        core = [p for p in presuppositions if p.is_core] or presuppositions
        grounding = _mean(_GROUNDING.get(p.verdict, 0.5) for p in core)

        # Only a CONTRADICTED core presupposition fails the check. See the
        # module docstring: UNVERIFIABLE is silence, not refutation.
        failures = [p for p in core if p.verdict == CONTRADICTED]
        correction = next((p.correction for p in failures if p.correction), "")

        # The rationale is surfaced in the decision explanation and quoted in
        # case studies, so it must distinguish "verified true" from "not
        # refuted". Reporting all-UNVERIFIABLE as "grounded" would be a small
        # dishonesty that propagates into the writeup.
        confirmed = sum(1 for p in core if p.verdict == SUPPORTED)
        silent = sum(1 for p in core if p.verdict == UNVERIFIABLE)

        if failures:
            rationale = (
                f"false premise: {failures[0].text}"
                + (f" — {correction}" if correction else "")
            )
        elif confirmed == 0:
            rationale = (
                f"{silent} presupposition(s) unverified — the corpus is "
                f"silent, not contradictory (f7={grounding:.2f})"
            )
        elif silent:
            rationale = (
                f"{confirmed} of {len(core)} presuppositions confirmed, "
                f"{silent} unverified (f7={grounding:.2f})"
            )
        else:
            rationale = f"all {confirmed} presuppositions confirmed"

        if grounding < self.grounding_min and not failures:
            rationale += (
                f"; grounding below the {self.grounding_min:.2f} floor"
            )

        result = PremiseCheckResult(
            ran=True,
            presuppositions=presuppositions,
            grounding_score=grounding,
            failed=bool(failures),
            correction=correction,
            rationale=rationale,
        )
        self._trace(result)
        return result

    # -- plumbing -----------------------------------------------------------

    def _charge_llm(self) -> None:
        if self.budget is not None and self.budget.can_spend("llm"):
            self.budget.spend("llm", 1)

    def _charge_retrieval(self) -> None:
        if self.budget is not None and self.budget.can_spend("retrieval"):
            self.budget.spend("retrieval", 1)

    def _trace(self, result: PremiseCheckResult) -> None:
        if self.tracer is None:
            return
        self.tracer.note(
            "premise_check",
            ran=result.ran,
            failed=result.failed,
            grounding_score=round(result.grounding_score, 4),
            n_presuppositions=len(result.presuppositions),
            verdicts=[p.verdict for p in result.presuppositions],
            rationale=result.rationale[:300],
        )


def premise_checker_node(checker: PremiseChecker):
    """LangGraph node: question -> state['premise_check']."""

    def node(state: dict) -> dict:
        result = checker.check(state.get("question", ""))
        return {"premise_check": result.to_dict()}

    return node


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 1.0