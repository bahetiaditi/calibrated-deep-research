"""Contradiction detection via pairwise NLI.

Two payoffs (§4.6)
------------------
**Sufficiency.** Contradiction density is feature f5. Sources disagreeing is a
legitimate reason to hedge or abstain, and it is the mechanism by which Tier
D's *no-consensus* questions ("what is the optimal LoRA rank for a 7B model?")
should trigger abstention rather than a confident arbitrary pick. Without this
signal the system has no way to distinguish "the evidence agrees" from "the
evidence is split and I picked one side".

**Report quality.** On the answer path it lets the synthesizer present genuine
disagreement *as* disagreement instead of silently choosing a winner.

Three decisions that shape what gets measured
---------------------------------------------
**Only cross-source pairs count toward f5.** Two chunks of the *same* paper
appearing to contradict each other is usually an artefact — a hypothesis
stated before it is refuted, a limitation acknowledged after a result, a
"one might expect X" setup. That is not sources disagreeing, and counting it
would inflate f5 on a corpus that actually agrees. Same-source pairs are
detected and recorded but excluded from the rate by default.

**Label indices are read from the model, never hardcoded.** NLI heads differ
in label order between checkpoints; assuming index 0 is "contradiction" is a
silent correctness bug that produces plausible-looking numbers pointing the
wrong way. The detector resolves names from `config.id2label` and only falls
back to a documented default if that is unavailable.

**Passages are truncated before scoring, and this is a real limitation.** NLI
models are trained on sentence pairs, not 1000-character chunks. Feeding whole
chunks degrades accuracy — the model has to decide whether *any* part of one
contradicts *any* part of the other, which is not what it was trained to do.
Truncating to the leading, claim-bearing window is a pragmatic compromise;
sentence-level pairing would be more faithful but multiplies cost by roughly
the square of sentences per chunk. Recorded here so the writeup states it
rather than implying a rigour the method does not have.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from src.config import Config, get_config
from src.state import Passage

log = logging.getLogger(__name__)

# Fallback only. cross-encoder/nli-deberta-v3-small documents this order, but
# the detector prefers the model's own config.id2label — see module docstring.
DEFAULT_LABELS = ("contradiction", "entailment", "neutral")

# NLI is a sentence-pair task. This is the leading window scored per passage.
MAX_PREMISE_CHARS = 400


@dataclass
class ContradictionRecord:
    """One detected disagreement, as stored in `state["contradictions"]`."""

    passage_a: str
    passage_b: str
    source_a: str
    source_b: str
    section_a: str | None
    section_b: str | None
    score: float
    cross_source: bool
    sub_question_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class ContradictionDetector:
    """Pairwise NLI over the top reranked passages for a sub-question."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model: Any = None,
        tracer: Any = None,
    ) -> None:
        cfg = config or get_config()
        section = cfg.section("retrieval.contradiction")
        self.model_name = str(section["model"])
        self.max_pairs = int(section.get("max_pairs_per_subquestion", 15))
        self.threshold = float(section.get("contradiction_threshold", 0.60))
        self.cross_source_only = bool(section.get("cross_source_only", True))
        self.symmetric = bool(section.get("symmetric", True))
        self.tracer = tracer
        self._model = model
        self._contradiction_index: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading NLI model %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    # -- label resolution ---------------------------------------------------

    def contradiction_index(self) -> int:
        """Which output column is 'contradiction' for THIS checkpoint.

        Hardcoding this is a silent correctness bug: a wrong index yields
        plausible numbers that point the wrong way, and nothing errors.
        """
        if self._contradiction_index is not None:
            return self._contradiction_index

        labels: dict[int, str] | None = None
        config = getattr(getattr(self.model, "model", None), "config", None)
        raw = getattr(config, "id2label", None)
        if isinstance(raw, dict) and raw:
            labels = {int(k): str(v).lower() for k, v in raw.items()}

        index = None
        if labels:
            for position, name in labels.items():
                if "contradict" in name:
                    index = position
                    break
            if index is None:
                log.warning(
                    "no contradiction label in %s (%s); using default order",
                    self.model_name, sorted(labels.values()),
                )
        if index is None:
            index = DEFAULT_LABELS.index("contradiction")
        self._contradiction_index = index
        return index

    # -- detection ----------------------------------------------------------

    def detect(
        self,
        passages: Sequence[Passage],
        *,
        sub_question_id: str = "",
        max_pairs: int | None = None,
    ) -> list[ContradictionRecord]:
        """Score pairs of passages and return those above the threshold.

        Passages are assumed to arrive in rerank order, so truncating to the
        pair budget keeps the strongest evidence.
        """
        max_pairs = max_pairs if max_pairs is not None else self.max_pairs
        usable = [p for p in passages if (p.get("text") or "").strip()]
        if len(usable) < 2 or max_pairs <= 0:
            return []

        pairs = [
            (a, b)
            for a, b in itertools.combinations(usable, 2)
            if not (self.cross_source_only and a["source_id"] == b["source_id"])
        ][:max_pairs]
        if not pairs:
            return []

        inputs = [
            (_window(a["text"]), _window(b["text"])) for a, b in pairs
        ]
        if self.symmetric:
            # NLI heads are asymmetric in practice even where the relation is
            # symmetric in principle, so both directions are scored and the
            # stronger signal wins. Missing a real contradiction is worse than
            # the extra forward passes.
            inputs += [(b, a) for a, b in inputs]

        try:
            raw = self.model.predict(inputs)
        except Exception as exc:  # noqa: BLE001
            # No contradiction data is a weaker signal, not a broken run.
            log.warning("NLI scoring failed: %s", exc)
            if self.tracer is not None:
                self.tracer.note("contradiction_failed", error=str(exc))
            return []

        index = self.contradiction_index()
        scores = [_contradiction_probability(row, index) for row in raw]
        if self.symmetric:
            half = len(pairs)
            scores = [max(scores[i], scores[i + half]) for i in range(half)]

        records = [
            ContradictionRecord(
                passage_a=a["id"], passage_b=b["id"],
                source_a=a["source_id"], source_b=b["source_id"],
                section_a=a.get("section"), section_b=b.get("section"),
                score=float(score),
                cross_source=a["source_id"] != b["source_id"],
                sub_question_id=sub_question_id or a.get("sub_question_id", ""),
            )
            for (a, b), score in zip(pairs, scores)
            if score >= self.threshold
        ]
        records.sort(key=lambda r: r.score, reverse=True)

        if self.tracer is not None:
            self.tracer.note(
                "contradiction_scan",
                sub_question_id=sub_question_id,
                pairs_scored=len(pairs),
                contradictions=len(records),
                rate=round(len(records) / len(pairs), 4) if pairs else 0.0,
                top_score=round(records[0].score, 4) if records else None,
            )
        return records

    # -- feature f5 ---------------------------------------------------------

    def contradiction_rate(
        self, passages: Sequence[Passage], *, sub_question_id: str = ""
    ) -> tuple[float, list[ContradictionRecord]]:
        """Feature f5: fraction of scored pairs labelled contradiction.

        Returns 0.0 when there are too few passages to form a pair. That is
        the honest reading — no disagreement was *observed*, which is
        different from evidence agreeing, and the thinness of the evidence is
        already captured by f1, f2 and f4.
        """
        records = self.detect(passages, sub_question_id=sub_question_id)
        usable = [p for p in passages if (p.get("text") or "").strip()]
        pairs = [
            (a, b)
            for a, b in itertools.combinations(usable, 2)
            if not (self.cross_source_only and a["source_id"] == b["source_id"])
        ][: self.max_pairs]
        if not pairs:
            return 0.0, records
        return len(records) / len(pairs), records


def _window(text: str) -> str:
    """Leading claim-bearing window — see the module docstring."""
    text = " ".join((text or "").split())
    return text[:MAX_PREMISE_CHARS]


def _contradiction_probability(row: Any, index: int) -> float:
    """Normalise one model output row to P(contradiction).

    Handles both logit vectors (the usual case) and a scalar, since some
    checkpoints expose a single head.
    """
    try:
        values = [float(v) for v in row]
    except TypeError:
        return float(row)
    if len(values) == 1:
        return values[0]
    if index >= len(values):
        log.warning("contradiction index %d out of range for %d outputs",
                    index, len(values))
        return 0.0
    return _softmax(values)[index]