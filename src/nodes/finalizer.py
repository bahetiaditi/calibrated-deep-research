"""The finalizer — the report a reader sees (C28).

Deterministic: no model call. Everything in the output is either the draft
the critic reviewed or a template filled from the decision record, so the
report cannot say anything the pipeline did not establish or decide. §2.5
lists report formatting among the things deliberately NOT agentic.

What it does with each decision
-------------------------------
  ANSWER   the reviewed draft, with each claim's verification applied
  PARTIAL  the reviewed draft, then what could not be established
  ABSTAIN  no draft — the reason, the premise correction if there is one,
           and what could not be established. A withheld draft is never
           shown: it lives in the decision record for C38, not here.

How the critic's verdicts reach the text
----------------------------------------
Each claim the critic judged is matched back to its sentence, and the
router's action is applied: DROP_CLAIM removes the sentence; anything less
than SUPPORTED that stays is marked in place ("partially supported",
"unverified", "contradicted by its source"). A reader can then see which
sentences rest on verified evidence without consulting a separate table —
and the verification summary counts exactly the marks in the text.

Citations become numbered footnotes in order of first use; the sources list
holds only passages actually cited in what remains.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config import Config, get_config
from src.nodes.critic import (
    CITATION_RE,
    CONTRADICTED,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    segment,
)
from src.nodes.critic_router import DROP_CLAIM, HEDGE
from src.state import ResearchState

log = logging.getLogger(__name__)

# Markers a model writes that are not citations: "[unresolved]",
# "[citation needed]". Lower-case words only, so a citation ([P…]) or a
# footnote ([3]) can never match.
STRAY_MARKER_RE = re.compile(r"\s*\[(?:[a-z][a-z\- ]{2,30})\]")

MARK = {
    PARTIAL: "*[partially supported]*",
    UNSUPPORTED: "*[unverified]*",
    CONTRADICTED: "*[contradicted by its cited source]*",
}

RULE_BANNER = {
    "premise_refuted": "NO ANSWER — the question rests on a premise the evidence contradicts",
    "system_failure": "NO ANSWER — the run failed before a report could be written",
    "no_evidence": "NO ANSWER — no evidence was found",
    "static_always_answer": "ANSWERED (abstention disabled for this run)",
}

BANNER = {
    "ANSWER": "ANSWERED",
    "PARTIAL": "PARTIAL ANSWER — some parts of the question could not be established",
    "ABSTAIN": "NO ANSWER — the evidence does not support one",
}


@dataclass
class Rendered:
    text: str
    footnotes: list[str] = field(default_factory=list)   # passage ids, in order
    dropped: int = 0
    marked: dict[str, int] = field(default_factory=dict)
    stray_markers: int = 0


class Finalizer:
    def __init__(self, config: Config | None = None, *, tracer: Any = None) -> None:
        self.cfg = config or get_config()
        self.tracer = tracer

    def finalize(self, state: ResearchState) -> str:
        decision = state.get("terminal_decision") or "ABSTAIN"
        record = state.get("decision") or {}
        lines = [f"# {state.get('question', '').strip()}", ""]
        lines += self._banner(decision, record)

        rendered = None
        if decision in ("ANSWER", "PARTIAL") and (state.get("draft") or "").strip():
            rendered = self._render_draft(state)
            heading = ("## Answer" if decision == "ANSWER"
                       else "## What the evidence establishes")
            lines += [heading, "", rendered.text, ""]
            if record.get("partial_rewrite") == "unaffordable":
                lines += [
                    "> This report was written to answer the whole question; "
                    "the budget did not allow it to be rewritten as a partial "
                    "answer. Sentences marked *[unverified]* should not be "
                    "relied on.", "",
                ]
        elif decision in ("ANSWER", "PARTIAL"):
            lines += ["## Answer", "", "_No report could be written._", ""]

        if decision == "ABSTAIN":
            lines += self._abstention(state, record)

        gaps = self._question_gaps(state, record)
        if gaps and decision != "ABSTAIN":
            lines += ["## What could not be established", ""]
            lines += [f"- {g}" for g in gaps] + [""]

        if rendered is not None:
            lines += self._verification(state, rendered)
            lines += self._sources(state, rendered.footnotes)

        report = "\n".join(lines).rstrip() + "\n"
        self._trace(decision, rendered, len(gaps))
        return report

    # -- sections -----------------------------------------------------------

    def _banner(self, decision: str, record: dict[str, Any]) -> list[str]:
        rule = record.get("rule")
        banner = RULE_BANNER.get(rule) or BANNER.get(decision, decision)
        out = [f"> **{banner}**"]
        if rule not in (None, "threshold"):
            # A hard rule decided, not the score: showing a score or the
            # calibration status here would imply the scorer made a decision
            # that bypassed it entirely.
            out.append(f"> Decided by rule: `{rule}`")
            return out + [""]
        if record.get("score") is not None:
            out.append(f"> Evidence-sufficiency score {record['score']:.2f} "
                       f"({record.get('scorer', '?')})")
        if record and not record.get("calibrated"):
            # Before C33 there is no fitted model; a reader must not mistake
            # the heuristic's decision for the calibrated layer's.
            out.append("> _Uncalibrated: decided by the heuristic scorer, not "
                       "a fitted model._")
        return out + [""]

    def _abstention(self, state: ResearchState, record: dict[str, Any]) -> list[str]:
        out = ["## Why there is no answer", ""]
        rule = record.get("rule")
        premise = state.get("premise_check") or {}
        if rule == "premise_refuted":
            refuted = [p for p in premise.get("presuppositions") or []
                       if p.get("verdict") == "CONTRADICTED"]
            if refuted:
                out.append("The question assumes something the evidence contradicts:")
                out.append("")
                out += [f"- *{p.get('text', '').strip()}*" for p in refuted]
                out.append("")
            correction = (premise.get("correction") or "").strip()
            if correction:
                out += ["**What the evidence says instead:** " + correction, ""]
        elif rule == "system_failure":
            out += ["The run failed before a report could be written "
                    f"(`{(record.get('error') or state.get('error') or '')[:200]}`). "
                    "This is a system failure, not a judgment about the evidence.", ""]
        else:
            out += [state.get("decision_rationale") or "The evidence was insufficient.", ""]
        gaps = self._question_gaps(state, record)
        if gaps:
            out += ["**Not established:**", ""] + [f"- {g}" for g in gaps] + [""]
        return out

    def _question_gaps(self, state: ResearchState, record: dict[str, Any]) -> list[str]:
        # The decision's gap list is authoritative: question-level gaps only
        # since C27.5, core sub-questions only since C27.8.
        gaps = record.get("gaps")
        if gaps is None:
            gaps = state.get("unresolved_gaps") or []
        out = []
        for gap in gaps:
            text = gap.removeprefix("could not establish: ").strip()
            out.append(text if text[-1:] in ".?!" else text + ".")
        return out

    def _verification(self, state: ResearchState, r: Rendered) -> list[str]:
        findings = state.get("critic_findings") or []
        kept = [f for f in findings if f.get("action_taken") != DROP_CLAIM]
        supported = sum(1 for f in kept if f.get("verdict") == SUPPORTED)
        out = ["## Verification", ""]
        if not kept:
            out += ["No checkable claims were made.", ""]
            return out
        out.append(f"{supported} of {len(kept)} claims are supported by the "
                   f"passages they cite.")
        for verdict, label in ((PARTIAL, "partially supported"),
                               (UNSUPPORTED, "unverified"),
                               (CONTRADICTED, "contradicted by their cited source")):
            n = r.marked.get(verdict, 0)
            if n:
                out.append(f"{n} marked *{label}*.")
        if r.dropped:
            out.append(f"{r.dropped} claim(s) removed after verification "
                       f"found them wrong.")
        return out + [""]

    def _sources(self, state: ResearchState, footnotes: list[str]) -> list[str]:
        if not footnotes:
            return []
        evidence = {p["id"]: p for p in state.get("evidence") or []}
        out = ["## Sources", ""]
        for n, pid in enumerate(footnotes, 1):
            p = evidence.get(pid) or {}
            bits = [p.get("title") or "(untitled)", str(p.get("source_id") or pid)]
            if p.get("section"):
                bits.append(str(p["section"]))
            if p.get("published"):
                bits.append(str(p["published"])[:10])
            out.append(f"[{n}] " + " — ".join(bits))
        return out + [""]

    # -- the draft ----------------------------------------------------------

    def _render_draft(self, state: ResearchState) -> Rendered:
        draft = state.get("draft") or ""
        stray = len(STRAY_MARKER_RE.findall(draft))
        draft = STRAY_MARKER_RE.sub("", draft)

        by_claim: dict[str, dict[str, Any]] = {}
        for finding in state.get("critic_findings") or []:
            # Keyed with stray markers stripped, exactly as the draft is: the
            # critic saw "…pattern [unresolved]." and the text now reads
            # "…pattern." — unstripped, that sentence's verdict would silently
            # go unapplied.
            claim = STRAY_MARKER_RE.sub("", str(finding.get("claim", "")))
            by_claim.setdefault(" ".join(claim.split()), finding)

        footnotes: list[str] = []
        marked: dict[str, int] = {}
        dropped = 0
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", draft):
            kept_sentences = []
            for sentence in segment(paragraph):
                finding = by_claim.get(sentence.text)
                text = sentence.text
                if finding is not None:
                    if finding.get("action_taken") == DROP_CLAIM:
                        dropped += 1
                        continue
                    verdict = finding.get("verdict")
                    if verdict in MARK:
                        # HEDGE keeps a PARTIAL claim, which is marked anyway;
                        # the mark is what makes the hedge visible.
                        text = f"{text} {MARK[verdict]}"
                        marked[verdict] = marked.get(verdict, 0) + 1
                kept_sentences.append(text)
            if kept_sentences:
                paragraphs.append(" ".join(kept_sentences))

        def number(match: re.Match) -> str:
            pid = match.group(1)
            if pid not in footnotes:
                footnotes.append(pid)
            return f"[{footnotes.index(pid) + 1}]"

        text = CITATION_RE.sub(number, "\n\n".join(paragraphs))
        return Rendered(text=text, footnotes=footnotes, dropped=dropped,
                        marked=marked, stray_markers=stray)

    def _trace(self, decision: str, r: Rendered | None, gaps: int) -> None:
        if self.tracer is None:
            return
        self.tracer.note(
            "finalize",
            decision=decision,
            footnotes=len(r.footnotes) if r else 0,
            dropped=r.dropped if r else 0,
            marked=(r.marked if r else {}),
            stray_markers=r.stray_markers if r else 0,
            gaps=gaps,
        )


def finalizer_node(finalizer: Finalizer):
    """LangGraph node: decided state -> state['final_report']."""

    def node(state: dict) -> dict:
        return {"final_report": finalizer.finalize(state)}  # type: ignore[arg-type]

    return node
