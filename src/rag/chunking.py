"""Chunking.

The idea that earns its place
-----------------------------
A claim sourced from a paper's **Related Work** section describes *someone
else's* contribution. A claim from **Results** is that paper's own
measurement. A retrieval system that cannot tell them apart will confidently
attribute the wrong result to the wrong paper, and it will do so fluently —
the citation resolves, the passage really is in that PDF, and the claim is
still wrong.

So chunks never cross a section boundary, and every chunk carries the section
it came from. Downstream this lets the critic weigh an abstract's summary
claim differently from a measured result, and lets the retriever filter to
Results when a sub-question asks for a number (§4.4).

Two smaller decisions with outsized effects:

**Bibliographies are dropped.** A chunk of References is never evidence, but
it is dense with exactly the tokens (author names, paper titles, venues) that
make BM25 rank it highly. Keeping it would poison sparse retrieval at C10.

**Hyphenation is repaired before anything else.** pypdf preserves the line
breaks of a justified two-column layout, so "atten-\ntion" arrives as two
fragments. Dense retrieval degrades quietly; BM25 breaks outright, because
"attention" no longer appears as a token in a paper that is entirely about
attention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Canonical section names. Variants seen in the wild map onto these so that
# "2.1 RELATED WORK", "Prior Work" and "Related work" are one section, not
# three. The canonical form is what ends up in Passage.section and what the
# retriever filters on.
SECTION_ALIASES: dict[str, str] = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "intro": "Introduction",
    "background": "Background",
    "preliminaries": "Background",
    "related work": "Related Work",
    "related works": "Related Work",
    "prior work": "Related Work",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "our approach": "Method",
    "model": "Method",
    "architecture": "Method",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "experimental settings": "Experiments",
    "setup": "Experiments",
    "evaluation": "Experiments",
    "results": "Results",
    "results and discussion": "Results",
    "main results": "Results",
    "findings": "Results",
    "analysis": "Analysis",
    "ablation": "Ablation",
    "ablations": "Ablation",
    "ablation study": "Ablation",
    "ablation studies": "Ablation",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "limitations and future work": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion",
    "future work": "Conclusion",
    "references": "References",
    "bibliography": "References",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "appendix": "Appendix",
    "supplementary material": "Appendix",
}

# Sections whose chunks are never evidence. References in particular is dense
# with high-IDF tokens and would dominate BM25 if retained.
NON_EVIDENCE_SECTIONS = frozenset({"References", "Acknowledgments"})

# Text appearing before any detected header.
FRONT_MATTER = "Front Matter"

_NUMBERED_HEADER = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Za-z][\w \-&:,]{2,70})\s*$")
_ROMAN_HEADER = re.compile(r"^\s*([IVXL]+)\.?\s+([A-Za-z][\w \-&:,]{2,70})\s*$")
_PAGE_NOISE = re.compile(r"^\s*(?:\d{1,4}|[ivxlIVXL]{1,6}|Page \d+.*)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str | None
    index: int


# ---------------------------------------------------------------------------
# Text repair
# ---------------------------------------------------------------------------


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break by justification.

    Must run before any other processing: every later step assumes words are
    whole. See the module docstring for why this matters more than it looks.
    """
    return re.sub(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)", r"\1\2", text)


def _is_noise(line: str) -> bool:
    """Page numbers and running heads left behind by PDF extraction."""
    return bool(_PAGE_NOISE.match(line))


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


def canonicalize_section(raw: str) -> str | None:
    """Map a detected heading onto a canonical section name, or None.

    Returning None for an unrecognised heading is deliberate: an unknown
    heading is still a real section boundary worth splitting on, but
    inventing a canonical label for it would let the retriever filter on a
    category that means nothing.
    """
    cleaned = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXL]+)\.?\s*", "", raw or "").strip()
    cleaned = cleaned.strip(":.- \t").lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None
    if cleaned in SECTION_ALIASES:
        return SECTION_ALIASES[cleaned]
    # "5 Experiments and Results" -> match the leading known phrase.
    for alias, canonical in SECTION_ALIASES.items():
        if cleaned.startswith(alias + " ") or cleaned == alias:
            return canonical
    return None


def _looks_like_header(line: str, canonical_only: bool) -> tuple[bool, str | None]:
    """Decide whether a line is a section heading.

    Two acceptance paths, because papers are inconsistent:
      - numbered or roman ("3 Method", "III. RESULTS")
      - a bare canonical name on its own line ("Abstract", "REFERENCES")

    A trailing period or comma disqualifies a line: real headings do not end
    in sentence punctuation, and this is what stops a body sentence beginning
    "3. We then measure..." from splitting the document.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False, None
    if stripped.endswith((".", ",", ";", ":")) and not _NUMBERED_HEADER.match(stripped):
        return False, None

    match = _NUMBERED_HEADER.match(stripped) or _ROMAN_HEADER.match(stripped)
    if match:
        canonical = canonicalize_section(stripped)
        if canonical_only and canonical is None:
            # Numbered but unrecognised: still a boundary, no canonical label.
            return True, None
        return True, canonical

    canonical = canonicalize_section(stripped)
    if canonical is not None and len(stripped.split()) <= 6:
        return True, canonical
    return False, None


def detect_sections(
    text: str, *, drop_non_evidence: bool = True
) -> list[tuple[str | None, str]]:
    """Split extracted text into (canonical_section, body) pairs.

    Text before the first heading becomes `Front Matter` — usually title,
    authors and abstract on papers whose abstract carries no heading.
    """
    text = dehyphenate(text)
    sections: list[tuple[str | None, list[str]]] = [(FRONT_MATTER, [])]

    for line in text.splitlines():
        if _is_noise(line):
            continue
        is_header, canonical = _looks_like_header(line, canonical_only=True)
        if is_header:
            sections.append((canonical, []))
        else:
            sections[-1][1].append(line)

    out: list[tuple[str | None, str]] = []
    for name, lines in sections:
        body = " ".join(" ".join(lines).split())
        if not body:
            continue
        if drop_non_evidence and name in NON_EVIDENCE_SECTIONS:
            continue
        out.append((name, body))
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def sliding_window(
    text: str, *, size_chars: int, overlap_chars: int, sentence_aware: bool = True
) -> list[str]:
    """Chunk a single block of text.

    Sentence-aware by default: cutting mid-sentence produces chunks that
    read as fragments to a cross-encoder reranker and to a critic asked
    whether a passage supports a claim.
    """
    # Validate before the early returns: a bad config must fail on the first
    # call, not silently pass until the first long document arrives.
    if overlap_chars >= size_chars:
        raise ValueError("overlap_chars must be smaller than size_chars")

    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= size_chars:
        return [text]

    if not sentence_aware:
        step = size_chars - overlap_chars
        return [text[i : i + size_chars] for i in range(0, len(text), step)]

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sentence in sentences:
        # A single sentence longer than the window gets hard-split rather
        # than silently dropped or allowed to blow past the size cap.
        if len(sentence) > size_chars:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            step = size_chars - overlap_chars
            chunks.extend(
                sentence[i : i + size_chars] for i in range(0, len(sentence), step)
            )
            continue

        if length + len(sentence) + 1 > size_chars and current:
            chunks.append(" ".join(current))
            # Carry back whole sentences worth roughly `overlap_chars`.
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap_chars:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current = carry
            length = carried

        current.append(sentence)
        length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def chunk_paper(
    text: str,
    *,
    size_chars: int = 1000,
    overlap_chars: int = 150,
    drop_non_evidence: bool = True,
) -> list[Chunk]:
    """Section-aware chunking. No chunk ever spans two sections."""
    chunks: list[Chunk] = []
    index = 0
    for section, body in detect_sections(text, drop_non_evidence=drop_non_evidence):
        for piece in sliding_window(
            body, size_chars=size_chars, overlap_chars=overlap_chars
        ):
            chunks.append(Chunk(text=piece, section=section, index=index))
            index += 1
    return chunks


def chunk_web(
    text: str, *, size_chars: int = 800, overlap_chars: int = 150
) -> list[Chunk]:
    """Web pages have no reliable section structure, so `section` is None."""
    return [
        Chunk(text=piece, section=None, index=i)
        for i, piece in enumerate(
            sliding_window(text, size_chars=size_chars, overlap_chars=overlap_chars)
        )
    ]


def sections_present(chunks: Iterable[Chunk]) -> list[str]:
    """Ordered unique section names — used by the C8 acceptance check."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.section and chunk.section not in seen:
            seen.append(chunk.section)
    return seen