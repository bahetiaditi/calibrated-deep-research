"""Tests for chunking.

The property that matters most: no chunk crosses a section boundary, and
every chunk knows which section it came from. A system that cannot tell
Related Work from Results will misattribute findings fluently — the citation
resolves, the passage really is in the PDF, and the claim is still wrong.
"""
import pytest

from src.rag.chunking import (
    FRONT_MATTER,
    NON_EVIDENCE_SECTIONS,
    Chunk,
    canonicalize_section,
    chunk_paper,
    chunk_web,
    dehyphenate,
    detect_sections,
    sections_present,
    sliding_window,
)

PAPER = """Attention Is All You Need
Ashish Vaswani, Noam Shazeer

Abstract
The dominant sequence transduction models are based on complex recurrent networks.

1 Introduction
Recurrent neural networks have been firmly established as state of the art.

2 Related Work
The goal of reducing sequential computation forms the foundation of the Extended
Neural GPU. Convolutional approaches were explored by prior authors.

3 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder
structure.

4 Results
Our model achieves 28.4 BLEU on the WMT 2014 English-to-German task.

References
[1] Some Author. Some Paper Title. In Proceedings, 2015.
"""


# --- text repair -----------------------------------------------------------


def test_dehyphenation_rejoins_split_words():
    """BM25 breaks outright without this: 'attention' stops being a token in
    a paper entirely about attention."""
    assert "attention" in dehyphenate("we study atten-\ntion mechanisms")


def test_dehyphenation_handles_unicode_hyphens():
    assert "multihead" in dehyphenate("multi\u2010\nhead")


def test_real_hyphenated_words_survive():
    assert "state-of-the-art" in dehyphenate("state-of-the-art results")


# --- canonicalisation ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("2 Related Work", "Related Work"),
    ("RELATED WORK", "Related Work"),
    ("Prior Work", "Related Work"),
    ("4.1 Results", "Results"),
    ("III. Experiments", "Experiments"),
    ("Methods", "Method"),
    ("Ablation Studies", "Ablation"),
    ("Conclusion and Future Work", "Conclusion"),
    ("Bibliography", "References"),
    ("5 Experiments and Ablations", "Experiments"),
])
def test_canonicalisation(raw, expected):
    assert canonicalize_section(raw) == expected


def test_unknown_heading_returns_none():
    """An unrecognised heading is still a boundary, but inventing a label
    would let the retriever filter on a category that means nothing."""
    assert canonicalize_section("7 Widget Calibration Protocol") is None


# --- section detection -----------------------------------------------------


def test_sections_detected_in_order():
    names = [name for name, _ in detect_sections(PAPER)]
    assert names[0] == FRONT_MATTER
    assert names[1:] == ["Abstract", "Introduction", "Related Work",
                         "Method", "Results"]


def test_references_dropped_by_default():
    """References is dense with high-IDF tokens and would dominate BM25."""
    names = [name for name, _ in detect_sections(PAPER)]
    assert "References" not in names
    assert NON_EVIDENCE_SECTIONS == {"References", "Acknowledgments"}


def test_references_retained_when_requested():
    names = [n for n, _ in detect_sections(PAPER, drop_non_evidence=False)]
    assert "References" in names


def test_body_sentence_starting_with_a_number_is_not_a_header():
    text = "1 Introduction\nWe show results.\n3. We then measure throughput.\nMore text."
    names = [n for n, _ in detect_sections(text)]
    # No front matter here (the text opens with the header), so exactly one
    # section: the numbered body sentence must not have created a second.
    assert names == ["Introduction"]
    body = dict(detect_sections(text))["Introduction"]
    assert "throughput" in body


def test_page_numbers_are_stripped():
    text = "1 Introduction\nSome content here.\n7\nMore content.\n"
    body = dict(detect_sections(text))["Introduction"]
    assert " 7 " not in f" {body} "


def test_text_before_first_header_is_front_matter():
    sections = dict(detect_sections("Title line\nAuthors\n\n1 Introduction\nBody."))
    assert "Title line" in sections[FRONT_MATTER]


# --- the core invariant ----------------------------------------------------


def test_no_chunk_crosses_a_section_boundary():
    """The C8 acceptance criterion."""
    chunks = chunk_paper(PAPER, size_chars=200, overlap_chars=40)
    related = " ".join(c.text for c in chunks if c.section == "Related Work")
    results = " ".join(c.text for c in chunks if c.section == "Results")
    assert "Extended" in related and "BLEU" not in related
    assert "BLEU" in results and "Extended" not in results


def test_every_chunk_carries_its_section():
    for chunk in chunk_paper(PAPER, size_chars=200, overlap_chars=40):
        assert chunk.section is not None


def test_sections_present_helper():
    chunks = chunk_paper(PAPER, size_chars=300, overlap_chars=50)
    assert sections_present(chunks)[:3] == [FRONT_MATTER, "Abstract", "Introduction"]


def test_chunk_indices_are_unique_and_ordered():
    chunks = chunk_paper(PAPER, size_chars=150, overlap_chars=30)
    assert [c.index for c in chunks] == list(range(len(chunks)))


# --- sliding window --------------------------------------------------------


def test_short_text_is_one_chunk():
    assert sliding_window("short.", size_chars=100, overlap_chars=20) == ["short."]


def test_long_text_is_split():
    text = "This is a sentence. " * 60
    chunks = sliding_window(text, size_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_chunks_do_not_cut_mid_sentence():
    """A chunk cut mid-sentence reads as a fragment to a cross-encoder and
    to a critic asked whether it supports a claim."""
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    for chunk in sliding_window(text, size_chars=40, overlap_chars=10):
        assert chunk.endswith((".", "!", "?"))


def test_overlap_carries_context_forward():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = sliding_window(text, size_chars=45, overlap_chars=20)
    assert len(chunks) > 1
    assert any(
        set(a.split()) & set(b.split())
        for a, b in zip(chunks, chunks[1:])
    )


def test_sentence_longer_than_window_is_hard_split_not_dropped():
    monster = "word " * 400
    chunks = sliding_window(monster, size_chars=200, overlap_chars=50)
    assert chunks and all(len(c) <= 200 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        sliding_window("text", size_chars=100, overlap_chars=100)


def test_empty_text_yields_no_chunks():
    assert sliding_window("   ", size_chars=100, overlap_chars=10) == []


def test_whitespace_is_normalised():
    assert "  " not in sliding_window("a\n\n  b\tc", size_chars=100,
                                      overlap_chars=10)[0]


# --- web -------------------------------------------------------------------


def test_web_chunks_have_no_section():
    for chunk in chunk_web("Some text. " * 200, size_chars=200, overlap_chars=40):
        assert chunk.section is None


def test_chunk_dataclass_is_immutable():
    with pytest.raises(Exception):
        Chunk(text="a", section=None, index=0).text = "b"