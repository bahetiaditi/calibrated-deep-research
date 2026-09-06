"""Tests for the passage schema and id scheme."""
from src.state import Passage, make_passage_id


def test_id_is_deterministic():
    a = make_passage_id("2312.00752v2", "some text", "Abstract")
    b = make_passage_id("2312.00752v2", "some text", "Abstract")
    assert a == b and a.startswith("P")


def test_id_changes_with_source():
    assert make_passage_id("a", "t") != make_passage_id("b", "t")


def test_id_changes_with_text():
    assert make_passage_id("a", "t1") != make_passage_id("a", "t2")


def test_id_changes_with_section():
    """A claim from Related Work describes someone else's contribution, so
    the same sentence in two sections is not the same evidence (§4.2)."""
    assert (make_passage_id("a", "t", "Results")
            != make_passage_id("a", "t", "Related Work"))


def test_none_section_differs_from_empty_string_source():
    assert make_passage_id("a", "t", None) == make_passage_id("a", "t", "")


def test_field_boundaries_cannot_collide():
    assert make_passage_id("ab", "c") != make_passage_id("a", "bc")


def test_passage_schema_fields():
    expected = {
        "id", "source_type", "source_id", "source_domain", "title", "text",
        "section", "published", "sub_question_id", "retrieval_score",
        "rerank_score",
    }
    assert set(Passage.__annotations__) == expected