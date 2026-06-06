from __future__ import annotations

from lbmm.scoring import extract_label, normalize, score_item


def test_normalize_basics() -> None:
    assert normalize("Yes.") == "yes"
    assert normalize("  NO  ") == "no"
    assert normalize("Plaintiff's-Side") == "plaintiff s side"


def test_extract_label_starts_with_choice() -> None:
    assert extract_label("yes, the contract holds.", ("yes", "no")) == "yes"
    assert extract_label("NO!", ("yes", "no")) == "no"


def test_extract_label_choice_anywhere() -> None:
    # the rationale comes first, the label is buried
    out = extract_label("the parties intended a release. answer: yes.", ("yes", "no"))
    assert out == "yes"


def test_extract_label_falls_back_to_first_line() -> None:
    assert extract_label("admissible\nbecause foo", ()) == "admissible"


def test_score_exact() -> None:
    ok, parsed = score_item("yes.", "yes", ("yes", "no"))
    assert ok is True
    assert parsed == "yes"


def test_score_wrong() -> None:
    ok, _ = score_item("no", "yes", ("yes", "no"))
    assert ok is False


def test_score_fuzzy_for_punctuation_only() -> None:
    ok, _ = score_item("admissible.", "admissible", ())
    assert ok is True


def test_score_empty_response() -> None:
    ok, parsed = score_item("", "yes", ("yes", "no"))
    assert ok is False
    assert parsed == ""
