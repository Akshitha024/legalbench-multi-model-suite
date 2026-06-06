from __future__ import annotations

from lbmm.judges.llm_judge import _parse_score


def test_clean_json() -> None:
    s = _parse_score('{"correctness": 4, "faithfulness": 5, "relevance": 3}')
    assert s is not None
    assert s.correctness == 4
    assert s.faithfulness == 5
    assert s.relevance == 3
    assert abs(s.mean - 4.0) < 1e-9


def test_json_in_code_fence() -> None:
    s = _parse_score(
        "Here is my evaluation:\n"
        "```json\n"
        '{"correctness": 2, "faithfulness": 3, "relevance": 4}\n'
        "```"
    )
    assert s is not None
    assert s.correctness == 2


def test_json_with_chatter() -> None:
    s = _parse_score(
        'The candidate is partial. {"correctness": 3, "faithfulness": 4, "relevance": 3} '
        "Hope this helps."
    )
    assert s is not None
    assert s.faithfulness == 4


def test_no_json_returns_none() -> None:
    assert _parse_score("Looks good to me!") is None


def test_missing_key_returns_none() -> None:
    assert _parse_score('{"correctness": 4, "faithfulness": 5}') is None
