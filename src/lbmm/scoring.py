"""Answer parsing + scoring.

LegalBench answers are short strings (``yes``/``no``, label names, sometimes
short noun phrases). Models love to answer with rationale + the answer + a
trailing period. The scorer here normalizes both sides aggressively and uses
a fuzzy match (rapidfuzz token-set ratio) as a fallback for the long-tail.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from .types import Prediction, TaskItem

_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = _NORMALIZE_RE.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def extract_label(raw: str, choices: tuple[str, ...] = ()) -> str:
    """Pick the most likely label from a model's free-form reply.

    Strategy:
      1. If the reply starts with a choice exactly (case-insensitive), use it.
      2. Else if a choice appears anywhere in the reply, use the first one.
      3. Else take the first line, stripped.
    """
    text = raw.strip()
    if not text:
        return ""
    norm = normalize(text)
    if choices:
        norm_choices = [normalize(c) for c in choices]
        for c, n in zip(choices, norm_choices, strict=True):
            if norm.startswith(n + " ") or norm == n:
                return c
        for c, n in zip(choices, norm_choices, strict=True):
            if n and n in norm:
                return c
    # fall back to the first line (most models put their answer first)
    first = text.splitlines()[0].strip()
    return first


def score_item(prediction_text: str, gold: str, choices: tuple[str, ...]) -> tuple[bool, str]:
    """Return (is_correct, parsed_label). Fuzzy for long-tail strings."""
    parsed = extract_label(prediction_text, choices)
    g = normalize(gold)
    p = normalize(parsed)
    if not p:
        return False, ""
    if p == g:
        return True, parsed
    # exact match against any choice that itself normalizes to gold
    if choices:
        for c in choices:
            if normalize(c) == g and normalize(parsed) == normalize(c):
                return True, parsed
    # last resort: high-overlap fuzzy match (handles "Yes." vs "yes")
    ratio = fuzz.token_set_ratio(p, g)
    return ratio >= 90, parsed


def apply_scoring(pred: Prediction, task_item: TaskItem) -> Prediction:
    correct, parsed = score_item(pred.response.text, task_item.answer, task_item.choices)
    pred.correct = correct
    pred.parsed = parsed
    return pred
