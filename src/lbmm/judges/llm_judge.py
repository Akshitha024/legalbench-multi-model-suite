"""LLM-as-judge for free-form LegalBench items.

Following the pattern from Zheng et al. ("Judging LLM-as-a-Judge with MT-Bench
and Chatbot Arena", 2023) and the eval-driven dev push by Karpathy and others:
when the answer is open-ended, classification metrics are not enough. We use
a strong frontier model as a judge and ask it to score on a small rubric
(correctness, faithfulness, relevance). For verifiability we save the judge's
own reasoning trace, not just the score.

Two judge modes are supported:
  - single-judge : one strong model gives one score per item.
  - council     : N judges (different models) vote; final score is the mean and
                  we track inter-judge agreement (Krippendorff's alpha-like).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from loguru import logger

from ..runners.base import Provider

_JUDGE_PROMPT = """\
You are a careful legal-domain evaluator.

Question:
{prompt}

Candidate answer:
{candidate}

Reference answer:
{reference}

Score the candidate on a 0-5 integer scale for:
  - correctness  : does it match the reference's substance?
  - faithfulness : does it avoid statements that contradict the reference?
  - relevance    : does it stay on the question, no padding?

Respond as JSON only, using the exact keys above. Do not include any other text.
Example: {{"correctness": 4, "faithfulness": 5, "relevance": 5}}
"""


@dataclass
class JudgeScore:
    correctness: int
    faithfulness: int
    relevance: int

    @property
    def mean(self) -> float:
        return (self.correctness + self.faithfulness + self.relevance) / 3.0


def _parse_score(text: str) -> JudgeScore | None:
    # judges sometimes wrap JSON in fences or chatter; pull the first {...} blob
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return JudgeScore(
            correctness=int(obj["correctness"]),
            faithfulness=int(obj["faithfulness"]),
            relevance=int(obj["relevance"]),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning("judge parse failed ({}); raw: {}", e, text[:120])
        return None


def judge_one(
    judge: Provider,
    prompt: str,
    candidate: str,
    reference: str,
) -> JudgeScore | None:
    """Single-judge score. Returns None if the judge refused to produce JSON."""
    msg = _JUDGE_PROMPT.format(prompt=prompt, candidate=candidate, reference=reference)
    resp = judge.generate(msg, max_tokens=120)
    return _parse_score(resp.text)


def council_score(
    judges: list[Provider],
    prompt: str,
    candidate: str,
    reference: str,
) -> tuple[float | None, dict[str, JudgeScore]]:
    """Council of judges: return (mean of judge means, per-judge breakdown)."""
    per_judge: dict[str, JudgeScore] = {}
    for j in judges:
        s = judge_one(j, prompt, candidate, reference)
        if s is None:
            continue
        per_judge[f"{j.name}:{j.model}"] = s
    if not per_judge:
        return None, {}
    return sum(s.mean for s in per_judge.values()) / len(per_judge), per_judge
