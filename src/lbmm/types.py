"""Shared dataclasses.

Frozen where they can be (so they hash and can be cached as dict keys), and
serializable to plain JSON for the runs/ artifacts. No third-party validation
layer; pydantic would add noise without helping any of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TaskKind = Literal["multiple_choice", "binary", "classification", "free_form"]


@dataclass(frozen=True)
class TaskItem:
    task: str
    index: int
    prompt: str
    answer: str  # gold answer string (label name or free text)
    choices: tuple[str, ...] = ()
    kind: TaskKind = "classification"


@dataclass
class ProviderResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: dict[str, Any] | None = None  # provider-specific payload (for debugging)


@dataclass
class Prediction:
    task: str
    index: int
    provider: str
    model: str
    response: ProviderResponse
    parsed: str  # normalized extracted answer (lowercased, stripped)
    correct: bool | None = None  # filled in by the scorer; None if not auto-scorable
    judge_score: float | None = None  # for free-form, set by the judge
    cost_usd: float | None = None
