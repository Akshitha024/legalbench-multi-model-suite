"""LegalBench task loader.

LegalBench (Guha et al., NeurIPS 2023) is collaboratively built: 162 tasks,
varying schemas, varying ground-truth conventions. The HuggingFace mirror at
``nguha/legalbench`` exposes each task as a separate config; columns differ
per task but the common pair is ``text`` + ``answer``. Multiple-choice tasks
carry their choices in the task's base_prompt (which lives in the LegalBench
GitHub repo, not in the HF dataset), so we either ship a prompt template per
task type or fall back to a generic instruction-following prompt.

This loader supports two strategies:
  - ``base_prompt``: official prompt from the LegalBench repo (when shipped
    in the local prompt cache under ``tasks/prompts/{task}.txt``)
  - ``generic``: a one-line instruction + the task's text column
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from ..types import TaskItem, TaskKind

PROMPT_DIR = Path(__file__).parent / "prompts"


_TASK_KIND_HINTS: dict[str, TaskKind] = {
    # tasks that are clearly binary yes/no
    "abercrombie": "multiple_choice",
    "proa": "binary",
    "nys_judicial_ethics": "binary",
    "successor_liability": "multiple_choice",
    "contract_qa": "classification",
    "personal_jurisdiction": "binary",
}


def _kind_of(task: str) -> TaskKind:
    return _TASK_KIND_HINTS.get(task, "classification")


def _generic_prompt(task: str, text: str, choices: tuple[str, ...] = ()) -> str:
    if choices:
        choice_block = "\n".join(f"  - {c}" for c in choices)
        return (
            f"You are evaluating a {task.replace('_', ' ')} item.\n\n"
            f"Question:\n{text}\n\n"
            f"Choose one of the following labels and respond with the label only.\n"
            f"{choice_block}\n\n"
            f"Answer:"
        )
    return (
        f"You are evaluating a {task.replace('_', ' ')} item.\n\n"
        f"Question:\n{text}\n\n"
        f"Respond with the most appropriate one-word or short-phrase label.\n\n"
        f"Answer:"
    )


def _base_prompt(task: str, text: str) -> str | None:
    cached = PROMPT_DIR / f"{task}.txt"
    if not cached.exists():
        return None
    template = cached.read_text()
    if "{{text}}" in template:
        return template.replace("{{text}}", text)
    return template + "\n\n" + text


def load_task(task: str, limit: int | None = None, split: str = "test") -> list[TaskItem]:
    """Pull one LegalBench sub-task and convert to TaskItem rows."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("load_task needs the `datasets` package") from e

    ds = load_dataset("nguha/legalbench", task, split=split, trust_remote_code=True)
    n = len(ds) if limit is None else min(len(ds), limit)
    rows: list[TaskItem] = []
    for i in range(n):
        row = ds[i]
        text = str(row.get("text") or row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not text or not answer:
            continue
        choices = _infer_choices(row)
        prompt = _base_prompt(task, text) or _generic_prompt(task, text, choices)
        rows.append(
            TaskItem(
                task=task,
                index=i,
                prompt=prompt,
                answer=answer,
                choices=choices,
                kind=_kind_of(task),
            )
        )
    logger.info("loaded task '{}' split='{}': {} items", task, split, len(rows))
    return rows


def _infer_choices(row: dict[str, object]) -> tuple[str, ...]:
    # LegalBench sometimes ships per-row choices, more often not.
    for key in ("options", "choices", "labels"):
        v = row.get(key)
        if isinstance(v, list | tuple):
            return tuple(str(x) for x in v)
    return ()


def known_tasks() -> Iterator[str]:
    """Iterate the LegalBench task names we have explicit kind hints for."""
    return iter(_TASK_KIND_HINTS.keys())
