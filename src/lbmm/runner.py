"""Orchestrate one eval run: (tasks, providers) → predictions on disk.

A run produces a directory under ``runs/<timestamp>__<provider>__<model>/``
containing one ``<task>.jsonl`` per task and a ``meta.json`` with the run
parameters. Re-running with the same parameters skips items already done
(idempotent enough for laptop use).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .cost import lookup
from .runners.base import Provider
from .scoring import apply_scoring
from .tasks.loader import load_task
from .types import Prediction


def run_one(
    provider: Provider,
    task: str,
    limit: int | None,
    out_dir: Path,
) -> list[Prediction]:
    items = load_task(task, limit=limit)
    task_file = out_dir / f"{task}.jsonl"

    # idempotency: skip indices already on disk
    already_done: set[int] = set()
    if task_file.exists():
        with task_file.open() as f:
            for line in f:
                try:
                    already_done.add(int(json.loads(line)["index"]))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
    if already_done:
        logger.info("{} already has {} items in {}, resuming", task, len(already_done), task_file)

    price = lookup(provider.name, provider.model)
    out: list[Prediction] = []
    with task_file.open("a") as f:
        for item in tqdm(items, desc=f"{provider.name}:{provider.model}/{task}"):
            if item.index in already_done:
                continue
            resp = provider.generate(item.prompt, max_tokens=64)
            pred = Prediction(
                task=item.task,
                index=item.index,
                provider=provider.name,
                model=provider.model,
                response=resp,
                parsed="",
            )
            apply_scoring(pred, item)
            if price is not None:
                pred.cost_usd = price.usd_for(resp.prompt_tokens, resp.completion_tokens)
            f.write(_to_jsonl(pred))
            f.write("\n")
            out.append(pred)
    return out


def _to_jsonl(p: Prediction) -> str:
    return json.dumps(
        {
            "task": p.task,
            "index": p.index,
            "provider": p.provider,
            "model": p.model,
            "answer_pred": p.parsed,
            "answer_raw": p.response.text,
            "correct": p.correct,
            "prompt_tokens": p.response.prompt_tokens,
            "completion_tokens": p.response.completion_tokens,
            "latency_ms": p.response.latency_ms,
            "cost_usd": p.cost_usd,
        }
    )


def make_run_dir(root: Path, provider: Provider) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_model = provider.model.replace("/", "_").replace(":", "_")
    d = root / f"{stamp}__{provider.name}__{safe_model}"
    d.mkdir(parents=True, exist_ok=True)
    return d
