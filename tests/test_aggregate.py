from __future__ import annotations

import json
from pathlib import Path

from lbmm.leaderboard.aggregate import collect_runs, per_model_summary, per_task_summary


def _write_run(root: Path, run_name: str, task: str, rows: list[dict]) -> None:
    d = root / run_name
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{task}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_collect_runs_empty(tmp_path: Path) -> None:
    df = collect_runs(tmp_path)
    assert df.empty
    assert "accuracy" not in df.columns  # sanity


def test_collect_and_summarize(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "20240501T120000__anthropic__haiku",
        "abercrombie",
        [
            _row(0, True, 0.001, 10, 5, 200),
            _row(1, True, 0.001, 12, 4, 210),
            _row(2, False, 0.001, 9, 6, 190),
        ],
    )
    _write_run(
        tmp_path,
        "20240501T120000__local__qwen0p5b",
        "abercrombie",
        [
            _row(0, False, 0.0, 11, 4, 5000),
            _row(1, True, 0.0, 12, 3, 5100),
        ],
    )

    df = collect_runs(tmp_path)
    assert len(df) == 5

    pm = per_model_summary(df)
    assert set(pm["provider"]) == {"anthropic", "local"}
    anth = pm[pm["provider"] == "anthropic"].iloc[0]
    loc = pm[pm["provider"] == "local"].iloc[0]
    assert abs(anth["accuracy"] - 2 / 3) < 1e-9
    assert abs(loc["accuracy"] - 0.5) < 1e-9
    # cost per correct for local is 0 (free)
    assert loc["cost_per_correct_usd"] == 0.0

    pt = per_task_summary(df)
    assert (pt["task"] == "abercrombie").all()


def _row(idx: int, correct: bool, cost: float, pt: int, ct: int, latency: int) -> dict:
    # provider/model are extracted from the run dir name in real life,
    # but our jsonl rows carry them too
    return {
        "task": "abercrombie",
        "index": idx,
        "provider": "anthropic" if cost > 0 else "local",
        "model": "haiku" if cost > 0 else "qwen0p5b",
        "answer_pred": "yes",
        "answer_raw": "yes.",
        "correct": correct,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "latency_ms": latency,
        "cost_usd": cost,
    }
