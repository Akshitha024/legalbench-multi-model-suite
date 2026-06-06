"""Aggregate runs/ into a leaderboard DataFrame."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    """Walk runs/ and concatenate every per-task jsonl into one frame."""
    frames: list[pd.DataFrame] = []
    for run in sorted(runs_dir.glob("*")):
        if not run.is_dir():
            continue
        for jf in run.glob("*.jsonl"):
            rows = [json.loads(line) for line in jf.open() if line.strip()]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["run"] = run.name
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "task",
                "index",
                "provider",
                "model",
                "answer_pred",
                "answer_raw",
                "correct",
                "prompt_tokens",
                "completion_tokens",
                "latency_ms",
                "cost_usd",
                "run",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (provider, model): accuracy, n, total_cost, latency p50/p99."""
    if df.empty:
        return df
    grp = df.groupby(["provider", "model"])
    summary = grp.agg(
        n=("index", "count"),
        accuracy=("correct", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        total_prompt_tokens=("prompt_tokens", "sum"),
        total_completion_tokens=("completion_tokens", "sum"),
        latency_p50_ms=("latency_ms", lambda s: float(s.quantile(0.5))),
        latency_p99_ms=("latency_ms", lambda s: float(s.quantile(0.99))),
    ).reset_index()
    # cost per correct answer; if total_cost == 0 (local), this is also 0
    summary["cost_per_correct_usd"] = summary.apply(
        lambda r: (
            0.0
            if r["total_cost_usd"] == 0
            else r["total_cost_usd"] / max(1, r["accuracy"] * r["n"])
        ),
        axis=1,
    )
    return summary.sort_values("accuracy", ascending=False).reset_index(drop=True)


def per_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, provider, model): per-task accuracy."""
    if df.empty:
        return df
    return (
        df.groupby(["task", "provider", "model"])
        .agg(n=("index", "count"), accuracy=("correct", "mean"))
        .reset_index()
        .sort_values(["task", "accuracy"], ascending=[True, False])
        .reset_index(drop=True)
    )
