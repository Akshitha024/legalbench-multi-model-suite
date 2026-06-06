"""Cost-vs-quality Pareto plot.

API providers form a real cost-quality frontier; local models live on the
zero-cost vertical line. The plot makes that frontier visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI

import matplotlib.pyplot as plt
import pandas as pd


def plot_cost_vs_accuracy(summary: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        out_path.write_bytes(b"")  # placeholder; nothing to plot
        return out_path

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in summary.iterrows():
        x = row["total_cost_usd"]
        y = row["accuracy"]
        label = f"{row['provider']}:{row['model']}"
        ax.scatter(x, y, s=80)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )
    ax.set_xlabel("Total run cost (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LegalBench multi-model: cost vs. accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_accuracy_bars(summary: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        out_path.write_bytes(b"")
        return out_path

    labels = [f"{r.provider}\n{r.model}" for r in summary.itertuples()]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 4))
    ax.bar(labels, summary["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("LegalBench multi-model: accuracy by provider")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
