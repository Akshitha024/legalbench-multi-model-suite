from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tabulate import tabulate

from ..leaderboard.aggregate import collect_runs, per_model_summary, per_task_summary
from ..leaderboard.plots import plot_accuracy_bars, plot_cost_vs_accuracy
from ..runner import make_run_dir, run_one
from ..runners.registry import build as build_provider

app = typer.Typer(add_completion=False, help="lbmm: LegalBench multi-model harness")


@app.command("run")
def cmd_run(
    tasks: Annotated[
        str, typer.Option(help="comma-separated LegalBench tasks")
    ] = "abercrombie,proa",
    providers: Annotated[
        str, typer.Option(help="comma-separated provider specs")
    ] = "local-qwen0p5b",
    limit: Annotated[int, typer.Option(help="cap items per task")] = 30,
    runs_dir: Annotated[Path, typer.Option(help="root dir for run artifacts")] = Path("runs"),
) -> None:
    """Run one or more providers across one or more tasks, save jsonl artifacts."""
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    for spec in provider_list:
        provider = build_provider(spec)
        out_dir = make_run_dir(runs_dir, provider)
        logger.info("run dir: {}", out_dir)
        for t in task_list:
            run_one(provider, t, limit=limit, out_dir=out_dir)
        provider.close()


@app.command("leaderboard")
def cmd_leaderboard(
    runs_dir: Annotated[Path, typer.Option(help="root dir of runs")] = Path("runs"),
    results_dir: Annotated[Path, typer.Option(help="where to write summary csvs")] = Path(
        "results"
    ),
) -> None:
    df = collect_runs(runs_dir)
    if df.empty:
        typer.echo("no runs yet; do `lbmm run ...` first")
        raise typer.Exit(code=1)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_model = per_model_summary(df)
    per_task = per_task_summary(df)
    (results_dir / "per_model.csv").write_text(per_model.to_csv(index=False))
    (results_dir / "per_task.csv").write_text(per_task.to_csv(index=False))
    typer.echo("")
    typer.echo(
        tabulate(per_model, headers="keys", floatfmt=".3f", tablefmt="github", showindex=False)
    )


@app.command("plots")
def cmd_plots(
    runs_dir: Annotated[Path, typer.Option(help="root dir of runs")] = Path("runs"),
    figures_dir: Annotated[Path, typer.Option(help="output dir for figures")] = Path(
        "results/figures"
    ),
) -> None:
    df = collect_runs(runs_dir)
    summary = per_model_summary(df)
    plot_cost_vs_accuracy(summary, figures_dir / "cost_vs_accuracy.png")
    plot_accuracy_bars(summary, figures_dir / "accuracy_by_model.png")
    typer.echo(f"wrote figures to {figures_dir}")


if __name__ == "__main__":
    app()
