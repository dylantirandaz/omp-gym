"""Render the platform report from the experiment ledger.

Two views: adapters (training runs with their loss curves) and
models (aggregated bench performance: pass rate, cost per pass,
tokens per solve). Use it to compare adapters and models across
recorded runs.
"""

from dataclasses import dataclass
from pathlib import Path

from .bench import wilson_interval
from .ledger import LedgerEntry, read_ledger


@dataclass(frozen=True)
class ModelStats:
    """Aggregated bench performance of one model."""

    model: str
    runs: int
    passes: int
    total_cost: float
    mean_tokens: float
    tokens_per_solve: float | None
    cost_per_pass: float | None
    low: float = 0.0
    high: float = 1.0


def _model_stats(entries: list[LedgerEntry]) -> list[ModelStats]:
    """Aggregate all bench rows in the ledger by model.

    Every bench row counts as one run. An error row is a scheduled
    episode that did not succeed, so it stays in the run count and
    lowers the pass rate. Cost sums the rows that carry a cost.
    The low/high fields hold the 95% Wilson interval of the pass
    rate, and models sort by its lower bound: a lucky 1/1 must not
    outrank a solid 9/10.
    """
    rows: list[dict[str, object]] = []
    for entry in entries:
        if entry.kind != "bench":
            continue
        for row in entry.metrics.get("rows", []):
            if isinstance(row, dict):
                rows.append(row)
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    stats: list[ModelStats] = []
    for model, model_rows in sorted(by_model.items()):
        passes = sum(1 for row in model_rows if row["reward"] >= 1.0)
        low, high = wilson_interval(passes, len(model_rows))
        cost = sum(
            float(row["cost_usd"])
            for row in model_rows
            if row.get("cost_usd") is not None
        )
        tokens = [int(row["total_tokens"]) for row in model_rows]
        solve_tokens = [
            int(row["total_tokens"])
            for row in model_rows
            if row["reward"] >= 1.0
        ]
        stats.append(
            ModelStats(
                model=model,
                runs=len(model_rows),
                passes=passes,
                total_cost=cost,
                mean_tokens=sum(tokens) / len(tokens),
                tokens_per_solve=(
                    sum(solve_tokens) / len(solve_tokens)
                    if solve_tokens
                    else None
                ),
                cost_per_pass=(cost / passes if passes else None),
                low=low,
                high=high,
            )
        )
    stats.sort(key=lambda s: (-s.low, s.mean_tokens))
    return stats


def render_report(entries: list[LedgerEntry]) -> str:
    """Render the full comparison report as markdown."""
    lines = ["# omp-gym report", ""]

    train_entries = [e for e in entries if e.kind == "train"]
    lines.append("## Adapters")
    lines.append("")
    if not train_entries:
        lines.append("no training runs recorded")
    else:
        lines.append(
            "| adapter | model | iters | train loss | val loss | when |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for entry in train_entries:
            metrics = entry.metrics
            adapter = str(entry.config.get("adapter", "?"))
            model = str(entry.config.get("model", "?")).split("/")[-1]
            iters = metrics.get("iterations", "?")
            first_t = metrics.get("first_train_loss")
            last_t = metrics.get("last_train_loss")
            first_v = metrics.get("first_val_loss")
            last_v = metrics.get("last_val_loss")
            train_cell = (
                f"{first_t} -> {last_t}"
                if first_t is not None
                else "-"
            )
            val_cell = (
                f"{first_v} -> {last_v}" if first_v is not None else "-"
            )
            lines.append(
                f"| {adapter} | {model} | {iters} | {train_cell} "
                f"| {val_cell} | {entry.timestamp[:16]} |"
            )
    lines.append("")

    stats = _model_stats(entries)
    lines.append("## Models")
    lines.append("")
    if not stats:
        lines.append("no bench runs recorded")
    else:
        lines.append(
            "| model | pass rate | cost per pass | tokens per solve "
            "| mean tokens | runs |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for stat in stats:
            rate = (
                f"{stat.passes / stat.runs:.0%} "
                f"[{stat.low:.0%}, {stat.high:.0%}] "
                f"({stat.passes}/{stat.runs})"
            )
            cpp = (
                f"${stat.cost_per_pass:.4f}"
                if stat.cost_per_pass is not None
                else "-"
            )
            tps = (
                f"{stat.tokens_per_solve:.0f}"
                if stat.tokens_per_solve is not None
                else "-"
            )
            lines.append(
                f"| {stat.model} | {rate} | {cpp} | {tps} "
                f"| {stat.mean_tokens:.0f} | {stat.runs} |"
            )
    lines.append("")

    lines.append("## Timeline")
    lines.append("")
    for entry in entries[-10:]:
        summary = ", ".join(
            f"{key}={value}"
            for key, value in list(entry.metrics.items())[:3]
            if not isinstance(value, (list, dict))
        )
        lines.append(
            f"- {entry.timestamp[:16]} {entry.kind}: {summary}"
        )
    lines.append("")
    return "\n".join(lines)


def report_from_ledger(ledger_path: Path) -> str:
    """Read the ledger and render the report."""
    entries, torn = read_ledger(ledger_path)
    report = render_report(entries)
    if torn:
        report += f"\n({torn} torn ledger lines skipped)\n"
    return report
