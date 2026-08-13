"""Benchmark models on the task suite.

Every model runs every task the same way: a fresh workspace copy,
one real omp session, then the task tests. The test result is the
score. Token and cost totals come from the session file, so the
report can compare pass rate, latency, and price side by side.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .runner import EpisodeFailure, EpisodeRecord, run_episode
from .task import TaskLoadError, TaskSpec, load_task
from .trajectory import AssistantStep, parse_session


@dataclass(frozen=True)
class BenchRow:
    """Result of one model on one task trial."""

    model: str
    task: str
    trial: int
    reward: float
    duration_seconds: float
    total_tokens: int
    cost_usd: float
    tool_calls: int
    error: str | None


def _read_usage(session_file: Path) -> tuple[int, float]:
    """Sum token and cost totals over all assistant messages."""
    total_tokens = 0
    total_cost = 0.0
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        total_tokens += int(usage.get("totalTokens", 0))
        cost = usage.get("cost")
        if isinstance(cost, dict):
            total_cost += float(cost.get("total", 0.0))
    return total_tokens, total_cost


def _count_tool_calls(session_file: Path) -> int:
    """Count tool calls across all assistant turns."""
    trajectory = parse_session(session_file)
    return sum(
        len(step.tool_calls)
        for step in trajectory.steps
        if isinstance(step, AssistantStep)
    )


def _read_provider_error(session_file: Path) -> str | None:
    """Return the first provider error recorded in the session."""
    for line in session_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        if message.get("stopReason") != "error":
            continue
        raw = str(message.get("errorMessage", "provider error"))
        return " ".join(raw.split())[:160]
    return None


def load_task_suite(tasks_dir: Path) -> list[TaskSpec] | TaskLoadError:
    """Load every task directory in the suite, or the first error."""
    tasks: list[TaskSpec] = []
    for entry in sorted(tasks_dir.iterdir()):
        if not (entry / "task.toml").is_file():
            continue
        loaded = load_task(entry)
        if isinstance(loaded, TaskLoadError):
            return loaded
        tasks.append(loaded)
    return tasks


def run_benchmark(
    models: list[str],
    tasks: list[TaskSpec],
    trials: int,
    runs_dir: Path,
) -> list[BenchRow]:
    """Run the full model x task x trial grid, one episode at a time."""
    rows: list[BenchRow] = []
    total = len(models) * len(tasks) * trials
    done = 0
    for model in models:
        for task in tasks:
            for trial in range(1, trials + 1):
                done += 1
                print(
                    f"[{done}/{total}] model={model} "
                    f"task={task.name} trial={trial}"
                )
                result = run_episode(task, runs_dir, model)
                if isinstance(result, EpisodeFailure):
                    rows.append(
                        BenchRow(
                            model=model,
                            task=task.name,
                            trial=trial,
                            reward=0.0,
                            duration_seconds=0.0,
                            total_tokens=0,
                            cost_usd=0.0,
                            tool_calls=0,
                            error=result.reason,
                        )
                    )
                    continue
                rows.append(_row_from_record(result, trial))
    return rows


def _row_from_record(record: EpisodeRecord, trial: int) -> BenchRow:
    session_file = Path(record.session_file)
    tokens, cost = _read_usage(session_file)
    tool_calls = _count_tool_calls(session_file)
    error = None
    if record.reward == 0.0 and tool_calls == 0:
        error = _read_provider_error(session_file)
    return BenchRow(
        model=record.model,
        task=record.task,
        trial=trial,
        reward=record.reward,
        duration_seconds=record.duration_seconds,
        total_tokens=tokens,
        cost_usd=round(cost, 6),
        tool_calls=tool_calls,
        error=error,
    )


def render_report(rows: list[BenchRow]) -> str:
    """Render the leaderboard and the task matrix as markdown.

    Rows with a provider error do not count as failed tasks. They
    appear in the errors column and as E cells in the matrix.
    """
    models = sorted({row.model for row in rows})
    tasks = sorted({row.task for row in rows})

    lines = ["# omp-gym bench", ""]
    lines.append(
        "| model | pass rate | errors | mean seconds | mean tokens "
        "| total cost usd | tool calls |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    def clean_rows(model: str) -> list[BenchRow]:
        return [
            row
            for row in rows
            if row.model == model and row.error is None
        ]

    def pass_rate(model: str) -> float:
        clean = clean_rows(model)
        if not clean:
            return -1.0
        return sum(row.reward for row in clean) / len(clean)

    for model in sorted(models, key=lambda m: -pass_rate(m)):
        clean = clean_rows(model)
        errored = [
            row
            for row in rows
            if row.model == model and row.error is not None
        ]
        if clean:
            count = len(clean)
            rate = f"{pass_rate(model):.0%} ({count} runs)"
            mean_seconds = f"{sum(r.duration_seconds for r in clean) / count:.1f}"
            mean_tokens = f"{sum(r.total_tokens for r in clean) / count:.0f}"
            total_cost = f"{sum(r.cost_usd for r in clean):.4f}"
            calls = str(sum(r.tool_calls for r in clean))
        else:
            rate, mean_seconds, mean_tokens, total_cost, calls = (
                "-", "-", "-", "-", "-",
            )
        lines.append(
            f"| {model} | {rate} | {len(errored)} | {mean_seconds} "
            f"| {mean_tokens} | {total_cost} | {calls} |"
        )

    lines.extend(["", "## Task matrix", ""])
    lines.append("| task | " + " | ".join(models) + " |")
    lines.append("| --- |" + " --- |" * len(models))
    for task in tasks:
        cells: list[str] = []
        for model in models:
            cell_rows = [
                row
                for row in rows
                if row.model == model and row.task == task
            ]
            if not cell_rows:
                cells.append("-")
                continue
            clean = [row for row in cell_rows if row.error is None]
            errors = len(cell_rows) - len(clean)
            if not clean:
                cells.append("E")
                continue
            wins = sum(1 for row in clean if row.reward >= 1.0)
            cell = f"{wins}/{len(clean)}"
            if errors:
                cell += "+E"
            cells.append(cell)
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    errored_rows = [row for row in rows if row.error is not None]
    if errored_rows:
        lines.extend(["", "## Provider errors", ""])
        for row in errored_rows:
            lines.append(f"- {row.model} / {row.task}: {row.error}")
    lines.append("")
    return "\n".join(lines)
