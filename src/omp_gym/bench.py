"""Benchmark models on the task suite.

Every model runs every task the same way: a fresh workspace copy,
one real omp session, then the task tests. The test result is the
score. Token and cost totals come from the session file, so the
report can compare pass rate, latency, and price side by side.

Scheduling: trials interleave across models round-robin (m1 t1,
m2 t1, ...) and each trial block is shuffled with the bench seed,
so drift over the run cannot align with the model or task order.
Every row records its schedule position as order_index.
"""

import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from .runner import EpisodeFailure, EpisodeRecord, run_episode
from .task import TaskLoadError, TaskSpec, load_task, workspace_digest
from .trajectory import AssistantStep, parse_session

BOOTSTRAP_RESAMPLES = 2000


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
    order_index: int | None = None
    error_class: str | None = None


FAILURE_CLASSES = (
    "provider_error",
    "baseline_timeout",
    "test_timeout",
    "no_session",
    "invalid_task",
    "sandbox",
    "other",
)


def classify_failure(reason: str) -> str:
    """Bucket one failure reason into the report taxonomy.

    Keyword matching in priority order; the first hit wins and
    unmatched text lands in "other". Session-derived provider
    errors are tagged provider_error at row creation, so this
    mainly serves runner-level failure reasons.
    """
    text = reason.lower()
    if "sandbox" in text or "seatbelt" in text or "isolation" in text:
        return "sandbox"
    if "baseline" in text and (
        "deadline" in text or "timeout" in text or "timed out" in text
    ):
        return "baseline_timeout"
    if "test" in text and (
        "deadline" in text or "timeout" in text or "timed out" in text
    ):
        return "test_timeout"
    if "no session" in text or "session file" in text:
        return "no_session"
    if "already passes" in text or "invalid task" in text or "not a valid task" in text:
        return "invalid_task"
    if (
        "provider" in text
        or "rate limit" in text
        or "overloaded" in text
        or "api error" in text
    ):
        return "provider_error"
    return "other"


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
        if not isinstance(entry, dict):
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
        if not isinstance(entry, dict):
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


def load_task_suite(
    tasks_dir: Path,
    recursive: bool = False,
    include_minted: bool = False,
) -> list[TaskSpec] | TaskLoadError:
    """Load the tasks under the root, or the first error.

    Only the root's immediate children are tasks by default;
    recursive=True opts into walking the whole tree. Directories
    named "minted" are excluded at any depth unless
    include_minted=True. The task id is the path relative to the
    root, so leaf-name collisions below nested pools cannot alias
    two tasks to one matrix row.
    """
    if not tasks_dir.is_dir():
        return []
    if recursive:
        candidates = [config.parent for config in tasks_dir.rglob("task.toml")]
    else:
        candidates = [
            child
            for child in tasks_dir.iterdir()
            if child.is_dir() and (child / "task.toml").is_file()
        ]
    selected: list[Path] = []
    for candidate in candidates:
        relative = candidate.relative_to(tasks_dir)
        if not include_minted and "minted" in relative.parts:
            continue
        selected.append(candidate)
    selected.sort(key=lambda path: path.relative_to(tasks_dir).as_posix())

    tasks: list[TaskSpec] = []
    for task_dir in selected:
        loaded = load_task(task_dir)
        if isinstance(loaded, TaskLoadError):
            return loaded
        task_id = task_dir.relative_to(tasks_dir).as_posix()
        tasks.append(replace(loaded, name=task_id))
    return tasks


def run_benchmark(
    models: list[str],
    tasks: list[TaskSpec],
    trials: int,
    runs_dir: Path,
    seed: int = 0,
) -> list[BenchRow]:
    """Run the full model x task x trial grid, one episode at a time.

    Trials interleave across models round-robin (m1 t1, m2 t1, ...),
    then each trial block is shuffled with the seed. order_index on
    each row records its final schedule position, so the report and
    the ledger can reconstruct execution order.
    """
    schedule: list[tuple[str, TaskSpec, int]] = []
    shuffler = random.Random(seed)  # noqa: S311 - seeded bench order
    for trial in range(1, trials + 1):
        block = [(model, task, trial) for model in models for task in tasks]
        shuffler.shuffle(block)
        schedule.extend(block)

    rows: list[BenchRow] = []
    total = len(schedule)
    for order_index, (model, task, trial) in enumerate(schedule):
        print(
            f"[{order_index + 1}/{total}] model={model} task={task.name} trial={trial}"
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
                    order_index=order_index,
                    error_class=classify_failure(result.reason),
                )
            )
            continue
        rows.append(_row_from_record(result, trial, order_index))
    return rows


def _record_duration(record: EpisodeRecord) -> float:
    """Wall time for one row: wall_seconds first, duration_seconds else.

    wall_seconds spans baseline plus episode plus scoring; older
    records lack it, so duration_seconds (episode only) is the
    fallback.
    """
    wall = getattr(record, "wall_seconds", None)
    if wall is not None:
        return float(wall)
    return record.duration_seconds


def _row_from_record(
    record: EpisodeRecord, trial: int, order_index: int | None = None
) -> BenchRow:
    session_file = Path(record.session_file)
    tokens, cost = _read_usage(session_file)
    tool_calls = _count_tool_calls(session_file)
    error = None
    error_class = None
    if record.reward == 0.0 and tool_calls == 0:
        error = _read_provider_error(session_file)
        if error is not None:
            error_class = "provider_error"
    return BenchRow(
        model=record.model,
        task=record.task,
        trial=trial,
        reward=record.reward,
        duration_seconds=_record_duration(record),
        total_tokens=tokens,
        cost_usd=round(cost, 6),
        tool_calls=tool_calls,
        error=error,
        order_index=order_index,
        error_class=error_class,
    )


def wilson_interval(passes: int, runs: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial pass rate.

    The point estimate overstates certainty at bench sample sizes:
    1/1 is not better than 9/10. With no runs the rate is fully
    unknown, so the interval is (0, 1).
    """
    if runs <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    rate = passes / runs
    denominator = 1.0 + z * z / runs
    center = (rate + z * z / (2 * runs)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / runs + z * z / (4 * runs * runs))
        / denominator
    )
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_pass_rate_ci(
    rows: list[BenchRow],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the overall pass rate.

    Per-task pass rates (wins over scheduled rows) are resampled
    with replacement; each replicate's statistic is the mean of the
    sampled task rates. Task-level resampling keeps the patient
    heterogeneity that row-level resampling would flatten. The
    interval is the 2.5/97.5 percentile of the replicate
    distribution; None when there are no rows.
    """
    by_task: dict[str, list[BenchRow]] = {}
    for row in rows:
        by_task.setdefault(row.task, []).append(row)
    if not by_task:
        return None
    rates = [
        sum(1 for row in task_rows if row.reward >= 1.0) / len(task_rows)
        for task_rows in by_task.values()
    ]
    rng = random.Random(seed)  # noqa: S311 - seeded resampling
    count = len(rates)
    estimates: list[float] = []
    for _ in range(resamples):
        total = sum(rates[rng.randrange(count)] for _ in range(count))
        estimates.append(total / count)
    estimates.sort()
    low = estimates[int(0.025 * resamples)]
    high = estimates[min(int(0.975 * resamples), resamples - 1)]
    return (low, high)


# Distribution names checked for the manifest, with their import
# module names. Only importable packages land in the manifest.
_MANIFEST_PACKAGES = (
    ("mlx", "mlx"),
    ("mlx_lm", "mlx-lm"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("yaml", "PyYAML"),
)


def _package_versions() -> dict[str, str]:
    """Installed versions of the importable known packages."""
    versions: dict[str, str] = {}
    for module_name, dist_name in _MANIFEST_PACKAGES:
        try:
            if importlib.util.find_spec(module_name) is None:
                continue
        except ModuleNotFoundError:
            continue
        try:
            versions[dist_name] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _git_sha(root: Path | None) -> str | None:
    """Current git HEAD, or None when git or the repo is unavailable."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed static argv
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed static argv
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def bench_manifest(
    tasks: list[TaskSpec], root: Path | None = None
) -> dict[str, object]:
    """The manifest payload: environment plus per-task digests.

    The digest binds the report to exact workspace contents, so a
    task edit after the bench cannot masquerade as the same task.
    """
    return {
        "git_sha": _git_sha(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "tasks": [
            {
                "name": task.name,
                "workspace_digest": workspace_digest(task.workspace),
            }
            for task in tasks
        ],
    }


def write_manifest(
    report_path: Path, tasks: list[TaskSpec], root: Path | None = None
) -> Path:
    """Write bench-manifest.json next to the bench report."""
    manifest_path = report_path.parent / "bench-manifest.json"
    payload = bench_manifest(tasks, root)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def _row_class(row: BenchRow) -> str | None:
    """The taxonomy class of an error row, derived when not tagged."""
    if row.error is None:
        return None
    return row.error_class or classify_failure(row.error)


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or None when the input is empty."""
    return sum(values) / len(values) if values else None


def summarize_report(rows: list[BenchRow], seed: int = 0) -> dict[str, object]:
    """JSON-able summary of one bench run.

    Aggregates always cover all scheduled rows; clean aggregates
    cover the rows without errors. No errored row is ever dropped
    from an all-rows number. failure_classes counts the taxonomy
    buckets; bootstrap_ci records the task-level CI and its
    resample parameters.
    """
    wins = sum(1 for row in rows if row.reward >= 1.0)
    ci = bootstrap_pass_rate_ci(rows, seed=seed)
    failures: dict[str, int] = {}
    for row in rows:
        klass = _row_class(row)
        if klass is not None:
            failures[klass] = failures.get(klass, 0) + 1

    models: dict[str, dict[str, object]] = {}
    for model in sorted({row.model for row in rows}):
        scheduled = [row for row in rows if row.model == model]
        clean = [row for row in scheduled if row.error is None]
        models[model] = {
            "runs": len(scheduled),
            "passes": sum(1 for row in scheduled if row.reward >= 1.0),
            "mean_seconds_all": _mean([row.duration_seconds for row in scheduled]),
            "mean_seconds_clean": _mean([row.duration_seconds for row in clean]),
            "mean_tokens_all": _mean([float(row.total_tokens) for row in scheduled]),
            "mean_tokens_clean": _mean([float(row.total_tokens) for row in clean]),
            "cost_usd_all": sum(row.cost_usd for row in scheduled),
            "cost_usd_clean": sum(row.cost_usd for row in clean),
        }
    return {
        "rows": len(rows),
        "passes": wins,
        "pass_rate": (wins / len(rows)) if rows else None,
        "bootstrap_ci": (
            {
                "level": 0.95,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": seed,
                "low": ci[0],
                "high": ci[1],
            }
            if ci is not None
            else None
        ),
        "failure_classes": dict(sorted(failures.items())),
        "models": models,
    }


def _tie_groups(bounds: list[tuple[float, float]]) -> list[int]:
    """Group id per position: connected components of interval overlap.

    Two intervals that overlap are linked; the groups are the
    transitive closure of that relation (union-find), so chained
    overlaps put every member in one group even when the end
    intervals do not touch each other. The mechanism is a
    heuristic: a long chain can lump models that are genuinely
    separated at the ends.
    """
    count = len(bounds)
    parent = list(range(count))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    for first in range(count):
        for second in range(first + 1, count):
            low_first, high_first = bounds[first]
            low_second, high_second = bounds[second]
            if low_first <= high_second and low_second <= high_first:
                root_first, root_second = find(first), find(second)
                if root_first != root_second:
                    parent[root_second] = root_first
    return [find(index) for index in range(count)]


def render_report(rows: list[BenchRow]) -> str:
    """Render the leaderboard and the task matrix as markdown.

    The headline pass rate divides successful episodes by all
    scheduled rows for the model, and carries a 95% Wilson
    interval over the same denominator; errored rows never leave
    the denominator. Aggregate columns come in all-rows and clean
    flavors, labeled, and the all-rows number counts every
    scheduled row. Models sort by the interval lower bound. The
    rank column is a heuristic: '=' marks membership in one
    connected component of the interval-overlap relation
    (transitive closure; a chain of overlaps groups the whole
    chain), and a bare number means the model's interval shares
    no overlap chain with any neighbor.
    """
    models = sorted({row.model for row in rows})
    tasks = sorted({row.task for row in rows})

    ci = bootstrap_pass_rate_ci(rows)
    lines = ["# omp-gym bench", ""]
    if ci is not None:
        lines.append(
            "Overall pass rate CI (task-level bootstrap, "
            f"{BOOTSTRAP_RESAMPLES} resamples, seed 0): "
            f"[{ci[0]:.0%}, {ci[1]:.0%}]"
        )
        lines.append("")
    lines.append(
        "| rank | model | pass rate | errors | mean s (all) "
        "| mean s (clean) | mean tokens (all) | mean tokens (clean) "
        "| cost usd (all) | cost usd (clean) | tool calls (all) "
        "| tool calls (clean) |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    def scheduled_rows(model: str) -> list[BenchRow]:
        return [row for row in rows if row.model == model]

    def interval(model: str) -> tuple[float, float]:
        scheduled = scheduled_rows(model)
        wins = sum(1 for row in scheduled if row.reward >= 1.0)
        return wilson_interval(wins, len(scheduled))

    ordered = sorted(models, key=lambda model: -interval(model)[0])
    bounds = [interval(model) for model in ordered]
    groups = _tie_groups(bounds)
    sizes: dict[int, int] = {}
    for group in groups:
        sizes[group] = sizes.get(group, 0) + 1

    def rank_cell(position: int) -> str:
        return "=" if sizes[groups[position]] > 1 else str(position + 1)

    for position, model in enumerate(ordered):
        scheduled = scheduled_rows(model)
        clean = [row for row in scheduled if row.error is None]
        errored = len(scheduled) - len(clean)
        wins = sum(1 for row in scheduled if row.reward >= 1.0)
        low, high = bounds[position]
        rate = (
            f"{wins / len(scheduled):.0%} [{low:.0%}, {high:.0%}] (n={len(scheduled)})"
        )

        def seconds_mean(pool: list[BenchRow]) -> str:
            mean = _mean([row.duration_seconds for row in pool])
            return f"{mean:.1f}" if mean is not None else "-"

        def tokens_mean(pool: list[BenchRow]) -> str:
            mean = _mean([float(row.total_tokens) for row in pool])
            return f"{mean:.0f}" if mean is not None else "-"

        cost_all = f"{sum(row.cost_usd for row in scheduled):.4f}"
        cost_clean = f"{sum(row.cost_usd for row in clean):.4f}" if clean else "-"
        calls_all = str(sum(row.tool_calls for row in scheduled))
        calls_clean = str(sum(row.tool_calls for row in clean)) if clean else "-"
        lines.append(
            f"| {rank_cell(position)} | {model} | {rate} | {errored} "
            f"| {seconds_mean(scheduled)} | {seconds_mean(clean)} "
            f"| {tokens_mean(scheduled)} | {tokens_mean(clean)} "
            f"| {cost_all} | {cost_clean} | {calls_all} | {calls_clean} |"
        )

    lines.extend(["", "## Task matrix", ""])
    lines.append("| task | " + " | ".join(models) + " |")
    lines.append("| --- |" + " --- |" * len(models))
    for task in tasks:
        cells: list[str] = []
        for model in models:
            cell_rows = [row for row in rows if row.model == model and row.task == task]
            if not cell_rows:
                cells.append("-")
                continue
            wins = sum(1 for row in cell_rows if row.reward >= 1.0)
            cells.append(f"{wins}/{len(cell_rows)}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    errored_rows = [row for row in rows if row.error is not None]
    if errored_rows:
        counts: dict[str, int] = {}
        for row in errored_rows:
            klass = _row_class(row) or "other"
            counts[klass] = counts.get(klass, 0) + 1
        lines.extend(["", "## Failures by class", ""])
        for klass in sorted(counts):
            lines.append(f"- {klass}: {counts[klass]}")
        lines.extend(["", "### Failure detail", ""])
        for row in errored_rows:
            klass = _row_class(row) or "other"
            lines.append(f"- {row.model} / {row.task} [{klass}]: {row.error}")

    lines.extend(["", "## Ranking method", ""])
    lines.append(
        "- sort: descending lower bound of the 95% Wilson interval over scheduled rows"
    )
    lines.append(
        "- ties: heuristic. '=' groups are connected components of the "
        "interval-overlap relation, so membership is transitive-closed; "
        "a chain of pairwise overlaps groups the whole chain even when "
        "the end intervals never touch. The limit of the heuristic: a "
        "long chain can group models that are genuinely separated."
    )
    lines.append(
        "- denominators are scheduled rows everywhere; error rows are "
        "never dropped from a denominator or an all-rows aggregate"
    )
    lines.append("")
    return "\n".join(lines)
