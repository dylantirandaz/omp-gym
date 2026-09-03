"""Benchmark models on minted OMP tasks and aggregate Prime traces into a leaderboard.

Trace row schema relied on (Prime Verifiers v1 `traces.jsonl`, one Episode per line;
see `verifiers/v1/cli/output.py`, `verifiers/v1/trace.py`, `verifiers/v1/types.py`):

    line                      -> Episode {"id", "env", "ok", "errors", "traces": [Trace, ...]}
    trace.id                  -> str
    trace.task.data           -> our TaskData: task_id, task_revision, task_digest, split, ...
    trace.agent.config.model  -> run model id (the env fills it in); `calls[].model` fallback
    trace.tools               -> [{name, description, parameters}] advertised to the agent
    trace.nodes               -> [{parent, sampled, message{role, content, tool_calls[]}}]
    trace.calls               -> [{model, finish_reason, sampling,
                                   usage{prompt_tokens, completion_tokens,
                                         cached_input_tokens, reasoning_tokens, cost},
                                   time{start, end}}]
    trace.rewards             -> {name: {score, weight} | null}; reward = sum(score * weight)
    trace.metrics             -> {passed_cases, total_cases, verifier_seconds}
    trace.info                -> {omp_tool_contract, omp_version, verifier{...}} (environment.py)
    trace.ok                  -> bool; trace.errors -> [{type, message, ...}]
    trace.stop_condition      -> str | null ("agent_completed" on a normal finish)
    trace.timing.agent        -> {start, end} epoch seconds (agent wall clock)

`Trace.reward`, `Trace.usage` and the token counters are pydantic properties and are
NOT serialized, so every derived number here is recomputed from the fields above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MAX_TRACE_FILE_BYTES = 128 * 1024 * 1024
MAX_PROVENANCE_BYTES = 1024 * 1024
MAX_TASK_CONFIG_BYTES = 1024 * 1024
PASSING_REWARD = 1.0
REFERENCE_MODEL_LABEL = "reference"
BENCH_JSON = "bench.json"
BENCH_MARKDOWN = "bench.md"
RUNS_DIR = "runs"
DEFAULT_SPLIT = "holdout"
DEFAULT_NUM_ROLLOUTS = 1
DEFAULT_MAX_CONCURRENT = 4
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_TOTAL_TOKENS = 200_000
MODEL_DIR_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

CommandRunner = Callable[[Sequence[str]], int]
"""Runs one command line and returns its exit status."""


@dataclass(frozen=True)
class BenchFailure:
    reason: str


@dataclass(frozen=True)
class Rollout:
    task_id: str
    task_revision: int
    task_digest: str
    model: str
    reward: float
    ok: bool
    input_tokens: int
    output_tokens: int
    tool_calls: int
    seconds: float


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    model: str
    tool_contract: str
    omp_version: str
    benchmark_digest: str
    rollouts: int


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    tasks: int
    rollouts: int
    mean_reward: float
    pass_rate: float
    pass_at_k: float | None
    k: int
    pass_at_k_by_task: dict[str, float]
    error_rate: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_tool_calls: float
    mean_seconds: float


@dataclass(frozen=True)
class ReferenceUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_tokens: int
    cost: float


@dataclass(frozen=True)
class ReferenceTask:
    task_id: str
    models: tuple[str, ...]
    usage: ReferenceUsage
    assistant_turns: int
    tool_calls: int
    reward: float


@dataclass(frozen=True)
class ReferenceModelMetrics:
    model: str
    tasks: int
    mean_reward: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_tool_calls: float
    mean_assistant_turns: float


@dataclass(frozen=True)
class BenchReport:
    schema_version: Literal[1]
    benchmark_digest: str | None
    warnings: tuple[str, ...]
    runs: tuple[RunSummary, ...]
    leaderboard: tuple[ModelMetrics, ...]
    matrix: dict[str, dict[str, float]]
    reference_tasks: tuple[ReferenceTask, ...]
    reference_models: tuple[ReferenceModelMetrics, ...]


@dataclass(frozen=True)
class _RunTraces:
    run_dir: Path
    model: str
    tool_contract: str
    omp_version: str
    rollouts: tuple[Rollout, ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_json_lines(path: Path) -> list[object] | BenchFailure:
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_TRACE_FILE_BYTES + 1)
    except OSError as error:
        return BenchFailure(f"trace file is not readable: {path}: {error}")
    if len(content) > MAX_TRACE_FILE_BYTES:
        return BenchFailure(f"trace file exceeds the size limit: {path}")
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        return BenchFailure(f"trace file is not UTF-8: {path}: {error}")
    rows: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            return BenchFailure(
                f"trace file has invalid JSON at {path}:{line_number}: {error}"
            )
    return rows


def _read_traces(path: Path) -> list[Mapping[str, object]] | BenchFailure:
    rows = _read_json_lines(path)
    if isinstance(rows, BenchFailure):
        return rows
    traces: list[Mapping[str, object]] = []
    for row in rows:
        raw_traces = row.get("traces") if isinstance(row, Mapping) else None
        if not isinstance(raw_traces, list):
            return BenchFailure(f"trace record is not an episode: {path}")
        for trace in raw_traces:
            if not isinstance(trace, Mapping):
                return BenchFailure(f"trace entry is not an object: {path}")
            traces.append(trace)
    if not traces:
        return BenchFailure(f"trace file contains no traces: {path}")
    return traces


def _trace_model(trace: Mapping[str, object]) -> str | BenchFailure:
    agent = trace.get("agent")
    config = agent.get("config") if isinstance(agent, Mapping) else None
    model = config.get("model") if isinstance(config, Mapping) else None
    if isinstance(model, str) and model:
        return model
    raw_calls = trace.get("calls")
    for raw_call in raw_calls if isinstance(raw_calls, list) else ():
        call_model = raw_call.get("model") if isinstance(raw_call, Mapping) else None
        if isinstance(call_model, str) and call_model:
            return call_model
    return BenchFailure("trace has no model identity")


def _trace_reward(trace: Mapping[str, object]) -> float | BenchFailure:
    rewards = trace.get("rewards")
    if not isinstance(rewards, Mapping):
        return BenchFailure("trace rewards are missing")
    total = 0.0
    for name, raw_reward in rewards.items():
        if raw_reward is None:
            continue
        if not isinstance(raw_reward, Mapping):
            return BenchFailure(f"trace reward is invalid: {name}")
        score = _number(raw_reward.get("score"))
        weight = _number(raw_reward.get("weight", 1.0))
        if score is None or weight is None:
            return BenchFailure(f"trace reward is invalid: {name}")
        total += score * weight
    return total


def _trace_tokens(trace: Mapping[str, object]) -> tuple[int, int] | BenchFailure:
    raw_calls = trace.get("calls")
    if not isinstance(raw_calls, list):
        return BenchFailure("trace model calls are missing")
    input_tokens = 0
    output_tokens = 0
    for raw_call in raw_calls:
        usage = raw_call.get("usage") if isinstance(raw_call, Mapping) else None
        if usage is None:
            continue
        if not isinstance(usage, Mapping):
            return BenchFailure("trace model call usage is invalid")
        prompt = _integer(usage.get("prompt_tokens"))
        completion = _integer(usage.get("completion_tokens"))
        cached = _integer(usage.get("cached_input_tokens", 0))
        if prompt is None or completion is None or cached is None:
            return BenchFailure("trace model call usage is invalid")
        input_tokens += prompt + cached
        output_tokens += completion
    return input_tokens, output_tokens


def _trace_tool_calls(trace: Mapping[str, object]) -> int | BenchFailure:
    raw_nodes = trace.get("nodes")
    if not isinstance(raw_nodes, list):
        return BenchFailure("trace nodes are missing")
    count = 0
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping) or raw_node.get("sampled") is not True:
            continue
        message = raw_node.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def _trace_seconds(trace: Mapping[str, object]) -> float:
    timing = trace.get("timing")
    span = timing.get("agent") if isinstance(timing, Mapping) else None
    start = _number(span.get("start")) if isinstance(span, Mapping) else None
    end = _number(span.get("end")) if isinstance(span, Mapping) else None
    if start is None or end is None or end <= start:
        return 0.0
    return end - start


def _task_identity(
    trace: Mapping[str, object],
) -> tuple[str, int, str] | BenchFailure:
    task = trace.get("task")
    data = task.get("data") if isinstance(task, Mapping) else None
    if not isinstance(data, Mapping):
        return BenchFailure("trace task data is missing")
    task_id = data.get("task_id")
    revision = _integer(data.get("task_revision"))
    digest = data.get("task_digest")
    if not isinstance(task_id, str) or revision is None or not isinstance(digest, str):
        return BenchFailure("trace task identity is invalid")
    return task_id, revision, digest


def _rollout(trace: Mapping[str, object]) -> Rollout | BenchFailure:
    identity = _task_identity(trace)
    if isinstance(identity, BenchFailure):
        return identity
    model = _trace_model(trace)
    if isinstance(model, BenchFailure):
        return model
    reward = _trace_reward(trace)
    if isinstance(reward, BenchFailure):
        return reward
    tokens = _trace_tokens(trace)
    if isinstance(tokens, BenchFailure):
        return tokens
    tool_calls = _trace_tool_calls(trace)
    if isinstance(tool_calls, BenchFailure):
        return tool_calls
    task_id, revision, digest = identity
    return Rollout(
        task_id=task_id,
        task_revision=revision,
        task_digest=digest,
        model=model,
        reward=reward,
        ok=trace.get("ok") is True,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        tool_calls=tool_calls,
        seconds=_trace_seconds(trace),
    )


def _info_field(trace: Mapping[str, object], key: str) -> str:
    info = trace.get("info")
    value = info.get(key) if isinstance(info, Mapping) else None
    return value if isinstance(value, str) else ""


def _run_trace_file(run_dir: Path) -> Path | BenchFailure:
    if not run_dir.is_dir():
        return BenchFailure(f"run directory does not exist: {run_dir}")
    trace_files = sorted(
        candidate for candidate in run_dir.rglob("traces.jsonl") if candidate.is_file()
    )
    if len(trace_files) != 1:
        return BenchFailure(
            f"run directory holds {len(trace_files)} trace files, expected one: "
            f"{run_dir}"
        )
    return trace_files[0]


def _read_run(run_dir: Path) -> _RunTraces | BenchFailure:
    trace_file = _run_trace_file(run_dir)
    if isinstance(trace_file, BenchFailure):
        return trace_file
    traces = _read_traces(trace_file)
    if isinstance(traces, BenchFailure):
        return traces
    rollouts: list[Rollout] = []
    first = traces[0]
    tool_contract = _info_field(first, "omp_tool_contract")
    omp_version = _info_field(first, "omp_version")
    for trace in traces:
        rollout = _rollout(trace)
        if isinstance(rollout, BenchFailure):
            return BenchFailure(f"{rollout.reason}: {trace_file}")
        if rollouts and rollout.model != rollouts[0].model:
            return BenchFailure(
                f"run mixes models {rollouts[0].model!r} and {rollout.model!r}: "
                f"{run_dir}"
            )
        if _info_field(trace, "omp_tool_contract") != tool_contract:
            return BenchFailure(f"run mixes OMP tool contracts: {run_dir}")
        rollouts.append(rollout)
    return _RunTraces(
        run_dir=run_dir,
        model=rollouts[0].model,
        tool_contract=tool_contract,
        omp_version=omp_version,
        rollouts=tuple(rollouts),
    )


def benchmark_digest(rollouts: Sequence[Rollout]) -> str:
    """Identify the task set a run exercised, independent of rollout order."""
    identities = sorted(
        {(r.task_id, r.task_revision, r.task_digest) for r in rollouts}
    )
    return hashlib.sha256(_canonical_json(identities)).hexdigest()


def pass_at_k(total: int, passed: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n - c, k) / C(n, k)."""
    if k < 1 or k > total:
        raise ValueError("k must be between 1 and the rollout count")
    if total - passed < k:
        return 1.0
    return 1.0 - math.comb(total - passed, k) / math.comb(total, k)


def _model_metrics(model: str, rollouts: Sequence[Rollout]) -> ModelMetrics:
    by_task: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        by_task.setdefault(rollout.task_id, []).append(rollout)
    k = min(len(task_rollouts) for task_rollouts in by_task.values())
    pass_at_k_by_task: dict[str, float] = {}
    if k > 1:
        for task_id, task_rollouts in sorted(by_task.items()):
            passed = sum(1 for r in task_rollouts if r.reward == PASSING_REWARD)
            pass_at_k_by_task[task_id] = pass_at_k(len(task_rollouts), passed, k)
    return ModelMetrics(
        model=model,
        tasks=len(by_task),
        rollouts=len(rollouts),
        mean_reward=_mean([r.reward for r in rollouts]),
        pass_rate=_mean([1.0 if r.reward == PASSING_REWARD else 0.0 for r in rollouts]),
        pass_at_k=_mean(list(pass_at_k_by_task.values())) if k > 1 else None,
        k=k,
        pass_at_k_by_task=pass_at_k_by_task,
        error_rate=_mean([0.0 if r.ok else 1.0 for r in rollouts]),
        mean_input_tokens=_mean([float(r.input_tokens) for r in rollouts]),
        mean_output_tokens=_mean([float(r.output_tokens) for r in rollouts]),
        mean_tool_calls=_mean([float(r.tool_calls) for r in rollouts]),
        mean_seconds=_mean([r.seconds for r in rollouts]),
    )


def _leaderboard(runs: Sequence[_RunTraces]) -> tuple[ModelMetrics, ...]:
    by_model: dict[str, list[Rollout]] = {}
    for run in runs:
        by_model.setdefault(run.model, []).extend(run.rollouts)
    metrics = [_model_metrics(model, rollouts) for model, rollouts in by_model.items()]
    metrics.sort(key=lambda entry: (-entry.mean_reward, -entry.pass_rate, entry.model))
    return tuple(metrics)


def _matrix(runs: Sequence[_RunTraces]) -> dict[str, dict[str, float]]:
    rewards: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        for rollout in run.rollouts:
            rewards.setdefault(rollout.task_id, {}).setdefault(run.model, []).append(
                rollout.reward
            )
    return {
        task_id: {model: _mean(values) for model, values in sorted(by_model.items())}
        for task_id, by_model in sorted(rewards.items())
    }


def _task_id(task_dir: Path) -> str | BenchFailure:
    config_path = task_dir / "task.toml"
    try:
        config_bytes = config_path.read_bytes()
    except OSError as error:
        return BenchFailure(f"task.toml is not readable: {config_path}: {error}")
    if len(config_bytes) > MAX_TASK_CONFIG_BYTES:
        return BenchFailure(f"task.toml exceeds its size limit: {config_path}")
    try:
        raw = tomllib.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return BenchFailure(f"task.toml is invalid: {config_path}: {error}")
    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return BenchFailure(f"task.toml has no task_id: {config_path}")
    return task_id


def _reference_usage(raw: object) -> ReferenceUsage | BenchFailure:
    if not isinstance(raw, Mapping):
        return BenchFailure("provenance usage is missing")
    counts: dict[str, int] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
    ):
        count = _integer(raw.get(key))
        if count is None:
            return BenchFailure(f"provenance usage field is invalid: {key}")
        counts[key] = count
    cost = _number(raw.get("cost"))
    if cost is None:
        return BenchFailure("provenance usage field is invalid: cost")
    return ReferenceUsage(cost=cost, **counts)


def _reference_task(task_dir: Path) -> ReferenceTask | BenchFailure:
    task_id = _task_id(task_dir)
    if isinstance(task_id, BenchFailure):
        return task_id
    path = task_dir / "provenance.json"
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_PROVENANCE_BYTES + 1)
    except OSError as error:
        return BenchFailure(f"provenance is not readable: {path}: {error}")
    if len(content) > MAX_PROVENANCE_BYTES:
        return BenchFailure(f"provenance exceeds its size limit: {path}")
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return BenchFailure(f"provenance is invalid: {path}: {error}")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        return BenchFailure(f"provenance schema version is invalid: {path}")
    models = raw.get("models")
    assistant_turns = _integer(raw.get("assistant_turns"))
    tool_calls = _integer(raw.get("tool_calls"))
    if (
        not isinstance(models, list)
        or not all(isinstance(model, str) and model for model in models)
        or assistant_turns is None
        or tool_calls is None
    ):
        return BenchFailure(f"provenance fields are invalid: {path}")
    usage = _reference_usage(raw.get("usage"))
    if isinstance(usage, BenchFailure):
        return BenchFailure(f"{usage.reason}: {path}")
    # The minting gate only keeps sessions whose reference patch passes.
    return ReferenceTask(
        task_id=task_id,
        models=tuple(models),
        usage=usage,
        assistant_turns=assistant_turns,
        tool_calls=tool_calls,
        reward=PASSING_REWARD,
    )


def _reference_tasks(tasks_dir: Path) -> tuple[ReferenceTask, ...] | BenchFailure:
    if not tasks_dir.is_dir():
        return BenchFailure(f"tasks directory does not exist: {tasks_dir}")
    references: list[ReferenceTask] = []
    for child in sorted(tasks_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or not (child / "task.toml").is_file():
            continue
        if not (child / "provenance.json").is_file():
            continue
        reference = _reference_task(child)
        if isinstance(reference, BenchFailure):
            return reference
        references.append(reference)
    return tuple(references)


def _reference_models(
    references: Sequence[ReferenceTask],
) -> tuple[ReferenceModelMetrics, ...]:
    by_model: dict[str, list[ReferenceTask]] = {}
    for reference in references:
        by_model.setdefault(" + ".join(reference.models), []).append(reference)
    return tuple(
        ReferenceModelMetrics(
            model=model,
            tasks=len(tasks),
            mean_reward=_mean([task.reward for task in tasks]),
            mean_input_tokens=_mean([float(t.usage.input_tokens) for t in tasks]),
            mean_output_tokens=_mean([float(t.usage.output_tokens) for t in tasks]),
            mean_tool_calls=_mean([float(t.tool_calls) for t in tasks]),
            mean_assistant_turns=_mean([float(t.assistant_turns) for t in tasks]),
        )
        for model, tasks in sorted(by_model.items())
    )


def aggregate_runs(
    run_dirs: Sequence[Path], *, tasks_dir: Path | None
) -> BenchReport | BenchFailure:
    """Fold one traces.jsonl per run directory into a leaderboard and task matrix."""
    if not run_dirs:
        return BenchFailure("at least one run directory is required")
    runs: list[_RunTraces] = []
    summaries: list[RunSummary] = []
    for run_dir in run_dirs:
        run = _read_run(run_dir)
        if isinstance(run, BenchFailure):
            return run
        runs.append(run)
        summaries.append(
            RunSummary(
                run_dir=str(run_dir),
                model=run.model,
                tool_contract=run.tool_contract,
                omp_version=run.omp_version,
                benchmark_digest=benchmark_digest(run.rollouts),
                rollouts=len(run.rollouts),
            )
        )
    digests = sorted({summary.benchmark_digest for summary in summaries})
    warnings: list[str] = []
    if len(digests) > 1:
        warnings.append(
            "runs exercised different task sets; the leaderboard is not comparable: "
            + ", ".join(
                f"{summary.run_dir} ({summary.benchmark_digest[:12]})"
                for summary in summaries
            )
        )
    references: tuple[ReferenceTask, ...] = ()
    if tasks_dir is not None:
        loaded = _reference_tasks(tasks_dir)
        if isinstance(loaded, BenchFailure):
            return loaded
        references = loaded
    return BenchReport(
        schema_version=1,
        benchmark_digest=digests[0] if len(digests) == 1 else None,
        warnings=tuple(warnings),
        runs=tuple(summaries),
        leaderboard=_leaderboard(runs),
        matrix=_matrix(runs),
        reference_tasks=references,
        reference_models=_reference_models(references),
    )


def _cell(value: float | None, *, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _leaderboard_markdown(report: BenchReport) -> str:
    header = (
        "model",
        "tasks",
        "rollouts",
        "mean reward",
        "pass rate",
        "pass@k",
        "error rate",
        "input tokens",
        "output tokens",
        "tool calls",
        "seconds",
    )
    rows = [
        (
            f"`{entry.model}`",
            str(entry.tasks),
            str(entry.rollouts),
            _cell(entry.mean_reward),
            _cell(entry.pass_rate),
            f"{_cell(entry.pass_at_k)} (k={entry.k})",
            _cell(entry.error_rate),
            _cell(entry.mean_input_tokens, digits=0),
            _cell(entry.mean_output_tokens, digits=0),
            _cell(entry.mean_tool_calls, digits=1),
            _cell(entry.mean_seconds, digits=1),
        )
        for entry in report.leaderboard
    ]
    return _table(header, rows)


def _reference_markdown(report: BenchReport) -> str:
    header = (
        "session model",
        "tasks",
        "mean reward",
        "input tokens",
        "output tokens",
        "tool calls",
        "assistant turns",
    )
    rows = [
        (
            f"`{entry.model}`",
            str(entry.tasks),
            _cell(entry.mean_reward),
            _cell(entry.mean_input_tokens, digits=0),
            _cell(entry.mean_output_tokens, digits=0),
            _cell(entry.mean_tool_calls, digits=1),
            _cell(entry.mean_assistant_turns, digits=1),
        )
        for entry in report.reference_models
    ]
    return _table(header, rows)


def _matrix_markdown(report: BenchReport) -> str:
    models = [entry.model for entry in report.leaderboard]
    reference_by_task = {task.task_id: task for task in report.reference_tasks}
    header = ["task", *(f"`{model}`" for model in models)]
    if reference_by_task:
        header.append(REFERENCE_MODEL_LABEL)
    task_ids = sorted(set(report.matrix) | set(reference_by_task))
    rows: list[list[str]] = []
    for task_id in task_ids:
        by_model = report.matrix.get(task_id, {})
        row = [f"`{task_id}`", *(_cell(by_model.get(model)) for model in models)]
        if reference_by_task:
            reference = reference_by_task.get(task_id)
            row.append(_cell(reference.reward if reference else None))
        rows.append(row)
    return _table(header, rows)


def _runs_markdown(report: BenchReport) -> str:
    header = ("run", "model", "tool contract", "omp version", "task set", "rollouts")
    rows = [
        (
            f"`{run.run_dir}`",
            f"`{run.model}`",
            run.tool_contract or "-",
            run.omp_version or "-",
            f"`{run.benchmark_digest[:12]}`",
            str(run.rollouts),
        )
        for run in report.runs
    ]
    return _table(header, rows)


def render_markdown(report: BenchReport) -> str:
    """Render the leaderboard, reference rows, per-task matrix, and run list."""
    digest = (
        f"`{report.benchmark_digest}`"
        if report.benchmark_digest is not None
        else "mixed (see warnings)"
    )
    sections = ["# omp-coding benchmark", "", f"Task set: {digest}", ""]
    for warning in report.warnings:
        sections.append(f"> Warning: {warning}")
        sections.append("")
    sections.extend(["## Leaderboard", "", _leaderboard_markdown(report), ""])
    if report.reference_models:
        sections.extend(["## Reference sessions", "", _reference_markdown(report), ""])
    sections.extend(["## Per-task mean reward", "", _matrix_markdown(report), ""])
    sections.extend(["## Runs", "", _runs_markdown(report), ""])
    return "\n".join(sections)


def _write_atomic(path: Path, content: str) -> BenchFailure | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
    except OSError as error:
        return BenchFailure(f"bench output could not be created: {error}")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return BenchFailure(f"bench output could not be written: {error}")
    return None


def write_report(report: BenchReport, output_dir: Path) -> BenchFailure | None:
    """Write `bench.json` (canonical, sorted keys) and `bench.md` into `output_dir`."""
    document = json.dumps(
        asdict(report), ensure_ascii=False, indent=2, sort_keys=True
    )
    json_failure = _write_atomic(output_dir / BENCH_JSON, document + "\n")
    if json_failure is not None:
        return json_failure
    return _write_atomic(output_dir / BENCH_MARKDOWN, render_markdown(report))


def _default_runner(command: Sequence[str]) -> int:
    completed = subprocess.run(  # noqa: S603 - fixed Prime executable.
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")
    return completed.returncode


def _eval_command(
    *,
    eval_executable: Path,
    model: str,
    tasks_dir: Path | None,
    split: str,
    base_url: str | None,
    api_key_var: str | None,
    num_rollouts: int,
    max_concurrent: int,
    max_tokens: int,
    max_total_tokens: int,
    output_dir: Path,
) -> list[str]:
    command = [
        str(eval_executable),
        "omp-coding",
        "--model",
        model,
        "--no-push",
        "--no-rich",
        "--env.taskset.split",
        split,
    ]
    if tasks_dir is not None:
        command.extend(["--env.taskset.tasks-dir", str(tasks_dir)])
    if base_url is not None:
        command.extend(["--client.base-url", base_url])
    if api_key_var is not None:
        command.extend(["--client.api-key-var", api_key_var])
    command.extend(
        [
            "--num-rollouts",
            str(num_rollouts),
            "--max-concurrent",
            str(max_concurrent),
            "--sampling.max-tokens",
            str(max_tokens),
            "--env.agent.max-total-tokens",
            str(max_total_tokens),
            "--output-dir",
            str(output_dir),
        ]
    )
    return command


def _model_run_dir(output_dir: Path, model: str) -> Path:
    return output_dir / RUNS_DIR / MODEL_DIR_PATTERN.sub("_", model)


def run_models(
    models: Sequence[str],
    *,
    tasks_dir: Path | None,
    split: str,
    base_url: str | None,
    api_key_var: str | None,
    num_rollouts: int,
    max_concurrent: int,
    max_tokens: int,
    max_total_tokens: int,
    output_dir: Path,
    eval_executable: Path = Path("eval"),
    runner: CommandRunner = _default_runner,
) -> BenchReport | BenchFailure:
    """Evaluate each model through the Prime `eval` command, then aggregate the runs."""
    if not models:
        return BenchFailure("at least one model is required")
    if len(set(models)) != len(models):
        return BenchFailure("models must be unique")
    if num_rollouts < 1 or max_concurrent < 1:
        return BenchFailure("rollouts and concurrency must be positive")
    if max_tokens < 1 or max_total_tokens < 1:
        return BenchFailure("token limits must be positive")
    run_dirs: list[Path] = []
    for model in models:
        run_dir = _model_run_dir(output_dir, model)
        if run_dir.exists():
            return BenchFailure(f"run directory already exists: {run_dir}")
        command = _eval_command(
            eval_executable=eval_executable,
            model=model,
            tasks_dir=tasks_dir,
            split=split,
            base_url=base_url,
            api_key_var=api_key_var,
            num_rollouts=num_rollouts,
            max_concurrent=max_concurrent,
            max_tokens=max_tokens,
            max_total_tokens=max_total_tokens,
            output_dir=run_dir,
        )
        status = runner(command)
        if status != 0:
            return BenchFailure(f"Prime evaluation of {model} exited with status {status}")
        run_dirs.append(run_dir)
    report = aggregate_runs(run_dirs, tasks_dir=tasks_dir)
    if isinstance(report, BenchFailure):
        return report
    write_failure = write_report(report, output_dir)
    if write_failure is not None:
        return write_failure
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--tasks-dir", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--models", required=True, help="comma-separated model ids")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--tasks-dir", type=Path)
    run.add_argument("--split", default=DEFAULT_SPLIT)
    run.add_argument("--client-base-url")
    run.add_argument("--client-api-key-var")
    run.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    run.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--max-total-tokens", type=int, default=DEFAULT_MAX_TOTAL_TOKENS)
    run.add_argument("--eval-executable", type=Path, default=Path("eval"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "aggregate":
        result: BenchReport | BenchFailure = aggregate_runs(
            arguments.run_dirs, tasks_dir=arguments.tasks_dir
        )
        if not isinstance(result, BenchFailure):
            write_failure = write_report(result, arguments.output)
            if write_failure is not None:
                result = write_failure
    elif arguments.command == "run":
        models = [model.strip() for model in arguments.models.split(",")]
        result = run_models(
            [model for model in models if model],
            tasks_dir=arguments.tasks_dir,
            split=arguments.split,
            base_url=arguments.client_base_url,
            api_key_var=arguments.client_api_key_var,
            num_rollouts=arguments.num_rollouts,
            max_concurrent=arguments.max_concurrent,
            max_tokens=arguments.max_tokens,
            max_total_tokens=arguments.max_total_tokens,
            output_dir=arguments.output,
            eval_executable=arguments.eval_executable,
        )
    else:
        raise AssertionError(f"unknown command: {arguments.command}")
    if isinstance(result, BenchFailure):
        print(json.dumps(asdict(result), sort_keys=True))
        return 1
    print(render_markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
