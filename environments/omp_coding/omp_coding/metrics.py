"""Measure generated tool behavior and sealed rewards from Prime traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .protocol import validate_tool_arguments


@dataclass(frozen=True)
class MetricsFailure:
    reason: str


@dataclass(frozen=True)
class TraceMetricsReport:
    schema_version: Literal[1]
    parser: str
    trace_files: int
    traces: int
    sampled_turns: int
    parsed_turns: int
    parsed_calls: int
    tool_call_attempts: int
    invalid_calls: int
    unavailable_tool_calls: int
    repeated_call_turns: int
    completed_model_calls: int
    model_calls: int
    length_limited_calls: int
    passed_cases: int
    total_cases: int
    sealed_validation_reward: float
    parsed_tool_call_rate: float
    invalid_tool_rate: float
    end_token_rate: float
    loop_rate: float
    comparison_sha256: str


@dataclass(frozen=True)
class MetricsComparison:
    schema_version: Literal[1]
    status: Literal["improved", "rejected"]
    reasons: tuple[str, ...]
    comparison_sha256: str
    baseline_reward: float
    candidate_reward: float
    reward_delta: float
    baseline_parsed_tool_call_rate: float
    candidate_parsed_tool_call_rate: float
    baseline_invalid_tool_rate: float
    candidate_invalid_tool_rate: float
    baseline_end_token_rate: float
    candidate_end_token_rate: float
    baseline_loop_rate: float
    candidate_loop_rate: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _trace_files(inputs: Sequence[Path]) -> tuple[Path, ...] | MetricsFailure:
    selected: set[Path] = set()
    for input_path in inputs:
        path = input_path.resolve()
        if path.is_file():
            selected.add(path)
        elif path.is_dir():
            selected.update(
                candidate
                for candidate in path.rglob("traces.jsonl")
                if candidate.is_file()
            )
        else:
            return MetricsFailure(f"trace input does not exist: {input_path}")
    if not selected:
        return MetricsFailure("no Prime trace files were found")
    return tuple(sorted(selected))


def _read_traces(paths: Sequence[Path]) -> list[Mapping[str, object]] | MetricsFailure:
    traces: list[Mapping[str, object]] = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            return MetricsFailure(f"trace file is not readable: {path}: {error}")
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                return MetricsFailure(
                    f"trace file has invalid JSON at {path}:{line_number}: {error}"
                )
            raw_traces = record.get("traces") if isinstance(record, Mapping) else None
            if not isinstance(raw_traces, list):
                return MetricsFailure(
                    f"trace record is invalid at {path}:{line_number}"
                )
            for trace in raw_traces:
                if not isinstance(trace, Mapping):
                    return MetricsFailure("Prime trace entry is not an object")
                traces.append(trace)
    if not traces:
        return MetricsFailure("Prime trace files contain no traces")
    return traces


def _tool_schemas(
    trace: Mapping[str, object],
) -> dict[str, Mapping[str, object]] | MetricsFailure:
    raw_tools = trace.get("tools")
    if not isinstance(raw_tools, list):
        return MetricsFailure("trace tool contract is missing")
    schemas: dict[str, Mapping[str, object]] = {}
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            return MetricsFailure("trace tool entry is invalid")
        name = raw_tool.get("name")
        parameters = raw_tool.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            return MetricsFailure("trace tool schema is invalid")
        schemas[name] = parameters
    return schemas


def _arguments(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _call_parts(value: object) -> tuple[str, Mapping[str, object]] | None:
    if not isinstance(value, Mapping):
        return None
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = _arguments(function.get("arguments"))
    else:
        name = value.get("name")
        arguments = _arguments(value.get("arguments"))
    if not isinstance(name, str) or arguments is None:
        return None
    return name, arguments


def _sampled_messages(
    trace: Mapping[str, object],
) -> list[Mapping[str, object]] | MetricsFailure:
    raw_nodes = trace.get("nodes")
    if not isinstance(raw_nodes, list):
        return MetricsFailure("trace nodes are missing")
    messages: list[Mapping[str, object]] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping) or raw_node.get("sampled") is not True:
            continue
        raw_message = raw_node.get("message")
        if not isinstance(raw_message, Mapping):
            return MetricsFailure("sampled trace message is invalid")
        if raw_message.get("role") == "assistant":
            messages.append(raw_message)
    return messages


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _comparison_entry(
    trace: Mapping[str, object], parser: str
) -> Mapping[str, object] | MetricsFailure:
    task = trace.get("task")
    task_data = task.get("data") if isinstance(task, Mapping) else None
    if not isinstance(task_data, Mapping):
        return MetricsFailure("trace task data is missing")
    task_id = task_data.get("task_id")
    task_revision = task_data.get("task_revision")
    task_digest = task_data.get("task_digest")
    prompt = task_data.get("prompt")
    system_prompt = task_data.get("system_prompt")
    split = task_data.get("split")
    if (
        not isinstance(task_id, str)
        or not isinstance(task_revision, int)
        or not isinstance(task_digest, str)
        or not isinstance(prompt, str)
        or not isinstance(system_prompt, str)
        or not isinstance(split, str)
    ):
        return MetricsFailure("trace task comparison identity is invalid")
    raw_calls = trace.get("calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return MetricsFailure("trace model calls are missing")
    sampling: Mapping[str, object] | None = None
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            return MetricsFailure("trace model call is invalid")
        current_sampling = raw_call.get("sampling")
        if not isinstance(current_sampling, Mapping):
            return MetricsFailure("trace model call settings are invalid")
        if sampling is None:
            sampling = current_sampling
        elif _canonical_json(current_sampling) != _canonical_json(sampling):
            return MetricsFailure("sampling settings changed within one trace")
    return {
        "task_id": task_id,
        "task_revision": task_revision,
        "task_digest": task_digest,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "split": split,
        "tools": trace.get("tools"),
        "parser": parser,
        "sampling": dict(sampling),
    }


def measure_traces(
    inputs: Sequence[Path],
    *,
    parser: str,
    baseline: TraceMetricsReport | None = None,
) -> TraceMetricsReport | MetricsFailure:
    """Measure model protocol behavior without parsing assistant text."""
    if not parser:
        return MetricsFailure("tool parser identity is required")
    paths = _trace_files(inputs)
    if isinstance(paths, MetricsFailure):
        return paths
    traces = _read_traces(paths)
    if isinstance(traces, MetricsFailure):
        return traces

    sampled_turns = 0
    parsed_turns = 0
    parsed_calls = 0
    invalid_calls = 0
    tool_call_attempts = 0
    unavailable_calls = 0
    repeated_turns = 0
    completed_calls = 0
    model_calls = 0
    length_calls = 0
    passed_cases = 0
    total_cases = 0
    rewards: list[float] = []
    comparison_entries: list[Mapping[str, object]] = []

    for trace in traces:
        schemas = _tool_schemas(trace)
        if isinstance(schemas, MetricsFailure):
            return schemas
        messages = _sampled_messages(trace)
        if isinstance(messages, MetricsFailure):
            return messages
        sampled_turns += len(messages)
        previous_signatures: tuple[bytes, ...] | None = None
        for message in messages:
            raw_calls = message.get("tool_calls", [])
            if not isinstance(raw_calls, list):
                invalid_calls += 1
                tool_call_attempts += 1
                continue
            if raw_calls:
                parsed_turns += 1
            signatures: list[bytes] = []
            for raw_call in raw_calls:
                tool_call_attempts += 1
                parts = _call_parts(raw_call)
                if parts is None:
                    invalid_calls += 1
                    continue
                parsed_calls += 1
                name, arguments = parts
                signatures.append(
                    _canonical_json({"name": name, "arguments": dict(arguments)})
                )
                schema = schemas.get(name)
                if schema is None:
                    invalid_calls += 1
                    unavailable_calls += 1
                    continue
                schema_error = validate_tool_arguments(schema, arguments)
                if schema_error is not None:
                    invalid_calls += 1
                    continue
            current_signatures = tuple(signatures)
            repeated = len(current_signatures) != len(set(current_signatures)) or (
                previous_signatures is not None
                and current_signatures
                and current_signatures == previous_signatures
            )
            if repeated:
                repeated_turns += 1
            if current_signatures:
                previous_signatures = current_signatures

        raw_model_calls = trace.get("calls")
        if not isinstance(raw_model_calls, list):
            return MetricsFailure("trace model calls are missing")
        for raw_call in raw_model_calls:
            if not isinstance(raw_call, Mapping):
                return MetricsFailure("trace model call is invalid")
            finish_reason = raw_call.get("finish_reason")
            if not isinstance(finish_reason, str):
                return MetricsFailure("trace finish reason is invalid")
            model_calls += 1
            if finish_reason in {"stop", "tool_calls"}:
                completed_calls += 1
            if finish_reason == "length":
                length_calls += 1

        reward_document = trace.get("rewards")
        tests = (
            reward_document.get("tests")
            if isinstance(reward_document, Mapping)
            else None
        )
        score = tests.get("score") if isinstance(tests, Mapping) else None
        numeric_score = _number(score)
        if numeric_score is None:
            return MetricsFailure("trace tests reward is missing")
        rewards.append(numeric_score)
        metrics = trace.get("metrics")
        raw_passed = (
            metrics.get("passed_cases") if isinstance(metrics, Mapping) else None
        )
        raw_total = metrics.get("total_cases") if isinstance(metrics, Mapping) else None
        current_passed = _integer(raw_passed)
        current_total = _integer(raw_total)
        if current_passed is not None:
            passed_cases += current_passed
        if current_total is not None:
            total_cases += current_total
        comparison_entry = _comparison_entry(trace, parser)
        if isinstance(comparison_entry, MetricsFailure):
            return comparison_entry
        comparison_entries.append(comparison_entry)

    comparison_sha256 = hashlib.sha256(
        _canonical_json(
            sorted(
                comparison_entries,
                key=lambda entry: (
                    str(entry["task_id"]),
                    _canonical_json(entry).decode("utf-8"),
                ),
            )
        )
    ).hexdigest()
    report = TraceMetricsReport(
        schema_version=1,
        parser=parser,
        trace_files=len(paths),
        traces=len(traces),
        sampled_turns=sampled_turns,
        parsed_turns=parsed_turns,
        parsed_calls=parsed_calls,
        tool_call_attempts=tool_call_attempts,
        invalid_calls=invalid_calls,
        unavailable_tool_calls=unavailable_calls,
        repeated_call_turns=repeated_turns,
        completed_model_calls=completed_calls,
        model_calls=model_calls,
        length_limited_calls=length_calls,
        passed_cases=passed_cases,
        total_cases=total_cases,
        sealed_validation_reward=sum(rewards) / len(rewards),
        parsed_tool_call_rate=(
            parsed_calls / tool_call_attempts if tool_call_attempts else 0.0
        ),
        invalid_tool_rate=(
            invalid_calls / tool_call_attempts if tool_call_attempts else 0.0
        ),
        end_token_rate=(completed_calls / model_calls if model_calls else 0.0),
        loop_rate=(repeated_turns / parsed_turns if parsed_turns else 0.0),
        comparison_sha256=comparison_sha256,
    )
    if baseline is not None and report.comparison_sha256 != baseline.comparison_sha256:
        return MetricsFailure("candidate comparison settings differ from the baseline")
    return report


def compare_metrics(
    baseline: TraceMetricsReport,
    candidate: TraceMetricsReport,
) -> MetricsComparison | MetricsFailure:
    """Accept only a sealed reward gain without tool behavior degradation."""
    if candidate.comparison_sha256 != baseline.comparison_sha256:
        return MetricsFailure("candidate comparison settings differ from the baseline")
    reasons: list[str] = []
    if candidate.sealed_validation_reward <= baseline.sealed_validation_reward:
        reasons.append("sealed validation reward did not increase")
    if candidate.parsed_calls == 0:
        reasons.append("candidate produced zero parsed tool calls")
    if candidate.parsed_tool_call_rate < baseline.parsed_tool_call_rate:
        reasons.append("parsed tool-call rate decreased")
    if candidate.invalid_tool_rate > baseline.invalid_tool_rate:
        reasons.append("invalid tool-call rate increased")
    if candidate.unavailable_tool_calls > 0:
        reasons.append("candidate called an unavailable tool")
    if candidate.end_token_rate < baseline.end_token_rate:
        reasons.append("end-token rate decreased")
    if candidate.loop_rate > baseline.loop_rate:
        reasons.append("tool-call loop rate increased")
    return MetricsComparison(
        schema_version=1,
        status="rejected" if reasons else "improved",
        reasons=tuple(reasons),
        comparison_sha256=baseline.comparison_sha256,
        baseline_reward=baseline.sealed_validation_reward,
        candidate_reward=candidate.sealed_validation_reward,
        reward_delta=(
            candidate.sealed_validation_reward - baseline.sealed_validation_reward
        ),
        baseline_parsed_tool_call_rate=baseline.parsed_tool_call_rate,
        candidate_parsed_tool_call_rate=candidate.parsed_tool_call_rate,
        baseline_invalid_tool_rate=baseline.invalid_tool_rate,
        candidate_invalid_tool_rate=candidate.invalid_tool_rate,
        baseline_end_token_rate=baseline.end_token_rate,
        candidate_end_token_rate=candidate.end_token_rate,
        baseline_loop_rate=baseline.loop_rate,
        candidate_loop_rate=candidate.loop_rate,
    )


def load_metrics_report(path: Path) -> TraceMetricsReport | MetricsFailure:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return MetricsFailure(f"metrics report is not readable: {path}: {error}")
    if not isinstance(raw, Mapping):
        return MetricsFailure("metrics report is not an object")
    try:
        return TraceMetricsReport(
            schema_version=1,
            parser=str(raw["parser"]),
            trace_files=int(raw["trace_files"]),
            traces=int(raw["traces"]),
            sampled_turns=int(raw["sampled_turns"]),
            parsed_turns=int(raw["parsed_turns"]),
            parsed_calls=int(raw["parsed_calls"]),
            tool_call_attempts=int(raw["tool_call_attempts"]),
            invalid_calls=int(raw["invalid_calls"]),
            unavailable_tool_calls=int(raw["unavailable_tool_calls"]),
            repeated_call_turns=int(raw["repeated_call_turns"]),
            completed_model_calls=int(raw["completed_model_calls"]),
            model_calls=int(raw["model_calls"]),
            length_limited_calls=int(raw["length_limited_calls"]),
            passed_cases=int(raw["passed_cases"]),
            total_cases=int(raw["total_cases"]),
            sealed_validation_reward=float(raw["sealed_validation_reward"]),
            parsed_tool_call_rate=float(raw["parsed_tool_call_rate"]),
            invalid_tool_rate=float(raw["invalid_tool_rate"]),
            end_token_rate=float(raw["end_token_rate"]),
            loop_rate=float(raw["loop_rate"]),
            comparison_sha256=str(raw["comparison_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        return MetricsFailure(f"metrics report fields are invalid: {error}")


def _write_report(path: Path, value: object) -> MetricsFailure | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
    except OSError as error:
        return MetricsFailure(f"metrics output could not be created: {error}")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                asdict(value),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return MetricsFailure(f"metrics output could not be written: {error}")
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser("measure")
    measure.add_argument("inputs", nargs="+", type=Path)
    measure.add_argument("--parser", required=True)
    measure.add_argument("--baseline", type=Path)
    measure.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "measure":
        baseline: TraceMetricsReport | None = None
        if arguments.baseline is not None:
            loaded_baseline = load_metrics_report(arguments.baseline)
            if isinstance(loaded_baseline, MetricsFailure):
                print(json.dumps(asdict(loaded_baseline), sort_keys=True))
                return 1
            baseline = loaded_baseline
        result: object = measure_traces(
            arguments.inputs,
            parser=arguments.parser,
            baseline=baseline,
        )
    elif arguments.command == "compare":
        loaded_baseline = load_metrics_report(arguments.baseline)
        if isinstance(loaded_baseline, MetricsFailure):
            result = loaded_baseline
        else:
            loaded_candidate = load_metrics_report(arguments.candidate)
            if isinstance(loaded_candidate, MetricsFailure):
                result = loaded_candidate
            else:
                result = compare_metrics(loaded_baseline, loaded_candidate)
    else:
        raise AssertionError(f"unknown command: {arguments.command}")
    if isinstance(result, MetricsFailure):
        print(json.dumps(asdict(result), sort_keys=True))
        return 1
    write_failure = _write_report(arguments.output, result)
    if write_failure is not None:
        print(json.dumps(asdict(write_failure), sort_keys=True))
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    if isinstance(result, MetricsComparison) and result.status == "rejected":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
