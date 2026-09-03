"""Export Prime Verifiers v1 traces and train one MLX LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from .hardware import MetalFailure, MetalReport, metal_preflight
from .protocol import (
    ProtocolFailure,
    ProtocolReport,
    load_protocol_context,
    protocol_gate_failure,
    run_local_protocol_gate,
)

LOSS_PATTERN = re.compile(r"(Train|Val) loss ([^\s,]+)")
DATA_FILES = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "holdout": "test.jsonl",
}

ActionKind: TypeAlias = Literal[
    "read",
    "write",
    "edit",
    "execute",
    "test",
    "recovery",
    "final",
]
ALL_ACTION_KINDS: frozenset[ActionKind] = frozenset(
    {"read", "write", "edit", "execute", "test", "recovery", "final"}
)
REQUIRED_ACTION_KINDS: frozenset[ActionKind] = ALL_ACTION_KINDS
RPC_HOST_TOOL_CONTRACT = "rpc-host-v1"
NATIVE_TOOL_CONTRACT = "omp-native-v1"
# Traces written before info.omp_tool_contract existed came from the sealed
# five-tool RPC host, so a missing contract means rpc-host-v1.
DEFAULT_TOOL_CONTRACT = RPC_HOST_TOOL_CONTRACT
# Native OMP tools expose no sealed test runner and report failures as plain
# text, so a native dataset cannot promise test or recovery turns.
REQUIRED_ACTION_KINDS_BY_CONTRACT: dict[str, frozenset[ActionKind]] = {
    RPC_HOST_TOOL_CONTRACT: ALL_ACTION_KINDS,
    NATIVE_TOOL_CONTRACT: frozenset({"read", "write", "edit", "execute", "final"}),
}
MINIMUM_SEQUENCE_LENGTH = 8192
MINIMUM_TRAIN_TRAJECTORIES = 200
MINIMUM_VALIDATION_TRAJECTORIES = 4
DATASET_MANIFEST = "manifest.json"
MAX_DATASET_FILE_BYTES = 256 * 1024 * 1024
MAX_TRACE_FILE_BYTES = 128 * 1024 * 1024
MAX_TRACE_INPUT_BYTES = 128 * 1024 * 1024
MAX_TRACE_FILES = 256
MAX_TRACES = 10_000
MAX_TRACE_NODES = 1024
MAX_RETAINED_SAMPLES = 10_000
ACTION_BY_TOOL: dict[str, ActionKind] = {
    "sandbox_read": "read",
    "sandbox_write": "write",
    "sandbox_edit": "edit",
    "sandbox_exec": "execute",
    "run_tests": "test",
    "read": "read",
    "grep": "read",
    "glob": "read",
    "write": "write",
    "edit": "edit",
    "bash": "execute",
}


@dataclass(frozen=True)
class ExportFailure:
    reason: str


@dataclass(frozen=True)
class ExportReport:
    trace_files: int
    traces_seen: int
    successful_traces: int
    train_trajectories: int
    valid_trajectories: int
    samples_written: int
    train_samples: int
    valid_samples: int
    test_samples: int
    action_counts: tuple[tuple[ActionKind, int], ...]
    dataset_sha256: str
    tool_contract: str


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: Literal[2]
    dataset_sha256: str
    train_trajectories: int
    valid_trajectories: int
    action_counts: tuple[tuple[ActionKind, int], ...]
    samples_per_trace_limit: int
    tool_contract: str = DEFAULT_TOOL_CONTRACT


@dataclass(frozen=True)
class TrainingFailure:
    reason: str


@dataclass(frozen=True)
class TrainingReport:
    model: str
    data: str
    adapter: str
    iterations: int
    max_sequence_length: int
    losses: tuple[float, ...]
    train_samples: int
    valid_samples: int
    test_samples: int
    dropped_samples: int
    adapter_bytes: int
    checkpoint_files: tuple[str, ...]
    protocol: ProtocolReport
    metal: MetalReport


def _report_path(path: Path) -> str:
    return path.name if path.is_absolute() else str(path)


@dataclass(frozen=True)
class _SampleLength:
    total_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class _PreparedTrainingData:
    path: Path
    train_samples: int
    valid_samples: int
    test_samples: int
    dropped_samples: int


@dataclass(frozen=True)
class _TraceConversion:
    split: str
    trace_id: str
    tool_contract: str
    samples: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _TraceFileContent:
    lines: tuple[str, ...]
    size_bytes: int


class _ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] | None,
        add_generation_prompt: bool = False,
        return_dict: Literal[False] = False,
    ) -> list[int]: ...


def _trace_files(inputs: Sequence[Path]) -> tuple[Path, ...] | ExportFailure:
    selected: set[Path] = set()
    for input_path in inputs:
        path = input_path.resolve()
        if path.is_file():
            candidates = (path,)
        elif path.is_dir():
            candidates = path.rglob("traces.jsonl")
        else:
            return ExportFailure(f"trace input does not exist: {input_path}")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            selected.add(candidate.resolve())
            if len(selected) > MAX_TRACE_FILES:
                return ExportFailure("too many trace files were selected")
    if not selected:
        return ExportFailure("no traces.jsonl files were found")
    return tuple(sorted(selected))


def _text_content(value: object) -> str | ExportFailure:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ExportFailure("message content is not text")
    parts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping) or item.get("type") != "text":
            return ExportFailure("message content has a non-text part")
        text = item.get("text")
        if not isinstance(text, str):
            return ExportFailure("message text is invalid")
        parts.append(text)
    return "\n".join(parts)


def _tool_calls(value: object) -> list[dict[str, object]] | ExportFailure:
    if not isinstance(value, list):
        return ExportFailure("assistant tool_calls is not a list")
    calls: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return ExportFailure("assistant tool call is invalid")
        call_id = item.get("id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            return ExportFailure("assistant tool call identity is invalid")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return ExportFailure("assistant tool arguments are invalid JSON")
        if not isinstance(arguments, Mapping):
            return ExportFailure("assistant tool arguments are not an object")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        )
    return calls


def _message(value: object) -> dict[str, object] | ExportFailure:
    if not isinstance(value, Mapping):
        return ExportFailure("trace node message is invalid")
    role = value.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        return ExportFailure("trace message role is invalid")
    content = _text_content(value.get("content", ""))
    if isinstance(content, ExportFailure):
        return content
    message: dict[str, object] = {"role": role, "content": content}
    if role == "assistant" and "tool_calls" in value:
        calls = _tool_calls(value["tool_calls"])
        if isinstance(calls, ExportFailure):
            return calls
        message["tool_calls"] = calls
    if role == "tool":
        call_id = value.get("tool_call_id")
        if not isinstance(call_id, str):
            return ExportFailure("tool result has no call identity")
        message["tool_call_id"] = call_id
    return message


def _tools(value: object) -> list[dict[str, object]] | ExportFailure:
    if not isinstance(value, list):
        return ExportFailure("trace tools is not a list")
    tools: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return ExportFailure("trace tool is invalid")
        name = item.get("name")
        description = item.get("description")
        parameters = item.get("parameters")
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(parameters, Mapping)
        ):
            return ExportFailure("trace tool contract is invalid")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": dict(parameters),
                },
            }
        )
    return tools


def _parent_chain(nodes: list[object], index: int) -> tuple[int, ...] | ExportFailure:
    chain: list[int] = []
    seen: set[int] = set()
    current: int | None = index
    while current is not None:
        if current in seen or not 0 <= current < len(nodes):
            return ExportFailure("trace node parent graph is invalid")
        seen.add(current)
        chain.append(current)
        node = nodes[current]
        if not isinstance(node, Mapping):
            return ExportFailure("trace node is invalid")
        parent = node.get("parent")
        if parent is not None and (
            isinstance(parent, bool) or not isinstance(parent, int)
        ):
            return ExportFailure("trace node parent is invalid")
        current = parent
    chain.reverse()
    return tuple(chain)


def _failed_tool_result(message: Mapping[str, object]) -> bool:
    if message.get("role") != "tool":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return False
    if not isinstance(result, Mapping):
        return False
    if result.get("ok") is False:
        return True
    return_code = result.get("returncode")
    if (
        isinstance(return_code, int)
        and not isinstance(return_code, bool)
        and return_code != 0
    ):
        return True
    return result.get("status") in {"failed", "error"}


def _target_kinds(
    messages: Sequence[Mapping[str, object]],
) -> frozenset[ActionKind] | ExportFailure:
    target = messages[-1]
    raw_calls = target.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return ExportFailure("sampled assistant tool calls are invalid")
    kinds: set[ActionKind] = set()
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            return ExportFailure("sampled assistant tool call is invalid")
        function = raw_call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or name not in ACTION_BY_TOOL:
            return ExportFailure(f"sampled assistant used unavailable tool: {name}")
        kinds.add(ACTION_BY_TOOL[name])
    content = target.get("content")
    if not raw_calls and isinstance(content, str) and content.strip():
        kinds.add("final")
    if len(messages) > 1 and _failed_tool_result(messages[-2]):
        kinds.add("recovery")
    return frozenset(kinds)


def trace_tool_contract(trace: Mapping[str, object]) -> str | ExportFailure:
    """Read the OMP tool contract a trace was produced under."""
    info = trace.get("info")
    contract = info.get("omp_tool_contract") if isinstance(info, Mapping) else None
    if contract is None:
        return DEFAULT_TOOL_CONTRACT
    if not isinstance(contract, str) or contract not in REQUIRED_ACTION_KINDS_BY_CONTRACT:
        return ExportFailure(f"trace tool contract is unsupported: {contract!r}")
    return contract


def _trace_samples(trace: object) -> _TraceConversion | ExportFailure:
    if not isinstance(trace, Mapping):
        return ExportFailure("trace is not an object")
    trace_id = trace.get("id")
    if not isinstance(trace_id, str) or not trace_id:
        return ExportFailure("trace identity is invalid")
    task = trace.get("task")
    task_data = task.get("data") if isinstance(task, Mapping) else None
    split = task_data.get("split") if isinstance(task_data, Mapping) else None
    task_id = task_data.get("task_id") if isinstance(task_data, Mapping) else None
    if split not in DATA_FILES or not isinstance(task_id, str):
        return ExportFailure("trace task identity is invalid")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        return ExportFailure("trace nodes is not a list")
    if len(nodes) > MAX_TRACE_NODES:
        return ExportFailure(f"trace has too many nodes: {trace_id}")
    tools = _tools(trace.get("tools"))
    if isinstance(tools, ExportFailure):
        return tools
    tool_contract = trace_tool_contract(trace)
    if isinstance(tool_contract, ExportFailure):
        return tool_contract

    target_indices: list[int] = []
    all_target_kinds: set[ActionKind] = set()
    final_chain: tuple[int, ...] | None = None
    final_messages: list[dict[str, object]] | None = None
    final_turn_index: int | None = None
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping) or node.get("sampled") is not True:
            continue
        raw_message = node.get("message")
        if (
            not isinstance(raw_message, Mapping)
            or raw_message.get("role") != "assistant"
        ):
            continue
        chain = _parent_chain(nodes, index)
        if isinstance(chain, ExportFailure):
            return chain
        messages: list[dict[str, object]] = []
        for node_index in chain:
            chain_node = nodes[node_index]
            if not isinstance(chain_node, Mapping):
                return ExportFailure("trace node is invalid")
            message = _message(chain_node.get("message"))
            if isinstance(message, ExportFailure):
                return message
            messages.append(message)
        kinds = _target_kinds(messages)
        if isinstance(kinds, ExportFailure):
            return kinds
        if not kinds:
            continue
        target_indices.append(index)
        all_target_kinds.update(kinds)
        final_chain = chain
        final_messages = messages
        final_turn_index = index
    if final_chain is None or final_messages is None or final_turn_index is None:
        return _TraceConversion(
            split=split, trace_id=trace_id, tool_contract=tool_contract, samples=()
        )
    if not set(target_indices).issubset(final_chain):
        return ExportFailure(
            f"successful trace contains branched target turns: {trace_id}"
        )
    sample = {
        "schema_version": 2,
        "task_id": task_id,
        "trace_id": trace_id,
        "turn_index": final_turn_index,
        "target_kinds": sorted(all_target_kinds),
        "messages": final_messages,
        "tools": tools,
    }
    return _TraceConversion(
        split=split, trace_id=trace_id, tool_contract=tool_contract, samples=(sample,)
    )


def _sample_kinds(
    sample: Mapping[str, object],
) -> tuple[ActionKind, ...] | ExportFailure:
    raw_kinds = sample.get("target_kinds")
    if not isinstance(raw_kinds, list) or not raw_kinds:
        return ExportFailure("training sample has no target action kind")
    kinds: list[ActionKind] = []
    for raw_kind in raw_kinds:
        if raw_kind not in ALL_ACTION_KINDS:
            return ExportFailure(f"training sample action kind is invalid: {raw_kind}")
        kinds.append(raw_kind)
    return tuple(kinds)


def _write_dataset_rows(
    *,
    output_dir: Path,
    samples_by_split: Mapping[str, Sequence[Mapping[str, object]]],
) -> str | ExportFailure:
    dataset_digest = hashlib.sha256(b"omp-coding-dataset-v2\0")
    for split, file_name in DATA_FILES.items():
        rows = samples_by_split[split]
        target = output_dir / file_name
        if not rows:
            target.unlink(missing_ok=True)
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{file_name}.", dir=output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for row in rows:
                    line = json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    stream.write(line + "\n")
                    encoded = line.encode("utf-8")
                    dataset_digest.update(file_name.encode())
                    dataset_digest.update(len(encoded).to_bytes(8, "big"))
                    dataset_digest.update(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            return ExportFailure(f"dataset write failed: {error}")
    return dataset_digest.hexdigest()


def _write_dataset_manifest(
    output_dir: Path, manifest: DatasetManifest
) -> ExportFailure | None:
    target = output_dir / DATASET_MANIFEST
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{DATASET_MANIFEST}.", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                asdict(manifest),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        return ExportFailure(f"dataset manifest write failed: {error}")
    return None


def _read_trace_lines(path: Path) -> _TraceFileContent | ExportFailure:
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_TRACE_FILE_BYTES + 1)
    except OSError as error:
        return ExportFailure(f"trace file is not readable: {path}: {error}")
    if len(content) > MAX_TRACE_FILE_BYTES:
        return ExportFailure(f"trace file exceeds the size limit: {path}")
    try:
        lines = tuple(content.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        return ExportFailure(f"trace file is not UTF-8: {path}: {error}")
    return _TraceFileContent(lines=lines, size_bytes=len(content))


def export_traces(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    minimum_train_trajectories: int = MINIMUM_TRAIN_TRAJECTORIES,
    minimum_validation_trajectories: int = MINIMUM_VALIDATION_TRAJECTORIES,
    required_action_kinds: frozenset[ActionKind] | None = None,
) -> ExportReport | ExportFailure:
    """Write diverse successful trajectories as bounded MLX chat data.

    `required_action_kinds` defaults to the set the traces' tool contract demands.
    """
    if minimum_train_trajectories < 1 or minimum_validation_trajectories < 1:
        return ExportFailure("minimum trajectory counts must be positive")
    if required_action_kinds is not None and not required_action_kinds:
        return ExportFailure("required action kinds must not be empty")
    dataset_contract: str | None = None
    trace_files = _trace_files(inputs)
    if isinstance(trace_files, ExportFailure):
        return trace_files
    samples_by_split: dict[str, list[dict[str, object]]] = {
        name: [] for name in DATA_FILES
    }
    trace_ids_by_split: dict[str, set[str]] = {name: set() for name in DATA_FILES}
    seen_trace_ids: set[str] = set()
    traces_seen = 0
    successful_traces = 0
    total_trace_bytes = 0
    retained_samples = 0
    for trace_file in trace_files:
        trace_content = _read_trace_lines(trace_file)
        if isinstance(trace_content, ExportFailure):
            return trace_content
        total_trace_bytes += trace_content.size_bytes
        if total_trace_bytes > MAX_TRACE_INPUT_BYTES:
            return ExportFailure("trace inputs exceed the aggregate size limit")
        for line_number, line in enumerate(trace_content.lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                return ExportFailure(
                    f"trace file has invalid JSON at {trace_file}:{line_number}: {error}"
                )
            traces = record.get("traces") if isinstance(record, Mapping) else None
            if not isinstance(traces, list):
                return ExportFailure(
                    f"trace record is invalid at {trace_file}:{line_number}"
                )
            for trace in traces:
                traces_seen += 1
                if traces_seen > MAX_TRACES:
                    return ExportFailure("trace input contains too many traces")
                if not isinstance(trace, Mapping):
                    return ExportFailure("trace is not an object")
                task = trace.get("task")
                task_data = task.get("data") if isinstance(task, Mapping) else None
                split = (
                    task_data.get("split") if isinstance(task_data, Mapping) else None
                )
                if split == "holdout":
                    return ExportFailure(
                        "holdout traces must not enter the training dataset"
                    )
                rewards = trace.get("rewards")
                tests = rewards.get("tests") if isinstance(rewards, Mapping) else None
                score = tests.get("score") if isinstance(tests, Mapping) else None
                completed = trace.get("stop_condition") == "agent_completed"
                tool_contract = trace_tool_contract(trace)
                if isinstance(tool_contract, ExportFailure):
                    return tool_contract
                if dataset_contract is None:
                    dataset_contract = tool_contract
                elif tool_contract != dataset_contract:
                    return ExportFailure(
                        "trace inputs mix tool contracts: "
                        f"{dataset_contract} and {tool_contract}"
                    )
                if trace.get("ok") is not True or score != 1.0 or not completed:
                    continue
                converted = _trace_samples(trace)
                if isinstance(converted, ExportFailure):
                    return converted
                if converted.trace_id in seen_trace_ids:
                    continue
                if not converted.samples:
                    return ExportFailure(
                        f"successful trace has no useful sampled turns: "
                        f"{converted.trace_id}"
                    )
                seen_trace_ids.add(converted.trace_id)
                trace_ids_by_split[converted.split].add(converted.trace_id)
                successful_traces += 1
                retained_samples += len(converted.samples)
                if retained_samples > MAX_RETAINED_SAMPLES:
                    return ExportFailure(
                        "trace input contains too many retained samples"
                    )
                samples_by_split[converted.split].extend(converted.samples)

    train_trajectories = len(trace_ids_by_split["train"])
    valid_trajectories = len(trace_ids_by_split["validation"])
    if train_trajectories < minimum_train_trajectories:
        return ExportFailure(
            f"need at least {minimum_train_trajectories} successful train "
            f"trajectories, found {train_trajectories}"
        )
    if valid_trajectories < minimum_validation_trajectories:
        return ExportFailure(
            f"need at least {minimum_validation_trajectories} successful validation "
            f"trajectories, found {valid_trajectories}"
        )
    if dataset_contract is None:
        return ExportFailure("trace inputs contain no traces")
    if required_action_kinds is None:
        required_action_kinds = REQUIRED_ACTION_KINDS_BY_CONTRACT[dataset_contract]

    action_counts: Counter[ActionKind] = Counter()
    for sample in samples_by_split["train"]:
        kinds = _sample_kinds(sample)
        if isinstance(kinds, ExportFailure):
            return kinds
        action_counts.update(kinds)
    missing_actions = required_action_kinds.difference(action_counts)
    if missing_actions:
        return ExportFailure(
            f"successful train trajectories lack actions: {sorted(missing_actions)}"
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return ExportFailure(f"dataset directory could not be created: {error}")
    dataset_sha256 = _write_dataset_rows(
        output_dir=output_dir,
        samples_by_split=samples_by_split,
    )
    if isinstance(dataset_sha256, ExportFailure):
        return dataset_sha256
    action_count_pairs = tuple(sorted(action_counts.items()))
    manifest = DatasetManifest(
        schema_version=2,
        dataset_sha256=dataset_sha256,
        train_trajectories=train_trajectories,
        valid_trajectories=valid_trajectories,
        action_counts=action_count_pairs,
        samples_per_trace_limit=1,
        tool_contract=dataset_contract,
    )
    manifest_failure = _write_dataset_manifest(output_dir, manifest)
    if manifest_failure is not None:
        return manifest_failure
    return ExportReport(
        trace_files=len(trace_files),
        traces_seen=traces_seen,
        successful_traces=successful_traces,
        train_trajectories=train_trajectories,
        valid_trajectories=valid_trajectories,
        samples_written=sum(len(rows) for rows in samples_by_split.values()),
        train_samples=len(samples_by_split["train"]),
        valid_samples=len(samples_by_split["validation"]),
        test_samples=len(samples_by_split["holdout"]),
        action_counts=action_count_pairs,
        dataset_sha256=dataset_sha256,
        tool_contract=dataset_contract,
    )


def _dataset_sha256(data_dir: Path) -> str | TrainingFailure:
    digest = hashlib.sha256(b"omp-coding-dataset-v2\0")
    for file_name in DATA_FILES.values():
        path = data_dir / file_name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            return TrainingFailure(f"dataset file is not readable: {path}: {error}")
        for line in lines:
            if not line:
                continue
            encoded = line.encode("utf-8")
            digest.update(file_name.encode())
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def load_dataset_manifest(data_dir: Path) -> DatasetManifest | TrainingFailure:
    path = data_dir / DATASET_MANIFEST
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return TrainingFailure(f"dataset manifest is not readable: {path}: {error}")
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 2:
        return TrainingFailure("dataset manifest schema version is invalid")
    dataset_sha256 = raw.get("dataset_sha256")
    train_trajectories = raw.get("train_trajectories")
    valid_trajectories = raw.get("valid_trajectories")
    samples_per_trace_limit = raw.get("samples_per_trace_limit")
    raw_action_counts = raw.get("action_counts")
    tool_contract = raw.get("tool_contract", DEFAULT_TOOL_CONTRACT)
    if (
        not isinstance(dataset_sha256, str)
        or not isinstance(train_trajectories, int)
        or isinstance(train_trajectories, bool)
        or not isinstance(valid_trajectories, int)
        or isinstance(valid_trajectories, bool)
        or not isinstance(samples_per_trace_limit, int)
        or isinstance(samples_per_trace_limit, bool)
        or not isinstance(raw_action_counts, list)
    ):
        return TrainingFailure("dataset manifest fields are invalid")
    if tool_contract not in REQUIRED_ACTION_KINDS_BY_CONTRACT:
        return TrainingFailure("dataset manifest tool contract is invalid")
    action_counts: list[tuple[ActionKind, int]] = []
    for raw_pair in raw_action_counts:
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or raw_pair[0] not in ALL_ACTION_KINDS
            or not isinstance(raw_pair[1], int)
            or isinstance(raw_pair[1], bool)
            or raw_pair[1] < 1
        ):
            return TrainingFailure("dataset manifest action counts are invalid")
        action_counts.append((raw_pair[0], raw_pair[1]))
    observed_sha256 = _dataset_sha256(data_dir)
    if isinstance(observed_sha256, TrainingFailure):
        return observed_sha256
    if observed_sha256 != dataset_sha256:
        return TrainingFailure("dataset content does not match its manifest")
    return DatasetManifest(
        schema_version=2,
        dataset_sha256=dataset_sha256,
        train_trajectories=train_trajectories,
        valid_trajectories=valid_trajectories,
        action_counts=tuple(action_counts),
        samples_per_trace_limit=samples_per_trace_limit,
        tool_contract=tool_contract,
    )


def _load_training_tokenizer(model: str) -> _ChatTokenizer | TrainingFailure:
    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError as error:
        return TrainingFailure(f"MLX-LM is not installed: {error}")
    try:
        return load_tokenizer(model)
    except Exception as error:
        return TrainingFailure(f"model tokenizer could not be loaded: {error}")


def _training_sample_length(
    tokenizer: _ChatTokenizer,
    sample: Mapping[str, object],
) -> _SampleLength | TrainingFailure:
    raw_messages = sample.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return TrainingFailure("training sample messages are missing or empty")
    messages: list[Mapping[str, object]] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            return TrainingFailure("training sample message is invalid")
        messages.append(message)
    if messages[-1].get("role") != "assistant":
        return TrainingFailure("training sample does not end with an assistant message")

    raw_tools = sample.get("tools")
    tools: list[Mapping[str, object]] | None
    if raw_tools is None:
        tools = None
    elif isinstance(raw_tools, list):
        tools = []
        for tool in raw_tools:
            if not isinstance(tool, Mapping):
                return TrainingFailure("training sample tool is invalid")
            tools.append(tool)
    else:
        return TrainingFailure("training sample tools are invalid")

    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            return_dict=False,
        )
        prompt_tokens = tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            add_generation_prompt=True,
            return_dict=False,
        )
    except Exception as error:
        return TrainingFailure(f"training sample tokenization failed: {error}")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        return TrainingFailure("training sample tokens are invalid")
    if not isinstance(prompt_tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool)
        for token in prompt_tokens
    ):
        return TrainingFailure("training prompt tokens are invalid")
    completion_tokens = len(tokens) - len(prompt_tokens)
    if completion_tokens < 2:
        return TrainingFailure("training sample has no trainable completion")
    return _SampleLength(
        total_tokens=len(tokens),
        completion_tokens=completion_tokens,
    )


def _read_training_rows(path: Path) -> list[dict[str, object]] | TrainingFailure:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        return TrainingFailure(f"training data file is not readable: {path}: {error}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            return TrainingFailure(
                f"training data has invalid JSON at {path}:{line_number}: {error}"
            )
        if not isinstance(row, Mapping):
            return TrainingFailure(
                f"training data row is not an object at {path}:{line_number}"
            )
        rows.append(dict(row))
    if not rows:
        return TrainingFailure(f"training data file is empty: {path}")
    return rows


def _prepare_training_data(
    *,
    data_dir: Path,
    output_dir: Path,
    tokenizer: _ChatTokenizer,
    max_sequence_length: int,
    required_action_kinds: frozenset[ActionKind],
) -> _PreparedTrainingData | TrainingFailure:
    counts = {split: 0 for split in DATA_FILES}
    trace_ids_by_split: dict[str, set[str]] = {split: set() for split in DATA_FILES}
    train_action_counts: Counter[ActionKind] = Counter()
    dropped_samples = 0
    try:
        output_dir.mkdir()
    except OSError as error:
        return TrainingFailure(f"prepared training data directory failed: {error}")
    for split, file_name in DATA_FILES.items():
        if split == "holdout":
            continue
        source = data_dir / file_name
        if not source.is_file():
            return TrainingFailure(f"training data file is missing: {source}")
        rows = _read_training_rows(source)
        if isinstance(rows, TrainingFailure):
            return rows
        retained: list[dict[str, object]] = []
        for row in rows:
            length = _training_sample_length(tokenizer, row)
            if isinstance(length, TrainingFailure):
                return length
            if length.total_tokens > max_sequence_length:
                dropped_samples += 1
                continue
            trace_id = row.get("trace_id")
            if not isinstance(trace_id, str) or not trace_id:
                return TrainingFailure("training sample trace identity is invalid")
            kinds = _sample_kinds(row)
            if isinstance(kinds, ExportFailure):
                return TrainingFailure(kinds.reason)
            trace_ids_by_split[split].add(trace_id)
            if split == "train":
                train_action_counts.update(kinds)
            retained.append(row)
        if not retained:
            return TrainingFailure(
                f"no {split} samples fit the maximum sequence length"
            )
        counts[split] = len(retained)
        target = output_dir / file_name
        content = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in retained
        )
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as error:
            return TrainingFailure(f"prepared training data write failed: {error}")
    if len(trace_ids_by_split["train"]) < MINIMUM_TRAIN_TRAJECTORIES:
        return TrainingFailure(
            "too few train trajectories fit the maximum sequence length"
        )
    if len(trace_ids_by_split["validation"]) < MINIMUM_VALIDATION_TRAJECTORIES:
        return TrainingFailure(
            "too few validation trajectories fit the maximum sequence length"
        )
    missing_actions = required_action_kinds.difference(train_action_counts)
    if missing_actions:
        return TrainingFailure(
            f"retained train data lacks required actions: {sorted(missing_actions)}"
        )
    return _PreparedTrainingData(
        path=output_dir,
        train_samples=counts["train"],
        valid_samples=counts["validation"],
        test_samples=counts["holdout"],
        dropped_samples=dropped_samples,
    )


def _training_losses(output: str) -> tuple[float, ...] | TrainingFailure:
    observed = tuple(LOSS_PATTERN.finditer(output))
    if not observed:
        return TrainingFailure("MLX training did not report losses")
    values: list[tuple[str, float]] = []
    for match in observed:
        try:
            value = float(match.group(2))
        except ValueError:
            return TrainingFailure("MLX training reported an invalid loss")
        if not math.isfinite(value):
            return TrainingFailure("MLX training reported a non-finite loss")
        values.append((match.group(1), value))
    training_losses = tuple(value for kind, value in values if kind == "Train")
    if not training_losses:
        return TrainingFailure("MLX training did not report training losses")
    return training_losses


def _sanitize_adapter_config(config_path: Path) -> TrainingFailure | None:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return TrainingFailure(f"MLX adapter configuration is invalid: {error}")
    if not isinstance(document, Mapping):
        return TrainingFailure("MLX adapter configuration is not an object")
    num_layers = document.get("num_layers")
    lora_parameters = document.get("lora_parameters")
    if (
        document.get("fine_tune_type") != "lora"
        or not isinstance(num_layers, int)
        or isinstance(num_layers, bool)
        or not isinstance(lora_parameters, Mapping)
    ):
        return TrainingFailure("MLX adapter configuration lacks LoRA parameters")
    private_path_fields = {
        "adapter_path",
        "data",
        "resume_adapter_file",
    }
    sanitized = {
        key: value for key, value in document.items() if key not in private_path_fields
    }
    try:
        config_path.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        return TrainingFailure(f"MLX adapter configuration write failed: {error}")
    return None


def _snapshot_dataset(
    source_dir: Path,
    destination_dir: Path,
) -> TrainingFailure | None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_descriptor = os.open(source_dir, flags)
    except OSError as error:
        return TrainingFailure(f"training dataset directory is not readable: {error}")
    try:
        destination_dir.mkdir(mode=0o700)
        for file_name in (
            DATASET_MANIFEST,
            DATA_FILES["train"],
            DATA_FILES["validation"],
        ):
            try:
                file_descriptor = os.open(
                    file_name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                return TrainingFailure(
                    f"training dataset file is not readable: {file_name}: {error}"
                )
            try:
                file_status = os.fstat(file_descriptor)
                if not stat.S_ISREG(file_status.st_mode):
                    return TrainingFailure(
                        f"training dataset file is not regular: {file_name}"
                    )
                with os.fdopen(file_descriptor, "rb", closefd=False) as source:
                    content = source.read(MAX_DATASET_FILE_BYTES + 1)
                if not content:
                    return TrainingFailure(
                        f"training dataset file is empty: {file_name}"
                    )
                if len(content) > MAX_DATASET_FILE_BYTES:
                    return TrainingFailure(
                        f"training dataset file exceeds the size limit: {file_name}"
                    )
                (destination_dir / file_name).write_bytes(content)
            except OSError as error:
                return TrainingFailure(
                    f"training dataset file could not be copied: {file_name}: {error}"
                )
            finally:
                os.close(file_descriptor)
        try:
            holdout_descriptor = os.open(
                DATA_FILES["holdout"],
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            return TrainingFailure(f"holdout data could not be inspected: {error}")
        else:
            os.close(holdout_descriptor)
            return TrainingFailure("training dataset must not contain holdout data")
    except OSError as error:
        return TrainingFailure(f"training dataset snapshot failed: {error}")
    finally:
        os.close(directory_descriptor)


def train_adapter(
    *,
    data_dir: Path,
    model: str,
    adapter_dir: Path,
    iterations: int,
    checkpoint_interval: int,
    max_sequence_length: int,
    number_of_layers: int,
    learning_rate: float,
) -> TrainingReport | TrainingFailure:
    """Gate one model, then train one LoRA adapter on Apple Metal."""
    if iterations < 2:
        return TrainingFailure("iterations must be at least 2")
    if checkpoint_interval < 1 or checkpoint_interval >= iterations:
        return TrainingFailure(
            "checkpoint interval must be positive and less than iterations"
        )
    if max_sequence_length < MINIMUM_SEQUENCE_LENGTH:
        return TrainingFailure(
            f"max sequence length must be at least {MINIMUM_SEQUENCE_LENGTH}"
        )
    if number_of_layers == 0 or number_of_layers < -1:
        return TrainingFailure("number of layers must be -1 or a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        return TrainingFailure("learning rate must be a positive finite number")
    if adapter_dir.exists() or adapter_dir.is_symlink():
        return TrainingFailure(f"adapter output already exists: {adapter_dir}")
    try:
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return TrainingFailure(
            f"adapter parent directory could not be created: {error}"
        )

    metal = metal_preflight()
    if isinstance(metal, MetalFailure):
        return TrainingFailure(metal.reason)
    with tempfile.TemporaryDirectory(
        prefix=f".{adapter_dir.name}.", dir=adapter_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        snapshot_dir = temporary / "dataset"
        snapshot_failure = _snapshot_dataset(data_dir, snapshot_dir)
        if snapshot_failure is not None:
            return snapshot_failure
        manifest = load_dataset_manifest(snapshot_dir)
        if isinstance(manifest, TrainingFailure):
            return manifest
        protocol_context = load_protocol_context(snapshot_dir)
        if isinstance(protocol_context, ProtocolFailure):
            return TrainingFailure(
                f"dataset protocol context is invalid: {protocol_context.reason}"
            )
        if manifest.train_trajectories < MINIMUM_TRAIN_TRAJECTORIES:
            return TrainingFailure(
                f"dataset needs at least {MINIMUM_TRAIN_TRAJECTORIES} "
                "train trajectories"
            )
        if manifest.valid_trajectories < MINIMUM_VALIDATION_TRAJECTORIES:
            return TrainingFailure(
                "dataset needs at least "
                f"{MINIMUM_VALIDATION_TRAJECTORIES} validation trajectories"
            )
        available_actions = {
            kind for kind, count in manifest.action_counts if count > 0
        }
        required_action_kinds = REQUIRED_ACTION_KINDS_BY_CONTRACT[manifest.tool_contract]
        missing_actions = required_action_kinds.difference(available_actions)
        if missing_actions:
            return TrainingFailure(
                f"dataset lacks required actions: {sorted(missing_actions)}"
            )
        if manifest.samples_per_trace_limit != 1:
            return TrainingFailure("dataset sample-per-trace limit is invalid")
        protocol = run_local_protocol_gate(
            model=model,
            context=protocol_context,
        )
        if isinstance(protocol, ProtocolFailure):
            return TrainingFailure(f"model protocol check failed: {protocol.reason}")
        gate_failure = protocol_gate_failure(protocol)
        if gate_failure is not None:
            return TrainingFailure(f"base model rejected: {gate_failure.reason}")

        tokenizer = _load_training_tokenizer(model)
        if isinstance(tokenizer, TrainingFailure):
            return tokenizer
        prepared = _prepare_training_data(
            data_dir=snapshot_dir,
            output_dir=temporary / "prepared",
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            required_action_kinds=required_action_kinds,
        )
        if isinstance(prepared, TrainingFailure):
            return prepared
        staged_adapter = temporary / "adapter"
        command = [
            sys.executable,
            "-m",
            "omp_coding.masked_lora",
            "--model",
            model,
            "--data",
            str(prepared.path),
            "--adapter-path",
            str(staged_adapter),
            "--num-layers",
            str(number_of_layers),
            "--iters",
            str(iterations),
            "--learning-rate",
            str(learning_rate),
            "--checkpoint-interval",
            str(checkpoint_interval),
            "--max-seq-length",
            str(max_sequence_length),
            "--seed",
            "0",
        ]
        completed = subprocess.run(  # noqa: S603 - fixed MLX module.
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(completed.stdout, end="")
        if completed.returncode != 0:
            return TrainingFailure(
                f"MLX training exited with status {completed.returncode}"
            )
        losses = _training_losses(completed.stdout)
        if isinstance(losses, TrainingFailure):
            return losses
        weights = staged_adapter / "adapters.safetensors"
        config = staged_adapter / "adapter_config.json"
        if not weights.is_file() or weights.stat().st_size == 0 or not config.is_file():
            return TrainingFailure("MLX training did not write a complete adapter")
        checkpoints = tuple(sorted(staged_adapter.glob("*_adapters.safetensors")))
        if not checkpoints:
            return TrainingFailure("MLX training did not write periodic checkpoints")
        config_failure = _sanitize_adapter_config(config)
        if config_failure is not None:
            return config_failure
        adapter_bytes = weights.stat().st_size
        checkpoint_files = tuple(path.name for path in checkpoints)
        try:
            staged_adapter.replace(adapter_dir)
        except OSError as error:
            return TrainingFailure(f"trained adapter could not be installed: {error}")
        return TrainingReport(
            model=model,
            data=_report_path(data_dir),
            adapter=_report_path(adapter_dir),
            iterations=iterations,
            max_sequence_length=max_sequence_length,
            losses=losses,
            train_samples=prepared.train_samples,
            valid_samples=prepared.valid_samples,
            test_samples=prepared.test_samples,
            dropped_samples=prepared.dropped_samples,
            adapter_bytes=adapter_bytes,
            checkpoint_files=checkpoint_files,
            protocol=protocol,
            metal=metal,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(command="preflight")
    export = subparsers.add_parser("export")
    export.add_argument("inputs", nargs="+", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--minimum-train-trajectories",
        type=int,
        default=MINIMUM_TRAIN_TRAJECTORIES,
    )
    export.add_argument(
        "--minimum-validation-trajectories",
        type=int,
        default=MINIMUM_VALIDATION_TRAJECTORIES,
    )
    train = subparsers.add_parser("run")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--adapter", type=Path, required=True)
    train.add_argument("--iters", type=int, required=True)
    train.add_argument("--checkpoint-interval", type=int, default=50)
    train.add_argument("--max-seq-length", type=int, default=MINIMUM_SEQUENCE_LENGTH)
    train.add_argument("--num-layers", type=int, default=-1)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        result: object = metal_preflight()
    elif arguments.command == "export":
        result = export_traces(
            arguments.inputs,
            arguments.output,
            minimum_train_trajectories=arguments.minimum_train_trajectories,
            minimum_validation_trajectories=(arguments.minimum_validation_trajectories),
        )
    elif arguments.command == "run":
        result = train_adapter(
            data_dir=arguments.data,
            model=arguments.model,
            adapter_dir=arguments.adapter,
            iterations=arguments.iters,
            checkpoint_interval=arguments.checkpoint_interval,
            max_sequence_length=arguments.max_seq_length,
            number_of_layers=arguments.num_layers,
            learning_rate=arguments.learning_rate,
        )
    else:
        raise AssertionError(f"unknown command: {arguments.command}")
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return (
        1 if isinstance(result, (ExportFailure, MetalFailure, TrainingFailure)) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
