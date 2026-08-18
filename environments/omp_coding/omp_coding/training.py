"""Export Prime Verifiers v1 traces and train one MLX LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol

LOSS_PATTERN = re.compile(r"(Train|Val) loss ([^\s,]+)")
DATA_FILES = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "holdout": "test.jsonl",
}


@dataclass(frozen=True)
class ExportFailure:
    reason: str


@dataclass(frozen=True)
class ExportReport:
    trace_files: int
    traces_seen: int
    successful_traces: int
    samples_written: int
    train_samples: int
    valid_samples: int
    test_samples: int
    dataset_sha256: str


@dataclass(frozen=True)
class MetalFailure:
    reason: str


@dataclass(frozen=True)
class MetalReport:
    backend: Literal["metal"]
    logical_device: Literal["gpu:0"]
    device_name: str
    architecture: str
    memory_bytes: int
    mlx_version: str
    dtype: Literal["float32"]
    check_value: float


@dataclass(frozen=True)
class TrainingFailure:
    reason: str


@dataclass(frozen=True)
class TrainingReport:
    model: str
    data: str
    adapter: str
    iterations: int
    losses: tuple[float, ...]
    train_samples: int
    valid_samples: int
    test_samples: int
    dropped_samples: int
    adapter_bytes: int
    metal: MetalReport


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
            selected.add(path)
            continue
        if path.is_dir():
            selected.update(
                candidate
                for candidate in path.rglob("traces.jsonl")
                if candidate.is_file()
            )
            continue
        return ExportFailure(f"trace input does not exist: {input_path}")
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


def _trace_samples(
    trace: object,
) -> tuple[str, list[dict[str, object]]] | ExportFailure:
    if not isinstance(trace, Mapping):
        return ExportFailure("trace is not an object")
    task = trace.get("task")
    task_data = task.get("data") if isinstance(task, Mapping) else None
    split = task_data.get("split") if isinstance(task_data, Mapping) else None
    task_id = task_data.get("task_id") if isinstance(task_data, Mapping) else None
    if split not in DATA_FILES or not isinstance(task_id, str):
        return ExportFailure("trace task identity is invalid")
    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        return ExportFailure("trace nodes is not a list")
    tools = _tools(trace.get("tools"))
    if isinstance(tools, ExportFailure):
        return tools

    samples: list[dict[str, object]] = []
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
        samples.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "trace_id": trace.get("id"),
                "messages": messages,
                "tools": tools,
            }
        )
    return split, samples


def export_traces(
    inputs: Sequence[Path], output_dir: Path
) -> ExportReport | ExportFailure:
    """Write successful sampled turns from Prime v1 traces as MLX chat data."""
    trace_files = _trace_files(inputs)
    if isinstance(trace_files, ExportFailure):
        return trace_files
    samples_by_split: dict[str, list[dict[str, object]]] = {
        name: [] for name in DATA_FILES
    }
    seen_samples: set[str] = set()
    traces_seen = 0
    successful_traces = 0
    for trace_file in trace_files:
        try:
            lines = trace_file.read_text().splitlines()
        except (OSError, UnicodeDecodeError) as error:
            return ExportFailure(f"trace file is not readable: {trace_file}: {error}")
        for line_number, line in enumerate(lines, start=1):
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
                if not isinstance(trace, Mapping):
                    return ExportFailure("trace is not an object")
                rewards = trace.get("rewards")
                tests = rewards.get("tests") if isinstance(rewards, Mapping) else None
                score = tests.get("score") if isinstance(tests, Mapping) else None
                if trace.get("ok") is not True or score != 1.0:
                    continue
                converted = _trace_samples(trace)
                if isinstance(converted, ExportFailure):
                    return converted
                split, samples = converted
                successful_traces += 1
                for sample in samples:
                    encoded = json.dumps(
                        sample,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    digest = hashlib.sha256(encoded.encode()).hexdigest()
                    if digest in seen_samples:
                        continue
                    seen_samples.add(digest)
                    samples_by_split[split].append(sample)

    if not samples_by_split["train"]:
        return ExportFailure("no successful train samples were found")
    if not samples_by_split["validation"]:
        return ExportFailure("no successful validation samples were found")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_digest = hashlib.sha256(b"omp-coding-dataset-v1\0")
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
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    stream.write(line + "\n")
                    dataset_digest.update(file_name.encode())
                    dataset_digest.update(len(line).to_bytes(8, "big"))
                    dataset_digest.update(line.encode())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            return ExportFailure(f"dataset write failed: {error}")

    return ExportReport(
        trace_files=len(trace_files),
        traces_seen=traces_seen,
        successful_traces=successful_traces,
        samples_written=sum(len(rows) for rows in samples_by_split.values()),
        train_samples=len(samples_by_split["train"]),
        valid_samples=len(samples_by_split["validation"]),
        test_samples=len(samples_by_split["holdout"]),
        dataset_sha256=dataset_digest.hexdigest(),
    )


def metal_preflight() -> MetalReport | MetalFailure:
    """Run one checked matrix operation on the Apple Metal GPU."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return MetalFailure("training requires Apple silicon on macOS")
    try:
        import mlx.core as mx

        mlx_version = version("mlx")
    except (ImportError, PackageNotFoundError) as error:
        return MetalFailure(f"MLX is not installed: {error}")
    if not mx.metal.is_available():
        return MetalFailure("the Metal GPU is not available")
    mx.set_default_device(mx.gpu)
    if str(mx.default_device()) != "Device(gpu, 0)":
        return MetalFailure(
            f"the default MLX device is not gpu:0: {mx.default_device()}"
        )
    left = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    right = mx.array([[2.0, 0.0], [1.0, 2.0]], dtype=mx.float32)
    result = mx.matmul(left, right)
    mx.eval(result)
    check_value = float(result[1, 1].item())
    if check_value != 8.0:
        return MetalFailure(f"the Metal operation returned {check_value}, expected 8.0")
    device = mx.device_info()
    return MetalReport(
        backend="metal",
        logical_device="gpu:0",
        device_name=str(device.get("device_name", "unknown")),
        architecture=str(device.get("architecture", "unknown")),
        memory_bytes=int(device.get("memory_size", 0)),
        mlx_version=mlx_version,
        dtype="float32",
        check_value=check_value,
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
) -> _PreparedTrainingData | TrainingFailure:
    counts = {split: 0 for split in DATA_FILES}
    dropped_samples = 0
    try:
        output_dir.mkdir()
    except OSError as error:
        return TrainingFailure(f"prepared training data directory failed: {error}")
    for split, file_name in DATA_FILES.items():
        source = data_dir / file_name
        if not source.is_file():
            if split == "holdout":
                continue
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
            retained.append(row)
        if not retained:
            if split == "holdout":
                continue
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


def _normalize_adapter_config(
    config_path: Path,
    *,
    data_dir: Path,
    adapter_dir: Path,
) -> TrainingFailure | None:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return TrainingFailure(f"MLX adapter configuration is invalid: {error}")
    if not isinstance(document, Mapping):
        return TrainingFailure("MLX adapter configuration is not an object")
    normalized = {
        **document,
        "adapter_path": str(adapter_dir.resolve()),
        "data": str(data_dir.resolve()),
    }
    try:
        config_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        return TrainingFailure(f"MLX adapter configuration write failed: {error}")
    return None


def train_adapter(
    *,
    data_dir: Path,
    model: str,
    adapter_dir: Path,
    iterations: int,
    max_sequence_length: int,
    number_of_layers: int,
    learning_rate: float,
) -> TrainingReport | TrainingFailure:
    """Train one LoRA adapter from an exported Prime v1 dataset."""
    metal = metal_preflight()
    if isinstance(metal, MetalFailure):
        return TrainingFailure(metal.reason)
    if iterations < 2:
        return TrainingFailure("iterations must be at least 2")
    if max_sequence_length < 256:
        return TrainingFailure("max sequence length must be at least 256")
    if number_of_layers == 0 or number_of_layers < -1:
        return TrainingFailure("number of layers must be -1 or a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        return TrainingFailure("learning rate must be a positive finite number")
    for file_name in ("train.jsonl", "valid.jsonl"):
        path = data_dir / file_name
        if not path.is_file() or path.stat().st_size == 0:
            return TrainingFailure(f"training data file is missing or empty: {path}")
    if adapter_dir.exists() or adapter_dir.is_symlink():
        return TrainingFailure(f"adapter output already exists: {adapter_dir}")
    try:
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return TrainingFailure(
            f"adapter parent directory could not be created: {error}"
        )

    tokenizer = _load_training_tokenizer(model)
    if isinstance(tokenizer, TrainingFailure):
        return tokenizer
    with tempfile.TemporaryDirectory(
        prefix=f".{adapter_dir.name}.", dir=adapter_dir.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        prepared = _prepare_training_data(
            data_dir=data_dir,
            output_dir=temporary / "data",
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )
        if isinstance(prepared, TrainingFailure):
            return prepared
        staged_adapter = temporary / "adapter"
        command = [
            sys.executable,
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            model,
            "--train",
            "--data",
            str(prepared.path),
            "--fine-tune-type",
            "lora",
            "--mask-prompt",
            "--num-layers",
            str(number_of_layers),
            "--batch-size",
            "1",
            "--iters",
            str(iterations),
            "--learning-rate",
            str(learning_rate),
            "--steps-per-report",
            "1",
            "--steps-per-eval",
            str(max(1, iterations // 2)),
            "--adapter-path",
            str(staged_adapter),
            "--save-every",
            str(iterations),
            "--max-seq-length",
            str(max_sequence_length),
            "--seed",
            "0",
        ]
        if prepared.test_samples > 0:
            command.extend(["--test", "--test-batches", "-1"])
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
        config_failure = _normalize_adapter_config(
            config,
            data_dir=data_dir,
            adapter_dir=adapter_dir,
        )
        if config_failure is not None:
            return config_failure
        adapter_bytes = weights.stat().st_size
        try:
            staged_adapter.replace(adapter_dir)
        except OSError as error:
            return TrainingFailure(f"trained adapter could not be installed: {error}")
    return TrainingReport(
        model=model,
        data=str(data_dir.resolve()),
        adapter=str(adapter_dir.resolve()),
        iterations=iterations,
        losses=losses,
        train_samples=prepared.train_samples,
        valid_samples=prepared.valid_samples,
        test_samples=prepared.test_samples,
        dropped_samples=prepared.dropped_samples,
        adapter_bytes=adapter_bytes,
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
    train = subparsers.add_parser("run")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--adapter", type=Path, required=True)
    train.add_argument("--iters", type=int, required=True)
    train.add_argument("--max-seq-length", type=int, default=4096)
    train.add_argument("--num-layers", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        result: object = metal_preflight()
    elif arguments.command == "export":
        result = export_traces(arguments.inputs, arguments.output)
    elif arguments.command == "run":
        result = train_adapter(
            data_dir=arguments.data,
            model=arguments.model,
            adapter_dir=arguments.adapter,
            iterations=arguments.iters,
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
