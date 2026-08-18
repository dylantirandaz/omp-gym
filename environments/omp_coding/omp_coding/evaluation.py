"""Compare fused MLX LoRA checkpoints through sealed Prime evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from .hardware import MetalFailure, MetalReport, metal_preflight
from .metrics import (
    MetricsComparison,
    MetricsFailure,
    TraceMetricsReport,
    compare_metrics,
    measure_traces,
)
from .protocol import (
    PROTOCOL_MODEL_NAME,
    ProtocolFailure,
    ProtocolReport,
    load_protocol_context,
    protocol_gate_failure,
    run_protocol_gate,
    start_local_model_server,
    stop_local_model_server,
)

EvaluationSplit: TypeAlias = Literal["validation", "holdout"]
LOCAL_API_KEY_VARIABLE = "OMP_CODING_LOCAL_API_KEY"
MLX_MODEL_FILE_PATTERNS = (
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)


@dataclass(frozen=True)
class EvaluationFailure:
    reason: str


@dataclass(frozen=True)
class EvaluatedCandidate:
    status: Literal["improved", "rejected"]
    checkpoint: str
    weights_sha256: str
    protocol: ProtocolReport
    metrics: TraceMetricsReport
    comparison: MetricsComparison


@dataclass(frozen=True)
class FailedCandidate:
    status: Literal["failed"]
    checkpoint: str
    weights_sha256: str
    reason: str
    protocol: ProtocolReport | None


CandidateResult: TypeAlias = EvaluatedCandidate | FailedCandidate


@dataclass(frozen=True)
class ImprovedEvaluationReport:
    schema_version: Literal[1]
    status: Literal["improved"]
    model: str
    adapter: str
    split: EvaluationSplit
    max_tokens: int
    num_rollouts: int
    selected_checkpoint: str
    baseline_protocol: ProtocolReport
    baseline_metrics: TraceMetricsReport
    candidates: tuple[CandidateResult, ...]
    metal: MetalReport


@dataclass(frozen=True)
class RejectedEvaluationReport:
    schema_version: Literal[1]
    status: Literal["rejected"]
    model: str
    adapter: str
    split: EvaluationSplit
    max_tokens: int
    num_rollouts: int
    baseline_protocol: ProtocolReport
    baseline_metrics: TraceMetricsReport
    candidates: tuple[CandidateResult, ...]
    metal: MetalReport


EvaluationReport: TypeAlias = ImprovedEvaluationReport | RejectedEvaluationReport


@dataclass(frozen=True)
class _ModelEvaluation:
    protocol: ProtocolReport
    metrics: TraceMetricsReport


@dataclass(frozen=True)
class _ModelEvaluationFailure:
    reason: str
    protocol: ProtocolReport | None


def _report_path(path: Path) -> str:
    return path.name if path.is_absolute() else str(path)


def _file_sha256(path: Path) -> str | EvaluationFailure:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        return EvaluationFailure(f"checkpoint is not readable: {path}: {error}")
    return digest.hexdigest()


def _resolve_model_source(model: str) -> Path | EvaluationFailure:
    local_path = Path(model)
    if local_path.is_dir():
        return local_path.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        return EvaluationFailure(f"Hugging Face Hub is not installed: {error}")
    try:
        return Path(
            snapshot_download(model, allow_patterns=MLX_MODEL_FILE_PATTERNS)
        ).resolve()
    except (OSError, ValueError) as error:
        return EvaluationFailure(f"model files could not be resolved: {model}: {error}")


def _fuse_checkpoint(
    *,
    model: str,
    adapter_dir: Path,
    weights: Path,
    output_dir: Path,
    staging_dir: Path,
) -> EvaluationFailure | None:
    model_source = _resolve_model_source(model)
    if isinstance(model_source, EvaluationFailure):
        return model_source
    config = adapter_dir / "adapter_config.json"
    if not config.is_file():
        return EvaluationFailure(f"adapter configuration is missing: {config}")
    if not weights.is_file() or weights.stat().st_size == 0:
        return EvaluationFailure(f"adapter checkpoint is missing or empty: {weights}")
    alias = staging_dir / "adapter"
    try:
        alias.mkdir()
        (alias / "adapter_config.json").symlink_to(config.resolve())
        (alias / "adapters.safetensors").symlink_to(weights.resolve())
    except OSError as error:
        return EvaluationFailure(f"checkpoint staging failed: {error}")
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        str(model_source),
        "--adapter-path",
        str(alias),
        "--save-path",
        str(output_dir),
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
        return EvaluationFailure(
            f"MLX fusion exited with status {completed.returncode}"
        )
    model_files = tuple(output_dir.glob("*.safetensors"))
    if not (output_dir / "config.json").is_file() or not model_files:
        return EvaluationFailure("MLX fusion did not write a complete model")
    if any(path.stat().st_size == 0 for path in model_files):
        return EvaluationFailure("MLX fusion wrote an empty model weight file")
    return None


def _prime_eval_executable() -> Path | EvaluationFailure:
    executable = shutil.which("eval")
    if executable is None:
        return EvaluationFailure("Prime Verifiers v1 eval command is not installed")
    return Path(executable)


def _run_prime_eval(
    *,
    executable: Path,
    workspace: Path,
    base_url: str,
    api_key: str,
    split: EvaluationSplit,
    max_tokens: int,
    num_rollouts: int,
    output_dir: Path,
) -> Path | EvaluationFailure:
    command = [
        str(executable),
        "omp-coding",
        "--model",
        PROTOCOL_MODEL_NAME,
        "--no-push",
        "--no-rich",
        "--env.taskset.split",
        split,
        "--client.base-url",
        base_url,
        "--client.api-key-var",
        LOCAL_API_KEY_VARIABLE,
        "--sampling.max-tokens",
        str(max_tokens),
        "--sampling.temperature",
        "0",
        "--max-concurrent",
        "1",
        "--num-rollouts",
        str(num_rollouts),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(  # noqa: S603 - fixed Prime executable.
        command,
        cwd=workspace,
        env={**os.environ, LOCAL_API_KEY_VARIABLE: api_key},
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        return EvaluationFailure(
            f"Prime evaluation exited with status {completed.returncode}"
        )
    trace_files = tuple(output_dir.rglob("traces.jsonl"))
    if len(trace_files) != 1:
        return EvaluationFailure(
            f"Prime evaluation wrote {len(trace_files)} trace files, expected one"
        )
    return trace_files[0]


def _evaluate_model(
    *,
    model: str,
    data_dir: Path,
    workspace: Path,
    split: EvaluationSplit,
    max_tokens: int,
    num_rollouts: int,
    output_dir: Path,
    executable: Path,
    baseline: TraceMetricsReport | None,
) -> _ModelEvaluation | _ModelEvaluationFailure:
    context = load_protocol_context(data_dir)
    if isinstance(context, ProtocolFailure):
        return _ModelEvaluationFailure(context.reason, None)
    server = start_local_model_server(model=model, max_tokens=max_tokens)
    if isinstance(server, ProtocolFailure):
        return _ModelEvaluationFailure(server.reason, None)
    result: tuple[ProtocolReport, Path] | _ModelEvaluationFailure
    try:
        raw_protocol = run_protocol_gate(
            base_url=server.base_url,
            api_key=server.api_key,
            model=PROTOCOL_MODEL_NAME,
            parser=server.parser,
            context=context,
            max_tokens=min(max_tokens, 512),
        )
        if isinstance(raw_protocol, ProtocolFailure):
            result = _ModelEvaluationFailure(raw_protocol.reason, None)
        else:
            protocol = ProtocolReport(
                model=model,
                parser=raw_protocol.parser,
                context_sha256=raw_protocol.context_sha256,
                probes=raw_protocol.probes,
                valid_call_rate=raw_protocol.valid_call_rate,
                parsed_call_rate=raw_protocol.parsed_call_rate,
                invalid_tool_rate=raw_protocol.invalid_tool_rate,
                end_token_rate=raw_protocol.end_token_rate,
                loop_rate=raw_protocol.loop_rate,
            )
            gate_failure = protocol_gate_failure(protocol)
            if gate_failure is not None:
                result = _ModelEvaluationFailure(gate_failure.reason, protocol)
            else:
                trace_file = _run_prime_eval(
                    executable=executable,
                    workspace=workspace,
                    base_url=server.base_url,
                    api_key=server.api_key,
                    split=split,
                    max_tokens=max_tokens,
                    num_rollouts=num_rollouts,
                    output_dir=output_dir,
                )
                if isinstance(trace_file, EvaluationFailure):
                    result = _ModelEvaluationFailure(trace_file.reason, protocol)
                else:
                    result = (protocol, trace_file)
    finally:
        stop_failure = stop_local_model_server(server)
    if stop_failure is not None:
        failure_protocol = (
            result.protocol
            if isinstance(result, _ModelEvaluationFailure)
            else result[0]
        )
        return _ModelEvaluationFailure(stop_failure.reason, failure_protocol)
    if isinstance(result, _ModelEvaluationFailure):
        return result
    protocol, trace_file = result
    metrics = measure_traces(
        [trace_file],
        parser=server.parser,
        baseline=baseline,
    )
    if isinstance(metrics, MetricsFailure):
        return _ModelEvaluationFailure(metrics.reason, protocol)
    return _ModelEvaluation(protocol=protocol, metrics=metrics)


def evaluate_checkpoints(
    *,
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    weights: Sequence[Path],
    workspace: Path,
    split: EvaluationSplit,
    max_tokens: int,
    num_rollouts: int,
) -> EvaluationReport | EvaluationFailure:
    """Fuse and compare checkpoints with identical sealed Prime settings."""
    if not weights:
        return EvaluationFailure("at least one adapter checkpoint is required")
    if split == "holdout" and len(weights) != 1:
        return EvaluationFailure("holdout permits exactly one candidate checkpoint")
    if max_tokens < 1 or num_rollouts < 1:
        return EvaluationFailure("evaluation limits must be positive")
    if len({path.resolve() for path in weights}) != len(weights):
        return EvaluationFailure("adapter checkpoints must be unique")
    executable = _prime_eval_executable()
    if isinstance(executable, EvaluationFailure):
        return executable
    metal = metal_preflight()
    if isinstance(metal, MetalFailure):
        return EvaluationFailure(metal.reason)
    with tempfile.TemporaryDirectory(prefix="omp-evaluation-") as temporary_name:
        temporary = Path(temporary_name)
        baseline = _evaluate_model(
            model=model,
            data_dir=data_dir,
            workspace=workspace,
            split=split,
            max_tokens=max_tokens,
            num_rollouts=num_rollouts,
            output_dir=temporary / "baseline-eval",
            executable=executable,
            baseline=None,
        )
        if isinstance(baseline, _ModelEvaluationFailure):
            return EvaluationFailure(f"baseline evaluation failed: {baseline.reason}")
        candidates: list[CandidateResult] = []
        for index, checkpoint in enumerate(weights):
            weights_sha256 = _file_sha256(checkpoint)
            if isinstance(weights_sha256, EvaluationFailure):
                return weights_sha256
            candidate_root = temporary / f"candidate-{index}"
            candidate_root.mkdir()
            fused_model = candidate_root / "fused"
            fusion_failure = _fuse_checkpoint(
                model=model,
                adapter_dir=adapter_dir,
                weights=checkpoint,
                output_dir=fused_model,
                staging_dir=candidate_root,
            )
            if fusion_failure is not None:
                candidates.append(
                    FailedCandidate(
                        status="failed",
                        checkpoint=_report_path(checkpoint),
                        weights_sha256=weights_sha256,
                        reason=fusion_failure.reason,
                        protocol=None,
                    )
                )
                continue
            candidate = _evaluate_model(
                model=str(fused_model),
                data_dir=data_dir,
                workspace=workspace,
                split=split,
                max_tokens=max_tokens,
                num_rollouts=num_rollouts,
                output_dir=candidate_root / "eval",
                executable=executable,
                baseline=baseline.metrics,
            )
            if isinstance(candidate, _ModelEvaluationFailure):
                candidates.append(
                    FailedCandidate(
                        status="failed",
                        checkpoint=_report_path(checkpoint),
                        weights_sha256=weights_sha256,
                        reason=candidate.reason,
                        protocol=candidate.protocol,
                    )
                )
                continue
            comparison = compare_metrics(baseline.metrics, candidate.metrics)
            if isinstance(comparison, MetricsFailure):
                return EvaluationFailure(comparison.reason)
            candidates.append(
                EvaluatedCandidate(
                    status=comparison.status,
                    checkpoint=_report_path(checkpoint),
                    weights_sha256=weights_sha256,
                    protocol=candidate.protocol,
                    metrics=candidate.metrics,
                    comparison=comparison,
                )
            )
    improved = [
        candidate
        for candidate in candidates
        if isinstance(candidate, EvaluatedCandidate) and candidate.status == "improved"
    ]
    if not improved:
        return RejectedEvaluationReport(
            schema_version=1,
            status="rejected",
            model=model,
            adapter=_report_path(adapter_dir),
            split=split,
            max_tokens=max_tokens,
            num_rollouts=num_rollouts,
            baseline_protocol=baseline.protocol,
            baseline_metrics=baseline.metrics,
            candidates=tuple(candidates),
            metal=metal,
        )
    selected = max(
        improved,
        key=lambda candidate: (
            candidate.metrics.sealed_validation_reward,
            candidate.metrics.parsed_tool_call_rate,
            -candidate.metrics.invalid_tool_rate,
            candidate.metrics.end_token_rate,
            -candidate.metrics.loop_rate,
        ),
    )
    return ImprovedEvaluationReport(
        schema_version=1,
        status="improved",
        model=model,
        adapter=_report_path(adapter_dir),
        split=split,
        max_tokens=max_tokens,
        num_rollouts=num_rollouts,
        selected_checkpoint=selected.checkpoint,
        baseline_protocol=baseline.protocol,
        baseline_metrics=baseline.metrics,
        candidates=tuple(candidates),
        metal=metal,
    )


def _write_report(path: Path, report: EvaluationReport) -> EvaluationFailure | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
    except OSError as error:
        return EvaluationFailure(f"evaluation report could not be created: {error}")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                asdict(report),
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
        return EvaluationFailure(f"evaluation report could not be written: {error}")
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-evaluate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--split", choices=("validation", "holdout"), required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = evaluate_checkpoints(
        model=arguments.model,
        data_dir=arguments.data,
        adapter_dir=arguments.adapter,
        weights=arguments.weights,
        workspace=arguments.workspace,
        split=arguments.split,
        max_tokens=arguments.max_tokens,
        num_rollouts=arguments.num_rollouts,
    )
    if isinstance(result, EvaluationFailure):
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
        return 1
    write_failure = _write_report(arguments.output, result)
    if write_failure is not None:
        print(json.dumps(asdict(write_failure), ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.status == "improved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
