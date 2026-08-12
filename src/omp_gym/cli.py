"""Command line interface for omp-gym."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .bench import load_task_suite, render_report, run_benchmark
from .export import export_dataset
from .preflight import require_metal_gpu
from .runner import EpisodeFailure, run_episode
from .task import TaskLoadError, load_task


def _cmd_preflight() -> int:
    require_metal_gpu()
    return 0


def _cmd_run(task_dir: Path, runs_dir: Path, model: str | None) -> int:
    task = load_task(task_dir)
    if isinstance(task, TaskLoadError):
        print(f"bad task {task.path}: {task.reason}", file=sys.stderr)
        return 1
    print(f"episode start: task={task.name}")
    result = run_episode(task, runs_dir, model)
    if isinstance(result, EpisodeFailure):
        print(
            f"episode failed: task={result.task} reason={result.reason}",
            file=sys.stderr,
        )
        return 1
    prompt_file = Path(result.episode_dir) / "prompt.txt"
    prompt_file.write_text(task.prompt + "\n")
    print(json.dumps(asdict(result), indent=2))
    return 0


def _cmd_export(
    runs_dir: Path,
    sessions_root: Path,
    out_dir: Path,
    min_reward: float,
    tokenizer_id: str,
    token_cap: int,
) -> int:
    stats = export_dataset(
        runs_dir,
        sessions_root,
        out_dir,
        min_reward,
        tokenizer_id,
        token_cap,
    )
    print(json.dumps(asdict(stats), indent=2))
    if stats.train_samples == 0:
        print("no trainable samples were found", file=sys.stderr)
        return 1
    return 0


def _cmd_train(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    max_seq_length: int,
) -> int:
    from .train import run_training

    run_training(
        data_dir=data_dir,
        model=model,
        iterations=iterations,
        adapter_dir=adapter_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    return 0


def _cmd_bench(
    models_arg: str,
    tasks_dir: Path,
    trials: int,
    runs_dir: Path,
    report_path: Path,
) -> int:
    models = [name.strip() for name in models_arg.split(",") if name.strip()]
    if not models:
        print("no models given", file=sys.stderr)
        return 1
    suite = load_task_suite(tasks_dir)
    if isinstance(suite, TaskLoadError):
        print(f"bad task {suite.path}: {suite.reason}", file=sys.stderr)
        return 1
    if not suite:
        print(f"no tasks found in {tasks_dir}", file=sys.stderr)
        return 1
    rows = run_benchmark(models, suite, trials, runs_dir)
    report = render_report(rows)
    report_path.write_text(report)
    rows_path = report_path.with_suffix(".jsonl")
    rows_path.write_text(
        "\n".join(json.dumps(asdict(row)) for row in rows) + "\n"
    )
    print(report)
    print(f"report: {report_path}\nrows: {rows_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="omp-gym")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight", help="verify the Metal GPU")

    run_parser = commands.add_parser("run", help="run one episode")
    run_parser.add_argument("--task", type=Path, required=True)
    run_parser.add_argument("--runs", type=Path, default=Path("runs"))
    run_parser.add_argument("--model", default=None)

    export_parser = commands.add_parser("export", help="export a dataset")
    export_parser.add_argument("--runs", type=Path, default=Path("runs"))
    export_parser.add_argument("--out", type=Path, default=Path("dataset"))
    export_parser.add_argument("--min-reward", type=float, default=1.0)
    export_parser.add_argument(
        "--sessions",
        type=Path,
        default=Path.home() / ".omp" / "agent" / "sessions",
        help="omp sessions root; every session below it is harvested",
    )
    export_parser.add_argument(
        "--tokenizer",
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="tokenizer that measures the sample token budget; "
        "must match the trainee model family",
    )
    export_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="sample token cap; must match the training sequence cap",
    )

    train_parser = commands.add_parser("train", help="train a LoRA adapter")
    train_parser.add_argument("--data", type=Path, default=Path("dataset"))
    train_parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    )
    train_parser.add_argument("--iters", type=int, default=60)
    train_parser.add_argument(
        "--adapter", type=Path, default=Path("adapters/v1")
    )
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--max-seq-length", type=int, default=4096)

    bench_parser = commands.add_parser(
        "bench", help="benchmark models on the task suite"
    )
    bench_parser.add_argument(
        "--models",
        required=True,
        help="comma-separated omp model names",
    )
    bench_parser.add_argument("--tasks", type=Path, default=Path("tasks"))
    bench_parser.add_argument("--trials", type=int, default=1)
    bench_parser.add_argument("--runs", type=Path, default=Path("runs"))
    bench_parser.add_argument(
        "--report", type=Path, default=Path("bench-report.md")
    )

    args = parser.parse_args()
    if args.command == "preflight":
        raise SystemExit(_cmd_preflight())
    if args.command == "run":
        raise SystemExit(_cmd_run(args.task, args.runs, args.model))
    if args.command == "export":
        raise SystemExit(
            _cmd_export(
                args.runs,
                args.sessions,
                args.out,
                args.min_reward,
                args.tokenizer,
                args.max_tokens,
            )
        )
    if args.command == "train":
        raise SystemExit(
            _cmd_train(
                args.data,
                args.model,
                args.iters,
                args.adapter,
                args.batch_size,
                args.max_seq_length,
            )
        )
    if args.command == "bench":
        raise SystemExit(
            _cmd_bench(
                args.models,
                args.tasks,
                args.trials,
                args.runs,
                args.report,
            )
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
