"""Command line interface for omp-gym."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

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


def _cmd_export(runs_dir: Path, out_dir: Path, min_reward: float) -> int:
    stats = export_dataset(runs_dir, out_dir, min_reward)
    print(json.dumps(asdict(stats), indent=2))
    if stats.episodes_exported == 0:
        print("no episodes reached the reward threshold", file=sys.stderr)
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

    args = parser.parse_args()
    if args.command == "preflight":
        raise SystemExit(_cmd_preflight())
    if args.command == "run":
        raise SystemExit(_cmd_run(args.task, args.runs, args.model))
    if args.command == "export":
        raise SystemExit(_cmd_export(args.runs, args.out, args.min_reward))
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
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
