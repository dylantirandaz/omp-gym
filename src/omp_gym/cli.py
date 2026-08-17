"""Command line interface for omp-gym."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .bench import (
    load_task_suite,
    render_report,
    run_benchmark,
    summarize_report,
    write_manifest,
)
from .config import default_model
from .export import export_dataset
from .layers import LayerSelection, parse_layer_selection
from .ledger import DEFAULT_LEDGER, append_entry
from .preflight import require_metal_gpu
from .runner import EpisodeFailure, run_episode
from .task import TaskLoadError, load_task


def _layer_selection_argument(value: str) -> LayerSelection:
    """Parse a positive layer count or the explicit word all."""
    parsed = parse_layer_selection(value)
    if parsed is None:
        try:
            parsed = parse_layer_selection(int(value))
        except ValueError:
            parsed = None
    if parsed is None:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'")
    return parsed


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
    append_entry(
        DEFAULT_LEDGER,
        kind="run",
        config={"task": task.name, "model": result.model},
        metrics={
            "reward": result.reward,
            "duration_seconds": result.duration_seconds,
            "test_exit_code": result.test_exit_code,
        },
        artifacts={
            "episode_dir": result.episode_dir,
            "session_file": result.session_file,
        },
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


def _cmd_export(
    runs_dir: Path,
    sessions_root: Path | None,
    out_dir: Path,
    min_reward: float,
    tokenizer_id: str,
    token_cap: int,
    pairs: bool,
    min_quality: bool,
    holdout_dir: Path,
    strict_torn: bool,
    max_pairs_per_task: int,
) -> int:
    if pairs:
        from .export import export_pairs

        pair_stats = export_pairs(
            runs_dir,
            out_dir,
            max_pairs_per_task=max_pairs_per_task,
            tokenizer_id=tokenizer_id,
            token_cap=token_cap,
            holdout_dir=holdout_dir,
        )
        print(json.dumps(asdict(pair_stats), indent=2))
        if pair_stats.pairs_written == 0:
            print("no task has both a win and a real loss", file=sys.stderr)
            return 1
        append_entry(
            DEFAULT_LEDGER,
            kind="export",
            config={
                "mode": "dpo-pairs",
                "runs": str(runs_dir),
                "holdout": str(holdout_dir),
                "max_pairs_per_task": max_pairs_per_task,
                "tokenizer": tokenizer_id,
                "token_cap": token_cap,
            },
            metrics=dict(asdict(pair_stats)),
            artifacts={"out_dir": str(out_dir)},
        )
        return 0
    stats = export_dataset(
        runs_dir=runs_dir,
        sessions_root=sessions_root,
        out_dir=out_dir,
        min_reward=min_reward,
        tokenizer=tokenizer_id,
        token_cap=token_cap,
        min_quality=min_quality,
        holdout_dir=holdout_dir,
        strict_torn=strict_torn,
    )
    print(json.dumps(asdict(stats), indent=2))
    if stats.train_samples == 0:
        print("no trainable samples were found", file=sys.stderr)
        return 1
    append_entry(
        DEFAULT_LEDGER,
        kind="export",
        config={
            "runs": str(runs_dir),
            "sessions": "" if sessions_root is None else str(sessions_root),
            "min_reward": min_reward,
            "tokenizer": tokenizer_id,
            "token_cap": token_cap,
            "holdout": str(holdout_dir),
            "strict_torn": strict_torn,
        },
        metrics=dict(asdict(stats)),
        artifacts={"out_dir": str(out_dir)},
    )
    return 0


def _cmd_train(
    data_dir: Path,
    model: str,
    iterations: int,
    adapter_dir: Path,
    batch_size: int,
    max_seq_length: int,
    num_layers: LayerSelection,
    learning_rate: float,
    method: str,
    resume_adapter: Path | None,
    max_train_seconds: int,
    train_drop_pct: float,
    val_drop_ratio: float,
    seed: int,
    dpo_beta: float,
    grad_clip: float,
) -> int:
    if method == "dpo":
        from .train import run_dpo_training

        report = run_dpo_training(
            data_dir=data_dir,
            model=model,
            iterations=iterations,
            adapter_dir=adapter_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            resume_adapter=resume_adapter,
            max_seq_length=max_seq_length,
            seed=seed,
            beta=dpo_beta,
            grad_clip=grad_clip,
            train_drop_pct=train_drop_pct,
            val_drop_ratio=val_drop_ratio,
        )
    else:
        from .train import run_training

        report = run_training(
            data_dir=data_dir,
            model=model,
            iterations=iterations,
            adapter_dir=adapter_dir,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            num_layers=num_layers,
            learning_rate=learning_rate,
            resume_adapter=resume_adapter,
            max_train_seconds=max_train_seconds,
            train_drop_pct=train_drop_pct,
            val_drop_ratio=val_drop_ratio,
        )
    append_entry(
        DEFAULT_LEDGER,
        kind="train",
        config={
            "model": model,
            "adapter": str(adapter_dir),
            "data": str(data_dir),
            "method": method,
            "resume_adapter": (str(resume_adapter) if resume_adapter else None),
            "iterations": iterations,
            "batch_size": batch_size,
            "max_seq_length": max_seq_length,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "max_train_seconds": max_train_seconds,
            "train_drop_pct": train_drop_pct,
            "val_drop_ratio": val_drop_ratio,
            "seed": seed if method == "dpo" else None,
            "dpo_beta": dpo_beta if method == "dpo" else None,
            "grad_clip": grad_clip if method == "dpo" else None,
        },
        metrics=dict(asdict(report)),
        artifacts={
            "adapter_dir": str(adapter_dir),
            "train_report": str(adapter_dir / "train_report.json"),
        },
    )
    return 0


def _cmd_bench(
    models_arg: str,
    tasks_dir: Path,
    trials: int,
    runs_dir: Path,
    report_path: Path,
    seed: int,
    recursive: bool,
    include_minted: bool,
) -> int:
    models = [name.strip() for name in models_arg.split(",") if name.strip()]
    if not models:
        print("no models given", file=sys.stderr)
        return 1
    suite = load_task_suite(
        tasks_dir,
        recursive=recursive,
        include_minted=include_minted,
    )
    if isinstance(suite, TaskLoadError):
        print(f"bad task {suite.path}: {suite.reason}", file=sys.stderr)
        return 1
    if not suite:
        print(f"no tasks found in {tasks_dir}", file=sys.stderr)
        return 1
    rows = run_benchmark(models, suite, trials, runs_dir, seed=seed)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(rows)
    report_path.write_text(report)
    rows_path = report_path.with_suffix(".jsonl")
    rows_path.write_text("\n".join(json.dumps(asdict(row)) for row in rows) + "\n")
    summary_path = report_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summarize_report(rows, seed=seed), indent=2) + "\n"
    )
    manifest_path = write_manifest(report_path, suite)
    print(report)
    print(
        f"report: {report_path}\nrows: {rows_path}\n"
        f"summary: {summary_path}\nmanifest: {manifest_path}"
    )
    append_entry(
        DEFAULT_LEDGER,
        kind="bench",
        config={
            "models": models,
            "tasks": [task.name for task in suite],
            "trials": trials,
            "seed": seed,
            "recursive": recursive,
            "include_minted": include_minted,
        },
        metrics={
            "rows": [asdict(row) for row in rows],
            "summary": summarize_report(rows, seed=seed),
        },
        artifacts={
            "report": str(report_path),
            "rows": str(rows_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    )
    return 0


def _cmd_serve(
    adapter_dir: Path,
    base_model: str,
    port: int,
    model_id: str,
    models_yml: Path,
) -> int:
    from .serve import run_server

    append_entry(
        DEFAULT_LEDGER,
        kind="serve",
        config={
            "adapter": str(adapter_dir),
            "base_model": base_model,
            "port": port,
            "model_id": model_id,
        },
        metrics={},
        artifacts={"adapter_dir": str(adapter_dir)},
    )
    return run_server(
        base_model=base_model,
        adapter_dir=adapter_dir,
        port=port,
        model_id=model_id,
        models_yml=models_yml,
    )


def main() -> None:
    parser = _build_parser()
    _dispatch(parser.parse_args())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-gym")
    model_default = default_model()
    model_help = "MLX model repo id or local path (default from gym.toml)"
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("preflight", help="verify the Metal GPU")

    report_parser = commands.add_parser(
        "report", help="compare adapters and models from the ledger"
    )
    report_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    improve_parser = commands.add_parser(
        "improve", help="hand a research goal to an operator agent"
    )
    improve_parser.add_argument("--goal", required=True)
    improve_parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="maximum omp-gym verb commands the operator may run",
    )
    improve_parser.add_argument(
        "--max-time",
        type=int,
        default=1800,
        help="hard wall-clock limit in seconds",
    )
    improve_parser.add_argument(
        "--operator-model",
        default=None,
        help="omp model for the operator; omitted uses omp config",
    )
    improve_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    inspect_parser = commands.add_parser(
        "inspect", help="logit-lens a local model on a prompt"
    )
    inspect_parser.add_argument("--prompt", required=True)
    inspect_parser.add_argument("--model", default=model_default, help=model_help)
    inspect_parser.add_argument("--adapter", type=Path, default=None)
    inspect_parser.add_argument("--top-k", type=int, default=3)
    inspect_parser.add_argument("--out", type=Path, default=Path("experiments"))
    inspect_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    steer_parser = commands.add_parser(
        "steer", help="A/B test activation steering with an SAE feature"
    )
    steer_parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="sae-weights-*.safetensors from a sae run",
    )
    steer_parser.add_argument("--feature", type=int, required=True)
    steer_parser.add_argument("--model", default=model_default, help=model_help)
    steer_parser.add_argument("--adapter", type=Path, default=None)
    steer_parser.add_argument("--alpha", type=float, default=2.0)
    steer_parser.add_argument("--max-tokens", type=int, default=80)
    steer_parser.add_argument("--seed", type=int, default=0)
    steer_parser.add_argument("--out", type=Path, default=Path("experiments"))
    steer_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    sae_parser = commands.add_parser(
        "sae", help="train a tiny SAE on residual activations"
    )
    sae_parser.add_argument("--data", type=Path, default=Path("dataset"))
    sae_parser.add_argument("--model", default=model_default, help=model_help)
    sae_parser.add_argument("--adapter", type=Path, default=None)
    sae_parser.add_argument("--layer", type=int, default=8)
    sae_parser.add_argument("--features", type=int, default=512)
    sae_parser.add_argument("--samples", type=int, default=256)
    sae_parser.add_argument("--steps", type=int, default=200)
    sae_parser.add_argument("--l1", type=float, default=1e-3)
    sae_parser.add_argument("--seed", type=int, default=0)
    sae_parser.add_argument("--out", type=Path, default=Path("experiments"))
    sae_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    gate_parser = commands.add_parser(
        "gate", help="flag adapters that memorize training tasks"
    )
    gate_parser.add_argument("--model", default=model_default, help=model_help)
    gate_parser.add_argument("--adapter", type=Path, default=None)
    gate_parser.add_argument(
        "--sae",
        type=Path,
        default=None,
        help="sae-weights-*.safetensors trained for the base model",
    )
    gate_parser.add_argument("--out", type=Path, default=Path("experiments"))
    gate_parser.add_argument("--holdout", type=Path, default=Path("holdout-tasks"))
    gate_parser.add_argument("--tasks", type=Path, default=Path("tasks"))
    gate_parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset"),
        help="dataset used when a fresh SAE must be trained",
    )
    gate_parser.add_argument("--train-drop-pct", type=float, default=0.15)
    gate_parser.add_argument("--val-drop-ratio", type=float, default=0.25)
    gate_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    rl_parser = commands.add_parser(
        "rl", help="group-relative policy gradient on live episodes"
    )
    rl_parser.add_argument(
        "--task",
        type=Path,
        action="append",
        required=True,
        help="task directory; repeat for a task mix",
    )
    rl_parser.add_argument(
        "--base-model",
        default=model_default,
        help=model_help,
    )
    rl_parser.add_argument("--adapter", type=Path, default=Path("adapters/v4"))
    rl_parser.add_argument("--out-adapter", type=Path, default=Path("adapters/rl-v1"))
    rl_parser.add_argument("--group", type=int, default=3)
    rl_parser.add_argument("--iters", type=int, default=2)
    rl_parser.add_argument("--port", type=int, default=8810)
    rl_parser.add_argument("--runs", type=Path, default=Path("runs"))
    rl_parser.add_argument(
        "--sample-temp",
        type=float,
        default=0.7,
        help="rollout sampling temperature",
    )
    rl_parser.add_argument("--kl-beta", type=float, default=0.0)
    rl_parser.add_argument("--seed", type=int, default=0)
    rl_parser.add_argument("--max-seq-length", type=int, default=4096)
    rl_parser.add_argument("--grad-clip", type=float, default=1.0)
    rl_parser.add_argument("--eval-episodes", type=int, default=0)
    rl_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    publish_parser = commands.add_parser(
        "publish", help="render the ledger page and optionally push it"
    )
    publish_parser.add_argument("--push", action="store_true")
    publish_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    doctor_parser = commands.add_parser(
        "doctor", help="check the environment and print fixes"
    )
    doctor_parser.add_argument("--env", type=Path, default=Path(".env"))
    doctor_parser.add_argument(
        "--sessions",
        type=Path,
        default=Path.home() / ".omp" / "agent" / "sessions",
    )

    init_parser = commands.add_parser(
        "init", help="doctor plus one scored first episode"
    )
    init_parser.add_argument("--env", type=Path, default=Path(".env"))
    init_parser.add_argument(
        "--sessions",
        type=Path,
        default=Path.home() / ".omp" / "agent" / "sessions",
    )
    init_parser.add_argument("--runs", type=Path, default=Path("runs"))
    init_parser.add_argument("--task", type=Path, default=Path("tasks/fizzbuzz-fix"))

    import_parser = commands.add_parser(
        "import", help="import sessions from other coding agents"
    )
    import_parser.add_argument(
        "--from",
        dest="source",
        choices=["claude", "codex"],
        required=True,
    )
    import_parser.add_argument("--out", type=Path, default=Path("imported"))
    import_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    clusters_parser = commands.add_parser(
        "clusters", help="cluster sessions and episodes by failure mode"
    )
    clusters_parser.add_argument(
        "--sessions",
        type=Path,
        default=Path.home() / ".omp" / "agent" / "sessions",
    )
    clusters_parser.add_argument("--runs", type=Path, default=Path("runs"))
    clusters_parser.add_argument("--out", type=Path, default=Path("experiments"))
    clusters_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    mint_parser = commands.add_parser(
        "mint", help="mint tasks from sessions where the agent failed"
    )
    mint_parser.add_argument(
        "--sessions",
        type=Path,
        default=Path.home() / ".omp" / "agent" / "sessions",
    )
    mint_parser.add_argument("--out", type=Path, default=Path("tasks") / "minted")
    mint_parser.add_argument("--limit", type=int, default=5)
    mint_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    ui_parser = commands.add_parser("ui", help="serve the experiment dashboard")
    ui_parser.add_argument("--port", type=int, default=8900)
    ui_parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    ui_parser.add_argument("--runs", type=Path, default=Path("runs"))

    serve_parser = commands.add_parser(
        "serve", help="serve an adapter as an omp provider"
    )
    serve_parser.add_argument("--adapter", type=Path, default=Path("adapters/v3"))
    serve_parser.add_argument("--base-model", default=model_default, help=model_help)
    serve_parser.add_argument("--port", type=int, default=8800)
    serve_parser.add_argument("--model-id", default="local-v3")
    serve_parser.add_argument(
        "--models-yml",
        type=Path,
        default=Path.home() / ".omp" / "agent" / "models.yml",
    )

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
        default=None,
        help="omp sessions root to harvest; personal session history "
        "enters the dataset only when this path is named",
    )
    export_parser.add_argument(
        "--tokenizer",
        default=model_default,
        help="tokenizer that measures the sample token budget; "
        "must match the trainee model family",
    )
    export_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="sample token cap; must match the training sequence cap",
    )
    export_parser.add_argument("--holdout", type=Path, default=Path("holdout-tasks"))
    export_parser.add_argument(
        "--strict-torn",
        action="store_true",
        help="fail when a session or episode JSONL line is torn",
    )
    export_parser.add_argument(
        "--max-pairs-per-task",
        type=int,
        default=16,
        help="maximum DPO pairs from one task",
    )
    export_parser.add_argument(
        "--pairs",
        action="store_true",
        help="export DPO preference pairs instead of SFT samples",
    )
    export_parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="include sessions that ended as failures in the dataset",
    )

    train_parser = commands.add_parser("train", help="train a LoRA adapter")
    train_parser.add_argument("--data", type=Path, default=Path("dataset"))
    train_parser.add_argument("--model", default=model_default, help=model_help)
    train_parser.add_argument("--iters", type=int, default=60)
    train_parser.add_argument("--adapter", type=Path, default=Path("adapters/v1"))
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--max-seq-length", type=int, default=4096)
    train_parser.add_argument(
        "--num-layers",
        type=_layer_selection_argument,
        default=16,
        help="last layer count, or 'all'",
    )
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--max-train-seconds", type=int, default=14400)
    train_parser.add_argument("--train-drop-pct", type=float, default=0.15)
    train_parser.add_argument("--val-drop-ratio", type=float, default=0.25)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--dpo-beta", type=float, default=0.1)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument(
        "--method",
        choices=["sft", "dpo"],
        default="sft",
        help="sft on chat samples or dpo on preference pairs",
    )
    train_parser.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help="continue from an existing adapter (dpo after sft)",
    )

    bench_parser = commands.add_parser(
        "bench", help="benchmark models on the task suite"
    )
    bench_parser.add_argument(
        "--models",
        required=True,
        help="comma-separated omp model names",
    )
    bench_parser.add_argument("--tasks", type=Path, default=Path("tasks"))
    bench_parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="episodes per model x task cell; 3 gives the Wilson "
        "interval something to work with",
    )
    bench_parser.add_argument("--runs", type=Path, default=Path("runs"))
    bench_parser.add_argument("--report", type=Path, default=Path("bench-report.md"))
    bench_parser.add_argument("--seed", type=int, default=0)
    bench_parser.add_argument("--recursive", action="store_true")
    bench_parser.add_argument("--include-minted", action="store_true")

    return parser


def _dispatch(args) -> None:
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
                args.pairs,
                not args.no_quality_filter,
                args.holdout,
                args.strict_torn,
                args.max_pairs_per_task,
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
                args.num_layers,
                args.learning_rate,
                args.method,
                args.resume_adapter,
                args.max_train_seconds,
                args.train_drop_pct,
                args.val_drop_ratio,
                args.seed,
                args.dpo_beta,
                args.grad_clip,
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
                args.seed,
                args.recursive,
                args.include_minted,
            )
        )
    if args.command == "serve":
        raise SystemExit(
            _cmd_serve(
                args.adapter,
                args.base_model,
                args.port,
                args.model_id,
                args.models_yml,
            )
        )
    if args.command == "report":
        from .report import report_from_ledger

        print(report_from_ledger(args.ledger))
        raise SystemExit(0)
    if args.command == "improve":
        from .improve import run_improve

        result = run_improve(
            goal=args.goal,
            budget=args.budget,
            max_time=args.max_time,
            ledger_path=args.ledger,
            model=args.operator_model,
        )
        print(json.dumps(asdict(result), indent=2))
        raise SystemExit(0 if result.summary_written else 1)
    if args.command == "inspect":
        from .inspect import run_lens

        result = run_lens(
            prompt=args.prompt,
            model_id=args.model,
            adapter_dir=args.adapter,
            top_k=args.top_k,
            out_dir=args.out,
        )
        append_entry(
            args.ledger,
            kind="inspect",
            config={"model": args.model, "adapter": str(args.adapter)},
            metrics={"layers": result["layers"]},
            artifacts={},
        )
        raise SystemExit(0)
    if args.command == "steer":
        from .export import SYSTEM_PROMPT
        from .steer import report_ab, run_ab

        def agent_prompt(task: str) -> str:
            return SYSTEM_PROMPT + "\n\n" + task

        prompts = [
            agent_prompt(
                "The test file test_fizzbuzz.py fails. "
                "Find the bug in fizzbuzz.py and fix it."
            ),
            agent_prompt("Implement total_column in report.py so the tests pass."),
            agent_prompt("Read the file main.py and explain it."),
            agent_prompt(
                "Write a function slugify that turns a title into a URL slug."
            ),
        ]
        result = run_ab(
            weights_path=args.weights,
            feature=args.feature,
            model_id=args.model,
            adapter_dir=args.adapter,
            alpha=args.alpha,
            prompts=prompts,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        payload = report_ab(result, args.out)
        append_entry(
            args.ledger,
            kind="steer",
            config={
                "feature": args.feature,
                "alpha": args.alpha,
                "model": args.model,
                "seed": args.seed,
            },
            metrics={
                "unsteered_tool_calls": result.unsteered_tool_calls,
                "steered_tool_calls": result.steered_tool_calls,
                "prompts": result.prompts,
            },
            artifacts={"steer": str(args.out)},
        )
        raise SystemExit(0)
    if args.command == "sae":
        from .sae import train_sae

        metrics = train_sae(
            args.data,
            args.model,
            args.adapter,
            args.out,
            layer=args.layer,
            features=args.features,
            samples=args.samples,
            steps=args.steps,
            l1=args.l1,
            seed=args.seed,
        )
        append_entry(
            args.ledger,
            kind="sae",
            config={
                "model": args.model,
                "adapter": str(args.adapter),
                "layer": args.layer,
                "features": args.features,
                "samples": args.samples,
                "steps": args.steps,
                "l1": args.l1,
                "seed": args.seed,
            },
            metrics={k: v for k, v in metrics.items() if k != "artifact"},
            artifacts={"sae": metrics["artifact"]},
        )
        raise SystemExit(0)
    if args.command == "gate":
        from .gate import run_gate

        payload = run_gate(
            model_id=args.model,
            adapter_dir=args.adapter,
            sae_weights=args.sae,
            out_dir=args.out,
            holdout_dir=args.holdout,
            tasks_dir=args.tasks,
            data_dir=args.data,
            train_drop_pct=args.train_drop_pct,
            val_drop_ratio=args.val_drop_ratio,
        )
        append_entry(
            args.ledger,
            kind="gate",
            config={
                "model": args.model,
                "adapter": str(args.adapter),
                "sae": payload["sae_weights"],
                "train_drop_pct": args.train_drop_pct,
                "val_drop_ratio": args.val_drop_ratio,
            },
            metrics={
                "drift_score": payload["drift_score"],
                "leakage_score": payload["leakage_score"],
                "memorization_score": payload["memorization_score"],
                "flagged": payload["flagged"],
            },
            artifacts={"gate": payload["artifact"]},
        )
        raise SystemExit(1 if payload["flagged"] else 0)
    if args.command == "rl":
        from .rl import run_rl

        summary = run_rl(
            task_dirs=args.task,
            base_model=args.base_model,
            adapter_dir=args.adapter,
            out_adapter=args.out_adapter,
            group_size=args.group,
            iterations=args.iters,
            port=args.port,
            runs_dir=args.runs,
            sample_temperature=args.sample_temp,
            ledger_path=args.ledger,
            kl_beta=args.kl_beta,
            seed=args.seed,
            max_seq_len=args.max_seq_length,
            grad_clip=args.grad_clip,
            eval_episodes=args.eval_episodes,
        )
        print(json.dumps(summary, indent=2))
        raise SystemExit(0)
    if args.command == "publish":
        from .publish import publish_report

        result = publish_report(Path.cwd(), args.ledger, args.push)
        print(json.dumps(result, indent=2))
        raise SystemExit(0)
    if args.command == "doctor":
        from .doctor import print_doctor, run_doctor

        raise SystemExit(print_doctor(run_doctor(args.env, args.sessions)))
    if args.command == "init":
        from .doctor import run_init

        raise SystemExit(run_init(args.env, args.sessions, args.runs, args.task))
    if args.command == "import":
        from .importers import import_sessions

        stats = import_sessions(args.source, args.out)
        print(json.dumps(asdict(stats), indent=2))
        append_entry(
            args.ledger,
            kind="import",
            config={"source": args.source},
            metrics={
                "files_seen": stats.files_seen,
                "files_written": stats.files_written,
            },
            artifacts={"out": stats.out_dir},
        )
        raise SystemExit(0)
    if args.command == "clusters":
        from .clusters import compute_clusters

        payload = compute_clusters(args.sessions, args.runs, args.out)
        append_entry(
            args.ledger,
            kind="clusters",
            config={"sessions": str(args.sessions)},
            metrics={mode: data["count"] for mode, data in payload["clusters"].items()},
            artifacts={"clusters": str(args.out / "clusters.json")},
        )
        for mode, data in payload["clusters"].items():
            print(f"{mode:24s} {data['count']}")
        raise SystemExit(0)
    if args.command == "mint":
        from .mint import mint_tasks

        minted = mint_tasks(args.sessions, args.out, args.limit)
        print(json.dumps([asdict(task) for task in minted], indent=2))
        append_entry(
            args.ledger,
            kind="mint",
            config={"sessions": str(args.sessions), "limit": args.limit},
            metrics={"minted": len(minted)},
            artifacts={"out": str(args.out)},
        )
        raise SystemExit(0)
    if args.command == "ui":
        from .ui import run_ui

        run_ui(args.port, args.ledger, args.runs)
        raise SystemExit(0)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
