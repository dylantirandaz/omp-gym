"""The fix pipeline: bench, gather wins, retrain, re-bench.

One command turns a failing task into a measured before/after
experiment: score the local policy on the task, collect winning
episodes from an anchor model, retrain the adapter on the
refreshed dataset, and score the policy again. Every step records
into the ledger, and the result is the published delta.
"""

from pathlib import Path

from .bench import run_benchmark
from .export import export_dataset
from .ledger import append_entry
from .runner import EpisodeFailure, run_episode
from .serve import ensure_provider
from .task import TaskLoadError, load_task
from .train import run_training


class FixError(SystemExit):
    """Raised when the fix pipeline cannot run a stage."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"fix failed: {reason}")


def run_fix(
    task_dir: Path,
    anchor_model: str,
    base_model: str,
    adapter_in: Path,
    adapter_out: Path,
    train_iters: int,
    win_episodes: int,
    trials: int,
    port: int,
    runs_dir: Path,
    sessions_root: Path,
    ledger_path,
) -> dict:
    """Run the full pipeline. Returns the before/after summary."""
    task = load_task(task_dir)
    if isinstance(task, TaskLoadError):
        raise FixError(f"bad task {task.path}: {task.reason}")
    if not (adapter_in / "adapters.safetensors").is_file():
        raise FixError(f"no adapter at {adapter_in}")

    ensure_provider(
        Path.home() / ".omp" / "agent" / "models.yml",
        port,
        f"fix-{adapter_out.name}",
        base_model,
    )

    from .rl import _start_policy_server

    def bench_policy(label: str) -> dict:
        server_proc, httpd, chosen_port = _start_policy_server(
            base_model, adapter_in if label == "before" else adapter_out,
            port,
        )
        ensure_provider(
            Path.home() / ".omp" / "agent" / "models.yml",
            chosen_port,
            f"fix-{adapter_out.name}",
            base_model,
        )
        try:
            rows = run_benchmark(
                [f"omp-gym/{base_model}"],
                [task],
                trials,
                runs_dir,
            )
        finally:
            httpd.shutdown()
            server_proc.terminate()
            server_proc.wait(timeout=15)
        clean = [row for row in rows if row.error is None]
        passed = sum(1 for row in clean if row.reward >= 1.0)
        mean_calls = (
            sum(row.tool_calls for row in clean) / len(clean)
            if clean
            else 0.0
        )
        return {
            "label": label,
            "runs": len(clean),
            "passed": passed,
            "mean_tool_calls": round(mean_calls, 1),
        }

    print("stage 1/5: bench the current policy")
    before = bench_policy("before")
    print(f"before: {before}")

    print("stage 2/5: collect winning episodes from the anchor model")
    won = 0
    for _ in range(win_episodes):
        result = run_episode(task, runs_dir, anchor_model)
        if isinstance(result, EpisodeFailure):
            continue
        if result.reward >= 1.0:
            won += 1
    print(f"anchor wins: {won}/{win_episodes}")
    if won == 0:
        raise FixError("anchor model scored no wins; no new data")

    print("stage 3/5: export the refreshed dataset")
    stats = export_dataset(
        runs_dir, sessions_root, Path("dataset"), 1.0,
        base_model, 2048,
    )
    print(f"dataset: {stats.train_samples} train samples")

    print("stage 4/5: retrain the adapter")
    report = run_training(
        data_dir=Path("dataset"),
        model=base_model,
        iterations=train_iters,
        adapter_dir=adapter_out,
        batch_size=1,
        max_seq_length=2048,
    )

    print("stage 5/5: bench the retrained policy")
    after = bench_policy("after")
    print(f"after: {after}")

    summary = {
        "task": task.name,
        "anchor_model": anchor_model,
        "before": before,
        "after": after,
        "train": {
            "first_train_loss": report.first_train_loss,
            "last_train_loss": report.last_train_loss,
        },
        "adapter_out": str(adapter_out),
    }
    append_entry(
        ledger_path,
        kind="fix",
        config={
            "task": task.name,
            "anchor": anchor_model,
            "adapter_in": str(adapter_in),
            "adapter_out": str(adapter_out),
        },
        metrics={
            "before_pass": f"{before['passed']}/{before['runs']}",
            "after_pass": f"{after['passed']}/{after['runs']}",
            "before_mean_calls": before["mean_tool_calls"],
            "after_mean_calls": after["mean_tool_calls"],
        },
        artifacts={},
    )
    return summary
