from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from omp_coding.bench import (
    BenchFailure,
    BenchReport,
    aggregate_runs,
    benchmark_digest,
    main,
    pass_at_k,
    run_models,
    write_report,
)

TASK_DIGESTS = {"alpha": "digest-alpha", "beta": "digest-beta"}


def _trace(
    *,
    task: str,
    model: str,
    score: float,
    ok: bool = True,
    tool_calls: int = 2,
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    cached_tokens: int = 5,
    seconds: float = 4.0,
    contract: str = "omp-native-v1",
) -> dict[str, object]:
    nodes: list[dict[str, object]] = [
        {"message": {"role": "system", "content": "Use tools."}, "sampled": False},
        {
            "parent": 0,
            "message": {"role": "user", "content": "Fix it."},
            "sampled": False,
        },
    ]
    calls: list[dict[str, object]] = []
    parent = 1
    for index in range(tool_calls):
        nodes.append(
            {
                "parent": parent,
                "sampled": True,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": f"call-{index}", "name": "read", "arguments": "{}"}
                    ],
                },
            }
        )
        parent = len(nodes) - 1
        nodes.append(
            {
                "parent": parent,
                "sampled": False,
                "message": {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "content": "1:pass",
                },
            }
        )
        parent = len(nodes) - 1
        calls.append(
            {
                "node": parent - 1,
                "model": model,
                "finish_reason": "tool_calls",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_input_tokens": cached_tokens,
                },
            }
        )
    nodes.append(
        {
            "parent": parent,
            "sampled": True,
            "message": {"role": "assistant", "content": "Done."},
        }
    )
    calls.append(
        {
            "node": len(nodes) - 1,
            "model": model,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_input_tokens": cached_tokens,
            },
        }
    )
    return {
        "id": f"{model}-{task}-{score}-{ok}",
        "ok": ok,
        "stop_condition": "agent_completed" if ok else "harness_error",
        "task": {
            "type": "OmpTask",
            "data": {
                "task_id": f"omp-session/{task}",
                "task_revision": 1,
                "task_digest": TASK_DIGESTS[task],
                "split": "holdout",
            },
        },
        "agent": {"config": {"model": model}, "name": "agent"},
        "tools": [{"name": "read", "description": "Read.", "parameters": {}}],
        "nodes": nodes,
        "calls": calls,
        "rewards": {"tests": {"score": score, "weight": 1.0}},
        "metrics": {"passed_cases": 1.0 if score == 1.0 else 0.0, "total_cases": 1.0},
        "info": {"omp_tool_contract": contract, "omp_version": "17.2.15"},
        "timing": {"agent": {"start": 100.0, "end": 100.0 + seconds}},
    }


def _write_run(run_dir: Path, traces: Sequence[dict[str, object]]) -> Path:
    run_dir.mkdir(parents=True)
    lines = [
        json.dumps({"id": f"episode-{index}", "ok": True, "traces": [trace]})
        for index, trace in enumerate(traces)
    ]
    (run_dir / "traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def _write_task(tasks_dir: Path, name: str, *, models: Sequence[str]) -> None:
    task_dir = tasks_dir / name
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        f'schema_version = 3\ntask_id = "omp-session/{name}"\n', encoding="utf-8"
    )
    provenance = {
        "schema_version": 1,
        "session_id": f"session-{name}",
        "session_cwd_sha256": "0" * 64,
        "episode_index": 0,
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:10:00Z",
        "models": list(models),
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "cache_read_tokens": 300,
            "cache_write_tokens": 0,
            "total_tokens": 1500,
            "cost": 0.25,
        },
        "assistant_turns": 6,
        "tool_calls": 9,
        "base_commit": "a" * 40,
        "repository": "owner/repo",
        "test_command": "pytest -q",
        "failing_run": {"exit_code": 1, "order": 3},
        "passing_run": {"exit_code": 0, "order": 5},
        "gate": {
            "before": {
                "status": "failed",
                "passed_cases": 0,
                "total_cases": 1,
                "exit_code": 1,
                "seconds": 1.0,
            },
            "reference": {
                "status": "passed",
                "passed_cases": 1,
                "total_cases": 1,
                "exit_code": 0,
                "seconds": 1.0,
            },
        },
        "minted_at": "2026-01-02T00:00:00Z",
        "mint_version": 1,
    }
    (task_dir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


def _two_model_runs(root: Path) -> tuple[Path, Path]:
    strong = _write_run(
        root / "strong",
        [
            _trace(task="alpha", model="strong", score=1.0),
            _trace(task="alpha", model="strong", score=1.0, tool_calls=4),
            _trace(task="beta", model="strong", score=1.0),
            _trace(task="beta", model="strong", score=0.5, ok=False, seconds=2.0),
        ],
    )
    weak = _write_run(
        root / "weak",
        [
            _trace(task="alpha", model="weak", score=0.0),
            _trace(task="alpha", model="weak", score=1.0),
            _trace(task="beta", model="weak", score=0.0),
            _trace(task="beta", model="weak", score=0.0),
        ],
    )
    return strong, weak


class PassAtKTests(unittest.TestCase):
    def test_estimator_matches_closed_form(self) -> None:
        self.assertEqual(pass_at_k(1, 1, 1), 1.0)
        self.assertEqual(pass_at_k(1, 0, 1), 0.0)
        self.assertAlmostEqual(pass_at_k(4, 1, 2), 0.5)
        self.assertEqual(pass_at_k(4, 3, 2), 1.0)
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)


class AggregateTests(unittest.TestCase):
    def test_leaderboard_metrics_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strong, weak = _two_model_runs(root)
            report = aggregate_runs([weak, strong], tasks_dir=None)

        self.assertNotIsInstance(report, BenchFailure)
        assert isinstance(report, BenchReport)
        self.assertEqual(report.warnings, ())
        self.assertIsNotNone(report.benchmark_digest)
        self.assertEqual([run.model for run in report.runs], ["weak", "strong"])
        self.assertEqual(report.runs[0].tool_contract, "omp-native-v1")
        self.assertEqual(report.runs[0].omp_version, "17.2.15")
        self.assertEqual([entry.model for entry in report.leaderboard], ["strong", "weak"])

        best = report.leaderboard[0]
        self.assertEqual(best.tasks, 2)
        self.assertEqual(best.rollouts, 4)
        self.assertAlmostEqual(best.mean_reward, 0.875)
        self.assertAlmostEqual(best.pass_rate, 0.75)
        self.assertEqual(best.k, 2)
        self.assertEqual(
            best.pass_at_k_by_task,
            {"omp-session/alpha": 1.0, "omp-session/beta": 1.0},
        )
        self.assertEqual(best.pass_at_k, 1.0)
        self.assertAlmostEqual(best.error_rate, 0.25)
        # Three calls per two-tool trace, five for the four-tool one: 14 calls.
        self.assertAlmostEqual(best.mean_input_tokens, 14 * 105 / 4)
        self.assertAlmostEqual(best.mean_output_tokens, 14 * 10 / 4)
        self.assertAlmostEqual(best.mean_tool_calls, 2.5)
        self.assertAlmostEqual(best.mean_seconds, 3.5)

        weak_entry = report.leaderboard[1]
        self.assertAlmostEqual(weak_entry.pass_at_k_by_task["omp-session/alpha"], 1.0)
        self.assertEqual(weak_entry.pass_at_k_by_task["omp-session/beta"], 0.0)
        self.assertEqual(
            report.matrix,
            {
                "omp-session/alpha": {"strong": 1.0, "weak": 0.5},
                "omp-session/beta": {"strong": 0.75, "weak": 0.0},
            },
        )

    def test_single_rollout_has_no_pass_at_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(
                Path(temporary) / "run",
                [_trace(task="alpha", model="m", score=1.0)],
            )
            report = aggregate_runs([run], tasks_dir=None)

        assert isinstance(report, BenchReport)
        self.assertEqual(report.leaderboard[0].k, 1)
        self.assertIsNone(report.leaderboard[0].pass_at_k)
        self.assertEqual(report.leaderboard[0].pass_at_k_by_task, {})

    def test_run_mixing_models_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(
                Path(temporary) / "run",
                [
                    _trace(task="alpha", model="one", score=1.0),
                    _trace(task="alpha", model="two", score=1.0),
                ],
            )
            result = aggregate_runs([run], tasks_dir=None)

        self.assertIsInstance(result, BenchFailure)
        assert isinstance(result, BenchFailure)
        self.assertIn("mixes models", result.reason)

    def test_model_falls_back_to_call_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = _trace(task="alpha", model="from-calls", score=1.0)
            trace["agent"] = {"config": {"model": None}}
            run = _write_run(Path(temporary) / "run", [trace])
            report = aggregate_runs([run], tasks_dir=None)

        assert isinstance(report, BenchReport)
        self.assertEqual(report.leaderboard[0].model, "from-calls")

    def test_different_task_sets_warn_instead_of_merging_silently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alpha_only = _write_run(
                root / "alpha", [_trace(task="alpha", model="a", score=1.0)]
            )
            both = _write_run(
                root / "both",
                [
                    _trace(task="alpha", model="b", score=1.0),
                    _trace(task="beta", model="b", score=1.0),
                ],
            )
            report = aggregate_runs([alpha_only, both], tasks_dir=None)

        assert isinstance(report, BenchReport)
        self.assertIsNone(report.benchmark_digest)
        self.assertEqual(len(report.warnings), 1)
        self.assertIn("different task sets", report.warnings[0])
        self.assertNotEqual(report.runs[0].benchmark_digest, report.runs[1].benchmark_digest)

    def test_benchmark_digest_ignores_rollout_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forward = _write_run(
                root / "forward",
                [
                    _trace(task="alpha", model="m", score=1.0),
                    _trace(task="beta", model="m", score=0.0),
                ],
            )
            backward = _write_run(
                root / "backward",
                [
                    _trace(task="beta", model="m", score=1.0),
                    _trace(task="alpha", model="m", score=1.0),
                    _trace(task="alpha", model="m", score=0.0),
                ],
            )
            report = aggregate_runs([forward, backward], tasks_dir=None)

        assert isinstance(report, BenchReport)
        self.assertEqual(report.runs[0].benchmark_digest, report.runs[1].benchmark_digest)
        self.assertEqual(report.warnings, ())
        self.assertEqual(report.leaderboard[0].rollouts, 5)

    def test_missing_trace_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            empty = Path(temporary) / "empty"
            empty.mkdir()
            result = aggregate_runs([empty], tasks_dir=None)

        self.assertIsInstance(result, BenchFailure)
        assert isinstance(result, BenchFailure)
        self.assertIn("expected one", result.reason)

    def test_digest_helper_is_stable(self) -> None:
        trace = _trace(task="alpha", model="m", score=1.0)
        with tempfile.TemporaryDirectory() as temporary:
            run = _write_run(Path(temporary) / "run", [trace, trace])
            report = aggregate_runs([run], tasks_dir=None)
        assert isinstance(report, BenchReport)
        self.assertEqual(len(report.runs[0].benchmark_digest), 64)
        self.assertEqual(report.runs[0].benchmark_digest, report.benchmark_digest)
        self.assertEqual(benchmark_digest(()), benchmark_digest(()))


class ReferenceAndReportTests(unittest.TestCase):
    def test_reference_rows_and_written_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strong, weak = _two_model_runs(root)
            tasks_dir = root / "tasks"
            _write_task(tasks_dir, "alpha", models=["claude-x"])
            _write_task(tasks_dir, "beta", models=["claude-x", "gpt-y"])
            (tasks_dir / "not-a-task").mkdir()
            report = aggregate_runs([strong, weak], tasks_dir=tasks_dir)
            assert isinstance(report, BenchReport)
            output = root / "report"
            self.assertIsNone(write_report(report, output))
            document = json.loads((output / "bench.json").read_text(encoding="utf-8"))
            markdown = (output / "bench.md").read_text(encoding="utf-8")

        self.assertEqual(
            [task.task_id for task in report.reference_tasks],
            ["omp-session/alpha", "omp-session/beta"],
        )
        self.assertEqual(report.reference_tasks[0].reward, 1.0)
        self.assertEqual(report.reference_tasks[0].usage.input_tokens, 1000)
        self.assertEqual(report.reference_tasks[1].models, ("claude-x", "gpt-y"))
        self.assertEqual(
            [entry.model for entry in report.reference_models],
            ["claude-x", "claude-x + gpt-y"],
        )
        self.assertEqual(report.reference_models[0].mean_tool_calls, 9.0)

        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(list(document), sorted(document))
        self.assertEqual(document["matrix"]["omp-session/beta"]["strong"], 0.75)
        self.assertEqual(document["reference_tasks"][0]["usage"]["cost"], 0.25)

        self.assertIn("## Leaderboard", markdown)
        self.assertIn("## Reference sessions", markdown)
        self.assertIn("## Per-task mean reward", markdown)
        self.assertIn("## Runs", markdown)
        leaderboard_rows = [
            line for line in markdown.splitlines() if line.startswith("| `")
        ]
        self.assertTrue(leaderboard_rows[0].startswith("| `strong` |"))
        self.assertTrue(leaderboard_rows[1].startswith("| `weak` |"))
        self.assertIn("| task | `strong` | `weak` | reference |", markdown)
        self.assertIn("| `omp-session/beta` | 0.750 | 0.000 | 1.000 |", markdown)

    def test_invalid_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _write_run(root / "run", [_trace(task="alpha", model="m", score=1.0)])
            tasks_dir = root / "tasks"
            _write_task(tasks_dir, "alpha", models=["claude-x"])
            provenance = tasks_dir / "alpha" / "provenance.json"
            broken = json.loads(provenance.read_text(encoding="utf-8"))
            del broken["usage"]["cost"]
            provenance.write_text(json.dumps(broken), encoding="utf-8")
            result = aggregate_runs([run], tasks_dir=tasks_dir)

        self.assertIsInstance(result, BenchFailure)
        assert isinstance(result, BenchFailure)
        self.assertIn("cost", result.reason)

    def test_cli_aggregate_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strong, weak = _two_model_runs(root)
            output = root / "out"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    ["aggregate", str(strong), str(weak), "--output", str(output)]
                )
                failing = main(
                    ["aggregate", str(root / "missing"), "--output", str(output)]
                )
            self.assertEqual(status, 0)
            self.assertTrue((output / "bench.json").is_file())
            self.assertTrue((output / "bench.md").is_file())
        self.assertEqual(failing, 1)
        self.assertIn("## Leaderboard", stdout.getvalue())
        self.assertIn("does not exist", stdout.getvalue())


class RunModelsTests(unittest.TestCase):
    def test_run_models_invokes_eval_per_model_and_aggregates(self) -> None:
        commands: list[list[str]] = []

        def runner(command: Sequence[str]) -> int:
            commands.append(list(command))
            output_dir = Path(command[command.index("--output-dir") + 1])
            model = command[command.index("--model") + 1]
            _write_run(
                output_dir / "run-uuid",
                [
                    _trace(task="alpha", model=model, score=1.0),
                    _trace(task="beta", model=model, score=0.0 if model == "b" else 1.0),
                ],
            )
            return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks_dir = root / "tasks"
            _write_task(tasks_dir, "alpha", models=["claude-x"])
            _write_task(tasks_dir, "beta", models=["claude-x"])
            output = root / "bench"
            report = run_models(
                ["org/a", "b"],
                tasks_dir=tasks_dir,
                split="holdout",
                base_url="http://127.0.0.1:8080/v1",
                api_key_var="BENCH_KEY",
                num_rollouts=2,
                max_concurrent=3,
                max_tokens=2048,
                max_total_tokens=50_000,
                output_dir=output,
                eval_executable=Path("/opt/bin/eval"),
                runner=runner,
            )
            self.assertNotIsInstance(report, BenchFailure)
            self.assertTrue((output / "bench.json").is_file())
            self.assertTrue((output / "bench.md").is_file())
            run_dirs = sorted(path.name for path in (output / "runs").iterdir())

        assert isinstance(report, BenchReport)
        self.assertEqual(run_dirs, ["b", "org_a"])
        self.assertEqual([entry.model for entry in report.leaderboard], ["org/a", "b"])
        self.assertEqual(len(report.reference_tasks), 2)
        self.assertEqual(len(commands), 2)
        first = commands[0]
        expected_head = [
            str(Path("/opt/bin/eval")),
            "omp-coding",
            "--model",
            "org/a",
            "--no-push",
            "--no-rich",
            "--env.taskset.split",
            "holdout",
            "--env.taskset.tasks-dir",
            str(tasks_dir),
            "--client.base-url",
            "http://127.0.0.1:8080/v1",
            "--client.api-key-var",
            "BENCH_KEY",
        ]
        self.assertEqual(first[:14], expected_head)
        self.assertEqual(first[14:16], ["--num-rollouts", "2"])
        self.assertEqual(first[16:18], ["--max-concurrent", "3"])
        self.assertEqual(first[18:20], ["--sampling.max-tokens", "2048"])
        self.assertEqual(first[20:22], ["--env.agent.max-total-tokens", "50000"])
        self.assertEqual(first[22], "--output-dir")
        self.assertTrue(first[23].endswith("org_a"))
        self.assertEqual(commands[1][3], "b")

    def test_run_models_omits_client_flags_when_unset(self) -> None:
        commands: list[list[str]] = []

        def runner(command: Sequence[str]) -> int:
            commands.append(list(command))
            return 3

        with tempfile.TemporaryDirectory() as temporary:
            result = run_models(
                ["m"],
                tasks_dir=None,
                split="validation",
                base_url=None,
                api_key_var=None,
                num_rollouts=1,
                max_concurrent=1,
                max_tokens=1,
                max_total_tokens=1,
                output_dir=Path(temporary) / "bench",
                runner=runner,
            )

        self.assertIsInstance(result, BenchFailure)
        assert isinstance(result, BenchFailure)
        self.assertIn("status 3", result.reason)
        self.assertNotIn("--client.base-url", commands[0])
        self.assertNotIn("--env.taskset.tasks-dir", commands[0])
        self.assertEqual(commands[0][0], "eval")

    def test_run_models_rejects_duplicates_and_existing_run_dirs(self) -> None:
        calls = 0

        def runner(command: Sequence[str]) -> int:
            nonlocal calls
            calls += 1
            return 0

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bench"
            duplicate = run_models(
                ["m", "m"],
                tasks_dir=None,
                split="holdout",
                base_url=None,
                api_key_var=None,
                num_rollouts=1,
                max_concurrent=1,
                max_tokens=1,
                max_total_tokens=1,
                output_dir=output,
                runner=runner,
            )
            (output / "runs" / "m").mkdir(parents=True)
            existing = run_models(
                ["m"],
                tasks_dir=None,
                split="holdout",
                base_url=None,
                api_key_var=None,
                num_rollouts=1,
                max_concurrent=1,
                max_tokens=1,
                max_total_tokens=1,
                output_dir=output,
                runner=runner,
            )

        self.assertIsInstance(duplicate, BenchFailure)
        self.assertIsInstance(existing, BenchFailure)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
