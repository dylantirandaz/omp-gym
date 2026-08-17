import json
import platform
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from omp_gym import bench
from omp_gym.bench import (
    BenchRow,
    bootstrap_pass_rate_ci,
    classify_failure,
    load_task_suite,
    render_report,
    run_benchmark,
    summarize_report,
    wilson_interval,
    write_manifest,
)
from omp_gym.ledger import LedgerEntry
from omp_gym.report import _model_stats
from omp_gym.report import render_report as render_ledger_report
from omp_gym.runner import EpisodeFailure, EpisodeRecord
from omp_gym.task import TaskSpec, workspace_digest


def _row(
    reward: float,
    trial: int,
    error: str | None = None,
    model: str = "m1",
    task: str = "t1",
    duration: float = 2.0,
    tokens: int = 100,
    cost: float = 0.01,
) -> BenchRow:
    """Build one bench row with fixed usage numbers."""
    return BenchRow(
        model=model,
        task=task,
        trial=trial,
        reward=reward,
        duration_seconds=duration,
        total_tokens=tokens,
        cost_usd=cost,
        tool_calls=4,
        error=error,
    )


def _bench_entry(rows: list[BenchRow]) -> LedgerEntry:
    """Wrap bench rows in one ledger entry."""
    return LedgerEntry(
        kind="bench",
        timestamp="2026-08-16T00:00:00+0000",
        config={},
        metrics={"rows": [asdict(row) for row in rows]},
        artifacts={},
    )


def _grid(model: str, passes: int, runs: int) -> list[BenchRow]:
    """One model's rows: the first `passes` trials succeed."""
    return [
        _row(1.0 if trial <= passes else 0.0, trial, model=model)
        for trial in range(1, runs + 1)
    ]


def _rate_cell(passes: int, runs: int) -> str:
    """The pass-rate cell render_report emits for passes/runs."""
    low, high = wilson_interval(passes, runs)
    return f"{passes / runs:.0%} [{low:.0%}, {high:.0%}] (n={runs})"


def _write_task(task_dir: Path) -> None:
    """Create one minimal valid task directory."""
    task_dir.mkdir(parents=True)
    (task_dir / "workspace").mkdir()
    (task_dir / "workspace" / "app.py").write_text("pass\n")
    (task_dir / "task.toml").write_text(
        'prompt = "Fix app.py."\n'
        'test_command = ["python3", "test_app.py"]\n'
        'max_time = "60"\n'
    )


def _fake_record(
    tmp: Path, task: TaskSpec, model: str, reward: float = 1.0
) -> EpisodeRecord:
    """One false episode record with an empty session file."""
    session = tmp / "session.jsonl"
    session.write_text("")
    episodes = tmp / "episodes"
    episodes.mkdir(exist_ok=True)
    return EpisodeRecord(
        task=task.name,
        model=model,
        episode_dir=str(episodes),
        session_file=str(session),
        omp_exit_code=0,
        test_exit_code=0,
        reward=reward,
        reward_partial=None,
        duration_seconds=1.5,
    )


class WilsonIntervalTest(unittest.TestCase):
    def test_zero_runs_is_fully_unknown(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_perfect_small_sample_keeps_uncertainty(self) -> None:
        low, high = wilson_interval(3, 3)
        self.assertGreater(low, 0.36)
        self.assertLess(low, 0.5)
        self.assertAlmostEqual(high, 1.0)

    def test_one_of_three_spot_values(self) -> None:
        low, high = wilson_interval(1, 3)
        self.assertAlmostEqual(low, 0.0615, places=3)
        self.assertAlmostEqual(high, 0.7924, places=3)

    def test_zero_passes_keep_a_zero_lower_bound(self) -> None:
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.31)


class ClassifyFailureTest(unittest.TestCase):
    def test_buckets(self) -> None:
        cases = {
            "baseline test run exceeded the 120s deadline": "baseline_timeout",
            "test command exceeded the 120s deadline": "test_timeout",
            "omp exited with 3 and wrote no session": "no_session",
            "task already passes before the agent runs": "invalid_task",
            "sandbox-exec refused the profile": "sandbox",
            "seatbelt profile failed": "sandbox",
            "isolation setup failed": "sandbox",
            "provider reset the stream": "provider_error",
            "rate limit hit": "provider_error",
            "something unexplained": "other",
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(classify_failure(reason), expected)

    def test_first_match_wins_in_priority_order(self) -> None:
        self.assertEqual(
            classify_failure("sandbox refused: baseline deadline"),
            "sandbox",
        )

    def test_all_classes_are_reachable(self) -> None:
        reachable = {
            classify_failure(reason)
            for reason in (
                "provider reset",
                "baseline deadline hit",
                "test deadline hit",
                "wrote no session",
                "task already passes",
                "sandbox denied",
                "???",
            )
        }
        self.assertEqual(reachable - {"other"}, set(bench.FAILURE_CLASSES) - {"other"})
        self.assertIn("other", reachable)


class RunBenchmarkOrderTest(unittest.TestCase):
    """Seeded interleaved scheduling with recorded order positions."""

    def setUp(self) -> None:
        self._stack = mock.patch.object(bench, "run_episode", autospec=True)
        self.fake_run = self._stack.start()
        self.addCleanup(self._stack.stop)
        self.tmp_root = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_root.cleanup)
        self.tmp = Path(self.tmp_root.name)
        self.fake_run.side_effect = lambda task, runs_dir, model: _fake_record(
            self.tmp, task, model
        )

    def _tasks(self, names: list[str]) -> list[TaskSpec]:
        return [
            TaskSpec(
                name=name,
                prompt="p",
                test_command=("python3", "t.py"),
                tools="bash",
                max_time="60",
                workspace=self.tmp,
            )
            for name in names
        ]

    def test_round_robin_blocks_then_seed_shuffle(self) -> None:
        tasks = self._tasks(["alpha", "beta"])
        rows = run_benchmark(["m1", "m2"], tasks, trials=2, runs_dir=self.tmp, seed=7)
        self.assertEqual(len(rows), 8)
        # Every schedule position is recorded exactly once.
        self.assertEqual(sorted(row.order_index for row in rows), list(range(8)))
        combos = {(row.model, row.task, row.trial) for row in rows}
        expected = {
            (model, task, trial)
            for model in ("m1", "m2")
            for task in ("alpha", "beta")
            for trial in (1, 2)
        }
        self.assertEqual(combos, expected)
        # Each trial block holds every model x task pair exactly
        # once: the round-robin interleave survives the shuffle.
        for trial in (1, 2):
            block = [(row.model, row.task) for row in rows if row.trial == trial]
            self.assertEqual(len(block), 4)
            self.assertEqual(
                set(block),
                {(model, task) for model in ("m1", "m2") for task in ("alpha", "beta")},
            )

    def test_seed_makes_order_deterministic(self) -> None:
        tasks = self._tasks(["alpha", "beta"])
        first = run_benchmark(["m1", "m2"], tasks, 2, self.tmp, seed=7)
        second = run_benchmark(["m1", "m2"], tasks, 2, self.tmp, seed=7)
        self.assertEqual(
            [(r.model, r.task, r.trial) for r in first],
            [(r.model, r.task, r.trial) for r in second],
        )

    def test_failure_rows_are_classified_and_recorded(self) -> None:
        tasks = self._tasks(["alpha"])
        self.fake_run.side_effect = lambda task, runs_dir, model: EpisodeFailure(
            task=task.name, reason="wrote no session"
        )
        rows = run_benchmark(["m1"], tasks, 2, self.tmp, seed=0)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row.error, "wrote no session")
            self.assertEqual(row.error_class, "no_session")
            self.assertIsNotNone(row.order_index)


class LoadTaskSuiteTest(unittest.TestCase):
    def test_only_immediate_children_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task(root / "top-task")
            _write_task(root / "nested" / "pool-task")
            suite = load_task_suite(root)
        self.assertIsInstance(suite, list)
        self.assertEqual([task.name for task in suite], ["top-task"])

    def test_recursive_opt_in_uses_relative_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task(root / "top-task")
            _write_task(root / "nested" / "pool-task")
            suite = load_task_suite(root, recursive=True)
        self.assertIsInstance(suite, list)
        self.assertEqual(
            [task.name for task in suite], ["nested/pool-task", "top-task"]
        )

    def test_default_excludes_minted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task(root / "top-task")
            _write_task(root / "minted" / "pool-task")
            suite = load_task_suite(root, recursive=True)
        self.assertIsInstance(suite, list)
        self.assertEqual([task.name for task in suite], ["top-task"])

    def test_include_minted_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task(root / "top-task")
            _write_task(root / "minted" / "pool-task")
            suite = load_task_suite(root, recursive=True, include_minted=True)
        self.assertIsInstance(suite, list)
        self.assertEqual(
            [task.name for task in suite],
            ["minted/pool-task", "top-task"],
        )

    def test_relative_ids_never_collide_on_leaf_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task(root / "pool-a" / "same-leaf")
            _write_task(root / "pool-b" / "same-leaf")
            suite = load_task_suite(root, recursive=True)
        self.assertIsInstance(suite, list)
        names = [task.name for task in suite]
        self.assertEqual(names, ["pool-a/same-leaf", "pool-b/same-leaf"])
        self.assertEqual(len(set(names)), 2)

    def test_first_bad_task_is_the_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "broken"
            broken.mkdir()
            # task.toml without a workspace/ fails to load.
            (broken / "task.toml").write_text(
                'prompt = "Fix app.py."\n'
                'test_command = ["python3", "test_app.py"]\n'
                'max_time = "60"\n'
            )
            suite = load_task_suite(root)
        self.assertNotIsInstance(suite, list)
        self.assertEqual(suite.path, broken)


class ManifestTest(unittest.TestCase):
    def test_manifest_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_task(root / "top-task")
            _write_task(root / "nested" / "pool-task")
            suite = load_task_suite(root, recursive=True)
            assert isinstance(suite, list)
            report = root / "bench-report.md"
            report.write_text("# bench\n")
            path = write_manifest(report, suite, root=root)

            self.assertEqual(path.name, "bench-manifest.json")
            self.assertEqual(path.parent, report.parent)
            payload = json.loads(path.read_text())
            self.assertIn("git_sha", payload)
            self.assertTrue(
                payload["git_sha"] is None or isinstance(payload["git_sha"], str)
            )
            self.assertEqual(payload["python"], platform.python_version())
            self.assertIn("platform", payload)
            self.assertIsInstance(payload["packages"], dict)
            by_name = {t["name"]: t for t in payload["tasks"]}
            self.assertEqual(set(by_name), {"top-task", "nested/pool-task"})
            for task in suite:
                self.assertEqual(
                    by_name[task.name]["workspace_digest"],
                    workspace_digest(task.workspace),
                )

    def test_manifest_tolerates_git_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = bench.bench_manifest([], root=Path(tmp))
        self.assertIn("git_sha", payload)


class RenderReportTest(unittest.TestCase):
    def test_error_rows_stay_in_denominator(self) -> None:
        rows = [
            _row(1.0, 1),
            _row(0.0, 2),
            _row(0.0, 3, error="provider exploded"),
        ]
        report = render_report(rows)
        self.assertIn(f"| 1 | m1 | {_rate_cell(1, 3)} | 1 |", report)
        self.assertNotIn("50%", report)
        self.assertIn("## Failures by class", report)
        self.assertIn("- provider_error: 1", report)
        self.assertIn("- m1 / t1 [provider_error]: provider exploded", report)

    def test_all_clean_rows(self) -> None:
        rows = [_row(1.0, 1), _row(1.0, 2), _row(1.0, 3)]
        report = render_report(rows)
        self.assertIn(f"| 1 | m1 | {_rate_cell(3, 3)} | 0 |", report)
        self.assertNotIn("## Failures by class", report)

    def test_all_rows_and_clean_aggregates_are_both_labeled(self) -> None:
        # The error row has duration 8 while clean rows have 2:
        # the all-rows mean (4.0) differs from the clean mean (2.0),
        # so swapping them would change the rendered numbers.
        rows = [
            _row(1.0, 1, duration=2.0, tokens=100),
            _row(1.0, 2, duration=2.0, tokens=100),
            _row(0.0, 3, error="wrote no session", duration=8.0, tokens=700),
        ]
        report = render_report(rows)
        header = next(
            line for line in report.splitlines() if line.startswith("| rank |")
        )
        self.assertIn("mean s (all)", header)
        self.assertIn("mean s (clean)", header)
        self.assertIn("mean tokens (all)", header)
        self.assertIn("mean tokens (clean)", header)
        body = next(
            line for line in report.splitlines() if line.startswith("| 1 | m1 |")
        )
        cells = [cell.strip() for cell in body.split("|")]
        # cells[0] is empty; rank, model, rate, errors, then the
        # four aggregate pairs in header order.
        self.assertEqual(cells[5], "4.0")  # mean s (all)
        self.assertEqual(cells[6], "2.0")  # mean s (clean)
        self.assertEqual(cells[7], "300")  # mean tokens (all)
        self.assertEqual(cells[8], "100")  # mean tokens (clean)

    def test_all_error_rows(self) -> None:
        rows = [
            _row(0.0, 1, error="timeout"),
            _row(0.0, 2, error="timeout"),
        ]
        report = render_report(rows)
        self.assertIn(f"| 1 | m1 | {_rate_cell(0, 2)} | 2 |", report)
        self.assertIn("| t1 | 0/2 |", report)
        self.assertIn("## Failures by class", report)

    def test_task_matrix_uses_scheduled_denominators(self) -> None:
        rows = [
            _row(1.0, 1),
            _row(0.0, 2),
            _row(0.0, 3, error="provider exploded"),
        ]
        report = render_report(rows)
        self.assertIn("| t1 | 1/3 |", report)
        self.assertNotIn("+E", report)

    def test_failure_classes_are_counted_per_class(self) -> None:
        rows = [
            _row(0.0, 1, error="wrote no session"),
            _row(0.0, 2, error="wrote no session"),
            _row(0.0, 3, error="baseline test run exceeded the 120s deadline"),
            _row(0.0, 4, error="episode blew up mysteriously"),
        ]
        report = render_report(rows)
        self.assertIn("- no_session: 2", report)
        self.assertIn("- baseline_timeout: 1", report)
        self.assertIn("- other: 1", report)

    def test_tagged_error_class_wins_over_classifier(self) -> None:
        row = BenchRow(
            model="m1",
            task="t1",
            trial=1,
            reward=0.0,
            duration_seconds=1.0,
            total_tokens=10,
            cost_usd=0.0,
            tool_calls=0,
            error="boom",
            error_class="sandbox",
        )
        report = render_report([row])
        self.assertIn("- sandbox: 1", report)

    def test_connected_component_ties_are_transitive(self) -> None:
        # A overlaps B, B overlaps C, A does not overlap C. Only
        # the transitive closure of overlap groups all three.
        runs = 30
        rows = (
            _grid("first", 20, runs)
            + _grid("bridge", 15, runs)
            + _grid("last", 9, runs)
        )
        first = wilson_interval(20, runs)
        bridge = wilson_interval(15, runs)
        last = wilson_interval(9, runs)
        # Precondition: the chain holds and the ends are separate.
        self.assertLessEqual(bridge[0], first[1])
        self.assertLessEqual(last[0], bridge[1])
        self.assertGreater(first[0], last[1])

        report = render_report(rows)
        for model in ("first", "bridge", "last"):
            line = next(line for line in report.splitlines() if f"| {model} |" in line)
            self.assertEqual(line.split("|")[1].strip(), "=", model)

    def test_separated_models_get_numeric_ranks(self) -> None:
        rows = _grid("clear", 30, 30) + _grid("behind", 10, 30)
        report = render_report(rows)
        table = report.splitlines()

        def leaderboard_rank(model: str) -> str:
            for line in table:
                if f"| {model} |" in line:
                    return line.split("|")[1].strip()
            raise AssertionError(f"no leaderboard row for {model}")

        self.assertEqual(leaderboard_rank("clear"), "1")
        self.assertEqual(leaderboard_rank("behind"), "2")

    def test_ranking_method_is_documented(self) -> None:
        report = render_report([_row(1.0, 1)])
        self.assertIn("## Ranking method", report)
        self.assertIn("heuristic", report)
        self.assertIn("transitive", report)

    def test_bootstrap_ci_line_rendered(self) -> None:
        report = render_report(_grid("m1", 5, 10))
        self.assertIn("task-level bootstrap", report)


class BootstrapCITest(unittest.TestCase):
    def test_shape_and_bounds(self) -> None:
        rows = [_row(1.0, 1, task="a"), _row(0.0, 2, task="a")] + [
            _row(1.0, 1, task="b"),
            _row(1.0, 2, task="b"),
        ]
        ci = bootstrap_pass_rate_ci(rows)
        assert ci is not None
        low, high = ci
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(low, high)
        self.assertLessEqual(high, 1.0)

    def test_seeded_resampling_is_deterministic(self) -> None:
        rows = _grid("m1", 5, 10)
        self.assertEqual(
            bootstrap_pass_rate_ci(rows, seed=3),
            bootstrap_pass_rate_ci(rows, seed=3),
        )

    def test_all_passes_ci_at_one(self) -> None:
        rows = _grid("m1", 10, 10)
        self.assertEqual(bootstrap_pass_rate_ci(rows), (1.0, 1.0))

    def test_empty_rows_have_no_ci(self) -> None:
        self.assertIsNone(bootstrap_pass_rate_ci([]))

    def test_summary_records_ci_parameters(self) -> None:
        summary = summarize_report(_grid("m1", 5, 10), seed=7)
        ci = summary["bootstrap_ci"]
        self.assertIsInstance(ci, dict)
        assert isinstance(ci, dict)
        self.assertEqual(ci["resamples"], 2000)
        self.assertEqual(ci["seed"], 7)
        self.assertLessEqual(ci["low"], ci["high"])
        self.assertEqual(summary["rows"], 10)
        self.assertEqual(summary["passes"], 5)


class SummarizeReportTest(unittest.TestCase):
    def test_all_and_clean_aggregates(self) -> None:
        rows = [
            _row(1.0, 1, duration=2.0, tokens=100, cost=0.02),
            _row(0.0, 2, error="boom", duration=4.0, tokens=200, cost=0.04),
        ]
        summary = summarize_report(rows)
        model = summary["models"]["m1"]
        self.assertEqual(model["runs"], 2)
        self.assertAlmostEqual(model["mean_seconds_all"], 3.0)
        self.assertAlmostEqual(model["mean_seconds_clean"], 2.0)
        self.assertAlmostEqual(model["mean_tokens_all"], 150.0)
        self.assertAlmostEqual(model["mean_tokens_clean"], 100.0)
        self.assertAlmostEqual(model["cost_usd_all"], 0.06)
        self.assertAlmostEqual(model["cost_usd_clean"], 0.02)

    def test_failure_classes_counted(self) -> None:
        rows = [
            _row(0.0, 1, error="wrote no session"),
            _row(0.0, 2, error="sandbox refused"),
        ]
        summary = summarize_report(rows)
        self.assertEqual(summary["failure_classes"], {"no_session": 1, "sandbox": 1})


class RowDurationTest(unittest.TestCase):
    """Wall duration preference for record-backed rows."""

    def _record_with_wall(
        self, tmp: Path, wall: float | None, duration: float
    ) -> EpisodeRecord:
        session = tmp / "session.jsonl"
        session.write_text("")
        record = EpisodeRecord(
            task="t1",
            model="m1",
            episode_dir=str(tmp),
            session_file=str(session),
            omp_exit_code=0,
            test_exit_code=0,
            reward=1.0,
            reward_partial=None,
            duration_seconds=duration,
        )
        if wall is not None:
            # Simulate the newer record shape without depending on
            # runner internals: the getter prefers wall_seconds.
            object.__setattr__(record, "wall_seconds", wall)
        return record

    def test_wall_seconds_preferred_over_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = self._record_with_wall(Path(tmp), wall=9.0, duration=2.0)
            row = bench._row_from_record(record, trial=1)
        self.assertEqual(row.duration_seconds, 9.0)

    def test_duration_fallback_for_records_without_wall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = self._record_with_wall(Path(tmp), wall=None, duration=2.0)
            row = bench._row_from_record(record, trial=1)
        self.assertEqual(row.duration_seconds, 2.0)


class ModelStatsTest(unittest.TestCase):
    def test_runs_count_every_row(self) -> None:
        entry = _bench_entry(
            [
                _row(1.0, 1),
                _row(0.0, 2),
                _row(0.0, 3, error="provider exploded"),
            ]
        )
        stats = _model_stats([entry])
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].runs, 3)
        self.assertEqual(stats[0].clean_runs, 2)
        self.assertEqual(stats[0].passes, 1)
        self.assertAlmostEqual(stats[0].total_cost, 0.03)

    def test_all_rows_and_clean_means(self) -> None:
        entry = _bench_entry(
            [
                _row(1.0, 1, duration=2.0, tokens=100),
                _row(0.0, 2, duration=4.0, tokens=300),
                _row(0.0, 3, error="boom", duration=8.0, tokens=500),
            ]
        )
        stat = _model_stats([entry])[0]
        self.assertAlmostEqual(stat.mean_tokens, 300.0)
        self.assertAlmostEqual(stat.mean_tokens_clean, 200.0)
        self.assertAlmostEqual(stat.mean_seconds_all, 14.0 / 3)
        self.assertAlmostEqual(stat.mean_seconds_clean, 3.0)

    def test_wall_seconds_preferred_when_present(self) -> None:
        row = asdict(_row(1.0, 1, duration=2.0))
        row["wall_seconds"] = 9.5
        entry = LedgerEntry(
            kind="bench",
            timestamp="2026-08-16T00:00:00+0000",
            config={},
            metrics={"rows": [row]},
            artifacts={},
        )
        stat = _model_stats([entry])[0]
        self.assertAlmostEqual(stat.mean_seconds_all, 9.5)

    def test_all_clean_rows(self) -> None:
        entry = _bench_entry([_row(1.0, 1), _row(1.0, 2), _row(1.0, 3)])
        stats = _model_stats([entry])
        self.assertEqual(stats[0].runs, 3)
        self.assertEqual(stats[0].passes, 3)
        self.assertIsNotNone(stats[0].cost_per_pass)

    def test_all_error_rows(self) -> None:
        entry = _bench_entry(
            [
                _row(0.0, 1, error="timeout"),
                _row(0.0, 2, error="timeout"),
            ]
        )
        stats = _model_stats([entry])
        self.assertEqual(stats[0].runs, 2)
        self.assertEqual(stats[0].passes, 0)
        self.assertIsNone(stats[0].cost_per_pass)
        self.assertIsNone(stats[0].tokens_per_solve)

    def test_interval_fields_and_lower_bound_sort(self) -> None:
        entry = _bench_entry(_grid("solid", 9, 10) + _grid("lucky", 1, 1))
        stats = _model_stats([entry])
        by_model = {stat.model: stat for stat in stats}
        low, high = wilson_interval(9, 10)
        self.assertAlmostEqual(by_model["solid"].low, low)
        self.assertAlmostEqual(by_model["solid"].high, high)
        self.assertEqual([stat.model for stat in stats], ["solid", "lucky"])


class LedgerReportTest(unittest.TestCase):
    def test_models_table_carries_the_interval(self) -> None:
        entry = _bench_entry([_row(1.0, 1), _row(0.0, 2)])
        report = render_ledger_report([entry])
        low, high = wilson_interval(1, 2)
        self.assertIn(f"50% [{low:.0%}, {high:.0%}] (1/2)", report)

    def test_models_table_labels_all_and_clean_columns(self) -> None:
        entry = _bench_entry([_row(1.0, 1), _row(0.0, 2, error="boom")])
        report = render_ledger_report([entry])
        self.assertIn("mean tokens (all)", report)
        self.assertIn("mean tokens (clean)", report)
        self.assertIn("mean seconds (all)", report)
        self.assertIn("mean seconds (clean)", report)


if __name__ == "__main__":
    unittest.main()
