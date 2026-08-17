import unittest
from dataclasses import asdict

from omp_gym.bench import BenchRow, render_report, wilson_interval
from omp_gym.ledger import LedgerEntry
from omp_gym.report import _model_stats
from omp_gym.report import render_report as render_ledger_report


def _row(
    reward: float,
    trial: int,
    error: str | None = None,
    model: str = "m1",
    task: str = "t1",
) -> BenchRow:
    """Build one bench row with fixed usage numbers."""
    return BenchRow(
        model=model,
        task=task,
        trial=trial,
        reward=reward,
        duration_seconds=2.0,
        total_tokens=100,
        cost_usd=0.01,
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
        self.assertIn("## Provider errors", report)
        self.assertIn("- m1 / t1: provider exploded", report)

    def test_all_clean_rows(self) -> None:
        rows = [_row(1.0, 1), _row(1.0, 2), _row(1.0, 3)]
        report = render_report(rows)
        self.assertIn(f"| 1 | m1 | {_rate_cell(3, 3)} | 0 |", report)
        self.assertNotIn("## Provider errors", report)

    def test_all_error_rows(self) -> None:
        rows = [
            _row(0.0, 1, error="timeout"),
            _row(0.0, 2, error="timeout"),
        ]
        report = render_report(rows)
        self.assertIn(
            f"| 1 | m1 | {_rate_cell(0, 2)} | 2 | - | - | - | - |",
            report,
        )
        self.assertIn("| t1 | E |", report)
        self.assertIn("## Provider errors", report)

    def test_sorts_by_lower_bound_and_suppresses_overlapping_ranks(
        self,
    ) -> None:
        rows = (
            _grid("clear", 30, 30)
            + _grid("mid-a", 15, 30)
            + _grid("mid-b", 14, 30)
        )
        report = render_report(rows)
        table = report.splitlines()

        def leaderboard_row(model: str) -> tuple[int, str]:
            for number, line in enumerate(table):
                if f"| {model} | " in line:
                    return number, line.split("|")[1].strip()
            raise AssertionError(f"no leaderboard row for {model}")

        clear_at, clear_rank = leaderboard_row("clear")
        mid_a_at, mid_a_rank = leaderboard_row("mid-a")
        mid_b_at, mid_b_rank = leaderboard_row("mid-b")
        self.assertEqual(clear_rank, "1")
        self.assertEqual(mid_a_rank, "=")
        self.assertEqual(mid_b_rank, "=")
        self.assertLess(clear_at, mid_a_at)
        self.assertLess(mid_a_at, mid_b_at)



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
        self.assertEqual(stats[0].passes, 1)
        self.assertAlmostEqual(stats[0].total_cost, 0.03)

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
        entry = _bench_entry(
            _grid("solid", 9, 10) + _grid("lucky", 1, 1)
        )
        stats = _model_stats([entry])
        by_model = {stat.model: stat for stat in stats}
        low, high = wilson_interval(9, 10)
        self.assertAlmostEqual(by_model["solid"].low, low)
        self.assertAlmostEqual(by_model["solid"].high, high)
        self.assertEqual(
            [stat.model for stat in stats], ["solid", "lucky"]
        )


class LedgerReportTest(unittest.TestCase):
    def test_models_table_carries_the_interval(self) -> None:
        entry = _bench_entry([_row(1.0, 1), _row(0.0, 2)])
        report = render_ledger_report([entry])
        low, high = wilson_interval(1, 2)
        self.assertIn(
            f"50% [{low:.0%}, {high:.0%}] (1/2)", report
        )


if __name__ == "__main__":
    unittest.main()
