import unittest
from dataclasses import asdict

from omp_gym.bench import BenchRow, render_report
from omp_gym.ledger import LedgerEntry
from omp_gym.report import _model_stats


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


class RenderReportTest(unittest.TestCase):
    def test_error_rows_stay_in_denominator(self) -> None:
        rows = [
            _row(1.0, 1),
            _row(0.0, 2),
            _row(0.0, 3, error="provider exploded"),
        ]
        report = render_report(rows)
        self.assertIn("| m1 | 33% (3 runs) | 1 |", report)
        self.assertNotIn("50%", report)
        self.assertIn("## Provider errors", report)
        self.assertIn("- m1 / t1: provider exploded", report)

    def test_all_clean_rows(self) -> None:
        rows = [_row(1.0, 1), _row(1.0, 2), _row(1.0, 3)]
        report = render_report(rows)
        self.assertIn("| m1 | 100% (3 runs) | 0 |", report)
        self.assertNotIn("## Provider errors", report)

    def test_all_error_rows(self) -> None:
        rows = [
            _row(0.0, 1, error="timeout"),
            _row(0.0, 2, error="timeout"),
        ]
        report = render_report(rows)
        self.assertIn("| m1 | 0% (2 runs) | 2 | - | - | - | - |", report)
        self.assertIn("| t1 | E |", report)
        self.assertIn("## Provider errors", report)


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


if __name__ == "__main__":
    unittest.main()
