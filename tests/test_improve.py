"""Tests for the harness-enforced improve budget.

The tests cover the kill contracts: `_entries_since` counts only
the entries appended after the recorded start, the operator
process group dies when the ledger reaches the verb budget (at
the limit, not past it), the wall-clock cap kills on its own, and
a clean operator exit records a normal ledger entry with the
limit configuration.
"""

import os
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from omp_gym import improve
from omp_gym.ledger import append_entry, read_ledger


def _append_fake_verbs(ledger_path: Path, count: int) -> None:
    for index in range(count):
        append_entry(
            ledger_path,
            kind="run",
            config={"fake": index},
            metrics={},
            artifacts={},
        )


class EntriesSinceTests(unittest.TestCase):
    def test_missing_ledger_counts_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "ledger.jsonl"
            self.assertEqual(improve._entries_since(missing, 0), 0)

    def test_counts_only_entries_past_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            _append_fake_verbs(ledger, 3)
            self.assertEqual(improve._entries_since(ledger, 0), 3)
            self.assertEqual(improve._entries_since(ledger, 1), 2)
            self.assertEqual(improve._entries_since(ledger, 3), 0)

    def test_never_negative_when_ledger_shrinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            _append_fake_verbs(ledger, 1)
            self.assertEqual(improve._entries_since(ledger, 5), 0)


class RunImproveTests(unittest.TestCase):
    """Drive run_improve with a fake operator command.

    The tests replace the omp invocation with plain binaries and
    pass a short poll interval, then run in a temp working
    directory so experiments/ artifacts never touch the repository.
    """

    def setUp(self) -> None:
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        tmp = self._stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(tmp)
        self.ledger = self.root / "ledger.jsonl"
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def _fake_operator(self, argv: list[str]) -> None:
        self._stack.enter_context(
            mock.patch.object(
                improve,
                "_operator_command",
                lambda prompt, max_time, model=None: argv,
            )
        )

    def _run(self, **overrides: object) -> improve.ImproveResult:
        options: dict[str, object] = {
            "goal": "test",
            "budget": 5,
            "max_time": 600,
            "ledger_path": self.ledger,
            "poll_seconds": 0.1,
        }
        options.update(overrides)
        return improve.run_improve(**options)  # type: ignore[arg-type]

    def test_budget_at_limit_kills_the_operator_group(self) -> None:
        self._fake_operator(["/bin/sleep", "60"])
        # Two new entries against a budget of two: the kill fires
        # at the limit, not one past it.
        pusher = threading.Timer(0.3, _append_fake_verbs, args=(self.ledger, 2))
        pusher.start()
        self.addCleanup(pusher.cancel)

        result = self._run(budget=2)

        self.assertLess(result.duration_seconds, 15.0)
        self.assertLess(result.exit_code, 0)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertEqual(record.kind, "improve")
        self.assertIs(record.metrics["budget_exceeded"], True)
        self.assertEqual(record.metrics["verbs_recorded"], 2)
        self.assertIs(record.metrics["timed_out"], False)

    def test_below_budget_lets_the_operator_finish(self) -> None:
        # The operator lives through several polls; one entry below
        # the budget never triggers the kill.
        self._fake_operator(["/bin/sleep", "0.6"])
        pusher = threading.Timer(0.2, _append_fake_verbs, args=(self.ledger, 1))
        pusher.start()
        self.addCleanup(pusher.cancel)

        result = self._run(budget=2)

        self.assertEqual(result.exit_code, 0)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertIs(record.metrics["budget_exceeded"], False)
        self.assertEqual(record.metrics["verbs_recorded"], 1)

    def test_deadline_expiry_kills_and_marks_timed_out(self) -> None:
        self._fake_operator(["/bin/sleep", "60"])
        self._stack.enter_context(
            mock.patch.object(improve, "_TIMEOUT_GRACE_SECONDS", 0.0)
        )

        result = self._run(max_time=0)

        self.assertEqual(result.exit_code, -1)
        self.assertLess(result.duration_seconds, 15.0)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertIs(record.metrics["timed_out"], True)
        self.assertIs(record.metrics["budget_exceeded"], False)

    def test_max_seconds_tightens_the_wall_clock(self) -> None:
        self._fake_operator(["/bin/sleep", "60"])

        # max_time alone would allow 600+120 seconds; the cap cuts
        # the session at one second.
        result = self._run(max_seconds=1)

        self.assertEqual(result.exit_code, -1)
        self.assertLess(result.duration_seconds, 15.0)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertIs(record.metrics["timed_out"], True)

    def test_limits_are_recorded_in_the_entry_config(self) -> None:
        self._fake_operator(["/bin/echo", "operator ran"])

        self._run(budget=3, max_time=42, max_seconds=90)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertEqual(record.config["budget"], 3)
        self.assertEqual(record.config["max_time"], 42)
        self.assertEqual(record.config["poll_seconds"], 0.1)
        self.assertEqual(record.config["max_seconds"], 90)
        self.assertIn("model", record.config)

    def test_max_seconds_defaults_to_none(self) -> None:
        self._fake_operator(["/bin/echo", "operator ran"])

        self._run()
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertIsNone(record.config["max_seconds"])

    def test_clean_exit_records_output_and_no_flags(self) -> None:
        self._fake_operator(["/bin/echo", "operator ran"])

        result = self._run()

        self.assertEqual(result.exit_code, 0)
        events = Path(result.work_dir) / "events.jsonl"
        self.assertIn("operator ran", events.read_text())
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertIs(record.metrics["budget_exceeded"], False)
        self.assertIs(record.metrics["timed_out"], False)
        self.assertEqual(record.metrics["verbs_recorded"], 0)


if __name__ == "__main__":
    unittest.main()
