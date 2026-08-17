"""Tests for the harness-enforced improve budget.

The tests cover three contracts: `_entries_since` counts only the
entries appended after the recorded start, the operator process
group dies when the ledger exceeds the verb budget, and a clean
operator exit records a normal ledger entry.
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
    shrink the poll interval, then run in a temp working directory
    so experiments/ artifacts never touch the repository.
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
        self._stack.enter_context(
            mock.patch.object(improve, "_POLL_SECONDS", 0.1)
        )

    def _fake_operator(self, argv: list[str]) -> None:
        self._stack.enter_context(
            mock.patch.object(
                improve,
                "_operator_command",
                lambda prompt, max_time: argv,
            )
        )

    def test_budget_excess_kills_the_operator_group(self) -> None:
        self._fake_operator(["/bin/sleep", "60"])
        # The fake verbs land after run_improve records its start
        # count, so the poll sees two new entries against a budget
        # of one.
        pusher = threading.Timer(
            0.3, _append_fake_verbs, args=(self.ledger, 2)
        )
        pusher.start()
        self.addCleanup(pusher.cancel)

        result = improve.run_improve(
            goal="test", budget=1, max_time=600, ledger_path=self.ledger
        )

        # The operator died from the kill, not from its own 60
        # second sleep, and well within one real poll interval.
        self.assertLess(result.duration_seconds, 15.0)
        self.assertLess(result.exit_code, 0)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertEqual(record.kind, "improve")
        self.assertIs(record.metrics["budget_exceeded"], True)
        self.assertEqual(record.metrics["verbs_recorded"], 2)
        self.assertIs(record.metrics["timed_out"], False)

    def test_deadline_expiry_kills_and_marks_timed_out(self) -> None:
        self._fake_operator(["/bin/sleep", "60"])
        self._stack.enter_context(
            mock.patch.object(improve, "_TIMEOUT_GRACE_SECONDS", 0.0)
        )

        result = improve.run_improve(
            goal="test", budget=5, max_time=0, ledger_path=self.ledger
        )

        self.assertEqual(result.exit_code, -1)
        self.assertLess(result.duration_seconds, 15.0)
        entries, _ = read_ledger(self.ledger)
        record = entries[-1]
        self.assertIs(record.metrics["timed_out"], True)
        self.assertIs(record.metrics["budget_exceeded"], False)

    def test_clean_exit_records_output_and_no_flags(self) -> None:
        self._fake_operator(["/bin/echo", "operator ran"])

        result = improve.run_improve(
            goal="test", budget=5, max_time=600, ledger_path=self.ledger
        )

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
