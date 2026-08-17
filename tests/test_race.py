"""Concurrency contract for the experiment ledger.

Concurrent appenders must not interleave partial lines: every writer
appends one JSON line per call, and readers see exactly that many
parseable entries with zero malformed lines.
"""

import tempfile
import threading
import unittest
from pathlib import Path

from omp_gym.ledger import append_entry, read_ledger

THREADS = 10
APPENDS_PER_THREAD = 50


class LedgerRaceTests(unittest.TestCase):
    def test_concurrent_appends_stay_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger_path = Path(temporary_directory) / "ledger.jsonl"
            barrier = threading.Barrier(THREADS)
            errors: list[BaseException] = []

            def writer(thread_index: int) -> None:
                try:
                    barrier.wait()
                    for entry_index in range(APPENDS_PER_THREAD):
                        append_entry(
                            ledger_path,
                            kind="race",
                            config={
                                "thread": thread_index,
                                "index": entry_index,
                            },
                            metrics={"value": thread_index * entry_index},
                            artifacts={},
                        )
                except BaseException as error:
                    errors.append(error)

            threads = [
                threading.Thread(target=writer, args=(index,))
                for index in range(THREADS)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            entries, malformed = read_ledger(ledger_path)

            expected = THREADS * APPENDS_PER_THREAD
            self.assertEqual(malformed, 0)
            self.assertEqual(len(entries), expected)
            lines = [
                line for line in ledger_path.read_text().splitlines() if line.strip()
            ]
            self.assertEqual(len(lines), expected)
            seen = {
                (entry.config.get("thread"), entry.config.get("index"))
                for entry in entries
            }
            self.assertEqual(len(seen), expected)


if __name__ == "__main__":
    unittest.main()
