"""Tests for ledger integrity: lock, fsync, strict schema, torn lines."""

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import omp_gym.ledger as ledger_module
from omp_gym.ledger import (
    SCHEMA_VERSION,
    LedgerEntry,
    append_entry,
    read_ledger,
)


def _canonical(payload: dict[str, object]) -> str:
    """The sorted-keys tight encoding append_entry writes."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _digestless_line() -> str:
    """One pre-digest ledger line; the strict schema rejects it."""
    return json.dumps(
        {
            "kind": "run",
            "timestamp": "2026-08-16T00:00:00+0000",
            "config": {},
            "metrics": {},
            "artifacts": {},
            "entry_id": "legacy01",
        }
    )


def _signed_line(tmp: Path) -> str:
    """One valid line, minted through append_entry itself."""
    append_entry(
        tmp / "ledger.jsonl",
        kind="run",
        config={},
        metrics={},
        artifacts={},
    )
    return (tmp / "ledger.jsonl").read_text().strip()


class AppendReadRoundTripTest(unittest.TestCase):
    def test_round_trip_fields_digest_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            append_entry(
                ledger,
                kind="bench",
                config={"models": ["m1"]},
                metrics={"rows": 3},
                artifacts={"report": "r.md"},
            )
            entries, malformed = read_ledger(ledger)
            self.assertEqual(malformed, 0)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertIsInstance(entry, LedgerEntry)
            self.assertEqual(entry.kind, "bench")
            self.assertEqual(entry.config, {"models": ["m1"]})
            self.assertEqual(entry.metrics, {"rows": 3})
            self.assertEqual(entry.artifacts, {"report": "r.md"})
            self.assertTrue(entry.entry_id)
            self.assertEqual(entry.schema_version, SCHEMA_VERSION)

            # The written line carries sha256 over its canonical
            # body, and the digest verifies.
            raw = json.loads(ledger.read_text().strip())
            digest = raw.pop("sha256")
            self.assertEqual(
                digest,
                hashlib.sha256(_canonical(raw).encode()).hexdigest(),
            )

    def test_lock_file_pairs_with_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            append_entry(ledger, kind="run", config={}, metrics={}, artifacts={})
            self.assertTrue(Path(str(ledger) + ".lock").is_file())

    def test_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "nested" / "deep" / "ledger.jsonl"
            append_entry(ledger, kind="run", config={}, metrics={}, artifacts={})
            entries, malformed = read_ledger(ledger)
            self.assertEqual(malformed, 0)
            self.assertEqual(len(entries), 1)

    def test_missing_ledger_reads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            entries, malformed = read_ledger(Path(tmp) / "ledger.jsonl")
            self.assertEqual(entries, [])
            self.assertEqual(malformed, 0)


class LockAndFsyncTest(unittest.TestCase):
    def test_exclusive_lock_and_unlock_around_the_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            calls: list[int] = []
            real_flock = ledger_module.fcntl.flock

            def recording_flock(handle: object, operation: int) -> None:
                calls.append(operation)
                real_flock(handle, operation)

            with mock.patch.object(ledger_module.fcntl, "flock", recording_flock):
                append_entry(ledger, kind="run", config={}, metrics={}, artifacts={})
            self.assertEqual(
                calls,
                [ledger_module.fcntl.LOCK_EX, ledger_module.fcntl.LOCK_UN],
            )

    def test_append_fsyncs_before_returning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            real_fsync = ledger_module.os.fsync

            with mock.patch.object(
                ledger_module.os, "fsync", wraps=real_fsync
            ) as fsync_mock:
                append_entry(ledger, kind="run", config={}, metrics={}, artifacts={})
            fsync_mock.assert_called_once()


class StrictSchemaTest(unittest.TestCase):
    """One rejection shape per rule; each line counts malformed."""

    def _read_one(self, ledger_dir_line: str) -> tuple[list[LedgerEntry], int]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ledger = Path(tmp.name) / "ledger.jsonl"
        ledger.write_text(ledger_dir_line + "\n")
        return read_ledger(ledger)

    def _assert_malformed(self, line: str) -> None:
        entries, malformed = self._read_one(line)
        self.assertEqual(entries, [], line)
        self.assertEqual(malformed, 1, line)

    def _signed_payload(self) -> dict[str, object]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raw = json.loads(_signed_line(Path(tmp.name)))
        return raw

    def test_torn_json_is_malformed(self) -> None:
        self._assert_malformed('{"kind": "run", "timestamp": "')

    def test_non_object_json_is_malformed(self) -> None:
        self._assert_malformed("[1, 2, 3]")

    def test_undecodable_bytes_line_is_malformed(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ledger = Path(tmp.name) / "ledger.jsonl"
        append_entry(ledger, kind="run", config={}, metrics={}, artifacts={})
        with ledger.open("ab") as handle:
            handle.write(b"\xff\xfe\xed garbage bytes\n")
        entries, malformed = read_ledger(ledger)
        self.assertEqual(len(entries), 1)
        self.assertEqual(malformed, 1)

    def test_digestless_line_is_malformed(self) -> None:
        self._assert_malformed(_digestless_line())

    def test_missing_schema_version_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload.pop("schema_version")
        payload.pop("sha256")
        payload["sha256"] = ledger_module._record_sha256(payload)
        self._assert_malformed(_canonical(payload))

    def test_wrong_schema_version_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload["schema_version"] = 2
        payload.pop("sha256")
        payload["sha256"] = ledger_module._record_sha256(payload)
        self._assert_malformed(_canonical(payload))

    def test_unexpected_top_level_field_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload["extra"] = {"nested": True}
        payload.pop("sha256")
        payload["sha256"] = ledger_module._record_sha256(payload)
        self._assert_malformed(_canonical(payload))

    def test_digest_mismatch_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload["sha256"] = "0" * 64
        self._assert_malformed(_canonical(payload))

    def test_empty_kind_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload["kind"] = ""
        payload.pop("sha256")
        payload["sha256"] = ledger_module._record_sha256(payload)
        self._assert_malformed(_canonical(payload))

    def test_non_dict_config_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload["config"] = "x"
        payload.pop("sha256")
        payload["sha256"] = ledger_module._record_sha256(payload)
        self._assert_malformed(_canonical(payload))

    def test_non_string_artifact_value_is_malformed(self) -> None:
        payload = self._signed_payload()
        payload["artifacts"] = {"count": 3}
        payload.pop("sha256")
        payload["sha256"] = ledger_module._record_sha256(payload)
        self._assert_malformed(_canonical(payload))

    def test_valid_lines_survive_malformed_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"
            append_entry(ledger, kind="run", config={}, metrics={}, artifacts={})
            append_entry(ledger, kind="bench", config={}, metrics={}, artifacts={})
            with ledger.open("a") as handle:
                handle.write("garbage\n")
            entries, malformed = read_ledger(ledger)
            self.assertEqual(malformed, 1)
            self.assertEqual(len(entries), 2)
            self.assertEqual([e.kind for e in entries], ["run", "bench"])


class ConcurrentAppendTest(unittest.TestCase):
    def test_ten_threads_of_fifty_appends_yield_500_valid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.jsonl"

            def worker(index: int) -> None:
                for record in range(50):
                    append_entry(
                        ledger,
                        kind="run",
                        config={"worker": index, "record": record},
                        metrics={},
                        artifacts={},
                    )

            threads = [
                threading.Thread(target=worker, args=(index,)) for index in range(10)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            entries, malformed = read_ledger(ledger)
            lines = ledger.read_text().splitlines()
            self.assertEqual(len(lines), 500)
            self.assertEqual(malformed, 0)
            self.assertEqual(len(entries), 500)
            ids = [entry.entry_id for entry in entries]
            self.assertEqual(len(set(ids)), 500)
            workers = {int(entry.config["worker"]) for entry in entries}
            self.assertEqual(workers, set(range(10)))


if __name__ == "__main__":
    unittest.main()
