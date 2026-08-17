"""Seeded mutation fuzzing of the parsers on the harness trust boundary.

Every loader reads bytes that a task author or an agent workspace can
control. The contract: malformed input becomes a declared error value
(TaskLoadError, torn/malformed line counts), never an escaping
exception. These tests mutate valid seeds with a fixed-seed RNG and
assert that contract across 200 iterations per loader.
"""

import random
import tempfile
import unittest
from pathlib import Path

from omp_gym.ledger import append_entry, read_ledger
from omp_gym.task import TaskLoadError, load_task
from omp_gym.trajectory import parse_session

SEED = 20260816
ITERATIONS = 200

_TASK_TOML = (
    b'prompt = "Fix app.py."\n'
    b'test_command = ["python3", "test_app.py"]\n'
    b'max_time = "60"\n'
)

_SESSION_JSONL = (
    b'{"type": "message", "message": {"role": "user", "content": "hi"}}\n'
    b'{"type": "message", "message": {"role": "assistant", "content": '
    b'[{"type": "text", "text": "working"}, {"type": "toolCall", '
    b'"id": "c1", "name": "read", "arguments": {"path": "app.py"}}]}}\n'
    b'{"type": "message", "message": {"role": "toolResult", '
    b'"toolCallId": "c1", "toolName": "read", '
    b'"content": [{"type": "text", "text": "code"}], "isError": false}}\n'
)

_LEDGER_JSONL = (
    b'{"kind": "run", "timestamp": "2026-08-16T00:00:00+0000", '
    b'"config": {}, "metrics": {"reward": 1.0}, "artifacts": {}, '
    b'"entry_id": "abc123"}\n'
)


def mutate(rng: random.Random, seed_bytes: bytes) -> bytes:
    """Apply a few seeded byte-level mutations to the seed input."""
    data = bytearray(seed_bytes)
    for _ in range(rng.randrange(1, 5)):
        if not data:
            data.append(rng.randrange(256))
            continue
        operation = rng.randrange(4)
        position = rng.randrange(len(data))
        if operation == 0:
            data[position] = rng.randrange(256)
        elif operation == 1:
            del data[position]
        elif operation == 2:
            data.insert(position, rng.randrange(256))
        else:
            data[position:] = data[position:][::-1]
    return bytes(data)


def _write_task_tree(root: Path, toml_bytes: bytes) -> Path:
    task_dir = root / "fuzz-task"
    (task_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_bytes(toml_bytes)
    return task_dir


class LoadTaskFuzzTests(unittest.TestCase):
    def test_mutated_toml_never_escapes_as_exception(self) -> None:
        rng = random.Random(SEED)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for _ in range(ITERATIONS):
                task_dir = _write_task_tree(root, mutate(rng, _TASK_TOML))
                loaded = load_task(task_dir)
                self.assertIsNotNone(loaded)
                if isinstance(loaded, TaskLoadError):
                    self.assertTrue(loaded.reason)


class ParseSessionFuzzTests(unittest.TestCase):
    def test_mutated_jsonl_never_escapes_as_exception(self) -> None:
        rng = random.Random(SEED + 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_path = Path(temporary_directory) / "session.jsonl"
            for _ in range(ITERATIONS):
                session_path.write_bytes(mutate(rng, _SESSION_JSONL))
                trajectory = parse_session(session_path)
                self.assertGreaterEqual(trajectory.torn_lines, 0)

    def test_non_object_json_lines_are_counted_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_path = Path(temporary_directory) / "session.jsonl"
            session_path.write_text('[]\nnull\n"message"\n42\n{}\n')

            trajectory = parse_session(session_path)

        self.assertEqual(trajectory.steps, ())
        self.assertEqual(trajectory.torn_lines, 4)


class ReadLedgerFuzzTests(unittest.TestCase):
    def test_mutated_jsonl_never_escapes_as_exception(self) -> None:
        rng = random.Random(SEED + 2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger_path = Path(temporary_directory) / "ledger.jsonl"
            for _ in range(ITERATIONS):
                ledger_path.write_bytes(mutate(rng, _LEDGER_JSONL))
                entries, malformed = read_ledger(ledger_path)
                self.assertGreaterEqual(malformed, 0)
                for entry in entries:
                    self.assertIsInstance(entry.kind, str)
                    self.assertIsInstance(entry.timestamp, str)

    def test_round_trip_entry_survives_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger_path = Path(temporary_directory) / "ledger.jsonl"
            append_entry(
                ledger_path,
                kind="run",
                config={"model": "m"},
                metrics={"reward": 1.0},
                artifacts={},
            )
            entries, malformed = read_ledger(ledger_path)
        self.assertEqual(malformed, 0)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].kind, "run")


if __name__ == "__main__":
    unittest.main()
