"""Tests for the session importers.

The tests cover three contracts: a failed Codex command produces a
tool result with isError true, a discoverable working directory
produces a session header line, and the import stats count results
without an error signal and sessions without a cwd. The fixtures
copy the payload shapes found in real local rollout files.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omp_gym.importers import (
    _convert_claude,
    _convert_codex,
    import_sessions,
)

_FAIL_OUTPUT = (
    "Chunk ID: 452f6a\n"
    "Wall time: 26.8167 seconds\n"
    "Process exited with code 1\n"
    "Original token count: 13\n"
    "Output:\ncurl: (6) Could not resolve host: example.invalid\r\n"
)
_PASS_OUTPUT = (
    "Chunk ID: d7a0f3\n"
    "Wall time: 0.0452 seconds\n"
    "Process exited with code 0\n"
    "Original token count: 2\n"
    "Output:\n959217"
)
_RUNNING_OUTPUT = (
    "Chunk ID: 33fcd1\n"
    "Wall time: 1.0024 seconds\n"
    "Process running with session ID 43780\n"
    "Original token count: 0\n"
    "Output:\n"
)


def _codex_line(kind: str, payload: dict[str, object]) -> str:
    return json.dumps({"type": kind, "payload": payload})


def _codex_call(call_id: str, command: str) -> str:
    return _codex_line(
        "response_item",
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": json.dumps({"cmd": command}),
            "call_id": call_id,
        },
    )


def _codex_output(call_id: str, output: object) -> str:
    return _codex_line(
        "response_item",
        {
            "type": "function_call_output",
            "call_id": call_id,
            "internal_chat_message_metadata_passthrough": {"turn_id": "t1"},
            "output": output,
        },
    )


def _codex_meta(cwd: str) -> str:
    return _codex_line(
        "session_meta",
        {
            "id": "0" * 8,
            "timestamp": "2026-08-16T00:00:00Z",
            "cwd": cwd,
            "originator": "codex_cli_rs",
            "cli_version": "0.0.0",
            "source": "cli",
            "model_provider": "openai",
        },
    )


def _write_rollout(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def _tool_results(entries: list[str]) -> list[dict[str, object]]:
    results = []
    for entry in entries:
        message = json.loads(entry).get("message", {})
        if message.get("role") == "toolResult":
            results.append(message)
    return results


class ConvertCodexTest(unittest.TestCase):
    def test_failed_command_marks_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            _write_rollout(
                rollout,
                [
                    _codex_call("c1", "curl https://example.invalid"),
                    _codex_output("c1", _FAIL_OUTPUT),
                    _codex_call("c2", "echo tip"),
                    _codex_output("c2", _PASS_OUTPUT),
                ],
            )
            converted = _convert_codex(rollout)
        results = _tool_results(converted.entries)
        self.assertEqual([r["isError"] for r in results], [True, False])
        self.assertEqual(converted.results_without_error_signal, 0)

    def test_missing_signal_stays_false_and_is_counted(self) -> None:
        no_signal_list = [
            {"type": "input_text", "text": "Internal Error ()"}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            _write_rollout(
                rollout,
                [
                    _codex_call("c1", "open url"),
                    _codex_output("c1", no_signal_list),
                    _codex_call("c2", "sleep 60"),
                    _codex_output("c2", _RUNNING_OUTPUT),
                ],
            )
            converted = _convert_codex(rollout)
        results = _tool_results(converted.entries)
        self.assertEqual([r["isError"] for r in results], [False, False])
        self.assertEqual(converted.results_without_error_signal, 2)
        self.assertIn(
            "Internal Error ()",
            results[0]["content"][0]["text"],
        )

    def test_json_output_signals(self) -> None:
        timed_out = json.dumps({"message": "Wait timed out.", "timed_out": True})
        wrapped = (
            "Wall time: 2.3812 seconds\n"
            "Output:\n" + json.dumps({"exit_code": 1, "output": "boom"})
        )
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            _write_rollout(
                rollout,
                [
                    _codex_call("c1", "wait"),
                    _codex_output("c1", timed_out),
                    _codex_call("c2", "remote run"),
                    _codex_output("c2", wrapped),
                ],
            )
            converted = _convert_codex(rollout)
        results = _tool_results(converted.entries)
        self.assertEqual([r["isError"] for r in results], [True, True])
        self.assertEqual(converted.results_without_error_signal, 0)

    def test_cwd_comes_from_session_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            _write_rollout(
                rollout,
                [
                    _codex_meta("/work/project"),
                    _codex_call("c1", "true"),
                    _codex_output("c1", _PASS_OUTPUT),
                ],
            )
            converted = _convert_codex(rollout)
        self.assertEqual(converted.cwd, "/work/project")


class ConvertClaudeTest(unittest.TestCase):
    def test_cwd_and_error_passthrough(self) -> None:
        lines = [
            json.dumps({"type": "mode", "sessionId": "s1", "mode": "code"}),
            json.dumps(
                {
                    "type": "user",
                    "cwd": "/work/claude",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "Run it."}],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "user",
                    "cwd": "/work/claude",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "boom",
                                "is_error": True,
                            }
                        ],
                    },
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "session.jsonl"
            _write_rollout(session, lines)
            converted = _convert_claude(session)
        self.assertEqual(converted.cwd, "/work/claude")
        results = _tool_results(converted.entries)
        self.assertEqual([r["isError"] for r in results], [True])
        self.assertEqual(converted.results_without_error_signal, 0)


class ImportSessionsTest(unittest.TestCase):
    def test_header_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            store = home / ".codex" / "sessions" / "2026" / "08"
            store.mkdir(parents=True)
            _write_rollout(
                store / "a-with-cwd.jsonl",
                [
                    _codex_meta("/work/project"),
                    _codex_call("c1", "false"),
                    _codex_output("c1", _FAIL_OUTPUT),
                ],
            )
            _write_rollout(
                store / "b-without-cwd.jsonl",
                [
                    _codex_call("c1", "spin"),
                    _codex_output("c1", _RUNNING_OUTPUT),
                ],
            )
            out_dir = Path(tmp) / "out"
            with mock.patch.object(Path, "home", return_value=home):
                stats = import_sessions("codex", out_dir)

            self.assertEqual(stats.files_seen, 2)
            self.assertEqual(stats.files_written, 2)
            self.assertEqual(stats.results_without_error_signal, 1)
            self.assertEqual(stats.sessions_without_cwd, 1)

            with_cwd = (out_dir / "codex" / "a-with-cwd.jsonl").read_text()
            header = json.loads(with_cwd.splitlines()[0])
            self.assertEqual(
                header, {"type": "session", "cwd": "/work/project"}
            )
            without_cwd = (
                out_dir / "codex" / "b-without-cwd.jsonl"
            ).read_text()
            first = json.loads(without_cwd.splitlines()[0])
            self.assertEqual(first["type"], "message")


if __name__ == "__main__":
    unittest.main()
