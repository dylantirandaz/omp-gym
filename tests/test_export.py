import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omp_gym.export import (
    TASK_PROMPT_PREFIX,
    _canonical_training_call,
    _collect_episode_messages,
    _collect_session_messages,
    _redact,
    _render_messages,
    export_dataset,
)
from omp_gym.trajectory import (
    AssistantStep,
    ToolCall,
    ToolResultStep,
    Trajectory,
)


def _session_line(role: str, text: str) -> str:
    """Build one session JSONL line with a single text block."""
    return json.dumps(
        {
            "type": "message",
            "message": {
                "role": role,
                "content": [{"type": "text", "text": text}],
            },
        }
    )


def _write_session(path: Path, user_text: str, assistant_text: str) -> None:
    """Write a minimal two-step session file."""
    path.write_text(
        _session_line("user", user_text)
        + "\n"
        + _session_line("assistant", assistant_text)
        + "\n"
    )


class RedactionTests(unittest.TestCase):
    UNCHANGED = (
        "tokens = tokenizer.encode(prompt)",
        "self.tokenizer = AutoTokenizer.from_pretrained(m)",
        "token_cap: int = 2048",
        "password_hash: str",
        "total_tokens += usage",
        "secret=short",
        "password = get_password(user)",
        "access_token_count = 12345678",
    )

    def test_code_shaped_text_is_unchanged(self) -> None:
        for text in self.UNCHANGED:
            with self.subTest(text=text):
                self.assertEqual(_redact(text), text)

    def test_known_token_literals_are_redacted(self) -> None:
        literals = (
            "sk-or-v1-abcdef1234567890",
            "ghp_abcdefghij1234567890",
            "github_pat_abcdefghij1234567890",
            "gho_abcdefghij1234567890",
            "xoxb-1234567890-abcdef",
            "AKIAIOSFODNN7EXAMPLE",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghij",
            "hf_abcdefghijABCDEFGHIJ",
            "gsk_abcdefghij1234567890",
            "AIzaSyA-abcdefghijklmnopqrstuvwxyz1234",
        )
        for literal in literals:
            with self.subTest(literal=literal):
                redacted = _redact(f"log: {literal} end")
                self.assertNotIn(literal, redacted)
                self.assertIn("[REDACTED]", redacted)
                self.assertIn("end", redacted)

    def test_json_api_key_value_is_redacted(self) -> None:
        self.assertEqual(
            _redact('{"api_key": "abcdef123456"}'),
            '{"api_key": "[REDACTED]"}',
        )

    def test_json_ghp_token_value_is_redacted(self) -> None:
        self.assertEqual(
            _redact('{"token": "ghp_abcdefghij1234567890"}'),
            '{"token": "[REDACTED]"}',
        )

    def test_bearer_header_keeps_the_word_bearer(self) -> None:
        header = (
            "Authorization: Bearer "
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdefghij"
        )
        self.assertEqual(
            _redact(header), "Authorization: Bearer [REDACTED]"
        )

    def test_bearer_header_with_opaque_token(self) -> None:
        self.assertEqual(
            _redact("Authorization: Bearer abc123def456ghij"),
            "Authorization: Bearer [REDACTED]",
        )

    def test_env_assignment_with_key_literal(self) -> None:
        self.assertEqual(
            _redact("OPENROUTER_API_KEY=sk-or-v1-abcdef1234567890"),
            "OPENROUTER_API_KEY=[REDACTED]",
        )

    def test_secret_word_assignments_are_redacted(self) -> None:
        cases = (
            ("password=hunter2hunter2", "password=[REDACTED]"),
            (
                "client_secret: verysecretvalue",
                "client_secret: [REDACTED]",
            ),
            (
                "'access_token': 'abcdefgh1234'",
                "'access_token': '[REDACTED]'",
            ),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_redact(text), expected)

    def test_upper_case_env_assignments_are_redacted(self) -> None:
        cases = (
            (
                "PREFIX_API_KEY=abcdefgh12345678",
                "PREFIX_API_KEY=[REDACTED]",
            ),
            ("HF_TOKEN=hf_abcdefghijABCDEFGHIJ", "HF_TOKEN=[REDACTED]"),
            ("DB_PASSWORD=hunter2hunter2", "DB_PASSWORD=[REDACTED]"),
            ("SERVICE_APIKEY=abcd1234efgh", "SERVICE_APIKEY=[REDACTED]"),
            (
                "APP_CLIENT_SECRET = supersecret99",
                "APP_CLIENT_SECRET = [REDACTED]",
            ),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_redact(text), expected)


class CurriculumRenderingTests(unittest.TestCase):
    def test_tool_turn_keeps_only_calls_and_relative_paths(self) -> None:
        trajectory = Trajectory(
            steps=(
                AssistantStep(
                    text="I will inspect the file.",
                    thinking="An internal plan that must not train the model.",
                    tool_calls=(
                        ToolCall(
                            call_id="call_1",
                            name="read",
                            arguments={
                                "path": "/tmp/run/ws/src/main.py",
                                "i": "Read source",
                            },
                        ),
                    ),
                ),
            ),
            torn_lines=0,
        )

        messages = _render_messages(trajectory, TASK_PROMPT_PREFIX)

        assistant_content = messages[-1]["content"]
        self.assertNotIn("I will inspect", assistant_content)
        self.assertNotIn("internal plan", assistant_content)
        payload = json.loads(assistant_content)
        self.assertEqual(payload["arguments"]["path"], "src/main.py")

    def test_redacts_credentials_from_tool_results(self) -> None:
        trajectory = Trajectory(
            steps=(
                ToolResultStep(
                    call_id="call_1",
                    tool_name="bash",
                    text=(
                        "OPENROUTER_API_KEY=sk-or-v1-abcdef1234567890\n"
                        "access_token: super-secret-value\n"
                        "exit 0"
                    ),
                    is_error=False,
                ),
            ),
            torn_lines=0,
        )

        messages = _render_messages(trajectory, TASK_PROMPT_PREFIX)

        rendered = "\n".join(message["content"] for message in messages)
        self.assertNotIn("sk-or-v1-abcdef1234567890", rendered)
        self.assertNotIn("super-secret-value", rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertIn("exit 0", rendered)

    def test_terminal_turn_keeps_its_result_text(self) -> None:
        trajectory = Trajectory(
            steps=(
                AssistantStep(
                    text="All tests pass.",
                    tool_calls=(),
                    thinking="This text stays private.",
                ),
            ),
            torn_lines=0,
        )

        messages = _render_messages(trajectory, TASK_PROMPT_PREFIX)

        self.assertEqual(messages[-1]["content"], "All tests pass.")

    def test_tool_result_uses_tokenizer_response_format(self) -> None:
        trajectory = Trajectory(
            steps=(
                ToolResultStep(
                    call_id="call_1",
                    tool_name="read",
                    text="source",
                    is_error=False,
                ),
            ),
            torn_lines=0,
        )

        messages = _render_messages(trajectory, TASK_PROMPT_PREFIX)

        self.assertEqual(messages[-1]["role"], "user")
        self.assertTrue(
            messages[-1]["content"].endswith(
                "<tool_response>\nsource\n</tool_response>"
            )
        )
        self.assertNotIn("status=", messages[-1]["content"])

    def test_failed_authoring_call_is_not_training_data(self) -> None:
        trajectory = Trajectory(
            steps=(
                AssistantStep(
                    text="I will try an invalid patch.",
                    tool_calls=(
                        ToolCall(
                            call_id="call_1",
                            name="edit",
                            arguments={"input": "invalid patch"},
                        ),
                    ),
                ),
                ToolResultStep(
                    call_id="call_1",
                    tool_name="edit",
                    text="invalid patch",
                    is_error=True,
                ),
                AssistantStep(
                    text="Use a valid patch.",
                    tool_calls=(),
                ),
            ),
            torn_lines=0,
        )

        messages = _render_messages(trajectory, TASK_PROMPT_PREFIX)

        self.assertEqual(messages[-1]["content"], "Use a valid patch.")
        self.assertNotIn("invalid patch", json.dumps(messages))
        self.assertNotIn("I will try", json.dumps(messages))

    def test_successful_edit_becomes_a_full_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("def answer():\n    return 42\n")
            trajectory = Trajectory(
                steps=(
                    AssistantStep(
                        text="",
                        tool_calls=(
                            ToolCall(
                                call_id="call_1",
                                name="edit",
                                arguments={
                                    "input": (
                                        "[app.py#ABCD]\n"
                                        "PUT 1.=2:\n"
                                        "+def answer():\n"
                                        "+    return 42\n"
                                    ),
                                    "i": "Fix answer",
                                },
                            ),
                        ),
                    ),
                    ToolResultStep(
                        call_id="call_1",
                        tool_name="edit",
                        text="updated",
                        is_error=False,
                    ),
                ),
                torn_lines=0,
            )

            messages = _render_messages(
                trajectory,
                TASK_PROMPT_PREFIX,
                workspace,
            )

        assistant_content = next(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        payload = json.loads(assistant_content)
        self.assertEqual(payload["name"], "write")
        self.assertEqual(
            payload["arguments"],
            {
                "path": "app.py",
                "content": "def answer():\n    return 42\n",
                "i": "Fix answer",
            },
        )

    def test_only_the_final_edit_becomes_a_full_file_write(self) -> None:
        first_patch = "[app.py#ABCD]\nPUT 1.=1:\n+draft = 1\n"
        second_patch = "[app.py#EF01]\nPUT 1.=1:\n+final = True\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("final = True\n")
            trajectory = Trajectory(
                steps=(
                    AssistantStep(
                        text="",
                        tool_calls=(
                            ToolCall(
                                call_id="call_1",
                                name="edit",
                                arguments={
                                    "input": first_patch,
                                    "i": "First pass",
                                },
                            ),
                        ),
                    ),
                    ToolResultStep(
                        call_id="call_1",
                        tool_name="edit",
                        text="updated",
                        is_error=False,
                    ),
                    AssistantStep(
                        text="",
                        tool_calls=(
                            ToolCall(
                                call_id="call_2",
                                name="edit",
                                arguments={
                                    "input": second_patch,
                                    "i": "Second pass",
                                },
                            ),
                        ),
                    ),
                    ToolResultStep(
                        call_id="call_2",
                        tool_name="edit",
                        text="updated",
                        is_error=False,
                    ),
                ),
                torn_lines=0,
            )

            messages = _render_messages(
                trajectory,
                TASK_PROMPT_PREFIX,
                workspace,
            )

        payloads = [
            json.loads(message["content"])
            for message in messages
            if message["role"] == "assistant"
        ]
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["name"], "edit")
        self.assertEqual(payloads[0]["arguments"]["input"], first_patch)
        self.assertEqual(payloads[1]["name"], "write")
        self.assertEqual(
            payloads[1]["arguments"]["content"], "final = True\n"
        )

    def test_write_after_edit_keeps_both_calls_as_written(self) -> None:
        patch = "[app.py#ABCD]\nPUT 1.=1:\n+draft = 1\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("on_disk = True\n")
            trajectory = Trajectory(
                steps=(
                    AssistantStep(
                        text="",
                        tool_calls=(
                            ToolCall(
                                call_id="call_1",
                                name="edit",
                                arguments={"input": patch},
                            ),
                        ),
                    ),
                    ToolResultStep(
                        call_id="call_1",
                        tool_name="edit",
                        text="updated",
                        is_error=False,
                    ),
                    AssistantStep(
                        text="",
                        tool_calls=(
                            ToolCall(
                                call_id="call_2",
                                name="write",
                                arguments={
                                    "path": "app.py",
                                    "content": "explicit = True\n",
                                },
                            ),
                        ),
                    ),
                    ToolResultStep(
                        call_id="call_2",
                        tool_name="write",
                        text="written",
                        is_error=False,
                    ),
                ),
                torn_lines=0,
            )

            messages = _render_messages(
                trajectory,
                TASK_PROMPT_PREFIX,
                workspace,
            )

        payloads = [
            json.loads(message["content"])
            for message in messages
            if message["role"] == "assistant"
        ]
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["name"], "edit")
        self.assertEqual(payloads[0]["arguments"]["input"], patch)
        self.assertEqual(payloads[1]["name"], "write")
        self.assertEqual(
            payloads[1]["arguments"]["content"], "explicit = True\n"
        )

    def test_legacy_edit_becomes_a_full_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("answer = 42\n")

            name, arguments = _canonical_training_call(
                ToolCall(
                    call_id="call_1",
                    name="edit",
                    arguments={
                        "path": "app.py",
                        "old_string": "answer = 0",
                        "new_string": "answer = 42",
                    },
                ),
                workspace,
            )

        self.assertEqual(name, "write")
        self.assertEqual(arguments["path"], "app.py")
        self.assertEqual(arguments["content"], "answer = 42\n")

    def test_failed_test_command_remains_training_data(self) -> None:
        trajectory = Trajectory(
            steps=(
                AssistantStep(
                    text="",
                    tool_calls=(
                        ToolCall(
                            call_id="call_1",
                            name="bash",
                            arguments={"command": "python3 test_app.py"},
                        ),
                    ),
                ),
                ToolResultStep(
                    call_id="call_1",
                    tool_name="bash",
                    text="1 test failed",
                    is_error=True,
                ),
            ),
            torn_lines=0,
        )

        messages = _render_messages(trajectory, TASK_PROMPT_PREFIX)

        contents = [message["content"] for message in messages]
        self.assertTrue(any('"name": "bash"' in content for content in contents))
        self.assertTrue(any("1 test failed" in content for content in contents))


class HoldoutExclusionTests(unittest.TestCase):
    def test_holdout_mention_excludes_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sessions = Path(temporary_directory)
            _write_session(
                sessions / "clean.jsonl",
                "Fix the parser bug.",
                "The parser now passes its tests.",
            )
            _write_session(
                sessions / "leak.jsonl",
                "Run the suite in holdout-tasks/json-fix.",
                "The holdout suite passes.",
            )

            rendered, seen, filtered, excluded, torn = (
                _collect_session_messages(sessions, min_quality=False)
            )

        self.assertEqual(seen, 2)
        self.assertEqual(filtered, 0)
        self.assertEqual(excluded, 1)
        self.assertEqual(torn, 0)
        self.assertEqual(len(rendered), 1)
        self.assertNotIn("holdout-tasks", json.dumps(rendered))


def _write_episode(runs: Path, sessions: Path, name: str, task: str) -> None:
    """Write one graded episode record with a minimal session."""
    episode_dir = runs / name
    episode_dir.mkdir(parents=True)
    session_file = sessions / f"{name}.jsonl"
    _write_session(session_file, "Fix the bug.", "The bug is fixed.")
    (episode_dir / "episode.json").write_text(
        json.dumps(
            {
                "task": task,
                "reward": 1.0,
                "session_file": str(session_file),
                "episode_dir": str(episode_dir),
            }
        )
    )


class EpisodeHoldoutExclusionTests(unittest.TestCase):
    def test_holdout_named_episode_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = root / "runs"
            sessions = root / "sessions"
            sessions.mkdir()
            holdout = root / "holdout-tasks"
            (holdout / "interval-union").mkdir(parents=True)
            _write_episode(runs, sessions, "ep-1", "interval-union")
            _write_episode(runs, sessions, "ep-2", "parser-fix")

            rendered, seen, excluded, torn = _collect_episode_messages(
                runs, 0.0, holdout
            )

        self.assertEqual(seen, 2)
        self.assertEqual(excluded, 1)
        self.assertEqual(torn, 0)
        self.assertEqual(len(rendered), 1)

    def test_missing_holdout_dir_means_no_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = root / "runs"
            sessions = root / "sessions"
            sessions.mkdir()
            _write_episode(runs, sessions, "ep-1", "interval-union")

            rendered, seen, excluded, torn = _collect_episode_messages(
                runs, 0.0, root / "holdout-tasks"
            )

        self.assertEqual(seen, 1)
        self.assertEqual(excluded, 0)
        self.assertEqual(torn, 0)
        self.assertEqual(len(rendered), 1)

    def test_export_dataset_counts_the_exclusion(self) -> None:
        def counter(text: str) -> int:
            return len(text) // 4 + 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = root / "runs"
            sessions = root / "sessions"
            sessions.mkdir()
            holdout = root / "holdout-tasks"
            (holdout / "interval-union").mkdir(parents=True)
            _write_episode(runs, sessions, "ep-1", "interval-union")
            _write_episode(runs, sessions, "ep-2", "parser-fix")

            with mock.patch(
                "omp_gym.export.load_token_counter",
                return_value=counter,
            ):
                stats = export_dataset(
                    runs,
                    root / "no-sessions",
                    root / "out",
                    0.0,
                    "unused-tokenizer",
                    2048,
                    min_quality=False,
                    holdout_dir=holdout,
                )

        self.assertEqual(stats.episodes_seen, 2)
        self.assertEqual(stats.episodes_excluded_holdout, 1)
        self.assertEqual(stats.trajectories_exported, 1)


class DatasetSplitTests(unittest.TestCase):
    def test_split_is_random_deterministic_and_disjoint(self) -> None:
        def counter(text: str) -> int:
            return len(text) // 4 + 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runs = root / "runs"
            runs.mkdir()
            sessions = root / "sessions"
            sessions.mkdir()
            for index in range(12):
                _write_session(
                    sessions / f"session-{index:02d}.jsonl",
                    f"Task number {index}.",
                    f"Reply number {index}.",
                )

            with mock.patch(
                "omp_gym.export.load_token_counter",
                return_value=counter,
            ):
                stats = export_dataset(
                    runs,
                    sessions,
                    root / "out_a",
                    1.0,
                    "unused-tokenizer",
                    2048,
                    min_quality=False,
                )
                export_dataset(
                    runs,
                    sessions,
                    root / "out_b",
                    1.0,
                    "unused-tokenizer",
                    2048,
                    min_quality=False,
                )

            train_a = (root / "out_a" / "train.jsonl").read_text()
            valid_a = (root / "out_a" / "valid.jsonl").read_text()
            train_b = (root / "out_b" / "train.jsonl").read_text()
            valid_b = (root / "out_b" / "valid.jsonl").read_text()

        self.assertEqual(train_a, train_b)
        self.assertEqual(valid_a, valid_b)
        self.assertEqual(stats.train_samples, 11)
        self.assertEqual(stats.valid_samples, 1)
        self.assertEqual(stats.sessions_excluded_holdout, 0)
        train_lines = set(train_a.splitlines())
        valid_lines = set(valid_a.splitlines())
        self.assertEqual(len(train_lines & valid_lines), 0)
        self.assertEqual(len(train_lines | valid_lines), 12)


if __name__ == "__main__":
    unittest.main()
