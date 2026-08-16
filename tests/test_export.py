import json
import tempfile
import unittest
from pathlib import Path

from omp_gym.export import (
    TASK_PROMPT_PREFIX,
    _canonical_training_call,
    _render_messages,
)
from omp_gym.trajectory import (
    AssistantStep,
    ToolCall,
    ToolResultStep,
    Trajectory,
)


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


if __name__ == "__main__":
    unittest.main()
