import json
import unittest

from omp_gym.export import TASK_PROMPT_PREFIX, _render_messages
from omp_gym.trajectory import AssistantStep, ToolCall, Trajectory


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
        payload_text = assistant_content.removeprefix("<tool_call>\n")
        payload_text = payload_text.removesuffix("\n</tool_call>")
        payload = json.loads(payload_text)
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


if __name__ == "__main__":
    unittest.main()
