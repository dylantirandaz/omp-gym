import unittest
import tempfile
from pathlib import Path

from omp_gym.runner import _episode_prompt, _without_parent_runtime
from omp_gym.task import TaskSpec


class EpisodeEnvironmentTests(unittest.TestCase):
    def test_removes_parent_session_and_tool_bridge_state(self) -> None:
        environment = {
            "ANTHROPIC_API_KEY": "provider-key",
            "PI_ARTIFACTS_DIR": "/tmp/parent-artifacts",
            "PI_EVAL_LOCAL_ROOTS": "/tmp/parent-roots",
            "PI_SESSION_FILE": "/tmp/parent-session.jsonl",
            "PI_TOOL_BRIDGE_SESSION": "parent-session",
            "PI_TOOL_BRIDGE_TOKEN": "parent-token",
            "PI_TOOL_BRIDGE_URL": "http://parent.invalid",
            "PI_SMOL_MODEL": "small-model",
        }

        cleaned = _without_parent_runtime(environment)

        self.assertEqual(
            cleaned,
            {
                "ANTHROPIC_API_KEY": "provider-key",
                "PI_SMOL_MODEL": "small-model",
            },
        )

    def test_episode_prompt_contains_explicit_context_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("answer = 0\n")
            task = TaskSpec(
                name="answer",
                prompt="Fix app.py.",
                test_command=("python3", "test_app.py"),
                tools="read,write,bash",
                max_time="60",
                workspace=workspace,
                context_files=("app.py",),
            )

            prompt = _episode_prompt(task, workspace)

        self.assertIn("Fix app.py.", prompt)
        self.assertIn("File: app.py\nanswer = 0", prompt)


if __name__ == "__main__":
    unittest.main()
