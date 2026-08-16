import unittest
import tempfile
from pathlib import Path

from omp_gym.runner import (
    _episode_environment,
    _episode_prompt,
    _partial_credit,
)
from omp_gym.task import TaskSpec


class EpisodeEnvironmentTests(unittest.TestCase):
    def test_child_gets_only_whitelisted_host_variables(self) -> None:
        host_environment = {
            "PATH": "/usr/bin",
            "HOME": "/Users/me",
            "AWS_SECRET_ACCESS_KEY": "cloud-secret",
            "PI_SESSION_FILE": "/tmp/parent-session.jsonl",
            "PI_TOOL_BRIDGE_TOKEN": "parent-token",
        }

        environment = _episode_environment(
            host_environment, {"EXTRA": "value"}
        )

        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["HOME"], "/Users/me")
        self.assertEqual(environment["EXTRA"], "value")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("PI_SESSION_FILE", environment)
        self.assertNotIn("PI_TOOL_BRIDGE_TOKEN", environment)

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


class PartialCreditTests(unittest.TestCase):
    def test_reads_custom_case_counts(self) -> None:
        self.assertEqual(_partial_credit("2 of 8 cases failed"), 0.75)

    def test_reads_unittest_failure_counts(self) -> None:
        output = "Ran 8 tests in 0.002s\n\nFAILED (failures=4)\n"
        self.assertEqual(_partial_credit(output), 0.5)

    def test_reads_unittest_failures_and_errors(self) -> None:
        output = "Ran 6 tests in 0.001s\n\nFAILED (failures=2, errors=1)\n"
        self.assertEqual(_partial_credit(output), 0.5)

    def test_reads_unittest_success(self) -> None:
        output = "Ran 8 tests in 0.002s\n\nOK\n"
        self.assertEqual(_partial_credit(output), 1.0)

    def test_reads_pytest_summary(self) -> None:
        self.assertEqual(_partial_credit("2 failed, 6 passed in 0.1s"), 0.75)

    def test_returns_none_without_counts(self) -> None:
        self.assertIsNone(_partial_credit("Segmentation fault"))


if __name__ == "__main__":
    unittest.main()
