import shutil
import tempfile
import unittest
from pathlib import Path

from omp_gym.runner import (
    _changed_protected_files,
    _episode_environment,
    _episode_prompt,
    _file_digests,
    _partial_credit,
    _protected_files,
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


def make_task(workspace: Path, test_command: tuple[str, ...]) -> TaskSpec:
    return TaskSpec(
        name="fix-app",
        prompt="Fix app.py.",
        test_command=test_command,
        tools="read,write,bash",
        max_time="60",
        workspace=workspace,
    )


class ProtectedFilesTests(unittest.TestCase):
    def test_collects_command_args_test_names_and_tests_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("answer = 0\n")
            (workspace / "cases.txt").write_text("1\n")
            (workspace / "test_app.py").write_text("import app\n")
            (workspace / "lib").mkdir()
            (workspace / "lib" / "app_test.js").write_text("check()\n")
            (workspace / "tests").mkdir()
            (workspace / "tests" / "helper.py").write_text("pass\n")
            task = make_task(
                workspace, ("python3", "test_app.py", "cases.txt")
            )

            protected = _protected_files(task)

        self.assertEqual(
            protected,
            (
                "cases.txt",
                "lib/app_test.js",
                "test_app.py",
                "tests/helper.py",
            ),
        )

    def test_ignores_command_args_that_are_not_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "app.py").write_text("answer = 0\n")
            task = make_task(workspace, ("python3", "-m", "unittest"))

            protected = _protected_files(task)

        self.assertEqual(protected, ())


class ChangedProtectedFilesTests(unittest.TestCase):
    def make_pristine(self, root: Path) -> Path:
        pristine = root / "pristine"
        pristine.mkdir()
        (pristine / "app.py").write_text("answer = 0\n")
        (pristine / "test_app.py").write_text(
            "import app\nassert app.answer == 42\n"
        )
        return pristine

    def test_truncated_test_file_is_caught_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine = self.make_pristine(root)
            task = make_task(pristine, ("python3", "test_app.py"))
            protected = _protected_files(task)
            digests = _file_digests(pristine, protected)
            episode = root / "ws"
            shutil.copytree(pristine, episode)
            (episode / "app.py").write_text("answer = 42\n")
            (episode / "test_app.py").write_text("")

            changed = _changed_protected_files(episode, digests)

        self.assertEqual(changed, ("test_app.py",))

    def test_deleted_test_file_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine = self.make_pristine(root)
            task = make_task(pristine, ("python3", "test_app.py"))
            digests = _file_digests(pristine, _protected_files(task))
            episode = root / "ws"
            shutil.copytree(pristine, episode)
            (episode / "test_app.py").unlink()

            changed = _changed_protected_files(episode, digests)

        self.assertEqual(changed, ("test_app.py",))

    def test_untouched_test_files_pass_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine = self.make_pristine(root)
            task = make_task(pristine, ("python3", "test_app.py"))
            digests = _file_digests(pristine, _protected_files(task))
            episode = root / "ws"
            shutil.copytree(pristine, episode)
            (episode / "app.py").write_text("answer = 42\n")

            changed = _changed_protected_files(episode, digests)

        self.assertEqual(changed, ())


if __name__ == "__main__":
    unittest.main()
