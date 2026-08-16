import tempfile
import unittest
from pathlib import Path

from omp_gym.bench import load_task_suite
from omp_gym.task import TaskLoadError, TaskSpec, load_task


def write_task(
    task_dir: Path,
    test_command: str = '["python3", "test_app.py"]',
    max_time: str = '"60"',
) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "workspace").mkdir()
    (task_dir / "task.toml").write_text(
        'prompt = "Fix app.py."\n'
        f"test_command = {test_command}\n"
        f"max_time = {max_time}\n"
    )


class LoadTaskTests(unittest.TestCase):
    def test_accepts_a_valid_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            task_dir = Path(temporary_directory) / "fix-app"
            write_task(task_dir)
            loaded = load_task(task_dir)
        self.assertIsInstance(loaded, TaskSpec)

    def test_rejects_a_non_integer_max_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            task_dir = Path(temporary_directory) / "fix-app"
            write_task(task_dir, max_time='"5m"')
            loaded = load_task(task_dir)
        self.assertIsInstance(loaded, TaskLoadError)
        self.assertIn("max_time", loaded.reason)

    def test_rejects_an_unknown_test_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            task_dir = Path(temporary_directory) / "fix-app"
            write_task(
                task_dir,
                test_command='["curl", "http://attacker.invalid"]',
            )
            loaded = load_task(task_dir)
        self.assertIsInstance(loaded, TaskLoadError)
        self.assertIn("test_command", loaded.reason)


class LoadTaskSuiteTests(unittest.TestCase):
    def test_finds_tasks_in_nested_pools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_task(root / "top-task")
            write_task(root / "minted" / "pool-task")
            suite = load_task_suite(root)
        self.assertEqual(
            [task.name for task in suite], ["pool-task", "top-task"]
        )


if __name__ == "__main__":
    unittest.main()
