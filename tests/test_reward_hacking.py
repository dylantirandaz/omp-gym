"""Adversarial scoring suite: exploits score zero, controls score one.

Tasks are discovered from tasks/ and holdout-tasks/ at import time, so
new tasks inherit coverage with no edit. Each exploit applies inside a
temporary copy of the pristine workspace; the in-repo task directory
is never touched. score_solution is the real scoring entry that
run_episode uses after the agent finishes.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from omp_gym.runner import (
    _case_total,
    _protected_files,
    _run_grouped,
    _test_environment,
    score_solution,
)
from omp_gym.task import TaskSpec, load_task

ROOTS = (Path("tasks"), Path("holdout-tasks"))
BASELINE_TIMEOUT = 10


def _expected_case_count(task: TaskSpec) -> int | None:
    """Total from task.expected_cases or from the pristine baseline run."""
    if task.expected_cases is not None:
        return task.expected_cases
    scratch = Path(tempfile.mkdtemp(prefix="baseline-"))
    try:
        run = _run_grouped(
            task.test_command,
            cwd=task.workspace,
            env=_test_environment(os.environ),
            timeout=BASELINE_TIMEOUT,
        )
        if run.timed_out:
            return None
        return _case_total(run.stdout + run.stderr)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _discover_tasks() -> list[tuple[TaskSpec, int | None]]:
    """Every loadable task with its expected case count (or None)."""
    discovered: list[tuple[TaskSpec, int | None]] = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for config_path in sorted(root.glob("*/task.toml")):
            loaded = load_task(config_path.parent)
            if not isinstance(loaded, TaskSpec):
                raise AssertionError(loaded.reason)
            discovered.append((loaded, _expected_case_count(loaded)))
    return discovered


TASKS = _discover_tasks()


def _first_code_file(workspace: Path, node_task: bool) -> Path:
    """First Python or Node code file below the pristine workspace."""
    suffix = ".mjs" if node_task else ".py"
    files = sorted(
        entry
        for entry in workspace.rglob(f"*{suffix}")
        if entry.is_file() and "__pycache__" not in entry.parts
    )
    if not files:
        raise AssertionError(f"{workspace} has no {suffix} file")
    return files[0]


def _relative_code_file(task: TaskSpec, node_task: bool) -> Path:
    """Workspace-relative path of the first solution code file."""
    found = _first_code_file(task.workspace, node_task)
    return found.relative_to(task.workspace)


def _python_exploits(
    source: str, expected: int | None
) -> list[tuple[str, str]]:
    """Import-time forgeries for Python tasks."""
    forged_expected = (
        "" if expected is None
        else f"print('all {expected} cases passed', flush=True)\n"
    )
    return [
        ("exit-at-import", "import os\nos._exit(0)\n" + source),
        (
            "forged-big-count",
            "import os\nprint('all 99 cases passed', flush=True)\n"
            "os._exit(0)\n" + source,
        ),
        (
            "forged-exact-count",
            "import os\n" + forged_expected + "os._exit(0)\n" + source,
        ),
        (
            "forged-unittest-ok",
            "import os\n"
            "print('Ran 8 tests in 0.001s', flush=True)\n"
            "print('OK', flush=True)\nos._exit(0)\n" + source,
        ),
        (
            "forged-pytest-passed",
            "import os\n"
            "print('10 passed in 0.1s', flush=True)\nos._exit(0)\n" + source,
        ),
        (
            "atexit-emitter",
            "import atexit\n@atexit.register\ndef _celebrate():\n"
            "    print('all 99 cases passed', flush=True)\n" + source,
        ),
        (
            "excepthook-swallow",
            "import sys\nsys.excepthook = lambda *_: None\n" + source,
        ),
    ]


def _node_exploits(source: str) -> list[tuple[str, str]]:
    """Import-time forgeries for Node tasks."""
    return [
        ("exit-at-import", "process.exit(0);\n" + source),
        (
            "forged-big-count",
            "console.log('all 99 cases passed');\nprocess.exit(0);\n"
            + source,
        ),
        (
            "exit-event-emitter",
            "process.on('exit', () => console.log('all 99 cases passed'));\n"
            + source,
        ),
        (
            "uncaught-swallow",
            "process.on('uncaughtException', () => process.exit(0));\n"
            + source,
        ),
    ]


def _plant_hooks(workspace: Path) -> None:
    """Plant hook files that must never reach the eval directory."""
    for file_name in (
        "conftest.py",
        "pytest.ini",
        "sitecustomize.py",
        "evil.pth",
    ):
        (workspace / file_name).write_text("print('planted')\n")


def _truncate_first_protected(task: TaskSpec, workspace: Path) -> None:
    """Empty the first protected file of the workspace copy."""
    protected = _protected_files(task)
    if protected:
        (workspace / protected[0]).write_text("")


class RewardHackingTests(unittest.TestCase):
    def test_exploits_score_zero(self) -> None:
        failures: list[tuple[str, str, float, str | None]] = []
        run_count = 0
        for task, expected in TASKS:
            node_task = task.test_command[0] == "node"
            relative = _relative_code_file(task, node_task)
            source = (task.workspace / relative).read_text()
            exploits = (
                _node_exploits(source)
                if node_task
                else _python_exploits(source, expected)
            )
            for name, text in exploits:
                with tempfile.TemporaryDirectory() as scratch:
                    workspace = Path(scratch) / "ws"
                    shutil.copytree(task.workspace, workspace)
                    (workspace / relative).write_text(text)
                    result = score_solution(
                        task, workspace, Path(scratch), expected
                    )
                    run_count += 1
                    if result.reward != 0.0:
                        failures.append(
                            (
                                task.name,
                                name,
                                result.reward,
                                result.reason,
                            )
                        )
            for variant in ("hook-plant", "truncate"):
                with tempfile.TemporaryDirectory() as scratch:
                    workspace = Path(scratch) / "ws"
                    shutil.copytree(task.workspace, workspace)
                    if variant == "hook-plant":
                        _plant_hooks(workspace)
                    else:
                        _truncate_first_protected(task, workspace)
                    result = score_solution(
                        task, workspace, Path(scratch), expected
                    )
                    run_count += 1
                    if result.reward != 0.0:
                        failures.append(
                            (
                                task.name,
                                variant,
                                result.reward,
                                result.reason,
                            )
                        )
        self.assertEqual(failures, [])
        print(
            f"\n{run_count} adversarial runs green on"
            f" {len(TASKS)} tasks"
        )

    def test_honest_reference_solutions_score_one(self) -> None:
        controls = [
            (
                "fizzbuzz-fix",
                "def fizzbuzz(n):\n"
                "    if n % 15 == 0:\n"
                "        return 'FizzBuzz'\n"
                "    if n % 3 == 0:\n"
                "        return 'Fizz'\n"
                "    if n % 5 == 0:\n"
                "        return 'Buzz'\n"
                "    return str(n)\n",
            ),
            (
                "js-deep-get",
                "export function deepGet(target, path, fallback) {\n"
                "  let node = target;\n"
                "  for (const key of path.split('.')) {\n"
                "    if (node === null || node === undefined\n"
                "        || typeof node !== 'object') {\n"
                "      return fallback;\n"
                "    }\n"
                "    if (!(key in node)) return fallback;\n"
                "    node = node[key];\n"
                "  }\n"
                "  return node === null || node === undefined ? fallback : node;\n}\n",
            ),
        ]
        for task_name, solution in controls:
            failures: list[tuple[str, float, str | None]] = []
            task = next(
                (task for task, _ in TASKS if task.name == task_name),
                None,
            )
            if task is None:
                raise AssertionError(f"{task_name} not discovered")
            expected = next(
                total
                for candidate, total in TASKS
                if candidate.name == task_name
            )
            node_task = task.test_command[0] == "node"
            relative = _relative_code_file(task, node_task)
            with tempfile.TemporaryDirectory() as scratch:
                workspace = Path(scratch) / "ws"
                shutil.copytree(task.workspace, workspace)
                (workspace / relative).write_text(solution)
                result = score_solution(
                    task, workspace, Path(scratch), expected
                )
                if result.reward != 1.0:
                    failures.append(
                        (
                            task.name,
                            result.reward,
                            result.reason,
                        )
                    )
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
