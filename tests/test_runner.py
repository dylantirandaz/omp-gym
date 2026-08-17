import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from omp_gym.runner import (
    EpisodeFailure,
    EpisodeProviderRoute,
    _baseline_passed,
    _build_eval_dir,
    _changed_protected_files,
    _episode_environment,
    _episode_prompt,
    _file_digests,
    _find_session_file,
    _overlay_files,
    _partial_credit,
    _prepare_agent_home,
    _protected_files,
    _run_grouped,
    _score_test_run,
    _session_provider_error,
    _test_environment,
    _test_evidence,
    run_episode,
)
from omp_gym.task import TaskSpec


class EpisodeEnvironmentTests(unittest.TestCase):
    def test_child_gets_only_scoped_variables(self) -> None:
        host_environment = {
            "PATH": "/usr/bin",
            "HOME": "/Users/me",
            "AWS_SECRET_ACCESS_KEY": "cloud-secret",
            "PI_SESSION_FILE": "/sessions/parent-session.jsonl",
            "PI_TOOL_BRIDGE_TOKEN": "parent-token",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "OPENAI_API_KEY": "openai-secret",
        }

        with unittest.mock.patch("omp_gym.runner.load_env_file", return_value={}):
            environment = _episode_environment(
                host_environment,
                {"EXTRA": "value"},
                model="anthropic/claude-sonnet",
                home=Path("/ep/home"),
                tmpdir=Path("/ep/tmp"),
            )

        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["HOME"], "/ep/home")
        self.assertEqual(environment["TMPDIR"], "/ep/tmp")
        self.assertEqual(environment["EXTRA"], "value")
        self.assertEqual(environment["ANTHROPIC_API_KEY"], "anthropic-secret")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("PI_SESSION_FILE", environment)
        self.assertNotIn("PI_TOOL_BRIDGE_TOKEN", environment)

    def test_unresolved_provider_gets_no_key(self) -> None:
        with unittest.mock.patch("omp_gym.runner.load_env_file", return_value={}):
            environment = _episode_environment(
                {"PATH": "/usr/bin"},
                None,
                model="vendor-x/model",
                home=Path("/ep/home"),
                tmpdir=Path("/ep/tmp"),
            )

        for name in environment:
            self.assertNotRegex(name, r"(_KEY|_TOKEN|_SECRET|_PASSWORD)$")

    def test_explicit_policy_secret_is_kept_only_when_supplied(self) -> None:
        with unittest.mock.patch("omp_gym.runner.load_env_file", return_value={}):
            kept = _episode_environment(
                {"PATH": "/usr/bin"},
                {"POLICY_KEY": "private"},
                model="omp-gym/model",
                home=Path("/ep/home"),
                tmpdir=Path("/ep/tmp"),
                extra_secret_names=("POLICY_KEY",),
            )
            scrubbed = _episode_environment(
                {"PATH": "/usr/bin"},
                {"POLICY_KEY": "private"},
                model="omp-gym/model",
                home=Path("/ep/home"),
                tmpdir=Path("/ep/tmp"),
            )

        self.assertEqual(kept["POLICY_KEY"], "private")
        self.assertNotIn("POLICY_KEY", scrubbed)

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


class ProviderConfigurationTests(unittest.TestCase):
    def test_private_home_uses_explicit_models_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_home = root / "source-home"
            source_agent = source_home / ".omp" / "agent"
            source_agent.mkdir(parents=True)
            (source_agent / "models.yml").write_text("source\n")
            supplied_models = root / "supplied-models.yml"
            supplied_models.write_text("supplied\n")

            with unittest.mock.patch(
                "omp_gym.runner.Path.home",
                return_value=source_home,
            ):
                home = _prepare_agent_home(
                    root / "episode",
                    "omp-gym/model",
                    EpisodeProviderRoute(
                        models_file=supplied_models,
                        network="open-443",
                    ),
                )

            installed_models = home / ".omp" / "agent" / "models.yml"
            self.assertEqual(installed_models.read_text(), "supplied\n")
            self.assertEqual((source_agent / "models.yml").read_text(), "source\n")

    def test_empty_assistant_session_reports_provider_error(self) -> None:
        session = (
            json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "stopReason": "error",
                        "errorMessage": "Unable to connect.",
                    },
                }
            ).encode()
            + b"\n"
        )

        self.assertEqual(_session_provider_error(session), "Unable to connect.")

    def test_assistant_tool_work_survives_an_earlier_provider_error(self) -> None:
        messages = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "stopReason": "error",
                    "errorMessage": "Temporary provider error.",
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "name": "read",
                            "arguments": {"path": "app.py"},
                        }
                    ],
                    "stopReason": "toolUse",
                },
            },
        ]
        session = b"\n".join(json.dumps(message).encode() for message in messages)

        self.assertIsNone(_session_provider_error(session))


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

    def test_pytest_errors_count_against_partial_credit(self) -> None:
        self.assertEqual(_partial_credit("1 error, 10 passed in 0.1s"), 10 / 11)
        self.assertEqual(_partial_credit("2 errors, 3 failed, 5 passed in 0.1s"), 0.5)

    def test_returns_none_without_counts(self) -> None:
        self.assertIsNone(_partial_credit("Segmentation fault"))


class TestEvidenceTests(unittest.TestCase):
    def test_reads_custom_all_passed_line(self) -> None:
        self.assertEqual(_test_evidence("all 10 cases passed\n"), 10)

    def test_reads_unittest_ok(self) -> None:
        output = "Ran 8 tests in 0.002s\n\nOK\n"
        self.assertEqual(_test_evidence(output), 8)

    def test_rejects_unittest_failed(self) -> None:
        output = "Ran 8 tests in 0.002s\n\nFAILED (failures=4)\n"
        self.assertEqual(_test_evidence(output), 0)

    def test_reads_pytest_summary(self) -> None:
        self.assertEqual(_test_evidence("6 passed in 0.12s\n"), 6)

    def test_rejects_pytest_summary_with_failures(self) -> None:
        self.assertEqual(_test_evidence("2 failed, 6 passed in 0.1s\n"), 0)

    def test_rejects_pytest_summary_with_errors(self) -> None:
        self.assertEqual(_test_evidence("1 error, 6 passed in 0.1s\n"), 0)

    def test_empty_output_gives_zero(self) -> None:
        self.assertEqual(_test_evidence(""), 0)


class ScoreTestRunTests(unittest.TestCase):
    def test_exit_zero_without_evidence_scores_zero(self) -> None:
        self.assertEqual(_score_test_run(0, ""), (0.0, None, 0))

    def test_exit_zero_with_evidence_scores_one(self) -> None:
        reward, partial, evidence = _score_test_run(0, "all 10 cases passed\n")
        self.assertEqual(reward, 1.0)
        self.assertIsNone(partial)
        self.assertEqual(evidence, 10)

    def test_exit_zero_keeps_partial_credit_from_unittest_output(self) -> None:
        output = "Ran 8 tests in 0.002s\n\nOK\n"
        self.assertEqual(_score_test_run(0, output), (1.0, 1.0, 8))

    def test_failure_keeps_partial_credit(self) -> None:
        reward, partial, evidence = _score_test_run(1, "2 of 8 cases failed\n")
        self.assertEqual(reward, 0.0)
        self.assertEqual(partial, 0.75)
        self.assertEqual(evidence, 0)


TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class ExploitSubprocessTests(unittest.TestCase):
    def run_in_copy(
        self,
        task_workspace: Path,
        solution_name: str,
        solution_text: str,
        command: tuple[str, ...],
    ) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / "ws"
            shutil.copytree(task_workspace, workspace)
            (workspace / solution_name).write_text(solution_text)
            completed = subprocess.run(  # noqa: S603 - trusted interpreter
                list(command),
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
        return completed.returncode, completed.stdout + completed.stderr

    def test_python_exit_zero_exploit_scores_zero(self) -> None:
        exploit = "import os\nos._exit(0)\n\n\ndef fizzbuzz(value):\n    return ''\n"
        returncode, output = self.run_in_copy(
            TASKS_DIR / "fizzbuzz-fix" / "workspace",
            "fizzbuzz.py",
            exploit,
            ("python3", "test_fizzbuzz.py"),
        )
        self.assertEqual(returncode, 0)
        reward, partial, evidence = _score_test_run(returncode, output)
        self.assertEqual(reward, 0.0)
        self.assertIsNone(partial)
        self.assertEqual(evidence, 0)

    def test_node_uncaught_exception_exploit_scores_zero(self) -> None:
        stub = (TASKS_DIR / "js-deep-get" / "workspace" / "deep_get.mjs").read_text()
        exploit = "process.on('uncaughtException', () => process.exit(0));\n" + stub
        returncode, output = self.run_in_copy(
            TASKS_DIR / "js-deep-get" / "workspace",
            "deep_get.mjs",
            exploit,
            ("node", "test_deep_get.mjs"),
        )
        self.assertEqual(returncode, 0)
        reward, partial, evidence = _score_test_run(returncode, output)
        self.assertEqual(reward, 0.0)
        self.assertIsNone(partial)
        self.assertEqual(evidence, 0)

    def test_honest_solve_scores_one(self) -> None:
        solution = (
            "def fizzbuzz(value):\n"
            "    if value % 15 == 0:\n"
            "        return 'FizzBuzz'\n"
            "    if value % 3 == 0:\n"
            "        return 'Fizz'\n"
            "    if value % 5 == 0:\n"
            "        return 'Buzz'\n"
            "    return str(value)\n"
        )
        returncode, output = self.run_in_copy(
            TASKS_DIR / "fizzbuzz-fix" / "workspace",
            "fizzbuzz.py",
            solution,
            ("python3", "test_fizzbuzz.py"),
        )
        self.assertEqual(returncode, 0)
        reward, partial, evidence = _score_test_run(returncode, output)
        self.assertEqual(reward, 1.0)
        self.assertEqual(evidence, 10)


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
            task = make_task(workspace, ("python3", "test_app.py", "cases.txt"))

            protected = _protected_files(task.workspace, task.test_command)

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

            protected = _protected_files(task.workspace, task.test_command)

        self.assertEqual(protected, ())


class ChangedProtectedFilesTests(unittest.TestCase):
    def make_pristine(self, root: Path) -> Path:
        pristine = root / "pristine"
        pristine.mkdir()
        (pristine / "app.py").write_text("answer = 0\n")
        (pristine / "test_app.py").write_text("import app\nassert app.answer == 42\n")
        return pristine

    def test_truncated_test_file_is_caught_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine = self.make_pristine(root)
            task = make_task(pristine, ("python3", "test_app.py"))
            protected = _protected_files(task.workspace, task.test_command)
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
            digests = _file_digests(
                pristine, _protected_files(task.workspace, task.test_command)
            )
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
            digests = _file_digests(
                pristine, _protected_files(task.workspace, task.test_command)
            )
            episode = root / "ws"
            shutil.copytree(pristine, episode)
            (episode / "app.py").write_text("answer = 42\n")

            changed = _changed_protected_files(episode, digests)

        self.assertEqual(changed, ())


class TestEnvironmentTests(unittest.TestCase):
    def test_test_process_gets_no_host_configuration(self) -> None:
        environment = _test_environment(Path("/ep/home"), Path("/ep/tmp"))

        self.assertEqual(environment["HOME"], "/ep/home")
        self.assertEqual(environment["TMPDIR"], "/ep/tmp")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        for name in environment:
            self.assertNotRegex(name, r"(_KEY|_TOKEN|_SECRET|_PASSWORD)$")


class BaselinePassedTests(unittest.TestCase):
    def test_passing_baseline_with_evidence_blocks_the_episode(self) -> None:
        self.assertTrue(_baseline_passed(0, "all 10 cases passed\n"))

    def test_exit_zero_without_evidence_does_not_block(self) -> None:
        self.assertFalse(_baseline_passed(0, ""))

    def test_failing_baseline_does_not_block(self) -> None:
        self.assertFalse(_baseline_passed(1, "2 of 8 cases failed\n"))


HOOK_NAMES = (
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "tox.ini",
    "sitecustomize.py",
    "usercustomize.py",
    "inject.pth",
)


class EvalOverlayTests(unittest.TestCase):
    def build_workspaces(self, root: Path) -> tuple[Path, Path, tuple[str, ...]]:
        pristine = root / "pristine"
        pristine.mkdir()
        (pristine / "app.py").write_text("answer = 0\n")
        (pristine / "test_app.py").write_text(
            "import app\nassert app.answer == 42\nprint('all 1 cases passed')\n"
        )
        episode = root / "ws"
        shutil.copytree(pristine, episode)
        (episode / "app.py").write_text("answer = 42\n")
        (episode / "helper.py").write_text("VALUE = 1\n")
        (episode / "pkg").mkdir()
        (episode / "pkg" / "util.py").write_text("VALUE = 2\n")
        (episode / "__pycache__").mkdir()
        (episode / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00")
        for name in HOOK_NAMES:
            (episode / name).write_text("print('all 99 cases passed')\n")
            (episode / "pkg" / name).write_text("print('all 99 cases passed')\n")
        task = make_task(pristine, ("python3", "test_app.py"))
        return pristine, episode, _protected_files(task.workspace, task.test_command)

    def test_overlay_excludes_hooks_and_protected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine, episode, protected = self.build_workspaces(root)

            overlay, refused = _overlay_files(pristine, episode, protected)
            self.assertEqual(refused, ())

        self.assertEqual(overlay, ("app.py", "helper.py", "pkg/util.py"))

    def test_eval_dir_keeps_pristine_tests_and_drops_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine, episode, protected = self.build_workspaces(root)
            # The agent also tampers with the protected test file
            # inside its workspace; the eval copy must not see it.
            (episode / "test_app.py").write_text("print('all 99 cases passed')\n")

            refused = _build_eval_dir(root / "eval", pristine, episode, protected)
            eval_dir = root / "eval"
            self.assertEqual(refused, ())

            self.assertEqual(
                (eval_dir / "test_app.py").read_text(),
                (pristine / "test_app.py").read_text(),
            )
            self.assertEqual((eval_dir / "app.py").read_text(), "answer = 42\n")
            self.assertEqual((eval_dir / "helper.py").read_text(), "VALUE = 1\n")
            for name in HOOK_NAMES:
                self.assertEqual(list(eval_dir.rglob(name)), [])
            self.assertEqual(list(eval_dir.rglob("*.pyc")), [])


SPAWN_AND_HANG = (
    "import subprocess, time\n"
    "child = subprocess.Popen(['/bin/sleep', '30'])\n"
    "print(child.pid, flush=True)\n"
    "time.sleep(30)\n"
)

SPAWN_AND_EXIT = (
    "import subprocess\n"
    "child = subprocess.Popen(['/bin/sleep', '30'])\n"
    "print(child.pid, flush=True)\n"
)


class RunGroupedTests(unittest.TestCase):
    def assert_process_gone(self, pid: int) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
        self.fail(f"process {pid} is still alive")

    def test_timeout_kills_the_whole_process_group(self) -> None:
        result = _run_grouped(
            ("python3", "-c", SPAWN_AND_HANG),
            env=_test_environment(Path("/ep/home"), Path("/ep/tmp")),
            timeout=2,
        )

        self.assertTrue(result.timed_out)
        helper_pid = int(result.stdout.strip())
        self.assert_process_gone(helper_pid)

    def test_normal_exit_still_kills_background_helpers(self) -> None:
        result = _run_grouped(
            ("python3", "-c", SPAWN_AND_EXIT),
            env=_test_environment(Path("/ep/home"), Path("/ep/tmp")),
            timeout=30,
        )

        self.assertFalse(result.timed_out)
        self.assertEqual(result.returncode, 0)
        helper_pid = int(result.stdout.strip())
        self.assert_process_gone(helper_pid)


class FindSessionFileTests(unittest.TestCase):
    def test_earliest_file_wins_even_after_late_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session_dir = Path(temporary_directory)
            real = session_dir / "real.jsonl"
            real.write_text('{"role": "system"}\n')
            time.sleep(0.05)
            forged_dir = session_dir / "sub"
            forged_dir.mkdir()
            forged = forged_dir / "forged.jsonl"
            forged.write_text('{"role": "assistant"}\n')
            if hasattr(real.stat(), "st_birthtime"):
                # omp appends to the real session until the episode
                # ends, so its last write is the newest of all.
                late = time.time() + 100
                os.utime(real, (late, late))

            self.assertEqual(_find_session_file(session_dir), real)

    def test_empty_session_dir_gives_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertIsNone(_find_session_file(Path(temporary_directory)))


class RunEpisodeBaselineTests(unittest.TestCase):
    def run_episode_no_sandbox(self, task: TaskSpec, runs_dir: Path) -> EpisodeFailure:
        """run_episode with sandboxing off: deterministic on any host."""
        with unittest.mock.patch.dict(os.environ, {"OMP_GYM_SANDBOX": "0"}):
            result = run_episode(task, runs_dir, model=None)
        self.assertIsInstance(result, EpisodeFailure)
        assert isinstance(result, EpisodeFailure)
        return result

    def make_baseline_task(self, root: Path, test_source: str) -> tuple[TaskSpec, Path]:
        workspace = root / "task"
        workspace.mkdir()
        (workspace / "app.py").write_text("answer = 42\n")
        (workspace / "test_app.py").write_text(test_source)
        task = make_task(workspace, ("python3", "test_app.py"))
        return task, root / "runs"

    def test_already_passing_task_fails_before_omp_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task, runs_dir = self.make_baseline_task(
                root,
                "import app\nassert app.answer == 42\nprint('all 1 cases passed')\n",
            )

            result = self.run_episode_no_sandbox(task, runs_dir)

            self.assertEqual(
                result.reason,
                "invalid task: already passes before the agent runs",
            )
            # The gate fires before omp starts, so no episode
            # artifacts beyond the baseline log exist.
            episode_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(episode_dirs), 1)
            self.assertFalse((episode_dirs[0] / "ws").exists())
            self.assertFalse((episode_dirs[0] / "prompt.txt").exists())

    def test_baseline_exit_zero_without_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task, runs_dir = self.make_baseline_task(root, "print('nothing ran')\n")

            result = self.run_episode_no_sandbox(task, runs_dir)

            self.assertEqual(
                result.reason,
                "invalid task: baseline exited 0 without test evidence",
            )
            episode_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(episode_dirs), 1)
            self.assertFalse((episode_dirs[0] / "ws").exists())

    def test_baseline_without_case_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task, runs_dir = self.make_baseline_task(
                root,
                "import sys\nprint('boom')\nsys.exit(1)\n",
            )

            result = self.run_episode_no_sandbox(task, runs_dir)

            self.assertEqual(
                result.reason,
                "invalid task: baseline output does not report a "
                "case count; set expected_cases in task.toml",
            )
            episode_dirs = list(runs_dir.iterdir())
            self.assertEqual(len(episode_dirs), 1)
            self.assertFalse((episode_dirs[0] / "ws").exists())


if __name__ == "__main__":
    unittest.main()
