import asyncio
import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock


class _TimeSpan:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0


class _Timing:
    def __init__(self) -> None:
        self.generation = _TimeSpan()


class _State(dict):
    pass


class _Client:
    pass


class _Rubric:
    def __init__(self, funcs) -> None:
        self.funcs = funcs


class _Environment:
    def __init__(self, dataset, rubric, **kwargs) -> None:
        self.dataset = dataset
        self.rubric = rubric

    async def init_state(self, input, client, model, sampling_args):
        return _State(
            input=input,
            prompt=input["prompt"],
            answer=input.get("answer", ""),
            client=client,
            model=model,
            sampling_args=sampling_args,
            timing=_Timing(),
            completion=None,
            reward=None,
            metrics=None,
            is_completed=False,
            is_truncated=False,
        )

    async def cleanup(self, state) -> None:
        return None


class _Dataset:
    @classmethod
    def from_list(cls, rows):
        return rows


def _normalize_messages(messages, *, field_name):
    return messages


def _load_prime_adapter():
    verifiers = types.ModuleType("verifiers")
    verifiers.__path__ = []
    verifiers.Client = _Client
    verifiers.Environment = _Environment
    verifiers.Rubric = _Rubric
    verifiers.State = _State

    verifiers_types = types.ModuleType("verifiers.types")
    verifiers_types.RolloutInput = dict
    verifiers_types.SamplingArgs = dict
    verifiers_utils = types.ModuleType("verifiers.utils")
    verifiers_utils.__path__ = []
    message_utils = types.ModuleType("verifiers.utils.message_utils")
    message_utils.normalize_messages = _normalize_messages

    datasets = types.ModuleType("datasets")
    datasets.Dataset = _Dataset
    module_path = Path("environments") / "omp-coding" / "omp_coding" / "__init__.py"
    spec = importlib.util.spec_from_file_location("omp_coding_under_test", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load the Prime adapter module")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "verifiers": verifiers,
            "verifiers.types": verifiers_types,
            "verifiers.utils": verifiers_utils,
            "verifiers.utils.message_utils": message_utils,
            "datasets": datasets,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _write_task(tasks_dir: Path, name: str = "task-slug") -> Path:
    task_dir = tasks_dir / name
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "solution.py").write_text("VALUE = 0\n")
    (task_dir / "task.toml").write_text(
        'prompt = "Implement the requested function."\n'
        'test_command = ["python3", "solution.py"]\n'
    )
    return task_dir


def _write_session(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "Fix it."}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call-1",
                                    "name": "read",
                                    "arguments": {"path": "app.py"},
                                }
                            ],
                        },
                    }
                ),
            )
        )
        + "\n"
    )


class PrimeContractTests(unittest.TestCase):
    def test_rollout_has_the_verifiers_v030_signature(self) -> None:
        adapter = _load_prime_adapter()

        signature = inspect.signature(adapter.OmpCodingEnv.rollout)

        self.assertEqual(
            list(signature.parameters),
            ["self", "input", "client", "model", "sampling_args"],
        )
        self.assertIsNone(signature.parameters["sampling_args"].default)

    def test_packaged_dataset_has_all_public_tasks(self) -> None:
        adapter = _load_prime_adapter()

        environment = adapter.OmpCodingEnv()

        task_names = {row["answer"] for row in environment.dataset}
        self.assertEqual(len(task_names), 18)
        self.assertIn("fizzbuzz-fix", task_names)
        self.assertIn("js-router-tree", task_names)
        self.assertEqual(len(environment.rubric.funcs), 1)

    def test_dataset_uses_the_task_prompt_and_slug_answer(self) -> None:
        adapter = _load_prime_adapter()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tasks_dir = Path(temporary_directory)
            _write_task(tasks_dir)

            environment = adapter.OmpCodingEnv(tasks_dir=tasks_dir)

        row = environment.dataset[0]
        self.assertEqual(row["answer"], "task-slug")
        self.assertEqual(
            row["prompt"],
            [
                {
                    "role": "user",
                    "content": "Implement the requested function.",
                }
            ],
        )

    def test_policy_config_is_temporary_and_keeps_secret_out_of_file(
        self,
    ) -> None:
        adapter = _load_prime_adapter()

        with adapter.policy_environment(
            "https://policy.example/v1/",
            "model-id",
            "POLICY_KEY",
            "private-value",
            {"X-Team": "team"},
        ) as environment:
            models_file = Path(environment["OMP_MODELS"])
            document = json.loads(models_file.read_text())
            provider = document["providers"]["omp-gym"]
            self.assertEqual(provider["baseUrl"], "https://policy.example/v1")
            self.assertEqual(provider["apiKey"], "POLICY_KEY")
            self.assertEqual(models_file.stat().st_mode & 0o077, 0)
            self.assertEqual(provider["headers"], {"X-Team": "team"})
            self.assertNotIn("private-value", models_file.read_text())
            self.assertEqual(environment["POLICY_KEY"], "private-value")

        self.assertFalse(models_file.exists())


class PrimeRolloutTests(unittest.TestCase):
    def test_success_returns_state_and_uses_binary_episode_reward(self) -> None:
        adapter = _load_prime_adapter()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks_dir = root / "tasks"
            _write_task(tasks_dir)
            session = root / "session.jsonl"
            _write_session(session)
            environment = adapter.OmpCodingEnv(tasks_dir=tasks_dir)
            record = adapter.EpisodeRecord(
                task="task-slug",
                model="omp-gym/policy",
                episode_dir=str(root / "episode"),
                session_file=str(session),
                omp_exit_code=0,
                test_exit_code=0,
                reward=1.0,
                reward_partial=0.25,
                duration_seconds=1.0,
                reward_improvement=0.5,
            )
            client = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    api_base_url="http://127.0.0.1:8000/v1",
                    api_key_var="POLICY_KEY",
                    extra_headers={"X-Run": "one"},
                )
            )
            input_row = {
                "prompt": [{"role": "user", "content": "Use this prompt."}],
                "answer": "task-slug",
                "example_id": 0,
            }
            with (
                mock.patch.dict(
                    adapter.os.environ,
                    {"POLICY_KEY": "secret"},
                ),
                mock.patch.object(
                    adapter,
                    "policy_environment",
                    return_value=nullcontext({"OPENAI_BASE_URL": "stub"}),
                ) as build_environment,
                mock.patch.object(adapter, "run_episode", return_value=record) as run,
            ):
                state = asyncio.run(
                    environment.rollout(input_row, client, "policy", {"n": 1})
                )

        task = run.call_args.args[0]
        self.assertEqual(task.prompt, "Use this prompt.")
        self.assertEqual(run.call_args.args[2], "omp-gym/policy")
        self.assertEqual(run.call_args.args[3], {"OPENAI_BASE_URL": "stub"})
        build_environment.assert_called_once_with(
            "http://127.0.0.1:8000/v1",
            "policy",
            "POLICY_KEY",
            "secret",
            {"X-Run": "one"},
        )
        self.assertEqual(
            run.call_args.kwargs["extra_secret_names"],
            ("POLICY_KEY",),
        )
        self.assertIsInstance(state, _State)
        self.assertEqual(state["reward"], 1.0)
        self.assertEqual(state["episode_reward"], 1.0)
        self.assertEqual(state["episode_result"]["status"], "success")
        self.assertTrue(state["is_completed"])
        self.assertFalse(state["is_truncated"])
        self.assertEqual(state["completion"][0]["role"], "assistant")
        self.assertEqual(environment.episode_reward(state), 1.0)

    def test_episode_failure_is_a_completed_zero_reward_state(self) -> None:
        adapter = _load_prime_adapter()
        with tempfile.TemporaryDirectory() as temporary_directory:
            tasks_dir = Path(temporary_directory)
            _write_task(tasks_dir)
            environment = adapter.OmpCodingEnv(tasks_dir=tasks_dir)
            failure = adapter.EpisodeFailure(
                task="task-slug",
                reason="provider endpoint connection failed",
            )
            client = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    api_base_url="http://127.0.0.1:8000/v1",
                    api_key_var="POLICY_KEY",
                    extra_headers={},
                )
            )
            input_row = {
                "prompt": "Implement the requested function.",
                "answer": "task-slug",
                "example_id": 0,
            }
            with (
                mock.patch.dict(
                    adapter.os.environ,
                    {"POLICY_KEY": "secret"},
                ),
                mock.patch.object(
                    adapter,
                    "policy_environment",
                    return_value=nullcontext({"OPENAI_BASE_URL": "stub"}),
                ),
                mock.patch.object(adapter, "run_episode", return_value=failure),
            ):
                state = asyncio.run(environment.rollout(input_row, client, "policy"))

        self.assertEqual(state["reward"], 0.0)
        self.assertEqual(state["episode_reward"], 0.0)
        self.assertEqual(state["episode_result"]["status"], "failure")
        self.assertEqual(state["episode_result"]["failure_class"], "provider_error")
        self.assertIn("endpoint", state["episode_result"]["reason"])
        self.assertTrue(state["is_completed"])
        self.assertEqual(state["completion"][0]["role"], "assistant")

    def test_completion_excludes_the_input_prompt_and_keeps_tool_calls(self) -> None:
        adapter = _load_prime_adapter()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = Path(temporary_directory) / "session.jsonl"
            _write_session(session)

            messages, error = adapter.episode_messages(str(session))

        self.assertIsNone(error)
        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "read")


if __name__ == "__main__":
    unittest.main()
