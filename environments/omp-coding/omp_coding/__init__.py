"""omp-coding: verifiers environment backed by the omp harness.

Each rollout is a real omp episode: the policy model drives omp's
tool surface (read, bash, edit, write, grep, glob) inside a fresh
workspace copy, and the task's test command produces the reward.

The policy endpoint comes from the verifiers client. omp discovers
it through its implicit LM Studio provider path (keyless,
openai-completions), so no omp configuration is needed.
"""

import asyncio
from pathlib import Path

from datasets import Dataset

import verifiers as vf
from omp_gym.runner import EpisodeFailure, run_episode
from omp_gym.task import TaskLoadError, load_task

DEFAULT_TASKS_DIR = Path(__file__).parent / "tasks"


def _episode_sync(
    task_dir: Path,
    runs_dir: Path,
    model: str,
    base_url: str,
) -> tuple[float, str]:
    """Run one omp episode against the policy endpoint."""
    loaded = load_task(task_dir)
    if isinstance(loaded, TaskLoadError):
        raise RuntimeError(f"bad task {loaded.path}: {loaded.reason}")
    result = run_episode(
        loaded,
        runs_dir,
        f"lm-studio/{model}",
        extra_env={"LM_STUDIO_BASE_URL": base_url},
    )
    if isinstance(result, EpisodeFailure):
        return 0.0, f"episode failed: {result.reason}"
    return result.reward, result.session_file


class OmpCodingEnv(vf.MultiTurnEnv):
    """Multi-turn coding environment driven by real omp episodes."""

    def __init__(
        self,
        tasks_dir: Path = DEFAULT_TASKS_DIR,
        runs_dir: Path = Path("vf-runs"),
        **kwargs,
    ) -> None:
        task_dirs = sorted(
            entry
            for entry in tasks_dir.iterdir()
            if (entry / "task.toml").is_file()
        )
        if not task_dirs:
            raise RuntimeError(f"no tasks found in {tasks_dir}")
        records = []
        for task_dir in task_dirs:
            loaded = load_task(task_dir)
            if isinstance(loaded, TaskLoadError):
                raise RuntimeError(
                    f"bad task {loaded.path}: {loaded.reason}"
                )
            records.append(
                {
                    "question": loaded.prompt,
                    "answer": loaded.name,
                    "info": {"task_dir": str(task_dir)},
                }
            )
        self._runs_dir = runs_dir
        rubric = vf.Rubric(funcs=[self._test_reward], weights=[1.0])
        super().__init__(
            dataset=Dataset.from_list(records),
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )

    def _test_reward(self, state, **kwargs) -> float:
        """The reward is the task test result recorded at rollout."""
        return float(state.get("reward", 0.0))

    async def is_completed(self, messages, state, **kwargs) -> bool:
        return True

    async def env_response(self, messages, state, **kwargs):
        return []

    async def rollout(self, client, model, prompt, answer="",
                      task="default", info=None, sampling_args=None,
                      **kwargs):
        """Run one omp episode with the policy as omp's model."""
        info = info or {}
        task_dir = Path(info["task_dir"])
        base_url = str(client.base_url).rstrip("/")
        reward, session_file = await asyncio.to_thread(
            _episode_sync, task_dir, self._runs_dir, model, base_url
        )
        state = {
            "reward": reward,
            "session_file": session_file,
            "task": answer,
        }
        completion = [
            {
                "role": "assistant",
                "content": (
                    f"omp episode complete; session: {session_file}"
                ),
            }
        ]
        return completion, state


def load_environment(
    tasks_dir: str | None = None,
    runs_dir: str = "vf-runs",
    **kwargs,
) -> vf.Environment:
    """Entrypoint required by the verifiers spec."""
    return OmpCodingEnv(
        tasks_dir=Path(tasks_dir) if tasks_dir else DEFAULT_TASKS_DIR,
        runs_dir=Path(runs_dir),
        **kwargs,
    )
