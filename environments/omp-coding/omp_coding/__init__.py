"""omp-coding: verifiers evaluation environment backed by the omp harness.

This package is an evaluation adapter. Each rollout runs one real
omp episode: the policy model drives omp's tool surface (read, bash,
edit, write, grep, glob) inside a fresh copy of the task workspace,
and the task's test command produces the reward.

This environment is for Verifiers evaluation. It is not a
policy-gradient training backend. omp runs as a separate process, so
trainer sampling arguments do not control its requests. Use the
``omp-gym export`` command to create training data from scored
episodes.

The Verifiers client supplies the policy endpoint. A small, generated
omp model file connects the episode process to that endpoint.
"""

import asyncio
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import verifiers as vf
from datasets import Dataset
from verifiers.types import RolloutInput, SamplingArgs
from verifiers.utils.message_utils import normalize_messages

from omp_gym.runner import EpisodeFailure, EpisodeRecord, run_episode
from omp_gym.task import TaskLoadError, TaskSpec, load_task
from omp_gym.trajectory import (
    AssistantStep,
    ToolResultStep,
    Trajectory,
    UserStep,
    parse_session,
)

DEFAULT_TASKS_DIR = Path(__file__).parent / "tasks"


FailureClass = Literal[
    "provider_error",
    "baseline_timeout",
    "test_timeout",
    "no_session",
    "invalid_task",
    "sandbox",
    "other",
]


@contextmanager
def policy_environment(
    policy_url: str,
    model: str,
    api_key_var: str,
    api_key: str,
    headers: Mapping[str, str],
) -> Iterator[dict[str, str]]:
    """Create an isolated omp model file for one rollout."""
    document = {
        "providers": {
            "omp-gym": {
                "baseUrl": policy_url.rstrip("/"),
                "apiKey": api_key_var,
                "auth": "apiKey",
                "api": "openai-completions",
                "models": [
                    {
                        "id": model,
                        "name": f"Verifiers policy {model}",
                        "contextWindow": 32768,
                        "maxTokens": 4096,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }
    provider = document["providers"]["omp-gym"]
    if headers:
        provider["headers"] = dict(headers)
    with TemporaryDirectory(prefix="omp-verifiers-") as temporary_directory:
        models_file = Path(temporary_directory) / "models.yml"
        models_file.write_text(json.dumps(document))
        models_file.chmod(0o600)
        episode_environment = {
            "OMP_MODELS": str(models_file),
            "OMP_PROVIDER": "omp-gym",
            "OMP_MODEL": f"omp-gym/{model}",
            api_key_var: api_key,
        }
        yield episode_environment


def failure_class(reason: str) -> FailureClass:
    """Classify one EpisodeFailure reason into a stable failure_class.

    The runner phrases reasons in plain English; this adapter maps the
    phrasing onto a small enum so training pipelines can group failures
    without parsing prose.
    """
    lowered = reason.lower()
    if "baseline" in lowered and "timed out" in lowered:
        return "baseline_timeout"
    if "timed out" in lowered or "timeout" in lowered:
        return "test_timeout"
    if "no session" in lowered or "session" in lowered:
        return "no_session"
    if "sandbox" in lowered or "seatbelt" in lowered or "resource limit" in lowered:
        return "sandbox"
    if "provider" in lowered or "endpoint" in lowered or "connection" in lowered:
        return "provider_error"
    if "invalid task" in lowered or "invalid" in lowered:
        return "invalid_task"
    return "other"


def trajectory_to_messages(
    trajectory: Trajectory,
) -> list[dict[str, object]]:
    """Convert a parsed omp session into OpenAI-style chat messages.

    User steps become user messages. Assistant steps become assistant
    messages whose tool calls use the OpenAI tool_calls shape; the
    visible text is the message content. Tool result steps become tool
    messages keyed by the originating call id.
    """
    messages: list[dict[str, object]] = []
    for step in trajectory.steps:
        if isinstance(step, UserStep):
            messages.append({"role": "user", "content": step.text})
        elif isinstance(step, AssistantStep):
            message: dict[str, object] = {
                "role": "assistant",
                "content": step.text,
            }
            if step.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in step.tool_calls
                ]
            messages.append(message)
        elif isinstance(step, ToolResultStep):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": step.call_id,
                    "name": step.tool_name,
                    "content": step.text,
                }
            )
    return messages


def episode_messages(
    session_file: str,
) -> tuple[list[dict[str, object]], str | None]:
    """Build completion messages from one recorded omp session.

    The first user message duplicates the Verifiers input prompt, so it
    is not part of the returned completion.
    """
    try:
        trajectory = parse_session(Path(session_file))
    except OSError:
        trajectory = None
    if trajectory is not None:
        messages = trajectory_to_messages(trajectory)
        if messages and messages[0].get("role") == "user":
            messages = messages[1:]
        if messages:
            return messages, None
    return (
        [
            {
                "role": "assistant",
                "content": f"omp episode complete; session: {session_file}",
            }
        ],
        "session_parse",
    )


def _completion_text(prompt: object) -> str:
    """Flatten a verifiers prompt value into plain text."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, Sequence):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
        return "\n\n".join(parts)
    return ""


class OmpCodingEnv(vf.Environment):
    """Verifiers environment that runs one omp episode per rollout."""

    def __init__(
        self,
        tasks_dir: Path = DEFAULT_TASKS_DIR,
        runs_dir: Path = Path("vf-runs"),
        **kwargs: object,
    ) -> None:
        task_paths = sorted(
            (path for path in tasks_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        loaded_tasks: list[TaskSpec] = []
        for task_path in task_paths:
            loaded = load_task(task_path)
            if isinstance(loaded, TaskLoadError):
                raise ValueError(loaded.reason)
            loaded_tasks.append(loaded)
        dataset = Dataset.from_list(
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": loaded.prompt,
                        }
                    ],
                    "answer": loaded.name,
                }
                for loaded in loaded_tasks
            ]
        )
        self._tasks_dir = tasks_dir
        self._runs_dir = runs_dir
        rubric = vf.Rubric(funcs=[self.episode_reward])
        super().__init__(dataset=dataset, rubric=rubric, **kwargs)

    @staticmethod
    def episode_reward(state: vf.State) -> float:
        """Return the reward from the completed omp episode."""
        value = state.get("episode_reward")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("episode_reward is not a number")
        return float(value)

    @staticmethod
    def _policy_config(
        client: vf.Client,
    ) -> tuple[str, str, str, dict[str, str]]:
        config = client.config
        if config is None:
            raise RuntimeError("the Verifiers client has no endpoint configuration")
        api_key = os.environ.get(config.api_key_var)
        if not api_key:
            raise RuntimeError(
                f"the policy key variable {config.api_key_var} is not set"
            )
        return (
            config.api_base_url,
            config.api_key_var,
            api_key,
            dict(config.extra_headers),
        )

    def _finish_failure(
        self,
        state: vf.State,
        failure: EpisodeFailure,
    ) -> None:
        state["completion"] = normalize_messages(
            [
                {
                    "role": "assistant",
                    "content": f"omp episode failed: {failure.reason}",
                }
            ],
            field_name="completion",
        )
        state["episode_reward"] = 0.0
        state["reward"] = 0.0
        state["episode_result"] = {
            "status": "failure",
            "task": failure.task,
            "failure_class": failure_class(failure.reason),
            "reason": failure.reason,
        }
        state["is_completed"] = True
        state["is_truncated"] = False

    def _finish_success(
        self,
        state: vf.State,
        record: EpisodeRecord,
    ) -> None:
        messages, completion_error = episode_messages(record.session_file)
        state["completion"] = normalize_messages(
            messages,
            field_name="completion",
        )
        state["episode_reward"] = record.reward
        state["reward"] = record.reward
        episode_result: dict[str, object] = {
            "status": "success",
            "record": asdict(record),
        }
        if completion_error is not None:
            episode_result["completion_error"] = completion_error
        state["episode_result"] = episode_result
        state["is_completed"] = True
        state["is_truncated"] = False

    async def rollout(
        self,
        input: RolloutInput,
        client: vf.Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> vf.State:
        """Run one complete omp episode against the policy endpoint."""
        state = await self.init_state(input, client, model, sampling_args)
        timing = state["timing"]
        timing.generation.start = time.time()
        try:
            answer = input.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("the rollout input has no task name")
            loaded = load_task(self._tasks_dir / answer)
            if isinstance(loaded, TaskLoadError):
                self._finish_failure(
                    state,
                    EpisodeFailure(task=answer, reason=loaded.reason),
                )
                return state

            prompt_text = _completion_text(input.get("prompt"))
            if prompt_text and prompt_text != loaded.prompt:
                loaded = replace(loaded, prompt=prompt_text)

            policy_url, api_key_var, api_key, headers = self._policy_config(client)
            with policy_environment(
                policy_url,
                model,
                api_key_var,
                api_key,
                headers,
            ) as episode_environment:
                result = await asyncio.to_thread(
                    run_episode,
                    loaded,
                    self._runs_dir,
                    f"omp-gym/{model}",
                    episode_environment,
                    extra_secret_names=(api_key_var,),
                )
            if isinstance(result, EpisodeFailure):
                self._finish_failure(state, result)
            else:
                self._finish_success(state, result)
            return state
        finally:
            timing.generation.end = time.time()
            await self.cleanup(state)


def load_environment(
    tasks_dir: str | None = None,
    runs_dir: str = "vf-runs",
    **kwargs: object,
) -> vf.Environment:
    """Entrypoint required by the verifiers spec."""
    return OmpCodingEnv(
        tasks_dir=Path(tasks_dir) if tasks_dir else DEFAULT_TASKS_DIR,
        runs_dir=Path(runs_dir),
        **kwargs,
    )
