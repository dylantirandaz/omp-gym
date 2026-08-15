"""Task specification.

A task is a directory with a `task.toml` file and a `workspace/`
directory. The workspace is the initial state of the episode.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    """One reproducible task for the environment."""

    name: str
    prompt: str
    test_command: tuple[str, ...]
    tools: str
    max_time: str
    workspace: Path
    context_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskLoadError:
    """The task directory is not a valid task."""

    path: Path
    reason: str


def load_task(task_dir: Path) -> TaskSpec | TaskLoadError:
    """Load one task from a directory. Return an error value on bad input."""
    config_path = task_dir / "task.toml"
    workspace = task_dir / "workspace"
    if not config_path.is_file():
        return TaskLoadError(task_dir, "task.toml not found")
    if not workspace.is_dir():
        return TaskLoadError(task_dir, "workspace/ not found")

    raw = tomllib.loads(config_path.read_text())
    prompt = raw.get("prompt")
    test_command = raw.get("test_command")
    if not isinstance(prompt, str) or not prompt.strip():
        return TaskLoadError(task_dir, "prompt must be a non-empty string")
    if (
        not isinstance(test_command, list)
        or not test_command
        or not all(isinstance(part, str) for part in test_command)
    ):
        return TaskLoadError(
            task_dir, "test_command must be a non-empty list of strings"
        )

    context_files = raw.get("context_files", [])
    if not isinstance(context_files, list) or not all(
        isinstance(relative_path, str) and relative_path
        for relative_path in context_files
    ):
        return TaskLoadError(task_dir, "context_files must be a list of strings")
    workspace_root = workspace.resolve()
    for relative_path in context_files:
        candidate = (workspace_root / relative_path).resolve()
        if (
            Path(relative_path).is_absolute()
            or not candidate.is_relative_to(workspace_root)
            or not candidate.is_file()
        ):
            return TaskLoadError(
                task_dir,
                f"context file is not a workspace file: {relative_path}",
            )
    tools = raw.get("tools", "read,bash,edit,write,grep,glob")
    max_time = raw.get("max_time", "300")
    if not isinstance(tools, str) or not isinstance(max_time, str):
        return TaskLoadError(task_dir, "tools and max_time must be strings")

    return TaskSpec(
        name=task_dir.name,
        prompt=prompt.strip(),
        test_command=tuple(test_command),
        tools=tools,
        max_time=max_time,
        context_files=tuple(context_files),
        workspace=workspace,
    )
