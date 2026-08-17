"""Task specification.

A task is a directory with a `task.toml` file and a `workspace/`
directory. The workspace is the initial state of the episode.
"""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ALLOWED_TEST_RUNNERS = frozenset({"python3", "python", "node", "pytest"})

MAX_TIME_SECONDS = 4 * 60 * 60
MAX_EXPECTED_CASES = 100_000
MAX_WORKSPACE_FILES = 4000
MAX_WORKSPACE_BYTES = 256 * 1024 * 1024


def runner_kind(test_command: tuple[str, ...]) -> Literal["python", "node"]:
    """Return the validated test command's language family.

    The task loader rejects path-qualified executable names. This
    prevents an arbitrary executable named `python3` or `node` from
    passing the runner allowlist.
    """
    return "node" if test_command[0] == "node" else "python"


def workspace_digest(workspace: Path) -> str:
    """Content hash of every file below the workspace.

    The digest covers the sorted relative path list and each file's
    bytes, so any rename, edit, deletion, or addition changes it.
    Symlinks that point outside the workspace are rejected by the
    caller before this runs; here a symlink contributes its target
    text, never the target's bytes.
    """
    root = workspace.resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            digest.update(b"L\0" + relative.encode())
            digest.update(b"\0" + str(path.readlink()).encode() + b"\0")
            continue
        if not path.is_file():
            continue
        digest.update(b"F\0" + relative.encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def workspace_size_error(workspace: Path) -> str | None:
    """A reason when the workspace exceeds the accepted footprint."""
    files = 0
    total = 0
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        total += path.stat().st_size
        if files > MAX_WORKSPACE_FILES:
            return f"workspace has more than {MAX_WORKSPACE_FILES} files"
        if total > MAX_WORKSPACE_BYTES:
            return (
                f"workspace is larger than {MAX_WORKSPACE_BYTES // (1024 * 1024)} MiB"
            )
    return None


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
    expected_cases: int | None = None
    version: str = "1"
    source: str = ""
    license: str = ""


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

    try:
        raw = tomllib.loads(config_path.read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        return TaskLoadError(task_dir, f"task.toml is invalid: {error}")
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
    runner_name = test_command[0]
    if runner_name not in ALLOWED_TEST_RUNNERS:
        return TaskLoadError(
            task_dir,
            "test_command must start with one of: "
            + ", ".join(sorted(ALLOWED_TEST_RUNNERS)),
        )

    context_files = raw.get("context_files", [])
    if not isinstance(context_files, list) or not all(
        isinstance(relative_path, str) and relative_path
        for relative_path in context_files
    ):
        return TaskLoadError(task_dir, "context_files must be a list of strings")
    workspace_root = workspace.resolve()
    for relative_path in context_files:
        try:
            candidate = (workspace_root / relative_path).resolve()
            valid_candidate = (
                not Path(relative_path).is_absolute()
                and candidate.is_relative_to(workspace_root)
                and candidate.is_file()
            )
        except (OSError, RuntimeError, ValueError):
            valid_candidate = False
        if not valid_candidate:
            return TaskLoadError(
                task_dir,
                f"context file is not a workspace file: {relative_path}",
            )
    tools = raw.get("tools", "read,bash,edit,write,grep,glob")
    max_time = raw.get("max_time", "300")
    if not isinstance(tools, str) or not isinstance(max_time, str):
        return TaskLoadError(task_dir, "tools and max_time must be strings")
    max_time_is_bounded = (
        max_time.isdigit()
        and len(max_time) <= len(str(MAX_TIME_SECONDS))
        and 0 < int(max_time) <= MAX_TIME_SECONDS
    )
    if not max_time_is_bounded:
        return TaskLoadError(
            task_dir,
            f"max_time must be an integer from 1 to {MAX_TIME_SECONDS}",
        )
    expected_cases = raw.get("expected_cases")
    if expected_cases is not None and (
        isinstance(expected_cases, bool)
        or not isinstance(expected_cases, int)
        or not 0 < expected_cases <= MAX_EXPECTED_CASES
    ):
        return TaskLoadError(
            task_dir,
            f"expected_cases must be an integer from 1 to {MAX_EXPECTED_CASES}",
        )
    size_error = workspace_size_error(workspace)
    if size_error is not None:
        return TaskLoadError(task_dir, size_error)
    version = raw.get("version", "1")
    source = raw.get("source", "")
    task_license = raw.get("license", "")
    if not all(isinstance(field, str) for field in (version, source, task_license)):
        return TaskLoadError(task_dir, "version, source, and license must be strings")

    return TaskSpec(
        name=task_dir.name,
        prompt=prompt.strip(),
        test_command=tuple(test_command),
        tools=tools,
        max_time=max_time,
        context_files=tuple(context_files),
        expected_cases=expected_cases,
        version=version,
        source=source,
        license=task_license,
        workspace=workspace,
    )
