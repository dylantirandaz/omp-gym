"""Episode runner.

One episode: copy the task workspace, run omp on the prompt without
supervision, then run the task tests. The test result is the reward.
"""

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .envfile import load_env_file
from .task import TaskSpec


@dataclass(frozen=True)
class EpisodeRecord:
    """Result of one completed episode."""

    task: str
    model: str
    episode_dir: str
    session_file: str
    omp_exit_code: int
    test_exit_code: int
    reward: float
    duration_seconds: float


@dataclass(frozen=True)
class EpisodeFailure:
    """The episode did not produce a usable session."""

    task: str
    reason: str


def _find_session_file(session_dir: Path) -> Path | None:
    """Find the newest session JSONL file below the session directory."""
    candidates = sorted(
        session_dir.rglob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    return candidates[-1]


def run_episode(
    task: TaskSpec,
    runs_dir: Path,
    model: str | None,
) -> EpisodeRecord | EpisodeFailure:
    """Run one real omp session on the task and score the result."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    episode_dir = (runs_dir / f"{task.name}-{stamp}").resolve()
    workspace = episode_dir / "ws"
    session_dir = episode_dir / "sess"
    episode_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(task.workspace, workspace)
    session_dir.mkdir()

    command = [
        "omp",
        "-p",
        task.prompt,
        "--cwd",
        str(workspace),
        "--session-dir",
        str(session_dir),
        "--mode",
        "json",
        "--auto-approve",
        "--no-extensions",
        "--no-skills",
        "--no-rules",
        "--no-title",
        "--tools",
        task.tools,
        "--max-time",
        task.max_time,
    ]
    if model is not None:
        command.extend(["--model", model])

    started = time.monotonic()
    omp_run = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=int(task.max_time) + 120,
        env={**os.environ, **load_env_file(Path(".env"))},
    )
    duration = time.monotonic() - started
    (episode_dir / "events.jsonl").write_text(omp_run.stdout)
    if omp_run.stderr:
        (episode_dir / "stderr.log").write_text(omp_run.stderr)

    session_file = _find_session_file(session_dir)
    if session_file is None:
        return EpisodeFailure(
            task=task.name,
            reason=(
                f"omp exited with {omp_run.returncode} "
                "and wrote no session"
            ),
        )

    test_run = subprocess.run(
        list(task.test_command),
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (episode_dir / "test_output.log").write_text(
        test_run.stdout + test_run.stderr
    )

    record = EpisodeRecord(
        task=task.name,
        model=model if model is not None else "default",
        episode_dir=str(episode_dir),
        session_file=str(session_file),
        omp_exit_code=omp_run.returncode,
        test_exit_code=test_run.returncode,
        reward=1.0 if test_run.returncode == 0 else 0.0,
        duration_seconds=round(duration, 1),
    )
    (episode_dir / "episode.json").write_text(
        json.dumps(asdict(record), indent=2)
    )
    return record
