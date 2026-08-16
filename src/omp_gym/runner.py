"""Episode runner.

One episode: copy the task workspace, run omp on the prompt without
supervision, then run the task tests. The test result is the reward.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path

from .envfile import load_env_file
from .export import SYSTEM_PROMPT, TASK_PROMPT_PREFIX
from .task import TaskSpec

_FAILED_PATTERN = re.compile(r"(\d+) of (\d+) cases failed")
_UNITTEST_RAN_PATTERN = re.compile(r"Ran (\d+) tests? in")
_UNITTEST_COUNT_PATTERN = re.compile(r"(?:failures|errors)=(\d+)")
_ALL_PASSED_PATTERN = re.compile(r"all (\d+) cases passed")
_PYTEST_PASSED_PATTERN = re.compile(r"(\d+) passed")
_EPISODE_HOST_ENV_NAMES = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "TMPDIR",
    "LANG",
    "TERM",
    "HF_HOME",
)


def _episode_environment(
    host_environment: Mapping[str, str],
    extra_env: dict[str, str] | None,
) -> dict[str, str]:
    """Build the minimal child environment for one episode.

    Episodes run real commands with auto-approval, so the child
    gets only basic host variables plus the provider keys from
    .env. The full parent environment, including unrelated
    credentials and parent OMP session state, stays out.
    """
    environment = {
        name: host_environment[name]
        for name in _EPISODE_HOST_ENV_NAMES
        if name in host_environment
    }
    environment.update(load_env_file(Path(".env")))
    environment.update(extra_env or {})
    return environment


def _episode_prompt(task: TaskSpec, workspace: Path) -> str:
    """Add the task's explicit source context to the episode prompt."""
    prompt = f"{TASK_PROMPT_PREFIX}\n\n{task.prompt}"
    if not task.context_files:
        return prompt
    sections = [prompt, "\n\nWorkspace context:"]
    for relative_path in task.context_files:
        content = (workspace / relative_path).read_text()
        sections.append(f"\n\nFile: {relative_path}\n{content}")
    return "".join(sections)


def _partial_credit(test_output: str) -> float | None:
    """Fraction of cases passed when the test reports a count.

    Three shapes are recognized: "N of M cases failed" (custom test
    scripts), unittest's "Ran N tests" with a FAILED counts line,
    and pytest's "X failed, Y passed" summary line. Others give
    None and only the binary reward applies.
    """
    found = _FAILED_PATTERN.search(test_output)
    if found:
        failed, total = int(found.group(1)), int(found.group(2))
        if total:
            return (total - failed) / total
    ran = _UNITTEST_RAN_PATTERN.search(test_output)
    if ran:
        total = int(ran.group(1))
        if total:
            failed = sum(
                int(count)
                for count in _UNITTEST_COUNT_PATTERN.findall(test_output)
            )
            return max(0, total - failed) / total
    for line in reversed(test_output.splitlines()):
        if "passed" not in line and "failed" not in line:
            continue
        failed_m = re.search(r"(\d+) failed", line)
        passed_m = re.search(r"(\d+) passed", line)
        failed = int(failed_m.group(1)) if failed_m else 0
        passed = int(passed_m.group(1)) if passed_m else 0
        total = failed + passed
        if total:
            return passed / total
    return None


def _test_evidence(test_output: str) -> int:
    """Count of test cases the output proves ran and passed.

    Three shapes give a positive count: a custom script's
    "all N cases passed", unittest's "Ran N tests" followed by
    "OK" with no "FAILED", and a pytest summary line "N passed"
    with no "failed" and no "error" token. All other output,
    including empty output, gives 0.
    """
    custom = _ALL_PASSED_PATTERN.search(test_output)
    if custom:
        return int(custom.group(1))
    ran = _UNITTEST_RAN_PATTERN.search(test_output)
    if ran:
        after_ran = test_output[ran.end() :]
        if "FAILED" not in test_output and re.search(r"\bOK\b", after_ran):
            return int(ran.group(1))
        return 0
    for line in reversed(test_output.splitlines()):
        if "passed" not in line:
            continue
        lowered = line.lower()
        if "failed" in lowered or "error" in lowered:
            return 0
        passed = _PYTEST_PASSED_PATTERN.search(line)
        if passed:
            return int(passed.group(1))
    return 0


def _score_test_run(
    returncode: int, test_output: str
) -> tuple[float, float | None, int]:
    """Score one test run from its exit code and captured output.

    Reward 1.0 requires exit code 0 and positive test evidence.
    Exit 0 without evidence scores 0.0 with no partial credit,
    because output that proves nothing must earn nothing. On a
    nonzero exit, partial credit reads the failure output as
    before.
    """
    evidence = _test_evidence(test_output)
    if returncode == 0:
        if evidence >= 1:
            return 1.0, _partial_credit(test_output), evidence
        return 0.0, None, 0
    return 0.0, _partial_credit(test_output), evidence


def _protected_files(task: TaskSpec) -> tuple[str, ...]:
    """Workspace-relative paths of files the episode must not change.

    A file is protected when the test command names it, when its
    name matches test_* or *_test.*, or when it sits under a
    tests/ directory. The set comes from the pristine workspace.
    """
    workspace = task.workspace.resolve()
    protected: set[str] = set()
    for argument in task.test_command:
        candidate = (workspace / argument).resolve()
        if candidate.is_file() and candidate.is_relative_to(workspace):
            protected.add(candidate.relative_to(workspace).as_posix())
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace)
        named_like_test = fnmatch(path.name, "test_*") or fnmatch(
            path.name, "*_test.*"
        )
        if named_like_test or "tests" in relative.parts[:-1]:
            protected.add(relative.as_posix())
    return tuple(sorted(protected))


def _file_digests(
    workspace: Path, relative_paths: tuple[str, ...]
) -> dict[str, str]:
    """SHA-256 digest of each named file inside the workspace."""
    return {
        relative: hashlib.sha256(
            (workspace / relative).read_bytes()
        ).hexdigest()
        for relative in relative_paths
    }


def _changed_protected_files(
    workspace: Path, pristine_digests: Mapping[str, str]
) -> tuple[str, ...]:
    """Protected files that changed or vanished in the workspace."""
    changed: list[str] = []
    for relative, digest in sorted(pristine_digests.items()):
        path = workspace / relative
        if not path.is_file():
            changed.append(relative)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    return tuple(changed)


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
    reward_partial: float | None
    duration_seconds: float
    test_files_changed: bool = False
    test_evidence: int = 0


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
    extra_env: dict[str, str] | None = None,
) -> EpisodeRecord | EpisodeFailure:
    """Run one real omp session on the task and score the result.

    extra_env is merged into the omp child environment last, so it
    wins over the .env file. Callers use it for per-episode routing
    such as pointing omp at a policy server.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:6]
    episode_dir = (runs_dir / f"{task.name}-{stamp}-{unique}").resolve()
    workspace = episode_dir / "ws"
    session_dir = episode_dir / "sess"
    episode_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(task.workspace, workspace)
    session_dir.mkdir()
    protected = _protected_files(task)
    pristine_digests = _file_digests(task.workspace, protected)
    prompt = _episode_prompt(task, workspace)
    (episode_dir / "prompt.txt").write_text(prompt + "\n")

    command = [
        "omp",
        "-p",
        prompt,
        "--system-prompt",
        SYSTEM_PROMPT,
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
    deadline = int(task.max_time) + 120
    try:
        omp_run = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=deadline,
            env=_episode_environment(os.environ, extra_env),
        )
    except subprocess.TimeoutExpired as timeout_error:
        stdout = timeout_error.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        (episode_dir / "events.jsonl").write_text(stdout or "")
        return EpisodeFailure(
            task=task.name,
            reason=f"omp exceeded the {deadline}s episode deadline",
        )
    duration = time.monotonic() - started
    (episode_dir / "events.jsonl").write_text(omp_run.stdout)
    if omp_run.stderr:
        (episode_dir / "stderr.log").write_text(omp_run.stderr)

    session_file = _find_session_file(session_dir)
    if session_file is None:
        return EpisodeFailure(
            task=task.name,
            reason=(f"omp exited with {omp_run.returncode} and wrote no session"),
        )

    changed = _changed_protected_files(workspace, pristine_digests)
    if changed:
        (episode_dir / "test_output.log").write_text(
            "protected test files changed: " + ", ".join(changed) + "\n"
        )
        record = EpisodeRecord(
            task=task.name,
            model=model if model is not None else "default",
            episode_dir=str(episode_dir),
            session_file=str(session_file),
            omp_exit_code=omp_run.returncode,
            test_exit_code=1,
            reward=0.0,
            reward_partial=None,
            duration_seconds=round(duration, 1),
            test_files_changed=True,
        )
        (episode_dir / "episode.json").write_text(
            json.dumps(asdict(record), indent=2)
        )
        return record

    try:
        test_run = subprocess.run(
            list(task.test_command),
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        (episode_dir / "test_output.log").write_text(
            "test command exceeded the 120s deadline\n"
        )
        return EpisodeFailure(
            task=task.name,
            reason="test command exceeded the 120s deadline",
        )
    test_output = test_run.stdout + test_run.stderr
    reward, partial, evidence = _score_test_run(test_run.returncode, test_output)
    if test_run.returncode == 0 and evidence == 0:
        if test_output and not test_output.endswith("\n"):
            test_output += "\n"
        test_output += "test exited 0 without test evidence\n"
    (episode_dir / "test_output.log").write_text(test_output)

    record = EpisodeRecord(
        task=task.name,
        model=model if model is not None else "default",
        episode_dir=str(episode_dir),
        session_file=str(session_file),
        omp_exit_code=omp_run.returncode,
        test_exit_code=test_run.returncode,
        reward=reward,
        reward_partial=partial,
        duration_seconds=round(duration, 1),
        test_evidence=evidence,
    )
    (episode_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))
    return record
