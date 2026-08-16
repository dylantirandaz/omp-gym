"""Episode runner.

One episode: copy the task workspace, run omp on the prompt without
supervision, then run the task tests in a rebuilt eval directory.

Residual risk: the trusted eval stage still imports the agent's
solution files, so import-time prints inside a solution file can
forge test evidence in the captured output. A VM boundary around
the eval stage is the real fix; this module only removes the cheap
escape paths (hook files, background processes, inherited env).
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import IO

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

_TEST_HOST_ENV_NAMES = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
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


def _test_environment(host_environment: Mapping[str, str]) -> dict[str, str]:
    """Build the minimal environment for a baseline or eval test run.

    The test process gets only PATH, HOME, TMPDIR, and LANG from
    the host and nothing else. Provider keys from .env and other
    parent variables stay out of the test process.
    """
    return {
        name: host_environment[name]
        for name in _TEST_HOST_ENV_NAMES
        if name in host_environment
    }


@dataclass(frozen=True)
class CompletedLike:
    """Result of one grouped subprocess run."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _kill_process_group(pgid: int) -> None:
    """SIGKILL every process in the group; a dead group is fine."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _drain_stream(stream: IO[str], parts: list[str]) -> None:
    """Collect chunks from a pipe until EOF; keep partial output."""
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        parts.append(chunk)


def _run_grouped(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str],
    timeout: float,
) -> CompletedLike:
    """Run a command in its own process group and reap the group.

    The child starts a new session, so its process group id equals
    its pid. The group gets SIGKILL on timeout, and also right
    after a normal exit. Background helpers the child spawned die
    with it, so they cannot outlive the run and touch files during
    scoring. Reader threads drain the pipes, so a leader with much
    output cannot deadlock and a background helper that holds the
    pipes open cannot stall the run past the leader's exit. Stdin
    is /dev/null because an open stdin pipe makes `omp -p` wait
    for EOF before it starts.
    """
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    readers = (
        threading.Thread(
            target=_drain_stream, args=(process.stdout, stdout_parts), daemon=True
        ),
        threading.Thread(
            target=_drain_stream, args=(process.stderr, stderr_parts), daemon=True
        ),
    )
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    _kill_process_group(process.pid)
    process.wait()
    for reader in readers:
        reader.join(timeout=5)
    for stream in (process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            try:
                stream.close()
            except OSError:
                pass
    return CompletedLike(
        returncode=process.returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        timed_out=timed_out,
    )


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
    and pytest's summary line. The pytest fraction is passed over
    passed plus failed plus errors, so "1 error, 10 passed" does
    not give 1.0. Others give None and only the binary reward
    applies.
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
        errors_m = re.search(r"(\d+) errors?\b", line)
        failed = int(failed_m.group(1)) if failed_m else 0
        passed = int(passed_m.group(1)) if passed_m else 0
        errors = int(errors_m.group(1)) if errors_m else 0
        total = failed + passed + errors
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


def _baseline_passed(returncode: int, test_output: str) -> bool:
    """True when the pristine workspace already passes the tests.

    A passing baseline means the task pays reward without any agent
    work, so the episode must not start.
    """
    return returncode == 0 and _test_evidence(test_output) >= 1


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


_HOOK_FILE_NAMES = frozenset(
    (
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "sitecustomize.py",
        "usercustomize.py",
    )
)


def _is_hook_file(name: str) -> bool:
    """True for file names that can inject code into a test run."""
    return name in _HOOK_FILE_NAMES or name.endswith(".pth")


def _overlay_files(
    pristine: Path,
    episode_workspace: Path,
    protected: tuple[str, ...],
) -> tuple[str, ...]:
    """Episode files that are safe to copy into the eval directory.

    Two groups qualify: files that exist in the pristine workspace
    and are not protected (the agent's edits to solution files),
    and new files whose names cannot hook the test runner. New
    conftest.py, pytest and packaging config, site customization,
    and .pth files stay out at every depth, as do __pycache__
    entries. Protected files never overlay; they come from the
    pristine copy.
    """
    protected_set = set(protected)
    pristine_files = {
        path.relative_to(pristine).as_posix()
        for path in pristine.rglob("*")
        if path.is_file()
    }
    selected: list[str] = []
    for path in sorted(episode_workspace.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(episode_workspace)
        if "__pycache__" in relative.parts:
            continue
        relative_name = relative.as_posix()
        if relative_name in protected_set:
            continue
        if relative_name in pristine_files:
            selected.append(relative_name)
            continue
        if _is_hook_file(path.name):
            continue
        selected.append(relative_name)
    return tuple(selected)


def _build_eval_dir(
    eval_dir: Path,
    pristine: Path,
    episode_workspace: Path,
    protected: tuple[str, ...],
) -> Path:
    """Build a fresh directory for the trusted test run.

    The base is a copy of the pristine task workspace; the safe
    episode files overlay it. Tests never run inside the agent's
    own workspace.
    """
    shutil.copytree(pristine, eval_dir)
    for relative in _overlay_files(pristine, episode_workspace, protected):
        target = eval_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(episode_workspace / relative, target)
    return eval_dir


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
    baseline_evidence: int = 0
    baseline_partial: float | None = None


@dataclass(frozen=True)
class EpisodeFailure:
    """The episode did not produce a usable session."""

    task: str
    reason: str


def _find_session_file(session_dir: Path) -> Path | None:
    """Find the earliest session JSONL file below the session directory.

    omp creates its session file when it starts. A file the agent
    forges appears later, so the earliest file is the real session
    even when more than one file exists. Creation time decides
    where the platform records it; the last write time is the
    fallback.
    """

    def start_time(path: Path) -> float:
        status = path.stat()
        return getattr(status, "st_birthtime", status.st_mtime)

    candidates = sorted(session_dir.rglob("*.jsonl"), key=start_time)
    if not candidates:
        return None
    return candidates[0]


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

    baseline_dir = episode_dir / "baseline"
    shutil.copytree(task.workspace, baseline_dir)
    baseline_run = _run_grouped(
        task.test_command,
        cwd=baseline_dir,
        env=_test_environment(os.environ),
        timeout=120,
    )
    shutil.rmtree(baseline_dir)
    baseline_output = baseline_run.stdout + baseline_run.stderr
    (episode_dir / "baseline_output.log").write_text(baseline_output)
    if baseline_run.timed_out:
        return EpisodeFailure(
            task=task.name,
            reason="baseline test run exceeded the 120s deadline",
        )
    if _baseline_passed(baseline_run.returncode, baseline_output):
        return EpisodeFailure(
            task=task.name,
            reason="task already passes before the agent runs",
        )
    baseline_evidence = _test_evidence(baseline_output)
    baseline_partial = _partial_credit(baseline_output)

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
    omp_run = _run_grouped(
        command,
        env=_episode_environment(os.environ, extra_env),
        timeout=deadline,
    )
    duration = time.monotonic() - started
    (episode_dir / "events.jsonl").write_text(omp_run.stdout)
    if omp_run.stderr:
        (episode_dir / "stderr.log").write_text(omp_run.stderr)
    if omp_run.timed_out:
        return EpisodeFailure(
            task=task.name,
            reason=f"omp exceeded the {deadline}s episode deadline",
        )

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
            baseline_evidence=baseline_evidence,
            baseline_partial=baseline_partial,
        )
        (episode_dir / "episode.json").write_text(
            json.dumps(asdict(record), indent=2)
        )
        return record

    eval_dir = _build_eval_dir(
        episode_dir / "eval", task.workspace, workspace, protected
    )
    test_run = _run_grouped(
        task.test_command,
        cwd=eval_dir,
        env=_test_environment(os.environ),
        timeout=120,
    )
    if test_run.timed_out:
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
        baseline_evidence=baseline_evidence,
        baseline_partial=baseline_partial,
    )
    (episode_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))
    return record
