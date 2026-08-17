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
import secrets
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
_ALL_PASSED_UNNUMBERED_PATTERN = re.compile(r"\ball .{0,24}cases passed\b")
_FAILURE_MARKER_PATTERN = re.compile(
    r"FAIL|FAILED|Traceback|AssertionError|\bfailed\b|\berrors?\b|\bError\b"
)
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

    Failure markers rule first: when FAIL, FAILED, Traceback,
    AssertionError, or an error token appears anywhere, the count
    is 0 no matter what positive lines coexist. A forged success
    line printed at import time sits next to the real failure
    summary, so this precedence kills print-only forgery. After
    that, three shapes give a positive count: a custom script's
    "all N cases passed", unittest's "Ran N tests" followed by
    "OK", and a pytest summary "N passed". A final unnumbered
    "all ... cases passed" line counts as one proven case for
    scripts that do not print a number.
    """
    if _FAILURE_MARKER_PATTERN.search(test_output):
        return 0
    custom = _ALL_PASSED_PATTERN.search(test_output)
    if custom:
        return int(custom.group(1))
    ran = _UNITTEST_RAN_PATTERN.search(test_output)
    if ran:
        after_ran = test_output[ran.end() :]
        if re.search(r"\bOK\b", after_ran):
            return int(ran.group(1))
        return 0
    for line in reversed(test_output.splitlines()):
        if "passed" not in line:
            continue
        passed = _PYTEST_PASSED_PATTERN.search(line)
        if passed:
            return int(passed.group(1))
        if _ALL_PASSED_UNNUMBERED_PATTERN.search(line):
            return 1
    return 0


def _case_total(test_output: str) -> int | None:
    """Total number of test cases named by the output, if any.

    The total anchors the exact-count rule: evidence from a later
    run must equal the pristine baseline's total. Custom scripts
    name it in "N of M cases failed", unittest in "Ran N tests",
    and pytest in the summed summary counts.
    """
    found = _FAILED_PATTERN.search(test_output)
    if found:
        return int(found.group(2))
    ran = _UNITTEST_RAN_PATTERN.search(test_output)
    if ran:
        return int(ran.group(1))
    for line in reversed(test_output.splitlines()):
        if "passed" not in line and "failed" not in line:
            continue
        failed_m = re.search(r"(\d+) failed", line)
        passed_m = re.search(r"(\d+) passed", line)
        errors_m = re.search(r"(\d+) errors?\b", line)
        total = sum(
            int(match.group(1))
            for match in (failed_m, passed_m, errors_m)
            if match
        )
        if total:
            return total
    custom = _ALL_PASSED_PATTERN.search(test_output)
    if custom:
        return int(custom.group(1))
    return None


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
class Evidence:
    """Proof of executed passing cases, bound to one episode nonce.

    Only score_solution builds this value, after the canary pass
    confirms the solution does not intercept process exit and the
    case count matches the expected total.
    """

    cases_passed: int
    nonce: str


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring one episode workspace."""

    reward: float
    reward_partial: float | None
    evidence: int
    expected_cases: int | None
    test_files_changed: bool
    test_exit_code: int
    test_output: str
    timed_out: bool = False
    reason: str | None = None


_PYTHON_RUNNERS = frozenset({"python3", "python", "pytest"})


def _solution_code_files(
    eval_dir: Path, protected: tuple[str, ...], node_task: bool
) -> tuple[Path, ...]:
    """Code files in the eval directory that the canary must import."""
    protected_set = set(protected)
    suffix = ".mjs" if node_task else ".py"
    files = [
        path
        for path in sorted(eval_dir.rglob(f"*{suffix}"))
        if path.is_file()
        and path.relative_to(eval_dir).as_posix() not in protected_set
        and "__pycache__" not in path.parts
    ]
    return tuple(files)


def _canary_command(
    eval_dir: Path, nonce: str, solution_files: tuple[Path, ...], node_task: bool
) -> tuple[str, ...]:
    """Write the canary script and return the command that runs it.

    The canary imports every solution module and then exits with
    code 13. Solution code that hijacks process exit (os._exit,
    sys.exit, process.exit, exit-event emitters) changes that code,
    and the score is withheld. The nonce in the file name marks the
    script as harness-made after the episode ended.
    """
    if node_task:
        script = eval_dir / f"canary_{nonce}.mjs"
        imports = "\n".join(
            f"try {{ await import({json.dumps(path.as_uri())}); }} catch {{}}"
            for path in solution_files
        )
        script.write_text(imports + "\nprocess.exit(13);\n")
        return ("node", str(script))
    script = eval_dir / f"canary_{nonce}.py"
    lines = [
        "import importlib.util",
        "import os",
        "",
    ]
    for index, path in enumerate(solution_files):
        lines.extend(
            [
                "try:",
                "    _spec = importlib.util.spec_from_file_location(",
                f"        'canary_module_{index}', {json.dumps(str(path))}",
                "    )",
                "    _module = importlib.util.module_from_spec(_spec)",
                "    _spec.loader.exec_module(_module)",
                "except Exception:",
                "    pass",
            ]
        )
    lines.append("os._exit(13)")
    script.write_text("\n".join(lines) + "\n")
    return ("python3", str(script))


def score_solution(
    task: TaskSpec,
    episode_workspace: Path,
    scratch_dir: Path,
    expected_cases: int | None,
) -> ScoreResult:
    """Score an episode workspace against the task's pristine tests.

    The pipeline: refuse changed protected files, rebuild a fresh
    eval directory from pristine tests plus safe solution files,
    run the real tests, run the canary, then grant reward only for
    a valid Evidence value: canary exit 13, positive evidence, and
    an exact case-count match when the expected total is known.
    """
    protected = _protected_files(task)
    pristine_digests = _file_digests(task.workspace, protected)
    changed = _changed_protected_files(episode_workspace, pristine_digests)
    if changed:
        line = "protected test files changed: " + ", ".join(changed) + "\n"
        return ScoreResult(
            reward=0.0,
            reward_partial=None,
            evidence=0,
            expected_cases=expected_cases,
            test_files_changed=True,
            test_exit_code=1,
            test_output=line,
        )

    eval_dir = scratch_dir / "eval"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    _build_eval_dir(eval_dir, task.workspace, episode_workspace, protected)
    nonce = secrets.token_hex(8)
    test_run = _run_grouped(
        task.test_command,
        cwd=eval_dir,
        env=_test_environment(os.environ),
        timeout=120,
    )
    if test_run.timed_out:
        return ScoreResult(
            reward=0.0,
            reward_partial=None,
            evidence=0,
            expected_cases=expected_cases,
            test_files_changed=False,
            test_exit_code=-1,
            test_output="test command exceeded the 120s deadline\n",
            timed_out=True,
            reason="test command exceeded the 120s deadline",
        )
    test_output = test_run.stdout + test_run.stderr
    reward, partial, evidence = _score_test_run(
        test_run.returncode, test_output
    )

    reason: str | None = None
    if reward == 1.0:
        node_task = task.test_command[0] not in _PYTHON_RUNNERS
        solution_files = _solution_code_files(eval_dir, protected, node_task)
        canary_run = _run_grouped(
            _canary_command(eval_dir, nonce, solution_files, node_task),
            cwd=eval_dir,
            env=_test_environment(os.environ),
            timeout=30,
        )
        if canary_run.timed_out or canary_run.returncode != 13:
            reward, partial, evidence = 0.0, None, 0
            reason = (
                "canary run exited "
                f"{'late' if canary_run.timed_out else canary_run.returncode}"
                "; solution code intercepts process exit"
            )
        elif expected_cases is not None and evidence != expected_cases:
            reward, partial = 0.0, None
            reason = (
                f"case count mismatch: evidence {evidence}, "
                f"expected {expected_cases}"
            )
            evidence = 0
        else:
            _ = Evidence(cases_passed=evidence, nonce=nonce)
    elif test_run.returncode == 0 and evidence == 0:
        reason = "test exited 0 without test evidence"

    if reason is not None:
        if test_output and not test_output.endswith("\n"):
            test_output += "\n"
        test_output += reason + "\n"
    return ScoreResult(
        reward=reward,
        reward_partial=partial,
        evidence=evidence,
        expected_cases=expected_cases,
        test_files_changed=False,
        test_exit_code=test_run.returncode,
        test_output=test_output,
        reason=reason,
    )


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
    expected_cases: int | None = None


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
    expected_cases = (
        task.expected_cases
        if task.expected_cases is not None
        else _case_total(baseline_output)
    )

    shutil.copytree(task.workspace, workspace)
    session_dir.mkdir()
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

    result = score_solution(task, workspace, episode_dir, expected_cases)
    if result.timed_out:
        (episode_dir / "test_output.log").write_text(result.test_output)
        return EpisodeFailure(
            task=task.name,
            reason=result.reason or "test command timed out",
        )
    (episode_dir / "test_output.log").write_text(result.test_output)

    record = EpisodeRecord(
        task=task.name,
        model=model if model is not None else "default",
        episode_dir=str(episode_dir),
        session_file=str(session_file),
        omp_exit_code=omp_run.returncode,
        test_exit_code=result.test_exit_code,
        reward=result.reward,
        reward_partial=result.reward_partial,
        duration_seconds=round(duration, 1),
        test_files_changed=result.test_files_changed,
        test_evidence=result.evidence,
        baseline_evidence=baseline_evidence,
        baseline_partial=baseline_partial,
        expected_cases=result.expected_cases,
    )
    (episode_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))
    return record
