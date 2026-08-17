"""Episode runner.

One episode: snapshot the task workspace (content-addressed), run
the baseline tests in a sandbox, run omp on the prompt inside a
sandbox, then score the result against the immutable snapshot in a
fresh eval directory, again in a sandbox.

Isolation model (see isolation.py): test processes get no network,
an empty HOME, a scrubbed environment, resource limits, and a
write scope limited to their own run directory. The agent process
gets the provider key its model needs and no other secret, a
per-episode HOME with a copy of the omp configuration, and a
network scope of loopback (local policy servers) or outbound 443
(remote providers). The pristine snapshot is content-hashed before
and after the episode; any change fails the episode.

Residual risk, stated plainly: the eval stage still imports the
agent's solution files, so import-time output forgery remains
possible in principle; the canary (separate directory, run first,
nonce-checked) removes the cheap versions. sandbox-exec is
deprecated by Apple, a process that escapes its group early can
outlive the killer, and outbound-443 episodes can reach any TLS
endpoint. A VM boundary is the real fix for those classes; this
module removes the host-level ones.
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

import yaml

from .envfile import load_env_file
from .export import SYSTEM_PROMPT, TASK_PROMPT_PREFIX
from .isolation import (
    MAX_STREAM_BYTES,
    ResourceLimits,
    fresh_home,
    limited_command,
    sandbox_available,
    sandbox_command,
    sandbox_enabled,
    scrub_secret_environment,
    write_sandbox_profile,
)
from .task import TaskSpec, runner_kind, workspace_digest

_FAILED_PATTERN = re.compile(r"(\d+) of (\d+) cases failed")
_UNITTEST_RAN_PATTERN = re.compile(r"Ran (\d+) tests? in")
_UNITTEST_COUNT_PATTERN = re.compile(r"(?:failures|errors)=(\d+)")
_ALL_PASSED_PATTERN = re.compile(r"all (\d+) cases passed")
_ALL_PASSED_UNNUMBERED_PATTERN = re.compile(r"\ball .{0,24}cases passed\b")
_FAILURE_MARKER_PATTERN = re.compile(
    r"\bFAILED\b"
    r"|^FAIL[:\s]"
    r"|\bFAIL:\s"
    r"|Traceback \(most recent call last\)"
    r"|AssertionError"
    r"|\b[1-9]\d* errors?\b"
    r"|\b[1-9]\d* failed\b",
    re.MULTILINE,
)
_PYTEST_PASSED_PATTERN = re.compile(r"(\d+) passed")
_PROVIDER_ERROR_PATTERN = re.compile(
    r"rate.?limit|overloaded|\b401\b|\b403\b|\b429\b|insufficient|quota",
    re.IGNORECASE,
)

_EPISODE_HOST_ENV_NAMES = ("PATH", "USER", "SHELL", "LANG", "TERM")

_TEST_ENV_EXTRAS = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}

# Known provider key names by provider id (the first path segment
# of a model id). Only the resolved provider's keys ever enter an
# episode environment; every other secret stays out.
_PROVIDER_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openai-codex": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "kimi-code": ("KIMI_CODE_API_KEY", "KIMI_API_KEY"),
    "moonshotai": ("MOONSHOT_API_KEY",),
}

_LOCAL_MODEL_PREFIXES = ("omp-gym/",)


def _provider_key_names(model: str | None) -> tuple[str, ...]:
    """Env key names the episode's provider may need, and no more."""
    if model is None:
        return ()
    raw_prefix = model.split("/", 1)[0].split(":", 1)[0].lower()
    if raw_prefix.startswith("claude"):
        prefix = "anthropic"
    elif raw_prefix.startswith(("gpt", "o1", "o3", "o4")):
        prefix = "openai"
    elif raw_prefix.startswith("gemini"):
        prefix = "gemini"
    elif raw_prefix.startswith("grok"):
        prefix = "xai"
    else:
        prefix = raw_prefix
    candidates = _PROVIDER_KEY_CANDIDATES.get(prefix)
    if candidates is None:
        normalized = prefix.upper().replace("-", "_")
        candidates = tuple(
            f"{normalized}_{suffix}" for suffix in ("API_KEY", "TOKEN", "KEY")
        )
    return candidates


def _configured_agent_model(config_file: Path) -> str | None:
    """Read the default omp model from one copied agent config."""
    if not config_file.is_file():
        return None
    try:
        document: object = yaml.safe_load(config_file.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(document, Mapping):
        return None
    roles = document.get("modelRoles")
    if not isinstance(roles, Mapping):
        return None
    model = roles.get("default")
    if not isinstance(model, str) or not model.strip():
        return None
    return model.strip()


def _is_local_model(model: str | None, extra_env: Mapping[str, str] | None) -> bool:
    """True when the episode talks to a loopback policy server."""
    if model is not None and model.startswith(_LOCAL_MODEL_PREFIXES):
        return True
    return bool(
        extra_env
        and any(
            "127.0.0.1" in value or "localhost" in value for value in extra_env.values()
        )
    )


def _episode_environment(
    host_environment: Mapping[str, str],
    extra_env: dict[str, str] | None,
    *,
    model: str | None,
    home: Path,
    tmpdir: Path,
    extra_secret_names: tuple[str, ...] = (),
) -> dict[str, str]:
    """Build the minimal child environment for one episode.

    The child gets basic host variables, the provider keys for the
    requested model only, and the caller's extra_env. Every other
    credential-shaped variable, from the host or from .env, stays
    out. HOME is a per-episode directory with a copy of the omp
    configuration, so agent writes never reach the real config.
    """
    environment = {
        name: host_environment[name]
        for name in _EPISODE_HOST_ENV_NAMES
        if name in host_environment
    }
    env_values = load_env_file(Path(".env"))
    key_names = _provider_key_names(model)
    for name in key_names:
        if name in env_values:
            environment[name] = env_values[name]
        elif name in host_environment:
            environment[name] = host_environment[name]
    environment["HOME"] = str(home)
    environment["TMPDIR"] = str(tmpdir)
    environment.update(extra_env or {})
    explicit_secret_names = tuple(
        name
        for name in extra_secret_names
        if extra_env is not None and name in extra_env
    )
    return scrub_secret_environment(
        environment,
        keep=(*key_names, *explicit_secret_names),
    )


def _test_environment(home: Path, tmpdir: Path) -> dict[str, str]:
    """Build the environment for a baseline, eval, or canary run.

    The test process gets PATH and LANG from the host, an empty
    HOME, a private TMPDIR, and interpreter hardening variables.
    No provider keys, no user configuration, nothing else.
    """
    environment = {
        name: os.environ[name] for name in ("PATH", "LANG") if name in os.environ
    }
    environment["HOME"] = str(home)
    environment["TMPDIR"] = str(tmpdir)
    environment.update(_TEST_ENV_EXTRAS)
    return environment


@dataclass(frozen=True)
class CompletedLike:
    """Result of one grouped subprocess run."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool = False


def _kill_process_group(pgid: int) -> None:
    """SIGKILL every process in the group; a dead group is fine."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _drain_stream(
    stream: IO[str], parts: list[str], cap: int, overflow: threading.Event
) -> None:
    """Collect chunks from a pipe until EOF, capped at `cap` bytes.

    Past the cap the reader keeps draining but discards, so the
    child cannot deadlock on a full pipe, and marks the run as
    overflowed. The caller kills the process group on overflow.
    """
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        encoded = chunk.encode("utf-8")
        if total >= cap:
            overflow.set()
            continue
        remaining = cap - total
        if len(encoded) <= remaining:
            parts.append(chunk)
            total += len(encoded)
            continue
        parts.append(encoded[:remaining].decode("utf-8", errors="ignore"))
        total = cap
        overflow.set()


def _run_grouped(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str],
    timeout: float,
    sandbox_profile: Path | None = None,
    limits: ResourceLimits | None = None,
    merge_stderr: bool = False,
    output_cap: int = MAX_STREAM_BYTES,
) -> CompletedLike:
    """Run a command in its own process group and reap the group.

    The child starts a new session, so its process group id equals
    its pid. The group gets SIGKILL on timeout, on output overflow,
    and in a finally block on every exit path, so no exception can
    strand the group. Reader threads drain the pipes with a byte
    cap. Stdin is /dev/null because an open stdin pipe makes
    `omp -p` wait for EOF before it starts. `limits` inserts a
    small launcher that applies rlimits and replaces itself with
    the target. This avoids Python's unsafe preexec hook after MLX
    starts worker threads. `sandbox_profile` puts that full process
    under sandbox-exec. `merge_stderr` keeps output order.
    """
    argv = list(command)
    if limits is not None:
        argv = limited_command(argv, limits)
    if sandbox_profile is not None:
        argv = sandbox_command(argv, sandbox_profile)
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    overflow = threading.Event()
    readers = [
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_parts, output_cap, overflow),
            daemon=True,
        )
    ]
    if not merge_stderr:
        readers.append(
            threading.Thread(
                target=_drain_stream,
                args=(process.stderr, stderr_parts, output_cap, overflow),
                daemon=True,
            )
        )
    for reader in readers:
        reader.start()
    timed_out = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if overflow.is_set():
                    break
    finally:
        _kill_process_group(process.pid)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for reader in readers:
            reader.join(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
    returncode = process.returncode if process.returncode is not None else -1
    return CompletedLike(
        returncode=returncode,
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        timed_out=timed_out,
        truncated=overflow.is_set(),
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


def _clamp01(value: float) -> float:
    """Clamp a partial-credit fraction into [0.0, 1.0]."""
    return min(1.0, max(0.0, value))


def _partial_credit(test_output: str) -> float | None:
    """Fraction of cases passed when the test reports a count.

    Three shapes are recognized: "N of M cases failed" (custom test
    scripts), unittest's "Ran N tests" with a FAILED counts line,
    and pytest's summary line. The result is clamped to [0, 1], so
    a forged "11 of 10 cases failed" cannot produce a negative
    reward. Other shapes give None and only the binary reward
    applies.
    """
    found = _FAILED_PATTERN.search(test_output)
    if found:
        failed, total = int(found.group(1)), int(found.group(2))
        if total:
            return _clamp01((total - failed) / total)
    ran = _UNITTEST_RAN_PATTERN.search(test_output)
    if ran:
        total = int(ran.group(1))
        if total:
            failed = sum(
                int(count) for count in _UNITTEST_COUNT_PATTERN.findall(test_output)
            )
            return _clamp01((total - failed) / total)
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
            return _clamp01(passed / total)
    return None


def _test_evidence(test_output: str) -> int:
    """Count of test cases the output proves ran and passed.

    Failure markers rule first: a FAILED banner, a traceback, an
    AssertionError, or a positive count of errors or failures
    anywhere sets the count to 0 no matter what positive lines
    coexist. Counted phrases require a nonzero number, so "0
    errors" in a passing run does not zero the score. After that,
    three shapes give a positive count: "all N cases passed",
    unittest's "Ran N tests" followed by "OK", and a pytest summary
    "N passed". A final unnumbered "all ... cases passed" line
    counts as one proven case.
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
            int(match.group(1)) for match in (failed_m, passed_m, errors_m) if match
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


_HOOK_FILE_NAMES = frozenset(
    (
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "sitecustomize.py",
        "usercustomize.py",
        "package.json",
        ".npmrc",
    )
)
_HOOK_FILE_PREFIXES = ("jest.config.", "vitest.config.")


def _is_hook_file(name: str) -> bool:
    """True for file names that can inject code into a test run."""
    if name in _HOOK_FILE_NAMES or name.endswith(".pth"):
        return True
    return any(name.startswith(prefix) for prefix in _HOOK_FILE_PREFIXES)


def _protected_files(root: Path, test_command: tuple[str, ...]) -> tuple[str, ...]:
    """Workspace-relative paths of files the episode must not change.

    A file is protected when the test command names it, when its
    name matches test_* or *_test.*, when it sits under a tests/
    directory, when it is a hook file already present in the
    pristine workspace, or when it is a symlink that escapes the
    workspace. The set comes from the immutable snapshot.
    """
    workspace = root.resolve()
    protected: set[str] = set()
    for argument in test_command:
        candidate = (workspace / argument).resolve()
        if candidate.is_file() and candidate.is_relative_to(workspace):
            protected.add(candidate.relative_to(workspace).as_posix())
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if path.is_symlink() and not path.resolve().is_relative_to(workspace):
            protected.add(relative.as_posix())
            continue
        if not path.is_file():
            continue
        named_like_test = fnmatch(path.name, "test_*") or fnmatch(path.name, "*_test.*")
        if (
            named_like_test
            or "tests" in relative.parts[:-1]
            or _is_hook_file(path.name)
        ):
            protected.add(relative.as_posix())
    return tuple(sorted(protected))


def _file_digests(root: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    """SHA-256 digest of each named file inside the root."""
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
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


def _safe_copytree(src: Path, dst: Path) -> tuple[str, ...]:
    """Copy a tree without following symlinks that escape it.

    Every entry resolves against the source root; a symlink whose
    target lands outside the root is skipped and reported. A
    directory symlink that stays inside contributes its content.
    Returns the workspace-relative paths that were refused.
    """
    src_root = src.resolve()
    refused: list[str] = []
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        resolved = path.resolve()
        if not resolved.is_relative_to(src_root):
            refused.append(relative.as_posix())
            continue
        if resolved.is_dir():
            (dst / relative).mkdir(parents=True, exist_ok=True)
            continue
        if not resolved.is_file():
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved, target)
    return tuple(refused)


def _overlay_files(
    pristine: Path,
    episode_workspace: Path,
    protected: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Episode files safe to copy into the eval dir, plus refused.

    Two groups qualify: files that exist in the pristine workspace
    and are not protected (the agent's edits to solution files),
    and new files whose names cannot hook the test runner. Hook
    files never overlay, new or pre-existing; pre-existing ones are
    also protected, so a modification already failed the episode
    before this runs. Symlinks escaping the episode workspace are
    refused and reported.
    """
    protected_set = set(protected)
    selected: list[str] = []
    refused: list[str] = []
    workspace_root = episode_workspace.resolve()
    for path in sorted(episode_workspace.rglob("*")):
        relative = path.relative_to(episode_workspace)
        if "__pycache__" in relative.parts:
            continue
        if not path.resolve().is_relative_to(workspace_root):
            refused.append(relative.as_posix())
            continue
        if not path.is_file():
            continue
        relative_name = relative.as_posix()
        if relative_name in protected_set or _is_hook_file(path.name):
            continue
        selected.append(relative_name)
    return tuple(selected), tuple(refused)


def _build_eval_dir(
    eval_dir: Path,
    pristine: Path,
    episode_workspace: Path,
    protected: tuple[str, ...],
) -> tuple[str, ...]:
    """Build a fresh directory for the trusted test run.

    The base is a copy of the pristine snapshot; the safe episode
    files overlay it. Tests never run inside the agent's own
    workspace. Returns the overlay files that were refused.
    """
    _safe_copytree(pristine, eval_dir)
    selected, refused = _overlay_files(pristine, episode_workspace, protected)
    for relative in selected:
        target = eval_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(episode_workspace / relative, target)
    return refused


@dataclass(frozen=True)
class Evidence:
    """Proof of executed passing cases, bound to one episode nonce.

    Only score_solution builds this value, after the canary pass
    confirms the solution does not intercept process exit and the
    case count matches the expected total. The nonce correlates the
    canary script with its output line; it is not a signature. The
    module docstring names the residual import-time forgery risk.
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


def _solution_code_files(
    eval_dir: Path, protected: tuple[str, ...], kind: str
) -> tuple[Path, ...]:
    """Code files in the eval directory that the canary must import."""
    protected_set = set(protected)
    suffixes = (".mjs", ".js", ".cjs") if kind == "node" else (".py",)
    files = [
        path
        for path in sorted(eval_dir.rglob("*"))
        if path.is_file()
        and path.suffix in suffixes
        and path.relative_to(eval_dir).as_posix() not in protected_set
        and "__pycache__" not in path.parts
    ]
    return tuple(files)


def _python_module_name(canary_dir: Path, path: Path) -> str:
    """The importable module name for one solution file."""
    relative = path.relative_to(canary_dir).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _canary_command(
    canary_dir: Path, nonce: str, solution_files: tuple[Path, ...], kind: str
) -> tuple[str, ...]:
    """Write the canary script and return the command that runs it.

    The canary imports every solution module under its real name,
    writes its nonce line with a raw fd write, and exits 13.
    Solution code that hijacks process exit changes that code, and
    solution code that silences stdout removes the nonce line; both
    forfeit the reward. Python modules import under their package
    names, so relative imports and package context work exactly as
    in the real test run. The nonce in the file name marks the
    script as harness-made after the episode ended.
    """
    marker = f"canary-ok {nonce}\n"
    if kind == "node":
        script = canary_dir / f"canary_{nonce}.mjs"
        imports = "\n".join(
            f"try {{ await import({json.dumps(path.as_uri())}); }} catch {{}}"
            for path in solution_files
        )
        script.write_text(
            "import { writeSync } from 'node:fs';\n"
            "const exitProcess = process.exit.bind(process);\n"
            + imports
            + f"\nwriteSync(1, {json.dumps(marker)});\nexitProcess(13);\n"
        )
        return ("node", str(script))
    script = canary_dir / f"canary_{nonce}.py"
    lines = [
        "import importlib",
        "import os",
        "import sys",
        "_write = os.write",
        "_exit = os._exit",
        f"sys.path.insert(0, {json.dumps(str(canary_dir))})",
        "",
    ]
    for path in solution_files:
        module = _python_module_name(canary_dir, path)
        if not module:
            continue
        lines.extend(
            [
                "try:",
                f"    importlib.import_module({json.dumps(module)})",
                "except Exception:",
                "    pass",
            ]
        )
    lines.append(f"_write(1, {marker.encode()!r})")
    lines.append("_exit(13)")
    script.write_text("\n".join(lines) + "\n")
    return ("python3", str(script))


def score_solution(
    task: TaskSpec,
    episode_workspace: Path,
    scratch_dir: Path,
    expected_cases: int | None,
    *,
    snapshot_dir: Path,
    snapshot_digest: str | None = None,
    use_sandbox: bool | None = None,
) -> ScoreResult:
    """Score an episode workspace against the task's pristine tests.

    The pipeline: verify the snapshot digest, refuse changed
    protected files, refuse symlink escapes, rebuild a fresh eval
    directory from the snapshot plus safe solution files, run the
    canary first in its own directory copy, run the real tests,
    then grant reward only for a valid Evidence value: canary exit
    13 with its nonce line, positive evidence, and an exact
    case-count match when the expected total is known. The
    snapshot, never the mutable task directory, supplies the tests.
    """
    if snapshot_digest is not None and (
        workspace_digest(snapshot_dir) != snapshot_digest
    ):
        return ScoreResult(
            reward=0.0,
            reward_partial=None,
            evidence=0,
            expected_cases=expected_cases,
            test_files_changed=True,
            test_exit_code=1,
            test_output="pristine snapshot digest mismatch\n",
            reason="sandbox: pristine snapshot digest mismatch",
        )
    protected = _protected_files(snapshot_dir, task.test_command)
    pristine_digests = _file_digests(snapshot_dir, protected)
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
            reason=line.strip(),
        )

    if use_sandbox is None:
        sandbox = sandbox_enabled(os.environ) and sandbox_available()
    else:
        sandbox = use_sandbox and sandbox_available()
    profiles_dir = scratch_dir / "profiles"
    kind = runner_kind(task.test_command)

    eval_dir = scratch_dir / "eval"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    refused = _build_eval_dir(eval_dir, snapshot_dir, episode_workspace, protected)
    if refused:
        line = "symlink or path escape refused: " + ", ".join(refused) + "\n"
        return ScoreResult(
            reward=0.0,
            reward_partial=None,
            evidence=0,
            expected_cases=expected_cases,
            test_files_changed=True,
            test_exit_code=1,
            test_output=line,
            reason=line.strip(),
        )

    canary_dir = scratch_dir / "canary"
    if canary_dir.exists():
        shutil.rmtree(canary_dir)
    shutil.copytree(eval_dir, canary_dir, symlinks=True)

    limits = ResourceLimits(cpu_seconds=150)
    nonce = secrets.token_hex(8)
    solution_files = _solution_code_files(canary_dir, protected, kind)
    canary_home = fresh_home(scratch_dir, "canary-home")
    canary_tmp = fresh_home(scratch_dir, "canary-tmp")
    canary_profile = (
        write_sandbox_profile(
            profiles_dir,
            "canary.sb",
            writable=(canary_dir, canary_tmp),
            network="deny",
        )
        if sandbox
        else None
    )
    canary_run = _run_grouped(
        _canary_command(canary_dir, nonce, solution_files, kind),
        cwd=canary_dir,
        env=_test_environment(canary_home, canary_tmp),
        timeout=30,
        sandbox_profile=canary_profile,
        limits=limits,
        merge_stderr=True,
    )
    canary_ok = (
        not canary_run.timed_out
        and not canary_run.truncated
        and canary_run.returncode == 13
        and canary_run.stdout.splitlines() == [f"canary-ok {nonce}"]
    )

    eval_home = fresh_home(scratch_dir, "eval-home")
    eval_tmp = fresh_home(scratch_dir, "eval-tmp")
    eval_profile = (
        write_sandbox_profile(
            profiles_dir,
            "eval.sb",
            writable=(eval_dir, eval_tmp),
            network="deny",
        )
        if sandbox
        else None
    )
    test_run = _run_grouped(
        task.test_command,
        cwd=eval_dir,
        env=_test_environment(eval_home, eval_tmp),
        timeout=120,
        sandbox_profile=eval_profile,
        limits=limits,
        merge_stderr=True,
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
            reason="test timeout: command exceeded the 120s deadline",
        )
    if test_run.truncated:
        return ScoreResult(
            reward=0.0,
            reward_partial=None,
            evidence=0,
            expected_cases=expected_cases,
            test_files_changed=False,
            test_exit_code=test_run.returncode,
            test_output="test output exceeded the 8 MiB cap\n",
            reason="invalid task: test output exceeded the 8 MiB cap",
        )
    test_output = test_run.stdout
    if test_run.stderr:
        test_output += "\n" + test_run.stderr
    reward, partial, evidence = _score_test_run(test_run.returncode, test_output)

    reason: str | None = None
    has_score = reward == 1.0 or (partial is not None and partial > 0.0)
    if has_score:
        reported_cases = evidence if reward == 1.0 else _case_total(test_output)
        if not canary_ok:
            reward, partial, evidence = 0.0, None, 0
            reason = (
                "canary failed: solution code writes output during import, "
                "intercepts process exit, or intercepts stdout"
            )
        elif expected_cases is None:
            reward, partial, evidence = 0.0, None, 0
            reason = "case count unavailable for nonzero score"
        elif reported_cases != expected_cases:
            reward, partial, evidence = 0.0, None, 0
            reason = (
                f"case count mismatch: evidence {reported_cases}, "
                f"expected {expected_cases}"
            )
        elif reward == 1.0:
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
    reward_improvement: float | None = None
    session_sha256: str | None = None
    workspace_digest: str | None = None
    wall_seconds: float | None = None


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
    fallback. The runner hashes the file the moment omp exits, so
    later tampering cannot change what export consumes.
    """

    def start_time(path: Path) -> float:
        status = path.stat()
        return getattr(status, "st_birthtime", status.st_mtime)

    candidates = sorted(session_dir.rglob("*.jsonl"), key=start_time)
    if not candidates:
        return None
    return candidates[0]


def _copy_selected_provider(
    source_file: Path,
    target_file: Path,
    model: str | None,
) -> None:
    """Copy only the custom provider used by one episode."""
    if model is None or "/" not in model or not source_file.is_file():
        return
    provider_id = model.split("/", 1)[0]
    try:
        document: object = yaml.safe_load(source_file.read_text())
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(document, Mapping):
        return
    providers = document.get("providers")
    if not isinstance(providers, Mapping):
        return
    provider = providers.get(provider_id)
    if not isinstance(provider, Mapping):
        return
    target_file.write_text(
        yaml.safe_dump(
            {"providers": {provider_id: dict(provider)}},
            sort_keys=False,
        )
    )


def _prepare_agent_home(episode_dir: Path, model: str | None = None) -> Path:
    """A per-episode HOME with the minimum omp configuration.

    config.yml and only the selected custom provider are copied.
    State databases, other provider definitions, caches, history,
    and MCP process configuration stay behind. The episode cannot
    read or write the operator's real omp state or inherit another
    provider's literal configuration.
    """
    home = episode_dir / "home"
    target = home / ".omp" / "agent"
    target.mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".omp" / "agent"
    source_config = source / "config.yml"
    target_config = target / "config.yml"
    if source_config.is_file():
        shutil.copyfile(source_config, target_config)
    resolved_model = (
        model if model is not None else _configured_agent_model(target_config)
    )
    _copy_selected_provider(
        source / "models.yml",
        target / "models.yml",
        resolved_model,
    )
    return home


def run_episode(
    task: TaskSpec,
    runs_dir: Path,
    model: str | None,
    extra_env: dict[str, str] | None = None,
    *,
    extra_secret_names: tuple[str, ...] = (),
) -> EpisodeRecord | EpisodeFailure:
    """Run one real omp session on the task and score the result.

    extra_env is merged into the omp child environment last, so it
    wins over the resolved provider keys. extra_secret_names keeps only
    explicitly supplied secret variables from that mapping. Callers use
    both values for per-episode policy routing.
    """
    wall_started = time.monotonic()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    unique = uuid.uuid4().hex[:6]
    episode_dir = (runs_dir / f"{task.name}-{stamp}-{unique}").resolve()
    workspace = episode_dir / "ws"
    session_dir = episode_dir / "sess"
    episode_dir.mkdir(parents=True, exist_ok=False)

    use_sandbox = sandbox_enabled(os.environ)
    if use_sandbox and not sandbox_available():
        return EpisodeFailure(
            task=task.name,
            reason=(
                "sandbox: sandbox-exec unavailable; set OMP_GYM_SANDBOX=0 "
                "to accept the documented risk"
            ),
        )

    snapshot_dir = episode_dir / "snapshot"
    refused = _safe_copytree(task.workspace, snapshot_dir)
    if refused:
        return EpisodeFailure(
            task=task.name,
            reason=(
                "invalid task: symlinks escape the workspace: " + ", ".join(refused)
            ),
        )
    pristine_digest = workspace_digest(snapshot_dir)

    baseline_dir = episode_dir / "baseline"
    _safe_copytree(snapshot_dir, baseline_dir)
    baseline_home = fresh_home(episode_dir, "baseline-home")
    baseline_tmp = fresh_home(episode_dir, "baseline-tmp")
    profiles_dir = episode_dir / "profiles"
    baseline_profile = (
        write_sandbox_profile(
            profiles_dir,
            "baseline.sb",
            writable=(baseline_dir, baseline_tmp),
            network="deny",
        )
        if use_sandbox
        else None
    )
    baseline_run = _run_grouped(
        task.test_command,
        cwd=baseline_dir,
        env=_test_environment(baseline_home, baseline_tmp),
        timeout=120,
        sandbox_profile=baseline_profile,
        limits=ResourceLimits(cpu_seconds=150),
        merge_stderr=True,
    )
    shutil.rmtree(baseline_dir, ignore_errors=True)
    baseline_output = baseline_run.stdout
    if baseline_run.timed_out:
        return EpisodeFailure(
            task=task.name,
            reason="baseline timeout: test run exceeded the 120s deadline",
        )
    if baseline_run.truncated:
        return EpisodeFailure(
            task=task.name,
            reason="invalid task: baseline output exceeded the 8 MiB cap",
        )
    if baseline_run.returncode == 0:
        if _baseline_passed(baseline_run.returncode, baseline_output):
            return EpisodeFailure(
                task=task.name,
                reason="invalid task: already passes before the agent runs",
            )
        return EpisodeFailure(
            task=task.name,
            reason="invalid task: baseline exited 0 without test evidence",
        )
    baseline_evidence = _test_evidence(baseline_output)
    baseline_partial = _partial_credit(baseline_output)
    expected_cases = (
        task.expected_cases
        if task.expected_cases is not None
        else _case_total(baseline_output)
    )
    if expected_cases is None:
        return EpisodeFailure(
            task=task.name,
            reason=(
                "invalid task: baseline output does not report a case count; "
                "set expected_cases in task.toml"
            ),
        )

    _safe_copytree(snapshot_dir, workspace)
    session_dir.mkdir()
    agent_home = _prepare_agent_home(episode_dir, model)
    agent_tmp = fresh_home(episode_dir, "agent-tmp")
    resolved_model = (
        model
        if model is not None
        else _configured_agent_model(agent_home / ".omp" / "agent" / "config.yml")
    )
    prompt = _episode_prompt(task, workspace)

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

    local_model = _is_local_model(resolved_model, extra_env)
    agent_profile = (
        write_sandbox_profile(
            profiles_dir,
            "agent.sb",
            writable=(workspace, session_dir, agent_home, agent_tmp),
            network="loopback" if local_model else "open-443",
        )
        if use_sandbox
        else None
    )
    episode_env = _episode_environment(
        os.environ,
        extra_env,
        model=resolved_model,
        home=agent_home,
        tmpdir=agent_tmp,
        extra_secret_names=extra_secret_names,
    )

    started = time.monotonic()
    deadline = int(task.max_time) + 120
    omp_run = _run_grouped(
        command,
        env=episode_env,
        timeout=deadline,
        sandbox_profile=agent_profile,
        limits=ResourceLimits(cpu_seconds=deadline + 60),
    )
    duration = time.monotonic() - started
    (episode_dir / "events.jsonl").write_text(omp_run.stdout)
    if omp_run.stderr:
        (episode_dir / "stderr.log").write_text(omp_run.stderr)
    if omp_run.timed_out:
        return EpisodeFailure(
            task=task.name,
            reason=(f"test timeout: omp exceeded the {deadline}s episode deadline"),
        )
    if omp_run.truncated:
        return EpisodeFailure(
            task=task.name,
            reason="provider error: omp output exceeded the 8 MiB cap",
        )

    session_file = _find_session_file(session_dir)
    if session_file is None:
        combined = omp_run.stdout + omp_run.stderr
        kind = (
            "provider error"
            if _PROVIDER_ERROR_PATTERN.search(combined)
            else "no session"
        )
        return EpisodeFailure(
            task=task.name,
            reason=(
                f"{kind}: omp exited with {omp_run.returncode} and wrote no session"
            ),
        )

    session_bytes = session_file.read_bytes()
    session_sha = hashlib.sha256(session_bytes).hexdigest()
    session_copy = episode_dir / "session.jsonl"
    session_copy.write_bytes(session_bytes)

    if workspace_digest(task.workspace) != pristine_digest:
        return EpisodeFailure(
            task=task.name,
            reason="sandbox: pristine task workspace modified during episode",
        )

    result = score_solution(
        task,
        workspace,
        episode_dir,
        expected_cases,
        snapshot_dir=snapshot_dir,
        snapshot_digest=pristine_digest,
        use_sandbox=use_sandbox,
    )
    if result.timed_out:
        (episode_dir / "test_output.log").write_text(result.test_output)
        return EpisodeFailure(
            task=task.name,
            reason=result.reason or "test timeout: command timed out",
        )
    (episode_dir / "test_output.log").write_text(result.test_output)
    (episode_dir / "baseline_output.log").write_text(baseline_output)
    (episode_dir / "prompt.txt").write_text(prompt + "\n")

    final_value = (
        result.reward_partial if result.reward_partial is not None else result.reward
    )
    improvement = round(final_value - (baseline_partial or 0.0), 4)
    record = EpisodeRecord(
        task=task.name,
        model=model if model is not None else "default",
        episode_dir=str(episode_dir),
        session_file=str(session_copy),
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
        reward_improvement=improvement,
        session_sha256=session_sha,
        workspace_digest=pristine_digest,
        wall_seconds=round(time.monotonic() - wall_started, 1),
    )
    (episode_dir / "episode.json").write_text(json.dumps(asdict(record), indent=2))
    return record
