"""Turn OMP session episodes into sealed test-command tasks.

An episode becomes a task only when the agent ran a recognized test command
that failed before its edits and passed after them, and when every edited
file can be anchored to one commit of a repository present on this host. The
start state is that commit plus the files earlier episodes of the same
session left behind; the end state adds this episode's mutations. The test
files of the end state are sealed and the start -> end diff is the reference
patch that the gate must turn green inside the task image.

This module is pure standard library plus ``git`` and ``docker`` processes so
it runs on the machine that holds the sessions, including Windows.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .sessions import (
    CommandRun,
    Episode,
    FileMutation,
    SessionLoadError,
    clean_prompt,
    discover_sessions,
    read_session,
    session_episodes,
)
from .task import (
    EMPTY_DEPENDENCY_LOCK_DIGEST,
    MAX_PROMPT_BYTES,
    MAX_SEALED_FILES,
    SEALED_FILES_DIR,
    TASK_ID_PATTERN,
    TEST_COMMAND_SCHEMA_VERSION,
    Split,
    TaskLoadError,
    TaskSpec,
    TestCommandVerifier,
    load_task,
)
from .testcommand import (
    EXCLUDES_FILE,
    MAX_OUTPUT_BYTES,
    START_TAG,
    CommandOutcome,
    TestCommandResult,
    grade_outcome,
    grading_script,
)

MINT_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_SESSIONS_DIR = Path.home() / ".omp" / "agent" / "sessions"
MAX_TIME_SECONDS = 1800
TOKEN_BUDGET = 400_000
CPU_LIMIT = 2.0
MEMORY_BYTES = 4 * 1024 * 1024 * 1024
PID_LIMIT = 512
WORKSPACE_BYTES = 1024 * 1024 * 1024
TEMP_BYTES = 256 * 1024 * 1024
HOME_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_FILE_BYTES = 32 * 1024 * 1024
MAX_BASE_COMMITS = 64
GIT_TIMEOUT_SECONDS = 300
DOCKER_BUILD_TIMEOUT_SECONDS = 3600
DOCKER_COMMAND_TIMEOUT_SECONDS = 120
PLACEHOLDER_IMAGE_DIGEST = "sha256:" + "0" * 64
OUTPUT_TAIL_CHARS = 4000
SESSION_ID_PREFIX_LENGTH = 8
TRAIN_SPLIT_BUCKETS = 7
VALIDATION_SPLIT_BUCKETS = 9
SPLIT_BUCKETS = 10
REPORT_NAME = "mint-report.json"
GIT_IDENTITY = ("-c", "user.name=omp-gym", "-c", "user.email=omp-gym@localhost")
GIT_NO_EOL_CONVERSION = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false")
CONTAINER_GIT = (
    "git -c user.name=omp-gym -c user.email=omp-gym@localhost"
    f" -c core.excludesFile={EXCLUDES_FILE}"
)
OFFLINE_ENV = (
    "UV_OFFLINE=1 UV_FROZEN=1 PIP_NO_INDEX=1 npm_config_offline=true"
    " CARGO_NET_OFFLINE=true GOPROXY=off CI=true PYTHONDONTWRITEBYTECODE=1"
)
EXCLUDED_DIRS = (
    ".venv/",
    "node_modules/",
    "target/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "dist/",
    "build/",
    "*.pyc",
    ".omp/",
)
BASE_PACKAGES = "git python3 util-linux ca-certificates"
LOCKFILE_NAMES = (
    "uv.lock",
    "poetry.lock",
    "package-lock.json",
    "bun.lock",
    "bun.lockb",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.lock",
    "go.sum",
)
TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__"})
TEST_FILE_GLOBS = ("test_*.py", "*_test.py", "*.test.*", "*.spec.*", "*_test.go")
DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"AKIA[0-9A-Z]{16}",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"ghp_[A-Za-z0-9]{30,}",
        r"xox[baprs]-",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
)
NETWORK_TOOLS = re.compile(
    r"(?<![\w-])(?:curl|wget|pip3? install|npm install|docker|sudo)(?![\w-])"
)
HOST_PATHS = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|/home/)")
_UV_FLAGS = r"(?: -[^\s]+(?: [^\s-][^\s]*)?)*"
_RUNNERS = (
    r"pytest",
    r"py\.test",
    r"python3? -m (?:pytest|unittest)",
    rf"uv run{_UV_FLAGS} (?:pytest|python3? -m (?:pytest|unittest))",
    r"npm test",
    r"npm run test",
    r"pnpm test",
    r"yarn test",
    r"bun test",
    r"(?:npx )?vitest",
    r"(?:npx )?jest",
    r"node --test",
    r"cargo test",
    r"go test",
)
TEST_COMMAND = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]*\s+)*"
    r"(?P<runner>" + "|".join(_RUNNERS) + r")(?:\s|$)"
)
GIT_MUTATION = re.compile(
    r"(?<![\w-])git (?:add|commit|push|checkout|switch|reset|stash|rebase|merge)(?![\w-])"
)
PYTEST_COMMAND = re.compile(r"(?<![\w-])pytest(?![\w-])|py\.test")

RuntimeKind = Literal["python", "node", "rust", "go", "shell"]
SplitChoice = Literal["auto", "train", "validation", "holdout"]


@dataclass(frozen=True)
class MintRejection:
    """Why one episode did not become a task."""

    reason: str


@dataclass(frozen=True)
class CommandResult:
    """Raw result of one helper process such as ``git`` or ``docker``."""

    exit_code: int
    stdout: bytes
    stderr: bytes

    @property
    def text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def error_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace").strip()


CommandRunner = Callable[[Sequence[str]], CommandResult]
ContainerRunner = Callable[[str, Path, int], CommandOutcome]


@dataclass(frozen=True)
class Candidate:
    """One episode with a recorded fail-then-pass test command."""

    episode: Episode
    failing_run: CommandRun
    passing_run: CommandRun

    @property
    def command(self) -> str:
        return self.passing_run.command.strip()


@dataclass(frozen=True)
class Anchor:
    """The repository commit and paths that reproduce the episode start state."""

    toplevel: str
    subdir: str
    base_commit: str
    slug: str
    mutations: tuple[tuple[str, FileMutation], ...]
    overlay: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class Runtime:
    """The language toolchain a workspace needs."""

    kind: RuntimeKind
    image: str
    version_label: str
    project_dir: str
    lockfiles: tuple[str, ...]
    uses_pytest: bool


@dataclass(frozen=True)
class GateRun:
    """The graded result of one containerized command run."""

    status: str
    passed_cases: int
    total_cases: int
    exit_code: int
    seconds: float
    reason: str
    output_tail: str


@dataclass(frozen=True)
class GateResult:
    """Both gate runs and the image identity they ran in."""

    image_digest: str
    architecture: str
    before: GateRun
    reference: GateRun


@dataclass(frozen=True)
class GateFailure:
    """The gate did not prove the task."""

    reason: str
    before: GateRun | None = None
    reference: GateRun | None = None


@dataclass(frozen=True)
class MintOptions:
    """User choices shared by every episode of one run."""

    split: SplitChoice = "auto"
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    gate: bool = True
    keep_failed: bool = False
    limit: int | None = None


@dataclass(frozen=True)
class EpisodeReport:
    """The decision for one episode."""

    session_id: str
    episode_index: int
    status: Literal["minted", "rejected"]
    reason: str
    task_dir: str | None


@dataclass(frozen=True)
class MintReport:
    """Everything one mint run decided."""

    sessions: int
    session_errors: tuple[str, ...]
    episodes: tuple[EpisodeReport, ...]
    minted: int
    rejected: int


def run_git(args: Sequence[str]) -> CommandResult:
    """Run ``git`` with binary capture so blobs and archives stay exact."""
    return _run_process(["git", *args], GIT_TIMEOUT_SECONDS)


def run_docker(args: Sequence[str]) -> CommandResult:
    """Run one ``docker`` CLI command."""
    timeout = (
        DOCKER_BUILD_TIMEOUT_SECONDS
        if args and args[0] == "build"
        else DOCKER_COMMAND_TIMEOUT_SECONDS
    )
    return _run_process(["docker", *args], timeout)


def _run_process(argv: Sequence[str], timeout: int) -> CommandResult:
    try:
        completed = subprocess.run(  # noqa: S603
            list(argv), capture_output=True, text=False, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return CommandResult(127, b"", f"{argv[0]} is not installed".encode())
    except subprocess.TimeoutExpired as error:
        return CommandResult(124, error.stdout or b"", error.stderr or b"")
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


# ---------------------------------------------------------------------------
# Stage 1: candidate detection


def is_test_command(command: str) -> bool:
    """Return whether ``command`` runs a known test runner and stays local.

    Sessions routinely chain a formatter or linter before the runner with
    ``&&``; those segments run offline too, so the runner may be any segment.
    """
    stripped = command.strip()
    if any(pattern.search(stripped) for pattern in (NETWORK_TOOLS, HOST_PATHS, GIT_MUTATION)):
        return False
    segments = [segment.strip() for segment in stripped.split("&&")]
    if segments[0].startswith("cd "):
        target = segments[0][3:].strip()
        if target.startswith("/") or DRIVE_PATH.match(target):
            return False
        segments = segments[1:]
    return any(TEST_COMMAND.match(segment) for segment in segments)


def _failed(run: CommandRun) -> bool:
    return run.is_error or (run.exit_code is not None and run.exit_code != 0)


def select_candidate(episode: Episode) -> Candidate | MintRejection:
    """Find the fail-then-pass test run that proves this episode's change."""
    if not episode.mutations:
        return MintRejection("episode has no file mutations")
    if episode.unresolved_mutations:
        return MintRejection(
            f"{episode.unresolved_mutations} file mutation(s) could not be reconstructed"
        )
    runs = [run for run in episode.commands if is_test_command(run.command)]
    if not runs:
        return MintRejection("no recognized test command was run")
    first_mutation = min(mutation.order for mutation in episode.mutations)
    passing = [run for run in runs if run.exit_code == 0 and run.order > first_mutation]
    if not passing:
        return MintRejection("no passing test run after the file mutations")
    passing_run = passing[-1]
    failing = [run for run in runs if run.order < passing_run.order and _failed(run)]
    if not failing:
        return MintRejection("no failing test run before the passing run")
    return Candidate(episode=episode, failing_run=failing[0], passing_run=passing_run)


# ---------------------------------------------------------------------------
# Stage 2: repository anchoring


def _relative_to(root: str, path: str) -> str | None:
    """Return the POSIX path of ``path`` under ``root`` or None when outside."""
    # realpath expands Windows 8.3 short names so temp and git paths compare.
    try:
        relative = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    except ValueError:
        return None
    if relative == ".":
        return ""
    if relative.startswith("..") or os.path.isabs(relative):
        return None
    return relative.replace(os.sep, "/")


def _toplevel(candidate: Candidate, git: CommandRunner) -> str | MintRejection:
    toplevels: set[str] = set()
    for mutation in candidate.episode.mutations:
        parent = os.path.dirname(mutation.absolute_path)
        if not os.path.isdir(parent):
            return MintRejection(f"directory no longer exists on this host: {parent}")
        result = git(["-C", parent, "rev-parse", "--show-toplevel"])
        if result.exit_code != 0:
            return MintRejection(f"not inside a git repository: {mutation.absolute_path}")
        toplevels.add(os.path.realpath(result.text.strip()))
    if len(toplevels) != 1:
        return MintRejection("file mutations span more than one repository")
    toplevel = toplevels.pop()
    if _relative_to(toplevel, candidate.episode.session.cwd) is None:
        return MintRejection("session working directory is outside the repository")
    return toplevel


def _slug(toplevel: str) -> str | MintRejection:
    name = os.path.basename(toplevel).lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", name).strip("-.")
    if TASK_ID_PATTERN.fullmatch(slug) is None or "/" in slug:
        return MintRejection(f"repository name does not form a task family: {name!r}")
    return slug


def _anchored_mutations(
    toplevel: str, mutations: Iterable[FileMutation]
) -> tuple[tuple[str, FileMutation], ...] | MintRejection:
    anchored: list[tuple[str, FileMutation]] = []
    for mutation in mutations:
        relative = _relative_to(toplevel, mutation.absolute_path)
        if not relative:
            return MintRejection(f"file is outside the repository: {mutation.absolute_path}")
        anchored.append((relative, mutation))
    return tuple(anchored)


def _overlay(
    toplevel: str, earlier: Sequence[Episode]
) -> tuple[tuple[str, str | None], ...]:
    """Final content per path left behind by earlier episodes of the session."""
    final: dict[str, str | None] = {}
    for episode in earlier:
        for mutation in episode.mutations:
            relative = _relative_to(toplevel, mutation.absolute_path)
            if relative:
                final[relative] = None if mutation.kind == "delete" else mutation.content
    return tuple(sorted(final.items()))


def _normalize_eol(text: str) -> str:
    return text.replace("\r\n", "\n")


def _blob(git: CommandRunner, toplevel: str, commit: str, relative: str) -> str | None:
    result = git(["-C", toplevel, "show", f"{commit}:{relative}"])
    return result.text if result.exit_code == 0 else None


def _commit_matches(
    git: CommandRunner,
    toplevel: str,
    commit: str,
    first_mutations: Mapping[str, FileMutation],
    overlay: Mapping[str, str | None],
) -> bool:
    for relative, mutation in first_mutations.items():
        if relative in overlay:
            pre_state = overlay[relative]
        else:
            pre_state = _blob(git, toplevel, commit, relative)
        if mutation.kind == "write":
            # A write records no old text, so an existing but different file
            # cannot contradict the commit; an identical file means the
            # commit already contains this episode's result.
            if pre_state is not None and _normalize_eol(pre_state) == _normalize_eol(
                mutation.content
            ):
                return False
            continue
        if pre_state is None or mutation.old_text is None:
            return False
        if _normalize_eol(pre_state) != _normalize_eol(mutation.old_text):
            return False
    return True


def _base_commits(git: CommandRunner, toplevel: str, before: datetime) -> tuple[str, ...]:
    result = git(
        [
            "-C",
            toplevel,
            "rev-list",
            "--all",
            f"--max-count={MAX_BASE_COMMITS}",
            f"--before={before.isoformat()}",
        ]
    )
    if result.exit_code != 0:
        return ()
    return tuple(line.strip() for line in result.text.splitlines() if line.strip())


def anchor_repository(
    candidate: Candidate,
    earlier: Sequence[Episode] = (),
    *,
    git: CommandRunner = run_git,
) -> Anchor | MintRejection:
    """Resolve the repository and the newest commit matching the pre-edit files."""
    toplevel = _toplevel(candidate, git)
    if isinstance(toplevel, MintRejection):
        return toplevel
    slug = _slug(toplevel)
    if isinstance(slug, MintRejection):
        return slug
    mutations = _anchored_mutations(toplevel, candidate.episode.mutations)
    if isinstance(mutations, MintRejection):
        return mutations
    overlay = dict(_overlay(toplevel, earlier))
    first: dict[str, FileMutation] = {}
    for relative, mutation in mutations:
        first.setdefault(relative, mutation)
    commits = _base_commits(git, toplevel, candidate.episode.started_at)
    if not commits:
        return MintRejection("no commit predates the episode")
    for commit in commits:
        if _commit_matches(git, toplevel, commit, first, overlay):
            subdir = _relative_to(toplevel, candidate.episode.session.cwd) or ""
            return Anchor(
                toplevel=toplevel,
                subdir=subdir,
                base_commit=commit,
                slug=slug,
                mutations=mutations,
                overlay=tuple(sorted(overlay.items())),
            )
    return MintRejection("no commit matches the pre-edit file contents")


def sealed_command(anchor: Anchor, command: str) -> str:
    """Return the shell command the verifier runs from ``/workspace``."""
    if anchor.subdir:
        return f"cd {shlex.quote(anchor.subdir)} && {command}"
    return command


# ---------------------------------------------------------------------------
# Stage 3: workspace materialization


def _safe_member_path(name: str) -> str | None:
    normalized = posixpath.normpath(name)
    parts = normalized.split("/")
    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        return None
    return normalized


def _extract_archive(archive: bytes, workspace: Path) -> MintRejection | None:
    try:
        tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:")
    except tarfile.TarError as error:
        return MintRejection(f"git archive is unreadable: {error}")
    with tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = _safe_member_path(member.name)
            if relative is None:
                return MintRejection(f"archive entry escapes the workspace: {member.name}")
            if posixpath.basename(relative).startswith(".env"):
                continue
            if member.size > MAX_ARCHIVE_FILE_BYTES:
                return MintRejection(f"repository file exceeds 32 MiB: {relative}")
            stream = tar.extractfile(member)
            if stream is None:
                continue
            target = workspace.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.read())
    return None


def _write_state(root: Path, relative: str, content: str | None) -> None:
    """Write or delete one file, keeping CRLF only where the file already has it."""
    target = root.joinpath(*relative.split("/"))
    if content is None:
        target.unlink(missing_ok=True)
        return
    keep_crlf = target.is_file() and b"\r\n" in target.read_bytes()
    text = _normalize_eol(content)
    if keep_crlf:
        text = text.replace("\n", "\r\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))


def materialize_workspace(
    anchor: Anchor, workspace: Path, *, git: CommandRunner = run_git
) -> MintRejection | None:
    """Fill ``workspace`` with the start state of the episode."""
    result = git(["-C", anchor.toplevel, "archive", "--format=tar", anchor.base_commit])
    if result.exit_code != 0:
        return MintRejection(f"git archive failed: {result.error_text}")
    workspace.mkdir(parents=True, exist_ok=True)
    failure = _extract_archive(result.stdout, workspace)
    if failure is not None:
        return failure
    for relative, content in anchor.overlay:
        _write_state(workspace, relative, content)
    return None


def apply_mutations(anchor: Anchor, root: Path) -> None:
    """Advance ``root`` from the start state to the end state of the episode."""
    for relative, mutation in anchor.mutations:
        _write_state(root, relative, None if mutation.kind == "delete" else mutation.content)


def _git_in(git: CommandRunner, repo: Path, *args: str) -> CommandResult:
    return git(["-C", str(repo), *GIT_NO_EOL_CONVERSION, *GIT_IDENTITY, *args])


def reference_patch(
    anchor: Anchor, workspace: Path, scratch: Path, *, git: CommandRunner = run_git
) -> tuple[bytes, Path] | MintRejection:
    """Return the start -> end binary patch and the end-state tree."""
    repo = scratch / "end"
    shutil.copytree(workspace, repo)
    steps = (
        ("init", "-q", "-b", "main"),
        ("add", "-A", "-f"),
        ("commit", "-q", "--allow-empty", "-m", "start"),
    )
    for step in steps:
        result = _git_in(git, repo, *step)
        if result.exit_code != 0:
            return MintRejection(f"git {step[0]} failed: {result.error_text}")
    apply_mutations(anchor, repo)
    result = _git_in(git, repo, "add", "-A", "-f")
    if result.exit_code != 0:
        return MintRejection(f"git add failed: {result.error_text}")
    result = _git_in(git, repo, "diff", "--cached", "--binary")
    if result.exit_code != 0:
        return MintRejection(f"git diff failed: {result.error_text}")
    if not result.stdout.strip():
        return MintRejection("reference patch is empty")
    return result.stdout, repo


def is_test_path(relative: str) -> bool:
    """Return whether one POSIX repository path looks like a test file."""
    parts = relative.split("/")
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in TEST_FILE_GLOBS)


def _command_files(command: str, subdir: str, end_state: Path) -> set[str]:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return set()
    found: set[str] = set()
    for token in tokens:
        candidate = token.split("::", 1)[0]
        if not candidate or candidate.startswith("-") or "\\" in candidate:
            continue
        relative = posixpath.normpath(posixpath.join(subdir, candidate))
        if relative.startswith(("..", "/")) or relative == ".":
            continue
        if end_state.joinpath(*relative.split("/")).is_file():
            found.add(relative)
    return found


def _all_test_files(end_state: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(end_state.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(end_state).as_posix()
        if relative.startswith(".git/"):
            continue
        if is_test_path(relative):
            found.append(relative)
    return found


def select_sealed_files(
    anchor: Anchor, command: str, end_state: Path
) -> tuple[str, ...] | MintRejection:
    """Choose the end-state files the verifier must not trust from the agent."""
    sealed = {
        relative
        for relative, mutation in anchor.mutations
        if is_test_path(relative) and end_state.joinpath(*relative.split("/")).is_file()
    }
    sealed |= _command_files(command, anchor.subdir, end_state)
    if not sealed:
        sealed = set(_all_test_files(end_state))
        if not sealed:
            return MintRejection("no test file to seal")
    if len(sealed) > MAX_SEALED_FILES:
        return MintRejection(f"more than {MAX_SEALED_FILES} test files to seal")
    return tuple(sorted(sealed))


# ---------------------------------------------------------------------------
# Stage 4: runtime detection and Dockerfile


def _has(workspace: Path, project_dir: str, *globs: str) -> bool:
    root = workspace.joinpath(*project_dir.split("/")) if project_dir else workspace
    return any(any(root.glob(pattern)) for pattern in globs)


def _runtime_kind(workspace: Path, project_dir: str) -> RuntimeKind | None:
    if _has(workspace, project_dir, "pyproject.toml", "requirements*.txt", "setup.py"):
        return "python"
    if _has(workspace, project_dir, "package.json"):
        return "node"
    if _has(workspace, project_dir, "Cargo.toml"):
        return "rust"
    if _has(workspace, project_dir, "go.mod"):
        return "go"
    return None


def _lockfiles(workspace: Path, project_dir: str) -> tuple[str, ...]:
    root = workspace.joinpath(*project_dir.split("/")) if project_dir else workspace
    names = [*LOCKFILE_NAMES, *sorted(path.name for path in root.glob("requirements*.txt"))]
    found = [name for name in names if (root / name).is_file()]
    return tuple(sorted(set(found)))


def detect_runtime(workspace: Path, subdir: str, command: str) -> Runtime:
    """Pick the base image from the project files nearest the session cwd."""
    images = {
        "python": ("python:3.12-slim-bookworm", "3.12"),
        "node": ("node:24-bookworm", "24"),
        "rust": ("rust:1-bookworm", "1"),
        "go": ("golang:1-bookworm", "1"),
    }
    for project_dir in dict.fromkeys((subdir, "")):
        kind = _runtime_kind(workspace, project_dir)
        if kind is not None:
            image, label = images[kind]
            return Runtime(
                kind=kind,
                image=image,
                version_label=label,
                project_dir=project_dir,
                lockfiles=_lockfiles(workspace, project_dir),
                uses_pytest=PYTEST_COMMAND.search(command) is not None,
            )
    return Runtime("shell", "debian:bookworm-slim", "bookworm", subdir, (), False)


def _python_install(runtime: Runtime) -> list[str]:
    if "uv.lock" in runtime.lockfiles:
        return [
            "RUN pip install --no-cache-dir uv"
            " && UV_PROJECT_ENVIRONMENT=/workspace/.venv uv sync --frozen",
            "ENV PATH=/workspace/.venv/bin:$PATH",
        ]
    steps = [
        f"pip install --no-cache-dir -r {name}"
        for name in runtime.lockfiles
        if name.startswith("requirements")
    ]
    if runtime.uses_pytest:
        steps.append("pip install --no-cache-dir pytest")
    steps.append("if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install --no-cache-dir -e .; fi")
    return ["RUN " + " && ".join(steps)]


def _node_install(runtime: Runtime) -> list[str]:
    locks = set(runtime.lockfiles)
    if "package-lock.json" in locks:
        return ["RUN npm ci"]
    if locks & {"bun.lock", "bun.lockb"}:
        return ["RUN npm i -g bun && bun install --frozen-lockfile"]
    if "pnpm-lock.yaml" in locks:
        return ["RUN corepack enable && pnpm install --frozen-lockfile"]
    if "yarn.lock" in locks:
        return ["RUN corepack enable && yarn install --frozen-lockfile"]
    return ["RUN npm install"]


def _install_lines(runtime: Runtime) -> list[str]:
    if runtime.kind == "python":
        return _python_install(runtime)
    if runtime.kind == "node":
        return _node_install(runtime)
    if runtime.kind == "rust":
        return ["RUN cargo fetch && cargo test --no-run || true"]
    if runtime.kind == "go":
        return ["RUN go mod download && go build ./... || true"]
    return []


def render_dockerfile(runtime: Runtime, workspace: Path) -> str:
    """Render the task image build: toolchain, dependencies, then offline."""
    extra = " build-essential" if runtime.kind in {"python", "rust"} else ""
    project = f"/workspace/{runtime.project_dir}" if runtime.project_dir else "/workspace"
    excludes = " ".join(shlex.quote(item) for item in EXCLUDED_DIRS)
    lines = [
        f"FROM {runtime.image}",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "RUN apt-get update"
        f" && apt-get install -y --no-install-recommends {BASE_PACKAGES}{extra}"
        " && rm -rf /var/lib/apt/lists/*",
        "COPY workspace/ /workspace/",
        f"WORKDIR {project}",
        *_install_lines(runtime),
        "WORKDIR /workspace",
        f"ENV {OFFLINE_ENV}",
        f"RUN mkdir -p {posixpath.dirname(EXCLUDES_FILE)}"
        f" && printf '%s\\n' {excludes} > {EXCLUDES_FILE}",
        f"RUN git init -q -b main && {CONTAINER_GIT} add -A"
        f" && {CONTAINER_GIT} commit -qm 'omp-gym start state'"
        f" && git tag {START_TAG}",
        "",
    ]
    return "\n".join(lines)


def dependency_lock_digest(runtime: Runtime, workspace: Path) -> str:
    """Hash the lockfiles that pin the image's dependencies."""
    if not runtime.lockfiles:
        return EMPTY_DEPENDENCY_LOCK_DIGEST
    root = (
        workspace.joinpath(*runtime.project_dir.split("/"))
        if runtime.project_dir
        else workspace
    )
    digest = hashlib.sha256()
    for name in runtime.lockfiles:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Stage 6: task assembly


def assign_split(slug: str, choice: SplitChoice) -> Split:
    """Split by repository so no family leaks between train and holdout."""
    if choice != "auto":
        return choice
    bucket = hashlib.sha256(slug.encode("utf-8")).digest()[-1] % SPLIT_BUCKETS
    if bucket < TRAIN_SPLIT_BUCKETS:
        return "train"
    if bucket < VALIDATION_SPLIT_BUCKETS:
        return "validation"
    return "holdout"


def task_name(slug: str, session_id: str, episode_index: int) -> str:
    return f"{slug}-{session_id[:SESSION_ID_PREFIX_LENGTH]}-e{episode_index}"


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def render_task_toml(document: Mapping[str, object]) -> str:
    """Render top-level scalars followed by one level of tables."""
    scalars = [
        f"{key} = {_toml_value(value)}"
        for key, value in document.items()
        if not isinstance(value, Mapping)
    ]
    tables: list[str] = []
    for key, value in document.items():
        if isinstance(value, Mapping):
            tables.append("")
            tables.append(f"[{key}]")
            tables.extend(f"{name} = {_toml_value(item)}" for name, item in value.items())
    return "\n".join([*scalars, *tables, ""])


def _task_document(
    *,
    name: str,
    anchor: Anchor,
    candidate: Candidate,
    runtime: Runtime,
    prompt: str,
    sealed: Sequence[str],
    lock_digest: str,
    options: MintOptions,
) -> dict[str, object]:
    episode = candidate.episode
    return {
        "schema_version": TEST_COMMAND_SCHEMA_VERSION,
        "task_id": f"omp-session/{name}",
        "task_revision": 1,
        "family": anchor.slug,
        "split": assign_split(anchor.slug, options.split),
        "prompt": prompt,
        "runtime": runtime.kind,
        "runtime_version": runtime.version_label,
        "max_time_seconds": MAX_TIME_SECONDS,
        "token_budget": TOKEN_BUDGET,
        "expected_cases": 1,
        "source": f"omp-session:{episode.session.id}/{episode.index}",
        "source_revision": anchor.base_commit,
        "license": "NOASSERTION",
        "sensitive_data": "private",
        "seed": 0,
        "environment": {
            "image": f"omp-gym/{name}:1",
            "image_digest": PLACEHOLDER_IMAGE_DIGEST,
            "os": "linux",
            "architecture": "amd64",
            "network": "none",
            "cpus": CPU_LIMIT,
            "memory_bytes": MEMORY_BYTES,
            "pids": PID_LIMIT,
            "workspace_bytes": WORKSPACE_BYTES,
            "temp_bytes": TEMP_BYTES,
            "home_bytes": HOME_BYTES,
            "dependency_lock_digest": lock_digest,
        },
        "verifier": {
            "protocol": "test-command-v1",
            "command": ["sh", "-c", sealed_command(anchor, candidate.command)],
            "timeout_seconds": options.timeout_seconds,
            "sealed_files": list(sealed),
            "reference": "verifier/reference.patch",
        },
    }


def _provenance(candidate: Candidate, anchor: Anchor, command: str) -> dict[str, object]:
    episode = candidate.episode
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "session_id": episode.session.id,
        "session_cwd_sha256": hashlib.sha256(
            episode.session.cwd.encode("utf-8")
        ).hexdigest(),
        "episode_index": episode.index,
        "started_at": episode.started_at.isoformat(),
        "ended_at": episode.ended_at.isoformat(),
        "models": list(episode.models),
        "usage": asdict(episode.usage),
        "assistant_turns": episode.assistant_turns,
        "tool_calls": episode.tool_calls,
        "base_commit": anchor.base_commit,
        "repository": anchor.slug,
        "test_command": command,
        "failing_run": {
            "exit_code": candidate.failing_run.exit_code,
            "order": candidate.failing_run.order,
        },
        "passing_run": {
            "exit_code": candidate.passing_run.exit_code,
            "order": candidate.passing_run.order,
        },
        "gate": None,
        "minted_at": datetime.now(UTC).isoformat(),
        "mint_version": MINT_VERSION,
    }


def _home_to_tilde(text: str) -> str:
    home = str(Path.home())
    for variant in dict.fromkeys((home, home.replace("\\", "/"), home.replace("/", "\\"))):
        text = text.replace(variant, "~")
    return text


def task_prompt(episode: Episode) -> str | MintRejection:
    """Return the user request as the task prompt."""
    prompt = _home_to_tilde(clean_prompt(episode.prompt)).strip()
    if not prompt:
        return MintRejection("prompt is empty after cleaning")
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        return MintRejection(f"prompt exceeds {MAX_PROMPT_BYTES // 1024} KiB")
    return prompt


def find_secret(text: str) -> str | None:
    """Return the name of the first secret-looking token in ``text``."""
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return match.group(0)[:8] + "..."
    return None


def _secret_rejection(
    prompt: str, patch: bytes, sealed: Sequence[str], end_state: Path
) -> MintRejection | None:
    if find_secret(prompt) is not None:
        return MintRejection("prompt contains a secret-looking token")
    if find_secret(patch.decode("utf-8", errors="replace")) is not None:
        return MintRejection("reference patch contains a secret-looking token")
    for relative in sealed:
        text = end_state.joinpath(*relative.split("/")).read_text(
            encoding="utf-8", errors="replace"
        )
        if find_secret(text) is not None:
            return MintRejection(f"sealed file contains a secret-looking token: {relative}")
    return None


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _copy_sealed(end_state: Path, sealed: Sequence[str], task_dir: Path) -> None:
    sealed_root = task_dir.joinpath(*SEALED_FILES_DIR.split("/"))
    for relative in sealed:
        target = sealed_root.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(end_state.joinpath(*relative.split("/")), target)


def _assemble(
    candidate: Candidate,
    anchor: Anchor,
    task_dir: Path,
    scratch: Path,
    options: MintOptions,
    git: CommandRunner,
) -> MintRejection | None:
    prompt = task_prompt(candidate.episode)
    if isinstance(prompt, MintRejection):
        return prompt
    workspace = task_dir / "workspace"
    failure = materialize_workspace(anchor, workspace, git=git)
    if failure is not None:
        return failure
    patch = reference_patch(anchor, workspace, scratch, git=git)
    if isinstance(patch, MintRejection):
        return patch
    patch_bytes, end_state = patch
    sealed = select_sealed_files(anchor, candidate.command, end_state)
    if isinstance(sealed, MintRejection):
        return sealed
    failure = _secret_rejection(prompt, patch_bytes, sealed, end_state)
    if failure is not None:
        return failure
    runtime = detect_runtime(workspace, anchor.subdir, candidate.command)
    name = task_dir.name
    document = _task_document(
        name=name,
        anchor=anchor,
        candidate=candidate,
        runtime=runtime,
        prompt=prompt,
        sealed=sealed,
        lock_digest=dependency_lock_digest(runtime, workspace),
        options=options,
    )
    (task_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (task_dir / "verifier" / "reference.patch").write_bytes(patch_bytes)
    _copy_sealed(end_state, sealed, task_dir)
    (task_dir / "Dockerfile").write_text(render_dockerfile(runtime, workspace), encoding="utf-8")
    (task_dir / ".dockerignore").write_text(
        "verifier/\nprovenance.json\ntask.toml\n", encoding="utf-8"
    )
    (task_dir / "task.toml").write_text(render_task_toml(document), encoding="utf-8")
    command = sealed_command(anchor, candidate.command)
    _write_json(task_dir / "provenance.json", _provenance(candidate, anchor, command))
    loaded = load_task(task_dir)
    if isinstance(loaded, TaskLoadError):
        return MintRejection(f"minted task does not load: {loaded.reason}")
    return None


def mint_episode(
    candidate: Candidate,
    anchor: Anchor,
    output_dir: Path,
    options: MintOptions,
    *,
    git: CommandRunner = run_git,
) -> Path | MintRejection:
    """Write one ungated task directory for an anchored candidate."""
    episode = candidate.episode
    task_dir = output_dir / task_name(anchor.slug, episode.session.id, episode.index)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="omp-mint-") as scratch:
        try:
            failure = _assemble(candidate, anchor, task_dir, Path(scratch), options, git)
        except OSError as error:
            failure = MintRejection(f"task write failed: {error}")
    if failure is not None:
        shutil.rmtree(task_dir, ignore_errors=True)
        return failure
    return task_dir


# ---------------------------------------------------------------------------
# Stage 5: gate


def _bounded(data: bytes) -> str:
    return data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")


def run_container(image: str, staging: Path, timeout_seconds: int) -> CommandOutcome:
    """Run the staged grading script inside a fresh offline container."""
    limits = _container_limits(staging)
    created = run_docker(
        ["create", "--network", "none", *limits, image, "sh", "/run/omp-gym/grade.sh"]
    )
    if created.exit_code != 0:
        return CommandOutcome(created.exit_code, created.text, created.error_text)
    container = created.text.strip()
    try:
        copied = run_docker(["cp", f"{staging}{os.sep}.", f"{container}:/run/omp-gym"])
        if copied.exit_code != 0:
            return CommandOutcome(copied.exit_code, copied.text, copied.error_text)
        try:
            completed = subprocess.run(  # noqa: S603 - fixed docker executable.
                ["docker", "start", "-a", container],  # noqa: S607
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return CommandOutcome(
                124, _bounded(error.stdout or b""), _bounded(error.stderr or b""), True
            )
        return CommandOutcome(
            completed.returncode, _bounded(completed.stdout), _bounded(completed.stderr)
        )
    finally:
        run_docker(["rm", "-f", container])


def _container_limits(staging: Path) -> list[str]:
    """Read the resource limits the minter staged next to the script."""
    limits_path = staging / "limits.json"
    if not limits_path.is_file():
        return []
    limits = _read_json(limits_path)
    return [
        "--memory",
        str(limits.get("memory_bytes", MEMORY_BYTES)),
        "--cpus",
        str(limits.get("cpus", CPU_LIMIT)),
        "--pids-limit",
        str(limits.get("pids", PID_LIMIT)),
    ]


def _stage(
    spec: TaskSpec, staging: Path, *, apply_patch: bool
) -> None:
    verifier = spec.verifier
    sealed_dir = staging / "sealed"
    for relative in verifier.sealed_files:
        target = sealed_dir.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verifier.sealed_root.joinpath(*relative.split("/")), target)
    if apply_patch:
        shutil.copyfile(verifier.reference_patch, staging / "changes.patch")
    script = grading_script(verifier.command, verifier.sealed_files, apply_patch=apply_patch)
    (staging / "grade.sh").write_bytes(script.encode("utf-8"))
    _write_json(
        staging / "limits.json",
        {
            "memory_bytes": spec.environment.memory_bytes,
            "cpus": spec.environment.cpus,
            "pids": spec.environment.pids,
        },
    )


def _gate_run(
    spec: TaskSpec,
    runner: ContainerRunner,
    *,
    apply_patch: bool,
    expected_cases: int,
) -> tuple[GateRun, TestCommandResult]:
    with tempfile.TemporaryDirectory(prefix="omp-gate-") as staging_name:
        staging = Path(staging_name)
        _stage(spec, staging, apply_patch=apply_patch)
        started = datetime.now(UTC)
        outcome = runner(spec.environment.image, staging, spec.verifier.timeout_seconds)
        seconds = (datetime.now(UTC) - started).total_seconds()
    graded = grade_outcome(outcome, expected_cases)
    run = GateRun(
        status=graded.status,
        passed_cases=graded.passed_cases,
        total_cases=graded.total_cases,
        exit_code=outcome.exit_code,
        seconds=round(seconds, 3),
        reason=graded.reason,
        output_tail=(outcome.stdout + "\n" + outcome.stderr)[-OUTPUT_TAIL_CHARS:],
    )
    return run, graded


def _build_image(task_dir: Path, image: str, docker: CommandRunner) -> tuple[str, str] | GateFailure:
    built = docker(["build", "-t", image, str(task_dir)])
    if built.exit_code != 0:
        return GateFailure(f"docker build failed: {built.error_text[-2000:]}")
    inspected = docker(["image", "inspect", "--format", "{{.Id}}|{{.Architecture}}", image])
    if inspected.exit_code != 0:
        return GateFailure(f"docker image inspect failed: {inspected.error_text}")
    digest, _, architecture = inspected.text.strip().partition("|")
    if not digest.startswith("sha256:") or architecture not in {"amd64", "arm64"}:
        return GateFailure(f"unexpected image identity: {inspected.text.strip()!r}")
    return digest, architecture


def _run_gate(
    spec: TaskSpec, docker: CommandRunner, runner: ContainerRunner
) -> GateResult | GateFailure:
    identity = _build_image(spec.task_root, spec.environment.image, docker)
    if isinstance(identity, GateFailure):
        return identity
    digest, architecture = identity
    reference, graded = _gate_run(spec, runner, apply_patch=True, expected_cases=1)
    if graded.status != "passed" or graded.passed_cases < 1:
        return GateFailure(
            f"reference run did not pass: {graded.reason}", reference=reference
        )
    before, graded_before = _gate_run(
        spec, runner, apply_patch=False, expected_cases=graded.passed_cases
    )
    if graded_before.status == "passed":
        return GateFailure(
            "tests already pass on the start state", before=before, reference=reference
        )
    return GateResult(digest, architecture, before, reference)


def _record_gate(task_dir: Path, result: GateResult) -> None:
    document = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    document["expected_cases"] = result.reference.passed_cases
    environment = dict(document["environment"])
    environment["image_digest"] = result.image_digest
    environment["architecture"] = result.architecture
    document["environment"] = environment
    (task_dir / "task.toml").write_text(render_task_toml(document), encoding="utf-8")
    provenance_path = task_dir / "provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
    provenance["gate"] = {"before": asdict(result.before), "reference": asdict(result.reference)}
    _write_json(provenance_path, provenance)


def gate_task(
    task_dir: Path,
    *,
    docker: CommandRunner = run_docker,
    runner: ContainerRunner = run_container,
    timeout_seconds: int | None = None,
    keep_failed: bool = False,
) -> GateResult | GateFailure:
    """Build the task image and prove the reference patch flips the tests.

    Works on freshly minted directories and on directories whose Dockerfile
    was edited by hand: the image is rebuilt and both runs are repeated.
    """
    spec = load_task(task_dir)
    if isinstance(spec, TaskLoadError):
        return GateFailure(f"task does not load: {spec.reason}")
    if not isinstance(spec.verifier, TestCommandVerifier):
        return GateFailure("only test-command-v1 tasks can be gated")
    if timeout_seconds is not None:
        spec = replace(spec, verifier=replace(spec.verifier, timeout_seconds=timeout_seconds))
    result = _run_gate(spec, docker, runner)
    if isinstance(result, GateFailure):
        if not keep_failed:
            shutil.rmtree(task_dir, ignore_errors=True)
        return result
    try:
        _record_gate(task_dir, result)
    except (OSError, tomllib.TOMLDecodeError) as error:
        return GateFailure(f"gate result could not be recorded: {error}")
    reloaded = load_task(task_dir)
    if isinstance(reloaded, TaskLoadError):
        return GateFailure(f"gated task does not load: {reloaded.reason}")
    return result


# ---------------------------------------------------------------------------
# Stage 7: driver and CLI


def _episodes(sessions_dir: Path) -> tuple[list[tuple[Path, tuple[Episode, ...]]], list[str]]:
    loaded: list[tuple[Path, tuple[Episode, ...]]] = []
    errors: list[str] = []
    for path in discover_sessions(sessions_dir):
        session = read_session(path)
        if isinstance(session, SessionLoadError):
            errors.append(f"{session.path}: {session.reason}")
            continue
        loaded.append((path, session_episodes(session)))
    return loaded, errors


def _rejected(episode: Episode, reason: str) -> EpisodeReport:
    return EpisodeReport(episode.session.id, episode.index, "rejected", reason, None)


def _mint_one(
    episode: Episode,
    earlier: Sequence[Episode],
    output_dir: Path,
    options: MintOptions,
    *,
    git: CommandRunner,
    docker: CommandRunner,
    runner: ContainerRunner,
) -> EpisodeReport:
    candidate = select_candidate(episode)
    if isinstance(candidate, MintRejection):
        return _rejected(episode, candidate.reason)
    anchor = anchor_repository(candidate, earlier, git=git)
    if isinstance(anchor, MintRejection):
        return _rejected(episode, anchor.reason)
    task_dir = mint_episode(candidate, anchor, output_dir, options, git=git)
    if isinstance(task_dir, MintRejection):
        return _rejected(episode, task_dir.reason)
    if options.gate:
        gated = gate_task(
            task_dir, docker=docker, runner=runner, keep_failed=options.keep_failed
        )
        if isinstance(gated, GateFailure):
            kept = str(task_dir) if options.keep_failed else None
            return EpisodeReport(
                episode.session.id, episode.index, "rejected", gated.reason, kept
            )
    return EpisodeReport(episode.session.id, episode.index, "minted", "", str(task_dir))


def mint_sessions(
    sessions_dir: Path,
    output_dir: Path,
    options: MintOptions | None = None,
    *,
    git: CommandRunner = run_git,
    docker: CommandRunner = run_docker,
    runner: ContainerRunner = run_container,
) -> MintReport:
    """Mint every eligible episode under ``sessions_dir`` into ``output_dir``."""
    options = options or MintOptions()
    sessions, errors = _episodes(sessions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[EpisodeReport] = []
    minted = 0
    for _, episodes in sessions:
        for index, episode in enumerate(episodes):
            if options.limit is not None and minted >= options.limit:
                break
            report = _mint_one(
                episode,
                episodes[:index],
                output_dir,
                options,
                git=git,
                docker=docker,
                runner=runner,
            )
            minted += report.status == "minted"
            reports.append(report)
    report = MintReport(
        sessions=len(sessions),
        session_errors=tuple(errors),
        episodes=tuple(reports),
        minted=minted,
        rejected=len(reports) - minted,
    )
    _write_json(output_dir / REPORT_NAME, asdict(report))
    return report


def scan_sessions(
    sessions_dir: Path, *, limit: int | None = None, git: CommandRunner = run_git
) -> tuple[EpisodeReport, ...]:
    """Report the decision for each episode without writing anything."""
    sessions, errors = _episodes(sessions_dir)
    decisions: list[EpisodeReport] = [
        EpisodeReport("", -1, "rejected", error, None) for error in errors
    ]
    for _, episodes in sessions:
        for index, episode in enumerate(episodes):
            if limit is not None and len(decisions) >= limit:
                return tuple(decisions)
            candidate = select_candidate(episode)
            if isinstance(candidate, MintRejection):
                decisions.append(_rejected(episode, candidate.reason))
                continue
            anchor = anchor_repository(candidate, episodes[:index], git=git)
            if isinstance(anchor, MintRejection):
                decisions.append(_rejected(episode, anchor.reason))
                continue
            decisions.append(
                EpisodeReport(
                    episode.session.id,
                    episode.index,
                    "minted",
                    f"candidate at {anchor.slug}@{anchor.base_commit[:12]}: {candidate.command}",
                    None,
                )
            )
    return tuple(decisions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omp-coding-mint")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="list episodes and their mint decision")
    scan.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_DIR)
    scan.add_argument("--limit", type=int, default=None)
    mint = subparsers.add_parser("mint", help="write sealed tasks from sessions")
    mint.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS_DIR)
    mint.add_argument("--output", type=Path, required=True)
    mint.add_argument("--limit", type=int, default=None)
    mint.add_argument(
        "--split", choices=("auto", "train", "validation", "holdout"), default="auto"
    )
    mint.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    mint.add_argument("--no-gate", action="store_true")
    mint.add_argument("--keep-failed", action="store_true")
    gate = subparsers.add_parser("gate", help="build and prove existing task directories")
    gate.add_argument("task_dirs", nargs="+", type=Path)
    gate.add_argument("--timeout-seconds", type=int, default=None)
    gate.add_argument("--keep-failed", action="store_true")
    return parser


def _print_decision(decision: EpisodeReport) -> None:
    label = "candidate" if decision.status == "minted" else "rejected"
    print(f"{decision.session_id or '-'}\t{decision.episode_index}\t{label}\t{decision.reason}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "scan":
        for decision in scan_sessions(arguments.sessions, limit=arguments.limit):
            _print_decision(decision)
        return 0
    if arguments.command == "mint":
        options = MintOptions(
            split=arguments.split,
            timeout_seconds=arguments.timeout_seconds,
            gate=not arguments.no_gate,
            keep_failed=arguments.keep_failed,
            limit=arguments.limit,
        )
        report = mint_sessions(arguments.sessions, arguments.output, options)
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return 0 if report.minted or not report.episodes else 1
    failures = 0
    for task_dir in arguments.task_dirs:
        result = gate_task(
            task_dir,
            timeout_seconds=arguments.timeout_seconds,
            keep_failed=arguments.keep_failed,
        )
        failures += isinstance(result, GateFailure)
        print(json.dumps({"task_dir": str(task_dir), **asdict(result)}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
