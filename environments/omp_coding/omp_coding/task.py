"""Versioned task contract with sealed verifier files.

Schema version 2 describes a sealed ``call-cases-v1`` task: a small public
workspace, declared editable files, and one JSON case file that calls the
candidate through a language worker.

Schema version 3 describes a ``test-command-v1`` task minted from an OMP
session: a complete repository workspace baked into a task image, one sealed
test command, sealed test files, and the reference patch that made the
command pass.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .runtime import (
    CPU_LIMIT,
    HOME_BYTES,
    MEMORY_BYTES,
    PID_LIMIT,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    TEMP_BYTES,
    WORKSPACE_BYTES,
    RuntimeFailure,
    inspect_plain_tree,
)

TASK_SCHEMA_VERSION = 2
TEST_COMMAND_SCHEMA_VERSION = 3
MAX_TASK_CONFIG_BYTES = 256 * 1024
MAX_PROMPT_BYTES = 32 * 1024
MAX_TIME_SECONDS = 30 * 60
MAX_EXPECTED_CASES = 1000
MAX_TOKEN_BUDGET = 1_000_000
MAX_COMMAND_ITEMS = 64
MAX_COMMAND_ITEM_BYTES = 1024
MAX_SEALED_FILES = 256
MAX_REPOSITORY_FILES = 20_000
MAX_REPOSITORY_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMORY_BYTES = 64 * 1024 * 1024 * 1024
MAX_SCRATCH_BYTES = 1024 * 1024 * 1024
MAX_PIDS = 4096
MAX_CPUS = 16.0
EMPTY_DEPENDENCY_LOCK_DIGEST = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
TASK_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._/-]{2,127}\Z")
LICENSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}\Z")
IMAGE_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
RUNTIME_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}\Z")
_SHARED_FIELDS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_revision",
        "family",
        "split",
        "prompt",
        "runtime",
        "runtime_version",
        "max_time_seconds",
        "token_budget",
        "expected_cases",
        "source",
        "source_revision",
        "license",
        "sensitive_data",
        "seed",
        "environment",
        "verifier",
    }
)
_CALL_CASES_FIELDS = _SHARED_FIELDS | {"editable_files", "context_files"}
_TEST_COMMAND_FIELDS = _SHARED_FIELDS
_ENVIRONMENT_FIELDS = frozenset(
    {
        "image",
        "image_digest",
        "os",
        "architecture",
        "network",
        "cpus",
        "memory_bytes",
        "pids",
        "workspace_bytes",
        "temp_bytes",
        "home_bytes",
        "dependency_lock_digest",
    }
)
_CALL_CASES_VERIFIER_FIELDS = frozenset({"protocol", "cases"})
_TEST_COMMAND_VERIFIER_FIELDS = frozenset(
    {"protocol", "command", "timeout_seconds", "sealed_files", "reference"}
)
_CALL_CASES_RUNTIMES = {"python": "3.11.2", "node": "24.19.0"}
TEST_COMMAND_RUNTIMES = frozenset({"python", "node", "rust", "go", "shell"})
SEALED_FILES_DIR = "verifier/files"

Split = Literal["train", "validation", "holdout"]
SensitiveData = Literal["public", "private"]
Architecture = Literal["arm64", "amd64"]


@dataclass(frozen=True)
class EnvironmentContract:
    """Exact runtime and resource identity for one task."""

    image: str
    image_digest: str
    os: Literal["linux"]
    architecture: Architecture
    network: Literal["none"]
    cpus: float
    memory_bytes: int
    pids: int
    workspace_bytes: int
    temp_bytes: int
    home_bytes: int
    dependency_lock_digest: str


@dataclass(frozen=True)
class CallCasesVerifier:
    """Sealed structured cases executed through a language worker."""

    protocol: Literal["call-cases-v1"]
    cases_path: Path


@dataclass(frozen=True)
class TestCommandVerifier:
    """One sealed test command with the files it must not trust from the agent."""

    protocol: Literal["test-command-v1"]
    command: tuple[str, ...]
    timeout_seconds: int
    sealed_files: tuple[str, ...]
    sealed_root: Path
    reference_patch: Path


Verifier = CallCasesVerifier | TestCommandVerifier


@dataclass(frozen=True)
class TaskSpec:
    """One complete and reproducible coding task."""

    schema_version: int
    task_id: str
    task_revision: int
    name: str
    family: str
    split: Split
    prompt: str
    runtime: str
    runtime_version: str
    max_time_seconds: int
    token_budget: int
    expected_cases: int
    editable_files: tuple[str, ...]
    context_files: tuple[str, ...]
    source: str
    source_revision: str
    license: str
    sensitive_data: SensitiveData
    seed: int
    environment: EnvironmentContract
    verifier: Verifier
    workspace: Path
    task_root: Path
    task_digest: str


@dataclass(frozen=True)
class TaskLoadError:
    """The task directory does not satisfy the current contract."""

    path: Path
    reason: str


@dataclass(frozen=True)
class _Scalars:
    task_id: str
    task_revision: int
    family: str
    split: Split
    prompt: str
    runtime: str
    runtime_version: str
    max_time_seconds: int
    token_budget: int
    expected_cases: int
    source: str
    source_revision: str
    license: str
    sensitive_data: SensitiveData
    seed: int


def workspace_digest(workspace: Path) -> str:
    """Return the verified content digest of one plain workspace."""
    inventory = inspect_plain_tree(workspace)
    if isinstance(inventory, RuntimeFailure):
        raise ValueError(inventory.reason)
    return inventory.digest


def _string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(item, str) and item for item in value):
        return None
    if len(set(value)) != len(value):
        return None
    return tuple(value)


def _bounded_integer(value: object, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if not minimum <= value <= maximum:
        return None
    return value


def _plain_relative_file(root: Path, value: str) -> Path | None:
    relative = Path(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    candidate = root.joinpath(*relative.parts)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve(strict=True)):
            return None
    except (OSError, RuntimeError):
        return None
    return resolved


def _exact_fields(value: object, expected: frozenset[str], section: str) -> str | None:
    if not isinstance(value, Mapping):
        return f"{section} must be a table"
    fields = set(value)
    missing = sorted(expected - fields)
    extra = sorted(fields - expected)
    if missing:
        return f"{section} is missing fields: {', '.join(missing)}"
    if extra:
        return f"{section} has unknown fields: {', '.join(extra)}"
    return None


def _pinned_environment(value: object) -> EnvironmentContract | str:
    field_error = _exact_fields(value, _ENVIRONMENT_FIELDS, "environment")
    if field_error is not None:
        return field_error
    if not isinstance(value, Mapping):
        return "environment must be a table"
    expected_values: dict[str, object] = {
        "image": RUNTIME_IMAGE,
        "image_digest": RUNTIME_IMAGE_DIGEST,
        "os": "linux",
        "architecture": "arm64",
        "network": "none",
        "cpus": CPU_LIMIT,
        "memory_bytes": MEMORY_BYTES,
        "pids": PID_LIMIT,
        "workspace_bytes": WORKSPACE_BYTES,
        "temp_bytes": TEMP_BYTES,
        "home_bytes": HOME_BYTES,
        "dependency_lock_digest": EMPTY_DEPENDENCY_LOCK_DIGEST,
    }
    for name, expected in expected_values.items():
        actual = value.get(name)
        if isinstance(expected, (int, float)) and isinstance(actual, bool):
            return f"environment.{name} does not match the runtime contract"
        if actual != expected:
            return f"environment.{name} does not match the runtime contract"
    return EnvironmentContract(
        image=RUNTIME_IMAGE,
        image_digest=RUNTIME_IMAGE_DIGEST,
        os="linux",
        architecture="arm64",
        network="none",
        cpus=CPU_LIMIT,
        memory_bytes=MEMORY_BYTES,
        pids=PID_LIMIT,
        workspace_bytes=WORKSPACE_BYTES,
        temp_bytes=TEMP_BYTES,
        home_bytes=HOME_BYTES,
        dependency_lock_digest=EMPTY_DEPENDENCY_LOCK_DIGEST,
    )


def _task_environment(value: object) -> EnvironmentContract | str:
    """Validate a per-task image contract for a minted repository task."""
    field_error = _exact_fields(value, _ENVIRONMENT_FIELDS, "environment")
    if field_error is not None:
        return field_error
    if not isinstance(value, Mapping):
        return "environment must be a table"
    image = value.get("image")
    image_digest = value.get("image_digest")
    architecture = value.get("architecture")
    lock_digest = value.get("dependency_lock_digest")
    if not isinstance(image, str) or not image or len(image) > 256:
        return "environment.image must be a bounded image reference"
    if not isinstance(image_digest, str) or not IMAGE_DIGEST_PATTERN.fullmatch(
        image_digest
    ):
        return "environment.image_digest must be a sha256 image digest"
    if value.get("os") != "linux":
        return "environment.os must be linux"
    if architecture not in {"arm64", "amd64"}:
        return "environment.architecture must be arm64 or amd64"
    if value.get("network") != "none":
        return "environment.network must be none"
    cpus = value.get("cpus")
    if isinstance(cpus, bool) or not isinstance(cpus, (int, float)):
        return "environment.cpus must be a number"
    if not 0.5 <= float(cpus) <= MAX_CPUS:
        return f"environment.cpus must be from 0.5 to {MAX_CPUS}"
    limits = {
        "memory_bytes": (256 * 1024 * 1024, MAX_MEMORY_BYTES),
        "pids": (16, MAX_PIDS),
        "workspace_bytes": (1024 * 1024, MAX_REPOSITORY_BYTES),
        "temp_bytes": (1024 * 1024, MAX_SCRATCH_BYTES),
        "home_bytes": (1024 * 1024, MAX_SCRATCH_BYTES),
    }
    sizes: dict[str, int] = {}
    for name, (minimum, maximum) in limits.items():
        parsed = _bounded_integer(value.get(name), minimum, maximum)
        if parsed is None:
            return f"environment.{name} is out of range"
        sizes[name] = parsed
    if not isinstance(lock_digest, str) or not HEX_DIGEST_PATTERN.fullmatch(lock_digest):
        return "environment.dependency_lock_digest must be a sha256 hex digest"
    return EnvironmentContract(
        image=image,
        image_digest=image_digest,
        os="linux",
        architecture=architecture,
        network="none",
        cpus=float(cpus),
        memory_bytes=sizes["memory_bytes"],
        pids=sizes["pids"],
        workspace_bytes=sizes["workspace_bytes"],
        temp_bytes=sizes["temp_bytes"],
        home_bytes=sizes["home_bytes"],
        dependency_lock_digest=lock_digest,
    )


def _scalars(raw: Mapping[str, object]) -> _Scalars | str:
    task_id = raw.get("task_id")
    family = raw.get("family")
    split = raw.get("split")
    prompt = raw.get("prompt")
    runtime = raw.get("runtime")
    runtime_version = raw.get("runtime_version")
    source = raw.get("source")
    source_revision = raw.get("source_revision")
    task_license = raw.get("license")
    sensitive_data = raw.get("sensitive_data")
    if not isinstance(task_id, str) or TASK_ID_PATTERN.fullmatch(task_id) is None:
        return "task_id is invalid"
    if not task_id.rsplit("/", 1)[-1]:
        return "task_id must end with a task name"
    task_revision = _bounded_integer(raw.get("task_revision"), 1, 1_000_000)
    if task_revision is None:
        return "task_revision must be a positive integer"
    if not isinstance(family, str) or TASK_ID_PATTERN.fullmatch(family) is None:
        return "family is invalid"
    if split not in {"train", "validation", "holdout"}:
        return "split is invalid"
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
    ):
        return "prompt must be bounded non-empty text"
    if not isinstance(runtime, str) or not runtime:
        return "runtime must be a non-empty string"
    if not isinstance(runtime_version, str) or not RUNTIME_VERSION_PATTERN.fullmatch(
        runtime_version
    ):
        return "runtime_version must be a bounded version label"
    max_time_seconds = _bounded_integer(raw.get("max_time_seconds"), 1, MAX_TIME_SECONDS)
    if max_time_seconds is None:
        return f"max_time_seconds must be from 1 to {MAX_TIME_SECONDS}"
    token_budget = _bounded_integer(raw.get("token_budget"), 256, MAX_TOKEN_BUDGET)
    if token_budget is None:
        return f"token_budget must be from 256 to {MAX_TOKEN_BUDGET}"
    expected_cases = _bounded_integer(raw.get("expected_cases"), 1, MAX_EXPECTED_CASES)
    if expected_cases is None:
        return f"expected_cases must be from 1 to {MAX_EXPECTED_CASES}"
    if not isinstance(source, str) or not source:
        return "source must be a non-empty string"
    if not isinstance(source_revision, str) or not source_revision:
        return "source_revision must be a non-empty string"
    if not isinstance(task_license, str) or LICENSE_PATTERN.fullmatch(task_license) is None:
        return "license must be one SPDX identifier"
    if sensitive_data not in {"public", "private"}:
        return "sensitive_data must be public or private"
    seed = _bounded_integer(raw.get("seed"), 0, 2**63 - 1)
    if seed is None:
        return "seed must be an integer from 0 to 2**63 - 1"
    return _Scalars(
        task_id=task_id,
        task_revision=task_revision,
        family=family,
        split=split,
        prompt=prompt.strip(),
        runtime=runtime,
        runtime_version=runtime_version,
        max_time_seconds=max_time_seconds,
        token_budget=token_budget,
        expected_cases=expected_cases,
        source=source,
        source_revision=source_revision,
        license=task_license,
        sensitive_data=sensitive_data,
        seed=seed,
    )


def _call_cases_verifier(
    raw: Mapping[str, object], task_root: Path, workspace: Path
) -> CallCasesVerifier | str:
    verifier = raw.get("verifier")
    field_error = _exact_fields(verifier, _CALL_CASES_VERIFIER_FIELDS, "verifier")
    if field_error is not None:
        return field_error
    if not isinstance(verifier, Mapping):
        return "verifier must be a table"
    if verifier.get("protocol") != "call-cases-v1":
        return "verifier.protocol must be call-cases-v1"
    cases_value = verifier.get("cases")
    if not isinstance(cases_value, str):
        return "verifier.cases must be a string"
    cases_path = _plain_relative_file(task_root, cases_value)
    if cases_path is None or cases_path.is_relative_to(workspace):
        return "verifier.cases must be a sealed task file"
    return CallCasesVerifier(protocol="call-cases-v1", cases_path=cases_path)


def _sealed_relative_path(value: str) -> str | None:
    relative = Path(value)
    if (
        "\\" in value
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    return relative.as_posix()


def _test_command_verifier(
    raw: Mapping[str, object], task_root: Path, workspace: Path
) -> TestCommandVerifier | str:
    verifier = raw.get("verifier")
    field_error = _exact_fields(verifier, _TEST_COMMAND_VERIFIER_FIELDS, "verifier")
    if field_error is not None:
        return field_error
    if not isinstance(verifier, Mapping):
        return "verifier must be a table"
    if verifier.get("protocol") != "test-command-v1":
        return "verifier.protocol must be test-command-v1"
    command = _string_list(verifier.get("command"))
    if (
        not command
        or len(command) > MAX_COMMAND_ITEMS
        or any(len(item.encode("utf-8")) > MAX_COMMAND_ITEM_BYTES for item in command)
    ):
        return "verifier.command must be a bounded non-empty argument list"
    timeout_seconds = _bounded_integer(verifier.get("timeout_seconds"), 1, MAX_TIME_SECONDS)
    if timeout_seconds is None:
        return f"verifier.timeout_seconds must be from 1 to {MAX_TIME_SECONDS}"
    sealed_files = _string_list(verifier.get("sealed_files"))
    if not sealed_files or len(sealed_files) > MAX_SEALED_FILES:
        return "verifier.sealed_files must be a bounded non-empty unique list"
    sealed_root = task_root.joinpath(*SEALED_FILES_DIR.split("/"))
    normalized: list[str] = []
    for relative in sealed_files:
        posix = _sealed_relative_path(relative)
        if posix is None or _plain_relative_file(sealed_root, posix) is None:
            return f"verifier.sealed_files entry is not a sealed file: {relative}"
        normalized.append(posix)
    reference_value = verifier.get("reference")
    if not isinstance(reference_value, str):
        return "verifier.reference must be a string"
    reference_patch = _plain_relative_file(task_root, reference_value)
    if reference_patch is None or reference_patch.is_relative_to(workspace):
        return "verifier.reference must be a sealed task file"
    return TestCommandVerifier(
        protocol="test-command-v1",
        command=command,
        timeout_seconds=timeout_seconds,
        sealed_files=tuple(normalized),
        sealed_root=sealed_root,
        reference_patch=reference_patch,
    )


def _declared_files(
    raw: Mapping[str, object], workspace: Path
) -> tuple[tuple[str, ...], tuple[str, ...]] | str:
    editable_files = _string_list(raw.get("editable_files"))
    context_files = _string_list(raw.get("context_files"))
    if not editable_files:
        return "editable_files must be a non-empty unique list"
    if context_files is None:
        return "context_files must be a unique list"
    if not set(editable_files).issubset(context_files):
        return "each editable file must also be a context file"
    for field_name, paths in (
        ("editable_files", editable_files),
        ("context_files", context_files),
    ):
        for relative_path in paths:
            if _plain_relative_file(workspace, relative_path) is None:
                return f"{field_name} path is not a plain workspace file: {relative_path}"
    return editable_files, context_files


def _read_config(task_root: Path) -> Mapping[str, object] | str:
    config_path = task_root / "task.toml"
    if not config_path.is_file():
        return "task.toml not found"
    try:
        config_bytes = config_path.read_bytes()
    except OSError as error:
        return f"task.toml is not readable: {error}"
    if len(config_bytes) > MAX_TASK_CONFIG_BYTES:
        return "task.toml exceeds its size limit"
    try:
        raw = tomllib.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return f"task.toml is invalid: {error}"
    return raw


def _load_call_cases_task(
    task_dir: Path, task_root: Path, raw: Mapping[str, object]
) -> TaskSpec | TaskLoadError:
    workspace = task_root / "workspace"
    field_error = _exact_fields(raw, _CALL_CASES_FIELDS, "task")
    if field_error is not None:
        return TaskLoadError(task_dir, field_error)
    scalars = _scalars(raw)
    if isinstance(scalars, str):
        return TaskLoadError(task_dir, scalars)
    if scalars.runtime not in _CALL_CASES_RUNTIMES:
        return TaskLoadError(task_dir, "runtime must be python or node")
    if scalars.runtime_version != _CALL_CASES_RUNTIMES[scalars.runtime]:
        return TaskLoadError(task_dir, "runtime_version does not match the image")
    if scalars.sensitive_data != "public":
        return TaskLoadError(task_dir, "sensitive_data must be public")
    declared = _declared_files(raw, workspace)
    if isinstance(declared, str):
        return TaskLoadError(task_dir, declared)
    editable_files, context_files = declared
    environment = _pinned_environment(raw.get("environment"))
    if isinstance(environment, str):
        return TaskLoadError(task_dir, environment)
    verifier = _call_cases_verifier(raw, task_root, workspace)
    if isinstance(verifier, str):
        return TaskLoadError(task_dir, verifier)
    inventory = inspect_plain_tree(task_root)
    if isinstance(inventory, RuntimeFailure):
        return TaskLoadError(task_dir, inventory.reason)
    return _task_spec(
        TASK_SCHEMA_VERSION,
        scalars,
        editable_files=editable_files,
        context_files=context_files,
        environment=environment,
        verifier=verifier,
        workspace=workspace,
        task_root=task_root,
        task_digest=inventory.digest,
    )


def _load_test_command_task(
    task_dir: Path, task_root: Path, raw: Mapping[str, object]
) -> TaskSpec | TaskLoadError:
    workspace = task_root / "workspace"
    field_error = _exact_fields(raw, _TEST_COMMAND_FIELDS, "task")
    if field_error is not None:
        return TaskLoadError(task_dir, field_error)
    scalars = _scalars(raw)
    if isinstance(scalars, str):
        return TaskLoadError(task_dir, scalars)
    if scalars.runtime not in TEST_COMMAND_RUNTIMES:
        return TaskLoadError(
            task_dir, f"runtime must be one of {', '.join(sorted(TEST_COMMAND_RUNTIMES))}"
        )
    environment = _task_environment(raw.get("environment"))
    if isinstance(environment, str):
        return TaskLoadError(task_dir, environment)
    verifier = _test_command_verifier(raw, task_root, workspace)
    if isinstance(verifier, str):
        return TaskLoadError(task_dir, verifier)
    if not (task_root / "Dockerfile").is_file():
        return TaskLoadError(task_dir, "Dockerfile not found")
    inventory = inspect_plain_tree(
        task_root,
        max_files=MAX_REPOSITORY_FILES,
        max_bytes=environment.workspace_bytes,
    )
    if isinstance(inventory, RuntimeFailure):
        return TaskLoadError(task_dir, inventory.reason)
    return _task_spec(
        TEST_COMMAND_SCHEMA_VERSION,
        scalars,
        editable_files=(),
        context_files=(),
        environment=environment,
        verifier=verifier,
        workspace=workspace,
        task_root=task_root,
        task_digest=inventory.digest,
    )


def _task_spec(
    schema_version: int,
    scalars: _Scalars,
    *,
    editable_files: tuple[str, ...],
    context_files: tuple[str, ...],
    environment: EnvironmentContract,
    verifier: Verifier,
    workspace: Path,
    task_root: Path,
    task_digest: str,
) -> TaskSpec:
    return TaskSpec(
        schema_version=schema_version,
        task_id=scalars.task_id,
        task_revision=scalars.task_revision,
        name=scalars.task_id.rsplit("/", 1)[-1],
        family=scalars.family,
        split=scalars.split,
        prompt=scalars.prompt,
        runtime=scalars.runtime,
        runtime_version=scalars.runtime_version,
        max_time_seconds=scalars.max_time_seconds,
        token_budget=scalars.token_budget,
        expected_cases=scalars.expected_cases,
        editable_files=editable_files,
        context_files=context_files,
        source=scalars.source,
        source_revision=scalars.source_revision,
        license=scalars.license,
        sensitive_data=scalars.sensitive_data,
        seed=scalars.seed,
        environment=environment,
        verifier=verifier,
        workspace=workspace,
        task_root=task_root,
        task_digest=task_digest,
    )


def load_task(task_dir: Path) -> TaskSpec | TaskLoadError:
    """Load one strict task without running task code."""
    task_root = task_dir.resolve()
    workspace = task_root / "workspace"
    if not workspace.is_dir():
        return TaskLoadError(task_dir, "workspace/ not found")
    raw = _read_config(task_root)
    if isinstance(raw, str):
        return TaskLoadError(task_dir, raw)
    schema_version = raw.get("schema_version")
    if schema_version == TASK_SCHEMA_VERSION:
        return _load_call_cases_task(task_dir, task_root, raw)
    if schema_version == TEST_COMMAND_SCHEMA_VERSION:
        return _load_test_command_task(task_dir, task_root, raw)
    return TaskLoadError(
        task_dir,
        f"schema_version must be {TASK_SCHEMA_VERSION} or {TEST_COMMAND_SCHEMA_VERSION}",
    )


def load_task_suite(tasks_dir: Path) -> tuple[TaskSpec, ...] | TaskLoadError:
    """Load each direct task child and reject ambiguous task identities."""
    if not tasks_dir.is_dir():
        return TaskLoadError(tasks_dir, "task directory does not exist")
    candidates = sorted(
        (
            child
            for child in tasks_dir.iterdir()
            if child.is_dir() and (child / "task.toml").is_file()
        ),
        key=lambda path: path.name,
    )
    tasks: list[TaskSpec] = []
    task_ids: set[str] = set()
    task_names: set[str] = set()
    for task_dir in candidates:
        loaded = load_task(task_dir)
        if isinstance(loaded, TaskLoadError):
            return loaded
        if task_dir.name != loaded.name:
            return TaskLoadError(
                task_dir,
                "task_id must end with the task directory name",
            )
        if loaded.task_id in task_ids:
            return TaskLoadError(task_dir, f"duplicate task_id: {loaded.task_id}")
        if loaded.name in task_names:
            return TaskLoadError(task_dir, f"duplicate task name: {loaded.name}")
        task_ids.add(loaded.task_id)
        task_names.add(loaded.name)
        tasks.append(loaded)
    return tuple(tasks)
