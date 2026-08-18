"""Versioned task contract with sealed verifier files."""

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
MAX_TASK_CONFIG_BYTES = 256 * 1024
MAX_PROMPT_BYTES = 32 * 1024
MAX_TIME_SECONDS = 30 * 60
MAX_EXPECTED_CASES = 1000
MAX_TOKEN_BUDGET = 1_000_000
EMPTY_DEPENDENCY_LOCK_DIGEST = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
TASK_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._/-]{2,127}\Z")
LICENSE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}\Z")
_TOP_LEVEL_FIELDS = frozenset(
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
        "editable_files",
        "context_files",
        "source",
        "source_revision",
        "license",
        "sensitive_data",
        "seed",
        "environment",
        "verifier",
    }
)
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
_VERIFIER_FIELDS = frozenset({"protocol", "cases"})
_RUNTIME_VERSIONS = {"python": "3.11.2", "node": "24.19.0"}


@dataclass(frozen=True)
class EnvironmentContract:
    """Exact runtime and resource identity for one task."""

    image: str
    image_digest: str
    os: Literal["linux"]
    architecture: Literal["arm64"]
    network: Literal["none"]
    cpus: float
    memory_bytes: int
    pids: int
    workspace_bytes: int
    temp_bytes: int
    home_bytes: int
    dependency_lock_digest: str


@dataclass(frozen=True)
class TaskSpec:
    """One complete and reproducible coding task."""

    schema_version: int
    task_id: str
    task_revision: int
    name: str
    family: str
    split: Literal["train", "validation", "holdout"]
    prompt: str
    runtime: Literal["python", "node"]
    runtime_version: str
    max_time_seconds: int
    token_budget: int
    expected_cases: int
    editable_files: tuple[str, ...]
    context_files: tuple[str, ...]
    source: str
    source_revision: str
    license: str
    sensitive_data: Literal["public"]
    seed: int
    environment: EnvironmentContract
    verifier_protocol: Literal["call-cases-v1"]
    verifier_path: Path
    workspace: Path
    task_root: Path
    task_digest: str


@dataclass(frozen=True)
class TaskLoadError:
    """The task directory does not satisfy the current contract."""

    path: Path
    reason: str


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


def _environment(value: object) -> EnvironmentContract | str:
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


def load_task(task_dir: Path) -> TaskSpec | TaskLoadError:
    """Load one strict schema-v2 task without running task code."""
    task_root = task_dir.resolve()
    config_path = task_root / "task.toml"
    workspace = task_root / "workspace"
    if not config_path.is_file():
        return TaskLoadError(task_dir, "task.toml not found")
    if not workspace.is_dir():
        return TaskLoadError(task_dir, "workspace/ not found")
    inventory = inspect_plain_tree(task_root)
    if isinstance(inventory, RuntimeFailure):
        return TaskLoadError(task_dir, inventory.reason)
    try:
        config_bytes = config_path.read_bytes()
    except OSError as error:
        return TaskLoadError(task_dir, f"task.toml is not readable: {error}")
    if len(config_bytes) > MAX_TASK_CONFIG_BYTES:
        return TaskLoadError(task_dir, "task.toml exceeds its size limit")
    try:
        raw = tomllib.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return TaskLoadError(task_dir, f"task.toml is invalid: {error}")
    field_error = _exact_fields(raw, _TOP_LEVEL_FIELDS, "task")
    if field_error is not None:
        return TaskLoadError(task_dir, field_error)
    schema_version = raw.get("schema_version")
    task_id = raw.get("task_id")
    task_revision = raw.get("task_revision")
    family = raw.get("family")
    split = raw.get("split")
    prompt = raw.get("prompt")
    runtime = raw.get("runtime")
    runtime_version = raw.get("runtime_version")
    max_time_seconds = raw.get("max_time_seconds")
    token_budget = raw.get("token_budget")
    expected_cases = raw.get("expected_cases")
    source = raw.get("source")
    source_revision = raw.get("source_revision")
    task_license = raw.get("license")
    sensitive_data = raw.get("sensitive_data")
    seed = raw.get("seed")
    if schema_version != TASK_SCHEMA_VERSION:
        return TaskLoadError(task_dir, f"schema_version must be {TASK_SCHEMA_VERSION}")
    if not isinstance(task_id, str) or TASK_ID_PATTERN.fullmatch(task_id) is None:
        return TaskLoadError(task_dir, "task_id is invalid")
    task_name = task_id.rsplit("/", 1)[-1]
    if not task_name:
        return TaskLoadError(task_dir, "task_id must end with a task name")
    if (
        isinstance(task_revision, bool)
        or not isinstance(task_revision, int)
        or not 1 <= task_revision <= 1_000_000
    ):
        return TaskLoadError(task_dir, "task_revision must be a positive integer")
    if not isinstance(family, str) or TASK_ID_PATTERN.fullmatch(family) is None:
        return TaskLoadError(task_dir, "family is invalid")
    if split not in {"train", "validation", "holdout"}:
        return TaskLoadError(task_dir, "split is invalid")
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES
    ):
        return TaskLoadError(task_dir, "prompt must be bounded non-empty text")
    if runtime not in _RUNTIME_VERSIONS:
        return TaskLoadError(task_dir, "runtime must be python or node")
    if runtime_version != _RUNTIME_VERSIONS[runtime]:
        return TaskLoadError(task_dir, "runtime_version does not match the image")
    if (
        isinstance(max_time_seconds, bool)
        or not isinstance(max_time_seconds, int)
        or not 1 <= max_time_seconds <= MAX_TIME_SECONDS
    ):
        return TaskLoadError(
            task_dir, f"max_time_seconds must be from 1 to {MAX_TIME_SECONDS}"
        )
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, int)
        or not 256 <= token_budget <= MAX_TOKEN_BUDGET
    ):
        return TaskLoadError(
            task_dir, f"token_budget must be from 256 to {MAX_TOKEN_BUDGET}"
        )
    if (
        isinstance(expected_cases, bool)
        or not isinstance(expected_cases, int)
        or not 1 <= expected_cases <= MAX_EXPECTED_CASES
    ):
        return TaskLoadError(
            task_dir, f"expected_cases must be from 1 to {MAX_EXPECTED_CASES}"
        )
    if not isinstance(source, str) or not source:
        return TaskLoadError(task_dir, "source must be a non-empty string")
    if not isinstance(source_revision, str) or not source_revision:
        return TaskLoadError(task_dir, "source_revision must be a non-empty string")
    if (
        not isinstance(task_license, str)
        or LICENSE_PATTERN.fullmatch(task_license) is None
    ):
        return TaskLoadError(task_dir, "license must be one SPDX identifier")
    if sensitive_data != "public":
        return TaskLoadError(task_dir, "sensitive_data must be public")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        return TaskLoadError(task_dir, "seed must be an integer from 0 to 2**63 - 1")
    editable_files = _string_list(raw.get("editable_files"))
    context_files = _string_list(raw.get("context_files"))
    if not editable_files:
        return TaskLoadError(task_dir, "editable_files must be a non-empty unique list")
    if context_files is None:
        return TaskLoadError(task_dir, "context_files must be a unique list")
    if not set(editable_files).issubset(context_files):
        return TaskLoadError(task_dir, "each editable file must also be a context file")
    for field_name, paths in (
        ("editable_files", editable_files),
        ("context_files", context_files),
    ):
        for relative_path in paths:
            if _plain_relative_file(workspace, relative_path) is None:
                return TaskLoadError(
                    task_dir,
                    f"{field_name} path is not a plain workspace file: {relative_path}",
                )
    environment = _environment(raw.get("environment"))
    if isinstance(environment, str):
        return TaskLoadError(task_dir, environment)
    verifier = raw.get("verifier")
    verifier_error = _exact_fields(verifier, _VERIFIER_FIELDS, "verifier")
    if verifier_error is not None:
        return TaskLoadError(task_dir, verifier_error)
    if not isinstance(verifier, Mapping):
        return TaskLoadError(task_dir, "verifier must be a table")
    if verifier.get("protocol") != "call-cases-v1":
        return TaskLoadError(task_dir, "verifier.protocol must be call-cases-v1")
    cases_value = verifier.get("cases")
    if not isinstance(cases_value, str):
        return TaskLoadError(task_dir, "verifier.cases must be a string")
    verifier_path = _plain_relative_file(task_root, cases_value)
    if verifier_path is None or verifier_path.is_relative_to(workspace):
        return TaskLoadError(task_dir, "verifier.cases must be a sealed task file")
    return TaskSpec(
        schema_version=TASK_SCHEMA_VERSION,
        task_id=task_id,
        task_revision=task_revision,
        name=task_name,
        family=family,
        split=split,
        prompt=prompt.strip(),
        runtime=runtime,
        runtime_version=runtime_version,
        max_time_seconds=max_time_seconds,
        token_budget=token_budget,
        expected_cases=expected_cases,
        editable_files=editable_files,
        context_files=context_files,
        source=source,
        source_revision=source_revision,
        license=task_license,
        sensitive_data="public",
        seed=seed,
        environment=environment,
        verifier_protocol="call-cases-v1",
        verifier_path=verifier_path,
        workspace=workspace,
        task_root=task_root,
        task_digest=inventory.digest,
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
