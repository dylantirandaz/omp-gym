"""Create deterministic training tasks outside the sealed package tree."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .runtime import (
    CPU_LIMIT,
    HOME_BYTES,
    MEMORY_BYTES,
    PID_LIMIT,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    TEMP_BYTES,
    WORKSPACE_BYTES,
)
from .task import EMPTY_DEPENDENCY_LOCK_DIGEST, TaskLoadError, TaskSpec, load_task


@dataclass(frozen=True)
class GenerationFailure:
    reason: str


@dataclass(frozen=True)
class _TaskDefinition:
    family: str
    prompt: str
    source: str
    public_test: str
    cases: tuple[Mapping[str, object], ...]


_GENERATED_ROOT = Path(tempfile.mkdtemp(prefix="omp-coding-generated-"))


def _number(seed: int, label: str, minimum: int, maximum: int) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    width = maximum - minimum + 1
    return minimum + int.from_bytes(digest[:8], "big") % width


def _values(seed: int, label: str, count: int) -> list[int]:
    return [_number(seed, f"{label}-{index}", -20, 30) for index in range(count)]


def _encode(value: object) -> Mapping[str, object]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "value": repr(value)}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item) for item in value]}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                [_encode(key), _encode(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        }
    raise TypeError(f"unsupported generated verifier value: {type(value).__name__}")


def _case(
    identifier: str,
    function_name: str,
    arguments: Sequence[object],
    expected: object,
) -> Mapping[str, object]:
    return {
        "id": identifier,
        "operations": [
            {
                "op": "call",
                "target": f"solution:{function_name}",
                "args": [_encode(argument) for argument in arguments],
                "store": "result",
            }
        ],
        "observe": ["result"],
        "expected": {
            "status": "ok",
            "values": {"result": _encode(expected)},
        },
    }


def _bounded_total(seed: int) -> _TaskDefinition:
    lower = _number(seed, "lower", -8, 2)
    upper = lower + _number(seed, "width", 5, 15)
    arguments = [
        _values(seed, f"bounded-{index}", _number(seed, f"size-{index}", 0, 8))
        for index in range(6)
    ]
    cases = tuple(
        _case(
            f"bounded-{index}",
            "bounded_total",
            [values, lower, upper],
            sum(min(max(value, lower), upper) for value in values),
        )
        for index, values in enumerate(arguments)
    )
    public_values = arguments[0]
    public_expected = sum(min(max(value, lower), upper) for value in public_values)
    return _TaskDefinition(
        family="generated-bounded-total",
        prompt=(
            "Implement `bounded_total` in `solution.py`. Clamp each input value to "
            "the inclusive lower and upper limits, and return the sum. Do not mutate "
            "the input list."
        ),
        source=(
            "def bounded_total(values, lower, upper):\n"
            "    return min(max(sum(values), lower), upper)\n"
        ),
        public_test=(
            "from solution import bounded_total\n\n"
            f"assert bounded_total({public_values!r}, {lower}, {upper}) == "
            f"{public_expected!r}\n"
            f"assert bounded_total([], {lower}, {upper}) == 0\n"
        ),
        cases=cases,
    )


def _window_sums(seed: int) -> _TaskDefinition:
    inputs: list[tuple[list[int], int]] = []
    for index in range(6):
        values = _values(
            seed, f"window-{index}", _number(seed, f"length-{index}", 1, 9)
        )
        size = _number(seed, f"window-size-{index}", 1, len(values) + 2)
        inputs.append((values, size))

    def expected(values: list[int], size: int) -> list[int]:
        if size > len(values):
            return []
        return [
            sum(values[index : index + size]) for index in range(len(values) - size + 1)
        ]

    cases = tuple(
        _case(f"window-{index}", "window_sums", [values, size], expected(values, size))
        for index, (values, size) in enumerate(inputs)
    )
    values, size = inputs[0]
    return _TaskDefinition(
        family="generated-window-sums",
        prompt=(
            "Implement `window_sums` in `solution.py`. Return the sum of each "
            "overlapping window. Return an empty list when the window is larger than "
            "the input. Raise `ValueError` when the size is not positive."
        ),
        source=(
            "def window_sums(values, size):\n"
            "    if size <= 0:\n"
            "        raise ValueError('size must be positive')\n"
            "    return [sum(values[index:index + size]) for index in range(0, len(values), size)]\n"
        ),
        public_test=(
            "from solution import window_sums\n\n"
            f"assert window_sums({values!r}, {size}) == {expected(values, size)!r}\n"
            "assert window_sums([1, 2], 3) == []\n"
        ),
        cases=cases,
    )


def _rotate(seed: int) -> _TaskDefinition:
    inputs: list[tuple[list[int], int]] = []
    for index in range(6):
        values = _values(
            seed, f"rotate-{index}", _number(seed, f"rotate-size-{index}", 0, 8)
        )
        distance = _number(seed, f"distance-{index}", -20, 20)
        inputs.append((values, distance))

    def expected(values: list[int], distance: int) -> list[int]:
        if not values:
            return []
        offset = distance % len(values)
        return values[-offset:] + values[:-offset] if offset else list(values)

    cases = tuple(
        _case(
            f"rotate-{index}", "rotate", [values, distance], expected(values, distance)
        )
        for index, (values, distance) in enumerate(inputs)
    )
    values, distance = inputs[0]
    return _TaskDefinition(
        family="generated-rotate",
        prompt=(
            "Implement `rotate` in `solution.py`. Rotate a list to the right by the "
            "specified distance. Support negative and large distances. Return a new "
            "list and support an empty input."
        ),
        source=(
            "def rotate(values, distance):\n"
            "    return values[-distance:] + values[:-distance]\n"
        ),
        public_test=(
            "from solution import rotate\n\n"
            f"assert rotate({values!r}, {distance}) == {expected(values, distance)!r}\n"
            "assert rotate([], 4) == []\n"
        ),
        cases=cases,
    )


def _parse_fields(seed: int) -> _TaskDefinition:
    delimiter = ["=", ":", "|"][_number(seed, "delimiter", 0, 2)]
    texts = (
        f" alpha {delimiter} one \n beta {delimiter} two ",
        f"name{delimiter}Ada{delimiter}Lovelace\nlang{delimiter}Python",
        f"\nleft{delimiter}1\n\nright{delimiter}2\n",
        f"key{delimiter} value with spaces ",
        "",
        f"x{delimiter}1\ny{delimiter}2\nz{delimiter}3",
    )

    def expected(text: str) -> dict[str, str]:
        output: dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            key, value = line.split(delimiter, 1)
            output[key.strip()] = value.strip()
        return output

    cases = tuple(
        _case(f"fields-{index}", "parse_fields", [text, delimiter], expected(text))
        for index, text in enumerate(texts)
    )
    return _TaskDefinition(
        family="generated-parse-fields",
        prompt=(
            "Implement `parse_fields` in `solution.py`. Ignore blank lines. Split each "
            "other line at the first delimiter only. Strip the key and value, and "
            "return a dictionary."
        ),
        source=(
            "def parse_fields(text, delimiter):\n"
            "    return dict(line.split(delimiter) for line in text.splitlines())\n"
        ),
        public_test=(
            "from solution import parse_fields\n\n"
            f"assert parse_fields({texts[0]!r}, {delimiter!r}) == {expected(texts[0])!r}\n"
            f"assert parse_fields('', {delimiter!r}) == {{}}\n"
        ),
        cases=cases,
    )


def _merge_counts(seed: int) -> _TaskDefinition:
    inputs: list[tuple[dict[str, int], dict[str, int]]] = []
    keys = ("alpha", "beta", "gamma", "delta")
    for index in range(6):
        left = {
            key: _number(seed, f"left-{index}-{key}", 0, 9)
            for key in keys[: _number(seed, f"left-size-{index}", 1, 4)]
        }
        right = {
            key: _number(seed, f"right-{index}-{key}", 0, 9)
            for key in keys[4 - _number(seed, f"right-size-{index}", 1, 4) :]
        }
        inputs.append((left, right))

    def expected(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        output = dict(left)
        for key, value in right.items():
            output[key] = output.get(key, 0) + value
        return output

    cases = tuple(
        _case(f"merge-{index}", "merge_counts", [left, right], expected(left, right))
        for index, (left, right) in enumerate(inputs)
    )
    left, right = inputs[0]
    return _TaskDefinition(
        family="generated-merge-counts",
        prompt=(
            "Implement `merge_counts` in `solution.py`. Return a new dictionary. Sum "
            "the values for keys that occur in both inputs. Do not mutate either input."
        ),
        source="def merge_counts(left, right):\n    return {**left, **right}\n",
        public_test=(
            "from solution import merge_counts\n\n"
            f"assert merge_counts({left!r}, {right!r}) == {expected(left, right)!r}\n"
            "assert merge_counts({}, {}) == {}\n"
        ),
        cases=cases,
    )


def _run_lengths(seed: int) -> _TaskDefinition:
    inputs: list[list[int]] = []
    for index in range(6):
        raw = _values(seed, f"runs-{index}", _number(seed, f"runs-size-{index}", 0, 10))
        inputs.append([value % 4 for value in raw])

    def expected(values: list[int]) -> list[tuple[int, int]]:
        output: list[tuple[int, int]] = []
        for value in values:
            if output and output[-1][0] == value:
                previous, count = output[-1]
                output[-1] = (previous, count + 1)
            else:
                output.append((value, 1))
        return output

    cases = tuple(
        _case(f"runs-{index}", "run_lengths", [values], expected(values))
        for index, values in enumerate(inputs)
    )
    values = inputs[0]
    return _TaskDefinition(
        family="generated-run-lengths",
        prompt=(
            "Implement `run_lengths` in `solution.py`. Return one `(value, count)` "
            "tuple for each adjacent run. Preserve run order. Return an empty list for "
            "an empty input."
        ),
        source=(
            "from collections import Counter\n\n"
            "def run_lengths(values):\n"
            "    return list(Counter(values).items())\n"
        ),
        public_test=(
            "from solution import run_lengths\n\n"
            f"assert run_lengths({values!r}) == {expected(values)!r}\n"
            "assert run_lengths([]) == []\n"
        ),
        cases=cases,
    )


_BUILDERS: tuple[Callable[[int], _TaskDefinition], ...] = (
    _bounded_total,
    _window_sums,
    _rotate,
    _parse_fields,
    _merge_counts,
    _run_lengths,
)


def _task_toml(
    *,
    task_id: str,
    definition: _TaskDefinition,
    seed: int,
    expected_cases: int,
) -> str:
    quote = json.dumps
    return "\n".join(
        (
            "schema_version = 2",
            f"task_id = {quote(task_id)}",
            "task_revision = 1",
            f"family = {quote(definition.family)}",
            'split = "train"',
            f"prompt = {quote(definition.prompt)}",
            'runtime = "python"',
            'runtime_version = "3.11.2"',
            "max_time_seconds = 300",
            "token_budget = 65536",
            f"expected_cases = {expected_cases}",
            'editable_files = ["solution.py"]',
            'context_files = ["solution.py", "test_solution.py"]',
            'source = "https://github.com/dylantirandaz/omp-gym"',
            f"source_revision = {quote(f'generated-v1-{seed}')} ",
            'license = "MIT"',
            'sensitive_data = "public"',
            f"seed = {seed}",
            "",
            "[environment]",
            f"image = {quote(RUNTIME_IMAGE)}",
            f"image_digest = {quote(RUNTIME_IMAGE_DIGEST)}",
            'os = "linux"',
            'architecture = "arm64"',
            'network = "none"',
            f"cpus = {CPU_LIMIT}",
            f"memory_bytes = {MEMORY_BYTES}",
            f"pids = {PID_LIMIT}",
            f"workspace_bytes = {WORKSPACE_BYTES}",
            f"temp_bytes = {TEMP_BYTES}",
            f"home_bytes = {HOME_BYTES}",
            f"dependency_lock_digest = {quote(EMPTY_DEPENDENCY_LOCK_DIGEST)}",
            "",
            "[verifier]",
            'protocol = "call-cases-v1"',
            'cases = "verifier/cases.json"',
            "",
        )
    )


def generate_training_tasks(
    *, count: int, generation_seed: int
) -> tuple[TaskSpec, ...] | GenerationFailure:
    """Create a deterministic set of generated train tasks."""
    if count < 0 or count > 1000:
        return GenerationFailure("generated task count must be from 0 through 1000")
    if generation_seed < 0:
        return GenerationFailure("generation seed must not be negative")
    tasks: list[TaskSpec] = []
    for index in range(count):
        task_seed = generation_seed * 10_000 + index
        definition = _BUILDERS[index % len(_BUILDERS)](task_seed)
        task_id = f"omp-gym/generated/{generation_seed}/{index:04d}-{definition.family}"
        task_root = _GENERATED_ROOT / str(generation_seed) / f"{index:04d}"
        workspace = task_root / "workspace"
        verifier = task_root / "verifier"
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            verifier.mkdir(parents=True, exist_ok=True)
            (workspace / "solution.py").write_text(definition.source, encoding="utf-8")
            (workspace / "test_solution.py").write_text(
                definition.public_test, encoding="utf-8"
            )
            (verifier / "cases.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "runtime": "python",
                        "cases": definition.cases,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (task_root / "task.toml").write_text(
                _task_toml(
                    task_id=task_id,
                    definition=definition,
                    seed=task_seed,
                    expected_cases=len(definition.cases),
                ),
                encoding="utf-8",
            )
        except OSError as error:
            return GenerationFailure(f"generated task write failed: {error}")
        loaded = load_task(task_root)
        if isinstance(loaded, TaskLoadError):
            return GenerationFailure(
                f"generated task is invalid: {task_id}: {loaded.reason}"
            )
        tasks.append(loaded)
    return tuple(tasks)
