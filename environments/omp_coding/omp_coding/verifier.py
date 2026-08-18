"""Structured black-box verification outside the solver container."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, Overflow, localcontext
from pathlib import Path
from typing import Literal

from .runtime import RuntimeFailure


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


MAX_CASE_FILE_BYTES = 8 * 1024 * 1024
MAX_CASES = 1000
MAX_OPERATIONS = 200
MAX_CASE_VALUE_DEPTH = 32
MAX_CASE_VALUE_NODES = 10_000
MAX_SUITE_NODES = 100_000
MAX_NUMERIC_TEXT = 4096
CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class VerifierSuite:
    """One validated set of sealed black-box cases."""

    runtime: Literal["python", "node"]
    cases: tuple[Mapping[str, object], ...]
    digest: str


@dataclass(frozen=True)
class VerifierLoadFailure:
    """A verifier case file is invalid."""

    reason: str


@dataclass(frozen=True)
class CaseFailure:
    """One candidate behavior did not match its sealed contract."""

    case_id: str
    reason: str
    actual: object | None = None


@dataclass(frozen=True)
class VerifierResult:
    """Trusted result from one sealed verifier run."""

    status: Literal["passed", "failed"]
    passed_cases: int
    total_cases: int
    failures: tuple[CaseFailure, ...]
    duration_seconds: float
    suite_digest: str

    @property
    def reward(self) -> float:
        return self.passed_cases / self.total_cases


def _json_shape_error(
    value: object,
    *,
    max_depth: int,
    max_nodes: int,
) -> str | None:
    """Validate one bounded JSON value without recursive Python calls."""
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > max_nodes:
            return f"JSON value has more than {max_nodes} nodes"
        if depth > max_depth:
            return f"JSON value is deeper than {max_depth} levels"
        if isinstance(current, dict):
            if not all(isinstance(key, str) for key in current):
                return "JSON object key is not a string"
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
        elif isinstance(current, float) and not math.isfinite(current):
            return "JSON number is not finite"
        elif current is not None and not isinstance(current, (bool, int, float, str)):
            return "value is not JSON"
    return None


def _valid_expected(value: object, observe: tuple[str, ...]) -> bool:
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    if status == "ok":
        values = value.get("values")
        return isinstance(values, dict) and set(values) == set(observe)
    if status == "error":
        operation = value.get("operation")
        error_type = value.get("type")
        message = value.get("message")
        return (
            isinstance(operation, int)
            and not isinstance(operation, bool)
            and operation >= 0
            and isinstance(error_type, str)
            and bool(error_type)
            and (message is None or isinstance(message, str))
        )
    return False


def load_verifier_suite(
    path: Path,
    *,
    runtime: Literal["python", "node"],
    expected_cases: int,
) -> VerifierSuite | VerifierLoadFailure:
    """Load and validate one sealed case file."""
    try:
        data = path.read_bytes()
    except OSError as error:
        return VerifierLoadFailure(f"verifier case file is not readable: {error}")
    if len(data) > MAX_CASE_FILE_BYTES:
        return VerifierLoadFailure("verifier case file exceeds its size limit")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        return VerifierLoadFailure(f"verifier case file is invalid JSON: {error}")
    shape_error = _json_shape_error(
        value,
        max_depth=MAX_CASE_VALUE_DEPTH + 4,
        max_nodes=MAX_SUITE_NODES,
    )
    if shape_error is not None:
        return VerifierLoadFailure(f"verifier case file is invalid: {shape_error}")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return VerifierLoadFailure("verifier schema version must be 1")
    if value.get("runtime") != runtime:
        return VerifierLoadFailure("verifier runtime does not match task runtime")
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= MAX_CASES:
        return VerifierLoadFailure(f"verifier must contain from 1 to {MAX_CASES} cases")
    if len(raw_cases) != expected_cases:
        return VerifierLoadFailure(
            "verifier case count does not match the task contract"
        )
    seen_ids: set[str] = set()
    cases: list[Mapping[str, object]] = []
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            return VerifierLoadFailure(f"verifier case {index} must be an object")
        case_id = case.get("id")
        operations = case.get("operations")
        observe = case.get("observe")
        expected = case.get("expected")
        if not isinstance(case_id, str) or CASE_ID_PATTERN.fullmatch(case_id) is None:
            return VerifierLoadFailure(f"verifier case {index} has an invalid id")
        if case_id in seen_ids:
            return VerifierLoadFailure(f"verifier case id is repeated: {case_id}")
        seen_ids.add(case_id)
        if (
            not isinstance(operations, list)
            or not 1 <= len(operations) <= MAX_OPERATIONS
        ):
            return VerifierLoadFailure(
                f"verifier case {case_id} has invalid operations"
            )
        if (
            not isinstance(observe, list)
            or not all(isinstance(name, str) and name for name in observe)
            or len(set(observe)) != len(observe)
        ):
            return VerifierLoadFailure(
                f"verifier case {case_id} has invalid observations"
            )
        for field_name, field_value in (
            ("operations", operations),
            ("expected", expected),
        ):
            shape_error = _json_shape_error(
                field_value,
                max_depth=MAX_CASE_VALUE_DEPTH,
                max_nodes=MAX_CASE_VALUE_NODES,
            )
            if shape_error is not None:
                return VerifierLoadFailure(
                    f"verifier case {case_id} has invalid {field_name}: {shape_error}"
                )
        observe_names = tuple(observe)
        if not _valid_expected(expected, observe_names):
            return VerifierLoadFailure(
                f"verifier case {case_id} has an invalid expectation"
            )
        cases.append(case)
    try:
        digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    except (RecursionError, TypeError, ValueError) as error:
        return VerifierLoadFailure(f"verifier case file is not canonical JSON: {error}")
    return VerifierSuite(runtime, tuple(cases), digest)


def _decimal_number(value: object) -> Decimal | None:
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    text = value.get("value")
    if (
        kind not in {"int", "float", "number", "bigint"}
        or not isinstance(text, str)
        or not 1 <= len(text) <= MAX_NUMERIC_TEXT
        or text.strip() != text
    ):
        return None
    if kind in {"int", "bigint"} and INTEGER_PATTERN.fullmatch(text) is None:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite() or abs(number.adjusted()) > 10_000:
        return None
    return number


def _close_number(actual: object, expected: object, tolerance: Decimal) -> bool:
    actual_number = _decimal_number(actual)
    expected_number = _decimal_number(expected)
    if actual_number is None or expected_number is None:
        return False
    try:
        with localcontext() as context:
            context.prec = MAX_NUMERIC_TEXT * 2 + 16
            return abs(actual_number - expected_number) <= tolerance
    except (InvalidOperation, Overflow):
        return False


def _close_recursive(actual: object, expected: object, tolerance: Decimal) -> bool:
    if _close_number(actual, expected, tolerance):
        return True
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(
            _close_recursive(actual[key], expected[key], tolerance) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _close_recursive(left, right, tolerance)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _matches_value(actual: object, expectation: object) -> bool:
    if not isinstance(expectation, dict) or "compare" not in expectation:
        return actual == expectation
    compare = expectation.get("compare")
    expected = expectation.get("value")
    tolerance = expectation.get("absolute")
    if (
        compare not in {"close", "close_recursive"}
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 <= float(tolerance) <= 1.0
    ):
        return False
    decimal_tolerance = Decimal(str(tolerance))
    if compare == "close":
        return _close_number(actual, expected, decimal_tolerance)
    return _close_recursive(actual, expected, decimal_tolerance)


def _case_failure(
    case: Mapping[str, object], reply: Mapping[str, object]
) -> CaseFailure | None:
    case_id = str(case["id"])
    expected = case["expected"]
    if not isinstance(expected, Mapping):
        return CaseFailure(case_id, "invalid expected result")
    expected_status = expected.get("status")
    actual_status = reply.get("status")
    if expected_status != actual_status:
        return CaseFailure(case_id, f"expected {expected_status}, got {actual_status}")
    if expected_status == "error":
        error = reply.get("error")
        if not isinstance(error, Mapping):
            return CaseFailure(case_id, "candidate error payload is invalid")
        actual_operation = reply.get("operation")
        actual_type = error.get("type")
        actual_message = error.get("message")
        expected_message = expected.get("message")
        if actual_operation != expected.get("operation"):
            return CaseFailure(case_id, "candidate failed at the wrong operation")
        if actual_type != expected.get("type"):
            return CaseFailure(case_id, f"candidate raised {actual_type}")
        if expected_message is not None and actual_message != expected_message:
            return CaseFailure(case_id, "candidate raised the wrong message")
        return None
    actual_values = reply.get("values")
    expected_values = expected.get("values")
    if not isinstance(actual_values, Mapping) or not isinstance(
        expected_values, Mapping
    ):
        return CaseFailure(case_id, "candidate value payload is invalid")
    if set(actual_values) != set(expected_values):
        return CaseFailure(case_id, "candidate observations are incomplete")
    for name, expected_value in expected_values.items():
        actual_value = actual_values[name]
        if not _matches_value(actual_value, expected_value):
            return CaseFailure(
                case_id,
                f"observation does not match: {name}",
                {"name": name, "value": actual_value},
            )
    return None


CandidateInvoker = Callable[
    [Literal["python", "node"], Mapping[str, object], float],
    Awaitable[Mapping[str, object] | RuntimeFailure],
]


async def run_verifier_cases(
    suite: VerifierSuite,
    invoke_candidate: CandidateInvoker,
    *,
    timeout_seconds: float,
) -> VerifierResult:
    """Run sealed cases through one asynchronous candidate boundary."""
    started = time.monotonic()
    passed = 0
    failures: list[CaseFailure] = []
    deadline = started + timeout_seconds
    for index, case in enumerate(suite.cases):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failures.extend(
                CaseFailure(str(item["id"]), "verifier deadline exceeded")
                for item in suite.cases[index:]
            )
            break
        request = {
            "schema_version": 1,
            "operations": case["operations"],
            "observe": case["observe"],
        }
        reply = await invoke_candidate(
            suite.runtime,
            request,
            min(remaining, 20.0),
        )
        if isinstance(reply, RuntimeFailure):
            failures.append(CaseFailure(str(case["id"]), reply.reason))
            failures.extend(
                CaseFailure(str(item["id"]), "not run after candidate failure")
                for item in suite.cases[index + 1 :]
            )
            break
        reply_shape_error = _json_shape_error(
            reply,
            max_depth=MAX_CASE_VALUE_DEPTH,
            max_nodes=MAX_CASE_VALUE_NODES,
        )
        if reply_shape_error is not None:
            failures.append(
                CaseFailure(str(case["id"]), "candidate reply exceeds its limits")
            )
            failures.extend(
                CaseFailure(str(item["id"]), "not run after candidate failure")
                for item in suite.cases[index + 1 :]
            )
            break
        failure = _case_failure(case, reply)
        if failure is None:
            passed += 1
        else:
            failures.append(failure)
    status: Literal["passed", "failed"] = (
        "passed" if passed == len(suite.cases) else "failed"
    )
    return VerifierResult(
        status=status,
        passed_cases=passed,
        total_cases=len(suite.cases),
        failures=tuple(failures),
        duration_seconds=time.monotonic() - started,
        suite_digest=suite.digest,
    )
