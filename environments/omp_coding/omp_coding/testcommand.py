"""Sealed test-command verification shared by the minter gate and the grader.

A ``test-command-v1`` task is graded by running one sealed command inside a
fresh container built from the task image. The grader first applies the
agent's workspace patch, then overwrites the sealed test files with their
trusted copies, then runs the command. This module owns the shell script that
does that, the parsers for the common test runners, and the reward rule. It
does not touch Docker or the Prime runtime, so it runs everywhere.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

WORKSPACE = "/workspace"
GRADING_DIR = "/run/omp-gym"
PATCH_PATH = f"{GRADING_DIR}/changes.patch"
SEALED_DIR = f"{GRADING_DIR}/sealed"
SCRIPT_PATH = f"{GRADING_DIR}/grade.sh"
START_TAG = "omp-gym-start"
EXCLUDES_FILE = "/opt/omp-gym/gitignore"
GIT_FLAGS = f"-c safe.directory='*' -c core.excludesFile={EXCLUDES_FILE}"
CHANGES_ARTIFACT = "/logs/artifacts/changes.patch"
PATCH_EXIT_CODE = 97
OVERLAY_EXIT_CODE = 98
MAX_OUTPUT_BYTES = 2 * 1024 * 1024

GradeStatus = Literal["passed", "failed", "error", "timeout"]


@dataclass(frozen=True)
class TestCounts:
    """Per-case counts parsed from one test runner summary."""

    passed: int
    failed: int
    errors: int
    skipped: int

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors


@dataclass(frozen=True)
class CommandOutcome:
    """The observable result of one sealed command run."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class TestCommandResult:
    """Trusted grading result for one workspace state."""

    status: GradeStatus
    passed_cases: int
    total_cases: int
    reward: float
    reason: str
    counts: TestCounts | None


_PYTEST_LABELS = r"passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected"
_PYTEST_SUMMARY = re.compile(
    r"^(?:=+ )?(?P<body>\d+ (?:" + _PYTEST_LABELS + r")(?:, \d+ (?:" + _PYTEST_LABELS + r"))*)"
    r" in [0-9.]+s(?: \([^)]*\))?(?: =+)?$",
    re.M,
)
_PYTEST_ITEM = re.compile(r"(?P<count>\d+) (?P<label>passed|failed|errors?|skipped|xfailed|xpassed)")
_UNITTEST_RAN = re.compile(r"^Ran (?P<count>\d+) tests? in [0-9.]+s$", re.M)
_UNITTEST_STATUS = re.compile(r"^(?:OK|FAILED) ?(?:\((?P<details>[^)]*)\))?$", re.M)
_UNITTEST_DETAIL = re.compile(r"(?P<label>failures|errors|skipped|expected failures|unexpected successes)=(?P<count>\d+)")
_CARGO_RESULT = re.compile(
    r"^test result: (?:ok|FAILED)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored",
    re.M,
)
_GO_PASS = re.compile(r"^\s*--- PASS: ", re.M)
_GO_FAIL = re.compile(r"^\s*--- FAIL: ", re.M)
_GO_SKIP = re.compile(r"^\s*--- SKIP: ", re.M)
_BUN_PASS = re.compile(r"^\s*(?P<count>\d+) pass\b", re.M)
_BUN_FAIL = re.compile(r"^\s*(?P<count>\d+) fail\b", re.M)
_BUN_SKIP = re.compile(r"^\s*(?P<count>\d+) skip\b", re.M)
_VITEST_TESTS = re.compile(r"^\s*Tests\s+(?P<body>.*?)\s+\((?P<total>\d+)\)", re.M)
_VITEST_ITEM = re.compile(r"(?P<count>\d+) (?P<label>passed|failed|skipped|todo)")
_JEST_TESTS = re.compile(r"^Tests:\s+(?P<body>.*?),\s+(?P<total>\d+) total", re.M)
_JEST_ITEM = re.compile(r"(?P<count>\d+) (?P<label>passed|failed|skipped|todo)")
_NODE_TEST_PASS = re.compile(r"^# pass (?P<count>\d+)", re.M)
_NODE_TEST_FAIL = re.compile(r"^# fail (?P<count>\d+)", re.M)
_NODE_TEST_SKIP = re.compile(r"^# skipped (?P<count>\d+)", re.M)


def _pytest(text: str) -> TestCounts | None:
    matches = list(_PYTEST_SUMMARY.finditer(text))
    if not matches:
        return None
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for item in _PYTEST_ITEM.finditer(matches[-1].group("body")):
        label = item.group("label")
        count = int(item.group("count"))
        if label in {"error", "errors"}:
            counts["errors"] += count
        elif label in {"skipped", "xfailed", "xpassed"}:
            counts["skipped"] += count
        else:
            counts[label] += count
    if counts["passed"] + counts["failed"] + counts["errors"] + counts["skipped"] == 0:
        return None
    return TestCounts(**counts)


def _unittest(text: str) -> TestCounts | None:
    ran = list(_UNITTEST_RAN.finditer(text))
    if not ran:
        return None
    total = int(ran[-1].group("count"))
    status = list(_UNITTEST_STATUS.finditer(text, ran[-1].end()))
    if not status:
        return None
    failed = errors = skipped = 0
    details = status[-1].group("details") or ""
    for item in _UNITTEST_DETAIL.finditer(details):
        count = int(item.group("count"))
        label = item.group("label")
        if label == "failures":
            failed += count
        elif label in {"errors", "unexpected successes"}:
            errors += count
        else:
            skipped += count
    passed = total - failed - errors - skipped
    if passed < 0:
        return None
    return TestCounts(passed=passed, failed=failed, errors=errors, skipped=skipped)


def _cargo(text: str) -> TestCounts | None:
    matches = list(_CARGO_RESULT.finditer(text))
    if not matches:
        return None
    passed = sum(int(match.group("passed")) for match in matches)
    failed = sum(int(match.group("failed")) for match in matches)
    skipped = sum(int(match.group("ignored")) for match in matches)
    return TestCounts(passed=passed, failed=failed, errors=0, skipped=skipped)


def _go(text: str) -> TestCounts | None:
    passed = len(_GO_PASS.findall(text))
    failed = len(_GO_FAIL.findall(text))
    skipped = len(_GO_SKIP.findall(text))
    if passed + failed + skipped == 0:
        return None
    return TestCounts(passed=passed, failed=failed, errors=0, skipped=skipped)


def _bun(text: str) -> TestCounts | None:
    passed = _BUN_PASS.findall(text)
    failed = _BUN_FAIL.findall(text)
    if not passed and not failed:
        return None
    skipped = _BUN_SKIP.findall(text)
    return TestCounts(
        passed=int(passed[-1]) if passed else 0,
        failed=int(failed[-1]) if failed else 0,
        errors=0,
        skipped=int(skipped[-1]) if skipped else 0,
    )


def _labelled(
    summary: re.Pattern[str], item: re.Pattern[str], text: str
) -> TestCounts | None:
    matches = list(summary.finditer(text))
    if not matches:
        return None
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for entry in item.finditer(matches[-1].group("body")):
        label = entry.group("label")
        count = int(entry.group("count"))
        counts["skipped" if label == "todo" else label] += count
    return TestCounts(
        passed=counts["passed"],
        failed=counts["failed"],
        errors=0,
        skipped=counts["skipped"],
    )


def _node_test(text: str) -> TestCounts | None:
    passed = _NODE_TEST_PASS.findall(text)
    failed = _NODE_TEST_FAIL.findall(text)
    if not passed and not failed:
        return None
    skipped = _NODE_TEST_SKIP.findall(text)
    return TestCounts(
        passed=int(passed[-1]) if passed else 0,
        failed=int(failed[-1]) if failed else 0,
        errors=0,
        skipped=int(skipped[-1]) if skipped else 0,
    )


def parse_test_output(text: str) -> TestCounts | None:
    """Parse the summary of the first recognized test runner in ``text``."""
    for parser in (
        _pytest,
        _unittest,
        _cargo,
        lambda value: _labelled(_VITEST_TESTS, _VITEST_ITEM, value),
        lambda value: _labelled(_JEST_TESTS, _JEST_ITEM, value),
        _bun,
        _node_test,
        _go,
    ):
        counts = parser(text)
        if counts is not None:
            return counts
    return None


def grade_outcome(outcome: CommandOutcome, expected_cases: int) -> TestCommandResult:
    """Turn one sealed command run into a reward.

    The reward is the fraction of passed cases when a runner summary is
    recognized. A run that reports fewer cases than the sealed reference run
    scores zero: the agent removed or skipped tests. A run without a
    recognizable summary scores on the exit code only when the sealed
    reference run had exactly one case, which is how exit-code-only commands
    are recorded.
    """
    if outcome.timed_out:
        return TestCommandResult("timeout", 0, expected_cases, 0.0, "command timed out", None)
    if outcome.exit_code in {PATCH_EXIT_CODE, OVERLAY_EXIT_CODE}:
        reason = "workspace patch did not apply" if outcome.exit_code == PATCH_EXIT_CODE else (
            "sealed files could not be restored"
        )
        return TestCommandResult("error", 0, expected_cases, 0.0, reason, None)
    counts = parse_test_output(outcome.stdout + "\n" + outcome.stderr)
    if counts is None:
        if expected_cases != 1:
            return TestCommandResult(
                "error", 0, expected_cases, 0.0, "no test summary was recognized", None
            )
        if outcome.exit_code == 0:
            return TestCommandResult("passed", 1, 1, 1.0, "command exited 0", None)
        return TestCommandResult(
            "failed", 0, 1, 0.0, f"command exited {outcome.exit_code}", None
        )
    if counts.total < expected_cases:
        return TestCommandResult(
            "error",
            counts.passed,
            expected_cases,
            0.0,
            f"test count shrank: {counts.total} < {expected_cases}",
            counts,
        )
    if counts.total == 0:
        return TestCommandResult("error", 0, expected_cases, 0.0, "no tests ran", counts)
    if outcome.exit_code != 0 and counts.failed + counts.errors == 0:
        return TestCommandResult(
            "error",
            counts.passed,
            counts.total,
            0.0,
            f"command exited {outcome.exit_code} without reported failures",
            counts,
        )
    reward = counts.passed / counts.total
    if outcome.exit_code == 0 and counts.failed + counts.errors == 0:
        return TestCommandResult("passed", counts.passed, counts.total, reward, "all tests passed", counts)
    return TestCommandResult(
        "failed",
        counts.passed,
        counts.total,
        reward,
        f"{counts.failed + counts.errors} of {counts.total} tests failed",
        counts,
    )


def grading_script(
    command: Sequence[str],
    sealed_files: Sequence[str],
    *,
    apply_patch: bool,
) -> str:
    """Return the POSIX shell script that grades one workspace state.

    The caller writes the agent or reference patch to ``PATCH_PATH`` and the
    trusted sealed files under ``SEALED_DIR`` before running the script. The
    script exits with the command's exit code, or with a reserved code when
    the patch or the sealed overlay cannot be applied.
    """
    lines = [
        "#!/bin/sh",
        "set -u",
        f"cd {WORKSPACE} || exit {OVERLAY_EXIT_CODE}",
        "export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null",
    ]
    if apply_patch:
        lines.append(
            f"if [ -s {PATCH_PATH} ]; then "
            f"git {GIT_FLAGS} apply --binary --whitespace=nowarn {PATCH_PATH} "
            f"|| exit {PATCH_EXIT_CODE}; fi"
        )
    for relative in sealed_files:
        source = shlex.quote(f"{SEALED_DIR}/{relative}")
        target = shlex.quote(f"{WORKSPACE}/{relative}")
        if "/" in relative:
            parent = shlex.quote(f"{WORKSPACE}/{relative.rsplit('/', 1)[0]}")
            lines.append(f"mkdir -p {parent} || exit {OVERLAY_EXIT_CODE}")
        lines.append(f"cp -f {source} {target} || exit {OVERLAY_EXIT_CODE}")
    lines.append("exec " + " ".join(shlex.quote(item) for item in command))
    return "\n".join(lines) + "\n"


def changes_script() -> str:
    """Return the script that records the agent's workspace patch.

    The task image commits the start state under ``START_TAG``. Everything the
    agent changed, added, or removed since then travels as one binary patch in
    the implicit Prime artifact directory.
    """
    return "\n".join(
        (
            "#!/bin/sh",
            "set -eu",
            f"cd {WORKSPACE}",
            "export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null",
            "mkdir -p /logs/artifacts",
            f"git {GIT_FLAGS} add -A",
            f"git {GIT_FLAGS} diff --cached --binary {START_TAG} > {CHANGES_ARTIFACT}",
        )
    ) + "\n"
