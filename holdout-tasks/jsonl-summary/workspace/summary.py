from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

InvalidReason: TypeAlias = Literal[
    "invalid JSON",
    "record must be an object",
    "invalid status",
    "invalid duration_ms",
]


@dataclass(frozen=True)
class Summary:
    accepted: int
    rejected: int
    total_duration_ms: int


@dataclass(frozen=True)
class InvalidRecord:
    line_number: int
    reason: InvalidReason


SummaryResult: TypeAlias = Summary | InvalidRecord


def summarize_lines(lines: Iterable[str]) -> SummaryResult:
    """Summarize JSON objects from JSON Lines text.

    Ignore blank lines. Each other line must contain one JSON object.
    The status must be accepted or rejected. The duration_ms value must be
    a nonnegative integer and must not be a Boolean value. Return the first
    invalid record. The line number starts at one and includes blank lines.
    """
    raise NotImplementedError
