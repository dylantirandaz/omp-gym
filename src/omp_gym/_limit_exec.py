"""Apply child resource limits, then replace this process."""

from __future__ import annotations

import os
import resource
import sys
from collections.abc import Sequence
from typing import NoReturn

_ARGUMENT_COUNT_BEFORE_COMMAND = 6


def _positive_integer(value: str, name: str) -> int:
    """Parse one positive internal limit."""
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _set_limit(kind: int, requested: int) -> None:
    """Set one limit without asking for more than the host permits."""
    _, hard = resource.getrlimit(kind)
    value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(kind, (value, value))


def main(arguments: Sequence[str]) -> NoReturn:
    """Apply the internal limit arguments and execute the target command."""
    if (
        len(arguments) <= _ARGUMENT_COUNT_BEFORE_COMMAND
        or arguments[_ARGUMENT_COUNT_BEFORE_COMMAND - 1] != "--"
    ):
        raise SystemExit(
            "usage: _limit_exec.py CPU MEMORY FILES OPEN_FILES PROCESSES -- COMMAND"
        )

    cpu_seconds = _positive_integer(arguments[0], "cpu seconds")
    memory_bytes = _positive_integer(arguments[1], "memory bytes")
    file_bytes = _positive_integer(arguments[2], "file bytes")
    open_files = _positive_integer(arguments[3], "open files")
    processes = _positive_integer(arguments[4], "processes")
    command = list(arguments[_ARGUMENT_COUNT_BEFORE_COMMAND:])

    _set_limit(resource.RLIMIT_CPU, cpu_seconds)
    if sys.platform != "darwin":
        _set_limit(resource.RLIMIT_AS, memory_bytes)
    _set_limit(resource.RLIMIT_FSIZE, file_bytes)
    _set_limit(resource.RLIMIT_NOFILE, open_files)
    _set_limit(resource.RLIMIT_NPROC, processes)
    os.execvpe(  # noqa: S606 - validated argv from the parent
        command[0], command, dict(os.environ)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
