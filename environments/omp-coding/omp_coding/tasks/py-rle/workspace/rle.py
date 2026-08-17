"""Run-length encoding."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RleError:
    """The text is not a valid run-length encoding."""

    reason: str


def encode(text: str) -> str:
    """Return the run-length encoding of the text.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError


def decode(encoded: str) -> str | RleError:
    """Return the text for a run-length encoding.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
