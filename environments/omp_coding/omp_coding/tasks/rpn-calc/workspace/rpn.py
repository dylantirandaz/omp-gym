"""Reverse Polish notation evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RpnError:
    """The token sequence is not a valid RPN expression."""

    reason: str


def evaluate_rpn(tokens: Sequence[str]) -> int | RpnError:
    """Evaluate integer tokens with the operators + - * /.

    Division truncates toward zero.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
