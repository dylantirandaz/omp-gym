from collections.abc import Sequence
from typing import TypeAlias

Interval: TypeAlias = tuple[int, int]


def merge_intervals(intervals: Sequence[Interval]) -> tuple[Interval, ...]:
    """Merge all closed intervals that overlap or touch.

    Each interval has a start value that is not more than its end value.
    Return the merged intervals in ascending order. Do not change the input.
    """
    raise NotImplementedError
