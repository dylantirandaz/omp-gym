"""Window arithmetic for the sliding-window limiter.

prune removes expired timestamps. A timestamp t is live at time now
when now - t < window. A timestamp at exactly now - window has
expired. The input list is sorted ascending; the output keeps the
live suffix in the same order.

next_free_time returns the earliest time at which one slot opens,
given the live timestamps of a full window. That time is
oldest + window.
"""


def prune(timestamps: list[float], now: float, window: float) -> list[float]:
    """Return only the live timestamps, oldest first.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError


def next_free_time(timestamps: list[float], window: float) -> float:
    """Return the time at which the oldest live timestamp expires.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
