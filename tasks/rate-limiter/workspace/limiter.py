"""A per-key sliding-window rate limiter.

The limiter allows at most `limit` events per key inside any window
of `window` seconds. It stores the timestamps of allowed events and
prunes them with window.prune on every call.

allow(key, now) returns True and records the event when fewer than
`limit` events are live for the key. It returns False and records
nothing when the key is full.

retry_after(key, now) returns 0.0 when a call to allow(key, now)
would succeed. Otherwise it returns the number of seconds from now
until the earliest time at which one slot opens
(window.next_free_time). retry_after never records an event.

Time comes only from the `now` arguments. The limiter holds no wall
clock. Keys are independent.
"""


class SlidingWindowLimiter:
    """Allow at most `limit` events per `window` seconds per key."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._events: dict[str, list[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        """Record and allow one event, or deny it.

        Not implemented yet. This is the task.
        """
        raise NotImplementedError

    def retry_after(self, key: str, now: float) -> float:
        """Return the wait in seconds until allow can succeed.

        Not implemented yet. This is the task.
        """
        raise NotImplementedError
