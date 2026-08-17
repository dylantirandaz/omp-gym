"""An exponential backoff planner with bounded jitter.

plan_retries returns the list of waits, one per retry attempt:

- For attempt n (counted from 0) the uncapped delay is
  base * factor**n. The cap is max_delay:
  capped = min(base * factor**n, max_delay).
- Jitter keeps the delay inside [capped / 2, capped):
  delay = capped * (0.5 + jitter.jitter_fraction(seed, n) / 2).
- Delays accumulate left to right into an elapsed float sum. When
  elapsed + delay is greater than max_elapsed, the plan stops and
  that delay is not included. An exact hit (elapsed + delay equal to
  max_elapsed) is still included.
- The plan never has more than max_attempts delays.
"""


def plan_retries(
    base: float,
    factor: float,
    max_delay: float,
    max_elapsed: float,
    max_attempts: int,
    seed: int,
) -> list[float]:
    """Return the planned waits in seconds.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
