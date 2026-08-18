"""Deterministic jitter from a 64-bit linear congruential generator.

jitter_fraction(seed, attempt) returns a float in [0.0, 1.0):

- state starts at seed modulo 2**64.
- The step function is
  state = (state * 6364136223846793005 + 1442695040888963407) mod 2**64.
- For attempt n (counted from 0) apply the step n + 1 times.
- The fraction is (state >> 11) / 2**53.

The same (seed, attempt) pair always gives the same fraction. There
is no global state and no use of the random module.
"""


def jitter_fraction(seed: int, attempt: int) -> float:
    """Return the deterministic jitter fraction in [0.0, 1.0).

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
