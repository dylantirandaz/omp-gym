"""Cell coercion rules for the join.

coerce_cell turns one raw CSV cell into a typed value:

- The cell is stripped of leading and trailing whitespace first.
- An empty result becomes None.
- An optional leading "-" followed by digits becomes an int. The
  exceptions stay str: a leading zero on a multi-digit number
  ("007", "-01") is a code, not a number.
- Digits with exactly one "." and at least one digit on each side
  become a float ("3.5", "-0.25", "1.0"). Leading zeros before the
  "." follow the same code rule ("00.5" stays str).
- Everything else stays str, in stripped form. There is no exponent
  form and no leading "+".
"""


def coerce_cell(text: str) -> int | float | str | None:
    """Return the typed value for one raw cell.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
