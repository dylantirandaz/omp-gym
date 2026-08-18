"""An inner join of two row lists on one key column.

Rows are dicts of raw string cells. join_rows coerces every cell
with coerce.coerce_cell and then joins:

- A row on either side that has no `key` column at all is a data
  error: return JoinError("{side} row {index} has no column {key}")
  where side is "left" or "right" and index is the position in that
  side, counted from 0. The first such row in left, then in right,
  wins.
- A row whose coerced key is None never matches; it is dropped.
- When several right rows share one key, the last one wins.
- The result has one dict per matching left row, in left order:
  the coerced left cells first, updated by the coerced right cells.
  On a shared column name the right value wins.
- Keys match by coerced value, so "7" matches " 7 " and "7.0" does
  not match "7" (int and float never compare equal here: match on
  (type, value)).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JoinError:
    """The input rows are not joinable."""

    reason: str


def join_rows(
    left: list[dict[str, str]],
    right: list[dict[str, str]],
    key: str,
) -> list[dict] | JoinError:
    """Return the joined rows, or an error value.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
