"""Replay and compaction on top of the event transition function.

replay folds a list of events into a final state:

- Events can arrive out of order. Replay applies them in ascending
  seq order.
- Two events with the same seq are duplicates. The first occurrence
  in the input list wins; later occurrences are dropped before the
  sort.
- The fold starts from INITIAL_STATE. The first LedgerError from
  apply_event stops the fold and is returned.

compact(events, upto_seq) shortens the log:

- It replays only the events with seq less than or equal to upto_seq
  (after the same dedup rule). A LedgerError from that replay is
  returned.
- It returns a new log: one Event(upto_seq, "snapshot", balance)
  followed by the surviving events with seq greater than upto_seq,
  in ascending seq order.
- Replaying the compacted log gives the same final state as
  replaying the original log.
"""

from events import Event, LedgerError, State


def replay(events: list[Event]) -> State | LedgerError:
    """Fold the events into a final state, or return an error value.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError


def compact(events: list[Event], upto_seq: int) -> list[Event] | LedgerError:
    """Return a shorter log with the same replay result.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
