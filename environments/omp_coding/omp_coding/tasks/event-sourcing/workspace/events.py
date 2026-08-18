"""Ledger events and the state transition function.

An event moves the ledger from one state to the next. Events carry a
sequence number. The rules are:

- An event with seq less than or equal to state.last_seq is a replayed
  duplicate. The transition returns the state unchanged (idempotency).
- kind "deposit": the balance increases by amount. last_seq becomes
  the event seq.
- kind "withdraw": the balance decreases by amount. A withdrawal of
  the exact balance is allowed. When amount is greater than the
  balance, the transition returns
  LedgerError("insufficient funds at seq {seq}").
- kind "snapshot": the balance becomes amount and last_seq becomes the
  event seq. A snapshot replaces the state; it does not add to it.
- Any other kind returns LedgerError("unknown kind: {kind}").
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """One ledger event."""

    seq: int
    kind: str
    amount: int


@dataclass(frozen=True)
class State:
    """The ledger state after some events."""

    balance: int
    last_seq: int


INITIAL_STATE = State(balance=0, last_seq=0)


@dataclass(frozen=True)
class LedgerError:
    """An event could not be applied."""

    reason: str


def apply_event(state: State, event: Event) -> State | LedgerError:
    """Return the next state, or an error value.

    Not implemented yet. This is the task.
    """
    raise NotImplementedError
