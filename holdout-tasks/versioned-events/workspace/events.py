from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    event_id: str
    revision: int
    payload: str


def latest_events(events: Sequence[Event]) -> tuple[Event, ...]:
    """Return the newest event for each ID.

    Keep the order in which each ID first occurs. A larger revision is newer.
    For equal revisions, use the event that occurs later in the input.
    Do not change the input.
    """
    latest_by_id: dict[str, Event] = {}
    first_id_order: list[str] = []
    for event in events:
        if event.event_id not in latest_by_id:
            latest_by_id[event.event_id] = event
            first_id_order.append(event.event_id)
        elif event.revision >= latest_by_id[event.event_id].revision:
            continue
    return tuple(latest_by_id[event_id] for event_id in first_id_order)
