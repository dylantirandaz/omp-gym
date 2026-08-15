import unittest

from events import Event, latest_events


class LatestEventsTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(latest_events(()), ())

    def test_one_event(self) -> None:
        event = Event("a", 1, "first")
        self.assertEqual(latest_events((event,)), (event,))

    def test_larger_revision_replaces_older_event(self) -> None:
        events = (
            Event("a", 1, "old"),
            Event("a", 3, "new"),
            Event("a", 2, "middle"),
        )
        self.assertEqual(latest_events(events), (Event("a", 3, "new"),))

    def test_first_id_order_is_stable(self) -> None:
        events = (
            Event("b", 1, "b-old"),
            Event("a", 5, "a"),
            Event("b", 2, "b-new"),
        )
        self.assertEqual(
            latest_events(events),
            (Event("b", 2, "b-new"), Event("a", 5, "a")),
        )

    def test_later_equal_revision_wins(self) -> None:
        events = (
            Event("a", 4, "first"),
            Event("a", 4, "second"),
        )
        self.assertEqual(latest_events(events), (Event("a", 4, "second"),))

    def test_lower_revision_does_not_replace_newer_event(self) -> None:
        events = (
            Event("a", 8, "new"),
            Event("a", 2, "old"),
        )
        self.assertEqual(latest_events(events), (Event("a", 8, "new"),))

    def test_multiple_ids_and_revisions(self) -> None:
        events = (
            Event("x", 1, "x1"),
            Event("y", 3, "y3"),
            Event("z", 2, "z2"),
            Event("y", 4, "y4"),
            Event("x", 1, "x1-later"),
        )
        self.assertEqual(
            latest_events(events),
            (
                Event("x", 1, "x1-later"),
                Event("y", 4, "y4"),
                Event("z", 2, "z2"),
            ),
        )

    def test_input_does_not_change(self) -> None:
        events = [Event("a", 1, "a"), Event("a", 2, "b")]
        original = list(events)
        latest_events(events)
        self.assertEqual(events, original)


if __name__ == "__main__":
    unittest.main()
