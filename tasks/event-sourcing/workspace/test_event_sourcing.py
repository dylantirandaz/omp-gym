import unittest

from events import INITIAL_STATE, Event, LedgerError, State, apply_event
from store import compact, replay


class ApplyEventTest(unittest.TestCase):
    def test_deposit(self) -> None:
        self.assertEqual(
            apply_event(INITIAL_STATE, Event(1, "deposit", 50)),
            State(balance=50, last_seq=1),
        )

    def test_withdraw(self) -> None:
        state = State(balance=80, last_seq=3)
        self.assertEqual(
            apply_event(state, Event(4, "withdraw", 30)),
            State(balance=50, last_seq=4),
        )

    def test_withdraw_exact_balance_is_allowed(self) -> None:
        state = State(balance=30, last_seq=1)
        self.assertEqual(
            apply_event(state, Event(2, "withdraw", 30)),
            State(balance=0, last_seq=2),
        )

    def test_withdraw_over_balance_is_an_error(self) -> None:
        state = State(balance=10, last_seq=6)
        self.assertEqual(
            apply_event(state, Event(7, "withdraw", 11)),
            LedgerError("insufficient funds at seq 7"),
        )

    def test_snapshot_replaces_the_balance(self) -> None:
        state = State(balance=999, last_seq=2)
        self.assertEqual(
            apply_event(state, Event(5, "snapshot", 40)),
            State(balance=40, last_seq=5),
        )

    def test_duplicate_seq_is_ignored(self) -> None:
        state = State(balance=70, last_seq=4)
        self.assertEqual(apply_event(state, Event(4, "deposit", 70)), state)

    def test_stale_seq_is_ignored(self) -> None:
        state = State(balance=70, last_seq=4)
        self.assertEqual(apply_event(state, Event(2, "withdraw", 500)), state)

    def test_unknown_kind_is_an_error(self) -> None:
        self.assertEqual(
            apply_event(INITIAL_STATE, Event(1, "transfer", 5)),
            LedgerError("unknown kind: transfer"),
        )


class ReplayTest(unittest.TestCase):
    def test_empty_log(self) -> None:
        self.assertEqual(replay([]), INITIAL_STATE)

    def test_in_order_log(self) -> None:
        log = [Event(1, "deposit", 100), Event(2, "withdraw", 40)]
        self.assertEqual(replay(log), State(balance=60, last_seq=2))

    def test_out_of_order_log_is_sorted_first(self) -> None:
        log = [Event(2, "withdraw", 40), Event(1, "deposit", 100)]
        self.assertEqual(replay(log), State(balance=60, last_seq=2))

    def test_duplicate_seq_first_occurrence_wins(self) -> None:
        log = [
            Event(1, "deposit", 100),
            Event(2, "deposit", 5),
            Event(2, "deposit", 900),
        ]
        self.assertEqual(replay(log), State(balance=105, last_seq=2))

    def test_duplicate_wins_even_when_listed_later_but_smaller_seq(self) -> None:
        log = [
            Event(2, "deposit", 7),
            Event(1, "deposit", 100),
            Event(2, "deposit", 900),
        ]
        self.assertEqual(replay(log), State(balance=107, last_seq=2))

    def test_error_stops_the_fold(self) -> None:
        log = [
            Event(1, "deposit", 10),
            Event(2, "withdraw", 25),
            Event(3, "deposit", 1000),
        ]
        self.assertEqual(replay(log), LedgerError("insufficient funds at seq 2"))

    def test_snapshot_inside_the_log(self) -> None:
        log = [
            Event(1, "deposit", 10),
            Event(2, "snapshot", 500),
            Event(3, "withdraw", 200),
        ]
        self.assertEqual(replay(log), State(balance=300, last_seq=3))


class CompactTest(unittest.TestCase):
    def test_compact_produces_snapshot_plus_tail(self) -> None:
        log = [
            Event(1, "deposit", 100),
            Event(2, "withdraw", 30),
            Event(3, "deposit", 5),
        ]
        self.assertEqual(
            compact(log, 2),
            [Event(2, "snapshot", 70), Event(3, "deposit", 5)],
        )

    def test_compact_sorts_and_dedupes_the_tail(self) -> None:
        log = [
            Event(4, "deposit", 8),
            Event(1, "deposit", 100),
            Event(3, "deposit", 1),
            Event(3, "deposit", 999),
        ]
        self.assertEqual(
            compact(log, 1),
            [
                Event(1, "snapshot", 100),
                Event(3, "deposit", 1),
                Event(4, "deposit", 8),
            ],
        )

    def test_compact_replays_to_the_same_state(self) -> None:
        log = [
            Event(2, "withdraw", 40),
            Event(1, "deposit", 100),
            Event(4, "deposit", 3),
            Event(3, "snapshot", 90),
        ]
        compacted = compact(log, 3)
        self.assertIsInstance(compacted, list)
        self.assertEqual(replay(compacted), replay(log))

    def test_compact_reports_a_replay_error(self) -> None:
        log = [Event(1, "withdraw", 1)]
        self.assertEqual(
            compact(log, 1), LedgerError("insufficient funds at seq 1")
        )

    def test_compact_of_an_empty_log(self) -> None:
        self.assertEqual(compact([], 5), [Event(5, "snapshot", 0)])


if __name__ == "__main__":
    unittest.main()
