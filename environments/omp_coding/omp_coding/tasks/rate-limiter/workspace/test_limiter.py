import unittest

from limiter import SlidingWindowLimiter
from window import next_free_time, prune


class PruneTest(unittest.TestCase):
    def test_keeps_live_timestamps(self) -> None:
        self.assertEqual(prune([1.0, 2.0, 3.0], 3.5, 10.0), [1.0, 2.0, 3.0])

    def test_drops_expired_timestamps(self) -> None:
        self.assertEqual(prune([1.0, 5.0, 9.0], 12.0, 10.0), [5.0, 9.0])

    def test_boundary_timestamp_has_expired(self) -> None:
        self.assertEqual(prune([2.0, 3.0], 12.0, 10.0), [3.0])

    def test_empty_input(self) -> None:
        self.assertEqual(prune([], 5.0, 10.0), [])

    def test_next_free_time(self) -> None:
        self.assertEqual(next_free_time([4.0, 6.0, 7.0], 10.0), 14.0)


class AllowTest(unittest.TestCase):
    def test_burst_up_to_the_limit(self) -> None:
        limiter = SlidingWindowLimiter(limit=3, window=10.0)
        self.assertEqual(
            [limiter.allow("k", 0.0) for _ in range(4)],
            [True, True, True, False],
        )

    def test_denied_events_are_not_recorded(self) -> None:
        limiter = SlidingWindowLimiter(limit=1, window=10.0)
        self.assertTrue(limiter.allow("k", 0.0))
        self.assertFalse(limiter.allow("k", 5.0))
        self.assertFalse(limiter.allow("k", 9.999))
        self.assertTrue(limiter.allow("k", 10.0))

    def test_slot_opens_exactly_at_the_window_edge(self) -> None:
        limiter = SlidingWindowLimiter(limit=1, window=10.0)
        self.assertTrue(limiter.allow("k", 3.0))
        self.assertFalse(limiter.allow("k", 12.999))
        self.assertTrue(limiter.allow("k", 13.0))

    def test_window_slides_one_slot_at_a_time(self) -> None:
        limiter = SlidingWindowLimiter(limit=2, window=10.0)
        self.assertTrue(limiter.allow("k", 0.0))
        self.assertTrue(limiter.allow("k", 6.0))
        self.assertFalse(limiter.allow("k", 9.0))
        self.assertTrue(limiter.allow("k", 10.0))
        self.assertFalse(limiter.allow("k", 15.0))
        self.assertTrue(limiter.allow("k", 16.0))

    def test_keys_are_independent(self) -> None:
        limiter = SlidingWindowLimiter(limit=1, window=10.0)
        self.assertTrue(limiter.allow("a", 0.0))
        self.assertTrue(limiter.allow("b", 0.0))
        self.assertFalse(limiter.allow("a", 1.0))
        self.assertFalse(limiter.allow("b", 1.0))


class RetryAfterTest(unittest.TestCase):
    def test_zero_for_a_fresh_key(self) -> None:
        limiter = SlidingWindowLimiter(limit=2, window=10.0)
        self.assertEqual(limiter.retry_after("k", 0.0), 0.0)

    def test_zero_when_a_slot_is_free(self) -> None:
        limiter = SlidingWindowLimiter(limit=2, window=10.0)
        limiter.allow("k", 0.0)
        self.assertEqual(limiter.retry_after("k", 1.0), 0.0)

    def test_wait_until_the_oldest_event_expires(self) -> None:
        limiter = SlidingWindowLimiter(limit=2, window=10.0)
        limiter.allow("k", 2.0)
        limiter.allow("k", 5.0)
        self.assertEqual(limiter.retry_after("k", 6.0), 6.0)

    def test_wait_shrinks_as_time_passes(self) -> None:
        limiter = SlidingWindowLimiter(limit=1, window=10.0)
        limiter.allow("k", 0.0)
        self.assertEqual(limiter.retry_after("k", 7.5), 2.5)

    def test_retry_after_records_nothing(self) -> None:
        limiter = SlidingWindowLimiter(limit=1, window=10.0)
        limiter.allow("k", 0.0)
        limiter.retry_after("k", 1.0)
        limiter.retry_after("k", 2.0)
        self.assertTrue(limiter.allow("k", 10.0))

    def test_denied_allow_does_not_push_the_wait_out(self) -> None:
        limiter = SlidingWindowLimiter(limit=1, window=10.0)
        limiter.allow("k", 0.0)
        self.assertFalse(limiter.allow("k", 4.0))
        self.assertEqual(limiter.retry_after("k", 4.0), 6.0)


if __name__ == "__main__":
    unittest.main()
