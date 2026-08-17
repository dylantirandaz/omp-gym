import unittest

from jitter import jitter_fraction
from policy import plan_retries


class JitterTest(unittest.TestCase):
    def test_first_attempt_for_seed_seven(self) -> None:
        self.assertAlmostEqual(jitter_fraction(7, 0), 0.4932122668392295, places=12)

    def test_second_attempt_for_seed_seven(self) -> None:
        self.assertAlmostEqual(jitter_fraction(7, 1), 0.9556595384052861, places=12)

    def test_third_attempt_for_seed_seven(self) -> None:
        self.assertAlmostEqual(jitter_fraction(7, 2), 0.9065758219926131, places=12)

    def test_seed_zero(self) -> None:
        self.assertAlmostEqual(jitter_fraction(0, 0), 0.07820865487829387, places=12)

    def test_seed_one_differs_from_seed_zero(self) -> None:
        self.assertAlmostEqual(jitter_fraction(1, 0), 0.42320917087271326, places=12)

    def test_fraction_stays_in_range(self) -> None:
        for seed in (0, 1, 7, 42, 2**63):
            for attempt in range(8):
                fraction = jitter_fraction(seed, attempt)
                self.assertGreaterEqual(fraction, 0.0)
                self.assertLess(fraction, 1.0)

    def test_repeat_calls_are_stable(self) -> None:
        self.assertEqual(jitter_fraction(42, 3), jitter_fraction(42, 3))


class PlanTest(unittest.TestCase):
    def test_exact_plan_for_seed_42(self) -> None:
        plan = plan_retries(1.0, 2.0, 60.0, 1000.0, 6, 42)
        expected = [
            0.7841151633219539,
            1.2254634289477513,
            2.825676637659024,
            6.521592199358391,
            13.441182457936925,
            16.419662571199012,
        ]
        self.assertEqual(len(plan), 6)
        for value, want in zip(plan, expected):
            self.assertAlmostEqual(value, want, places=9)

    def test_delays_stay_inside_the_jitter_band(self) -> None:
        plan = plan_retries(1.0, 2.0, 60.0, 1000.0, 6, 42)
        for attempt, delay in enumerate(plan):
            capped = min(1.0 * 2.0**attempt, 60.0)
            self.assertGreaterEqual(delay, capped / 2)
            self.assertLess(delay, capped)

    def test_max_delay_caps_late_attempts(self) -> None:
        plan = plan_retries(1.0, 3.0, 4.0, 1000.0, 5, 7)
        self.assertAlmostEqual(plan[3], 2.5454930227720336, places=9)
        self.assertAlmostEqual(plan[4], 2.532758930113479, places=9)
        for delay in plan:
            self.assertLess(delay, 4.0)

    def test_max_elapsed_excludes_the_crossing_delay(self) -> None:
        plan = plan_retries(1.0, 2.0, 60.0, 24.0, 6, 42)
        self.assertEqual(len(plan), 4)

    def test_exact_elapsed_hit_is_included(self) -> None:
        plan = plan_retries(1.0, 2.0, 60.0, 24.798029887224047, 6, 42)
        self.assertEqual(len(plan), 5)

    def test_max_attempts_caps_the_plan(self) -> None:
        plan = plan_retries(1.0, 2.0, 60.0, 1000.0, 3, 42)
        self.assertEqual(len(plan), 3)

    def test_zero_attempts_gives_an_empty_plan(self) -> None:
        self.assertEqual(plan_retries(1.0, 2.0, 60.0, 1000.0, 0, 42), [])

    def test_tiny_budget_gives_an_empty_plan(self) -> None:
        self.assertEqual(plan_retries(1.0, 2.0, 60.0, 0.1, 6, 42), [])

    def test_seed_changes_the_plan(self) -> None:
        plan_a = plan_retries(1.0, 2.0, 60.0, 1000.0, 1, 0)
        plan_b = plan_retries(1.0, 2.0, 60.0, 1000.0, 1, 1)
        self.assertNotEqual(plan_a, plan_b)


if __name__ == "__main__":
    unittest.main()
