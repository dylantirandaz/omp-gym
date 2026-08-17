import unittest

from omp_gym.rl import (
    COMPLETION_COLLAPSE_RATIO,
    _completion_length_collapsed,
    _normalized_advantages,
)


class NormalizedAdvantagesTests(unittest.TestCase):
    def test_advantages_have_zero_mean(self) -> None:
        advantages = _normalized_advantages([1.0, 2.0, 3.0, 6.0])
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0)

    def test_advantages_have_unit_std(self) -> None:
        advantages = _normalized_advantages([0.0, 1.0])
        mean = sum(advantages) / len(advantages)
        variance = sum(
            (value - mean) ** 2 for value in advantages
        ) / len(advantages)
        self.assertAlmostEqual(variance**0.5, 1.0)

    def test_equal_rewards_give_zero_advantages(self) -> None:
        # The zero-std guard divides by 1.0 instead of 0.0.
        advantages = _normalized_advantages([0.5, 0.5, 0.5])
        self.assertEqual(advantages, [0.0, 0.0, 0.0])

    def test_higher_reward_gets_higher_advantage(self) -> None:
        low, high = _normalized_advantages([0.0, 1.0])
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)


class CompletionLengthCollapseTests(unittest.TestCase):
    def test_ratio_is_the_documented_value(self) -> None:
        self.assertEqual(COMPLETION_COLLAPSE_RATIO, 0.3)

    def test_stops_when_length_falls_below_the_floor(self) -> None:
        self.assertTrue(_completion_length_collapsed(100.0, 25.0))

    def test_continues_above_the_floor(self) -> None:
        self.assertFalse(_completion_length_collapsed(100.0, 45.0))

    def test_the_floor_itself_does_not_stop(self) -> None:
        self.assertFalse(_completion_length_collapsed(100.0, 30.0))

    def test_first_iteration_never_stops(self) -> None:
        self.assertFalse(_completion_length_collapsed(None, 1.0))

    def test_zero_reference_disables_the_check(self) -> None:
        self.assertFalse(_completion_length_collapsed(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
