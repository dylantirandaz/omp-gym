import unittest

from omp_gym.dpo import DPO_BETA, LORA_CONFIG, _pad_length


class PadLengthTests(unittest.TestCase):
    def _pair(self, prompt: int, chosen: int, rejected: int):
        return ([0] * prompt, [0] * chosen, [0] * rejected)

    def test_rounds_up_to_a_multiple_of_128(self) -> None:
        self.assertEqual(_pad_length([self._pair(10, 20, 5)]), 128)

    def test_an_exact_multiple_stays(self) -> None:
        self.assertEqual(_pad_length([self._pair(100, 28, 1)]), 128)

    def test_one_token_over_adds_a_full_block(self) -> None:
        self.assertEqual(_pad_length([self._pair(100, 29, 1)]), 256)

    def test_the_longer_completion_side_wins(self) -> None:
        # rejected is longer than chosen: 100 + 60 = 160 -> 256.
        self.assertEqual(_pad_length([self._pair(100, 10, 60)]), 256)

    def test_the_longest_pair_sets_the_length(self) -> None:
        pairs = [
            self._pair(10, 10, 10),
            self._pair(300, 100, 20),
        ]
        self.assertEqual(_pad_length(pairs), 512)


class ConstantsTests(unittest.TestCase):
    def test_lora_config_is_importable_without_mlx(self) -> None:
        # rl.py and the config artifacts depend on these keys.
        self.assertEqual(
            set(LORA_CONFIG), {"rank", "scale", "dropout"}
        )
        self.assertGreater(DPO_BETA, 0.0)


if __name__ == "__main__":
    unittest.main()
