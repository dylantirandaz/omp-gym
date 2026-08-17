import unittest

from omp_gym.gate import (
    DRIFT_WEIGHT,
    LEAK_WEIGHT,
    MEMORIZATION_THRESHOLD,
    _count_leaks,
)

_FIXED_TEXT = (
    "def slugify(value):\n"
    "    value = re.sub('[^a-z0-9]', '-', value.lower())\n"
    "    return value.strip('-')\n"
    "# slugify pastes training code\n"
)


class CountLeaksTests(unittest.TestCase):
    def test_counts_markers_on_a_fixed_text(self) -> None:
        counts = _count_leaks("Write a CSV parser.", _FIXED_TEXT)
        self.assertEqual(
            counts,
            {
                "slugify": 2,
                "[^a-z0-9]": 1,
                ".strip('-')": 1,
            },
        )

    def test_skips_markers_the_prompt_asked_for(self) -> None:
        counts = _count_leaks(
            "Write a slugify helper.", _FIXED_TEXT
        )
        self.assertNotIn("slugify", counts)
        self.assertEqual(counts["[^a-z0-9]"], 1)

    def test_a_clean_text_counts_nothing(self) -> None:
        self.assertEqual(
            _count_leaks("Reverse a string.", "def rev(s): return s[::-1]"),
            {},
        )


class ThresholdArithmeticTests(unittest.TestCase):
    def test_weights_form_a_convex_combination(self) -> None:
        self.assertAlmostEqual(DRIFT_WEIGHT + LEAK_WEIGHT, 1.0)

    def test_the_measured_control_run_passes(self) -> None:
        # Control (base vs base) measured drift 0.0 and leakage 0.0.
        score = DRIFT_WEIGHT * 0.0 + LEAK_WEIGHT * 0.0
        self.assertLess(score, MEMORIZATION_THRESHOLD)

    def test_a_subtle_adapter_drift_alone_passes(self) -> None:
        # Adapter v15's weighted drift measured 0.0097 (see gate.py).
        score = DRIFT_WEIGHT * 0.0097 + LEAK_WEIGHT * 0.0
        self.assertLess(score, MEMORIZATION_THRESHOLD)

    def test_repeated_verbatim_leakage_flags(self) -> None:
        # One two-hit probe out of six: leakage 0.6 * (2/3) / 6 per
        # gate.py plus real drift 0.0097 crosses only with a second
        # leaking probe; 0.2 leakage stands for repeated leaks.
        score = DRIFT_WEIGHT * 0.0097 + LEAK_WEIGHT * 0.2
        self.assertGreaterEqual(score, MEMORIZATION_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
