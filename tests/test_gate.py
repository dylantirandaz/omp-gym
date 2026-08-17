import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from omp_gym.gate import (
    DRIFT_WEIGHT,
    LEAK_WEIGHT,
    MEMORIZATION_THRESHOLD,
    _count_leaks,
    _train_curve_verdict,
    memorization_shaped_run,
)
from omp_gym.sae import _excerpt_fingerprint, _sample_hits

try:
    import mlx.core as mx
except ModuleNotFoundError:
    mx = None

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
        counts = _count_leaks("Write a slugify helper.", _FIXED_TEXT)
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


class MemorizationShapeTests(unittest.TestCase):
    def test_default_thresholds_flag_the_measured_run(self) -> None:
        # Measured run: train 2.058 -> 1.404 (-31.8%) while val
        # only moved 2.151 -> 2.001 (-7.0%).
        reason = memorization_shaped_run([2.058, 1.404], [2.151, 2.001])
        self.assertIsNotNone(reason)
        assert reason is not None
        self.assertIn("memorization-shaped", reason)
        self.assertIn("31.8%", reason)
        self.assertIn("7.0%", reason)

    def test_train_drop_pct_is_threaded_through(self) -> None:
        # The observed 31.8% train drop clears a 40% bar.
        self.assertIsNone(
            memorization_shaped_run([2.058, 1.404], [2.151, 2.001], train_drop_pct=0.40)
        )

    def test_val_drop_ratio_is_threaded_through(self) -> None:
        # Train -30% with val -10%: val covers a third of the train
        # drop, which passes the 0.25 ratio but fails a 0.5 ratio.
        self.assertIsNone(memorization_shaped_run([2.0, 1.4], [2.0, 1.8]))
        self.assertIsNotNone(
            memorization_shaped_run([2.0, 1.4], [2.0, 1.8], val_drop_ratio=0.5)
        )

    def test_no_val_losses_means_no_verdict(self) -> None:
        self.assertIsNone(memorization_shaped_run([2.0, 1.0], []))

    def test_a_zero_first_loss_cannot_compute_a_drop(self) -> None:
        self.assertIsNone(memorization_shaped_run([0.0, 0.0], [1.0, 0.5]))


class TrainCurveVerdictTests(unittest.TestCase):
    def _report(self, directory: Path, **fields) -> None:
        (directory / "train_report.json").write_text(json.dumps(fields))

    def test_replays_the_curve_check_on_the_adapter_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._report(
                Path(tmp),
                first_train_loss=2.058,
                last_train_loss=1.404,
                first_val_loss=2.151,
                last_val_loss=2.001,
            )
            verdict = _train_curve_verdict(
                Path(tmp), train_drop_pct=0.15, val_drop_ratio=0.25
            )
            self.assertIsNotNone(verdict)
            assert verdict is not None
            self.assertIn("memorization-shaped", verdict)
            self.assertIsNone(
                _train_curve_verdict(
                    Path(tmp), train_drop_pct=0.40, val_drop_ratio=0.25
                )
            )

    def test_a_clean_report_gives_no_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._report(
                Path(tmp),
                first_train_loss=2.0,
                last_train_loss=1.4,
                first_val_loss=2.0,
                last_val_loss=1.5,
            )
            self.assertIsNone(
                _train_curve_verdict(
                    Path(tmp), train_drop_pct=0.15, val_drop_ratio=0.25
                )
            )

    def test_missing_or_malformed_report_is_no_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(
                _train_curve_verdict(root, train_drop_pct=0.15, val_drop_ratio=0.25)
            )
            self.assertIsNone(
                _train_curve_verdict(None, train_drop_pct=0.15, val_drop_ratio=0.25)
            )
            (root / "train_report.json").write_text("not json")
            self.assertIsNone(
                _train_curve_verdict(root, train_drop_pct=0.15, val_drop_ratio=0.25)
            )


class SaeExcerptRedactionTests(unittest.TestCase):
    def test_fingerprint_is_sha256_and_length_only(self) -> None:
        text = "def total_column(rows):\n    return sum(r['x'] for r in rows)\n"
        fingerprint = _excerpt_fingerprint(text)
        self.assertEqual(set(fingerprint), {"sha256", "length"})
        self.assertEqual(
            fingerprint["sha256"],
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(fingerprint["length"], len(text))
        self.assertNotIn(text, json.dumps(fingerprint))

    def test_sample_hits_carry_fingerprints_not_text(self) -> None:
        if mx is None:
            self.skipTest("mlx is not available on this machine")
        z = mx.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        fingerprint = {"sha256": "ab" * 32, "length": 42}
        hits = _sample_hits(z, 1, [(0, 3, 0)], [fingerprint])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["excerpt_sha256"], "ab" * 32)
        self.assertEqual(hits[0]["excerpt_length"], 42)
        self.assertNotIn("excerpt", hits[0])
        self.assertNotIn("raw text", json.dumps(hits))


if __name__ == "__main__":
    unittest.main()
