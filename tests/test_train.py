import os
import tempfile
import time
import unittest
from pathlib import Path

from omp_gym.train import (
    TrainError,
    _finish_report,
    _require_fresh_adapter,
    _validate_loss_curves,
)


class ValidateLossCurvesTests(unittest.TestCase):
    def test_accepts_an_improving_run(self) -> None:
        _validate_loss_curves([2.0, 1.5, 1.0], [2.0, 1.8, 1.6])

    def test_rejects_a_flat_train_loss(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([1.0, 1.0], [])
        self.assertIn("train loss did not go down", str(caught.exception))
        self.assertIn("1.0 -> 1.0", str(caught.exception))

    def test_rejects_a_rising_val_loss(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([2.0, 1.0], [1.2, 1.7])
        self.assertIn("val loss went up", str(caught.exception))
        self.assertIn("1.2", str(caught.exception))
        self.assertIn("1.7", str(caught.exception))

    def test_accepts_a_run_without_val_losses(self) -> None:
        _validate_loss_curves([2.0, 1.0], [])

    def test_flags_the_observed_memorization_run(self) -> None:
        # Measured run: train 2.058 -> 1.404 (-31.8%) while val
        # only moved 2.151 -> 2.001 (-7.0%).
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves(
                [2.058, 1.404], [2.151, 2.001]
            )
        message = str(caught.exception)
        self.assertIn("memorization-shaped", message)
        self.assertIn("31.8%", message)
        self.assertIn("7.0%", message)

    def test_accepts_a_balanced_run(self) -> None:
        # Train -30% with val -25% is healthy generalization.
        _validate_loss_curves([2.0, 1.4], [2.0, 1.5])

    def test_accepts_a_small_train_drop_with_flat_val(self) -> None:
        # A 10% train drop is below the 15% bar; flat val passes.
        _validate_loss_curves([2.0, 1.8], [2.0, 2.0])

    def test_rejects_a_run_with_no_loss_reports(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([1.0], [])
        self.assertIn("no loss reports", str(caught.exception))


class AdapterFreshnessTests(unittest.TestCase):
    def test_rejects_a_missing_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time())
        self.assertIn("was not written", str(caught.exception))

    def test_rejects_a_stale_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_file = Path(tmp) / "adapters.safetensors"
            adapter_file.write_bytes(b"weights")
            old = time.time() - 3600
            os.utime(adapter_file, (old, old))
            with self.assertRaises(TrainError) as caught:
                _require_fresh_adapter(Path(tmp), time.time())
        self.assertIn("stale artifact", str(caught.exception))

    def test_accepts_a_fresh_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_file = Path(tmp) / "adapters.safetensors"
            adapter_file.write_bytes(b"weights")
            _require_fresh_adapter(Path(tmp), time.time() - 60)


class FinishReportTests(unittest.TestCase):
    def _finish(self, adapter_dir: Path, started_at: float):
        return _finish_report(
            "test-model",
            Path("data"),
            2,
            adapter_dir,
            [2.0, 1.0],
            [],
            "test-gpu",
            started_at,
        )

    def test_rejects_a_stale_adapter_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_file = Path(tmp) / "adapters.safetensors"
            adapter_file.write_bytes(b"weights")
            old = time.time() - 3600
            os.utime(adapter_file, (old, old))
            with self.assertRaises(TrainError) as caught:
                self._finish(Path(tmp), time.time())
        self.assertIn("stale artifact", str(caught.exception))

    def test_accepts_a_fresh_adapter_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_file = Path(tmp) / "adapters.safetensors"
            adapter_file.write_bytes(b"weights")
            report = self._finish(Path(tmp), time.time() - 60)
            self.assertEqual(report.last_train_loss, 1.0)
            report_file = Path(tmp) / "train_report.json"
            self.assertTrue(report_file.is_file())


if __name__ == "__main__":
    unittest.main()
