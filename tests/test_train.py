import unittest

from omp_gym.train import TrainError, _validate_loss_curves


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

    def test_rejects_a_run_with_no_loss_reports(self) -> None:
        with self.assertRaises(TrainError) as caught:
            _validate_loss_curves([1.0], [])
        self.assertIn("no loss reports", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
