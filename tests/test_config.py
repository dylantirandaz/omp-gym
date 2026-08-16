import tempfile
import unittest
from pathlib import Path

from omp_gym.config import DEFAULT_MODEL, default_model


class DefaultModelTests(unittest.TestCase):
    def test_missing_file_gives_builtin_default(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "gym.toml"
        self.assertEqual(default_model(missing), DEFAULT_MODEL)

    def test_configured_model_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_file = Path(temporary_directory) / "gym.toml"
            config_file.write_text('model = "models/my-local-mlx"\n')
            self.assertEqual(default_model(config_file), "models/my-local-mlx")

    def test_blank_model_is_a_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_file = Path(temporary_directory) / "gym.toml"
            config_file.write_text('model = ""\n')
            with self.assertRaises(SystemExit):
                default_model(config_file)

    def test_malformed_file_is_a_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_file = Path(temporary_directory) / "gym.toml"
            config_file.write_text("model = [unclosed\n")
            with self.assertRaises(SystemExit):
                default_model(config_file)


if __name__ == "__main__":
    unittest.main()
