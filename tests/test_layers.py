import argparse
import unittest

from omp_gym.cli import _layer_selection_argument
from omp_gym.layers import (
    adapter_layer_selection,
    layer_config_fields,
    mlx_num_layers,
    parse_layer_selection,
)


class LayerSelectionTests(unittest.TestCase):
    def test_all_is_explicit_inside_the_harness(self) -> None:
        self.assertEqual(parse_layer_selection("all"), "all")
        self.assertEqual(mlx_num_layers("all"), 0)
        self.assertEqual(
            layer_config_fields("all"),
            {"num_layers": 0, "layer_selection": "all"},
        )

    def test_positive_counts_are_preserved(self) -> None:
        self.assertEqual(parse_layer_selection(16), 16)
        self.assertEqual(mlx_num_layers(16), 16)

    def test_numeric_sentinels_are_not_internal_values(self) -> None:
        for value in (0, -1, True, "-1", "0"):
            self.assertIsNone(parse_layer_selection(value))

    def test_cli_accepts_all_and_positive_counts(self) -> None:
        self.assertEqual(_layer_selection_argument("all"), "all")
        self.assertEqual(_layer_selection_argument("24"), 24)

    def test_cli_rejects_numeric_sentinels(self) -> None:
        for value in ("0", "-1", "none"):
            with self.assertRaises(argparse.ArgumentTypeError):
                _layer_selection_argument(value)

    def test_adapter_config_cross_checks_the_explicit_field(self) -> None:
        self.assertEqual(
            adapter_layer_selection({"num_layers": 0, "layer_selection": "all"}),
            "all",
        )
        self.assertEqual(
            adapter_layer_selection({"num_layers": 12, "layer_selection": 12}),
            12,
        )
        self.assertIsNone(
            adapter_layer_selection({"num_layers": 8, "layer_selection": "all"})
        )

    def test_legacy_zero_is_normalized_only_at_adapter_boundary(self) -> None:
        self.assertEqual(adapter_layer_selection({"num_layers": 0}), "all")


if __name__ == "__main__":
    unittest.main()
