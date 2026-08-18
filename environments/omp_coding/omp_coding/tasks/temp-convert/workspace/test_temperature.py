import unittest

from temperature import ConvertError, convert


class ConvertTest(unittest.TestCase):
    def test_freezing_point_to_celsius(self) -> None:
        self.assertEqual(convert("32F"), "0C")

    def test_boiling_point_to_celsius(self) -> None:
        self.assertEqual(convert("212F"), "100C")

    def test_freezing_point_to_fahrenheit(self) -> None:
        self.assertEqual(convert("0C"), "32F")

    def test_boiling_point_to_fahrenheit(self) -> None:
        self.assertEqual(convert("100C"), "212F")

    def test_negative_crossover_point(self) -> None:
        self.assertEqual(convert("-40F"), "-40C")

    def test_negative_celsius_to_fahrenheit(self) -> None:
        self.assertEqual(convert("-10C"), "14F")

    def test_result_rounds_to_nearest_degree(self) -> None:
        self.assertEqual(convert("98F"), "37C")

    def test_unknown_unit(self) -> None:
        self.assertEqual(convert("12K"), ConvertError("invalid value"))

    def test_unit_before_degrees(self) -> None:
        self.assertEqual(convert("F32"), ConvertError("invalid value"))

    def test_fractional_degrees(self) -> None:
        self.assertEqual(convert("3.5C"), ConvertError("invalid value"))

    def test_empty_value(self) -> None:
        self.assertEqual(convert(""), ConvertError("invalid value"))


if __name__ == "__main__":
    unittest.main()
