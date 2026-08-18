import unittest

from coerce import coerce_cell
from join import JoinError, join_rows


class CoerceTest(unittest.TestCase):
    def test_empty_cell_is_none(self) -> None:
        self.assertIsNone(coerce_cell(""))

    def test_whitespace_cell_is_none(self) -> None:
        self.assertIsNone(coerce_cell("   "))

    def test_int(self) -> None:
        self.assertEqual(coerce_cell(" 42 "), 42)

    def test_negative_int(self) -> None:
        self.assertEqual(coerce_cell("-3"), -3)

    def test_zero_is_an_int(self) -> None:
        self.assertEqual(coerce_cell("0"), 0)

    def test_leading_zero_code_stays_str(self) -> None:
        self.assertEqual(coerce_cell("007"), "007")

    def test_negative_leading_zero_stays_str(self) -> None:
        self.assertEqual(coerce_cell("-01"), "-01")

    def test_float(self) -> None:
        value = coerce_cell("3.5")
        self.assertIsInstance(value, float)
        self.assertEqual(value, 3.5)

    def test_whole_float_stays_float(self) -> None:
        value = coerce_cell("1.0")
        self.assertIsInstance(value, float)
        self.assertEqual(value, 1.0)

    def test_float_needs_digits_on_both_sides(self) -> None:
        self.assertEqual(coerce_cell(".5"), ".5")
        self.assertEqual(coerce_cell("5."), "5.")

    def test_float_with_leading_zero_code_stays_str(self) -> None:
        self.assertEqual(coerce_cell("00.5"), "00.5")

    def test_plus_sign_stays_str(self) -> None:
        self.assertEqual(coerce_cell("+5"), "+5")

    def test_exponent_stays_str(self) -> None:
        self.assertEqual(coerce_cell("1e3"), "1e3")

    def test_word_is_stripped(self) -> None:
        self.assertEqual(coerce_cell("  ada  "), "ada")


class JoinTest(unittest.TestCase):
    def test_simple_match(self) -> None:
        left = [{"id": "1", "name": "ada"}]
        right = [{"id": "1", "score": "9.5"}]
        self.assertEqual(
            join_rows(left, right, "id"),
            [{"id": 1, "name": "ada", "score": 9.5}],
        )

    def test_left_order_is_kept(self) -> None:
        left = [{"id": "2"}, {"id": "1"}]
        right = [{"id": "1", "v": "a"}, {"id": "2", "v": "b"}]
        self.assertEqual(
            join_rows(left, right, "id"),
            [{"id": 2, "v": "b"}, {"id": 1, "v": "a"}],
        )

    def test_unmatched_left_rows_are_dropped(self) -> None:
        left = [{"id": "1"}, {"id": "9"}]
        right = [{"id": "1", "v": "a"}]
        self.assertEqual(join_rows(left, right, "id"), [{"id": 1, "v": "a"}])

    def test_last_duplicate_right_row_wins(self) -> None:
        left = [{"id": "1"}]
        right = [{"id": "1", "v": "old"}, {"id": "1", "v": "new"}]
        self.assertEqual(join_rows(left, right, "id"), [{"id": 1, "v": "new"}])

    def test_keys_match_after_coercion(self) -> None:
        left = [{"id": " 7 "}]
        right = [{"id": "7", "v": "x"}]
        self.assertEqual(join_rows(left, right, "id"), [{"id": 7, "v": "x"}])

    def test_int_key_never_matches_float_key(self) -> None:
        left = [{"id": "7"}]
        right = [{"id": "7.0", "v": "x"}]
        self.assertEqual(join_rows(left, right, "id"), [])

    def test_none_keys_never_match(self) -> None:
        left = [{"id": "", "v": "l"}]
        right = [{"id": "", "v": "r"}, {"id": "  ", "v": "r2"}]
        self.assertEqual(join_rows(left, right, "id"), [])

    def test_right_value_wins_on_shared_columns(self) -> None:
        left = [{"id": "1", "v": "left"}]
        right = [{"id": "1", "v": "right"}]
        self.assertEqual(join_rows(left, right, "id"), [{"id": 1, "v": "right"}])

    def test_missing_key_column_on_the_left(self) -> None:
        left = [{"id": "1"}, {"name": "no-key"}]
        right = [{"id": "1"}]
        self.assertEqual(
            join_rows(left, right, "id"),
            JoinError("left row 1 has no column id"),
        )

    def test_missing_key_column_on_the_right(self) -> None:
        left = [{"id": "1"}]
        right = [{"v": "a"}]
        self.assertEqual(
            join_rows(left, right, "id"),
            JoinError("right row 0 has no column id"),
        )

    def test_left_error_wins_over_right_error(self) -> None:
        left = [{"x": "1"}]
        right = [{"y": "2"}]
        self.assertEqual(
            join_rows(left, right, "id"),
            JoinError("left row 0 has no column id"),
        )

    def test_empty_inputs(self) -> None:
        self.assertEqual(join_rows([], [], "id"), [])


if __name__ == "__main__":
    unittest.main()
