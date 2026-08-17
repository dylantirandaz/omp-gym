import unittest

from rle import RleError, decode, encode


class EncodeTest(unittest.TestCase):
    def test_empty_text(self) -> None:
        self.assertEqual(encode(""), "")

    def test_single_letter(self) -> None:
        self.assertEqual(encode("a"), "1a")

    def test_short_runs(self) -> None:
        self.assertEqual(encode("aaabb"), "3a2b")

    def test_runs_of_one(self) -> None:
        self.assertEqual(encode("abc"), "1a1b1c")

    def test_repeated_run_letters(self) -> None:
        self.assertEqual(encode("aabbaa"), "2a2b2a")


class DecodeTest(unittest.TestCase):
    def test_empty_encoding(self) -> None:
        self.assertEqual(decode(""), "")

    def test_short_runs(self) -> None:
        self.assertEqual(decode("3a2b"), "aaabb")

    def test_multi_digit_count(self) -> None:
        self.assertEqual(decode("12x"), "x" * 12)

    def test_trailing_count_is_invalid(self) -> None:
        self.assertEqual(decode("3a2"), RleError("invalid encoding"))

    def test_missing_counts_are_invalid(self) -> None:
        self.assertEqual(decode("abc"), RleError("invalid encoding"))


class RoundTripTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        self.assertEqual(decode(encode("hellooo")), "hellooo")


if __name__ == "__main__":
    unittest.main()
