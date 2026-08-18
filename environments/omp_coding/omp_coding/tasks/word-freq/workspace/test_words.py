import unittest

from words import word_freq


class WordFreqTest(unittest.TestCase):
    def test_empty_text(self) -> None:
        self.assertEqual(word_freq(""), ())

    def test_one_word(self) -> None:
        self.assertEqual(word_freq("hello"), (("hello", 1),))

    def test_case_folding(self) -> None:
        self.assertEqual(word_freq("Dog dog DOG"), (("dog", 3),))

    def test_punctuation_splits_words(self) -> None:
        self.assertEqual(
            word_freq("stop, go; stop!"),
            (("stop", 2), ("go", 1)),
        )

    def test_tie_breaks_by_word(self) -> None:
        self.assertEqual(
            word_freq("pear apple pear apple"),
            (("apple", 2), ("pear", 2)),
        )

    def test_digits_stay_inside_words(self) -> None:
        self.assertEqual(word_freq("area51 area51"), (("area51", 2),))

    def test_mixed_sentence(self) -> None:
        self.assertEqual(
            word_freq("The cat and the dog saw the cat."),
            (("the", 3), ("cat", 2), ("and", 1), ("dog", 1), ("saw", 1)),
        )


if __name__ == "__main__":
    unittest.main()
