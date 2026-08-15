import unittest

from intervals import merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(merge_intervals([]), ())

    def test_one_interval(self) -> None:
        self.assertEqual(merge_intervals([(2, 5)]), ((2, 5),))

    def test_unsorted_overlapping_intervals(self) -> None:
        intervals = [(8, 10), (1, 4), (3, 7)]
        self.assertEqual(merge_intervals(intervals), ((1, 7), (8, 10)))

    def test_touching_intervals_merge(self) -> None:
        self.assertEqual(
            merge_intervals([(1, 3), (3, 5), (9, 9)]),
            ((1, 5), (9, 9)),
        )

    def test_nested_and_duplicate_intervals(self) -> None:
        self.assertEqual(
            merge_intervals([(1, 9), (2, 3), (1, 9), (4, 8)]),
            ((1, 9),),
        )

    def test_negative_values(self) -> None:
        self.assertEqual(
            merge_intervals([(-3, -1), (-8, -5), (-5, -3)]),
            ((-8, -1),),
        )

    def test_disjoint_intervals_are_sorted(self) -> None:
        self.assertEqual(
            merge_intervals([(10, 12), (1, 2), (5, 6)]),
            ((1, 2), (5, 6), (10, 12)),
        )

    def test_input_does_not_change(self) -> None:
        intervals = [(5, 7), (1, 2)]
        original = list(intervals)
        merge_intervals(intervals)
        self.assertEqual(intervals, original)


if __name__ == "__main__":
    unittest.main()
