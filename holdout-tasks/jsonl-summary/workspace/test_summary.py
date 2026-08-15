import unittest

from summary import InvalidRecord, Summary, summarize_lines


class SummarizeLinesTests(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(summarize_lines([]), Summary(0, 0, 0))

    def test_valid_records(self) -> None:
        lines = [
            '{"status": "accepted", "duration_ms": 12}\n',
            '{"status": "rejected", "duration_ms": 5}\n',
            '{"status": "accepted", "duration_ms": 8}\n',
        ]
        self.assertEqual(summarize_lines(lines), Summary(2, 1, 25))

    def test_blank_lines_are_ignored(self) -> None:
        lines = [
            "\n",
            "   \n",
            '{"status": "accepted", "duration_ms": 3}\n',
        ]
        self.assertEqual(summarize_lines(lines), Summary(1, 0, 3))

    def test_invalid_json_has_physical_line_number(self) -> None:
        lines = [
            "\n",
            '{"status": "accepted", "duration_ms": 3}\n',
            "{not-json}\n",
        ]
        self.assertEqual(
            summarize_lines(lines),
            InvalidRecord(3, "invalid JSON"),
        )

    def test_record_must_be_an_object(self) -> None:
        self.assertEqual(
            summarize_lines(['["accepted", 2]\n']),
            InvalidRecord(1, "record must be an object"),
        )

    def test_status_is_checked(self) -> None:
        line = '{"status": "pending", "duration_ms": 2}\n'
        self.assertEqual(
            summarize_lines([line]),
            InvalidRecord(1, "invalid status"),
        )

    def test_duration_is_checked(self) -> None:
        cases = [
            '{"status": "accepted"}\n',
            '{"status": "accepted", "duration_ms": -1}\n',
            '{"status": "accepted", "duration_ms": 1.5}\n',
            '{"status": "accepted", "duration_ms": true}\n',
        ]
        for line in cases:
            with self.subTest(line=line):
                self.assertEqual(
                    summarize_lines([line]),
                    InvalidRecord(1, "invalid duration_ms"),
                )

    def test_first_invalid_record_stops_processing(self) -> None:
        lines = [
            '{"status": "accepted", "duration_ms": 4}\n',
            '{"status": "bad", "duration_ms": 2}\n',
            "{not-json}\n",
        ]
        self.assertEqual(
            summarize_lines(lines),
            InvalidRecord(2, "invalid status"),
        )


if __name__ == "__main__":
    unittest.main()
