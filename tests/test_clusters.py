import unittest

from omp_gym.clusters import _CORRECTION, _EDIT_FAIL, _GAVE_UP

# Real error strings from the edit tool, copied from session data
# under runs/bench-v10-100-early/csv-total-20260815-075334-b80b88/.
REAL_EDIT_ERRORS = [
    'input must begin with "[PATH#HASH]" on the first non-blank '
    'line for anchored edits; got: "PUT 1=". '
    'Example: "[src/foo.ts#1A2B]" then edit ops.',
    "line 1: payload line has no preceding hunk header. "
    "Use `PUT N.=M:`, `CUT N.=M`, or `PUT <N:`/`PUT >N:` "
    'above the body. Got "PUT 1=".',
]

# Real correction from a local session under
# ~/.omp/agent/sessions/-macmini projects-givemeanode/.
REAL_CORRECTION = (
    "hmm mcp server still failed to connect, is there a way u "
    "could connect to it from here?"
)


class CorrectionPatternTests(unittest.TestCase):
    def test_ignores_a_question_that_contains_wrong(self) -> None:
        self.assertIsNone(
            _CORRECTION.search("what's wrong with this regex?")
        )

    def test_ignores_a_task_that_contains_stop(self) -> None:
        self.assertIsNone(_CORRECTION.search("add a --stop flag"))

    def test_matches_a_real_correction(self) -> None:
        self.assertIsNotNone(_CORRECTION.search(REAL_CORRECTION))

    def test_matches_second_person_corrections(self) -> None:
        for text in (
            "no, that's wrong, the flag is --iters",
            "you are wrong about the schema",
            "stop doing that and revert the change",
            "undo that edit",
            "this still doesn't work",
        ):
            self.assertIsNotNone(_CORRECTION.search(text), text)


class EditFailPatternTests(unittest.TestCase):
    def test_ignores_a_missing_file_traceback(self) -> None:
        self.assertIsNone(
            _EDIT_FAIL.search("FileNotFoundError: config.yaml not found")
        )

    def test_ignores_an_html_anchor(self) -> None:
        self.assertIsNone(_EDIT_FAIL.search('<a href="#anchor">'))

    def test_ignores_prose_about_stale_data(self) -> None:
        self.assertIsNone(
            _EDIT_FAIL.search(
                "The stale canonical environment controls had "
                "been removed."
            )
        )

    def test_matches_real_edit_tool_errors(self) -> None:
        for text in REAL_EDIT_ERRORS:
            self.assertIsNotNone(_EDIT_FAIL.search(text), text)


class GaveUpPatternTests(unittest.TestCase):
    def test_ignores_an_apology_with_a_fix(self) -> None:
        self.assertIsNone(
            _GAVE_UP.search(
                "I'm sorry, that was my mistake, here is the fix:"
            )
        )

    def test_matches_phrases_of_abandonment(self) -> None:
        for text in (
            "I cannot proceed without the API key.",
            "I am giving up on this approach.",
            "I am unable to complete the task.",
        ):
            self.assertIsNotNone(_GAVE_UP.search(text), text)


if __name__ == "__main__":
    unittest.main()
