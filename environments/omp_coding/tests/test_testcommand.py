from __future__ import annotations

import unittest

from omp_coding.testcommand import (
    CHANGES_ARTIFACT,
    OVERLAY_EXIT_CODE,
    PATCH_EXIT_CODE,
    PATCH_PATH,
    SEALED_DIR,
    START_TAG,
    CommandOutcome,
    TestCounts,
    changes_script,
    grade_outcome,
    grading_script,
    parse_test_output,
)

PYTEST_VERBOSE = (
    "collected 4 items\n\n"
    "tests/test_a.py ..F.\n\n"
    "=================================== FAILURES ===================================\n"
    "___ test_c ___\n"
    "========================= 1 failed, 3 passed in 0.12s ==========================\n"
)
PYTEST_QUIET = "..F.\n1 failed, 3 passed, 2 skipped, 1 warning in 0.04s\n"
UNITTEST_FAILED = (
    "..F\n"
    "======================================================================\n"
    "FAIL: test_c (tests.test_a.Case)\n"
    "----------------------------------------------------------------------\n"
    "Ran 3 tests in 0.002s\n\n"
    "FAILED (failures=1)\n"
)
CARGO = (
    "running 2 tests\ntest a ... ok\ntest b ... FAILED\n\n"
    "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n"
    "running 1 test\ntest doc ... ok\n\n"
    "test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n"
)
GO = "=== RUN   TestA\n--- PASS: TestA (0.00s)\n=== RUN   TestB\n--- FAIL: TestB (0.00s)\n--- SKIP: TestC (0.00s)\nFAIL\n"
BUN = "bun test v1.1.0\n\n 5 pass\n 1 skip\n 2 fail\nRan 8 tests across 2 files.\n"
VITEST = " Test Files  1 failed | 1 passed (2)\n      Tests  1 failed | 7 passed | 1 skipped (9)\n   Start at  10:00:00\n"
JEST = "Test Suites: 1 failed, 1 passed, 2 total\nTests:       1 failed, 7 passed, 8 total\nSnapshots:   0 total\n"
NODE_TEST = "# tests 4\n# suites 0\n# pass 3\n# fail 1\n# skipped 0\n"


class ParserTests(unittest.TestCase):
    def test_pytest_bars_and_quiet_summaries(self) -> None:
        self.assertEqual(parse_test_output(PYTEST_VERBOSE), TestCounts(3, 1, 0, 0))
        self.assertEqual(parse_test_output(PYTEST_QUIET), TestCounts(3, 1, 0, 2))
        self.assertEqual(parse_test_output("2 passed in 0.01s\n"), TestCounts(2, 0, 0, 0))

    def test_pytest_errors_count_separately(self) -> None:
        counts = parse_test_output("=== 2 passed, 1 error in 0.5s ===\n")
        self.assertEqual(counts, TestCounts(2, 0, 1, 0))

    def test_other_runners(self) -> None:
        self.assertEqual(parse_test_output(UNITTEST_FAILED), TestCounts(2, 1, 0, 0))
        self.assertEqual(parse_test_output(CARGO), TestCounts(2, 1, 0, 1))
        self.assertEqual(parse_test_output(GO), TestCounts(1, 1, 0, 1))
        self.assertEqual(parse_test_output(BUN), TestCounts(5, 2, 0, 1))
        self.assertEqual(parse_test_output(VITEST), TestCounts(7, 1, 0, 1))
        self.assertEqual(parse_test_output(JEST), TestCounts(7, 1, 0, 0))
        self.assertEqual(parse_test_output(NODE_TEST), TestCounts(3, 1, 0, 0))

    def test_unrecognized_output_yields_none(self) -> None:
        self.assertIsNone(parse_test_output("Traceback (most recent call last):\nImportError\n"))
        self.assertIsNone(parse_test_output(""))


class GradeTests(unittest.TestCase):
    def test_all_passed_is_full_reward(self) -> None:
        result = grade_outcome(CommandOutcome(0, "4 passed in 0.1s\n", ""), expected_cases=4)
        self.assertEqual((result.status, result.reward, result.passed_cases, result.total_cases), ("passed", 1.0, 4, 4))

    def test_partial_credit_is_the_passed_fraction(self) -> None:
        result = grade_outcome(CommandOutcome(1, PYTEST_QUIET, ""), expected_cases=4)
        self.assertEqual(result.status, "failed")
        self.assertAlmostEqual(result.reward, 0.75)

    def test_shrunken_test_count_scores_zero(self) -> None:
        result = grade_outcome(CommandOutcome(0, "2 passed in 0.1s\n", ""), expected_cases=4)
        self.assertEqual((result.status, result.reward), ("error", 0.0))
        self.assertIn("shrank", result.reason)

    def test_exit_code_mode_only_for_single_case_references(self) -> None:
        passed = grade_outcome(CommandOutcome(0, "ok\n", ""), expected_cases=1)
        failed = grade_outcome(CommandOutcome(2, "boom\n", ""), expected_cases=1)
        unparsed = grade_outcome(CommandOutcome(0, "ok\n", ""), expected_cases=3)
        self.assertEqual((passed.status, passed.reward), ("passed", 1.0))
        self.assertEqual((failed.status, failed.reward), ("failed", 0.0))
        self.assertEqual((unparsed.status, unparsed.reward), ("error", 0.0))

    def test_nonzero_exit_without_reported_failures_is_an_error(self) -> None:
        result = grade_outcome(CommandOutcome(3, "4 passed in 0.1s\n", "crash"), expected_cases=4)
        self.assertEqual((result.status, result.reward), ("error", 0.0))

    def test_reserved_exit_codes_and_timeouts(self) -> None:
        patch = grade_outcome(CommandOutcome(PATCH_EXIT_CODE, "", ""), expected_cases=2)
        overlay = grade_outcome(CommandOutcome(OVERLAY_EXIT_CODE, "", ""), expected_cases=2)
        timeout = grade_outcome(CommandOutcome(0, "", "", timed_out=True), expected_cases=2)
        self.assertEqual([patch.status, overlay.status, timeout.status], ["error", "error", "timeout"])
        self.assertEqual([patch.reward, overlay.reward, timeout.reward], [0.0, 0.0, 0.0])


class ScriptTests(unittest.TestCase):
    def test_grading_script_applies_patch_then_overlays_then_execs(self) -> None:
        script = grading_script(["sh", "-c", "pytest -q"], ["tests/test_a.py", "conftest.py"], apply_patch=True)
        lines = script.splitlines()
        patch_line = next(index for index, line in enumerate(lines) if PATCH_PATH in line)
        overlay_lines = [index for index, line in enumerate(lines) if line.startswith("cp -f")]
        self.assertIn(f"exit {PATCH_EXIT_CODE}", lines[patch_line])
        self.assertTrue(all(index > patch_line for index in overlay_lines))
        self.assertIn(f"mkdir -p /workspace/tests || exit {OVERLAY_EXIT_CODE}", lines)
        self.assertIn(f"cp -f {SEALED_DIR}/tests/test_a.py /workspace/tests/test_a.py || exit {OVERLAY_EXIT_CODE}", lines)
        self.assertEqual(lines[-1], "exec sh -c 'pytest -q'")

    def test_before_run_script_never_touches_the_patch(self) -> None:
        script = grading_script(["pytest"], ["tests/test_a.py"], apply_patch=False)
        self.assertNotIn(PATCH_PATH, script)

    def test_changes_script_diffs_against_the_start_tag(self) -> None:
        script = changes_script()
        self.assertIn(f"diff --cached --binary {START_TAG} > {CHANGES_ARTIFACT}", script)
        self.assertIn("add -A", script)


if __name__ == "__main__":
    unittest.main()
