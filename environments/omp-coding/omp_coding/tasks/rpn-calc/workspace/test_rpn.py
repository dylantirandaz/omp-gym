import unittest

from rpn import RpnError, evaluate_rpn


class EvaluateRpnTest(unittest.TestCase):
    def test_single_number(self) -> None:
        self.assertEqual(evaluate_rpn(["42"]), 42)

    def test_addition(self) -> None:
        self.assertEqual(evaluate_rpn("3 4 +".split()), 7)

    def test_nested_expression(self) -> None:
        self.assertEqual(evaluate_rpn("5 1 2 + 4 * + 3 -".split()), 14)

    def test_multiplication(self) -> None:
        self.assertEqual(evaluate_rpn("6 7 *".split()), 42)

    def test_division_truncates_toward_zero(self) -> None:
        self.assertEqual(evaluate_rpn("7 -2 /".split()), -3)

    def test_negative_dividend_truncates_toward_zero(self) -> None:
        self.assertEqual(evaluate_rpn("-7 2 /".split()), -3)

    def test_unknown_token(self) -> None:
        self.assertEqual(evaluate_rpn("3 x +".split()), RpnError("unknown token"))

    def test_stack_underflow(self) -> None:
        self.assertEqual(evaluate_rpn("3 +".split()), RpnError("stack underflow"))

    def test_leftover_operands(self) -> None:
        self.assertEqual(evaluate_rpn("1 2".split()), RpnError("leftover operands"))

    def test_division_by_zero(self) -> None:
        self.assertEqual(evaluate_rpn("1 0 /".split()), RpnError("division by zero"))

    def test_empty_expression(self) -> None:
        self.assertEqual(evaluate_rpn([]), RpnError("empty expression"))


if __name__ == "__main__":
    unittest.main()
